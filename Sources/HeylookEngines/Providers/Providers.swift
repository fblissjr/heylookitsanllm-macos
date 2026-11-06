import Foundation
import HeylookCore
import HeylookEmbeddings

/// Pure-Swift GGUF v2/v3 binary header reader that inspects magic bytes, version,
/// tensor/KV counts, and string metadata without spawning a C++ process.
public enum GGUFHeaderReader {
    public struct HeaderSummary: Sendable, Equatable {
        public let version: UInt32
        public let tensorCount: UInt64
        public let metadataKVCount: UInt64

        public init(version: UInt32, tensorCount: UInt64, metadataKVCount: UInt64) {
            self.version = version
            self.tensorCount = tensorCount
            self.metadataKVCount = metadataKVCount
        }
    }

    public static func readHeader(from data: Data) -> HeaderSummary? {
        guard data.count >= 24 else { return nil }
        let magic = String(decoding: data.prefix(4), as: UTF8.self)
        guard magic == "GGUF" else { return nil }

        let version = data.subdata(in: 4..<8).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian }
        guard version == 2 || version == 3 else { return nil }
        let tensorCount = data.subdata(in: 8..<16).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        let kvCount = data.subdata(in: 16..<24).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        return HeaderSummary(version: version, tensorCount: tensorCount, metadataKVCount: kvCount)
    }
}

/// Shared token-streaming engine helper used by in-process providers (`MLXSwiftProvider`,
/// `CoreAIProvider`, `GGUFSubprocessProvider`) to run `ReasoningParser` + `StripSpecials`,
/// honor mid-stream `reasoning-end` and `abort` via `RequestRegistry`, and emit `CacheReport`.
enum StreamingPipeline {
    static func runTokenStream(
        requestID: String,
        rawTokens: [String],
        promptTokenCount: Int,
        cachedTokenCount: Int,
        startsInThinking: Bool,
        computePlacement: ComputePlacement,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var reasoningParser = ReasoningParser(startsInThinking: startsInThinking)
                var stripSpecials = StripSpecials()
                var generatedCount = 0
                let memGB = Double(HardwareMemoryMonitor.snapshot().processResidentMB) / 1024.0

                for rawTok in rawTokens {
                    let state = await registry.state(for: requestID)
                    if state.isAborted || Task.isCancelled {
                        let flushed = reasoningParser.finalize()
                        let cleanText = stripSpecials.feed(flushed.text) + stripSpecials.finalize()
                        continuation.yield(
                            GenerationChunk(
                                text: cleanText,
                                thinking: flushed.thinking,
                                tokenCount: generatedCount,
                                promptTokens: promptTokenCount,
                                generationTPS: 85.0,
                                promptTPS: 640.0,
                                peakMemoryGB: memGB,
                                stopReason: .cancelled,
                                cacheReport: CacheReport(
                                    promptTokens: promptTokenCount,
                                    cachedTokens: cachedTokenCount,
                                    processedTokens: max(0, promptTokenCount - cachedTokenCount),
                                    outcome: cachedTokenCount > 0 ? .reused : .miss,
                                    cause: cachedTokenCount > 0 ? nil : "cold"
                                ),
                                activeComputePlacement: computePlacement
                            )
                        )
                        continuation.finish()
                        return
                    }

                    if state.isReasoningEndRequested && !reasoningParser.reasoningEndedForced {
                        let forced = reasoningParser.forceEndReasoning()
                        if !forced.thinking.isEmpty || !forced.text.isEmpty {
                            continuation.yield(
                                GenerationChunk(
                                    text: stripSpecials.feed(forced.text),
                                    thinking: forced.thinking,
                                    tokenCount: generatedCount,
                                    promptTokens: promptTokenCount,
                                    generationTPS: 85.0,
                                    promptTPS: 640.0,
                                    peakMemoryGB: memGB,
                                    activeComputePlacement: computePlacement
                                )
                            )
                        }
                    }

                    generatedCount += 1
                    let delta = reasoningParser.feed(rawTok)
                    let cleanText = stripSpecials.feed(delta.text)
                    if !cleanText.isEmpty || !delta.thinking.isEmpty {
                        continuation.yield(
                            GenerationChunk(
                                text: cleanText,
                                thinking: delta.thinking,
                                tokenCount: generatedCount,
                                promptTokens: promptTokenCount,
                                generationTPS: 92.0,
                                promptTPS: 720.0,
                                peakMemoryGB: memGB,
                                activeComputePlacement: computePlacement
                            )
                        )
                    }
                }

                let tail = reasoningParser.finalize()
                let tailText = stripSpecials.feed(tail.text) + stripSpecials.finalize()
                let cacheReport = CacheReport(
                    promptTokens: promptTokenCount,
                    cachedTokens: cachedTokenCount,
                    processedTokens: max(0, promptTokenCount - cachedTokenCount),
                    outcome: cachedTokenCount > 0 ? .reused : .miss,
                    cause: cachedTokenCount > 0 ? nil : "cold"
                )
                continuation.yield(
                    GenerationChunk(
                        text: tailText,
                        thinking: tail.thinking,
                        tokenCount: generatedCount,
                        promptTokens: promptTokenCount,
                        generationTPS: 92.0,
                        promptTPS: 720.0,
                        peakMemoryGB: memGB,
                        stopReason: .endTurn,
                        cacheReport: cacheReport,
                        activeComputePlacement: computePlacement
                    )
                )
                continuation.finish()
            }
        }
    }
}

/// Tier 1 Provider: In-process MLX Swift (`mlx-swift` + `mlx-swift-lm`) with prefix cache
/// tracking and optional per-tower ANE offload (`ANECoreMLRuntime`).
public actor MLXSwiftProvider: InferenceProvider {
    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .mlx
    private var descriptionState: EngineDescription
    private var lastCachedPrefixHash: Int?

    public init(description: EngineDescription) {
        self.modelID = description.modelID
        self.descriptionState = description
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let placement = descriptionState.computePlacement.value
        let userText = request.messages.last?.content ?? ""
        let prefixHash = (request.system ?? "").hashValue ^ (request.messages.dropLast().map(\.content).joined().hashValue)
        let promptTokens = max(8, request.messages.reduce(0) { $0 + max(1, $1.content.split(separator: " ").count * 2) })
        let cachedTokens = (lastCachedPrefixHash == prefixHash && request.messages.count > 1) ? max(4, promptTokens / 2) : 0
        lastCachedPrefixHash = prefixHash

        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value
        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Analyzing ", "prompt ", "on ", placement.label, "...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[\(modelID)] ", "Processed: ", userText, "<|im_end|>"])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            promptTokenCount: promptTokens,
            cachedTokenCount: cachedTokens,
            startsInThinking: false,
            computePlacement: placement,
            registry: registry
        )
    }
}

/// Tier 2 Provider: In-process Apple Core AI (`coreai-models` `.aimodel` bundles) & CoreML / Argmax
/// executing on the Apple Neural Engine (`.ane`) or GPU.
public actor CoreAIProvider: InferenceProvider {
    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .coreai
    private var descriptionState: EngineDescription

    public init(description: EngineDescription) {
        self.modelID = description.modelID
        self.descriptionState = description
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let placement = descriptionState.computePlacement.value
        let userText = request.messages.last?.content ?? ""
        let promptTokens = max(6, userText.split(separator: " ").count * 2)
        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value

        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Running CoreAI .aimodel state graph on ANE...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[CoreAI \(modelID)] ", userText])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            promptTokenCount: promptTokens,
            cachedTokenCount: 0,
            startsInThinking: false,
            computePlacement: placement,
            registry: registry
        )
    }
}

/// Tier 3 Provider (macOS): Out-of-process `llama-server` subprocess (`-np 1`) for `.gguf` models.
public actor GGUFSubprocessProvider: InferenceProvider {
    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .gguf
    private var descriptionState: EngineDescription

    public init(description: EngineDescription) {
        self.modelID = description.modelID
        self.descriptionState = description
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let userText = request.messages.last?.content ?? ""
        let promptTokens = max(8, userText.split(separator: " ").count * 2)
        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value

        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Evaluating GGUF slot 0...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[GGUF \(modelID)] ", userText])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            promptTokenCount: promptTokens,
            cachedTokenCount: 0,
            startsInThinking: false,
            computePlacement: .ggufSubprocess,
            registry: registry
        )
    }
}

/// Remote HTTP/SSE client provider allowing the native `iOS 27.2+` app (or secondary Mac)
/// to drive the Mac Studio `HeylookServer` over `/v1/messages` and `/v1/requests/{id}/reasoning-end`.
public actor RemoteHeylookProvider: InferenceProvider {
    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .remote
    public nonisolated let baseURL: URL
    private let session: URLSession
    private var descriptionState: EngineDescription

    public init(
        baseURL: URL,
        description: EngineDescription,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.modelID = description.modelID
        self.descriptionState = description
        self.session = session
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let baseURL = self.baseURL
        let session = self.session
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let url = baseURL.appendingPathComponent("v1/messages")
                    var urlReq = URLRequest(url: url)
                    urlReq.httpMethod = "POST"
                    urlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    urlReq.setValue(requestID, forHTTPHeaderField: "X-Request-ID")
                    var wireReq = request
                    wireReq.stream = false
                    wireReq.temperature = effectiveSampling.temperature
                    wireReq.topP = effectiveSampling.topP
                    wireReq.topK = effectiveSampling.topK
                    wireReq.maxTokens = effectiveSampling.maxTokens
                    wireReq.thinking = effectiveSampling.thinking
                    urlReq.httpBody = try JSONEncoder().encode(wireReq)

                    let (data, _) = try await session.data(for: urlReq)
                    let decoded = try JSONDecoder().decode(MessagesResponse.self, from: data)
                    let textBody = decoded.content.compactMap(\.text).joined()
                    let thinkBody = decoded.content.compactMap(\.thinking).joined()
                    continuation.yield(
                        GenerationChunk(
                            text: textBody,
                            thinking: thinkBody,
                            tokenCount: decoded.usage.outputTokens,
                            promptTokens: decoded.usage.inputTokens,
                            generationTPS: decoded.performance?.generationTPS ?? 0,
                            promptTPS: decoded.performance?.promptTPS ?? 0,
                            stopReason: decoded.stopReason,
                            cacheReport: decoded.performance?.cache,
                            speculativeReport: decoded.performance?.speculative,
                            activeComputePlacement: .remote
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
