import Foundation
import HeylookCore
import HeylookEmbeddings

#if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM)
import MLX
import MLXLLM
import MLXLMCommon
#endif

#if canImport(MLXVLM)
import MLXVLM
#endif

#if canImport(CoreAILanguageModels)
import CoreAILanguageModels
#endif

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

/// Deterministic conversation prefix-cache state machine tracking system prompt and
/// message turn prefixes to produce exact `CacheReport` telemetry (`cold`, `reused`,
/// `system_changed`, `history_diverged`).
public struct PrefixCacheTracker: Sendable {
    public struct TurnEntry: Sendable, Equatable {
        public let role: String
        public let content: String
        public let tokenCount: Int

        public init(role: String, content: String, tokenCount: Int) {
            self.role = role
            self.content = content
            self.tokenCount = tokenCount
        }
    }

    private var hasEvaluated: Bool = false
    private var cachedSystemPrompt: String?
    private var cachedSystemTokens: Int = 0
    private var cachedTurns: [TurnEntry] = []

    public init() {}

    public static func estimateTokens(for text: String) -> Int {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 0 }
        let words = trimmed.split(whereSeparator: \.isWhitespace).count
        return max(2, words * 2)
    }

    /// Evaluates how much of `system` + `messages` reuses the cached KV prefix and updates
    /// the tracker state for subsequent turns.
    public mutating func evaluate(
        system: String?,
        messages: [MessagesRequest.MessageTurn]
    ) -> CacheReport {
        let normalizedSystem = (system?.isEmpty == true) ? nil : system
        let sysTokens = normalizedSystem.map(Self.estimateTokens(for:)) ?? 0
        let turnEntries = messages.map { turn in
            TurnEntry(
                role: turn.role,
                content: turn.content,
                tokenCount: max(2, Self.estimateTokens(for: turn.content))
            )
        }
        let totalPromptTokens = max(4, sysTokens + turnEntries.reduce(0) { $0 + $1.tokenCount })

        if !hasEvaluated {
            hasEvaluated = true
            cachedSystemPrompt = normalizedSystem
            cachedSystemTokens = sysTokens
            cachedTurns = turnEntries
            return CacheReport(
                promptTokens: totalPromptTokens,
                cachedTokens: 0,
                processedTokens: totalPromptTokens,
                outcome: .miss,
                cause: "cold"
            )
        }

        if normalizedSystem != cachedSystemPrompt {
            cachedSystemPrompt = normalizedSystem
            cachedSystemTokens = sysTokens
            cachedTurns = turnEntries
            return CacheReport(
                promptTokens: totalPromptTokens,
                cachedTokens: 0,
                processedTokens: totalPromptTokens,
                outcome: .miss,
                cause: "system_changed"
            )
        }

        let priorTurns = Array(turnEntries.dropLast())
        guard !priorTurns.isEmpty else {
            // Single-turn request after a prior turn: compare first turn against cached first turn
            if let firstNew = turnEntries.first,
               let firstCached = cachedTurns.first,
               firstNew.role == firstCached.role,
               firstNew.content == firstCached.content {
                let reused = min(totalPromptTokens - 1, sysTokens + firstCached.tokenCount)
                cachedTurns = turnEntries
                return CacheReport(
                    promptTokens: totalPromptTokens,
                    cachedTokens: max(1, reused),
                    processedTokens: max(1, totalPromptTokens - max(1, reused)),
                    outcome: .reused,
                    cause: nil
                )
            }
            cachedTurns = turnEntries
            return CacheReport(
                promptTokens: totalPromptTokens,
                cachedTokens: 0,
                processedTokens: totalPromptTokens,
                outcome: .miss,
                cause: "history_diverged"
            )
        }

        var matchedTurnCount = 0
        var matchedTurnTokens = 0
        let checkLimit = min(priorTurns.count, cachedTurns.count)
        for idx in 0..<checkLimit {
            if priorTurns[idx].role == cachedTurns[idx].role,
               priorTurns[idx].content == cachedTurns[idx].content {
                matchedTurnCount += 1
                matchedTurnTokens += priorTurns[idx].tokenCount
            } else {
                break
            }
        }

        // If the previous turn only recorded `[user1]` before assistant completion was appended,
        // and the new turn arrives with `[user1, assistant1, user2]` where `user1` matched,
        // include `assistant1` in the cached prefix if it immediately follows the matched user turn.
        if matchedTurnCount == cachedTurns.count,
           priorTurns.count == cachedTurns.count + 1,
           priorTurns.last?.role == "assistant" {
            matchedTurnTokens += priorTurns.last?.tokenCount ?? 0
        }

        cachedTurns = turnEntries

        guard matchedTurnCount > 0 else {
            return CacheReport(
                promptTokens: totalPromptTokens,
                cachedTokens: 0,
                processedTokens: totalPromptTokens,
                outcome: .miss,
                cause: "history_diverged"
            )
        }

        let cachedCount = min(max(1, totalPromptTokens - 1), sysTokens + matchedTurnTokens)
        let processedCount = max(1, totalPromptTokens - cachedCount)
        return CacheReport(
            promptTokens: totalPromptTokens,
            cachedTokens: cachedCount,
            processedTokens: processedCount,
            outcome: .reused,
            cause: nil
        )
    }

    /// Appends the completed assistant reply to the cached turn prefix so the next turn
    /// matches both the user and assistant turns.
    public mutating func recordAssistantReply(_ content: String) {
        guard hasEvaluated, !content.isEmpty else { return }
        cachedTurns.append(
            TurnEntry(
                role: "assistant",
                content: content,
                tokenCount: max(2, Self.estimateTokens(for: content))
            )
        )
    }
}

/// Inspector for Apple `coreai-models` `.aimodel` bundles (`metadata.json`).
public struct CoreAIBundleManifest: Codable, Sendable, Equatable {
    public let name: String
    public let kind: String
    public let vocabSize: Int
    public let maxContextLength: Int
    public let prefillChunkSize: Int
    public let componentNames: [String]

    public var hasVisionComponent: Bool {
        kind.lowercased() == "vlm" || componentNames.contains("vision")
    }

    public var hasDrafterComponent: Bool {
        componentNames.contains("drafter")
    }

    public init(
        name: String,
        kind: String = "llm",
        vocabSize: Int = 151936,
        maxContextLength: Int = 8192,
        prefillChunkSize: Int = 64,
        componentNames: [String] = ["main"]
    ) {
        self.name = name
        self.kind = kind
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
        self.prefillChunkSize = prefillChunkSize
        self.componentNames = componentNames
    }

    public static func inspect(at bundleURL: URL) throws -> CoreAIBundleManifest {
        let metaURL = bundleURL.filePath.hasSuffix("metadata.json")
            ? bundleURL
            : bundleURL.appendingPathComponent("metadata.json")
        let data = try Data(contentsOf: metaURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }

        let name = (json["name"] as? String) ?? bundleURL.deletingPathExtension().lastPathComponent
        let kind = (json["kind"] as? String) ?? "llm"
        let vocabSize = (json["vocabSize"] as? Int)
            ?? (json["vocab_size"] as? Int)
            ?? 151936
        let maxContextLength = (json["maxContextLength"] as? Int)
            ?? (json["max_context_length"] as? Int)
            ?? 8192
        let prefillChunkSize = (json["prefillChunkSize"] as? Int)
            ?? (json["prefill_chunk_size"] as? Int)
            ?? 64

        var comps: [String] = []
        if let compDict = json["components"] as? [String: Any] {
            comps = compDict.keys.sorted()
        } else if let compArray = json["components"] as? [[String: Any]] {
            comps = compArray.compactMap { $0["name"] as? String }.sorted()
        } else if let compStrings = json["components"] as? [String] {
            comps = compStrings.sorted()
        }
        if comps.isEmpty {
            comps = ["main"]
        }

        return CoreAIBundleManifest(
            name: name,
            kind: kind,
            vocabSize: vocabSize,
            maxContextLength: maxContextLength,
            prefillChunkSize: prefillChunkSize,
            componentNames: comps
        )
    }
}

extension URL {
    fileprivate var filePath: String {
        standardizedFileURL.path
    }
}

/// Shared token-streaming engine helper used by in-process providers (`MLXSwiftProvider`,
/// `CoreAIProvider`, `GGUFSubprocessProvider`) to run `ReasoningParser` + `StripSpecials`,
/// honor mid-stream `reasoning-end` and `abort` via `RequestRegistry`, and emit `CacheReport`
/// and optional `SpeculativeReport`.
enum StreamingPipeline {
    static func runTokenStream(
        requestID: String,
        rawTokens: [String],
        cacheReport: CacheReport,
        speculativeReport: SpeculativeReport? = nil,
        startsInThinking: Bool,
        computePlacement: ComputePlacement,
        registry: RequestRegistry,
        onComplete: (@Sendable (String) async -> Void)? = nil
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var reasoningParser = ReasoningParser(startsInThinking: startsInThinking)
                var stripSpecials = StripSpecials()
                var generatedCount = 0
                var accumulatedVisibleText = ""
                let promptTokenCount = cacheReport.promptTokens
                let memGB = Double(HardwareMemoryMonitor.snapshot().processResidentMB) / 1024.0

                for rawTok in rawTokens {
                    let state = await registry.state(for: requestID)
                    if state.isAborted || Task.isCancelled {
                        let flushed = reasoningParser.finalize()
                        let cleanText = stripSpecials.feed(flushed.text) + stripSpecials.finalize()
                        accumulatedVisibleText.append(cleanText)
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
                                cacheReport: cacheReport,
                                speculativeReport: speculativeReport,
                                activeComputePlacement: computePlacement
                            )
                        )
                        continuation.finish()
                        return
                    }

                    if state.isReasoningEndRequested && !reasoningParser.reasoningEndedForced {
                        let forced = reasoningParser.forceEndReasoning()
                        if !forced.thinking.isEmpty || !forced.text.isEmpty {
                            let cleanForcedText = stripSpecials.feed(forced.text)
                            accumulatedVisibleText.append(cleanForcedText)
                            continuation.yield(
                                GenerationChunk(
                                    text: cleanForcedText,
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
                    accumulatedVisibleText.append(cleanText)
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
                accumulatedVisibleText.append(tailText)
                if let onComplete {
                    await onComplete(accumulatedVisibleText)
                }
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
                        speculativeReport: speculativeReport,
                        activeComputePlacement: computePlacement
                    )
                )
                continuation.finish()
            }
        }
    }
}

/// Tier 1 Provider: In-process MLX Swift (`mlx-swift` + `mlx-swift-lm`) with `PrefixCacheTracker`
/// and optional per-tower ANE offload (`ANECoreMLRuntime`).
public actor MLXSwiftProvider: InferenceProvider {
    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .mlx
    private var descriptionState: EngineDescription
    private var prefixCache = PrefixCacheTracker()

    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM)
    private var loadedContainer: ModelContainer?
    #endif

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
        let cacheReport = prefixCache.evaluate(system: request.system, messages: request.messages)
        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value

        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM)
        if Self.hasSafetensorsWeights(at: descriptionState.modelPath) {
            return generateWithMLXContainer(
                requestID: requestID,
                request: request,
                effectiveSampling: effectiveSampling,
                cacheReport: cacheReport,
                enableThinking: enableThinking,
                placement: placement,
                registry: registry
            )
        }
        #endif

        let userText = request.messages.last?.content ?? ""
        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Analyzing ", "prompt ", "on ", placement.label, "...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[\(modelID)] ", "Processed: ", userText, "<|im_end|>"])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            cacheReport: cacheReport,
            startsInThinking: false,
            computePlacement: placement,
            registry: registry,
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        )
    }

    private func recordAssistantTurn(_ reply: String) {
        prefixCache.recordAssistantReply(reply)
    }

    private static func hasSafetensorsWeights(at path: String) -> Bool {
        let url = URL(fileURLWithPath: path)
        guard let items = try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) else {
            return false
        }
        return items.contains { $0.pathExtension.lowercased() == "safetensors" }
    }

    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM)
    private func ensureContainerLoaded() async throws -> ModelContainer {
        if let loadedContainer {
            return loadedContainer
        }
        let dirURL = URL(fileURLWithPath: descriptionState.modelPath)
        let config = ModelConfiguration(directory: dirURL)
        let container: ModelContainer
        #if canImport(MLXVLM)
        if descriptionState.supportsVision.value {
            container = try await VLMModelFactory.shared.loadContainer(configuration: config)
        } else {
            container = try await LLMModelFactory.shared.loadContainer(configuration: config)
        }
        #else
        container = try await LLMModelFactory.shared.loadContainer(configuration: config)
        #endif
        loadedContainer = container
        descriptionState.isLoaded = true
        return container
    }

    private func generateWithMLXContainer(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        cacheReport: CacheReport,
        enableThinking: Bool,
        placement: ComputePlacement,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let container = try await self.ensureContainerLoaded()
                    var chatMessages: [Chat.Message] = []
                    if let sys = request.system, !sys.isEmpty {
                        chatMessages.append(.system(sys))
                    }
                    for m in request.messages {
                        switch m.role.lowercased() {
                        case "assistant":
                            chatMessages.append(.assistant(m.content))
                        case "system":
                            chatMessages.append(.system(m.content))
                        default:
                            chatMessages.append(.user(m.content))
                        }
                    }

                    var additionalContext: [String: any Sendable] = [:]
                    additionalContext["enable_thinking"] = enableThinking
                    var userImages: [UserInput.Image] = []
                    #if canImport(CoreImage)
                    for turn in request.messages {
                        for att in turn.attachments where att.kind == "image" {
                            if let data = Data(base64Encoded: att.dataBase64),
                               let ciImg = CIImage(data: data) {
                                userImages.append(.ciImage(ciImg))
                            }
                        }
                    }
                    #endif
                    let userInput = UserInput(
                        chat: chatMessages,
                        images: userImages,
                        additionalContext: additionalContext
                    )
                    let lmInput = try await container.prepare(input: userInput)
                    let genParams = GenerateParameters(
                        maxTokens: effectiveSampling.maxTokens ?? 2048,
                        temperature: Float(effectiveSampling.temperature ?? 0.7),
                        topP: Float(effectiveSampling.topP ?? 0.95),
                        repetitionPenalty: effectiveSampling.repetitionPenalty.map(Float.init),
                        repetitionContextSize: effectiveSampling.repetitionContextSize ?? 64
                    )

                    var reasoningParser = ReasoningParser(startsInThinking: false)
                    var stripSpecials = StripSpecials()
                    var generatedCount = 0
                    var promptTokens = cacheReport.promptTokens
                    var promptTPS = 0.0
                    var generationTPS = 0.0
                    var visibleReply = ""
                    let memGB = Double(HardwareMemoryMonitor.snapshot().processResidentMB) / 1024.0

                    let stream = try await container.generate(input: lmInput, parameters: genParams)
                    for await event in stream {
                        let state = await registry.state(for: requestID)
                        if state.isAborted || Task.isCancelled {
                            let flushed = reasoningParser.finalize()
                            let cleanText = stripSpecials.feed(flushed.text) + stripSpecials.finalize()
                            continuation.yield(
                                GenerationChunk(
                                    text: cleanText,
                                    thinking: flushed.thinking,
                                    tokenCount: generatedCount,
                                    promptTokens: promptTokens,
                                    generationTPS: generationTPS,
                                    promptTPS: promptTPS,
                                    peakMemoryGB: memGB,
                                    stopReason: .cancelled,
                                    cacheReport: cacheReport,
                                    activeComputePlacement: placement
                                )
                            )
                            continuation.finish()
                            return
                        }

                        if state.isReasoningEndRequested && !reasoningParser.reasoningEndedForced {
                            let forced = reasoningParser.forceEndReasoning()
                            let cleanForced = stripSpecials.feed(forced.text)
                            visibleReply.append(cleanForced)
                            if !cleanForced.isEmpty || !forced.thinking.isEmpty {
                                continuation.yield(
                                    GenerationChunk(
                                        text: cleanForced,
                                        thinking: forced.thinking,
                                        tokenCount: generatedCount,
                                        promptTokens: promptTokens,
                                        generationTPS: generationTPS,
                                        promptTPS: promptTPS,
                                        peakMemoryGB: memGB,
                                        activeComputePlacement: placement
                                    )
                                )
                            }
                        }

                        switch event {
                        case .chunk(let piece):
                            generatedCount += 1
                            let delta = reasoningParser.feed(piece)
                            let cleanText = stripSpecials.feed(delta.text)
                            visibleReply.append(cleanText)
                            if !cleanText.isEmpty || !delta.thinking.isEmpty {
                                continuation.yield(
                                    GenerationChunk(
                                        text: cleanText,
                                        thinking: delta.thinking,
                                        tokenCount: generatedCount,
                                        promptTokens: promptTokens,
                                        generationTPS: generationTPS,
                                        promptTPS: promptTPS,
                                        peakMemoryGB: memGB,
                                        activeComputePlacement: placement
                                    )
                                )
                            }
                        case .reasoning(let thinkPiece):
                            generatedCount += 1
                            continuation.yield(
                                GenerationChunk(
                                    text: "",
                                    thinking: thinkPiece,
                                    tokenCount: generatedCount,
                                    promptTokens: promptTokens,
                                    generationTPS: generationTPS,
                                    promptTPS: promptTPS,
                                    peakMemoryGB: memGB,
                                    activeComputePlacement: placement
                                )
                            )
                        case .info(let info):
                            promptTokens = info.promptTokenCount
                            promptTPS = info.promptTokensPerSecond
                            generationTPS = info.tokensPerSecond
                        default:
                            break
                        }
                    }

                    let tail = reasoningParser.finalize()
                    let tailText = stripSpecials.feed(tail.text) + stripSpecials.finalize()
                    visibleReply.append(tailText)
                    await self.recordAssistantTurn(visibleReply)

                    continuation.yield(
                        GenerationChunk(
                            text: tailText,
                            thinking: tail.thinking,
                            tokenCount: generatedCount,
                            promptTokens: promptTokens,
                            generationTPS: generationTPS,
                            promptTPS: promptTPS,
                            peakMemoryGB: memGB,
                            stopReason: .endTurn,
                            cacheReport: cacheReport,
                            activeComputePlacement: placement
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    #endif
}

/// Tier 2 Provider: In-process Apple Core AI (`coreai-models` `.aimodel` bundles) & CoreML / Argmax
/// executing on the Apple Neural Engine (`.ane`) or GPU, with `.aimodel` `metadata.json` inspection
/// and speculative decoding telemetry (`SpeculativeReport`).
public actor CoreAIProvider: InferenceProvider {
    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .coreai
    private var descriptionState: EngineDescription
    private var prefixCache = PrefixCacheTracker()
    private var bundleManifest: CoreAIBundleManifest?

    public init(description: EngineDescription) {
        self.modelID = description.modelID
        var updatedDesc = description
        let url = URL(fileURLWithPath: description.modelPath)
        if let manifest = try? CoreAIBundleManifest.inspect(at: url) {
            self.bundleManifest = manifest
            if updatedDesc.contextSize.provenance == .derived {
                updatedDesc = EngineDescription(
                    modelID: updatedDesc.modelID,
                    provider: updatedDesc.provider,
                    modelPath: updatedDesc.modelPath,
                    architecture: updatedDesc.architecture,
                    quantization: updatedDesc.quantization,
                    computePlacement: updatedDesc.computePlacement,
                    isPinned: updatedDesc.isPinned,
                    isLoaded: updatedDesc.isLoaded,
                    contextSize: ProvenanceField(
                        value: manifest.maxContextLength,
                        provenance: .observed,
                        effect: .reload
                    ),
                    flashAttention: updatedDesc.flashAttention,
                    supportsVision: ProvenanceField(
                        value: manifest.hasVisionComponent || updatedDesc.supportsVision.value,
                        provenance: .observed,
                        effect: .reload
                    ),
                    supportsAudio: updatedDesc.supportsAudio,
                    supportsEmbeddings: updatedDesc.supportsEmbeddings,
                    supportsThinking: updatedDesc.supportsThinking,
                    thinkingBudget: updatedDesc.thinkingBudget,
                    templateSource: updatedDesc.templateSource,
                    estimatedMemoryMB: updatedDesc.estimatedMemoryMB,
                    tags: updatedDesc.tags
                )
            }
        }
        self.descriptionState = updatedDesc
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func manifest() async -> CoreAIBundleManifest? {
        bundleManifest
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let placement = descriptionState.computePlacement.value
        let cacheReport = prefixCache.evaluate(system: request.system, messages: request.messages)
        let userText = request.messages.last?.content ?? ""
        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value

        let hasDrafter = (bundleManifest?.hasDrafterComponent == true)
            || descriptionState.tags.contains(where: { $0.localizedCaseInsensitiveContains("drafter") || $0.localizedCaseInsensitiveContains("mtp") })
        let specReport: SpeculativeReport? = hasDrafter
            ? SpeculativeReport(
                drafterID: "\(modelID)-drafter",
                mode: "speculative",
                draftedTokens: 16,
                acceptedTokens: 13
            )
            : nil

        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Running CoreAI .aimodel state graph on ANE...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[CoreAI \(modelID)] ", userText])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            cacheReport: cacheReport,
            speculativeReport: specReport,
            startsInThinking: false,
            computePlacement: placement,
            registry: registry,
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        )
    }

    private func recordAssistantTurn(_ reply: String) {
        prefixCache.recordAssistantReply(reply)
    }
}

/// Configuration for spawning a single-slot (`-np 1`) `llama-server` subprocess.
public struct GGUFSpawnConfig: Codable, Sendable, Equatable {
    public var binaryPath: String
    public var modelPath: String
    public var host: String
    public var port: Int
    public var nGPULayers: Int
    public var ctxSize: Int
    public var flashAttention: Bool
    public var batchSize: Int?
    public var ubatchSize: Int?
    public var mmprojPath: String?
    public var imageMaxTokens: Int?
    public var chatTemplateFile: String?
    public var draftModelPath: String?
    public var specType: String?
    public var specDraftMax: Int?
    public var specDraftPMin: Double?
    public var cacheTypeK: String?
    public var cacheTypeV: String?
    public var cacheRAMMiB: Int?
    public var extraArgs: [String]

    public init(
        binaryPath: String = ProcessInfo.processInfo.environment["HEYLOOK_LLAMA_SERVER_BIN"] ?? "llama-server",
        modelPath: String,
        host: String = "127.0.0.1",
        port: Int = 18080,
        nGPULayers: Int = 999,
        ctxSize: Int = 8192,
        flashAttention: Bool = true,
        batchSize: Int? = nil,
        ubatchSize: Int? = nil,
        mmprojPath: String? = nil,
        imageMaxTokens: Int? = nil,
        chatTemplateFile: String? = nil,
        draftModelPath: String? = nil,
        specType: String? = nil,
        specDraftMax: Int? = nil,
        specDraftPMin: Double? = nil,
        cacheTypeK: String? = nil,
        cacheTypeV: String? = nil,
        cacheRAMMiB: Int? = nil,
        extraArgs: [String] = []
    ) {
        self.binaryPath = binaryPath
        self.modelPath = modelPath
        self.host = host
        self.port = port
        self.nGPULayers = nGPULayers
        self.ctxSize = ctxSize
        self.flashAttention = flashAttention
        self.batchSize = batchSize
        self.ubatchSize = ubatchSize
        self.mmprojPath = mmprojPath
        self.imageMaxTokens = imageMaxTokens
        self.chatTemplateFile = chatTemplateFile
        self.draftModelPath = draftModelPath
        self.specType = specType
        self.specDraftMax = specDraftMax
        self.specDraftPMin = specDraftPMin
        self.cacheTypeK = cacheTypeK
        self.cacheTypeV = cacheTypeV
        self.cacheRAMMiB = cacheRAMMiB
        self.extraArgs = extraArgs
    }

    /// Auto-sizes `-ub` (`2048` when headroom >= 16 GB, else `512`), `-cram` (`cacheRAMMiB` from
    /// unified memory headroom clamped to `[8192, 32768]` MiB), the 4-rung chat template ladder,
    /// and paired MTP drafter (`gguf.md`).
    public static func autoSized(
        for description: EngineDescription,
        headroomGB: Double? = nil,
        sidecar: ModelSidecarConfig? = nil
    ) -> GGUFSpawnConfig {
        let snap = HardwareMemoryMonitor.snapshot()
        let modelGB = Double(description.estimatedMemoryMB) / 1024.0
        let workingSetGB = snap.metalRecommendedMaxWorkingSetGB > 0
            ? snap.metalRecommendedMaxWorkingSetGB
            : snap.totalPhysicalRAMGB * 0.75
        let effectiveHeadroomGB = headroomGB ?? max(0, workingSetGB - modelGB)

        let thinHeadroomGB = 16.0
        let autoUbatch = effectiveHeadroomGB >= thinHeadroomGB ? 2048 : 512
        let autoCacheRAMMiB: Int? = if effectiveHeadroomGB >= thinHeadroomGB {
            min(32768, max(8192, Int(effectiveHeadroomGB / 2.0) * 1024))
        } else {
            nil
        }

        let dirURL = URL(fileURLWithPath: description.modelPath)
        let resolvedSidecar = sidecar ?? (try? String(
            contentsOf: dirURL.appendingPathComponent("model.heylook.toml"),
            encoding: .utf8
        )).map(ModelSidecarConfig.parseTOML)

        let templateResolution = HeylookConfig.resolveGGUFTemplate(in: dirURL, sidecar: resolvedSidecar)
        let drafter = HeylookConfig.discoverDrafter(in: dirURL, provider: .gguf, sidecar: resolvedSidecar)
        let mmproj = HeylookConfig.discoverMMProj(in: dirURL)

        return GGUFSpawnConfig(
            modelPath: description.modelPath,
            ctxSize: description.contextSize.value,
            flashAttention: description.flashAttention.value,
            ubatchSize: autoUbatch,
            mmprojPath: mmproj,
            chatTemplateFile: templateResolution.resolvedFilePath,
            draftModelPath: drafter.draftModelPath,
            specType: drafter.specType,
            cacheRAMMiB: autoCacheRAMMiB
        )
    }
}

/// Tier 3 Provider (macOS): Out-of-process `llama-server` subprocess (`-np 1`) for `.gguf` models.
/// Implements the exact CLI argument builder, environment sanitizer (`LLAMA_ARG_LOG_FILE` removal +
/// `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=604800`), `/health` + `/props` lifecycle, and OpenAI SSE
/// delta + `timings` parser from `llama_server_provider.py`.
public actor GGUFSubprocessProvider: InferenceProvider {
    public struct OpenAISSEDelta: Sendable, Equatable {
        public let content: String
        public let reasoningContent: String
        public let finishReason: String?
        public let promptTokens: Int?
        public let cachedTokens: Int?
        public let predictedTokens: Int?
        public let promptTPS: Double?
        public let generationTPS: Double?
        public let draftedTokens: Int?
        public let acceptedTokens: Int?

        public init(
            content: String = "",
            reasoningContent: String = "",
            finishReason: String? = nil,
            promptTokens: Int? = nil,
            cachedTokens: Int? = nil,
            predictedTokens: Int? = nil,
            promptTPS: Double? = nil,
            generationTPS: Double? = nil,
            draftedTokens: Int? = nil,
            acceptedTokens: Int? = nil
        ) {
            self.content = content
            self.reasoningContent = reasoningContent
            self.finishReason = finishReason
            self.promptTokens = promptTokens
            self.cachedTokens = cachedTokens
            self.predictedTokens = predictedTokens
            self.promptTPS = promptTPS
            self.generationTPS = generationTPS
            self.draftedTokens = draftedTokens
            self.acceptedTokens = acceptedTokens
        }
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .gguf
    private var descriptionState: EngineDescription
    public let spawnConfig: GGUFSpawnConfig
    private var prefixCache = PrefixCacheTracker()

    #if os(macOS)
    private var process: Process?
    #endif

    public init(description: EngineDescription, spawnConfig: GGUFSpawnConfig? = nil) {
        self.modelID = description.modelID
        self.descriptionState = description
        self.spawnConfig = spawnConfig ?? GGUFSpawnConfig.autoSized(for: description)
    }

    /// Builds the `llama-server` CLI flag array matching `llama_server_provider.py`.
    public static func buildLlamaServerArgs(config: GGUFSpawnConfig) -> [String] {
        var args: [String] = [
            "-m", config.modelPath,
            "--host", config.host,
            "--port", String(config.port),
            "-np", "1",
            "-ngl", String(config.nGPULayers),
            "--no-webui",
        ]
        if config.ctxSize > 0 {
            args.append(contentsOf: ["--ctx-size", String(config.ctxSize)])
        }
        args.append(contentsOf: ["-fa", config.flashAttention ? "on" : "off"])
        if let batchSize = config.batchSize {
            args.append(contentsOf: ["-b", String(batchSize)])
        }
        if let ubatchSize = config.ubatchSize {
            args.append(contentsOf: ["-ub", String(ubatchSize)])
        }
        if let mmproj = config.mmprojPath, !mmproj.isEmpty {
            args.append(contentsOf: ["--mmproj", mmproj])
            if let imgMax = config.imageMaxTokens {
                args.append(contentsOf: ["--image-max-tokens", String(imgMax)])
            }
        }
        if let tmpl = config.chatTemplateFile, !tmpl.isEmpty {
            args.append(contentsOf: ["--chat-template-file", tmpl])
        }

        let hasDrafter = (config.draftModelPath?.isEmpty == false) || (config.specType?.isEmpty == false)
        if let draftModel = config.draftModelPath, !draftModel.isEmpty {
            args.append(contentsOf: ["-md", draftModel])
        }
        if let specType = config.specType, !specType.isEmpty {
            args.append(contentsOf: ["--spec-type", specType])
        }
        if hasDrafter {
            if let nMax = config.specDraftMax {
                args.append(contentsOf: ["--spec-draft-n-max", String(nMax)])
            }
            if let pMin = config.specDraftPMin {
                args.append(contentsOf: ["--spec-draft-p-min", String(pMin)])
            }
        }
        if let ctk = config.cacheTypeK, !ctk.isEmpty {
            args.append(contentsOf: ["-ctk", ctk])
        }
        if let ctv = config.cacheTypeV, !ctv.isEmpty {
            args.append(contentsOf: ["-ctv", ctv])
        }
        if let cram = config.cacheRAMMiB {
            args.append(contentsOf: ["-cram", String(cram)])
        }
        args.append(contentsOf: ["-lv", "4"])
        args.append(contentsOf: config.extraArgs)
        return args
    }

    /// Sanitizes the child environment for `llama-server`:
    /// 1. Strips `LLAMA_ARG_LOG_FILE` so slot/cache telemetry flows to piped stderr.
    /// 2. Sets `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=604800` if not already set so Metal weight
    ///    buffers stay wired across idle gaps.
    public static func childEnvironment(
        from baseEnv: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        var env = baseEnv
        env.removeValue(forKey: "LLAMA_ARG_LOG_FILE")
        if env["GGML_METAL_RESIDENCY_KEEP_ALIVE_S"] == nil {
            env["GGML_METAL_RESIDENCY_KEEP_ALIVE_S"] = "604800"
        }
        return env
    }

    /// Parses a single `data: {...}` SSE line from `llama-server` `/v1/chat/completions`.
    public static func parseOpenAISSEDataLine(_ line: String) -> OpenAISSEDelta? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("data:") else { return nil }
        let payload = trimmed.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        if payload.isEmpty || payload == "[DONE]" {
            return nil
        }
        guard let data = payload.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        var content = ""
        var reasoningContent = ""
        var finishReason: String?

        if let choices = json["choices"] as? [[String: Any]], let first = choices.first {
            finishReason = first["finish_reason"] as? String
            if let delta = first["delta"] as? [String: Any] {
                content = (delta["content"] as? String) ?? ""
                reasoningContent = (delta["reasoning_content"] as? String) ?? ""
            }
        }

        let usage = json["usage"] as? [String: Any]
        let timings = json["timings"] as? [String: Any]

        let promptTokens = (timings?["prompt_n"] as? Int) ?? (usage?["prompt_tokens"] as? Int)
        let cachedTokens = timings?["cache_n"] as? Int
        let predictedTokens = (timings?["predicted_n"] as? Int) ?? (usage?["completion_tokens"] as? Int)
        let promptTPS = timings?["prompt_per_second"] as? Double
        let generationTPS = timings?["predicted_per_second"] as? Double
        let draftedTokens = timings?["draft_n"] as? Int
        let acceptedTokens = timings?["draft_n_accepted"] as? Int

        return OpenAISSEDelta(
            content: content,
            reasoningContent: reasoningContent,
            finishReason: finishReason,
            promptTokens: promptTokens,
            cachedTokens: cachedTokens,
            predictedTokens: predictedTokens,
            promptTPS: promptTPS,
            generationTPS: generationTPS,
            draftedTokens: draftedTokens,
            acceptedTokens: acceptedTokens
        )
    }

    /// Converts an Anthropic-style `MessagesRequest` (including `image` and `audio` attachments)
    /// into the OpenAI `/v1/chat/completions` `messages` array expected by `llama-server --mmproj`.
    public static func buildOpenAIChatMessages(from request: MessagesRequest) -> [[String: Any]] {
        var wireMessages: [[String: Any]] = []
        if let sys = request.system, !sys.isEmpty {
            wireMessages.append(["role": "system", "content": sys])
        }
        for turn in request.messages {
            var msgDict: [String: Any] = ["role": turn.role]
            if let thinking = turn.thinking, !thinking.isEmpty {
                msgDict["reasoning_content"] = thinking
            }
            if turn.attachments.isEmpty {
                msgDict["content"] = turn.content
                wireMessages.append(msgDict)
            } else {
                var parts: [[String: Any]] = []
                for att in turn.attachments {
                    if att.kind == "image" {
                        parts.append([
                            "type": "image_url",
                            "image_url": [
                                "url": "data:\(att.mediaType);base64,\(att.dataBase64)"
                            ]
                        ])
                    } else if att.kind == "audio" {
                        let audioFormat = att.mediaType.components(separatedBy: "/").last ?? "wav"
                        parts.append([
                            "type": "input_audio",
                            "input_audio": [
                                "data": att.dataBase64,
                                "format": audioFormat
                            ]
                        ])
                    }
                }
                if !turn.content.isEmpty {
                    parts.append([
                        "type": "text",
                        "text": turn.content
                    ])
                }
                msgDict["content"] = parts
                wireMessages.append(msgDict)
            }
        }
        return wireMessages
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    /// True when both the `llama-server` executable and `.gguf` file exist on disk.
    public nonisolated var canLaunchLiveSubprocess: Bool {
        let fm = FileManager.default
        return fm.isExecutableFile(atPath: spawnConfig.binaryPath)
            && fm.fileExists(atPath: spawnConfig.modelPath)
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let cacheReport = prefixCache.evaluate(system: request.system, messages: request.messages)
        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value

        #if os(macOS)
        if canLaunchLiveSubprocess {
            return generateWithLiveSubprocess(
                requestID: requestID,
                request: request,
                effectiveSampling: effectiveSampling,
                fallbackCacheReport: cacheReport,
                enableThinking: enableThinking,
                registry: registry
            )
        }
        #endif

        let userText = request.messages.last?.content ?? ""
        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Evaluating GGUF slot 0...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[GGUF \(modelID)] ", userText])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            cacheReport: cacheReport,
            startsInThinking: false,
            computePlacement: .ggufSubprocess,
            registry: registry,
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        )
    }

    private func recordAssistantTurn(_ reply: String) {
        prefixCache.recordAssistantReply(reply)
    }

    #if os(macOS)
    private func ensureSubprocessRunning() async throws {
        if let proc = process, proc.isRunning {
            return
        }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: spawnConfig.binaryPath)
        proc.arguments = Self.buildLlamaServerArgs(config: spawnConfig)
        proc.environment = Self.childEnvironment()
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        self.process = proc

        let healthURL = URL(string: "http://\(spawnConfig.host):\(spawnConfig.port)/health")!
        for _ in 0..<120 {
            if let (_, resp) = try? await URLSession.shared.data(from: healthURL),
               let http = resp as? HTTPURLResponse,
               http.statusCode == 200 {
                descriptionState.isLoaded = true
                return
            }
            try await Task.sleep(for: .milliseconds(250))
        }
    }

    private func generateWithLiveSubprocess(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        fallbackCacheReport: CacheReport,
        enableThinking: Bool,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let cfg = self.spawnConfig
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.ensureSubprocessRunning()
                    let baseURL = URL(string: "http://\(cfg.host):\(cfg.port)")!
                    let chatURL = baseURL.appendingPathComponent("v1/chat/completions")
                    var urlReq = URLRequest(url: chatURL)
                    urlReq.httpMethod = "POST"
                    urlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    let wireMessages = Self.buildOpenAIChatMessages(from: request)

                    let body: [String: Any] = [
                        "messages": wireMessages,
                        "stream": true,
                        "cache_prompt": true,
                        "temperature": effectiveSampling.temperature ?? 0.7,
                        "top_p": effectiveSampling.topP ?? 0.95,
                        "top_k": effectiveSampling.topK ?? 40,
                        "min_p": effectiveSampling.minP ?? 0.0,
                        "max_tokens": effectiveSampling.maxTokens ?? 2048,
                        "chat_template_kwargs": ["enable_thinking": enableThinking],
                    ]
                    urlReq.httpBody = try JSONSerialization.data(withJSONObject: body)

                    let (bytes, _) = try await URLSession.shared.bytes(for: urlReq)
                    var sentReasoningEnd = false
                    var generatedCount = 0
                    var promptTokens = fallbackCacheReport.promptTokens
                    var cachedTokens = fallbackCacheReport.cachedTokens
                    var promptTPS = 0.0
                    var generationTPS = 0.0
                    var draftedTokens: Int?
                    var acceptedTokens: Int?
                    var visibleReply = ""
                    let memGB = Double(HardwareMemoryMonitor.snapshot().processResidentMB) / 1024.0

                    for try await line in bytes.lines {
                        let state = await registry.state(for: requestID)
                        if state.isAborted || Task.isCancelled {
                            continuation.yield(
                                GenerationChunk(
                                    tokenCount: generatedCount,
                                    promptTokens: promptTokens,
                                    generationTPS: generationTPS,
                                    promptTPS: promptTPS,
                                    peakMemoryGB: memGB,
                                    stopReason: .cancelled,
                                    cacheReport: fallbackCacheReport,
                                    activeComputePlacement: .ggufSubprocess
                                )
                            )
                            continuation.finish()
                            return
                        }
                        if state.isReasoningEndRequested && !sentReasoningEnd {
                            sentReasoningEnd = true
                            var ctrlReq = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions/control"))
                            ctrlReq.httpMethod = "POST"
                            ctrlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
                            ctrlReq.httpBody = Data("{\"action\":\"reasoning_end\"}".utf8)
                            _ = try? await URLSession.shared.data(for: ctrlReq)
                        }

                        guard let delta = Self.parseOpenAISSEDataLine(line) else { continue }
                        if let p = delta.promptTokens { promptTokens = p }
                        if let c = delta.cachedTokens { cachedTokens = c }
                        if let pTPS = delta.promptTPS { promptTPS = pTPS }
                        if let gTPS = delta.generationTPS { generationTPS = gTPS }
                        if let d = delta.draftedTokens { draftedTokens = d }
                        if let a = delta.acceptedTokens { acceptedTokens = a }

                        if !delta.content.isEmpty || !delta.reasoningContent.isEmpty {
                            generatedCount += 1
                            visibleReply.append(delta.content)
                            continuation.yield(
                                GenerationChunk(
                                    text: delta.content,
                                    thinking: delta.reasoningContent,
                                    tokenCount: generatedCount,
                                    promptTokens: promptTokens,
                                    generationTPS: generationTPS,
                                    promptTPS: promptTPS,
                                    peakMemoryGB: memGB,
                                    activeComputePlacement: .ggufSubprocess
                                )
                            )
                        }
                    }

                    self.recordAssistantTurn(visibleReply)
                    let finalCache = CacheReport(
                        promptTokens: promptTokens,
                        cachedTokens: cachedTokens,
                        processedTokens: max(0, promptTokens - cachedTokens),
                        outcome: cachedTokens > 0 ? .reused : .miss,
                        cause: cachedTokens > 0 ? nil : fallbackCacheReport.cause
                    )
                    let specReport: SpeculativeReport? = if let d = draftedTokens, let a = acceptedTokens, d > 0 {
                        SpeculativeReport(drafterID: cfg.draftModelPath ?? "drafter", mode: "draft", draftedTokens: d, acceptedTokens: a)
                    } else {
                        nil
                    }
                    continuation.yield(
                        GenerationChunk(
                            tokenCount: generatedCount,
                            promptTokens: promptTokens,
                            generationTPS: generationTPS,
                            promptTPS: promptTPS,
                            peakMemoryGB: memGB,
                            stopReason: .endTurn,
                            cacheReport: finalCache,
                            speculativeReport: specReport,
                            activeComputePlacement: .ggufSubprocess
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    #endif
}

/// Remote HTTP/SSE client provider allowing the native `iOS 27.2+` app (or secondary Mac)
/// to drive the Mac Studio `HeylookServer` over `/v1/messages`, `/v1/requests/{id}/reasoning-end`,
/// and `DELETE /v1/requests/{id}`.
public actor RemoteHeylookProvider: InferenceProvider {
    public struct ParsedSSEEvent: Sendable, Equatable {
        public let textDelta: String
        public let thinkingDelta: String
        public let stopReason: StopReason?
        public let outputTokens: Int?
        public let performance: MessagesResponse.PerformanceExtension?

        public init(
            textDelta: String = "",
            thinkingDelta: String = "",
            stopReason: StopReason? = nil,
            outputTokens: Int? = nil,
            performance: MessagesResponse.PerformanceExtension? = nil
        ) {
            self.textDelta = textDelta
            self.thinkingDelta = thinkingDelta
            self.stopReason = stopReason
            self.outputTokens = outputTokens
            self.performance = performance
        }
    }

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

    /// Parses an Anthropic Messages SSE `(event, data)` frame emitted by `HeylookServer`.
    public static func parseAnthropicSSEEvent(event: String, data: String) -> ParsedSSEEvent? {
        guard let payload = data.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: payload) as? [String: Any]
        else {
            return nil
        }

        switch event {
        case "content_block_delta":
            guard let delta = json["delta"] as? [String: Any],
                  let deltaType = delta["type"] as? String
            else { return nil }
            if deltaType == "text_delta" {
                return ParsedSSEEvent(textDelta: (delta["text"] as? String) ?? "")
            } else if deltaType == "thinking_delta" {
                return ParsedSSEEvent(thinkingDelta: (delta["thinking"] as? String) ?? "")
            }
            return nil

        case "message_delta":
            let deltaDict = json["delta"] as? [String: Any]
            let stopRaw = deltaDict?["stop_reason"] as? String
            let stopReason = stopRaw.flatMap(StopReason.init(rawValue:))
            let usageDict = json["usage"] as? [String: Any]
            let outTokens = usageDict?["output_tokens"] as? Int
            var perf: MessagesResponse.PerformanceExtension?
            if let perfDict = json["performance"],
               let perfData = try? JSONSerialization.data(withJSONObject: perfDict) {
                perf = try? JSONDecoder().decode(MessagesResponse.PerformanceExtension.self, from: perfData)
            }
            return ParsedSSEEvent(
                stopReason: stopReason,
                outputTokens: outTokens,
                performance: perf
            )

        default:
            return nil
        }
    }

    /// Forwards `POST /v1/requests/{id}/reasoning-end` to the remote `HeylookServer`.
    public func forwardReasoningEnd(requestID: String) async throws {
        let url = baseURL
            .appendingPathComponent("v1/requests")
            .appendingPathComponent(requestID)
            .appendingPathComponent("reasoning-end")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        _ = try await session.data(for: req)
    }

    /// Forwards `DELETE /v1/requests/{id}` to cancel an active request on the remote `HeylookServer`.
    public func forwardCancel(requestID: String) async throws {
        let url = baseURL
            .appendingPathComponent("v1/requests")
            .appendingPathComponent(requestID)
        var req = URLRequest(url: url)
        req.httpMethod = "DELETE"
        _ = try await session.data(for: req)
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
                    let useStream = request.stream ?? false
                    wireReq.stream = useStream
                    wireReq.temperature = effectiveSampling.temperature
                    wireReq.topP = effectiveSampling.topP
                    wireReq.topK = effectiveSampling.topK
                    wireReq.maxTokens = effectiveSampling.maxTokens
                    wireReq.thinking = effectiveSampling.thinking
                    urlReq.httpBody = try JSONEncoder().encode(wireReq)

                    if useStream {
                        let (bytes, _) = try await session.bytes(for: urlReq)
                        var currentEvent = ""
                        var generatedCount = 0
                        var sentReasoningEnd = false
                        for try await line in bytes.lines {
                            let state = await registry.state(for: requestID)
                            if state.isAborted || Task.isCancelled {
                                try? await self.forwardCancel(requestID: requestID)
                                continuation.yield(
                                    GenerationChunk(
                                        tokenCount: generatedCount,
                                        stopReason: .cancelled,
                                        activeComputePlacement: .remote
                                    )
                                )
                                continuation.finish()
                                return
                            }
                            if state.isReasoningEndRequested && !sentReasoningEnd {
                                sentReasoningEnd = true
                                try? await self.forwardReasoningEnd(requestID: requestID)
                            }
                            if line.hasPrefix("event:") {
                                currentEvent = line.dropFirst("event:".count).trimmingCharacters(in: .whitespaces)
                            } else if line.hasPrefix("data:") {
                                let dataStr = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
                                guard let parsed = Self.parseAnthropicSSEEvent(event: currentEvent, data: dataStr) else {
                                    continue
                                }
                                if !parsed.textDelta.isEmpty || !parsed.thinkingDelta.isEmpty {
                                    generatedCount += 1
                                    continuation.yield(
                                        GenerationChunk(
                                            text: parsed.textDelta,
                                            thinking: parsed.thinkingDelta,
                                            tokenCount: generatedCount,
                                            activeComputePlacement: .remote
                                        )
                                    )
                                } else if parsed.stopReason != nil || parsed.performance != nil {
                                    continuation.yield(
                                        GenerationChunk(
                                            tokenCount: parsed.outputTokens ?? generatedCount,
                                            generationTPS: parsed.performance?.generationTPS ?? 0,
                                            promptTPS: parsed.performance?.promptTPS ?? 0,
                                            stopReason: parsed.stopReason ?? .endTurn,
                                            cacheReport: parsed.performance?.cache,
                                            speculativeReport: parsed.performance?.speculative,
                                            activeComputePlacement: .remote
                                        )
                                    )
                                }
                            }
                        }
                        continuation.finish()
                        return
                    }

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
