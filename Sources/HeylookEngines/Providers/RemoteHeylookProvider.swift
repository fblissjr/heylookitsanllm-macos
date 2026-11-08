import Foundation
import HeylookCore

/// Remote HTTP/SSE client provider allowing the native `iOS 27.2+` app (or secondary Mac)
/// to stream inference over LAN/Tailscale from a Mac Studio `HeylookServer` via `/v1/messages`,
/// `/v1/conversations/{id}/generate`, `POST /v1/requests/{id}/reasoning-end`,
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

    public func unload() async {
        descriptionState.isLoaded = false
        let unloadURL = baseURL
            .appendingPathComponent("v1/admin/models")
            .appendingPathComponent(modelID)
            .appendingPathComponent("unload")
        var req = URLRequest(url: unloadURL)
        req.httpMethod = "POST"
        _ = try? await session.data(for: req)
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
                    wireReq.minP = effectiveSampling.minP
                    wireReq.maxTokens = effectiveSampling.maxTokens
                    wireReq.repetitionPenalty = effectiveSampling.repetitionPenalty
                    wireReq.repetitionContextSize = effectiveSampling.repetitionContextSize
                    wireReq.presencePenalty = effectiveSampling.presencePenalty
                    wireReq.seed = effectiveSampling.seed
                    wireReq.thinking = effectiveSampling.thinking
                    wireReq.thinkingBudget = effectiveSampling.thinkingBudget
                    wireReq.thinkingDepth = effectiveSampling.thinkingDepth ?? request.thinkingDepth
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
