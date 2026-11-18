import Foundation
import HeylookCore

/// Tier 4 Provider: Remote HTTP peer (`HeylookServer` over LAN / Tailscale) proxying `/v1/messages`,
/// `heylook_progress` SSE events, `/v1/models` `EngineContract` projection, and remote abort / reasoning-end.
public actor RemoteHeylookProvider: InferenceProvider {
    public struct ParsedSSEEvent: Sendable, Equatable {
        public let textDelta: String
        public let thinkingDelta: String
        public let stopReason: StopReason?
        public let outputTokens: Int?
        public let performance: MessagesResponse.PerformanceExtension?
        public let progressProcessed: Int?
        public let progressTotal: Int?
        public let errorMessage: String?

        public init(
            textDelta: String = "",
            thinkingDelta: String = "",
            stopReason: StopReason? = nil,
            outputTokens: Int? = nil,
            performance: MessagesResponse.PerformanceExtension? = nil,
            progressProcessed: Int? = nil,
            progressTotal: Int? = nil,
            errorMessage: String? = nil
        ) {
            self.textDelta = textDelta
            self.thinkingDelta = thinkingDelta
            self.stopReason = stopReason
            self.outputTokens = outputTokens
            self.performance = performance
            self.progressProcessed = progressProcessed
            self.progressTotal = progressTotal
            self.errorMessage = errorMessage
        }
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .remote
    public nonisolated let baseURL: URL
    private var descriptionState: EngineDescription
    private var observedContract: EngineContract?
    private let session: URLSession

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

    public func observedContractState() async -> ObservedContractState {
        ObservedContractState(
            runningContextLength: observedContract?.context.running.value ?? (descriptionState.isLoaded ? descriptionState.contextSize.value : nil),
            loadedTemplateText: nil,
            runtimeCaps: observedContract?.template.caps.value,
            speculativeInForce: observedContract?.speculative.inForce.value,
            isPrefixCacheWarm: false,
            contextUsed: nil
        )
    }

    public func preload() async throws {
        let loadURL = baseURL
            .appendingPathComponent("v1/models")
            .appendingPathComponent(modelID)
            .appendingPathComponent("load")
        var req = URLRequest(url: loadURL)
        req.httpMethod = "POST"
        let (data, resp) = try await session.data(for: req)
        if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            let msg = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Remote load HTTP \(http.statusCode)\(msg.isEmpty ? "" : ": \(msg)")"
            )
        }
        if let decodedAdmin = try? JSONDecoder().decode(AdminModelWireRow.self, from: data) {
            observedContract = decodedAdmin.engine
            descriptionState = Self.applyContract(decodedAdmin.engine, onto: descriptionState, loaded: decodedAdmin.loaded)
        } else if let decodedRow = try? JSONDecoder().decode(PublicModelWireRow.self, from: data) {
            observedContract = decodedRow.engine
            descriptionState = Self.applyContract(decodedRow.engine, onto: descriptionState, loaded: true)
        } else if let decoded = try? JSONDecoder().decode(EngineDescription.self, from: data) {
            descriptionState = decoded
        } else {
            descriptionState.isLoaded = true
        }
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

    /// Queries `GET /v1/models` on a remote `HeylookServer` peer, decoding full `PublicModelWireRow` +
    /// `EngineContract` fields when exposed, or direct `EngineDescription` envelopes.
    public static func fetchRemoteModels(
        from baseURL: URL,
        session: URLSession = .shared
    ) async throws -> [EngineDescription] {
        let url = baseURL.appendingPathComponent("v1/models")
        let (data, resp) = try await session.data(from: url)
        if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: "remote",
                detail: "Remote /v1/models returned HTTP \(http.statusCode)"
            )
        }
        struct AdminWireEnvelope: Decodable {
            let data: [AdminModelWireRow]
        }
        if let adminEnv = try? JSONDecoder().decode(AdminWireEnvelope.self, from: data) {
            return adminEnv.data.map { row in
                descriptionFromContract(
                    id: row.id,
                    baseURL: baseURL,
                    modalities: row.engine.embedding?.modalities ?? [],
                    capabilities: row.capabilities,
                    thinkingDefault: row.thinkingDefault,
                    tags: row.tags,
                    loaded: row.loaded,
                    contract: row.engine
                )
            }
        }
        struct PublicWireEnvelope: Decodable {
            let data: [PublicModelWireRow]
        }
        if let wireEnv = try? JSONDecoder().decode(PublicWireEnvelope.self, from: data) {
            return wireEnv.data.map { row in
                descriptionFromContract(
                    id: row.id,
                    baseURL: baseURL,
                    modalities: row.modalities,
                    capabilities: row.capabilities,
                    thinkingDefault: row.thinkingDefault,
                    tags: [],
                    loaded: false,
                    contract: row.engine
                )
            }
        }
        struct ModelsEnvelope: Decodable {
            let data: [EngineDescription]
        }
        if let env = try? JSONDecoder().decode(ModelsEnvelope.self, from: data) {
            return env.data
        }
        return try JSONDecoder().decode([EngineDescription].self, from: data)
    }

    private static func descriptionFromContract(
        id: String,
        baseURL: URL,
        modalities: [String],
        capabilities: [String],
        thinkingDefault: Bool,
        tags: [String],
        loaded: Bool,
        contract: EngineContract
    ) -> EngineDescription {
        let base = EngineDescription(
            modelID: id,
            provider: .remote,
            modelPath: baseURL.absoluteString,
            architecture: ProvenanceField(
                value: contract.runtime.value ?? "remote",
                provenance: contract.runtime.provenance,
                effect: .reload
            ),
            quantization: ProvenanceField(
                value: "remote",
                provenance: .observed,
                effect: .reload
            ),
            computePlacement: ProvenanceField(
                value: .remote,
                provenance: .observed,
                effect: .reload,
                computePlacement: .remote
            ),
            isPinned: false,
            isLoaded: loaded,
            contextSize: ProvenanceField(
                value: contract.context.running.value ?? contract.context.length.value ?? 8192,
                provenance: contract.context.length.provenance,
                effect: .reload
            ),
            flashAttention: ProvenanceField(value: false, provenance: .notApplicable, effect: .reload),
            supportsVision: ProvenanceField(
                value: modalities.contains("vision") || capabilities.contains("vision"),
                provenance: .observed,
                effect: .reload
            ),
            supportsAudio: ProvenanceField(
                value: modalities.contains("audio") || capabilities.contains("audio"),
                provenance: .observed,
                effect: .reload
            ),
            supportsEmbeddings: ProvenanceField(
                value: contract.embedding != nil || capabilities.contains("embeddings"),
                provenance: .observed,
                effect: .reload
            ),
            supportsThinking: ProvenanceField(
                value: contract.thinking != nil || thinkingDefault || capabilities.contains("thinking"),
                provenance: .observed,
                effect: .perRequest
            ),
            thinkingBudget: ProvenanceField(value: nil, provenance: .observed, effect: .perRequest),
            reasoningFormat: ProvenanceField(
                value: contract.thinking != nil ? "tag_pair" : "none",
                provenance: .observed,
                effect: .reload
            ),
            thinkingControls: nil,
            templateSource: ProvenanceField(
                value: contract.template.origin.value ?? "remote",
                provenance: contract.template.origin.provenance,
                effect: .reload
            ),
            speculativeDrafter: ProvenanceField(
                value: contract.speculative.drafter.value,
                provenance: contract.speculative.drafter.provenance,
                effect: .reload
            ),
            estimatedMemoryMB: 0,
            tags: tags
        )
        return applyContract(contract, onto: base, loaded: loaded)
    }

    private static func applyContract(
        _ contract: EngineContract?,
        onto base: EngineDescription,
        loaded: Bool
    ) -> EngineDescription {
        guard let contract else {
            var copy = base
            copy.isLoaded = loaded
            return copy
        }
        return EngineDescription(
            modelID: base.modelID,
            provider: base.provider,
            modelPath: base.modelPath,
            architecture: ProvenanceField(
                value: contract.runtime.value ?? base.architecture.value,
                provenance: contract.runtime.provenance,
                effect: .reload
            ),
            quantization: base.quantization,
            computePlacement: base.computePlacement,
            isPinned: base.isPinned,
            isLoaded: loaded,
            contextSize: ProvenanceField(
                value: contract.context.running.value ?? contract.context.length.value ?? base.contextSize.value,
                provenance: contract.context.length.provenance,
                effect: .reload
            ),
            flashAttention: base.flashAttention,
            supportsVision: base.supportsVision,
            supportsAudio: base.supportsAudio,
            supportsEmbeddings: ProvenanceField(
                value: contract.embedding != nil || base.supportsEmbeddings.value,
                provenance: .observed,
                effect: .reload
            ),
            supportsThinking: ProvenanceField(
                value: contract.thinking != nil || base.supportsThinking.value,
                provenance: contract.thinking != nil ? .observed : base.supportsThinking.provenance,
                effect: .perRequest
            ),
            thinkingBudget: base.thinkingBudget,
            reasoningFormat: base.reasoningFormat,
            thinkingControls: base.thinkingControls,
            templateSource: ProvenanceField(
                value: contract.template.origin.value ?? base.templateSource.value,
                provenance: contract.template.origin.provenance,
                effect: .reload
            ),
            speculativeDrafter: ProvenanceField(
                value: contract.speculative.drafter.value ?? base.speculativeDrafter.value,
                provenance: contract.speculative.drafter.provenance,
                effect: .reload
            ),
            estimatedMemoryMB: base.estimatedMemoryMB,
            tags: base.tags
        )
    }

    /// Parses an Anthropic Messages SSE `(event, data)` frame (plus `heylook_progress`) emitted by `HeylookServer`.
    public static func parseAnthropicSSEEvent(event: String, data: String) -> ParsedSSEEvent? {
        guard let payload = data.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: payload) as? [String: Any]
        else {
            return nil
        }

        switch event {
        case "error":
            if let errDict = json["error"] as? [String: Any],
               let msg = errDict["message"] as? String {
                return ParsedSSEEvent(errorMessage: msg)
            }
            return ParsedSSEEvent(errorMessage: (json["message"] as? String) ?? "Remote SSE stream error")

        case "heylook_progress":
            if let prefill = json["prefill"] as? [String: Any],
               let processed = prefill["processed"] as? Int,
               let total = prefill["total"] as? Int,
               total > 0 {
                return ParsedSSEEvent(progressProcessed: max(0, processed), progressTotal: total)
            }
            if let processed = json["processed"] as? Int,
               let total = json["total"] as? Int,
               total > 0 {
                return ParsedSSEEvent(progressProcessed: max(0, processed), progressTotal: total)
            }
            return nil

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

        case "message_stop":
            guard let perfDict = json["performance"],
                  let perfData = try? JSONSerialization.data(withJSONObject: perfDict),
                  let perf = try? JSONDecoder().decode(MessagesResponse.PerformanceExtension.self, from: perfData)
            else {
                return nil
            }
            return ParsedSSEEvent(performance: perf)

        default:
            return nil
        }
    }

    public func forwardReasoningEnd(requestID: String) async throws {
        let url = baseURL
            .appendingPathComponent("v1/requests")
            .appendingPathComponent(requestID)
            .appendingPathComponent("reasoning-end")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        _ = try await session.data(for: req)
    }

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
        let currentModelID = self.modelID
        let progressCallback = PrefillProgressBridge.current

        return AsyncThrowingStream { continuation in
            let workTask = Task {
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
                    wireReq.thinking = effectiveSampling.thinking ?? request.thinking
                    wireReq.thinkingBudget = effectiveSampling.thinkingBudget ?? request.thinkingBudget
                    wireReq.thinkingDepth = effectiveSampling.thinkingDepth ?? request.thinkingDepth
                    wireReq.stopSequences = effectiveSampling.stopSequences ?? request.stopSequences
                    wireReq.responseFormat = request.responseFormat
                    urlReq.httpBody = try JSONEncoder().encode(wireReq)

                    if useStream {
                        let (bytes, resp) = try await session.bytes(for: urlReq)
                        if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                            var errBytes = Data()
                            for try await b in bytes {
                                errBytes.append(b)
                                if errBytes.count >= 4096 { break }
                            }
                            let msg = String(decoding: errBytes, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
                            throw ProviderExecutionError.runtimeUnavailable(
                                modelID: currentModelID,
                                detail: "Remote HTTP \(http.statusCode)\(msg.isEmpty ? "" : ": \(msg)")"
                            )
                        }
                        var currentEvent = ""
                        var generatedCount = 0
                        var sentReasoningEnd = false
                        var deferredStopReason: StopReason?
                        var deferredOutputTokens: Int?
                        var emittedTerminalTelemetry = false
                        for try await line in bytes.lines {
                            let state = registry.state(for: requestID)
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
                                if let errMsg = parsed.errorMessage {
                                    throw ProviderExecutionError.runtimeUnavailable(
                                        modelID: currentModelID,
                                        detail: errMsg
                                    )
                                }
                                if let p = parsed.progressProcessed, let t = parsed.progressTotal, t > 0 {
                                    if let progressCallback {
                                        await progressCallback(p, t)
                                    }
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
                                } else if let perf = parsed.performance {
                                    emittedTerminalTelemetry = true
                                    continuation.yield(
                                        GenerationChunk(
                                            tokenCount: parsed.outputTokens ?? deferredOutputTokens ?? generatedCount,
                                            generationTPS: perf.generationTPS ?? 0,
                                            promptTPS: perf.promptTPS ?? 0,
                                            stopReason: parsed.stopReason ?? deferredStopReason ?? .endTurn,
                                            cacheReport: perf.cache,
                                            speculativeReport: perf.speculative,
                                            activeComputePlacement: .remote
                                        )
                                    )
                                } else if let stopReason = parsed.stopReason {
                                    deferredStopReason = stopReason
                                    if let outTok = parsed.outputTokens {
                                        deferredOutputTokens = outTok
                                    }
                                }
                            }
                        }
                        if !emittedTerminalTelemetry, let stopReason = deferredStopReason {
                            continuation.yield(
                                GenerationChunk(
                                    tokenCount: deferredOutputTokens ?? generatedCount,
                                    stopReason: stopReason,
                                    activeComputePlacement: .remote
                                )
                            )
                        }
                        continuation.finish()
                        return
                    }

                    let (data, resp) = try await session.data(for: urlReq)
                    if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                        let msg = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
                        throw ProviderExecutionError.runtimeUnavailable(
                            modelID: currentModelID,
                            detail: "Remote HTTP \(http.statusCode)\(msg.isEmpty ? "" : ": \(msg)")"
                        )
                    }
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
                } catch is CancellationError {
                    try? await self.forwardCancel(requestID: requestID)
                    continuation.yield(GenerationChunk(stopReason: .cancelled, activeComputePlacement: .remote))
                    continuation.finish()
                } catch let urlErr as URLError where urlErr.code == .cancelled {
                    try? await self.forwardCancel(requestID: requestID)
                    continuation.yield(GenerationChunk(stopReason: .cancelled, activeComputePlacement: .remote))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            registry.addCancelHandler(for: requestID) {
                workTask.cancel()
            }
            continuation.onTermination = { _ in
                workTask.cancel()
            }
        }
    }
}
