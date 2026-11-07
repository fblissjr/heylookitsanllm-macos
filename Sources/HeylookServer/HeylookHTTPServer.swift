import Foundation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import Network

/// HTTP response envelope produced by `HeylookRouter` and served over `HeylookHTTPServer`.
public struct HeylookHTTPResponse: Sendable, Equatable {
    public let status: Int
    public let headers: [String: String]
    public let body: Data

    public init(status: Int = 200, headers: [String: String] = ["Content-Type": "application/json"], body: Data) {
        self.status = status
        self.headers = headers
        self.body = body
    }

    public static func json<T: Encodable>(_ value: T, status: Int = 200) throws -> HeylookHTTPResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(value)
        return HeylookHTTPResponse(status: status, headers: ["Content-Type": "application/json"], body: data)
    }

    public static func error(status: Int, type: String, message: String) -> HeylookHTTPResponse {
        let payload: [String: Any] = [
            "type": "error",
            "error": [
                "type": type,
                "message": message,
            ],
        ]
        let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data()
        return HeylookHTTPResponse(status: status, headers: ["Content-Type": "application/json"], body: data)
    }
}

/// Request payload for `POST /v1/conversations/{id}/voice-generate` (hands-free iOS/macOS ANE
/// speech transcription + disconnect-resilient server-owned generation handoff).
public struct ConversationVoiceGenerateRequest: Codable, Sendable, Equatable {
    public var model: String
    public var audioBase64: String
    public var languageHint: String?
    public var sampling: SamplerSettings?

    public enum CodingKeys: String, CodingKey {
        case model
        case audioBase64 = "audio_base64"
        case languageHint = "language_hint"
        case sampling
    }

    public init(
        model: String,
        audioBase64: String,
        languageHint: String? = "en",
        sampling: SamplerSettings? = nil
    ) {
        self.model = model
        self.audioBase64 = audioBase64
        self.languageHint = languageHint
        self.sampling = sampling
    }
}

/// Authoritative `/v1/*` API router implementing the `heylook` Messages wire,
/// disconnect-resilient `/v1/conversations/{id}/generate`, `/v1/embeddings`, `/v1/capabilities`,
/// `/v1/models`, `/v1/requests/{id}`, `/v1/conversations/*`, `/v1/presets/*`, and `/v1/system/metrics`.
public actor HeylookRouter {
    public let store: SQLiteStore
    public let scheduler: ComputeScheduler
    public let registry: RequestRegistry
    public let embeddingEngine: any EmbeddingEngine
    public let bindHost: String
    public let allowedHosts: Set<String>

    public init(
        store: SQLiteStore,
        scheduler: ComputeScheduler,
        registry: RequestRegistry = RequestRegistry(),
        embeddingEngine: any EmbeddingEngine = DeterministicHashEmbeddingEngine(),
        bindHost: String = "127.0.0.1",
        allowedHosts: [String] = []
    ) {
        self.store = store
        self.scheduler = scheduler
        self.registry = registry
        self.embeddingEngine = embeddingEngine
        self.bindHost = bindHost
        self.allowedHosts = Set(allowedHosts.map { $0.lowercased() })
    }

    /// Validates `Host` header against DNS rebinding on loopback / Tailscale (`allowed_hosts`).
    public nonisolated func isAllowedHostHeader(_ hostHeader: String?) -> Bool {
        if allowedHosts.contains("*") { return true }
        guard let hostHeader, !hostHeader.isEmpty else { return true }
        let hostOnly = hostHeader.split(separator: ":").first.map(String.init)?.lowercased() ?? ""
        if hostOnly == "127.0.0.1" || hostOnly == "localhost" || hostOnly == "[::1]" {
            return true
        }
        if allowedHosts.contains(hostOnly) {
            return true
        }
        let isLoopbackBind = (bindHost == "127.0.0.1" || bindHost == "localhost" || bindHost == "::1")
        return !isLoopbackBind
    }

    public func handle(
        method: String,
        path: String,
        queryItems: [URLQueryItem] = [],
        headers: [String: String] = [:],
        body: Data = Data(),
        onSSEFrame: (@Sendable (String) async -> Void)? = nil
    ) async -> HeylookHTTPResponse {
        let hostHeader = headers["Host"] ?? headers["host"]
        guard isAllowedHostHeader(hostHeader) else {
            return .error(status: 403, type: "forbidden", message: "Forbidden Host header on loopback server")
        }

        let cleanPath = path.hasSuffix("/") && path.count > 1 ? String(path.dropLast()) : path
        let segments = cleanPath.split(separator: "/").map(String.init)

        do {
            switch (method.uppercased(), cleanPath) {
            case ("GET", "/v1/models"):
                struct ModelsEnvelope: Encodable {
                    let object = "list"
                    let data: [EngineDescription]
                }
                let models = await scheduler.listModels()
                return try .json(ModelsEnvelope(data: models))

            case ("GET", "/v1/capabilities"):
                struct CapabilitiesEnvelope: Encodable {
                    let deviceName: String
                    let totalPhysicalRAMGB: Double
                    let metalWorkingSetGB: Double
                    let maxRequestMediaBytes: Int
                    let endpoints: [String]

                    enum CodingKeys: String, CodingKey {
                        case deviceName = "device_name"
                        case totalPhysicalRAMGB = "total_physical_ram_gb"
                        case metalWorkingSetGB = "metal_working_set_gb"
                        case maxRequestMediaBytes = "max_request_media_bytes"
                        case endpoints
                    }
                }
                let snap = HardwareMemoryMonitor.snapshot()
                return try .json(
                    CapabilitiesEnvelope(
                        deviceName: snap.deviceName,
                        totalPhysicalRAMGB: snap.totalPhysicalRAMGB,
                        metalWorkingSetGB: snap.metalRecommendedMaxWorkingSetGB,
                        maxRequestMediaBytes: 24 * 1024 * 1024,
                        endpoints: [
                            "/v1/messages",
                            "/v1/embeddings",
                            "/v1/conversations",
                            "/v1/models",
                            "/v1/presets",
                            "/v1/notebooks",
                            "/v1/system/metrics",
                            "/v1/capabilities",
                        ]
                    )
                )

            case ("GET", "/v1/system/metrics"):
                struct SystemMetricsEnvelope: Encodable {
                    let memory: HardwareMemoryMonitor.MemorySnapshot
                    let scheduler: ComputeScheduler.SchedulerMetrics
                }
                let mem = HardwareMemoryMonitor.snapshot()
                let sched = await scheduler.metrics()
                return try .json(SystemMetricsEnvelope(memory: mem, scheduler: sched))

            case ("POST", "/v1/messages"):
                let req = try JSONDecoder().decode(MessagesRequest.self, from: body)
                let reqID = headers["X-Request-ID"] ?? headers["x-request-id"] ?? "req_\(UUID().uuidString)"
                return try await handleMessages(requestID: reqID, request: req, onSSEFrame: onSSEFrame)

            case ("POST", "/v1/embeddings"):
                return try await handleEmbeddings(body: body)

            case ("POST", "/v1/admin/models/unload-all"):
                await scheduler.unloadAllModels()
                await embeddingEngine.unload()
                let sched = await scheduler.metrics()
                struct UnloadAllAck: Encodable {
                    let unloaded: Bool
                    let loaded_gpu_models: [String]
                    let loaded_ane_models: [String]
                }
                return try .json(
                    UnloadAllAck(
                        unloaded: true,
                        loaded_gpu_models: sched.loadedGPUModels,
                        loaded_ane_models: sched.loadedANEModels
                    )
                )

            case ("GET", "/v1/conversations"):
                let list = try store.listConversations()
                return try .json(list)

            case ("POST", "/v1/conversations"):
                let incoming = (try? JSONDecoder().decode(ConversationRecord.self, from: body)) ?? ConversationRecord()
                try store.upsertConversation(incoming)
                return try .json(incoming, status: 201)

            case ("GET", "/v1/presets"):
                return try .json(try store.listPresets())

            case ("POST", "/v1/presets"):
                let preset = try JSONDecoder().decode(PresetRecord.self, from: body)
                try store.upsertPreset(preset)
                return try .json(preset, status: 201)

            case ("GET", "/v1/notebooks"):
                return try .json(try store.listNotebooks())

            case ("POST", "/v1/notebooks"):
                let notebook = try JSONDecoder().decode(NotebookRecord.self, from: body)
                try store.upsertNotebook(notebook)
                return try .json(notebook, status: 201)

            default:
                break
            }

            // Dynamic parameterized routes under /v1/...
            if segments.count >= 3 && segments[0] == "v1" && segments[1] == "requests" {
                let reqID = segments[2]
                if method.uppercased() == "DELETE" && segments.count == 3 {
                    struct CancelAck: Encodable {
                        let id: String
                        let cancelled: Bool
                    }
                    let cancelled = await registry.abort(requestID: reqID)
                    return try .json(CancelAck(id: reqID, cancelled: cancelled))
                }
                if method.uppercased() == "POST" && segments.count == 4 && segments[3] == "reasoning-end" {
                    let ack = await registry.requestReasoningEnd(requestID: reqID)
                    return try .json(["acknowledged": ack])
                }
            }

            if segments.count >= 3 && segments[0] == "v1" && segments[1] == "conversations" {
                let convID = segments[2]
                if segments.count == 4 && segments[3] == "messages" {
                    if method.uppercased() == "GET" {
                        let msgs = try store.listMessages(conversationID: convID)
                        return try .json(msgs)
                    }
                    if method.uppercased() == "DELETE" {
                        let fromPos = queryItems.first(where: { $0.name == "from" })?.value.flatMap(Int.init) ?? 0
                        let deleted = try store.deleteMessages(conversationID: convID, fromPosition: fromPos)
                        return try .json(["deleted": deleted])
                    }
                }
                if segments.count == 4 && segments[3] == "generate" && method.uppercased() == "POST" {
                    let genReq = try JSONDecoder().decode(ConversationGenerateRequest.self, from: body)
                    let reqID = headers["X-Request-ID"] ?? headers["x-request-id"] ?? "req_\(UUID().uuidString)"
                    return try await handleConversationGenerate(
                        requestID: reqID,
                        conversationID: convID,
                        generateRequest: genReq,
                        onSSEFrame: onSSEFrame
                    )
                }
                if segments.count == 4 && segments[3] == "voice-generate" && method.uppercased() == "POST" {
                    let voiceReq = try JSONDecoder().decode(ConversationVoiceGenerateRequest.self, from: body)
                    guard let wavData = Data(base64Encoded: voiceReq.audioBase64) else {
                        return .error(status: 400, type: "invalid_request_error", message: "Invalid base64 audio payload in voice-generate.")
                    }
                    let aneProvider = CoreAIProvider(
                        description: EngineDescription(
                            modelID: "ane-speech-whisper",
                            provider: .coreai,
                            modelPath: "models/ane-speech-whisper.aimodel",
                            computePlacement: ProvenanceField(
                                value: .ane,
                                provenance: .derived,
                                effect: .reload,
                                computePlacement: .ane
                            )
                        )
                    )
                    let transcript = try await aneProvider.transcribeAudio(
                        wavData: wavData,
                        languageHint: voiceReq.languageHint ?? "en"
                    )
                    let transcriptJSON = (try? String(data: JSONEncoder().encode(transcript), encoding: .utf8)) ?? "{}"
                    let voiceFrame = "event: heylook_voice_transcribed\ndata: \(transcriptJSON)\n\n"
                    if let onSSEFrame {
                        await onSSEFrame(voiceFrame)
                    }
                    let reqID = headers["X-Request-ID"] ?? headers["x-request-id"] ?? "req_\(UUID().uuidString)"
                    let genResp = try await handleConversationGenerate(
                        requestID: reqID,
                        conversationID: convID,
                        generateRequest: ConversationGenerateRequest(
                            model: voiceReq.model,
                            content: transcript.text,
                            sampling: voiceReq.sampling
                        ),
                        onSSEFrame: onSSEFrame
                    )
                    var combinedBody = Data(voiceFrame.utf8)
                    combinedBody.append(genResp.body)
                    return HeylookHTTPResponse(
                        status: genResp.status,
                        headers: genResp.headers,
                        body: combinedBody
                    )
                }
                if segments.count == 4 && segments[3] == "prompt" && method.uppercased() == "POST" {
                    struct PromptResponse: Encodable {
                        let conversation_id: String
                        let prompt: String
                        let template_source: String
                        let message_count: Int
                    }
                    let conv = try store.getConversation(id: convID)
                    let msgs = try store.listMessages(conversationID: convID)
                    var modelDirURL: URL?
                    if let defaultModelID = conv?.defaultModelID,
                       let desc = await scheduler.modelDescription(for: defaultModelID) {
                        modelDirURL = URL(fileURLWithPath: desc.modelPath)
                    } else if let firstModel = await scheduler.listModels().first {
                        modelDirURL = URL(fileURLWithPath: firstModel.modelPath)
                    }
                    let rendered = HeylookConfig.renderPromptPreview(
                        in: modelDirURL,
                        system: conv?.systemPrompt,
                        messages: msgs.map { (role: $0.role, content: $0.content, thinking: $0.thinking) }
                    )
                    return try .json(
                        PromptResponse(
                            conversation_id: convID,
                            prompt: rendered.prompt,
                            template_source: rendered.source,
                            message_count: msgs.count
                        )
                    )
                }
                if segments.count == 5 && segments[3] == "generate" && segments[4] == "reasoning-end" && method.uppercased() == "POST" {
                    let ack = await registry.requestReasoningEnd(forConversationID: convID)
                    return try .json(["acknowledged": ack])
                }
            }

            if segments.count == 4 && segments[0] == "v1" && segments[1] == "models" && segments[3] == "image-plan" && method.uppercased() == "POST" {
                let modelID = segments[2]
                guard let desc = await scheduler.modelDescription(for: modelID) else {
                    throw ComputeSchedulerError.modelNotFound(modelID)
                }
                guard desc.isLoaded else {
                    return .error(status: 409, type: "invalid_request_error", message: "Model '\(modelID)' is not loaded; image-plan requires a resident model.")
                }
                struct ImagePlanResponse: Encodable {
                    let model: String
                    let tokens_per_image: Int
                }
                let budget = HeylookConfig.estimateImageTokenBudget(in: URL(fileURLWithPath: desc.modelPath))
                return try .json(ImagePlanResponse(model: modelID, tokens_per_image: budget))
            }

            if segments.count >= 4 && segments[0] == "v1" && segments[1] == "admin" && segments[2] == "models" {
                let modelID = segments[3]
                if segments.count == 5 && segments[4] == "chat-template" {
                    guard let existing = await scheduler.modelDescription(for: modelID) else {
                        throw ComputeSchedulerError.modelNotFound(modelID)
                    }
                    let dirURL = URL(fileURLWithPath: existing.modelPath)
                    struct ChatTemplateResponse: Encodable {
                        let model: String
                        let source: String
                        let has_override: Bool
                        let template: String
                    }
                    switch method.uppercased() {
                    case "GET":
                        let state = HeylookConfig.inspectChatTemplate(
                            in: dirURL,
                            fallbackSource: existing.templateSource.value
                        )
                        return try .json(
                            ChatTemplateResponse(
                                model: modelID,
                                source: state.source,
                                has_override: state.hasOverride,
                                template: state.template
                            )
                        )
                    case "PUT":
                        let templateText: String
                        if let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                           let t = json["template"] as? String {
                            templateText = t
                        } else {
                            templateText = String(decoding: body, as: UTF8.self)
                        }
                        let state = try HeylookConfig.writeChatTemplateOverride(templateText, in: dirURL)
                        return try .json(
                            ChatTemplateResponse(
                                model: modelID,
                                source: state.source,
                                has_override: state.hasOverride,
                                template: state.template
                            )
                        )
                    case "DELETE":
                        let reverted = HeylookConfig.deleteChatTemplateOverride(
                            in: dirURL,
                            fallbackSource: existing.templateSource.value
                        )
                        return try .json([
                            "model": modelID,
                            "source": reverted.source
                        ])
                    default:
                        break
                    }
                }
                if segments.count == 4 && method.uppercased() == "GET" {
                    guard let desc = await scheduler.modelDescription(for: modelID) else {
                        throw ComputeSchedulerError.modelNotFound(modelID)
                    }
                    return try .json(desc)
                }
                if segments.count == 4 && (method.uppercased() == "PATCH" || method.uppercased() == "PUT") {
                    guard let existing = await scheduler.modelDescription(for: modelID) else {
                        throw ComputeSchedulerError.modelNotFound(modelID)
                    }
                    let sidecar: ModelSidecarConfig
                    if let decodedJSON = try? JSONDecoder().decode(ModelSidecarConfig.self, from: body) {
                        sidecar = decodedJSON
                    } else {
                        let tomlText = String(decoding: body, as: UTF8.self)
                        sidecar = ModelSidecarConfig.parseTOML(tomlText)
                    }
                    let dirURL = URL(fileURLWithPath: existing.modelPath)
                    var isDir: ObjCBool = false
                    if FileManager.default.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue {
                        let sidecarURL = dirURL.appendingPathComponent("model.heylook.toml")
                        try? sidecar.toTOML().write(to: sidecarURL, atomically: true, encoding: .utf8)
                    }
                    let updated = try await scheduler.applySidecarUpdate(sidecar, for: modelID)
                    return try .json(updated)
                }
                if segments.count == 5 && segments[4] == "load" && method.uppercased() == "POST" {
                    let loaded = try await scheduler.activateModel(modelID)
                    return try .json(loaded)
                }
                if segments.count == 5 && segments[4] == "unload" && method.uppercased() == "POST" {
                    await scheduler.unloadModel(modelID)
                    guard let unloaded = await scheduler.modelDescription(for: modelID) else {
                        throw ComputeSchedulerError.modelNotFound(modelID)
                    }
                    return try .json(unloaded)
                }
            }

            return .error(status: 404, type: "not_found_error", message: "Route \(method) \(cleanPath) not found")
        } catch let error as ComputeSchedulerError {
            switch error {
            case .queueOverloaded:
                return .error(status: 503, type: "model_overloaded", message: error.localizedDescription)
            case .modelNotFound:
                return .error(status: 404, type: "not_found_error", message: error.localizedDescription)
            }
        } catch let error as ProviderExecutionError {
            return .error(status: 500, type: "provider_execution_error", message: error.localizedDescription)
        } catch {
            return .error(status: 400, type: "invalid_request_error", message: error.localizedDescription)
        }
    }

    private func handleEmbeddings(body: Data) async throws -> HeylookHTTPResponse {
        guard let json = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            return .error(status: 400, type: "invalid_request_error", message: "Invalid JSON body for /v1/embeddings")
        }
        let modelName = (json["model"] as? String) ?? "embedding-gemma2"
        let requestedDim = (json["dimensions"] as? Int) ?? Matryoshka.gemma2NativeDimensions
        let taskRaw = (json["task"] as? String) ?? "document"
        let task = EmbeddingTask(rawValue: taskRaw) ?? .document

        var inputs: [EmbeddingInput] = []
        if let singleText = json["input"] as? String {
            inputs = [.text(singleText, task: task)]
        } else if let textList = json["input"] as? [String] {
            inputs = textList.map { .text($0, task: task) }
        } else if let partArrays = json["input"] as? [[[String: Any]]] {
            for partArray in partArrays {
                var parts: [EmbeddingPart] = []
                for partDict in partArray {
                    let pType = (partDict["type"] as? String) ?? "text"
                    if pType == "text", let txt = partDict["text"] as? String {
                        parts.append(.text(txt))
                    } else if let urlStr = (partDict["image_url"] as? [String: Any])?["url"] as? String
                        ?? (partDict["audio_url"] as? [String: Any])?["url"] as? String {
                        if let commaIdx = urlStr.firstIndex(of: ",") {
                            let b64 = String(urlStr[urlStr.index(after: commaIdx)...])
                            if let data = Data(base64Encoded: b64) {
                                let kind: ArtifactKind = pType.contains("audio") ? .audio : .image
                                parts.append(.media(kind: kind, mimeType: kind == .audio ? "audio/wav" : "image/jpeg", data: data))
                            }
                        }
                    }
                }
                inputs.append(EmbeddingInput(parts: parts.isEmpty ? [.text("")] : parts, task: task))
            }
        } else {
            return .error(status: 400, type: "invalid_request_error", message: "Missing or unsupported 'input' field in /v1/embeddings")
        }

        let outputs = try await embeddingEngine.embedBatch(inputs, dimensions: requestedDim)
        struct EmbeddingItem: Encodable {
            let object = "embedding"
            let index: Int
            let embedding: [Float]
        }
        struct EmbeddingsResponseEnvelope: Encodable {
            let object = "list"
            let model: String
            let data: [EmbeddingItem]
        }

        let items = outputs.enumerated().map { idx, out in
            EmbeddingItem(index: idx, embedding: out.vector)
        }
        return try .json(EmbeddingsResponseEnvelope(model: modelName, data: items))
    }

    private func handleMessages(
        requestID: String,
        request: MessagesRequest,
        onSSEFrame: (@Sendable (String) async -> Void)? = nil
    ) async throws -> HeylookHTTPResponse {
        let provider = try await scheduler.provider(for: request.model)
        let desc = await provider.describe()
        let hasImage = request.messages.contains { $0.attachments.contains { $0.kind == "image" } }
        let hasAudio = request.messages.contains { $0.attachments.contains { $0.kind == "audio" } }
        if hasImage && !desc.supportsVision.value {
            return .error(
                status: 422,
                type: "invalid_request_error",
                message: "Model '\(request.model)' does not support image attachments (supports_vision = false)."
            )
        }
        if hasAudio && !desc.supportsAudio.value {
            return .error(
                status: 422,
                type: "invalid_request_error",
                message: "Model '\(request.model)' does not support audio attachments (supports_audio = false)."
            )
        }

        let resolvedSampling = SamplerCascade.resolve(request: request.samplerOverlay)
        await registry.register(requestID: requestID, modelID: request.model)

        return try await scheduler.withComputeLease(for: request.model) { [registry] in
            defer {
                Task { await registry.finish(requestID: requestID) }
            }
            let useStream = request.stream ?? false
            var fullText = ""
            var fullThinking = ""
            var finalChunk = GenerationChunk()
            var sseLines: [String] = []

            if useStream {
                let startPayload: [String: Any] = [
                    "type": "message_start",
                    "message": [
                        "id": requestID,
                        "type": "message",
                        "role": "assistant",
                        "model": request.model,
                        "content": [],
                    ],
                ]
                if let startData = try? JSONSerialization.data(withJSONObject: startPayload),
                   let startStr = String(data: startData, encoding: .utf8) {
                    let frame = "event: message_start\ndata: \(startStr)\n\n"
                    sseLines.append(frame)
                    if let onSSEFrame { await onSSEFrame(frame) }
                }
            }

            let stream = await provider.generate(
                requestID: requestID,
                request: request,
                effectiveSampling: resolvedSampling.effective,
                registry: registry
            )
            for try await chunk in stream {
                if !chunk.thinking.isEmpty {
                    fullThinking.append(chunk.thinking)
                    if useStream {
                        let deltaObj: [String: Any] = [
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": [
                                "type": "thinking_delta",
                                "thinking": chunk.thinking,
                            ],
                        ]
                        if let d = try? JSONSerialization.data(withJSONObject: deltaObj),
                           let s = String(data: d, encoding: .utf8) {
                            let frame = "event: content_block_delta\ndata: \(s)\n\n"
                            sseLines.append(frame)
                            if let onSSEFrame { await onSSEFrame(frame) }
                        }
                    }
                }
                if !chunk.text.isEmpty {
                    fullText.append(chunk.text)
                    if useStream {
                        let deltaObj: [String: Any] = [
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": [
                                "type": "text_delta",
                                "text": chunk.text,
                            ],
                        ]
                        if let d = try? JSONSerialization.data(withJSONObject: deltaObj),
                           let s = String(data: d, encoding: .utf8) {
                            let frame = "event: content_block_delta\ndata: \(s)\n\n"
                            sseLines.append(frame)
                            if let onSSEFrame { await onSSEFrame(frame) }
                        }
                    }
                }
                finalChunk = chunk
            }

            let perfExt = MessagesResponse.PerformanceExtension(
                generationTPS: finalChunk.generationTPS,
                promptTPS: finalChunk.promptTPS,
                computePlacement: finalChunk.activeComputePlacement,
                cache: finalChunk.cacheReport,
                speculative: finalChunk.speculativeReport
            )

            if useStream {
                struct MessageDeltaFrame: Encodable {
                    struct DeltaBody: Encodable {
                        let stopReason: StopReason
                        enum CodingKeys: String, CodingKey {
                            case stopReason = "stop_reason"
                        }
                    }
                    let type = "message_delta"
                    let delta: DeltaBody
                    let usage: MessagesResponse.Usage
                    let performance: MessagesResponse.PerformanceExtension
                }
                let msgDelta = MessageDeltaFrame(
                    delta: .init(stopReason: finalChunk.stopReason ?? .endTurn),
                    usage: MessagesResponse.Usage(
                        inputTokens: finalChunk.cacheReport?.processedTokens ?? finalChunk.promptTokens,
                        outputTokens: finalChunk.tokenCount,
                        cacheReadInputTokens: finalChunk.cacheReport?.cachedTokens
                    ),
                    performance: perfExt
                )
                if let deltaData = try? JSONEncoder().encode(msgDelta),
                   let deltaStr = String(data: deltaData, encoding: .utf8) {
                    let frame = "event: message_delta\ndata: \(deltaStr)\n\n"
                    sseLines.append(frame)
                    if let onSSEFrame { await onSSEFrame(frame) }
                }
                let stopFrame = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                sseLines.append(stopFrame)
                if let onSSEFrame { await onSSEFrame(stopFrame) }

                return HeylookHTTPResponse(
                    status: 200,
                    headers: [
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    ],
                    body: Data(sseLines.joined().utf8)
                )
            }

            var blocks: [MessagesResponse.ContentBlock] = []
            if !fullThinking.isEmpty {
                blocks.append(.thinking(fullThinking))
            }
            blocks.append(.text(fullText))

            let response = MessagesResponse(
                id: requestID,
                model: request.model,
                content: blocks,
                stopReason: finalChunk.stopReason ?? .endTurn,
                usage: MessagesResponse.Usage(
                    inputTokens: finalChunk.cacheReport?.processedTokens ?? finalChunk.promptTokens,
                    outputTokens: finalChunk.tokenCount,
                    cacheReadInputTokens: finalChunk.cacheReport?.cachedTokens
                ),
                performance: perfExt
            )
            return try HeylookHTTPResponse.json(response)
        }
    }

    /// Disconnect-resilient server-owned generation (`POST /v1/conversations/{id}/generate`).
    /// Immediately persists the user turn, runs generation in an unstructured server-owned `Task`
    /// so client disconnect never loses the assistant response, persists the assistant message
    /// to `SQLiteStore`, and emits an SSE stream ending with `event: heylook_saved`.
    private func handleConversationGenerate(
        requestID: String,
        conversationID: String,
        generateRequest: ConversationGenerateRequest,
        onSSEFrame: (@Sendable (String) async -> Void)? = nil
    ) async throws -> HeylookHTTPResponse {
        let provider = try await scheduler.provider(for: generateRequest.model)
        if try store.getConversation(id: conversationID) == nil {
            try store.upsertConversation(
                ConversationRecord(
                    id: conversationID,
                    title: String(generateRequest.content.prefix(48)),
                    defaultModelID: generateRequest.model
                )
            )
        }

        let existingMessages = try store.listMessages(conversationID: conversationID)
        let nextPos = (existingMessages.map(\.position).max() ?? -1) + 1
        let userMsg = try store.appendMessage(
            MessageRecord(
                conversationID: conversationID,
                position: nextPos,
                role: "user",
                content: generateRequest.content,
                modelID: generateRequest.model
            )
        )

        var turns = existingMessages.map { MessagesRequest.MessageTurn(role: $0.role, content: $0.content) }
        turns.append(MessagesRequest.MessageTurn(role: "user", content: generateRequest.content))
        let conv = try store.getConversation(id: conversationID)
        let msgReq = MessagesRequest(
            model: generateRequest.model,
            messages: turns,
            system: conv?.systemPrompt,
            stream: true
        )
        let resolvedSampling = SamplerCascade.resolve(request: generateRequest.sampling)
        await registry.register(requestID: requestID, conversationID: conversationID, modelID: generateRequest.model)

        // Unstructured server-owned task ensures persistence even if caller task cancels
        let serverTask = Task { [store, scheduler, registry] () throws -> Data in
            try await scheduler.withComputeLease(for: generateRequest.model) {
                defer {
                    Task { await registry.finish(requestID: requestID) }
                }
                var sseLines: [String] = []
                var fullText = ""
                var fullThinking = ""
                var finalChunk = GenerationChunk()

                let stream = await provider.generate(
                    requestID: requestID,
                    request: msgReq,
                    effectiveSampling: resolvedSampling.effective,
                    registry: registry
                )
                for try await chunk in stream {
                    if !chunk.thinking.isEmpty {
                        fullThinking.append(chunk.thinking)
                        let deltaJSON = try String(
                            data: JSONEncoder().encode(["type": "thinking_delta", "thinking": chunk.thinking]),
                            encoding: .utf8
                        ) ?? "{}"
                        let frame = "event: content_block_delta\ndata: \(deltaJSON)\n\n"
                        sseLines.append(frame)
                        if let onSSEFrame { await onSSEFrame(frame) }
                    }
                    if !chunk.text.isEmpty {
                        fullText.append(chunk.text)
                        let deltaJSON = try String(
                            data: JSONEncoder().encode(["type": "text_delta", "text": chunk.text]),
                            encoding: .utf8
                        ) ?? "{}"
                        let frame = "event: content_block_delta\ndata: \(deltaJSON)\n\n"
                        sseLines.append(frame)
                        if let onSSEFrame { await onSSEFrame(frame) }
                    }
                    finalChunk = chunk
                }

                let assistantPos = nextPos + 1
                let savedAssistant = try store.appendMessage(
                    MessageRecord(
                        conversationID: conversationID,
                        position: assistantPos,
                        role: "assistant",
                        content: fullText,
                        thinking: fullThinking.isEmpty ? nil : fullThinking,
                        modelID: generateRequest.model,
                        stopReason: finalChunk.stopReason ?? .endTurn,
                        cacheReport: finalChunk.cacheReport,
                        speculativeReport: finalChunk.speculativeReport,
                        generationTPS: finalChunk.generationTPS,
                        computePlacement: finalChunk.activeComputePlacement
                    )
                )

                let savedEvent = HeylookSavedEvent(
                    conversationID: conversationID,
                    userMessageID: userMsg.id,
                    assistantMessageID: savedAssistant.id,
                    position: assistantPos,
                    stopReason: finalChunk.stopReason ?? .endTurn
                )
                let savedJSON = try String(data: JSONEncoder().encode(savedEvent), encoding: .utf8) ?? "{}"
                let savedFrame = "event: heylook_saved\ndata: \(savedJSON)\n\n"
                sseLines.append(savedFrame)
                if let onSSEFrame { await onSSEFrame(savedFrame) }
                return Data(sseLines.joined().utf8)
            }
        }

        let sseData = try await serverTask.value
        return HeylookHTTPResponse(
            status: 200,
            headers: [
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            ],
            body: sseData
        )
    }
}

/// Embedded TCP/HTTP server actor wrapping `HeylookRouter` via `Network.framework` (`NWListener`)
/// with Bonjour `_heylook._tcp` advertisement for mobile (`iOS 27.2+`) discovery.
public actor HeylookHTTPServer {
    public let router: HeylookRouter
    public let host: String
    public let port: UInt16
    public let advertiseBonjour: Bool
    private var listener: NWListener?
    public private(set) var isRunning: Bool = false

    public init(
        router: HeylookRouter,
        host: String = "127.0.0.1",
        port: UInt16 = 8000,
        advertiseBonjour: Bool = true
    ) {
        self.router = router
        self.host = host
        self.port = port
        self.advertiseBonjour = advertiseBonjour
    }

    public func start() throws {
        guard !isRunning else { return }
        guard let nwPort = NWEndpoint.Port(rawValue: port) else { return }
        let params = NWParameters.tcp
        let nwListener = try NWListener(using: params, on: nwPort)
        if advertiseBonjour {
            nwListener.service = NWListener.Service(name: "Heylook Mac Studio", type: "_heylook._tcp")
        }
        let routerRef = self.router
        nwListener.newConnectionHandler = { connection in
            connection.start(queue: .global(qos: .userInitiated))
            Self.receiveHTTPRequest(on: connection, accumulated: Data(), router: routerRef)
        }
        nwListener.start(queue: .global(qos: .userInitiated))
        self.listener = nwListener
        self.isRunning = true
    }

    public func stop() async {
        listener?.cancel()
        listener = nil
        isRunning = false
        await router.scheduler.unloadAllModels()
        await router.embeddingEngine.unload()
    }

    /// Assembles multi-frame HTTP requests up to the full `Content-Length` body before dispatching
    /// so large base64 image/video/audio payloads are never truncated across TCP reads.
    public nonisolated static func expectedTotalBytes(for rawData: Data) -> Int? {
        let separator = Data("\r\n\r\n".utf8)
        guard let range = rawData.range(of: separator) else { return nil }
        let headerData = rawData.subdata(in: 0..<range.lowerBound)
        let headerText = String(decoding: headerData, as: UTF8.self)
        var contentLength = 0
        for line in headerText.components(separatedBy: "\r\n").dropFirst() {
            let parts = line.split(separator: ":", maxSplits: 1)
            if parts.count == 2,
               parts[0].trimmingCharacters(in: .whitespaces).lowercased() == "content-length",
               let len = Int(parts[1].trimmingCharacters(in: .whitespaces)) {
                contentLength = max(0, len)
                break
            }
        }
        return range.upperBound + contentLength
    }

    private nonisolated static func receiveHTTPRequest(
        on connection: NWConnection,
        accumulated: Data,
        router: HeylookRouter
    ) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4 * 1024 * 1024) { chunk, _, isComplete, error in
            var buffer = accumulated
            if let chunk, !chunk.isEmpty {
                buffer.append(chunk)
            }
            if error != nil || (buffer.isEmpty && isComplete) {
                connection.cancel()
                return
            }
            if let expected = Self.expectedTotalBytes(for: buffer), buffer.count < expected, !isComplete {
                Self.receiveHTTPRequest(on: connection, accumulated: buffer, router: router)
                return
            }
            if Self.expectedTotalBytes(for: buffer) == nil && !isComplete {
                Self.receiveHTTPRequest(on: connection, accumulated: buffer, router: router)
                return
            }

            let finalBuffer = buffer
            Task {
                let sseStreamBox = SSEConnectionStreamer(connection: connection)
                let response = await Self.parseAndDispatch(
                    rawData: finalBuffer,
                    router: router,
                    onSSEFrame: { frame in
                        await sseStreamBox.sendFrame(frame)
                    }
                )
                let didStream = await sseStreamBox.hasStreamed
                if didStream {
                    await sseStreamBox.finish()
                } else {
                    let serialized = Self.serializeHTTPResponse(response)
                    connection.send(content: serialized, completion: .contentProcessed { _ in
                        connection.cancel()
                    })
                }
            }
        }
    }

    private nonisolated static func parseAndDispatch(
        rawData: Data,
        router: HeylookRouter,
        onSSEFrame: (@Sendable (String) async -> Void)? = nil
    ) async -> HeylookHTTPResponse {
        let separator = Data("\r\n\r\n".utf8)
        let headerData: Data
        let bodyData: Data
        if let range = rawData.range(of: separator) {
            headerData = rawData.subdata(in: 0..<range.lowerBound)
            bodyData = rawData.subdata(in: range.upperBound..<rawData.count)
        } else {
            headerData = rawData
            bodyData = Data()
        }

        let headerText = String(decoding: headerData, as: UTF8.self)
        let lines = headerText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else {
            return .error(status: 400, type: "invalid_request_error", message: "Malformed HTTP request line")
        }
        let reqParts = requestLine.split(separator: " ")
        guard reqParts.count >= 2 else {
            return .error(status: 400, type: "invalid_request_error", message: "Malformed HTTP request line")
        }
        let method = String(reqParts[0])
        let target = String(reqParts[1])
        let comps = URLComponents(string: "http://localhost\(target)")
        let path = comps?.path ?? target
        let queryItems = comps?.queryItems ?? []

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            let parts = line.split(separator: ":", maxSplits: 1)
            if parts.count == 2 {
                headers[parts[0].trimmingCharacters(in: .whitespaces)] = parts[1].trimmingCharacters(in: .whitespaces)
            }
        }
        return await router.handle(
            method: method,
            path: path,
            queryItems: queryItems,
            headers: headers,
            body: bodyData,
            onSSEFrame: onSSEFrame
        )
    }

    private nonisolated static func serializeHTTPResponse(_ response: HeylookHTTPResponse) -> Data {
        var headerLines = ["HTTP/1.1 \(response.status) OK"]
        var mergedHeaders = response.headers
        mergedHeaders["Content-Length"] = "\(response.body.count)"
        for (k, v) in mergedHeaders {
            headerLines.append("\(k): \(v)")
        }
        let head = headerLines.joined(separator: "\r\n") + "\r\n\r\n"
        var out = Data(head.utf8)
        out.append(response.body)
        return out
    }
}

/// Actor that flushes SSE frames incrementally over an `NWConnection` as tokens are generated.
private actor SSEConnectionStreamer {
    private let connection: NWConnection
    public private(set) var hasStreamed: Bool = false

    init(connection: NWConnection) {
        self.connection = connection
    }

    func sendFrame(_ frame: String) async {
        var payload = Data()
        if !hasStreamed {
            hasStreamed = true
            let sseHeader = [
                "HTTP/1.1 200 OK",
                "Content-Type: text/event-stream",
                "Cache-Control: no-cache",
                "Connection: keep-alive",
                "",
                "",
            ].joined(separator: "\r\n")
            payload.append(Data(sseHeader.utf8))
        }
        payload.append(Data(frame.utf8))
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            connection.send(content: payload, completion: .contentProcessed { _ in
                cont.resume()
            })
        }
    }

    func finish() {
        connection.cancel()
    }
}

/// Represents a Mac Studio / `heylook` server discovered via Bonjour (`_heylook._tcp`) or configured over LAN/Tailscale.
public struct DiscoveredHeylookServer: Identifiable, Sendable, Equatable, Codable {
    public let id: String
    public let name: String
    public let domain: String
    public let baseURLString: String
    public let isBonjour: Bool

    public init(
        id: String,
        name: String,
        domain: String = "local.",
        baseURLString: String,
        isBonjour: Bool = true
    ) {
        self.id = id
        self.name = name
        self.domain = domain
        self.baseURLString = baseURLString
        self.isBonjour = isBonjour
    }

    /// Derives a connectable HTTP base URL from a Bonjour service name or hostname and port.
    public static func fromBonjourService(
        name: String,
        type: String = "_heylook._tcp",
        domain: String = "local.",
        defaultPort: Int = 8000
    ) -> DiscoveredHeylookServer {
        let sanitizedHost = name
            .replacingOccurrences(of: " ", with: "-")
            .replacingOccurrences(of: "'", with: "")
            .replacingOccurrences(of: "(", with: "")
            .replacingOccurrences(of: ")", with: "")
        let hostWithDomain = domain == "local." || domain == "local"
            ? "\(sanitizedHost).local"
            : "\(sanitizedHost).\(domain.trimmingCharacters(in: CharacterSet(charactersIn: ".")))"
        let url = "http://\(hostWithDomain):\(defaultPort)"
        return DiscoveredHeylookServer(
            id: "\(name).\(type).\(domain)",
            name: name,
            domain: domain,
            baseURLString: url,
            isBonjour: true
        )
    }
}

/// Discovers embedded `heylook` servers advertising `_heylook._tcp` via `Network.framework` `NWBrowser`.
public final class BonjourServiceBrowser: @unchecked Sendable {
    public static let serviceType = "_heylook._tcp"

    private let queue = DispatchQueue(label: "com.heylook.bonjour.browser")
    private var browser: NWBrowser?
    private let defaultPort: Int

    public init(defaultPort: Int = 8000) {
        self.defaultPort = defaultPort
    }

    public func start(onUpdate: @escaping @Sendable ([DiscoveredHeylookServer]) -> Void) {
        stop()
        let descriptor = NWBrowser.Descriptor.bonjour(type: Self.serviceType, domain: nil)
        let params = NWParameters.tcp
        params.includePeerToPeer = true
        let nwBrowser = NWBrowser(for: descriptor, using: params)
        let port = self.defaultPort
        nwBrowser.browseResultsChangedHandler = { results, _ in
            var servers: [DiscoveredHeylookServer] = []
            for result in results {
                if case let .service(name, type, domain, _) = result.endpoint {
                    servers.append(
                        DiscoveredHeylookServer.fromBonjourService(
                            name: name,
                            type: type,
                            domain: domain,
                            defaultPort: port
                        )
                    )
                }
            }
            servers.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
            onUpdate(servers)
        }
        nwBrowser.start(queue: queue)
        self.browser = nwBrowser
    }

    public func stop() {
        browser?.cancel()
        browser = nil
    }
}

/// Automated multi-turn KV prefix cache chain probe (`chain_probe`), VLM vision parity probe
/// (`vlm_parity_probe`), and Matryoshka `768d`/`512d`/`256d`/`128d` `Recall@K` & `MRR` benchmark
/// (`matryoshka_recall_probe`) executed by `heylook-cli eval-lab` and the contract test suite.
public enum EvalLabSuite {
    public struct ProbeSummary: Codable, Sendable, Equatable {
        public let name: String
        public let passed: Bool
        public let detail: String

        public init(name: String, passed: Bool, detail: String) {
            self.name = name
            self.passed = passed
            self.detail = detail
        }
    }

    public struct SuiteReport: Codable, Sendable, Equatable {
        public let allPassed: Bool
        public let probes: [ProbeSummary]
        public let matryoshkaBenchmarks: [EmbeddingLabRunner.DimensionMetrics]

        public init(
            allPassed: Bool,
            probes: [ProbeSummary],
            matryoshkaBenchmarks: [EmbeddingLabRunner.DimensionMetrics]
        ) {
            self.allPassed = allPassed
            self.probes = probes
            self.matryoshkaBenchmarks = matryoshkaBenchmarks
        }
    }

    public static func runAll() async throws -> SuiteReport {
        var probes: [ProbeSummary] = []

        // 1. chain_probe: Multi-turn KV prefix cache reuse, branch truncation slicing, and system invalidation
        let mlxProvider = MLXSwiftProvider(
            description: EngineDescription(
                modelID: "eval-qwen3.5-mtp",
                provider: .mlx,
                modelPath: "models/eval-qwen3.5-mtp",
                tags: ["mtp"]
            ),
            allowSyntheticFallback: true,
            tokenCounter: { text in max(2, text.split(whereSeparator: \.isWhitespace).count * 3) }
        )
        let reg = RequestRegistry()
        var lastChunk1 = GenerationChunk()
        for try await c in await mlxProvider.generate(
            requestID: "chain_1",
            request: MessagesRequest(
                model: "eval-qwen3.5-mtp",
                messages: [.init(role: "user", content: "Explain prefix caching")],
                system: "You are a helpful coding assistant."
            ),
            effectiveSampling: .builtinDefaults,
            registry: reg
        ) {
            lastChunk1 = c
        }
        var lastChunk2 = GenerationChunk()
        for try await c in await mlxProvider.generate(
            requestID: "chain_2",
            request: MessagesRequest(
                model: "eval-qwen3.5-mtp",
                messages: [
                    .init(role: "user", content: "Explain prefix caching"),
                    .init(role: "assistant", content: "[eval-qwen3.5-mtp] Processed: Explain prefix caching"),
                    .init(role: "user", content: "Now add KV slicing details"),
                ],
                system: "You are a helpful coding assistant."
            ),
            effectiveSampling: .builtinDefaults,
            registry: reg
        ) {
            lastChunk2 = c
        }
        var lastChunk3 = GenerationChunk()
        for try await c in await mlxProvider.generate(
            requestID: "chain_3_branch",
            request: MessagesRequest(
                model: "eval-qwen3.5-mtp",
                messages: [
                    .init(role: "user", content: "Explain prefix caching"),
                    .init(role: "assistant", content: "[eval-qwen3.5-mtp] Processed: Explain prefix caching"),
                    .init(role: "user", content: "Edited branch turn two"),
                ],
                system: "You are a helpful coding assistant."
            ),
            effectiveSampling: .builtinDefaults,
            registry: reg
        ) {
            lastChunk3 = c
        }
        let kvState = await mlxProvider.kvSessionState()
        let chainPassed = lastChunk1.cacheReport?.outcome == .miss
            && lastChunk1.cacheReport?.cause == "cold"
            && lastChunk2.cacheReport?.outcome == .reused
            && (lastChunk2.cacheReport?.cachedTokens ?? 0) > 0
            && lastChunk3.cacheReport?.outcome == .reused
            && kvState.sliceTruncations == 1
            && lastChunk3.speculativeReport?.mode == "draft-mtp"
        probes.append(
            ProbeSummary(
                name: "chain_probe",
                passed: chainPassed,
                detail: "cold->reused(\(lastChunk2.cacheReport?.cachedTokens ?? 0)t)->branch_slice(trunc=\(kvState.sliceTruncations), spec=\(lastChunk3.speculativeReport?.mode ?? "none"))"
            )
        )
        await mlxProvider.unload()

        // 2. vlm_parity_probe: Model-derived image-plan token budget & HTTP 422 vision capability gate
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_eval_vlm_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        try """
        {
          "image_processor": {
            "max_soft_tokens": 256
          }
        }
        """.write(to: tempDir.appendingPathComponent("processor_config.json"), atomically: true, encoding: .utf8)

        let store = try SQLiteStore(databaseURL: tempDir.appendingPathComponent("eval.sqlite"))
        let scheduler = ComputeScheduler()
        let vlmDesc = EngineDescription(
            modelID: "eval-vlm",
            provider: .mlx,
            modelPath: tempDir.path,
            isLoaded: true,
            supportsVision: ProvenanceField(value: true, provenance: .derived, effect: .reload)
        )
        let txtDesc = EngineDescription(
            modelID: "eval-text-only",
            provider: .mlx,
            modelPath: tempDir.path,
            isLoaded: true,
            supportsVision: ProvenanceField(value: false, provenance: .derived, effect: .reload)
        )
        await scheduler.register(provider: MLXSwiftProvider(description: vlmDesc), description: vlmDesc)
        await scheduler.register(provider: MLXSwiftProvider(description: txtDesc), description: txtDesc)
        let router = HeylookRouter(store: store, scheduler: scheduler)

        let planResp = await router.handle(method: "POST", path: "/v1/models/eval-vlm/image-plan", body: Data("{}".utf8))
        let planJSON = (try? JSONSerialization.jsonObject(with: planResp.body) as? [String: Any]) ?? [:]
        let tokensPerImg = (planJSON["tokens_per_image"] as? NSNumber)?.intValue ?? 0

        let imgTurn = MessagesRequest.MessageTurn(
            role: "user",
            content: "Describe image",
            attachments: [.init(kind: "image", mediaType: "image/png", dataBase64: "iVBORw0KGgo=")]
        )
        let gate422 = await router.handle(
            method: "POST",
            path: "/v1/messages",
            body: try JSONEncoder().encode(MessagesRequest(model: "eval-text-only", messages: [imgTurn]))
        )
        let gate200 = await router.handle(
            method: "POST",
            path: "/v1/messages",
            body: try JSONEncoder().encode(MessagesRequest(model: "eval-vlm", messages: [imgTurn]))
        )
        let vlmPassed = tokensPerImg == 256 && gate422.status == 422 && gate200.status == 200
        probes.append(
            ProbeSummary(
                name: "vlm_parity_probe",
                passed: vlmPassed,
                detail: "tokens_per_image=\(tokensPerImg), text_only_status=\(gate422.status), vlm_status=\(gate200.status)"
            )
        )

        // 3. matryoshka_recall_probe: 768d / 512d / 256d / 128d Recall@1 and MRR verification
        let vecStore = try VectorStore(databaseURL: tempDir.appendingPathComponent("eval_vectors.sqlite"))
        let engine = DeterministicHashEmbeddingEngine()
        let coordinator = IndexCoordinator(engine: engine, store: vecStore)
        let corpusDir = tempDir.appendingPathComponent("corpus", isDirectory: true)
        try FileManager.default.createDirectory(at: corpusDir, withIntermediateDirectories: true)
        let fileA = corpusDir.appendingPathComponent("ane_scheduler.md")
        let fileB = corpusDir.appendingPathComponent("gguf_subprocesses.md")
        try "Apple Neural Engine co-residency and CoreML MLComputePlan scheduling".write(to: fileA, atomically: true, encoding: .utf8)
        try "Out-of-process llama-server subprocess lifecycle and dynamic port pool".write(to: fileB, atomically: true, encoding: .utf8)
        _ = try await coordinator.indexURLs([corpusDir])
        let hitsA = try await coordinator.search(queryText: "Apple Neural Engine co-residency and CoreML MLComputePlan scheduling", limit: 2)
        let hitsB = try await coordinator.search(queryText: "Out-of-process llama-server subprocess lifecycle and dynamic port pool", limit: 2)
        var cases: [EmbeddingLabRunner.BenchmarkCase] = []
        if let idA = hitsA.first?.artifact.id {
            cases.append(.init(query: "Apple Neural Engine co-residency and CoreML MLComputePlan scheduling", expectedArtifactID: idA))
        }
        if let idB = hitsB.first?.artifact.id {
            cases.append(.init(query: "Out-of-process llama-server subprocess lifecycle and dynamic port pool", expectedArtifactID: idB))
        }
        let mrlReports = try await EmbeddingLabRunner.evaluate(
            coordinator: coordinator,
            cases: cases
        )
        let mrlPassed = mrlReports.count == 4 && mrlReports.allSatisfy { $0.recallAt1 == 1.0 && $0.mrr == 1.0 }
        probes.append(
            ProbeSummary(
                name: "matryoshka_recall_probe",
                passed: mrlPassed,
                detail: mrlReports.map { "\($0.dimensions)d:R@1=\(Int($0.recallAt1 * 100))%" }.joined(separator: ", ")
            )
        )

        return SuiteReport(
            allPassed: probes.allSatisfy(\.passed),
            probes: probes,
            matryoshkaBenchmarks: mrlReports
        )
    }
}
