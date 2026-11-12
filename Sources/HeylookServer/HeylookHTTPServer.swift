import Foundation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import Network

/// HTTP response envelope produced by `HeylookRouter` and served over `HeylookHTTPServer`.
public struct HeylookHTTPResponse: Sendable, Equatable {
    public let status: Int
    public var headers: [String: String]
    public let body: Data

    public init(status: Int = 200, headers: [String: String] = ["Content-Type": "application/json"], body: Data) {
        self.status = status
        self.headers = headers
        self.body = body
    }

    public static func statusReasonPhrase(for status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 201: return "Created"
        case 202: return "Accepted"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 409: return "Conflict"
        case 413: return "Payload Too Large"
        case 422: return "Unprocessable Entity"
        case 429: return "Too Many Requests"
        case 500: return "Internal Server Error"
        case 502: return "Bad Gateway"
        case 503: return "Service Unavailable"
        default: return status >= 500 ? "Server Error" : (status >= 400 ? "Client Error" : "OK")
        }
    }

    public static func json<T: Encodable>(
        _ value: T,
        status: Int = 200,
        extraHeaders: [String: String] = [:]
    ) throws -> HeylookHTTPResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(value)
        var hdrs = ["Content-Type": "application/json"]
        for (k, v) in extraHeaders { hdrs[k] = v }
        return HeylookHTTPResponse(status: status, headers: hdrs, body: data)
    }

    public static func jsonObject(
        _ object: Any,
        status: Int = 200,
        extraHeaders: [String: String] = [:]
    ) throws -> HeylookHTTPResponse {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        var hdrs = ["Content-Type": "application/json"]
        for (k, v) in extraHeaders { hdrs[k] = v }
        return HeylookHTTPResponse(status: status, headers: hdrs, body: data)
    }

    public static func error(
        status: Int,
        type: String,
        message: String,
        extraHeaders: [String: String] = [:]
    ) -> HeylookHTTPResponse {
        let payload: [String: Any] = [
            "type": "error",
            "error": [
                "type": type,
                "message": message,
            ],
        ]
        let data = (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])) ?? Data()
        var hdrs = ["Content-Type": "application/json"]
        for (k, v) in extraHeaders { hdrs[k] = v }
        return HeylookHTTPResponse(status: status, headers: hdrs, body: data)
    }
}

/// Request payload for `POST /v1/conversations/{id}/voice-generate` (hands-free iOS/macOS ANE
/// speech transcription + disconnect-resilient server-owned generation handoff).
public struct ConversationVoiceGenerateRequest: Codable, Sendable, Equatable {
    public var model: String
    public var audioBase64: String
    public var languageHint: String?
    public var asrModel: String?
    public var sampling: SamplerSettings?

    public enum CodingKeys: String, CodingKey {
        case model
        case audioBase64 = "audio_base64"
        case languageHint = "language_hint"
        case asrModel = "asr_model"
        case sampling
    }

    public init(
        model: String,
        audioBase64: String,
        languageHint: String? = "en",
        asrModel: String? = nil,
        sampling: SamplerSettings? = nil
    ) {
        self.model = model
        self.audioBase64 = audioBase64
        self.languageHint = languageHint
        self.asrModel = asrModel
        self.sampling = sampling
    }
}

/// Request payload for `POST /v1/audio/transcriptions` (standalone ANE speech transcription).
public struct AudioTranscriptionRequest: Codable, Sendable, Equatable {
    public var audioBase64: String
    public var languageHint: String?
    public var model: String?

    public enum CodingKeys: String, CodingKey {
        case audioBase64 = "audio_base64"
        case languageHint = "language_hint"
        case model
    }

    public init(
        audioBase64: String,
        languageHint: String? = "en",
        model: String? = nil
    ) {
        self.audioBase64 = audioBase64
        self.languageHint = languageHint
        self.model = model
    }
}

/// Authoritative `/v1/*` API router implementing the `heylook` Messages wire,
/// stateful `ConversationGenerationCoordinator` (`append` / `regenerate` / `continue` + `media_blobs` + `409` lock),
/// `/v1/embeddings`, `/v1/capabilities`, `/v1/models`, `/v1/admin/*`, `/v1/conversations/*`,
/// `/v1/notebooks/*`, `/v1/presets/*`, `/v1/performance/profile/*`, and `/v1/system/metrics`.
public actor HeylookRouter {
    public let store: SQLiteStore
    public let scheduler: ComputeScheduler
    public let registry: RequestRegistry
    public private(set) var embeddingEngine: any EmbeddingEngine
    public let bindHost: String
    public let allowedHosts: Set<String>
    public let adminToken: String?
    public let isReadOnlyConfig: Bool
    public let allowSyntheticFallback: Bool
    private let startedAtUnixMs: Int64

    public init(
        store: SQLiteStore,
        scheduler: ComputeScheduler,
        registry: RequestRegistry = .shared,
        embeddingEngine: (any EmbeddingEngine)? = nil,
        bindHost: String = "127.0.0.1",
        allowedHosts: [String] = [],
        adminToken: String? = ProcessInfo.processInfo.environment["HEYLOOK_ADMIN_TOKEN"],
        isReadOnlyConfig: Bool = HeylookConfig.isModelConfigReadOnly,
        allowSyntheticFallback: Bool = false
    ) {
        self.store = store
        self.scheduler = scheduler
        self.registry = registry
        self.embeddingEngine = embeddingEngine ?? NativeMLXEmbeddingGemma2.makeBestAvailableEngine(
            computeOptions: .hybridANETowers,
            allowSyntheticFallback: allowSyntheticFallback
        )
        self.bindHost = bindHost
        self.allowedHosts = Set(allowedHosts.map { $0.lowercased() })
        let trimmedToken = adminToken?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.adminToken = (trimmedToken?.isEmpty == false) ? trimmedToken : nil
        self.isReadOnlyConfig = isReadOnlyConfig
        self.allowSyntheticFallback = allowSyntheticFallback
        self.startedAtUnixMs = Int64(Date().timeIntervalSince1970 * 1000)
    }

    public func setEmbeddingEngine(_ engine: any EmbeddingEngine) {
        self.embeddingEngine = engine
    }

    /// Validates `Host` header against DNS rebinding on BOTH loopback (`127.0.0.1`) AND wildcard (`0.0.0.0` / `::`).
    /// Arbitrary DNS hostnames (e.g. `evil.example`) are ALWAYS rejected unless explicitly listed in `allowedHosts`.
    public nonisolated func isAllowedHostHeader(_ hostHeader: String?) -> Bool {
        if allowedHosts.contains("*") { return true }
        guard let raw = hostHeader?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return true
        }

        let hostOnly: String
        if raw.hasPrefix("[") {
            if let closeBracket = raw.firstIndex(of: "]") {
                hostOnly = String(raw[...closeBracket]).lowercased()
            } else {
                return false
            }
        } else {
            let colonCount = raw.filter { $0 == ":" }.count
            if colonCount > 1 {
                hostOnly = "[\(raw.lowercased())]"
            } else {
                hostOnly = raw.split(separator: ":", maxSplits: 1).first.map(String.init)?.lowercased() ?? ""
            }
        }
        guard !hostOnly.isEmpty else { return false }

        if hostOnly == "127.0.0.1" || hostOnly == "localhost" || hostOnly == "[::1]" || hostOnly == "::1" {
            return true
        }
        let unbracketed = hostOnly.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        if allowedHosts.contains(hostOnly) || allowedHosts.contains(unbracketed) {
            return true
        }

        let isLoopbackBind = (bindHost == "127.0.0.1" || bindHost == "localhost" || bindHost == "::1" || bindHost == "[::1]")
        if isLoopbackBind {
            return false
        }

        // Non-loopback bind (`0.0.0.0`, `::`, LAN IP, Tailscale IP): allow local mDNS `.local`,
        // machine hostname, and raw IPv4/IPv6 literals, while rejecting external DNS rebinding domains!
        if hostOnly.hasSuffix(".local") {
            return true
        }
        let localHost = ProcessInfo.processInfo.hostName.lowercased()
        if !localHost.isEmpty && (hostOnly == localHost || hostOnly == "\(localHost).local") {
            return true
        }
        if Self.isIPv4Literal(unbracketed) || Self.isIPv6Literal(unbracketed) {
            return true
        }
        return false
    }

    private nonisolated static func isIPv4Literal(_ s: String) -> Bool {
        let parts = s.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return false }
        for p in parts {
            guard let octet = Int(p), (0...255).contains(octet), String(octet) == p else {
                return false
            }
        }
        return true
    }

    private nonisolated static func isIPv6Literal(_ s: String) -> Bool {
        guard s.contains(":") else { return false }
        return s.unicodeScalars.allSatisfy { scalar in
            CharacterSet(charactersIn: "0123456789abcdefABCDEF:.").contains(scalar)
        }
    }

    public struct RequestIDValidationError: Error, Sendable, Equatable {
        public let message: String
        public init(_ message: String) { self.message = message }
    }

    /// Validates incoming `X-Request-ID` (`1...128` ASCII `[A-Za-z0-9._:-]`).
    public nonisolated static func validateOrGenerateRequestID(from headers: [String: String]) -> Result<String, RequestIDValidationError> {
        let incoming = headers["X-Request-ID"] ?? headers["x-request-id"]
        guard let raw = incoming else {
            return .success("req_\(UUID().uuidString)")
        }
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, trimmed.count <= 128 else {
            return .failure(RequestIDValidationError("X-Request-ID must be between 1 and 128 characters"))
        }
        let validScalars = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-")
        guard trimmed.unicodeScalars.allSatisfy({ validScalars.contains($0) }) else {
            return .failure(RequestIDValidationError("X-Request-ID contains invalid characters (allowed: A-Z, a-z, 0-9, ., _, :, -)"))
        }
        return .success(trimmed)
    }

    private nonisolated func checkAdminToken(headers: [String: String]) -> HeylookHTTPResponse? {
        guard let expected = adminToken, !expected.isEmpty else { return nil }
        let providedHeader = headers["X-Heylook-Admin-Token"] ?? headers["x-heylook-admin-token"]
        let bearer: String? = {
            guard let auth = headers["Authorization"] ?? headers["authorization"] else { return nil }
            let prefix = "bearer "
            if auth.lowercased().hasPrefix(prefix) {
                return String(auth.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
            }
            return nil
        }()
        guard let candidate = providedHeader ?? bearer, !candidate.isEmpty else {
            return .error(status: 401, type: "authentication_error", message: "Missing X-Heylook-Admin-Token header")
        }
        guard candidate == expected else {
            return .error(status: 403, type: "permission_error", message: "Invalid X-Heylook-Admin-Token")
        }
        return nil
    }

    private nonisolated func checkWritableConfig() -> HeylookHTTPResponse? {
        if isReadOnlyConfig || HeylookConfig.isModelConfigReadOnly {
            return .error(
                status: 409,
                type: "read_only_config",
                message: "Configuration mutations are disabled (HEYLOOK_READ_ONLY_MODEL_CONFIG=1)"
            )
        }
        return nil
    }

    public func handle(
        method: String,
        path: String,
        queryItems: [URLQueryItem] = [],
        headers: [String: String] = [:],
        body: Data = Data(),
        onSSEFrame: (@Sendable (String) async -> Void)? = nil
    ) async -> HeylookHTTPResponse {
        let reqIDResult = Self.validateOrGenerateRequestID(from: headers)
        let reqID: String
        switch reqIDResult {
        case .success(let id):
            reqID = id
        case .failure(let err):
            return .error(status: 422, type: "invalid_request_error", message: err.message)
        }

        func finalize(_ resp: HeylookHTTPResponse) -> HeylookHTTPResponse {
            var out = resp
            out.headers["X-Request-ID"] = reqID
            return out
        }

        let hostHeader = headers["Host"] ?? headers["host"]
        guard isAllowedHostHeader(hostHeader) else {
            return finalize(.error(status: 403, type: "forbidden", message: "Forbidden Host header"))
        }

        if method.uppercased() == "OPTIONS" {
            return finalize(
                HeylookHTTPResponse(
                    status: 204,
                    headers: [
                        "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Heylook-Admin-Token, X-Request-ID",
                    ],
                    body: Data()
                )
            )
        }

        let cleanPath = path.hasSuffix("/") && path.count > 1 ? String(path.dropLast()) : path
        let segments = cleanPath.split(separator: "/").map(String.init)
        let upperMethod = method.uppercased()

        // Admin token gate on /v1/admin/*, /v1/data/clear, and /v1/cache/clear
        if cleanPath.hasPrefix("/v1/admin") || cleanPath == "/v1/data/clear" || cleanPath == "/v1/cache/clear" {
            if let authErr = checkAdminToken(headers: headers) {
                return finalize(authErr)
            }
        }

        do {
            let response = try await routeValidated(
                method: upperMethod,
                cleanPath: cleanPath,
                segments: segments,
                queryItems: queryItems,
                headers: headers,
                body: body,
                requestID: reqID,
                onSSEFrame: onSSEFrame
            )
            return finalize(response)
        } catch let error as ComputeSchedulerError {
            switch error {
            case .queueOverloaded:
                return finalize(
                    .error(
                        status: 503,
                        type: "model_overloaded",
                        message: error.localizedDescription,
                        extraHeaders: ["Retry-After": "2"]
                    )
                )
            case .modelNotFound:
                return finalize(.error(status: 404, type: "not_found_error", message: error.localizedDescription))
            }
        } catch let error as ProviderExecutionError {
            switch error {
            case let .invalidRequest(detail):
                return finalize(.error(status: 400, type: "invalid_request_error", message: detail))
            default:
                return finalize(.error(status: 500, type: "provider_execution_error", message: error.localizedDescription))
            }
        } catch let error as EmbeddingEngineError {
            switch error {
            case .runtimeUnavailable:
                return finalize(.error(status: 503, type: "runtime_unavailable", message: error.localizedDescription))
            default:
                return finalize(.error(status: 422, type: "invalid_request_error", message: error.localizedDescription))
            }
        } catch let error as JinjaTemplateError {
            return finalize(.error(status: 422, type: "template_syntax_error", message: error.localizedDescription))
        } catch let error as SQLiteStoreError {
            switch error {
            case .conflict(let detail):
                return finalize(.error(status: 409, type: "conflict_error", message: detail))
            case .notFound(let detail):
                return finalize(.error(status: 404, type: "not_found_error", message: detail))
            default:
                return finalize(.error(status: 500, type: "store_error", message: error.localizedDescription))
            }
        } catch is DecodingError {
            return finalize(.error(status: 400, type: "invalid_request_error", message: "Malformed JSON request payload"))
        } catch {
            return finalize(.error(status: 500, type: "internal_server_error", message: error.localizedDescription))
        }
    }

    private func routeValidated(
        method: String,
        cleanPath: String,
        segments: [String],
        queryItems: [URLQueryItem],
        headers: [String: String],
        body: Data,
        requestID: String,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> HeylookHTTPResponse {
        switch (method, cleanPath) {
        case ("GET", "/health"), ("GET", "/v1/health"), ("GET", "/v1/status"):
            let sched = await scheduler.metrics()
            let nowMs = Int64(Date().timeIntervalSince1970 * 1000)
            let payload: [String: Any] = [
                "status": "ok",
                "uptime_ms": max(0, nowMs - startedAtUnixMs),
                "loaded_gpu_models": sched.loadedGPUModels,
                "loaded_ane_models": sched.loadedANEModels,
                "gpu_queue_depth": sched.gpuQueueDepth,
                "embedding_engine": embeddingEngine.modelID,
            ]
            return try .jsonObject(payload)

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
                let embeddingLimits: EmbeddingRequestLimits
                let voiceASR: ComputeScheduler.VoiceASRSummary
                let readOnlyModelConfig: Bool
                let endpoints: [String]

                enum CodingKeys: String, CodingKey {
                    case deviceName = "device_name"
                    case totalPhysicalRAMGB = "total_physical_ram_gb"
                    case metalWorkingSetGB = "metal_working_set_gb"
                    case maxRequestMediaBytes = "max_request_media_bytes"
                    case embeddingLimits = "embedding_limits"
                    case voiceASR = "voice_asr"
                    case readOnlyModelConfig = "read_only_model_config"
                    case endpoints
                }
            }
            let snap = HardwareMemoryMonitor.snapshot()
            let asrSummary = await scheduler.voiceASRSummary(allowSyntheticFallback: allowSyntheticFallback)
            return try .json(
                CapabilitiesEnvelope(
                    deviceName: snap.deviceName,
                    totalPhysicalRAMGB: snap.totalPhysicalRAMGB,
                    metalWorkingSetGB: snap.metalRecommendedMaxWorkingSetGB,
                    maxRequestMediaBytes: EmbeddingRequestLimits.maxRequestMediaBytes,
                    embeddingLimits: .default,
                    voiceASR: asrSummary,
                    readOnlyModelConfig: isReadOnlyConfig || HeylookConfig.isModelConfigReadOnly,
                    endpoints: [
                        "/v1/messages",
                        "/v1/embeddings",
                        "/v1/audio/transcriptions",
                        "/v1/conversations",
                        "/v1/models",
                        "/v1/presets",
                        "/v1/notebooks",
                        "/v1/system/metrics",
                        "/v1/performance/profile",
                        "/v1/capabilities",
                        "/v1/status",
                    ]
                )
            )

        case ("GET", "/v1/system/metrics"):
            struct SystemMetricsEnvelope: Encodable {
                let memory: HardwareMemoryMonitor.MemorySnapshot
                let scheduler: ComputeScheduler.SchedulerMetrics
                let activeRequestCount: Int

                enum CodingKeys: String, CodingKey {
                    case memory
                    case scheduler
                    case activeRequestCount = "active_request_count"
                }
            }
            let mem = HardwareMemoryMonitor.snapshot()
            let sched = await scheduler.metrics()
            return try .json(
                SystemMetricsEnvelope(
                    memory: mem,
                    scheduler: sched,
                    activeRequestCount: registry.activeRequests().count
                )
            )

        case ("POST", "/v1/messages"):
            let req = try JSONDecoder().decode(MessagesRequest.self, from: body)
            return try await handleMessages(requestID: requestID, request: req, onSSEFrame: onSSEFrame)

        case ("POST", "/v1/embeddings"):
            return try await handleEmbeddings(body: body)

        case ("POST", "/v1/audio/transcriptions"):
            let req = try JSONDecoder().decode(AudioTranscriptionRequest.self, from: body)
            guard let wavData = Data(base64Encoded: req.audioBase64) else {
                return .error(status: 400, type: "invalid_request_error", message: "Invalid base64 audio payload in /v1/audio/transcriptions.")
            }
            do {
                let transcript = try await scheduler.transcribeVoice(
                    wavData: wavData,
                    languageHint: req.languageHint ?? "en",
                    preferredASRModelID: req.model,
                    allowSyntheticFallback: allowSyntheticFallback
                )
                return try .json(transcript)
            } catch let schedErr as ComputeSchedulerError {
                if case .modelNotFound(let missingID) = schedErr {
                    return .error(status: 404, type: "model_not_found", message: "ASR model '\(missingID)' not found.")
                }
                return .error(status: 503, type: "speech_runtime_unavailable", message: schedErr.localizedDescription)
            } catch {
                return .error(status: 503, type: "speech_runtime_unavailable", message: error.localizedDescription)
            }

        case ("POST", "/v1/cache/clear"):
            let clearedCount = await scheduler.clearAllCaches()
            return try .jsonObject(["cleared": true, "providers_cleared": clearedCount])

        case ("POST", "/v1/telemetry/events"):
            let payloadText = String(decoding: body, as: UTF8.self)
            HeylookObservabilityLog.shared.record(
                level: "INFO",
                category: "telemetry",
                message: "client_event",
                requestID: requestID,
                metadata: ["payload": String(payloadText.prefix(2048))]
            )
            return try .jsonObject(["accepted": true], status: 202)

        case ("POST", "/v1/data/clear"):
            try store.clearAllData()
            return try .jsonObject(["cleared": true])

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

        case ("GET", "/v1/admin/models"):
            struct AdminModelsEnvelope: Encodable {
                let object = "list"
                let data: [EngineDescription]
            }
            let models = await scheduler.listModels()
            return try .json(AdminModelsEnvelope(data: models))

        case ("POST", "/v1/admin/models"):
            if let ro = checkWritableConfig() { return ro }
            guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any],
                  let rawPath = (obj["model_path"] as? String) ?? (obj["path"] as? String),
                  !rawPath.isEmpty else {
                return .error(status: 422, type: "invalid_request_error", message: "Missing 'model_path' or 'path'")
            }
            let targetURL = URL(fileURLWithPath: rawPath)
            guard FileManager.default.fileExists(atPath: targetURL.path) else {
                return .error(status: 404, type: "not_found_error", message: "Model path not found: \(targetURL.path)")
            }
            let customID = obj["id"] as? String
            let discovered = HeylookConfig.discoverModels(in: [targetURL])
            var desc = discovered.first ?? EngineDescription(
                modelID: customID ?? targetURL.deletingPathExtension().lastPathComponent,
                provider: targetURL.pathExtension.lowercased() == "gguf" ? .gguf : (targetURL.pathExtension.lowercased() == "aimodel" ? .coreai : .mlx),
                modelPath: targetURL.standardizedFileURL.path
            )
            if let customID, !customID.isEmpty, customID != desc.modelID {
                desc = EngineDescription(
                    modelID: customID,
                    provider: desc.provider,
                    modelPath: desc.modelPath,
                    architecture: desc.architecture,
                    quantization: desc.quantization,
                    computePlacement: desc.computePlacement,
                    isPinned: desc.isPinned,
                    isLoaded: desc.isLoaded,
                    contextSize: desc.contextSize,
                    flashAttention: desc.flashAttention,
                    supportsVision: desc.supportsVision,
                    supportsAudio: desc.supportsAudio,
                    supportsEmbeddings: desc.supportsEmbeddings,
                    supportsThinking: desc.supportsThinking,
                    thinkingBudget: desc.thinkingBudget,
                    reasoningFormat: desc.reasoningFormat,
                    thinkingControls: desc.thinkingControls,
                    templateSource: desc.templateSource,
                    speculativeDrafter: desc.speculativeDrafter,
                    estimatedMemoryMB: desc.estimatedMemoryMB,
                    tags: desc.tags
                )
            }
            let provider = ProviderFactory.makeProvider(for: desc, allowSyntheticFallback: allowSyntheticFallback)
            await scheduler.register(provider: provider, description: desc)
            return try .json(desc, status: 201)

        case ("POST", "/v1/admin/models/validate"):
            guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any],
                  let rawPath = (obj["path"] as? String) ?? (obj["model_path"] as? String),
                  !rawPath.isEmpty else {
                return .error(status: 422, type: "invalid_request_error", message: "Missing 'path' in /v1/admin/models/validate")
            }
            let url = URL(fileURLWithPath: rawPath)
            guard FileManager.default.fileExists(atPath: url.path) else {
                return try .jsonObject(["valid": false, "path": url.path, "reason": "Path does not exist"])
            }
            let found = HeylookConfig.discoverModels(in: [url])
            if let first = found.first {
                let data = try JSONEncoder().encode(first)
                let dict = (try? JSONSerialization.jsonObject(with: data)) ?? [:]
                return try .jsonObject(["valid": true, "path": url.path, "model": dict])
            }
            if url.pathExtension.lowercased() == "gguf", let hdr = GGUFHeaderReader.readFileHeader(at: url) {
                return try .jsonObject([
                    "valid": true,
                    "path": url.path,
                    "provider": "gguf",
                    "architecture": hdr.architecture ?? "gguf",
                    "context_length": hdr.contextLength ?? 8192,
                ])
            }
            return try .jsonObject(["valid": false, "path": url.path, "reason": "No recognizable MLX, GGUF, or CoreAI weights found"])

        case ("POST", "/v1/admin/models/fit"):
            let obj = body.isEmpty ? [:] : ((try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:])
            let requestedCtx = obj["ctx_size"] as? Int
            let cacheTypeK = obj["cache_type_k"] as? String
            if let modelID = obj["model_id"] as? String ?? obj["model"] as? String,
               let desc = await scheduler.modelDescription(for: modelID) {
                let fit = GGUFSpawnConfig.assessRAMFit(for: desc, ctxSize: requestedCtx, cacheTypeK: cacheTypeK)
                return try .json(fit)
            }
            if let rawPath = obj["path"] as? String ?? obj["model_path"] as? String, !rawPath.isEmpty {
                let tempDesc = EngineDescription(
                    modelID: URL(fileURLWithPath: rawPath).lastPathComponent,
                    provider: .gguf,
                    modelPath: rawPath,
                    contextSize: ProvenanceField(value: requestedCtx ?? 8192, provenance: .configured)
                )
                let fit = GGUFSpawnConfig.assessRAMFit(for: tempDesc, ctxSize: requestedCtx, cacheTypeK: cacheTypeK)
                return try .json(fit)
            }
            return .error(status: 422, type: "invalid_request_error", message: "Provide 'model_id' or 'path' for /v1/admin/models/fit")

        case ("POST", "/v1/admin/models/reload"):
            let obj = body.isEmpty ? [:] : ((try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:])
            var cfg = HeylookConfig.loadDefault()
            if let customFolders = obj["scan_folders"] as? [String] {
                cfg.scanFolders = customFolders
            }
            if let maxLoaded = obj["max_loaded_models"] as? Int {
                cfg.maxLoadedModels = maxLoaded
            }
            await scheduler.reconfigureLimits(maxLoadedGPUModels: cfg.maxLoadedModels, maxQueueDepth: cfg.maxQueueDepth)
            let discovered = cfg.discoverConfiguredModels()
            await scheduler.reloadRegistry(discovered, allowSyntheticFallback: allowSyntheticFallback)
            let models = await scheduler.listModels()
            return try .jsonObject(["reloaded": true, "count": models.count])

        case ("GET", "/v1/admin/models/scan-config"):
            let cfg = HeylookConfig.loadDefault()
            return try .jsonObject([
                "scan_folders": cfg.scanFolders,
                "default_model": cfg.defaultModel as Any,
                "max_loaded_models": cfg.maxLoadedModels,
                "config_path": HeylookConfig.activeConfigURL().path,
            ])

        case ("PATCH", "/v1/admin/models/scan-config"):
            if let ro = checkWritableConfig() { return ro }
            guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                return .error(status: 400, type: "invalid_request_error", message: "Invalid JSON body")
            }
            var cfg = HeylookConfig.loadDefault()
            if let folders = obj["scan_folders"] as? [String] {
                cfg.scanFolders = folders
            }
            if let defModel = obj["default_model"] as? String {
                cfg.defaultModel = defModel.isEmpty ? nil : defModel
            }
            let cfgURL = HeylookConfig.activeConfigURL()
            try FileManager.default.createDirectory(at: cfgURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try cfg.toTOML().write(to: cfgURL, atomically: true, encoding: .utf8)
            return try .jsonObject([
                "scan_folders": cfg.scanFolders,
                "default_model": cfg.defaultModel as Any,
            ])

        case ("GET", "/v1/admin/models/model-options"):
            return try .jsonObject([
                "providers": EngineProviderKind.allCases.map(\.rawValue),
                "compute_placements": ComputePlacement.allCases.map(\.rawValue),
                "quantization_modes": MLXQuantizationMode.allCases.map(\.rawValue),
                "sampler_fields": [
                    "temperature", "top_p", "top_k", "min_p",
                    "repetition_penalty", "repetition_context_size", "presence_penalty",
                    "max_tokens", "seed", "stop_sequences",
                    "thinking", "thinking_depth", "thinking_budget",
                ],
            ])

        case ("GET", "/v1/admin/config"):
            let cfg = HeylookConfig.loadDefault()
            return try .json(cfg)

        case ("PATCH", "/v1/admin/config"):
            if let ro = checkWritableConfig() { return ro }
            guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                return .error(status: 400, type: "invalid_request_error", message: "Invalid JSON body")
            }
            var cfg = HeylookConfig.loadDefault()
            if let dm = obj["default_model"] as? String ?? obj["defaultModel"] as? String {
                cfg.defaultModel = dm.isEmpty ? nil : dm
            }
            if let ml = obj["max_loaded_models"] as? Int ?? obj["maxLoadedModels"] as? Int {
                cfg.maxLoadedModels = max(1, ml)
            }
            if let mq = obj["max_queue_depth"] as? Int ?? obj["maxQueueDepth"] as? Int {
                cfg.maxQueueDepth = max(1, mq)
            }
            if let sf = obj["scan_folders"] as? [String] ?? obj["scanFolders"] as? [String] {
                cfg.scanFolders = sf
            }
            if let ah = obj["allowed_hosts"] as? [String] ?? obj["allowedHosts"] as? [String] {
                cfg.allowedHosts = ah
            }
            await scheduler.reconfigureLimits(maxLoadedGPUModels: cfg.maxLoadedModels, maxQueueDepth: cfg.maxQueueDepth)
            let cfgURL = HeylookConfig.activeConfigURL()
            try FileManager.default.createDirectory(at: cfgURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try cfg.toTOML().write(to: cfgURL, atomically: true, encoding: .utf8)
            return try .json(cfg)

        case ("GET", "/v1/admin/config/logs"):
            let limit = queryItems.first(where: { $0.name == "limit" })?.value.flatMap(Int.init) ?? 100
            let levelFilter = queryItems.first(where: { $0.name == "level" })?.value
            let categoryFilter = queryItems.first(where: { $0.name == "category" })?.value
            let searchFilter = queryItems.first(where: { $0.name == "search" })?.value
            let log = HeylookObservabilityLog.shared
            let entries = log.recentEntries(
                limit: max(1, min(1000, limit)),
                level: levelFilter,
                category: categoryFilter,
                search: searchFilter
            )
            struct LogsResponse: Encodable {
                let level: String
                let retention_limit: Int
                let entries: [HeylookObservabilityLog.Entry]
            }
            return try .json(
                LogsResponse(
                    level: log.level,
                    retention_limit: log.retentionLimit,
                    entries: entries
                )
            )

        case ("PATCH", "/v1/admin/config/logs"):
            if let ro = checkWritableConfig() { return ro }
            guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                return .error(status: 400, type: "invalid_request_error", message: "Invalid JSON body")
            }
            let log = HeylookObservabilityLog.shared
            if let lvl = obj["level"] as? String {
                log.setLevel(lvl)
            }
            if let ret = (obj["retention_limit"] as? NSNumber)?.intValue ?? (obj["retentionLimit"] as? NSNumber)?.intValue {
                log.setRetentionLimit(ret)
            }
            struct LogsPatchResponse: Encodable {
                let level: String
                let retention_limit: Int
            }
            return try .json(LogsPatchResponse(level: log.level, retention_limit: log.retentionLimit))

        case ("DELETE", "/v1/admin/config/logs"):
            if let ro = checkWritableConfig() { return ro }
            HeylookObservabilityLog.shared.clear()
            return try .json(["cleared": true])

        case ("GET", "/v1/conversations"):
            let list = try store.listConversations()
            return try .json(list)

        case ("POST", "/v1/conversations"):
            let incoming = body.isEmpty
                ? ConversationRecord()
                : try JSONDecoder().decode(ConversationRecord.self, from: body)
            try store.upsertConversation(incoming)
            return try .json(incoming, status: 201)

        case ("GET", "/v1/presets"):
            return try .json(try store.listPresets())

        case ("POST", "/v1/presets"):
            let preset = try JSONDecoder().decode(PresetRecord.self, from: body)
            let created = try store.createPreset(preset)
            return try .json(created, status: 201)

        case ("GET", "/v1/notebooks"):
            return try .json(try store.listNotebooks())

        case ("POST", "/v1/notebooks"):
            let notebook = try JSONDecoder().decode(NotebookRecord.self, from: body)
            try store.upsertNotebook(notebook)
            return try .json(notebook, status: 201)

        default:
            break
        }

        // DELETE /v1/admin/config/{key}
        if segments.count == 4 && segments[0] == "v1" && segments[1] == "admin" && segments[2] == "config" && method == "DELETE" {
            if let ro = checkWritableConfig() { return ro }
            let key = segments[3]
            var cfg = HeylookConfig.loadDefault()
            switch key {
            case "default_model", "defaultModel":
                cfg.defaultModel = nil
            case "max_loaded_models", "maxLoadedModels":
                cfg.maxLoadedModels = 3
            case "max_queue_depth", "maxQueueDepth":
                cfg.maxQueueDepth = 2
            case "allowed_hosts", "allowedHosts":
                cfg.allowedHosts = []
            default:
                return .error(status: 404, type: "not_found_error", message: "Unknown config key '\(key)'")
            }
            let cfgURL = HeylookConfig.activeConfigURL()
            try FileManager.default.createDirectory(at: cfgURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try cfg.toTOML().write(to: cfgURL, atomically: true, encoding: .utf8)
            return try .json(cfg)
        }

        // GET /v1/performance/profile/{time_range}
        if segments.count == 4 && segments[0] == "v1" && segments[1] == "performance" && segments[2] == "profile" && method == "GET" {
            let rangeSpec = segments[3].lowercased()
            let summary = try store.queryPerformanceProfile(window: rangeSpec)
            struct ProfileResponse: Encodable {
                let time_range: String
                let sample_count: Int
                let average_generation_tps: Double
                let average_prompt_tps: Double
                let average_ttft_ms: Double
                let total_output_tokens: Int
                let total_prompt_tokens: Int
                let total_cached_tokens: Int
                let by_model: [SQLiteStore.ModelPerformanceSummary]
                let recent: [MessageStatsRecord]
            }
            return try .json(
                ProfileResponse(
                    time_range: rangeSpec,
                    sample_count: summary.sampleCount,
                    average_generation_tps: summary.averageGenerationTPS,
                    average_prompt_tps: summary.averagePromptTPS,
                    average_ttft_ms: summary.averageTTFTMs,
                    total_output_tokens: summary.totalOutputTokens,
                    total_prompt_tokens: summary.totalPromptTokens,
                    total_cached_tokens: summary.totalCachedTokens,
                    by_model: summary.byModel,
                    recent: summary.recent
                )
            )
        }

        // /v1/requests/{id}[/reasoning-end]
        if segments.count >= 3 && segments[0] == "v1" && segments[1] == "requests" {
            let targetReqID = segments[2]
            if method == "DELETE" && segments.count == 3 {
                struct CancelAck: Encodable {
                    let id: String
                    let cancelled: Bool
                }
                let cancelled = registry.cancel(id: targetReqID)
                return try .json(CancelAck(id: targetReqID, cancelled: cancelled))
            }
            if method == "POST" && segments.count == 4 && segments[3] == "reasoning-end" {
                let ack = registry.endThinking(id: targetReqID)
                return try .jsonObject(["acknowledged": ack])
            }
        }

        // /v1/notebooks/{id}[/generate]
        if segments.count >= 3 && segments[0] == "v1" && segments[1] == "notebooks" {
            let nbID = segments[2]
            if segments.count == 3 {
                switch method {
                case "GET":
                    guard let nb = try store.getNotebook(id: nbID) else {
                        return .error(status: 404, type: "not_found_error", message: "Notebook '\(nbID)' not found")
                    }
                    return try .json(nb)
                case "PATCH", "PUT":
                    guard var existing = try store.getNotebook(id: nbID) else {
                        return .error(status: 404, type: "not_found_error", message: "Notebook '\(nbID)' not found")
                    }
                    guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                        return .error(status: 400, type: "invalid_request_error", message: "Invalid JSON body")
                    }
                    if let t = obj["title"] as? String { existing.title = t }
                    if let b = obj["body"] as? String { existing.body = b }
                    if obj.keys.contains("system_prompt") || obj.keys.contains("systemPrompt") {
                        existing.systemPrompt = (obj["system_prompt"] as? String) ?? (obj["systemPrompt"] as? String) ?? ""
                    }
                    if obj.keys.contains("model_id") || obj.keys.contains("modelID") {
                        let rawModel = obj["model_id"] ?? obj["modelID"]
                        if rawModel is NSNull {
                            existing.modelID = nil
                        } else if let m = rawModel as? String {
                            existing.modelID = m.isEmpty ? nil : m
                        }
                    }
                    if obj.keys.contains("params") {
                        if obj["params"] is NSNull {
                            existing.params = nil
                        } else if let pObj = obj["params"],
                                  let pData = try? JSONSerialization.data(withJSONObject: pObj),
                                  let p = try? JSONDecoder().decode(SamplerSettings.self, from: pData) {
                            existing.params = p
                        }
                    }
                    if obj.keys.contains("applied_preset_id") || obj.keys.contains("appliedPresetID") {
                        let rawPreset = obj["applied_preset_id"] ?? obj["appliedPresetID"]
                        if rawPreset is NSNull {
                            existing.appliedPresetID = nil
                        } else if let pid = rawPreset as? String {
                            existing.appliedPresetID = pid.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : pid
                        }
                    }
                    existing.updatedUnixMs = Int64(Date().timeIntervalSince1970 * 1000)
                    try store.upsertNotebook(existing)
                    return try .json(existing)
                case "DELETE":
                    guard try store.getNotebook(id: nbID) != nil else {
                        return .error(status: 404, type: "not_found_error", message: "Notebook '\(nbID)' not found")
                    }
                    try store.deleteNotebook(id: nbID)
                    return try .jsonObject(["deleted": true, "id": nbID])
                default:
                    break
                }
            }
            if segments.count == 4 && segments[3] == "generate" && method == "POST" {
                guard var nb = try store.getNotebook(id: nbID) else {
                    return .error(status: 404, type: "not_found_error", message: "Notebook '\(nbID)' not found")
                }
                let obj = body.isEmpty ? [:] : ((try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:])
                let fallbackModelID = await scheduler.listModels().first?.modelID ?? ""
                let modelID = (obj["model"] as? String) ?? nb.modelID ?? fallbackModelID
                guard !modelID.isEmpty else {
                    return .error(status: 422, type: "invalid_request_error", message: "No model specified for notebook generation")
                }
                let promptText = (obj["prompt"] as? String) ?? nb.body
                let sysText = (obj["system"] as? String) ?? (obj["system_prompt"] as? String) ?? (nb.systemPrompt.isEmpty ? nil : nb.systemPrompt)

                var presetSettings: SamplerSettings? = nil
                if let presetID = nb.appliedPresetID,
                   let preset = (try? store.getPreset(id: presetID)) {
                    presetSettings = preset.settings
                }
                let topLevelReqParams = body.isEmpty ? SamplerSettings() : ((try? JSONDecoder().decode(SamplerSettings.self, from: body)) ?? SamplerSettings())
                var nestedReqParams = SamplerSettings()
                if let nestedObj = obj["params"] ?? obj["sampling"],
                   let nestedData = try? JSONSerialization.data(withJSONObject: nestedObj),
                   let decodedNested = try? JSONDecoder().decode(SamplerSettings.self, from: nestedData) {
                    nestedReqParams = decodedNested
                }
                // Cascade: preset -> nb.params -> nested params -> top-level request sampler fields
                let nbLayered = SamplerCascade.resolve(
                    builtin: SamplerSettings(),
                    preset: presetSettings,
                    request: nb.params ?? SamplerSettings()
                ).effective
                let reqLayered = SamplerCascade.resolve(
                    builtin: nbLayered,
                    preset: nestedReqParams,
                    request: topLevelReqParams
                ).effective

                let msgReq = MessagesRequest(
                    model: modelID,
                    messages: [.init(role: "user", content: promptText)],
                    system: sysText,
                    maxTokens: reqLayered.maxTokens ?? 2048,
                    temperature: reqLayered.temperature,
                    topP: reqLayered.topP,
                    topK: reqLayered.topK,
                    minP: reqLayered.minP,
                    repetitionPenalty: reqLayered.repetitionPenalty,
                    repetitionContextSize: reqLayered.repetitionContextSize,
                    presencePenalty: reqLayered.presencePenalty,
                    seed: reqLayered.seed,
                    stream: (obj["stream"] as? Bool) ?? true,
                    thinking: reqLayered.thinking,
                    thinkingDepth: reqLayered.thinkingDepth,
                    thinkingBudget: reqLayered.thinkingBudget,
                    stopSequences: reqLayered.stopSequences
                )
                let resp = try await handleMessages(requestID: requestID, request: msgReq, onSSEFrame: onSSEFrame)
                if let latest = try? store.getNotebook(id: nbID) {
                    nb = latest
                }
                return resp
            }
        }

        // /v1/presets/{id}
        if segments.count == 3 && segments[0] == "v1" && segments[1] == "presets" {
            let presetID = segments[2]
            switch method {
            case "GET":
                guard let p = try store.getPreset(id: presetID) else {
                    return .error(status: 404, type: "not_found_error", message: "Preset '\(presetID)' not found")
                }
                return try .json(p)
            case "PATCH", "PUT":
                guard var existing = try store.getPreset(id: presetID) else {
                    return .error(status: 404, type: "not_found_error", message: "Preset '\(presetID)' not found")
                }
                guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                    return .error(status: 400, type: "invalid_request_error", message: "Invalid JSON body")
                }
                if let n = obj["name"] as? String { existing.name = n }
                if let d = obj["description"] as? String { existing.description = d }
                if let sc = obj["scope"] as? String { existing.scope = sc }
                if obj.keys.contains("system_prompt") || obj.keys.contains("systemPrompt") {
                    let rawSys = obj["system_prompt"] ?? obj["systemPrompt"]
                    if rawSys is NSNull {
                        existing.systemPrompt = nil
                    } else if let sysStr = rawSys as? String {
                        existing.systemPrompt = sysStr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : sysStr
                    }
                }
                if let sObj = obj["settings"],
                   let sData = try? JSONSerialization.data(withJSONObject: sObj),
                   let s = try? JSONDecoder().decode(SamplerSettings.self, from: sData) {
                    existing.settings = s
                }
                existing.updatedUnixMs = Int64(Date().timeIntervalSince1970 * 1000)
                let updated = try store.updatePreset(id: presetID, existing)
                return try .json(updated)
            case "DELETE":
                let deleted = try store.deletePreset(id: presetID)
                guard deleted else {
                    return .error(status: 404, type: "not_found_error", message: "Preset '\(presetID)' not found")
                }
                return try .jsonObject(["deleted": true, "id": presetID])
            default:
                break
            }
        }

        // /v1/conversations/{id}...
        if segments.count >= 3 && segments[0] == "v1" && segments[1] == "conversations" {
            let convID = segments[2]
            if segments.count == 3 {
                switch method {
                case "GET":
                    guard let conv = try store.getConversation(id: convID) else {
                        return .error(status: 404, type: "not_found_error", message: "Conversation '\(convID)' not found")
                    }
                    let msgs = try store.listMessages(conversationID: convID)
                    struct ConversationDetailEnvelope: Encodable {
                        let conversation: ConversationRecord
                        let messages: [MessageRecord]
                    }
                    return try .json(ConversationDetailEnvelope(conversation: conv, messages: msgs))
                case "PATCH", "PUT":
                    guard var conv = try store.getConversation(id: convID) else {
                        return .error(status: 404, type: "not_found_error", message: "Conversation '\(convID)' not found")
                    }
                    guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                        return .error(status: 400, type: "invalid_request_error", message: "Invalid JSON body")
                    }
                    if let t = obj["title"] as? String { conv.title = t }
                    if obj.keys.contains("system_prompt") || obj.keys.contains("systemPrompt") {
                        conv.systemPrompt = (obj["system_prompt"] as? String) ?? (obj["systemPrompt"] as? String) ?? ""
                    }
                    if obj.keys.contains("default_model_id") || obj.keys.contains("defaultModelID") {
                        let rawModel = obj["default_model_id"] ?? obj["defaultModelID"]
                        if rawModel is NSNull {
                            conv.defaultModelID = nil
                        } else if let m = rawModel as? String {
                            conv.defaultModelID = m.isEmpty ? nil : m
                        }
                    }
                    if obj.keys.contains("params") {
                        if obj["params"] is NSNull {
                            conv.params = nil
                        } else if let pObj = obj["params"],
                                  let pData = try? JSONSerialization.data(withJSONObject: pObj),
                                  let p = try? JSONDecoder().decode(SamplerSettings.self, from: pData) {
                            conv.params = p
                        }
                    }
                    if obj.keys.contains("applied_preset_id") || obj.keys.contains("appliedPresetID") {
                        let rawPreset = obj["applied_preset_id"] ?? obj["appliedPresetID"]
                        if rawPreset is NSNull {
                            conv.appliedPresetID = nil
                        } else if let pid = rawPreset as? String {
                            conv.appliedPresetID = pid.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : pid
                        }
                    }
                    conv.updatedUnixMs = Int64(Date().timeIntervalSince1970 * 1000)
                    try store.upsertConversation(conv)
                    return try .json(conv)
                case "DELETE":
                    guard try store.getConversation(id: convID) != nil else {
                        return .error(status: 404, type: "not_found_error", message: "Conversation '\(convID)' not found")
                    }
                    try store.deleteConversation(id: convID)
                    return try .jsonObject(["deleted": true, "id": convID])
                default:
                    break
                }
            }

            if segments.count == 4 && segments[3] == "clone" && method == "POST" {
                let obj = body.isEmpty ? [:] : ((try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:])
                let upTo = obj["up_to_position"] as? Int
                let title = obj["title"] as? String
                let cloned = try store.cloneConversation(id: convID, upToPosition: upTo, newTitle: title)
                return try .json(cloned, status: 201)
            }

            if segments.count == 4 && segments[3] == "messages" {
                if method == "GET" {
                    let afterPos = queryItems.first(where: { $0.name == "after" })?.value.flatMap(Int.init)
                    var msgs = try store.listMessages(conversationID: convID)
                    if let afterPos {
                        msgs = msgs.filter { $0.position > afterPos }
                    }
                    return try .json(msgs)
                }
                if method == "POST" {
                    if try store.getConversation(id: convID) == nil {
                        try store.upsertConversation(ConversationRecord(id: convID))
                    }
                    let existing = try store.listMessages(conversationID: convID)
                    let nextPos = (existing.map(\.position).max() ?? -1) + 1
                    if let decoded = try? JSONDecoder().decode(MessageRecord.self, from: body) {
                        var rec = decoded
                        rec = MessageRecord(
                            id: rec.id,
                            conversationID: convID,
                            position: nextPos,
                            role: rec.role,
                            content: rec.content,
                            thinking: rec.thinking,
                            modelID: rec.modelID,
                            contentBlocks: rec.contentBlocks
                        )
                        let saved = try store.appendMessage(rec)
                        return try .json(saved, status: 201)
                    }
                    let turn = try JSONDecoder().decode(MessagesRequest.MessageTurn.self, from: body)
                    let rec = MessageRecord(
                        conversationID: convID,
                        position: nextPos,
                        role: turn.role,
                        content: turn.content,
                        thinking: turn.thinking,
                        contentBlocks: turn.parts
                    )
                    let saved = try store.appendMessage(rec)
                    return try .json(saved, status: 201)
                }
                if method == "DELETE" {
                    let fromPos = queryItems.first(where: { $0.name == "from" })?.value.flatMap(Int.init) ?? 0
                    let deleted = try store.deleteMessages(conversationID: convID, fromPosition: fromPos)
                    return try .json(["deleted": deleted])
                }
            }

            if segments.count == 5 && segments[3] == "messages" {
                let msgID = segments[4]
                if method == "PATCH" || method == "PUT" {
                    guard var existing = try store.getMessage(id: msgID), existing.conversationID == convID else {
                        return .error(status: 404, type: "not_found_error", message: "Message '\(msgID)' not found")
                    }
                    guard let obj = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                        return .error(status: 400, type: "invalid_request_error", message: "Invalid JSON body")
                    }
                    if let explicitBlocksObj = obj["content_blocks"] ?? obj["contentBlocks"] {
                        if explicitBlocksObj is NSNull {
                            existing.contentBlocks = nil
                        } else if let bData = try? JSONSerialization.data(withJSONObject: explicitBlocksObj),
                                  let blocks = try? JSONDecoder().decode([MessagesRequest.MessageTurn.ContentPart].self, from: bData) {
                            existing.contentBlocks = blocks
                        }
                    } else if let c = obj["content"] as? String, let oldBlocks = existing.contentBlocks {
                        var updatedBlocks: [MessagesRequest.MessageTurn.ContentPart] = []
                        var insertedText = false
                        for block in oldBlocks {
                            switch block {
                            case .text:
                                if !insertedText {
                                    if !c.isEmpty {
                                        updatedBlocks.append(.text(c))
                                    }
                                    insertedText = true
                                }
                            default:
                                updatedBlocks.append(block)
                            }
                        }
                        if !insertedText && !c.isEmpty {
                            updatedBlocks.append(.text(c))
                        }
                        existing.contentBlocks = updatedBlocks.isEmpty ? nil : updatedBlocks
                    }
                    if let c = obj["content"] as? String { existing.content = c }
                    if obj.keys.contains("thinking") {
                        existing.thinking = obj["thinking"] as? String
                    }
                    let truncateAfter = (obj["truncate_after"] as? Bool)
                        ?? (queryItems.first(where: { $0.name == "truncate_after" })?.value == "true")
                    if truncateAfter {
                        _ = try store.deleteMessages(conversationID: convID, fromPosition: existing.position + 1)
                    }
                    let saved = try store.appendMessage(existing)
                    return try .json(saved)
                }
                if method == "DELETE" {
                    let removed = try store.deleteMessage(conversationID: convID, messageID: msgID)
                    guard removed else {
                        return .error(status: 404, type: "not_found_error", message: "Message '\(msgID)' not found")
                    }
                    return try .jsonObject(["deleted": true, "id": msgID])
                }
            }

            if segments.count == 5 && segments[3] == "media" && method == "GET" {
                let mediaID = segments[4]
                guard let blob = try store.getMediaBlob(id: mediaID) else {
                    return .error(status: 404, type: "not_found_error", message: "Media blob '\(mediaID)' not found")
                }
                return HeylookHTTPResponse(
                    status: 200,
                    headers: [
                        "Content-Type": blob.mediaType,
                        "Cache-Control": "public, max-age=31536000, immutable",
                    ],
                    body: blob.data
                )
            }

            if segments.count == 4 && segments[3] == "generate" {
                if method == "POST" {
                    let genReq = try JSONDecoder().decode(ConversationGenerateRequest.self, from: body)
                    return try await handleConversationGenerate(
                        requestID: requestID,
                        conversationID: convID,
                        generateRequest: genReq,
                        onSSEFrame: onSSEFrame
                    )
                }
                if method == "DELETE" {
                    let cancelled = registry.abort(forConversationID: convID)
                    return try .jsonObject(["conversation_id": convID, "cancelled": cancelled])
                }
            }

            if segments.count == 4 && segments[3] == "voice-generate" && method == "POST" {
                let voiceReq = try JSONDecoder().decode(ConversationVoiceGenerateRequest.self, from: body)
                guard let wavData = Data(base64Encoded: voiceReq.audioBase64) else {
                    return .error(status: 400, type: "invalid_request_error", message: "Invalid base64 audio payload in voice-generate.")
                }
                let transcript: CoreAIProvider.ANESpeechTranscriptionResult
                do {
                    transcript = try await scheduler.transcribeVoice(
                        wavData: wavData,
                        languageHint: voiceReq.languageHint ?? "en",
                        preferredASRModelID: voiceReq.asrModel,
                        allowSyntheticFallback: allowSyntheticFallback
                    )
                } catch let schedErr as ComputeSchedulerError {
                    if case .modelNotFound(let missingID) = schedErr {
                        return .error(status: 404, type: "model_not_found", message: "ASR model '\(missingID)' not found.")
                    }
                    return .error(status: 503, type: "speech_runtime_unavailable", message: schedErr.localizedDescription)
                } catch {
                    return .error(status: 503, type: "speech_runtime_unavailable", message: error.localizedDescription)
                }
                guard !transcript.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return .error(status: 422, type: "empty_transcription", message: "No speech detected in audio payload.")
                }
                let transcriptJSON = (try? String(data: JSONEncoder().encode(transcript), encoding: .utf8)) ?? "{}"
                let voiceFrame = "event: heylook_voice_transcribed\ndata: \(transcriptJSON)\n\n"
                if let onSSEFrame {
                    await onSSEFrame(voiceFrame)
                }
                let genResp = try await handleConversationGenerate(
                    requestID: requestID,
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

            if segments.count == 4 && segments[3] == "prompt" && (method == "GET" || method == "POST") {
                struct PromptRequest: Decodable {
                    var model: String?
                    var system: String?
                    var thinking: Bool?
                    var thinking_depth: String?
                    var reasoning_effort: String?
                    var continuation: Bool?
                }
                struct PromptResponse: Encodable {
                    let conversation_id: String
                    let prompt: String
                    let source: String
                    let template_source: String
                    let markers: [PromptMarkerSpan]
                    let continuation: Bool
                    let dropped_media: Int
                    let unrendered_media: Int
                    let message_count: Int
                    let thinking_enabled: Bool
                    let thinking_depth: String?
                }
                let bodyPromptReq = body.isEmpty ? nil : (try? JSONDecoder().decode(PromptRequest.self, from: body))
                func queryParam(_ key: String) -> String? {
                    queryItems.first(where: { $0.name == key })?.value
                }
                let queryBool: (String) -> Bool? = { key in
                    guard let raw = queryParam(key)?.lowercased() else { return nil }
                    if raw == "true" || raw == "1" { return true }
                    if raw == "false" || raw == "0" { return false }
                    return nil
                }
                let promptReq = PromptRequest(
                    model: bodyPromptReq?.model ?? queryParam("model"),
                    system: bodyPromptReq?.system ?? queryParam("system"),
                    thinking: bodyPromptReq?.thinking ?? queryBool("thinking"),
                    thinking_depth: bodyPromptReq?.thinking_depth ?? queryParam("thinking_depth"),
                    reasoning_effort: bodyPromptReq?.reasoning_effort ?? queryParam("reasoning_effort"),
                    continuation: bodyPromptReq?.continuation ?? queryBool("continuation")
                )
                let conv = try store.getConversation(id: convID)
                let msgs = try store.listMessages(conversationID: convID, hydrateMedia: true)
                let requestedModelID = promptReq.model ?? conv?.defaultModelID
                var modelDesc: EngineDescription?
                if let requestedModelID,
                   let desc = await scheduler.modelDescription(for: requestedModelID) {
                    modelDesc = desc
                } else if let firstModel = await scheduler.listModels().first {
                    modelDesc = firstModel
                }
                let modelDirURL = modelDesc.map { HeylookConfig.resolveModelDirectoryURL(from: $0.modelPath) }
                let requestOverlay = SamplerSettings(
                    thinking: promptReq.thinking,
                    thinkingDepth: promptReq.thinking_depth ?? promptReq.reasoning_effort
                )
                let resolvedSamplers = HeylookConfig.resolveEffectiveSampling(
                    modelPath: modelDesc?.modelPath,
                    request: requestOverlay
                )
                let effectiveThinking = promptReq.thinking
                    ?? resolvedSamplers.effective.thinking
                    ?? modelDesc?.thinkingControls?.defaultEnabled
                    ?? modelDesc?.supportsThinking.value
                    ?? true
                let effectiveDepth = (promptReq.thinking_depth ?? promptReq.reasoning_effort)
                    ?? resolvedSamplers.effective.thinkingDepth
                    ?? (modelDesc?.thinkingControls?.depthValues.contains("medium") == true
                        ? "medium"
                        : modelDesc?.thinkingControls?.depthValues.first)
                let hydratedTurns: [MessagesRequest.MessageTurn] = msgs.map { m in
                    let blocks = m.contentBlocks ?? []
                    let mediaAtts: [MessagesRequest.MediaAttachment] = blocks.compactMap {
                        if case .media(let att) = $0 { return att }
                        return nil
                    }
                    return MessagesRequest.MessageTurn(
                        role: m.role,
                        content: m.content,
                        thinking: m.thinking,
                        attachments: mediaAtts,
                        parts: blocks.isEmpty ? nil : blocks
                    )
                }
                let isContinuation = promptReq.continuation ?? false
                let detailed = try HeylookConfig.renderPromptDetailed(
                    in: modelDirURL,
                    system: promptReq.system ?? conv?.systemPrompt,
                    turns: hydratedTurns,
                    thinkingEnabled: effectiveThinking,
                    thinkingDepth: effectiveDepth,
                    continuation: isContinuation,
                    modelSupportsVision: modelDesc?.supportsVision.value ?? true,
                    modelSupportsAudio: modelDesc?.supportsAudio.value ?? true
                )
                return try .json(
                    PromptResponse(
                        conversation_id: convID,
                        prompt: detailed.prompt,
                        source: detailed.source,
                        template_source: detailed.source,
                        markers: detailed.markers,
                        continuation: detailed.continuation,
                        dropped_media: detailed.droppedMedia,
                        unrendered_media: detailed.unrenderedMedia,
                        message_count: msgs.count,
                        thinking_enabled: effectiveThinking,
                        thinking_depth: effectiveDepth
                    )
                )
            }

            if segments.count == 5 && segments[3] == "generate" && segments[4] == "reasoning-end" && method == "POST" {
                let ack = registry.requestReasoningEnd(forConversationID: convID)
                return try .jsonObject(["acknowledged": ack])
            }
        }

        // /v1/models/{id}[/load | /unload | /image-plan]
        if segments.count >= 3 && segments[0] == "v1" && segments[1] == "models" {
            let modelID = segments[2]
            if segments.count == 3 && method == "GET" {
                guard let desc = await scheduler.modelDescription(for: modelID) else {
                    throw ComputeSchedulerError.modelNotFound(modelID)
                }
                return try .json(desc)
            }
            if segments.count == 4 && segments[3] == "load" && method == "POST" {
                let warmParam = queryItems.first(where: { $0.name == "warm" })?.value?.lowercased()
                let warm = (warmParam == nil) ? true : (warmParam == "true" || warmParam == "1")
                let loaded = try await scheduler.activateModel(modelID, warm: warm)
                return try .json(loaded)
            }
            if segments.count == 4 && segments[3] == "unload" && method == "POST" {
                await scheduler.unloadModel(modelID)
                guard let unloaded = await scheduler.modelDescription(for: modelID) else {
                    throw ComputeSchedulerError.modelNotFound(modelID)
                }
                return try .json(unloaded)
            }
            if segments.count == 4 && segments[3] == "image-plan" && method == "POST" {
                guard let desc = await scheduler.modelDescription(for: modelID) else {
                    throw ComputeSchedulerError.modelNotFound(modelID)
                }
                guard desc.isLoaded else {
                    return .error(
                        status: 409,
                        type: "invalid_request_error",
                        message: "Model '\(modelID)' is not loaded; image-plan requires a resident model."
                    )
                }
                guard desc.supportsVision.value else {
                    return .error(
                        status: 422,
                        type: "invalid_request_error",
                        message: "Model '\(modelID)' does not support vision."
                    )
                }
                let modelDirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
                let defaultBudget = HeylookConfig.estimateImageTokenBudget(in: modelDirURL)
                let obj = body.isEmpty ? [:] : ((try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:])
                let requestedCap = (obj["image_max_tokens"] as? Int)
                    ?? HeylookConfig.loadSidecar(in: modelDirURL)?.imageMaxTokens
                let effectiveCap = requestedCap.map { min($0, max($0, 16)) } ?? defaultBudget

                struct ResizedSize: Encodable {
                    let width: Int
                    let height: Int
                    let tokens: Int
                }
                struct ImagePlanResponse: Encodable {
                    let model: String
                    let tokens_per_image: Int
                    let total_image_tokens: Int
                    let resized_dimensions: [ResizedSize]
                }

                var resizedList: [ResizedSize] = []
                if let sizeArr = obj["sizes"] as? [[String: Any]], !sizeArr.isEmpty {
                    let cellSide = 28.0 // 14px patch * 2x2 spatial merge
                    for entry in sizeArr {
                        let w = max(1, (entry["width"] as? Int) ?? 896)
                        let h = max(1, (entry["height"] as? Int) ?? 896)
                        let maxPixels = Double(effectiveCap) * cellSide * cellSide
                        let curPixels = Double(w * h)
                        let scale = curPixels > maxPixels ? (maxPixels / curPixels).squareRoot() : 1.0
                        let rw = max(28, Int((Double(w) * scale / cellSide).rounded()) * 28)
                        let rh = max(28, Int((Double(h) * scale / cellSide).rounded()) * 28)
                        let toks = max(1, (rw / 28) * (rh / 28))
                        resizedList.append(ResizedSize(width: rw, height: rh, tokens: toks))
                    }
                }
                let totalToks = resizedList.isEmpty ? defaultBudget : resizedList.map(\.tokens).reduce(0, +)
                let perImg = resizedList.first?.tokens ?? (requestedCap ?? defaultBudget)
                return try .json(
                    ImagePlanResponse(
                        model: modelID,
                        tokens_per_image: perImg,
                        total_image_tokens: totalToks,
                        resized_dimensions: resizedList
                    )
                )
            }
        }

        // /v1/admin/models/{id}...
        if segments.count >= 4 && segments[0] == "v1" && segments[1] == "admin" && segments[2] == "models" {
            let modelID = segments[3]
            if segments.count == 5 && segments[4] == "chat-template" {
                guard let existing = await scheduler.modelDescription(for: modelID) else {
                    throw ComputeSchedulerError.modelNotFound(modelID)
                }
                let dirURL = HeylookConfig.resolveModelDirectoryURL(from: existing.modelPath)
                struct ChatTemplateResponse: Encodable {
                    let model: String
                    let source: String
                    let origin: String
                    let has_override: Bool
                    let override_present: Bool
                    let writable: Bool
                    let inert_reason: String?
                    let stale: Bool
                    let prefix_stable: Bool
                    let prefix_note: String?
                    let modified_since_download: Bool?
                    let caps: TemplateCapabilities
                    let template: String
                }
                func makeResponse(from state: HeylookConfig.ChatTemplateState) -> ChatTemplateResponse {
                    ChatTemplateResponse(
                        model: modelID,
                        source: state.source,
                        origin: state.origin,
                        has_override: state.hasOverride,
                        override_present: state.overridePresent,
                        writable: state.writable && !isReadOnlyConfig,
                        inert_reason: state.inertReason,
                        stale: state.stale,
                        prefix_stable: state.prefixStable,
                        prefix_note: state.prefixNote,
                        modified_since_download: state.modifiedSinceDownload,
                        caps: state.caps,
                        template: state.template
                    )
                }

                switch method {
                case "GET":
                    let state = HeylookConfig.inspectChatTemplate(
                        in: dirURL,
                        fallbackSource: existing.templateSource.value
                    )
                    return try .json(makeResponse(from: state))

                case "PUT":
                    if let ro = checkWritableConfig() { return ro }
                    let templateText: String
                    if let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                       let t = json["template"] as? String {
                        templateText = t
                    } else {
                        templateText = String(decoding: body, as: UTF8.self)
                    }
                    let state = try HeylookConfig.writeChatTemplateOverride(templateText, in: dirURL)
                    let sidecar = HeylookConfig.loadSidecar(in: dirURL) ?? ModelSidecarConfig(id: modelID)
                    _ = try? await scheduler.applySidecarUpdate(sidecar, for: modelID)
                    return try .json(makeResponse(from: state))

                case "DELETE":
                    if let ro = checkWritableConfig() { return ro }
                    let priorState = HeylookConfig.inspectChatTemplate(
                        in: dirURL,
                        fallbackSource: existing.templateSource.value
                    )
                    guard priorState.hasOverride else {
                        return .error(
                            status: 404,
                            type: "not_found_error",
                            message: "No chat_template.heylook.jinja override exists for '\(modelID)'"
                        )
                    }
                    let reverted = HeylookConfig.deleteChatTemplateOverride(
                        in: dirURL,
                        fallbackSource: existing.templateSource.value == "chat_template.heylook.jinja"
                            ? (existing.provider == .gguf ? "gguf_embedded" : "tokenizer_config.json")
                            : existing.templateSource.value
                    )
                    let sidecar = HeylookConfig.loadSidecar(in: dirURL) ?? ModelSidecarConfig(id: modelID)
                    _ = try? await scheduler.applySidecarUpdate(sidecar, for: modelID)
                    return try .json(makeResponse(from: reverted))

                default:
                    break
                }
            }

            if segments.count == 4 && method == "GET" {
                guard let desc = await scheduler.modelDescription(for: modelID) else {
                    throw ComputeSchedulerError.modelNotFound(modelID)
                }
                return try .json(desc)
            }
            if segments.count == 4 && (method == "PATCH" || method == "PUT") {
                if let ro = checkWritableConfig() { return ro }
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
                let dirURL = HeylookConfig.resolveModelDirectoryURL(from: existing.modelPath)
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue {
                    try? HeylookConfig.writeSidecar(sidecar, in: dirURL)
                }
                let updated = try await scheduler.applySidecarUpdate(sidecar, for: modelID)
                return try .json(updated)
            }
            if segments.count == 4 && method == "DELETE" {
                if let ro = checkWritableConfig() { return ro }
                let removed = await scheduler.removeModel(modelID)
                guard removed else {
                    throw ComputeSchedulerError.modelNotFound(modelID)
                }
                return try .jsonObject(["deleted": true, "id": modelID])
            }
            if segments.count == 5 && segments[4] == "load" && method == "POST" {
                let warmParam = queryItems.first(where: { $0.name == "warm" })?.value?.lowercased()
                let warm = (warmParam == nil) ? true : (warmParam == "true" || warmParam == "1")
                let loaded = try await scheduler.activateModel(modelID, warm: warm)
                return try .json(loaded)
            }
            if segments.count == 5 && segments[4] == "unload" && method == "POST" {
                await scheduler.unloadModel(modelID)
                guard let unloaded = await scheduler.modelDescription(for: modelID) else {
                    throw ComputeSchedulerError.modelNotFound(modelID)
                }
                return try .json(unloaded)
            }
        }

        return .error(status: 404, type: "not_found_error", message: "Route \(method) \(cleanPath) not found")
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
        var promptNames: [String?] = []

        if let singleText = json["input"] as? String {
            if singleText.count > EmbeddingRequestLimits.maxTextChars {
                return .error(status: 422, type: "invalid_request_error", message: "Text block exceeds \(EmbeddingRequestLimits.maxTextChars) characters")
            }
            inputs = [.text(singleText, task: task)]
            promptNames = [task.promptName]
        } else if let textList = json["input"] as? [String] {
            if textList.isEmpty || textList.count > EmbeddingRequestLimits.maxItems {
                return .error(status: 422, type: "invalid_request_error", message: "Embedding input count (\(textList.count)) must be between 1 and \(EmbeddingRequestLimits.maxItems)")
            }
            for txt in textList where txt.count > EmbeddingRequestLimits.maxTextChars {
                return .error(status: 422, type: "invalid_request_error", message: "Text block exceeds \(EmbeddingRequestLimits.maxTextChars) characters")
            }
            inputs = textList.map { .text($0, task: task) }
            promptNames = Array(repeating: task.promptName, count: textList.count)
        } else if let itemObjects = json["input"] as? [[String: Any]] {
            if itemObjects.isEmpty || itemObjects.count > EmbeddingRequestLimits.maxItems {
                return .error(
                    status: 422,
                    type: "invalid_request_error",
                    message: "Embedding input count (\(itemObjects.count)) must be between 1 and \(EmbeddingRequestLimits.maxItems)"
                )
            }
            var totalRequestMediaBytes = 0
            for itemObj in itemObjects {
                let promptName = itemObj["prompt_name"] as? String
                let resolvedTask: EmbeddingTask
                if let promptName {
                    resolvedTask = EmbeddingTask.allCases.first { $0.promptName == promptName || $0.rawValue == promptName } ?? task
                } else {
                    resolvedTask = task
                }
                guard let contentBlocks = itemObj["content"] as? [[String: Any]], !contentBlocks.isEmpty else {
                    return .error(status: 400, type: "invalid_request_error", message: "Each embedding item must contain a non-empty 'content' array")
                }
                if contentBlocks.count > EmbeddingRequestLimits.maxBlocksPerItem {
                    return .error(
                        status: 422,
                        type: "invalid_request_error",
                        message: "Embedding item content block count (\(contentBlocks.count)) exceeds limit of \(EmbeddingRequestLimits.maxBlocksPerItem)"
                    )
                }
                let hasWindows = contentBlocks.contains { $0["windows"] != nil && !($0["windows"] is NSNull) }
                if hasWindows && contentBlocks.count != 1 {
                    return .error(
                        status: 422,
                        type: "invalid_request_error",
                        message: "An item with video windows must contain only that video block"
                    )
                }
                var parts: [EmbeddingPart] = []
                for block in contentBlocks {
                    let bType = (block["type"] as? String) ?? "text"
                    if bType == "text" {
                        let txt = (block["text"] as? String) ?? ""
                        if txt.count > EmbeddingRequestLimits.maxTextChars {
                            return .error(status: 422, type: "invalid_request_error", message: "Text block exceeds \(EmbeddingRequestLimits.maxTextChars) characters")
                        }
                        parts.append(.text(txt))
                    } else if bType == "image" || bType == "audio" || bType == "video" {
                        guard let source = block["source"] as? [String: Any],
                              let b64 = source["data"] as? String,
                              let data = Data(base64Encoded: b64) else {
                            return .error(status: 400, type: "invalid_request_error", message: "Invalid base64 media source in /v1/embeddings")
                        }
                        let mimeType = (source["media_type"] as? String)
                            ?? (bType == "video" ? "video/mp4" : (bType == "audio" ? "audio/wav" : "image/jpeg"))
                        if !mimeType.lowercased().hasPrefix("\(bType)/") {
                            return .error(status: 422, type: "invalid_request_error", message: "media_type '\(mimeType)' must match block type '\(bType)'")
                        }
                        let kindCap = EmbeddingRequestLimits.mediaBytesByKind[bType] ?? EmbeddingRequestLimits.maxImageOrAudioBytes
                        if data.count > kindCap {
                            return .error(status: 422, type: "invalid_request_error", message: "\(bType) payload (\(data.count) bytes) exceeds limit of \(kindCap) bytes")
                        }
                        totalRequestMediaBytes += data.count
                        if totalRequestMediaBytes > EmbeddingRequestLimits.maxRequestMediaBytes {
                            return .error(
                                status: 422,
                                type: "invalid_request_error",
                                message: "Total request media (\(totalRequestMediaBytes) bytes) exceeds \(EmbeddingRequestLimits.maxRequestMediaBytes) bytes"
                            )
                        }
                        var windowSampling: VideoWindowSampling = .off
                        if let winDict = block["windows"] as? [String: Any] {
                            guard bType == "video" else {
                                return .error(status: 422, type: "invalid_request_error", message: "windows applies to video blocks only")
                            }
                            let samplingStr = (winDict["sampling"] as? String) ?? "fps"
                            windowSampling = VideoWindowSampling(rawValue: samplingStr) ?? .fps
                        }
                        let kind: ArtifactKind = bType == "video" ? .video : (bType == "audio" ? .audio : .image)
                        parts.append(.media(kind: kind, mimeType: mimeType, data: data, windowSampling: windowSampling))
                    } else {
                        return .error(status: 400, type: "invalid_request_error", message: "Unsupported embedding block type '\(bType)'")
                    }
                }
                inputs.append(EmbeddingInput(parts: parts, task: resolvedTask))
                promptNames.append(promptName)
            }
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
                promptNames.append(task.promptName)
            }
        } else {
            return .error(status: 400, type: "invalid_request_error", message: "Missing or unsupported 'input' field in /v1/embeddings")
        }

        let outputs = try await embeddingEngine.embedBatch(inputs, dimensions: requestedDim)
        struct WindowRow: Encodable {
            let start: Double
            let end: Double
            let frame_times: [Double]
            let embedding: [Float]
        }
        struct EmbeddingItem: Encodable {
            let object = "embedding"
            let index: Int
            let embedding: [Float]
            let prompt_name: String?
            let windows: [WindowRow]?
        }
        struct EmbeddingsResponseEnvelope: Encodable {
            let object = "list"
            let model: String
            let dimensions: Int
            let data: [EmbeddingItem]
        }

        let items = outputs.enumerated().map { idx, out in
            let winRows: [WindowRow]? = out.windows.isEmpty ? nil : out.windows.map {
                WindowRow(start: $0.start, end: $0.end, frame_times: $0.frameTimes, embedding: $0.vector)
            }
            let pName = idx < promptNames.count ? promptNames[idx] : nil
            return EmbeddingItem(index: idx, embedding: out.vector, prompt_name: pName, windows: winRows)
        }
        return try .json(EmbeddingsResponseEnvelope(model: modelName, dimensions: requestedDim, data: items))
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
        if let requestedDepth = request.thinkingDepth, !requestedDepth.isEmpty,
           let controls = desc.thinkingControls {
            if controls.depthVariable == nil && controls.depthValues.isEmpty {
                return .error(
                    status: 400,
                    type: "invalid_request_error",
                    message: "Model '\(request.model)' does not support variable thinking_depth / reasoning_effort."
                )
            }
            if !controls.depthValues.isEmpty && !controls.depthValues.contains(requestedDepth) {
                return .error(
                    status: 400,
                    type: "invalid_request_error",
                    message: "Model '\(request.model)' does not support thinking_depth '\(requestedDepth)'. Allowed values: \(controls.depthValues.joined(separator: ", "))."
                )
            }
        }

        let resolvedSampling = HeylookConfig.resolveEffectiveSampling(
            modelPath: desc.modelPath,
            request: request.samplerOverlay
        )
        registry.register(requestID: requestID, modelID: request.model)

        return try await scheduler.withComputeLease(for: request.model) { [registry] in
            defer {
                registry.unregister(id: requestID)
            }
            let useStream = request.stream ?? false
            let stream = await provider.generate(
                requestID: requestID,
                request: request,
                effectiveSampling: resolvedSampling.effective,
                registry: registry
            )
            let consumed = try await Self.consumeStreamAndFormatSSE(
                requestID: requestID,
                modelID: request.model,
                stream: stream,
                emitSSE: useStream,
                onSSEFrame: onSSEFrame
            )

            if useStream {
                return HeylookHTTPResponse(
                    status: 200,
                    headers: [
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    ],
                    body: Data(consumed.sseLines.joined().utf8)
                )
            }

            var blocks: [MessagesResponse.ContentBlock] = []
            if !consumed.fullThinking.isEmpty {
                blocks.append(.thinking(consumed.fullThinking))
            }
            blocks.append(.text(consumed.fullText))

            let response = MessagesResponse(
                id: requestID,
                model: request.model,
                content: blocks,
                stopReason: consumed.finalChunk.stopReason ?? .endTurn,
                usage: consumed.usage,
                performance: consumed.performance
            )
            return try HeylookHTTPResponse.json(response)
        }
    }

    private static func consumeStreamAndFormatSSE(
        requestID: String,
        modelID: String,
        stream: AsyncThrowingStream<GenerationChunk, Error>,
        emitSSE: Bool,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> (
        fullText: String,
        fullThinking: String,
        finalChunk: GenerationChunk,
        usage: MessagesResponse.Usage,
        performance: MessagesResponse.PerformanceExtension,
        sseLines: [String]
    ) {
        var fullText = ""
        var fullThinking = ""
        var finalChunk = GenerationChunk()
        var sseLines: [String] = []
        var emittedMessageStart = false
        var nextBlockIndex = 0
        var activeThinkingIndex: Int?
        var activeTextIndex: Int?

        func appendSSE(_ frame: String) async {
            sseLines.append(frame)
            if let onSSEFrame {
                await onSSEFrame(frame)
            }
        }

        func ensureMessageStart(inputTokens: Int = 0) async {
            guard emitSSE && !emittedMessageStart else { return }
            emittedMessageStart = true
            let startPayload: [String: Any] = [
                "type": "message_start",
                "message": [
                    "id": requestID,
                    "type": "message",
                    "role": "assistant",
                    "model": modelID,
                    "content": [],
                    "stop_reason": NSNull(),
                    "usage": [
                        "input_tokens": inputTokens,
                        "output_tokens": 0,
                    ],
                ],
            ]
            if let startData = try? JSONSerialization.data(withJSONObject: startPayload, options: [.sortedKeys]),
               let startStr = String(data: startData, encoding: .utf8) {
                await appendSSE("event: message_start\ndata: \(startStr)\n\n")
            }
        }

        do {
            for try await chunk in stream {
                await ensureMessageStart(inputTokens: chunk.cacheReport?.processedTokens ?? chunk.promptTokens)

                if !chunk.thinking.isEmpty {
                    fullThinking.append(chunk.thinking)
                    if emitSSE {
                        let idx: Int
                        if let existing = activeThinkingIndex {
                            idx = existing
                        } else {
                            idx = nextBlockIndex
                            nextBlockIndex += 1
                            activeThinkingIndex = idx
                            let blockStart: [String: Any] = [
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": [
                                    "type": "thinking",
                                    "thinking": "",
                                ],
                            ]
                            if let d = try? JSONSerialization.data(withJSONObject: blockStart, options: [.sortedKeys]),
                               let s = String(data: d, encoding: .utf8) {
                                await appendSSE("event: content_block_start\ndata: \(s)\n\n")
                            }
                        }
                        let deltaObj: [String: Any] = [
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": [
                                "type": "thinking_delta",
                                "thinking": chunk.thinking,
                            ],
                        ]
                        if let d = try? JSONSerialization.data(withJSONObject: deltaObj, options: [.sortedKeys]),
                           let s = String(data: d, encoding: .utf8) {
                            await appendSSE("event: content_block_delta\ndata: \(s)\n\n")
                        }
                    }
                }

                if !chunk.text.isEmpty {
                    fullText.append(chunk.text)
                    if emitSSE {
                        if let thinkIdx = activeThinkingIndex {
                            let stopThinking: [String: Any] = [
                                "type": "content_block_stop",
                                "index": thinkIdx,
                            ]
                            if let d = try? JSONSerialization.data(withJSONObject: stopThinking, options: [.sortedKeys]),
                               let s = String(data: d, encoding: .utf8) {
                                await appendSSE("event: content_block_stop\ndata: \(s)\n\n")
                            }
                            activeThinkingIndex = nil
                        }
                        let idx: Int
                        if let existing = activeTextIndex {
                            idx = existing
                        } else {
                            idx = nextBlockIndex
                            nextBlockIndex += 1
                            activeTextIndex = idx
                            let blockStart: [String: Any] = [
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": [
                                    "type": "text",
                                    "text": "",
                                ],
                            ]
                            if let d = try? JSONSerialization.data(withJSONObject: blockStart, options: [.sortedKeys]),
                               let s = String(data: d, encoding: .utf8) {
                                await appendSSE("event: content_block_start\ndata: \(s)\n\n")
                            }
                        }
                        let deltaObj: [String: Any] = [
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": [
                                "type": "text_delta",
                                "text": chunk.text,
                            ],
                        ]
                        if let d = try? JSONSerialization.data(withJSONObject: deltaObj, options: [.sortedKeys]),
                           let s = String(data: d, encoding: .utf8) {
                            await appendSSE("event: content_block_delta\ndata: \(s)\n\n")
                        }
                    }
                }
                finalChunk = chunk
            }
        } catch {
            if emitSSE && emittedMessageStart {
                let errPayload: [String: Any] = [
                    "type": "error",
                    "error": [
                        "type": "api_error",
                        "message": error.localizedDescription,
                    ],
                ]
                if let d = try? JSONSerialization.data(withJSONObject: errPayload, options: [.sortedKeys]),
                   let s = String(data: d, encoding: .utf8) {
                    await appendSSE("event: error\ndata: \(s)\n\n")
                }
                let usage = MessagesResponse.Usage(
                    inputTokens: finalChunk.cacheReport?.processedTokens ?? finalChunk.promptTokens,
                    outputTokens: finalChunk.tokenCount
                )
                let perf = MessagesResponse.PerformanceExtension(
                    generationTPS: finalChunk.generationTPS,
                    promptTPS: finalChunk.promptTPS,
                    computePlacement: finalChunk.activeComputePlacement
                )
                return (fullText, fullThinking, finalChunk, usage, perf, sseLines)
            }
            throw error
        }

        await ensureMessageStart(inputTokens: finalChunk.cacheReport?.processedTokens ?? finalChunk.promptTokens)

        let usage = MessagesResponse.Usage(
            inputTokens: finalChunk.cacheReport?.processedTokens ?? finalChunk.promptTokens,
            outputTokens: finalChunk.tokenCount,
            cacheReadInputTokens: finalChunk.cacheReport?.cachedTokens,
            thinkingTokens: finalChunk.thinkingTokens > 0 ? finalChunk.thinkingTokens : nil,
            contentTokens: finalChunk.contentTokens > 0 ? finalChunk.contentTokens : nil
        )
        let perfExt = MessagesResponse.PerformanceExtension(
            generationTPS: finalChunk.generationTPS,
            promptTPS: finalChunk.promptTPS,
            ttftMs: finalChunk.ttftMs,
            thinkingDurationMs: finalChunk.thinkingDurationMs,
            contentDurationMs: finalChunk.contentDurationMs,
            generationDurationMs: finalChunk.generationDurationMs,
            computePlacement: finalChunk.activeComputePlacement,
            cache: finalChunk.cacheReport,
            speculative: finalChunk.speculativeReport
        )

        if emitSSE {
            if let thinkIdx = activeThinkingIndex {
                let stopThinking: [String: Any] = ["type": "content_block_stop", "index": thinkIdx]
                if let d = try? JSONSerialization.data(withJSONObject: stopThinking, options: [.sortedKeys]),
                   let s = String(data: d, encoding: .utf8) {
                    await appendSSE("event: content_block_stop\ndata: \(s)\n\n")
                }
            }
            if let txtIdx = activeTextIndex {
                let stopText: [String: Any] = ["type": "content_block_stop", "index": txtIdx]
                if let d = try? JSONSerialization.data(withJSONObject: stopText, options: [.sortedKeys]),
                   let s = String(data: d, encoding: .utf8) {
                    await appendSSE("event: content_block_stop\ndata: \(s)\n\n")
                }
            }

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
                usage: usage,
                performance: perfExt
            )
            if let deltaData = try? JSONEncoder().encode(msgDelta),
               let deltaStr = String(data: deltaData, encoding: .utf8) {
                await appendSSE("event: message_delta\ndata: \(deltaStr)\n\n")
            }
            await appendSSE("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
        }

        return (fullText, fullThinking, finalChunk, usage, perfExt, sseLines)
    }

    /// Stateful `ConversationGenerationCoordinator` (`POST /v1/conversations/{id}/generate`).
    /// Supports `mode`: `"append"` / `"regenerate"` / `"continue"`, `409 generation_in_progress`
    /// per-conversation locking, content-addressed `media_blobs` externalization + hydration,
    /// `dropped_media` reporting when switching to a text-only model, `MessageStatsRecord`
    /// persistence, and disconnect-resilient server-owned completion ending with `heylook_saved`.
    private func handleConversationGenerate(
        requestID: String,
        conversationID: String,
        generateRequest: ConversationGenerateRequest,
        onSSEFrame: (@Sendable (String) async -> Void)? = nil
    ) async throws -> HeylookHTTPResponse {
        let provider = try await scheduler.provider(for: generateRequest.model)
        let desc = await provider.describe()

        let requestedDepth = generateRequest.thinkingDepth ?? generateRequest.sampling?.thinkingDepth
        if let requestedDepth, !requestedDepth.isEmpty,
           let controls = desc.thinkingControls {
            if controls.depthVariable == nil && controls.depthValues.isEmpty {
                return .error(
                    status: 400,
                    type: "invalid_request_error",
                    message: "Model '\(generateRequest.model)' does not support variable thinking_depth / reasoning_effort."
                )
            }
            if !controls.depthValues.isEmpty && !controls.depthValues.contains(requestedDepth) {
                return .error(
                    status: 400,
                    type: "invalid_request_error",
                    message: "Model '\(generateRequest.model)' does not support thinking_depth '\(requestedDepth)'. Allowed values: \(controls.depthValues.joined(separator: ", "))."
                )
            }
        }

        guard registry.tryAcquireConversationLock(
            conversationID: conversationID,
            requestID: requestID,
            modelID: generateRequest.model
        ) != nil else {
            return .error(
                status: 409,
                type: "generation_in_progress",
                message: "Conversation '\(conversationID)' already has an active generation."
            )
        }

        do {
            let mode = generateRequest.mode
            if try store.getConversation(id: conversationID) == nil {
                guard mode == .append else {
                    registry.unregister(id: requestID)
                    return .error(status: 404, type: "not_found_error", message: "Conversation '\(conversationID)' not found")
                }
                let titlePreview = generateRequest.content.isEmpty ? "New Conversation" : String(generateRequest.content.prefix(48))
                try store.upsertConversation(
                    ConversationRecord(
                        id: conversationID,
                        title: titlePreview,
                        systemPrompt: generateRequest.system ?? "",
                        defaultModelID: generateRequest.model
                    )
                )
            } else if let sysOverride = generateRequest.system {
                if var conv = try store.getConversation(id: conversationID) {
                    conv.systemPrompt = sysOverride
                    try store.upsertConversation(conv)
                }
            }

            let existingMessages = try store.listMessages(conversationID: conversationID, hydrateMedia: true)
            var savedUserMsg: MessageRecord?
            var targetAssistantMsg: MessageRecord?
            let assistantPos: Int
            var historyRecords: [MessageRecord] = []

            switch mode {
            case .regenerate:
                let target: MessageRecord?
                if let targetID = generateRequest.messageID {
                    target = existingMessages.first(where: { $0.id == targetID })
                } else {
                    target = existingMessages.last(where: { $0.role == "assistant" })
                }
                guard let targetRecord = target else {
                    registry.unregister(id: requestID)
                    return .error(status: 404, type: "not_found_error", message: "Target message for regenerate not found")
                }
                let regenPos = targetRecord.role == "assistant" ? targetRecord.position : (targetRecord.position + 1)
                _ = try store.deleteMessages(conversationID: conversationID, fromPosition: regenPos)
                historyRecords = try store.listMessages(conversationID: conversationID, hydrateMedia: true)
                assistantPos = regenPos

            case .continue:
                let target: MessageRecord?
                if let targetID = generateRequest.messageID {
                    target = existingMessages.first(where: { $0.id == targetID && $0.role == "assistant" })
                } else {
                    target = existingMessages.last(where: { $0.role == "assistant" })
                }
                guard let targetRecord = target else {
                    registry.unregister(id: requestID)
                    return .error(status: 404, type: "not_found_error", message: "Target assistant message for continue not found")
                }
                targetAssistantMsg = targetRecord
                historyRecords = existingMessages.filter { $0.position <= targetRecord.position }
                assistantPos = targetRecord.position

            case .append:
                let nextPos = (existingMessages.map(\.position).max() ?? -1) + 1
                let rawUserParts = generateRequest.userParts
                let userMsg = try store.appendMessage(
                    MessageRecord(
                        conversationID: conversationID,
                        position: nextPos,
                        role: "user",
                        content: generateRequest.content,
                        modelID: generateRequest.model,
                        contentBlocks: rawUserParts.isEmpty ? nil : rawUserParts
                    )
                )
                let hydratedUser = (try? store.getMessage(id: userMsg.id, hydrateMedia: true)) ?? userMsg
                savedUserMsg = userMsg
                historyRecords = existingMessages + [hydratedUser]
                assistantPos = nextPos + 1
            }

            // Hydrate media_blobs and strip unsupported media when switching to a text-only model
            var droppedMediaCount = 0
            var wireTurns: [MessagesRequest.MessageTurn] = []
            for rec in historyRecords {
                let rawParts = rec.contentBlocks ?? (rec.content.isEmpty ? [] : [.text(rec.content)])
                var keptAttachments: [MessagesRequest.MediaAttachment] = []
                var keptParts: [MessagesRequest.MessageTurn.ContentPart] = []

                for part in rawParts {
                    switch part {
                    case .media(let att):
                        if att.kind == "image" && !desc.supportsVision.value {
                            droppedMediaCount += 1
                        } else if att.kind == "audio" && !desc.supportsAudio.value {
                            droppedMediaCount += 1
                        } else {
                            keptAttachments.append(att)
                            keptParts.append(.media(att))
                        }
                    default:
                        keptParts.append(part)
                    }
                }
                if keptParts.isEmpty {
                    keptParts = [.text(rec.content)]
                }
                wireTurns.append(
                    MessagesRequest.MessageTurn(
                        role: rec.role,
                        content: rec.content,
                        thinking: rec.thinking,
                        attachments: keptAttachments,
                        parts: keptParts
                    )
                )
            }

            let conv = try store.getConversation(id: conversationID)
            var presetSampler: SamplerSettings? = nil
            if let presetName = generateRequest.presetName ?? conv?.appliedPresetID,
               let preset = (try? store.listPresets())?.first(where: { $0.id == presetName || $0.name.caseInsensitiveCompare(presetName) == .orderedSame }) {
                presetSampler = preset.settings
            }
            let convBaseOverlay = conv?.params ?? SamplerSettings()
            var reqOverlay = generateRequest.sampling ?? SamplerSettings()
            if generateRequest.thinking != nil { reqOverlay.thinking = generateRequest.thinking }
            if generateRequest.thinkingBudget != nil { reqOverlay.thinkingBudget = generateRequest.thinkingBudget }
            if generateRequest.thinkingDepth != nil { reqOverlay.thinkingDepth = generateRequest.thinkingDepth }
            let mergedOverlay = SamplerCascade.resolve(
                builtin: convBaseOverlay,
                request: reqOverlay
            ).effective

            let resolvedSampling = HeylookConfig.resolveEffectiveSampling(
                modelPath: desc.modelPath,
                preset: presetSampler,
                request: mergedOverlay
            )
            let msgReq = MessagesRequest(
                model: generateRequest.model,
                messages: wireTurns,
                system: generateRequest.system ?? conv?.systemPrompt,
                stream: true,
                thinking: resolvedSampling.effective.thinking,
                thinkingDepth: resolvedSampling.effective.thinkingDepth,
                thinkingBudget: resolvedSampling.effective.thinkingBudget,
                stopSequences: resolvedSampling.effective.stopSequences,
                responseFormat: generateRequest.responseFormat
            )

            let continuePrefixText = targetAssistantMsg?.content ?? ""
            let continuePrefixThinking = targetAssistantMsg?.thinking ?? ""
            let targetAssistantID = targetAssistantMsg?.id ?? UUID().uuidString
            let userMsgID = savedUserMsg?.id

            // Unstructured server-owned task ensures persistence even if HTTP caller disconnects mid-stream
            let serverTask = Task { [store, scheduler, registry, droppedMediaCount] () throws -> Data in
                defer {
                    registry.unregister(id: requestID)
                }
                return try await scheduler.withComputeLease(for: generateRequest.model) {
                    let stream = await provider.generate(
                        requestID: requestID,
                        request: msgReq,
                        effectiveSampling: resolvedSampling.effective,
                        registry: registry
                    )
                    var consumed = try await Self.consumeStreamAndFormatSSE(
                        requestID: requestID,
                        modelID: generateRequest.model,
                        stream: stream,
                        emitSSE: true,
                        onSSEFrame: onSSEFrame
                    )

                    let mergedText = continuePrefixText + consumed.fullText
                    let mergedThinking = continuePrefixThinking + consumed.fullThinking
                    let savedAssistant = try store.appendMessage(
                        MessageRecord(
                            id: targetAssistantID,
                            conversationID: conversationID,
                            position: assistantPos,
                            role: "assistant",
                            content: mergedText,
                            thinking: mergedThinking.isEmpty ? nil : mergedThinking,
                            modelID: generateRequest.model,
                            stopReason: consumed.finalChunk.stopReason ?? .endTurn,
                            cacheReport: consumed.finalChunk.cacheReport,
                            speculativeReport: consumed.finalChunk.speculativeReport,
                            generationTPS: consumed.finalChunk.generationTPS,
                            computePlacement: consumed.finalChunk.activeComputePlacement
                        )
                    )

                    try? store.upsertMessageStats(
                        MessageStatsRecord(
                            messageID: savedAssistant.id,
                            conversationID: conversationID,
                            modelID: generateRequest.model,
                            promptTokens: consumed.finalChunk.promptTokens,
                            outputTokens: consumed.finalChunk.tokenCount,
                            thinkingTokens: consumed.finalChunk.thinkingTokens,
                            contentTokens: consumed.finalChunk.contentTokens,
                            cachedTokens: consumed.finalChunk.cacheReport?.cachedTokens ?? 0,
                            promptTPS: consumed.finalChunk.promptTPS,
                            generationTPS: consumed.finalChunk.generationTPS,
                            ttftMs: consumed.finalChunk.ttftMs,
                            thinkingDurationMs: consumed.finalChunk.thinkingDurationMs,
                            contentDurationMs: consumed.finalChunk.contentDurationMs,
                            generationDurationMs: consumed.finalChunk.generationDurationMs
                        )
                    )

                    let savedEvent = HeylookSavedEvent(
                        conversationID: conversationID,
                        userMessageID: userMsgID,
                        assistantMessageID: savedAssistant.id,
                        position: assistantPos,
                        stopReason: consumed.finalChunk.stopReason ?? .endTurn,
                        mode: mode,
                        droppedMedia: droppedMediaCount
                    )
                    let savedJSON = try String(data: JSONEncoder().encode(savedEvent), encoding: .utf8) ?? "{}"
                    let savedFrame = "event: heylook_saved\ndata: \(savedJSON)\n\n"
                    consumed.sseLines.append(savedFrame)
                    if let onSSEFrame { await onSSEFrame(savedFrame) }
                    return Data(consumed.sseLines.joined().utf8)
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
        } catch {
            registry.unregister(id: requestID)
            throw error
        }
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
        params.allowLocalEndpointReuse = true
        let isWildcard = host.isEmpty || host == "0.0.0.0" || host == "::" || host == "*"
        let nwListener: NWListener
        if !isWildcard {
            let bindAddr = (host == "localhost") ? "127.0.0.1" : host
            params.requiredLocalEndpoint = NWEndpoint.hostPort(
                host: NWEndpoint.Host(bindAddr),
                port: nwPort
            )
            // When `requiredLocalEndpoint` already carries a specific port, passing `on: nwPort`
            // as well causes `libnetwork` to fail with `NWError.posix(.EINVAL)` (error 22).
            nwListener = try NWListener(using: params)
        } else {
            nwListener = try NWListener(using: params, on: nwPort)
        }
        if advertiseBonjour && isWildcard {
            nwListener.service = NWListener.Service(name: "Heylook Server", type: "_heylook._tcp")
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
                await Self.dispatchRawHTTPBuffer(
                    finalBuffer,
                    router: router,
                    sendBytes: { payload in
                        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                            connection.send(content: payload, completion: .contentProcessed { _ in
                                cont.resume()
                            })
                        }
                    },
                    closeConnection: {
                        connection.cancel()
                    }
                )
            }
        }
    }

    /// Shared live-connection HTTP/SSE dispatcher used by `receiveHTTPRequest` and wire contract tests.
    /// Ensures chunked SSE responses always include a valid `X-Request-ID` even when omitted by the client.
    internal nonisolated static func dispatchRawHTTPBuffer(
        _ finalBuffer: Data,
        router: HeylookRouter,
        sendBytes: @escaping @Sendable (Data) async -> Void,
        closeConnection: @escaping @Sendable () -> Void
    ) async {
        let incomingReqID = Self.peekRequestID(in: finalBuffer)
        let fallbackReqID = incomingReqID == nil
            ? "req_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16))"
            : nil
        let sseStreamBox = SSEConnectionStreamer(
            requestID: incomingReqID ?? fallbackReqID,
            sendBytes: sendBytes,
            closeConnection: closeConnection
        )
        let response = await Self.parseAndDispatch(
            rawData: finalBuffer,
            router: router,
            fallbackRequestID: fallbackReqID,
            onSSEFrame: { frame in
                await sseStreamBox.sendFrame(frame)
            }
        )
        let didStream = await sseStreamBox.hasStreamed
        if didStream {
            await sseStreamBox.finish()
        } else {
            let serialized = Self.serializeHTTPResponse(response)
            await sendBytes(serialized)
            closeConnection()
        }
    }

    private nonisolated static func peekRequestID(in rawData: Data) -> String? {
        let separator = Data("\r\n\r\n".utf8)
        let headerData = rawData.range(of: separator).map { rawData.subdata(in: 0..<$0.lowerBound) } ?? rawData
        let headerText = String(decoding: headerData, as: UTF8.self)
        for line in headerText.components(separatedBy: "\r\n").dropFirst() {
            let parts = line.split(separator: ":", maxSplits: 1)
            if parts.count == 2,
               parts[0].trimmingCharacters(in: .whitespaces).lowercased() == "x-request-id" {
                return parts[1].trimmingCharacters(in: .whitespaces)
            }
        }
        return nil
    }

    private nonisolated static func parseAndDispatch(
        rawData: Data,
        router: HeylookRouter,
        fallbackRequestID: String? = nil,
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
        if let fallbackRequestID,
           !headers.keys.contains(where: { $0.lowercased() == "x-request-id" }) {
            headers["X-Request-ID"] = fallbackRequestID
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

    public nonisolated static func serializeHTTPResponse(_ response: HeylookHTTPResponse) -> Data {
        let reason = HeylookHTTPResponse.statusReasonPhrase(for: response.status)
        var headerLines = ["HTTP/1.1 \(response.status) \(reason)"]
        var mergedHeaders = response.headers
        mergedHeaders["Content-Length"] = "\(response.body.count)"
        for (k, v) in mergedHeaders.sorted(by: { $0.key < $1.key }) {
            headerLines.append("\(k): \(v)")
        }
        let head = headerLines.joined(separator: "\r\n") + "\r\n\r\n"
        var out = Data(head.utf8)
        out.append(response.body)
        return out
    }
}

/// Actor that flushes SSE frames incrementally over an `NWConnection` using HTTP/1.1
/// `Transfer-Encoding: chunked` framing + periodic `: ping` keepalive comments.
private actor SSEConnectionStreamer {
    private let requestID: String?
    private let sendBytes: @Sendable (Data) async -> Void
    private let closeConnection: @Sendable () -> Void
    public private(set) var hasStreamed: Bool = false
    private var pingTask: Task<Void, Never>?

    init(
        requestID: String?,
        sendBytes: @escaping @Sendable (Data) async -> Void,
        closeConnection: @escaping @Sendable () -> Void
    ) {
        self.requestID = requestID
        self.sendBytes = sendBytes
        self.closeConnection = closeConnection
    }

    func sendFrame(_ frame: String) async {
        let frameBytes = Data(frame.utf8)
        guard !frameBytes.isEmpty else { return }

        var wirePayload = Data()
        if !hasStreamed {
            hasStreamed = true
            var headerLines = [
                "HTTP/1.1 200 OK",
                "Content-Type: text/event-stream",
                "Cache-Control: no-cache",
                "Connection: close",
                "Transfer-Encoding: chunked",
            ]
            if let requestID, !requestID.isEmpty {
                headerLines.append("X-Request-ID: \(requestID)")
            }
            headerLines.append("")
            headerLines.append("")
            wirePayload.append(Data(headerLines.joined(separator: "\r\n").utf8))
            startKeepaliveLoop()
        }

        let hexLen = String(frameBytes.count, radix: 16)
        wirePayload.append(Data("\(hexLen)\r\n".utf8))
        wirePayload.append(frameBytes)
        wirePayload.append(Data("\r\n".utf8))

        await sendBytes(wirePayload)
    }

    private func startKeepaliveLoop() {
        pingTask?.cancel()
        pingTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(15))
                if Task.isCancelled { break }
                await self?.sendFrame(": ping\n\n")
            }
        }
    }

    func finish() async {
        pingTask?.cancel()
        pingTask = nil
        if hasStreamed {
            let trailer = Data("0\r\n\r\n".utf8)
            await sendBytes(trailer)
        }
        closeConnection()
    }
}

/// Represents a `heylook` server discovered via Bonjour (`_heylook._tcp`) or configured over LAN/Tailscale.
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
/// (`matryoshka_recall_probe`).
///
/// Truthful Runtime: By default (`allowSyntheticSelfTest == false`), `runAll()` requires genuine
/// discovered model weights and compiled ANE `.mlmodelc` towers and reports honest `FAIL` status
/// when absent rather than silently masquerading synthetic echo/hash fixtures as live model evals.
/// Pass `allowSyntheticSelfTest: true` ONLY for offline pipeline contract verification.
public enum EvalLabSuite {
    public struct ProbeSummary: Codable, Sendable, Equatable, Identifiable {
        public var id: String { name }
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

    public static func runAll(allowSyntheticSelfTest: Bool = false) async throws -> SuiteReport {
        var probes: [ProbeSummary] = []

        if !allowSyntheticSelfTest {
            let discovered = HeylookConfig.discoverConfiguredModels()
            let embEngine = NativeMLXEmbeddingGemma2.makeBestAvailableEngine(
                computeOptions: .hybridANETowers,
                allowSyntheticFallback: false
            )
            var embAvailable = true
            do {
                _ = try await embEngine.embedBatch([.text("probe", task: .query)], dimensions: 128)
            } catch {
                embAvailable = false
            }

            if discovered.isEmpty || !embAvailable {
                if discovered.isEmpty {
                    probes.append(
                        ProbeSummary(
                            name: "chain_probe",
                            passed: false,
                            detail: "No configured model weights discovered in scan_folders (pass --synthetic-self-test for offline pipeline check)"
                        )
                    )
                    probes.append(
                        ProbeSummary(
                            name: "vlm_parity_probe",
                            passed: false,
                            detail: "No configured VLM weights discovered in scan_folders"
                        )
                    )
                } else {
                    probes.append(
                        ProbeSummary(
                            name: "chain_probe",
                            passed: false,
                            detail: "Live model evaluation requires native MLX/GGUF runtime execution against \(discovered.first!.modelID)"
                        )
                    )
                    probes.append(
                        ProbeSummary(
                            name: "vlm_parity_probe",
                            passed: false,
                            detail: "Live VLM evaluation requires loaded vision model weights"
                        )
                    )
                }
                if !embAvailable {
                    probes.append(
                        ProbeSummary(
                            name: "matryoshka_recall_probe",
                            passed: false,
                            detail: "Compiled EmbeddingGemma 2 ANE .mlmodelc towers & tokenizer.json not found"
                        )
                    )
                }
                return SuiteReport(
                    allPassed: false,
                    probes: probes,
                    matryoshkaBenchmarks: []
                )
            }
        }

        // Offline pipeline contract self-test (`allowSyntheticSelfTest == true`)
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
            && kvState.speculativeMode == "draft-mtp"
            && lastChunk3.speculativeReport == nil
        probes.append(
            ProbeSummary(
                name: "chain_probe",
                passed: chainPassed,
                detail: "cold->reused(\(lastChunk2.cacheReport?.cachedTokens ?? 0)t)->branch_slice(trunc=\(kvState.sliceTruncations), specCfg=\(kvState.speculativeMode ?? "none"))"
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
        await scheduler.register(
            provider: MLXSwiftProvider(description: vlmDesc, allowSyntheticFallback: true),
            description: vlmDesc
        )
        await scheduler.register(
            provider: MLXSwiftProvider(description: txtDesc, allowSyntheticFallback: true),
            description: txtDesc
        )
        let router = HeylookRouter(store: store, scheduler: scheduler, allowSyntheticFallback: true)

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
