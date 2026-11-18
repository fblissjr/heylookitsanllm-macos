import Foundation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

/// Structured error raised by `RemoteWorkbenchBackend` when a remote `HeylookServer` peer
/// returns an HTTP error, invalid payload, or transport failure.
public enum RemoteWorkbenchError: LocalizedError, Sendable, Equatable {
    case invalidBaseURL(String)
    case httpStatus(statusCode: Int, code: String?, message: String, retryAfter: String?)
    case invalidPayload(String)

    public var errorDescription: String? {
        switch self {
        case .invalidBaseURL(let raw):
            return "Invalid remote server URL '\(raw)'."
        case .httpStatus(let statusCode, let code, let message, _):
            if let code, !code.isEmpty {
                return "Remote HTTP \(statusCode) (\(code)): \(message)"
            }
            return "Remote HTTP \(statusCode): \(message)"
        case .invalidPayload(let detail):
            return "Remote server returned unexpected payload: \(detail)"
        }
    }
}

/// Truthful HTTP/SSE client implementation of `WorkbenchBackend` plus remote workbench CRUD/telemetry
/// endpoints (`/v1/conversations/*`, `/v1/notebooks/*`, `/v1/presets/*`, `/v1/admin/*`, `/v1/system/metrics`,
/// `/v1/performance/profile/{range}`, `/v1/audio/transcriptions`, `/v1/requests/*`).
///
/// Parses real `PublicModelWireRow` and `AdminModelWireRow` `EngineContract` metadata and fails loudly
/// whenever the peer is unreachable or returns an error — never synthesizing fake fallback models.
public final class RemoteWorkbenchBackend: WorkbenchBackend, Sendable {
    public let baseURL: URL
    public let adminToken: String?
    public let session: URLSession

    private actor MetadataCache {
        var publicRows: [String: PublicModelWireRow] = [:]
        var adminRows: [String: AdminModelWireRow] = [:]

        func setPublicRows(_ rows: [PublicModelWireRow]) {
            var next: [String: PublicModelWireRow] = [:]
            for row in rows {
                next[row.id] = row
            }
            publicRows = next
        }

        func setAdminRows(_ rows: [AdminModelWireRow]) {
            var next: [String: AdminModelWireRow] = [:]
            for row in rows {
                next[row.id] = row
            }
            adminRows = next
        }

        func upsertAdminRow(_ row: AdminModelWireRow) {
            adminRows[row.id] = row
        }

        func contract(for modelID: String) -> EngineContract? {
            adminRows[modelID]?.engine ?? publicRows[modelID]?.engine
        }

        func samplerDefaults(for modelID: String) -> [String: JSONValue]? {
            adminRows[modelID]?.samplerDefaults ?? publicRows[modelID]?.samplerDefaults
        }

        func samplerSources(for modelID: String) -> [String: String]? {
            adminRows[modelID]?.samplerSources ?? publicRows[modelID]?.samplerSources
        }

        func publicRow(for modelID: String) -> PublicModelWireRow? {
            publicRows[modelID]
        }

        func adminRow(for modelID: String) -> AdminModelWireRow? {
            adminRows[modelID]
        }

        func cachedAdminRows() -> [AdminModelWireRow] {
            Array(adminRows.values).sorted { $0.id.localizedCaseInsensitiveCompare($1.id) == .orderedAscending }
        }
    }

    private let cache = MetadataCache()

    public init(baseURL: URL, adminToken: String? = nil, session: URLSession = .shared) {
        self.baseURL = baseURL
        let trimmed = adminToken?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.adminToken = (trimmed?.isEmpty == true) ? nil : trimmed
        self.session = session
    }

    // MARK: - Cached Remote Contract Queries

    public func engineContract(for modelID: String) async -> EngineContract? {
        await cache.contract(for: modelID)
    }

    public func samplerDefaults(for modelID: String) async -> (defaults: [String: JSONValue], sources: [String: String])? {
        guard let defaults = await cache.samplerDefaults(for: modelID) else { return nil }
        let sources = await cache.samplerSources(for: modelID) ?? [:]
        return (defaults, sources)
    }

    public func publicWireRow(for modelID: String) async -> PublicModelWireRow? {
        await cache.publicRow(for: modelID)
    }

    public func adminWireRow(for modelID: String) async -> AdminModelWireRow? {
        await cache.adminRow(for: modelID)
    }

    // MARK: - Truthful Remote Catalog Discovery

    /// Fetches `/v1/models` (and optionally enriches with `/v1/admin/models` when accessible) and maps
    /// each entry strictly from its returned `EngineContract`, capabilities, and modalities.
    public func fetchModelDescriptions() async throws -> [EngineDescription] {
        let publicURL = endpoint(["v1", "models"])
        let (publicData, publicResp) = try await performDataRequest(makeRequest(url: publicURL, method: "GET"))
        try validateHTTPResponse(publicResp, data: publicData, allowCodes: [200])

        let publicRows = try decodePublicModelList(from: publicData)
        await cache.setPublicRows(publicRows)

        // Attempt admin enrichment so loaded flags, config provenance, and tags stay accurate when permitted.
        var adminMap: [String: AdminModelWireRow] = [:]
        let adminURL = endpoint(["v1", "admin", "models"])
        if let (adminData, adminResp) = try? await performDataRequest(makeRequest(url: adminURL, method: "GET", includeAdminToken: true)),
           let http = adminResp as? HTTPURLResponse,
           (200..<300).contains(http.statusCode),
           let adminRows = try? decodeAdminModelList(from: adminData) {
            await cache.setAdminRows(adminRows)
            for row in adminRows {
                adminMap[row.id] = row
            }
        }

        if publicRows.isEmpty && !adminMap.isEmpty {
            return adminMap.values
                .sorted { $0.id.localizedCaseInsensitiveCompare($1.id) == .orderedAscending }
                .map { Self.engineDescription(fromAdmin: $0, baseURL: baseURL) }
        }

        return publicRows.map { pub in
            Self.engineDescription(fromPublic: pub, admin: adminMap[pub.id], baseURL: baseURL)
        }
    }

    public static func engineDescription(
        fromPublic row: PublicModelWireRow,
        admin: AdminModelWireRow? = nil,
        baseURL: URL
    ) -> EngineDescription {
        let contract = admin?.engine ?? row.engine
        let capabilities = admin?.capabilities ?? row.capabilities
        let modalities = row.modalities
        let loaded = admin?.loaded ?? false
        let tags = admin?.tags ?? []
        return makeEngineDescription(
            id: row.id,
            baseURL: baseURL,
            modalities: modalities,
            capabilities: capabilities,
            thinkingDefault: admin?.thinkingDefault ?? row.thinkingDefault,
            tags: tags,
            loaded: loaded,
            contract: contract
        )
    }

    public static func engineDescription(fromAdmin row: AdminModelWireRow, baseURL: URL) -> EngineDescription {
        makeEngineDescription(
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

    private static func makeEngineDescription(
        id: String,
        baseURL: URL,
        modalities: [String],
        capabilities: [String],
        thinkingDefault: Bool,
        tags: [String],
        loaded: Bool,
        contract: EngineContract
    ) -> EngineDescription {
        let supportsVision = modalities.contains("vision") || capabilities.contains("vision")
        let supportsAudio = modalities.contains("audio") || capabilities.contains("audio")
        let supportsEmbeddings = contract.embedding != nil
            || modalities.contains("embeddings")
            || capabilities.contains("embeddings")
        let supportsThinking = capabilities.contains("thinking") || contract.thinking != nil
        let contextLength = contract.context.running.value ?? contract.context.length.value ?? 0
        let contextProv: ProvenanceSource = (contract.context.running.value != nil)
            ? contract.context.running.provenance.legacySource
            : contract.context.length.provenance.legacySource

        let controls = makeThinkingControls(
            contract: contract,
            capabilities: capabilities,
            thinkingDefault: thinkingDefault,
            supportsThinking: supportsThinking
        )

        let drafter = contract.speculative.drafter.value
        let runtimeArch = contract.runtime.value ?? "remote"
        let templateOrigin = contract.template.origin.value ?? "remote"

        let allTags = Array(Set(tags + ["remote"])).sorted()
        let quantizationLabel = contract.settings["quantization"]?.value?.stringValue ?? "remote"

        return EngineDescription(
            modelID: id,
            provider: .remote,
            modelPath: baseURL.absoluteString,
            architecture: ProvenanceField(
                value: runtimeArch,
                provenance: contract.runtime.provenance.legacySource,
                effect: .reload
            ),
            quantization: ProvenanceField(
                value: quantizationLabel,
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
                value: contextLength,
                provenance: contextProv,
                effect: .reload
            ),
            flashAttention: ProvenanceField(
                value: false,
                provenance: .notApplicable,
                effect: .reload
            ),
            supportsVision: ProvenanceField(
                value: supportsVision,
                provenance: .observed,
                effect: .reload
            ),
            supportsAudio: ProvenanceField(
                value: supportsAudio,
                provenance: .observed,
                effect: .reload
            ),
            supportsEmbeddings: ProvenanceField(
                value: supportsEmbeddings,
                provenance: .observed,
                effect: .reload
            ),
            supportsThinking: ProvenanceField(
                value: supportsThinking,
                provenance: .observed,
                effect: .perRequest
            ),
            thinkingBudget: ProvenanceField(
                value: nil,
                provenance: (contract.thinking?.budget != nil) ? .observed : .notApplicable,
                effect: .perRequest
            ),
            reasoningFormat: ProvenanceField(
                value: supportsThinking ? "tag_pair" : "none",
                provenance: .observed,
                effect: .reload
            ),
            thinkingControls: controls,
            templateSource: ProvenanceField(
                value: templateOrigin,
                provenance: contract.template.origin.provenance.legacySource,
                effect: .reload
            ),
            speculativeDrafter: ProvenanceField(
                value: drafter,
                provenance: contract.speculative.drafter.provenance.legacySource,
                effect: .reload
            ),
            estimatedMemoryMB: 0,
            tags: allTags
        )
    }

    private static func makeThinkingControls(
        contract: EngineContract,
        capabilities: [String],
        thinkingDefault: Bool,
        supportsThinking: Bool
    ) -> ThinkingControls? {
        guard supportsThinking || contract.thinking != nil else {
            return .unsupported
        }
        let tf = contract.thinking
        let depth = tf?.depth
        let offValues = depth?.off ?? []
        let hasSwitch = tf?.switchVariable != nil || capabilities.contains("thinking")
        let canDisable = hasSwitch || !offValues.isEmpty
        return ThinkingControls(
            supportsThinking: supportsThinking || tf != nil,
            canDisableThinking: canDisable,
            defaultEnabled: thinkingDefault,
            switchVariable: tf?.switchVariable,
            depthVariable: depth?.variable,
            depthValues: depth?.values ?? [],
            depthAliases: depth?.aliases ?? [:],
            depthDefault: depth?.defaultValue,
            depthUnknown: depth?.unknown,
            depthChangesPrefix: depth?.changesPrefix,
            depthOffValues: offValues,
            budgetEnforced: tf?.budget?.enforced,
            budgetReason: tf?.budget?.reason,
            templateSourceName: contract.template.origin.value
        )
    }

    // MARK: - WorkbenchBackend Conformance

    public func listPublicModels() async -> [PublicModelWireRow] {
        do {
            let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "models"]), method: "GET"))
            try validateHTTPResponse(resp, data: data, allowCodes: [200])
            let rows = try decodePublicModelList(from: data)
            await cache.setPublicRows(rows)
            return rows
        } catch {
            return []
        }
    }

    public func listAdminModels() async -> [AdminModelWireRow] {
        do {
            let (data, resp) = try await performDataRequest(
                makeRequest(url: endpoint(["v1", "admin", "models"]), method: "GET", includeAdminToken: true)
            )
            try validateHTTPResponse(resp, data: data, allowCodes: [200])
            let rows = try decodeAdminModelList(from: data)
            await cache.setAdminRows(rows)
            return rows
        } catch {
            return await cache.cachedAdminRows()
        }
    }

    public func adminModel(id: String) async throws -> AdminModelWireRow {
        let url = endpoint(["v1", "admin", "models"] + modelPathComponents(id))
        let (data, resp) = try await performDataRequest(makeRequest(url: url, method: "GET", includeAdminToken: true))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        let row = try JSONDecoder().decode(AdminModelWireRow.self, from: data)
        await cache.upsertAdminRow(row)
        return row
    }

    public func loadModel(id: String, warm: Bool = false) async throws -> ModelLoadResult {
        var components = URLComponents(url: endpoint(["v1", "models"] + modelPathComponents(id) + ["load"]), resolvingAgainstBaseURL: false)!
        if warm {
            components.queryItems = [URLQueryItem(name: "warm", value: "true")]
        }
        guard let url = components.url else { throw RemoteWorkbenchError.invalidBaseURL(baseURL.absoluteString) }
        let (data, resp) = try await performDataRequest(makeRequest(url: url, method: "POST"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(ModelLoadResult.self, from: data)
    }

    public func reloadModel(id: String, loadSettings: [String: JSONValue]?, warm: Bool = false) async throws -> ModelLoadResult {
        var components = URLComponents(url: endpoint(["v1", "admin", "models"] + modelPathComponents(id) + ["reload"]), resolvingAgainstBaseURL: false)!
        if warm {
            components.queryItems = [URLQueryItem(name: "warm", value: "true")]
        }
        guard let url = components.url else { throw RemoteWorkbenchError.invalidBaseURL(baseURL.absoluteString) }
        let body: Data? = try loadSettings.map { try JSONEncoder().encode($0) }
        let (data, resp) = try await performDataRequest(
            makeRequest(url: url, method: "POST", body: body, includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(ModelLoadResult.self, from: data)
    }

    public func unloadModel(id: String) async throws -> ModelUnloadResult {
        let url = endpoint(["v1", "admin", "models"] + modelPathComponents(id) + ["unload"])
        let (data, resp) = try await performDataRequest(makeRequest(url: url, method: "POST", includeAdminToken: true))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(ModelUnloadResult.self, from: data)
    }

    public func modelStatus(id: String) async -> [String: JSONValue] {
        let url = endpoint(["v1", "admin", "models"] + modelPathComponents(id) + ["status"])
        guard let (data, resp) = try? await performDataRequest(makeRequest(url: url, method: "GET", includeAdminToken: true)),
              let http = resp as? HTTPURLResponse,
              (200..<300).contains(http.statusCode),
              let dict = try? JSONDecoder().decode([String: JSONValue].self, from: data) else {
            return [:]
        }
        return dict
    }

    public func assessModelFit(
        id: String,
        configOverrides: [String: JSONValue],
        headroomGB: Double
    ) async throws -> RAMFitEvaluator.FitReport {
        struct Body: Encodable {
            let configOverrides: [String: JSONValue]
            let headroomGB: Double
            enum CodingKeys: String, CodingKey {
                case configOverrides = "config_overrides"
                case headroomGB = "headroom_gb"
            }
        }
        let url = endpoint(["v1", "admin", "models"] + modelPathComponents(id) + ["fit"])
        let encoded = try JSONEncoder().encode(Body(configOverrides: configOverrides, headroomGB: headroomGB))
        let (data, resp) = try await performDataRequest(
            makeRequest(url: url, method: "POST", body: encoded, includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(RAMFitEvaluator.FitReport.self, from: data)
    }

    public func planImages(id: String, sizes: [[Int]]) async throws -> ImagePlanWireResponse {
        let url = endpoint(["v1", "models"] + modelPathComponents(id) + ["image-plan"])
        let encoded = try JSONEncoder().encode(ImagePlanRequestPayload(sizes: sizes))
        let (data, resp) = try await performDataRequest(makeRequest(url: url, method: "POST", body: encoded))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(ImagePlanWireResponse.self, from: data)
    }

    public func previewPrompt(
        conversationID: String,
        request: PromptPreviewRequestPayload
    ) async throws -> PromptPreviewWireResponse {
        let url = endpoint(["v1", "conversations", conversationID, "prompt"])
        let encoded = try JSONEncoder().encode(request)
        let (data, resp) = try await performDataRequest(makeRequest(url: url, method: "POST", body: encoded))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(PromptPreviewWireResponse.self, from: data)
    }

    public func generateInConversation(
        conversationID: String,
        request: ConversationGenerateRequest,
        requestID: String,
        detachOnSubscriberExit: Bool,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> HeylookSavedEvent {
        let url = endpoint(["v1", "conversations", conversationID, "generate"])
        let bodyData = try JSONEncoder().encode(request)
        var req = makeRequest(url: url, method: "POST", body: bodyData)
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        req.setValue(requestID, forHTTPHeaderField: "X-Request-ID")

        if onSSEFrame == nil {
            let (data, resp) = try await performDataRequest(req)
            try validateHTTPResponse(resp, data: data, allowCodes: [200])
            return try JSONDecoder().decode(HeylookSavedEvent.self, from: data)
        }

        let (bytes, resp) = try await session.bytes(for: req)
        if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            var errData = Data()
            for try await byte in bytes {
                errData.append(byte)
                if errData.count >= 8192 { break }
            }
            try validateHTTPResponse(resp, data: errData, allowCodes: [200])
        }

        var currentEvent = ""
        var currentDataLines: [String] = []
        var savedEvent: HeylookSavedEvent?
        var streamErrorMessage: String?

        func flushFrame() async throws {
            guard !currentDataLines.isEmpty || !currentEvent.isEmpty else { return }
            let dataPayload = currentDataLines.joined(separator: "\n")
            let rawFrame: String
            if currentEvent.isEmpty {
                rawFrame = "data: \(dataPayload)\n\n"
            } else {
                rawFrame = "event: \(currentEvent)\ndata: \(dataPayload)\n\n"
            }
            if let onSSEFrame {
                await onSSEFrame(rawFrame)
            }
            if currentEvent == "heylook_saved",
               let payloadData = dataPayload.data(using: .utf8),
               let decoded = try? JSONDecoder().decode(HeylookSavedEvent.self, from: payloadData) {
                savedEvent = decoded
            } else if currentEvent == "error",
                      let payloadData = dataPayload.data(using: .utf8),
                      let errObj = try? JSONDecoder().decode(StreamErrorSSEEvent.self, from: payloadData) {
                streamErrorMessage = errObj.error.message
            }
            currentEvent = ""
            currentDataLines.removeAll(keepingCapacity: true)
        }

        for try await line in bytes.lines {
            if line.isEmpty {
                try await flushFrame()
            } else if line.hasPrefix("event:") {
                if !currentDataLines.isEmpty || !currentEvent.isEmpty {
                    try await flushFrame()
                }
                currentEvent = line.dropFirst("event:".count).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("data:") {
                let raw = line.dropFirst("data:".count)
                let trimmed = raw.first == " " ? String(raw.dropFirst()) : String(raw)
                currentDataLines.append(trimmed)
            }
        }
        try await flushFrame()

        if let streamErrorMessage {
            throw RemoteWorkbenchError.httpStatus(statusCode: 500, code: "stream_error", message: streamErrorMessage, retryAfter: nil)
        }
        if let savedEvent {
            return savedEvent
        }
        throw RemoteWorkbenchError.invalidPayload("Conversation stream ended without terminal heylook_saved event.")
    }

    public func createMessageResponse(
        request: MessagesRequest,
        requestID: String,
        cancellation: StreamCancellationSignal,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> MessagesResponse? {
        let url = endpoint(["v1", "messages"])
        var wireReq = request
        let streaming = (request.stream ?? false) || (onSSEFrame != nil)
        wireReq.stream = streaming
        let bodyData = try JSONEncoder().encode(wireReq)
        var req = makeRequest(url: url, method: "POST", body: bodyData)
        req.setValue(requestID, forHTTPHeaderField: "X-Request-ID")

        if !streaming {
            let (data, resp) = try await performDataRequest(req)
            try validateHTTPResponse(resp, data: data, allowCodes: [200])
            return try JSONDecoder().decode(MessagesResponse.self, from: data)
        }

        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        let (bytes, resp) = try await session.bytes(for: req)
        if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            var errData = Data()
            for try await byte in bytes {
                errData.append(byte)
                if errData.count >= 8192 { break }
            }
            try validateHTTPResponse(resp, data: errData, allowCodes: [200])
        }

        var currentEvent = ""
        var currentDataLines: [String] = []
        var sentAbort = false

        func flushFrame() async {
            guard !currentDataLines.isEmpty || !currentEvent.isEmpty else { return }
            let dataPayload = currentDataLines.joined(separator: "\n")
            let rawFrame = currentEvent.isEmpty
                ? "data: \(dataPayload)\n\n"
                : "event: \(currentEvent)\ndata: \(dataPayload)\n\n"
            await onSSEFrame?(rawFrame)
            currentEvent = ""
            currentDataLines.removeAll(keepingCapacity: true)
        }

        for try await line in bytes.lines {
            if cancellation.isCancelled || Task.isCancelled {
                if !sentAbort {
                    sentAbort = true
                    try? await abortRequest(id: requestID)
                }
                break
            }
            if line.isEmpty {
                await flushFrame()
            } else if line.hasPrefix("event:") {
                if !currentDataLines.isEmpty || !currentEvent.isEmpty {
                    await flushFrame()
                }
                currentEvent = line.dropFirst("event:".count).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("data:") {
                let raw = line.dropFirst("data:".count)
                let trimmed = raw.first == " " ? String(raw.dropFirst()) : String(raw)
                currentDataLines.append(trimmed)
            }
        }
        await flushFrame()
        return nil
    }

    public func embed(request: EmbeddingWireRequest, requestID: String) async throws -> EmbeddingWireResponse {
        let url = endpoint(["v1", "embeddings"])
        let itemsJSON: [JSONValue] = request.items.map { item in
            var obj: [String: JSONValue] = [
                "content": encodeContentPartsToJSON(item.content)
            ]
            if item.hasExplicitPromptName, let name = item.promptName {
                obj["prompt_name"] = .string(name)
            }
            return .object(obj)
        }
        var bodyObj: [String: JSONValue] = [
            "model": .string(request.model),
            "input": .array(itemsJSON)
        ]
        if let dims = request.dimensions {
            bodyObj["dimensions"] = .int(dims)
        }
        let bodyData = try JSONEncoder().encode(bodyObj)
        var req = makeRequest(url: url, method: "POST", body: bodyData)
        req.setValue(requestID, forHTTPHeaderField: "X-Request-ID")
        let (data, resp) = try await performDataRequest(req)
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(EmbeddingWireResponse.self, from: data)
    }

    public func getChatTemplate(id: String) async throws -> HeylookConfig.ChatTemplateState {
        let url = endpoint(["v1", "admin", "models"] + modelPathComponents(id) + ["chat-template"])
        let (data, resp) = try await performDataRequest(makeRequest(url: url, method: "GET", includeAdminToken: true))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(HeylookConfig.ChatTemplateState.self, from: data)
    }

    public func setChatTemplate(id: String, template: String) async throws -> HeylookConfig.ChatTemplateState {
        struct Body: Encodable { let template: String }
        let url = endpoint(["v1", "admin", "models"] + modelPathComponents(id) + ["chat-template"])
        let body = try JSONEncoder().encode(Body(template: template))
        let (data, resp) = try await performDataRequest(
            makeRequest(url: url, method: "PUT", body: body, includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(HeylookConfig.ChatTemplateState.self, from: data)
    }

    public func deleteChatTemplate(id: String) async throws -> HeylookConfig.ChatTemplateState {
        let url = endpoint(["v1", "admin", "models"] + modelPathComponents(id) + ["chat-template"])
        let (data, resp) = try await performDataRequest(
            makeRequest(url: url, method: "DELETE", includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(HeylookConfig.ChatTemplateState.self, from: data)
    }

    // MARK: - Conversations REST API

    public func listConversations() async throws -> ConversationListResponseDTO {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "conversations"]), method: "GET"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(ConversationListResponseDTO.self, from: data)
    }

    public func createConversation(
        title: String? = nil,
        modelID: String? = nil,
        systemPrompt: String? = nil,
        params: SamplerSettings? = nil,
        appliedPresetID: String? = nil
    ) async throws -> ConversationDetailDTO {
        var obj: [String: JSONValue] = [:]
        if let title { obj["title"] = .string(title) }
        if let modelID { obj["model_id"] = .string(modelID) }
        if let systemPrompt { obj["system_prompt"] = .string(systemPrompt) }
        if let params { obj["params"] = encodeSamplerSettingsToJSON(params) }
        if let appliedPresetID { obj["applied_preset_id"] = .string(appliedPresetID) }
        let body = try JSONEncoder().encode(obj)
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "conversations"]), method: "POST", body: body)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 201])
        return try JSONDecoder().decode(ConversationDetailDTO.self, from: data)
    }

    public func getConversation(id: String) async throws -> ConversationDetailDTO {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "conversations", id]), method: "GET"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(ConversationDetailDTO.self, from: data)
    }

    public func updateConversation(
        id: String,
        title: String? = nil,
        modelID: String? = nil,
        systemPrompt: String? = nil,
        params: SamplerSettings?? = nil,
        appliedPresetID: String?? = nil
    ) async throws -> ConversationDetailDTO {
        var obj: [String: JSONValue] = [:]
        if let title { obj["title"] = .string(title) }
        if let modelID { obj["model_id"] = .string(modelID) }
        if let systemPrompt { obj["system_prompt"] = .string(systemPrompt) }
        if let paramsWrapper = params {
            obj["params"] = paramsWrapper.map(encodeSamplerSettingsToJSON) ?? .null
        }
        if let presetWrapper = appliedPresetID {
            obj["applied_preset_id"] = presetWrapper.map { .string($0) } ?? .null
        }
        let body = try JSONEncoder().encode(obj)
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "conversations", id]), method: "PATCH", body: body)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(ConversationDetailDTO.self, from: data)
    }

    public func deleteConversation(id: String) async throws {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "conversations", id]), method: "DELETE"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 204])
    }

    public func cloneConversation(id: String, title: String? = nil, upToPosition: Int? = nil) async throws -> ConversationDetailDTO {
        var obj: [String: JSONValue] = [:]
        if let title { obj["title"] = .string(title) }
        if let upToPosition { obj["up_to_position"] = .int(upToPosition) }
        let body = try JSONEncoder().encode(obj)
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "conversations", id, "clone"]), method: "POST", body: body)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 201])
        return try JSONDecoder().decode(ConversationDetailDTO.self, from: data)
    }

    public func addMessage(
        conversationID: String,
        role: String,
        content: String,
        contentBlocks: [MessagesRequest.MessageTurn.ContentPart]? = nil,
        thinking: String? = nil
    ) async throws -> WireMessageDTO {
        var obj: [String: JSONValue] = ["role": .string(role)]
        if let contentBlocks, !contentBlocks.isEmpty {
            obj["content"] = encodeContentPartsToJSON(contentBlocks)
        } else {
            obj["content"] = .string(content)
        }
        if let thinking { obj["thinking"] = .string(thinking) }
        let body = try JSONEncoder().encode(obj)
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "conversations", conversationID, "messages"]), method: "POST", body: body)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 201])
        return try JSONDecoder().decode(WireMessageDTO.self, from: data)
    }

    public func updateMessage(
        conversationID: String,
        messageID: String,
        content: String? = nil,
        contentBlocks: [MessagesRequest.MessageTurn.ContentPart]? = nil,
        thinking: String? = nil,
        truncateAfter: Bool = false
    ) async throws -> WireMessageDTO {
        var obj: [String: JSONValue] = ["truncate_after": .bool(truncateAfter)]
        if let contentBlocks {
            obj["content"] = encodeContentPartsToJSON(contentBlocks)
        } else if let content {
            obj["content"] = .string(content)
        }
        if let thinking {
            obj["thinking"] = .string(thinking)
        }
        let body = try JSONEncoder().encode(obj)
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "conversations", conversationID, "messages", messageID]), method: "PATCH", body: body)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(WireMessageDTO.self, from: data)
    }

    public func deleteMessagesAfter(conversationID: String, afterPosition: Int) async throws {
        var components = URLComponents(url: endpoint(["v1", "conversations", conversationID, "messages"]), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "after", value: "\(afterPosition)")]
        guard let url = components.url else { throw RemoteWorkbenchError.invalidBaseURL(baseURL.absoluteString) }
        let (data, resp) = try await performDataRequest(makeRequest(url: url, method: "DELETE"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 204])
    }

    public func deleteMessage(conversationID: String, messageID: String) async throws {
        let url = endpoint(["v1", "conversations", conversationID, "messages", messageID])
        let (data, resp) = try await performDataRequest(makeRequest(url: url, method: "DELETE"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 204])
    }

    // MARK: - Notebooks REST API

    public func listNotebooks() async throws -> NotebookListResponseDTO {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "notebooks"]), method: "GET"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(NotebookListResponseDTO.self, from: data)
    }

    public func createNotebook(
        title: String,
        content: String,
        systemPrompt: String?,
        modelID: String?,
        params: SamplerSettings?,
        appliedPresetID: String?
    ) async throws -> NotebookDetailDTO {
        var obj: [String: JSONValue] = [
            "title": .string(title),
            "content": .string(content)
        ]
        if let systemPrompt { obj["system_prompt"] = .string(systemPrompt) }
        if let modelID { obj["model_id"] = .string(modelID) }
        if let params { obj["params"] = encodeSamplerSettingsToJSON(params) }
        if let appliedPresetID { obj["applied_preset_id"] = .string(appliedPresetID) }
        let body = try JSONEncoder().encode(obj)
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "notebooks"]), method: "POST", body: body))
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 201])
        return try JSONDecoder().decode(NotebookDetailDTO.self, from: data)
    }

    public func getNotebook(id: String) async throws -> NotebookDetailDTO {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "notebooks", id]), method: "GET"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(NotebookDetailDTO.self, from: data)
    }

    public func updateNotebook(
        id: String,
        title: String? = nil,
        content: String? = nil,
        systemPrompt: String? = nil,
        modelID: String? = nil,
        params: SamplerSettings?? = nil,
        appliedPresetID: String?? = nil
    ) async throws -> NotebookDetailDTO {
        var obj: [String: JSONValue] = [:]
        if let title { obj["title"] = .string(title) }
        if let content { obj["content"] = .string(content) }
        if let systemPrompt { obj["system_prompt"] = .string(systemPrompt) }
        if let modelID { obj["model_id"] = .string(modelID) }
        if let paramsWrapper = params {
            obj["params"] = paramsWrapper.map(encodeSamplerSettingsToJSON) ?? .null
        }
        if let presetWrapper = appliedPresetID {
            obj["applied_preset_id"] = presetWrapper.map { .string($0) } ?? .null
        }
        let body = try JSONEncoder().encode(obj)
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "notebooks", id]), method: "PATCH", body: body))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(NotebookDetailDTO.self, from: data)
    }

    public func deleteNotebook(id: String) async throws {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "notebooks", id]), method: "DELETE"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 204])
    }

    // MARK: - Presets REST API

    public func listPresets() async throws -> PresetListResponseDTO {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "presets"]), method: "GET"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(PresetListResponseDTO.self, from: data)
    }

    public func createPreset(name: String, systemPrompt: String?, params: SamplerSettings) async throws -> PresetDTO {
        var obj: [String: JSONValue] = [
            "name": .string(name),
            "params": encodeSamplerSettingsToJSON(params)
        ]
        if let systemPrompt { obj["system_prompt"] = .string(systemPrompt) }
        let body = try JSONEncoder().encode(obj)
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "presets"]), method: "POST", body: body))
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 201])
        return try JSONDecoder().decode(PresetDTO.self, from: data)
    }

    public func getPreset(id: String) async throws -> PresetDTO {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "presets", id]), method: "GET"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(PresetDTO.self, from: data)
    }

    public func updatePreset(id: String, name: String? = nil, systemPrompt: String?? = nil, params: SamplerSettings? = nil) async throws -> PresetDTO {
        var obj: [String: JSONValue] = [:]
        if let name { obj["name"] = .string(name) }
        if let promptWrapper = systemPrompt {
            obj["system_prompt"] = promptWrapper.map { .string($0) } ?? .null
        }
        if let params { obj["params"] = encodeSamplerSettingsToJSON(params) }
        let body = try JSONEncoder().encode(obj)
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "presets", id]), method: "PATCH", body: body))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(PresetDTO.self, from: data)
    }

    public func deletePreset(id: String) async throws {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "presets", id]), method: "DELETE"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 204])
    }

    // MARK: - Admin, Telemetry, Audio & Request Control Endpoints

    public func getModelOptions() async throws -> [String: JSONValue] {
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "admin", "model-options"]), method: "GET", includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode([String: JSONValue].self, from: data)
    }

    public func patchAdminModel(id: String, updates: ModelConfigPatchPayload) async throws -> AdminModelPatchResult {
        let url = endpoint(["v1", "admin", "models"] + modelPathComponents(id))
        let body = try JSONEncoder().encode(updates)
        let (data, resp) = try await performDataRequest(
            makeRequest(url: url, method: "PATCH", body: body, includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(AdminModelPatchResult.self, from: data)
    }

    public func getScanConfig() async throws -> ScanConfigPayload {
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "admin", "models", "scan-config"]), method: "GET", includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(ScanConfigPayload.self, from: data)
    }

    public func updateScanConfig(folders: [String]?, scanIntervalSeconds: Int?) async throws -> ScanConfigPayload {
        var obj: [String: JSONValue] = [:]
        if let folders { obj["folders"] = .array(folders.map { .string($0) }) }
        if let scanIntervalSeconds { obj["scan_interval_seconds"] = .int(scanIntervalSeconds) }
        let body = try JSONEncoder().encode(obj)
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "admin", "models", "scan-config"]), method: "PUT", body: body, includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(ScanConfigPayload.self, from: data)
    }

    public func reloadAllModels() async throws -> [String: JSONValue] {
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "admin", "reload"]), method: "POST", includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode([String: JSONValue].self, from: data)
    }

    public func getOperationalConfig() async throws -> [String: JSONValue] {
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "admin", "config"]), method: "GET", includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode([String: JSONValue].self, from: data)
    }

    public func updateOperationalConfig(_ patch: [String: JSONValue]) async throws -> [String: JSONValue] {
        let body = try JSONEncoder().encode(patch)
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "admin", "config"]), method: "PUT", body: body, includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode([String: JSONValue].self, from: data)
    }

    public func resetOperationalConfigKey(_ key: String) async throws -> [String: JSONValue] {
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "admin", "config", key]), method: "DELETE", includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode([String: JSONValue].self, from: data)
    }

    public func readObservabilityLogs(stream: ObservabilityStream? = nil, limit: Int? = nil) async throws -> ObservabilityLogsResponseDTO {
        var components = URLComponents(url: endpoint(["v1", "admin", "config", "logs"]), resolvingAgainstBaseURL: false)!
        var items: [URLQueryItem] = []
        if let stream { items.append(URLQueryItem(name: "stream", value: stream.rawValue)) }
        if let limit { items.append(URLQueryItem(name: "limit", value: "\(limit)")) }
        if !items.isEmpty { components.queryItems = items }
        guard let url = components.url else { throw RemoteWorkbenchError.invalidBaseURL(baseURL.absoluteString) }
        let (data, resp) = try await performDataRequest(makeRequest(url: url, method: "GET", includeAdminToken: true))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(ObservabilityLogsResponseDTO.self, from: data)
    }

    public func systemMetrics() async throws -> [String: JSONValue] {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "system", "metrics"]), method: "GET"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode([String: JSONValue].self, from: data)
    }

    public func performanceProfile(range: String) async throws -> SQLiteStore.PerformanceProfileSummary {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "performance", "profile", range]), method: "GET"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(SQLiteStore.PerformanceProfileSummary.self, from: data)
    }

    public func capabilities() async throws -> [String: JSONValue] {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "capabilities"]), method: "GET"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode([String: JSONValue].self, from: data)
    }

    public func clearCaches(modelID: String? = nil) async throws -> [String: JSONValue] {
        let body: Data? = try modelID.map { try JSONEncoder().encode(["model": $0]) }
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "cache", "clear"]), method: "POST", body: body, includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode([String: JSONValue].self, from: data)
    }

    public func clearAllData() async throws -> [String: JSONValue] {
        let (data, resp) = try await performDataRequest(
            makeRequest(url: endpoint(["v1", "data", "clear"]), method: "POST", includeAdminToken: true)
        )
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode([String: JSONValue].self, from: data)
    }

    public func transcribeAudio(_ request: AudioTranscriptionRequest) async throws -> CoreAIProvider.ANESpeechTranscriptionResult {
        let body = try JSONEncoder().encode(request)
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "audio", "transcriptions"]), method: "POST", body: body))
        try validateHTTPResponse(resp, data: data, allowCodes: [200])
        return try JSONDecoder().decode(CoreAIProvider.ANESpeechTranscriptionResult.self, from: data)
    }

    public func abortRequest(id: String) async throws {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "requests", id]), method: "DELETE"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 204, 404])
    }

    public func endReasoning(id: String) async throws {
        let (data, resp) = try await performDataRequest(makeRequest(url: endpoint(["v1", "requests", id, "reasoning-end"]), method: "POST"))
        try validateHTTPResponse(resp, data: data, allowCodes: [200, 404])
    }

    // MARK: - Private HTTP & Wire Helpers

    private func endpoint(_ components: [String]) -> URL {
        var url = baseURL
        for component in components where !component.isEmpty {
            url.appendPathComponent(component)
        }
        return url
    }

    private func modelPathComponents(_ modelID: String) -> [String] {
        modelID.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
    }

    private func makeRequest(
        url: URL,
        method: String,
        body: Data? = nil,
        includeAdminToken: Bool = false
    ) -> URLRequest {
        var req = URLRequest(url: url)
        req.httpMethod = method
        if let body {
            req.httpBody = body
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        if includeAdminToken, let adminToken {
            req.setValue(adminToken, forHTTPHeaderField: "X-Heylook-Admin-Token")
        }
        return req
    }

    private func performDataRequest(_ req: URLRequest) async throws -> (Data, URLResponse) {
        try await session.data(for: req)
    }

    private func validateHTTPResponse(_ response: URLResponse, data: Data, allowCodes: Set<Int>) throws {
        guard let http = response as? HTTPURLResponse else {
            throw RemoteWorkbenchError.invalidPayload("Non-HTTP response received.")
        }
        guard allowCodes.contains(http.statusCode) else {
            let retryAfter = http.value(forHTTPHeaderField: "Retry-After")
            let parsed = parseErrorPayload(data: data, statusCode: http.statusCode)
            throw RemoteWorkbenchError.httpStatus(
                statusCode: http.statusCode,
                code: parsed.code,
                message: parsed.message,
                retryAfter: retryAfter
            )
        }
    }

    private func parseErrorPayload(data: Data, statusCode: Int) -> (code: String?, message: String) {
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let errorObj = json["error"] as? [String: Any] {
                let code = errorObj["code"] as? String ?? errorObj["type"] as? String
                let msg = (errorObj["message"] as? String) ?? HTTPURLResponse.localizedString(forStatusCode: statusCode)
                return (code, msg)
            }
            if let detail = json["detail"] as? String {
                return (json["code"] as? String, detail)
            }
            if let msg = json["message"] as? String {
                return (json["code"] as? String, msg)
            }
        }
        let raw = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        return (nil, raw.isEmpty ? HTTPURLResponse.localizedString(forStatusCode: statusCode) : raw)
    }

    private func decodePublicModelList(from data: Data) throws -> [PublicModelWireRow] {
        struct PublicListEnvelope: Decodable {
            let data: [PublicModelWireRow]
        }
        if let env = try? JSONDecoder().decode(PublicListEnvelope.self, from: data) {
            return env.data
        }
        if let direct = try? JSONDecoder().decode([PublicModelWireRow].self, from: data) {
            return direct
        }
        throw RemoteWorkbenchError.invalidPayload("Expected {\"object\": \"list\", \"data\": [...]} from /v1/models.")
    }

    private func decodeAdminModelList(from data: Data) throws -> [AdminModelWireRow] {
        struct AdminModelsEnvelope: Decodable {
            let models: [AdminModelWireRow]?
            let data: [AdminModelWireRow]?
        }
        if let env = try? JSONDecoder().decode(AdminModelsEnvelope.self, from: data) {
            if let models = env.models { return models }
            if let dataRows = env.data { return dataRows }
        }
        if let direct = try? JSONDecoder().decode([AdminModelWireRow].self, from: data) {
            return direct
        }
        throw RemoteWorkbenchError.invalidPayload("Expected {\"models\": [...]} from /v1/admin/models.")
    }

    private func encodeSamplerSettingsToJSON(_ settings: SamplerSettings) -> JSONValue {
        var dict: [String: JSONValue] = [:]
        if let temperature = settings.temperature { dict["temperature"] = .double(temperature) }
        if let topP = settings.topP { dict["top_p"] = .double(topP) }
        if let topK = settings.topK { dict["top_k"] = .int(topK) }
        if let minP = settings.minP { dict["min_p"] = .double(minP) }
        if let rep = settings.repetitionPenalty { dict["repetition_penalty"] = .double(rep) }
        if let repCtx = settings.repetitionContextSize { dict["repetition_context_size"] = .int(repCtx) }
        if let pres = settings.presencePenalty { dict["presence_penalty"] = .double(pres) }
        if let maxTokens = settings.maxTokens { dict["max_tokens"] = .int(maxTokens) }
        if let seed = settings.seed { dict["seed"] = .int(seed) }
        if let thinking = settings.thinking { dict["enable_thinking"] = .bool(thinking) }
        if let depth = settings.thinkingDepth { dict["thinking_depth"] = .string(depth) }
        if let budget = settings.thinkingBudget { dict["thinking_budget"] = .int(budget) }
        if let stops = settings.stopSequences { dict["stop_sequences"] = .array(stops.map { .string($0) }) }
        return .object(dict)
    }

    private func encodeContentPartsToJSON(_ parts: [MessagesRequest.MessageTurn.ContentPart]) -> JSONValue {
        guard let encoded = try? JSONEncoder().encode(parts),
              let value = try? JSONDecoder().decode(JSONValue.self, from: encoded) else {
            return .array([])
        }
        return value
    }
}

private extension ProvenanceSource {
    var legacySource: ProvenanceSource { self }
}

private extension JSONValue {
    var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }
}
