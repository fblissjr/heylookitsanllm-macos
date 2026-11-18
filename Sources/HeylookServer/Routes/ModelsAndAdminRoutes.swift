import Foundation
import HeylookCore

public enum ModelsAndAdminRoutes {
    public static func register(on router: inout PatternRouter, service: WorkbenchService) {
        // MARK: Public /v1/models Discovery & Client Ops (no admin token)

        router.register("GET", "/v1/models") { _ in
            let rows = await service.listPublicModels()
            struct ListEnvelope: Encodable {
                let object = "list"
                let data: [PublicModelWireRow]
            }
            return try .json(ListEnvelope(data: rows), status: 200)
        }

        // Fixed/sub-resource routes BEFORE any greedy catch-all
        router.register("POST", "/v1/models/*id/image-plan") { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            let req = try decodeImagePlanRequest(from: ctx.body)
            let plan = try await service.planImages(id: modelID, sizes: req.sizes)
            return try .json(plan, status: 200)
        }

        router.register("POST", "/v1/models/*id/load") { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            let warm = ctx.boolQuery("warm", default: false)
            let result = try await service.loadModel(id: modelID, warm: warm)
            return try .json(result, status: 200)
        }

        // MARK: Admin Endpoints (X-Heylook-Admin-Token required when HEYLOOK_ADMIN_TOKEN is configured)

        router.register("GET", "/v1/admin/model-options", requiresAdmin: true) { _ in
            let options = ProviderConfigSchema.modelOptionsJSON(includeCoreAI: true)
            return try .jsonObject(options, status: 200)
        }

        // Compatibility alias for callers hitting the older nested path
        router.register("GET", "/v1/admin/models/model-options", requiresAdmin: true) { _ in
            let options = ProviderConfigSchema.modelOptionsJSON(includeCoreAI: true)
            return try .jsonObject(options, status: 200)
        }

        router.register("POST", "/v1/admin/reload", requiresAdmin: true) { _ in
            let result = try await service.reloadAllModels()
            return try .json(result, status: 200)
        }

        router.register("GET", "/v1/admin/config", requiresAdmin: true) { _ in
            let snap = await service.getOperationalConfig()
            return try .json(snap, status: 200)
        }

        router.register("PUT", "/v1/admin/config", requiresAdmin: true) { ctx in
            let snap = try await service.updateOperationalConfig(patchData: ctx.body)
            return try .json(snap, status: 200)
        }

        router.register("GET", "/v1/admin/config/logs", requiresAdmin: true) { ctx in
            let logs = try await service.readObservabilityLogs(
                streamRaw: ctx.queryValue("stream"),
                limit: ctx.intQuery("limit")
            )
            return try .json(logs, status: 200)
        }

        router.register("DELETE", "/v1/admin/config/:key", requiresAdmin: true) { ctx in
            guard let key = ctx.param("key"), !key.isEmpty else {
                throw WorkbenchError.badRequest("Missing operational config key")
            }
            let snap = try await service.resetOperationalConfigKey(key)
            return try .json(snap, status: 200)
        }

        // Admin models fixed routes MUST precede greedy /v1/admin/models/*id
        router.register("GET", "/v1/admin/models", requiresAdmin: true) { _ in
            let rows = await service.listAdminModels()
            struct AdminListEnvelope: Encodable {
                let models: [AdminModelWireRow]
                let total: Int
            }
            return try .json(AdminListEnvelope(models: rows, total: rows.count), status: 200)
        }

        router.register("POST", "/v1/admin/models", requiresAdmin: true) { ctx in
            let payload = try JSONDecoder().decode([String: JSONValue].self, from: ctx.body)
            let created = try await service.addAdminModel(payload)
            return try .json(created, status: 201)
        }

        router.register("GET", "/v1/admin/models/scan-config", requiresAdmin: true) { _ in
            let scanCfg = await service.getScanConfig()
            return try .json(scanCfg, status: 200)
        }

        router.register("PUT", "/v1/admin/models/scan-config", requiresAdmin: true) { ctx in
            struct ScanPatchBody: Decodable {
                let folders: [String]?
                let scanIntervalSeconds: Int?
                enum CodingKeys: String, CodingKey {
                    case folders
                    case scanIntervalSeconds = "scan_interval_seconds"
                }
            }
            let raw = try JSONSerialization.jsonObject(with: ctx.body)
            guard let dict = raw as? [String: Any] else {
                throw WorkbenchError.unprocessable(message: "scan-config body must be a JSON object")
            }
            let unknown = dict.keys.filter { $0 != "folders" && $0 != "scan_interval_seconds" }.sorted()
            if !unknown.isEmpty {
                throw WorkbenchError.unprocessable(message: "Unknown field(s): \(unknown.joined(separator: ", "))")
            }
            let body = try JSONDecoder().decode(ScanPatchBody.self, from: ctx.body)
            let updated = try await service.updateScanConfig(
                folders: body.folders,
                scanIntervalSeconds: body.scanIntervalSeconds
            )
            return try .json(updated, status: 200)
        }

        router.register("POST", "/v1/admin/models/validate", requiresAdmin: true) { ctx in
            let payload = try JSONDecoder().decode([String: JSONValue].self, from: ctx.body)
            let result = await service.validateModelConfig(payload)
            return try .json(result, status: 200)
        }

        // Sub-resource routes on *id BEFORE bare *id catch-all
        router.register("GET", "/v1/admin/models/*id/status", requiresAdmin: true) { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            let status = await service.modelStatus(id: modelID)
            return try .json(status, status: 200)
        }

        router.register("POST", "/v1/admin/models/*id/reload", requiresAdmin: true) { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            let warm = ctx.boolQuery("warm", default: false)
            var loadSettings: [String: JSONValue]? = nil
            if !ctx.body.isEmpty {
                let decoded = try JSONDecoder().decode(JSONValue.self, from: ctx.body)
                switch decoded {
                case .null:
                    loadSettings = nil
                case .object(let obj):
                    loadSettings = obj
                default:
                    throw WorkbenchError.badRequest("reload body must be a JSON object or null")
                }
            }
            let res = try await service.reloadModel(id: modelID, loadSettings: loadSettings, warm: warm)
            return try .json(res, status: 200)
        }

        router.register("POST", "/v1/admin/models/*id/unload", requiresAdmin: true) { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            let res = try await service.unloadModel(id: modelID)
            return try .json(res, status: 200)
        }

        router.register("POST", "/v1/admin/models/*id/fit", requiresAdmin: true) { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            struct FitBody: Decodable {
                let configOverrides: [String: JSONValue]?
                let headroomGB: Double?
                enum CodingKeys: String, CodingKey {
                    case configOverrides = "config_overrides"
                    case headroomGB = "headroom_gb"
                }
            }
            let parsed = ctx.body.isEmpty
                ? FitBody(configOverrides: nil, headroomGB: nil)
                : try JSONDecoder().decode(FitBody.self, from: ctx.body)
            let report = try await service.assessModelFit(
                id: modelID,
                configOverrides: parsed.configOverrides ?? [:],
                headroomGB: parsed.headroomGB ?? 8.0
            )
            return try .json(report, status: 200)
        }

        router.register("GET", "/v1/admin/models/*id/chat-template", requiresAdmin: true) { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            let state = try await service.getChatTemplate(id: modelID)
            let row = try await service.adminModel(id: modelID)
            return try serializeChatTemplateState(state, modelID: modelID, provider: row.provider.rawValue)
        }

        router.register("PUT", "/v1/admin/models/*id/chat-template", requiresAdmin: true) { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            struct UpdateBody: Decodable {
                let template: String
            }
            let body = try JSONDecoder().decode(UpdateBody.self, from: ctx.body)
            let state = try await service.setChatTemplate(id: modelID, template: body.template)
            let row = try await service.adminModel(id: modelID)
            return try serializeChatTemplateState(state, modelID: modelID, provider: row.provider.rawValue)
        }

        router.register("DELETE", "/v1/admin/models/*id/chat-template", requiresAdmin: true) { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            let state = try await service.deleteChatTemplate(id: modelID)
            let row = try await service.adminModel(id: modelID)
            return try serializeChatTemplateState(state, modelID: modelID, provider: row.provider.rawValue)
        }

        // Greedy *id catch-alls MUST be registered LAST
        router.register("GET", "/v1/admin/models/*id", requiresAdmin: true) { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            let row = try await service.adminModel(id: modelID)
            return try .json(row, status: 200)
        }

        router.register("PATCH", "/v1/admin/models/*id", requiresAdmin: true) { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            let updates = try JSONDecoder().decode(ModelConfigPatchPayload.self, from: ctx.body)
            let result = try await service.patchAdminModel(id: modelID, updates: updates)
            return try .json(result, status: 200)
        }

        router.register("DELETE", "/v1/admin/models/*id", requiresAdmin: true) { ctx in
            guard let modelID = ctx.param("id"), !modelID.isEmpty else {
                throw WorkbenchError.badRequest("Missing model id")
            }
            let result = try await service.deleteAdminModel(id: modelID)
            return try .json(result, status: 200)
        }
    }

    private static func decodeImagePlanRequest(from data: Data) throws -> ImagePlanRequestPayload {
        guard !data.isEmpty else {
            throw WorkbenchError.badRequest("sizes array is required")
        }
        return try JSONDecoder().decode(ImagePlanRequestPayload.self, from: data)
    }

    private static func serializeChatTemplateState(
        _ state: HeylookConfig.ChatTemplateState,
        modelID: String,
        provider: String
    ) throws -> HeylookHTTPResponse {
        let data = try JSONEncoder().encode(state)
        var obj = (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        obj["model_id"] = modelID
        obj["provider"] = provider
        obj["override_present"] = state.hasOverride
        // Truthful staleness: explicit JSON null when model is not loaded
        if let observed = state.observedStale {
            obj["stale"] = observed
        } else {
            obj["stale"] = NSNull()
        }
        return try .jsonObject(obj, status: 200)
    }
}
