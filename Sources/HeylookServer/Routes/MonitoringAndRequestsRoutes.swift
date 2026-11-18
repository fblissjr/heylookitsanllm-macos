import Foundation
import HeylookCore

public enum MonitoringAndRequestsRoutes {
    public static func register(
        on router: inout PatternRouter,
        service: WorkbenchService,
        registry: RequestRegistry
    ) {
        router.register("GET", "/v1/system/metrics") { _ in
            let snapshot = await service.systemMetrics()
            return try .json(snapshot, status: 200)
        }

        router.register("GET", "/v1/performance/profile/:range") { ctx in
            guard let range = ctx.param("range"), !range.isEmpty else {
                throw WorkbenchError.badRequest("Missing time range")
            }
            let profile = try await service.performanceProfile(range: range)
            return try .json(profile, status: 200)
        }

        router.register("GET", "/v1/capabilities") { _ in
            let caps = await service.capabilities()
            return try .json(caps, status: 200)
        }

        router.register("POST", "/v1/cache/clear", requiresAdmin: true) { ctx in
            var modelID: String? = nil
            if !ctx.body.isEmpty {
                struct Body: Decodable { let model: String? }
                modelID = try JSONDecoder().decode(Body.self, from: ctx.body).model
            }
            let result = try await service.clearCaches(modelID: modelID)
            return try .json(result, status: 200)
        }

        router.register("POST", "/v1/data/clear", requiresAdmin: true) { _ in
            let result = try await service.clearAllData()
            return try .json(result, status: 200)
        }

        router.register("POST", "/v1/telemetry/events") { ctx in
            let obj = (try? JSONSerialization.jsonObject(with: ctx.body) as? [String: Any]) ?? [:]
            let result = await service.ingestTelemetryEvents(obj)
            return try .json(result, status: 200)
        }

        // Active Request Lifecycle Controls
        router.register("DELETE", "/v1/requests/:id") { ctx in
            guard let reqID = ctx.param("id"),
                  RequestRegistry.isValidClientRequestID(reqID) else {
                throw WorkbenchError.unprocessable(
                    message: "Invalid request_id: must match ^[A-Za-z0-9._:-]{1,128}$",
                    field: "request_id"
                )
            }
            let countBefore = registry.activeRequests().filter { $0.requestID == reqID }.count
            guard registry.abort(requestID: reqID) else {
                throw WorkbenchError.notFound("No active request with id '\(reqID)'")
            }
            let cancelled = max(1, countBefore)
            return try .jsonObject([
                "cancelled": cancelled,
                "request_id": reqID,
            ], status: 200)
        }

        router.register("POST", "/v1/requests/:id/reasoning-end") { ctx in
            guard let reqID = ctx.param("id"),
                  RequestRegistry.isValidClientRequestID(reqID) else {
                throw WorkbenchError.unprocessable(
                    message: "Invalid request_id: must match ^[A-Za-z0-9._:-]{1,128}$",
                    field: "request_id"
                )
            }
            let countBefore = registry.activeRequests().filter { $0.requestID == reqID }.count
            guard registry.requestReasoningEnd(requestID: reqID) else {
                throw WorkbenchError.notFound("No active request with id '\(reqID)'")
            }
            let signalled = max(1, countBefore)
            return try .jsonObject([
                "signalled": signalled,
                "request_id": reqID,
            ], status: 200)
        }
    }
}
