import CryptoKit
import Darwin
import Foundation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines

/// Domain-level backend errors mapped cleanly to HTTP status codes, headers, and error envelopes.
public enum WorkbenchError: LocalizedError, Sendable {
    case badRequest(String)
    case unauthorized(String)
    case forbidden(String)
    case notFound(String)
    case conflict(code: String? = nil, message: String)
    case unprocessable(message: String, field: String? = nil)
    case modelOverloaded(message: String, queueCapacity: Int? = nil)
    case serviceUnavailable(String)
    case internalError(String)

    public var errorDescription: String? {
        switch self {
        case .badRequest(let msg),
             .unauthorized(let msg),
             .forbidden(let msg),
             .notFound(let msg),
             .unprocessable(let msg, _),
             .modelOverloaded(let msg, _),
             .serviceUnavailable(let msg),
             .internalError(let msg):
            return msg
        case .conflict(_, let msg):
            return msg
        }
    }
}

/// Request validation, DNS-rebinding defense, constant-time admin auth, and HTTP error mapping.
public enum RequestMiddleware {
    public static let queueFullMessage = "The generation queue is full. Retry shortly."
    private static let hmacKey = SymmetricKey(data: Data("heylook-constant-time-token-compare-v1".utf8))

    public static func headerValue(named targetName: String, in headers: [String: String]) -> String? {
        let lower = targetName.lowercased()
        for (k, v) in headers where k.lowercased() == lower {
            return v
        }
        return nil
    }

    /// Always returns a valid request ID: accepts client-supplied `X-Request-ID` when valid
    /// per `RequestRegistry.isValidClientRequestID`, otherwise silently mints a safe `req_<hex>`.
    public static func resolveRequestID(from headers: [String: String]) -> String {
        if let raw = headerValue(named: "X-Request-ID", in: headers)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           RequestRegistry.isValidClientRequestID(raw) {
            return raw
        }
        return "req_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16))"
    }

    /// Constant-time comparison against `X-Heylook-Admin-Token` ONLY (no `Authorization: Bearer`).
    public static func verifyAdminToken(headers: [String: String], expectedToken: String?) -> Bool {
        guard let expected = expectedToken?.trimmingCharacters(in: .whitespacesAndNewlines),
              !expected.isEmpty else {
            return true
        }
        let provided = headerValue(named: "X-Heylook-Admin-Token", in: headers) ?? ""
        return constantTimeEqual(provided, expected)
    }

    public static func constantTimeEqual(_ lhs: String, _ rhs: String) -> Bool {
        let macA = Array(HMAC<SHA256>.authenticationCode(for: Data(lhs.utf8), using: hmacKey))
        let macB = Array(HMAC<SHA256>.authenticationCode(for: Data(rhs.utf8), using: hmacKey))
        var diff: UInt8 = 0
        for i in 0..<min(macA.count, macB.count) {
            diff |= macA[i] ^ macB[i]
        }
        return diff == 0 && macA.count == macB.count && lhs.utf8.count == rhs.utf8.count
    }

    public static func hostOnly(_ value: String) -> String {
        var v = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if v.hasPrefix("[") {
            if let end = v.firstIndex(of: "]") {
                return String(v[v.index(after: v.startIndex)..<end])
            }
            return v
        }
        if v.filter({ $0 == ":" }).count == 1, let colon = v.lastIndex(of: ":") {
            v = String(v[..<colon])
        }
        while v.hasSuffix(".") {
            v.removeLast()
        }
        return v
    }

    public static func ownHostNames() -> Set<String> {
        var names: Set<String> = ["localhost"]
        let rawHost = ProcessInfo.processInfo.hostName.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "."))
        if !rawHost.isEmpty {
            names.insert(rawHost)
            if let short = rawHost.split(separator: ".").first.map(String.init), !short.isEmpty {
                names.insert(short)
                names.insert("\(short).local")
            }
        }
        return names
    }

    public static func isIPLiteral(_ host: String) -> Bool {
        var addr4 = in_addr()
        if host.withCString({ inet_pton(AF_INET, $0, &addr4) }) == 1 {
            return true
        }
        var addr6 = in6_addr()
        if host.withCString({ inet_pton(AF_INET6, $0, &addr6) }) == 1 {
            return true
        }
        return false
    }

    /// Validates `Host` against DNS rebinding on both loopback and wildcard listeners.
    public static func isAllowedHostHeader(_ hostHeader: String?, allowedHosts: Set<String>) -> Bool {
        if allowedHosts.contains("*") { return true }
        guard let raw = hostHeader?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            // Non-browser callers without Host pass cleanly; HTTP/1.1 browser requests always carry Host
            return true
        }
        let normalized = hostOnly(raw)
        if isIPLiteral(normalized) {
            return true
        }
        if ownHostNames().contains(normalized) {
            return true
        }
        return allowedHosts.contains(normalized)
    }

    public static func mapError(_ error: Error, defaultCapacity: Int? = nil) -> HeylookHTTPResponse {
        if let wbErr = error as? WorkbenchError {
            switch wbErr {
            case .badRequest(let msg):
                return .error(status: 400, type: "invalid_request_error", message: msg)
            case .unauthorized(let msg):
                return .error(status: 401, type: "authentication_error", message: msg)
            case .forbidden(let msg):
                return .error(status: 403, type: "permission_error", message: msg)
            case .notFound(let msg):
                return .error(status: 404, type: "not_found_error", message: msg)
            case .conflict(let code, let msg):
                return .error(status: 409, type: "invalid_request_error", message: msg, code: code)
            case .unprocessable(let msg, let field):
                let detail: Any = field.map { ["field": $0, "error": msg] } ?? msg
                return .error(status: 422, type: "validation_error", message: msg, detail: detail)
            case .modelOverloaded(let msg, let cap):
                return busyResponse(message: msg, capacity: cap ?? defaultCapacity)
            case .serviceUnavailable(let msg):
                return .error(status: 503, type: "server_error", message: msg)
            case .internalError(let msg):
                return .error(status: 500, type: "server_error", message: msg)
            }
        }

        if let schedErr = error as? ComputeSchedulerError {
            switch schedErr {
            case .queueOverloaded(_, let maxDepth):
                return busyResponse(message: queueFullMessage, capacity: maxDepth)
            case .modelNotFound(let modelID):
                return .error(
                    status: 400,
                    type: "invalid_request_error",
                    message: "Model '\(modelID)' not found or not enabled"
                )
            case .modelBusy(let modelID):
                return .error(
                    status: 409,
                    type: "invalid_request_error",
                    message: "Model '\(modelID)' has active generation traffic and cannot be unloaded",
                    code: "generation_in_progress"
                )
            }
        }

        if let wireErr = error as? WireValidationError {
            return .error(
                status: wireErr.statusCode,
                type: wireErr.errorType,
                message: wireErr.localizedDescription
            )
        }

        if let sqlErr = error as? SQLiteStoreError {
            switch sqlErr {
            case .notFound(let msg):
                return .error(status: 404, type: "not_found_error", message: msg)
            case .conflict(let msg):
                return .error(status: 409, type: "conflict_error", message: msg)
            case .invalidInput(let msg):
                return .error(status: 400, type: "invalid_request_error", message: msg)
            case .openFailed(let msg), .sqlFailed(let msg):
                return .error(status: 500, type: "server_error", message: msg)
            }
        }

        if let execErr = error as? ProviderExecutionError {
            switch execErr {
            case .invalidRequest:
                return .error(status: 400, type: "invalid_request_error", message: execErr.localizedDescription)
            case .checkpointNotFound, .binaryNotFound, .runtimeUnavailable:
                return .error(status: 500, type: "server_error", message: execErr.localizedDescription)
            }
        }

        if let embValidation = error as? EmbeddingValidationError {
            let status = embValidation.statusCode
            return .error(
                status: status,
                type: status == 422 ? "validation_error" : "invalid_request_error",
                message: embValidation.localizedDescription
            )
        }

        if let embErr = error as? EmbeddingEngineError {
            switch embErr {
            case .invalidResponse(let status, let detail):
                return .error(status: status, type: status >= 500 ? "server_error" : "invalid_request_error", message: detail)
            case .vectorCountMismatch:
                return .error(status: 500, type: "server_error", message: embErr.localizedDescription)
            case .checkpointNotFound, .runtimeUnavailable:
                return .error(status: 503, type: "server_error", message: embErr.localizedDescription)
            }
        }

        if let jinjaErr = error as? JinjaTemplateError {
            return .error(status: 400, type: "invalid_request_error", message: jinjaErr.localizedDescription)
        }

        if let decodingErr = error as? DecodingError {
            return .error(
                status: 422,
                type: "validation_error",
                message: "Validation failed: \(decodingErr.localizedDescription)"
            )
        }

        if error is CancellationError {
            return .error(status: 409, type: "invalid_request_error", message: "Request was cancelled", code: "cancelled")
        }

        return .error(status: 500, type: "server_error", message: error.localizedDescription)
    }

    public static func busyResponse(message: String, capacity: Int?) -> HeylookHTTPResponse {
        let trimmed = message
            .replacingOccurrences(of: "MODEL_BUSY:", with: "")
            .replacingOccurrences(of: "MODEL_BUSY", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let finalMessage = trimmed.isEmpty ? queueFullMessage : trimmed
        let capStr = (capacity ?? 0) > 0 ? "\(capacity!)" : "1"
        return .error(
            status: 503,
            type: "server_error",
            message: finalMessage,
            code: "model_overloaded",
            extraHeaders: [
                "Retry-After": "1",
                "X-RateLimit-Limit": capStr,
                "X-RateLimit-Remaining": "0",
            ]
        )
    }
}
