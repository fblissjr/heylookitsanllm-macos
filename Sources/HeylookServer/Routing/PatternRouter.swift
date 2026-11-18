import Foundation

/// HTTP response envelope produced by `HeylookRouter` and serialized over `HTTPConnectionHandler`.
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
        case 304: return "Not Modified"
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
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
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
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
        var hdrs = ["Content-Type": "application/json"]
        for (k, v) in extraHeaders { hdrs[k] = v }
        return HeylookHTTPResponse(status: status, headers: hdrs, body: data)
    }

    public static func error(
        status: Int,
        type: String,
        message: String,
        code: String? = nil,
        detail: Any? = nil,
        extraHeaders: [String: String] = [:]
    ) -> HeylookHTTPResponse {
        var errorDict: [String: Any] = [
            "type": type,
            "message": message,
        ]
        if let code, !code.isEmpty {
            errorDict["code"] = code
        }
        var payload: [String: Any] = [
            "type": "error",
            "error": errorDict,
        ]
        if let detail {
            payload["detail"] = detail
        } else {
            payload["detail"] = message
        }
        let data = (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys, .withoutEscapingSlashes])) ?? Data()
        var hdrs = ["Content-Type": "application/json"]
        for (k, v) in extraHeaders { hdrs[k] = v }
        return HeylookHTTPResponse(status: status, headers: hdrs, body: data)
    }
}

/// Request context passed to registered route handlers.
public struct RouteRequestContext: Sendable {
    public let method: String
    public let path: String
    public let queryItems: [URLQueryItem]
    public let headers: [String: String]
    public let body: Data
    public let params: [String: String]
    public let requestID: String
    public let cancellationSignal: StreamCancellationSignal
    public let onSSEFrame: (@Sendable (String) async -> Void)?

    public func queryValue(_ name: String) -> String? {
        queryItems.first(where: { $0.name == name })?.value
    }

    public func boolQuery(_ name: String, default defaultValue: Bool = false) -> Bool {
        guard let raw = queryValue(name)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !raw.isEmpty else {
            return defaultValue
        }
        switch raw {
        case "1", "true", "yes", "on": return true
        case "0", "false", "no", "off": return false
        default: return defaultValue
        }
    }

    public func intQuery(_ name: String) -> Int? {
        queryValue(name).flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    }

    public func param(_ name: String) -> String? {
        params[name]
    }
}

/// Ordered HTTP pattern router supporting exact segments, single-segment `:name` captures,
/// and greedy multi-segment `*name` captures (decoding both slash-separated `org/model` and
/// percent-encoded `org%2Fmodel` transparently).
public struct PatternRouter: Sendable {
    public typealias Handler = @Sendable (RouteRequestContext) async throws -> HeylookHTTPResponse

    private enum Segment: Sendable {
        case literal(String)
        case param(String)
        case greedy(String)
    }

    private struct RouteEntry: Sendable {
        let method: String
        let requiresAdmin: Bool
        let pattern: [Segment]
        let handler: Handler
    }

    private var routes: [RouteEntry] = []

    public init() {}

    public mutating func register(
        _ method: String,
        _ template: String,
        requiresAdmin: Bool = false,
        handler: @escaping Handler
    ) {
        routes.append(
            RouteEntry(
                method: method.uppercased(),
                requiresAdmin: requiresAdmin,
                pattern: Self.compile(template),
                handler: handler
            )
        )
    }

    public mutating func registerMethods(
        _ methods: [String],
        _ template: String,
        requiresAdmin: Bool = false,
        handler: @escaping Handler
    ) {
        for m in methods {
            register(m, template, requiresAdmin: requiresAdmin, handler: handler)
        }
    }

    public struct MatchResult: Sendable {
        public let params: [String: String]
        public let requiresAdmin: Bool
        public let handler: Handler
    }

    public enum LookupResult: Sendable {
        case matched(MatchResult)
        case methodNotAllowed(allowedMethods: [String])
        case notFound
    }

    public func match(method: String, path: String) -> LookupResult {
        let normalizedMethod = method.uppercased()
        let pathSegments = Self.splitPath(path)

        var matchingMethodsForPath: [String] = []
        for entry in routes {
            guard let captured = Self.match(pattern: entry.pattern, against: pathSegments) else {
                continue
            }
            if entry.method == normalizedMethod {
                return .matched(
                    MatchResult(
                        params: captured,
                        requiresAdmin: entry.requiresAdmin,
                        handler: entry.handler
                    )
                )
            }
            if !matchingMethodsForPath.contains(entry.method) {
                matchingMethodsForPath.append(entry.method)
            }
        }

        if !matchingMethodsForPath.isEmpty {
            return .methodNotAllowed(allowedMethods: matchingMethodsForPath)
        }
        return .notFound
    }

    private static func compile(_ template: String) -> [Segment] {
        splitPath(template).map { raw in
            if raw.hasPrefix("*") && raw.count > 1 {
                return .greedy(String(raw.dropFirst()))
            } else if raw.hasPrefix(":") && raw.count > 1 {
                return .param(String(raw.dropFirst()))
            } else {
                return .literal(raw)
            }
        }
    }

    private static func splitPath(_ rawPath: String) -> [String] {
        var trimmed = rawPath
        if let qIdx = trimmed.firstIndex(of: "?") {
            trimmed = String(trimmed[..<qIdx])
        }
        if trimmed == "/" || trimmed.isEmpty {
            return []
        }
        return trimmed.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
    }

    private static func match(pattern: [Segment], against segments: [String]) -> [String: String]? {
        var params: [String: String] = [:]
        return matchRecursive(pattern: pattern, pIdx: 0, segments: segments, sIdx: 0, params: &params)
            ? params
            : nil
    }

    private static func matchRecursive(
        pattern: [Segment],
        pIdx: Int,
        segments: [String],
        sIdx: Int,
        params: inout [String: String]
    ) -> Bool {
        if pIdx == pattern.count {
            return sIdx == segments.count
        }

        switch pattern[pIdx] {
        case .literal(let expected):
            guard sIdx < segments.count else { return false }
            let actual = decodeSegment(segments[sIdx])
            guard actual == expected || segments[sIdx] == expected else { return false }
            return matchRecursive(pattern: pattern, pIdx: pIdx + 1, segments: segments, sIdx: sIdx + 1, params: &params)

        case .param(let name):
            guard sIdx < segments.count else { return false }
            guard let decoded = safeDecodeSegment(segments[sIdx]), !decoded.isEmpty else { return false }
            let previous = params[name]
            params[name] = decoded
            if matchRecursive(pattern: pattern, pIdx: pIdx + 1, segments: segments, sIdx: sIdx + 1, params: &params) {
                return true
            }
            if let previous { params[name] = previous } else { params.removeValue(forKey: name) }
            return false

        case .greedy(let name):
            let remainingPattern = pattern.count - (pIdx + 1)
            let maxEnd = segments.count - remainingPattern
            guard maxEnd > sIdx else { return false }
            // Greedy match tries longest slice first so trailing sub-routes match accurately
            for endIdx in stride(from: maxEnd, through: sIdx + 1, by: -1) {
                let slice = segments[sIdx..<endIdx]
                var decodedParts: [String] = []
                decodedParts.reserveCapacity(slice.count)
                var valid = true
                for part in slice {
                    guard let d = safeDecodeSegment(part) else {
                        valid = false
                        break
                    }
                    decodedParts.append(d)
                }
                guard valid else { continue }
                let joined = decodedParts.joined(separator: "/")
                guard !joined.isEmpty else { continue }
                let previous = params[name]
                params[name] = joined
                if matchRecursive(pattern: pattern, pIdx: pIdx + 1, segments: segments, sIdx: endIdx, params: &params) {
                    return true
                }
                if let previous { params[name] = previous } else { params.removeValue(forKey: name) }
            }
            return false
        }
    }

    private static func decodeSegment(_ segment: String) -> String {
        segment.removingPercentEncoding ?? segment
    }

    private static func safeDecodeSegment(_ segment: String) -> String? {
        if segment.contains("\0") || segment.lowercased().contains("%00") {
            return nil
        }
        let decoded = segment.removingPercentEncoding ?? segment
        if decoded.contains("\0") {
            return nil
        }
        return decoded
    }
}
