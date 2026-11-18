import CryptoKit
import Foundation
import UniformTypeIdentifiers
import zlib

/// Optional static asset serving from `HEYLOOK_FRONTEND_DIR` (when configured and present on disk),
/// enforcing strict `no-cache`, per-encoding `ETag` + `304`, `Vary: Accept-Encoding`, gzip compression,
/// symlink confinement, NUL-byte `404` defense, and ZERO SPA fallback so unmatched API routes return `404`.
public enum StaticFrontendRoutes {
    private static let gzipExtensions: Set<String> = ["js", "mjs", "css", "html", "json", "svg"]
    private static let gzipMinBytes = 1024

    public static func resolvedFrontendDirectory() -> URL? {
        guard let raw = ProcessInfo.processInfo.environment["HEYLOOK_FRONTEND_DIR"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty else {
            return nil
        }
        let expanded = (raw as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded, isDirectory: true).resolvingSymlinksInPath()
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }
        return url
    }

    public static func register(on router: inout PatternRouter, frontendRoot: URL? = resolvedFrontendDirectory()) {
        guard let rootDir = frontendRoot else { return }
        let canonicalRoot = rootDir.resolvingSymlinksInPath()

        router.registerMethods(["GET", "HEAD"], "/") { ctx in
            serveAsset(canonicalRoot.appendingPathComponent("index.html"), root: canonicalRoot, ctx: ctx)
        }
        router.registerMethods(["GET", "HEAD"], "/index.html") { ctx in
            serveAsset(canonicalRoot.appendingPathComponent("index.html"), root: canonicalRoot, ctx: ctx)
        }
        router.registerMethods(["GET", "HEAD"], "/icon.svg") { ctx in
            serveAsset(canonicalRoot.appendingPathComponent("icon.svg"), root: canonicalRoot, ctx: ctx)
        }
        router.registerMethods(["GET", "HEAD"], "/favicon.ico") { ctx in
            serveAsset(canonicalRoot.appendingPathComponent("icon.svg"), root: canonicalRoot, ctx: ctx)
        }
        router.registerMethods(["GET", "HEAD"], "/manifest.json") { ctx in
            serveAsset(canonicalRoot.appendingPathComponent("manifest.json"), root: canonicalRoot, ctx: ctx)
        }
        router.registerMethods(["GET", "HEAD"], "/apple-touch-icon.png") { ctx in
            serveAsset(canonicalRoot.appendingPathComponent("apple-touch-icon.png"), root: canonicalRoot, ctx: ctx)
        }
        router.registerMethods(["GET", "HEAD"], "/js/*rest") { ctx in
            guard let rest = ctx.param("rest"), !rest.isEmpty else {
                throw WorkbenchError.notFound("Not found")
            }
            return serveAsset(canonicalRoot.appendingPathComponent("js").appendingPathComponent(rest), root: canonicalRoot, ctx: ctx)
        }
        router.registerMethods(["GET", "HEAD"], "/css/*rest") { ctx in
            guard let rest = ctx.param("rest"), !rest.isEmpty else {
                throw WorkbenchError.notFound("Not found")
            }
            return serveAsset(canonicalRoot.appendingPathComponent("css").appendingPathComponent(rest), root: canonicalRoot, ctx: ctx)
        }
    }

    private static func serveAsset(_ candidate: URL, root: URL, ctx: RouteRequestContext) -> HeylookHTTPResponse {
        if candidate.path.contains("\0") {
            return .error(status: 404, type: "not_found_error", message: "Not found")
        }
        let resolved = candidate.resolvingSymlinksInPath()
        let rootPrefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard resolved.path == root.path || resolved.path.hasPrefix(rootPrefix) else {
            return .error(status: 404, type: "not_found_error", message: "Not found")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: resolved.path, isDirectory: &isDir),
              !isDir.boolValue,
              let attrs = try? FileManager.default.attributesOfItem(atPath: resolved.path),
              let mtime = attrs[.modificationDate] as? Date,
              let size = (attrs[.size] as? NSNumber)?.intValue else {
            return .error(status: 404, type: "not_found_error", message: "Not found")
        }

        let baseTagSeed = "\(mtime.timeIntervalSince1970)-\(size)"
        let digest = Insecure.MD5.hash(data: Data(baseTagSeed.utf8))
        let baseHex = digest.map { String(format: "%02x", $0) }.joined()

        let acceptEnc = RequestMiddleware.headerValue(named: "Accept-Encoding", in: ctx.headers)?.lowercased() ?? ""
        let ext = resolved.pathExtension.lowercased()
        let wantsGzip = gzipExtensions.contains(ext) && size >= gzipMinBytes && acceptEnc.contains("gzip")
        let etag = wantsGzip ? "\"\(baseHex)-gzip\"" : "\"\(baseHex)\""

        var baseHeaders: [String: String] = [
            "Cache-Control": "no-cache",
            "ETag": etag,
            "Vary": "Accept-Encoding",
        ]

        if let ifNoneMatch = RequestMiddleware.headerValue(named: "If-None-Match", in: ctx.headers) {
            let candidates = ifNoneMatch.split(separator: ",").map {
                var token = $0.trimmingCharacters(in: .whitespacesAndNewlines)
                if token.hasPrefix("W/") {
                    token = String(token.dropFirst(2))
                }
                return token
            }
            if candidates.contains(etag) {
                return HeylookHTTPResponse(status: 304, headers: baseHeaders, body: Data())
            }
        }

        guard let fileData = try? Data(contentsOf: resolved) else {
            return .error(status: 404, type: "not_found_error", message: "Not found")
        }

        baseHeaders["Content-Type"] = mimeType(forExtension: ext)
        baseHeaders["Last-Modified"] = httpDateString(mtime)

        if wantsGzip, let compressed = gzipCompress(fileData) {
            baseHeaders["Content-Encoding"] = "gzip"
            let body = ctx.method == "HEAD" ? Data() : compressed
            return HeylookHTTPResponse(status: 200, headers: baseHeaders, body: body)
        }

        let body = ctx.method == "HEAD" ? Data() : fileData
        return HeylookHTTPResponse(status: 200, headers: baseHeaders, body: body)
    }

    private static func mimeType(forExtension ext: String) -> String {
        switch ext {
        case "html": return "text/html; charset=utf-8"
        case "js", "mjs": return "application/javascript; charset=utf-8"
        case "css": return "text/css; charset=utf-8"
        case "json": return "application/json; charset=utf-8"
        case "svg": return "image/svg+xml"
        case "png": return "image/png"
        case "ico": return "image/x-icon"
        default:
            return UTType(filenameExtension: ext)?.preferredMIMEType ?? "application/octet-stream"
        }
    }

    private static func httpDateString(_ date: Date) -> String {
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.timeZone = TimeZone(secondsFromGMT: 0)
        fmt.dateFormat = "EEE, dd MMM yyyy HH:mm:ss 'GMT'"
        return fmt.string(from: date)
    }

    private static func gzipCompress(_ data: Data) -> Data? {
        guard !data.isEmpty else { return Data() }
        var stream = z_stream()
        let initStatus = deflateInit2_(
            &stream,
            6,
            Z_DEFLATED,
            MAX_WBITS + 16,
            8,
            Z_DEFAULT_STRATEGY,
            ZLIB_VERSION,
            Int32(MemoryLayout<z_stream>.size)
        )
        guard initStatus == Z_OK else { return nil }
        defer { deflateEnd(&stream) }

        var compressed = Data()
        let chunkSize = 16_384
        var outputBuffer = [UInt8](repeating: 0, count: chunkSize)

        let result: Int32 = data.withUnsafeBytes { rawBuffer in
            guard let basePtr = rawBuffer.bindMemory(to: Bytef.self).baseAddress else {
                return Z_STREAM_ERROR
            }
            stream.next_in = UnsafeMutablePointer<Bytef>(mutating: basePtr)
            stream.avail_in = uInt(data.count)

            repeat {
                let status: Int32 = outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    stream.next_out = outPtr.baseAddress
                    stream.avail_out = uInt(chunkSize)
                    return deflate(&stream, Z_FINISH)
                }
                let produced = chunkSize - Int(stream.avail_out)
                if produced > 0 {
                    compressed.append(outputBuffer, count: produced)
                }
                if status != Z_OK && status != Z_STREAM_END {
                    return status
                }
            } while stream.avail_out == 0

            return Z_STREAM_END
        }
        return result == Z_STREAM_END ? compressed : nil
    }
}
