import Foundation
import HeylookCore
import Network

/// HTTP/1.1 connection transport handling `Content-Length`, `Transfer-Encoding: chunked`,
/// true Keep-Alive connection reuse loops, and peer-disconnect `StreamCancellationSignal`
/// propagation without CORS preflight hacks.
public enum HTTPConnectionHandler {
    public struct ParsedHTTPRequest: Sendable {
        public let method: String
        public let path: String
        public let queryItems: [URLQueryItem]
        public let headers: [String: String]
        public let body: Data
        public let wantsKeepAlive: Bool
        public let consumedBytes: Int
    }

    /// Computes total frame byte length for `Content-Length` requests (used by transport loop and
    /// preserved via `HeylookHTTPServer.expectedTotalBytes`). Returns `nil` for chunked requests
    /// when headers are incomplete or body decoding determines completeness.
    public static func expectedTotalBytes(for rawData: Data) -> Int? {
        let separator = Data("\r\n\r\n".utf8)
        guard let range = rawData.range(of: separator) else { return nil }
        let headerData = rawData.subdata(in: 0..<range.lowerBound)
        let headerText = String(decoding: headerData, as: UTF8.self)
        var contentLength = 0
        for line in headerText.components(separatedBy: "\r\n").dropFirst() {
            let parts = line.split(separator: ":", maxSplits: 1)
            if parts.count == 2 {
                let key = parts[0].trimmingCharacters(in: .whitespaces).lowercased()
                let val = parts[1].trimmingCharacters(in: .whitespaces)
                if key == "transfer-encoding", val.lowercased().contains("chunked") {
                    return decodeChunkedBody(in: rawData, from: range.upperBound)?.consumedEndIndex
                }
                if key == "content-length", let len = Int(val) {
                    contentLength = max(0, len)
                    break
                }
            }
        }
        return range.upperBound + contentLength
    }

    public static func splitTarget(_ rawTarget: String) -> (path: String, queryItems: [URLQueryItem]) {
        if let qIdx = rawTarget.firstIndex(of: "?") {
            let path = String(rawTarget[..<qIdx])
            let queryStr = String(rawTarget[rawTarget.index(after: qIdx)...])
            let comps = URLComponents(string: "http://localhost/?\(queryStr)")
            return (path.isEmpty ? "/" : path, comps?.queryItems ?? [])
        }
        return (rawTarget.isEmpty ? "/" : rawTarget, [])
    }

    private static func parseHeaderBlock(_ headerText: String) -> (
        method: String,
        target: String,
        httpVersion: String,
        headers: [String: String]
    )? {
        let lines = headerText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        let reqParts = requestLine.split(separator: " ")
        guard reqParts.count >= 2 else { return nil }

        let method = String(reqParts[0]).uppercased()
        let rawTarget = String(reqParts[1])
        let httpVersion = reqParts.count >= 3 ? String(reqParts[2]).uppercased() : "HTTP/1.1"
        var headers: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            let parts = line.split(separator: ":", maxSplits: 1)
            guard parts.count == 2 else { continue }
            headers[parts[0].trimmingCharacters(in: .whitespaces)] = parts[1].trimmingCharacters(in: .whitespaces)
        }
        return (method, rawTarget, httpVersion, headers)
    }

    public static func parseNextRequest(from buffer: Data) -> ParsedHTTPRequest? {
        let separator = Data("\r\n\r\n".utf8)
        guard let range = buffer.range(of: separator) else { return nil }
        let headerData = buffer.subdata(in: 0..<range.lowerBound)
        let headerText = String(decoding: headerData, as: UTF8.self)
        guard let parsedHead = parseHeaderBlock(headerText) else { return nil }

        let (path, queryItems) = splitTarget(parsedHead.target)
        let headers = parsedHead.headers
        var isChunked = false
        var contentLength = 0
        var connectionValue: String? = nil

        for (key, val) in headers {
            let lowerKey = key.lowercased()
            if lowerKey == "transfer-encoding", val.lowercased().contains("chunked") {
                isChunked = true
            } else if lowerKey == "content-length", let len = Int(val) {
                contentLength = max(0, len)
            } else if lowerKey == "connection" {
                connectionValue = val.lowercased()
            }
        }

        let bodyStart = range.upperBound
        let bodyData: Data
        let consumedBytes: Int
        if isChunked {
            guard let decoded = decodeChunkedBody(in: buffer, from: bodyStart) else {
                return nil
            }
            bodyData = decoded.body
            consumedBytes = decoded.consumedEndIndex
        } else {
            let totalNeeded = bodyStart + contentLength
            guard buffer.count >= totalNeeded else { return nil }
            bodyData = buffer.subdata(in: bodyStart..<totalNeeded)
            consumedBytes = totalNeeded
        }

        let wantsKeepAlive: Bool
        if let connectionValue {
            wantsKeepAlive = !connectionValue.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces) }).contains("close")
        } else {
            wantsKeepAlive = parsedHead.httpVersion != "HTTP/1.0"
        }

        return ParsedHTTPRequest(
            method: parsedHead.method,
            path: path,
            queryItems: queryItems,
            headers: headers,
            body: bodyData,
            wantsKeepAlive: wantsKeepAlive,
            consumedBytes: consumedBytes
        )
    }

    private static func decodeChunkedBody(in buffer: Data, from startIndex: Int) -> (body: Data, consumedEndIndex: Int)? {
        var cursor = startIndex
        var out = Data()
        let crlf = Data("\r\n".utf8)
        let doubleCRLF = Data("\r\n\r\n".utf8)

        while cursor <= buffer.count {
            guard let lineEndRange = buffer.range(of: crlf, in: cursor..<buffer.count) else {
                return nil
            }
            let sizeLine = String(decoding: buffer.subdata(in: cursor..<lineEndRange.lowerBound), as: UTF8.self)
            let hexPart = sizeLine.split(separator: ";", maxSplits: 1).first.map {
                String($0).trimmingCharacters(in: .whitespaces)
            } ?? ""
            guard let chunkSize = Int(hexPart, radix: 16), chunkSize >= 0 else {
                return nil
            }
            cursor = lineEndRange.upperBound
            if chunkSize == 0 {
                // Consume optional trailer headers terminated by CRLF (or immediate CRLF if no trailers)
                if cursor + 2 <= buffer.count && buffer[cursor..<(cursor + 2)] == crlf {
                    return (out, cursor + 2)
                }
                if let trailerEnd = buffer.range(of: doubleCRLF, in: cursor..<buffer.count) {
                    return (out, trailerEnd.upperBound)
                }
                return nil
            }
            guard cursor + chunkSize + 2 <= buffer.count else {
                return nil
            }
            out.append(buffer.subdata(in: cursor..<(cursor + chunkSize)))
            cursor += chunkSize
            guard buffer[cursor..<(cursor + 2)] == crlf else {
                return nil
            }
            cursor += 2
        }
        return nil
    }

    private actor ConnectionBufferActor {
        private var buffer = Data()
        private var eof = false
        private var waiters: [CheckedContinuation<Void, Never>] = []
        private var activeCancelSignal: StreamCancellationSignal?

        func setActiveCancelSignal(_ signal: StreamCancellationSignal?) {
            activeCancelSignal = signal
            if eof {
                signal?.cancel()
            }
        }

        func append(_ data: Data, isComplete: Bool, hadError: Bool) {
            if !data.isEmpty {
                buffer.append(data)
            }
            if isComplete || hadError {
                eof = true
                activeCancelSignal?.cancel()
            }
            let pending = waiters
            waiters.removeAll()
            for w in pending {
                w.resume()
            }
        }

        func nextRequest() async -> (request: ParsedHTTPRequest?, remainingAfterEOF: Data?) {
            while true {
                if let parsed = parseNextRequest(from: buffer) {
                    buffer.removeFirst(parsed.consumedBytes)
                    return (parsed, nil)
                }
                if eof {
                    if !buffer.isEmpty {
                        let tail = buffer
                        buffer.removeAll(keepingCapacity: false)
                        return (nil, tail)
                    }
                    return (nil, nil)
                }
                await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                    waiters.append(cont)
                }
            }
        }
    }

    /// Drives persistent HTTP/1.1 Keep-Alive request loops on an accepted `NWConnection`,
    /// using one persistent socket reader so peer TCP FIN / abort immediately cancels
    /// in-flight `/v1/messages` generation while keeping HTTP/1.1 Keep-Alive safe.
    public static func handleConnection(
        _ connection: NWConnection,
        router: HeylookRouter
    ) {
        let bufferActor = ConnectionBufferActor()
        connection.stateUpdateHandler = { state in
            switch state {
            case .cancelled, .failed:
                Task { await bufferActor.append(Data(), isComplete: true, hadError: true) }
            default:
                break
            }
        }
        connection.start(queue: .global(qos: .userInitiated))
        startSocketReader(on: connection, into: bufferActor)

        Task {
            await runConnectionLoop(on: connection, bufferActor: bufferActor, router: router)
        }
    }

    private static func startSocketReader(
        on connection: NWConnection,
        into bufferActor: ConnectionBufferActor
    ) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4 * 1024 * 1024) { chunk, _, isComplete, error in
            let data = chunk ?? Data()
            let hadErr = (error != nil)
            Task {
                await bufferActor.append(data, isComplete: isComplete, hadError: hadErr)
            }
            if !isComplete && !hadErr {
                startSocketReader(on: connection, into: bufferActor)
            }
        }
    }

    private static func runConnectionLoop(
        on connection: NWConnection,
        bufferActor: ConnectionBufferActor,
        router: HeylookRouter
    ) async {
        while true {
            let next = await bufferActor.nextRequest()
            if let tail = next.remainingAfterEOF {
                let signal = StreamCancellationSignal()
                await bufferActor.setActiveCancelSignal(signal)
                await dispatchRawHTTPBuffer(
                    tail,
                    router: router,
                    cancellationSignal: signal,
                    sendBytes: { payload in
                        await sendData(payload, on: connection, cancelSignal: signal)
                    },
                    closeConnection: {
                        connection.cancel()
                    }
                )
                await bufferActor.setActiveCancelSignal(nil)
                return
            }

            guard let parsed = next.request else {
                connection.cancel()
                return
            }

            let cancelSignal = StreamCancellationSignal()
            await bufferActor.setActiveCancelSignal(cancelSignal)

            let keepAlive = await dispatchParsedRequest(
                parsed,
                on: connection,
                router: router,
                cancelSignal: cancelSignal
            )
            await bufferActor.setActiveCancelSignal(nil)

            if !keepAlive {
                return
            }
        }
    }

    private static func dispatchParsedRequest(
        _ parsed: ParsedHTTPRequest,
        on connection: NWConnection,
        router: HeylookRouter,
        cancelSignal: StreamCancellationSignal
    ) async -> Bool {
        let incomingReqID = parsed.headers.first { $0.key.lowercased() == "x-request-id" }?.value
        let resolvedReqID = HeylookRouter.resolveRequestID(from: parsed.headers)
        var headers = parsed.headers
        if incomingReqID != resolvedReqID {
            headers["X-Request-ID"] = resolvedReqID
        }

        let sseStreamer = SSEConnectionStreamer(
            requestID: resolvedReqID,
            sendBytes: { payload in
                await sendData(payload, on: connection, cancelSignal: cancelSignal)
            },
            closeConnection: {
                connection.cancel()
            }
        )

        let response = await router.handle(
            method: parsed.method,
            path: parsed.path,
            queryItems: parsed.queryItems,
            headers: headers,
            body: parsed.body,
            cancellationSignal: cancelSignal,
            onSSEFrame: { frame in
                await sseStreamer.sendFrame(frame)
            }
        )

        let didStream = await sseStreamer.hasStreamed
        if didStream {
            await sseStreamer.finish()
            return false
        }

        var finalResponse = response
        if finalResponse.headers["X-Request-ID"] == nil && finalResponse.headers["x-request-id"] == nil {
            finalResponse.headers["X-Request-ID"] = resolvedReqID
        }
        let keepAlive = parsed.wantsKeepAlive
            && (finalResponse.headers["Connection"]?.lowercased() != "close")
            && !cancelSignal.isCancelled
        finalResponse.headers["Connection"] = keepAlive ? "keep-alive" : "close"

        let serialized = serializeHTTPResponse(finalResponse)
        await sendData(serialized, on: connection, cancelSignal: cancelSignal)
        if !keepAlive {
            connection.cancel()
        }
        return keepAlive
    }

    private static func sendData(
        _ payload: Data,
        on connection: NWConnection,
        cancelSignal: StreamCancellationSignal
    ) async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            connection.send(content: payload, completion: .contentProcessed { error in
                if error != nil {
                    cancelSignal.cancel()
                }
                cont.resume()
            })
        }
    }

    /// Dispatches a raw HTTP buffer directly (also exercised by contract tests).
    public static func dispatchRawHTTPBuffer(
        _ finalBuffer: Data,
        router: HeylookRouter,
        cancellationSignal: StreamCancellationSignal = StreamCancellationSignal(),
        sendBytes: @escaping @Sendable (Data) async -> Void,
        closeConnection: @escaping @Sendable () -> Void
    ) async {
        let separator = Data("\r\n\r\n".utf8)
        let headerData: Data
        let bodyData: Data
        if let range = finalBuffer.range(of: separator) {
            headerData = finalBuffer.subdata(in: 0..<range.lowerBound)
            bodyData = finalBuffer.subdata(in: range.upperBound..<finalBuffer.count)
        } else {
            headerData = finalBuffer
            bodyData = Data()
        }

        let headerText = String(decoding: headerData, as: UTF8.self)
        guard let parsedHead = parseHeaderBlock(headerText) else {
            let err = HeylookHTTPResponse.error(status: 400, type: "invalid_request_error", message: "Malformed HTTP request line")
            await sendBytes(serializeHTTPResponse(err))
            closeConnection()
            return
        }

        let method = parsedHead.method
        let (path, queryItems) = splitTarget(parsedHead.target)
        var headers = parsedHead.headers
        let isChunked = RequestMiddleware.headerValue(named: "Transfer-Encoding", in: headers)?
            .lowercased()
            .contains("chunked") ?? false

        let decodedBody: Data
        if isChunked, let parsed = parseNextRequest(from: finalBuffer) {
            decodedBody = parsed.body
        } else {
            decodedBody = bodyData
        }

        let resolvedReqID = HeylookRouter.resolveRequestID(from: headers)
        headers["X-Request-ID"] = resolvedReqID

        let sseStreamBox = SSEConnectionStreamer(
            requestID: resolvedReqID,
            sendBytes: sendBytes,
            closeConnection: closeConnection
        )
        var response = await router.handle(
            method: method,
            path: path,
            queryItems: queryItems,
            headers: headers,
            body: decodedBody,
            cancellationSignal: cancellationSignal,
            onSSEFrame: { frame in
                await sseStreamBox.sendFrame(frame)
            }
        )
        let didStream = await sseStreamBox.hasStreamed
        if didStream {
            await sseStreamBox.finish()
        } else {
            if response.headers["X-Request-ID"] == nil && response.headers["x-request-id"] == nil {
                response.headers["X-Request-ID"] = resolvedReqID
            }
            let serialized = serializeHTTPResponse(response)
            await sendBytes(serialized)
            closeConnection()
        }
    }

    public static func serializeHTTPResponse(_ response: HeylookHTTPResponse) -> Data {
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
