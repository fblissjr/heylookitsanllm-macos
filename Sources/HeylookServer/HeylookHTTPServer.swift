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

/// Authoritative `/v1/*` API router implementing the `heylook` Messages wire,
/// disconnect-resilient `/v1/conversations/{id}/generate`, `/v1/models`, `/v1/requests/{id}`,
/// `/v1/conversations/*`, `/v1/presets/*`, and `/v1/system/metrics`.
public actor HeylookRouter {
    public let store: SQLiteStore
    public let scheduler: ComputeScheduler
    public let registry: RequestRegistry
    public let bindHost: String

    public init(
        store: SQLiteStore,
        scheduler: ComputeScheduler,
        registry: RequestRegistry = RequestRegistry(),
        bindHost: String = "127.0.0.1"
    ) {
        self.store = store
        self.scheduler = scheduler
        self.registry = registry
        self.bindHost = bindHost
    }

    /// Validates `Host` header against DNS rebinding when bound to loopback.
    public nonisolated func isAllowedHostHeader(_ hostHeader: String?) -> Bool {
        let isLoopbackBind = (bindHost == "127.0.0.1" || bindHost == "localhost" || bindHost == "::1")
        guard isLoopbackBind else { return true }
        guard let hostHeader, !hostHeader.isEmpty else { return true }
        let hostOnly = hostHeader.split(separator: ":").first.map(String.init)?.lowercased() ?? ""
        return hostOnly == "127.0.0.1" || hostOnly == "localhost" || hostOnly == "[::1]"
    }

    public func handle(
        method: String,
        path: String,
        queryItems: [URLQueryItem] = [],
        headers: [String: String] = [:],
        body: Data = Data()
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
                return try await handleMessages(requestID: reqID, request: req)

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
                        generateRequest: genReq
                    )
                }
                if segments.count == 5 && segments[3] == "generate" && segments[4] == "reasoning-end" && method.uppercased() == "POST" {
                    let ack = await registry.requestReasoningEnd(forConversationID: convID)
                    return try .json(["acknowledged": ack])
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
        } catch {
            return .error(status: 400, type: "invalid_request_error", message: error.localizedDescription)
        }
    }

    private func handleMessages(
        requestID: String,
        request: MessagesRequest
    ) async throws -> HeylookHTTPResponse {
        let provider = try await scheduler.provider(for: request.model)
        let resolvedSampling = SamplerCascade.resolve(request: request.samplerOverlay)
        await registry.register(requestID: requestID, modelID: request.model)

        return try await scheduler.withComputeLease(for: request.model) { [registry] in
            defer {
                Task { await registry.finish(requestID: requestID) }
            }
            var fullText = ""
            var fullThinking = ""
            var finalChunk = GenerationChunk()

            let stream = await provider.generate(
                requestID: requestID,
                request: request,
                effectiveSampling: resolvedSampling.effective,
                registry: registry
            )
            for try await chunk in stream {
                fullText.append(chunk.text)
                fullThinking.append(chunk.thinking)
                finalChunk = chunk
            }

            var blocks: [MessagesResponse.ContentBlock] = []
            if !fullThinking.isEmpty {
                blocks.append(.thinking(fullThinking))
            }
            blocks.append(.text(fullText))

            let response = MessagesResponse(
                model: request.model,
                content: blocks,
                stopReason: finalChunk.stopReason ?? .endTurn,
                usage: MessagesResponse.Usage(
                    inputTokens: finalChunk.cacheReport?.processedTokens ?? finalChunk.promptTokens,
                    outputTokens: finalChunk.tokenCount,
                    cacheReadInputTokens: finalChunk.cacheReport?.cachedTokens
                ),
                performance: MessagesResponse.PerformanceExtension(
                    generationTPS: finalChunk.generationTPS,
                    promptTPS: finalChunk.promptTPS,
                    computePlacement: finalChunk.activeComputePlacement,
                    cache: finalChunk.cacheReport,
                    speculative: finalChunk.speculativeReport
                )
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
        generateRequest: ConversationGenerateRequest
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
                        sseLines.append("event: content_block_delta\ndata: \(deltaJSON)\n\n")
                    }
                    if !chunk.text.isEmpty {
                        fullText.append(chunk.text)
                        let deltaJSON = try String(
                            data: JSONEncoder().encode(["type": "text_delta", "text": chunk.text]),
                            encoding: .utf8
                        ) ?? "{}"
                        sseLines.append("event: content_block_delta\ndata: \(deltaJSON)\n\n")
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
                sseLines.append("event: heylook_saved\ndata: \(savedJSON)\n\n")
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
            Self.receiveHTTPRequest(on: connection, router: routerRef)
        }
        nwListener.start(queue: .global(qos: .userInitiated))
        self.listener = nwListener
        self.isRunning = true
    }

    public func stop() {
        listener?.cancel()
        listener = nil
        isRunning = false
    }

    private nonisolated static func receiveHTTPRequest(on connection: NWConnection, router: HeylookRouter) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4 * 1024 * 1024) { data, _, _, error in
            guard let data, error == nil, !data.isEmpty else {
                connection.cancel()
                return
            }
            Task {
                let response = await Self.parseAndDispatch(rawData: data, router: router)
                let serialized = Self.serializeHTTPResponse(response)
                connection.send(content: serialized, completion: .contentProcessed { _ in
                    connection.cancel()
                })
            }
        }
    }

    private nonisolated static func parseAndDispatch(rawData: Data, router: HeylookRouter) async -> HeylookHTTPResponse {
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
        return await router.handle(method: method, path: path, queryItems: queryItems, headers: headers, body: bodyData)
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
