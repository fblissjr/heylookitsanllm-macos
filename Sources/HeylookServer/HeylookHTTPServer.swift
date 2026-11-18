import Foundation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import Network

/// Authoritative `/v1/*` HTTP/SSE router backed by `WorkbenchService` and `PatternRouter`.
/// Preserves Wave 1 convenience initializers for `HeylookUI` / `HeylookTests` compatibility while
/// enforcing DNS-rebinding protection, silent invalid `X-Request-ID` replacement, constant-time
/// `X-Heylook-Admin-Token` authentication (`401`), and unified `heylookitsanllm` `v2.0.251` routes.
public actor HeylookRouter {
    public nonisolated let service: WorkbenchService
    public nonisolated let store: SQLiteStore
    public nonisolated let scheduler: ComputeScheduler
    public nonisolated let registry: RequestRegistry
    public nonisolated let bindHost: String
    public nonisolated let allowedHosts: Set<String>
    public nonisolated let adminToken: String?
    public nonisolated let isReadOnlyConfig: Bool

    private let patternRouter: PatternRouter

    public var embeddingEngine: any EmbeddingEngine {
        get async { await service.currentEmbeddingEngine() }
    }

    public init(
        service: WorkbenchService,
        bindHost: String = "127.0.0.1",
        allowedHosts: [String] = [],
        adminToken: String? = ProcessInfo.processInfo.environment["HEYLOOK_ADMIN_TOKEN"]
    ) {
        self.service = service
        self.store = service.store
        self.scheduler = service.scheduler
        self.registry = service.registry
        self.bindHost = bindHost
        self.allowedHosts = Set(allowedHosts.map { RequestMiddleware.hostOnly($0) == "*" ? "*" : RequestMiddleware.hostOnly($0) })
        let trimmedToken = adminToken?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.adminToken = (trimmedToken?.isEmpty == false) ? trimmedToken : nil
        self.isReadOnlyConfig = service.isReadOnlyConfig
        self.patternRouter = Self.buildPatternRouter(service: service, registry: service.registry)
    }

    public init(
        store: SQLiteStore,
        scheduler: ComputeScheduler,
        registry: RequestRegistry = .shared,
        embeddingEngine: (any EmbeddingEngine)? = nil,
        bindHost: String = "127.0.0.1",
        allowedHosts: [String] = [],
        adminToken: String? = ProcessInfo.processInfo.environment["HEYLOOK_ADMIN_TOKEN"],
        isReadOnlyConfig: Bool = HeylookConfig.isReadOnlyModelConfig
    ) {
        let svc = WorkbenchService(
            store: store,
            scheduler: scheduler,
            registry: registry,
            embeddingEngine: embeddingEngine,
            isReadOnlyConfig: isReadOnlyConfig
        )
        self.init(
            service: svc,
            bindHost: bindHost,
            allowedHosts: allowedHosts,
            adminToken: adminToken
        )
    }

    private static func buildPatternRouter(
        service: WorkbenchService,
        registry: RequestRegistry
    ) -> PatternRouter {
        var router = PatternRouter()
        MessagesAndAudioRoutes.register(on: &router, service: service)
        ConversationsRoutes.register(on: &router, service: service)
        NotebooksAndPresetsRoutes.register(on: &router, service: service)
        ModelsAndAdminRoutes.register(on: &router, service: service)
        MonitoringAndRequestsRoutes.register(on: &router, service: service, registry: registry)
        EmbeddingsRoutes.register(on: &router, service: service)
        StaticFrontendRoutes.register(on: &router)
        return router
    }

    public func setEmbeddingEngine(_ engine: any EmbeddingEngine) async {
        await service.setEmbeddingEngine(engine)
    }

    public nonisolated func isAllowedHostHeader(_ hostHeader: String?) -> Bool {
        RequestMiddleware.isAllowedHostHeader(hostHeader, allowedHosts: allowedHosts)
    }

    public nonisolated static func resolveRequestID(from headers: [String: String]) -> String {
        RequestMiddleware.resolveRequestID(from: headers)
    }

    public func handle(
        method: String,
        path: String,
        queryItems: [URLQueryItem] = [],
        headers: [String: String] = [:],
        body: Data = Data(),
        onSSEFrame: (@Sendable (String) async -> Void)? = nil
    ) async -> HeylookHTTPResponse {
        await handle(
            method: method,
            path: path,
            queryItems: queryItems,
            headers: headers,
            body: body,
            cancellationSignal: StreamCancellationSignal(),
            onSSEFrame: onSSEFrame
        )
    }

    public func handle(
        method: String,
        path: String,
        queryItems: [URLQueryItem] = [],
        headers: [String: String] = [:],
        body: Data = Data(),
        cancellationSignal: StreamCancellationSignal,
        onSSEFrame: (@Sendable (String) async -> Void)? = nil
    ) async -> HeylookHTTPResponse {
        let requestID = RequestMiddleware.resolveRequestID(from: headers)

        func finish(_ response: HeylookHTTPResponse) -> HeylookHTTPResponse {
            var res = response
            if RequestMiddleware.headerValue(named: "X-Request-ID", in: res.headers) == nil {
                res.headers["X-Request-ID"] = requestID
            }
            return res
        }

        let hostHeader = RequestMiddleware.headerValue(named: "Host", in: headers)
        guard isAllowedHostHeader(hostHeader) else {
            let name = hostHeader.map(RequestMiddleware.hostOnly) ?? ""
            return finish(
                .error(
                    status: 403,
                    type: "permission_error",
                    message: "This server does not answer to the host name '\(name)'. Add it to allowed_hosts in heylook.toml (DNS-rebinding guard)."
                )
            )
        }

        switch patternRouter.match(method: method, path: path) {
        case .notFound:
            return finish(.error(status: 404, type: "not_found_error", message: "Route not found: \(method.uppercased()) \(path)"))

        case .methodNotAllowed(let allowed):
            return finish(
                .error(
                    status: 405,
                    type: "invalid_request_error",
                    message: "Method \(method.uppercased()) not allowed for \(path)",
                    extraHeaders: ["Allow": allowed.joined(separator: ", ")]
                )
            )

        case .matched(let matched):
            if matched.requiresAdmin && !RequestMiddleware.verifyAdminToken(headers: headers, expectedToken: adminToken) {
                return finish(
                    .error(
                        status: 401,
                        type: "authentication_error",
                        message: "Admin endpoint requires X-Heylook-Admin-Token header."
                    )
                )
            }

            let ctx = RouteRequestContext(
                method: method.uppercased(),
                path: path,
                queryItems: queryItems,
                headers: headers,
                body: body,
                params: matched.params,
                requestID: requestID,
                cancellationSignal: cancellationSignal,
                onSSEFrame: onSSEFrame
            )

            do {
                return finish(try await matched.handler(ctx))
            } catch {
                let metrics = await scheduler.metrics()
                return finish(RequestMiddleware.mapError(error, defaultCapacity: metrics.maxQueueDepth))
            }
        }
    }
}

/// Embedded Network.framework HTTP/1.1 Keep-Alive & chunked SSE server exposing `HeylookRouter`
/// with Bonjour `_heylook._tcp` advertisement.
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
            // Avoid POSIX EINVAL when requiredLocalEndpoint already carries a port
            nwListener = try NWListener(using: params)
        } else {
            nwListener = try NWListener(using: params, on: nwPort)
        }
        if advertiseBonjour && isWildcard {
            nwListener.service = NWListener.Service(name: "Heylook Server", type: "_heylook._tcp")
        }
        let routerRef = self.router
        nwListener.newConnectionHandler = { connection in
            HTTPConnectionHandler.handleConnection(connection, router: routerRef)
        }
        nwListener.start(queue: .global(qos: .userInitiated))
        self.listener = nwListener
        self.isRunning = true
    }

    public func stop() async {
        listener?.cancel()
        listener = nil
        isRunning = false
        await router.service.unloadAllForShutdown()
    }

    public nonisolated static func expectedTotalBytes(for rawData: Data) -> Int? {
        HTTPConnectionHandler.expectedTotalBytes(for: rawData)
    }

    internal nonisolated static func dispatchRawHTTPBuffer(
        _ finalBuffer: Data,
        router: HeylookRouter,
        sendBytes: @escaping @Sendable (Data) async -> Void,
        closeConnection: @escaping @Sendable () -> Void
    ) async {
        await HTTPConnectionHandler.dispatchRawHTTPBuffer(
            finalBuffer,
            router: router,
            sendBytes: sendBytes,
            closeConnection: closeConnection
        )
    }

    public nonisolated static func serializeHTTPResponse(_ response: HeylookHTTPResponse) -> Data {
        HTTPConnectionHandler.serializeHTTPResponse(response)
    }
}
