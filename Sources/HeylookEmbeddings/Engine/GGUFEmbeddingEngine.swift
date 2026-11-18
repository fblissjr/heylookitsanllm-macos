#if os(macOS)
import Darwin
#endif
import Foundation
import HeylookCore

/// Out-of-process GGUF `EmbeddingEngine` executing against `llama-server --embeddings --pooling <mode>`
/// (`gguf_embeddings.py` parity).
///
/// Supports both attaching to a managed `llama-server` HTTP base URL (all Apple platforms) and spawning
/// a dedicated `llama-server --embeddings` subprocess on macOS when initialized from a `.gguf` file path.
/// Video `windows` requests are explicitly rejected with HTTP `400` because `llama-server` samples
/// video frames internally without exposing frame-picker indices.
public actor GGUFEmbeddingEngine: EmbeddingEngine {
    public nonisolated let modelID: String
    public nonisolated let nativeDimensions: Int
    public nonisolated let supportedDimensions: [Int]
    public nonisolated let facts: EngineEmbeddingFacts?
    public nonisolated let pooling: String
    public nonisolated var availablePromptNames: [String] { promptCatalog.prompts.keys.sorted() }
    public nonisolated var defaultPromptName: String? { promptCatalog.defaultPromptName }

    private nonisolated let promptCatalog: EmbeddingModelFacts.PromptCatalog
    private nonisolated let fixedBaseURL: URL?
    private nonisolated let timeoutSeconds: Double
    private nonisolated let session: URLSession

    #if os(macOS)
    private nonisolated let managedLaunchConfig: LaunchConfig?
    private let processHolder = SubprocessHolder()

    private struct LaunchConfig: Sendable {
        let binaryURL: URL
        let modelFile: URL
        let mmprojFile: URL?
        let ctxSize: Int
        let batchSize: Int
        let ubatchSize: Int
        let nGPULayers: Int
        let flashAttention: Bool
        let pooling: String
    }

    private final class SubprocessHolder: @unchecked Sendable {
        private let lock = NSLock()
        private var process: Process?
        private var baseURL: URL?

        func currentBaseURL() -> URL? {
            lock.lock()
            defer { lock.unlock() }
            if let proc = process, !proc.isRunning {
                process = nil
                baseURL = nil
                return nil
            }
            return baseURL
        }

        func store(process: Process, baseURL: URL) {
            lock.lock()
            self.process = process
            self.baseURL = baseURL
            lock.unlock()
        }

        func terminateIfNeeded() {
            lock.lock()
            let proc = process
            process = nil
            baseURL = nil
            lock.unlock()

            guard let proc else { return }
            if proc.isRunning {
                proc.terminate()
                let deadline = Date().addingTimeInterval(2.0)
                while proc.isRunning && Date() < deadline {
                    Thread.sleep(forTimeInterval: 0.05)
                }
                if proc.isRunning {
                    kill(proc.processIdentifier, SIGKILL)
                }
            }
        }

        deinit {
            terminateIfNeeded()
        }
    }
    #endif

    /// Connects to an already-running `llama-server --embeddings` instance over HTTP.
    public init(
        baseURL: URL,
        modelID: String,
        facts: EngineEmbeddingFacts? = nil,
        prompts: [String: String] = [:],
        defaultPromptName: String? = nil,
        pooling: String = "mean",
        timeoutSeconds: Double = 300.0,
        session: URLSession? = nil
    ) {
        self.fixedBaseURL = baseURL
        self.modelID = modelID
        self.facts = facts
        self.nativeDimensions = facts?.nativeDimensions ?? Matryoshka.gemma2NativeDimensions
        self.supportedDimensions = facts?.dimensions ?? Matryoshka.gemma2Dimensions
        let mergedPrompts = prompts.isEmpty ? (facts?.prompts ?? [:]) : prompts
        self.promptCatalog = EmbeddingModelFacts.PromptCatalog(
            prompts: mergedPrompts,
            defaultPromptName: defaultPromptName,
            maxSeqLength: facts?.maxInputTokens
        )
        self.pooling = pooling
        self.timeoutSeconds = timeoutSeconds
        if let session {
            self.session = session
        } else {
            let cfg = URLSessionConfiguration.ephemeral
            cfg.timeoutIntervalForRequest = timeoutSeconds
            cfg.timeoutIntervalForResource = max(timeoutSeconds, 3600)
            self.session = URLSession(configuration: cfg)
        }
        #if os(macOS)
        self.managedLaunchConfig = nil
        #endif
    }

    #if os(macOS)
    /// Manages a dedicated `llama-server --embeddings --pooling <mode>` subprocess on macOS.
    public init(
        modelPath: URL,
        mmprojPath: URL? = nil,
        modelID: String? = nil,
        llamaServerBinary: URL? = nil,
        ctxSize: Int? = nil,
        nBatch: Int? = nil,
        nUBatch: Int? = nil,
        nGPULayers: Int = -1,
        flashAttention: Bool = true,
        pooling: String = "mean",
        timeoutSeconds: Double = 300.0,
        session: URLSession? = nil
    ) throws {
        guard let primaryGGUF = EmbeddingModelFacts.resolvePrimaryGGUFFile(at: modelPath) else {
            throw EmbeddingEngineError.checkpointNotFound(modelPath)
        }
        let dir = primaryGGUF.deletingLastPathComponent()
        let resolvedMMProj = mmprojPath ?? EmbeddingModelFacts.resolveMMProjFile(in: dir)
        let derivedFacts = EmbeddingModelFacts.deriveFacts(
            provider: .gguf,
            modelPath: primaryGGUF,
            mmprojPath: resolvedMMProj,
            computePlacement: "cpu_gpu_subprocess",
            ctxSize: ctxSize,
            nBatch: nBatch,
            nUBatch: nUBatch
        )
        let maxCtx = derivedFacts?.maxInputTokens ?? EmbeddingModelFacts.gemma2MaxInputTokens
        let profile = try EmbeddingModelFacts.ggufEmbeddingProfile(
            ctxSize: ctxSize,
            nBatch: nBatch,
            nUBatch: nUBatch,
            maxContext: maxCtx
        )
        guard let binary = Self.resolveLlamaServerBinary(explicit: llamaServerBinary) else {
            throw EmbeddingEngineError.runtimeUnavailable(
                "llama-server executable not found (checked HEYLOOK_LLAMA_SERVER_BIN, /opt/homebrew/bin/llama-server, /usr/local/bin/llama-server, PATH)"
            )
        }

        self.fixedBaseURL = nil
        self.modelID = modelID ?? primaryGGUF.deletingPathExtension().lastPathComponent
        self.facts = derivedFacts
        self.nativeDimensions = derivedFacts?.nativeDimensions ?? Matryoshka.gemma2NativeDimensions
        self.supportedDimensions = derivedFacts?.dimensions ?? Matryoshka.gemma2Dimensions
        self.promptCatalog = EmbeddingModelFacts.PromptCatalog.load(from: dir)
        self.pooling = pooling
        self.timeoutSeconds = timeoutSeconds
        if let session {
            self.session = session
        } else {
            let cfg = URLSessionConfiguration.ephemeral
            cfg.timeoutIntervalForRequest = timeoutSeconds
            cfg.timeoutIntervalForResource = max(timeoutSeconds, 3600)
            self.session = URLSession(configuration: cfg)
        }
        self.managedLaunchConfig = LaunchConfig(
            binaryURL: binary,
            modelFile: primaryGGUF,
            mmprojFile: resolvedMMProj,
            ctxSize: profile["ctx_size"] ?? maxCtx,
            batchSize: profile["n_batch"] ?? maxCtx,
            ubatchSize: profile["n_ubatch"] ?? maxCtx,
            nGPULayers: nGPULayers,
            flashAttention: flashAttention,
            pooling: pooling
        )
    }
    #endif

    public func unload() async {
        #if os(macOS)
        processHolder.terminateIfNeeded()
        #endif
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        guard !inputs.isEmpty else { return [] }
        let targetDim = dimensions ?? nativeDimensions
        guard supportedDimensions.contains(targetDim) else {
            throw EmbeddingValidationError.unsupportedDimensions(
                requested: targetDim,
                supported: supportedDimensions
            )
        }

        let allowedModalities = facts?.modalities ?? ["text", "image", "audio", "video"]
        var resolvedPrompts: [(prefix: String, promptName: String?)] = []
        resolvedPrompts.reserveCapacity(inputs.count)

        for (idx, item) in inputs.enumerated() {
            guard !item.parts.isEmpty else {
                throw EmbeddingValidationError.invalidInput(
                    statusCode: 422,
                    detail: "input[\(idx)]: embedding item content cannot be empty"
                )
            }
            if try item.resolvedVideoWindows(itemIndex: idx) != nil {
                // llama-server samples video frames internally; video windows run on MLX embedding models only.
                throw EmbeddingValidationError.videoWindowsUnsupported(
                    itemIndex: idx,
                    detail: "video windows run on MLX embedding models only"
                )
            }
            for part in item.parts {
                if case let .media(kind, _, _, _, _) = part,
                   !allowedModalities.contains(kind.rawValue) {
                    throw EmbeddingValidationError.unsupportedModality(
                        itemIndex: idx,
                        modality: kind.rawValue,
                        supported: allowedModalities
                    )
                }
            }
            let prompt = try EmbeddingModelFacts.resolvePrompt(
                for: item,
                catalog: promptCatalog,
                itemIndex: idx
            )
            resolvedPrompts.append(prompt)
        }

        let baseURL = try await ensureEndpointReady()
        var outputs: [EmbeddedOutput] = []
        outputs.reserveCapacity(inputs.count)

        for (idx, item) in inputs.enumerated() {
            try Task.checkCancellation()
            let prompt = resolvedPrompts[idx]
            var contentBlocks: [[String: Any]] = []
            var plainTextBuffer = ""
            var hasMedia = false

            if !prompt.prefix.isEmpty && !item.hasNonWhitespaceText {
                contentBlocks.append(["type": "text", "text": prompt.prefix])
                plainTextBuffer += prompt.prefix
            }

            var didApplyPrompt = false
            for part in item.parts {
                switch part {
                case .text(let s):
                    let finalText: String
                    if !didApplyPrompt && !prompt.prefix.isEmpty {
                        didApplyPrompt = true
                        finalText = (s.hasPrefix("task:") || s.hasPrefix("title:")) ? s : (prompt.prefix + s)
                    } else {
                        finalText = s
                    }
                    contentBlocks.append(["type": "text", "text": finalText])
                    plainTextBuffer += finalText

                case let .media(kind, mimeType, _, _, _):
                    hasMedia = true
                    let rawData = try part.resolvedData()
                    let uri = "data:\(mimeType);base64,\(rawData.base64EncodedString())"
                    switch kind {
                    case .image:
                        contentBlocks.append([
                            "type": "image_url",
                            "image_url": ["url": uri],
                        ])
                    case .audio, .video:
                        let key = "input_\(kind.rawValue)"
                        contentBlocks.append([
                            "type": key,
                            key: ["data": uri],
                        ])
                    default:
                        let str = String(data: rawData, encoding: .utf8) ?? ""
                        contentBlocks.append(["type": "text", "text": str])
                    }
                }
            }

            let response = try await postEmbeddingRequest(
                baseURL: baseURL,
                contentBlocks: contentBlocks,
                plainText: hasMedia ? nil : plainTextBuffer,
                itemIndex: idx
            )
            let projected = Matryoshka.project(response.embedding, dimensions: targetDim)
            guard projected.allSatisfy(\.isFinite), projected.contains(where: { $0 != 0 }) else {
                throw EmbeddingEngineError.invalidResponse(
                    status: 500,
                    detail: "input[\(idx)]: llama-server returned non-finite or zero embedding"
                )
            }
            outputs.append(
                EmbeddedOutput(
                    vector: projected,
                    windows: [],
                    inputTokens: response.promptTokens,
                    promptName: prompt.promptName
                )
            )
        }
        return outputs
    }

    // MARK: - HTTP Execution

    private func postEmbeddingRequest(
        baseURL: URL,
        contentBlocks: [[String: Any]],
        plainText: String?,
        itemIndex: Int
    ) async throws -> (embedding: [Float], promptTokens: Int?) {
        let primaryURL = baseURL.appendingPathComponent("v1/embeddings")
        let body: [String: Any] = [
            "input": [["content": contentBlocks]],
            "encoding_format": "float",
        ]

        do {
            let result = try await executeJSONPost(url: primaryURL, body: body, itemIndex: itemIndex)
            if let parsed = Self.parseOpenAIEmbeddingResponse(result) {
                return parsed
            }
            throw EmbeddingEngineError.invalidResponse(
                status: 502,
                detail: "input[\(itemIndex)]: malformed llama-server embedding response"
            )
        } catch let embErr as EmbeddingValidationError {
            throw embErr
        } catch let embErr as EmbeddingEngineError {
            // Only retry plain text against OpenAI string / /embedding if llama-server rejects `content` blocks on text-only runs.
            if case let .invalidResponse(status, _) = embErr,
               status == 400 || status == 404 || status == 422,
               let plainText {
                let fallbackBody: [String: Any] = [
                    "input": [plainText],
                    "encoding_format": "float",
                ]
                if status != 404,
                   let data = try? await executeJSONPost(url: primaryURL, body: fallbackBody, itemIndex: itemIndex),
                   let parsed = Self.parseOpenAIEmbeddingResponse(data) {
                    return parsed
                }
                let legacyURL = baseURL.appendingPathComponent("embedding")
                if let legacyData = try? await executeJSONPost(url: legacyURL, body: ["content": plainText], itemIndex: itemIndex),
                   let parsedLegacy = Self.parseLegacyEmbeddingResponse(legacyData) {
                    return parsedLegacy
                }
            }
            throw embErr
        }
    }

    private func executeJSONPost(
        url: URL,
        body: [String: Any],
        itemIndex: Int
    ) async throws -> Data {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = timeoutSeconds
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        do {
            let (data, response) = try await session.data(for: req)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 200
            guard (200..<300).contains(status) else {
                let snippet = String(data: data.prefix(4096), encoding: .utf8) ?? "HTTP \(status)"
                if status == 400 {
                    throw EmbeddingValidationError.invalidInput(
                        statusCode: 400,
                        detail: "input[\(itemIndex)]: llama-server 400: \(snippet)"
                    )
                }
                throw EmbeddingEngineError.invalidResponse(
                    status: status,
                    detail: "input[\(itemIndex)]: llama-server \(status): \(snippet)"
                )
            }
            return data
        } catch is CancellationError {
            throw CancellationError()
        } catch let validation as EmbeddingValidationError {
            throw validation
        } catch let engineErr as EmbeddingEngineError {
            throw engineErr
        } catch {
            // Mirror gguf_embeddings.py: terminate managed subprocess on timeout / unknown GPU completion.
            await unload()
            throw EmbeddingEngineError.runtimeUnavailable(
                "llama-server unreachable for embedding model '\(modelID)': \(error.localizedDescription)"
            )
        }
    }

    static func parseOpenAIEmbeddingResponse(_ data: Data) -> (embedding: [Float], promptTokens: Int?)? {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = root["data"] as? [[String: Any]],
              rows.count == 1,
              let rawNums = rows[0]["embedding"] as? [NSNumber],
              !rawNums.isEmpty
        else {
            return nil
        }
        let vec = rawNums.map(\.floatValue)
        let usage = root["usage"] as? [String: Any]
        let promptTokens = (usage?["prompt_tokens"] as? NSNumber)?.intValue
        return (vec, promptTokens)
    }

    static func parseLegacyEmbeddingResponse(_ data: Data) -> (embedding: [Float], promptTokens: Int?)? {
        guard let json = try? JSONSerialization.jsonObject(with: data) else {
            return nil
        }
        if let arr = json as? [[String: Any]], let first = arr.first {
            if let nested = first["embedding"] as? [[NSNumber]], let row = nested.first, !row.isEmpty {
                return (row.map(\.floatValue), (first["tokens_evaluated"] as? NSNumber)?.intValue)
            }
            if let flat = first["embedding"] as? [NSNumber], !flat.isEmpty {
                return (flat.map(\.floatValue), (first["tokens_evaluated"] as? NSNumber)?.intValue)
            }
        }
        if let obj = json as? [String: Any], let flat = obj["embedding"] as? [NSNumber], !flat.isEmpty {
            return (flat.map(\.floatValue), nil)
        }
        return nil
    }

    // MARK: - Subprocess Lifecycle (macOS)

    /// Builds `llama-server --embeddings --pooling <mode>` launch arguments cleanly for testing & spawning.
    public static func buildLlamaServerArgs(
        modelPath: String,
        host: String = "127.0.0.1",
        port: Int,
        mmprojPath: String? = nil,
        ctxSize: Int,
        batchSize: Int,
        ubatchSize: Int,
        nGPULayers: Int = -1,
        flashAttention: Bool = true,
        pooling: String = "mean"
    ) -> [String] {
        var args = [
            "-m", modelPath,
            "--host", host,
            "--port", String(port),
            "--embeddings",
            "--pooling", pooling,
            "-np", "1",
            "-ngl", String(nGPULayers),
            "--no-webui",
            "--ctx-size", String(ctxSize),
            "-b", String(batchSize),
            "-ub", String(ubatchSize),
            "-fa", flashAttention ? "on" : "off",
        ]
        if let mmprojPath, !mmprojPath.isEmpty {
            args.append(contentsOf: ["--mmproj", mmprojPath])
        }
        return args
    }

    private func ensureEndpointReady() async throws -> URL {
        if let fixedBaseURL {
            return fixedBaseURL
        }
        #if os(macOS)
        if let existing = processHolder.currentBaseURL() {
            return existing
        }
        guard let launch = managedLaunchConfig else {
            throw EmbeddingEngineError.runtimeUnavailable("No llama-server URL or model path configured.")
        }
        let port = Self.pickAvailableLoopbackPort()
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let proc = Process()
        proc.executableURL = launch.binaryURL
        proc.arguments = Self.buildLlamaServerArgs(
            modelPath: launch.modelFile.path,
            port: port,
            mmprojPath: launch.mmprojFile?.path,
            ctxSize: launch.ctxSize,
            batchSize: launch.batchSize,
            ubatchSize: launch.ubatchSize,
            nGPULayers: launch.nGPULayers,
            flashAttention: launch.flashAttention,
            pooling: launch.pooling
        )
        var env = ProcessInfo.processInfo.environment
        env.removeValue(forKey: "DYLD_LIBRARY_PATH")
        env.removeValue(forKey: "DYLD_FALLBACK_LIBRARY_PATH")
        proc.environment = env

        let stderrPipe = Pipe()
        proc.standardError = stderrPipe
        proc.standardOutput = FileHandle.nullDevice

        do {
            try proc.run()
        } catch {
            throw EmbeddingEngineError.runtimeUnavailable(
                "Failed to spawn llama-server at \(launch.binaryURL.path): \(error.localizedDescription)"
            )
        }
        processHolder.store(process: proc, baseURL: baseURL)

        let healthURL = baseURL.appendingPathComponent("health")
        let deadline = Date().addingTimeInterval(min(timeoutSeconds, 90.0))
        while Date() < deadline {
            try Task.checkCancellation()
            if !proc.isRunning {
                let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                let msg = String(data: errData.suffix(1024), encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                await unload()
                throw EmbeddingEngineError.runtimeUnavailable(
                    "llama-server exited prematurely (code \(proc.terminationStatus))\(msg.isEmpty ? "" : ": \(msg)")"
                )
            }
            var req = URLRequest(url: healthURL)
            req.timeoutInterval = 2.0
            if let (_, resp) = try? await session.data(for: req),
               let http = resp as? HTTPURLResponse,
               http.statusCode == 200 {
                return baseURL
            }
            try await Task.sleep(nanoseconds: 150_000_000)
        }
        await unload()
        throw EmbeddingEngineError.runtimeUnavailable(
            "llama-server embedding startup timed out for '\(modelID)'"
        )
        #else
        throw EmbeddingEngineError.runtimeUnavailable(
            "Local llama-server subprocess launch is only supported on macOS; pass baseURL on iOS."
        )
        #endif
    }

    #if os(macOS)
    private static func resolveLlamaServerBinary(explicit: URL?) -> URL? {
        let fm = FileManager.default
        if let explicit, fm.isExecutableFile(atPath: explicit.path) {
            return explicit
        }
        for key in ["HEYLOOK_LLAMA_SERVER_BIN", "HEYLOOK_LLAMA_SERVER"] {
            if let val = ProcessInfo.processInfo.environment[key],
               !val.isEmpty,
               fm.isExecutableFile(atPath: val) {
                return URL(fileURLWithPath: val)
            }
        }
        let fixedCandidates = [
            "/opt/homebrew/bin/llama-server",
            "/usr/local/bin/llama-server",
        ]
        for path in fixedCandidates where fm.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        if let pathEnv = ProcessInfo.processInfo.environment["PATH"] {
            for dir in pathEnv.split(separator: ":") {
                let candidate = URL(fileURLWithPath: String(dir)).appendingPathComponent("llama-server")
                if fm.isExecutableFile(atPath: candidate.path) {
                    return candidate
                }
            }
        }
        return nil
    }

    private static func pickAvailableLoopbackPort() -> Int {
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return 18290 }
        defer { Darwin.close(fd) }
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr.s_addr = CFSwapInt32HostToBig(INADDR_LOOPBACK)
        let bindRes = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.bind(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindRes == 0 else { return 18290 }
        var assigned = sockaddr_in()
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameRes = withUnsafeMutablePointer(to: &assigned) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.getsockname(fd, sockPtr, &len)
            }
        }
        guard nameRes == 0 else { return 18290 }
        return Int(UInt16(bigEndian: assigned.sin_port))
    }
    #endif
}
