import Foundation
import HeylookCore

/// Launch configuration for a `llama-server` subprocess (`GGUFSubprocessProvider`).
public struct GGUFSpawnConfig: Sendable, Equatable {
    public var binaryPath: String
    public var modelPath: String
    public var host: String
    public var port: Int
    public var nGPULayers: Int
    public var ctxSize: Int
    public var flashAttention: Bool
    public var batchSize: Int?
    public var ubatchSize: Int?
    public var mmprojPath: String?
    public var imageMaxTokens: Int?
    public var chatTemplateFile: String?
    public var draftModelPath: String?
    public var specType: String?
    public var specDraftMax: Int?
    public var specDraftPMin: Double?
    public var cacheTypeK: String?
    public var cacheTypeV: String?
    public var cacheRAMMiB: Int?
    public var extraArgs: [String]

    public init(
        binaryPath: String = "/opt/homebrew/bin/llama-server",
        modelPath: String,
        host: String = "127.0.0.1",
        port: Int = 18080,
        nGPULayers: Int = 999,
        ctxSize: Int = 8192,
        flashAttention: Bool = true,
        batchSize: Int? = nil,
        ubatchSize: Int? = nil,
        mmprojPath: String? = nil,
        imageMaxTokens: Int? = nil,
        chatTemplateFile: String? = nil,
        draftModelPath: String? = nil,
        specType: String? = nil,
        specDraftMax: Int? = nil,
        specDraftPMin: Double? = nil,
        cacheTypeK: String? = nil,
        cacheTypeV: String? = nil,
        cacheRAMMiB: Int? = nil,
        extraArgs: [String] = []
    ) {
        self.binaryPath = binaryPath
        self.modelPath = modelPath
        self.host = host
        self.port = port
        self.nGPULayers = nGPULayers
        self.ctxSize = ctxSize
        self.flashAttention = flashAttention
        self.batchSize = batchSize
        self.ubatchSize = ubatchSize
        self.mmprojPath = mmprojPath
        self.imageMaxTokens = imageMaxTokens
        self.chatTemplateFile = chatTemplateFile
        self.draftModelPath = draftModelPath
        self.specType = specType
        self.specDraftMax = specDraftMax
        self.specDraftPMin = specDraftPMin
        self.cacheTypeK = cacheTypeK
        self.cacheTypeV = cacheTypeV
        self.cacheRAMMiB = cacheRAMMiB
        self.extraArgs = extraArgs
    }

    /// Auto-sizes `-ub` (`2048` when headroom >= 16 GB, else `512`), `-cram` (`cacheRAMMiB` from
    /// unified memory headroom clamped to `[8192, 32768]` MiB), the 4-rung chat template ladder,
    /// non-causal vision projector `--image-max-tokens` clamping to `ubatchSize`, and paired MTP drafter.
    public static func autoSized(
        for description: EngineDescription,
        headroomGB: Double? = nil,
        sidecar: ModelSidecarConfig? = nil
    ) -> GGUFSpawnConfig {
        let snap = HardwareMemoryMonitor.snapshot()
        let modelGB = Double(description.estimatedMemoryMB) / 1024.0
        let workingSetGB = snap.metalRecommendedMaxWorkingSetGB > 0
            ? snap.metalRecommendedMaxWorkingSetGB
            : snap.totalPhysicalRAMGB * 0.75
        let effectiveHeadroomGB = headroomGB ?? max(0, workingSetGB - modelGB)

        let thinHeadroomGB = 16.0
        let autoUbatch = effectiveHeadroomGB >= thinHeadroomGB ? 2048 : 512
        let autoCacheRAMMiB: Int? = if effectiveHeadroomGB >= thinHeadroomGB {
            min(32768, max(8192, Int(effectiveHeadroomGB / 2.0) * 1024))
        } else {
            nil
        }

        let dirURL = URL(fileURLWithPath: description.modelPath)
        let resolvedSidecar = sidecar ?? (try? String(
            contentsOf: dirURL.appendingPathComponent("model.heylook.toml"),
            encoding: .utf8
        )).map(ModelSidecarConfig.parseTOML)

        let templateResolution = HeylookConfig.resolveGGUFTemplate(in: dirURL, sidecar: resolvedSidecar)
        let drafter = HeylookConfig.discoverDrafter(in: dirURL, provider: .gguf, sidecar: resolvedSidecar)
        let mmproj = HeylookConfig.discoverMMProj(in: dirURL)
        let assignedPort = GGUFPortPool.shared.acquirePort(for: description.modelID)

        // Check primary GGUF header for hybrid-recurrent SSM state (`--cache-ram 8192`) and built-in MTP
        var ggufHeader: GGUFHeaderReader.HeaderSummary?
        if dirURL.pathExtension.lowercased() == "gguf" {
            ggufHeader = GGUFHeaderReader.readFileHeader(at: dirURL)
        } else if let items = try? FileManager.default.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil),
                  let firstGGUF = items.first(where: { $0.pathExtension.lowercased() == "gguf" && !$0.lastPathComponent.lowercased().contains("mmproj") }) {
            ggufHeader = GGUFHeaderReader.readFileHeader(at: firstGGUF)
        }

        // Check mmproj header for non-causal vision projectors (`clip.projector_type`) that require
        // `--image-max-tokens <= n_ubatch` so `llama_context::decode` never aborts.
        var clampedImageMaxTokens: Int? = resolvedSidecar?.imageMaxTokens
        if let mmproj,
           let mmprojHeader = GGUFHeaderReader.readFileHeader(at: URL(fileURLWithPath: mmproj)),
           mmprojHeader.hasNonCausalVisionProjector {
            clampedImageMaxTokens = min(clampedImageMaxTokens ?? autoUbatch, autoUbatch)
        }

        let effectiveCacheRAM: Int?
        if let autoCacheRAMMiB {
            effectiveCacheRAM = autoCacheRAMMiB
        } else if ggufHeader?.isHybridRecurrent == true {
            effectiveCacheRAM = 8192
        } else {
            effectiveCacheRAM = nil
        }

        let unset = Set(resolvedSidecar?.unsetKeys ?? [])
        let effectiveSpecType: String?
        if let spec = drafter.specType {
            effectiveSpecType = spec
        } else if drafter.draftModelPath == nil,
                  ggufHeader?.hasBuiltInMTP == true,
                  !unset.contains("spec_type") {
            effectiveSpecType = "draft-mtp"
        } else {
            effectiveSpecType = nil
        }

        return GGUFSpawnConfig(
            modelPath: description.modelPath,
            port: assignedPort,
            ctxSize: description.contextSize.value,
            flashAttention: description.flashAttention.value,
            ubatchSize: autoUbatch,
            mmprojPath: mmproj,
            imageMaxTokens: clampedImageMaxTokens,
            chatTemplateFile: templateResolution.resolvedFilePath,
            draftModelPath: drafter.draftModelPath,
            specType: effectiveSpecType,
            cacheRAMMiB: effectiveCacheRAM
        )
    }
}

/// Tier 3 Provider (macOS): Out-of-process `llama-server` subprocess (`-np 1`) for `.gguf` models.
/// Implements the CLI argument builder, environment sanitizer (`LLAMA_ARG_LOG_FILE` removal +
/// `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=604800`), `/health` + `/props` lifecycle, and OpenAI SSE
/// delta + `timings` + mid-stream `error` parser.
public actor GGUFSubprocessProvider: InferenceProvider {
    public struct OpenAISSEDelta: Sendable, Equatable {
        public let content: String
        public let reasoningContent: String
        public let finishReason: String?
        public let promptTokens: Int?
        public let cachedTokens: Int?
        public let predictedTokens: Int?
        public let promptTPS: Double?
        public let generationTPS: Double?
        public let draftedTokens: Int?
        public let acceptedTokens: Int?
        public let errorMessage: String?

        public init(
            content: String = "",
            reasoningContent: String = "",
            finishReason: String? = nil,
            promptTokens: Int? = nil,
            cachedTokens: Int? = nil,
            predictedTokens: Int? = nil,
            promptTPS: Double? = nil,
            generationTPS: Double? = nil,
            draftedTokens: Int? = nil,
            acceptedTokens: Int? = nil,
            errorMessage: String? = nil
        ) {
            self.content = content
            self.reasoningContent = reasoningContent
            self.finishReason = finishReason
            self.promptTokens = promptTokens
            self.cachedTokens = cachedTokens
            self.predictedTokens = predictedTokens
            self.promptTPS = promptTPS
            self.generationTPS = generationTPS
            self.draftedTokens = draftedTokens
            self.acceptedTokens = acceptedTokens
            self.errorMessage = errorMessage
        }
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .gguf
    public nonisolated let allowSyntheticFallback: Bool
    private var templateInfo: ModelTemplateInfo
    private var descriptionState: EngineDescription
    public nonisolated let spawnConfig: GGUFSpawnConfig
    private var prefixCache = PrefixCacheTracker()
    private var unloadCount: Int = 0

    #if os(macOS)
    private var process: Process?
    #endif

    public init(
        description: EngineDescription,
        spawnConfig: GGUFSpawnConfig? = nil,
        allowSyntheticFallback: Bool = false
    ) {
        self.modelID = description.modelID
        self.descriptionState = description
        self.spawnConfig = spawnConfig ?? GGUFSpawnConfig.autoSized(for: description)
        self.allowSyntheticFallback = allowSyntheticFallback
        self.templateInfo = ModelTemplateInfo.inspect(
            modelDirectory: URL(fileURLWithPath: description.modelPath),
            modelID: description.modelID
        )
    }

    /// Builds the `llama-server` CLI flag array matching `llama_server_provider.py`.
    public static func buildLlamaServerArgs(config: GGUFSpawnConfig) -> [String] {
        var args: [String] = [
            "-m", config.modelPath,
            "--host", config.host,
            "--port", String(config.port),
            "-np", "1",
            "-ngl", String(config.nGPULayers),
            "--no-webui",
        ]
        if config.ctxSize > 0 {
            args.append(contentsOf: ["--ctx-size", String(config.ctxSize)])
        }
        args.append(contentsOf: ["-fa", config.flashAttention ? "on" : "off"])
        if let batchSize = config.batchSize {
            args.append(contentsOf: ["-b", String(batchSize)])
        }
        if let ubatchSize = config.ubatchSize {
            args.append(contentsOf: ["-ub", String(ubatchSize)])
        }
        if let mmproj = config.mmprojPath, !mmproj.isEmpty {
            args.append(contentsOf: ["--mmproj", mmproj])
            if let imgMax = config.imageMaxTokens {
                args.append(contentsOf: ["--image-max-tokens", String(imgMax)])
            }
        }
        if let tmpl = config.chatTemplateFile, !tmpl.isEmpty {
            args.append(contentsOf: ["--chat-template-file", tmpl])
        }

        let hasDrafter = (config.draftModelPath?.isEmpty == false) || (config.specType?.isEmpty == false)
        if let draftModel = config.draftModelPath, !draftModel.isEmpty {
            args.append(contentsOf: ["-md", draftModel])
        }
        if let specType = config.specType, !specType.isEmpty {
            args.append(contentsOf: ["--spec-type", specType])
        }
        if hasDrafter {
            if let nMax = config.specDraftMax {
                args.append(contentsOf: ["--spec-draft-n-max", String(nMax)])
            }
            if let pMin = config.specDraftPMin {
                args.append(contentsOf: ["--spec-draft-p-min", String(pMin)])
            }
        }
        if let ctk = config.cacheTypeK, !ctk.isEmpty {
            args.append(contentsOf: ["-ctk", ctk])
        }
        if let ctv = config.cacheTypeV, !ctv.isEmpty {
            args.append(contentsOf: ["-ctv", ctv])
        }
        if let cram = config.cacheRAMMiB {
            args.append(contentsOf: ["-cram", String(cram)])
        }
        args.append(contentsOf: ["-lv", "4"])
        args.append(contentsOf: config.extraArgs)
        return args
    }

    /// Sanitizes the child environment for `llama-server`:
    /// 1. Strips `LLAMA_ARG_LOG_FILE` so slot/cache telemetry flows to piped stderr.
    /// 2. Sets `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=604800` if not already set so Metal weight
    ///    buffers stay wired across idle gaps.
    public static func childEnvironment(
        from baseEnv: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        var env = baseEnv
        env.removeValue(forKey: "LLAMA_ARG_LOG_FILE")
        if env["GGML_METAL_RESIDENCY_KEEP_ALIVE_S"] == nil {
            env["GGML_METAL_RESIDENCY_KEEP_ALIVE_S"] = "604800"
        }
        return env
    }

    /// Parses a single `data: {...}` SSE line from `llama-server` `/v1/chat/completions`,
    /// including mid-stream `data: {"error": ...}` frames so decode OOM/slot errors are never swallowed.
    public static func parseOpenAISSEDataLine(_ line: String) -> OpenAISSEDelta? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("data:") else { return nil }
        let payload = trimmed.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        if payload.isEmpty || payload == "[DONE]" {
            return nil
        }
        guard let data = payload.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        if let errDict = json["error"] as? [String: Any] {
            let msg = (errDict["message"] as? String)
                ?? (errDict["type"] as? String)
                ?? "llama-server mid-stream error"
            return OpenAISSEDelta(errorMessage: msg)
        } else if let errStr = json["error"] as? String, !errStr.isEmpty {
            return OpenAISSEDelta(errorMessage: errStr)
        }

        var content = ""
        var reasoningContent = ""
        var finishReason: String?

        if let choices = json["choices"] as? [[String: Any]], let first = choices.first {
            finishReason = first["finish_reason"] as? String
            if let delta = first["delta"] as? [String: Any] {
                content = (delta["content"] as? String) ?? ""
                reasoningContent = (delta["reasoning_content"] as? String) ?? ""
            }
        }

        let usage = json["usage"] as? [String: Any]
        let timings = json["timings"] as? [String: Any]

        let promptTokens = (timings?["prompt_n"] as? Int) ?? (usage?["prompt_tokens"] as? Int)
        let cachedTokens = timings?["cache_n"] as? Int
        let predictedTokens = (timings?["predicted_n"] as? Int) ?? (usage?["completion_tokens"] as? Int)
        let promptTPS = timings?["prompt_per_second"] as? Double
        let generationTPS = timings?["predicted_per_second"] as? Double
        let draftedTokens = timings?["draft_n"] as? Int
        let acceptedTokens = timings?["draft_n_accepted"] as? Int

        return OpenAISSEDelta(
            content: content,
            reasoningContent: reasoningContent,
            finishReason: finishReason,
            promptTokens: promptTokens,
            cachedTokens: cachedTokens,
            predictedTokens: predictedTokens,
            promptTPS: promptTPS,
            generationTPS: generationTPS,
            draftedTokens: draftedTokens,
            acceptedTokens: acceptedTokens
        )
    }

    /// Converts an Anthropic-style `MessagesRequest` (including `image` and `audio` attachments)
    /// into the OpenAI `/v1/chat/completions` `messages` array expected by `llama-server --mmproj`.
    public static func buildOpenAIChatMessages(from request: MessagesRequest) -> [[String: Any]] {
        var wireMessages: [[String: Any]] = []
        if let sys = request.system, !sys.isEmpty {
            wireMessages.append(["role": "system", "content": sys])
        }
        for turn in request.messages {
            var msgDict: [String: Any] = ["role": turn.role]
            if let thinking = turn.thinking, !thinking.isEmpty {
                msgDict["reasoning_content"] = thinking
            }
            if turn.attachments.isEmpty {
                msgDict["content"] = turn.content
                wireMessages.append(msgDict)
            } else {
                var parts: [[String: Any]] = []
                for att in turn.attachments {
                    if att.kind == "image" {
                        parts.append([
                            "type": "image_url",
                            "image_url": [
                                "url": "data:\(att.mediaType);base64,\(att.dataBase64)"
                            ]
                        ])
                    } else if att.kind == "audio" {
                        let audioFormat = att.mediaType.components(separatedBy: "/").last ?? "wav"
                        parts.append([
                            "type": "input_audio",
                            "input_audio": [
                                "data": att.dataBase64,
                                "format": audioFormat
                            ]
                        ])
                    }
                }
                if !turn.content.isEmpty {
                    parts.append([
                        "type": "text",
                        "text": turn.content
                    ])
                }
                msgDict["content"] = parts
                wireMessages.append(msgDict)
            }
        }
        return wireMessages
    }

    public struct GGUFServerProps: Sendable, Equatable {
        public let contextSize: Int?
        public let chatTemplate: String?
        public let totalSlots: Int?
        public let modelPath: String?

        public init(
            contextSize: Int? = nil,
            chatTemplate: String? = nil,
            totalSlots: Int? = nil,
            modelPath: String? = nil
        ) {
            self.contextSize = contextSize
            self.chatTemplate = chatTemplate
            self.totalSlots = totalSlots
            self.modelPath = modelPath
        }
    }

    /// Parses `GET /props` JSON from a running `llama-server` instance to extract runtime-observed
    /// `n_ctx` (`default_generation_settings.n_ctx`), `chat_template`, and `total_slots`.
    public static func parsePropsJSON(_ data: Data) -> GGUFServerProps? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        let defaultGen = json["default_generation_settings"] as? [String: Any]
        let nCtx = (defaultGen?["n_ctx"] as? NSNumber)?.intValue
            ?? (json["n_ctx"] as? NSNumber)?.intValue
        let chatTmpl = (json["chat_template"] as? String)
            ?? (defaultGen?["chat_template"] as? String)
        let slots = (json["total_slots"] as? NSNumber)?.intValue
        let modelPath = json["model_path"] as? String
        return GGUFServerProps(
            contextSize: nCtx,
            chatTemplate: chatTmpl,
            totalSlots: slots,
            modelPath: modelPath
        )
    }

    /// Upgrades `descriptionState.contextSize` and `descriptionState.templateSource` (plus
    /// `reasoningFormat` and `thinkingControls` when `props.chatTemplate` is present) to `.observed`
    /// provenance from `llama-server` `GET /props`.
    public func applyObservedProps(_ props: GGUFServerProps) {
        let newCtx = (props.contextSize ?? 0) > 0
            ? ProvenanceField(value: props.contextSize!, provenance: .observed, effect: .reload)
            : descriptionState.contextSize
        let newTmpl = (props.chatTemplate?.isEmpty == false)
            ? ProvenanceField(value: "llama-server:/props", provenance: .observed, effect: .reload)
            : descriptionState.templateSource
        if let observedTemplate = props.chatTemplate, !observedTemplate.isEmpty {
            self.templateInfo = ModelTemplateInfo.analyze(
                templateText: observedTemplate,
                specialTokens: self.templateInfo.specialTokens,
                eosTokenIDs: self.templateInfo.eosTokenIDs,
                templateSource: "llama-server:/props",
                modelID: descriptionState.modelID
            )
        }
        let newReasoningFmt: ProvenanceField<String> = (props.chatTemplate?.isEmpty == false)
            ? ProvenanceField(value: self.templateInfo.reasoningFormat.wireIdentifier, provenance: .observed, effect: .reload)
            : descriptionState.reasoningFormat
        descriptionState = EngineDescription(
            modelID: descriptionState.modelID,
            provider: descriptionState.provider,
            modelPath: descriptionState.modelPath,
            architecture: descriptionState.architecture,
            quantization: descriptionState.quantization,
            computePlacement: descriptionState.computePlacement,
            isPinned: descriptionState.isPinned,
            isLoaded: descriptionState.isLoaded,
            contextSize: newCtx,
            flashAttention: descriptionState.flashAttention,
            supportsVision: descriptionState.supportsVision,
            supportsAudio: descriptionState.supportsAudio,
            supportsEmbeddings: descriptionState.supportsEmbeddings,
            supportsThinking: descriptionState.supportsThinking,
            thinkingBudget: descriptionState.thinkingBudget,
            reasoningFormat: newReasoningFmt,
            thinkingControls: self.templateInfo.thinkingControls,
            templateSource: newTmpl,
            estimatedMemoryMB: descriptionState.estimatedMemoryMB,
            tags: descriptionState.tags
        )
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func isPrefixCacheWarm() -> Bool {
        prefixCache.isWarm
    }

    public func totalUnloadCount() -> Int {
        unloadCount
    }

    public func unload() async {
        unloadCount += 1
        GGUFPortPool.shared.releasePort(for: modelID)
        #if os(macOS)
        SubprocessLifecycleRegistry.shared.terminate(for: modelID)
        process = nil
        #endif
        prefixCache.reset()
        descriptionState.isLoaded = false
    }

    /// True when both the `llama-server` executable and `.gguf` file exist on disk.
    public nonisolated var canLaunchLiveSubprocess: Bool {
        let fm = FileManager.default
        return fm.isExecutableFile(atPath: spawnConfig.binaryPath)
            && fm.fileExists(atPath: spawnConfig.modelPath)
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let cacheReport = prefixCache.evaluate(system: request.system, messages: request.messages)
        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value

        #if os(macOS)
        if canLaunchLiveSubprocess {
            return generateWithLiveSubprocess(
                requestID: requestID,
                request: request,
                effectiveSampling: effectiveSampling,
                fallbackCacheReport: cacheReport,
                enableThinking: enableThinking,
                registry: registry
            )
        }
        #endif

        guard allowSyntheticFallback else {
            let fm = FileManager.default
            let err: ProviderExecutionError = !fm.isExecutableFile(atPath: spawnConfig.binaryPath)
                ? .binaryNotFound(modelID: modelID, binaryPath: spawnConfig.binaryPath)
                : .checkpointNotFound(modelID: modelID, path: spawnConfig.modelPath)
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: err)
            }
        }

        let userText = request.messages.last?.content ?? ""
        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Evaluating GGUF slot 0...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[GGUF \(modelID)] ", userText])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            cacheReport: cacheReport,
            startsInThinking: false,
            reasoningFormat: templateInfo.reasoningFormat,
            thinkingEnabled: enableThinking,
            strippableSpecialTokens: Array(templateInfo.strippableSpecialTokens),
            computePlacement: .ggufSubprocess,
            registry: registry,
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        )
    }

    private func recordAssistantTurn(_ reply: String) {
        prefixCache.recordAssistantReply(reply)
    }

    #if os(macOS)
    /// Spawns and registers the child subprocess in `SubprocessLifecycleRegistry.shared`.
    @discardableResult
    public func launchSubprocess(
        customArguments: [String]? = nil,
        checkHealth: Bool = true
    ) async throws -> Int32 {
        if let proc = process, proc.isRunning {
            return proc.processIdentifier
        }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: spawnConfig.binaryPath)
        proc.arguments = customArguments ?? Self.buildLlamaServerArgs(config: spawnConfig)
        proc.environment = Self.childEnvironment()
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        self.process = proc
        SubprocessLifecycleRegistry.shared.register(proc, for: modelID)

        guard checkHealth else {
            descriptionState.isLoaded = true
            return proc.processIdentifier
        }

        let healthURL = URL(string: "http://\(spawnConfig.host):\(spawnConfig.port)/health")!
        let propsURL = URL(string: "http://\(spawnConfig.host):\(spawnConfig.port)/props")!
        for _ in 0..<120 {
            if let (_, resp) = try? await URLSession.shared.data(from: healthURL),
               let http = resp as? HTTPURLResponse,
               http.statusCode == 200 {
                descriptionState.isLoaded = true
                if let (propsData, propsResp) = try? await URLSession.shared.data(from: propsURL),
                   let propsHTTP = propsResp as? HTTPURLResponse,
                   propsHTTP.statusCode == 200,
                   let parsedProps = Self.parsePropsJSON(propsData) {
                    applyObservedProps(parsedProps)
                }
                return proc.processIdentifier
            }
            try await Task.sleep(for: .milliseconds(250))
        }
        return proc.processIdentifier
    }

    private func ensureSubprocessRunning() async throws {
        _ = try await launchSubprocess(checkHealth: true)
    }

    private func generateWithLiveSubprocess(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        fallbackCacheReport: CacheReport,
        enableThinking: Bool,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let cfg = self.spawnConfig
        let currentModelID = self.modelID
        let activeTemplate = self.templateInfo
        let baseURL = URL(string: "http://\(cfg.host):\(cfg.port)")!

        return StreamingPipeline.transformEventStream(
            requestID: requestID,
            cacheReport: fallbackCacheReport,
            defaultSpeculativeReport: nil,
            startsInThinking: false,
            reasoningFormat: activeTemplate.reasoningFormat,
            thinkingEnabled: enableThinking,
            strippableSpecialTokens: Array(activeTemplate.strippableSpecialTokens),
            computePlacement: .ggufSubprocess,
            registry: registry,
            onReasoningEndHook: {
                var ctrlReq = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions/control"))
                ctrlReq.httpMethod = "POST"
                ctrlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
                ctrlReq.httpBody = Data("{\"action\":\"reasoning_end\"}".utf8)
                _ = try? await URLSession.shared.data(for: ctrlReq)
            },
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        ) { [weak self] eventCont in
            guard let self else { return }
            try await self.ensureSubprocessRunning()
            let chatURL = baseURL.appendingPathComponent("v1/chat/completions")
            var urlReq = URLRequest(url: chatURL)
            urlReq.httpMethod = "POST"
            urlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")

            let wireMessages = Self.buildOpenAIChatMessages(from: request)
            var templateKwargs: [String: Any] = [
                activeTemplate.thinkingControls.switchVariable ?? "enable_thinking": enableThinking
            ]
            if let depthVar = activeTemplate.thinkingControls.depthVariable,
               let depthVal = effectiveSampling.thinkingDepth ?? request.thinkingDepth,
               !depthVal.isEmpty {
                templateKwargs[depthVar] = depthVal
            }

            var body: [String: Any] = [
                "messages": wireMessages,
                "stream": true,
                "cache_prompt": true,
                "temperature": effectiveSampling.temperature ?? 0.7,
                "top_p": effectiveSampling.topP ?? 0.95,
                "top_k": effectiveSampling.topK ?? 40,
                "min_p": effectiveSampling.minP ?? 0.0,
                "max_tokens": effectiveSampling.maxTokens ?? 2048,
                "chat_template_kwargs": templateKwargs,
            ]
            if let repPen = effectiveSampling.repetitionPenalty {
                body["repeat_penalty"] = repPen
            }
            if let presPen = effectiveSampling.presencePenalty {
                body["presence_penalty"] = presPen
            }
            if let seed = effectiveSampling.seed {
                body["seed"] = seed
            }
            urlReq.httpBody = try JSONSerialization.data(withJSONObject: body)

            let (bytes, _) = try await URLSession.shared.bytes(for: urlReq)
            var sawPreSplitReasoning = false

            for try await line in bytes.lines {
                guard let delta = Self.parseOpenAISSEDataLine(line) else { continue }
                if let errMsg = delta.errorMessage {
                    throw ProviderExecutionError.runtimeUnavailable(
                        modelID: currentModelID,
                        detail: "llama-server error: \(errMsg)"
                    )
                }

                let specReport: SpeculativeReport? = if let d = delta.draftedTokens, let a = delta.acceptedTokens, d > 0 {
                    SpeculativeReport(
                        drafterID: cfg.draftModelPath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "\(currentModelID)-drafter",
                        mode: cfg.specType ?? "draft",
                        draftedTokens: d,
                        acceptedTokens: a
                    )
                } else {
                    nil
                }

                if delta.promptTokens != nil || delta.cachedTokens != nil || delta.promptTPS != nil || delta.generationTPS != nil || specReport != nil {
                    eventCont.yield(
                        .telemetry(
                            promptTokens: delta.promptTokens,
                            cachedTokens: delta.cachedTokens,
                            promptTPS: delta.promptTPS,
                            generationTPS: delta.generationTPS,
                            speculativeReport: specReport
                        )
                    )
                }

                if !delta.reasoningContent.isEmpty {
                    sawPreSplitReasoning = true
                    eventCont.yield(.splitReasoning(delta.reasoningContent))
                    if !delta.content.isEmpty {
                        eventCont.yield(.splitContent(delta.content))
                    }
                } else if !delta.content.isEmpty {
                    // When `llama-server` already split reasoning into `reasoning_content`,
                    // treat `delta.content` as pre-split visible content so we never double-parse.
                    if sawPreSplitReasoning {
                        eventCont.yield(.splitContent(delta.content))
                    } else {
                        eventCont.yield(.rawToken(delta.content))
                    }
                }
            }
        }
    }
    #endif
}
