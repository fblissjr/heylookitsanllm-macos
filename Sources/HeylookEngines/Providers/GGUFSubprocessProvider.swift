import CoreGraphics
import Foundation
import HeylookCore
import ImageIO
import UniformTypeIdentifiers

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
    public var cacheReuse: Int?
    public var ctxCheckpoints: Int?
    public var checkpointMinStep: Int?
    public var reasoningFormatFlag: String?
    public var reasoningBudget: Int?
    public var reasoningPreserve: Bool
    public var extraArgs: [String]

    public init(
        binaryPath: String = GGUFSubprocessProvider.resolveBinaryPath(),
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
        cacheReuse: Int? = nil,
        ctxCheckpoints: Int? = nil,
        checkpointMinStep: Int? = nil,
        reasoningFormatFlag: String? = nil,
        reasoningBudget: Int? = nil,
        reasoningPreserve: Bool = true,
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
        self.cacheReuse = cacheReuse
        self.ctxCheckpoints = ctxCheckpoints
        self.checkpointMinStep = checkpointMinStep
        self.reasoningFormatFlag = reasoningFormatFlag
        self.reasoningBudget = reasoningBudget
        self.reasoningPreserve = reasoningPreserve
        self.extraArgs = extraArgs
    }

    public struct RAMFitAssessment: Sendable, Equatable, Codable {
        public let modelWeightGB: Double
        public let mmprojWeightGB: Double
        public let drafterWeightGB: Double
        public let estimatedKVGB: Double
        public let availableWorkingSetGB: Double
        public let remainingHeadroomGB: Double
        public let fitsInMemory: Bool
        public let droppedDrafterForMemory: Bool
        public let recommendedCtxSize: Int

        public init(
            modelWeightGB: Double,
            mmprojWeightGB: Double,
            drafterWeightGB: Double,
            estimatedKVGB: Double,
            availableWorkingSetGB: Double,
            remainingHeadroomGB: Double,
            fitsInMemory: Bool,
            droppedDrafterForMemory: Bool,
            recommendedCtxSize: Int
        ) {
            self.modelWeightGB = modelWeightGB
            self.mmprojWeightGB = mmprojWeightGB
            self.drafterWeightGB = drafterWeightGB
            self.estimatedKVGB = estimatedKVGB
            self.availableWorkingSetGB = availableWorkingSetGB
            self.remainingHeadroomGB = remainingHeadroomGB
            self.fitsInMemory = fitsInMemory
            self.droppedDrafterForMemory = droppedDrafterForMemory
            self.recommendedCtxSize = recommendedCtxSize
        }
    }

    /// Evaluates real disk weight footprint + KV cache sizing against Metal working-set headroom
    /// by delegating to `RAMFitCalculator.assess(...)`.
    public static func assessRAMFit(
        for description: EngineDescription,
        ctxSize: Int? = nil,
        cacheTypeK: String? = nil,
        headroomOverrideGB: Double? = nil,
        sidecar: ModelSidecarConfig? = nil
    ) -> RAMFitAssessment {
        let report = RAMFitCalculator.assess(
            for: description,
            overrides: RAMFitOverrides(ctxSize: ctxSize, cacheTypeK: cacheTypeK),
            headroomOverrideGB: headroomOverrideGB,
            sidecar: sidecar
        )
        return RAMFitAssessment(
            modelWeightGB: report.weightsGB,
            mmprojWeightGB: report.mmprojGB,
            drafterWeightGB: report.drafterGB,
            estimatedKVGB: report.kvGB,
            availableWorkingSetGB: report.workingSetGB,
            remainingHeadroomGB: report.headroomGB,
            fitsInMemory: report.fits,
            droppedDrafterForMemory: report.droppedDrafterForMemory,
            recommendedCtxSize: report.recommendedCtxSize
        )
    }

    /// Auto-sizes `ubatchSize`, `cacheRAMMiB`, template resolution (with media-blind sidecar fallback),
    /// `--image-max-tokens` clamping for non-causal vision projectors, composable specialization flags
    /// (`Gemma4Specialization` / `HybridRecurrentSWASpecialization`: `--cache-reuse` off with `--mmproj` or
    /// SWA/SSM, `--ctx-checkpoints 32 --checkpoint-min-step 256`, `--reasoning-preserve`), and drafter flags.
    public static func autoSized(
        for description: EngineDescription,
        headroomGB: Double? = nil,
        sidecar: ModelSidecarConfig? = nil
    ) -> GGUFSpawnConfig {
        let rawURL = URL(fileURLWithPath: description.modelPath)
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: description.modelPath)
        let resolvedSidecar = sidecar ?? HeylookConfig.loadSidecar(in: dirURL)

        let fitReport = RAMFitCalculator.assess(
            for: description,
            overrides: RAMFitOverrides(ctxSize: resolvedSidecar?.ctxSize ?? description.contextSize.value),
            headroomOverrideGB: headroomGB,
            sidecar: resolvedSidecar
        )
        let effectiveHeadroomGB = headroomGB ?? max(0, fitReport.workingSetGB - fitReport.weightsGB)

        let thinHeadroomGB = 16.0
        let autoUbatch = effectiveHeadroomGB >= thinHeadroomGB ? 2048 : 512
        let autoCacheRAMMiB: Int? = if effectiveHeadroomGB >= thinHeadroomGB {
            min(32768, max(8192, Int(effectiveHeadroomGB / 2.0) * 1024))
        } else {
            nil
        }

        let templateResolution = HeylookConfig.resolveGGUFTemplate(in: dirURL, sidecar: resolvedSidecar)
        let drafter = HeylookConfig.discoverDrafter(in: dirURL, provider: .gguf, sidecar: resolvedSidecar)
        let mmproj = HeylookConfig.discoverMMProj(in: dirURL)
        let assignedPort = GGUFPortPool.shared.acquirePort(for: description.modelID)

        var ggufHeader: GGUFHeaderReader.HeaderSummary?
        var resolvedGGUFModelPath = description.modelPath
        if rawURL.pathExtension.lowercased() == "gguf" {
            ggufHeader = GGUFHeaderReader.readFileHeader(at: rawURL)
        } else if let items = try? FileManager.default.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil) {
            let ggufFiles = items
                .filter { url in
                    let lower = url.lastPathComponent.lowercased()
                    return url.pathExtension.lowercased() == "gguf"
                        && !lower.contains("mmproj")
                        && !lower.hasPrefix("draft")
                        && !lower.contains("-draft")
                        && !lower.hasPrefix("mtp-")
                        && !lower.contains("-mtp")
                        && !lower.hasPrefix("eagle")
                        && !lower.contains("-dspark-")
                }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            if let firstGGUF = ggufFiles.first {
                ggufHeader = GGUFHeaderReader.readFileHeader(at: firstGGUF)
                resolvedGGUFModelPath = firstGGUF.standardizedFileURL.path
            }
        }

        var resolvedTemplateFile = templateResolution.resolvedFilePath
        if mmproj != nil,
           let candidateFile = resolvedTemplateFile,
           templateResolution.source.provenance != .configured,
           let sidecarText = try? String(contentsOfFile: candidateFile, encoding: .utf8) {
            let sidecarCaps = HeylookJinjaRenderer.detectCapabilities(template: sidecarText)
            if !sidecarCaps.supportsVision && !sidecarCaps.supportsAudio,
               let embedded = ggufHeader?.embeddedChatTemplate {
                let embeddedCaps = HeylookJinjaRenderer.detectCapabilities(template: embedded)
                if embeddedCaps.supportsVision || embeddedCaps.supportsAudio {
                    resolvedTemplateFile = nil
                }
            }
        }

        var clampedImageMaxTokens: Int? = resolvedSidecar?.imageMaxTokens
        if let mmproj,
           let mmprojHeader = GGUFHeaderReader.readFileHeader(at: URL(fileURLWithPath: mmproj)),
           mmprojHeader.hasNonCausalVisionProjector {
            clampedImageMaxTokens = min(clampedImageMaxTokens ?? autoUbatch, autoUbatch)
        }

        let signature = CheckpointSignature.inspect(
            modelDirectoryOrFile: URL(fileURLWithPath: resolvedGGUFModelPath),
            drafterDirectory: drafter.draftModelPath.map { URL(fileURLWithPath: $0) },
            mmprojPath: mmproj
        )
        let specialization = ModelSpecializationRegistry.resolve(for: signature)
        let specFlags = specialization.ggufFlags(for: signature)

        let isHybrid = ggufHeader?.isHybridRecurrent == true || signature.isHybridRecurrentOrSWA
        let effectiveCacheRAM: Int?
        if let autoCacheRAMMiB {
            effectiveCacheRAM = autoCacheRAMMiB
        } else if isHybrid {
            effectiveCacheRAM = 8192
        } else {
            effectiveCacheRAM = nil
        }

        let unset = Set(resolvedSidecar?.unsetKeys ?? [])
        let effectiveDraftModelPath = fitReport.droppedDrafterForMemory ? nil : drafter.draftModelPath
        let effectiveSpecType: String?
        if fitReport.droppedDrafterForMemory {
            effectiveSpecType = (ggufHeader?.hasBuiltInMTP == true && !unset.contains("spec_type")) ? "draft-mtp" : nil
        } else if let spec = drafter.specType {
            effectiveSpecType = spec
        } else if effectiveDraftModelPath == nil,
                  ggufHeader?.hasBuiltInMTP == true,
                  !unset.contains("spec_type") {
            effectiveSpecType = "draft-mtp"
        } else {
            effectiveSpecType = nil
        }

        let effectiveCacheReuse: Int? = (mmproj != nil || specFlags.disableCacheReuse) ? nil : 256
        let effectiveCtxCheckpoints: Int? = specFlags.ctxCheckpoints
        let effectiveCheckpointMinStep: Int? = specFlags.checkpointMinStep
        let reasoningFlag: String? = description.supportsThinking.value ? "deepseek" : nil

        let configuredServerBinary: String? = {
            if case .string(let s)? = resolvedSidecar?.extraConfig["server_binary"] {
                let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.isEmpty ? nil : trimmed
            }
            return nil
        }()

        return GGUFSpawnConfig(
            binaryPath: GGUFSubprocessProvider.resolveBinaryPath(configuredPath: configuredServerBinary),
            modelPath: resolvedGGUFModelPath,
            port: assignedPort,
            ctxSize: resolvedSidecar?.ctxSize ?? description.contextSize.value,
            flashAttention: resolvedSidecar?.flashAttn ?? description.flashAttention.value,
            ubatchSize: autoUbatch,
            mmprojPath: mmproj,
            imageMaxTokens: clampedImageMaxTokens,
            chatTemplateFile: resolvedTemplateFile,
            draftModelPath: effectiveDraftModelPath,
            specType: effectiveSpecType,
            cacheTypeK: { if case .string(let s)? = resolvedSidecar?.extraConfig["cache_type_k"] { return s }; return nil }(),
            cacheTypeV: { if case .string(let s)? = resolvedSidecar?.extraConfig["cache_type_v"] { return s }; return nil }(),
            cacheRAMMiB: effectiveCacheRAM,
            cacheReuse: effectiveCacheReuse,
            ctxCheckpoints: effectiveCtxCheckpoints,
            checkpointMinStep: effectiveCheckpointMinStep,
            reasoningFormatFlag: reasoningFlag,
            reasoningBudget: resolvedSidecar?.sampling.thinkingBudget,
            reasoningPreserve: specFlags.reasoningPreserve
        )
    }
}

/// Tier 3 Provider (macOS): Out-of-process `llama-server` subprocess (`-np 1`) for `.gguf` models.
/// Strictly truthful (never synthesizes fallback outputs), wires `reasoning_control`, `prompt_progress`,
/// continuation echo stripping, `/apply-template`, `/v1/chat/completions/input_tokens`, and `/props` caps.
public actor GGUFSubprocessProvider: InferenceProvider {
    public struct OpenAISSEDelta: Sendable, Equatable {
        public let completionID: String?
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
        public let progressProcessed: Int?
        public let progressTotal: Int?
        public let errorMessage: String?

        public init(
            completionID: String? = nil,
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
            progressProcessed: Int? = nil,
            progressTotal: Int? = nil,
            errorMessage: String? = nil
        ) {
            self.completionID = completionID
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
            self.progressProcessed = progressProcessed
            self.progressTotal = progressTotal
            self.errorMessage = errorMessage
        }
    }

    private final class StderrRingBuffer: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []
        private let maxLines: Int

        init(maxLines: Int = 64) {
            self.maxLines = maxLines
        }

        func append(_ data: Data) {
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            let newLines = text
                .split(whereSeparator: \.isNewline)
                .map(String.init)
                .filter { !$0.isEmpty }
            guard !newLines.isEmpty else { return }
            lock.lock()
            lines.append(contentsOf: newLines)
            if lines.count > maxLines {
                lines.removeFirst(lines.count - maxLines)
            }
            lock.unlock()
        }

        func snapshot(limit: Int = 12) -> String {
            lock.lock()
            defer { lock.unlock() }
            return lines.suffix(limit).joined(separator: " | ")
        }
    }

    private final class CompletionIDBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: String?
        func setIfUnset(_ id: String?) {
            guard let id, !id.isEmpty else { return }
            lock.lock()
            if value == nil { value = id }
            lock.unlock()
        }
        func get() -> String? {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .gguf
    private var templateInfo: ModelTemplateInfo
    private var descriptionState: EngineDescription
    public nonisolated let spawnConfig: GGUFSpawnConfig
    private var activeSpawnConfig: GGUFSpawnConfig
    private var prefixCache = PrefixCacheTracker()
    private var unloadCount: Int = 0
    private var observedProps: GGUFServerProps?
    private var lastContextUsed: Int?
    private let stderrBuffer = StderrRingBuffer()

    #if os(macOS)
    private var process: Process?
    private var stderrPipe: Pipe?
    #endif

    /// Resolves `llama-server` via `server_binary` (`configuredPath`) -> `$HEYLOOK_LLAMA_SERVER` -> `~/.heylook/llama.cpp/build/bin/llama-server` -> `/opt/homebrew/bin/llama-server`.
    /// Never silently masks an explicit missing `server_binary` or `$HEYLOOK_LLAMA_SERVER` override.
    public static func resolveBinaryPath(
        configuredPath: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        if let configured = configuredPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !configured.isEmpty {
            return configured
        }
        if let envPath = environment["HEYLOOK_LLAMA_SERVER"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !envPath.isEmpty {
            return envPath
        }
        let fm = FileManager.default
        #if os(macOS)
        let localBuild = fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".heylook/llama.cpp/build/bin/llama-server")
            .path
        if fm.isExecutableFile(atPath: localBuild) {
            return localBuild
        }
        #endif
        let homebrew = "/opt/homebrew/bin/llama-server"
        if fm.isExecutableFile(atPath: homebrew) {
            return homebrew
        }
        return homebrew
    }

    public init(
        description: EngineDescription,
        spawnConfig: GGUFSpawnConfig? = nil
    ) {
        self.modelID = description.modelID
        self.descriptionState = description
        let resolved = spawnConfig ?? GGUFSpawnConfig.autoSized(for: description)
        self.spawnConfig = resolved
        self.activeSpawnConfig = resolved
        self.templateInfo = ModelTemplateInfo.inspect(
            modelDirectory: HeylookConfig.resolveModelDirectoryURL(from: description.modelPath),
            modelID: description.modelID
        )
    }

    @available(*, deprecated, message: "allowSyntheticFallback has been removed; GGUFSubprocessProvider always requires a real llama-server executable and GGUF checkpoint.")
    public init(
        description: EngineDescription,
        spawnConfig: GGUFSpawnConfig? = nil,
        allowSyntheticFallback: Bool
    ) {
        _ = allowSyntheticFallback
        self.init(description: description, spawnConfig: spawnConfig)
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
        // Disable `--cache-reuse` whenever `--mmproj` is present
        let hasMMProj = config.mmprojPath?.isEmpty == false
        if !hasMMProj, let reuse = config.cacheReuse, reuse > 0 {
            args.append(contentsOf: ["--cache-reuse", String(reuse)])
        }
        if let ckpts = config.ctxCheckpoints, ckpts > 0 {
            args.append(contentsOf: ["--ctx-checkpoints", String(ckpts)])
        }
        if let minStep = config.checkpointMinStep, minStep > 0 {
            args.append(contentsOf: ["--checkpoint-min-step", String(minStep)])
        }
        if let rFmt = config.reasoningFormatFlag, !rFmt.isEmpty {
            args.append(contentsOf: ["--reasoning-format", rFmt])
        }
        if let rBudget = config.reasoningBudget {
            args.append(contentsOf: ["--reasoning-budget", String(rBudget)])
        }
        if config.reasoningPreserve {
            args.append("--reasoning-preserve")
        }
        args.append(contentsOf: ["-lv", "4"])
        args.append(contentsOf: config.extraArgs)
        return args
    }

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

        let completionID = json["id"] as? String
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

        var progressProcessed: Int?
        var progressTotal: Int?
        if let progress = json["prompt_progress"] as? [String: Any] {
            let cacheHit = (progress["cache"] as? Int) ?? 0
            let rawTotal = (progress["total"] as? Int) ?? 0
            let rawProc = (progress["processed"] as? Int) ?? 0
            let workTotal = rawTotal - cacheHit
            if workTotal > 0 {
                progressTotal = workTotal
                progressProcessed = max(0, rawProc - cacheHit)
            }
        }

        let usage = json["usage"] as? [String: Any]
        let usageDetails = usage?["prompt_tokens_details"] as? [String: Any]
        let timings = json["timings"] as? [String: Any]

        let usagePrompt = usage?["prompt_tokens"] as? Int
        let timingsCached = timings?["cache_n"] as? Int
        let usageCached = usageDetails?["cached_tokens"] as? Int
        let cachedTokens = usageCached ?? timingsCached

        let promptTokens: Int?
        if let usagePrompt {
            promptTokens = usagePrompt
        } else if let promptN = timings?["prompt_n"] as? Int {
            promptTokens = promptN + (cachedTokens ?? 0)
        } else {
            promptTokens = nil
        }

        let predictedTokens = (usage?["completion_tokens"] as? Int) ?? (timings?["predicted_n"] as? Int)
        let promptTPS = timings?["prompt_per_second"] as? Double
        let generationTPS = timings?["predicted_per_second"] as? Double
        let draftedTokens = timings?["draft_n"] as? Int
        let acceptedTokens = timings?["draft_n_accepted"] as? Int

        return OpenAISSEDelta(
            completionID: completionID,
            content: content,
            reasoningContent: reasoningContent,
            finishReason: finishReason,
            promptTokens: promptTokens,
            cachedTokens: cachedTokens,
            predictedTokens: predictedTokens,
            promptTPS: promptTPS,
            generationTPS: generationTPS,
            draftedTokens: draftedTokens,
            acceptedTokens: acceptedTokens,
            progressProcessed: progressProcessed,
            progressTotal: progressTotal
        )
    }

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
            let hasMedia = turn.attachments.isEmpty == false || turn.parts.contains(where: {
                if case .media = $0 { return true }
                return false
            })
            if !hasMedia {
                msgDict["content"] = turn.content
                wireMessages.append(msgDict)
            } else {
                var wireParts: [[String: Any]] = []
                for part in turn.parts {
                    switch part {
                    case .thinking:
                        break
                    case .text(let txt):
                        if !txt.isEmpty {
                            wireParts.append(["type": "text", "text": txt])
                        }
                    case .media(let att):
                        if att.kind == "image" {
                            wireParts.append([
                                "type": "image_url",
                                "image_url": [
                                    "url": "data:\(att.mediaType);base64,\(att.dataBase64)"
                                ],
                            ])
                        } else if att.kind == "audio" {
                            let audioFormat = att.mediaType.components(separatedBy: "/").last ?? "wav"
                            wireParts.append([
                                "type": "input_audio",
                                "input_audio": [
                                    "data": att.dataBase64,
                                    "format": audioFormat,
                                ],
                            ])
                        }
                    }
                }
                msgDict["content"] = wireParts
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
        public let supportsTypedContent: Bool?
        public let supportsTools: Bool?
        public let supportsParallelToolCalls: Bool?
        public let supportsSystemRole: Bool?
        public let rawCaps: [String: Bool]

        public init(
            contextSize: Int? = nil,
            chatTemplate: String? = nil,
            totalSlots: Int? = nil,
            modelPath: String? = nil,
            supportsTypedContent: Bool? = nil,
            supportsTools: Bool? = nil,
            supportsParallelToolCalls: Bool? = nil,
            supportsSystemRole: Bool? = nil,
            rawCaps: [String: Bool] = [:]
        ) {
            self.contextSize = contextSize
            self.chatTemplate = chatTemplate
            self.totalSlots = totalSlots
            self.modelPath = modelPath
            self.supportsTypedContent = supportsTypedContent
            self.supportsTools = supportsTools
            self.supportsParallelToolCalls = supportsParallelToolCalls
            self.supportsSystemRole = supportsSystemRole
            self.rawCaps = rawCaps
        }
    }

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
        let capsAny = (json["chat_template_caps"] as? [String: Any])
            ?? (defaultGen?["chat_template_caps"] as? [String: Any])
        var boolCaps: [String: Bool] = [:]
        if let capsAny {
            for (k, v) in capsAny {
                if let b = v as? Bool {
                    boolCaps[k] = b
                }
            }
        }
        return GGUFServerProps(
            contextSize: nCtx,
            chatTemplate: chatTmpl,
            totalSlots: slots,
            modelPath: modelPath,
            supportsTypedContent: boolCaps["supports_typed_content"],
            supportsTools: boolCaps["supports_tools"],
            supportsParallelToolCalls: boolCaps["supports_parallel_tool_calls"],
            supportsSystemRole: boolCaps["supports_system_role"],
            rawCaps: boolCaps
        )
    }

    public func applyObservedProps(_ props: GGUFServerProps) {
        self.observedProps = props
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
            speculativeDrafter: descriptionState.speculativeDrafter,
            estimatedMemoryMB: descriptionState.estimatedMemoryMB,
            tags: descriptionState.tags
        )
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func observedContractState() async -> ObservedContractState {
        ObservedContractState(
            runningContextLength: observedProps?.contextSize ?? (descriptionState.isLoaded ? activeSpawnConfig.ctxSize : nil),
            loadedTemplateText: observedProps?.chatTemplate,
            runtimeCaps: observedProps?.rawCaps.isEmpty == false ? observedProps?.rawCaps : nil,
            speculativeInForce: descriptionState.isLoaded
                ? ((activeSpawnConfig.draftModelPath != nil) || (activeSpawnConfig.specType != nil))
                : nil,
            isPrefixCacheWarm: prefixCache.isWarm,
            contextUsed: lastContextUsed
        )
    }

    public func isPrefixCacheWarm() -> Bool {
        prefixCache.isWarm
    }

    public func totalUnloadCount() -> Int {
        unloadCount
    }

    public func recentStderrOutput() -> String {
        stderrBuffer.snapshot()
    }

    public func preload() async throws {
        #if os(macOS)
        if canLaunchLiveSubprocess {
            _ = try await launchSubprocess(checkHealth: true)
            return
        }
        #endif
        let fm = FileManager.default
        if !fm.isExecutableFile(atPath: activeSpawnConfig.binaryPath) {
            throw ProviderExecutionError.binaryNotFound(
                modelID: modelID,
                binaryPath: activeSpawnConfig.binaryPath
            )
        }
        throw ProviderExecutionError.checkpointNotFound(
            modelID: modelID,
            path: activeSpawnConfig.modelPath
        )
    }

    public func clearCache() async -> Bool {
        prefixCache.reset()
        lastContextUsed = nil
        #if os(macOS)
        if let proc = process, proc.isRunning {
            let slotURL = URL(string: "http://\(activeSpawnConfig.host):\(activeSpawnConfig.port)/slots/0?action=erase")!
            var req = URLRequest(url: slotURL)
            req.httpMethod = "POST"
            _ = try? await URLSession.shared.data(for: req)
        }
        #endif
        return true
    }

    public func unload() async {
        unloadCount += 1
        GGUFPortPool.shared.releasePort(for: modelID)
        #if os(macOS)
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe = nil
        SubprocessLifecycleRegistry.shared.terminate(for: modelID)
        process = nil
        #endif
        prefixCache.reset()
        observedProps = nil
        lastContextUsed = nil
        descriptionState.isLoaded = false
    }

    public nonisolated var canLaunchLiveSubprocess: Bool {
        let fm = FileManager.default
        return fm.isExecutableFile(atPath: spawnConfig.binaryPath)
            && fm.fileExists(atPath: spawnConfig.modelPath)
    }

    public func renderPromptPreview(request: MessagesRequest) async throws -> PromptPreviewResult {
        #if os(macOS)
        if let proc = process, proc.isRunning {
            let baseURL = URL(string: "http://\(activeSpawnConfig.host):\(activeSpawnConfig.port)")!
            let enableThinking = ModelSpecializationRegistry.resolveThinkingEnabled(
                requestThinking: request.thinking,
                sidecarThinking: nil,
                modelCapable: descriptionState.supportsThinking.value
            )
            var templateKwargs: [String: Any] = [
                templateInfo.thinkingControls.switchVariable ?? "enable_thinking": enableThinking
            ]
            if let depthVar = templateInfo.thinkingControls.depthVariable,
               let depthVal = request.thinkingDepth,
               !depthVal.isEmpty {
                templateKwargs[depthVar] = depthVal
            }
            let body: [String: Any] = [
                "messages": Self.buildOpenAIChatMessages(from: request),
                "chat_template_kwargs": templateKwargs,
            ]
            var req = URLRequest(url: baseURL.appendingPathComponent("apply-template"))
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (data, resp) = try await URLSession.shared.data(for: req)
            if let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let prompt = json["prompt"] as? String {
                let tokCount = try? await queryGGUFInputTokenCount(baseURL: baseURL, messages: Self.buildOpenAIChatMessages(from: request))
                return PromptPreviewResult(
                    prompt: prompt,
                    tokenCount: tokCount,
                    tokens: nil,
                    templateSource: "llama-server:/apply-template",
                    chatTemplateCaps: observedProps?.rawCaps
                )
            }
        }
        #endif
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: descriptionState.modelPath)
        let enableThinking = ModelSpecializationRegistry.resolveThinkingEnabled(
            requestThinking: request.thinking,
            sidecarThinking: nil,
            modelCapable: descriptionState.supportsThinking.value
        )
        let rendered = try HeylookConfig.renderPromptDetailed(
            in: dirURL,
            system: request.system,
            turns: request.messages,
            thinkingEnabled: enableThinking,
            thinkingDepth: request.thinkingDepth
        )
        return PromptPreviewResult(
            prompt: rendered.prompt,
            tokenCount: nil,
            tokens: nil,
            templateSource: rendered.source,
            chatTemplateCaps: observedProps?.rawCaps
        )
    }

    public func planImages(sizes: [[Int]]) async throws -> ImagePlanResult {
        try ModelSpecializationRegistry.validateImagePlanSizes(sizes)
        guard descriptionState.supportsVision.value else {
            throw ProviderExecutionError.invalidRequest(
                detail: "\(modelID) is served as text; it takes no images"
            )
        }
        #if os(macOS)
        try await ensureSubprocessRunning()
        let baseURL = URL(string: "http://\(activeSpawnConfig.host):\(activeSpawnConfig.port)")!
        let textOnlyMessages: [[String: Any]] = [["role": "user", "content": "x"]]
        let baseCount = try await queryGGUFInputTokenCount(baseURL: baseURL, messages: textOnlyMessages)
        var rows: [ImagePlanRow] = []
        rows.reserveCapacity(sizes.count)
        for pair in sizes {
            let w = pair[0]
            let h = pair[1]
            let pngBase64 = try Self.encodeSyntheticPNGBase64(width: w, height: h)
            let imgMessages: [[String: Any]] = [[
                "role": "user",
                "content": [
                    ["type": "image_url", "image_url": ["url": "data:image/png;base64,\(pngBase64)"]],
                    ["type": "text", "text": "x"],
                ],
            ]]
            let withImgCount = try await queryGGUFInputTokenCount(baseURL: baseURL, messages: imgMessages)
            rows.append(ImagePlanRow(size: [w, h], tokens: max(0, withImgCount - baseCount), target: nil))
        }
        return ImagePlanResult(
            engine: "gguf",
            images: rows,
            source: "llama-server /v1/chat/completions/input_tokens"
        )
        #else
        throw ProviderExecutionError.runtimeUnavailable(
            modelID: modelID,
            detail: "GGUF llama-server subprocesses run on macOS only."
        )
        #endif
    }

    private func queryGGUFInputTokenCount(baseURL: URL, messages: [[String: Any]]) async throws -> Int {
        var req = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions/input_tokens"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: [
            "messages": messages,
            "add_generation_prompt": true,
        ])
        let (data, resp) = try await URLSession.shared.data(for: req)
        if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            let msg = String(decoding: data, as: UTF8.self)
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "llama-server /input_tokens HTTP \(http.statusCode): \(msg)"
            )
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Malformed /v1/chat/completions/input_tokens payload from llama-server"
            )
        }
        if let n = json["input_tokens"] as? Int { return n }
        if let arr = json["tokens"] as? [Any] { return arr.count }
        throw ProviderExecutionError.runtimeUnavailable(
            modelID: modelID,
            detail: "llama-server /input_tokens omitted count"
        )
    }

    private static func encodeSyntheticPNGBase64(width: Int, height: Int) throws -> String {
        let clampedW = max(1, width)
        let clampedH = max(1, height)
        let bytesPerRow = clampedW * 4
        var pixelBuffer = [UInt8](repeating: 128, count: bytesPerRow * clampedH)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixelBuffer,
            width: clampedW,
            height: clampedH,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ),
              let cgImage = ctx.makeImage(),
              let mutData = CFDataCreateMutable(nil, 0),
              let dest = CGImageDestinationCreateWithData(
                  mutData,
                  UTType.png.identifier as CFString,
                  1,
                  nil
              ) else {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: "image-plan",
                detail: "Failed to synthesize PNG buffer for \(width)x\(height)"
            )
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: "image-plan",
                detail: "Failed to finalize PNG buffer for \(width)x\(height)"
            )
        }
        return (mutData as Data).base64EncodedString()
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let cacheReport = prefixCache.evaluate(system: request.system, messages: request.messages)
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: descriptionState.modelPath)
        let sidecar = HeylookConfig.loadSidecar(in: dirURL)
        let enableThinking = ModelSpecializationRegistry.resolveThinkingEnabled(
            requestThinking: effectiveSampling.thinking ?? request.thinking,
            sidecarThinking: sidecar?.supportsThinking,
            modelCapable: descriptionState.supportsThinking.value
        )
        let resolvedStops = effectiveSampling.stopSequences ?? request.stopSequences ?? []

        #if os(macOS)
        if canLaunchLiveSubprocess {
            return generateWithLiveSubprocess(
                requestID: requestID,
                request: request,
                effectiveSampling: effectiveSampling,
                fallbackCacheReport: cacheReport,
                enableThinking: enableThinking,
                stopSequences: resolvedStops,
                registry: registry
            )
        }
        #endif

        let fm = FileManager.default
        let err: ProviderExecutionError = !fm.isExecutableFile(atPath: spawnConfig.binaryPath)
            ? .binaryNotFound(modelID: modelID, binaryPath: spawnConfig.binaryPath)
            : .checkpointNotFound(modelID: modelID, path: spawnConfig.modelPath)
        return AsyncThrowingStream { continuation in
            continuation.finish(throwing: err)
        }
    }

    private func recordAssistantTurn(_ reply: String, usedContext: Int? = nil) {
        prefixCache.recordAssistantReply(reply)
        if let usedContext {
            lastContextUsed = usedContext
        }
    }

    private func recordContextUsedOnly(_ total: Int) {
        lastContextUsed = total
    }

    #if os(macOS)
    @discardableResult
    public func launchSubprocess(
        customArguments: [String]? = nil,
        checkHealth: Bool = true,
        startupTimeoutSeconds: Double = 120.0
    ) async throws -> Int32 {
        if let proc = process, proc.isRunning {
            return proc.processIdentifier
        }
        return try await launchAttempt(
            config: activeSpawnConfig,
            customArguments: customArguments,
            checkHealth: checkHealth,
            startupTimeoutSeconds: startupTimeoutSeconds,
            allowDrafterRetry: customArguments == nil
        )
    }

    private func launchAttempt(
        config: GGUFSpawnConfig,
        customArguments: [String]?,
        checkHealth: Bool,
        startupTimeoutSeconds: Double,
        allowDrafterRetry: Bool
    ) async throws -> Int32 {
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        let pipe = Pipe()
        let ring = self.stderrBuffer
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty {
                ring.append(chunk)
            }
        }
        self.stderrPipe = pipe

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: config.binaryPath)
        proc.arguments = customArguments ?? Self.buildLlamaServerArgs(config: config)
        proc.environment = Self.childEnvironment()
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = pipe
        try proc.run()
        self.process = proc
        self.activeSpawnConfig = config
        SubprocessLifecycleRegistry.shared.register(proc, for: modelID)

        guard checkHealth else {
            descriptionState.isLoaded = true
            return proc.processIdentifier
        }

        let healthURL = URL(string: "http://\(config.host):\(config.port)/health")!
        let propsURL = URL(string: "http://\(config.host):\(config.port)/props")!
        let maxPolls = max(4, Int((startupTimeoutSeconds / 0.25).rounded(.up)))

        for _ in 0..<maxPolls {
            if !proc.isRunning {
                pipe.fileHandleForReading.readabilityHandler = nil
                let exitCode = proc.terminationStatus
                let hasDrafter = (config.draftModelPath?.isEmpty == false) || (config.specType?.isEmpty == false)
                if allowDrafterRetry && hasDrafter {
                    var fallbackCfg = config
                    fallbackCfg.draftModelPath = nil
                    fallbackCfg.specType = nil
                    HeylookObservabilityLog.shared.record(
                        level: "WARN",
                        category: "gguf",
                        message: "llama-server exited with drafter; retrying without drafter",
                        metadata: [
                            "model_id": modelID,
                            "exit_code": String(exitCode),
                            "stderr": stderrBuffer.snapshot(limit: 4),
                        ]
                    )
                    return try await launchAttempt(
                        config: fallbackCfg,
                        customArguments: nil,
                        checkHealth: checkHealth,
                        startupTimeoutSeconds: startupTimeoutSeconds,
                        allowDrafterRetry: false
                    )
                }
                let detail = stderrBuffer.snapshot()
                throw ProviderExecutionError.runtimeUnavailable(
                    modelID: modelID,
                    detail: "llama-server exited prematurely (code \(exitCode))\(detail.isEmpty ? "" : ": \(detail)")"
                )
            }

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

        pipe.fileHandleForReading.readabilityHandler = nil
        SubprocessLifecycleRegistry.shared.terminate(for: modelID)
        self.process = nil
        descriptionState.isLoaded = false
        let detail = stderrBuffer.snapshot()
        throw ProviderExecutionError.runtimeUnavailable(
            modelID: modelID,
            detail: "llama-server health check timed out after \(Int(startupTimeoutSeconds))s\(detail.isEmpty ? "" : ": \(detail)")"
        )
    }

    private func ensureSubprocessRunning() async throws {
        _ = try await launchSubprocess(checkHealth: true)
    }

    private func describeComputeError(_ msg: String) -> String {
        let lower = msg.lowercased()
        guard lower.contains("compute error") else {
            return "llama-server error for '\(modelID)': \(msg)"
        }
        let fit = RAMFitCalculator.assess(for: descriptionState)
        var text = "llama-server compute error for '\(modelID)' -- Metal working-set headroom is \(String(format: "%.1f", fit.kvHeadroomGB)) GiB at \(String(format: "%.0f", fit.workingSetGB)) GiB limit"
        if let suggest = fit.sysctlSuggestMB {
            text += ". Raise wired limit: sudo sysctl iogpu.wired_limit_mb=\(suggest)"
        }
        return text + ", or reduce ctx_size / n_ubatch."
    }

    private func generateWithLiveSubprocess(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        fallbackCacheReport: CacheReport,
        enableThinking: Bool,
        stopSequences: [String],
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let cfg = self.activeSpawnConfig
        let currentModelID = self.modelID
        let activeTemplate = self.templateInfo
        let descSnapshot = self.descriptionState
        let baseURL = URL(string: "http://\(cfg.host):\(cfg.port)")!
        let cmplBox = CompletionIDBox()

        let lastTurn = request.messages.last
        let isAssistantContinuation = lastTurn?.role.lowercased() == "assistant"
        if let rawCont = request.metadata?["continue_final_message"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           !rawCont.isEmpty {
            let requestedContinue: Bool
            if ["true", "1", "yes"].contains(rawCont) {
                requestedContinue = true
            } else if ["false", "0", "no"].contains(rawCont) {
                requestedContinue = false
            } else {
                let err = ProviderExecutionError.invalidRequest(
                    detail: "Invalid metadata.continue_final_message value '\(rawCont)'."
                )
                return AsyncThrowingStream { $0.finish(throwing: err) }
            }
            if requestedContinue && !isAssistantContinuation {
                let err = ProviderExecutionError.invalidRequest(
                    detail: "continue_final_message=true requires final message role to be 'assistant' for GGUF."
                )
                return AsyncThrowingStream { $0.finish(throwing: err) }
            }
            if !requestedContinue && isAssistantContinuation {
                let err = ProviderExecutionError.invalidRequest(
                    detail: "continue_final_message=false conflicts with trailing 'assistant' message."
                )
                return AsyncThrowingStream { $0.finish(throwing: err) }
            }
        }

        let strippedTrailingThinking = lastTurn?.thinking.map { String($0.drop(while: \.isWhitespace)) }
        let hasTrailingThinking = isAssistantContinuation && !(strippedTrailingThinking?.isEmpty ?? true)
        let hasTrailingContent = isAssistantContinuation && !(lastTurn?.content.isEmpty ?? true)
        let startsInThinking = enableThinking && hasTrailingThinking && !hasTrailingContent

        let initialEchoChars = hasTrailingContent ? (lastTurn?.content.count ?? 0) : 0
        let initialEchoThinkingChars = hasTrailingThinking ? (strippedTrailingThinking?.count ?? 0) : 0
        let initialClosedThought = (initialEchoThinkingChars > 0) && (initialEchoChars > 0)

        let reasoningControlEnabled = enableThinking || (hasTrailingThinking && !hasTrailingContent)

        return StreamingPipeline.transformEventStream(
            requestID: requestID,
            cacheReport: fallbackCacheReport,
            defaultSpeculativeReport: nil,
            startsInThinking: startsInThinking,
            reasoningFormat: activeTemplate.reasoningFormat,
            thinkingEnabled: enableThinking,
            strippableSpecialTokens: Array(activeTemplate.strippableSpecialTokens),
            stopSequences: stopSequences,
            maxTokens: effectiveSampling.maxTokens,
            thinkingBudget: effectiveSampling.thinkingBudget,
            computePlacement: .ggufSubprocess,
            registry: registry,
            onReasoningEndHook: {
                guard reasoningControlEnabled, let cid = cmplBox.get() else { return }
                var ctrlReq = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions/control"))
                ctrlReq.httpMethod = "POST"
                ctrlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
                let bodyObj: [String: String] = ["id": cid, "action": "reasoning_end"]
                ctrlReq.httpBody = try? JSONSerialization.data(withJSONObject: bodyObj)
                _ = try? await URLSession.shared.data(for: ctrlReq)
            },
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        ) { [weak self] eventCont in
            guard let self else { return }
            var echoChars = initialEchoChars
            var echoThinkingChars = initialEchoThinkingChars
            var closedThought = initialClosedThought
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

            // NOTE: Intentionally DO NOT forward `stop_sequences` in `body["stop"]`:
            // `StreamingPipeline`'s `StopSequenceFilter` cuts visible reply only and never aborts `<think>`.
            var body: [String: Any] = [
                "messages": wireMessages,
                "stream": true,
                "stream_options": ["include_usage": true],
                "cache_prompt": true,
                "return_progress": true,
                "temperature": effectiveSampling.temperature ?? SamplerCascade.fallbackTemperature,
                "top_p": effectiveSampling.topP ?? SamplerCascade.fallbackTopP,
                "min_p": effectiveSampling.minP ?? 0.0,
                "max_tokens": effectiveSampling.maxTokens ?? SamplerCascade.defaultMaxTokens,
                "chat_template_kwargs": templateKwargs,
            ]
            if let topK = effectiveSampling.topK {
                body["top_k"] = topK
            }
            if let repCtx = effectiveSampling.repetitionContextSize {
                body["repeat_last_n"] = repCtx
            }
            if reasoningControlEnabled {
                body["reasoning_control"] = true
            }
            if let repPen = effectiveSampling.repetitionPenalty {
                body["repeat_penalty"] = repPen
            }
            if let presPen = effectiveSampling.presencePenalty {
                body["presence_penalty"] = presPen
            }
            if let seed = effectiveSampling.seed {
                body["seed"] = seed
            }
            if let thinkBudget = effectiveSampling.thinkingBudget {
                body["reasoning_budget_tokens"] = enableThinking ? thinkBudget : 0
            } else if !enableThinking {
                body["reasoning_budget_tokens"] = 0
            }
            if let rf = request.responseFormat {
                body["response_format"] = rf.toOpenAIObject()
            }
            urlReq.httpBody = try JSONSerialization.data(withJSONObject: body)

            let (bytes, resp) = try await URLSession.shared.bytes(for: urlReq)
            if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                var errData = Data()
                for try await b in bytes {
                    errData.append(b)
                    if errData.count >= 4096 { break }
                }
                let errText = String(decoding: errData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
                throw ProviderExecutionError.runtimeUnavailable(
                    modelID: currentModelID,
                    detail: "llama-server HTTP \(http.statusCode)\(errText.isEmpty ? "" : ": \(errText)")"
                )
            }

            var sawPreSplitReasoning = false
            var lastPromptTotal = 0
            var lastGenTotal = 0

            for try await line in bytes.lines {
                try Task.checkCancellation()
                guard let delta = Self.parseOpenAISSEDataLine(line) else { continue }
                if let errMsg = delta.errorMessage {
                    let diag = await self.describeComputeError(errMsg)
                    throw ProviderExecutionError.runtimeUnavailable(
                        modelID: descSnapshot.modelID,
                        detail: diag
                    )
                }

                if reasoningControlEnabled {
                    cmplBox.setIfUnset(delta.completionID)
                }

                if let proc = delta.progressProcessed, let tot = delta.progressTotal, tot > 0 {
                    eventCont.yield(.progress(processed: proc, total: tot))
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

                if let pTok = delta.promptTokens { lastPromptTotal = pTok }
                if let gTok = delta.predictedTokens { lastGenTotal = gTok }
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

                // Strip leading continuation prefill echo from both content and reasoning channels
                var contentText = delta.content
                var thinkText = delta.reasoningContent

                if echoChars > 0 && !contentText.isEmpty {
                    let cut = min(echoChars, contentText.count)
                    echoChars -= cut
                    contentText = String(contentText.dropFirst(cut))
                }
                if echoThinkingChars > 0 && !thinkText.isEmpty {
                    let cut = min(echoThinkingChars, thinkText.count)
                    echoThinkingChars -= cut
                    thinkText = String(thinkText.dropFirst(cut))
                }
                if closedThought && echoThinkingChars == 0 && !thinkText.isEmpty {
                    let trimmed = String(thinkText.drop(while: { $0.isWhitespace }))
                    thinkText = trimmed
                    if !thinkText.isEmpty {
                        closedThought = false
                    }
                }

                if !thinkText.isEmpty {
                    sawPreSplitReasoning = true
                    eventCont.yield(.splitReasoning(thinkText))
                    if !contentText.isEmpty {
                        eventCont.yield(.splitContent(contentText))
                    }
                } else if !contentText.isEmpty {
                    if sawPreSplitReasoning {
                        eventCont.yield(.splitContent(contentText))
                    } else {
                        eventCont.yield(.rawToken(contentText))
                    }
                }
            }

            if lastPromptTotal > 0 {
                await self.recordContextUsedOnly(lastPromptTotal + lastGenTotal)
            }
        }
    }
    #endif
}
