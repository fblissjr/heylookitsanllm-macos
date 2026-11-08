import Foundation
import HeylookCore
import HeylookEmbeddings

#if canImport(CoreImage)
import CoreImage
#endif

#if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM)
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers
#endif

#if canImport(MLXVLM)
import MLXVLM
#endif

/// Tier 1 Provider: In-process MLX Swift (`mlx-swift` + `mlx-swift-lm`) with `PrefixCacheTracker`,
/// stateful multi-turn KV cache session slicing, speculative drafter discovery (`draft-mtp` /
/// `draft-model`), and optional per-tower ANE offload (`ANECoreMLRuntime`).
public actor MLXSwiftProvider: InferenceProvider {
    public struct KVCacheSessionState: Sendable, Equatable {
        public var activeTurnCount: Int
        public var cachedTokenCount: Int
        public var sliceTruncations: Int
        public var speculativeDrafterPath: String?
        public var speculativeMode: String?

        public init(
            activeTurnCount: Int = 0,
            cachedTokenCount: Int = 0,
            sliceTruncations: Int = 0,
            speculativeDrafterPath: String? = nil,
            speculativeMode: String? = nil
        ) {
            self.activeTurnCount = activeTurnCount
            self.cachedTokenCount = cachedTokenCount
            self.sliceTruncations = sliceTruncations
            self.speculativeDrafterPath = speculativeDrafterPath
            self.speculativeMode = speculativeMode
        }
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .mlx
    public nonisolated let allowSyntheticFallback: Bool
    private let tokenCounter: (@Sendable (String) -> Int)?
    private let templateInfo: ModelTemplateInfo
    private var descriptionState: EngineDescription
    private var prefixCache = PrefixCacheTracker()
    private var kvSession = KVCacheSessionState()
    private var unloadCount: Int = 0

    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXHuggingFace)
    private final class MLXKVCacheBox: @unchecked Sendable {
        var caches: [KVCache]? = nil
    }
    private var loadedContainer: ModelContainer?
    private var loadedTokenizer: (any MLXLMCommon.Tokenizer)?
    private let kvCacheBox = MLXKVCacheBox()
    #endif

    private var effectiveTokenCounter: (@Sendable (String) -> Int)? {
        if let tokenCounter {
            return tokenCounter
        }
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXHuggingFace)
        if let tok = loadedTokenizer {
            return { text in
                max(1, tok.encode(text: text).count)
            }
        }
        #endif
        return nil
    }

    public init(
        description: EngineDescription,
        allowSyntheticFallback: Bool = false,
        tokenCounter: (@Sendable (String) -> Int)? = nil
    ) {
        self.modelID = description.modelID
        self.descriptionState = description
        self.allowSyntheticFallback = allowSyntheticFallback
        self.tokenCounter = tokenCounter

        let dirURL = URL(fileURLWithPath: description.modelPath)
        self.templateInfo = ModelTemplateInfo.inspect(
            modelDirectory: dirURL,
            modelID: description.modelID
        )
        let sidecar = (try? String(
            contentsOf: dirURL.appendingPathComponent("model.heylook.toml"),
            encoding: .utf8
        )).map(ModelSidecarConfig.parseTOML)
        let discoveredDrafter = HeylookConfig.discoverDrafter(
            in: dirURL,
            provider: .mlx,
            sidecar: sidecar
        )
        if discoveredDrafter.draftModelPath != nil || discoveredDrafter.specType != nil {
            self.kvSession = KVCacheSessionState(
                speculativeDrafterPath: discoveredDrafter.draftModelPath,
                speculativeMode: discoveredDrafter.specType ?? "draft-mtp"
            )
        } else if description.tags.contains(where: { $0.localizedCaseInsensitiveContains("mtp") || $0.localizedCaseInsensitiveContains("drafter") }) {
            self.kvSession = KVCacheSessionState(
                speculativeDrafterPath: description.modelPath,
                speculativeMode: "draft-mtp"
            )
        }
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func isPrefixCacheWarm() -> Bool {
        prefixCache.isWarm
    }

    public func kvSessionState() -> KVCacheSessionState {
        kvSession
    }

    public func totalUnloadCount() -> Int {
        unloadCount
    }

    public func unload() async {
        unloadCount += 1
        prefixCache.reset()
        kvSession.activeTurnCount = 0
        kvSession.cachedTokenCount = 0
        kvSession.sliceTruncations = 0
        descriptionState.isLoaded = false
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXHuggingFace)
        loadedContainer = nil
        loadedTokenizer = nil
        kvCacheBox.caches = nil
        MLX.Memory.clearCache()
        #endif
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let placement = descriptionState.computePlacement.value
        let cacheReport = prefixCache.evaluate(
            system: request.system,
            messages: request.messages,
            tokenCounter: effectiveTokenCounter
        )
        kvSession.activeTurnCount = prefixCache.cachedTurnCount
        kvSession.cachedTokenCount = cacheReport.cachedTokens
        kvSession.sliceTruncations = prefixCache.branchTruncationsCount

        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value

        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXHuggingFace)
        if Self.hasSafetensorsWeights(at: descriptionState.modelPath) {
            if cacheReport.outcome == .miss {
                kvCacheBox.caches = nil
            }
            return generateWithMLXContainer(
                requestID: requestID,
                request: request,
                effectiveSampling: effectiveSampling,
                cacheReport: cacheReport,
                enableThinking: enableThinking,
                placement: placement,
                registry: registry
            )
        }
        #endif

        guard allowSyntheticFallback else {
            let missingError = ProviderExecutionError.checkpointNotFound(
                modelID: modelID,
                path: descriptionState.modelPath
            )
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: missingError)
            }
        }

        let syntheticSpecReport: SpeculativeReport? = if let mode = kvSession.speculativeMode {
            SpeculativeReport(
                drafterID: kvSession.speculativeDrafterPath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "\(modelID)-mtp",
                mode: mode,
                draftedTokens: 12,
                acceptedTokens: 10
            )
        } else {
            nil
        }

        let userText = request.messages.last?.content ?? ""
        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Analyzing ", "prompt ", "on ", placement.label, "...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[\(modelID)] ", "Processed: ", userText, "<|im_end|>"])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            cacheReport: cacheReport,
            speculativeReport: syntheticSpecReport,
            startsInThinking: false,
            reasoningFormat: templateInfo.reasoningFormat,
            thinkingEnabled: enableThinking,
            strippableSpecialTokens: Array(templateInfo.strippableSpecialTokens),
            computePlacement: placement,
            registry: registry,
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        )
    }

    private func recordAssistantTurn(_ reply: String) {
        prefixCache.recordAssistantReply(reply, tokenCounter: effectiveTokenCounter)
        kvSession.activeTurnCount = prefixCache.cachedTurnCount
    }

    private static func hasSafetensorsWeights(at path: String) -> Bool {
        let url = URL(fileURLWithPath: path)
        guard let items = try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) else {
            return false
        }
        return items.contains { $0.pathExtension.lowercased() == "safetensors" }
    }

    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXHuggingFace)
    private func updateCachedTokenCount(_ count: Int) {
        kvSession.cachedTokenCount = max(kvSession.cachedTokenCount, count)
    }

    private func ensureContainerLoaded() async throws -> ModelContainer {
        if let loadedContainer {
            return loadedContainer
        }
        let dirURL = URL(fileURLWithPath: descriptionState.modelPath)
        let loader: any TokenizerLoader = #huggingFaceTokenizerLoader()
        // Pass `reasoningConfig: nil` so `mlx-swift-lm` yields raw token chunks without running a
        // competing internal `StreamingReasoningParser`; `HeylookCore.ReasoningParser` handles all
        // 4 reasoning protocols (`.tagPair`, `.gemmaChannel`, `.harmonyChannel`, `.recipientChannel`)
        // and mid-stream `forceEndReasoning()` in a single canonical pipeline.
        let modelConfig = ModelConfiguration(
            directory: dirURL,
            extraEOSTokens: self.templateInfo.strippableSpecialTokens,
            eosTokenIds: Set(self.templateInfo.eosTokenIDs),
            reasoningConfig: nil
        )
        let resolvedConfig = modelConfig.resolved(
            modelDirectory: dirURL,
            tokenizerDirectory: dirURL
        )
        let container: ModelContainer
        #if canImport(MLXVLM)
        if descriptionState.supportsVision.value {
            do {
                let ctx = try await VLMModelFactory.shared._load(
                    configuration: resolvedConfig,
                    tokenizerLoader: loader
                )
                container = VLMModelFactory.shared._wrap(ctx)
            } catch {
                let ctx = try await LLMModelFactory.shared._load(
                    configuration: resolvedConfig,
                    tokenizerLoader: loader
                )
                container = LLMModelFactory.shared._wrap(ctx)
                descriptionState.supportsVision = ProvenanceField(
                    value: false,
                    provenance: .observed,
                    effect: .reload
                )
            }
        } else {
            let ctx = try await LLMModelFactory.shared._load(
                configuration: resolvedConfig,
                tokenizerLoader: loader
            )
            container = LLMModelFactory.shared._wrap(ctx)
        }
        #else
        let ctx = try await LLMModelFactory.shared._load(
            configuration: resolvedConfig,
            tokenizerLoader: loader
        )
        container = LLMModelFactory.shared._wrap(ctx)
        #endif
        loadedContainer = container
        loadedTokenizer = await container.tokenizer
        descriptionState.isLoaded = true
        return container
    }

    private func generateWithMLXContainer(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        cacheReport: CacheReport,
        enableThinking: Bool,
        placement: ComputePlacement,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let activeTemplate = self.templateInfo
        let cacheBox = self.kvCacheBox

        return StreamingPipeline.transformEventStream(
            requestID: requestID,
            cacheReport: cacheReport,
            defaultSpeculativeReport: nil,
            startsInThinking: false,
            reasoningFormat: activeTemplate.reasoningFormat,
            thinkingEnabled: enableThinking,
            strippableSpecialTokens: Array(activeTemplate.strippableSpecialTokens),
            computePlacement: placement,
            registry: registry,
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        ) { [weak self] eventCont in
            guard let self else { return }
            let container = try await self.ensureContainerLoaded()
            var additionalContextDraft: [String: any Sendable] = [:]
            let switchKey = activeTemplate.thinkingControls.switchVariable ?? "enable_thinking"
            additionalContextDraft[switchKey] = enableThinking
            if let depthVar = activeTemplate.thinkingControls.depthVariable,
               let depthVal = effectiveSampling.thinkingDepth ?? request.thinkingDepth,
               !depthVal.isEmpty {
                additionalContextDraft[depthVar] = depthVal
            }
            let additionalContext = additionalContextDraft
            var genParamsDraft = GenerateParameters(
                maxTokens: effectiveSampling.maxTokens ?? 2048,
                temperature: Float(effectiveSampling.temperature ?? 0.7),
                topP: Float(effectiveSampling.topP ?? 0.95)
            )
            genParamsDraft.topK = effectiveSampling.topK ?? 0
            genParamsDraft.minP = Float(effectiveSampling.minP ?? 0)
            if let seed = effectiveSampling.seed {
                genParamsDraft.seed = UInt64(bitPattern: Int64(seed))
            }
            if let repPen = effectiveSampling.repetitionPenalty {
                genParamsDraft.repetitionPenalty = Float(repPen)
            }
            genParamsDraft.repetitionContextSize = effectiveSampling.repetitionContextSize ?? 64
            if let presPen = effectiveSampling.presencePenalty {
                genParamsDraft.presencePenalty = Float(presPen)
            }
            let genParams = genParamsDraft

            let (stream, totalPromptTokens): (AsyncStream<Generation>, Int) = try await container.perform { context in
                var chatMessages: [Chat.Message] = []
                if let sys = request.system, !sys.isEmpty {
                    chatMessages.append(.system(sys))
                }
                for m in request.messages {
                    switch m.role.lowercased() {
                    case "assistant":
                        chatMessages.append(.assistant(m.content))
                    case "system":
                        chatMessages.append(.system(m.content))
                    default:
                        var turnImages: [UserInput.Image] = []
                        #if canImport(CoreImage)
                        for att in m.attachments where att.kind == "image" {
                            if let data = Data(base64Encoded: att.dataBase64),
                               let ciImg = CIImage(data: data) {
                                turnImages.append(.ciImage(ciImg))
                            }
                        }
                        #endif
                        chatMessages.append(.user(m.content, images: turnImages))
                    }
                }
                let userInput = UserInput(
                    chat: chatMessages,
                    additionalContext: additionalContext
                )
                let lmInput = try await context.processor.prepare(input: userInput)
                let promptTokenCount = lmInput.text.tokens.size
                let activeCaches: [KVCache]
                let effectiveInput: LMInput
                if cacheReport.outcome == .reused,
                   let existingCaches = cacheBox.caches,
                   lmInput.image == nil,
                   lmInput.video == nil,
                   lmInput.audio == nil,
                   cacheReport.cachedTokens > 0,
                   cacheReport.cachedTokens < promptTokenCount
                {
                    for cache in existingCaches where cache.isTrimmable && cache.offset > cacheReport.cachedTokens {
                        _ = cache.trim(cache.offset - cacheReport.cachedTokens)
                    }
                    activeCaches = existingCaches
                    effectiveInput = LMInput(text: lmInput.text[cacheReport.cachedTokens...])
                } else {
                    activeCaches = try context.model.newCache(parameters: genParams)
                    effectiveInput = lmInput
                }
                cacheBox.caches = activeCaches
                let genStream = try MLXLMCommon.generate(
                    input: effectiveInput,
                    cache: activeCaches,
                    parameters: genParams,
                    context: context
                )
                return (genStream, promptTokenCount)
            }
            await self.updateCachedTokenCount(totalPromptTokens)

            for await event in stream {
                switch event {
                case .chunk(let piece):
                    eventCont.yield(.rawToken(piece))
                case .reasoning(let thinkPiece):
                    eventCont.yield(.splitReasoning(thinkPiece))
                case .info(let info):
                    eventCont.yield(
                        .telemetry(
                            promptTokens: info.promptTokenCount,
                            promptTPS: info.promptTokensPerSecond,
                            generationTPS: info.tokensPerSecond
                        )
                    )
                default:
                    break
                }
            }
        }
    }
    #endif
}

// MARK: - Pure-Swift MLX Model Conversion (`MLXModelConverter` -> `MLXLMCommon.convert`)

public enum MLXQuantizationMode: String, Sendable, Codable, CaseIterable {
    case affine = "affine"
    case mxfp4 = "mxfp4"
    case mxfp8 = "mxfp8"
    case nvfp4 = "nvfp4"

    public var defaultBits: Int {
        switch self {
        case .mxfp8:
            return 8
        case .affine, .mxfp4, .nvfp4:
            return 4
        }
    }

    public var defaultGroupSize: Int {
        switch self {
        case .mxfp4, .mxfp8, .nvfp4:
            return 32
        case .affine:
            return 64
        }
    }
}

public enum MLXQuantizationCalibration: String, Sendable, Codable, CaseIterable {
    case standard = "standard"
    case q4Zero = "q4_zero"
}

public struct MLXConversionSpec: Sendable, Equatable {
    public var bits: Int?
    public var groupSize: Int?
    public var mode: MLXQuantizationMode
    public var calibration: MLXQuantizationCalibration
    public var maxShardSizeBytes: Int64
    public var overwriteExistingOutput: Bool

    public init(
        bits: Int? = nil,
        groupSize: Int? = nil,
        mode: MLXQuantizationMode = .affine,
        calibration: MLXQuantizationCalibration = .standard,
        maxShardSizeBytes: Int64 = 5 * 1024 * 1024 * 1024,
        overwriteExistingOutput: Bool = false
    ) {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
        self.calibration = calibration
        self.maxShardSizeBytes = maxShardSizeBytes
        self.overwriteExistingOutput = overwriteExistingOutput
    }

    public var resolvedBits: Int {
        if calibration == .q4Zero { return 4 }
        return bits ?? mode.defaultBits
    }

    public var resolvedGroupSize: Int {
        if calibration == .q4Zero { return 32 }
        return groupSize ?? mode.defaultGroupSize
    }
}

public enum MLXModelConversionError: LocalizedError, Sendable, Equatable {
    case missingConfigJSON(URL)
    case noSafetensorsFiles(URL)
    case unsupportedPyTorchWeights(URL)
    case sourceAlreadyQuantized(URL, String)
    case outputDirectoryExists(URL)
    case outputDirectoryOverlapsSource(URL)
    case incompatibleCalibration(MLXQuantizationCalibration, Int, Int, MLXQuantizationMode)
    case runtimeUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfigJSON(let dir):
            return "Missing config.json in source directory \(dir.path)"
        case .noSafetensorsFiles(let dir):
            return "No .safetensors weight files found in \(dir.path)"
        case .unsupportedPyTorchWeights(let dir):
            return "PyTorch .bin weights in \(dir.path) are not supported; convert from .safetensors"
        case .sourceAlreadyQuantized(let dir, let detail):
            return "Source model in \(dir.path) is already quantized (\(detail))"
        case .outputDirectoryExists(let dir):
            return "Output directory \(dir.path) already exists (pass --overwrite to replace)"
        case .outputDirectoryOverlapsSource(let dir):
            return "Output directory \(dir.path) cannot overlap the source checkpoint directory"
        case .incompatibleCalibration(let cal, let bits, let gs, let mode):
            return "Calibration '\(cal.rawValue)' requires bits=4, groupSize=32, mode=affine (got bits=\(bits), groupSize=\(gs), mode=\(mode.rawValue))"
        case .runtimeUnavailable(let detail):
            return detail
        }
    }
}

public struct MLXConversionPlan: Sendable, Equatable {
    public let sourceDirectory: URL
    public let outputDirectory: URL
    public let modelType: String
    public let isVisionModel: Bool
    public let resolvedBits: Int
    public let resolvedGroupSize: Int
    public let mode: MLXQuantizationMode
    public let calibration: MLXQuantizationCalibration
    public let safetensorsFiles: [String]
    public let sidecarFiles: [String]
    public let reasoningFormat: String
}

public enum MLXModelConverter {
    private static let sidecarExtensions: Set<String> = [
        "json", "jsonl", "jinja", "txt", "model", "tiktoken", "py",
    ]

    /// Validates a source `.safetensors` directory and returns the resolved `MLXConversionPlan`
    /// matching `mlx-swift-lm`'s `ModelConversion.swift` preflight rules.
    public static func planConversion(
        from sourceDirectory: URL,
        to outputDirectory: URL,
        spec: MLXConversionSpec = .init()
    ) throws -> MLXConversionPlan {
        let fm = FileManager.default
        let srcCanonical = sourceDirectory.resolvingSymlinksInPath().standardizedFileURL
        let dstCanonical = outputDirectory.resolvingSymlinksInPath().standardizedFileURL

        if srcCanonical.pathComponents.starts(with: dstCanonical.pathComponents)
            || dstCanonical.pathComponents.starts(with: srcCanonical.pathComponents) {
            throw MLXModelConversionError.outputDirectoryOverlapsSource(outputDirectory)
        }

        if fm.fileExists(atPath: dstCanonical.path) && !spec.overwriteExistingOutput {
            throw MLXModelConversionError.outputDirectoryExists(outputDirectory)
        }

        if spec.calibration == .q4Zero {
            let b = spec.bits ?? 4
            let g = spec.groupSize ?? 32
            if b != 4 || g != 32 || spec.mode != .affine {
                throw MLXModelConversionError.incompatibleCalibration(.q4Zero, b, g, spec.mode)
            }
        }

        let configURL = srcCanonical.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL),
              let configJSON = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw MLXModelConversionError.missingConfigJSON(sourceDirectory)
        }

        if let qDict = configJSON["quantization"] as? [String: Any] {
            let bits = qDict["bits"] as? Int ?? 4
            let mode = qDict["mode"] as? String ?? "affine"
            throw MLXModelConversionError.sourceAlreadyQuantized(sourceDirectory, "\(bits)-bit \(mode)")
        }
        if let qCfg = configJSON["quantization_config"] as? [String: Any] {
            let method = qCfg["quant_method"] as? String ?? "quantized"
            throw MLXModelConversionError.sourceAlreadyQuantized(sourceDirectory, method)
        }

        let entries = (try? fm.contentsOfDirectory(
            at: srcCanonical,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []

        let safetensors = entries
            .filter { $0.pathExtension.lowercased() == "safetensors" }
            .map(\.lastPathComponent)
            .sorted()

        if safetensors.isEmpty {
            if entries.contains(where: { $0.pathExtension.lowercased() == "bin" }) {
                throw MLXModelConversionError.unsupportedPyTorchWeights(sourceDirectory)
            }
            throw MLXModelConversionError.noSafetensorsFiles(sourceDirectory)
        }

        let sidecars = entries
            .filter { sidecarExtensions.contains($0.pathExtension.lowercased()) }
            .map(\.lastPathComponent)
            .sorted()

        let modelType = (configJSON["model_type"] as? String) ?? "unknown"
        let isVision = ModelTemplateInfo.hasVisionConfiguration(in: srcCanonical, configJSON: configJSON)
        let templateInfo = ModelTemplateInfo.inspect(modelDirectory: srcCanonical)

        return MLXConversionPlan(
            sourceDirectory: srcCanonical,
            outputDirectory: dstCanonical,
            modelType: modelType,
            isVisionModel: isVision,
            resolvedBits: spec.resolvedBits,
            resolvedGroupSize: spec.resolvedGroupSize,
            mode: spec.mode,
            calibration: spec.calibration,
            safetensorsFiles: safetensors,
            sidecarFiles: sidecars,
            reasoningFormat: templateInfo.reasoningFormat.wireIdentifier
        )
    }

    /// Executes 100% pure-Swift `.safetensors` quantization and sharded export using `mlx-swift-lm`
    /// (`LLMModelFactory.shared.convert` / `MLXLMCommon.convert`).
    public static func convert(
        from sourceDirectory: URL,
        to outputDirectory: URL,
        spec: MLXConversionSpec = .init(),
        progressHandler: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> (plan: MLXConversionPlan, weightsURLs: [URL]) {
        let plan = try planConversion(from: sourceDirectory, to: outputDirectory, spec: spec)

        #if canImport(MLX) && canImport(MLXNN) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXVLM)
        let mlxMode: QuantizationMode
        switch spec.mode {
        case .affine:
            mlxMode = .affine
        case .mxfp4:
            mlxMode = .mxfp4
        case .mxfp8:
            mlxMode = .mxfp8
        case .nvfp4:
            mlxMode = .nvfp4
        }

        let mlxCalibration: ModelConversionQuantizationCalibration =
            (spec.calibration == .q4Zero) ? .q4Zero : .standard

        let options = ModelConversionOptions(
            bits: plan.resolvedBits,
            groupSize: plan.resolvedGroupSize,
            mode: mlxMode,
            calibration: mlxCalibration,
            maxShardSize: spec.maxShardSizeBytes,
            overwriteExistingOutput: spec.overwriteExistingOutput
        )

        let result: ModelConversionResult
        if plan.isVisionModel {
            let configURL = plan.sourceDirectory.appendingPathComponent("config.json")
            let configData = try Data(contentsOf: configURL)
            if let vlmModel = try? await VLMModelFactory.shared.typeRegistry.createModel(
                configuration: configData,
                modelType: plan.modelType
            ) {
                result = try MLXLMCommon.convert(
                    modelDirectory: plan.sourceDirectory,
                    tokenizerDirectory: plan.sourceDirectory,
                    model: vlmModel,
                    to: plan.outputDirectory,
                    options: options,
                    progressHandler: { prog in
                        progressHandler("[\(prog.stage.rawValue)] \(prog.message ?? "")")
                    }
                )
            } else {
                result = try await LLMModelFactory.shared.convert(
                    from: plan.sourceDirectory,
                    to: plan.outputDirectory,
                    options: options,
                    progressHandler: { prog in
                        progressHandler("[\(prog.stage.rawValue)] \(prog.message ?? "")")
                    }
                )
            }
        } else {
            result = try await LLMModelFactory.shared.convert(
                from: plan.sourceDirectory,
                to: plan.outputDirectory,
                options: options,
                progressHandler: { prog in
                    progressHandler("[\(prog.stage.rawValue)] \(prog.message ?? "")")
                }
            )
        }
        return (plan, result.weightsURLs)
        #else
        throw MLXModelConversionError.runtimeUnavailable(
            "Pure-Swift MLX weight quantization requires MLXLMCommon & Metal shaders. Run via `./scripts/build_native.sh cli convert-mlx ...`."
        )
        #endif
    }
}
