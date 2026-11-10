import Foundation
import HeylookCore
import HeylookEmbeddings

#if canImport(CoreImage)
import CoreImage
#endif

#if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM)
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers
#endif

#if canImport(MLXVLM)
import MLXVLM
#endif

/// Tier 1 Provider: In-process MLX Swift (`mlx-swift` + `mlx-swift-lm`) with `PrefixCacheTracker`,
/// exact post-template token-ID KV cache session slicing (guarded against non-trimmable hybrid
/// recurrent SSM caches), VLM Lanczos `imageMaxTokens` resizing, and optional per-tower ANE offload.
public actor MLXSwiftProvider: InferenceProvider {
    public struct KVCacheSessionState: Sendable, Equatable {
        public var activeTurnCount: Int
        public var cachedTokenCount: Int
        public var sliceTruncations: Int
        public var speculativeDrafterPath: String?
        public var speculativeMode: String?
        public var isHybridNonTrimmable: Bool

        public init(
            activeTurnCount: Int = 0,
            cachedTokenCount: Int = 0,
            sliceTruncations: Int = 0,
            speculativeDrafterPath: String? = nil,
            speculativeMode: String? = nil,
            isHybridNonTrimmable: Bool = false
        ) {
            self.activeTurnCount = activeTurnCount
            self.cachedTokenCount = cachedTokenCount
            self.sliceTruncations = sliceTruncations
            self.speculativeDrafterPath = speculativeDrafterPath
            self.speculativeMode = speculativeMode
            self.isHybridNonTrimmable = isHybridNonTrimmable
        }
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .mlx
    public nonisolated let allowSyntheticFallback: Bool
    private let tokenCounter: (@Sendable (String) -> Int)?
    private var templateInfo: ModelTemplateInfo
    private var descriptionState: EngineDescription
    private var prefixCache = PrefixCacheTracker()
    private var kvSession = KVCacheSessionState()
    private var unloadCount: Int = 0

    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
    private final class MLXKVCacheBox: @unchecked Sendable {
        var caches: [KVCache]? = nil
        var tokenIDs: [Int] = []
        var mtpDrafter: (any MTPDrafterModel)? = nil
        var draftLanguageModel: (any LanguageModel)? = nil
        var didAttemptDrafterLoad: Bool = false
    }
    private var loadedContainer: ModelContainer?
    private var loadedTokenizer: (any MLXLMCommon.Tokenizer)?
    private let kvCacheBox = MLXKVCacheBox()
    #endif

    private var effectiveTokenCounter: (@Sendable (String) -> Int)? {
        if let tokenCounter {
            return tokenCounter
        }
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
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

    public func preload() async throws {
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
        if Self.hasSafetensorsWeights(at: descriptionState.modelPath) {
            _ = try await ensureContainerLoaded()
            return
        }
        #endif
        if allowSyntheticFallback {
            descriptionState.isLoaded = true
            return
        }
        if Self.hasSafetensorsWeights(at: descriptionState.modelPath) {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "MLX Swift runtime is not linked in this build; build with ./scripts/build_native.sh"
            )
        }
        throw ProviderExecutionError.checkpointNotFound(
            modelID: modelID,
            path: descriptionState.modelPath
        )
    }

    public func clearCache() async -> Bool {
        prefixCache.reset()
        kvSession.activeTurnCount = 0
        kvSession.cachedTokenCount = 0
        kvSession.sliceTruncations = 0
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
        let hadMLXState = loadedContainer != nil
            || kvCacheBox.caches != nil
            || kvCacheBox.mtpDrafter != nil
            || kvCacheBox.draftLanguageModel != nil
        kvCacheBox.caches = nil
        kvCacheBox.tokenIDs.removeAll()
        if hadMLXState {
            MLX.Memory.clearCache()
        }
        #endif
        return true
    }

    public func unload() async {
        unloadCount += 1
        _ = await clearCache()
        descriptionState.isLoaded = false
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
        loadedContainer = nil
        loadedTokenizer = nil
        kvCacheBox.mtpDrafter = nil
        kvCacheBox.draftLanguageModel = nil
        kvCacheBox.didAttemptDrafterLoad = false
        #endif
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        // Reject unsupported audio/vision modalities truthfully rather than silently dropping attachments.
        let hasAudioAttachment = request.messages.contains {
            $0.attachments.contains { $0.kind == "audio" }
        }
        if hasAudioAttachment && !descriptionState.supportsAudio.value {
            let err = ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Model '\(modelID)' does not support audio attachments"
            )
            return AsyncThrowingStream { $0.finish(throwing: err) }
        }
        let hasImageAttachment = request.messages.contains {
            $0.attachments.contains { $0.kind == "image" }
        }
        if hasImageAttachment && !descriptionState.supportsVision.value {
            let err = ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Model '\(modelID)' does not support image attachments"
            )
            return AsyncThrowingStream { $0.finish(throwing: err) }
        }

        // Refresh templateInfo from disk so live operator edits to chat_template.heylook.jinja take effect.
        let dirURL = URL(fileURLWithPath: descriptionState.modelPath)
        self.templateInfo = ModelTemplateInfo.inspect(modelDirectory: dirURL, modelID: modelID)

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
        let resolvedStops = effectiveSampling.stopSequences ?? request.stopSequences ?? []

        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
        if Self.hasSafetensorsWeights(at: descriptionState.modelPath) {
            if cacheReport.outcome == .miss {
                kvCacheBox.caches = nil
                kvCacheBox.tokenIDs.removeAll()
            }
            return generateWithMLXContainer(
                requestID: requestID,
                request: request,
                effectiveSampling: effectiveSampling,
                cacheReport: cacheReport,
                enableThinking: enableThinking,
                stopSequences: resolvedStops,
                placement: placement,
                registry: registry
            )
        }
        #endif

        guard allowSyntheticFallback else {
            let missingError: ProviderExecutionError = Self.hasSafetensorsWeights(at: descriptionState.modelPath)
                ? .runtimeUnavailable(
                    modelID: modelID,
                    detail: "MLX Swift runtime is not linked in this build; build with ./scripts/build_native.sh"
                )
                : .checkpointNotFound(
                    modelID: modelID,
                    path: descriptionState.modelPath
                )
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: missingError)
            }
        }

        // Truthful Runtime: never fabricate SpeculativeReport token counts when no real drafter executed.
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
            speculativeReport: nil,
            startsInThinking: false,
            reasoningFormat: templateInfo.reasoningFormat,
            thinkingEnabled: enableThinking,
            strippableSpecialTokens: Array(templateInfo.strippableSpecialTokens),
            stopSequences: resolvedStops,
            maxTokens: effectiveSampling.maxTokens,
            thinkingBudget: effectiveSampling.thinkingBudget,
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

    #if canImport(CoreImage)
    /// Resizes `image` with Lanczos scaling when its pixel count exceeds `maxTokens * (patchSize * spatialMerge)^2`.
    public static func resizeCIImageIfNeeded(
        _ image: CIImage,
        maxTokens: Int?,
        patchSize: Int = 14,
        spatialMerge: Int = 2
    ) -> CIImage {
        guard let maxTokens, maxTokens > 0 else { return image }
        let width = image.extent.width
        let height = image.extent.height
        guard width > 0 && height > 0 else { return image }
        let cellSide = Double(max(1, patchSize * spatialMerge))
        let maxPixels = Double(maxTokens) * cellSide * cellSide
        let currentPixels = Double(width * height)
        guard currentPixels > maxPixels else { return image }
        let scale = (maxPixels / currentPixels).squareRoot()
        guard scale > 0.01 && scale < 0.999 else { return image }
        if let filter = CIFilter(name: "CILanczosScaleTransform") {
            filter.setValue(image, forKey: kCIInputImageKey)
            filter.setValue(scale, forKey: kCIInputScaleKey)
            filter.setValue(1.0, forKey: kCIInputAspectRatioKey)
            if let out = filter.outputImage {
                return out
            }
        }
        return image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }
    #endif

    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
    private func updateObservedKVState(cachedTokens: Int, sliceTruncated: Bool, isHybrid: Bool) {
        kvSession.cachedTokenCount = cachedTokens
        if sliceTruncated {
            kvSession.sliceTruncations += 1
        }
        kvSession.isHybridNonTrimmable = isHybrid
    }

    private func ensureContainerLoaded() async throws -> ModelContainer {
        if let loadedContainer {
            return loadedContainer
        }
        let dirURL = URL(fileURLWithPath: descriptionState.modelPath)
        let loader: any TokenizerLoader = HeylookTransformersTokenizerLoader()
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
        await ensureSpeculativeDrafterLoadedIfAvailable(loader: loader)
        return container
    }

    private func ensureSpeculativeDrafterLoadedIfAvailable(loader: any TokenizerLoader) async {
        guard !kvCacheBox.didAttemptDrafterLoad else { return }
        kvCacheBox.didAttemptDrafterLoad = true
        guard let drafterPath = kvSession.speculativeDrafterPath,
              !drafterPath.isEmpty,
              Self.hasSafetensorsWeights(at: drafterPath)
        else { return }

        let drafterDir = URL(fileURLWithPath: drafterPath)
        let drafterCfg = ModelConfiguration(directory: drafterDir)
            .resolved(modelDirectory: drafterDir, tokenizerDirectory: drafterDir)
        let mode = kvSession.speculativeMode ?? "draft-mtp"

        if mode == "draft-mtp" {
            if descriptionState.supportsVision.value,
               let vlmMTP = try? await MTPDrafterModelFactory.visionLanguage._load(
                   configuration: drafterCfg,
                   tokenizerLoader: loader
               ) {
                kvCacheBox.mtpDrafter = vlmMTP.model
                return
            }
            if let llmMTP = try? await MTPDrafterModelFactory.shared._load(
                configuration: drafterCfg,
                tokenizerLoader: loader
            ) {
                kvCacheBox.mtpDrafter = llmMTP.model
                return
            }
        }
        if drafterDir.standardizedFileURL.path != URL(fileURLWithPath: descriptionState.modelPath).standardizedFileURL.path,
           let draftCtx = try? await LLMModelFactory.shared._load(
               configuration: drafterCfg,
               tokenizerLoader: loader
           ) {
            kvCacheBox.draftLanguageModel = draftCtx.model
            if kvSession.speculativeMode == nil {
                kvSession.speculativeMode = "draft-model"
            }
        }
    }

    private func generateWithMLXContainer(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        cacheReport: CacheReport,
        enableThinking: Bool,
        stopSequences: [String],
        placement: ComputePlacement,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let activeTemplate = self.templateInfo
        let cacheBox = self.kvCacheBox
        let modelDirURL = URL(fileURLWithPath: self.descriptionState.modelPath)

        return StreamingPipeline.transformEventStream(
            requestID: requestID,
            cacheReport: cacheReport,
            defaultSpeculativeReport: nil,
            startsInThinking: false,
            reasoningFormat: activeTemplate.reasoningFormat,
            thinkingEnabled: enableThinking,
            strippableSpecialTokens: Array(activeTemplate.strippableSpecialTokens),
            stopSequences: stopSequences,
            maxTokens: effectiveSampling.maxTokens,
            thinkingBudget: effectiveSampling.thinkingBudget,
            computePlacement: placement,
            registry: registry,
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        ) { [weak self] eventCont in
            guard let self else { return }
            let container = try await self.ensureContainerLoaded()
            let sidecar = HeylookConfig.loadSidecar(in: modelDirURL)
            let maxImgTokens = sidecar?.imageMaxTokens
            let templateState = HeylookConfig.inspectChatTemplate(in: modelDirURL)

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

            let (stream, totalPromptTokens, exactCachedTokens, didTrim, isHybrid): (AsyncStream<Generation>, Int, Int, Bool, Bool) = try await container.perform { context in
                var allImages: [UserInput.Image] = []
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
                        var orderedSegments: [String] = []
                        for part in m.parts {
                            switch part {
                            case .text(let txt):
                                if !txt.isEmpty { orderedSegments.append(txt) }
                            case .thinking:
                                break
                            case .media(let att) where att.kind == "image":
                                #if canImport(CoreImage)
                                if let data = Data(base64Encoded: att.dataBase64),
                                   let rawCI = CIImage(data: data) {
                                    let resized = Self.resizeCIImageIfNeeded(rawCI, maxTokens: maxImgTokens)
                                    let wrapped = UserInput.Image.ciImage(resized)
                                    turnImages.append(wrapped)
                                    allImages.append(wrapped)
                                }
                                #endif
                            case .media:
                                break
                            }
                        }
                        let mergedText = orderedSegments.isEmpty ? m.content : orderedSegments.joined(separator: "\n")
                        chatMessages.append(.user(mergedText, images: turnImages))
                    }
                }

                // Honor operator chat_template.heylook.jinja or model.heylook.toml sidecar template override when text-only
                let userInput: UserInput
                if (templateState.hasOverride || templateState.origin == "sidecar" || templateState.origin == "explicit_path") && allImages.isEmpty {
                    let rendered = try HeylookConfig.renderPromptDetailed(
                        in: modelDirURL,
                        system: request.system,
                        turns: request.messages,
                        thinkingEnabled: enableThinking,
                        thinkingDepth: effectiveSampling.thinkingDepth ?? request.thinkingDepth
                    )
                    userInput = UserInput(prompt: .text(rendered.prompt), additionalContext: additionalContext)
                } else {
                    userInput = UserInput(chat: chatMessages, additionalContext: additionalContext)
                }

                let lmInput = try await context.processor.prepare(input: userInput)
                let promptTokenIDs = lmInput.text.tokens.asArray(Int.self)
                let promptTokenCount = promptTokenIDs.count

                let activeCaches: [KVCache]
                let effectiveInput: LMInput
                var reusedTokens = 0
                var performedTrim = false
                var hybridNonTrimmable = false

                if let existingCaches = cacheBox.caches,
                   !existingCaches.isEmpty,
                   !cacheBox.tokenIDs.isEmpty,
                   lmInput.image == nil,
                   lmInput.video == nil,
                   lmInput.audio == nil,
                   promptTokenCount > 1
                {
                    let allTrimmable = existingCaches.allSatisfy(\.isTrimmable)
                    hybridNonTrimmable = !allTrimmable
                    let existingOffset = existingCaches.first?.offset ?? 0
                    let limit = min(min(cacheBox.tokenIDs.count, existingOffset), promptTokenCount - 1)
                    var common = 0
                    while common < limit && cacheBox.tokenIDs[common] == promptTokenIDs[common] {
                        common += 1
                    }

                    if common > 0 && (allTrimmable || common == existingOffset) {
                        if allTrimmable && existingOffset > common {
                            for cache in existingCaches where cache.offset > common {
                                _ = cache.trim(cache.offset - common)
                            }
                            performedTrim = true
                        }
                        activeCaches = existingCaches
                        effectiveInput = LMInput(text: lmInput.text[common...])
                        reusedTokens = common
                        cacheBox.tokenIDs = promptTokenIDs
                    } else {
                        activeCaches = try context.model.newCache(parameters: genParams)
                        effectiveInput = lmInput
                        cacheBox.tokenIDs = promptTokenIDs
                    }
                } else {
                    let fresh = try context.model.newCache(parameters: genParams)
                    hybridNonTrimmable = !fresh.allSatisfy(\.isTrimmable)
                    activeCaches = fresh
                    effectiveInput = lmInput
                    cacheBox.tokenIDs = promptTokenIDs
                }

                cacheBox.caches = activeCaches
                let canSpeculate = lmInput.image == nil && lmInput.video == nil && lmInput.audio == nil
                let genStream: AsyncStream<Generation>
                if canSpeculate,
                   let mtp = cacheBox.mtpDrafter,
                   mtp.isCompatible(with: context.model),
                   let specStream = try? MLXLMCommon.generate(
                       input: effectiveInput,
                       cache: activeCaches,
                       parameters: genParams,
                       context: context,
                       mtpDrafter: mtp
                   ) {
                    genStream = specStream
                } else if canSpeculate,
                          let draftLM = cacheBox.draftLanguageModel,
                          let specStream = try? MLXLMCommon.generate(
                              input: effectiveInput,
                              cache: activeCaches,
                              parameters: genParams,
                              context: context,
                              draftModel: draftLM
                          ) {
                    genStream = specStream
                } else {
                    genStream = try MLXLMCommon.generate(
                        input: effectiveInput,
                        cache: activeCaches,
                        parameters: genParams,
                        context: context
                    )
                }
                return (genStream, promptTokenCount, reusedTokens, performedTrim, hybridNonTrimmable)
            }

            let kvSnap = await self.kvSessionState()
            let specModeLabel = kvSnap.speculativeMode ?? "draft-mtp"
            let specDrafterID = kvSnap.speculativeDrafterPath
                .map { URL(fileURLWithPath: $0).lastPathComponent }
                ?? self.modelID
            await self.updateObservedKVState(
                cachedTokens: exactCachedTokens,
                sliceTruncated: didTrim,
                isHybrid: isHybrid
            )
            eventCont.yield(
                .telemetry(
                    promptTokens: totalPromptTokens,
                    cachedTokens: exactCachedTokens
                )
            )

            var generatedText = ""
            var sawReasoning = false
            for await event in stream {
                try Task.checkCancellation()
                switch event {
                case .chunk(let piece):
                    generatedText.append(piece)
                    eventCont.yield(.rawToken(piece))
                case .reasoning(let thinkPiece):
                    sawReasoning = true
                    generatedText.append(thinkPiece)
                    eventCont.yield(.splitReasoning(thinkPiece))
                case .info(let info):
                    let specReport: SpeculativeReport?
                    if let specTel = info.speculativeDecodingTelemetry, specTel.draftTokenCount > 0 {
                        specReport = SpeculativeReport(
                            drafterID: specDrafterID,
                            mode: specModeLabel,
                            draftedTokens: specTel.draftTokenCount,
                            acceptedTokens: specTel.acceptedDraftTokenCount
                        )
                    } else if let proposed = info.proposedDraftTokens,
                              let accepted = info.acceptedDraftTokens,
                              proposed > 0 {
                        specReport = SpeculativeReport(
                            drafterID: specDrafterID,
                            mode: specModeLabel,
                            draftedTokens: proposed,
                            acceptedTokens: accepted
                        )
                    } else {
                        specReport = nil
                    }
                    eventCont.yield(
                        .telemetry(
                            promptTokens: totalPromptTokens,
                            cachedTokens: exactCachedTokens,
                            promptTPS: info.promptTokensPerSecond,
                            generationTPS: info.tokensPerSecond,
                            speculativeReport: specReport
                        )
                    )
                default:
                    break
                }
            }
            // When `.reasoning` fired, `mlx-swift-lm` stripped `<think>`/`</think>` delimiters that ARE inside
            // `activeCaches`, so re-encoding `generatedText` would misalign `cacheBox.tokenIDs[promptTokenCount...]`
            // against KV slot offsets. Keep `cacheBox.tokenIDs = promptTokenIDs` when `sawReasoning == true`:
            // Jinja strips prior-turn thinking anyway, so Turn 2 reuses `0..<promptTokenCount` and trims cleanly.
            if !sawReasoning, let tok = await self.loadedTokenizer, !generatedText.isEmpty {
                cacheBox.tokenIDs.append(contentsOf: tok.encode(text: generatedText))
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
