import CryptoKit
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

/// Tier 1 Provider: In-process MLX Swift (`mlx-swift` + `mlx-swift-lm`) with multi-branch `APCSnapshotCache`,
/// bounded LRU `VisionFeatureCache`, lossless Leviathan MTP rejection sampling (`Gemma4Specialization`),
/// GPU closing-thinking token injection, `chat_template.heylook.jinja` VLM/LLM parity, `response_format`
/// validation, and live processor `planImages` / `renderPromptPreview`.
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
    private let tokenCounter: (@Sendable (String) -> Int)?
    private var templateInfo: ModelTemplateInfo
    private var descriptionState: EngineDescription
    private var prefixCache = PrefixCacheTracker()
    private var kvSession = KVCacheSessionState()
    private var unloadCount: Int = 0
    private var lastContextUsed: Int?
    private var checkpointSignature: CheckpointSignature
    private var specialization: any ModelSpecialization
    private var isDiffusionCheckpoint: Bool = false

    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
    private final class MLXRuntimeBox: @unchecked Sendable {
        let apcCache = APCSnapshotCache(maxEntries: 16)
        let visionCache = VisionFeatureCache(maxEntries: 16)
        var mtpDrafter: (any MTPDrafterModel)?
        var draftLanguageModel: (any LanguageModel)?
        var didAttemptDrafterLoad: Bool = false
    }
    private var loadedContainer: ModelContainer?
    private var loadedTokenizer: (any MLXLMCommon.Tokenizer)?
    private let runtimeBox = MLXRuntimeBox()
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
        tokenCounter: (@Sendable (String) -> Int)? = nil
    ) {
        self.modelID = description.modelID
        self.descriptionState = description
        self.tokenCounter = tokenCounter

        let dirURL = URL(fileURLWithPath: description.modelPath)
        self.templateInfo = ModelTemplateInfo.inspect(
            modelDirectory: dirURL,
            modelID: description.modelID
        )
        let sidecar = HeylookConfig.loadSidecar(in: dirURL)
        let discoveredDrafter = HeylookConfig.discoverDrafter(
            in: dirURL,
            provider: .mlx,
            sidecar: sidecar
        )
        let drafterURL = discoveredDrafter.draftModelPath.map { URL(fileURLWithPath: $0) }
        let sig = CheckpointSignature.inspect(modelDirectoryOrFile: dirURL, drafterDirectory: drafterURL)
        self.checkpointSignature = sig
        self.specialization = ModelSpecializationRegistry.resolve(for: sig)
        self.isDiffusionCheckpoint = Self.detectDiffusionCheckpoint(in: dirURL)

        if discoveredDrafter.draftModelPath != nil || discoveredDrafter.specType != nil {
            self.kvSession = KVCacheSessionState(
                speculativeDrafterPath: discoveredDrafter.draftModelPath,
                speculativeMode: discoveredDrafter.specType ?? "draft-mtp",
                isHybridNonTrimmable: sig.isHybridRecurrentOrSWA
            )
        } else {
            self.kvSession = KVCacheSessionState(
                isHybridNonTrimmable: sig.isHybridRecurrentOrSWA
            )
        }
    }

    @available(*, deprecated, message: "allowSyntheticFallback has been removed; MLXSwiftProvider enforces real checkpoints and runtime.")
    public init(
        description: EngineDescription,
        allowSyntheticFallback: Bool,
        tokenCounter: (@Sendable (String) -> Int)? = nil
    ) {
        _ = allowSyntheticFallback
        self.init(description: description, tokenCounter: tokenCounter)
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

    public func observedContractState() async -> ObservedContractState {
        let hasLiveSpec: Bool
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
        hasLiveSpec = (runtimeBox.mtpDrafter != nil || runtimeBox.draftLanguageModel != nil)
        #else
        hasLiveSpec = false
        #endif
        let runningCtx = descriptionState.isLoaded ? descriptionState.contextSize.value : nil
        return ObservedContractState(
            runningContextLength: runningCtx,
            loadedTemplateText: nil,
            runtimeCaps: nil,
            speculativeInForce: descriptionState.isLoaded ? hasLiveSpec : nil,
            isPrefixCacheWarm: prefixCache.isWarm,
            contextUsed: lastContextUsed
        )
    }

    public func preload() async throws {
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
        if Self.hasSafetensorsWeights(at: descriptionState.modelPath) {
            _ = try await ensureContainerLoaded()
            return
        }
        #endif
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
        lastContextUsed = nil
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
        let hadMLXState = loadedContainer != nil
            || !runtimeBox.apcCache.isEmpty
            || runtimeBox.mtpDrafter != nil
            || runtimeBox.draftLanguageModel != nil
        runtimeBox.apcCache.reset()
        runtimeBox.visionCache.clear()
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
        runtimeBox.mtpDrafter = nil
        runtimeBox.draftLanguageModel = nil
        runtimeBox.didAttemptDrafterLoad = false
        #endif
    }

    public func renderPromptPreview(request: MessagesRequest) async throws -> PromptPreviewResult {
        let dirURL = URL(fileURLWithPath: descriptionState.modelPath)
        let sidecar = HeylookConfig.loadSidecar(in: dirURL)
        let enableThinking = ModelSpecializationRegistry.resolveThinkingEnabled(
            requestThinking: request.thinking,
            sidecarThinking: sidecar?.supportsThinking,
            modelCapable: descriptionState.supportsThinking.value
        )
        let isContinuation = Self.resolveContinuationFlag(for: request)
        let rendered = try HeylookConfig.renderPromptDetailed(
            in: dirURL,
            system: request.system,
            turns: request.messages,
            thinkingEnabled: enableThinking,
            thinkingDepth: request.thinkingDepth,
            continuation: isContinuation,
            modelSupportsVision: descriptionState.supportsVision.value,
            modelSupportsAudio: descriptionState.supportsAudio.value
        )
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
        if let tok = loadedTokenizer {
            let ids = tok.encode(text: rendered.prompt, addSpecialTokens: false)
            return PromptPreviewResult(
                prompt: rendered.prompt,
                tokenCount: ids.count,
                tokens: ids,
                templateSource: rendered.source,
                chatTemplateCaps: nil
            )
        }
        #endif
        return PromptPreviewResult(
            prompt: rendered.prompt,
            tokenCount: nil,
            tokens: nil,
            templateSource: rendered.source,
            chatTemplateCaps: nil
        )
    }

    public func planImages(sizes: [[Int]]) async throws -> ImagePlanResult {
        try ModelSpecializationRegistry.validateImagePlanSizes(sizes)
        guard descriptionState.supportsVision.value else {
            throw ProviderExecutionError.invalidRequest(
                detail: "\(modelID) is served as text; it takes no images"
            )
        }
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers) && canImport(CoreImage)
        let container = try await ensureContainerLoaded()
        let dirURL = URL(fileURLWithPath: descriptionState.modelPath)
        let sidecar = HeylookConfig.loadSidecar(in: dirURL)
        let maxImgTokens = sidecar?.imageMaxTokens
        let visionGeom = specialization.visionGeometry(for: checkpointSignature)

        let rows: [ImagePlanRow] = try await container.perform { context in
            let textInput = UserInput(chat: [.user("x")])
            let baseTokens = try await context.processor.prepare(input: textInput).text.tokens.size
            var out: [ImagePlanRow] = []
            out.reserveCapacity(sizes.count)
            for pair in sizes {
                let w = pair[0]
                let h = pair[1]
                let synthetic = Self.makeSyntheticCIImage(width: w, height: h)
                let resized = Self.resizeCIImageIfNeeded(
                    synthetic,
                    maxTokens: maxImgTokens,
                    alignmentMultiplePx: visionGeom?.alignmentMultiplePx ?? 28
                )
                let imgInput = UserInput(chat: [.user("x", images: [.ciImage(resized)])])
                let prepared = try await context.processor.prepare(input: imgInput)
                let totalTokens = prepared.text.tokens.size
                let targetDims: [Int]?
                if let frame = prepared.image?.frames?.first {
                    targetDims = [frame.w, frame.h]
                } else if let pixels = prepared.image?.pixels, pixels.ndim >= 4 {
                    targetDims = [pixels.dim(-1), pixels.dim(-2)]
                } else {
                    targetDims = [Int(resized.extent.width.rounded()), Int(resized.extent.height.rounded())]
                }
                out.append(ImagePlanRow(size: [w, h], tokens: max(0, totalTokens - baseTokens), target: targetDims))
            }
            return out
        }
        return ImagePlanResult(
            engine: "mlx",
            images: rows,
            source: "live processor"
        )
        #else
        throw ProviderExecutionError.runtimeUnavailable(
            modelID: modelID,
            detail: "MLX Vision runtime is unavailable in this build."
        )
        #endif
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let dirURL = URL(fileURLWithPath: descriptionState.modelPath)
        self.templateInfo = ModelTemplateInfo.inspect(modelDirectory: dirURL, modelID: modelID)
        let sidecar = HeylookConfig.loadSidecar(in: dirURL)

        let hasAudioAttachment = request.messages.contains {
            $0.attachments.contains { $0.kind == "audio" }
        }
        if hasAudioAttachment && !descriptionState.supportsAudio.value {
            let err = ProviderExecutionError.invalidRequest(
                detail: "Model '\(modelID)' is served by the MLX provider and does not support audio input."
            )
            return AsyncThrowingStream { $0.finish(throwing: err) }
        }
        let hasImageAttachment = request.messages.contains {
            $0.attachments.contains { $0.kind == "image" }
        }
        if hasImageAttachment && !descriptionState.supportsVision.value {
            let err = ProviderExecutionError.invalidRequest(
                detail: "Model '\(modelID)' is text-only and cannot process image attachments."
            )
            return AsyncThrowingStream { $0.finish(throwing: err) }
        }

        // Non-user image roles guard (matching mlx_provider.py)
        if hasImageAttachment {
            let badRoles = request.messages.filter { turn in
                turn.role.lowercased() != "user" && turn.attachments.contains { $0.kind == "image" }
            }.map(\.role)
            if !badRoles.isEmpty {
                let err = ProviderExecutionError.invalidRequest(
                    detail: "MLX provider places images only on user turns (unsupported on role(s): \(Set(badRoles).sorted().joined(separator: ", ")))."
                )
                return AsyncThrowingStream { $0.finish(throwing: err) }
            }
        }

        let isContinuation = Self.resolveContinuationFlag(for: request)
        if isContinuation && isDiffusionCheckpoint {
            let err = ProviderExecutionError.invalidRequest(
                detail: "continue_final_message is not supported on diffusion models."
            )
            return AsyncThrowingStream { $0.finish(throwing: err) }
        }

        // Honor response_format rejections (400): Harmony / channel-hop / diffusion / continuation
        if let rf = request.responseFormat, rf.type != "text" {
            if isDiffusionCheckpoint {
                let err = ProviderExecutionError.invalidRequest(
                    detail: "response_format is not supported for diffusion models."
                )
                return AsyncThrowingStream { $0.finish(throwing: err) }
            }
            if templateInfo.reasoningFormat == .harmonyChannel || templateInfo.reasoningFormat == .recipientChannel {
                let err = ProviderExecutionError.invalidRequest(
                    detail: "response_format is not supported for channel-hop models ('\(templateInfo.reasoningFormat.wireIdentifier)')."
                )
                return AsyncThrowingStream { $0.finish(throwing: err) }
            }
            if isContinuation {
                let err = ProviderExecutionError.invalidRequest(
                    detail: "response_format cannot be combined with a continuation request."
                )
                return AsyncThrowingStream { $0.finish(throwing: err) }
            }
        }

        let placement = descriptionState.computePlacement.value
        let cacheReport = prefixCache.evaluate(
            system: request.system,
            messages: request.messages,
            tokenCounter: effectiveTokenCounter
        )
        kvSession.activeTurnCount = prefixCache.cachedTurnCount
        kvSession.cachedTokenCount = cacheReport.cachedTokens
        kvSession.sliceTruncations = prefixCache.branchTruncationsCount

        let enableThinking = ModelSpecializationRegistry.resolveThinkingEnabled(
            requestThinking: effectiveSampling.thinking ?? request.thinking,
            sidecarThinking: sidecar?.supportsThinking,
            modelCapable: descriptionState.supportsThinking.value
        )
        let resolvedStops = effectiveSampling.stopSequences ?? request.stopSequences ?? []

        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
        if Self.hasSafetensorsWeights(at: descriptionState.modelPath) {
            return generateWithMLXContainer(
                requestID: requestID,
                request: request,
                effectiveSampling: effectiveSampling,
                cacheReport: cacheReport,
                enableThinking: enableThinking,
                isContinuation: isContinuation,
                stopSequences: resolvedStops,
                placement: placement,
                registry: registry
            )
        }
        #endif

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

    private static func resolveContinuationFlag(for request: MessagesRequest) -> Bool {
        if let meta = request.metadata?["continue_final_message"]?.lowercased() {
            return meta == "true" || meta == "1" || meta == "yes"
        }
        return request.messages.last?.role.lowercased() == "assistant"
    }

    private static func resolveStartsInThinking(
        request: MessagesRequest,
        enableThinking: Bool,
        isContinuation: Bool,
        templateInfo: ModelTemplateInfo
    ) -> Bool {
        guard enableThinking else { return false }
        if isContinuation, let last = request.messages.last {
            let hasThinking = !(last.thinking?.isEmpty ?? true)
            let emptyContent = last.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            if hasThinking && emptyContent {
                return true
            }
            if last.role.lowercased() == "user" {
                return false
            }
        }
        return templateInfo.reasoningFormat.prefillsThinking
    }

    private static func detectDiffusionCheckpoint(in dirURL: URL) -> Bool {
        guard let data = try? Data(contentsOf: dirURL.appendingPathComponent("config.json")),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let mt = (raw["model_type"] as? String)?.lowercased() else {
            return false
        }
        return mt.contains("diffusion") || mt.contains("llada") || mt.contains("dream")
    }

    private func recordAssistantTurn(_ reply: String, usedTokens: Int? = nil) {
        prefixCache.recordAssistantReply(reply, tokenCounter: effectiveTokenCounter)
        kvSession.activeTurnCount = prefixCache.cachedTurnCount
        if let usedTokens {
            lastContextUsed = usedTokens
        }
    }

    private static func hasSafetensorsWeights(at path: String) -> Bool {
        let url = URL(fileURLWithPath: path)
        guard let items = try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) else {
            return false
        }
        return items.contains { $0.pathExtension.lowercased() == "safetensors" }
    }

    #if canImport(CoreImage)
    private static func makeSyntheticCIImage(width: Int, height: Int) -> CIImage {
        CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0))
            .cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    /// Resizes `image` with Lanczos scaling when pixel count exceeds `maxTokens * (alignmentMultiplePx)^2`,
    /// defaulting to `28px` (14 * 2) for Qwen or `48px` (16 * 3) from `Gemma4Specialization`.
    public static func resizeCIImageIfNeeded(
        _ image: CIImage,
        maxTokens: Int?,
        patchSize: Int = 14,
        spatialMerge: Int = 2
    ) -> CIImage {
        resizeCIImageIfNeeded(image, maxTokens: maxTokens, alignmentMultiplePx: max(1, patchSize * spatialMerge))
    }

    public static func resizeCIImageIfNeeded(
        _ image: CIImage,
        maxTokens: Int?,
        alignmentMultiplePx: Int
    ) -> CIImage {
        guard let maxTokens, maxTokens > 0 else { return image }
        let width = image.extent.width
        let height = image.extent.height
        guard width > 0 && height > 0 else { return image }
        let cellSide = Double(max(1, alignmentMultiplePx))
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
        guard !runtimeBox.didAttemptDrafterLoad else { return }
        runtimeBox.didAttemptDrafterLoad = true
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
                runtimeBox.mtpDrafter = vlmMTP.model
                return
            }
            if let llmMTP = try? await MTPDrafterModelFactory.shared._load(
                configuration: drafterCfg,
                tokenizerLoader: loader
            ) {
                runtimeBox.mtpDrafter = llmMTP.model
                return
            }
        }
        if drafterDir.standardizedFileURL.path != URL(fileURLWithPath: descriptionState.modelPath).standardizedFileURL.path,
           let draftCtx = try? await LLMModelFactory.shared._load(
               configuration: drafterCfg,
               tokenizerLoader: loader
           ) {
            runtimeBox.draftLanguageModel = draftCtx.model
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
        isContinuation: Bool,
        stopSequences: [String],
        placement: ComputePlacement,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let activeTemplate = self.templateInfo
        let box = self.runtimeBox
        let sig = self.checkpointSignature
        let spec = self.specialization
        let modelDirURL = URL(fileURLWithPath: self.descriptionState.modelPath)
        let startsInThinking = Self.resolveStartsInThinking(
            request: request,
            enableThinking: enableThinking,
            isContinuation: isContinuation,
            templateInfo: activeTemplate
        )

        return StreamingPipeline.transformEventStream(
            requestID: requestID,
            cacheReport: cacheReport,
            defaultSpeculativeReport: nil,
            startsInThinking: startsInThinking,
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
            let visionGeom = spec.visionGeometry(for: sig)
            let alignmentPx = visionGeom?.alignmentMultiplePx ?? 28

            var genParamsDraft = GenerateParameters(
                maxTokens: effectiveSampling.maxTokens ?? SamplerCascade.defaultMaxTokens,
                temperature: Float(effectiveSampling.temperature ?? SamplerCascade.fallbackTemperature),
                topP: Float(effectiveSampling.topP ?? SamplerCascade.fallbackTopP)
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
            genParamsDraft.prefill.progress = { processed, total in
                eventCont.yield(.progress(processed: processed, total: total))
            }
            let genParams = genParamsDraft

            let rendered = try HeylookConfig.renderPromptDetailed(
                in: modelDirURL,
                system: request.system,
                turns: request.messages,
                thinkingEnabled: enableThinking,
                thinkingDepth: effectiveSampling.thinkingDepth ?? request.thinkingDepth,
                continuation: isContinuation,
                modelSupportsVision: await self.descriptionState.supportsVision.value,
                modelSupportsAudio: await self.descriptionState.supportsAudio.value
            )

            struct PreparedMLXImagePayload: Sendable {
                let turnIndex: Int
                let data: Data
                let cacheKey: String
            }

            var imagePayloads: [PreparedMLXImagePayload] = []
            var imageSaltKeys: [String] = []
            #if canImport(CoreImage)
            let geomKey = "max\(maxImgTokens ?? 0)_align\(alignmentPx)"
            for (turnIdx, m) in request.messages.enumerated() {
                for part in m.parts {
                    if case .media(let att) = part, att.kind == "image",
                       let imgData = Data(base64Encoded: att.dataBase64) {
                        let key = VisionFeatureCache.cacheKey(imageData: imgData, geometryKey: geomKey)
                        imageSaltKeys.append(key)
                        imagePayloads.append(PreparedMLXImagePayload(turnIndex: turnIdx, data: imgData, cacheKey: key))
                    }
                }
            }
            #endif
            let mediaSalt = VisionFeatureCache.mediaSalt(forKeys: imageSaltKeys)
            let immutableImagePayloads = imagePayloads

            let specModeLabel = await self.kvSessionState().speculativeMode ?? "draft-mtp"
            let specDrafterID = await self.kvSessionState().speculativeDrafterPath
                .map { URL(fileURLWithPath: $0).lastPathComponent }
                ?? self.modelID
            let stopIDs = Set(activeTemplate.eosTokenIDs)
            let policy = spec.apcCheckpointPolicy(for: sig)

            struct ContainerRunSummary: Sendable {
                let reusedTokens: Int
                let didTrim: Bool
                let isHybrid: Bool
                let fullSequenceCount: Int
            }

            let summary: ContainerRunSummary = try await container.perform { context in
                enum PreparedIterator: TokenIteratorProtocol {
                    case standard(TokenIterator)
                    case speculative(SpeculativeTokenIterator)
                    case leviathanMTP(LeviathanMTPTokenIterator)
                    case standardMTP(MTPSpeculativeTokenIterator)

                    mutating func next() -> Int? {
                        switch self {
                        case .standard(var it):
                            defer { self = .standard(it) }
                            return it.next()
                        case .speculative(var it):
                            defer { self = .speculative(it) }
                            return it.next()
                        case .leviathanMTP(var it):
                            defer { self = .leviathanMTP(it) }
                            return it.next()
                        case .standardMTP(var it):
                            defer { self = .standardMTP(it) }
                            return it.next()
                        }
                    }

                    var maxTokens: Int? {
                        switch self {
                        case .standard(let it): it.maxTokens
                        case .speculative(let it): it.maxTokens
                        case .leviathanMTP(let it): it.maxTokens
                        case .standardMTP(let it): it.maxTokens
                        }
                    }

                    var tokenCount: Int {
                        switch self {
                        case .standard(let it): it.tokenCount
                        case .speculative(let it): it.tokenCount
                        case .leviathanMTP(let it): it.tokenCount
                        case .standardMTP(let it): it.tokenCount
                        }
                    }

                    var promptPrefillTime: TimeInterval {
                        switch self {
                        case .standard(let it): it.promptPrefillTime
                        case .speculative(let it): it.promptPrefillTime
                        case .leviathanMTP(let it): it.promptPrefillTime
                        case .standardMTP(let it): it.promptPrefillTime
                        }
                    }

                    var state: LMOutput.State? {
                        switch self {
                        case .standard(let it): it.state
                        case .speculative(let it): it.state
                        case .leviathanMTP(let it): it.state
                        case .standardMTP(let it): it.state
                        }
                    }

                    var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
                        switch self {
                        case .standard(let it): it.speculativeDecodingTelemetry
                        case .speculative(let it): it.speculativeDecodingTelemetry
                        case .leviathanMTP(let it): it.speculativeDecodingTelemetry
                        case .standardMTP(let it): it.speculativeDecodingTelemetry
                        }
                    }

                    var lastLogProbabilities: DeferredTokenLogProbabilities? {
                        switch self {
                        case .standard(let it): it.lastLogProbabilities
                        case .speculative(let it): it.lastLogProbabilities
                        case .leviathanMTP(let it): it.lastLogProbabilities
                        case .standardMTP(let it): it.lastLogProbabilities
                        }
                    }

                    var evictedTokenCount: Int {
                        switch self {
                        case .standard(let it): it.evictedTokenCount
                        case .speculative(let it): it.evictedTokenCount
                        case .leviathanMTP(let it): it.evictedTokenCount
                        case .standardMTP(let it): it.evictedTokenCount
                        }
                    }

                    var reasoningTokenCount: Int? {
                        switch self {
                        case .standard(let it): it.reasoningTokenCount
                        case .speculative(let it): it.reasoningTokenCount
                        case .leviathanMTP(let it): it.reasoningTokenCount
                        case .standardMTP(let it): it.reasoningTokenCount
                        }
                    }

                    mutating func recordEmittedToken(_ token: Int) {
                        switch self {
                        case .standard(var it):
                            it.recordEmittedToken(token)
                            self = .standard(it)
                        case .speculative(var it):
                            it.recordEmittedToken(token)
                            self = .speculative(it)
                        case .leviathanMTP(var it):
                            it.recordEmittedToken(token)
                            self = .leviathanMTP(it)
                        case .standardMTP(var it):
                            it.recordEmittedToken(token)
                            self = .standardMTP(it)
                        }
                    }

                    mutating func discardGeneratedToken() {
                        switch self {
                        case .standard(var it):
                            it.discardGeneratedToken()
                            self = .standard(it)
                        case .speculative(var it):
                            it.discardGeneratedToken()
                            self = .speculative(it)
                        case .leviathanMTP(var it):
                            it.discardGeneratedToken()
                            self = .leviathanMTP(it)
                        case .standardMTP(var it):
                            it.discardGeneratedToken()
                            self = .standardMTP(it)
                        }
                    }

                    mutating func finalizeIfSpeculative() {
                        if case .leviathanMTP(var it) = self {
                            it.finalizeUnconsumedLookahead()
                            self = .leviathanMTP(it)
                        }
                    }
                }

                var resolvedCIImages: [UserInput.Image] = []
                var resolvedImagesByTurn: [Int: [UserInput.Image]] = [:]
                #if canImport(CoreImage)
                for payload in immutableImagePayloads {
                    let img: UserInput.Image?
                    if let cached = box.visionCache.get(key: payload.cacheKey) {
                        img = .ciImage(cached.resizedImage)
                    } else if let rawCI = CIImage(data: payload.data) {
                        let resized = Self.resizeCIImageIfNeeded(
                            rawCI,
                            maxTokens: maxImgTokens,
                            alignmentMultiplePx: alignmentPx
                        )
                        box.visionCache.insert(
                            key: payload.cacheKey,
                            resizedImage: resized,
                            width: Int(resized.extent.width.rounded()),
                            height: Int(resized.extent.height.rounded())
                        )
                        img = .ciImage(resized)
                    } else {
                        img = nil
                    }
                    if let img {
                        resolvedCIImages.append(img)
                        resolvedImagesByTurn[payload.turnIndex, default: []].append(img)
                    }
                }
                #endif

                let prefillStart = Date.timeIntervalSinceReferenceDate
                let lmInput: LMInput
                if resolvedCIImages.isEmpty {
                    let tokens = context.tokenizer.encode(text: rendered.prompt, addSpecialTokens: false)
                    lmInput = LMInput(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
                } else {
                    #if canImport(MLXVLM)
                    let preTokens = context.tokenizer.encode(text: rendered.prompt, addSpecialTokens: false)
                    if let customProc = try? await Self.buildOverriddenVLMProcessor(
                        in: modelDirURL,
                        tokenizer: PreRenderedVLMTokenizerProxy(base: context.tokenizer, preRenderedTokens: preTokens)
                    ) {
                        let userIn = UserInput(chat: [.user(rendered.prompt, images: resolvedCIImages)])
                        lmInput = try await customProc.prepare(input: userIn)
                    } else {
                        var chatMessages: [Chat.Message] = []
                        if let sys = request.system, !sys.isEmpty {
                            chatMessages.append(.system(sys))
                        }
                        for (turnIdx, m) in request.messages.enumerated() {
                            switch m.role.lowercased() {
                            case "assistant": chatMessages.append(.assistant(m.content))
                            case "system": chatMessages.append(.system(m.content))
                            default:
                                let turnImgs = resolvedImagesByTurn[turnIdx] ?? []
                                var segs: [String] = []
                                for part in m.parts {
                                    if case .text(let t) = part, !t.isEmpty { segs.append(t) }
                                }
                                let merged = segs.isEmpty ? m.content : segs.joined(separator: "\n")
                                chatMessages.append(.user(merged, images: turnImgs))
                            }
                        }
                        lmInput = try await context.processor.prepare(
                            input: UserInput(chat: chatMessages)
                        )
                    }
                    #else
                    let tokens = context.tokenizer.encode(text: rendered.prompt, addSpecialTokens: false)
                    lmInput = LMInput(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
                    #endif
                }

                let promptTokenIDs = lmInput.text.tokens.asArray(Int.self)
                let probeHybrid = sig.isHybridRecurrentOrSWA
                let apcMatch = box.apcCache.match(
                    promptTokenIDs: promptTokenIDs,
                    mediaSalt: mediaSalt,
                    isHybridOrSWA: probeHybrid
                )

                let activeCaches: [any KVCache]
                let effInput: LMInput
                let reusedTokens: Int
                if let matchedCaches = apcMatch.caches, apcMatch.reusedPrefixTokens > 0 {
                    activeCaches = matchedCaches
                    effInput = LMInput(text: lmInput.text[apcMatch.reusedPrefixTokens...])
                    reusedTokens = apcMatch.reusedPrefixTokens
                } else {
                    activeCaches = try context.model.newCache(parameters: genParams)
                    effInput = lmInput
                    reusedTokens = 0
                }
                let isHybrid = !activeCaches.allSatisfy(\.isTrimmable)

                // Compose ThinkingControlLogitProcessor for GPU </think>/<channel|> injection
                let baseProc = genParams.processor()
                let thinkingProc = ThinkingControlLogitProcessor(
                    base: baseProc,
                    requestID: requestID,
                    reasoningFormat: activeTemplate.reasoningFormat,
                    thinkingEnabled: enableThinking,
                    startsInThinking: startsInThinking,
                    thinkingBudget: effectiveSampling.thinkingBudget,
                    tokenizer: context.tokenizer
                )

                let hasMedia = (lmInput.image != nil || lmInput.video != nil || lmInput.audio != nil)
                let allowMMMTP = spec.allowsMTPAfterMultimodalPrefill(sig)
                var mutableIter: PreparedIterator
                if (!hasMedia || allowMMMTP),
                   let mtp = box.mtpDrafter,
                   mtp.isCompatible(with: context.model) {
                    if !mtp.requiresGreedySampling,
                       let leviathan = try? LeviathanMTPTokenIterator(
                           input: effInput,
                           mainModel: context.model,
                           drafter: mtp,
                           mainCache: activeCaches,
                           mainState: apcMatch.state,
                           parameters: genParams,
                           processor: thinkingProc,
                           blockSize: 4,
                           draftMaxContext: sig.draftMaxContext
                       ) {
                        mutableIter = .leviathanMTP(leviathan)
                    } else {
                        let stdMTP = try MTPSpeculativeTokenIterator(
                            input: effInput,
                            mainModel: context.model,
                            drafter: mtp,
                            mainCache: activeCaches,
                            parameters: genParams,
                            blockSize: 4
                        )
                        mutableIter = .standardMTP(stdMTP)
                    }
                } else if !hasMedia,
                          let draftLM = box.draftLanguageModel {
                    let specIt = try SpeculativeTokenIterator(
                        input: effInput,
                        mainModel: context.model,
                        draftModel: draftLM,
                        mainCache: activeCaches,
                        mainState: apcMatch.state,
                        parameters: genParams,
                        numDraftTokens: 2
                    )
                    mutableIter = .speculative(specIt)
                } else {
                    let stdIt = try TokenIterator(
                        input: effInput,
                        model: context.model,
                        cache: activeCaches,
                        state: apcMatch.state,
                        processor: thinkingProc,
                        sampler: genParams.sampler(),
                        prefill: genParams.prefill,
                        maxTokens: genParams.maxTokens
                    )
                    mutableIter = .standard(stdIt)
                }

                let totalPromptTokens = promptTokenIDs.count
                let processedPromptTokens = max(1, totalPromptTokens - reusedTokens)
                let prefillDuration = max(1e-4, Date.timeIntervalSinceReferenceDate - prefillStart)
                let prefillTPS = Double(processedPromptTokens) / prefillDuration

                eventCont.yield(
                    .cacheDetail(
                        promptTokens: totalPromptTokens,
                        cachedTokens: reusedTokens,
                        outcome: reusedTokens > 0 ? .reused : .miss,
                        cause: reusedTokens > 0 ? nil : (apcMatch.missCause ?? "cold")
                    )
                )
                eventCont.yield(
                    .telemetry(
                        promptTokens: totalPromptTokens,
                        cachedTokens: reusedTokens,
                        promptTPS: prefillTPS
                    )
                )

                // Snapshot the prefilled prompt boundary before decode mutates the tail, preserving M-RoPE/model state
                if policy.snapshottingAtTurnAndSystemBoundaries {
                    let boundaryOffset = activeCaches.map(\.offset).min() ?? 0
                    var boundaryTokens: [Int]?
                    if boundaryOffset <= promptTokenIDs.count {
                        boundaryTokens = Array(promptTokenIDs.prefix(boundaryOffset))
                    } else if boundaryOffset == promptTokenIDs.count + 1,
                              case .leviathanMTP(let leviathan) = mutableIter,
                              let primed = leviathan.primedTokenID {
                        boundaryTokens = promptTokenIDs + [primed]
                    }
                    if let boundaryTokens, boundaryTokens.count == boundaryOffset {
                        box.apcCache.recordSnapshot(
                            tokenIDs: boundaryTokens,
                            mediaSalt: mediaSalt,
                            caches: activeCaches,
                            state: mutableIter.state,
                            isHybridOrSWA: isHybrid
                        )
                    }
                }

                let eosID = context.tokenizer.eosTokenId
                let unknownID = context.tokenizer.unknownTokenId
                var detokenizer = context.tokenizer.makeStreamingDetokenizer()
                var emittedTokenIDs: [Int] = []
                var terminalSteppedTokenID: Int?
                let decodeStart = Date.timeIntervalSinceReferenceDate

                while let token = autoreleasepool(invoking: { mutableIter.next() }) {
                    try Task.checkCancellation()
                    if let unknownID, token == unknownID {
                        mutableIter.discardGeneratedToken()
                        break
                    }
                    if token == eosID || stopIDs.contains(token) {
                        mutableIter.discardGeneratedToken()
                        terminalSteppedTokenID = token
                        break
                    }
                    mutableIter.recordEmittedToken(token)
                    emittedTokenIDs.append(token)
                    detokenizer.append(token: token)
                    if let piece = detokenizer.next(), !piece.isEmpty {
                        eventCont.yield(.rawToken(piece))
                    }
                    if let limit = genParams.maxTokens, limit > 0, emittedTokenIDs.count >= limit {
                        break
                    }
                }
                mutableIter.finalizeIfSpeculative()
                if let flushPiece = detokenizer.finish(), !flushPiece.isEmpty {
                    eventCont.yield(.rawToken(flushPiece))
                }

                // Record exact post-generation token sequence reconciled with actual cache offsets (zero re-encode or stop-token drift)
                let fullSequenceIDs = promptTokenIDs + emittedTokenIDs
                var finalOffset = activeCaches.map(\.offset).min() ?? 0
                if finalOffset > fullSequenceIDs.count {
                    let excess = finalOffset - fullSequenceIDs.count
                    if activeCaches.allSatisfy(\.isTrimmable),
                       (activeCaches.map(\.maxTrimCount).min() ?? 0) >= excess {
                        let trimmed = trimPromptCache(activeCaches, numTokens: excess)
                        finalOffset = max(0, finalOffset - trimmed)
                    }
                }
                let apcRecordedIDs: [Int]?
                if finalOffset <= fullSequenceIDs.count {
                    apcRecordedIDs = Array(fullSequenceIDs.prefix(finalOffset))
                } else if finalOffset == fullSequenceIDs.count + 1,
                          let steppedStop = terminalSteppedTokenID {
                    apcRecordedIDs = fullSequenceIDs + [steppedStop]
                } else {
                    apcRecordedIDs = nil
                }
                if let apcRecordedIDs, apcRecordedIDs.count == finalOffset {
                    box.apcCache.recordSnapshot(
                        tokenIDs: apcRecordedIDs,
                        mediaSalt: mediaSalt,
                        caches: activeCaches,
                        state: mutableIter.state,
                        isHybridOrSWA: isHybrid
                    )
                }

                let decodeDur = max(1e-4, Date.timeIntervalSinceReferenceDate - decodeStart)
                let genTPS = Double(emittedTokenIDs.count) / decodeDur
                let specReport: SpeculativeReport?
                if let specTel = mutableIter.speculativeDecodingTelemetry, specTel.draftTokenCount > 0 {
                    specReport = SpeculativeReport(
                        drafterID: specDrafterID,
                        mode: specModeLabel,
                        draftedTokens: specTel.draftTokenCount,
                        acceptedTokens: specTel.acceptedDraftTokenCount
                    )
                } else {
                    specReport = nil
                }
                eventCont.yield(
                    .telemetry(
                        promptTokens: totalPromptTokens,
                        cachedTokens: reusedTokens,
                        promptTPS: prefillTPS,
                        generationTPS: genTPS,
                        speculativeReport: specReport
                    )
                )

                return ContainerRunSummary(
                    reusedTokens: reusedTokens,
                    didTrim: apcMatch.didTrim,
                    isHybrid: isHybrid,
                    fullSequenceCount: fullSequenceIDs.count
                )
            }

            await self.updateObservedKVState(
                cachedTokens: summary.reusedTokens,
                sliceTruncated: summary.didTrim,
                isHybrid: summary.isHybrid
            )
            await self.recordContextUsedOnly(summary.fullSequenceCount)
        }
    }

    private func recordContextUsedOnly(_ total: Int) {
        lastContextUsed = total
    }

    #if canImport(MLXVLM)
    private struct OptionalProcessorConfiguration: Codable, Sendable {
        let processorClass: String?

        enum CodingKeys: String, CodingKey {
            case processorClass = "processor_class"
        }
    }

    private struct ConfigModelTypeProbe: Codable, Sendable {
        let modelType: String?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
        }
    }

    private static func mapModelTypeToVLMProcessorClass(_ modelType: String?) -> String? {
        switch modelType?.lowercased() {
        case "qwen3_5", "qwen3_5_moe", "qwen3_8", "qwen3_8_moe", "qwen3_vl", "qwen3_vl_moe":
            return "Qwen3VLProcessor"
        case "qwen2_5_vl":
            return "Qwen2_5_VLProcessor"
        case "qwen2_vl":
            return "Qwen2VLProcessor"
        case "mistral3":
            return "Mistral3Processor"
        case "gemma4_unified":
            return "Gemma4UnifiedProcessor"
        case "gemma4":
            return "Gemma4Processor"
        case "gemma3", "gemma3_text":
            return "Gemma3Processor"
        case "paligemma":
            return "PaliGemmaProcessor"
        case "pixtral":
            return "PixtralProcessor"
        case "lfm2_vl", "lfm2-vl":
            return "Lfm2VlProcessor"
        case "glm_ocr":
            return "Glm46VProcessor"
        case "muse_glimmer":
            return "MuseGlimmerProcessor"
        case "idefics3":
            return "Idefics3Processor"
        case "smolvlm":
            return "SmolVLMProcessor"
        case "llava_qwen2":
            return "FastVLMProcessor"
        default:
            return nil
        }
    }

    private static func buildOverriddenVLMProcessor(
        in dirURL: URL,
        tokenizer: any MLXLMCommon.Tokenizer
    ) async throws -> any UserInputProcessor {
        let modelType: String? = {
            let cfgURL = dirURL.appendingPathComponent("config.json")
            guard let cfgData = try? Data(contentsOf: cfgURL),
                  let probe = try? JSONDecoder().decode(ConfigModelTypeProbe.self, from: cfgData) else {
                return nil
            }
            return probe.modelType
        }()

        let candidates = ["preprocessor_config.json", "processor_config.json"]
        for name in candidates {
            let file = dirURL.appendingPathComponent(name)
            guard let data = try? Data(contentsOf: file) else { continue }
            let explicitClass = (try? JSONDecoder().decode(OptionalProcessorConfiguration.self, from: data))?.processorClass?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let resolvedClass = (explicitClass?.isEmpty == false ? explicitClass : nil)
                ?? mapModelTypeToVLMProcessorClass(modelType)
            if let resolvedClass {
                return try await VLMProcessorTypeRegistry.shared.createModel(
                    configuration: data,
                    processorType: resolvedClass,
                    tokenizer: tokenizer
                )
            }
        }
        throw ModelFactoryError.unsupportedProcessorType("missing preprocessor_config/processor_config")
    }
    #endif
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
        "json", "jsonl", "jinja", "txt", "model", "tiktoken", "py", "md", "toml",
    ]

    public static func planConversion(
        from sourceDirectory: URL,
        to outputDirectory: URL,
        spec: MLXConversionSpec = .init()
    ) throws -> MLXConversionPlan {
        let fm = FileManager.default
        let srcCanonical = sourceDirectory.standardizedFileURL
        let dstCanonical = outputDirectory.standardizedFileURL

        if srcCanonical.path == dstCanonical.path {
            throw MLXModelConversionError.outputDirectoryOverlapsSource(dstCanonical)
        }
        if fm.fileExists(atPath: dstCanonical.path) && !spec.overwriteExistingOutput {
            throw MLXModelConversionError.outputDirectoryExists(dstCanonical)
        }
        if spec.calibration == .q4Zero,
           (spec.resolvedBits != 4 || spec.resolvedGroupSize != 32 || spec.mode != .affine) {
            throw MLXModelConversionError.incompatibleCalibration(
                spec.calibration,
                spec.resolvedBits,
                spec.resolvedGroupSize,
                spec.mode
            )
        }

        let configURL = srcCanonical.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL),
              let configJSON = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw MLXModelConversionError.missingConfigJSON(srcCanonical)
        }

        if let quantDict = configJSON["quantization"] as? [String: Any],
           let qBits = quantDict["bits"] as? Int {
            throw MLXModelConversionError.sourceAlreadyQuantized(srcCanonical, "bits=\(qBits)")
        }
        if let quantConfig = configJSON["quantization_config"] as? [String: Any],
           !quantConfig.isEmpty {
            throw MLXModelConversionError.sourceAlreadyQuantized(srcCanonical, "quantization_config present")
        }

        let entries = (try? fm.contentsOfDirectory(at: srcCanonical, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        let safetensors = entries.filter { $0.pathExtension.lowercased() == "safetensors" }
            .map(\.lastPathComponent)
            .sorted()
        let hasPyTorchBins = entries.contains { $0.pathExtension.lowercased() == "bin" && $0.lastPathComponent.hasPrefix("pytorch_model") }
        if safetensors.isEmpty {
            if hasPyTorchBins {
                throw MLXModelConversionError.unsupportedPyTorchWeights(srcCanonical)
            }
            throw MLXModelConversionError.noSafetensorsFiles(srcCanonical)
        }

        let sidecars = entries.filter { url in
            let ext = url.pathExtension.lowercased()
            return sidecarExtensions.contains(ext) && url.lastPathComponent != "config.json"
        }.map(\.lastPathComponent).sorted()

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
        case .affine: mlxMode = .affine
        case .mxfp4: mlxMode = .mxfp4
        case .mxfp8: mlxMode = .mxfp8
        case .nvfp4: mlxMode = .nvfp4
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
