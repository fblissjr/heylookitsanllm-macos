import Accelerate
import Foundation
import HeylookCore

#if canImport(CoreAILanguageModels)
import CoreAILanguageModels
import Tokenizers
#endif

/// Inspector for Apple `coreai-models` `.aimodel` bundles (`metadata.json`, supporting Schema 0.1, Schema 0.2 nested layouts,
/// `speech_recognizer` Parakeet TDT / Whisper manifests, and legacy `encoder.aimodel` + `decoder.aimodel` Whisper directories).
public struct CoreAIBundleManifest: Codable, Sendable, Equatable {
    public let name: String
    public let kind: String
    public let schemaVersion: String
    public let vocabSize: Int
    public let maxContextLength: Int
    public let prefillChunkSize: Int
    public let componentNames: [String]
    public let visionBackbone: String?
    public let drafterKind: String?
    public let speechArchitecture: String?

    public var hasVisionComponent: Bool {
        kind.lowercased() == "vlm" || visionBackbone != nil || componentNames.contains("vision")
    }

    public var hasDrafterComponent: Bool {
        drafterKind != nil || componentNames.contains("drafter")
    }

    public var isSpeechRecognizer: Bool {
        kind.lowercased() == "speech_recognizer" || speechArchitecture != nil
    }

    public init(
        name: String,
        kind: String = "llm",
        schemaVersion: String = "0.2",
        vocabSize: Int = 151936,
        maxContextLength: Int = 8192,
        prefillChunkSize: Int = 64,
        componentNames: [String] = ["main"],
        visionBackbone: String? = nil,
        drafterKind: String? = nil,
        speechArchitecture: String? = nil
    ) {
        self.name = name
        self.kind = kind
        self.schemaVersion = schemaVersion
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
        self.prefillChunkSize = prefillChunkSize
        self.componentNames = componentNames
        self.visionBackbone = visionBackbone
        self.drafterKind = drafterKind
        self.speechArchitecture = speechArchitecture
    }

    /// Derives a truthful model ID for a `.aimodel` or CoreAI bundle directory without truncating version dots (e.g. `0.6b`).
    public static func canonicalBundleID(for bundleURL: URL) -> String {
        let std = bundleURL.standardizedFileURL
        let dir = std.path.hasSuffix("metadata.json") ? std.deletingLastPathComponent() : std
        if let data = try? Data(contentsOf: dir.appendingPathComponent("metadata.json")),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let metaName = json["name"] as? String,
           !metaName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return metaName
        }
        let ext = dir.pathExtension.lowercased()
        if ext == "aimodel" || ext == "mlmodelc" {
            return dir.deletingPathExtension().lastPathComponent
        }
        return dir.lastPathComponent
    }

    public static func inspect(at bundleURL: URL) throws -> CoreAIBundleManifest {
        let fm = FileManager.default
        let isMetaFile = bundleURL.standardizedFileURL.path.hasSuffix("metadata.json")
        let metaURL = isMetaFile
            ? bundleURL
            : bundleURL.appendingPathComponent("metadata.json")
        let bundleDir = isMetaFile ? bundleURL.deletingLastPathComponent() : bundleURL

        if !fm.fileExists(atPath: metaURL.path) {
            let encURL = bundleDir.appendingPathComponent("encoder.aimodel")
            let decURL = bundleDir.appendingPathComponent("decoder.aimodel")
            if fm.fileExists(atPath: encURL.path) && fm.fileExists(atPath: decURL.path) {
                return CoreAIBundleManifest(
                    name: Self.canonicalBundleID(for: bundleDir),
                    kind: "speech_recognizer",
                    schemaVersion: "0.1",
                    vocabSize: 51866,
                    maxContextLength: 448,
                    prefillChunkSize: 64,
                    componentNames: ["decoder.aimodel", "encoder.aimodel"],
                    visionBackbone: nil,
                    drafterKind: nil,
                    speechArchitecture: "whisper"
                )
            }
        }

        let data = try Data(contentsOf: metaURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }

        let schemaVersion = (json["schema_version"] as? String)
            ?? (json["schemaVersion"] as? String)
            ?? "0.1"
        let langDict = json["language"] as? [String: Any]
        let visionDict = json["vision"] as? [String: Any]
        let specDict = json["speculative"] as? [String: Any]
        let assetsDict = json["assets"] as? [String: Any]
        let configDict = json["config"] as? [String: Any]

        let name = (json["name"] as? String)
            ?? (json["tokenizer"] as? String)?.components(separatedBy: "/").last
            ?? Self.canonicalBundleID(for: bundleDir)
        let rawKind = (json["kind"] as? String)
            ?? (json["model_type"] as? String)
            ?? (visionDict != nil ? "vlm" : "llm")
        let speechArch: String?
        if rawKind.lowercased() == "speech_recognizer" {
            speechArch = ((configDict?["architecture"] as? String) ?? "whisper").lowercased()
        } else {
            speechArch = nil
        }
        let vocabSize = (json["vocabSize"] as? Int)
            ?? (json["vocab_size"] as? Int)
            ?? (langDict?["vocab_size"] as? Int)
            ?? ((langDict?["overrides"] as? [String: Any])?["vocab_size"] as? Int)
            ?? (configDict?["vocab_size"] as? Int)
            ?? (speechArch != nil ? 51866 : 151936)
        let maxContextLength = (json["maxContextLength"] as? Int)
            ?? (json["max_context_length"] as? Int)
            ?? (langDict?["max_context_length"] as? Int)
            ?? (speechArch != nil ? 448 : 8192)
        let prefillChunkSize = (json["prefillChunkSize"] as? Int)
            ?? (json["prefill_chunk_size"] as? Int)
            ?? (langDict?["prefill_chunk_size"] as? Int)
            ?? 64
        let visionBackbone = (visionDict?["backbone"] as? String)
        let drafterKind = (specDict?["drafter_kind"] as? String)

        var compsSet = Set<String>()
        if let compDict = json["components"] as? [String: Any] {
            compsSet.formUnion(compDict.keys)
        } else if let compArray = json["components"] as? [[String: Any]] {
            compsSet.formUnion(compArray.compactMap { $0["name"] as? String })
        } else if let compStrings = json["components"] as? [String] {
            compsSet.formUnion(compStrings)
        }
        if let assetsDict {
            for (k, _) in assetsDict where k != "tokenizer_dir" {
                compsSet.insert(k)
            }
        }
        if visionDict != nil {
            compsSet.insert("vision")
        }
        if specDict != nil {
            compsSet.insert("drafter")
        }
        if compsSet.isEmpty {
            compsSet.insert("main")
        }

        return CoreAIBundleManifest(
            name: name,
            kind: rawKind,
            schemaVersion: schemaVersion,
            vocabSize: vocabSize,
            maxContextLength: maxContextLength,
            prefillChunkSize: prefillChunkSize,
            componentNames: compsSet.sorted(),
            visionBackbone: visionBackbone,
            drafterKind: drafterKind,
            speechArchitecture: speechArch
        )
    }
}

/// Tier 2 Provider: In-process Apple Core AI (`coreai-models` `.aimodel` bundles) for ANE/GPU LLM inference.
/// Speech recognition is delegated cleanly to `SpeechTranscriptionService`.
/// Strictly truthful:
/// - `supportsVision` is reported `false` until `CoreAIVision` tensor inputs are wired.
/// - `speculativeReport` stays `nil` unless live speculative token counts are returned.
/// - Throws `ProviderExecutionError` when compiled `.aimodel` / runtime components are missing (never emits synthetic echo text).
public actor CoreAIProvider: InferenceProvider {
    // Compatibility bridges preserved for HeylookUI / HeylookServer / HeylookTests while speech lives in SpeechTranscriptionService.
    public typealias ANESpeechTranscriptionResult = HeylookEngines.ANESpeechTranscriptionResult

    public static let systemSpeechModelID = SpeechTranscriptionService.systemSpeechModelID
    public static let systemSpeechModelPath = "system://speech-transcriber"

    public static var isSystemSpeechAvailable: Bool {
        SpeechTranscriptionService.isSystemSpeechAvailable
    }

    public static var isCoreAISpeechLinked: Bool {
        SpeechTranscriptionService.isCoreAISpeechLinked
    }

    public static func systemSpeechDescription() -> EngineDescription {
        SpeechTranscriptionService.systemSpeechDescription()
    }

    public static func decodeAudioMonoPCM(
        from audioData: Data,
        targetSampleRate: Int = 16_000
    ) async -> [Float] {
        await SpeechTranscriptionService.decodeAudioMonoPCM(from: audioData, targetSampleRate: targetSampleRate)
    }

    public static func decodeRIFFChunksMonoPCM(
        from data: Data,
        targetSampleRate: Int = 16_000
    ) -> [Float]? {
        SpeechTranscriptionService.decodeRIFFChunksMonoPCM(from: data, targetSampleRate: targetSampleRate)
    }

    public static func encodeCanonicalPCM16WAV(samples: [Float], sampleRate: Int = 16_000) -> Data {
        SpeechTranscriptionService.encodeCanonicalPCM16WAV(samples: samples, sampleRate: sampleRate)
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .coreai
    public nonisolated let isSystemSpeechProvider: Bool
    private let templateInfo: ModelTemplateInfo
    private var descriptionState: EngineDescription
    private var prefixCache = PrefixCacheTracker()
    private var bundleManifest: CoreAIBundleManifest?
    private let speechService: SpeechTranscriptionService?
    private var lastContextUsed: Int?

    #if canImport(CoreAILanguageModels)
    private var loadedCoreAIBundle: LanguageModelBundle?
    private var loadedCoreAITokenizer: (any Tokenizers.Tokenizer)?
    private var loadedCoreAIEngine: (any InferenceEngine)?
    private var engineCachedTokenIDs: [Int] = []
    #endif

    public init(description: EngineDescription) {
        self.modelID = description.modelID
        self.isSystemSpeechProvider = description.modelID == Self.systemSpeechModelID
            || description.modelPath == Self.systemSpeechModelPath
        var updatedDesc = description
        if self.isSystemSpeechProvider {
            self.templateInfo = ModelTemplateInfo(
                reasoningFormat: .none,
                thinkingControls: ThinkingControls(
                    supportsThinking: false,
                    canDisableThinking: false,
                    defaultEnabled: false,
                    switchVariable: nil,
                    depthVariable: nil,
                    depthValues: []
                ),
                specialTokens: [],
                strippableSpecialTokens: [],
                eosTokenIDs: [],
                templateSource: "speech_recognizer"
            )
            self.bundleManifest = nil
            self.speechService = SpeechTranscriptionService(description: updatedDesc)
            self.descriptionState = updatedDesc
            return
        }

        let url = URL(fileURLWithPath: description.modelPath)
        let inspectedTemplate = ModelTemplateInfo.inspect(
            modelDirectory: url,
            modelID: description.modelID
        )
        self.templateInfo = inspectedTemplate
        let inspectedManifest = try? CoreAIBundleManifest.inspect(at: url)
        self.bundleManifest = inspectedManifest

        if let manifest = inspectedManifest {
            let resolvedArch = manifest.speechArchitecture ?? updatedDesc.architecture.value
            var updatedTags = updatedDesc.tags
            if manifest.isSpeechRecognizer {
                if !updatedTags.contains("asr") { updatedTags.append("asr") }
                if !updatedTags.contains("speech") { updatedTags.append("speech") }
            }
            // Truthful CoreAI contract: `supportsVision` remains `false` until CoreAIVision tensor inputs are wired.
            updatedDesc = EngineDescription(
                modelID: updatedDesc.modelID,
                provider: updatedDesc.provider,
                modelPath: updatedDesc.modelPath,
                architecture: ProvenanceField(
                    value: resolvedArch,
                    provenance: manifest.speechArchitecture != nil ? .observed : updatedDesc.architecture.provenance,
                    effect: .reload
                ),
                quantization: updatedDesc.quantization,
                computePlacement: updatedDesc.computePlacement,
                isPinned: updatedDesc.isPinned,
                isLoaded: updatedDesc.isLoaded,
                contextSize: updatedDesc.contextSize.provenance == .derived
                    ? ProvenanceField(value: manifest.maxContextLength, provenance: .observed, effect: .reload)
                    : updatedDesc.contextSize,
                flashAttention: updatedDesc.flashAttention,
                supportsVision: ProvenanceField(value: false, provenance: .observed, effect: .reload),
                supportsAudio: ProvenanceField(
                    value: manifest.isSpeechRecognizer || updatedDesc.supportsAudio.value,
                    provenance: manifest.isSpeechRecognizer ? .observed : updatedDesc.supportsAudio.provenance,
                    effect: .reload
                ),
                supportsEmbeddings: updatedDesc.supportsEmbeddings,
                supportsThinking: ProvenanceField(
                    value: manifest.isSpeechRecognizer ? false : updatedDesc.supportsThinking.value,
                    provenance: manifest.isSpeechRecognizer ? .observed : updatedDesc.supportsThinking.provenance,
                    effect: .perRequest
                ),
                thinkingBudget: updatedDesc.thinkingBudget,
                reasoningFormat: ProvenanceField(
                    value: manifest.isSpeechRecognizer ? "none" : inspectedTemplate.reasoningFormat.wireIdentifier,
                    provenance: manifest.isSpeechRecognizer ? .notApplicable : .derived,
                    effect: .reload
                ),
                thinkingControls: manifest.isSpeechRecognizer ? nil : inspectedTemplate.thinkingControls,
                templateSource: manifest.isSpeechRecognizer
                    ? ProvenanceField(value: "speech_recognizer", provenance: .notApplicable, effect: .reload)
                    : updatedDesc.templateSource,
                speculativeDrafter: updatedDesc.speculativeDrafter,
                estimatedMemoryMB: updatedDesc.estimatedMemoryMB,
                tags: updatedTags
            )
            self.speechService = manifest.isSpeechRecognizer
                ? SpeechTranscriptionService(description: updatedDesc, manifest: manifest)
                : nil
        } else {
            updatedDesc = EngineDescription(
                modelID: updatedDesc.modelID,
                provider: updatedDesc.provider,
                modelPath: updatedDesc.modelPath,
                architecture: updatedDesc.architecture,
                quantization: updatedDesc.quantization,
                computePlacement: updatedDesc.computePlacement,
                isPinned: updatedDesc.isPinned,
                isLoaded: updatedDesc.isLoaded,
                contextSize: updatedDesc.contextSize,
                flashAttention: updatedDesc.flashAttention,
                supportsVision: ProvenanceField(value: false, provenance: .observed, effect: .reload),
                supportsAudio: updatedDesc.supportsAudio,
                supportsEmbeddings: updatedDesc.supportsEmbeddings,
                supportsThinking: updatedDesc.supportsThinking,
                thinkingBudget: updatedDesc.thinkingBudget,
                reasoningFormat: updatedDesc.reasoningFormat,
                thinkingControls: updatedDesc.thinkingControls,
                templateSource: updatedDesc.templateSource,
                speculativeDrafter: updatedDesc.speculativeDrafter,
                estimatedMemoryMB: updatedDesc.estimatedMemoryMB,
                tags: updatedDesc.tags
            )
            self.speechService = nil
        }

        self.descriptionState = updatedDesc
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func manifest() async -> CoreAIBundleManifest? {
        bundleManifest
    }

    public func observedContractState() async -> ObservedContractState {
        ObservedContractState(
            runningContextLength: descriptionState.isLoaded ? descriptionState.contextSize.value : nil,
            loadedTemplateText: nil,
            runtimeCaps: nil,
            speculativeInForce: nil,
            isPrefixCacheWarm: prefixCache.isWarm,
            contextUsed: lastContextUsed
        )
    }

    public func renderPromptPreview(request: MessagesRequest) async throws -> PromptPreviewResult {
        let modelURL = URL(fileURLWithPath: descriptionState.modelPath)
        let enableThinking = ModelSpecializationRegistry.resolveThinkingEnabled(
            requestThinking: request.thinking,
            sidecarThinking: nil,
            modelCapable: descriptionState.supportsThinking.value
        )
        let rendered = try HeylookConfig.renderPromptDetailed(
            in: FileManager.default.fileExists(atPath: modelURL.path) ? modelURL : nil,
            system: request.system,
            turns: request.messages,
            thinkingEnabled: enableThinking,
            thinkingDepth: request.thinkingDepth
        )
        #if canImport(CoreAILanguageModels)
        if let tokenizer = loadedCoreAITokenizer {
            let ids = tokenizer.encode(text: rendered.prompt)
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

    public func preload() async throws {
        if let speechService {
            try await speechService.preload()
            descriptionState = await speechService.describe()
            return
        }

        #if canImport(CoreAILanguageModels)
        if let mainComp = bundleManifest?.mainComponent,
           FileManager.default.fileExists(
               atPath: URL(fileURLWithPath: descriptionState.modelPath).appendingPathComponent(mainComp).path
           ) {
            _ = try await ensureCoreAILoaded()
            return
        }
        #endif

        if bundleManifest == nil {
            throw ProviderExecutionError.checkpointNotFound(
                modelID: modelID,
                path: descriptionState.modelPath
            )
        }
        throw ProviderExecutionError.runtimeUnavailable(
            modelID: modelID,
            detail: "Compiled Core AI runtime unavailable at \(descriptionState.modelPath) (build with HEYLOOK_ENABLE_COREAI=1 for .aimodel bundles)."
        )
    }

    public func clearCache() async -> Bool {
        prefixCache.reset()
        lastContextUsed = nil
        #if canImport(CoreAILanguageModels)
        engineCachedTokenIDs.removeAll()
        if let engine = loadedCoreAIEngine {
            try? await engine.reset()
        }
        #endif
        return true
    }

    public func unload() async {
        _ = await clearCache()
        if let speechService {
            await speechService.unload()
        }
        descriptionState.isLoaded = false
        #if canImport(CoreAILanguageModels)
        loadedCoreAIEngine = nil
        loadedCoreAITokenizer = nil
        loadedCoreAIBundle = nil
        #endif
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        if bundleManifest?.isSpeechRecognizer == true || isSystemSpeechProvider {
            let err = ProviderExecutionError.invalidRequest(
                detail: "Speech recognizer model '\(modelID)' accepts audio transcription, not /v1/messages chat generation."
            )
            return AsyncThrowingStream { $0.finish(throwing: err) }
        }

        let hasImageAttachment = request.messages.contains { turn in
            turn.parts.contains {
                if case .media(let att) = $0 { return att.kind == "image" }
                return false
            }
        }
        if hasImageAttachment {
            let err = ProviderExecutionError.invalidRequest(
                detail: "CoreAIProvider does not yet wire image tensor inputs for '\(modelID)' (supportsVision=false until CoreAIVision is implemented)."
            )
            return AsyncThrowingStream { $0.finish(throwing: err) }
        }

        let placement = descriptionState.computePlacement.value
        let cacheReport = prefixCache.evaluate(system: request.system, messages: request.messages)
        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value
        let resolvedStops = effectiveSampling.stopSequences ?? request.stopSequences ?? []

        #if canImport(CoreAILanguageModels)
        if let mainComp = bundleManifest?.mainComponent,
           FileManager.default.fileExists(
               atPath: URL(fileURLWithPath: descriptionState.modelPath).appendingPathComponent(mainComp).path
           ) {
            return generateWithCoreAIEngine(
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

        let err: ProviderExecutionError = bundleManifest == nil
            ? .checkpointNotFound(modelID: modelID, path: descriptionState.modelPath)
            : .runtimeUnavailable(
                modelID: modelID,
                detail: "Compiled .mlmodelc component not found in \(descriptionState.modelPath) or CoreAILanguageModels not linked."
            )
        return AsyncThrowingStream { continuation in
            continuation.finish(throwing: err)
        }
    }

    public func transcribeAudio(
        wavData: Data,
        languageHint: String = "en"
    ) async throws -> ANESpeechTranscriptionResult {
        if let speechService {
            return try await speechService.transcribeAudio(wavData: wavData, languageHint: languageHint)
        }
        let ephemeralService = SpeechTranscriptionService(description: descriptionState, manifest: bundleManifest)
        return try await ephemeralService.transcribeAudio(wavData: wavData, languageHint: languageHint)
    }

    private func recordAssistantTurn(_ reply: String) {
        prefixCache.recordAssistantReply(reply)
    }

    private func recordContextUsed(_ total: Int) {
        lastContextUsed = total
    }

    #if canImport(CoreAILanguageModels)
    private func ensureCoreAILoaded() async throws -> (LanguageModelBundle, any Tokenizers.Tokenizer, any InferenceEngine) {
        if let b = loadedCoreAIBundle, let t = loadedCoreAITokenizer, let e = loadedCoreAIEngine {
            return (b, t, e)
        }
        let bundleURL = URL(fileURLWithPath: descriptionState.modelPath)
        let bundle = try LanguageModelBundle(at: bundleURL)
        let tokenizer = try await bundle.loadTokenizer()
        let runner = CoreAIRunner(bundle: bundle)
        let engine = try await runner.makeInferenceEngine()
        loadedCoreAIBundle = bundle
        loadedCoreAITokenizer = tokenizer
        loadedCoreAIEngine = engine
        engineCachedTokenIDs.removeAll()
        descriptionState.isLoaded = true
        return (bundle, tokenizer, engine)
    }

    private func cancelEngineIfNeeded() async {
        try? await loadedCoreAIEngine?.cancel()
    }

    private func prepareEngineTokens(
        engine: any InferenceEngine,
        fullPromptTokens: [Int]
    ) async throws -> (deltaTokens: [Int], cachedCount: Int) {
        if !engineCachedTokenIDs.isEmpty,
           fullPromptTokens.count > engineCachedTokenIDs.count,
           Array(fullPromptTokens.prefix(engineCachedTokenIDs.count)) == engineCachedTokenIDs {
            let delta = Array(fullPromptTokens.dropFirst(engineCachedTokenIDs.count))
            engineCachedTokenIDs = fullPromptTokens
            return (delta, fullPromptTokens.count - delta.count)
        }
        try await engine.reset()
        engineCachedTokenIDs = fullPromptTokens
        return (fullPromptTokens, 0)
    }

    private func appendGeneratedEngineTokenID(_ tokenID: Int) {
        engineCachedTokenIDs.append(tokenID)
    }

    private func generateWithCoreAIEngine(
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
        let modelPath = self.descriptionState.modelPath
        let lastTurn = request.messages.last
        let isAssistantContinuation = lastTurn?.role.lowercased() == "assistant"
        let startsInThinking = enableThinking
            && isAssistantContinuation
            && !(lastTurn?.thinking?.isEmpty ?? true)
            && (lastTurn?.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)

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
            onAbortHook: { [weak self] in
                await self?.cancelEngineIfNeeded()
            },
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        ) { [weak self] eventCont in
            guard let self else { return }
            let (_, tokenizer, engine) = try await self.ensureCoreAILoaded()
            let rendered = try HeylookConfig.renderPromptDetailed(
                in: URL(fileURLWithPath: modelPath),
                system: request.system,
                turns: request.messages,
                thinkingEnabled: enableThinking,
                thinkingDepth: effectiveSampling.thinkingDepth ?? request.thinkingDepth
            )
            let promptTokenIds = tokenizer.encode(text: rendered.prompt)
            let (inputTokens, reusedCount) = try await self.prepareEngineTokens(
                engine: engine,
                fullPromptTokens: promptTokenIds
            )
            let samplingCfg = SamplingConfiguration(
                temperature: effectiveSampling.temperature ?? 0.7,
                topK: effectiveSampling.topK,
                topP: effectiveSampling.topP,
                minP: effectiveSampling.minP,
                repetitionPenalty: effectiveSampling.repetitionPenalty,
                repetitionPenaltyWindow: effectiveSampling.repetitionContextSize,
                seed: effectiveSampling.seed.map { UInt64(bitPattern: Int64($0)) }
            )
            let maxTokens = effectiveSampling.maxTokens ?? SamplerCascade.defaultMaxTokens
            let eosSet = Set(activeTemplate.eosTokenIDs)
            let startTime = Date.timeIntervalSinceReferenceDate
            var firstTokenTime: TimeInterval?
            var generatedCount = 0

            let tokenStream = try await engine.generate(
                with: inputTokens.map(Int32.init),
                samplingConfiguration: samplingCfg,
                inferenceOptions: .extend(maxTokens: maxTokens)
            )
            for try await step in tokenStream {
                try Task.checkCancellation()
                let tokenInt = Int(step.tokenId)
                if tokenInt == tokenizer.eosTokenId || eosSet.contains(tokenInt) {
                    break
                }
                if firstTokenTime == nil {
                    firstTokenTime = Date.timeIntervalSinceReferenceDate
                }
                generatedCount += 1
                await self.appendGeneratedEngineTokenID(tokenInt)
                let piece = tokenizer.decode(tokens: [tokenInt], skipSpecialTokens: false)
                eventCont.yield(.rawToken(piece))
            }

            await self.recordContextUsed(promptTokenIds.count + generatedCount)
            let now = Date.timeIntervalSinceReferenceDate
            let prefillDur = max(1e-4, (firstTokenTime ?? now) - startTime)
            let decodeDur = max(1e-4, now - (firstTokenTime ?? startTime))
            eventCont.yield(
                .cacheDetail(
                    promptTokens: promptTokenIds.count,
                    cachedTokens: reusedCount,
                    outcome: reusedCount > 0 ? .reused : .miss,
                    cause: reusedCount > 0 ? nil : "cold"
                )
            )
            eventCont.yield(
                .telemetry(
                    promptTokens: promptTokenIds.count,
                    cachedTokens: reusedCount,
                    promptTPS: Double(max(1, inputTokens.count)) / prefillDur,
                    generationTPS: Double(generatedCount) / decodeDur,
                    speculativeReport: nil
                )
            )
        }
    }
    #endif
}

#if canImport(CoreAILanguageModels)
extension CoreAIBundleManifest {
    var mainComponent: String? {
        if componentNames.contains("main.mlmodelc") { return "main.mlmodelc" }
        if componentNames.contains("main") { return "main.mlmodelc" }
        return componentNames.first
    }
}
#endif
