import Accelerate
import AVFoundation
import CoreMedia
import Foundation
import HeylookCore
import HeylookEmbeddings
import Speech

#if canImport(CoreAILanguageModels)
import CoreAILanguageModels
import Tokenizers
#endif

#if canImport(CoreAISpeech)
import CoreAISpeech
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
            // Support legacy `coreai-models` Whisper bundle layout (`encoder.aimodel` + `decoder.aimodel` without `metadata.json`)
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

/// Tier 2 Provider: In-process Apple Core AI (`coreai-models` `.aimodel` bundles) & CoreML
/// executing on the Apple Neural Engine (`.ane`) or GPU, with `.aimodel` `metadata.json` inspection.
public actor CoreAIProvider: InferenceProvider {
    public static let systemSpeechModelID = "apple-speech-transcriber"
    public static let systemSpeechModelPath = "system://speech-transcriber"

    public static var isSystemSpeechAvailable: Bool {
        SpeechTranscriber.isAvailable
    }

    public static var isCoreAISpeechLinked: Bool {
        #if canImport(CoreAISpeech)
        return true
        #else
        return false
        #endif
    }

    public static func systemSpeechDescription() -> EngineDescription {
        EngineDescription(
            modelID: systemSpeechModelID,
            provider: .coreai,
            modelPath: systemSpeechModelPath,
            architecture: ProvenanceField(value: "apple_speech_transcriber", provenance: .observed, effect: .reload),
            quantization: ProvenanceField(value: "native", provenance: .observed, effect: .reload),
            computePlacement: ProvenanceField(value: .ane, provenance: .observed, effect: .reload, computePlacement: .ane),
            isPinned: false,
            isLoaded: false,
            contextSize: ProvenanceField(value: 448, provenance: .notApplicable, effect: .reload),
            flashAttention: ProvenanceField(value: false, provenance: .notApplicable, effect: .reload),
            supportsVision: ProvenanceField(value: false, provenance: .observed, effect: .reload),
            supportsAudio: ProvenanceField(value: true, provenance: .observed, effect: .reload),
            supportsEmbeddings: ProvenanceField(value: false, provenance: .observed, effect: .reload),
            supportsThinking: ProvenanceField(value: false, provenance: .observed, effect: .perRequest),
            thinkingBudget: ProvenanceField(value: nil, provenance: .notApplicable, effect: .perRequest),
            reasoningFormat: ProvenanceField(value: "none", provenance: .notApplicable, effect: .reload),
            thinkingControls: nil,
            templateSource: ProvenanceField(value: "speech_recognizer", provenance: .notApplicable, effect: .reload),
            speculativeDrafter: ProvenanceField(value: nil, provenance: .notApplicable, effect: .reload),
            estimatedMemoryMB: 0,
            tags: ["asr", "speech", "system"]
        )
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .coreai
    public nonisolated let allowSyntheticFallback: Bool
    public nonisolated let isSystemSpeechProvider: Bool
    private let templateInfo: ModelTemplateInfo
    private var descriptionState: EngineDescription
    private var prefixCache = PrefixCacheTracker()
    private var bundleManifest: CoreAIBundleManifest?
    private var unloadCount: Int = 0

    #if canImport(CoreAILanguageModels)
    private var loadedCoreAIBundle: LanguageModelBundle?
    private var loadedCoreAITokenizer: (any Tokenizers.Tokenizer)?
    private var loadedCoreAIEngine: (any InferenceEngine)?
    private var engineCachedTokenIDs: [Int] = []
    #endif

    #if canImport(CoreAISpeech)
    private var loadedSpeechModel: SpeechRecognitionModel?
    #endif

    public init(description: EngineDescription, allowSyntheticFallback: Bool = false) {
        self.modelID = description.modelID
        self.allowSyntheticFallback = allowSyntheticFallback
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
            self.descriptionState = updatedDesc
            return
        }
        let url = URL(fileURLWithPath: description.modelPath)
        let inspectedTemplate = ModelTemplateInfo.inspect(
            modelDirectory: url,
            modelID: description.modelID
        )
        self.templateInfo = inspectedTemplate
        if let manifest = try? CoreAIBundleManifest.inspect(at: url) {
            self.bundleManifest = manifest
            if updatedDesc.contextSize.provenance == .derived {
                let resolvedArch = manifest.speechArchitecture ?? updatedDesc.architecture.value
                var updatedTags = updatedDesc.tags
                if manifest.isSpeechRecognizer {
                    if !updatedTags.contains("asr") { updatedTags.append("asr") }
                    if !updatedTags.contains("speech") { updatedTags.append("speech") }
                }
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
                    contextSize: ProvenanceField(
                        value: manifest.maxContextLength,
                        provenance: .observed,
                        effect: .reload
                    ),
                    flashAttention: updatedDesc.flashAttention,
                    supportsVision: ProvenanceField(
                        value: !manifest.isSpeechRecognizer && (manifest.hasVisionComponent || updatedDesc.supportsVision.value),
                        provenance: .observed,
                        effect: .reload
                    ),
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
                    estimatedMemoryMB: updatedDesc.estimatedMemoryMB,
                    tags: updatedTags
                )
            }
        }
        self.descriptionState = updatedDesc
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func manifest() async -> CoreAIBundleManifest? {
        bundleManifest
    }

    public func preload() async throws {
        if isSystemSpeechProvider {
            guard SpeechTranscriber.isAvailable || allowSyntheticFallback else {
                throw ProviderExecutionError.runtimeUnavailable(
                    modelID: modelID,
                    detail: "Apple SpeechTranscriber is unavailable on this host."
                )
            }
            descriptionState.isLoaded = true
            return
        }
        #if canImport(CoreAISpeech)
        if bundleManifest?.isSpeechRecognizer == true,
           (!allowSyntheticFallback || Self.hasCompiledSpeechWeights(at: descriptionState.modelPath)) {
            _ = try await ensureCoreAISpeechLoaded()
            return
        }
        #endif
        #if canImport(CoreAILanguageModels)
        if bundleManifest?.isSpeechRecognizer != true,
           let mainComp = bundleManifest?.mainComponent,
           FileManager.default.fileExists(
               atPath: URL(fileURLWithPath: descriptionState.modelPath).appendingPathComponent(mainComp).path
           ) {
            _ = try await ensureCoreAILoaded()
            return
        }
        #endif
        if allowSyntheticFallback {
            descriptionState.isLoaded = true
            return
        }
        if bundleManifest == nil {
            throw ProviderExecutionError.checkpointNotFound(
                modelID: modelID,
                path: descriptionState.modelPath
            )
        }
        throw ProviderExecutionError.runtimeUnavailable(
            modelID: modelID,
            detail: "Compiled Core AI runtime unavailable at \(descriptionState.modelPath) (build with HEYLOOK_ENABLE_COREAI=1 for .aimodel bundles)"
        )
    }

    public func clearCache() async -> Bool {
        prefixCache.reset()
        #if canImport(CoreAILanguageModels)
        engineCachedTokenIDs.removeAll()
        if let engine = loadedCoreAIEngine {
            try? await engine.reset()
        }
        #endif
        return true
    }

    public func unload() async {
        unloadCount += 1
        _ = await clearCache()
        descriptionState.isLoaded = false
        #if canImport(CoreAILanguageModels)
        loadedCoreAIEngine = nil
        loadedCoreAITokenizer = nil
        loadedCoreAIBundle = nil
        #endif
        #if canImport(CoreAISpeech)
        loadedSpeechModel = nil
        #endif
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
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

        guard allowSyntheticFallback else {
            let err: ProviderExecutionError = bundleManifest == nil
                ? .checkpointNotFound(modelID: modelID, path: descriptionState.modelPath)
                : .runtimeUnavailable(
                    modelID: modelID,
                    detail: "Compiled .mlmodelc component not found in \(descriptionState.modelPath)"
                )
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: err)
            }
        }

        let hasDrafter = (bundleManifest?.hasDrafterComponent == true)
            || descriptionState.tags.contains(where: { $0.localizedCaseInsensitiveContains("drafter") || $0.localizedCaseInsensitiveContains("mtp") })
        let syntheticSpecReport: SpeculativeReport? = hasDrafter
            ? SpeculativeReport(
                drafterID: "\(modelID)-drafter",
                mode: bundleManifest?.drafterKind ?? "speculative",
                draftedTokens: 16,
                acceptedTokens: 13
            )
            : nil

        let userText = request.messages.last?.content ?? ""
        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Running CoreAI .aimodel state graph on ANE...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[CoreAI \(modelID)] ", userText])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            cacheReport: cacheReport,
            speculativeReport: syntheticSpecReport,
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
        prefixCache.recordAssistantReply(reply)
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
        // Only extend stateful InferenceEngine when the prior cached sequence is an exact prefix
        // of `fullPromptTokens` with at least 1 new token to evaluate; otherwise reset cleanly!
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
            let maxTokens = effectiveSampling.maxTokens ?? 2048
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

            let now = Date.timeIntervalSinceReferenceDate
            let prefillDur = max(1e-4, (firstTokenTime ?? now) - startTime)
            let decodeDur = max(1e-4, now - (firstTokenTime ?? startTime))
            eventCont.yield(
                .telemetry(
                    promptTokens: promptTokenIds.count,
                    cachedTokens: reusedCount,
                    promptTPS: Double(max(1, inputTokens.count)) / prefillDur,
                    generationTPS: Double(generatedCount) / decodeDur
                )
            )
        }
    }
    #endif

    // MARK: - ANE Speech / Audio Frontend (CoreAISpeech Whisper & Parakeet TDT + System SpeechTranscriber)

    public struct ANESpeechTranscriptionResult: Sendable, Equatable, Codable {
        public let text: String
        public let durationSeconds: Double
        public let sampleRateHz: Int
        public let rmsEnergy: Float
        public let melFrames: Int
        public let computePlacement: ComputePlacement
        public let backend: String
        public let modelID: String

        enum CodingKeys: String, CodingKey {
            case text
            case durationSeconds
            case sampleRateHz
            case rmsEnergy
            case melFrames
            case computePlacement
            case backend
            case modelID = "model_id"
        }

        public init(
            text: String,
            durationSeconds: Double,
            sampleRateHz: Int = 16_000,
            rmsEnergy: Float = 0,
            melFrames: Int = 0,
            computePlacement: ComputePlacement = .ane,
            backend: String = "coreai_speech",
            modelID: String = ""
        ) {
            self.text = text
            self.durationSeconds = durationSeconds
            self.sampleRateHz = sampleRateHz
            self.rmsEnergy = rmsEnergy
            self.melFrames = melFrames
            self.computePlacement = computePlacement
            self.backend = backend
            self.modelID = modelID
        }

        public init(from decoder: any Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.text = try c.decode(String.self, forKey: .text)
            self.durationSeconds = try c.decode(Double.self, forKey: .durationSeconds)
            self.sampleRateHz = try c.decodeIfPresent(Int.self, forKey: .sampleRateHz) ?? 16_000
            self.rmsEnergy = try c.decodeIfPresent(Float.self, forKey: .rmsEnergy) ?? 0
            self.melFrames = try c.decodeIfPresent(Int.self, forKey: .melFrames) ?? 0
            self.computePlacement = try c.decodeIfPresent(ComputePlacement.self, forKey: .computePlacement) ?? .ane
            self.backend = try c.decodeIfPresent(String.self, forKey: .backend) ?? "coreai_speech"
            self.modelID = try c.decodeIfPresent(String.self, forKey: .modelID) ?? ""
        }
    }

    private static func hasCompiledSpeechWeights(at bundlePath: String) -> Bool {
        let fm = FileManager.default
        let root = URL(fileURLWithPath: bundlePath)
        let candidates = [
            root.appendingPathComponent("encoder.aimodel/coremldata.bin"),
            root.appendingPathComponent("encoder.aimodel/model.mil"),
            root.appendingPathComponent("encoder.aimodel/weights"),
            root.appendingPathComponent("coremldata.bin"),
            root.appendingPathComponent("model.mil"),
        ]
        return candidates.contains { fm.fileExists(atPath: $0.path) }
    }

    #if canImport(CoreAISpeech)
    private func ensureCoreAISpeechLoaded() async throws -> SpeechRecognitionModel {
        if let existing = loadedSpeechModel {
            return existing
        }
        let bundleURL = URL(fileURLWithPath: descriptionState.modelPath)
        guard FileManager.default.fileExists(atPath: bundleURL.path) else {
            throw ProviderExecutionError.checkpointNotFound(
                modelID: modelID,
                path: descriptionState.modelPath
            )
        }
        do {
            let model = try await SpeechRecognitionModel(resourcesAt: bundleURL)
            loadedSpeechModel = model
            descriptionState.isLoaded = true
            return model
        } catch {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Failed to load CoreAISpeech bundle at \(descriptionState.modelPath): \(error.localizedDescription)"
            )
        }
    }
    #endif

    /// Transcribes `wavData` on the ANE lane via either:
    /// 1. `CoreAISpeech.SpeechRecognitionModel` (`Whisper` / `Parakeet TDT` `.aimodel` bundles) when linked and configured, or
    /// 2. Apple's system `SpeechAnalyzer` + `SpeechTranscriber` (`Speech.framework`) when `isSystemSpeechProvider == true`.
    ///
    /// In production (`allowSyntheticFallback == false`), ALWAYS throws `ProviderExecutionError`
    /// when the requested ASR runtime is missing — never fabricates or echoes WAV marker bytes.
    public func transcribeAudio(
        wavData: Data,
        languageHint: String = "en"
    ) async throws -> ANESpeechTranscriptionResult {
        // 1. Tier B: Custom CoreAISpeech `.aimodel` bundle (`Whisper` or `Parakeet TDT`)
        #if canImport(CoreAISpeech)
        if !isSystemSpeechProvider,
           bundleManifest?.isSpeechRecognizer == true,
           (!allowSyntheticFallback || Self.hasCompiledSpeechWeights(at: descriptionState.modelPath)) {
            let speechModel = try await ensureCoreAISpeechLoaded()
            let modelSampleRate = await speechModel.sampleRate
            let targetRate = max(8_000, Int(modelSampleRate.rounded()))
            let samples = await Self.decodeAudioMonoPCM(
                from: wavData,
                targetSampleRate: targetRate,
                stripSyntheticMarker: false
            )
            let stats = Self.computeAudioStats(samples: samples, sampleRate: targetRate)
            guard !samples.isEmpty else {
                return ANESpeechTranscriptionResult(
                    text: "",
                    durationSeconds: 0,
                    sampleRateHz: targetRate,
                    rmsEnergy: 0,
                    melFrames: 0,
                    computePlacement: descriptionState.computePlacement.value,
                    backend: "coreai_speech",
                    modelID: modelID
                )
            }
            do {
                let (text, _) = try await speechModel.transcribe(pcm: samples)
                return ANESpeechTranscriptionResult(
                    text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                    durationSeconds: stats.durationSeconds,
                    sampleRateHz: targetRate,
                    rmsEnergy: stats.rmsEnergy,
                    melFrames: stats.melFrames,
                    computePlacement: descriptionState.computePlacement.value,
                    backend: "coreai_speech",
                    modelID: modelID
                )
            } catch {
                throw ProviderExecutionError.runtimeUnavailable(
                    modelID: modelID,
                    detail: "CoreAISpeech transcription failed: \(error.localizedDescription)"
                )
            }
        }
        #endif

        // 2. Tier A: Out-of-the-Box System `SpeechAnalyzer` + `SpeechTranscriber` (`apple-speech-transcriber`)
        if isSystemSpeechProvider {
            if allowSyntheticFallback,
               let markerRange = wavData.range(of: Data("TEXT:".utf8)),
               markerRange.upperBound < wavData.endIndex {
                let samples = await Self.decodeAudioMonoPCM(
                    from: wavData,
                    targetSampleRate: 16_000,
                    stripSyntheticMarker: true
                )
                let stats = Self.computeAudioStats(samples: samples, sampleRate: 16_000)
                let tail = wavData.subdata(in: markerRange.upperBound..<wavData.endIndex)
                let extractedText = String(decoding: tail, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return ANESpeechTranscriptionResult(
                    text: extractedText,
                    durationSeconds: stats.durationSeconds,
                    sampleRateHz: 16_000,
                    rmsEnergy: stats.rmsEnergy,
                    melFrames: stats.melFrames,
                    computePlacement: .ane,
                    backend: "synthetic_fixture",
                    modelID: modelID
                )
            }
            return try await transcribeWithSystemSpeechAnalyzer(
                wavData: wavData,
                languageHint: languageHint
            )
        }

        // 3. Strict production gate for explicit non-system paths when CoreAISpeech is absent or bundle is missing
        guard allowSyntheticFallback else {
            if bundleManifest == nil && !FileManager.default.fileExists(atPath: descriptionState.modelPath) {
                throw ProviderExecutionError.checkpointNotFound(
                    modelID: modelID,
                    path: descriptionState.modelPath
                )
            }
            let detail: String
            if bundleManifest?.isSpeechRecognizer == true {
                detail = "Speech recognizer bundle found at \(descriptionState.modelPath), but CoreAISpeech is not linked (build with HEYLOOK_ENABLE_COREAI=1)."
            } else {
                detail = "Speech ASR model runtime is not loaded at \(descriptionState.modelPath)"
            }
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: detail
            )
        }

        // 4. Synthetic test-fixture mode (`allowSyntheticFallback == true` ONLY)
        let sampleRate = 16_000
        let samples = await Self.decodeAudioMonoPCM(
            from: wavData,
            targetSampleRate: sampleRate,
            stripSyntheticMarker: true
        )
        let stats = Self.computeAudioStats(samples: samples, sampleRate: sampleRate)

        if let markerRange = wavData.range(of: Data("TEXT:".utf8)),
           markerRange.upperBound < wavData.endIndex {
            let tail = wavData.subdata(in: markerRange.upperBound..<wavData.endIndex)
            let extractedText = String(decoding: tail, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ANESpeechTranscriptionResult(
                text: extractedText,
                durationSeconds: stats.durationSeconds,
                sampleRateHz: sampleRate,
                rmsEnergy: stats.rmsEnergy,
                melFrames: stats.melFrames,
                computePlacement: descriptionState.computePlacement.value,
                backend: "synthetic_fixture",
                modelID: modelID
            )
        }

        let formattedSecs = String(format: "%.2fs", stats.durationSeconds)
        let syntheticText = "[ANE Whisper (\(languageHint), \(formattedSecs), \(stats.melFrames) mel frames)] Hands-free voice prompt"
        return ANESpeechTranscriptionResult(
            text: syntheticText,
            durationSeconds: stats.durationSeconds,
            sampleRateHz: sampleRate,
            rmsEnergy: stats.rmsEnergy,
            melFrames: stats.melFrames,
            computePlacement: descriptionState.computePlacement.value,
            backend: "synthetic_fixture",
            modelID: modelID
        )
    }

    private func transcribeWithSystemSpeechAnalyzer(
        wavData: Data,
        languageHint: String
    ) async throws -> ANESpeechTranscriptionResult {
        guard SpeechTranscriber.isAvailable else {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Apple SpeechTranscriber is unavailable on this host."
            )
        }

        let sampleRate = 16_000
        let samples = await Self.decodeAudioMonoPCM(
            from: wavData,
            targetSampleRate: sampleRate,
            stripSyntheticMarker: false
        )
        let stats = Self.computeAudioStats(samples: samples, sampleRate: sampleRate)
        // Sub-10ms or empty audio clips carry no speech; return truthful empty transcript without error.
        guard samples.count >= 160 else {
            return ANESpeechTranscriptionResult(
                text: "",
                durationSeconds: stats.durationSeconds,
                sampleRateHz: sampleRate,
                rmsEnergy: stats.rmsEnergy,
                melFrames: stats.melFrames,
                computePlacement: .ane,
                backend: "apple_speech_transcriber",
                modelID: modelID
            )
        }

        let requestedLocale = Locale(identifier: languageHint)
        var resolvedLocale = await SpeechTranscriber.supportedLocale(equivalentTo: requestedLocale)
        if resolvedLocale == nil {
            let langPrefix = languageHint
                .split(whereSeparator: { $0 == "-" || $0 == "_" })
                .first
                .map(String.init)?
                .lowercased() ?? "en"
            let supported = await SpeechTranscriber.supportedLocales
            resolvedLocale = supported.first { loc in
                loc.identifier.lowercased().hasPrefix(langPrefix)
            }
        }
        guard let targetLocale = resolvedLocale else {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Locale '\(languageHint)' is not supported by Apple SpeechTranscriber."
            )
        }

        let transcriber = SpeechTranscriber(locale: targetLocale, preset: .transcription)
        let assetStatus = await AssetInventory.status(forModules: [transcriber])
        switch assetStatus {
        case .unsupported:
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "On-device SpeechTranscriber assets are unsupported for locale '\(targetLocale.identifier)'."
            )
        case .supported:
            do {
                if let installReq = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                    try await installReq.downloadAndInstall()
                }
            } catch {
                throw ProviderExecutionError.runtimeUnavailable(
                    modelID: modelID,
                    detail: "On-device SpeechTranscriber assets for '\(targetLocale.identifier)' are not installed (\(error.localizedDescription))."
                )
            }
        case .downloading, .installed:
            break
        @unknown default:
            break
        }

        let canonicalWAV = Self.encodeCanonicalPCM16WAV(samples: samples, sampleRate: sampleRate)
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("heylook-speech-analyzer-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try canonicalWAV.write(to: tempURL, options: .atomic)

        let audioFile = try AVAudioFile(forReading: tempURL)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let collectTask = Task<[String], Error> {
            var finals: [String] = []
            for try await result in transcriber.results {
                if result.isFinal {
                    let seg = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !seg.isEmpty {
                        finals.append(seg)
                    }
                }
            }
            return finals
        }

        do {
            if let lastTime = try await analyzer.analyzeSequence(from: audioFile) {
                try await analyzer.finalizeAndFinish(through: lastTime)
            } else {
                try await analyzer.finalizeAndFinishThroughEndOfInput()
            }
        } catch {
            await analyzer.cancelAndFinishNow()
            collectTask.cancel()
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "SpeechAnalyzer transcription failed: \(error.localizedDescription)"
            )
        }

        let segments = (try? await collectTask.value) ?? []
        descriptionState.isLoaded = true
        return ANESpeechTranscriptionResult(
            text: segments.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines),
            durationSeconds: stats.durationSeconds,
            sampleRateHz: sampleRate,
            rmsEnergy: stats.rmsEnergy,
            melFrames: stats.melFrames,
            computePlacement: .ane,
            backend: "apple_speech_transcriber",
            modelID: modelID
        )
    }

    private static func computeAudioStats(
        samples: [Float],
        sampleRate: Int
    ) -> (durationSeconds: Double, rmsEnergy: Float, melFrames: Int) {
        guard !samples.isEmpty, sampleRate > 0 else { return (0, 0, 0) }
        let duration = Double(samples.count) / Double(sampleRate)
        var meanSq: Float = 0
        vDSP_measqv(samples, 1, &meanSq, vDSP_Length(samples.count))
        let rms = meanSq.squareRoot()
        let melFramesCount = ANECoreMLRuntime.computeAudioLogMel(samples: samples).count
        return (duration, rms, melFramesCount)
    }

    /// Decodes arbitrary audio containers (`WAV` with `AVAudioRecorder` 4 KB `FLLR` alignment chunks,
    /// `24/44.1/48 kHz`, stereo, `M4A`, `CAF`, `FLAC`, `MP3`) into resampled mono `Float32` PCM at `targetSampleRate`.
    public static func decodeAudioMonoPCM(
        from audioData: Data,
        targetSampleRate: Int = 16_000,
        stripSyntheticMarker: Bool = false
    ) async -> [Float] {
        var cleanData = audioData
        if stripSyntheticMarker, let markerRange = cleanData.range(of: Data("TEXT:".utf8)) {
            cleanData = cleanData.subdata(in: 0..<markerRange.lowerBound)
        }
        guard !cleanData.isEmpty else { return [] }

        // 1. Fast, chunk-accurate RIFF/WAVE parser first when valid RIFF header is present (handles `FLLR`/`JUNK`/`LIST`,
        //    8/16/24/32-bit PCM & IEEE Float32, multi-channel stereo downmix, and arbitrary sample rates without disk I/O).
        if let riffSamples = decodeRIFFChunksMonoPCM(from: cleanData, targetSampleRate: targetSampleRate),
           !riffSamples.isEmpty {
            return riffSamples
        }

        // 2. Fallback to `ArtifactProcessor.decodeMonoPCM` via temporary file for compressed containers (`M4A`, `MP3`, `FLAC`, `CAF`).
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("heylook-audio-decode-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        do {
            try cleanData.write(to: tempURL, options: .atomic)
            let decoded = try await ArtifactProcessor.decodeMonoPCM(for: tempURL, sampleRate: targetSampleRate)
            if !decoded.isEmpty {
                return decoded
            }
        } catch {
            // Fall through cleanly
        }
        return []
    }

    /// Parses standard RIFF/WAVE containers by walking subchunks (`fmt `, `FLLR`, `JUNK`, `LIST`, `data`)
    /// rather than assuming a fixed 44-byte header offset.
    public static func decodeRIFFChunksMonoPCM(
        from data: Data,
        targetSampleRate: Int = 16_000
    ) -> [Float]? {
        guard data.count >= 12,
              String(decoding: data.prefix(4), as: UTF8.self) == "RIFF",
              String(decoding: data.subdata(in: 8..<12), as: UTF8.self) == "WAVE"
        else { return nil }

        func readUInt16LE(_ offset: Int) -> UInt16 {
            UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
        }
        func readUInt32LE(_ offset: Int) -> UInt32 {
            UInt32(data[offset])
                | (UInt32(data[offset + 1]) << 8)
                | (UInt32(data[offset + 2]) << 16)
                | (UInt32(data[offset + 3]) << 24)
        }

        var audioFormat: UInt16 = 1
        var channels: Int = 1
        var srcSampleRate: Int = 16_000
        var bitsPerSample: Int = 16
        var dataRange: Range<Int>?

        var cursor = 12
        while cursor + 8 <= data.count {
            let chunkID = String(decoding: data.subdata(in: cursor..<(cursor + 4)), as: UTF8.self)
            let rawChunkSize = Int(readUInt32LE(cursor + 4))
            let payloadStart = cursor + 8
            let clampedEnd = min(data.count, payloadStart + max(0, rawChunkSize))

            if chunkID == "fmt " && clampedEnd - payloadStart >= 16 {
                audioFormat = readUInt16LE(payloadStart)
                channels = max(1, Int(readUInt16LE(payloadStart + 2)))
                srcSampleRate = max(1_000, Int(readUInt32LE(payloadStart + 4)))
                bitsPerSample = max(8, Int(readUInt16LE(payloadStart + 14)))
                // WAVE_FORMAT_EXTENSIBLE (0xFFFE) stores subformat code at byte 24..26
                if audioFormat == 0xFFFE && clampedEnd - payloadStart >= 26 {
                    audioFormat = readUInt16LE(payloadStart + 24)
                }
            } else if chunkID == "data" {
                dataRange = payloadStart..<clampedEnd
                break
            }

            // RIFF pads odd-sized chunks with 1 alignment byte
            let paddedSize = rawChunkSize + (rawChunkSize % 2)
            guard paddedSize >= 0, payloadStart + paddedSize > cursor else { break }
            cursor = payloadStart + paddedSize
        }

        // Legacy synthetic fixture fallback if a truncated test buffer had RIFF/WAVE without a well-formed `data` chunk
        let resolvedRange: Range<Int>
        if let dataRange {
            resolvedRange = dataRange
        } else if data.count > 44 {
            resolvedRange = 44..<data.count
        } else {
            return []
        }

        let bytesPerSample = max(1, bitsPerSample / 8)
        let frameStride = bytesPerSample * channels
        let frameCount = resolvedRange.count / frameStride
        guard frameCount > 0 else { return [] }

        // Only decode uncompressed Linear PCM (1) or IEEE Float (3); defer compressed codecs to AVAudioFile
        guard audioFormat == 1 || audioFormat == 3 else { return nil }

        var mono = [Float]()
        mono.reserveCapacity(frameCount)
        for frame in 0..<frameCount {
            let frameBase = resolvedRange.lowerBound + frame * frameStride
            var accum: Float = 0
            for ch in 0..<channels {
                let off = frameBase + ch * bytesPerSample
                let sample: Float
                if audioFormat == 3 && bytesPerSample == 4 {
                    let bits = readUInt32LE(off)
                    sample = Float(bitPattern: bits)
                } else if bytesPerSample == 1 {
                    sample = (Float(data[off]) - 128.0) / 128.0
                } else if bytesPerSample == 2 {
                    let s16 = Int16(bitPattern: readUInt16LE(off))
                    sample = Float(s16) / 32768.0
                } else if bytesPerSample == 3 {
                    let b0 = Int32(data[off])
                    let b1 = Int32(data[off + 1]) << 8
                    let b2 = Int32(Int8(bitPattern: data[off + 2])) << 16
                    sample = Float(b0 | b1 | b2) / 8_388_608.0
                } else if bytesPerSample == 4 {
                    let s32 = Int32(bitPattern: readUInt32LE(off))
                    sample = Float(s32) / 2_147_483_648.0
                } else {
                    sample = 0
                }
                accum += sample
            }
            mono.append(accum / Float(channels))
        }

        if abs(Double(srcSampleRate - targetSampleRate)) < 0.5 {
            return mono
        }
        return ArtifactProcessor.resampleMonoFloatPCM(
            samples: mono,
            sourceSampleRate: Double(srcSampleRate),
            targetSampleRate: Double(max(1_000, targetSampleRate))
        )
    }

    /// Encodes mono `Float32` samples (`[-1, 1]`) into a canonical 44-byte 16-bit PCM RIFF WAV container.
    public static func encodeCanonicalPCM16WAV(samples: [Float], sampleRate: Int = 16_000) -> Data {
        let clampedRate = UInt32(max(1_000, sampleRate))
        let dataBytesCount = UInt32(samples.count * 2)
        let riffChunkSize = 36 + dataBytesCount
        let byteRate = clampedRate * 2

        var out = Data()
        out.reserveCapacity(44 + Int(dataBytesCount))

        func appendU16(_ v: UInt16) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { out.append(contentsOf: $0) }
        }
        func appendU32(_ v: UInt32) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { out.append(contentsOf: $0) }
        }

        out.append(contentsOf: "RIFF".utf8)
        appendU32(riffChunkSize)
        out.append(contentsOf: "WAVE".utf8)
        out.append(contentsOf: "fmt ".utf8)
        appendU32(16)
        appendU16(1) // PCM
        appendU16(1) // Mono
        appendU32(clampedRate)
        appendU32(byteRate)
        appendU16(2) // Block align
        appendU16(16) // Bits per sample
        out.append(contentsOf: "data".utf8)
        appendU32(dataBytesCount)

        for s in samples {
            let clamped = max(-1.0, min(1.0, s))
            let i16 = Int16((clamped * 32767.0).rounded())
            appendU16(UInt16(bitPattern: i16))
        }
        return out
    }
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
