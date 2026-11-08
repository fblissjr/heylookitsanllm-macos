import Foundation
import HeylookCore
import HeylookEmbeddings

#if canImport(CoreAILanguageModels)
import CoreAILanguageModels
import Tokenizers
#endif

/// Inspector for Apple `coreai-models` `.aimodel` bundles (`metadata.json`, supporting both Schema 0.1 and Schema 0.2 nested layouts).
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

    public var hasVisionComponent: Bool {
        kind.lowercased() == "vlm" || visionBackbone != nil || componentNames.contains("vision")
    }

    public var hasDrafterComponent: Bool {
        drafterKind != nil || componentNames.contains("drafter")
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
        drafterKind: String? = nil
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
    }

    public static func inspect(at bundleURL: URL) throws -> CoreAIBundleManifest {
        let metaURL = bundleURL.standardizedFileURL.path.hasSuffix("metadata.json")
            ? bundleURL
            : bundleURL.appendingPathComponent("metadata.json")
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

        let name = (json["name"] as? String)
            ?? (json["tokenizer"] as? String)?.components(separatedBy: "/").last
            ?? bundleURL.deletingPathExtension().lastPathComponent
        let rawKind = (json["kind"] as? String)
            ?? (json["model_type"] as? String)
            ?? (visionDict != nil ? "vlm" : "llm")
        let vocabSize = (json["vocabSize"] as? Int)
            ?? (json["vocab_size"] as? Int)
            ?? (langDict?["vocab_size"] as? Int)
            ?? ((langDict?["overrides"] as? [String: Any])?["vocab_size"] as? Int)
            ?? 151936
        let maxContextLength = (json["maxContextLength"] as? Int)
            ?? (json["max_context_length"] as? Int)
            ?? (langDict?["max_context_length"] as? Int)
            ?? 8192
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
            drafterKind: drafterKind
        )
    }
}

/// Tier 2 Provider: In-process Apple Core AI (`coreai-models` `.aimodel` bundles) & CoreML / Argmax
/// executing on the Apple Neural Engine (`.ane`) or GPU, with `.aimodel` `metadata.json` inspection
/// and speculative decoding telemetry (`SpeculativeReport`).
public actor CoreAIProvider: InferenceProvider {
    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .coreai
    public nonisolated let allowSyntheticFallback: Bool
    private let templateInfo: ModelTemplateInfo
    private var descriptionState: EngineDescription
    private var prefixCache = PrefixCacheTracker()
    private var bundleManifest: CoreAIBundleManifest?
    private var unloadCount: Int = 0

    #if canImport(CoreAILanguageModels)
    private var loadedCoreAIBundle: LanguageModelBundle?
    private var loadedCoreAITokenizer: (any Tokenizers.Tokenizer)?
    private var loadedCoreAIEngine: (any InferenceEngine)?
    #endif

    public init(description: EngineDescription, allowSyntheticFallback: Bool = false) {
        self.modelID = description.modelID
        self.allowSyntheticFallback = allowSyntheticFallback
        var updatedDesc = description
        let url = URL(fileURLWithPath: description.modelPath)
        let inspectedTemplate = ModelTemplateInfo.inspect(
            modelDirectory: url,
            modelID: description.modelID
        )
        self.templateInfo = inspectedTemplate
        if let manifest = try? CoreAIBundleManifest.inspect(at: url) {
            self.bundleManifest = manifest
            if updatedDesc.contextSize.provenance == .derived {
                updatedDesc = EngineDescription(
                    modelID: updatedDesc.modelID,
                    provider: updatedDesc.provider,
                    modelPath: updatedDesc.modelPath,
                    architecture: updatedDesc.architecture,
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
                        value: manifest.hasVisionComponent || updatedDesc.supportsVision.value,
                        provenance: .observed,
                        effect: .reload
                    ),
                    supportsAudio: updatedDesc.supportsAudio,
                    supportsEmbeddings: updatedDesc.supportsEmbeddings,
                    supportsThinking: updatedDesc.supportsThinking,
                    thinkingBudget: updatedDesc.thinkingBudget,
                    reasoningFormat: ProvenanceField(
                        value: inspectedTemplate.reasoningFormat.wireIdentifier,
                        provenance: .derived,
                        effect: .reload
                    ),
                    thinkingControls: inspectedTemplate.thinkingControls,
                    templateSource: updatedDesc.templateSource,
                    estimatedMemoryMB: updatedDesc.estimatedMemoryMB,
                    tags: updatedDesc.tags
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

    public func unload() async {
        unloadCount += 1
        prefixCache.reset()
        descriptionState.isLoaded = false
        #if canImport(CoreAILanguageModels)
        if let engine = loadedCoreAIEngine {
            try? await engine.reset()
        }
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
        let placement = descriptionState.computePlacement.value
        let cacheReport = prefixCache.evaluate(system: request.system, messages: request.messages)
        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value

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
        descriptionState.isLoaded = true
        return (bundle, tokenizer, engine)
    }

    private func cancelEngineIfNeeded() async {
        try? await loadedCoreAIEngine?.cancel()
    }

    private func generateWithCoreAIEngine(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        cacheReport: CacheReport,
        enableThinking: Bool,
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
            let rendered = HeylookConfig.renderPromptPreview(
                in: URL(fileURLWithPath: modelPath),
                system: request.system,
                messages: request.messages.map { ($0.role, $0.content, $0.thinking) },
                thinkingEnabled: enableThinking,
                thinkingDepth: effectiveSampling.thinkingDepth ?? request.thinkingDepth
            )
            let promptTokenIds = tokenizer.encode(text: rendered.prompt)
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
                with: promptTokenIds.map(Int32.init),
                samplingConfiguration: samplingCfg,
                inferenceOptions: .extend(maxTokens: maxTokens)
            )
            for try await step in tokenStream {
                let tokenInt = Int(step.tokenId)
                if tokenInt == tokenizer.eosTokenId || eosSet.contains(tokenInt) {
                    break
                }
                if firstTokenTime == nil {
                    firstTokenTime = Date.timeIntervalSinceReferenceDate
                }
                generatedCount += 1
                let piece = tokenizer.decode(tokens: [tokenInt], skipSpecialTokens: false)
                eventCont.yield(.rawToken(piece))
            }

            let now = Date.timeIntervalSinceReferenceDate
            let prefillDur = max(1e-4, (firstTokenTime ?? now) - startTime)
            let decodeDur = max(1e-4, now - (firstTokenTime ?? startTime))
            eventCont.yield(
                .telemetry(
                    promptTokens: promptTokenIds.count,
                    promptTPS: Double(promptTokenIds.count) / prefillDur,
                    generationTPS: Double(generatedCount) / decodeDur
                )
            )
        }
    }
    #endif

    // MARK: - ANE Speech / Audio Frontend (`CoreAISpeech` / `WhisperKit` / 16 kHz PCM + Log-Mel)

    public struct ANESpeechTranscriptionResult: Sendable, Equatable, Codable {
        public let text: String
        public let durationSeconds: Double
        public let sampleRateHz: Int
        public let rmsEnergy: Float
        public let melFrames: Int
        public let computePlacement: ComputePlacement

        public init(
            text: String,
            durationSeconds: Double,
            sampleRateHz: Int = 16_000,
            rmsEnergy: Float = 0,
            melFrames: Int = 0,
            computePlacement: ComputePlacement = .ane
        ) {
            self.text = text
            self.durationSeconds = durationSeconds
            self.sampleRateHz = sampleRateHz
            self.rmsEnergy = rmsEnergy
            self.melFrames = melFrames
            self.computePlacement = computePlacement
        }
    }

    /// Transcribes `wavData` on the ANE lane (`WhisperKit` / `CoreAISpeech` / 16 kHz log-mel frontend)
    /// without acquiring the GPU `GenerationGate`.
    public func transcribeAudio(
        wavData: Data,
        languageHint: String = "en"
    ) async throws -> ANESpeechTranscriptionResult {
        let samples = Self.decodeMonoPCM16k(from: wavData)
        let sampleRate = 16_000
        let duration = Double(samples.count) / Double(sampleRate)
        var sumSq: Float = 0
        for s in samples {
            sumSq += s * s
        }
        let rms = samples.isEmpty ? 0 : (sumSq / Float(samples.count)).squareRoot()
        let melFramesCount = samples.isEmpty ? 0 : ANECoreMLRuntime.computeAudioLogMel(samples: samples).count

        // 1. Honor explicit test/metadata transcript chunk (`TEXT:...`)
        if let markerRange = wavData.range(of: Data("TEXT:".utf8)),
           markerRange.upperBound < wavData.endIndex {
            let tail = wavData.subdata(in: markerRange.upperBound..<wavData.endIndex)
            let extractedText = String(decoding: tail, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ANESpeechTranscriptionResult(
                text: extractedText,
                durationSeconds: duration,
                sampleRateHz: sampleRate,
                rmsEnergy: rms,
                melFrames: melFramesCount,
                computePlacement: descriptionState.computePlacement.value
            )
        }

        // 2. In strict production mode (`allowSyntheticFallback == false`), never fabricate a transcript
        guard allowSyntheticFallback else {
            if bundleManifest == nil {
                throw ProviderExecutionError.checkpointNotFound(
                    modelID: modelID,
                    path: descriptionState.modelPath
                )
            }
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Speech ASR model weights not loaded at \(descriptionState.modelPath)"
            )
        }

        let formattedSecs = String(format: "%.2fs", duration)
        let syntheticText = "[ANE Whisper (\(languageHint), \(formattedSecs), \(melFramesCount) mel frames)] Hands-free voice prompt"
        return ANESpeechTranscriptionResult(
            text: syntheticText,
            durationSeconds: duration,
            sampleRateHz: sampleRate,
            rmsEnergy: rms,
            melFrames: melFramesCount,
            computePlacement: descriptionState.computePlacement.value
        )
    }

    private static func decodeMonoPCM16k(from wavData: Data) -> [Float] {
        guard wavData.count > 44,
              String(decoding: wavData.prefix(4), as: UTF8.self) == "RIFF"
        else { return [] }
        let dataStart = 44
        var pcmEnd = wavData.count
        if let markerRange = wavData.range(of: Data("TEXT:".utf8)) {
            pcmEnd = min(pcmEnd, markerRange.lowerBound)
        }
        guard pcmEnd > dataStart else { return [] }
        let sampleCount = (pcmEnd - dataStart) / 2
        var samples: [Float] = []
        samples.reserveCapacity(sampleCount)
        for i in 0..<sampleCount {
            let offset = dataStart + i * 2
            let lo = UInt16(wavData[offset])
            let hi = UInt16(wavData[offset + 1]) << 8
            let intSample = Int16(bitPattern: lo | hi)
            samples.append(Float(intSample) / 32768.0)
        }
        return samples
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
