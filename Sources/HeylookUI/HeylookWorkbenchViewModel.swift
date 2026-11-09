import AVFoundation
import Foundation
import Observation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

@MainActor
@Observable
public final class HeylookWorkbenchViewModel {
    public var activeSurface: HeylookSurface = .chat

    // Core dependencies
    public let store: SQLiteStore
    public let vectorStore: VectorStore
    public let scheduler: ComputeScheduler
    public let registry: RequestRegistry
    public let router: HeylookRouter
    public let server: HeylookHTTPServer
    public var config: HeylookConfig

    // Models & Engine State
    public var registeredModels: [EngineDescription] = [] {
        didSet { recomputeFilteredModels() }
    }
    public private(set) var filteredModels: [EngineDescription] = []
    public private(set) var modelCountsByProvider: [EngineProviderKind: Int] = [:]
    public var selectedModelID: String = ""
    public var editingSidecarModel: EngineDescription?
    public var showMLXConverterSheet: Bool = false
    public var convertingSourceModel: EngineDescription?
    public var memorySnapshot: HardwareMemoryMonitor.MemorySnapshot = HardwareMemoryMonitor.snapshot()
    public var schedulerMetrics: ComputeScheduler.SchedulerMetrics = ComputeScheduler.SchedulerMetrics(
        gpuGateBusy: false,
        gpuQueueDepth: 0,
        maxQueueDepth: 2,
        activeANETasks: 0,
        loadedGPUModels: [],
        loadedANEModels: [],
        pinnedModels: []
    )

    // Chat Workbench State
    public var conversations: [ConversationRecord] = []
    public var selectedConversationID: String?
    public var currentMessages: [MessageRecord] = []
    public var draftPrompt: String = ""
    public var draftSystemPrompt: String = ""
    public var draftAttachments: [MessagesRequest.MediaAttachment] = []
    public var isGenerating: Bool = false
    public var activeRequestID: String?
    public var streamingThinking: String = ""
    public var streamingContent: String = ""
    public var showPromptPreview: Bool = false
    public var showLoadPanel: Bool = false
    public var showInspector: Bool = false
    public var selectedProviderFilter: EngineProviderKind? = nil {
        didSet { recomputeFilteredModels() }
    }
    public var isRecordingVoice: Bool = false
    public var lastVoiceTranscription: CoreAIProvider.ANESpeechTranscriptionResult?
    private var voiceRecorder: AVAudioRecorder?
    private var voiceRecordingURL: URL?

    // Sampler & Thinking Override Controls
    public var presets: [PresetRecord] = []
    public var selectedPresetName: String = "Balanced Daily Driver"
    public var temperatureOverride: Double = 0.7
    public var topPOverride: Double = 0.95
    public var topKOverride: Int = 40
    public var minPOverride: Double = 0.05
    public var maxTokensOverride: Int = 2048
    public var thinkingEnabled: Bool = true
    public var thinkingBudgetOverride: Int = 4096
    public var thinkingDepth: String = "medium"
    public var seedOverride: String = ""
    public var userOverriddenSamplerKeys: Set<String> = []

    // Message Edit State
    public var editingMessageID: String?
    public var editingContent: String = ""

    // Notebook State
    public var notebooks: [NotebookRecord] = []
    public var selectedNotebookID: String?
    public var notebookTitle: String = "Scratchpad"
    public var notebookBody: String = "Once upon a time in unified memory, the GPU and the Neural Engine shared a 192 GB pool..."
    public var isNotebookGenerating: Bool = false

    // Embeddings & ANE Media Finder Studio State
    public var selectedBackend: EngineBackend = .nativeMLX
    public var modelComputeOptions: ModelComputeOptions = .hybridANETowers
    public var embeddingServerURLString: String = "http://127.0.0.1:8000"
    public var checkpointPath: String = NativeMLXEmbeddingGemma2.defaultCheckpointURL.path
    public var serverCapabilities: HeylookServerCapabilities?
    public var embeddingQuery: String = ""
    public var queryAttachmentURL: URL?
    public var selectedDimension: Int = Matryoshka.gemma2NativeDimensions
    public var selectedSampling: VideoWindowSampling = .fps
    public var loadAudioFromVideo: Bool = false
    public var selectedKindFilter: ArtifactKind?
    public var selectedTaskOverride: EmbeddingTask?
    public var searchHits: [SearchHit] = []
    public var selectedHit: SearchHit?
    public var selectedSpanIndex: Int = 0
    public var embeddingStatusMessage: String = "Ready"
    public var isEmbeddingBusy: Bool = false
    public var isIndexing: Bool = false
    public var artifactCount: Int = 0
    public var chunkCount: Int = 0
    public var countsByKind: [ArtifactKind: Int] = [:]
    public var watchedFolders: [VectorStore.WatchedFolder] = []
    public var showEvalLabDrawer: Bool = false
    public var compareLeftText: String = "Swift 6.2 actor isolation for unified memory inference on Apple Silicon"
    public var compareRightText: String = "Running MLX and CoreML ANE vision towers concurrently on Mac Studio"
    public var compareCosineScore: Float?
    public var aneInspectionReports: [ANETowerInspector.ComponentComputeReport] = []
    public var benchmarkCases: [EmbeddingLabRunner.BenchmarkCase] = [
        .init(query: "Apple Neural Engine co-residency", expectedArtifactID: "doc-vision"),
        .init(query: "evict 70B LLM background indexing", expectedArtifactID: "doc-scheduler"),
        .init(query: "Matryoshka 768d to 128d truncation", expectedArtifactID: "doc-towers"),
        .init(query: "heylook_saved SQLite WAL mobile lock screen", expectedArtifactID: "doc-sse")
    ]
    public var labMetrics: [EmbeddingLabRunner.DimensionMetrics] = []
    public var evalLabSuiteReport: EvalLabSuite.SuiteReport?
    public var isRunningEvalLab: Bool = false
    public var activeGGUFPorts: [String: Int] = [:]
    public var activeSubprocessCount: Int = 0

    public var hasActiveQuery: Bool {
        !embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || queryAttachmentURL != nil
    }

    public var effectiveQueryTask: EmbeddingTask {
        selectedTaskOverride ?? (selectedKindFilter == .code ? .codeRetrieval : .query)
    }

    public var activeQueryTowersLabel: String {
        let hasText = !embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let attachmentKind = queryAttachmentURL.flatMap { ArtifactKind.detect(from: $0) }
        switch (hasText, attachmentKind) {
        case (true, nil):
            return "Text Tower"
        case (true, .some(.image)):
            return "Text + Vision (Image <=280 tokens)"
        case (true, .some(.video)):
            return "Text + Vision (Video <=140 tokens/f)"
        case (true, .some(.audio)):
            return "Text + Audio (16 kHz Conformer)"
        case (true, .some(let k)):
            return "Text + \(k.rawValue.capitalized)"
        case (false, .some(.image)):
            return "Vision Tower (Image <=280 tokens)"
        case (false, .some(.video)):
            return "Vision Tower (Video <=140 tokens/f)"
        case (false, .some(.audio)):
            return "Audio Tower (16 kHz Conformer)"
        case (false, .some(let k)):
            return "Text Tower (\(k.rawValue.capitalized))"
        case (false, nil):
            return "Text / Vision / Audio"
        }
    }

    public var activeQueryPromptLabel: String {
        let hasText = !embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if !hasText, let url = queryAttachmentURL, let kind = ArtifactKind.detect(from: url), kind.isMedia {
            return "Pure Media (<bos><|media|><eos>)"
        }
        return effectiveQueryTask.prefixText.trimmingCharacters(in: .whitespaces)
    }

    public var activeWindowFeaturesLabel: String {
        let samplingText: String
        switch selectedSampling {
        case .fps: samplingText = "Video: Fixed 1 fps (32s)"
        case .content: samplingText = "Video: Content Novelty (32s)"
        case .off: samplingText = "Video: Whole Only"
        }
        return loadAudioFromVideo ? "\(samplingText) + 16 kHz Soundtrack" : samplingText
    }

    // Server & Bonjour Remote Connection State
    public var isServerRunning: Bool = false
    public var serverPort: UInt16 = 8000
    public var statusBanner: String?
    public var discoveredServers: [DiscoveredHeylookServer] = []
    public var isBrowsingBonjour: Bool = false
    public var remoteServerURLString: String = "http://mac-studio.local:8000"
    public var connectedRemoteServerURL: String?

    private let bonjourBrowser = BonjourServiceBrowser()
    private let hashEmbeddingEngine = DeterministicHashEmbeddingEngine()
    private var activeEmbeddingEngine: any EmbeddingEngine
    private var indexCoordinator: IndexCoordinator
    private var indexTask: Task<Void, Never>?
    #if os(macOS)
    private var folderWatcher: FolderWatcher?
    #endif

    public init() {
        let dbURL = HeylookConfig.defaultStoreURL
        let vecDbURL = HeylookConfig.defaultVectorStoreURL
        let cfg = HeylookConfig.loadDefault()
        self.config = cfg
        self.store = (try? SQLiteStore(databaseURL: dbURL))
            ?? (try! SQLiteStore(databaseURL: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("heylook_fallback.sqlite")))
        self.vectorStore = (try? VectorStore(databaseURL: vecDbURL))
            ?? (try! VectorStore(databaseURL: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("heylook_vec_fallback.sqlite")))
        self.scheduler = ComputeScheduler(maxLoadedGPUModels: cfg.maxLoadedModels, maxQueueDepth: cfg.maxQueueDepth)
        self.registry = RequestRegistry()
        let initialEngine = NativeMLXEmbeddingGemma2.makeBestAvailableEngine(computeOptions: .hybridANETowers)
        self.activeEmbeddingEngine = initialEngine
        self.indexCoordinator = IndexCoordinator(engine: initialEngine, store: self.vectorStore)
        self.router = HeylookRouter(
            store: self.store,
            scheduler: self.scheduler,
            registry: self.registry,
            embeddingEngine: initialEngine,
            bindHost: cfg.serverHost,
            allowedHosts: cfg.allowedHosts
        )
        self.server = HeylookHTTPServer(
            router: self.router,
            host: cfg.serverHost,
            port: UInt16(cfg.serverPort)
        )
        self.serverPort = UInt16(cfg.serverPort)

        Task {
            await bootstrap()
        }
    }

    public func bootstrap() async {
        let discovered = config.discoverConfiguredModels()
        await scheduler.registerDiscoveredModels(discovered)

        registeredModels = await scheduler.listModels()
        if let first = registeredModels.first {
            selectedModelID = first.modelID
        }

        setupEmbeddingCoordinator()
        refreshStoreData()
        await refreshMetrics()
    }

    /// Explicitly loads the synthetic demo workspace (`defaultDemoModels()` + sample corpus)
    /// when requested by the user or an offline preview/test.
    public func loadSampleDemoWorkspace() async {
        let demos = Self.defaultDemoModels()
        await scheduler.registerDiscoveredModels(demos, allowSyntheticFallback: true)
        registeredModels = await scheduler.listModels()
        if selectedModelID.isEmpty, let first = registeredModels.first {
            selectedModelID = first.modelID
        }
        await seedSampleCorpusIfNeeded()
        await refreshMetrics()
        statusBanner = "Loaded sample demo workspace (\(demos.count) synthetic models + 4 sample artifacts)."
    }

    public static func defaultDemoModels() -> [EngineDescription] {
        [
            EngineDescription(
                modelID: "gemma-4-26b-a4b-it-mlx",
                provider: .mlx,
                modelPath: "models/mlx/Gemma-4-26B-A4B-IT",
                architecture: ProvenanceField(value: "gemma4", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "4bit", provenance: .derived, effect: .reload),
                computePlacement: ProvenanceField(value: .hybridANEGPU, provenance: .configured, effect: .reload, computePlacement: .hybridANEGPU),
                isPinned: false,
                isLoaded: true,
                contextSize: ProvenanceField(value: 32768, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
                estimatedMemoryMB: 14800,
                tags: ["chat", "vision", "thinking"]
            ),
            EngineDescription(
                modelID: "qwen3.5-0.8b-mlx-8bit",
                provider: .mlx,
                modelPath: "models/mlx/Qwen3.5-0.8B-MLX-8bit",
                architecture: ProvenanceField(value: "qwen3_5", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "8bit", provenance: .derived, effect: .reload),
                computePlacement: ProvenanceField(value: .mlxGPU, provenance: .derived, effect: .reload, computePlacement: .mlxGPU),
                isPinned: false,
                isLoaded: false,
                contextSize: ProvenanceField(value: 32768, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: false, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
                estimatedMemoryMB: 980,
                tags: ["fast", "thinking_depth"]
            ),
            EngineDescription(
                modelID: "coreai-llama-3.2-3b-ane",
                provider: .coreai,
                modelPath: "models/coreai/Llama-3.2-3B.aimodel",
                architecture: ProvenanceField(value: "llama", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "int4-palettized", provenance: .derived, effect: .reload),
                computePlacement: ProvenanceField(value: .ane, provenance: .configured, effect: .reload, computePlacement: .ane),
                isPinned: true,
                isLoaded: true,
                contextSize: ProvenanceField(value: 8192, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: false, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
                estimatedMemoryMB: 2100,
                tags: ["ane", "coreai"]
            ),
            EngineDescription(
                modelID: "embeddinggemma-2-MLX-BF16",
                provider: .mlx,
                modelPath: "models/embeddings/embeddinggemma-2-MLX-BF16",
                architecture: ProvenanceField(value: "embedding_gemma2", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "bf16+ane", provenance: .configured, effect: .reload),
                computePlacement: ProvenanceField(value: .ane, provenance: .configured, effect: .reload, computePlacement: .ane),
                isPinned: true,
                isLoaded: true,
                contextSize: ProvenanceField(value: 8192, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsAudio: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsEmbeddings: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: false, provenance: .derived, effect: .perRequest),
                estimatedMemoryMB: 1488,
                tags: ["embeddings", "ane", "multimodal"]
            )
        ]
    }

    public func refreshStoreData() {
        do {
            conversations = try store.listConversations()
            presets = try store.listPresets()
            notebooks = try store.listNotebooks()

            if selectedConversationID == nil, let first = conversations.first {
                selectConversation(first.id)
            }
            if selectedNotebookID == nil, let firstNb = notebooks.first {
                selectedNotebookID = firstNb.id
                notebookTitle = firstNb.title
                notebookBody = firstNb.body
            }
        } catch {
            statusBanner = "Store error: \(error.localizedDescription)"
        }
    }

    public func refreshMetrics() async {
        memorySnapshot = HardwareMemoryMonitor.snapshot()
        schedulerMetrics = await scheduler.metrics()
        registeredModels = await scheduler.listModels()
        aneInspectionReports = ANETowerInspector.statusSnapshot(options: modelComputeOptions)
        activeGGUFPorts = GGUFPortPool.shared.activeAllocations()
        #if os(macOS)
        activeSubprocessCount = SubprocessLifecycleRegistry.shared.activeSubprocessCount
        #endif
    }

    // MARK: - Chat Operations

    public func createNewConversation() {
        do {
            let model = selectedModelID.isEmpty ? (registeredModels.first?.modelID ?? "") : selectedModelID
            let conv = ConversationRecord(
                title: "New Chat",
                systemPrompt: draftSystemPrompt,
                defaultModelID: model
            )
            try store.upsertConversation(conv)
            conversations = try store.listConversations()
            selectConversation(conv.id)
        } catch {
            statusBanner = "Could not create conversation: \(error.localizedDescription)"
        }
    }

    public func selectConversation(_ id: String) {
        selectedConversationID = id
        do {
            currentMessages = try store.listMessages(conversationID: id)
        } catch {
            statusBanner = "Failed to load messages: \(error.localizedDescription)"
        }
    }

    public func deleteConversation(_ id: String) {
        try? store.deleteConversation(id: id)
        conversations = (try? store.listConversations()) ?? []
        if selectedConversationID == id {
            selectedConversationID = conversations.first?.id
            if let next = selectedConversationID {
                selectConversation(next)
            } else {
                currentMessages = []
            }
        }
    }

    public func applyPreset(_ preset: PresetRecord) {
        selectedPresetName = preset.name
        userOverriddenSamplerKeys.removeAll()
        if let t = preset.settings.temperature { temperatureOverride = t }
        if let p = preset.settings.topP { topPOverride = p }
        if let k = preset.settings.topK { topKOverride = k }
        if let mp = preset.settings.minP { minPOverride = mp }
        if let m = preset.settings.maxTokens { maxTokensOverride = m }
        if let th = preset.settings.thinking { thinkingEnabled = th }
        if let tb = preset.settings.thinkingBudget { thinkingBudgetOverride = tb }
    }

    public func resetSamplerOverrides() {
        userOverriddenSamplerKeys.removeAll()
        if let activePreset = presets.first(where: { $0.name == selectedPresetName }) {
            applyPreset(activePreset)
        } else {
            temperatureOverride = SamplerSettings.builtinDefaults.temperature ?? 0.7
            topPOverride = SamplerSettings.builtinDefaults.topP ?? 0.95
            topKOverride = SamplerSettings.builtinDefaults.topK ?? 40
            minPOverride = SamplerSettings.builtinDefaults.minP ?? 0.05
            maxTokensOverride = SamplerSettings.builtinDefaults.maxTokens ?? 2048
            thinkingEnabled = SamplerSettings.builtinDefaults.thinking ?? true
            thinkingBudgetOverride = SamplerSettings.builtinDefaults.thinkingBudget ?? 4096
        }
    }

    // MARK: - KeyPath-Bindable Projections (swiftui-specialist / dataflow.md)

    public var boundTemperature: Double {
        get { temperatureOverride }
        set {
            temperatureOverride = newValue
            userOverriddenSamplerKeys.insert("temperature")
        }
    }

    public var boundTopP: Double {
        get { topPOverride }
        set {
            topPOverride = newValue
            userOverriddenSamplerKeys.insert("top_p")
        }
    }

    public var boundTopK: Int {
        get { topKOverride }
        set {
            topKOverride = newValue
            userOverriddenSamplerKeys.insert("top_k")
        }
    }

    public var boundMinP: Double {
        get { minPOverride }
        set {
            minPOverride = newValue
            userOverriddenSamplerKeys.insert("min_p")
        }
    }

    public var boundMaxTokens: Int {
        get { maxTokensOverride }
        set {
            maxTokensOverride = newValue
            userOverriddenSamplerKeys.insert("max_tokens")
        }
    }

    public var boundThinkingBudget: Int {
        get { thinkingBudgetOverride }
        set {
            thinkingBudgetOverride = newValue
            userOverriddenSamplerKeys.insert("thinking_budget")
        }
    }

    public var boundEffectiveQueryTask: EmbeddingTask {
        get { effectiveQueryTask }
        set {
            selectedTaskOverride = newValue
            Task { await runEmbeddingSearch() }
        }
    }

    public var boundComputePreset: ComputePreset {
        get { modelComputeOptions.preset }
        set { applyComputePreset(newValue) }
    }

    public subscript(computeUnitFor component: EmbeddingComponent) -> TowerComputeUnit {
        get { modelComputeOptions[component] }
        set {
            modelComputeOptions[component] = newValue
            setupEmbeddingCoordinator()
        }
    }

    public var activeModelDescription: EngineDescription? {
        registeredModels.first(where: { $0.modelID == selectedModelID }) ?? registeredModels.first
    }

    private func recomputeFilteredModels() {
        var counts: [EngineProviderKind: Int] = [:]
        for model in registeredModels {
            counts[model.provider, default: 0] += 1
        }
        modelCountsByProvider = counts
        if let provider = selectedProviderFilter {
            filteredModels = registeredModels.filter { $0.provider == provider }
        } else {
            filteredModels = registeredModels
        }
    }

    public var currentResolvedSamplers: SamplerCascade.ResolvedSampling {
        let activePreset = presets.first(where: { $0.name == selectedPresetName })?.settings
        let requestOverlay = SamplerSettings(
            temperature: userOverriddenSamplerKeys.contains("temperature") ? temperatureOverride : nil,
            topP: userOverriddenSamplerKeys.contains("top_p") ? topPOverride : nil,
            topK: userOverriddenSamplerKeys.contains("top_k") ? topKOverride : nil,
            minP: userOverriddenSamplerKeys.contains("min_p") ? minPOverride : nil,
            maxTokens: userOverriddenSamplerKeys.contains("max_tokens") ? maxTokensOverride : nil,
            thinking: userOverriddenSamplerKeys.contains("thinking") ? thinkingEnabled : nil,
            thinkingDepth: userOverriddenSamplerKeys.contains("thinking_depth") ? thinkingDepth : nil,
            thinkingBudget: userOverriddenSamplerKeys.contains("thinking_budget") ? thinkingBudgetOverride : nil
        )
        return HeylookConfig.resolveEffectiveSampling(
            modelPath: activeModelDescription?.modelPath,
            preset: activePreset,
            request: requestOverlay
        )
    }

    public func attachChatMedia(from urls: [URL]) {
        for url in urls {
            if let attachment = try? MessagesRequest.MediaAttachment.fromFileURL(url) {
                draftAttachments.append(attachment)
            }
        }
    }

    public func removeChatAttachment(id: String) {
        draftAttachments.removeAll { $0.id == id }
    }

    public var chatAttachmentCapabilityWarning: String? {
        guard let desc = activeModelDescription, !draftAttachments.isEmpty else { return nil }
        let hasImage = draftAttachments.contains { $0.kind == "image" }
        let hasAudio = draftAttachments.contains { $0.kind == "audio" }
        if hasImage && !desc.supportsVision.value {
            return "Model '\(desc.modelID)' does not support image inputs (supports_vision = false)."
        }
        if hasAudio && !desc.supportsAudio.value {
            return "Model '\(desc.modelID)' does not support audio inputs (supports_audio = false)."
        }
        return nil
    }

    /// Transcribes a recorded 16 kHz WAV utterance on the ANE lane (`CoreAIProvider.transcribeAudio`)
    /// without acquiring the Metal GPU `GenerationGate`, and dispatches the resulting prompt turn.
    @discardableResult
    public func sendVoiceTurn(
        wavData: Data,
        languageHint: String = "en"
    ) async -> CoreAIProvider.ANESpeechTranscriptionResult? {
        let aneProvider = CoreAIProvider(
            description: EngineDescription(
                modelID: "ane-speech-whisper",
                provider: .coreai,
                modelPath: "models/ane-speech-whisper.aimodel",
                computePlacement: ProvenanceField(
                    value: .ane,
                    provenance: .derived,
                    effect: .reload,
                    computePlacement: .ane
                )
            )
        )
        do {
            let transcript = try await aneProvider.transcribeAudio(wavData: wavData, languageHint: languageHint)
            lastVoiceTranscription = transcript
            draftPrompt = transcript.text
            await sendChatPrompt()
            return transcript
        } catch {
            statusBanner = "ANE voice transcription failed: \(error.localizedDescription)"
            return nil
        }
    }

    public func toggleVoiceRecording() async {
        if isRecordingVoice {
            voiceRecorder?.stop()
            voiceRecorder = nil
            isRecordingVoice = false
            guard let url = voiceRecordingURL else { return }
            voiceRecordingURL = nil
            defer { try? FileManager.default.removeItem(at: url) }
            if let wavData = try? Data(contentsOf: url), wavData.count > 44 {
                _ = await sendVoiceTurn(wavData: wavData)
            }
            return
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("heylook-voice-\(UUID().uuidString).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16_000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false
        ]
        do {
            #if os(iOS)
            let session = AVAudioSession.sharedInstance()
            try? session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try? session.setActive(true)
            #endif
            let recorder = try AVAudioRecorder(url: tempURL, settings: settings)
            recorder.prepareToRecord()
            if recorder.record() {
                voiceRecordingURL = tempURL
                voiceRecorder = recorder
                isRecordingVoice = true
            }
        } catch {
            statusBanner = "Microphone capture failed: \(error.localizedDescription)"
        }
    }

    public func sendChatPrompt() async {
        let promptText = draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard (!promptText.isEmpty || !draftAttachments.isEmpty), !isGenerating else { return }
        let pendingAttachments = draftAttachments
        draftPrompt = ""
        draftAttachments = []

        if selectedConversationID == nil {
            createNewConversation()
        }
        guard let convID = selectedConversationID else { return }

        do {
            let nextPos = (currentMessages.map(\.position).max() ?? -1) + 1
            let fallbackAttachmentLabel = pendingAttachments.map { $0.filename ?? $0.mediaType }.joined(separator: ", ")
            let storedContent = promptText.isEmpty ? fallbackAttachmentLabel : promptText
            let userMsg = MessageRecord(
                conversationID: convID,
                position: nextPos,
                role: "user",
                content: storedContent,
                modelID: selectedModelID
            )
            try store.appendMessage(userMsg)
            currentMessages = try store.listMessages(conversationID: convID)
            if var conv = try store.getConversation(id: convID), conv.title == "New Chat" {
                conv.title = String(storedContent.prefix(38))
                try store.upsertConversation(conv)
                conversations = try store.listConversations()
            }
            await streamAssistantReply(for: convID, latestAttachments: pendingAttachments)
        } catch {
            statusBanner = "Chat error: \(error.localizedDescription)"
        }
    }

    public func streamAssistantReply(
        for convID: String,
        latestAttachments: [MessagesRequest.MediaAttachment] = []
    ) async {
        isGenerating = true
        streamingThinking = ""
        streamingContent = ""
        let reqID = "req_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        activeRequestID = reqID

        let modelID = selectedModelID.isEmpty ? (registeredModels.first?.modelID ?? "") : selectedModelID
        guard !modelID.isEmpty else {
            statusBanner = "No model selected or configured. Add scan_folders in heylook.toml or load the sample workspace."
            isGenerating = false
            activeRequestID = nil
            return
        }
        do {
            let provider = try await scheduler.provider(for: modelID)
            let lastIdx = currentMessages.indices.last
            let turns = currentMessages.enumerated().map { idx, msg in
                MessagesRequest.MessageTurn(
                    role: msg.role,
                    content: msg.content,
                    thinking: msg.thinking,
                    attachments: idx == lastIdx ? latestAttachments : []
                )
            }
            let resolved = currentResolvedSamplers
            let effective = resolved.effective
            let req = MessagesRequest(
                model: modelID,
                messages: turns,
                system: draftSystemPrompt.isEmpty ? nil : draftSystemPrompt,
                maxTokens: effective.maxTokens,
                temperature: effective.temperature,
                topP: effective.topP,
                topK: effective.topK,
                minP: effective.minP,
                stream: true,
                thinking: effective.thinking,
                thinkingDepth: activeModelDescription?.thinkingControls?.depthVariable != nil
                    ? (effective.thinkingDepth ?? thinkingDepth)
                    : nil,
                thinkingBudget: effective.thinkingBudget
            )
            await registry.register(requestID: reqID, conversationID: convID, modelID: modelID)

            let finalChunk = try await scheduler.withComputeLease(for: modelID) { [registry] () throws -> GenerationChunk in
                defer {
                    Task { await registry.finish(requestID: reqID) }
                }
                var lastChunk = GenerationChunk()
                let stream = await provider.generate(
                    requestID: reqID,
                    request: req,
                    effectiveSampling: effective,
                    registry: registry
                )
                for try await chunk in stream {
                    await MainActor.run {
                        self.streamingThinking.append(chunk.thinking)
                        self.streamingContent.append(chunk.text)
                    }
                    lastChunk = chunk
                }
                return lastChunk
            }

            let nextPos = (currentMessages.map(\.position).max() ?? -1) + 1
            let assistantMsg = MessageRecord(
                conversationID: convID,
                position: nextPos,
                role: "assistant",
                content: streamingContent,
                thinking: streamingThinking.isEmpty ? nil : streamingThinking,
                modelID: modelID,
                stopReason: finalChunk.stopReason ?? .endTurn,
                cacheReport: finalChunk.cacheReport,
                speculativeReport: finalChunk.speculativeReport,
                generationTPS: finalChunk.generationTPS,
                computePlacement: finalChunk.activeComputePlacement
            )
            try store.appendMessage(assistantMsg)
            currentMessages = try store.listMessages(conversationID: convID)
        } catch {
            statusBanner = "Generation failed: \(error.localizedDescription)"
        }

        isGenerating = false
        activeRequestID = nil
        streamingThinking = ""
        streamingContent = ""
        await refreshMetrics()
    }

    public func endThinkingNow() async {
        guard let reqID = activeRequestID else { return }
        _ = await registry.requestReasoningEnd(requestID: reqID)
    }

    public func cancelGeneration() async {
        guard let reqID = activeRequestID else { return }
        _ = await registry.abort(requestID: reqID)
    }

    public func saveAndContinueEditedMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID else { return }
        do {
            var updated = message
            updated.content = editingContent
            try store.appendMessage(updated)
            try store.deleteMessages(conversationID: convID, fromPosition: message.position + 1)
            editingMessageID = nil
            currentMessages = try store.listMessages(conversationID: convID)
            if message.role == "user" {
                await streamAssistantReply(for: convID)
            }
        } catch {
            statusBanner = "Failed to edit message: \(error.localizedDescription)"
        }
    }

    public func renderedPromptPreview() -> String {
        var turns: [(role: String, content: String, thinking: String?)] = currentMessages.map {
            (role: $0.role, content: $0.content, thinking: $0.thinking)
        }
        if !draftPrompt.isEmpty {
            turns.append((role: "user", content: draftPrompt, thinking: nil))
        }
        let dirURL = activeModelDescription.map { URL(fileURLWithPath: $0.modelPath) }
        return HeylookConfig.renderPromptPreview(
            in: dirURL,
            system: draftSystemPrompt.isEmpty ? nil : draftSystemPrompt,
            messages: turns,
            thinkingEnabled: thinkingEnabled,
            thinkingDepth: activeModelDescription?.thinkingControls?.depthVariable != nil ? thinkingDepth : nil
        ).prompt
    }

    // MARK: - Notebook Operations

    public func createNewNotebook() {
        do {
            let model = selectedModelID.isEmpty ? (registeredModels.first?.modelID ?? "") : selectedModelID
            let nb = NotebookRecord(
                title: "Untitled Notebook",
                body: "",
                modelID: model
            )
            try store.upsertNotebook(nb)
            notebooks = try store.listNotebooks()
            selectNotebook(nb.id)
        } catch {
            statusBanner = "Could not create notebook: \(error.localizedDescription)"
        }
    }

    public func selectNotebook(_ id: String) {
        selectedNotebookID = id
        if let nb = notebooks.first(where: { $0.id == id }) {
            notebookTitle = nb.title
            notebookBody = nb.body
        }
    }

    public func deleteNotebook(_ id: String) {
        try? store.deleteNotebook(id: id)
        notebooks = (try? store.listNotebooks()) ?? []
        if selectedNotebookID == id {
            if let first = notebooks.first {
                selectNotebook(first.id)
            } else {
                selectedNotebookID = nil
                notebookTitle = "Scratchpad"
                notebookBody = ""
            }
        }
    }

    public func saveNotebook() {
        do {
            let model = selectedModelID.isEmpty ? (registeredModels.first?.modelID ?? "") : selectedModelID
            let nb = NotebookRecord(
                id: selectedNotebookID ?? UUID().uuidString,
                title: notebookTitle,
                body: notebookBody,
                modelID: model
            )
            try store.upsertNotebook(nb)
            selectedNotebookID = nb.id
            notebooks = try store.listNotebooks()
        } catch {
            statusBanner = "Notebook save failed: \(error.localizedDescription)"
        }
    }

    public func continueNotebookGeneration() async {
        guard !isNotebookGenerating else { return }
        let modelID = selectedModelID.isEmpty ? (registeredModels.first?.modelID ?? "") : selectedModelID
        guard !modelID.isEmpty else {
            statusBanner = "No model selected for notebook generation."
            return
        }
        isNotebookGenerating = true
        let reqID = "nb_\(UUID().uuidString.prefix(8))"
        do {
            let provider = try await scheduler.provider(for: modelID)
            let desc = await provider.describe()
            let req = MessagesRequest(
                model: modelID,
                messages: [MessagesRequest.MessageTurn(role: "user", content: "Continue writing seamlessly from:\n\n\(notebookBody)")],
                maxTokens: 256,
                temperature: temperatureOverride,
                thinking: false
            )
            let resolved = HeylookConfig.resolveEffectiveSampling(
                modelPath: desc.modelPath,
                request: req.samplerOverlay
            )
            notebookBody += "\n\n"
            try await scheduler.withComputeLease(for: modelID) { [registry] in
                let stream = await provider.generate(
                    requestID: reqID,
                    request: req,
                    effectiveSampling: resolved.effective,
                    registry: registry
                )
                for try await chunk in stream {
                    await MainActor.run {
                        self.notebookBody.append(chunk.text)
                    }
                }
            }
            saveNotebook()
        } catch {
            statusBanner = "Notebook generation error: \(error.localizedDescription)"
        }
        isNotebookGenerating = false
    }

    public func saveNotebookToConversations() {
        do {
            let model = selectedModelID.isEmpty ? (registeredModels.first?.modelID ?? "") : selectedModelID
            let conv = ConversationRecord(
                title: "Notebook: \(notebookTitle)",
                systemPrompt: draftSystemPrompt,
                defaultModelID: model
            )
            try store.upsertConversation(conv)
            try store.appendMessage(
                MessageRecord(
                    conversationID: conv.id,
                    position: 0,
                    role: "user",
                    content: notebookBody,
                    modelID: model
                )
            )
            conversations = try store.listConversations()
            selectConversation(conv.id)
            activeSurface = .chat
        } catch {
            statusBanner = "Could not export notebook: \(error.localizedDescription)"
        }
    }

    // MARK: - Embeddings & ANE Media Finder Studio Operations

    public func setupEmbeddingCoordinator() {
        let dir = URL(fileURLWithPath: checkpointPath)
        let engine: any EmbeddingEngine
        switch selectedBackend {
        case .heylook:
            let baseURL = URL(string: embeddingServerURLString) ?? URL(string: "http://127.0.0.1:8000")!
            let heylook = HeylookMLXEngine(baseURL: baseURL)
            engine = heylook
            Task {
                self.serverCapabilities = try? await heylook.fetchCapabilities()
            }
        case .nativeMLX:
            engine = NativeMLXEmbeddingGemma2.makeBestAvailableEngine(
                checkpointURL: dir,
                computeOptions: modelComputeOptions
            )
        }
        self.activeEmbeddingEngine = engine
        self.indexCoordinator = IndexCoordinator(engine: engine, store: vectorStore)
        refreshEmbeddingCounts()
        refreshWatchedFolders()
        refreshComputeReports()
    }

    public func refreshComputeReports() {
        let dir = URL(fileURLWithPath: checkpointPath)
        aneInspectionReports = ANETowerInspector.statusSnapshot(
            options: modelComputeOptions,
            checkpointDirectory: dir
        )
        Task {
            let full = await ANETowerInspector.inspect(
                checkpointDirectory: dir,
                options: modelComputeOptions
            )
            if full.count == EmbeddingComponent.allCases.count {
                self.aneInspectionReports = full
            }
        }
    }

    public func applyComputePreset(_ preset: ComputePreset) {
        switch preset {
        case .allMLXGPU:
            modelComputeOptions = .allMLXGPU
        case .hybridANETowers:
            modelComputeOptions = .hybridANETowers
        case .allNeuralEngine:
            modelComputeOptions = .allNeuralEngine
        case .custom:
            break
        }
        setupEmbeddingCoordinator()
    }

    public func refreshEmbeddingCounts() {
        guard let stats = try? vectorStore.stats() else { return }
        artifactCount = stats.artifactCount
        chunkCount = stats.chunkCount
        countsByKind = stats.countsByKind
    }

    public func refreshWatchedFolders() {
        guard let folders = try? vectorStore.listWatchedFolders() else { return }
        watchedFolders = folders
        #if os(macOS)
        let folderURLs = folders.map(\.url)
        if folderWatcher == nil {
            folderWatcher = FolderWatcher { [weak self] changedDirs in
                Task { @MainActor [weak self] in
                    guard let self, !self.isIndexing else { return }
                    self.startIndexing(changedDirs, persistFolders: false)
                }
            }
        }
        folderWatcher?.watch(urls: folderURLs)
        #endif
    }

    public func removeWatchedFolder(_ folder: VectorStore.WatchedFolder) {
        try? vectorStore.removeWatchedFolder(folder.url)
        refreshWatchedFolders()
    }

    public func reindexWatchedFolders() {
        let urls = watchedFolders.map(\.url)
        guard !urls.isEmpty else { return }
        startIndexing(urls, persistFolders: false)
    }

    public func startIndexing(_ urls: [URL], persistFolders: Bool = true) {
        indexTask?.cancel()
        if persistFolders {
            for url in urls {
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
                    try? vectorStore.addWatchedFolder(url, windowSampling: selectedSampling)
                }
            }
            refreshWatchedFolders()
        }
        indexTask = Task {
            await indexURLs(urls)
        }
    }

    public func cancelIndexing() {
        indexTask?.cancel()
        indexTask = nil
        isIndexing = false
        isEmbeddingBusy = false
        embeddingStatusMessage = "Indexing cancelled."
    }

    public func indexURLs(_ urls: [URL]) async {
        isEmbeddingBusy = true
        isIndexing = true
        defer {
            isEmbeddingBusy = false
            isIndexing = false
            refreshEmbeddingCounts()
            refreshWatchedFolders()
        }

        do {
            let result = try await indexCoordinator.indexURLs(
                urls,
                videoWindowSampling: selectedSampling,
                loadAudioFromVideo: loadAudioFromVideo
            ) { progress in
                Task { @MainActor in
                    self.embeddingStatusMessage = "Indexing (\(progress.processed)/\(progress.total)): \(progress.currentTitle)"
                }
            }
            embeddingStatusMessage = "Indexed \(result.indexed) artifacts (\(result.skipped.count) skipped)."
        } catch is CancellationError {
            embeddingStatusMessage = "Indexing cancelled."
        } catch {
            embeddingStatusMessage = "Index error: \(error.localizedDescription)"
        }
    }

    public func clearEmbeddingIndex() {
        do {
            try vectorStore.clearAll()
            searchHits = []
            selectedHit = nil
            refreshEmbeddingCounts()
            embeddingStatusMessage = "Index cleared."
        } catch {
            embeddingStatusMessage = "Clear failed: \(error.localizedDescription)"
        }
    }

    public func seedSampleCorpusIfNeeded() async {
        setupEmbeddingCoordinator()
        guard ((try? vectorStore.stats().artifactCount) ?? 0) == 0 else {
            refreshEmbeddingCounts()
            return
        }

        let samples: [(String, ArtifactKind, String, String)] = [
            (
                "doc-vision",
                .text,
                "VISION.md — Native Apple Silicon Workbench",
                "Heylook is a single-user multimodal AI workbench for macOS 27.2 and iOS 27.2 unifying MLX Metal GPU generation with ANE CoreML co-residency."
            ),
            (
                "doc-scheduler",
                .code,
                "ComputeScheduler.swift — Dual-Lane GPU & ANE Co-Residency",
                "The ComputeScheduler separates heavy autoregressive GPU models from ANE-resident embedding and vision towers so background media indexing never evicts a 70B LLM."
            ),
            (
                "doc-towers",
                .code,
                "ModelComputeOptions.swift — Per-Tower Compute Unit Selection",
                "Assign Vision Tower and Audio Conformer to CPU + Apple Neural Engine while keeping Text Projection on GPU or ANE with Matryoshka 768d to 128d truncation."
            ),
            (
                "doc-sse",
                .code,
                "MessagesWire.swift — Disconnect-Resilient SSE Generation",
                "Mobile iOS clients can lock their screen mid-generation while the Mac Studio finishes writing the assistant turn to SQLite WAL and emits heylook_saved."
            )
        ]

        let batchInputs = samples.map { _, _, title, snippet in
            EmbeddingInput(parts: [.text(snippet)], task: .document, title: title)
        }
        let batchOutputs = try? await activeEmbeddingEngine.embedBatch(
            batchInputs,
            dimensions: Matryoshka.gemma2NativeDimensions
        )

        for (idx, (id, kind, title, snippet)) in samples.enumerated() {
            let artifact = Artifact(
                id: id,
                collectionID: "default",
                kind: kind,
                url: URL(fileURLWithPath: "/workspace/\(id).swift"),
                title: title,
                fingerprint: ContentFingerprint(byteCount: Int64(snippet.utf8.count), modifiedUnixMs: 1_700_000_000_000),
                modelID: activeEmbeddingEngine.modelID
            )
            let vec = (batchOutputs != nil && idx < batchOutputs!.count)
                ? batchOutputs![idx].vector
                : hashEmbeddingEngine.embed(text: snippet)
            let chunk = EmbeddedChunk(
                id: "\(id)#whole",
                artifactID: id,
                span: .whole,
                vector: vec,
                previewText: snippet
            )
            try? vectorStore.upsert(artifact: artifact, chunks: [chunk], modelID: activeEmbeddingEngine.modelID)
        }
        refreshEmbeddingCounts()
        embeddingQuery = "Apple Neural Engine co-residency"
        await runEmbeddingSearch()
        await runEmbeddingLabEvaluation()
    }

    public func runEmbeddingSearch() async {
        let q = embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty || queryAttachmentURL != nil else {
            searchHits = []
            selectedHit = nil
            return
        }
        isEmbeddingBusy = true
        embeddingStatusMessage = "Searching (\(selectedDimension)d)..."
        defer { isEmbeddingBusy = false }

        do {
            let kinds: Set<ArtifactKind>? = selectedKindFilter.map { [$0] }
            let results = try await indexCoordinator.search(
                queryText: q,
                exampleFileURL: queryAttachmentURL,
                dimensions: selectedDimension,
                kinds: kinds,
                taskOverride: selectedTaskOverride,
                limit: 30
            )
            searchHits = results
            selectedHit = results.first
            selectedSpanIndex = 0
            if !results.isEmpty {
                showInspector = true
            }
            embeddingStatusMessage = "Found \(results.count) results (\(selectedDimension)d Matryoshka slice)."
        } catch {
            embeddingStatusMessage = "Search failed: \(error.localizedDescription)"
            statusBanner = "Search failed: \(error.localizedDescription)"
        }
    }

    public func runPairwiseCompare() {
        let dim = selectedDimension
        let leftText = compareLeftText
        let rightText = compareRightText
        let engine = activeEmbeddingEngine
        let fallback = hashEmbeddingEngine
        Task { @MainActor in
            if let outputs = try? await engine.embedBatch(
                [.text(leftText, task: .sts), .text(rightText, task: .sts)],
                dimensions: dim
            ), outputs.count == 2 {
                self.compareCosineScore = Matryoshka.cosine(outputs[0].vector, outputs[1].vector, dimensions: dim)
            } else {
                let left = fallback.embed(text: leftText, dimensions: dim)
                let right = fallback.embed(text: rightText, dimensions: dim)
                self.compareCosineScore = Matryoshka.cosine(left, right, dimensions: dim)
            }
        }
    }

    public func runEmbeddingLabEvaluation() async {
        labMetrics = (try? await EmbeddingLabRunner.evaluate(coordinator: indexCoordinator, cases: benchmarkCases)) ?? []
    }

    public func sendHitToPairwiseCompare(_ hit: SearchHit) {
        let q = embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        compareLeftText = q.isEmpty ? hit.artifact.title : q
        compareRightText = hit.previewText ?? hit.artifact.title
        showEvalLabDrawer = true
        runPairwiseCompare()
    }

    public func pinHitToRetrievalLab(_ hit: SearchHit) async {
        let q = embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        let queryText = q.isEmpty ? hit.artifact.title : q
        if !benchmarkCases.contains(where: { $0.query == queryText && $0.expectedArtifactID == hit.artifact.id }) {
            benchmarkCases.append(.init(query: queryText, expectedArtifactID: hit.artifact.id))
        }
        showEvalLabDrawer = true
        await runEmbeddingLabEvaluation()
    }

    public func promoteComparePairToRetrievalLab() async {
        guard let hit = selectedHit ?? searchHits.first else { return }
        let queryText = compareLeftText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !queryText.isEmpty else { return }
        if !benchmarkCases.contains(where: { $0.query == queryText && $0.expectedArtifactID == hit.artifact.id }) {
            benchmarkCases.append(.init(query: queryText, expectedArtifactID: hit.artifact.id))
        }
        await runEmbeddingLabEvaluation()
    }

    @discardableResult
    public func runFullEvalLabSuite() async -> EvalLabSuite.SuiteReport? {
        isRunningEvalLab = true
        defer { isRunningEvalLab = false }
        guard let report = try? await EvalLabSuite.runAll() else {
            return nil
        }
        evalLabSuiteReport = report
        if !report.matryoshkaBenchmarks.isEmpty {
            labMetrics = report.matryoshkaBenchmarks
        }
        return report
    }

    // MARK: - Server Lifecycle & Bonjour Remote Discovery

    public func toggleServer() async {
        do {
            if isServerRunning {
                await server.stop()
                isServerRunning = false
                statusBanner = "Embedded server stopped."
            } else {
                try await server.start()
                isServerRunning = true
                statusBanner = "Embedded server listening on http://127.0.0.1:\(serverPort)"
            }
        } catch {
            statusBanner = "Server toggle error: \(error.localizedDescription)"
        }
    }

    public func startBonjourDiscovery() {
        isBrowsingBonjour = true
        bonjourBrowser.start { [weak self] servers in
            Task { @MainActor [weak self] in
                self?.discoveredServers = servers
            }
        }
    }

    public func stopBonjourDiscovery() {
        bonjourBrowser.stop()
        isBrowsingBonjour = false
    }

    public func connectToRemoteServer(_ urlString: String) async {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let baseURL = URL(string: trimmed) else {
            statusBanner = "Invalid remote server URL: \(urlString)"
            return
        }
        remoteServerURLString = trimmed
        embeddingServerURLString = trimmed
        connectedRemoteServerURL = trimmed

        // Query /v1/models on the remote Mac Studio if reachable; register RemoteHeylookProvider
        var remoteDescriptions: [EngineDescription] = []
        let modelsURL = baseURL.appendingPathComponent("v1/models")
        if let (data, resp) = try? await URLSession.shared.data(from: modelsURL),
           let http = resp as? HTTPURLResponse,
           http.statusCode == 200,
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let list = json["data"] as? [[String: Any]] {
            for item in list {
                if let id = (item["id"] as? String) ?? (item["modelID"] as? String) {
                    let remoteDesc = EngineDescription(
                        modelID: id,
                        provider: .remote,
                        modelPath: trimmed,
                        architecture: ProvenanceField(value: "remote-heylook", provenance: .observed),
                        quantization: ProvenanceField(value: "remote", provenance: .observed),
                        computePlacement: ProvenanceField(value: .remote, provenance: .observed),
                        contextSize: ProvenanceField(value: 32768, provenance: .observed),
                        flashAttention: ProvenanceField(value: true, provenance: .observed),
                        supportsVision: ProvenanceField(value: true, provenance: .observed),
                        supportsThinking: ProvenanceField(value: true, provenance: .observed),
                        tags: ["remote", "bonjour"]
                    )
                    remoteDescriptions.append(remoteDesc)
                }
            }
        }

        if remoteDescriptions.isEmpty {
            let fallbackDesc = EngineDescription(
                modelID: "remote-mac-studio-default",
                provider: .remote,
                modelPath: trimmed,
                architecture: ProvenanceField(value: "remote-heylook", provenance: .configured),
                quantization: ProvenanceField(value: "remote", provenance: .configured),
                computePlacement: ProvenanceField(value: .remote, provenance: .configured),
                contextSize: ProvenanceField(value: 32768, provenance: .configured),
                flashAttention: ProvenanceField(value: true, provenance: .configured),
                supportsVision: ProvenanceField(value: true, provenance: .configured),
                supportsThinking: ProvenanceField(value: true, provenance: .configured),
                tags: ["remote", "bonjour"]
            )
            remoteDescriptions = [fallbackDesc]
        }

        for desc in remoteDescriptions {
            let provider = RemoteHeylookProvider(baseURL: baseURL, description: desc)
            await scheduler.register(provider: provider, description: desc)
        }
        registeredModels = await scheduler.listModels()
        if let firstRemote = remoteDescriptions.first {
            selectedModelID = firstRemote.modelID
        }
        statusBanner = "Connected to remote Mac Studio at \(trimmed) (\(remoteDescriptions.count) models)."
    }

    // MARK: - Model Lifecycle & Sidecar Administration

    public func toggleModelPin(_ model: EngineDescription) async {
        await scheduler.setPinned(!model.isPinned, for: model.modelID)
        await refreshMetrics()
    }

    public func toggleModelLoaded(_ model: EngineDescription) async {
        if model.isLoaded {
            await scheduler.unloadModel(model.modelID)
        } else {
            _ = try? await scheduler.activateModel(model.modelID)
        }
        await refreshMetrics()
    }

    public func unloadAllModels() async {
        await scheduler.unloadAllModels()
        await activeEmbeddingEngine.unload()
        await refreshMetrics()
        statusBanner = "Unloaded all models, cleared Metal/ANE caches, and terminated background subprocesses."
    }

    public func rescanConfiguredModels() async {
        config = HeylookConfig.loadDefault()
        let discovered = config.discoverConfiguredModels()
        await scheduler.registerDiscoveredModels(discovered)
        registeredModels = await scheduler.listModels()
        if selectedModelID.isEmpty, let first = registeredModels.first {
            selectedModelID = first.modelID
        }
        await refreshMetrics()
        statusBanner = "Rescanned heylook.toml (\(registeredModels.count) models across \(config.scanFolders.count) scan folders)."
    }

    public func shutdown() async {
        indexTask?.cancel()
        indexTask = nil
        #if os(macOS)
        folderWatcher?.stop()
        #endif
        stopBonjourDiscovery()
        if isServerRunning {
            await server.stop()
            isServerRunning = false
        } else {
            await scheduler.unloadAllModels()
            await activeEmbeddingEngine.unload()
        }
        await refreshMetrics()
    }

    public func loadSidecarConfig(for model: EngineDescription) -> ModelSidecarConfig {
        let dirURL = URL(fileURLWithPath: model.modelPath)
        let sidecarURL = dirURL.appendingPathComponent("model.heylook.toml")
        if let text = try? String(contentsOf: sidecarURL, encoding: .utf8) {
            return ModelSidecarConfig.parseTOML(text)
        }
        return ModelSidecarConfig(
            id: model.modelID,
            provider: model.provider,
            pinned: model.isPinned,
            computePlacement: model.computePlacement.value,
            ctxSize: model.contextSize.value,
            flashAttn: model.flashAttention.value,
            supportsVision: model.supportsVision.value,
            supportsThinking: model.supportsThinking.value,
            sampling: SamplerSettings(
                temperature: 0.7,
                topP: 0.95,
                thinkingBudget: model.thinkingBudget.value
            ),
            tags: model.tags
        )
    }

    public func saveModelSidecar(_ sidecar: ModelSidecarConfig, for model: EngineDescription) async {
        let dirURL = URL(fileURLWithPath: model.modelPath)
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue {
            let sidecarURL = dirURL.appendingPathComponent("model.heylook.toml")
            try? sidecar.toTOML().write(to: sidecarURL, atomically: true, encoding: .utf8)
        }
        _ = try? await scheduler.applySidecarUpdate(sidecar, for: model.modelID)
        await refreshMetrics()
        statusBanner = "Saved model.heylook.toml sidecar for \(model.modelID)."
    }

    public func loadChatTemplateState(for model: EngineDescription) -> HeylookConfig.ChatTemplateState {
        let dirURL = URL(fileURLWithPath: model.modelPath)
        return HeylookConfig.inspectChatTemplate(
            in: dirURL,
            fallbackSource: model.templateSource.value
        )
    }

    @discardableResult
    public func saveChatTemplateOverride(_ template: String, for model: EngineDescription) async -> HeylookConfig.ChatTemplateState {
        let dirURL = URL(fileURLWithPath: model.modelPath)
        let state = (try? HeylookConfig.writeChatTemplateOverride(template, in: dirURL))
            ?? HeylookConfig.ChatTemplateState(source: "chat_template.heylook.jinja", hasOverride: true, template: template)
        let sidecar = loadSidecarConfig(for: model)
        _ = try? await scheduler.applySidecarUpdate(sidecar, for: model.modelID)
        await refreshMetrics()
        statusBanner = "Saved chat_template.heylook.jinja override for \(model.modelID)."
        return state
    }

    @discardableResult
    public func revertChatTemplateOverride(for model: EngineDescription) async -> HeylookConfig.ChatTemplateState {
        let dirURL = URL(fileURLWithPath: model.modelPath)
        let reverted = HeylookConfig.deleteChatTemplateOverride(
            in: dirURL,
            fallbackSource: model.templateSource.value
        )
        let sidecar = loadSidecarConfig(for: model)
        _ = try? await scheduler.applySidecarUpdate(sidecar, for: model.modelID)
        await refreshMetrics()
        statusBanner = "Reverted chat_template.heylook.jinja override for \(model.modelID) (active: \(reverted.source))."
        return reverted
    }
}
