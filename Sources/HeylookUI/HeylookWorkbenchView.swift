import AVFoundation
import AVKit
import Foundation
import Observation
import SwiftUI
import UniformTypeIdentifiers
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

#if os(macOS)
import AppKit
#endif

public enum HeylookSurface: String, CaseIterable, Identifiable, Sendable {
    case chat = "Chat"
    case notebook = "Notebook"
    case embeddings = "Embeddings & ANE"
    case models = "Models"
    case perf = "Perf & Server"

    public var id: String { rawValue }

    public var icon: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        case .notebook: return "doc.text"
        case .embeddings: return "sparkles.rectangle.stack"
        case .models: return "cpu"
        case .perf: return "gauge.with.dots.needle.67percent"
        }
    }
}

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
    public let config: HeylookConfig

    // Models & Engine State
    public var registeredModels: [EngineDescription] = []
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
        var discovered = config.discoverConfiguredModels()
        if discovered.isEmpty {
            discovered = Self.defaultDemoModels()
        }

        for desc in discovered {
            let provider: any InferenceProvider
            switch desc.provider {
            case .mlx:
                provider = MLXSwiftProvider(description: desc)
            case .coreai:
                provider = CoreAIProvider(description: desc)
            case .gguf:
                provider = GGUFSubprocessProvider(description: desc)
            case .remote:
                provider = MLXSwiftProvider(description: desc)
            }
            await scheduler.register(provider: provider, description: desc)
        }

        registeredModels = await scheduler.listModels()
        if let first = registeredModels.first {
            selectedModelID = first.modelID
        }

        refreshStoreData()
        await seedSampleCorpusIfNeeded()
        await refreshMetrics()
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
            let model = selectedModelID.isEmpty ? "gemma-4-26b-a4b-it-mlx" : selectedModelID
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

    public var activeModelDescription: EngineDescription? {
        registeredModels.first(where: { $0.modelID == selectedModelID }) ?? registeredModels.first
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
        guard let transcript = try? await aneProvider.transcribeAudio(wavData: wavData, languageHint: languageHint) else {
            return nil
        }
        lastVoiceTranscription = transcript
        draftPrompt = transcript.text
        await sendChatPrompt()
        return transcript
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
            let attachmentSummary = pendingAttachments.isEmpty
                ? ""
                : pendingAttachments.map { "[\($0.kind.uppercased()): \($0.filename ?? $0.mediaType)]" }.joined(separator: " ")
            let storedContent = [attachmentSummary, promptText]
                .filter { !$0.isEmpty }
                .joined(separator: "\n")
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

        let modelID = selectedModelID.isEmpty ? "gemma-4-26b-a4b-it-mlx" : selectedModelID
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
            let req = MessagesRequest(
                model: modelID,
                messages: turns,
                system: draftSystemPrompt.isEmpty ? nil : draftSystemPrompt,
                maxTokens: maxTokensOverride,
                temperature: temperatureOverride,
                topP: topPOverride,
                topK: topKOverride,
                minP: minPOverride,
                stream: true,
                thinking: thinkingEnabled,
                thinkingDepth: activeModelDescription?.thinkingControls?.depthVariable != nil ? thinkingDepth : nil,
                thinkingBudget: thinkingBudgetOverride
            )
            let resolved = currentResolvedSamplers
            await registry.register(requestID: reqID, conversationID: convID, modelID: modelID)

            let finalChunk = try await scheduler.withComputeLease(for: modelID) { [registry] () throws -> GenerationChunk in
                defer {
                    Task { await registry.finish(requestID: reqID) }
                }
                var lastChunk = GenerationChunk()
                let stream = await provider.generate(
                    requestID: reqID,
                    request: req,
                    effectiveSampling: resolved.effective,
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

    public func saveNotebook() {
        do {
            let model = selectedModelID.isEmpty ? "gemma-4-26b-a4b-it-mlx" : selectedModelID
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
        isNotebookGenerating = true
        let modelID = selectedModelID.isEmpty ? "gemma-4-26b-a4b-it-mlx" : selectedModelID
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
            let model = selectedModelID.isEmpty ? "gemma-4-26b-a4b-it-mlx" : selectedModelID
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
        let discovered = config.discoverConfiguredModels()
        for desc in discovered {
            if await scheduler.modelDescription(for: desc.modelID) == nil {
                let provider: any InferenceProvider
                switch desc.provider {
                case .mlx:
                    provider = MLXSwiftProvider(description: desc)
                case .coreai:
                    provider = CoreAIProvider(description: desc)
                case .gguf:
                    provider = GGUFSubprocessProvider(description: desc)
                case .remote:
                    provider = MLXSwiftProvider(description: desc)
                }
                await scheduler.register(provider: provider, description: desc)
            }
        }
        registeredModels = await scheduler.listModels()
        await refreshMetrics()
        statusBanner = "Rescanned model directories (\(registeredModels.count) models registered)."
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

// MARK: - Root Multiplatform Workbench View

public struct HeylookRootView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    public init(viewModel: HeylookWorkbenchViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        NavigationSplitView {
            List(HeylookSurface.allCases, selection: $viewModel.activeSurface) { surface in
                NavigationLink(value: surface) {
                    Label(surface.rawValue, systemImage: surface.icon)
                }
            }
            .navigationTitle("Heylook")
            .safeAreaInset(edge: .bottom) {
                VStack(alignment: .leading, spacing: 6) {
                    Divider()
                    HStack(spacing: 6) {
                        Circle()
                            .fill(viewModel.isServerRunning ? Color.green : Color.secondary.opacity(0.5))
                            .frame(width: 8, height: 8)
                        Text(viewModel.isServerRunning ? "Server :\(viewModel.serverPort)" : "Server Offline")
                            .font(.caption.monospaced())
                        Spacer()
                        Button(viewModel.isServerRunning ? "Stop" : "Serve") {
                            Task { await viewModel.toggleServer() }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    }
                    Text("\(viewModel.schedulerMetrics.loadedGPUModels.count) GPU • \(viewModel.schedulerMetrics.loadedANEModels.count) ANE")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(10)
            }
        } detail: {
            Group {
                switch viewModel.activeSurface {
                case .chat:
                    ChatWorkbenchSurface(viewModel: viewModel)
                case .notebook:
                    NotebookSurface(viewModel: viewModel)
                case .embeddings:
                    EmbeddingsStudioSurface(viewModel: viewModel)
                case .models:
                    ModelsInspectorSurface(viewModel: viewModel)
                case .perf:
                    PerfServerSurface(viewModel: viewModel)
                }
            }
            .toolbar {
                ToolbarItemGroup(placement: .automatic) {
                    Picker("Active Model", selection: $viewModel.selectedModelID) {
                        ForEach(viewModel.registeredModels) { model in
                            Text(model.modelID).tag(model.modelID)
                        }
                    }
                    .frame(maxWidth: 280)

                    Button {
                        viewModel.showLoadPanel.toggle()
                    } label: {
                        Label("Load & Samplers", systemImage: "slider.horizontal.3")
                    }
                }
            }
            .sheet(isPresented: $viewModel.showLoadPanel) {
                LoadAndSamplerPanelSheet(viewModel: viewModel)
            }
        }
    }
}

// MARK: - 1. Chat Workbench Surface

struct ChatWorkbenchSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        HSplitView {
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text("Conversations")
                        .font(.headline)
                    Spacer()
                    Button {
                        viewModel.createNewConversation()
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                    .buttonStyle(.borderless)
                    .keyboardShortcut("n", modifiers: [.command])
                    .help("New Conversation (⌘N)")
                }
                .padding(12)

                Divider()

                List(selection: $viewModel.selectedConversationID) {
                    ForEach(viewModel.conversations) { conv in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(conv.title)
                                .font(.subheadline.weight(.medium))
                                .lineLimit(1)
                            Text(conv.defaultModelID ?? "")
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                        }
                        .tag(Optional(conv.id))
                        .contextMenu {
                            Button("Delete", role: .destructive) {
                                viewModel.deleteConversation(conv.id)
                            }
                        }
                    }
                }
                .onChange(of: viewModel.selectedConversationID) { _, newValue in
                    if let id = newValue {
                        viewModel.selectConversation(id)
                    }
                }
            }
            .frame(minWidth: 210, idealWidth: 240, maxWidth: 300)

            VStack(spacing: 0) {
                HStack(spacing: 10) {
                    ForEach(viewModel.presets) { preset in
                        Button(preset.name) {
                            viewModel.applyPreset(preset)
                        }
                        .buttonStyle(.bordered)
                        .tint(viewModel.selectedPresetName == preset.name ? .accentColor : .secondary)
                        .controlSize(.small)
                    }

                    Divider().frame(height: 16)

                    Toggle("Thinking", isOn: $viewModel.thinkingEnabled)
                        .toggleStyle(.button)
                        .controlSize(.small)
                        .disabled(viewModel.activeModelDescription?.supportsThinking.value == false)

                    if let controls = viewModel.activeModelDescription?.thinkingControls,
                       !controls.depthValues.isEmpty {
                        Picker("Depth", selection: $viewModel.thinkingDepth) {
                            ForEach(controls.depthValues, id: \.self) { level in
                                Text(level).tag(level)
                            }
                        }
                        .pickerStyle(.segmented)
                        .controlSize(.small)
                        .frame(maxWidth: 210)
                        .onChange(of: viewModel.thinkingDepth) { _, _ in
                            viewModel.userOverriddenSamplerKeys.insert("thinking_depth")
                        }
                    }

                    Spacer()

                    Toggle("Prompt Preview", isOn: $viewModel.showPromptPreview)
                        .toggleStyle(.button)
                        .controlSize(.small)
                        .keyboardShortcut("p", modifiers: [.command, .shift])
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.bar)

                Divider()

                if viewModel.showPromptPreview {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Live Rendered Template Preview")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        ScrollView {
                            Text(viewModel.renderedPromptPreview())
                                .font(.system(.caption, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                        }
                        .frame(maxHeight: 130)
                    }
                    .padding(10)
                    .background(Color.secondary.opacity(0.08))
                    Divider()
                }

                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        ForEach(viewModel.currentMessages) { msg in
                            MessageBubbleView(message: msg, viewModel: viewModel)
                        }

                        if viewModel.isGenerating {
                            StreamingAssistantView(viewModel: viewModel)
                        }
                    }
                    .padding(16)
                }

                Divider()

                VStack(alignment: .leading, spacing: 8) {
                    if let warning = viewModel.chatAttachmentCapabilityWarning {
                        HStack(spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                            Text(warning)
                                .font(.caption)
                                .foregroundStyle(.orange)
                            Spacer()
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                    }

                    if !viewModel.draftAttachments.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(viewModel.draftAttachments) { att in
                                    HStack(spacing: 6) {
                                        Image(systemName: att.kind == "audio" ? "waveform" : "photo")
                                            .font(.caption)
                                            .foregroundStyle(.tint)
                                        Text(att.filename ?? att.mediaType)
                                            .font(.caption.weight(.medium))
                                            .lineLimit(1)
                                        Text(att.mediaType)
                                            .font(.caption2.monospaced())
                                            .foregroundStyle(.secondary)
                                        Text("~\(att.estimatedTokens) tok")
                                            .font(.caption2.monospaced().weight(.semibold))
                                            .foregroundStyle(.secondary)
                                            .padding(.horizontal, 5)
                                            .padding(.vertical, 2)
                                            .background(Color.secondary.opacity(0.12), in: Capsule())
                                        Button {
                                            viewModel.removeChatAttachment(id: att.id)
                                        } label: {
                                            Image(systemName: "xmark.circle.fill")
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                        .buttonStyle(.plain)
                                    }
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 5)
                                    .background(Color.accentColor.opacity(0.12), in: Capsule())
                                }
                            }
                        }
                    }

                    if viewModel.isRecordingVoice {
                        HStack(spacing: 6) {
                            Image(systemName: "waveform.badge.mic")
                                .foregroundStyle(.red)
                            Text("Recording 16 kHz mono WAV for ANE transcription — click Stop Mic to send")
                                .font(.caption.weight(.medium))
                                .foregroundStyle(.red)
                            Spacer()
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color.red.opacity(0.1), in: Capsule())
                    } else if let voiceMeta = viewModel.lastVoiceTranscription {
                        HStack(spacing: 6) {
                            Image(systemName: "waveform.circle.fill")
                                .foregroundStyle(.green)
                            Text(String(format: "ANE Voice (%@): %.1fs, %d mel frames", voiceMeta.computePlacement.label, voiceMeta.durationSeconds, voiceMeta.melFrames))
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button {
                                viewModel.lastVoiceTranscription = nil
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(Color.green.opacity(0.1), in: Capsule())
                    }

                    HStack(alignment: .bottom, spacing: 10) {
                        #if os(macOS)
                        Button {
                            pickChatAttachments()
                        } label: {
                            Image(systemName: "paperclip")
                                .font(.system(size: 14, weight: .semibold))
                        }
                        .buttonStyle(.bordered)
                        .help("Attach image or audio files")
                        #endif

                        Button {
                            Task { await viewModel.toggleVoiceRecording() }
                        } label: {
                            Image(systemName: viewModel.isRecordingVoice ? "stop.circle.fill" : "mic.fill")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(viewModel.isRecordingVoice ? .red : .primary)
                        }
                        .buttonStyle(.bordered)
                        .disabled(viewModel.isGenerating)
                        .help(viewModel.isRecordingVoice ? "Stop recording & transcribe on ANE" : "Record voice turn (16 kHz ANE transcription)")

                        TextField("Ask anything or drop images/audio (⌘↩ to send)...", text: $viewModel.draftPrompt, axis: .vertical)
                            .lineLimit(1...6)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit {
                                Task { await viewModel.sendChatPrompt() }
                            }

                        if viewModel.isGenerating {
                            Button("Stop", role: .destructive) {
                                Task { await viewModel.cancelGeneration() }
                            }
                            .buttonStyle(.borderedProminent)
                            .keyboardShortcut(.escape, modifiers: [])
                            .help("Cancel generation (Esc)")
                        } else {
                            Button("Send") {
                                Task { await viewModel.sendChatPrompt() }
                            }
                            .buttonStyle(.borderedProminent)
                            .keyboardShortcut(.return, modifiers: [.command])
                            .disabled(
                                viewModel.draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                    && viewModel.draftAttachments.isEmpty
                            )
                            .help("Send message (⌘↩)")
                        }
                    }
                }
                .padding(12)
                .onDrop(of: [.fileURL], isTargeted: nil) { providers in
                    handleDroppedChatFiles(providers)
                }
            }
        }
    }

    #if os(macOS)
    private func pickChatAttachments() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image, .audio]
        if panel.runModal() == .OK {
            viewModel.attachChatMedia(from: panel.urls)
        }
    }
    #endif

    private func handleDroppedChatFiles(_ providers: [NSItemProvider]) -> Bool {
        var handled = false
        for provider in providers where provider.canLoadObject(ofClass: URL.self) {
            handled = true
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url {
                    Task { @MainActor in
                        viewModel.attachChatMedia(from: [url])
                    }
                }
            }
        }
        return handled
    }
}

struct MessageBubbleView: View {
    let message: MessageRecord
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(message.role.uppercased())
                    .font(.caption.weight(.bold))
                    .foregroundStyle(message.role == "user" ? .blue : .purple)
                if let modelID = message.modelID {
                    Text(modelID)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
                if let placement = message.computePlacement {
                    Text(placement.label)
                        .font(.caption2.monospaced())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.green.opacity(0.14), in: Capsule())
                }
                if let cache = message.cacheReport {
                    Text(cache.summaryBadge)
                        .font(.caption2.monospaced())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.blue.opacity(0.12), in: Capsule())
                }
                if let spec = message.speculativeReport {
                    Text(spec.summaryBadge)
                        .font(.caption2.monospaced())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.purple.opacity(0.14), in: Capsule())
                }
                Spacer()
                if let tps = message.generationTPS, tps > 0 {
                    Text(String(format: "%.1f tok/s", tps))
                        .font(.caption2.monospaced())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.secondary.opacity(0.12), in: Capsule())
                }
                Button {
                    viewModel.editingMessageID = message.id
                    viewModel.editingContent = message.content
                } label: {
                    Image(systemName: "pencil")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
            }

            if let thinking = message.thinking, !thinking.isEmpty {
                DisclosureGroup("Reasoning Trace (\(thinking.count) chars)") {
                    Text(thinking)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 4)
                }
                .font(.caption.weight(.medium))
                .padding(8)
                .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
            }

            if viewModel.editingMessageID == message.id {
                VStack(alignment: .leading, spacing: 8) {
                    TextEditor(text: $viewModel.editingContent)
                        .font(.body)
                        .frame(minHeight: 80)
                    HStack {
                        Button("Save & Continue") {
                            Task { await viewModel.saveAndContinueEditedMessage(message) }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)

                        Button("Cancel") {
                            viewModel.editingMessageID = nil
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            } else {
                Text(message.content)
                    .font(.body)
                    .textSelection(.enabled)
            }
        }
        .padding(12)
        .background(
            message.role == "user"
                ? Color.blue.opacity(0.07)
                : Color.secondary.opacity(0.07),
            in: RoundedRectangle(cornerRadius: 12)
        )
    }
}

struct StreamingAssistantView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                ProgressView().controlSize(.small)
                Text("ASSISTANT STREAMING")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.purple)
                Spacer()
                Button("End thinking") {
                    Task { await viewModel.endThinkingNow() }
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }

            if !viewModel.streamingThinking.isEmpty {
                Text(viewModel.streamingThinking)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .padding(8)
                    .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }

            if !viewModel.streamingContent.isEmpty {
                Text(viewModel.streamingContent)
                    .font(.body)
            }
        }
        .padding(12)
        .background(Color.purple.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - 2. Notebook Surface

struct NotebookSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                TextField("Notebook Title", text: $viewModel.notebookTitle)
                    .font(.title3.weight(.semibold))
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 340)

                Spacer()

                Button("Continue Writing") {
                    Task { await viewModel.continueNotebookGeneration() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isNotebookGenerating)

                Button("Save Notebook") {
                    viewModel.saveNotebook()
                }
                .buttonStyle(.bordered)

                Button("Save to Conversations") {
                    viewModel.saveNotebookToConversations()
                }
                .buttonStyle(.bordered)
            }

            TextEditor(text: $viewModel.notebookBody)
                .font(.system(.body, design: .monospaced))
                .padding(8)
                .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
        }
        .padding(16)
    }
}

// MARK: - 3. Embeddings & ANE Media Finder Studio Surface

struct EmbeddingsStudioSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        #if os(macOS)
        HSplitView {
            EmbeddingsSidebarView(viewModel: viewModel)
                .frame(minWidth: 260, idealWidth: 290, maxWidth: 340)

            EmbeddingsSearchAndResultsView(viewModel: viewModel)
                .frame(minWidth: 420, idealWidth: 560)

            EmbeddingsInspectorView(
                hit: viewModel.selectedHit,
                selectedDimension: viewModel.selectedDimension,
                onSendToCompare: { hit in
                    viewModel.sendHitToPairwiseCompare(hit)
                },
                onPinToLab: { hit in
                    Task { await viewModel.pinHitToRetrievalLab(hit) }
                }
            )
            .frame(minWidth: 340, idealWidth: 400)
        }
        #else
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                EmbeddingsSidebarView(viewModel: viewModel)
                EmbeddingsSearchAndResultsView(viewModel: viewModel)
                EmbeddingsInspectorView(
                    hit: viewModel.selectedHit,
                    selectedDimension: viewModel.selectedDimension,
                    onSendToCompare: { hit in
                        viewModel.sendHitToPairwiseCompare(hit)
                    },
                    onPinToLab: { hit in
                        Task { await viewModel.pinHitToRetrievalLab(hit) }
                    }
                )
            }
            .padding(12)
        }
        #endif
    }
}

private struct EmbeddingsSidebarView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var isTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Image(systemName: "sparkle.magnifyingglass")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 1) {
                    Text("mbed Studio")
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                    Text("Multimodal ANE / GPU Finder")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Modality Filter
                    VStack(alignment: .leading, spacing: 4) {
                        Text("MODALITY FILTER")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 14)

                        modalityFilterButton(title: "All Artifacts", icon: "square.grid.2x2", kind: nil, count: viewModel.artifactCount)

                        ForEach(ArtifactKind.allCases) { kind in
                            modalityFilterButton(
                                title: kind.displayName,
                                icon: kind.systemImage,
                                kind: kind,
                                count: viewModel.countsByKind[kind] ?? 0
                            )
                        }
                    }

                    Divider()

                    // Engine & Compute Placement
                    VStack(alignment: .leading, spacing: 10) {
                        Text("ENGINE & COMPUTE PLACEMENT")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        Picker("Engine", selection: $viewModel.selectedBackend) {
                            ForEach(EngineBackend.allCases, id: \.self) { engine in
                                Text(engine.label).tag(engine)
                            }
                        }
                        .pickerStyle(.menu)
                        .onChange(of: viewModel.selectedBackend) {
                            viewModel.setupEmbeddingCoordinator()
                        }

                        Picker("Dimensions", selection: $viewModel.selectedDimension) {
                            Text("768d (Full)").tag(768)
                            Text("512d").tag(512)
                            Text("256d (Fast)").tag(256)
                            Text("128d (Compact)").tag(128)
                        }
                        .pickerStyle(.menu)
                        .onChange(of: viewModel.selectedDimension) {
                            Task { await viewModel.runEmbeddingSearch() }
                        }

                        Picker(
                            "Query Task",
                            selection: Binding(
                                get: { viewModel.effectiveQueryTask },
                                set: { newTask in
                                    viewModel.selectedTaskOverride = newTask
                                    Task { await viewModel.runEmbeddingSearch() }
                                }
                            )
                        ) {
                            Text("Retrieval Query (Asymmetric)").tag(EmbeddingTask.query)
                            Text("Semantic Similarity (Symmetric)").tag(EmbeddingTask.sts)
                            Text("Code Retrieval").tag(EmbeddingTask.codeRetrieval)
                            Text("Question Answering").tag(EmbeddingTask.questionAnswering)
                            Text("Fact Verification").tag(EmbeddingTask.factVerification)
                        }
                        .pickerStyle(.menu)

                        Picker("Video Sampling", selection: $viewModel.selectedSampling) {
                            ForEach(VideoWindowSampling.allCases, id: \.self) { mode in
                                Text(mode.label).tag(mode)
                            }
                        }
                        .pickerStyle(.menu)

                        Toggle("Include Video Soundtrack", isOn: $viewModel.loadAudioFromVideo)
                            .font(.caption)
                            .toggleStyle(.switch)
                            .controlSize(.small)

                        Divider()

                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Label("Per-Tower Compute", systemImage: "cpu.fill")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text(viewModel.modelComputeOptions.shortSummaryBadge)
                                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }

                            Picker(
                                "Preset",
                                selection: Binding(
                                    get: { viewModel.modelComputeOptions.preset },
                                    set: { viewModel.applyComputePreset($0) }
                                )
                            ) {
                                ForEach(ComputePreset.allCases, id: \.self) { preset in
                                    Text(preset.label).tag(preset)
                                }
                            }
                            .pickerStyle(.menu)
                            .controlSize(.small)

                            ForEach(EmbeddingComponent.allCases) { component in
                                HStack(spacing: 6) {
                                    Image(systemName: component.systemImage)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .frame(width: 14)
                                    Text(component.title)
                                        .font(.caption2.weight(.medium))
                                    Spacer()
                                    Picker(
                                        "",
                                        selection: Binding(
                                            get: { viewModel.modelComputeOptions[component] },
                                            set: { newUnit in
                                                viewModel.modelComputeOptions[component] = newUnit
                                                viewModel.setupEmbeddingCoordinator()
                                            }
                                        )
                                    ) {
                                        ForEach(TowerComputeUnit.allCases, id: \.self) { unit in
                                            Text(unit.label).tag(unit)
                                        }
                                    }
                                    .labelsHidden()
                                    .pickerStyle(.menu)
                                    .controlSize(.mini)
                                    .frame(width: 122)
                                }

                                if let report = viewModel.aneInspectionReports.first(where: { $0.component == component }) {
                                    HStack(spacing: 4) {
                                        Circle()
                                            .fill(report.isCompiledBundlePresent ? Color.green : Color.orange)
                                            .frame(width: 6, height: 6)
                                        Text(
                                            report.isCompiledBundlePresent
                                                ? "CoreML Ready (\(Int((report.aneResidencyFraction * 100).rounded()))% ANE ops)"
                                                : "CoreML missing — run scripts/export_ane_coreml.py"
                                        )
                                        .font(.system(size: 9))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                    }
                                    .padding(.leading, 20)
                                }
                            }

                            ComputeMemorySplitBar(split: viewModel.modelComputeOptions.memorySplitMB)
                                .padding(.top, 4)
                        }

                        if viewModel.selectedBackend == .heylook {
                            TextField("Server URL", text: $viewModel.embeddingServerURLString)
                                .textFieldStyle(.roundedBorder)
                                .font(.caption)
                                .onSubmit {
                                    viewModel.setupEmbeddingCoordinator()
                                }
                        } else if viewModel.selectedBackend == .nativeMLX {
                            TextField("Checkpoint Path", text: $viewModel.checkpointPath)
                                .textFieldStyle(.roundedBorder)
                                .font(.caption)
                                .onSubmit {
                                    viewModel.setupEmbeddingCoordinator()
                                }
                        }
                    }
                    .padding(.horizontal, 14)

                    Divider()

                    // Index Actions
                    VStack(alignment: .leading, spacing: 8) {
                        Text("INDEXING & EVALUATION")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        #if os(macOS)
                        Button {
                            openFilePickerForIndexing()
                        } label: {
                            Label("Add Files or Folders…", systemImage: "plus.rectangle.on.folder")
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.regular)
                        #endif

                        HStack(spacing: 6) {
                            Button {
                                viewModel.reindexWatchedFolders()
                            } label: {
                                Label("Re-scan", systemImage: "arrow.clockwise")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(viewModel.watchedFolders.isEmpty || viewModel.isIndexing)

                            Button(role: .destructive) {
                                viewModel.clearEmbeddingIndex()
                            } label: {
                                Label("Clear", systemImage: "trash")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(viewModel.artifactCount == 0 || viewModel.isIndexing)
                        }

                        Button {
                            viewModel.showEvalLabDrawer.toggle()
                        } label: {
                            Label(
                                viewModel.showEvalLabDrawer ? "Hide Pairwise & Eval Lab" : "Show Pairwise & Eval Lab",
                                systemImage: "chart.bar.xaxis"
                            )
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)

                        if !viewModel.watchedFolders.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Image(systemName: "eye.fill")
                                        .font(.caption2)
                                        .foregroundStyle(.green)
                                    Text("WATCHED FOLDERS (FSEVENTS)")
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundStyle(.secondary)
                                }
                                .padding(.top, 4)

                                ForEach(viewModel.watchedFolders) { folder in
                                    HStack(spacing: 4) {
                                        Image(systemName: "folder")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                        Text(folder.url.lastPathComponent)
                                            .font(.caption2)
                                            .lineLimit(1)
                                            .help(folder.path)
                                        Spacer()
                                        Button {
                                            viewModel.removeWatchedFolder(folder)
                                        } label: {
                                            Image(systemName: "xmark.circle.fill")
                                                .font(.caption2)
                                                .foregroundStyle(.tertiary)
                                        }
                                        .buttonStyle(.plain)
                                        .help("Stop watching \(folder.path)")
                                    }
                                }
                            }
                        }

                        Text("Drop folders, videos, audio, PDFs, images, or code here to index.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 14)
                    .padding(.bottom, 14)
                }
            }
        }
        .background(isTargeted ? Color.accentColor.opacity(0.12) : Color.secondary.opacity(0.04))
        .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
            handleDrop(providers: providers)
        }
    }

    private func modalityFilterButton(title: String, icon: String, kind: ArtifactKind?, count: Int) -> some View {
        let isSelected = viewModel.selectedKindFilter == kind
        return Button {
            viewModel.selectedKindFilter = kind
            Task { await viewModel.runEmbeddingSearch() }
        } label: {
            HStack {
                Image(systemName: icon)
                    .frame(width: 18)
                    .foregroundStyle(isSelected ? .white : .primary)
                Text(title)
                    .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? .white : .primary)
                Spacer()
                Text("\(count)")
                    .font(.caption2.monospacedDigit())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        Capsule()
                            .fill(isSelected ? Color.white.opacity(0.25) : Color.secondary.opacity(0.15))
                    )
                    .foregroundStyle(isSelected ? .white : .secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 7)
                    .fill(isSelected ? Color.accentColor : Color.clear)
            )
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 8)
    }

    #if os(macOS)
    private func openFilePickerForIndexing() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.prompt = "Index"
        if panel.runModal() == .OK {
            viewModel.startIndexing(panel.urls)
        }
    }
    #endif

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in
                    viewModel.startIndexing([url])
                }
            }
        }
        return true
    }
}

private struct EmbeddingsSearchAndResultsView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        VStack(spacing: 0) {
            // Search Bar + Query Attachment
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                        .font(.system(size: 15, weight: .medium))

                    TextField("Search across text, code, PDFs, images, audio & video…", text: $viewModel.embeddingQuery)
                        .textFieldStyle(.plain)
                        .font(.system(size: 15))
                        .onSubmit {
                            Task { await viewModel.runEmbeddingSearch() }
                        }
                        .onChange(of: viewModel.embeddingQuery) {
                            Task { await viewModel.runEmbeddingSearch() }
                        }

                    #if os(macOS)
                    Button {
                        pickQueryAttachment()
                    } label: {
                        Image(systemName: "paperclip")
                            .foregroundStyle(viewModel.queryAttachmentURL != nil ? Color.accentColor : Color.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Attach an image or audio clip for multimodal query search")
                    #endif

                    if !viewModel.embeddingQuery.isEmpty || viewModel.queryAttachmentURL != nil {
                        Button {
                            viewModel.embeddingQuery = ""
                            viewModel.queryAttachmentURL = nil
                            Task { await viewModel.runEmbeddingSearch() }
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }

                if let attachment = viewModel.queryAttachmentURL {
                    let kind = ArtifactKind.detect(from: attachment) ?? .image
                    HStack(spacing: 6) {
                        Image(systemName: kind.systemImage)
                            .font(.caption)
                            .foregroundStyle(.tint)
                        Text("Query Media: \(attachment.lastPathComponent)")
                            .font(.caption.weight(.medium))
                            .lineLimit(1)
                        Button {
                            viewModel.queryAttachmentURL = nil
                            Task { await viewModel.runEmbeddingSearch() }
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(Color.accentColor.opacity(0.14)))
                }

                ScoringPipelineBanner(
                    dimension: viewModel.selectedDimension,
                    task: viewModel.effectiveQueryTask,
                    hasQueryMedia: viewModel.queryAttachmentURL != nil,
                    includeSoundtrack: viewModel.loadAudioFromVideo,
                    videoSampling: viewModel.selectedSampling
                )
            }
            .padding(14)

            Divider()

            // Status / Progress Bar
            HStack {
                if viewModel.isIndexing || viewModel.isEmbeddingBusy {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.75)
                }
                Text(viewModel.embeddingStatusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                if viewModel.isIndexing {
                    Button {
                        viewModel.cancelIndexing()
                    } label: {
                        Label("Stop", systemImage: "stop.circle.fill")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.plain)
                    .help("Cancel current indexing job")
                }
                Text("\(viewModel.searchHits.count) results")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(Color.secondary.opacity(0.05))

            Divider()

            // Results List
            if viewModel.searchHits.isEmpty {
                ContentUnavailableView(
                    "No Matching Artifacts",
                    systemImage: "doc.text.magnifyingglass",
                    description: Text("Try another query, attach an image/audio clip, or drop folders onto the sidebar to index.")
                )
                .frame(maxHeight: .infinity)
            } else {
                List(selection: $viewModel.selectedHit) {
                    ForEach(viewModel.searchHits) { hit in
                        HitRowView(hit: hit)
                            .tag(hit)
                    }
                }
                .listStyle(.inset)
            }

            if viewModel.showEvalLabDrawer {
                Divider()
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label("Pairwise Embedding Cosine & Retrieval Lab (Recall@1 / MRR)", systemImage: "chart.bar.xaxis")
                            .font(.caption.weight(.semibold))
                        Spacer()
                        Button {
                            Task { await viewModel.runFullEvalLabSuite() }
                        } label: {
                            Label(
                                viewModel.isRunningEvalLab ? "Running Probes..." : "Run Eval Suite (chain + VLM + MRL)",
                                systemImage: "checkmark.shield"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.mini)
                        .disabled(viewModel.isRunningEvalLab)

                        ForEach(viewModel.labMetrics, id: \.dimensions) { metric in
                            Text(String(format: "%dd R@1: %.0f%% (MRR %.2f)", metric.dimensions, metric.recallAt1 * 100, metric.mrr))
                                .font(.caption2.monospaced())
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.12), in: Capsule())
                        }
                    }

                    if let suite = viewModel.evalLabSuiteReport {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(suite.probes, id: \.name) { probe in
                                    HStack(spacing: 4) {
                                        Image(systemName: probe.passed ? "checkmark.circle.fill" : "xmark.octagon.fill")
                                            .foregroundStyle(probe.passed ? .green : .red)
                                            .font(.caption2)
                                        Text("\(probe.name): \(probe.detail)")
                                            .font(.caption2.monospaced())
                                    }
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(
                                        (probe.passed ? Color.green : Color.red).opacity(0.12),
                                        in: Capsule()
                                    )
                                }
                            }
                        }
                    }

                    HStack(spacing: 10) {
                        TextField("Left text", text: $viewModel.compareLeftText)
                            .textFieldStyle(.roundedBorder)
                            .font(.caption)
                        TextField("Right text", text: $viewModel.compareRightText)
                            .textFieldStyle(.roundedBorder)
                            .font(.caption)
                        Button("Compare") {
                            viewModel.runPairwiseCompare()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)

                        Button {
                            Task { await viewModel.promoteComparePairToRetrievalLab() }
                        } label: {
                            Label("Add to Lab (\(viewModel.benchmarkCases.count))", systemImage: "plus.square.on.square")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(viewModel.selectedHit == nil && viewModel.searchHits.isEmpty)
                        .help("Promote Left query + selected artifact into the Retrieval Lab benchmark suite")

                        if let score = viewModel.compareCosineScore {
                            Text(String(format: "%dd Cosine: %.4f", viewModel.selectedDimension, score))
                                .font(.caption.monospaced().weight(.bold))
                                .foregroundStyle(.tint)
                        }
                    }
                }
                .padding(12)
                .background(Color.secondary.opacity(0.04))
            }
        }
    }

    #if os(macOS)
    private func pickQueryAttachment() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Attach Query Media"
        if panel.runModal() == .OK, let url = panel.url {
            viewModel.queryAttachmentURL = url
            Task { await viewModel.runEmbeddingSearch() }
        }
    }
    #endif
}

private struct ScoringPipelineBanner: View {
    let dimension: Int
    let task: EmbeddingTask
    let hasQueryMedia: Bool
    let includeSoundtrack: Bool
    let videoSampling: VideoWindowSampling

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                FeaturePill(
                    icon: "cpu",
                    label: "Gemma-2 2B Multimodal",
                    tint: .purple
                )
                FeaturePill(
                    icon: "ruler",
                    label: "Matryoshka \(dimension)d L2 Cosine",
                    tint: .blue
                )
                FeaturePill(
                    icon: "arrow.triangle.branch",
                    label: task == .query ? "Asymmetric Query→Doc" : "Symmetric Similarity",
                    tint: .indigo
                )
                FeaturePill(
                    icon: "film.stack",
                    label: "Video: \(videoSampling.label)",
                    tint: .orange
                )
                if includeSoundtrack {
                    FeaturePill(
                        icon: "waveform.and.magnifyingglass",
                        label: "Dual Video+Soundtrack Spans",
                        tint: .pink
                    )
                }
                if hasQueryMedia {
                    FeaturePill(
                        icon: "photo.on.rectangle.angled",
                        label: "Cross-Modal Media Query",
                        tint: .green
                    )
                }
            }
        }
    }
}

private struct ComputeMemorySplitBar: View {
    let split: (aneMB: Int, gpuMB: Int, cpuMB: Int)

    var body: some View {
        let total = max(1, split.aneMB + split.gpuMB + split.cpuMB)
        let aneFrac = CGFloat(split.aneMB) / CGFloat(total)
        let gpuFrac = CGFloat(split.gpuMB) / CGFloat(total)

        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Est. Residency Split")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("ANE ~\(split.aneMB)MB • GPU ~\(split.gpuMB)MB")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            GeometryReader { geo in
                HStack(spacing: 1) {
                    if split.aneMB > 0 {
                        Rectangle()
                            .fill(Color.green.gradient)
                            .frame(width: max(4, geo.size.width * aneFrac))
                    }
                    if split.gpuMB > 0 {
                        Rectangle()
                            .fill(Color.purple.gradient)
                            .frame(width: max(4, geo.size.width * gpuFrac))
                    }
                    if split.cpuMB > 0 {
                        Rectangle()
                            .fill(Color.blue.gradient)
                    }
                }
                .clipShape(Capsule())
            }
            .frame(height: 6)
        }
    }
}

private struct FeaturePill: View {
    let icon: String
    let label: String
    let tint: Color

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 9, weight: .semibold))
            Text(label)
                .font(.system(size: 10, weight: .medium))
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .foregroundStyle(tint)
        .background(
            Capsule()
                .fill(tint.opacity(0.12))
        )
    }
}

private struct HitRowView: View {
    let hit: SearchHit

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ThumbnailBadge(artifact: hit.artifact)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(hit.artifact.title)
                        .font(.system(size: 14, weight: .semibold))
                        .lineLimit(1)

                    if let spanBadge = hit.bestSpan.displayBadge {
                        Text(spanBadge)
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.accentColor.opacity(0.15)))
                            .foregroundStyle(Color.accentColor)
                    }

                    Text(hit.bestScoredFeature.title)
                        .font(.system(size: 10, weight: .medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.secondary.opacity(0.14)))
                        .foregroundStyle(.secondary)

                    if hit.hasMultiFeatureMatch {
                        Text("+ Audio & Vision")
                            .font(.system(size: 9, weight: .bold))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.pink.opacity(0.15)))
                            .foregroundStyle(.pink)
                    }

                    Spacer()
                    ScoreBadgeView(score: hit.bestScore)
                }

                if let preview = hit.previewText, !preview.isEmpty {
                    Text(preview.replacingOccurrences(of: "\n", with: " "))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                HStack(spacing: 6) {
                    Text(hit.artifact.kind.displayName)
                        .font(.caption2.weight(.medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.secondary.opacity(0.15)))

                    if hit.spanMatches.count > 1 {
                        Text("\(hit.spanMatches.count) spans matched")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    Text(hit.artifact.url.path)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        }
        .padding(.vertical, 6)
    }
}

private struct ScoreBadgeView: View {
    let score: Float

    var body: some View {
        let clamped = max(0, min(1, score))
        let color: Color = clamped > 0.55 ? .green : (clamped > 0.35 ? .blue : .secondary)
        return Text(String(format: "%.3f", score))
            .font(.caption2.monospacedDigit().weight(.semibold))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(Capsule().fill(color.opacity(0.16)))
            .foregroundStyle(color)
    }
}

private struct ThumbnailBadge: View {
    let artifact: Artifact

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.secondary.opacity(0.12))
                .frame(width: 46, height: 46)

            #if os(macOS)
            if artifact.kind == .image,
               let nsImage = NSImage(contentsOf: artifact.url) {
                Image(nsImage: nsImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 46, height: 46)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                Image(systemName: artifact.kind.systemImage)
                    .font(.system(size: 20))
                    .foregroundStyle(.tint)
            }
            #else
            Image(systemName: artifact.kind.systemImage)
                .font(.system(size: 20))
                .foregroundStyle(.tint)
            #endif
        }
    }
}

private struct EmbeddingsInspectorView: View {
    let hit: SearchHit?
    let selectedDimension: Int
    var onSendToCompare: ((SearchHit) -> Void)? = nil
    var onPinToLab: ((SearchHit) -> Void)? = nil
    @State private var player: AVPlayer?
    @State private var selectedChunkID: String?

    private var activeSpanMatch: SearchHit.SpanMatch? {
        guard let hit else { return nil }
        if let id = selectedChunkID, let match = hit.spanMatches.first(where: { $0.chunkID == id }) {
            return match
        }
        return hit.spanMatches.first
    }

    private var activeSpan: ArtifactSpan? {
        activeSpanMatch?.span ?? hit?.bestSpan
    }

    var body: some View {
        if let hit {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Header
                    HStack(spacing: 12) {
                        ThumbnailBadge(artifact: hit.artifact)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(hit.artifact.title)
                                .font(.headline)
                            HStack(spacing: 6) {
                                Text(hit.artifact.kind.displayName)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                ScoreBadgeView(score: hit.bestScore)
                            }
                        }
                        Spacer()
                    }

                    // Compare-to-Lab Bridge Actions
                    if onSendToCompare != nil || onPinToLab != nil {
                        HStack(spacing: 8) {
                            if let onSendToCompare {
                                Button {
                                    onSendToCompare(hit)
                                } label: {
                                    Label("Compare in Lab", systemImage: "arrow.left.and.right.square")
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                            if let onPinToLab {
                                Button {
                                    onPinToLab(hit)
                                } label: {
                                    Label("Pin to Retrieval Lab", systemImage: "chart.bar.doc.horizontal")
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                            Spacer()
                        }
                    }

                    Divider()

                    // Transparent Feature & Score Attribution Card
                    if let match = activeSpanMatch {
                        ScoredFeatureBreakdownCard(
                            artifact: hit.artifact,
                            spanMatch: match,
                            selectedDimension: selectedDimension
                        )
                    }

                    // Preview by Modality
                    switch hit.artifact.kind {
                    case .image:
                        #if os(macOS)
                        if let nsImage = NSImage(contentsOf: hit.artifact.url) {
                            Image(nsImage: nsImage)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 280)
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                        }
                        #endif

                    case .video, .audio:
                        VStack(alignment: .leading, spacing: 8) {
                            if let player {
                                #if os(macOS)
                                MacVideoPlayerView(player: player)
                                    .frame(height: hit.artifact.kind == .video ? 240 : 70)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                #else
                                VideoPlayer(player: player)
                                    .frame(height: hit.artifact.kind == .video ? 240 : 70)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                #endif
                            }

                            if let span = activeSpan, !span.frameTimes.isEmpty {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("EXTRACTED KEYFRAME TIMESTAMPS (CLICK TO SEEK)")
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundStyle(.secondary)

                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 6) {
                                            ForEach(Array(span.frameTimes.enumerated()), id: \.offset) { idx, frameTime in
                                                Button {
                                                    seekAndPlay(url: hit.artifact.url, startTime: frameTime)
                                                } label: {
                                                    HStack(spacing: 3) {
                                                        Image(systemName: "camera.viewfinder")
                                                            .font(.system(size: 9))
                                                        Text("#\(idx + 1) \(ArtifactSpan.formatTimestampMillis(frameTime))")
                                                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                                                    }
                                                    .padding(.horizontal, 7)
                                                    .padding(.vertical, 3)
                                                    .background(Capsule().fill(Color.orange.opacity(0.16)))
                                                    .foregroundStyle(.orange)
                                                }
                                                .buttonStyle(.plain)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        .onAppear {
                            seekAndPlay(url: hit.artifact.url, startTime: activeSpan?.startTimeSeconds)
                        }
                        .onChange(of: hit.artifact.id) {
                            selectedChunkID = nil
                            seekAndPlay(url: hit.artifact.url, startTime: hit.bestSpan.startTimeSeconds)
                        }

                    case .text, .code, .document:
                        if let snippet = activeSpanMatch?.previewText ?? hit.previewText {
                            ScrollView {
                                Text(snippet)
                                    .font(.system(size: 12, design: hit.artifact.kind == .code ? .monospaced : .default))
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(10)
                            }
                            .frame(maxHeight: 240)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color.secondary.opacity(0.08))
                            )
                        }
                    }

                    // Matching Spans Timeline / Pages / Lines
                    if !hit.spanMatches.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("SCORED SPANS & FEATURES (\(hit.spanMatches.count))")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text("Click span to inspect / seek")
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                            }

                            ForEach(hit.spanMatches.prefix(8)) { spanMatch in
                                let isSelected = (selectedChunkID ?? hit.spanMatches.first?.chunkID) == spanMatch.chunkID
                                let feature = spanMatch.scoredFeature(for: hit.artifact)
                                Button {
                                    selectedChunkID = spanMatch.chunkID
                                    if let startTime = spanMatch.span.startTimeSeconds {
                                        seekAndPlay(url: hit.artifact.url, startTime: startTime)
                                    }
                                } label: {
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack {
                                            Image(systemName: feature.systemImage)
                                                .font(.caption)
                                                .foregroundStyle(isSelected ? .white : Color.accentColor)
                                            Text(spanMatch.span.displayBadge ?? "Full Artifact")
                                                .font(.caption.monospacedDigit().weight(.medium))
                                                .foregroundStyle(isSelected ? .white : .primary)
                                            Text(feature.title)
                                                .font(.system(size: 10))
                                                .padding(.horizontal, 5)
                                                .padding(.vertical, 1)
                                                .background(
                                                    Capsule().fill(
                                                        isSelected ? Color.white.opacity(0.2) : Color.secondary.opacity(0.15)
                                                    )
                                                )
                                                .foregroundStyle(isSelected ? .white : .secondary)
                                            Spacer()
                                            Text(String(format: "%.3f", spanMatch.score))
                                                .font(.caption2.monospacedDigit().weight(.bold))
                                                .foregroundStyle(isSelected ? .white : .secondary)
                                        }

                                        if let snippet = spanMatch.previewText, !snippet.isEmpty {
                                            Text(snippet.replacingOccurrences(of: "\n", with: " "))
                                                .font(.caption2)
                                                .foregroundStyle(isSelected ? .white.opacity(0.85) : .secondary)
                                                .lineLimit(1)
                                        }
                                    }
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 7)
                                    .background(
                                        RoundedRectangle(cornerRadius: 7)
                                            .fill(isSelected ? Color.accentColor : Color.secondary.opacity(0.08))
                                    )
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }

                    Divider()

                    // Metadata
                    VStack(alignment: .leading, spacing: 6) {
                        Text("METADATA")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        LabeledContent("Path", value: hit.artifact.url.path)
                            .font(.caption)
                        LabeledContent("Size", value: ByteCountFormatter.string(fromByteCount: hit.artifact.fingerprint.byteCount, countStyle: .file))
                            .font(.caption)

                        ForEach(hit.artifact.attributes.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                            LabeledContent(key.capitalized, value: value)
                                .font(.caption)
                        }
                    }

                    #if os(macOS)
                    HStack {
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([hit.artifact.url])
                        } label: {
                            Label("Reveal in Finder", systemImage: "folder")
                        }

                        Button {
                            NSWorkspace.shared.open(hit.artifact.url)
                        } label: {
                            Label("Open File", systemImage: "arrow.up.forward.app")
                        }
                    }
                    .padding(.top, 6)
                    #endif
                }
                .padding(16)
            }
        } else {
            ContentUnavailableView(
                "Select an Artifact",
                systemImage: "sidebar.right",
                description: Text("Inspect multimodal previews, Matryoshka cosine attribution, and jump directly to matching timestamps or pages.")
            )
        }
    }

    private func seekAndPlay(url: URL, startTime: Double?) {
        let avPlayer: AVPlayer
        if let existing = player,
           let asset = existing.currentItem?.asset as? AVURLAsset,
           asset.url == url {
            avPlayer = existing
        } else {
            avPlayer = AVPlayer(url: url)
            player = avPlayer
        }
        if let startTime, startTime > 0 {
            let target = CMTime(seconds: startTime, preferredTimescale: 600)
            avPlayer.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
        }
    }
}

private struct ScoredFeatureBreakdownCard: View {
    let artifact: Artifact
    let spanMatch: SearchHit.SpanMatch
    let selectedDimension: Int

    var body: some View {
        let feature = spanMatch.scoredFeature(for: artifact)
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("WHY THIS MATCHED (EMBEDDING ATTRIBUTION)", systemImage: "sparkles.rectangle.stack")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(feature.subtitle)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }

            HStack(spacing: 8) {
                Text(feature.title)
                    .font(.caption.weight(.semibold))
                if let badge = spanMatch.span.displayBadge {
                    Text("• \(badge)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Matryoshka Cosine Across Subspaces (L2 Re-Normalized)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                ForEach([128, 256, 512, 768], id: \.self) { dim in
                    if let dimScore = spanMatch.scoresByDimension[dim] {
                        HStack(spacing: 8) {
                            Text("\(dim)d")
                                .font(.system(size: 10, weight: dim == selectedDimension ? .bold : .regular, design: .monospaced))
                                .frame(width: 34, alignment: .leading)
                                .foregroundStyle(dim == selectedDimension ? Color.accentColor : .secondary)

                            GeometryReader { geo in
                                ZStack(alignment: .leading) {
                                    Capsule()
                                        .fill(Color.secondary.opacity(0.15))
                                    Capsule()
                                        .fill(dim == selectedDimension ? Color.accentColor : Color.blue.opacity(0.6))
                                        .frame(width: max(4, geo.size.width * CGFloat(max(0, min(1, dimScore)))))
                                }
                            }
                            .frame(height: 6)

                            Text(String(format: "%.3f", dimScore))
                                .font(.system(size: 10, weight: dim == selectedDimension ? .bold : .regular, design: .monospaced))
                                .frame(width: 42, alignment: .trailing)
                        }
                    }
                }
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 9)
                .fill(Color.accentColor.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 9)
                .strokeBorder(Color.accentColor.opacity(0.18), lineWidth: 1)
        )
    }
}

#if os(macOS)
private struct MacVideoPlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .inline
        view.showsFullScreenToggleButton = true
        view.player = player
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
    }
}
#endif

// MARK: - 4. Models & Provenance Inspector Surface

struct ModelsInspectorSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Text("Registered Models (\(viewModel.registeredModels.count))")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    viewModel.convertingSourceModel = nil
                    viewModel.showMLXConverterSheet = true
                } label: {
                    Label("Quantize / Convert MLX…", systemImage: "cube.transparent")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    Task { await viewModel.rescanConfiguredModels() }
                } label: {
                    Label("Re-scan", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    Task { await viewModel.unloadAllModels() }
                } label: {
                    Label("Unload All Models", systemImage: "eject.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)

            Divider()

            List(viewModel.registeredModels) { model in
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(model.isLoaded ? Color.green : Color.secondary.opacity(0.35))
                        .frame(width: 8, height: 8)
                    Text(model.modelID)
                        .font(.headline)
                    Text(model.provider.displayName)
                        .font(.caption2.monospaced().weight(.bold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.15), in: Capsule())
                    Text(model.computePlacement.value.label)
                        .font(.caption2.monospaced())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.green.opacity(0.15), in: Capsule())
                    if model.isPinned {
                        Text("PINNED")
                            .font(.caption2.monospaced().weight(.bold))
                            .foregroundStyle(.orange)
                    }
                    Spacer()

                    Button(model.isPinned ? "Unpin" : "Pin") {
                        Task { await viewModel.toggleModelPin(model) }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Button(model.isLoaded ? "Unload" : "Load") {
                        Task { await viewModel.toggleModelLoaded(model) }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    if model.provider == .mlx {
                        Button {
                            viewModel.convertingSourceModel = model
                            viewModel.showMLXConverterSheet = true
                        } label: {
                            Label("Quantize…", systemImage: "cube.transparent")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    Button {
                        viewModel.editingSidecarModel = model
                    } label: {
                        Label("Sidecar", systemImage: "doc.badge.gearshape")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Button("Select") {
                        viewModel.selectedModelID = model.modelID
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }

                Text(
                    "Path: \(model.modelPath) • Arch: \(model.architecture.value) (\(model.architecture.provenance.rawValue)) • Quant: \(model.quantization.value) • Ctx: \(model.contextSize.value) (\(model.contextSize.provenance.rawValue)) • Reasoning: \(model.reasoningFormat.value) • Template: \(model.templateSource.value)"
                )
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    ForEach(model.tags, id: \.self) { tag in
                        Text(tag)
                            .font(.caption2.monospaced())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.secondary.opacity(0.12), in: Capsule())
                    }
                }
            }
            .padding(.vertical, 6)
        }
        }
        .sheet(item: $viewModel.editingSidecarModel) { model in
            ModelSidecarEditorSheet(viewModel: viewModel, model: model)
        }
        .sheet(isPresented: $viewModel.showMLXConverterSheet) {
            MLXModelConverterSheet(
                viewModel: viewModel,
                initialModel: viewModel.convertingSourceModel
            )
        }
    }
}

private struct MLXModelConverterSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let initialModel: EngineDescription?
    @Environment(\.dismiss) private var dismiss

    @State private var sourcePath: String = ""
    @State private var outputPath: String = ""
    @State private var mode: MLXQuantizationMode = .affine
    @State private var bits: Int = 4
    @State private var groupSize: Int = 64
    @State private var calibration: MLXQuantizationCalibration = .standard
    @State private var overwriteExisting: Bool = false
    @State private var isConverting: Bool = false
    @State private var progressMessage: String = ""
    @State private var conversionError: String?

    private var activeSpec: MLXConversionSpec {
        MLXConversionSpec(
            bits: bits,
            groupSize: groupSize,
            mode: mode,
            calibration: calibration,
            overwriteExistingOutput: overwriteExisting
        )
    }

    private var preflightResult: Result<MLXConversionPlan, Error>? {
        let src = sourcePath.trimmingCharacters(in: .whitespacesAndNewlines)
        let dst = outputPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !src.isEmpty, !dst.isEmpty else { return nil }
        return Result {
            try MLXModelConverter.planConversion(
                from: URL(fileURLWithPath: src),
                to: URL(fileURLWithPath: dst),
                spec: activeSpec
            )
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Pure-Swift MLX Checkpoint Quantizer (MLXLMCommon.convert)")
                        .font(.headline)
                    Text("Quantize unquantized BF16/FP16 .safetensors checkpoints in-process without Python")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    TextField("Source .safetensors directory", text: $sourcePath)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption.monospaced())
                        .onChange(of: sourcePath) {
                            autoUpdateOutputPathIfEmptyOrDefault()
                        }
                    #if os(macOS)
                    Button("Browse…") {
                        pickFolder(isOutput: false)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    #endif
                }

                HStack {
                    TextField("Output quantized directory", text: $outputPath)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption.monospaced())
                    #if os(macOS)
                    Button("Browse…") {
                        pickFolder(isOutput: true)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    #endif
                }
            }

            Divider()

            HStack(spacing: 14) {
                Picker("Mode", selection: $mode) {
                    ForEach(MLXQuantizationMode.allCases, id: \.self) { qMode in
                        Text(qMode.rawValue).tag(qMode)
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: mode) {
                    bits = mode.defaultBits
                    groupSize = mode.defaultGroupSize
                }

                Picker("Bits", selection: $bits) {
                    Text("4-bit").tag(4)
                    Text("8-bit").tag(8)
                }
                .pickerStyle(.segmented)
                .disabled(calibration == .q4Zero)

                Picker("Group Size", selection: $groupSize) {
                    Text("32").tag(32)
                    Text("64").tag(64)
                    Text("128").tag(128)
                }
                .pickerStyle(.segmented)
                .disabled(calibration == .q4Zero)
            }

            HStack(spacing: 14) {
                Picker("Calibration", selection: $calibration) {
                    Text("standard").tag(MLXQuantizationCalibration.standard)
                    Text("q4_zero (4-bit gs=32 affine)").tag(MLXQuantizationCalibration.q4Zero)
                }
                .pickerStyle(.menu)
                .onChange(of: calibration) {
                    if calibration == .q4Zero {
                        mode = .affine
                        bits = 4
                        groupSize = 32
                    }
                }

                Toggle("Overwrite existing output directory", isOn: $overwriteExisting)
                    .font(.caption)
            }

            if let preflightResult {
                switch preflightResult {
                case .success(let plan):
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Image(systemName: "checkmark.seal.fill")
                                .foregroundStyle(.green)
                            Text("Preflight Ready: \(plan.modelType) (\(plan.isVisionModel ? "VLM" : "LLM")) • \(plan.resolvedBits)-bit \(plan.mode.rawValue) (group_size=\(plan.resolvedGroupSize), \(plan.calibration.rawValue))")
                                .font(.caption.weight(.semibold))
                        }
                        Text("Shards: \(plan.safetensorsFiles.count) .safetensors • Sidecars: \(plan.sidecarFiles.count) files • Reasoning: \(plan.reasoningFormat)")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.green.opacity(0.1)))

                case .failure(let err):
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(err.localizedDescription)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.orange.opacity(0.1)))
                }
            }

            if !progressMessage.isEmpty {
                HStack(spacing: 8) {
                    if isConverting {
                        ProgressView().controlSize(.small)
                    }
                    Text(progressMessage)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }

            if let conversionError {
                Text(conversionError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            HStack {
                Button("Close") { dismiss() }
                    .buttonStyle(.bordered)
                    .disabled(isConverting)
                Spacer()
                Button("Quantize Checkpoint") {
                    runConversion()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isConverting || (try? preflightResult?.get()) == nil)
            }
        }
        .padding(20)
        .frame(minWidth: 580)
        .onAppear {
            if let initialModel {
                sourcePath = initialModel.modelPath
                outputPath = "\(initialModel.modelPath)-MLX-\(bits)bit"
            }
        }
    }

    private func autoUpdateOutputPathIfEmptyOrDefault() {
        let trimmed = sourcePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            outputPath = "\(trimmed)-MLX-\(activeSpec.resolvedBits)bit"
        }
    }

    private func runConversion() {
        isConverting = true
        conversionError = nil
        progressMessage = "Starting MLX quantization…"
        let srcURL = URL(fileURLWithPath: sourcePath.trimmingCharacters(in: .whitespacesAndNewlines))
        let dstURL = URL(fileURLWithPath: outputPath.trimmingCharacters(in: .whitespacesAndNewlines))
        let spec = activeSpec

        Task {
            do {
                let (_, weights) = try await MLXModelConverter.convert(
                    from: srcURL,
                    to: dstURL,
                    spec: spec,
                    progressHandler: { msg in
                        Task { @MainActor in
                            self.progressMessage = msg
                        }
                    }
                )
                await viewModel.rescanConfiguredModels()
                isConverting = false
                progressMessage = "Completed: wrote \(weights.count) quantized .safetensors shard(s) to \(dstURL.lastPathComponent)."
            } catch {
                isConverting = false
                conversionError = error.localizedDescription
            }
        }
    }

    #if os(macOS)
    private func pickFolder(isOutput: Bool) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = isOutput
        panel.allowsMultipleSelection = false
        panel.prompt = isOutput ? "Select Output Folder" : "Select Source Checkpoint"
        if panel.runModal() == .OK, let url = panel.url {
            if isOutput {
                outputPath = url.path
            } else {
                sourcePath = url.path
                autoUpdateOutputPathIfEmptyOrDefault()
            }
        }
    }
    #endif
}

private struct ModelSidecarEditorSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let model: EngineDescription
    @Environment(\.dismiss) private var dismiss

    @State private var pinned: Bool = false
    @State private var ctxSize: Int = 8192
    @State private var flashAttn: Bool = true
    @State private var supportsVision: Bool = false
    @State private var supportsThinking: Bool = true
    @State private var temperature: Double = 0.7
    @State private var topP: Double = 0.95
    @State private var thinkingBudget: Int = 4096
    @State private var tagsText: String = ""
    @State private var unsetText: String = ""
    @State private var templateSourceLabel: String = "default_chatml"
    @State private var hasTemplateOverride: Bool = false
    @State private var chatTemplateText: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Edit model.heylook.toml Sidecar & Chat Template")
                        .font(.headline)
                    Text(model.modelID)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(templateSourceLabel)
                    .font(.caption2.monospaced().weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(
                        (hasTemplateOverride ? Color.purple : Color.secondary).opacity(0.15),
                        in: Capsule()
                    )
                    .foregroundStyle(hasTemplateOverride ? .purple : .secondary)
            }

            Divider()

            Toggle("Pin Model in Unified Memory (Exempt from GPU LRU Eviction)", isOn: $pinned)
            Stepper("Context Size (ctx_size): \(ctxSize)", value: $ctxSize, in: 2048...1_048_576, step: 4096)
                .font(.subheadline.monospacedDigit())
            Toggle("Flash Attention (flash_attn)", isOn: $flashAttn)
            Toggle("Supports Vision (supports_vision)", isOn: $supportsVision)
            Toggle("Supports Thinking (supports_thinking)", isOn: $supportsThinking)

            Divider()

            HStack {
                Text("Default Temperature: \(String(format: "%.2f", temperature))")
                    .font(.subheadline.monospacedDigit())
                Slider(value: $temperature, in: 0...2.0)
            }
            HStack {
                Text("Default Top-P: \(String(format: "%.2f", topP))")
                    .font(.subheadline.monospacedDigit())
                Slider(value: $topP, in: 0.05...1.0)
            }
            Stepper("Default Thinking Budget: \(thinkingBudget)", value: $thinkingBudget, in: 256...65536, step: 512)
                .font(.subheadline.monospacedDigit())

            TextField("Tags (comma-separated, e.g. chat, vision, mtp)", text: $tagsText)
                .textFieldStyle(.roundedBorder)
                .font(.caption.monospaced())

            TextField("Unset Keys (comma-separated, e.g. draft_model_path, spec_type)", text: $unsetText)
                .textFieldStyle(.roundedBorder)
                .font(.caption.monospaced())

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("4-Rung Chat Template Override (chat_template.heylook.jinja)")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    if hasTemplateOverride {
                        Button("Revert Override", role: .destructive) {
                            Task {
                                let reverted = await viewModel.revertChatTemplateOverride(for: model)
                                templateSourceLabel = reverted.source
                                hasTemplateOverride = reverted.hasOverride
                                chatTemplateText = reverted.template
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    }
                    Button("Save Template Override") {
                        Task {
                            let saved = await viewModel.saveChatTemplateOverride(chatTemplateText, for: model)
                            templateSourceLabel = saved.source
                            hasTemplateOverride = saved.hasOverride
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .disabled(chatTemplateText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                TextEditor(text: $chatTemplateText)
                    .font(.system(.caption, design: .monospaced))
                    .frame(minHeight: 110, maxHeight: 150)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(Color.secondary.opacity(0.22), lineWidth: 1)
                    )
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .buttonStyle(.bordered)
                Spacer()
                Button("Save model.heylook.toml") {
                    let parsedTags = tagsText
                        .split(separator: ",")
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }
                    let parsedUnset = unsetText
                        .split(separator: ",")
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }
                    let updatedSidecar = ModelSidecarConfig(
                        id: model.modelID,
                        provider: model.provider,
                        pinned: pinned,
                        computePlacement: model.computePlacement.value,
                        ctxSize: ctxSize,
                        flashAttn: flashAttn,
                        supportsVision: supportsVision,
                        supportsThinking: supportsThinking,
                        unsetKeys: parsedUnset,
                        sampling: SamplerSettings(
                            temperature: temperature,
                            topP: topP,
                            thinkingBudget: thinkingBudget
                        ),
                        tags: parsedTags
                    )
                    Task {
                        await viewModel.saveModelSidecar(updatedSidecar, for: model)
                        dismiss()
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 560)
        .onAppear {
            let loaded = viewModel.loadSidecarConfig(for: model)
            pinned = loaded.pinned ?? model.isPinned
            ctxSize = loaded.ctxSize ?? model.contextSize.value
            flashAttn = loaded.flashAttn ?? model.flashAttention.value
            supportsVision = loaded.supportsVision ?? model.supportsVision.value
            supportsThinking = loaded.supportsThinking ?? model.supportsThinking.value
            temperature = loaded.sampling.temperature ?? 0.7
            topP = loaded.sampling.topP ?? 0.95
            thinkingBudget = loaded.sampling.thinkingBudget ?? (model.thinkingBudget.value ?? 4096)
            tagsText = (loaded.tags.isEmpty ? model.tags : loaded.tags).joined(separator: ", ")
            unsetText = loaded.unsetKeys.joined(separator: ", ")
            let tmplState = viewModel.loadChatTemplateState(for: model)
            templateSourceLabel = tmplState.source
            hasTemplateOverride = tmplState.hasOverride
            chatTemplateText = tmplState.template
        }
    }
}

// MARK: - 5. Perf & Server Surface

struct PerfServerSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                GroupBox("Unified Memory & Dual-Lane Compute Residency (\(viewModel.memorySnapshot.deviceName))") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 24) {
                            VStack(alignment: .leading) {
                                Text("Physical Unified RAM")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Text(String(format: "%.1f GB", viewModel.memorySnapshot.totalPhysicalRAMGB))
                                    .font(.title3.monospaced().weight(.semibold))
                            }
                            VStack(alignment: .leading) {
                                Text("Metal Max Working Set")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Text(String(format: "%.1f GB", viewModel.memorySnapshot.metalRecommendedMaxWorkingSetGB))
                                    .font(.title3.monospaced().weight(.semibold))
                            }
                            VStack(alignment: .leading) {
                                Text("GPU Models / ANE Models")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Text("\(viewModel.schedulerMetrics.loadedGPUModels.count) GPU / \(viewModel.schedulerMetrics.loadedANEModels.count) ANE")
                                    .font(.title3.monospaced().weight(.semibold))
                            }
                            Spacer()
                            Button("Refresh Telemetry") {
                                Task { await viewModel.refreshMetrics() }
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    .padding(8)
                }

                GroupBox("Lifecycle Reclamation, Subprocess Registry & GGUF Port Pool") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 24) {
                            VStack(alignment: .leading) {
                                Text("Active llama-server Subprocesses")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Text("\(viewModel.activeSubprocessCount)")
                                    .font(.title3.monospaced().weight(.semibold))
                                    .foregroundStyle(viewModel.activeSubprocessCount > 0 ? .orange : .green)
                            }
                            VStack(alignment: .leading) {
                                Text("Allocated GGUF Loopback Ports")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Text(
                                    viewModel.activeGGUFPorts.isEmpty
                                        ? "None (18080+ free)"
                                        : viewModel.activeGGUFPorts
                                            .sorted { $0.value < $1.value }
                                            .map { "\($0.key)::\($0.value)" }
                                            .joined(separator: ", ")
                                )
                                .font(.caption.monospaced().weight(.medium))
                                .lineLimit(1)
                            }
                            Spacer()
                            Button("Unload All & Reclaim Memory", role: .destructive) {
                                Task { await viewModel.unloadAllModels() }
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        Text("Calls `provider.unload()`, `MLX.Memory.clearCache()`, `ANECoreMLRuntime.unload()`, and `SubprocessLifecycleRegistry.shared.terminateAll()` (`SIGTERM` -> `SIGKILL`).")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(8)
                }

                GroupBox("Embedded HTTP & SSE Server (/v1/messages, /v1/embeddings, /v1/conversations)") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Circle()
                                .fill(viewModel.isServerRunning ? Color.green : Color.secondary)
                                .frame(width: 10, height: 10)
                            Text(
                                viewModel.isServerRunning
                                    ? "Listening on http://127.0.0.1:\(viewModel.serverPort)"
                                    : "Embedded server stopped"
                            )
                            .font(.subheadline.monospaced())
                            Spacer()
                            Button(viewModel.isServerRunning ? "Stop Server" : "Start Server") {
                                Task { await viewModel.toggleServer() }
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        Text("Supports disconnect-resilient background generation (`heylook_saved`) for iOS 27.2+ mobile clients over LAN/Tailscale.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(8)
                }

                GroupBox("Bonjour (_heylook._tcp) Auto-Discovery & Remote Mac Studio Connection (iOS 27.2+ / LAN / Tailscale)") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Label(
                                viewModel.isBrowsingBonjour ? "Scanning LAN for _heylook._tcp peers..." : "Bonjour Auto-Discovery",
                                systemImage: "antenna.radiowaves.left.and.right"
                            )
                            .font(.subheadline.weight(.medium))
                            Spacer()
                            Button(viewModel.isBrowsingBonjour ? "Stop Scan" : "Scan Bonjour") {
                                if viewModel.isBrowsingBonjour {
                                    viewModel.stopBonjourDiscovery()
                                } else {
                                    viewModel.startBonjourDiscovery()
                                }
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }

                        if !viewModel.discoveredServers.isEmpty {
                            VStack(alignment: .leading, spacing: 6) {
                                ForEach(viewModel.discoveredServers) { peer in
                                    HStack {
                                        Image(systemName: "desktopcomputer")
                                            .foregroundStyle(.tint)
                                        Text(peer.name)
                                            .font(.caption.weight(.semibold))
                                        Text(peer.baseURLString)
                                            .font(.caption2.monospaced())
                                            .foregroundStyle(.secondary)
                                        Spacer()
                                        Button("Connect") {
                                            Task { await viewModel.connectToRemoteServer(peer.baseURLString) }
                                        }
                                        .buttonStyle(.borderedProminent)
                                        .controlSize(.mini)
                                    }
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 5)
                                    .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
                                }
                            }
                        }

                        HStack(spacing: 8) {
                            TextField("http://mac-studio.local:8000 or Tailscale URL", text: $viewModel.remoteServerURLString)
                                .textFieldStyle(.roundedBorder)
                                .font(.caption.monospaced())
                            Button("Connect Remote") {
                                Task { await viewModel.connectToRemoteServer(viewModel.remoteServerURLString) }
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                        }

                        if let connected = viewModel.connectedRemoteServerURL {
                            HStack(spacing: 6) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                                Text("Connected to remote Mac Studio: \(connected)")
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(8)
                }
            }
            .padding(16)
        }
    }
}

struct LoadAndSamplerPanelSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        let resolved = viewModel.currentResolvedSamplers
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Sampler Cascade & Thinking Controls")
                        .font(.headline)
                    Text("Precedence: builtin → model.heylook.toml → preset (\(viewModel.selectedPresetName)) → request")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Reset Overrides") {
                    viewModel.resetSamplerOverrides()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(viewModel.userOverriddenSamplerKeys.isEmpty)
            }

            Divider()

            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Temperature: \(String(format: "%.2f", viewModel.temperatureOverride))")
                        .font(.subheadline.monospacedDigit())
                    Spacer()
                    SamplerProvenancePill(
                        source: resolved.provenanceByField[.temperature] ?? .derived,
                        isRequestOverride: viewModel.userOverriddenSamplerKeys.contains("temperature")
                    )
                }
                Slider(
                    value: Binding(
                        get: { viewModel.temperatureOverride },
                        set: {
                            viewModel.temperatureOverride = $0
                            viewModel.userOverriddenSamplerKeys.insert("temperature")
                        }
                    ),
                    in: 0...2
                )

                HStack {
                    Text("Top-P: \(String(format: "%.2f", viewModel.topPOverride))")
                        .font(.subheadline.monospacedDigit())
                    Spacer()
                    SamplerProvenancePill(
                        source: resolved.provenanceByField[.topP] ?? .derived,
                        isRequestOverride: viewModel.userOverriddenSamplerKeys.contains("top_p")
                    )
                }
                Slider(
                    value: Binding(
                        get: { viewModel.topPOverride },
                        set: {
                            viewModel.topPOverride = $0
                            viewModel.userOverriddenSamplerKeys.insert("top_p")
                        }
                    ),
                    in: 0.05...1.0
                )

                HStack {
                    Stepper(
                        "Top-K: \(viewModel.topKOverride)",
                        value: Binding(
                            get: { viewModel.topKOverride },
                            set: {
                                viewModel.topKOverride = $0
                                viewModel.userOverriddenSamplerKeys.insert("top_k")
                            }
                        ),
                        in: 1...200,
                        step: 5
                    )
                    .font(.subheadline.monospacedDigit())
                    Spacer()
                    SamplerProvenancePill(
                        source: resolved.provenanceByField[.topK] ?? .derived,
                        isRequestOverride: viewModel.userOverriddenSamplerKeys.contains("top_k")
                    )
                }

                HStack {
                    Text("Min-P: \(String(format: "%.2f", viewModel.minPOverride))")
                        .font(.subheadline.monospacedDigit())
                    Spacer()
                    SamplerProvenancePill(
                        source: resolved.provenanceByField[.minP] ?? .derived,
                        isRequestOverride: viewModel.userOverriddenSamplerKeys.contains("min_p")
                    )
                }
                Slider(
                    value: Binding(
                        get: { viewModel.minPOverride },
                        set: {
                            viewModel.minPOverride = $0
                            viewModel.userOverriddenSamplerKeys.insert("min_p")
                        }
                    ),
                    in: 0.0...0.5
                )

                HStack {
                    Stepper(
                        "Max Output Tokens: \(viewModel.maxTokensOverride)",
                        value: Binding(
                            get: { viewModel.maxTokensOverride },
                            set: {
                                viewModel.maxTokensOverride = $0
                                viewModel.userOverriddenSamplerKeys.insert("max_tokens")
                            }
                        ),
                        in: 64...32768,
                        step: 256
                    )
                    .font(.subheadline.monospacedDigit())
                    Spacer()
                    SamplerProvenancePill(
                        source: resolved.provenanceByField[.maxTokens] ?? .derived,
                        isRequestOverride: viewModel.userOverriddenSamplerKeys.contains("max_tokens")
                    )
                }

                HStack {
                    Stepper(
                        "Thinking Budget: \(viewModel.thinkingBudgetOverride) tokens",
                        value: Binding(
                            get: { viewModel.thinkingBudgetOverride },
                            set: {
                                viewModel.thinkingBudgetOverride = $0
                                viewModel.userOverriddenSamplerKeys.insert("thinking_budget")
                            }
                        ),
                        in: 256...32768,
                        step: 512
                    )
                    .font(.subheadline.monospacedDigit())
                    Spacer()
                    SamplerProvenancePill(
                        source: resolved.provenanceByField[.thinkingBudget] ?? .derived,
                        isRequestOverride: viewModel.userOverriddenSamplerKeys.contains("thinking_budget")
                    )
                }
            }

            Divider()

            TextField("System Prompt Override", text: $viewModel.draftSystemPrompt, axis: .vertical)
                .lineLimit(2...5)
                .textFieldStyle(.roundedBorder)

            HStack {
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 480)
    }
}

private struct SamplerProvenancePill: View {
    let source: ProvenanceSource
    let isRequestOverride: Bool

    var body: some View {
        let label = isRequestOverride ? "request" : source.rawValue
        Text(label)
            .font(.system(size: 10, weight: .semibold, design: .monospaced))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(pillColor.opacity(0.16), in: Capsule())
            .foregroundStyle(pillColor)
    }

    private var pillColor: Color {
        if isRequestOverride {
            return .orange
        }
        switch source {
        case .configured:
            return .purple
        case .observed, .observedCached:
            return .green
        case .derived, .unknown, .notApplicable:
            return .secondary
        }
    }
}
