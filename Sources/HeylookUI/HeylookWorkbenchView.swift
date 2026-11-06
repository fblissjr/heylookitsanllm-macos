import Foundation
import Observation
import SwiftUI
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

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
    public var isGenerating: Bool = false
    public var activeRequestID: String?
    public var streamingThinking: String = ""
    public var streamingContent: String = ""
    public var showPromptPreview: Bool = false
    public var showLoadPanel: Bool = false

    // Sampler & Thinking Override Controls
    public var presets: [PresetRecord] = []
    public var selectedPresetName: String = "Balanced Daily Driver"
    public var temperatureOverride: Double = 0.7
    public var topPOverride: Double = 0.95
    public var maxTokensOverride: Int = 2048
    public var thinkingEnabled: Bool = true
    public var thinkingDepth: String = "medium"
    public var seedOverride: String = ""

    // Message Edit State
    public var editingMessageID: String?
    public var editingContent: String = ""

    // Notebook State
    public var notebooks: [NotebookRecord] = []
    public var selectedNotebookID: String?
    public var notebookTitle: String = "Scratchpad"
    public var notebookBody: String = "Once upon a time in unified memory, the GPU and the Neural Engine shared a 192 GB pool..."
    public var isNotebookGenerating: Bool = false

    // Embeddings & ANE Studio State
    public var modelComputeOptions: ModelComputeOptions = .hybridANETowers
    public var embeddingQuery: String = ""
    public var selectedDimension: Int = Matryoshka.gemma2NativeDimensions
    public var searchHits: [SearchHit] = []
    public var compareLeftText: String = "Swift 6.2 actor isolation for unified memory inference on Apple Silicon"
    public var compareRightText: String = "Running MLX and CoreML ANE vision towers concurrently on Mac Studio"
    public var compareCosineScore: Float?
    public var aneInspectionReports: [ANETowerInspector.ComponentComputeReport] = []
    public var labMetrics: [EmbeddingLabRunner.DimensionMetrics] = []

    // Server State
    public var isServerRunning: Bool = false
    public var serverPort: UInt16 = 8000
    public var statusBanner: String?

    private let embeddingEngine = DeterministicHashEmbeddingEngine()
    private let indexCoordinator: IndexCoordinator

    public init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let heylookDir = appSupport.appendingPathComponent("Heylook", isDirectory: true)
        try? FileManager.default.createDirectory(at: heylookDir, withIntermediateDirectories: true)

        let dbURL = heylookDir.appendingPathComponent("heylook.sqlite")
        let vecDbURL = heylookDir.appendingPathComponent("heylook_vectors.sqlite")
        let tomlURL = heylookDir.appendingPathComponent("heylook.toml")

        let cfg = (try? String(contentsOf: tomlURL, encoding: .utf8)).map(HeylookConfig.parseTOML) ?? HeylookConfig()
        self.config = cfg
        self.store = (try? SQLiteStore(databaseURL: dbURL))
            ?? (try! SQLiteStore(databaseURL: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("heylook_fallback.sqlite")))
        self.vectorStore = (try? VectorStore(databaseURL: vecDbURL))
            ?? (try! VectorStore(databaseURL: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("heylook_vec_fallback.sqlite")))
        self.scheduler = ComputeScheduler(maxLoadedGPUModels: cfg.maxLoadedModels, maxQueueDepth: cfg.maxQueueDepth)
        self.registry = RequestRegistry()
        self.router = HeylookRouter(
            store: self.store,
            scheduler: self.scheduler,
            registry: self.registry,
            bindHost: cfg.serverHost
        )
        self.server = HeylookHTTPServer(
            router: self.router,
            host: cfg.serverHost,
            port: UInt16(cfg.serverPort)
        )
        self.serverPort = UInt16(cfg.serverPort)
        self.indexCoordinator = IndexCoordinator(engine: self.embeddingEngine, store: self.vectorStore)

        Task {
            await bootstrap()
        }
    }

    public func bootstrap() async {
        let folderURLs = config.scanFolders.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        var discovered = HeylookConfig.discoverModels(in: folderURLs)
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
        if let t = preset.settings.temperature { temperatureOverride = t }
        if let p = preset.settings.topP { topPOverride = p }
        if let m = preset.settings.maxTokens { maxTokensOverride = m }
        if let th = preset.settings.thinking { thinkingEnabled = th }
    }

    public func sendChatPrompt() async {
        let promptText = draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !promptText.isEmpty, !isGenerating else { return }
        draftPrompt = ""

        if selectedConversationID == nil {
            createNewConversation()
        }
        guard let convID = selectedConversationID else { return }

        do {
            let nextPos = (currentMessages.map(\.position).max() ?? -1) + 1
            let userMsg = MessageRecord(
                conversationID: convID,
                position: nextPos,
                role: "user",
                content: promptText,
                modelID: selectedModelID
            )
            try store.appendMessage(userMsg)
            currentMessages = try store.listMessages(conversationID: convID)
            if var conv = try store.getConversation(id: convID), conv.title == "New Chat" {
                conv.title = String(promptText.prefix(38))
                try store.upsertConversation(conv)
                conversations = try store.listConversations()
            }
            await streamAssistantReply(for: convID)
        } catch {
            statusBanner = "Chat error: \(error.localizedDescription)"
        }
    }

    public func streamAssistantReply(for convID: String) async {
        isGenerating = true
        streamingThinking = ""
        streamingContent = ""
        let reqID = "req_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        activeRequestID = reqID

        let modelID = selectedModelID.isEmpty ? "gemma-4-26b-a4b-it-mlx" : selectedModelID
        do {
            let provider = try await scheduler.provider(for: modelID)
            let turns = currentMessages.map { MessagesRequest.MessageTurn(role: $0.role, content: $0.content) }
            let req = MessagesRequest(
                model: modelID,
                messages: turns,
                system: draftSystemPrompt.isEmpty ? nil : draftSystemPrompt,
                maxTokens: maxTokensOverride,
                temperature: temperatureOverride,
                topP: topPOverride,
                stream: true,
                thinking: thinkingEnabled
            )
            let resolved = SamplerCascade.resolve(request: req.samplerOverlay)
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
        var lines: [String] = []
        if !draftSystemPrompt.isEmpty {
            lines.append("<|im_start|>system\n\(draftSystemPrompt)<|im_end|>")
        }
        for msg in currentMessages {
            lines.append("<|im_start|>\(msg.role)\n\(msg.content)<|im_end|>")
        }
        if !draftPrompt.isEmpty {
            lines.append("<|im_start|>user\n\(draftPrompt)<|im_end|>")
        }
        lines.append("<|im_start|>assistant")
        if thinkingEnabled {
            lines.append("<think>")
        }
        return lines.joined(separator: "\n")
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
            let req = MessagesRequest(
                model: modelID,
                messages: [MessagesRequest.MessageTurn(role: "user", content: "Continue writing seamlessly from:\n\n\(notebookBody)")],
                maxTokens: 256,
                temperature: temperatureOverride,
                thinking: false
            )
            let resolved = SamplerCascade.resolve(request: req.samplerOverlay)
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

    // MARK: - Embeddings & ANE Studio Operations

    public func seedSampleCorpusIfNeeded() async {
        guard ((try? vectorStore.stats().artifactCount) ?? 0) == 0 else { return }

        let samples: [(String, String, String)] = [
            (
                "doc-vision",
                "VISION.md — Native Apple Silicon Workbench",
                "Heylook is a single-user multimodal AI workbench for macOS 27.2 and iOS 27.2 unifying MLX Metal GPU generation with ANE CoreML co-residency."
            ),
            (
                "doc-scheduler",
                "ComputeScheduler.swift — Dual-Lane GPU & ANE Co-Residency",
                "The ComputeScheduler separates heavy autoregressive GPU models from ANE-resident embedding and vision towers so background media indexing never evicts a 70B LLM."
            ),
            (
                "doc-towers",
                "ModelComputeOptions.swift — Per-Tower Compute Unit Selection",
                "Assign Vision Tower and Audio Conformer to CPU + Apple Neural Engine while keeping Text Projection on GPU or ANE with Matryoshka 768d to 128d truncation."
            ),
            (
                "doc-sse",
                "MessagesWire.swift — Disconnect-Resilient SSE Generation",
                "Mobile iOS clients can lock their screen mid-generation while the Mac Studio finishes writing the assistant turn to SQLite WAL and emits heylook_saved."
            )
        ]

        for (id, title, snippet) in samples {
            let artifact = Artifact(
                id: id,
                collectionID: "default",
                kind: .text,
                url: URL(fileURLWithPath: "/workspace/\(id).md"),
                title: title,
                fingerprint: ContentFingerprint(byteCount: Int64(snippet.utf8.count), modifiedUnixMs: 1_700_000_000_000),
                modelID: embeddingEngine.modelID
            )
            let vec = embeddingEngine.embed(text: snippet)
            let chunk = EmbeddedChunk(
                id: "\(id)#0",
                artifactID: id,
                span: .whole,
                vector: vec,
                previewText: snippet
            )
            try? vectorStore.upsert(artifact: artifact, chunks: [chunk], modelID: embeddingEngine.modelID)
        }
        embeddingQuery = "Apple Neural Engine co-residency"
        await runEmbeddingSearch()
        await runEmbeddingLabEvaluation()
    }

    public func runEmbeddingSearch() async {
        let q = embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else {
            searchHits = []
            return
        }
        do {
            searchHits = try await indexCoordinator.search(
                queryText: q,
                dimensions: selectedDimension,
                limit: 10
            )
        } catch {
            statusBanner = "Search failed: \(error.localizedDescription)"
        }
    }

    public func runPairwiseCompare() {
        let left = embeddingEngine.embed(text: compareLeftText, dimensions: selectedDimension)
        let right = embeddingEngine.embed(text: compareRightText, dimensions: selectedDimension)
        compareCosineScore = Matryoshka.cosine(left, right, dimensions: selectedDimension)
    }

    public func runEmbeddingLabEvaluation() async {
        let cases: [EmbeddingLabRunner.BenchmarkCase] = [
            .init(query: "Apple Neural Engine co-residency", expectedArtifactID: "doc-vision"),
            .init(query: "evict 70B LLM background indexing", expectedArtifactID: "doc-scheduler"),
            .init(query: "Matryoshka 768d to 128d truncation", expectedArtifactID: "doc-towers"),
            .init(query: "heylook_saved SQLite WAL mobile lock screen", expectedArtifactID: "doc-sse")
        ]
        labMetrics = (try? await EmbeddingLabRunner.evaluate(coordinator: indexCoordinator, cases: cases)) ?? []
    }

    // MARK: - Server Lifecycle

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

                    Spacer()

                    Toggle("Prompt Preview", isOn: $viewModel.showPromptPreview)
                        .toggleStyle(.button)
                        .controlSize(.small)
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

                HStack(alignment: .bottom, spacing: 10) {
                    TextField("Ask anything or drop images/audio...", text: $viewModel.draftPrompt, axis: .vertical)
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
                    } else {
                        Button("Send") {
                            Task { await viewModel.sendChatPrompt() }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
                .padding(12)
            }
        }
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

// MARK: - 3. Embeddings & ANE Studio Surface

struct EmbeddingsStudioSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                GroupBox("Per-Tower Compute Unit Assignment & 192 GB Unified Memory Split") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack(spacing: 16) {
                            ForEach(EmbeddingComponent.allCases) { comp in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(comp.title)
                                        .font(.caption.weight(.semibold))
                                    Picker(
                                        comp.title,
                                        selection: Binding(
                                            get: { viewModel.modelComputeOptions[comp] },
                                            set: { viewModel.modelComputeOptions[comp] = $0 }
                                        )
                                    ) {
                                        ForEach(TowerComputeUnit.allCases) { unit in
                                            Text(unit.label).tag(unit)
                                        }
                                    }
                                    .labelsHidden()
                                }
                            }

                            Spacer()

                            VStack(alignment: .trailing, spacing: 4) {
                                Text("Matryoshka Dim")
                                    .font(.caption.weight(.semibold))
                                Picker("Dim", selection: $viewModel.selectedDimension) {
                                    ForEach(Matryoshka.gemma2Dimensions, id: \.self) { dim in
                                        Text("\(dim)d").tag(dim)
                                    }
                                }
                                .pickerStyle(.segmented)
                                .frame(width: 240)
                            }
                        }

                        let split = viewModel.modelComputeOptions.memorySplitMB
                        HStack(spacing: 16) {
                            Label(
                                "ANE Resident: \(split.aneMB) MB",
                                systemImage: "brain.head.profile"
                            )
                            .font(.caption.monospaced())
                            .foregroundStyle(.green)

                            Label(
                                "GPU: \(split.gpuMB) MB • CPU: \(split.cpuMB) MB",
                                systemImage: "memorychip"
                            )
                            .font(.caption.monospaced())
                            .foregroundStyle(.blue)

                            Spacer()

                            Text(viewModel.modelComputeOptions.shortSummaryBadge)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(8)
                }

                GroupBox("Multimodal Span Search (Matryoshka \(viewModel.selectedDimension)d Cosine)") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            TextField("Search indexed text, images, PDFs, audio, video...", text: $viewModel.embeddingQuery)
                                .textFieldStyle(.roundedBorder)
                                .onSubmit {
                                    Task { await viewModel.runEmbeddingSearch() }
                                }
                            Button("Search") {
                                Task { await viewModel.runEmbeddingSearch() }
                            }
                            .buttonStyle(.borderedProminent)
                        }

                        ForEach(viewModel.searchHits) { hit in
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(hit.artifact.title)
                                        .font(.subheadline.weight(.semibold))
                                    if let preview = hit.previewText {
                                        Text(preview)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                Spacer()
                                Text(String(format: "%.4f", hit.bestScore))
                                    .font(.caption.monospaced().weight(.bold))
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.accentColor.opacity(0.15), in: Capsule())
                            }
                            .padding(8)
                            .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
                        }
                    }
                    .padding(8)
                }

                GroupBox("Pairwise Embedding Cosine & Retrieval Lab (Recall@1 / MRR)") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 12) {
                            TextField("Left text", text: $viewModel.compareLeftText, axis: .vertical)
                                .lineLimit(2...4)
                                .textFieldStyle(.roundedBorder)
                            TextField("Right text", text: $viewModel.compareRightText, axis: .vertical)
                                .lineLimit(2...4)
                                .textFieldStyle(.roundedBorder)
                        }
                        HStack {
                            Button("Compute Cosine Similarity") {
                                viewModel.runPairwiseCompare()
                            }
                            .buttonStyle(.bordered)

                            if let score = viewModel.compareCosineScore {
                                Text(String(format: "Cosine (%dd): %.4f", viewModel.selectedDimension, score))
                                    .font(.subheadline.monospaced().weight(.semibold))
                            }

                            Spacer()

                            ForEach(viewModel.labMetrics, id: \.dimensions) { metric in
                                Text(String(format: "%dd R@1: %.0f%% (MRR %.2f)", metric.dimensions, metric.recallAt1 * 100, metric.mrr))
                                    .font(.caption2.monospaced())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                                    .background(Color.secondary.opacity(0.12), in: Capsule())
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

// MARK: - 4. Models & Provenance Inspector Surface

struct ModelsInspectorSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        List(viewModel.registeredModels) { model in
            VStack(alignment: .leading, spacing: 6) {
                HStack {
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
                            .font(.caption2.monospaced())
                            .foregroundStyle(.orange)
                    }
                    Spacer()
                    Button("Select") {
                        viewModel.selectedModelID = model.modelID
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                Text("Path: \(model.modelPath) • Arch: \(model.architecture.value) (\(model.architecture.provenance.rawValue)) • Quant: \(model.quantization.value)")
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
            }
            .padding(16)
        }
    }
}

struct LoadAndSamplerPanelSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Sampler Cascade & Thinking Controls")
                .font(.headline)

            HStack {
                Text("Temperature: \(String(format: "%.2f", viewModel.temperatureOverride))")
                Slider(value: $viewModel.temperatureOverride, in: 0...2)
            }
            HStack {
                Text("Top-P: \(String(format: "%.2f", viewModel.topPOverride))")
                Slider(value: $viewModel.topPOverride, in: 0.05...1.0)
            }
            Stepper("Max Output Tokens: \(viewModel.maxTokensOverride)", value: $viewModel.maxTokensOverride, in: 64...32768, step: 256)

            TextField("System Prompt Override", text: $viewModel.draftSystemPrompt, axis: .vertical)
                .lineLimit(2...5)
                .textFieldStyle(.roundedBorder)

            HStack {
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(minWidth: 420)
    }
}
