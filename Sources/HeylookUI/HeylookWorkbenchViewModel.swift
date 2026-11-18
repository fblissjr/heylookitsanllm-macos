import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import Observation
import UniformTypeIdentifiers
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

/// Staged multimodal attachment supporting optional ≤2048px max-edge downscaling (`Shrink` vs `Full res`)
/// and in-session multi-turn replay across Chat turns.
public struct StagedChatAttachment: Identifiable, Sendable, Equatable {
    public let id: String
    public let kind: MessagesRequest.MediaKind
    public let mimeType: String
    public let filename: String
    public let originalData: Data
    public let shrunk2048Data: Data?
    public let shrunkMimeType: String?
    public let originalPixelWidth: Int?
    public let originalPixelHeight: Int?
    public let shrunkPixelWidth: Int?
    public let shrunkPixelHeight: Int?
    public var useShrunk2048: Bool

    public init(
        id: String = UUID().uuidString,
        kind: MessagesRequest.MediaKind,
        mimeType: String,
        filename: String,
        originalData: Data,
        shrunk2048Data: Data? = nil,
        shrunkMimeType: String? = nil,
        originalPixelWidth: Int? = nil,
        originalPixelHeight: Int? = nil,
        shrunkPixelWidth: Int? = nil,
        shrunkPixelHeight: Int? = nil,
        useShrunk2048: Bool? = nil
    ) {
        self.id = id
        self.kind = kind
        self.mimeType = mimeType
        self.filename = filename
        self.originalData = originalData
        self.shrunk2048Data = shrunk2048Data
        self.shrunkMimeType = shrunkMimeType
        self.originalPixelWidth = originalPixelWidth
        self.originalPixelHeight = originalPixelHeight
        self.shrunkPixelWidth = shrunkPixelWidth
        self.shrunkPixelHeight = shrunkPixelHeight
        self.useShrunk2048 = useShrunk2048 ?? (shrunk2048Data != nil)
    }

    public var canToggleShrink: Bool {
        kind == .image && shrunk2048Data != nil
    }

    public var effectiveData: Data {
        if useShrunk2048, let shrunk2048Data {
            return shrunk2048Data
        }
        return originalData
    }

    public var effectiveMimeType: String {
        if useShrunk2048, shrunk2048Data != nil, let shrunkMimeType {
            return shrunkMimeType
        }
        return mimeType
    }

    public var dimensionsLabel: String? {
        if useShrunk2048, let sw = shrunkPixelWidth, let sh = shrunkPixelHeight {
            return "\(sw)×\(sh)"
        }
        if let ow = originalPixelWidth, let oh = originalPixelHeight {
            return "\(ow)×\(oh)"
        }
        return nil
    }

    public var wireAttachment: MessagesRequest.MediaAttachment {
        MessagesRequest.MediaAttachment(
            id: id,
            kind: kind,
            mimeType: effectiveMimeType,
            base64Data: effectiveData.base64EncodedString(),
            filename: filename
        )
    }

    public static func make(
        kind: MessagesRequest.MediaKind,
        mimeType: String,
        filename: String,
        data: Data
    ) -> StagedChatAttachment {
        guard kind == .image,
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = props[kCGImagePropertyPixelWidth] as? Int,
              let height = props[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0 else {
            return StagedChatAttachment(
                kind: kind,
                mimeType: mimeType,
                filename: filename,
                originalData: data
            )
        }

        guard max(width, height) > 2048 else {
            return StagedChatAttachment(
                kind: .image,
                mimeType: mimeType,
                filename: filename,
                originalData: data,
                originalPixelWidth: width,
                originalPixelHeight: height
            )
        }

        let thumbOpts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: 2048
        ]
        guard let cgThumb = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOpts as CFDictionary) else {
            return StagedChatAttachment(
                kind: .image,
                mimeType: mimeType,
                filename: filename,
                originalData: data,
                originalPixelWidth: width,
                originalPixelHeight: height
            )
        }

        let mutableData = NSMutableData()
        let outUTType = mimeType == "image/png" ? UTType.png.identifier : UTType.jpeg.identifier
        let outMime = mimeType == "image/png" ? "image/png" : "image/jpeg"
        if let dest = CGImageDestinationCreateWithData(mutableData, outUTType as CFString, 1, nil) {
            let destProps: [CFString: Any] = outMime == "image/jpeg"
                ? [kCGImageDestinationLossyCompressionQuality: 0.9]
                : [:]
            CGImageDestinationAddImage(dest, cgThumb, destProps as CFDictionary)
            if CGImageDestinationFinalize(dest) {
                return StagedChatAttachment(
                    kind: .image,
                    mimeType: mimeType,
                    filename: filename,
                    originalData: data,
                    shrunk2048Data: mutableData as Data,
                    shrunkMimeType: outMime,
                    originalPixelWidth: width,
                    originalPixelHeight: height,
                    shrunkPixelWidth: cgThumb.width,
                    shrunkPixelHeight: cgThumb.height,
                    useShrunk2048: true
                )
            }
        }

        return StagedChatAttachment(
            kind: .image,
            mimeType: mimeType,
            filename: filename,
            originalData: data,
            originalPixelWidth: width,
            originalPixelHeight: height
        )
    }
}

public enum DestructiveActionPrompt: Identifiable, Equatable, Sendable {
    case clearEmbeddingIndex
    case unloadAllModels
    case deleteConversation(id: String, title: String)
    case deleteNotebook(id: String, title: String)
    case deleteMessage(id: String, conversationID: String)
    case deletePreset(id: String, name: String)
    case clearAllConversationsAndNotebooks

    public var id: String {
        switch self {
        case .clearEmbeddingIndex:
            return "clearEmbeddingIndex"
        case .unloadAllModels:
            return "unloadAllModels"
        case .deleteConversation(let id, _):
            return "deleteConversation:\(id)"
        case .deleteNotebook(let id, _):
            return "deleteNotebook:\(id)"
        case .deleteMessage(let id, _):
            return "deleteMessage:\(id)"
        case .deletePreset(let id, _):
            return "deletePreset:\(id)"
        case .clearAllConversationsAndNotebooks:
            return "clearAllConversationsAndNotebooks"
        }
    }

    public var title: String {
        switch self {
        case .clearEmbeddingIndex:
            return "Clear Embedding Index?"
        case .unloadAllModels:
            return "Unload All Resident Models?"
        case .deleteConversation:
            return "Delete Conversation?"
        case .deleteNotebook:
            return "Delete Notebook?"
        case .deleteMessage:
            return "Delete Message?"
        case .deletePreset:
            return "Delete Preset?"
        case .clearAllConversationsAndNotebooks:
            return "Clear All Conversations & Notebooks?"
        }
    }

    public var confirmLabel: String {
        switch self {
        case .clearEmbeddingIndex:
            return "Clear Index"
        case .unloadAllModels:
            return "Unload All Models"
        case .deleteConversation:
            return "Delete Conversation"
        case .deleteNotebook:
            return "Delete Notebook"
        case .deleteMessage:
            return "Delete Message"
        case .deletePreset:
            return "Delete Preset"
        case .clearAllConversationsAndNotebooks:
            return "Clear All Data"
        }
    }

    public var message: String {
        switch self {
        case .clearEmbeddingIndex:
            return "This removes all indexed artifacts and Matryoshka vectors from VectorStore SQLite."
        case .unloadAllModels:
            return "This unloads all GPU and ANE models, clears Metal/ANE caches, and terminates llama-server subprocesses."
        case .deleteConversation(_, let title):
            return "Permanently delete \"\(title)\" and all of its messages?"
        case .deleteNotebook(_, let title):
            return "Permanently delete notebook \"\(title)\"?"
        case .deleteMessage:
            return "Remove this message from the conversation? Surrounding turns will be preserved."
        case .deletePreset(_, let name):
            return "Permanently delete preset \"\(name)\"?"
        case .clearAllConversationsAndNotebooks:
            return "Permanently delete all conversations, messages, media blobs, turn stats, and notebooks? Saved presets and indexed embeddings are preserved."
        }
    }
}

public enum StructuredOutputMode: String, CaseIterable, Identifiable, Sendable {
    case text = "text"
    case jsonObject = "json_object"
    case jsonSchema = "json_schema"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .text: return "Text"
        case .jsonObject: return "JSON Object"
        case .jsonSchema: return "JSON Schema"
        }
    }
}

public struct StagedImagePlanEstimate: Sendable, Equatable {
    public let resizedWidth: Int
    public let resizedHeight: Int
    public let estimatedTokens: Int
    public let tokenCap: Int

    public init(resizedWidth: Int, resizedHeight: Int, estimatedTokens: Int, tokenCap: Int) {
        self.resizedWidth = resizedWidth
        self.resizedHeight = resizedHeight
        self.estimatedTokens = estimatedTokens
        self.tokenCap = tokenCap
    }
}

public enum EmbeddingsSubMode: String, CaseIterable, Identifiable, Sendable {
    case search = "Media Search"
    case compareMatrix = "N×N Compare"
    case evalLab = "Embedding Lab"

    public var id: String { rawValue }

    public var icon: String {
        switch self {
        case .search: return "magnifyingglass"
        case .compareMatrix: return "square.grid.3x3.fill"
        case .evalLab: return "chart.bar.xaxis"
        }
    }
}

public struct CompareMatrixItem: Identifiable, Sendable, Equatable {
    public let id: String
    public var label: String
    public var kind: ArtifactKind
    public var textContent: String
    public var fileURL: URL?

    public init(
        id: String = UUID().uuidString,
        label: String,
        kind: ArtifactKind = .text,
        textContent: String = "",
        fileURL: URL? = nil
    ) {
        self.id = id
        self.label = label
        self.kind = kind
        self.textContent = textContent
        self.fileURL = fileURL
    }
}

public struct EmbeddingLabConfigRun: Identifiable, Sendable, Equatable {
    public let id: String
    public let label: String
    public let checkpointName: String
    public let computePresetLabel: String
    public let shortPlacementBadge: String
    public let metricsByDim: [EmbeddingLabRunner.DimensionMetrics]
    public let meanTop1Margin: Double
    public let worstTop1Margin: Double
    public let evalLatencyMs: Double
    public let aneMemoryMB: Int
    public let gpuMemoryMB: Int
    public let timestampUnixMs: Int64

    public init(
        id: String = UUID().uuidString,
        label: String,
        checkpointName: String,
        computePresetLabel: String,
        shortPlacementBadge: String,
        metricsByDim: [EmbeddingLabRunner.DimensionMetrics],
        meanTop1Margin: Double,
        worstTop1Margin: Double,
        evalLatencyMs: Double,
        aneMemoryMB: Int,
        gpuMemoryMB: Int,
        timestampUnixMs: Int64 = SQLiteStore.nowMillis()
    ) {
        self.id = id
        self.label = label
        self.checkpointName = checkpointName
        self.computePresetLabel = computePresetLabel
        self.shortPlacementBadge = shortPlacementBadge
        self.metricsByDim = metricsByDim
        self.meanTop1Margin = meanTop1Margin
        self.worstTop1Margin = worstTop1Margin
        self.evalLatencyMs = evalLatencyMs
        self.aneMemoryMB = aneMemoryMB
        self.gpuMemoryMB = gpuMemoryMB
        self.timestampUnixMs = timestampUnixMs
    }
}

public enum PerfTimeWindow: String, CaseIterable, Identifiable, Sendable {
    case oneHour = "1h"
    case sixHours = "6h"
    case twentyFourHours = "24h"
    case sevenDays = "7d"

    public var id: String { rawValue }

    public var label: String { rawValue }
}

public typealias WorkbenchShellController = HeylookWorkbenchViewModel

/// Lean `@Observable` shell coordinator delegating domain work to modular controllers:
/// - `chat: ChatController` + `chatSettings: DocumentSettingsController`
/// - `notebookPresets: NotebookAndPresetsController` + `notebookSettings: DocumentSettingsController`
/// - `embeddings: EmbeddingsStudioController`
/// - `modelsPerf: ModelsAndPerfController`
///
/// Zero background 3-second polling: metrics, perf profiles, and observability update strictly on
/// demand and after model/generation lifecycle events.
@MainActor
@Observable
public final class HeylookWorkbenchViewModel {
    // MARK: - Core Services & Shell Navigation

    public let store: SQLiteStore
    public let vectorStore: VectorStore
    public let scheduler: ComputeScheduler
    public let registry: RequestRegistry
    public let router: HeylookRouter
    public let server: HeylookHTTPServer

    public var activeSurface: HeylookSurface = .chat {
        didSet {
            guard oldValue != activeSurface else { return }
            handleActiveSurfaceTransition(from: oldValue, to: activeSurface)
        }
    }

    public var showLoadPanel: Bool = false
    public var showInspector: Bool = true
    public var pendingConfirmation: DestructiveActionPrompt?
    public var statusBanner: String?
    public var lastChatError: String?

    // MARK: - Focused Domain Controllers

    public let chatSettings: DocumentSettingsController
    public let notebookSettings: DocumentSettingsController
    public let chat: ChatController
    public let notebookPresets: NotebookAndPresetsController
    public let embeddings: EmbeddingsStudioController
    public let modelsPerf: ModelsAndPerfController

    /// Returns the settings controller belonging to the active document surface (`chatSettings` vs `notebookSettings`).
    public var activeDocumentSettings: DocumentSettingsController {
        activeSurface == .notebook ? notebookSettings : chatSettings
    }

    @ObservationIgnored private var storeMutationTask: Task<Void, Never>?

    public init(autoBootstrap: Bool = true) {
        let dbURL = HeylookConfig.defaultStoreURL
        let vecDbURL = HeylookConfig.defaultVectorStoreURL
        let cfg = HeylookConfig.loadDefault()
        let sqliteStore = (try? SQLiteStore(databaseURL: dbURL))
            ?? (try! SQLiteStore(databaseURL: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("heylook_fallback.sqlite")))
        let vecStore = (try? VectorStore(databaseURL: vecDbURL))
            ?? (try! VectorStore(databaseURL: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("heylook_vec_fallback.sqlite")))
        let computeScheduler = ComputeScheduler(maxLoadedGPUModels: cfg.maxLoadedModels, maxQueueDepth: cfg.maxQueueDepth)
        let reqRegistry = RequestRegistry.shared

        self.store = sqliteStore
        self.vectorStore = vecStore
        self.scheduler = computeScheduler
        self.registry = reqRegistry

        let embControllerInitEngine = NativeMLXEmbeddingGemma2.makeBestAvailableEngine(computeOptions: .hybridANETowers)
        let httpRouter = HeylookRouter(
            store: sqliteStore,
            scheduler: computeScheduler,
            registry: reqRegistry,
            embeddingEngine: embControllerInitEngine,
            bindHost: cfg.serverHost,
            allowedHosts: cfg.allowedHosts,
            adminToken: cfg.adminToken ?? ProcessInfo.processInfo.environment["HEYLOOK_ADMIN_TOKEN"],
            isReadOnlyConfig: HeylookConfig.isReadOnlyModelConfig
        )
        self.router = httpRouter
        self.server = HeylookHTTPServer(
            router: httpRouter,
            host: cfg.serverHost,
            port: UInt16(cfg.serverPort)
        )

        self.chatSettings = DocumentSettingsController(scopeKind: .conversation, store: sqliteStore)
        self.notebookSettings = DocumentSettingsController(scopeKind: .notebook, store: sqliteStore)
        self.chat = ChatController(
            store: sqliteStore,
            scheduler: computeScheduler,
            registry: reqRegistry,
            chatSettings: self.chatSettings
        )
        self.notebookPresets = NotebookAndPresetsController(
            store: sqliteStore,
            scheduler: computeScheduler,
            registry: reqRegistry,
            notebookSettings: self.notebookSettings
        )
        self.embeddings = EmbeddingsStudioController(vectorStore: vecStore, router: httpRouter)
        self.modelsPerf = ModelsAndPerfController(
            config: cfg,
            store: sqliteStore,
            scheduler: computeScheduler,
            router: httpRouter,
            server: self.server
        )

        wireControllers()

        if autoBootstrap {
            Task {
                await bootstrap()
            }
        }
    }

    private func wireControllers() {
        chatSettings.activeModelProvider = { [weak self] in self?.modelsPerf.activeModelDescription }
        chatSettings.livePresetsProvider = { [weak self] in self?.notebookPresets.presets ?? [] }
        chatSettings.onPresetsChanged = { [weak self] next in
            self?.notebookPresets.presets = next
        }
        chatSettings.onPersistDocumentParams = { [weak self] docID, params, presetID in
            self?.chat.persistConversationParamsCallback(docID: docID, params: params, appliedPresetID: presetID)
        }
        chatSettings.onPersistDocumentSystemPrompt = { [weak self] docID, prompt in
            self?.chat.persistConversationSystemPromptCallback(docID: docID, systemPrompt: prompt)
        }
        chatSettings.onSchedulePromptPreview = { [weak self] in
            if self?.chat.showPromptPreview == true {
                self?.chat.schedulePromptPreviewRefresh()
            }
        }
        chatSettings.onStatusMessage = { [weak self] msg in self?.statusBanner = msg }
        chatSettings.onErrorMessage = { [weak self] msg in self?.lastChatError = msg }
        chatSettings.onRequestDeletePresetConfirmation = { [weak self] preset in
            self?.pendingConfirmation = .deletePreset(id: preset.id, name: preset.name)
        }

        notebookSettings.activeModelProvider = { [weak self] in self?.modelsPerf.activeModelDescription }
        notebookSettings.livePresetsProvider = { [weak self] in self?.notebookPresets.presets ?? [] }
        notebookSettings.onPresetsChanged = { [weak self] next in
            self?.notebookPresets.presets = next
        }
        notebookSettings.onPersistDocumentParams = { [weak self] docID, params, presetID in
            self?.notebookPresets.persistNotebookParamsCallback(docID: docID, params: params, appliedPresetID: presetID)
        }
        notebookSettings.onPersistDocumentSystemPrompt = { [weak self] docID, prompt in
            self?.notebookPresets.persistNotebookSystemPromptCallback(docID: docID, systemPrompt: prompt)
        }
        notebookSettings.onStatusMessage = { [weak self] msg in self?.statusBanner = msg }
        notebookSettings.onErrorMessage = { [weak self] msg in self?.lastChatError = msg }
        notebookSettings.onRequestDeletePresetConfirmation = { [weak self] preset in
            self?.pendingConfirmation = .deletePreset(id: preset.id, name: preset.name)
        }

        chat.configureDependencies(
            remoteBackend: { [weak self] in self?.modelsPerf.activeRemoteBackend },
            connectedRemoteServerURL: { [weak self] in self?.modelsPerf.connectedRemoteServerURL },
            registeredModels: { [weak self] in self?.modelsPerf.registeredModels ?? [] },
            generationModels: { [weak self] in self?.modelsPerf.generationModels ?? [] },
            activeModelID: { [weak self] in self?.modelsPerf.selectedModelID ?? "" },
            activeModel: { [weak self] in self?.modelsPerf.activeModelDescription },
            activeEngineContract: { [weak self] in
                guard let self else { return nil }
                return await self.modelsPerf.activeEngineContract(for: self.modelsPerf.selectedModelID)
            },
            sidecarLoader: { [weak self] desc in
                self?.modelsPerf.loadSidecarConfig(for: desc) ?? ModelSidecarConfig()
            }
        )
        chat.onSelectModelID = { [weak self] id in self?.modelsPerf.selectedModelID = id }
        chat.onStatusMessage = { [weak self] msg in self?.statusBanner = msg }
        chat.onErrorMessage = { [weak self] msg in self?.lastChatError = msg }
        chat.onSwitchToChatSurface = { [weak self] in self?.activeSurface = .chat }
        chat.onGenerationCompleted = { [weak self] in
            await self?.modelsPerf.refreshMetrics()
        }

        notebookPresets.configureDependencies(
            remoteBackend: { [weak self] in self?.modelsPerf.activeRemoteBackend },
            activeModelID: { [weak self] in self?.modelsPerf.selectedModelID ?? "" },
            activeModel: { [weak self] in self?.modelsPerf.activeModelDescription },
            generationModels: { [weak self] in self?.modelsPerf.generationModels ?? [] },
            activeEngineContract: { [weak self] in
                guard let self else { return nil }
                return await self.modelsPerf.activeEngineContract(for: self.modelsPerf.selectedModelID)
            }
        )
        notebookPresets.onSelectModelID = { [weak self] id in self?.modelsPerf.selectedModelID = id }
        notebookPresets.onStatusMessage = { [weak self] msg in self?.statusBanner = msg }
        notebookPresets.onErrorMessage = { [weak self] msg in self?.lastChatError = msg }
        notebookPresets.onPresetsUpdated = { [weak self] in
            self?.chatSettings.recomputeDriftState()
            self?.notebookSettings.recomputeDriftState()
        }
        notebookPresets.onSendToChatRequested = { [weak self] conv in
            guard let self else { return }
            self.chat.conversations = (try? self.store.listConversations()) ?? self.chat.conversations
            self.chat.selectConversation(conv.id)
            self.activeSurface = .chat
        }
        notebookPresets.onGenerationCompleted = { [weak self] in
            await self?.modelsPerf.refreshMetrics()
        }

        embeddings.onStatusMessage = { [weak self] msg in self?.statusBanner = msg }

        modelsPerf.onStatusMessage = { [weak self] msg in self?.statusBanner = msg }
        modelsPerf.onModelsChanged = { [weak self] _ in
            self?.chat.refreshASRAvailability()
        }
        modelsPerf.onSelectedModelChanged = { [weak self] _, desc in
            guard let self else { return }
            if self.chat.showPromptPreview {
                self.chat.schedulePromptPreviewRefresh()
            }
            if self.activeSurface == .notebook {
                _ = self.notebookSettings.reconcileSettings(for: desc, engineContract: nil)
            } else {
                _ = self.chatSettings.reconcileSettings(for: desc, engineContract: nil)
            }
        }
        modelsPerf.onRefreshANEReportsIfNeeded = { [weak self] in
            await self?.embeddings.refreshComputeReportsIfNeeded()
        }
        modelsPerf.onUnloadEmbeddingEngine = { [weak self] in
            await self?.embeddings.unloadActiveEngine()
        }
        modelsPerf.onSchedulePromptPreview = { [weak self] in
            if self?.chat.showPromptPreview == true {
                self?.chat.schedulePromptPreviewRefresh()
            }
        }
        modelsPerf.onRemoteBackendChanged = { [weak self] remote in
            guard let self else { return }
            if let remote {
                self.embeddings.embeddingServerURLString = remote.baseURL.absoluteString
                await self.notebookPresets.syncWithRemoteBackend(remote)
                await self.chat.syncWithRemoteBackend(remote)
                self.wireRemotePresetCallbacks(remote)
            } else {
                self.wireRemotePresetCallbacks(nil)
                self.refreshStoreData()
            }
        }
    }

    private func wireRemotePresetCallbacks(_ remote: RemoteWorkbenchBackend?) {
        if let remote {
            let createHandler: (PresetRecord) async throws -> PresetRecord = { record in
                let dto = try await remote.createPreset(
                    name: record.name,
                    systemPrompt: record.systemPrompt,
                    params: record.settings
                )
                return PresetRecord(
                    id: dto.id,
                    name: dto.name,
                    scope: record.scope,
                    settings: dto.params.settings,
                    systemPrompt: dto.systemPrompt,
                    description: record.description
                )
            }
            let updateHandler: (PresetRecord) async throws -> PresetRecord = { record in
                let dto = try await remote.updatePreset(
                    id: record.id,
                    name: record.name,
                    systemPrompt: record.systemPrompt,
                    params: record.settings
                )
                return PresetRecord(
                    id: dto.id,
                    name: dto.name,
                    scope: record.scope,
                    settings: dto.params.settings,
                    systemPrompt: dto.systemPrompt,
                    description: record.description
                )
            }
            chatSettings.remoteCreatePresetHandler = createHandler
            chatSettings.remoteUpdatePresetHandler = updateHandler
            notebookSettings.remoteCreatePresetHandler = createHandler
            notebookSettings.remoteUpdatePresetHandler = updateHandler
        } else {
            chatSettings.remoteCreatePresetHandler = nil
            chatSettings.remoteUpdatePresetHandler = nil
            notebookSettings.remoteCreatePresetHandler = nil
            notebookSettings.remoteUpdatePresetHandler = nil
        }
    }

    // MARK: - Bootstrap, Lifecycle & Surface Transitions (Zero Polling)

    public func bootstrap() async {
        notebookPresets.seedDefaultPresetsIfNeeded()

        let discovered = modelsPerf.config.discoverConfiguredModels()
        await scheduler.registerDiscoveredModels(discovered)

        modelsPerf.registeredModels = await scheduler.listModels()
        if modelsPerf.selectedModelID.isEmpty {
            modelsPerf.selectedModelID = modelsPerf.generationModels.first(where: { $0.isLoaded })?.modelID
                ?? modelsPerf.generationModels.first?.modelID
                ?? modelsPerf.registeredModels.first?.modelID
                ?? ""
        }

        embeddings.setupEmbeddingCoordinator()
        await embeddings.syncEmbeddingEngineToRouter()
        refreshStoreData()
        await modelsPerf.refreshMetrics()

        storeMutationTask?.cancel()
        let stream = store.mutationStream()
        storeMutationTask = Task { [weak self] in
            for await event in stream {
                guard !Task.isCancelled, let self else { break }
                self.handleStoreMutationEvent(event)
            }
        }
    }

    private func handleStoreMutationEvent(_ event: StoreMutationEvent) {
        notebookPresets.handleStoreMutationEvent(event)
        chat.handleStoreMutationEvent(event, activeSurface: activeSurface)
        if case .messagesChanged = event {
            modelsPerf.refreshPerfAndObservability()
        } else if case .dataCleared = event {
            modelsPerf.refreshPerfAndObservability()
        }
    }

    private func handleActiveSurfaceTransition(from oldSurface: HeylookSurface, to newSurface: HeylookSurface) {
        if oldSurface == .notebook {
            notebookPresets.flushNotebookSaveIfDirty()
        } else if oldSurface == .chat {
            chatSettings.flushPendingWrites()
        }

        if newSurface == .notebook {
            if let modelID = notebookPresets.notebooks.first(where: { $0.id == notebookPresets.selectedNotebookID })?.modelID,
               !modelID.isEmpty,
               modelsPerf.generationModels.contains(where: { $0.modelID == modelID }) {
                modelsPerf.selectedModelID = modelID
            }
            notebookSettings.recomputeDriftState()
        } else if newSurface == .chat {
            if let convID = chat.selectedConversationID,
               let conv = chat.conversations.first(where: { $0.id == convID }),
               let defModel = conv.defaultModelID,
               !defModel.isEmpty,
               modelsPerf.generationModels.contains(where: { $0.modelID == defModel }) {
                modelsPerf.selectedModelID = defModel
            }
            chatSettings.recomputeDriftState()
        } else if newSurface == .perf || newSurface == .models {
            modelsPerf.refreshPerfAndObservability()
            Task { await modelsPerf.refreshMetrics() }
        }
    }

    public func flushPendingEdits() {
        chatSettings.flushPendingWrites()
        notebookSettings.flushPendingWrites()
        notebookPresets.flushNotebookSaveIfDirty()
    }

    public func refreshStoreData() {
        notebookPresets.refreshFromStore()
        chat.refreshFromStore()
    }

    public func confirmDestructiveAction(_ action: DestructiveActionPrompt) {
        switch action {
        case .clearEmbeddingIndex:
            embeddings.clearEmbeddingIndex()
        case .unloadAllModels:
            Task { await modelsPerf.unloadAllModels() }
        case .deleteConversation(let id, _):
            chat.deleteConversation(id)
        case .deleteNotebook(let id, _):
            notebookPresets.deleteNotebook(id)
        case .deleteMessage(let id, let conversationID):
            chat.deleteSingleMessage(id: id, conversationID: conversationID)
        case .deletePreset(let id, _):
            notebookPresets.deletePresetByID(id)
        case .clearAllConversationsAndNotebooks:
            clearAllStoreData()
        }
    }

    public func clearAllStoreData() {
        if chat.isGenerating {
            chat.cancelGeneration()
        }
        if notebookPresets.isNotebookGenerating {
            notebookPresets.cancelNotebookGeneration()
        }
        do {
            try store.clearAllData()
            refreshStoreData()
            modelsPerf.refreshPerfAndObservability()
            statusBanner = "Cleared all conversations, messages, media blobs, stats, and notebooks."
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Clear all data failed: \(error.localizedDescription)"
        }
    }

    public func shutdown() async {
        flushPendingEdits()
        storeMutationTask?.cancel()
        storeMutationTask = nil
        chat.cancelBackgroundTasks()
        embeddings.stopBackgroundTasks()
        modelsPerf.stopBonjourDiscovery()
        if modelsPerf.isServerRunning {
            await server.stop()
            modelsPerf.isServerRunning = false
        } else {
            await scheduler.unloadAllModels()
            await embeddings.unloadActiveEngine()
        }
        await modelsPerf.refreshMetrics()
    }

    // MARK: - Thin Step-3B Compatibility Bridges (Models & Perf)

    public var config: HeylookConfig {
        get { modelsPerf.config }
        set { modelsPerf.config = newValue }
    }
    public var registeredModels: [EngineDescription] {
        get { modelsPerf.registeredModels }
        set { modelsPerf.registeredModels = newValue }
    }
    public var generationModels: [EngineDescription] { modelsPerf.generationModels }
    public var filteredModels: [EngineDescription] { modelsPerf.filteredModels }
    public var modelCountsByProvider: [EngineProviderKind: Int] { modelsPerf.modelCountsByProvider }
    public var selectedModelID: String {
        get { modelsPerf.selectedModelID }
        set { modelsPerf.selectedModelID = newValue }
    }
    public var activeModelDescription: EngineDescription? { modelsPerf.activeModelDescription }
    public var loadingModelIDs: Set<String> { modelsPerf.loadingModelIDs }
    public var selectedProviderFilter: EngineProviderKind? {
        get { modelsPerf.selectedProviderFilter }
        set { modelsPerf.selectedProviderFilter = newValue }
    }
    public var editingSidecarModel: EngineDescription? {
        get { modelsPerf.editingSidecarModel }
        set { modelsPerf.editingSidecarModel = newValue }
    }
    public var showMLXConverterSheet: Bool {
        get { modelsPerf.showMLXConverterSheet }
        set { modelsPerf.showMLXConverterSheet = newValue }
    }
    public var convertingSourceModel: EngineDescription? {
        get { modelsPerf.convertingSourceModel }
        set { modelsPerf.convertingSourceModel = newValue }
    }
    public var memorySnapshot: HardwareMemoryMonitor.MemorySnapshot { modelsPerf.memorySnapshot }
    public var schedulerMetrics: ComputeScheduler.SchedulerMetrics { modelsPerf.schedulerMetrics }
    public var selectedPerfWindow: PerfTimeWindow {
        get { modelsPerf.selectedPerfWindow }
        set { modelsPerf.selectedPerfWindow = newValue }
    }
    public var perfProfileSummary: SQLiteStore.PerformanceProfileSummary? { modelsPerf.perfProfileSummary }
    public var observabilityMinLevelFilter: String? {
        get { modelsPerf.observabilityMinLevelFilter }
        set { modelsPerf.observabilityMinLevelFilter = newValue }
    }
    public var observabilityCategoryFilter: String {
        get { modelsPerf.observabilityCategoryFilter }
        set { modelsPerf.observabilityCategoryFilter = newValue }
    }
    public var observabilitySearchQuery: String {
        get { modelsPerf.observabilitySearchQuery }
        set { modelsPerf.observabilitySearchQuery = newValue }
    }
    public var observabilityEntries: [HeylookObservabilityLog.Entry] { modelsPerf.observabilityEntries }
    public var boundObservabilityLoggerLevel: String {
        get { modelsPerf.boundObservabilityLoggerLevel }
        set { modelsPerf.boundObservabilityLoggerLevel = newValue }
    }
    public var activeGGUFPorts: [String: Int] {
        get { modelsPerf.activeGGUFPorts }
        set { modelsPerf.activeGGUFPorts = newValue }
    }
    public var sortedActiveGGUFPorts: [String] { modelsPerf.sortedActiveGGUFPorts }
    public var activeSubprocessCount: Int { modelsPerf.activeSubprocessCount }
    public var isServerRunning: Bool {
        get { modelsPerf.isServerRunning }
        set { modelsPerf.isServerRunning = newValue }
    }
    public var serverPort: UInt16 {
        get { modelsPerf.serverPort }
        set { modelsPerf.serverPort = newValue }
    }
    public var discoveredServers: [DiscoveredHeylookServer] { modelsPerf.discoveredServers }
    public var isBrowsingBonjour: Bool { modelsPerf.isBrowsingBonjour }
    public var remoteServerURLString: String {
        get { modelsPerf.remoteServerURLString }
        set { modelsPerf.remoteServerURLString = newValue }
    }
    public var connectedRemoteServerURL: String? { modelsPerf.connectedRemoteServerURL }

    public static func defaultDemoModels() -> [EngineDescription] { ModelsAndPerfController.defaultDemoModels() }
    public func loadSampleDemoWorkspace() async { await modelsPerf.loadSampleDemoWorkspace() }
    public func refreshMetrics() async { await modelsPerf.refreshMetrics() }
    public func refreshPerfAndObservability() { modelsPerf.refreshPerfAndObservability() }
    public func refreshObservabilityEntries() { modelsPerf.refreshObservabilityEntries() }
    public func clearObservabilityEntries() { modelsPerf.clearObservabilityEntries() }
    public func exportObservabilityJSONL() -> String { modelsPerf.exportObservabilityJSONL() }
    public func toggleServer() async { await modelsPerf.toggleServer() }
    public func startBonjourDiscovery() { modelsPerf.startBonjourDiscovery() }
    public func stopBonjourDiscovery() { modelsPerf.stopBonjourDiscovery() }
    public func connectToRemoteServer(_ urlString: String) async { await modelsPerf.connectToRemoteServer(urlString) }
    public func disconnectRemoteServer() async { await modelsPerf.disconnectRemoteServer() }
    public func assessModelRAMFit(
        _ model: EngineDescription,
        ctxSizeOverride: Int? = nil,
        sidecarOverride: ModelSidecarConfig? = nil
    ) -> GGUFSpawnConfig.RAMFitAssessment {
        modelsPerf.assessModelRAMFit(model, ctxSizeOverride: ctxSizeOverride, sidecarOverride: sidecarOverride)
    }
    public func applyQuickContextSize(_ ctxSize: Int, for model: EngineDescription) async {
        await modelsPerf.applyQuickContextSize(ctxSize, for: model)
    }
    public func addModelScanFolder(_ url: URL) async { await modelsPerf.addModelScanFolder(url) }
    public func removeModelScanFolder(_ folder: String) async { await modelsPerf.removeModelScanFolder(folder) }
    public func toggleModelPin(_ model: EngineDescription) async { await modelsPerf.toggleModelPin(model) }
    public func toggleModelLoaded(_ model: EngineDescription) async { await modelsPerf.toggleModelLoaded(model) }
    public func unloadAllModels() async { await modelsPerf.unloadAllModels() }
    public func rescanConfiguredModels() async { await modelsPerf.rescanConfiguredModels() }
    public func unloadGGUFAllocation(_ allocationEntry: String) async { await modelsPerf.unloadGGUFAllocation(allocationEntry) }
    public func hasSidecarFileOnDisk(for model: EngineDescription) -> Bool { modelsPerf.hasSidecarFileOnDisk(for: model) }
    public func loadSidecarConfig(for model: EngineDescription) -> ModelSidecarConfig { modelsPerf.loadSidecarConfig(for: model) }
    public func saveModelSidecar(_ sidecar: ModelSidecarConfig, for model: EngineDescription) async {
        await modelsPerf.saveModelSidecar(sidecar, for: model)
    }
    public func loadChatTemplateState(for model: EngineDescription) -> HeylookConfig.ChatTemplateState {
        modelsPerf.loadChatTemplateState(for: model)
    }
    public func validateJinjaTemplateSyntax(_ template: String) -> String? {
        modelsPerf.validateJinjaTemplateSyntax(template)
    }
    @discardableResult
    public func saveChatTemplateOverride(_ template: String, for model: EngineDescription) async throws -> HeylookConfig.ChatTemplateState {
        try await modelsPerf.saveChatTemplateOverride(template, for: model)
    }
    @discardableResult
    public func revertChatTemplateOverride(for model: EngineDescription) async -> HeylookConfig.ChatTemplateState {
        await modelsPerf.revertChatTemplateOverride(for: model)
    }

    // MARK: - Thin Step-3B Compatibility Bridges (Chat)

    public var conversations: [ConversationRecord] {
        get { chat.conversations }
        set { chat.conversations = newValue }
    }
    public var conversationSearchQuery: String {
        get { chat.conversationSearchQuery }
        set { chat.conversationSearchQuery = newValue }
    }
    public var filteredConversations: [ConversationRecord] { chat.filteredConversations }
    public var selectedConversationID: String? {
        get { chat.selectedConversationID }
        set { chat.selectedConversationID = newValue }
    }
    public var currentMessages: [MessageRecord] {
        get { chat.currentMessages }
        set { chat.currentMessages = newValue }
    }
    public var messages: [MessageRecord] {
        get { chat.currentMessages }
        set { chat.currentMessages = newValue }
    }
    public var draftPrompt: String {
        get { chat.draftPrompt }
        set { chat.draftPrompt = newValue }
    }
    public var draftSystemPrompt: String {
        get { chat.draftSystemPrompt }
        set { chat.draftSystemPrompt = newValue }
    }
    public var stagedAttachments: [StagedChatAttachment] {
        get { chat.stagedAttachments }
        set { chat.stagedAttachments = newValue }
    }
    public var draftAttachments: [MessagesRequest.MediaAttachment] {
        chat.stagedAttachments.map(\.wireAttachment)
    }
    public var messageAttachmentsByID: [String: [StagedChatAttachment]] { chat.messageAttachmentsByID }
    public var lastDroppedMediaCount: Int { chat.lastDroppedMediaCount }
    public var responseFormatMode: StructuredOutputMode {
        get { chat.responseFormatMode }
        set { chat.responseFormatMode = newValue }
    }
    public var responseFormatSchemaName: String {
        get { chat.responseFormatSchemaName }
        set { chat.responseFormatSchemaName = newValue }
    }
    public var responseFormatSchemaJSON: String {
        get { chat.responseFormatSchemaJSON }
        set { chat.responseFormatSchemaJSON = newValue }
    }
    public var responseFormatStrict: Bool {
        get { chat.responseFormatStrict }
        set { chat.responseFormatStrict = newValue }
    }
    public var responseFormatValidationError: String? { chat.responseFormatValidationError }
    public func buildValidatedResponseFormat() throws -> ResponseFormat? { try chat.buildValidatedResponseFormat() }
    public var isGenerating: Bool {
        get { chat.isGenerating }
        set { chat.isGenerating = newValue }
    }
    public var generatingConversationID: String? { chat.generatingConversationID }
    public var activeRequestID: String? {
        get { chat.activeRequestID }
        set { chat.activeRequestID = newValue }
    }
    public var isEndingThinking: Bool { chat.isEndingThinking }
    public var streamStatusLine: String? { chat.streamStatusLine }
    public var streamingThinking: String {
        get { chat.streamingThinking }
        set { chat.streamingThinking = newValue }
    }
    public var streamingContent: String {
        get { chat.streamingContent }
        set { chat.streamingContent = newValue }
    }
    public var streamingTPS: Double {
        get { chat.streamingTPS }
        set { chat.streamingTPS = newValue }
    }
    public var streamingTokens: Int {
        get { chat.streamingTokens }
        set { chat.streamingTokens = newValue }
    }
    public var showPromptPreview: Bool {
        get { chat.showPromptPreview }
        set { chat.showPromptPreview = newValue }
    }
    public var promptPreviewText: String { chat.promptPreviewText }
    public var promptPreviewSource: String { chat.promptPreviewSource }
    public var promptPreviewMarkers: [PromptMarkerSpan] { chat.promptPreviewMarkers }
    public var isRenderingPromptPreview: Bool { chat.isRenderingPromptPreview }
    public func schedulePromptPreviewRefresh() { chat.schedulePromptPreviewRefresh() }

    public var editingMessageID: String? {
        get { chat.editingMessageID }
        set { chat.editingMessageID = newValue }
    }
    public var editingContent: String {
        get { chat.editingContent }
        set { chat.editingContent = newValue }
    }
    public var editingThinking: String {
        get { chat.editingThinking }
        set { chat.editingThinking = newValue }
    }
    public var pendingModelSwitchTargetID: String? {
        get { chat.pendingModelSwitchTargetID }
        set { chat.pendingModelSwitchTargetID = newValue }
    }
    public var modelSwitchWarningMessage: String? {
        get { chat.modelSwitchWarningMessage }
        set { chat.modelSwitchWarningMessage = newValue }
    }

    public func selectGenerationModel(_ targetModelID: String) { chat.selectGenerationModel(targetModelID) }
    public func confirmPendingModelSwitch() { chat.confirmPendingModelSwitch() }
    public func cancelPendingModelSwitch() { chat.cancelPendingModelSwitch() }
    public func omittedMediaNote(for messageID: String) -> String? { chat.omittedMediaNote(for: messageID) }
    public func flushSystemPromptSave() { chatSettings.flushPendingPromptSave() }
    public func createNewConversation() { chat.createNewConversation() }
    public func selectConversation(_ id: String) { chat.selectConversation(id) }
    public func cloneConversation(id: String, upToPosition: Int? = nil, newTitle: String? = nil) {
        chat.cloneConversation(id: id, upToPosition: upToPosition, newTitle: newTitle)
    }
    public func renameConversation(id: String, newTitle: String) { chat.renameConversation(id: id, newTitle: newTitle) }
    public func deleteConversation(_ id: String) { chat.deleteConversation(id) }
    public func estimateStagedImagePlan(for attachment: StagedChatAttachment) -> StagedImagePlanEstimate? {
        chat.estimateStagedImagePlan(for: attachment)
    }
    public var totalStagedImageTokens: Int { chat.totalStagedImageTokens }
    public func attachChatMedia(from urls: [URL]) { chat.attachChatMedia(from: urls) }
    public func attachChatMediaFile(url: URL) { chat.attachChatMediaFile(url: url) }
    #if os(macOS)
    @discardableResult
    public func pasteAttachmentFromClipboard() -> Bool { chat.pasteAttachmentFromClipboard() }
    #endif
    public func toggleAttachmentShrink(id: String) { chat.toggleAttachmentShrink(id: id) }
    public func removeDraftAttachment(id: String) { chat.removeDraftAttachment(id: id) }
    public func removeChatAttachment(id: String) { chat.removeChatAttachment(id: id) }
    public var chatAttachmentCapabilityWarning: String? { chat.chatAttachmentCapabilityWarning }

    public var isRecordingVoice: Bool {
        get { chat.isRecordingVoice }
        set { chat.isRecordingVoice = newValue }
    }
    public var autoSendVoiceTurn: Bool {
        get { chat.autoSendVoiceTurn }
        set { chat.autoSendVoiceTurn = newValue }
    }
    public var selectedASRModelID: String? {
        get { chat.selectedASRModelID }
        set { chat.selectedASRModelID = newValue }
    }
    public var availableASRModels: [EngineDescription] { chat.availableASRModels }
    public var activeASRBackendLabel: String { chat.activeASRBackendLabel }
    public var lastVoiceTranscription: CoreAIProvider.ANESpeechTranscriptionResult? {
        get { chat.lastVoiceTranscription }
        set { chat.lastVoiceTranscription = newValue }
    }
    public var resolvedASRModelPath: String? { chat.resolvedASRModelPath }
    public var isVoiceASRConfigured: Bool { chat.isVoiceASRConfigured }
    @discardableResult
    public func transcribeVoiceToComposer(
        wavData: Data,
        languageHint: String = "en",
        autoSend: Bool? = nil
    ) async -> CoreAIProvider.ANESpeechTranscriptionResult? {
        await chat.transcribeVoiceToComposer(wavData: wavData, languageHint: languageHint, autoSend: autoSend)
    }
    @discardableResult
    public func sendVoiceTurn(
        wavData: Data,
        languageHint: String = "en"
    ) async -> CoreAIProvider.ANESpeechTranscriptionResult? {
        await chat.sendVoiceTurn(wavData: wavData, languageHint: languageHint)
    }
    public func toggleVoiceRecording(autoSendOnStop: Bool? = nil) { chat.toggleVoiceRecording(autoSendOnStop: autoSendOnStop) }
    public func sendChatPrompt() async { await chat.sendChatPrompt() }
    public func beginEditingMessage(_ message: MessageRecord) { chat.beginEditingMessage(message) }
    public func cancelEditingMessage() { chat.cancelEditingMessage() }
    public func saveEditedMessageInPlace(_ message: MessageRecord) { chat.saveEditedMessageInPlace(message) }
    public func saveAndRegenerateEditedMessage(_ message: MessageRecord) async { await chat.saveAndRegenerateEditedMessage(message) }
    public func saveAndContinueEditedMessage(_ message: MessageRecord) async { await chat.saveAndContinueEditedMessage(message) }
    public func regenerateAssistantMessage(_ message: MessageRecord) async { await chat.regenerateAssistantMessage(message) }
    public func continueAssistantMessage(_ message: MessageRecord) async { await chat.continueAssistantMessage(message) }
    public func deleteSingleMessage(id: String, conversationID: String) { chat.deleteSingleMessage(id: id, conversationID: conversationID) }
    public func streamAssistantReply(
        for convID: String,
        modelOverride: String? = nil,
        continuingAssistantMessage: MessageRecord? = nil
    ) async {
        await chat.streamAssistantReply(for: convID, modelOverride: modelOverride, continuingAssistantMessage: continuingAssistantMessage)
    }
    public func endThinkingNow() { chat.endThinkingNow() }
    public func cancelGeneration() { chat.cancelGeneration() }

    // MARK: - Thin Step-3B Compatibility Bridges (Notebook & Presets)

    public var notebooks: [NotebookRecord] {
        get { notebookPresets.notebooks }
        set { notebookPresets.notebooks = newValue }
    }
    public var selectedNotebookID: String? {
        get { notebookPresets.selectedNotebookID }
        set { notebookPresets.selectedNotebookID = newValue }
    }
    public var notebookTitle: String {
        get { notebookPresets.notebookTitle }
        set { notebookPresets.notebookTitle = newValue }
    }
    public var notebookSystemPrompt: String {
        get { notebookPresets.notebookSystemPrompt }
        set { notebookPresets.notebookSystemPrompt = newValue }
    }
    public var notebookBody: String {
        get { notebookPresets.notebookBody }
        set { notebookPresets.notebookBody = newValue }
    }
    public var notebookSelectedRange: NSRange {
        get { notebookPresets.notebookSelectedRange }
        set { notebookPresets.notebookSelectedRange = newValue }
    }
    public var isNotebookGenerating: Bool { notebookPresets.isNotebookGenerating }
    public var activeNotebookRequestID: String? { notebookPresets.activeNotebookRequestID }
    public var activeNotebookStreamingID: String? { notebookPresets.activeNotebookStreamingID }
    public var isNotebookDirty: Bool { notebookPresets.isNotebookDirty }
    public var hasNotebookSelection: Bool { notebookPresets.hasNotebookSelection }
    public func noteNotebookUserEdited() { notebookPresets.noteNotebookUserEdited() }
    public func flushNotebookSaveIfDirty() { notebookPresets.flushNotebookSaveIfDirty() }
    public func createNewNotebook() { notebookPresets.createNewNotebook() }
    public func selectNotebook(_ id: String) { notebookPresets.selectNotebook(id) }
    public func deleteNotebook(_ id: String) { notebookPresets.deleteNotebook(id) }
    public func saveNotebook() { notebookPresets.saveNotebook() }
    public func cancelNotebookGeneration() { notebookPresets.cancelNotebookGeneration() }
    public func continueNotebookGeneration() async { await notebookPresets.continueNotebookGeneration() }
    public func summarizeNotebook() async { await notebookPresets.summarizeNotebook() }
    public func analyzeNotebookSelection() async { await notebookPresets.analyzeNotebookSelection() }
    public func saveNotebookToConversations() { notebookPresets.saveNotebookToConversations() }

    public var presets: [PresetRecord] {
        get { notebookPresets.presets }
        set { notebookPresets.presets = newValue }
    }
    public var filteredPresets: [PresetRecord] { notebookPresets.filteredPresets }
    public var presetSearchQuery: String {
        get { notebookPresets.presetSearchQuery }
        set { notebookPresets.presetSearchQuery = newValue }
    }
    public var presetScopeFilter: String? {
        get { notebookPresets.presetScopeFilter }
        set { notebookPresets.presetScopeFilter = newValue }
    }
    public var selectedPresetRecordID: String? {
        get { notebookPresets.selectedPresetRecordID }
        set { notebookPresets.selectedPresetRecordID = newValue }
    }
    @discardableResult
    public func createNewPresetRecord(
        name: String,
        scope: String = "sampler",
        description: String = "",
        systemPrompt: String? = nil,
        settings: SamplerSettings = SamplerSettings()
    ) -> PresetRecord? {
        notebookPresets.createNewPresetRecord(name: name, scope: scope, description: description, systemPrompt: systemPrompt, settings: settings)
    }
    @discardableResult
    public func updatePresetRecord(_ updated: PresetRecord) -> PresetRecord? { notebookPresets.updatePresetRecord(updated) }
    @discardableResult
    public func duplicatePresetRecord(_ source: PresetRecord) -> PresetRecord? { notebookPresets.duplicatePresetRecord(source) }
    public func deletePresetByID(_ id: String) { notebookPresets.deletePresetByID(id) }

    // MARK: - Thin Step-3B Compatibility Bridges (Per-Document Settings / Samplers)

    public var pendingPromptReplacement: PendingPromptReplacement? {
        get { activeDocumentSettings.pendingPromptReplacement }
        set { activeDocumentSettings.pendingPromptReplacement = newValue }
    }
    public func confirmPendingPromptReplacement(replacePrompt: Bool) {
        activeDocumentSettings.confirmPendingPromptReplacement(replacePrompt: replacePrompt)
    }
    public func cancelPendingPromptReplacement() {
        activeDocumentSettings.cancelPendingPromptReplacement()
    }
    public var reconcileDisclosureMessage: String? { activeDocumentSettings.reconcileDisclosureMessage }

    public var selectedPresetName: String? { activeDocumentSettings.selectedPresetName }
    public var isPresetModified: Bool {
        get { activeDocumentSettings.isPresetModified }
        set { activeDocumentSettings.isPresetModified = newValue }
    }
    public var temperatureOverride: Double? {
        get { activeDocumentSettings.temperatureOverride }
        set { activeDocumentSettings.temperatureOverride = newValue }
    }
    public var topPOverride: Double? {
        get { activeDocumentSettings.topPOverride }
        set { activeDocumentSettings.topPOverride = newValue }
    }
    public var topKOverride: Int? {
        get { activeDocumentSettings.topKOverride }
        set { activeDocumentSettings.topKOverride = newValue }
    }
    public var minPOverride: Double? {
        get { activeDocumentSettings.minPOverride }
        set { activeDocumentSettings.minPOverride = newValue }
    }
    public var repetitionPenaltyOverride: Double? {
        get { activeDocumentSettings.repetitionPenaltyOverride }
        set { activeDocumentSettings.repetitionPenaltyOverride = newValue }
    }
    public var repetitionContextSizeOverride: Int? {
        get { activeDocumentSettings.repetitionContextSizeOverride }
        set { activeDocumentSettings.repetitionContextSizeOverride = newValue }
    }
    public var presencePenaltyOverride: Double? {
        get { activeDocumentSettings.presencePenaltyOverride }
        set { activeDocumentSettings.presencePenaltyOverride = newValue }
    }
    public var maxTokensOverride: Int? {
        get { activeDocumentSettings.maxTokensOverride }
        set { activeDocumentSettings.maxTokensOverride = newValue }
    }
    public var seedOverride: Int? {
        get { activeDocumentSettings.seedOverride }
        set { activeDocumentSettings.seedOverride = newValue }
    }
    public var stopSequencesOverride: [String]? {
        get { activeDocumentSettings.stopSequencesOverride }
        set { activeDocumentSettings.stopSequencesOverride = newValue }
    }
    public var thinkingEnabledOverride: Bool? {
        get { activeDocumentSettings.thinkingEnabledOverride }
        set { activeDocumentSettings.thinkingEnabledOverride = newValue }
    }
    public var thinkingDepthOverride: String? {
        get { activeDocumentSettings.thinkingDepthOverride }
        set { activeDocumentSettings.thinkingDepthOverride = newValue }
    }
    public var thinkingBudgetOverride: Int? {
        get { activeDocumentSettings.thinkingBudgetOverride }
        set { activeDocumentSettings.thinkingBudgetOverride = newValue }
    }

    public var unmaskedManualSamplerOverlay: SamplerSettings { activeDocumentSettings.unmaskedManualSamplerOverlay }
    public var currentManualParamsForPersistence: SamplerSettings? { activeDocumentSettings.currentManualParamsForPersistence }
    public var activePresetIDForPersistence: String? { activeDocumentSettings.appliedPresetID }

    public func applyDocumentSamplingState(params: SamplerSettings?, appliedPresetID: String?) {
        activeDocumentSettings.hydrateDocument(
            id: activeDocumentSettings.activeDocumentID,
            systemPrompt: activeDocumentSettings.systemPrompt,
            params: params,
            appliedPresetID: appliedPresetID,
            activeModel: modelsPerf.activeModelDescription,
            engineContract: nil
        )
    }

    public func persistActiveConversationSamplingState() {
        chatSettings.flushPendingParamsSave()
    }

    public func applyPreset(_ preset: PresetRecord) {
        _ = activeDocumentSettings.applyPreset(preset)
    }

    public func clearPresetSelection() { activeDocumentSettings.clearPresetSelection() }
    public var hasAnyManualSamplerOverride: Bool { activeDocumentSettings.hasAnyManualSamplerOverride }
    public func resetManualSamplerOverrides() { activeDocumentSettings.resetManualSamplerOverrides() }
    public func resetSamplerOverrides() { activeDocumentSettings.resetManualSamplerOverrides() }
    public func updateActivePreset() { _ = activeDocumentSettings.updateActivePreset() }
    public func saveCurrentAsPreset(name: String, scope: String? = nil, description: String = "") {
        _ = activeDocumentSettings.saveAsPreset(name: name, scope: scope, description: description)
    }
    /// Never silently deletes in 1 click — arms confirmation via `pendingConfirmation`.
    public func deleteActivePreset() { activeDocumentSettings.requestDeleteActivePreset() }

    public var activeModelSupportsThinking: Bool { activeDocumentSettings.activeModelSupportsThinking }
    public var activeModelCanDisableThinking: Bool { activeDocumentSettings.activeModelCanDisableThinking }
    public var activeModelDepthOptions: [String] { activeDocumentSettings.activeModelDepthOptions }
    public var effectiveThinkingEnabled: Bool { activeDocumentSettings.effectiveThinkingEnabled }
    public var effectiveThinkingDepth: String { activeDocumentSettings.effectiveThinkingDepth }
    public var requestSamplerOverlay: SamplerSettings { activeDocumentSettings.requestSamplerOverlay }
    public var currentResolvedSamplers: SamplerCascade.ResolvedSampling { activeDocumentSettings.currentResolvedSamplers }

    // MARK: - Thin Step-3B Compatibility Bridges (Embeddings Studio)

    public static var maxCompareMatrixItems: Int { EmbeddingsStudioController.maxCompareMatrixItems }
    public var embeddingsSubMode: EmbeddingsSubMode {
        get { embeddings.embeddingsSubMode }
        set { embeddings.embeddingsSubMode = newValue }
    }
    public var selectedBackend: EngineBackend {
        get { embeddings.selectedBackend }
        set { embeddings.selectedBackend = newValue }
    }
    public var modelComputeOptions: ModelComputeOptions {
        get { embeddings.modelComputeOptions }
        set { embeddings.modelComputeOptions = newValue }
    }
    public var embeddingServerURLString: String {
        get { embeddings.embeddingServerURLString }
        set { embeddings.embeddingServerURLString = newValue }
    }
    public var checkpointPath: String {
        get { embeddings.checkpointPath }
        set { embeddings.checkpointPath = newValue }
    }
    public var serverCapabilities: HeylookServerCapabilities? { embeddings.serverCapabilities }
    public var embeddingQuery: String {
        get { embeddings.embeddingQuery }
        set { embeddings.embeddingQuery = newValue }
    }
    public var queryAttachmentURL: URL? {
        get { embeddings.queryAttachmentURL }
        set { embeddings.queryAttachmentURL = newValue }
    }
    public var selectedDimension: Int {
        get { embeddings.selectedDimension }
        set { embeddings.selectedDimension = newValue }
    }
    public var selectedSampling: VideoWindowSampling {
        get { embeddings.selectedSampling }
        set { embeddings.selectedSampling = newValue }
    }
    public var loadAudioFromVideo: Bool {
        get { embeddings.loadAudioFromVideo }
        set { embeddings.loadAudioFromVideo = newValue }
    }
    public var selectedKindFilter: ArtifactKind? {
        get { embeddings.selectedKindFilter }
        set { embeddings.selectedKindFilter = newValue }
    }
    public var selectedTaskOverride: EmbeddingTask? {
        get { embeddings.selectedTaskOverride }
        set { embeddings.selectedTaskOverride = newValue }
    }
    public var searchHits: [SearchHit] { embeddings.searchHits }
    public var selectedHit: SearchHit? {
        get { embeddings.selectedHit }
        set { embeddings.selectedHit = newValue }
    }
    public var selectedSpanIndex: Int {
        get { embeddings.selectedSpanIndex }
        set { embeddings.selectedSpanIndex = newValue }
    }
    public var embeddingStatusMessage: String { embeddings.embeddingStatusMessage }
    public var isEmbeddingBusy: Bool { embeddings.isEmbeddingBusy }
    public var isIndexing: Bool { embeddings.isIndexing }
    public var artifactCount: Int { embeddings.artifactCount }
    public var chunkCount: Int { embeddings.chunkCount }
    public var countsByKind: [ArtifactKind: Int] { embeddings.countsByKind }
    public var watchedFolders: [VectorStore.WatchedFolder] { embeddings.watchedFolders }
    public var showEvalLabDrawer: Bool {
        get { embeddings.showEvalLabDrawer }
        set { embeddings.showEvalLabDrawer = newValue }
    }
    public var compareLeftText: String {
        get { embeddings.compareLeftText }
        set { embeddings.compareLeftText = newValue }
    }
    public var compareRightText: String {
        get { embeddings.compareRightText }
        set { embeddings.compareRightText = newValue }
    }
    public var compareCosineScore: Float? { embeddings.compareCosineScore }
    public var compareItems: [CompareMatrixItem] {
        get { embeddings.compareItems }
        set { embeddings.compareItems = newValue }
    }
    public var compareAsymmetricMode: Bool {
        get { embeddings.compareAsymmetricMode }
        set { embeddings.compareAsymmetricMode = newValue }
    }
    public var compareMatrixDimension: Int {
        get { embeddings.compareMatrixDimension }
        set { embeddings.compareMatrixDimension = newValue }
    }
    public var compareRowVectors768: [[Float]] { embeddings.compareRowVectors768 }
    public var compareColVectors768: [[Float]] { embeddings.compareColVectors768 }
    public var compareMatrixScores: [[Float]] { embeddings.compareMatrixScores }
    public var selectedCompareCellRow: Int {
        get { embeddings.selectedCompareCellRow }
        set { embeddings.selectedCompareCellRow = newValue }
    }
    public var selectedCompareCellCol: Int {
        get { embeddings.selectedCompareCellCol }
        set { embeddings.selectedCompareCellCol = newValue }
    }
    public var aneInspectionReports: [ANETowerInspector.ComponentComputeReport] { embeddings.aneInspectionReports }
    public var benchmarkCases: [EmbeddingLabRunner.BenchmarkCase] {
        get { embeddings.benchmarkCases }
        set { embeddings.benchmarkCases = newValue }
    }
    public var labMetrics: [EmbeddingLabRunner.DimensionMetrics] { embeddings.labMetrics }
    public var labConfigRuns: [EmbeddingLabConfigRun] { embeddings.labConfigRuns }
    public var evalLabSuiteReport: EvalLabSuite.SuiteReport? { embeddings.evalLabSuiteReport }
    public var isRunningEvalLab: Bool { embeddings.isRunningEvalLab }

    public var hasActiveQuery: Bool { embeddings.hasActiveQuery }
    public var effectiveQueryTask: EmbeddingTask { embeddings.effectiveQueryTask }
    public var boundEffectiveQueryTask: EmbeddingTask {
        get { embeddings.boundEffectiveQueryTask }
        set { embeddings.boundEffectiveQueryTask = newValue }
    }
    public var boundComputePreset: ComputePreset {
        get { embeddings.boundComputePreset }
        set { embeddings.boundComputePreset = newValue }
    }
    public subscript(computeUnitFor component: EmbeddingComponent) -> TowerComputeUnit {
        get { embeddings[computeUnitFor: component] }
        set { embeddings[computeUnitFor: component] = newValue }
    }
    public var activeQueryTowersLabel: String { embeddings.activeQueryTowersLabel }
    public var isActiveEmbeddingEngineSynthetic: Bool { embeddings.isActiveEmbeddingEngineSynthetic }
    public func syncEmbeddingEngineToRouter() async { await embeddings.syncEmbeddingEngineToRouter() }
    public func setupEmbeddingCoordinator() { embeddings.setupEmbeddingCoordinator() }
    public func refreshComputeReports() { embeddings.refreshComputeReports() }
    public func applyComputePreset(_ preset: ComputePreset) { embeddings.applyComputePreset(preset) }
    public func refreshEmbeddingCounts() { embeddings.refreshEmbeddingCounts() }
    public func refreshWatchedFolders() { embeddings.refreshWatchedFolders() }
    public func removeWatchedFolder(_ folder: VectorStore.WatchedFolder) { embeddings.removeWatchedFolder(folder) }
    public func reindexWatchedFolders() { embeddings.reindexWatchedFolders() }
    public func startIndexing(_ urls: [URL], persistFolders: Bool = true) { embeddings.startIndexing(urls, persistFolders: persistFolders) }
    public func cancelIndexing() { embeddings.cancelIndexing() }
    public func indexURLs(_ urls: [URL]) async { await embeddings.indexURLs(urls) }
    public func clearEmbeddingIndex() { embeddings.clearEmbeddingIndex() }
    public func seedSampleCorpusIfNeeded() async { await embeddings.seedSampleCorpusIfNeeded() }
    public func scheduleEmbeddingSearchDebounced() { embeddings.scheduleEmbeddingSearchDebounced() }
    public func runEmbeddingSearch() async { await embeddings.runEmbeddingSearch() }
    public func addCompareTextItem(_ initialText: String = "") { embeddings.addCompareTextItem(initialText) }
    public func addCompareMediaFiles(_ urls: [URL]) { embeddings.addCompareMediaFiles(urls) }
    public func removeCompareItem(id: String) { embeddings.removeCompareItem(id: id) }
    public func recomputeCompareMatrixScores() { embeddings.recomputeCompareMatrixScores() }
    public func runCompareMatrix() async { await embeddings.runCompareMatrix() }
    public func scoresAcrossDimensionsForCompareCell(row: Int, col: Int) -> [(dim: Int, score: Float)] {
        embeddings.scoresAcrossDimensionsForCompareCell(row: row, col: col)
    }
    public func compareMatrixCSV() -> String { embeddings.compareMatrixCSV() }
    public func compareMatrixJSON() -> String { embeddings.compareMatrixJSON() }
    public func sendComparePairToLab(row: Int, col: Int) async { await embeddings.sendComparePairToLab(row: row, col: col) }
    public func runPairwiseCompare() { embeddings.runPairwiseCompare() }
    public func runEmbeddingLabEvaluation() async { await embeddings.runEmbeddingLabEvaluation() }
    public func evaluateCurrentConfigInLab(customLabel: String? = nil) async { await embeddings.evaluateCurrentConfigInLab(customLabel: customLabel) }
    public func sweepComputePresetsInLab() async { await embeddings.sweepComputePresetsInLab() }
    public func clearLabConfigRuns() { embeddings.clearLabConfigRuns() }
    public func importBenchmarkSetJSON(from url: URL) { embeddings.importBenchmarkSetJSON(from: url) }
    public func sendHitToPairwiseCompare(_ hit: SearchHit) { embeddings.sendHitToPairwiseCompare(hit) }
    public func pinHitToRetrievalLab(_ hit: SearchHit) async { await embeddings.pinHitToRetrievalLab(hit) }
    public func promoteComparePairToRetrievalLab() async { await embeddings.promoteComparePairToRetrievalLab() }
    @discardableResult
    public func runFullEvalLabSuite() async -> EvalLabSuite.SuiteReport? { await embeddings.runFullEvalLabSuite() }
}
