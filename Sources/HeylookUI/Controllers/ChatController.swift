import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import Observation
import UniformTypeIdentifiers
import HeylookCore
import HeylookEngines
import HeylookServer

#if os(macOS)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

/// Resident server or fallback image plan telemetry and target resize recommendation for a staged image.
public struct StagedImagePlanState: Sendable, Equatable {
    public let tokens: Int
    public let targetWidth: Int?
    public let targetHeight: Int?
    public let currentWidth: Int
    public let currentHeight: Int
    public let originalWidth: Int
    public let originalHeight: Int
    public let isExactServerPlan: Bool
    public let note: String?

    public init(
        tokens: Int,
        targetWidth: Int? = nil,
        targetHeight: Int? = nil,
        currentWidth: Int,
        currentHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        isExactServerPlan: Bool,
        note: String? = nil
    ) {
        self.tokens = tokens
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight
        self.currentWidth = currentWidth
        self.currentHeight = currentHeight
        self.originalWidth = originalWidth
        self.originalHeight = originalHeight
        self.isExactServerPlan = isExactServerPlan
        self.note = note
    }

    public var actionLabel: String? {
        guard isExactServerPlan else { return nil }
        if currentWidth != originalWidth || currentHeight != originalHeight {
            return "Full res"
        }
        if let tw = targetWidth, let th = targetHeight, (tw != currentWidth || th != currentHeight) {
            return "Fit \(tw)×\(th)"
        }
        return nil
    }
}

/// Editable attachment entry used inside inline turn editing, preserving kept stored media blocks
/// verbatim alongside newly staged attachments.
public struct EditableChatAttachment: Identifiable, Sendable, Equatable {
    public enum Source: Sendable, Equatable {
        case kept(MessagesRequest.MediaAttachment)
        case staged(StagedChatAttachment)
    }

    public let id: String
    public let kind: MessagesRequest.MediaKind
    public let filename: String
    public let dimensionsLabel: String?
    public let previewData: Data?
    public let source: Source

    public init(kept attachment: MessagesRequest.MediaAttachment, hydrated: StagedChatAttachment? = nil) {
        self.id = hydrated?.id ?? UUID().uuidString
        self.kind = (attachment.kind == "audio") ? .audio : .image
        self.filename = attachment.filename
            ?? hydrated?.filename
            ?? (self.kind == .audio ? "audio" : "image")
        self.dimensionsLabel = hydrated?.dimensionsLabel
        if let data = hydrated?.effectiveData {
            self.previewData = data
        } else if !attachment.dataBase64.isEmpty {
            self.previewData = Data(base64Encoded: attachment.dataBase64)
        } else {
            self.previewData = nil
        }
        self.source = .kept(attachment)
    }

    public init(staged attachment: StagedChatAttachment) {
        self.id = attachment.id
        self.kind = attachment.kind
        self.filename = attachment.filename
        self.dimensionsLabel = attachment.dimensionsLabel
        self.previewData = attachment.effectiveData
        self.source = .staged(attachment)
    }

    public var contentPart: MessagesRequest.MessageTurn.ContentPart {
        switch source {
        case .kept(let att):
            return .media(att)
        case .staged(let staged):
            return .media(staged.wireAttachment)
        }
    }
}

/// Domain controller for Chat conversations, multimodal staged/historical media, voice ASR,
/// off-MainActor prompt preview, turn editing/forking/continuation, and token streaming.
///
/// Strictly uses its dedicated `chatSettings: DocumentSettingsController` instance so switching
/// surfaces or streaming mid-switch never shares or corrupts Notebook settings.
@MainActor
@Observable
public final class ChatController {
    private static let orphanConversationKey = "__orphan__"
    public static let maxStagedImages = 8
    public static let maxStagedAudio = 2

    public let store: SQLiteStore
    public let scheduler: ComputeScheduler
    public let registry: RequestRegistry
    public let chatSettings: DocumentSettingsController
    @ObservationIgnored public let localService: WorkbenchService

    // MARK: - Conversation & Message State

    public var conversations: [ConversationRecord] = []
    public var conversationSearchQuery: String = ""
    public var selectedConversationID: String?
    public var currentMessages: [MessageRecord] = []

    public var draftPrompt: String = "" {
        didSet {
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
    }

    public var draftSystemPrompt: String {
        get { chatSettings.systemPrompt }
        set { chatSettings.systemPrompt = newValue }
    }

    public var stagedAttachments: [StagedChatAttachment] = [] {
        didSet {
            scheduleStagedImagePlanRefresh()
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
    }
    public private(set) var stagedImagePlansByID: [String: StagedImagePlanState] = [:]
    public private(set) var isRefreshingImagePlans: Bool = false
    public private(set) var messageAttachmentsByID: [String: [StagedChatAttachment]] = [:]
    public private(set) var lastDroppedMediaCount: Int = 0

    // MARK: - Structured Output (`response_format`)

    public var responseFormatMode: StructuredOutputMode = .text
    public var responseFormatSchemaName: String = "structured_response"
    public var responseFormatSchemaJSON: String = """
    {
      "type": "object",
      "properties": {
        "answer": { "type": "string" }
      },
      "required": ["answer"],
      "additionalProperties": false
    }
    """
    public var responseFormatStrict: Bool = true

    // MARK: - Streaming, Prefill & Background Re-Adoption State

    public var isGenerating: Bool = false
    public private(set) var generatingConversationID: String?
    public var activeRequestID: String?
    public private(set) var isEndingThinking: Bool = false
    public private(set) var streamStatusLine: String?
    public private(set) var prefillProcessed: Int = 0
    public private(set) var prefillTotal: Int = 0
    public var streamingThinking: String = ""
    public var streamingContent: String = ""
    public var streamingTPS: Double = 0
    public var streamingTokens: Int = 0

    /// Non-destructive visual masking for `regenerate` / `continue` until `heylook_saved` commits.
    public private(set) var streamingHiddenFromPosition: Int?
    public private(set) var streamingContinueAnchorMessageID: String?
    public private(set) var pendingStreamingUserMessage: MessageRecord?

    /// Re-adopted live background generation when re-entering an active conversation.
    public private(set) var backgroundGeneratingConversationID: String?

    public var isSelectedConversationGenerating: Bool {
        guard let sel = selectedConversationID else { return false }
        return (isGenerating && generatingConversationID == sel)
            || backgroundGeneratingConversationID == sel
    }

    public var isActivelyStreamingSelectedConversation: Bool {
        guard let sel = selectedConversationID else { return false }
        return isGenerating && generatingConversationID == sel
    }

    public var visibleConversationMessages: [MessageRecord] {
        var rows = currentMessages.filter { msg in
            if let sel = selectedConversationID,
               isGenerating,
               generatingConversationID == sel {
                if let anchorID = streamingContinueAnchorMessageID, msg.id == anchorID {
                    return false
                }
                if let hideFrom = streamingHiddenFromPosition, msg.position >= hideFrom {
                    return false
                }
            }
            return true
        }
        if let pending = pendingStreamingUserMessage,
           pending.conversationID == selectedConversationID,
           !rows.contains(where: { $0.id == pending.id }) {
            rows.append(pending)
        }
        return rows
    }

    public var showPromptPreview: Bool = false {
        didSet {
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
    }
    public private(set) var promptPreviewText: String = ""
    public private(set) var promptPreviewSource: String = ""
    public private(set) var promptPreviewMarkers: [PromptMarkerSpan] = []
    public private(set) var isRenderingPromptPreview: Bool = false

    // MARK: - Inline Model Load / Reload Panel State

    public private(set) var activeAdminRow: AdminModelWireRow?
    public var loadPanelCtxSize: Int? = nil
    public var loadPanelFlashAttn: String? = nil
    public private(set) var isLoadingPanelModel: Bool = false
    public private(set) var loadPanelStatusMessage: String?
    public private(set) var loadPanelStatusIsError: Bool = false

    // MARK: - Inline Message Edit & Model Switch Guard State

    public var editingMessageID: String?
    public var editingContent: String = ""
    public var editingThinking: String = ""
    public var editingAttachments: [EditableChatAttachment] = []
    public private(set) var editorPreviewResponse: PromptPreviewWireResponse?
    public private(set) var editorPreviewError: String?
    public private(set) var isRenderingEditorPreview: Bool = false

    public var pendingModelSwitchTargetID: String?
    public var modelSwitchWarningMessage: String?

    // MARK: - Voice / ANE ASR State

    public var isRecordingVoice: Bool = false
    public var autoSendVoiceTurn: Bool = false
    public var selectedASRModelID: String? = nil {
        didSet { refreshASRAvailability() }
    }
    public private(set) var availableASRModels: [EngineDescription] = []
    public private(set) var activeASRBackendLabel: String = "Apple SpeechTranscriber (ANE)"
    public var lastVoiceTranscription: CoreAIProvider.ANESpeechTranscriptionResult?
    public private(set) var resolvedASRModelPath: String?

    public var isVoiceASRConfigured: Bool {
        resolvedASRModelPath != nil
            || CoreAIProvider.isSystemSpeechAvailable
            || connectedRemoteServerURLProvider() != nil
    }

    // MARK: - Private Tasks & Composer Stash

    @ObservationIgnored private var composerDraftByConversationID: [String: String] = [:]
    @ObservationIgnored private var composerMediaByConversationID: [String: [StagedChatAttachment]] = [:]
    @ObservationIgnored private var promptPreviewTask: Task<Void, Never>?
    @ObservationIgnored private var imagePlanTask: Task<Void, Never>?
    @ObservationIgnored private var activeGenerationTask: Task<Void, Never>?
    @ObservationIgnored private var resyncPollTask: Task<Void, Never>?
    @ObservationIgnored private var activeCancellationSignal: StreamCancellationSignal?
    @ObservationIgnored private var pendingVoiceAutoSend: Bool?
    @ObservationIgnored private var voiceRecorder: AVAudioRecorder?
    @ObservationIgnored private var voiceRecordingURL: URL?

    // MARK: - Shell Dependencies & Callbacks

    @ObservationIgnored private var remoteBackendProvider: () -> RemoteWorkbenchBackend? = { nil }
    @ObservationIgnored private var connectedRemoteServerURLProvider: () -> String? = { nil }
    @ObservationIgnored private var registeredModelsProvider: () -> [EngineDescription] = { [] }
    @ObservationIgnored private var generationModelsProvider: () -> [EngineDescription] = { [] }
    @ObservationIgnored private var activeModelIDProvider: () -> String = { "" }
    @ObservationIgnored private var activeModelProvider: () -> EngineDescription? = { nil }
    @ObservationIgnored private var activeEngineContractProvider: () async -> EngineContract? = { nil }
    @ObservationIgnored private var sidecarLoader: (EngineDescription) -> ModelSidecarConfig = { _ in ModelSidecarConfig() }

    @ObservationIgnored public var onSelectModelID: (String) -> Void = { _ in }
    @ObservationIgnored public var onStatusMessage: (String?) -> Void = { _ in }
    @ObservationIgnored public var onErrorMessage: (String?) -> Void = { _ in }
    @ObservationIgnored public var onSwitchToChatSurface: () -> Void = {}
    @ObservationIgnored public var onGenerationCompleted: (() async -> Void)?

    public init(
        store: SQLiteStore,
        scheduler: ComputeScheduler,
        registry: RequestRegistry,
        chatSettings: DocumentSettingsController,
        localService: WorkbenchService? = nil
    ) {
        self.store = store
        self.scheduler = scheduler
        self.registry = registry
        self.chatSettings = chatSettings
        self.localService = localService ?? WorkbenchService(
            store: store,
            scheduler: scheduler,
            registry: registry
        )
    }

    public func configureDependencies(
        remoteBackend: @escaping () -> RemoteWorkbenchBackend?,
        connectedRemoteServerURL: @escaping () -> String?,
        registeredModels: @escaping () -> [EngineDescription],
        generationModels: @escaping () -> [EngineDescription],
        activeModelID: @escaping () -> String,
        activeModel: @escaping () -> EngineDescription?,
        activeEngineContract: @escaping () async -> EngineContract?,
        sidecarLoader: @escaping (EngineDescription) -> ModelSidecarConfig
    ) {
        self.remoteBackendProvider = remoteBackend
        self.connectedRemoteServerURLProvider = connectedRemoteServerURL
        self.registeredModelsProvider = registeredModels
        self.generationModelsProvider = generationModels
        self.activeModelIDProvider = activeModelID
        self.activeModelProvider = activeModel
        self.activeEngineContractProvider = activeEngineContract
        self.sidecarLoader = sidecarLoader
        refreshASRAvailability()
        Task { [weak self] in
            await self?.refreshActiveAdminModel(resetDrafts: true)
        }
    }

    // MARK: - Filtered Conversations & Store Sync

    public var filteredConversations: [ConversationRecord] {
        let q = conversationSearchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return conversations }
        return conversations.filter {
            $0.title.localizedCaseInsensitiveContains(q) || $0.systemPrompt.localizedCaseInsensitiveContains(q)
        }
    }

    public func refreshFromStore() {
        conversations = (try? store.listConversations()) ?? conversations
        if selectedConversationID == nil {
            let lastID = chatSettings.readLastDocumentID()
            if let lastID, conversations.contains(where: { $0.id == lastID }) {
                selectConversation(lastID)
            } else if let first = conversations.first {
                selectConversation(first.id)
            }
        } else if let sel = selectedConversationID {
            checkBackgroundGenerationStatus(for: sel)
        }
        Task { [weak self] in
            await self?.refreshActiveAdminModel(resetDrafts: false)
        }
    }

    public func syncWithRemoteBackend(_ remote: RemoteWorkbenchBackend) async {
        do {
            let list = try await remote.listConversations()
            var loadedConvs: [ConversationRecord] = []
            var generatingIDs: Set<String> = []
            for summary in list.conversations {
                if let detail = try? await remote.getConversation(id: summary.id) {
                    let updatedMs = WireTimestamp.unixMs(fromISO8601: detail.updatedAt) ?? SQLiteStore.nowMillis()
                    let createdMs = WireTimestamp.unixMs(fromISO8601: detail.createdAt) ?? updatedMs
                    let record = ConversationRecord(
                        id: detail.id,
                        title: detail.title,
                        systemPrompt: detail.systemPrompt ?? "",
                        defaultModelID: detail.modelID,
                        params: detail.params.settings == SamplerSettings() ? nil : detail.params.settings,
                        appliedPresetID: detail.appliedPresetID,
                        createdUnixMs: createdMs,
                        updatedUnixMs: updatedMs
                    )
                    if detail.generating {
                        generatingIDs.insert(detail.id)
                    }
                    loadedConvs.append(record)
                    try? store.upsertConversation(record)
                }
            }
            conversations = loadedConvs
            let preferredID = selectedConversationID ?? chatSettings.readLastDocumentID()
            if let targetID = preferredID, loadedConvs.contains(where: { $0.id == targetID }) {
                selectedConversationID = nil
                selectConversation(targetID)
                if generatingIDs.contains(targetID) {
                    adoptBackgroundGeneration(for: targetID)
                }
            } else if let first = loadedConvs.first {
                selectedConversationID = nil
                selectConversation(first.id)
                if generatingIDs.contains(first.id) {
                    adoptBackgroundGeneration(for: first.id)
                }
            }
            await refreshActiveAdminModel(resetDrafts: true)
        } catch {
            onStatusMessage("Remote conversation list failed: \(error.localizedDescription)")
        }
    }

    public func handleStoreMutationEvent(_ event: StoreMutationEvent, activeSurface: HeylookSurface) {
        switch event {
        case .conversationUpserted(let upsertedID):
            conversations = (try? store.listConversations()) ?? conversations
            if selectedConversationID == upsertedID && activeSurface == .chat && !isActivelyStreamingSelectedConversation {
                if let conv = conversations.first(where: { $0.id == upsertedID }) {
                    chatSettings.hydrateDocument(
                        id: conv.id,
                        systemPrompt: conv.systemPrompt,
                        params: conv.params,
                        appliedPresetID: conv.appliedPresetID,
                        activeModel: activeModelProvider(),
                        engineContract: nil
                    )
                }
            }
        case .conversationDeleted:
            conversations = (try? store.listConversations()) ?? conversations
            if let sel = selectedConversationID, !conversations.contains(where: { $0.id == sel }) {
                selectedConversationID = nil
                if let next = conversations.first {
                    selectConversation(next.id)
                } else {
                    currentMessages = []
                    chatSettings.saveLastDocumentID(nil)
                }
            }
        case .messagesChanged(let convID):
            if selectedConversationID == convID {
                if isGenerating && generatingConversationID == convID {
                    // During `.append`, orchestrator commits the user turn right before streaming;
                    // clear the optimistic placeholder once SQLite already contains it, while
                    // keeping visual masking intact for `regenerate` / `continue`.
                    if let loaded = try? loadAndHydrateMessages(conversationID: convID) {
                        if let pending = pendingStreamingUserMessage,
                           loaded.last?.role == "user",
                           loaded.last?.content == pending.content {
                            pendingStreamingUserMessage = nil
                        }
                        if streamingHiddenFromPosition != nil {
                            currentMessages = loaded
                        } else if loaded.count >= currentMessages.count {
                            currentMessages = loaded
                        }
                    }
                } else if let loaded = try? loadAndHydrateMessages(conversationID: convID) {
                    currentMessages = loaded
                }
            }
        case .dataCleared:
            selectedConversationID = nil
            currentMessages = []
            messageAttachmentsByID.removeAll()
            composerDraftByConversationID.removeAll()
            composerMediaByConversationID.removeAll()
            pendingStreamingUserMessage = nil
            streamingHiddenFromPosition = nil
            streamingContinueAnchorMessageID = nil
            backgroundGeneratingConversationID = nil
            chatSettings.saveLastDocumentID(nil)
            chatSettings.hydrateDocument(
                id: nil,
                systemPrompt: "",
                params: nil,
                appliedPresetID: nil,
                activeModel: activeModelProvider(),
                engineContract: nil
            )
            refreshFromStore()
        default:
            break
        }
    }

    public func cancelBackgroundTasks() {
        promptPreviewTask?.cancel()
        promptPreviewTask = nil
        imagePlanTask?.cancel()
        imagePlanTask = nil
        resyncPollTask?.cancel()
        resyncPollTask = nil
        activeCancellationSignal?.cancel()
        activeGenerationTask?.cancel()
        activeGenerationTask = nil
    }

    // MARK: - Model Selection, Load Panel & Multimodal Switch Guard

    public var isActiveModelGGUF: Bool {
        if let row = activeAdminRow {
            return row.provider == .gguf
        }
        return activeModelProvider()?.provider == .gguf
    }

    public var activeModelContextCeiling: Int {
        if let len = activeAdminRow?.engine.context.length.value, len > 0 {
            return len
        }
        if let ctx = activeModelProvider()?.contextSize.value, ctx > 0 {
            return ctx
        }
        return 32_768
    }

    public var activeModelRunningContext: Int? {
        guard let row = activeAdminRow, row.loaded,
              let running = row.engine.context.running.value, running > 0 else {
            return nil
        }
        return running
    }

    public var activeModelStaleReloadFields: [String] {
        activeAdminRow?.staleReloadFields ?? []
    }

    public var activeModelFlashAttnDefaultLabel: String {
        if case .string(let val)? = activeAdminRow?.engine.settings["flash_attn"]?.auto, !val.isEmpty {
            return "Default (\(val))"
        }
        return "Default (on)"
    }

    public var activeModelContextStepChoices: [Int] {
        let ceiling = max(512, activeModelContextCeiling)
        var values = Set<Int>()
        var power = 4096
        while power < ceiling {
            if power >= 512 {
                values.insert(power)
            }
            power *= 2
        }
        values.insert(ceiling)
        if case .int(let cfgCtx)? = activeAdminRow?.config["ctx_size"], cfgCtx >= 512 {
            values.insert(min(cfgCtx, ceiling))
        }
        if let running = activeModelRunningContext, running >= 512 {
            values.insert(min(running, ceiling))
        }
        if let draft = loadPanelCtxSize, draft >= 512 {
            values.insert(min(draft, ceiling))
        }
        return values.sorted()
    }

    private var savedLoadCtxSize: Int? {
        if case .int(let val)? = activeAdminRow?.config["ctx_size"], val >= 512 {
            return val
        }
        return nil
    }

    private var savedLoadFlashAttn: String? {
        if case .string(let val)? = activeAdminRow?.config["flash_attn"], !val.isEmpty {
            return val
        }
        return nil
    }

    public var loadPanelHasPendingChanges: Bool {
        guard isActiveModelGGUF else { return false }
        let ceiling = max(512, activeModelContextCeiling)
        let clampedDraftCtx = loadPanelCtxSize.map { min(max(512, $0), ceiling) }
        return clampedDraftCtx != savedLoadCtxSize || loadPanelFlashAttn != savedLoadFlashAttn
    }

    public var canReloadActiveModel: Bool {
        let isLoaded = activeAdminRow?.loaded ?? activeModelProvider()?.isLoaded ?? false
        guard isLoaded else { return false }
        return !activeModelStaleReloadFields.isEmpty || loadPanelHasPendingChanges
    }

    public func refreshActiveAdminModel(resetDrafts: Bool = false) async {
        let modelID = activeModelIDProvider()
        guard !modelID.isEmpty else {
            activeAdminRow = nil
            if resetDrafts {
                loadPanelCtxSize = nil
                loadPanelFlashAttn = nil
                loadPanelStatusMessage = nil
                loadPanelStatusIsError = false
            }
            return
        }
        do {
            let row: AdminModelWireRow
            if let remote = remoteBackendProvider() {
                row = try await remote.adminModel(id: modelID)
            } else {
                row = try await localService.adminModel(id: modelID)
            }
            guard activeModelIDProvider() == modelID else { return }
            activeAdminRow = row
            if resetDrafts {
                loadPanelCtxSize = savedLoadCtxSize
                loadPanelFlashAttn = savedLoadFlashAttn
                loadPanelStatusMessage = nil
                loadPanelStatusIsError = false
            }
        } catch {
            if resetDrafts {
                activeAdminRow = nil
                loadPanelCtxSize = nil
                loadPanelFlashAttn = nil
            }
        }
    }

    public func performLoadOrReloadActiveModel(warm: Bool = true, reloadAnyway: Bool = false) async {
        let modelID = activeModelIDProvider()
        guard !modelID.isEmpty, !isLoadingPanelModel else { return }
        isLoadingPanelModel = true
        loadPanelStatusMessage = activeAdminRow?.loaded == true ? "Reloading \(modelID)…" : "Loading \(modelID)…"
        loadPanelStatusIsError = false
        defer { isLoadingPanelModel = false }

        do {
            let result: ModelLoadResult
            let isLoaded = activeAdminRow?.loaded ?? activeModelProvider()?.isLoaded ?? false
            var loadSettings: [String: JSONValue] = [:]
            if isActiveModelGGUF {
                let ceiling = max(512, activeModelContextCeiling)
                let targetCtx = loadPanelCtxSize.map { min(max(512, $0), ceiling) }
                if targetCtx != savedLoadCtxSize {
                    loadSettings["ctx_size"] = targetCtx.map { .int($0) } ?? .null
                }
                if loadPanelFlashAttn != savedLoadFlashAttn {
                    loadSettings["flash_attn"] = loadPanelFlashAttn.map { .string($0) } ?? .null
                }
            }

            if isLoaded || reloadAnyway || !loadSettings.isEmpty {
                let settingsPayload: [String: JSONValue]? = loadSettings.isEmpty ? nil : loadSettings
                if let remote = remoteBackendProvider() {
                    result = try await remote.reloadModel(id: modelID, loadSettings: settingsPayload, warm: warm)
                } else {
                    result = try await localService.reloadModel(id: modelID, loadSettings: settingsPayload, warm: warm)
                }
            } else {
                if let remote = remoteBackendProvider() {
                    result = try await remote.loadModel(id: modelID, warm: warm)
                } else {
                    result = try await localService.loadModel(id: modelID, warm: warm)
                }
            }

            await refreshActiveAdminModel(resetDrafts: true)
            scheduleStagedImagePlanRefresh()
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }

            if let warmErr = result.warmError, !warmErr.isEmpty {
                loadPanelStatusMessage = "Loaded, warm-up failed: \(warmErr)"
                loadPanelStatusIsError = true
            } else if result.warmed == true, let ms = result.warmMs {
                loadPanelStatusMessage = "Loaded & warmed in \(ms)ms"
                loadPanelStatusIsError = false
            } else {
                loadPanelStatusMessage = "Model loaded"
                loadPanelStatusIsError = false
            }
            await onGenerationCompleted?()
        } catch {
            loadPanelStatusMessage = error.localizedDescription
            loadPanelStatusIsError = true
            onStatusMessage("Model load failed for \(modelID): \(error.localizedDescription)")
        }
    }

    public func selectGenerationModel(_ targetModelID: String) {
        let currentID = activeModelIDProvider()
        guard targetModelID != currentID else {
            pendingModelSwitchTargetID = nil
            modelSwitchWarningMessage = nil
            return
        }
        guard let target = registeredModelsProvider().first(where: { $0.modelID == targetModelID }) else {
            onSelectModelID(targetModelID)
            Task { [weak self] in
                await self?.refreshActiveAdminModel(resetDrafts: true)
                self?.scheduleStagedImagePlanRefresh()
            }
            return
        }

        // Strict parity: switch warning checks ONLY persisted history media (never staged composer media).
        let dropped = incompatibilitiesForSwitch(to: target)
        if !dropped.isEmpty {
            pendingModelSwitchTargetID = targetModelID
            modelSwitchWarningMessage = "Switching to '\(targetModelID)' will omit \(dropped.joined(separator: " and ")) from persisted conversation history."
            return
        }

        commitModelSelection(targetModelID)
    }

    public func confirmPendingModelSwitch() {
        guard let target = pendingModelSwitchTargetID else { return }
        pendingModelSwitchTargetID = nil
        modelSwitchWarningMessage = nil
        commitModelSelection(target)
    }

    public func cancelPendingModelSwitch() {
        pendingModelSwitchTargetID = nil
        modelSwitchWarningMessage = nil
    }

    private func commitModelSelection(_ targetModelID: String) {
        // Mid-stream model switch stops the active stream first.
        if isGenerating || isSelectedConversationGenerating {
            cancelGeneration()
        }

        onSelectModelID(targetModelID)
        let targetDesc = registeredModelsProvider().first(where: { $0.modelID == targetModelID })
        _ = chatSettings.reconcileSettings(for: targetDesc, engineContract: nil)
        Task { [weak self] in
            guard let self else { return }
            if let contract = await self.activeEngineContractProvider() {
                _ = self.chatSettings.reconcileSettings(for: targetDesc, engineContract: contract)
            }
            await self.refreshActiveAdminModel(resetDrafts: true)
            self.scheduleStagedImagePlanRefresh()
        }

        if let convID = selectedConversationID,
           var conv = try? store.getConversation(id: convID) {
            conv.defaultModelID = targetModelID
            conv.updatedUnixMs = SQLiteStore.nowMillis()
            try? store.upsertConversation(conv)
            conversations = (try? store.listConversations()) ?? conversations
            if let remote = remoteBackendProvider() {
                Task { _ = try? await remote.updateConversation(id: convID, modelID: targetModelID) }
            }
        }
        if showPromptPreview {
            schedulePromptPreviewRefresh()
        }
    }

    private func incompatibilitiesForSwitch(to target: EngineDescription) -> [String] {
        let convImages = countPersistedHistoryAttachments(ofKind: .image)
        let convAudio = countPersistedHistoryAttachments(ofKind: .audio)
        var dropped: [String] = []
        if convImages > 0 && !target.supportsVision.value {
            dropped.append("\(convImages) image\(convImages == 1 ? "" : "s") (no vision)")
        }
        if convAudio > 0 && !target.supportsAudio.value {
            dropped.append("\(convAudio) audio clip\(convAudio == 1 ? "" : "s") (no audio)")
        }
        return dropped
    }

    private func countPersistedHistoryAttachments(ofKind kind: MessagesRequest.MediaKind) -> Int {
        var count = 0
        for msg in currentMessages {
            if let hydrated = messageAttachmentsByID[msg.id], !hydrated.isEmpty {
                count += hydrated.filter { $0.kind == kind }.count
            } else if let blocks = msg.contentBlocks {
                count += blocks.filter { part in
                    if case .media(let att) = part {
                        return (att.kind == "audio" ? MessagesRequest.MediaKind.audio : .image) == kind
                    }
                    return false
                }.count
            }
        }
        return count
    }

    public func omittedMediaNote(for messageID: String) -> String? {
        let msg = currentMessages.first(where: { $0.id == messageID })
        let hydrated = messageAttachmentsByID[messageID] ?? []
        let mediaParts: [MessagesRequest.MediaKind]
        if !hydrated.isEmpty {
            mediaParts = hydrated.map(\.kind)
        } else if let blocks = msg?.contentBlocks {
            mediaParts = blocks.compactMap { part in
                if case .media(let att) = part {
                    return att.kind == "audio" ? .audio : .image
                }
                return nil
            }
        } else {
            return nil
        }
        guard !mediaParts.isEmpty else { return nil }
        let activeDesc = activeModelProvider()
        let supportsVis = activeDesc?.supportsVision.value ?? false
        let supportsAud = activeDesc?.supportsAudio.value ?? false
        let droppedImages = supportsVis ? 0 : mediaParts.filter { $0 == .image }.count
        let droppedAudio = supportsAud ? 0 : mediaParts.filter { $0 == .audio }.count
        var parts: [String] = []
        if droppedImages > 0 {
            parts.append("\(droppedImages) image\(droppedImages == 1 ? "" : "s")")
        }
        if droppedAudio > 0 {
            parts.append("\(droppedAudio) audio clip\(droppedAudio == 1 ? "" : "s")")
        }
        guard !parts.isEmpty else { return nil }
        let selectedID = activeModelIDProvider()
        let modelLabel = selectedID.isEmpty ? "current model" : selectedID
        return "\(parts.joined(separator: " & ")) not sent to \(modelLabel)"
    }

    // MARK: - Conversation Lifecycle, Background Re-Adoption & System Prompt Persistence

    private func stashComposerState(for key: String) {
        composerDraftByConversationID[key] = draftPrompt
        composerMediaByConversationID[key] = stagedAttachments
    }

    private func restoreComposerState(for key: String) {
        draftPrompt = composerDraftByConversationID[key] ?? ""
        stagedAttachments = composerMediaByConversationID[key] ?? []
    }

    public func createNewConversation() {
        chatSettings.flushPendingWrites()
        if let prevID = selectedConversationID {
            stashComposerState(for: prevID)
        }

        let inherited = chatSettings.presetForNewDoc()
        let selectedID = activeModelIDProvider()
        let model = selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID

        if let remote = remoteBackendProvider() {
            Task { [weak self] in
                guard let self else { return }
                do {
                    let detail = try await remote.createConversation(
                        title: "New Chat",
                        modelID: model.isEmpty ? nil : model,
                        systemPrompt: inherited.systemPrompt.isEmpty ? nil : inherited.systemPrompt,
                        params: inherited.params,
                        appliedPresetID: inherited.appliedPresetID
                    )
                    let updatedMs = WireTimestamp.unixMs(fromISO8601: detail.updatedAt) ?? SQLiteStore.nowMillis()
                    let createdMs = WireTimestamp.unixMs(fromISO8601: detail.createdAt) ?? updatedMs
                    let record = ConversationRecord(
                        id: detail.id,
                        title: detail.title,
                        systemPrompt: detail.systemPrompt ?? "",
                        defaultModelID: detail.modelID,
                        params: detail.params.settings == SamplerSettings() ? nil : detail.params.settings,
                        appliedPresetID: detail.appliedPresetID,
                        createdUnixMs: createdMs,
                        updatedUnixMs: updatedMs
                    )
                    try? store.upsertConversation(record)
                    conversations = (try? store.listConversations()) ?? conversations
                    chatSettings.clearPreDocDraftPrompt()
                    selectConversation(record.id)
                } catch {
                    onStatusMessage("Could not create remote conversation: \(error.localizedDescription)")
                }
            }
            return
        }

        do {
            let conv = ConversationRecord(
                title: "New Chat",
                systemPrompt: inherited.systemPrompt,
                defaultModelID: model.isEmpty ? nil : model,
                params: inherited.params,
                appliedPresetID: inherited.appliedPresetID
            )
            try store.upsertConversation(conv)
            conversations = try store.listConversations()
            chatSettings.clearPreDocDraftPrompt()
            selectedConversationID = conv.id
            chatSettings.saveLastDocumentID(conv.id)
            currentMessages = []
            editingMessageID = nil
            draftPrompt = ""
            stagedAttachments = []
            chatSettings.hydrateDocument(
                id: conv.id,
                systemPrompt: conv.systemPrompt,
                params: conv.params,
                appliedPresetID: conv.appliedPresetID,
                activeModel: activeModelProvider(),
                engineContract: nil
            )
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        } catch {
            onStatusMessage("Could not create conversation: \(error.localizedDescription)")
        }
    }

    public func selectConversation(_ id: String) {
        if selectedConversationID == id && !currentMessages.isEmpty {
            checkBackgroundGenerationStatus(for: id)
            return
        }
        chatSettings.flushPendingWrites()
        if let prevID = selectedConversationID, prevID != id {
            stashComposerState(for: prevID)
        } else if selectedConversationID == nil {
            stashComposerState(for: Self.orphanConversationKey)
        }

        selectedConversationID = id
        chatSettings.saveLastDocumentID(id)
        editingMessageID = nil
        editingAttachments = []
        editorPreviewResponse = nil
        editorPreviewError = nil
        pendingModelSwitchTargetID = nil
        modelSwitchWarningMessage = nil
        backgroundGeneratingConversationID = nil
        resyncPollTask?.cancel()
        resyncPollTask = nil

        do {
            currentMessages = try loadAndHydrateMessages(conversationID: id)
        } catch {
            onStatusMessage("Failed to load messages: \(error.localizedDescription)")
        }

        if let conv = conversations.first(where: { $0.id == id }) ?? (try? store.getConversation(id: id)) {
            if let defModel = conv.defaultModelID,
               !defModel.isEmpty,
               defModel != activeModelIDProvider(),
               generationModelsProvider().contains(where: { $0.modelID == defModel }) {
                onSelectModelID(defModel)
                Task { [weak self] in
                    await self?.refreshActiveAdminModel(resetDrafts: true)
                    self?.scheduleStagedImagePlanRefresh()
                }
            }
            chatSettings.hydrateDocument(
                id: conv.id,
                systemPrompt: conv.systemPrompt,
                params: conv.params,
                appliedPresetID: conv.appliedPresetID,
                activeModel: activeModelProvider(),
                engineContract: nil
            )
            if let remote = remoteBackendProvider() {
                Task { [weak self] in
                    guard let self, let detail = try? await remote.getConversation(id: id) else { return }
                    let hydrated = detail.messages.map { wire in
                        Self.messageRecord(from: wire, conversationID: id)
                    }
                    if self.selectedConversationID == id {
                        self.currentMessages = hydrated
                        self.hydrateRemoteMessageAttachments(from: hydrated)
                        if detail.generating && !(self.isGenerating && self.generatingConversationID == id) {
                            self.adoptBackgroundGeneration(for: id)
                        }
                    }
                }
            }
        } else {
            chatSettings.hydrateDocument(
                id: id,
                systemPrompt: "",
                params: nil,
                appliedPresetID: nil,
                activeModel: activeModelProvider(),
                engineContract: nil
            )
        }
        restoreComposerState(for: id)
        checkBackgroundGenerationStatus(for: id)

        if showPromptPreview {
            schedulePromptPreviewRefresh()
        }
    }

    private func checkBackgroundGenerationStatus(for conversationID: String) {
        if !(isGenerating && generatingConversationID == conversationID) && registry.isConversationGenerating(conversationID) {
            adoptBackgroundGeneration(for: conversationID)
        }
    }

    private func adoptBackgroundGeneration(for conversationID: String) {
        backgroundGeneratingConversationID = conversationID
        if selectedConversationID == conversationID {
            onStatusMessage("Generation running in background — syncing when complete.")
        }
        startResyncUntilSettled(for: conversationID)
    }

    private func startResyncUntilSettled(for conversationID: String) {
        resyncPollTask?.cancel()
        resyncPollTask = Task { [weak self] in
            guard let self else { return }
            let maxAttempts = 120
            for _ in 0..<maxAttempts {
                try? await Task.sleep(for: .milliseconds(750))
                guard !Task.isCancelled else { return }

                if let remote = self.remoteBackendProvider() {
                    if let detail = try? await remote.getConversation(id: conversationID) {
                        let hydrated = detail.messages.map { Self.messageRecord(from: $0, conversationID: conversationID) }
                        if self.selectedConversationID == conversationID {
                            self.currentMessages = hydrated
                            self.hydrateRemoteMessageAttachments(from: hydrated)
                        }
                        if !detail.generating {
                            if self.backgroundGeneratingConversationID == conversationID {
                                self.backgroundGeneratingConversationID = nil
                            }
                            self.onStatusMessage(nil)
                            await self.onGenerationCompleted?()
                            return
                        }
                    }
                } else {
                    if let loaded = try? self.loadAndHydrateMessages(conversationID: conversationID),
                       self.selectedConversationID == conversationID {
                        self.currentMessages = loaded
                    }
                    if !self.registry.isConversationGenerating(conversationID) {
                        if self.backgroundGeneratingConversationID == conversationID {
                            self.backgroundGeneratingConversationID = nil
                        }
                        self.onStatusMessage(nil)
                        await self.onGenerationCompleted?()
                        return
                    }
                }
            }
            if self.backgroundGeneratingConversationID == conversationID {
                self.backgroundGeneratingConversationID = nil
            }
        }
    }

    public func cloneConversation(id: String, upToPosition: Int? = nil, newTitle: String? = nil) {
        chatSettings.flushPendingWrites()
        if let remote = remoteBackendProvider() {
            Task { [weak self] in
                guard let self else { return }
                do {
                    let detail = try await remote.cloneConversation(id: id, title: newTitle, upToPosition: upToPosition)
                    let updatedMs = WireTimestamp.unixMs(fromISO8601: detail.updatedAt) ?? SQLiteStore.nowMillis()
                    let createdMs = WireTimestamp.unixMs(fromISO8601: detail.createdAt) ?? updatedMs
                    let record = ConversationRecord(
                        id: detail.id,
                        title: detail.title,
                        systemPrompt: detail.systemPrompt ?? "",
                        defaultModelID: detail.modelID,
                        params: detail.params.settings == SamplerSettings() ? nil : detail.params.settings,
                        appliedPresetID: detail.appliedPresetID,
                        createdUnixMs: createdMs,
                        updatedUnixMs: updatedMs
                    )
                    try? store.upsertConversation(record)
                    conversations = (try? store.listConversations()) ?? conversations
                    selectConversation(detail.id)
                    onSwitchToChatSurface()
                    if let upToPosition {
                        onStatusMessage("Forked conversation at turn #\(upToPosition + 1) → \"\(detail.title)\".")
                    } else {
                        onStatusMessage("Cloned conversation → \"\(detail.title)\".")
                    }
                } catch {
                    onErrorMessage(error.localizedDescription)
                    onStatusMessage("Clone failed: \(error.localizedDescription)")
                }
            }
            return
        }

        do {
            let cloned = try store.cloneConversation(id: id, upToPosition: upToPosition, newTitle: newTitle)
            conversations = try store.listConversations()
            selectConversation(cloned.id)
            onSwitchToChatSurface()
            if let upToPosition {
                onStatusMessage("Forked conversation at turn #\(upToPosition + 1) → \"\(cloned.title)\".")
            } else {
                onStatusMessage("Cloned conversation → \"\(cloned.title)\".")
            }
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Clone failed: \(error.localizedDescription)")
        }
    }

    public func renameConversation(id: String, newTitle: String) {
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, var conv = try? store.getConversation(id: id) else { return }
        conv.title = trimmed
        conv.updatedUnixMs = SQLiteStore.nowMillis()
        try? store.upsertConversation(conv)
        conversations = (try? store.listConversations()) ?? conversations
        if let remote = remoteBackendProvider() {
            Task { _ = try? await remote.updateConversation(id: id, title: trimmed) }
        }
    }

    public func deleteConversation(_ id: String) {
        if backgroundGeneratingConversationID == id {
            backgroundGeneratingConversationID = nil
            resyncPollTask?.cancel()
            resyncPollTask = nil
        }
        let msgIDs = ((try? store.listMessages(conversationID: id)) ?? []).map(\.id)
        for mID in msgIDs {
            messageAttachmentsByID[mID] = nil
        }
        composerDraftByConversationID[id] = nil
        composerMediaByConversationID[id] = nil

        _ = try? store.deleteConversation(id: id)
        conversations = (try? store.listConversations()) ?? []
        if let remote = remoteBackendProvider() {
            Task { try? await remote.deleteConversation(id: id) }
        }
        if selectedConversationID == id {
            selectedConversationID = nil
            chatSettings.saveLastDocumentID(nil)
            if let next = conversations.first {
                selectConversation(next.id)
            } else {
                currentMessages = []
                draftPrompt = ""
                stagedAttachments = []
                chatSettings.hydrateDocument(
                    id: nil,
                    systemPrompt: "",
                    params: nil,
                    appliedPresetID: nil,
                    activeModel: activeModelProvider(),
                    engineContract: nil
                )
            }
        }
    }

    public func persistConversationParamsCallback(docID: String, params: SamplerSettings?, appliedPresetID: String?) {
        guard var conv = try? store.getConversation(id: docID) else { return }
        guard conv.params != params || conv.appliedPresetID != appliedPresetID else { return }
        conv.params = params
        conv.appliedPresetID = appliedPresetID
        conv.updatedUnixMs = SQLiteStore.nowMillis()
        try? store.upsertConversation(conv)
        conversations = (try? store.listConversations()) ?? conversations
        if let remote = remoteBackendProvider() {
            Task {
                _ = try? await remote.updateConversation(
                    id: docID,
                    params: params,
                    appliedPresetID: appliedPresetID
                )
            }
        }
    }

    public func persistConversationSystemPromptCallback(docID: String, systemPrompt: String) {
        guard var conv = try? store.getConversation(id: docID) else { return }
        guard conv.systemPrompt != systemPrompt else { return }
        conv.systemPrompt = systemPrompt
        conv.updatedUnixMs = SQLiteStore.nowMillis()
        try? store.upsertConversation(conv)
        conversations = (try? store.listConversations()) ?? conversations
        if let remote = remoteBackendProvider() {
            Task { _ = try? await remote.updateConversation(id: docID, systemPrompt: systemPrompt) }
        }
    }

    // MARK: - Off-MainActor Debounced Prompt Preview

    public func schedulePromptPreviewRefresh() {
        promptPreviewTask?.cancel()
        guard showPromptPreview else { return }

        let activeModel = activeModelProvider()
        let sysPrompt = chatSettings.systemPrompt
        let msgs = currentMessages
        let attachmentsMap = messageAttachmentsByID
        let staged = stagedAttachments
        let draft = draftPrompt
        let thinkOn = chatSettings.effectiveThinkingEnabled
        let depthOptions = chatSettings.activeModelDepthOptions
        let depthVal = depthOptions.isEmpty ? nil : chatSettings.effectiveThinkingDepth
        let thinkingBudget = chatSettings.thinkingBudgetOverride
        let remoteBackend = remoteBackendProvider()
        let activeConvID = selectedConversationID
        let service = localService

        isRenderingPromptPreview = true
        promptPreviewTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(150))
            guard !Task.isCancelled else { return }

            if let activeConvID, let activeModel, activeModel.isLoaded {
                let userParts: [MessagesRequest.MessageTurn.ContentPart] =
                    staged.map { .media($0.wireAttachment) }
                    + (draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? [] : [.text(draft)])
                let payload = PromptPreviewRequestPayload(
                    mode: .append,
                    userContent: draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : draft,
                    userParts: userParts,
                    overrides: PromptPreviewOverrides(
                        model: activeModel.modelID,
                        system: sysPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : sysPrompt,
                        thinking: activeModel.supportsThinking.value ? .bool(thinkOn) : nil,
                        thinkingDepth: depthVal,
                        thinkingBudget: thinkingBudget
                    )
                )
                let serverPreview: PromptPreviewWireResponse?
                if activeModel.provider == .remote, let remoteBackend {
                    serverPreview = try? await remoteBackend.previewPrompt(conversationID: activeConvID, request: payload)
                } else {
                    serverPreview = try? await service.previewPrompt(conversationID: activeConvID, request: payload)
                }
                if let serverPreview {
                    guard !Task.isCancelled else { return }
                    self?.promptPreviewText = serverPreview.prompt
                    self?.promptPreviewSource = Self.formatPromptPreviewSummary(serverPreview)
                    self?.promptPreviewMarkers = Self.computeMarkerSpans(in: serverPreview.prompt, markers: serverPreview.markers)
                    self?.isRenderingPromptPreview = false
                    return
                }
            }

            let dirURL = activeModel.map { HeylookConfig.resolveModelDirectoryURL(from: $0.modelPath) }
            let supportsVis = activeModel?.supportsVision.value
            let supportsAud = activeModel?.supportsAudio.value

            let rendered = await Task.detached(priority: .utility) { () -> (prompt: String, source: String, markers: [PromptMarkerSpan]) in
                var turns: [MessagesRequest.MessageTurn] = msgs.map { msg in
                    let turnStaged = attachmentsMap[msg.id] ?? []
                    return MessagesRequest.MessageTurn(
                        role: msg.role,
                        content: msg.content,
                        thinking: msg.thinking,
                        attachments: turnStaged.map(\.wireAttachment),
                        parts: turnStaged.isEmpty ? msg.contentBlocks : nil
                    )
                }
                let trimmedDraft = draft.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedDraft.isEmpty || !staged.isEmpty {
                    turns.append(
                        MessagesRequest.MessageTurn(
                            role: "user",
                            content: draft,
                            thinking: nil,
                            attachments: staged.map(\.wireAttachment)
                        )
                    )
                }

                let trimmedSys = sysPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                do {
                    let detailed = try HeylookConfig.renderPromptDetailed(
                        in: dirURL,
                        system: trimmedSys.isEmpty ? nil : sysPrompt,
                        turns: turns,
                        thinkingEnabled: thinkOn,
                        thinkingDepth: depthVal,
                        modelSupportsVision: supportsVis,
                        modelSupportsAudio: supportsAud
                    )
                    var badges: [String] = [detailed.source]
                    if detailed.continuation {
                        badges.append("continuation")
                    }
                    if !detailed.markers.isEmpty {
                        badges.append("\(detailed.markers.count) markers")
                    }
                    if detailed.droppedMedia > 0 {
                        badges.append("\(detailed.droppedMedia) media omitted")
                    }
                    if detailed.unrenderedMedia > 0 {
                        badges.append("\(detailed.unrenderedMedia) media unrendered by template")
                    }
                    return (detailed.prompt, badges.joined(separator: " • "), detailed.markers)
                } catch {
                    return ("/* Jinja template error: \(error.localizedDescription) */", "template_error", [])
                }
            }.value

            guard !Task.isCancelled else { return }
            self?.promptPreviewText = rendered.prompt
            self?.promptPreviewSource = rendered.source
            self?.promptPreviewMarkers = rendered.markers
            self?.isRenderingPromptPreview = false
        }
    }

    public static func formatPromptPreviewSummary(_ response: PromptPreviewWireResponse) -> String {
        var badges: [String] = [response.source]
        if let tokens = response.tokenCount {
            badges.append("\(tokens) tokens")
        }
        if let cont = response.continuation, !cont.isEmpty {
            badges.append("continue:\(cont)")
        }
        if !response.markers.isEmpty {
            badges.append("\(response.markers.count) markers")
        }
        if response.droppedMedia.total > 0 {
            badges.append("\(response.droppedMedia.total) media omitted")
        }
        if response.unrenderedMedia > 0 {
            badges.append("\(response.unrenderedMedia) unrendered media")
        }
        return badges.joined(separator: " • ")
    }

    public static func computeMarkerSpans(in prompt: String, markers: [String]) -> [PromptMarkerSpan] {
        let unique = Array(Set(markers.filter { !$0.isEmpty })).sorted { $0.count > $1.count }
        guard !prompt.isEmpty, !unique.isEmpty else { return [] }
        var spans: [PromptMarkerSpan] = []
        let chars = Array(prompt)
        var occupied = Array(repeating: false, count: chars.count)
        for token in unique {
            let tokenChars = Array(token)
            let tLen = tokenChars.count
            guard tLen > 0, tLen <= chars.count else { continue }
            var i = 0
            while i <= chars.count - tLen {
                if !occupied[i] && Array(chars[i..<(i + tLen)]) == tokenChars {
                    var free = true
                    for k in i..<(i + tLen) where occupied[k] {
                        free = false
                        break
                    }
                    if free {
                        for k in i..<(i + tLen) {
                            occupied[k] = true
                        }
                        spans.append(PromptMarkerSpan(start: i, end: i + tLen, token: token))
                        i += tLen
                        continue
                    }
                }
                i += 1
            }
        }
        return spans.sorted { $0.start < $1.start }
    }

    // MARK: - Multimodal Attachments, Resident Image Plan, Structured Output & Paste

    public func scheduleStagedImagePlanRefresh() {
        imagePlanTask?.cancel()
        let imageAttachments = stagedAttachments.filter { $0.kind == .image }
        guard !imageAttachments.isEmpty else {
            stagedImagePlansByID = [:]
            isRefreshingImagePlans = false
            return
        }

        let activeModel = activeModelProvider()
        let remote = remoteBackendProvider()
        let service = localService

        guard let activeModel else {
            stagedImagePlansByID = [:]
            isRefreshingImagePlans = false
            return
        }
        guard activeModel.supportsVision.value else {
            var next: [String: StagedImagePlanState] = [:]
            for att in imageAttachments {
                let dims = Self.currentPixelSize(for: att)
                let orig = Self.originalPixelSize(for: att)
                next[att.id] = StagedImagePlanState(
                    tokens: 0,
                    currentWidth: dims.width,
                    currentHeight: dims.height,
                    originalWidth: orig.width,
                    originalHeight: orig.height,
                    isExactServerPlan: false,
                    note: "Model does not support images"
                )
            }
            stagedImagePlansByID = next
            isRefreshingImagePlans = false
            return
        }

        guard activeModel.isLoaded else {
            var next: [String: StagedImagePlanState] = [:]
            for att in imageAttachments {
                let dims = Self.currentPixelSize(for: att)
                let orig = Self.originalPixelSize(for: att)
                next[att.id] = StagedImagePlanState(
                    tokens: 0,
                    currentWidth: dims.width,
                    currentHeight: dims.height,
                    originalWidth: orig.width,
                    originalHeight: orig.height,
                    isExactServerPlan: false,
                    note: "Load \(activeModel.modelID) for exact image tokens"
                )
            }
            stagedImagePlansByID = next
            isRefreshingImagePlans = false
            return
        }

        isRefreshingImagePlans = true
        imagePlanTask = Task { [weak self] in
            var sizesPayload: [[Int]] = []
            var entries: [(id: String, curW: Int, curH: Int, origW: Int, origH: Int, hasOrigPair: Bool)] = []
            for att in imageAttachments {
                let cur = Self.currentPixelSize(for: att)
                let orig = Self.originalPixelSize(for: att)
                guard cur.width > 0, cur.height > 0 else { continue }
                let hasOrig = orig.width > 0 && orig.height > 0 && (orig.width != cur.width || orig.height != cur.height)
                entries.append((att.id, cur.width, cur.height, orig.width, orig.height, hasOrig))
                sizesPayload.append([cur.width, cur.height])
                if hasOrig {
                    sizesPayload.append([orig.width, orig.height])
                }
            }
            guard !sizesPayload.isEmpty else {
                self?.stagedImagePlansByID = [:]
                self?.isRefreshingImagePlans = false
                return
            }

            do {
                let plan: ImagePlanWireResponse
                if activeModel.provider == .remote, let remote {
                    plan = try await remote.planImages(id: activeModel.modelID, sizes: sizesPayload)
                } else {
                    plan = try await service.planImages(id: activeModel.modelID, sizes: sizesPayload)
                }
                guard !Task.isCancelled else { return }

                var result: [String: StagedImagePlanState] = [:]
                var rowIdx = 0
                for entry in entries {
                    guard rowIdx < plan.images.count else { break }
                    let curRow = plan.images[rowIdx]
                    rowIdx += 1
                    let origRow: ImagePlanRow
                    if entry.hasOrigPair, rowIdx < plan.images.count {
                        origRow = plan.images[rowIdx]
                        rowIdx += 1
                    } else {
                        origRow = curRow
                    }
                    let targetW = (origRow.target?.count == 2) ? origRow.target?[0] : ((curRow.target?.count == 2) ? curRow.target?[0] : nil)
                    let targetH = (origRow.target?.count == 2) ? origRow.target?[1] : ((curRow.target?.count == 2) ? curRow.target?[1] : nil)
                    result[entry.id] = StagedImagePlanState(
                        tokens: curRow.tokens,
                        targetWidth: targetW,
                        targetHeight: targetH,
                        currentWidth: entry.curW,
                        currentHeight: entry.curH,
                        originalWidth: entry.origW > 0 ? entry.origW : entry.curW,
                        originalHeight: entry.origH > 0 ? entry.origH : entry.curH,
                        isExactServerPlan: true,
                        note: nil
                    )
                }
                self?.stagedImagePlansByID = result
                self?.isRefreshingImagePlans = false
            } catch {
                guard !Task.isCancelled else { return }
                var unavailable: [String: StagedImagePlanState] = [:]
                for entry in entries {
                    unavailable[entry.id] = StagedImagePlanState(
                        tokens: 0,
                        currentWidth: entry.curW,
                        currentHeight: entry.curH,
                        originalWidth: entry.origW > 0 ? entry.origW : entry.curW,
                        originalHeight: entry.origH > 0 ? entry.origH : entry.curH,
                        isExactServerPlan: false,
                        note: "Cost unavailable"
                    )
                }
                self?.stagedImagePlansByID = unavailable
                self?.isRefreshingImagePlans = false
            }
        }
    }

    private static func currentPixelSize(for att: StagedChatAttachment) -> (width: Int, height: Int) {
        if att.useShrunk2048, let sw = att.shrunkPixelWidth, let sh = att.shrunkPixelHeight {
            return (sw, sh)
        }
        return (att.originalPixelWidth ?? 0, att.originalPixelHeight ?? 0)
    }

    private static func originalPixelSize(for att: StagedChatAttachment) -> (width: Int, height: Int) {
        (att.originalPixelWidth ?? 0, att.originalPixelHeight ?? 0)
    }

    public func estimateStagedImagePlan(for attachment: StagedChatAttachment) -> StagedImagePlanEstimate? {
        guard attachment.kind == .image,
              let exact = stagedImagePlansByID[attachment.id],
              exact.isExactServerPlan else {
            return nil
        }
        return StagedImagePlanEstimate(
            resizedWidth: exact.targetWidth ?? exact.currentWidth,
            resizedHeight: exact.targetHeight ?? exact.currentHeight,
            estimatedTokens: exact.tokens,
            tokenCap: exact.tokens
        )
    }

    public var totalStagedImageTokens: Int {
        stagedAttachments.reduce(0) { sum, att in
            guard att.kind == .image,
                  let plan = stagedImagePlansByID[att.id],
                  plan.isExactServerPlan else {
                return sum
            }
            return sum + plan.tokens
        }
    }

    public var hasExactTotalStagedImageTokens: Bool {
        let images = stagedAttachments.filter { $0.kind == .image }
        guard !images.isEmpty else { return false }
        return images.allSatisfy { stagedImagePlansByID[$0.id]?.isExactServerPlan == true }
    }

    public var responseFormatValidationError: String? {
        guard responseFormatMode == .jsonSchema else { return nil }
        let trimmedName = responseFormatSchemaName.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedName.isEmpty {
            return "Schema name cannot be empty."
        }
        let raw = responseFormatSchemaJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = raw.data(using: .utf8) else {
            return "Invalid UTF-8 in JSON schema."
        }
        guard let decoded = try? JSONDecoder().decode(JSONValue.self, from: data),
              case .object = decoded else {
            return "JSON Schema must be a valid JSON object ({ ... })."
        }
        return nil
    }

    public func buildValidatedResponseFormat() throws -> ResponseFormat? {
        switch responseFormatMode {
        case .text:
            return nil
        case .jsonObject:
            return ResponseFormat(type: "json_object")
        case .jsonSchema:
            if let err = responseFormatValidationError {
                throw ProviderExecutionError.invalidRequest(detail: err)
            }
            let data = Data(responseFormatSchemaJSON.utf8)
            let schemaObj = try JSONDecoder().decode(JSONValue.self, from: data)
            let trimmedName = responseFormatSchemaName.trimmingCharacters(in: .whitespacesAndNewlines)
            return ResponseFormat(
                type: "json_schema",
                jsonSchema: ResponseFormat.JSONSchemaSpec(
                    name: trimmedName,
                    schema: schemaObj,
                    strict: responseFormatStrict
                )
            )
        }
    }

    public func attachChatMedia(from urls: [URL]) {
        for url in urls {
            attachChatMediaFile(url: url)
        }
    }

    public func attachChatMediaFile(url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer {
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
        guard let data = try? Data(contentsOf: url), !data.isEmpty else {
            onStatusMessage("Unable to read attachment '\(url.lastPathComponent)'.")
            return
        }
        let ext = url.pathExtension.lowercased()
        let audioExts: Set<String> = ["wav", "mp3", "m4a", "aac", "flac"]
        let imageExts: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "gif", "tiff"]

        let kind: MessagesRequest.MediaKind
        let mime: String
        if audioExts.contains(ext) {
            kind = .audio
            switch ext {
            case "mp3": mime = "audio/mp3"
            case "m4a", "aac": mime = "audio/mp4"
            case "flac": mime = "audio/flac"
            default: mime = "audio/wav"
            }
        } else if imageExts.contains(ext) {
            kind = .image
            switch ext {
            case "png": mime = "image/png"
            case "webp": mime = "image/webp"
            case "gif": mime = "image/gif"
            default: mime = "image/jpeg"
            }
        } else {
            onStatusMessage("Unsupported chat attachment '\(url.lastPathComponent)' (attach an image or audio file).")
            return
        }

        _ = stageMediaDataIfAllowed(kind: kind, mimeType: mime, filename: url.lastPathComponent, data: data)
    }

    @discardableResult
    public func stageMediaDataIfAllowed(
        kind: MessagesRequest.MediaKind,
        mimeType: String,
        filename: String,
        data: Data
    ) -> Bool {
        if let active = activeModelProvider() {
            if kind == .image && !active.supportsVision.value {
                onStatusMessage("Model '\(active.modelID)' does not accept image attachments.")
                return false
            }
            if kind == .audio && !active.supportsAudio.value {
                onStatusMessage("Model '\(active.modelID)' does not accept audio attachments.")
                return false
            }
        }
        let currentCount = stagedAttachments.filter { $0.kind == kind }.count
        let limit = (kind == .image) ? Self.maxStagedImages : Self.maxStagedAudio
        if currentCount >= limit {
            let noun = (kind == .image) ? "images" : "audio clips"
            onStatusMessage("Maximum \(limit) \(noun) per message.")
            return false
        }
        stagedAttachments.append(
            StagedChatAttachment.make(
                kind: kind,
                mimeType: mimeType,
                filename: filename,
                data: data
            )
        )
        return true
    }

    #if os(macOS)
    @discardableResult
    public func pasteAttachmentFromClipboard() -> Bool {
        let pb = NSPasteboard.general
        if let urls = pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL],
           !urls.isEmpty {
            let validExts: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "gif", "tiff", "wav", "mp3", "m4a", "aac", "flac"]
            let mediaURLs = urls.filter { validExts.contains($0.pathExtension.lowercased()) }
            if !mediaURLs.isEmpty {
                for url in mediaURLs {
                    attachChatMediaFile(url: url)
                }
                return true
            }
        }
        if let pngData = pb.data(forType: .png), !pngData.isEmpty {
            let stamp = Int(Date().timeIntervalSince1970)
            return stageMediaDataIfAllowed(
                kind: .image,
                mimeType: "image/png",
                filename: "pasted-\(stamp).png",
                data: pngData
            )
        }
        if let tiffData = pb.data(forType: .tiff),
           let bitmap = NSBitmapImageRep(data: tiffData),
           let pngData = bitmap.representation(using: .png, properties: [:]) {
            let stamp = Int(Date().timeIntervalSince1970)
            return stageMediaDataIfAllowed(
                kind: .image,
                mimeType: "image/png",
                filename: "pasted-\(stamp).png",
                data: pngData
            )
        }
        return false
    }
    #endif

    #if canImport(UIKit)
    @discardableResult
    public func pasteAttachmentFromIOSClipboard() -> Bool {
        let pb = UIPasteboard.general
        if let image = pb.image, let pngData = image.pngData(), !pngData.isEmpty {
            let stamp = Int(Date().timeIntervalSince1970)
            return stageMediaDataIfAllowed(
                kind: .image,
                mimeType: "image/png",
                filename: "pasted-\(stamp).png",
                data: pngData
            )
        }
        return false
    }
    #endif

    public func toggleAttachmentShrink(id: String) {
        guard let idx = stagedAttachments.firstIndex(where: { $0.id == id }) else { return }
        let att = stagedAttachments[idx]
        if let plan = stagedImagePlansByID[id], plan.isExactServerPlan {
            if att.useShrunk2048 || plan.currentWidth != plan.originalWidth || plan.currentHeight != plan.originalHeight {
                stagedAttachments[idx].useShrunk2048 = false
                scheduleStagedImagePlanRefresh()
                return
            }
            if let tw = plan.targetWidth, let th = plan.targetHeight,
               (tw != plan.currentWidth || th != plan.currentHeight),
               let resized = Self.resizeImageDataExact(att.originalData, width: tw, height: th, mimeType: att.mimeType) {
                stagedAttachments[idx] = StagedChatAttachment(
                    id: att.id,
                    kind: att.kind,
                    mimeType: att.mimeType,
                    filename: att.filename,
                    originalData: att.originalData,
                    shrunk2048Data: resized.data,
                    shrunkMimeType: resized.mimeType,
                    originalPixelWidth: att.originalPixelWidth,
                    originalPixelHeight: att.originalPixelHeight,
                    shrunkPixelWidth: tw,
                    shrunkPixelHeight: th,
                    useShrunk2048: true
                )
                scheduleStagedImagePlanRefresh()
                return
            }
        }
        guard att.canToggleShrink else { return }
        stagedAttachments[idx].useShrunk2048.toggle()
        scheduleStagedImagePlanRefresh()
    }

    private static func resizeImageDataExact(
        _ data: Data,
        width targetW: Int,
        height targetH: Int,
        mimeType: String
    ) -> (data: Data, mimeType: String)? {
        guard targetW > 0, targetH > 0,
              let src = CGImageSourceCreateWithData(data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            return nil
        }
        let colorSpace = cgImage.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: targetW,
            height: targetH,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetW, height: targetH))
        guard let scaled = ctx.makeImage() else { return nil }

        let preferPNG = mimeType.lowercased().contains("png")
        let utType = preferPNG ? UTType.png.identifier : UTType.jpeg.identifier
        let outMime = preferPNG ? "image/png" : "image/jpeg"
        let outData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(outData, utType as CFString, 1, nil) else {
            return nil
        }
        let options: [CFString: Any] = preferPNG ? [:] : [kCGImageDestinationLossyCompressionQuality: 0.9]
        CGImageDestinationAddImage(dest, scaled, options as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return (outData as Data, outMime)
    }

    public func removeDraftAttachment(id: String) {
        stagedAttachments.removeAll { $0.id == id }
    }

    public func removeChatAttachment(id: String) {
        removeDraftAttachment(id: id)
    }

    public var chatAttachmentCapabilityWarning: String? {
        guard !stagedAttachments.isEmpty else { return nil }
        guard let desc = activeModelProvider() else {
            return "Select a generation model before sending media attachments."
        }
        let hasImage = stagedAttachments.contains { $0.kind == .image }
        let hasAudio = stagedAttachments.contains { $0.kind == .audio }
        if hasImage && !desc.supportsVision.value {
            return "Model '\(desc.modelID)' does not support image inputs (supports_vision = false). Remove staged images or switch models."
        }
        if hasAudio && !desc.supportsAudio.value {
            return "Model '\(desc.modelID)' does not support audio inputs (supports_audio = false). Remove staged audio or switch models."
        }
        return nil
    }

    // MARK: - ANE / Remote Speech Recognition

    public func refreshASRAvailability() {
        let fm = FileManager.default
        let registered = registeredModelsProvider()
        var models: [EngineDescription] = registered.filter {
            $0.provider == .coreai && $0.isSpeechRecognizer
        }
        if CoreAIProvider.isSystemSpeechAvailable,
           !models.contains(where: { $0.modelID == CoreAIProvider.systemSpeechModelID }) {
            models.append(CoreAIProvider.systemSpeechDescription())
        }
        availableASRModels = models

        var resolvedPath: String?
        if let envPath = ProcessInfo.processInfo.environment["HEYLOOK_ASR_MODEL_PATH"],
           !envPath.isEmpty,
           fm.fileExists(atPath: (envPath as NSString).expandingTildeInPath) {
            resolvedPath = (envPath as NSString).expandingTildeInPath
        } else if fm.fileExists(atPath: "models/ane-speech-whisper.aimodel") {
            resolvedPath = "models/ane-speech-whisper.aimodel"
        } else if let candidate = registered.first(where: { desc in
            guard desc.isSpeechRecognizer else { return false }
            let expanded = (desc.modelPath as NSString).expandingTildeInPath
            return fm.fileExists(atPath: expanded)
        }) {
            resolvedPath = (candidate.modelPath as NSString).expandingTildeInPath
        }
        resolvedASRModelPath = resolvedPath

        let chosenModel: EngineDescription? = {
            if let explicitID = selectedASRModelID {
                return models.first(where: { $0.modelID == explicitID })
            }
            if CoreAIProvider.isCoreAISpeechLinked,
               let coreAISpeech = models.first(where: { $0.modelID != CoreAIProvider.systemSpeechModelID }) {
                return coreAISpeech
            }
            return models.first(where: { $0.modelID == CoreAIProvider.systemSpeechModelID }) ?? models.first
        }()

        if let chosenModel {
            switch chosenModel.architecture.value.lowercased() {
            case "parakeet_tdt":
                activeASRBackendLabel = "Parakeet TDT (ANE)"
            case "whisper":
                activeASRBackendLabel = "Whisper (ANE)"
            case "apple_speech_transcriber":
                activeASRBackendLabel = "Apple SpeechTranscriber (ANE)"
            default:
                activeASRBackendLabel = "\(chosenModel.modelID) (ANE)"
            }
        } else if resolvedPath != nil {
            activeASRBackendLabel = "CoreAI Speech (ANE)"
        } else if connectedRemoteServerURLProvider() != nil {
            activeASRBackendLabel = "Remote Server ASR"
        } else {
            activeASRBackendLabel = "Voice ASR Unavailable"
        }
    }

    @discardableResult
    public func transcribeVoiceToComposer(
        wavData: Data,
        languageHint: String = "en",
        autoSend: Bool? = nil
    ) async -> CoreAIProvider.ANESpeechTranscriptionResult? {
        guard isVoiceASRConfigured else {
            onStatusMessage("Voice input unavailable: neither Apple SpeechTranscriber nor a CoreAISpeech .aimodel bundle is available.")
            return nil
        }

        do {
            let transcript: CoreAIProvider.ANESpeechTranscriptionResult
            if let remoteBackend = remoteBackendProvider() {
                let payload = AudioTranscriptionRequest(
                    audioBase64: wavData.base64EncodedString(),
                    languageHint: languageHint,
                    model: selectedASRModelID
                )
                transcript = try await remoteBackend.transcribeAudio(payload)
            } else {
                transcript = try await scheduler.transcribeVoice(
                    wavData: wavData,
                    languageHint: languageHint,
                    preferredASRModelID: selectedASRModelID
                )
            }

            lastVoiceTranscription = transcript
            await onGenerationCompleted?()

            let cleanedText = transcript.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleanedText.isEmpty else {
                onStatusMessage("No speech detected in recording.")
                return transcript
            }

            let trimmedDraft = draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedDraft.isEmpty {
                draftPrompt = cleanedText
            } else {
                draftPrompt = "\(trimmedDraft) \(cleanedText)"
            }

            let shouldAutoSend = autoSend ?? autoSendVoiceTurn
            if shouldAutoSend {
                await sendChatPrompt()
            } else {
                onStatusMessage(String(
                    format: "Dictated %.1fs via %@ (%@).",
                    transcript.durationSeconds,
                    transcript.modelID,
                    transcript.computePlacement.rawValue.uppercased()
                ))
            }
            return transcript
        } catch {
            onStatusMessage("ANE voice transcription failed: \(error.localizedDescription)")
            return nil
        }
    }

    @discardableResult
    public func sendVoiceTurn(
        wavData: Data,
        languageHint: String = "en"
    ) async -> CoreAIProvider.ANESpeechTranscriptionResult? {
        await transcribeVoiceToComposer(
            wavData: wavData,
            languageHint: languageHint,
            autoSend: true
        )
    }

    public func toggleVoiceRecording(autoSendOnStop: Bool? = nil) {
        Task { await performToggleVoiceRecording(autoSendOnStop: autoSendOnStop) }
    }

    private func performToggleVoiceRecording(autoSendOnStop: Bool? = nil) async {
        if isRecordingVoice {
            voiceRecorder?.stop()
            voiceRecorder = nil
            isRecordingVoice = false
            let resolvedAutoSend = autoSendOnStop ?? pendingVoiceAutoSend ?? autoSendVoiceTurn
            pendingVoiceAutoSend = nil
            guard let url = voiceRecordingURL else { return }
            voiceRecordingURL = nil
            defer { try? FileManager.default.removeItem(at: url) }
            if let wavData = try? Data(contentsOf: url), wavData.count > 44 {
                _ = await transcribeVoiceToComposer(wavData: wavData, autoSend: resolvedAutoSend)
            }
            return
        }

        guard isVoiceASRConfigured else {
            onStatusMessage("Voice input unavailable: no ASR runtime configured.")
            return
        }

        let granted = await AVAudioApplication.requestRecordPermission()
        guard granted else {
            onStatusMessage("Microphone permission denied. Grant Microphone access in System Settings > Privacy & Security.")
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
                pendingVoiceAutoSend = autoSendOnStop
                voiceRecordingURL = tempURL
                voiceRecorder = recorder
                isRecordingVoice = true
                onStatusMessage("Recording 16 kHz mono voice input (\(activeASRBackendLabel))…")
            }
        } catch {
            onStatusMessage("Microphone capture failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Send, Canonical Saga Streaming & Turn Editing/Regeneration

    public var activeModelCanEndThinking: Bool {
        guard chatSettings.activeModelSupportsThinking else { return false }
        if let adminEnforced = activeAdminRow?.engine.thinking?.budget?.enforced {
            return adminEnforced
        }
        if let ctrlEnforced = activeModelProvider()?.thinkingControls?.budgetEnforced {
            return ctrlEnforced
        }
        if let row = activeAdminRow {
            return row.capabilities.contains("thinking_budget")
        }
        return false
    }

    public var activeModelBudgetDisabledReason: String? {
        guard chatSettings.activeModelSupportsThinking else { return nil }
        if let reason = activeAdminRow?.engine.thinking?.budget?.reason,
           activeAdminRow?.engine.thinking?.budget?.enforced == false {
            return reason
        }
        if let ctrl = activeModelProvider()?.thinkingControls,
           ctrl.budgetEnforced == false {
            return ctrl.budgetReason
        }
        return nil
    }

    public var canContinueEditedUserTurn: Bool {
        if let row = activeAdminRow {
            return row.provider != .gguf
        }
        return activeModelProvider()?.provider != .gguf
    }

    public func sendChatPrompt() async {
        // Pressing Send while a model-switch confirmation banner is pending commits the switch first.
        if pendingModelSwitchTargetID != nil {
            confirmPendingModelSwitch()
        }

        let promptText = draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !promptText.isEmpty || !stagedAttachments.isEmpty else { return }
        guard !isGenerating && !isSelectedConversationGenerating else {
            onStatusMessage("Generation already in progress in this conversation.")
            return
        }

        if let warning = chatAttachmentCapabilityWarning {
            onStatusMessage(warning)
            return
        }
        if let schemaErr = responseFormatValidationError {
            onStatusMessage("Structured output schema invalid: \(schemaErr)")
            return
        }

        let selectedID = activeModelIDProvider()
        let modelID = selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID
        guard !modelID.isEmpty else {
            onStatusMessage("No generation model selected or configured. Add scan_folders in heylook.toml or load the sample workspace.")
            return
        }

        if let existingConvID = selectedConversationID, registry.isConversationGenerating(existingConvID) {
            adoptBackgroundGeneration(for: existingConvID)
            onStatusMessage("Conversation is already generating in background.")
            return
        }

        chatSettings.flushPendingWrites()
        let pendingStaged = stagedAttachments
        let rawDraftPrompt = draftPrompt

        if selectedConversationID == nil {
            let inherited = chatSettings.presetForNewDoc()
            let initialTitle = Self.derivedConversationTitle(promptText: promptText, attachments: pendingStaged)

            if let remote = remoteBackendProvider() {
                do {
                    let detail = try await remote.createConversation(
                        title: initialTitle,
                        modelID: modelID,
                        systemPrompt: inherited.systemPrompt.isEmpty ? nil : inherited.systemPrompt,
                        params: inherited.params,
                        appliedPresetID: inherited.appliedPresetID
                    )
                    let updatedMs = WireTimestamp.unixMs(fromISO8601: detail.updatedAt) ?? SQLiteStore.nowMillis()
                    let createdMs = WireTimestamp.unixMs(fromISO8601: detail.createdAt) ?? updatedMs
                    let conv = ConversationRecord(
                        id: detail.id,
                        title: detail.title,
                        systemPrompt: detail.systemPrompt ?? "",
                        defaultModelID: detail.modelID,
                        params: detail.params.settings == SamplerSettings() ? nil : detail.params.settings,
                        appliedPresetID: detail.appliedPresetID,
                        createdUnixMs: createdMs,
                        updatedUnixMs: updatedMs
                    )
                    try? store.upsertConversation(conv)
                    conversations = (try? store.listConversations()) ?? conversations
                    chatSettings.clearPreDocDraftPrompt()
                    selectedConversationID = conv.id
                    chatSettings.saveLastDocumentID(conv.id)
                    chatSettings.hydrateDocument(
                        id: conv.id,
                        systemPrompt: conv.systemPrompt,
                        params: conv.params,
                        appliedPresetID: conv.appliedPresetID,
                        activeModel: activeModelProvider(),
                        engineContract: nil
                    )
                    currentMessages = []
                } catch {
                    onStatusMessage("Failed to create remote conversation: \(error.localizedDescription)")
                    return
                }
            } else {
                let conv = ConversationRecord(
                    title: initialTitle,
                    systemPrompt: inherited.systemPrompt,
                    defaultModelID: modelID,
                    params: inherited.params,
                    appliedPresetID: inherited.appliedPresetID
                )
                try? store.upsertConversation(conv)
                conversations = (try? store.listConversations()) ?? []
                chatSettings.clearPreDocDraftPrompt()
                selectedConversationID = conv.id
                chatSettings.saveLastDocumentID(conv.id)
                chatSettings.hydrateDocument(
                    id: conv.id,
                    systemPrompt: conv.systemPrompt,
                    params: conv.params,
                    appliedPresetID: conv.appliedPresetID,
                    activeModel: activeModelProvider(),
                    engineContract: nil
                )
                currentMessages = []
            }
        }
        guard let convID = selectedConversationID else { return }

        if var conv = try? store.getConversation(id: convID),
           currentMessages.isEmpty,
           (conv.title == "New Chat" || conv.title == "New Conversation") {
            let derived = Self.derivedConversationTitle(promptText: promptText, attachments: pendingStaged)
            if derived != conv.title {
                conv.title = derived
                conv.defaultModelID = modelID
                conv.updatedUnixMs = SQLiteStore.nowMillis()
                try? store.upsertConversation(conv)
                conversations = (try? store.listConversations()) ?? conversations
                if let remote = remoteBackendProvider() {
                    Task { _ = try? await remote.updateConversation(id: convID, title: derived, modelID: modelID) }
                }
            }
        }

        // Media-only messages store empty string "" + contentBlocks (never "[Attached: ...]").
        let storedContent = promptText
        let wireParts: [MessagesRequest.MessageTurn.ContentPart] =
            pendingStaged.map { .media($0.wireAttachment) }
            + (storedContent.isEmpty ? [] : [.text(storedContent)])
        let pendingUser = MessageRecord(
            conversationID: convID,
            position: (currentMessages.map(\.position).max() ?? -1) + 1,
            role: "user",
            content: storedContent,
            modelID: modelID,
            contentBlocks: wireParts.isEmpty ? nil : wireParts
        )
        if !pendingStaged.isEmpty {
            messageAttachmentsByID[pendingUser.id] = pendingStaged
        }
        pendingStreamingUserMessage = pendingUser

        draftPrompt = ""
        stagedAttachments = []
        composerDraftByConversationID[convID] = ""
        composerMediaByConversationID[convID] = []
        composerDraftByConversationID[Self.orphanConversationKey] = nil
        composerMediaByConversationID[Self.orphanConversationKey] = nil

        await streamConversationGeneration(
            for: convID,
            mode: .append,
            messageID: nil,
            content: storedContent,
            userParts: wireParts,
            modelOverride: modelID,
            rollbackDraft: (rawDraftPrompt, pendingStaged, pendingUser.id)
        )
    }

    private static func derivedConversationTitle(
        promptText: String,
        attachments: [StagedChatAttachment]
    ) -> String {
        if !promptText.isEmpty {
            return String(promptText.prefix(44))
        }
        let hasImages = attachments.contains { $0.kind == .image }
        let hasAudio = attachments.contains { $0.kind == .audio }
        if hasImages && hasAudio { return "Multimodal chat" }
        if hasImages { return "Image chat" }
        if hasAudio { return "Audio chat" }
        return "New Chat"
    }

    private func loadAndHydrateMessages(conversationID: String) throws -> [MessageRecord] {
        let loaded = try store.listMessages(conversationID: conversationID, hydrateMedia: true)
        hydrateAttachmentsIfNeeded(from: loaded)
        return loaded
    }

    private func hydrateRemoteMessageAttachments(from messages: [MessageRecord]) {
        hydrateAttachmentsIfNeeded(from: messages)
    }

    private func hydrateAttachmentsIfNeeded(from messages: [MessageRecord]) {
        for msg in messages {
            guard messageAttachmentsByID[msg.id] == nil, let blocks = msg.contentBlocks else { continue }
            var hydrated: [StagedChatAttachment] = []
            for block in blocks {
                if case .media(let att) = block,
                   !att.dataBase64.isEmpty,
                   let data = Data(base64Encoded: att.dataBase64),
                   !data.isEmpty {
                    let kind: MessagesRequest.MediaKind = (att.kind == "audio") ? .audio : .image
                    hydrated.append(
                        StagedChatAttachment.make(
                            kind: kind,
                            mimeType: att.mediaType,
                            filename: att.filename ?? (kind == .audio ? "audio.wav" : "image.jpg"),
                            data: data
                        )
                    )
                }
            }
            if !hydrated.isEmpty {
                messageAttachmentsByID[msg.id] = hydrated
            }
        }
    }

    private static func messageRecord(from wire: WireMessageDTO, conversationID: String) -> MessageRecord {
        let createdMs = WireTimestamp.unixMs(fromISO8601: wire.createdAt) ?? SQLiteStore.nowMillis()
        let updatedMs = WireTimestamp.unixMs(fromISO8601: wire.updatedAt)
        return MessageRecord(
            id: wire.id,
            conversationID: conversationID,
            position: wire.position,
            role: wire.role,
            content: wire.content,
            thinking: wire.thinking,
            modelID: wire.modelID,
            tokenCount: wire.stats?.outputTokens,
            generationTPS: wire.stats?.generationTPS,
            ttftMs: wire.stats?.ttftMs,
            cacheReport: wire.stats?.cache,
            speculativeReport: wire.stats?.speculative,
            contentBlocks: wire.contentBlocks.isEmpty ? nil : wire.contentBlocks,
            createdUnixMs: createdMs,
            updatedUnixMs: updatedMs,
            promptTPS: wire.stats?.promptTPS,
            peakMemoryGB: wire.stats?.peakMemoryGB,
            queueWaitMs: wire.stats?.queueWaitMs,
            stats: wire.stats
        )
    }

    private static func rebuiltContentBlocks(
        for attachments: [EditableChatAttachment],
        newContent: String,
        newThinking: String?
    ) -> [MessagesRequest.MessageTurn.ContentPart]? {
        let hasThinking = (newThinking?.isEmpty == false)
        if attachments.isEmpty && !hasThinking {
            return nil
        }
        var rebuilt: [MessagesRequest.MessageTurn.ContentPart] = []
        if let th = newThinking, !th.isEmpty {
            rebuilt.append(.thinking(th))
        }
        for att in attachments {
            rebuilt.append(att.contentPart)
        }
        if !newContent.isEmpty || rebuilt.isEmpty {
            rebuilt.append(.text(newContent))
        }
        return rebuilt
    }

    public func canEditMedia(for message: MessageRecord) -> Bool {
        if message.role == "user" {
            return true
        }
        let provider = activeAdminRow?.provider ?? activeModelProvider()?.provider
        return provider != .mlx
    }

    public func beginEditingMessage(_ message: MessageRecord) {
        editingMessageID = message.id
        editingContent = message.content
        editingThinking = message.thinking ?? ""
        editorPreviewResponse = nil
        editorPreviewError = nil

        let hydrated = messageAttachmentsByID[message.id] ?? []
        var items: [EditableChatAttachment] = []
        if let blocks = message.contentBlocks, !blocks.isEmpty {
            var hydratedIdx = 0
            for block in blocks {
                if case .media(let att) = block {
                    let matched: StagedChatAttachment?
                    if hydratedIdx < hydrated.count {
                        matched = hydrated[hydratedIdx]
                        hydratedIdx += 1
                    } else {
                        matched = nil
                    }
                    items.append(EditableChatAttachment(kept: att, hydrated: matched))
                }
            }
        } else {
            items = hydrated.map { EditableChatAttachment(staged: $0) }
        }
        editingAttachments = items
    }

    public func cancelEditingMessage() {
        editingMessageID = nil
        editingContent = ""
        editingThinking = ""
        editingAttachments = []
        editorPreviewResponse = nil
        editorPreviewError = nil
        isRenderingEditorPreview = false
    }

    public func removeEditingAttachment(id: String) {
        editingAttachments.removeAll { $0.id == id }
    }

    public func moveEditingAttachment(id: String, delta: Int) {
        guard let idx = editingAttachments.firstIndex(where: { $0.id == id }) else { return }
        let dest = idx + delta
        guard editingAttachments.indices.contains(dest) else { return }
        editingAttachments.swapAt(idx, dest)
    }

    public func addMediaToEditingMessage(from urls: [URL], message: MessageRecord) {
        guard canEditMedia(for: message) else {
            onStatusMessage("MLX templates reject media inside assistant turns; attach media on user turns.")
            return
        }
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer {
                if scoped { url.stopAccessingSecurityScopedResource() }
            }
            guard let data = try? Data(contentsOf: url), !data.isEmpty else { continue }
            let ext = url.pathExtension.lowercased()
            let audioExts: Set<String> = ["wav", "mp3", "m4a", "aac", "flac"]
            let imageExts: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "gif", "tiff"]

            let kind: MessagesRequest.MediaKind
            let mime: String
            if audioExts.contains(ext) {
                kind = .audio
                switch ext {
                case "mp3": mime = "audio/mp3"
                case "m4a", "aac": mime = "audio/mp4"
                case "flac": mime = "audio/flac"
                default: mime = "audio/wav"
                }
            } else if imageExts.contains(ext) {
                kind = .image
                switch ext {
                case "png": mime = "image/png"
                case "webp": mime = "image/webp"
                case "gif": mime = "image/gif"
                default: mime = "image/jpeg"
                }
            } else {
                onStatusMessage("Unsupported media format '\(url.lastPathComponent)'.")
                continue
            }

            if let active = activeModelProvider() {
                if kind == .image && !active.supportsVision.value {
                    onStatusMessage("Model '\(active.modelID)' does not accept image attachments.")
                    continue
                }
                if kind == .audio && !active.supportsAudio.value {
                    onStatusMessage("Model '\(active.modelID)' does not accept audio attachments.")
                    continue
                }
            }
            let currentCount = editingAttachments.filter { $0.kind == kind }.count
            let limit = (kind == .image) ? Self.maxStagedImages : Self.maxStagedAudio
            if currentCount >= limit {
                onStatusMessage("Maximum \(limit) \(kind == .image ? "images" : "audio clips") per turn.")
                continue
            }

            let staged = StagedChatAttachment.make(
                kind: kind,
                mimeType: mime,
                filename: url.lastPathComponent,
                data: data
            )
            editingAttachments.append(EditableChatAttachment(staged: staged))
        }
    }

    public func previewEditorPrompt(for message: MessageRecord) async {
        guard let convID = selectedConversationID else { return }
        isRenderingEditorPreview = true
        editorPreviewError = nil

        let modelID = activeModelIDProvider()
        let activeModel = activeModelProvider()
        let sysPrompt = chatSettings.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let depthOptions = chatSettings.activeModelDepthOptions
        let depthVal = depthOptions.isEmpty ? nil : chatSettings.effectiveThinkingDepth

        let payload = PromptPreviewRequestPayload(
            mode: .continue,
            messageID: message.id,
            overrides: PromptPreviewOverrides(
                model: modelID.isEmpty ? nil : modelID,
                system: sysPrompt.isEmpty ? nil : chatSettings.systemPrompt,
                thinking: (activeModel?.supportsThinking.value == true) ? .bool(chatSettings.effectiveThinkingEnabled) : nil,
                thinkingDepth: depthVal,
                thinkingBudget: chatSettings.thinkingBudgetOverride
            ),
            edits: [message.id: editingContent]
        )

        do {
            let response: PromptPreviewWireResponse
            if let remote = remoteBackendProvider() {
                response = try await remote.previewPrompt(conversationID: convID, request: payload)
            } else {
                response = try await localService.previewPrompt(conversationID: convID, request: payload)
            }
            editorPreviewResponse = response
            editorPreviewError = nil
        } catch {
            editorPreviewResponse = nil
            editorPreviewError = error.localizedDescription
        }
        isRenderingEditorPreview = false
    }

    public func saveEditedMessageInPlace(_ message: MessageRecord) {
        guard let convID = selectedConversationID else { return }
        if registry.isConversationGenerating(convID) || isSelectedConversationGenerating {
            onStatusMessage("Cannot edit while conversation is actively generating.")
            return
        }
        Task { [weak self] in
            guard let self else { return }
            do {
                _ = try await self.persistEditedMessageAsync(message, conversationID: convID)
                self.cancelEditingMessage()
                if self.showPromptPreview {
                    self.schedulePromptPreviewRefresh()
                }
            } catch {
                self.onErrorMessage(error.localizedDescription)
                self.onStatusMessage("Failed to save message edit: \(error.localizedDescription)")
            }
        }
    }

    @discardableResult
    private func persistEditedMessageAsync(
        _ message: MessageRecord,
        conversationID convID: String
    ) async throws -> MessageRecord {
        let trimmedThinking = editingThinking.trimmingCharacters(in: .whitespacesAndNewlines)
        var updated = message
        updated.content = editingContent
        if message.role == "assistant" {
            updated.thinking = trimmedThinking.isEmpty ? nil : editingThinking
        }
        updated.contentBlocks = Self.rebuiltContentBlocks(
            for: editingAttachments,
            newContent: updated.content,
            newThinking: updated.thinking
        )

        if let remote = remoteBackendProvider() {
            let wire = try await remote.updateMessage(
                conversationID: convID,
                messageID: updated.id,
                content: updated.content,
                contentBlocks: updated.contentBlocks ?? [.text(updated.content)],
                thinking: updated.role == "assistant" ? updated.thinking : nil,
                truncateAfter: false
            )
            let mapped = Self.messageRecord(from: wire, conversationID: convID)
            messageAttachmentsByID[mapped.id] = nil
            hydrateAttachmentsIfNeeded(from: [mapped])
            if let idx = currentMessages.firstIndex(where: { $0.id == mapped.id }) {
                currentMessages[idx] = mapped
            }
            _ = try? store.appendMessage(mapped)
            return mapped
        } else {
            let saved = try store.appendMessage(updated)
            messageAttachmentsByID[saved.id] = nil
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            return saved
        }
    }

    public func saveAndRegenerateEditedMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating, !isSelectedConversationGenerating else { return }
        if registry.isConversationGenerating(convID) {
            onStatusMessage("Conversation is already generating.")
            return
        }
        do {
            if message.role == "assistant" {
                cancelEditingMessage()
                streamingHiddenFromPosition = message.position
                await streamConversationGeneration(
                    for: convID,
                    mode: .regenerate,
                    messageID: message.id
                )
                return
            }

            let priorIndex = currentMessages.firstIndex(where: { $0.id == message.id })
            let nextMessage = priorIndex.flatMap { idx in
                (idx + 1 < currentMessages.count) ? currentMessages[idx + 1] : nil
            }
            let updated = try await persistEditedMessageAsync(message, conversationID: convID)
            cancelEditingMessage()

            if let nextMessage {
                // Non-destructive saga: hide downstream visually; SQLite tail commits atomically on heylook_saved.
                streamingHiddenFromPosition = nextMessage.position
                await streamConversationGeneration(
                    for: convID,
                    mode: .regenerate,
                    messageID: nextMessage.id
                )
            } else {
                streamingHiddenFromPosition = updated.position + 1
                await streamConversationGeneration(
                    for: convID,
                    mode: .continue,
                    messageID: updated.id
                )
            }
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Failed to regenerate message: \(error.localizedDescription)")
        }
    }

    public func saveAndContinueEditedMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating, !isSelectedConversationGenerating else { return }
        if registry.isConversationGenerating(convID) {
            onStatusMessage("Conversation is already generating.")
            return
        }
        if message.role == "user" && !canContinueEditedUserTurn {
            onStatusMessage("GGUF models cannot continue directly from a user turn; use Save & Regenerate.")
            return
        }
        do {
            let updated = try await persistEditedMessageAsync(message, conversationID: convID)
            cancelEditingMessage()

            if updated.role == "assistant" {
                streamingContinueAnchorMessageID = updated.id
                streamingHiddenFromPosition = updated.position + 1
                await streamConversationGeneration(
                    for: convID,
                    mode: .continue,
                    messageID: updated.id,
                    initialThinking: updated.thinking ?? "",
                    initialContent: updated.content
                )
            } else {
                streamingHiddenFromPosition = updated.position + 1
                await streamConversationGeneration(
                    for: convID,
                    mode: .continue,
                    messageID: updated.id
                )
            }
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Failed to continue edited message: \(error.localizedDescription)")
        }
    }

    public func regenerateAssistantMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating, !isSelectedConversationGenerating else { return }
        if registry.isConversationGenerating(convID) {
            onStatusMessage("Conversation is already generating.")
            return
        }
        // Non-destructive: mask replaced turns in UI while streaming; atomic replace happens on heylook_saved.
        streamingHiddenFromPosition = message.position
        await streamConversationGeneration(
            for: convID,
            mode: .regenerate,
            messageID: message.id
        )
    }

    public func continueAssistantMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating, !isSelectedConversationGenerating, message.role == "assistant" else {
            return
        }
        if registry.isConversationGenerating(convID) {
            onStatusMessage("Conversation is already generating.")
            return
        }
        // Non-destructive: keep SQLite intact until heylook_saved commits anchor update + tail trim.
        streamingContinueAnchorMessageID = message.id
        streamingHiddenFromPosition = message.position + 1
        await streamConversationGeneration(
            for: convID,
            mode: .continue,
            messageID: message.id,
            initialThinking: message.thinking ?? "",
            initialContent: message.content
        )
    }

    public func deleteSingleMessage(id: String, conversationID: String) {
        if registry.isConversationGenerating(conversationID) || isSelectedConversationGenerating {
            onStatusMessage("Cannot delete messages while conversation is actively generating.")
            return
        }
        _ = try? store.deleteMessage(conversationID: conversationID, messageID: id)
        messageAttachmentsByID[id] = nil
        if editingMessageID == id {
            cancelEditingMessage()
        }
        if selectedConversationID == conversationID {
            currentMessages = (try? loadAndHydrateMessages(conversationID: conversationID)) ?? []
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
        if let remote = remoteBackendProvider() {
            Task { try? await remote.deleteMessage(conversationID: conversationID, messageID: id) }
        }
    }

    public func streamAssistantReply(
        for convID: String,
        modelOverride: String? = nil,
        continuingAssistantMessage: MessageRecord? = nil
    ) async {
        if let cont = continuingAssistantMessage {
            streamingContinueAnchorMessageID = cont.id
            streamingHiddenFromPosition = cont.position + 1
            await streamConversationGeneration(
                for: convID,
                mode: .continue,
                messageID: cont.id,
                modelOverride: modelOverride,
                initialThinking: cont.thinking ?? "",
                initialContent: cont.content
            )
        } else {
            await streamConversationGeneration(
                for: convID,
                mode: .append,
                messageID: nil,
                modelOverride: modelOverride
            )
        }
    }

    private func streamConversationGeneration(
        for convID: String,
        mode: ConversationGenerateMode,
        messageID: String?,
        content: String = "",
        userParts: [MessagesRequest.MessageTurn.ContentPart] = [],
        modelOverride: String? = nil,
        initialThinking: String = "",
        initialContent: String = "",
        rollbackDraft: (prompt: String, attachments: [StagedChatAttachment], tempUserID: String)? = nil
    ) async {
        let selectedID = activeModelIDProvider()
        let modelID = modelOverride
            ?? (selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID)
        guard !modelID.isEmpty else {
            onStatusMessage("No generation model selected or configured. Add scan_folders in heylook.toml or load the sample workspace.")
            clearStreamingPresentationState()
            return
        }

        // Flush any debounced `chatSettings` edit BEFORE reading samplers/prompt for generation!
        chatSettings.flushPendingWrites()
        let sparseOverlay = chatSettings.requestSamplerOverlay
        let supportsThinking = chatSettings.activeModelSupportsThinking
        let thinkingEnabled = chatSettings.effectiveThinkingEnabled
        let depthOptions = chatSettings.activeModelDepthOptions
        let depthValue = depthOptions.isEmpty ? nil : chatSettings.effectiveThinkingDepth
        let convSystemPrompt = (try? store.getConversation(id: convID))?.systemPrompt ?? chatSettings.systemPrompt
        let trimmedSys = convSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)

        let validatedResponseFormat: ResponseFormat?
        do {
            validatedResponseFormat = try buildValidatedResponseFormat()
        } catch {
            onStatusMessage("Structured output error: \(error.localizedDescription)")
            if let rb = rollbackDraft {
                restoreFailedDraft(rb, conversationID: convID)
            }
            clearStreamingPresentationState()
            return
        }

        let reqID = "req_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        let requestStartedAt = Date()

        activeRequestID = reqID
        isGenerating = true
        generatingConversationID = convID
        isEndingThinking = false
        prefillProcessed = 0
        prefillTotal = 0
        streamingThinking = initialThinking
        streamingContent = initialContent
        streamingTPS = 0
        streamingTokens = 0

        let targetDesc = activeModelProvider() ?? registeredModelsProvider().first(where: { $0.modelID == modelID })
        if targetDesc?.isLoaded == false {
            streamStatusLine = "Loading \(modelID)…"
        } else {
            streamStatusLine = "Waiting for the first token…"
        }

        let req = ConversationGenerateRequest(
            model: modelID,
            content: content,
            mode: mode,
            messageID: messageID,
            userParts: userParts.isEmpty ? nil : userParts,
            system: trimmedSys.isEmpty ? nil : convSystemPrompt,
            thinking: supportsThinking ? thinkingEnabled : nil,
            thinkingDepth: depthValue,
            thinkingBudget: supportsThinking ? sparseOverlay.thinkingBudget : nil,
            presetName: chatSettings.selectedPresetName,
            sampling: sparseOverlay,
            stopSequences: sparseOverlay.stopSequences,
            responseFormat: validatedResponseFormat
        )

        let cancelSignal = StreamCancellationSignal()
        activeCancellationSignal = cancelSignal

        var firstTokenDate: Date?
        var generatedTextAndThinkTokensApprox = 0
        var inBandError: String?
        var receivedSavedEvent: HeylookSavedEvent?

        let handleFrame: @Sendable (String) async -> Void = { [weak self] frame in
            guard let parsed = Self.parseSSEFrame(frame),
                  let data = parsed.data.data(using: .utf8) else { return }

            await MainActor.run {
                guard let self else { return }
                switch parsed.event {
                case "heylook_progress":
                    if let ev = try? JSONDecoder().decode(HeylookProgressSSEEvent.self, from: data) {
                        self.streamStatusLine = nil
                        self.prefillProcessed = ev.prefill.processed
                        self.prefillTotal = ev.prefill.total
                    }
                case "content_block_delta":
                    if let ev = try? JSONDecoder().decode(ContentBlockDeltaSSEEvent.self, from: data) {
                        self.streamStatusLine = nil
                        self.prefillProcessed = 0
                        self.prefillTotal = 0
                        if firstTokenDate == nil {
                            firstTokenDate = Date()
                        }
                        if ev.delta.type == "thinking_delta" {
                            let piece = ev.delta.thinking ?? ev.delta.text ?? ""
                            if !piece.isEmpty {
                                self.streamingThinking.append(piece)
                                generatedTextAndThinkTokensApprox += 1
                            }
                        } else {
                            let piece = ev.delta.text ?? ""
                            if !piece.isEmpty {
                                self.isEndingThinking = false
                                self.streamingContent.append(piece)
                                generatedTextAndThinkTokensApprox += 1
                            }
                        }
                        self.streamingTokens = max(self.streamingTokens, generatedTextAndThinkTokensApprox)
                        if let ft = firstTokenDate {
                            let elapsed = Date().timeIntervalSince(ft)
                            if elapsed > 0.08 && self.streamingTokens > 0 {
                                self.streamingTPS = Double(self.streamingTokens) / elapsed
                            }
                        }
                    }
                case "message_delta":
                    if let ev = try? JSONDecoder().decode(MessageDeltaSSEEvent.self, from: data),
                       let out = ev.usage?.outputTokens, out > 0 {
                        self.streamingTokens = out
                        let elapsed = Date().timeIntervalSince(firstTokenDate ?? requestStartedAt)
                        if elapsed > 0.05 {
                            self.streamingTPS = Double(out) / elapsed
                        }
                    }
                case "message_stop":
                    if let ev = try? JSONDecoder().decode(MessageStopSSEEvent.self, from: data),
                       let perf = ev.performance {
                        if let tps = perf.generationTPS ?? perf.tokensPerSecond, tps > 0 {
                            self.streamingTPS = tps
                        }
                    }
                case "error":
                    if let ev = try? JSONDecoder().decode(StreamErrorSSEEvent.self, from: data) {
                        inBandError = ev.error.message
                    }
                case "heylook_saved":
                    if let ev = try? JSONDecoder().decode(HeylookSavedEvent.self, from: data) {
                        receivedSavedEvent = ev
                    }
                default:
                    break
                }
            }
        }

        do {
            let returnedSaved: HeylookSavedEvent
            if let remote = remoteBackendProvider() {
                returnedSaved = try await remote.generateInConversation(
                    conversationID: convID,
                    request: req,
                    requestID: reqID,
                    detachOnSubscriberExit: true,
                    onSSEFrame: handleFrame
                )
            } else {
                returnedSaved = try await localService.generateInConversation(
                    conversationID: convID,
                    request: req,
                    requestID: reqID,
                    cancellationSignal: cancelSignal,
                    detachOnSubscriberExit: true,
                    onSSEFrame: handleFrame
                )
            }

            let saved = receivedSavedEvent ?? returnedSaved
            lastDroppedMediaCount = saved.droppedMedia.total
            if saved.droppedMedia.total > 0 {
                HeylookObservabilityLog.shared.record(
                    level: "INFO",
                    category: "ui.chat",
                    message: "Omitted \(saved.droppedMedia.total) unsupported media attachment(s) on \(modelID)",
                    requestID: reqID,
                    metadata: ["model": modelID, "dropped_media": "\(saved.droppedMedia.total)"]
                )
            }

            if saved.endReason == "detached" {
                adoptBackgroundGeneration(for: convID)
            } else {
                if let err = saved.error ?? inBandError, !err.isEmpty {
                    onStatusMessage("Generation stopped: \(err)")
                }
                if remoteBackendProvider() != nil {
                    let hydrated = saved.messages.map { Self.messageRecord(from: $0, conversationID: convID) }
                    if selectedConversationID == convID {
                        currentMessages = hydrated
                        hydrateRemoteMessageAttachments(from: hydrated)
                    }
                    for msg in hydrated {
                        _ = try? store.appendMessage(msg)
                    }
                } else if selectedConversationID == convID {
                    currentMessages = (try? loadAndHydrateMessages(conversationID: convID)) ?? currentMessages
                }
                if let rb = rollbackDraft {
                    messageAttachmentsByID[rb.tempUserID] = nil
                }
                pendingStreamingUserMessage = nil
                conversations = (try? store.listConversations()) ?? conversations
            }
        } catch {
            if let errMsg = inBandError, !errMsg.isEmpty {
                onStatusMessage("Generation failed: \(errMsg)")
            } else {
                onStatusMessage("Generation failed: \(error.localizedDescription)")
            }
            HeylookObservabilityLog.shared.record(
                level: "ERROR",
                category: "ui.chat",
                message: "Chat generation failed: \(error.localizedDescription)",
                requestID: reqID,
                metadata: ["conversation_id": convID, "model": modelID]
            )
            if let rb = rollbackDraft,
               (try? store.listMessages(conversationID: convID))?.last?.role != "user" {
                restoreFailedDraft(rb, conversationID: convID)
            }
            if selectedConversationID == convID {
                currentMessages = (try? loadAndHydrateMessages(conversationID: convID)) ?? currentMessages
            }
        }

        clearStreamingPresentationState()
        activeCancellationSignal = nil
        if selectedConversationID == convID && showPromptPreview {
            schedulePromptPreviewRefresh()
        }
        await refreshActiveAdminModel(resetDrafts: false)
        await onGenerationCompleted?()
    }

    private func restoreFailedDraft(
        _ rollback: (prompt: String, attachments: [StagedChatAttachment], tempUserID: String),
        conversationID: String
    ) {
        messageAttachmentsByID[rollback.tempUserID] = nil
        if pendingStreamingUserMessage?.id == rollback.tempUserID {
            pendingStreamingUserMessage = nil
        }
        if selectedConversationID == conversationID, draftPrompt.isEmpty, stagedAttachments.isEmpty {
            draftPrompt = rollback.prompt
            stagedAttachments = rollback.attachments
        } else {
            composerDraftByConversationID[conversationID] = rollback.prompt
            composerMediaByConversationID[conversationID] = rollback.attachments
        }
    }

    private func clearStreamingPresentationState() {
        isGenerating = false
        generatingConversationID = nil
        activeRequestID = nil
        isEndingThinking = false
        streamStatusLine = nil
        prefillProcessed = 0
        prefillTotal = 0
        streamingThinking = ""
        streamingContent = ""
        streamingHiddenFromPosition = nil
        streamingContinueAnchorMessageID = nil
        pendingStreamingUserMessage = nil
    }

    nonisolated private static func parseSSEFrame(_ frame: String) -> (event: String, data: String)? {
        var eventName = ""
        var dataLines: [String] = []
        for rawLine in frame.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.hasSuffix("\r") ? String(rawLine.dropLast()) : String(rawLine)
            if line.hasPrefix("event:") {
                eventName = line.dropFirst("event:".count).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("data:") {
                let tail = line.dropFirst("data:".count)
                let cleaned = tail.first == " " ? String(tail.dropFirst()) : String(tail)
                dataLines.append(cleaned)
            }
        }
        guard !dataLines.isEmpty else { return nil }
        return (eventName, dataLines.joined(separator: "\n"))
    }

    public func endThinkingNow() {
        guard activeModelCanEndThinking else { return }
        guard let reqID = activeRequestID, !isEndingThinking else { return }
        isEndingThinking = true
        _ = registry.requestReasoningEnd(requestID: reqID)
        if let remote = remoteBackendProvider() {
            Task { try? await remote.endReasoning(id: reqID) }
        }
    }

    public func cancelGeneration() {
        if let reqID = activeRequestID {
            _ = registry.abort(requestID: reqID)
            if let remote = remoteBackendProvider() {
                Task { try? await remote.abortRequest(id: reqID) }
            }
            return
        }
        if let bgConvID = backgroundGeneratingConversationID {
            _ = registry.abort(forConversationID: bgConvID)
            if remoteBackendProvider() != nil {
                startResyncUntilSettled(for: bgConvID)
            }
        }
    }
}
