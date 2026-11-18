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
#endif

/// Domain controller for Chat conversations, multimodal staged/historical media, voice ASR,
/// off-MainActor prompt preview, turn editing/forking/continuation, and token streaming.
///
/// Strictly uses its dedicated `chatSettings: DocumentSettingsController` instance so switching
/// surfaces or streaming mid-switch never shares or corrupts Notebook settings.
@MainActor
@Observable
public final class ChatController {
    private static let orphanConversationKey = "__orphan__"

    public let store: SQLiteStore
    public let scheduler: ComputeScheduler
    public let registry: RequestRegistry
    public let chatSettings: DocumentSettingsController

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
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
    }
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

    // MARK: - Streaming & Prompt Preview State

    public var isGenerating: Bool = false
    public private(set) var generatingConversationID: String?
    public var activeRequestID: String?
    public private(set) var isEndingThinking: Bool = false
    public private(set) var streamStatusLine: String?
    public var streamingThinking: String = ""
    public var streamingContent: String = ""
    public var streamingTPS: Double = 0
    public var streamingTokens: Int = 0

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

    // MARK: - Inline Message Edit & Model Switch Guard State

    public var editingMessageID: String?
    public var editingContent: String = ""
    public var editingThinking: String = ""

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
        chatSettings: DocumentSettingsController
    ) {
        self.store = store
        self.scheduler = scheduler
        self.registry = registry
        self.chatSettings = chatSettings
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
        }
    }

    public func syncWithRemoteBackend(_ remote: RemoteWorkbenchBackend) async {
        do {
            let list = try await remote.listConversations()
            var loadedConvs: [ConversationRecord] = []
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
                    loadedConvs.append(record)
                    try? store.upsertConversation(record)
                }
            }
            conversations = loadedConvs
            if let first = loadedConvs.first {
                selectedConversationID = nil
                selectConversation(first.id)
            }
        } catch {
            onStatusMessage("Remote conversation list failed: \(error.localizedDescription)")
        }
    }

    public func handleStoreMutationEvent(_ event: StoreMutationEvent, activeSurface: HeylookSurface) {
        switch event {
        case .conversationUpserted(let upsertedID):
            conversations = (try? store.listConversations()) ?? conversations
            if selectedConversationID == upsertedID && activeSurface == .chat && !isGenerating {
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
            if selectedConversationID == convID && !(isGenerating && generatingConversationID == convID) {
                if let loaded = try? loadAndHydrateMessages(conversationID: convID) {
                    currentMessages = loaded
                }
            }
        case .dataCleared:
            selectedConversationID = nil
            currentMessages = []
            messageAttachmentsByID.removeAll()
            composerDraftByConversationID.removeAll()
            composerMediaByConversationID.removeAll()
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
    }

    // MARK: - Model Selection & Multimodal Switch Guard

    public func selectGenerationModel(_ targetModelID: String) {
        let currentID = activeModelIDProvider()
        guard targetModelID != currentID else {
            pendingModelSwitchTargetID = nil
            modelSwitchWarningMessage = nil
            return
        }
        guard let target = registeredModelsProvider().first(where: { $0.modelID == targetModelID }) else {
            onSelectModelID(targetModelID)
            return
        }

        let convImages = countConversationAttachments(ofKind: .image)
        let convAudio = countConversationAttachments(ofKind: .audio)
        var dropped: [String] = []
        if convImages > 0 && !target.supportsVision.value {
            dropped.append("\(convImages) image\(convImages == 1 ? "" : "s") (no vision)")
        }
        if convAudio > 0 && !target.supportsAudio.value {
            dropped.append("\(convAudio) audio clip\(convAudio == 1 ? "" : "s") (no audio)")
        }

        if !dropped.isEmpty {
            pendingModelSwitchTargetID = targetModelID
            modelSwitchWarningMessage = "Switching to '\(targetModelID)' will omit \(dropped.joined(separator: " and ")) in this conversation."
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
        onSelectModelID(targetModelID)
        let targetDesc = registeredModelsProvider().first(where: { $0.modelID == targetModelID })
        _ = chatSettings.reconcileSettings(for: targetDesc, engineContract: nil)
        Task { [weak self] in
            guard let self else { return }
            if let contract = await self.activeEngineContractProvider() {
                _ = self.chatSettings.reconcileSettings(for: targetDesc, engineContract: contract)
            }
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

    private func countConversationAttachments(ofKind kind: MessagesRequest.MediaKind) -> Int {
        var count = stagedAttachments.filter { $0.kind == kind }.count
        for msg in currentMessages {
            count += (messageAttachmentsByID[msg.id] ?? []).filter { $0.kind == kind }.count
        }
        return count
    }

    public func omittedMediaNote(for messageID: String) -> String? {
        guard let attachments = messageAttachmentsByID[messageID], !attachments.isEmpty else {
            return nil
        }
        let activeDesc = activeModelProvider()
        let supportsVis = activeDesc?.supportsVision.value ?? false
        let supportsAud = activeDesc?.supportsAudio.value ?? false
        let droppedImages = supportsVis ? 0 : attachments.filter { $0.kind == .image }.count
        let droppedAudio = supportsAud ? 0 : attachments.filter { $0.kind == .audio }.count
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

    // MARK: - Conversation Lifecycle & System Prompt Persistence

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

        // Inheritance per presetForNewDoc(): stamped preset of open doc, or pre-doc draft.
        let inherited = chatSettings.presetForNewDoc()
        let selectedID = activeModelIDProvider()
        let model = selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID

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

            if let remote = remoteBackendProvider() {
                Task {
                    _ = try? await remote.createConversation(
                        title: conv.title,
                        modelID: conv.defaultModelID,
                        systemPrompt: conv.systemPrompt.isEmpty ? nil : conv.systemPrompt,
                        params: conv.params,
                        appliedPresetID: conv.appliedPresetID
                    )
                }
            }
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        } catch {
            onStatusMessage("Could not create conversation: \(error.localizedDescription)")
        }
    }

    public func selectConversation(_ id: String) {
        if selectedConversationID == id && !currentMessages.isEmpty {
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
        pendingModelSwitchTargetID = nil
        modelSwitchWarningMessage = nil

        do {
            currentMessages = try loadAndHydrateMessages(conversationID: id)
        } catch {
            onStatusMessage("Failed to load messages: \(error.localizedDescription)")
        }

        if let conv = conversations.first(where: { $0.id == id }) ?? (try? store.getConversation(id: id)) {
            if let defModel = conv.defaultModelID,
               !defModel.isEmpty,
               generationModelsProvider().contains(where: { $0.modelID == defModel }) {
                onSelectModelID(defModel)
            }
            chatSettings.hydrateDocument(
                id: conv.id,
                systemPrompt: conv.systemPrompt,
                params: conv.params,
                appliedPresetID: conv.appliedPresetID,
                activeModel: activeModelProvider(),
                engineContract: nil
            )
            if let remote = remoteBackendProvider(), currentMessages.isEmpty {
                Task { [weak self] in
                    guard let self, let detail = try? await remote.getConversation(id: id) else { return }
                    let hydrated = detail.messages.map { wire in
                        MessageRecord(
                            id: wire.id,
                            conversationID: id,
                            position: wire.position,
                            role: wire.role,
                            content: wire.content,
                            thinking: wire.thinking,
                            modelID: wire.modelID,
                            contentBlocks: wire.contentBlocks,
                            stats: wire.stats
                        )
                    }
                    if self.selectedConversationID == id {
                        self.currentMessages = hydrated
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

        if showPromptPreview {
            schedulePromptPreviewRefresh()
        }
    }

    public func cloneConversation(id: String, upToPosition: Int? = nil, newTitle: String? = nil) {
        chatSettings.flushPendingWrites()
        do {
            let cloned = try store.cloneConversation(id: id, upToPosition: upToPosition, newTitle: newTitle)
            conversations = try store.listConversations()
            selectConversation(cloned.id)
            onSwitchToChatSurface()
            if let remote = remoteBackendProvider() {
                Task { _ = try? await remote.cloneConversation(id: id, title: newTitle, upToPosition: upToPosition) }
            }
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
        let remoteBackend = remoteBackendProvider()
        let activeConvID = selectedConversationID

        isRenderingPromptPreview = true
        promptPreviewTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(150))
            guard !Task.isCancelled else { return }

            if activeModel?.provider == .remote,
               let remoteBackend,
               let activeConvID,
               let activeModel {
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
                        thinkingDepth: depthVal
                    )
                )
                if let remotePreview = try? await remoteBackend.previewPrompt(conversationID: activeConvID, request: payload) {
                    guard !Task.isCancelled else { return }
                    var badges = [remotePreview.source]
                    if remotePreview.droppedMedia.total > 0 {
                        badges.append("\(remotePreview.droppedMedia.total) media omitted")
                    }
                    self?.promptPreviewText = remotePreview.prompt
                    self?.promptPreviewSource = badges.joined(separator: " • ")
                    self?.promptPreviewMarkers = []
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

    // MARK: - Multimodal Attachments, Image Plan, Structured Output & Paste

    public func estimateStagedImagePlan(for attachment: StagedChatAttachment) -> StagedImagePlanEstimate? {
        guard attachment.kind == .image else { return nil }
        let width: Int
        let height: Int
        if attachment.useShrunk2048, let sw = attachment.shrunkPixelWidth, let sh = attachment.shrunkPixelHeight {
            width = sw
            height = sh
        } else if let ow = attachment.originalPixelWidth, let oh = attachment.originalPixelHeight {
            width = ow
            height = oh
        } else {
            return nil
        }
        guard width > 0, height > 0 else { return nil }

        var tokenCap = 280
        if let active = activeModelProvider() {
            if active.provider == .remote {
                tokenCap = 1024
            } else {
                let dirURL = HeylookConfig.resolveModelDirectoryURL(from: active.modelPath)
                let sidecarCap = sidecarLoader(active).imageMaxTokens
                tokenCap = max(16, sidecarCap ?? HeylookConfig.estimateImageTokenBudget(in: dirURL))
            }
        }
        let maxPixels = tokenCap * 28 * 28
        let rawPixels = width * height
        let scale = rawPixels > maxPixels ? sqrt(Double(maxPixels) / Double(rawPixels)) : 1.0
        let targetW = max(28, Int((Double(width) * scale / 28.0).rounded() * 28.0))
        let targetH = max(28, Int((Double(height) * scale / 28.0).rounded() * 28.0))
        let estTokens = min(tokenCap, max(16, (targetW / 28) * (targetH / 28)))
        return StagedImagePlanEstimate(
            resizedWidth: targetW,
            resizedHeight: targetH,
            estimatedTokens: estTokens,
            tokenCap: tokenCap
        )
    }

    public var totalStagedImageTokens: Int {
        stagedAttachments.reduce(0) { sum, att in
            guard att.kind == .image else { return sum }
            return sum + (estimateStagedImagePlan(for: att)?.estimatedTokens ?? 280)
        }
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

        stagedAttachments.append(
            StagedChatAttachment.make(
                kind: kind,
                mimeType: mime,
                filename: url.lastPathComponent,
                data: data
            )
        )
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
            stagedAttachments.append(
                StagedChatAttachment.make(
                    kind: .image,
                    mimeType: "image/png",
                    filename: "pasted-\(stamp).png",
                    data: pngData
                )
            )
            return true
        }
        if let tiffData = pb.data(forType: .tiff),
           let bitmap = NSBitmapImageRep(data: tiffData),
           let pngData = bitmap.representation(using: .png, properties: [:]) {
            let stamp = Int(Date().timeIntervalSince1970)
            stagedAttachments.append(
                StagedChatAttachment.make(
                    kind: .image,
                    mimeType: "image/png",
                    filename: "pasted-\(stamp).png",
                    data: pngData
                )
            )
            return true
        }
        return false
    }
    #endif

    public func toggleAttachmentShrink(id: String) {
        guard let idx = stagedAttachments.firstIndex(where: { $0.id == id }),
              stagedAttachments[idx].canToggleShrink else { return }
        stagedAttachments[idx].useShrunk2048.toggle()
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
            return "Model '\(desc.modelID)' does not support image inputs (supports_vision = false)."
        }
        if hasAudio && !desc.supportsAudio.value {
            return "Model '\(desc.modelID)' does not support audio inputs (supports_audio = false)."
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

    // MARK: - Send, Streaming & Turn Editing/Regeneration

    public func sendChatPrompt() async {
        let promptText = draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard (!promptText.isEmpty || !stagedAttachments.isEmpty), !isGenerating else { return }

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
            onStatusMessage("Conversation is already generating (HTTP client or another session holds the conversation lock).")
            return
        }

        chatSettings.flushPendingWrites()
        let pendingStaged = stagedAttachments

        if selectedConversationID == nil {
            let inherited = chatSettings.presetForNewDoc()
            let conv = ConversationRecord(
                title: "New Chat",
                systemPrompt: inherited.systemPrompt,
                defaultModelID: modelID,
                params: inherited.params,
                appliedPresetID: inherited.appliedPresetID
            )
            try? store.upsertConversation(conv)
            conversations = (try? store.listConversations()) ?? []
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
            if let remote = remoteBackendProvider() {
                _ = try? await remote.createConversation(
                    title: conv.title,
                    modelID: conv.defaultModelID,
                    systemPrompt: conv.systemPrompt.isEmpty ? nil : conv.systemPrompt,
                    params: conv.params,
                    appliedPresetID: conv.appliedPresetID
                )
            }
        }
        guard let convID = selectedConversationID else { return }

        draftPrompt = ""
        stagedAttachments = []
        composerDraftByConversationID[convID] = ""
        composerMediaByConversationID[convID] = []
        composerDraftByConversationID[Self.orphanConversationKey] = nil
        composerMediaByConversationID[Self.orphanConversationKey] = nil

        do {
            let nextPos = (currentMessages.map(\.position).max() ?? -1) + 1
            let fallbackAttachmentLabel = pendingStaged.map(\.filename).joined(separator: ", ")
            let storedContent = promptText.isEmpty ? "[Attached: \(fallbackAttachmentLabel)]" : promptText
            let contentBlocks: [MessagesRequest.MessageTurn.ContentPart]? = pendingStaged.isEmpty
                ? nil
                : (pendingStaged.map { .media($0.wireAttachment) } + (promptText.isEmpty ? [] : [.text(promptText)]))
            let userMsg = MessageRecord(
                conversationID: convID,
                position: nextPos,
                role: "user",
                content: storedContent,
                modelID: modelID,
                contentBlocks: contentBlocks
            )
            if !pendingStaged.isEmpty {
                messageAttachmentsByID[userMsg.id] = pendingStaged
            }
            try store.appendMessage(userMsg)
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            if var conv = try store.getConversation(id: convID), conv.title == "New Chat" {
                conv.title = String(storedContent.prefix(44))
                conv.defaultModelID = modelID
                conv.updatedUnixMs = SQLiteStore.nowMillis()
                try store.upsertConversation(conv)
                conversations = try store.listConversations()
            }
            await streamAssistantReply(for: convID, modelOverride: modelID)
        } catch {
            onStatusMessage("Chat error: \(error.localizedDescription)")
        }
    }

    private func loadAndHydrateMessages(conversationID: String) throws -> [MessageRecord] {
        let loaded = try store.listMessages(conversationID: conversationID, hydrateMedia: true)
        for msg in loaded {
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
        return loaded
    }

    private static func updatingTextInContentBlocks(
        _ existingBlocks: [MessagesRequest.MessageTurn.ContentPart]?,
        newContent: String,
        newThinking: String?
    ) -> [MessagesRequest.MessageTurn.ContentPart]? {
        guard let existingBlocks else { return nil }
        var rebuilt: [MessagesRequest.MessageTurn.ContentPart] = []
        if let th = newThinking, !th.isEmpty {
            rebuilt.append(.thinking(th))
        }
        for block in existingBlocks {
            if case .media = block {
                rebuilt.append(block)
            }
        }
        if !newContent.isEmpty || rebuilt.isEmpty {
            rebuilt.append(.text(newContent))
        }
        return rebuilt
    }

    public func beginEditingMessage(_ message: MessageRecord) {
        editingMessageID = message.id
        editingContent = message.content
        editingThinking = message.thinking ?? ""
    }

    public func cancelEditingMessage() {
        editingMessageID = nil
        editingContent = ""
        editingThinking = ""
    }

    public func saveEditedMessageInPlace(_ message: MessageRecord) {
        guard let convID = selectedConversationID else { return }
        if registry.isConversationGenerating(convID) {
            onStatusMessage("Cannot edit while conversation is actively generating.")
            return
        }
        do {
            let trimmedThinking = editingThinking.trimmingCharacters(in: .whitespacesAndNewlines)
            var updated = message
            updated.content = editingContent
            if message.role == "assistant" {
                updated.thinking = trimmedThinking.isEmpty ? nil : editingThinking
            }
            updated.contentBlocks = Self.updatingTextInContentBlocks(
                updated.contentBlocks,
                newContent: updated.content,
                newThinking: updated.thinking
            )
            try store.appendMessage(updated)
            cancelEditingMessage()
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Failed to save message edit: \(error.localizedDescription)")
        }
    }

    public func saveAndRegenerateEditedMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating else { return }
        if registry.isConversationGenerating(convID) {
            onStatusMessage("Conversation is already generating.")
            return
        }
        do {
            if message.role == "assistant" {
                cancelEditingMessage()
                try store.deleteMessages(conversationID: convID, fromPosition: message.position)
                currentMessages = try loadAndHydrateMessages(conversationID: convID)
                await streamAssistantReply(for: convID)
                return
            }

            var updated = message
            updated.content = editingContent
            updated.contentBlocks = Self.updatingTextInContentBlocks(
                updated.contentBlocks,
                newContent: updated.content,
                newThinking: updated.thinking
            )
            try store.appendMessage(updated)
            try store.deleteMessages(conversationID: convID, fromPosition: message.position + 1)
            cancelEditingMessage()
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            await streamAssistantReply(for: convID)
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Failed to regenerate message: \(error.localizedDescription)")
        }
    }

    public func saveAndContinueEditedMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating else { return }
        if registry.isConversationGenerating(convID) {
            onStatusMessage("Conversation is already generating.")
            return
        }
        do {
            let trimmedThinking = editingThinking.trimmingCharacters(in: .whitespacesAndNewlines)
            var updated = message
            updated.content = editingContent
            if message.role == "assistant" {
                updated.thinking = trimmedThinking.isEmpty ? nil : editingThinking
            }
            updated.contentBlocks = Self.updatingTextInContentBlocks(
                updated.contentBlocks,
                newContent: updated.content,
                newThinking: updated.thinking
            )
            try store.appendMessage(updated)
            try store.deleteMessages(conversationID: convID, fromPosition: message.position + 1)
            cancelEditingMessage()
            currentMessages = try loadAndHydrateMessages(conversationID: convID)

            if updated.role == "assistant" {
                await streamAssistantReply(for: convID, continuingAssistantMessage: updated)
            } else {
                await streamAssistantReply(for: convID)
            }
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Failed to continue edited message: \(error.localizedDescription)")
        }
    }

    public func regenerateAssistantMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating else { return }
        if registry.isConversationGenerating(convID) {
            onStatusMessage("Conversation is already generating.")
            return
        }
        do {
            try store.deleteMessages(conversationID: convID, fromPosition: message.position)
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            await streamAssistantReply(for: convID)
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Failed to regenerate turn: \(error.localizedDescription)")
        }
    }

    public func continueAssistantMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating, message.role == "assistant" else { return }
        if registry.isConversationGenerating(convID) {
            onStatusMessage("Conversation is already generating.")
            return
        }
        do {
            try store.deleteMessages(conversationID: convID, fromPosition: message.position + 1)
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            await streamAssistantReply(for: convID, continuingAssistantMessage: message)
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Failed to continue assistant turn: \(error.localizedDescription)")
        }
    }

    public func deleteSingleMessage(id: String, conversationID: String) {
        if registry.isConversationGenerating(conversationID) {
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
    }

    public func streamAssistantReply(
        for convID: String,
        modelOverride: String? = nil,
        continuingAssistantMessage: MessageRecord? = nil
    ) async {
        let selectedID = activeModelIDProvider()
        let modelID = modelOverride
            ?? (selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID)
        guard !modelID.isEmpty else {
            onStatusMessage("No generation model selected or configured. Add scan_folders in heylook.toml or load the sample workspace.")
            return
        }

        let reqID = "req_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        guard let execID = registry.tryAcquireConversationLock(
            conversationID: convID,
            requestID: reqID,
            modelID: modelID
        ) else {
            onStatusMessage("Conversation '\(convID)' is already generating (shared lock active).")
            HeylookObservabilityLog.shared.record(
                level: "WARN",
                category: "ui.chat",
                message: "Blocked concurrent generation on locked conversation",
                requestID: reqID,
                metadata: ["conversation_id": convID, "model": modelID]
            )
            return
        }
        defer {
            registry.releaseConversationLock(
                conversationID: convID,
                requestID: reqID,
                executionID: execID
            )
        }

        // Flush any debounced `chatSettings` edit BEFORE reading samplers/prompt for generation!
        chatSettings.flushPendingWrites()
        let sparseOverlay = chatSettings.requestSamplerOverlay
        let resolvedEffective = chatSettings.currentResolvedSamplers.effective
        let supportsThinking = chatSettings.activeModelSupportsThinking
        let thinkingEnabled = chatSettings.effectiveThinkingEnabled
        let depthOptions = chatSettings.activeModelDepthOptions
        let depthValue = depthOptions.isEmpty ? nil : chatSettings.effectiveThinkingDepth

        let requestStartedAt = Date()
        activeRequestID = reqID
        isGenerating = true
        generatingConversationID = convID
        isEndingThinking = false
        streamingThinking = continuingAssistantMessage?.thinking ?? ""
        streamingContent = continuingAssistantMessage?.content ?? ""
        streamingTPS = 0
        streamingTokens = 0

        let targetDesc = await scheduler.modelDescription(for: modelID)
        if targetDesc?.isLoaded == false {
            streamStatusLine = "Loading \(modelID)… (first token follows model load)"
        } else {
            streamStatusLine = "Waiting for first token…"
        }

        do {
            let convMessages = try loadAndHydrateMessages(conversationID: convID)
            if let cont = continuingAssistantMessage, selectedConversationID == convID {
                currentMessages = convMessages.filter { $0.id != cont.id }
            }

            let supportsVis = targetDesc?.supportsVision.value ?? false
            let supportsAud = targetDesc?.supportsAudio.value ?? false
            var droppedMedia = 0

            let turns: [MessagesRequest.MessageTurn] = convMessages.map { msg in
                let staged = messageAttachmentsByID[msg.id] ?? []
                if !staged.isEmpty {
                    let allowedWire = staged.compactMap { att -> MessagesRequest.MediaAttachment? in
                        if att.kind == .image && !supportsVis {
                            droppedMedia += 1
                            return nil
                        }
                        if att.kind == .audio && !supportsAud {
                            droppedMedia += 1
                            return nil
                        }
                        return att.wireAttachment
                    }
                    return MessagesRequest.MessageTurn(
                        role: msg.role,
                        content: msg.content,
                        thinking: msg.thinking,
                        attachments: allowedWire
                    )
                } else if let blocks = msg.contentBlocks, !blocks.isEmpty {
                    var keptBlocks: [MessagesRequest.MessageTurn.ContentPart] = []
                    for block in blocks {
                        if case .media(let att) = block {
                            if att.kind == "image" && !supportsVis {
                                droppedMedia += 1
                                continue
                            }
                            if att.kind == "audio" && !supportsAud {
                                droppedMedia += 1
                                continue
                            }
                        }
                        keptBlocks.append(block)
                    }
                    return MessagesRequest.MessageTurn(
                        role: msg.role,
                        content: msg.content,
                        thinking: msg.thinking,
                        attachments: [],
                        parts: keptBlocks
                    )
                } else {
                    return MessagesRequest.MessageTurn(
                        role: msg.role,
                        content: msg.content,
                        thinking: msg.thinking,
                        attachments: []
                    )
                }
            }
            lastDroppedMediaCount = droppedMedia
            if droppedMedia > 0 {
                HeylookObservabilityLog.shared.record(
                    level: "INFO",
                    category: "ui.chat",
                    message: "Omitted \(droppedMedia) historical media attachment(s) unsupported by \(modelID)",
                    requestID: reqID,
                    metadata: ["model": modelID, "dropped_media": "\(droppedMedia)"]
                )
            }

            let provider = try await scheduler.provider(for: modelID)
            let isRemoteProvider = provider is RemoteHeylookProvider
            // Null-Means-Cascade on remote requests; resolved 5-layer cascade locally.
            let effective = isRemoteProvider ? sparseOverlay : resolvedEffective
            let convSystemPrompt = (try? store.getConversation(id: convID))?.systemPrompt ?? chatSettings.systemPrompt
            let trimmedSys = convSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            let validatedResponseFormat = try buildValidatedResponseFormat()

            let req = MessagesRequest(
                model: modelID,
                messages: turns,
                system: trimmedSys.isEmpty ? nil : convSystemPrompt,
                maxTokens: effective.maxTokens,
                temperature: effective.temperature,
                topP: effective.topP,
                topK: effective.topK,
                minP: effective.minP,
                repetitionPenalty: effective.repetitionPenalty,
                repetitionContextSize: effective.repetitionContextSize,
                presencePenalty: effective.presencePenalty,
                seed: effective.seed,
                stream: true,
                thinking: isRemoteProvider ? effective.thinking : (supportsThinking ? thinkingEnabled : nil),
                thinkingDepth: isRemoteProvider ? effective.thinkingDepth : depthValue,
                thinkingBudget: isRemoteProvider ? effective.thinkingBudget : (supportsThinking ? effective.thinkingBudget : nil),
                stopSequences: effective.stopSequences,
                responseFormat: validatedResponseFormat
            )

            let finalChunk = try await scheduler.withComputeLease(for: modelID) { [registry] () throws -> GenerationChunk in
                var lastChunk = GenerationChunk()
                let stream = await provider.generate(
                    requestID: reqID,
                    request: req,
                    effectiveSampling: effective,
                    registry: registry
                )
                for try await chunk in stream {
                    await MainActor.run {
                        if !chunk.thinking.isEmpty || !chunk.text.isEmpty {
                            self.streamStatusLine = nil
                        } else if chunk.promptTokens > 0 && self.streamStatusLine != nil {
                            self.streamStatusLine = "Prefilling \(chunk.promptTokens) prompt tokens…"
                        }
                        if !chunk.thinking.isEmpty {
                            self.streamingThinking.append(chunk.thinking)
                        }
                        if !chunk.text.isEmpty {
                            self.isEndingThinking = false
                            self.streamingContent.append(chunk.text)
                        }
                        if chunk.tokenCount > 0 { self.streamingTokens = chunk.tokenCount }
                        if chunk.generationTPS > 0 { self.streamingTPS = chunk.generationTPS }
                    }
                    lastChunk = chunk
                }
                return lastChunk
            }

            let requestDurationMs = Date().timeIntervalSince(requestStartedAt) * 1000.0

            if !streamingContent.isEmpty || !streamingThinking.isEmpty {
                let savedMessageID: String
                if var cont = continuingAssistantMessage {
                    cont.content = streamingContent
                    cont.thinking = streamingThinking.isEmpty ? nil : streamingThinking
                    cont.contentBlocks = Self.updatingTextInContentBlocks(
                        cont.contentBlocks,
                        newContent: cont.content,
                        newThinking: cont.thinking
                    )
                    cont.modelID = modelID
                    cont.stopReason = (finalChunk.stopReason ?? .endTurn).rawValue
                    cont.tokenCount = finalChunk.tokenCount > 0 ? finalChunk.tokenCount : cont.tokenCount
                    cont.ttftMs = finalChunk.ttftMs ?? cont.ttftMs
                    cont.cacheReport = finalChunk.cacheReport ?? cont.cacheReport
                    cont.speculativeReport = finalChunk.speculativeReport ?? cont.speculativeReport
                    cont.generationTPS = finalChunk.generationTPS > 0 ? finalChunk.generationTPS : cont.generationTPS
                    cont.computePlacement = finalChunk.activeComputePlacement
                    try store.appendMessage(cont)
                    savedMessageID = cont.id
                } else {
                    let nextPos = (convMessages.map(\.position).max() ?? -1) + 1
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
                        computePlacement: finalChunk.activeComputePlacement,
                        tokenCount: finalChunk.tokenCount > 0 ? finalChunk.tokenCount : nil,
                        ttftMs: finalChunk.ttftMs
                    )
                    try store.appendMessage(assistantMsg)
                    savedMessageID = assistantMsg.id
                }

                let stats = MessageStatsRecord(
                    messageID: savedMessageID,
                    conversationID: convID,
                    modelID: modelID,
                    promptTokens: finalChunk.promptTokens,
                    outputTokens: finalChunk.tokenCount,
                    thinkingTokens: finalChunk.thinkingTokens,
                    contentTokens: finalChunk.contentTokens,
                    cachedTokens: finalChunk.cacheReport?.cachedTokens ?? 0,
                    promptTPS: finalChunk.promptTPS,
                    generationTPS: finalChunk.generationTPS,
                    ttftMs: finalChunk.ttftMs,
                    thinkingDurationMs: finalChunk.thinkingDurationMs,
                    contentDurationMs: finalChunk.contentDurationMs,
                    generationDurationMs: finalChunk.generationDurationMs,
                    requestDurationMs: requestDurationMs
                )
                try? store.upsertMessageStats(stats)
                HeylookObservabilityLog.shared.record(
                    level: "INFO",
                    category: "ui.chat",
                    message: "Completed assistant turn (\(finalChunk.tokenCount) tok @ \(String(format: "%.1f", finalChunk.generationTPS)) tok/s)",
                    requestID: reqID,
                    metadata: [
                        "conversation_id": convID,
                        "model": modelID,
                        "output_tokens": "\(finalChunk.tokenCount)",
                        "prompt_tokens": "\(finalChunk.promptTokens)",
                        "stop_reason": (finalChunk.stopReason ?? .endTurn).rawValue
                    ]
                )
            }
            if selectedConversationID == convID {
                currentMessages = try loadAndHydrateMessages(conversationID: convID)
            }
        } catch {
            onStatusMessage("Generation failed: \(error.localizedDescription)")
            HeylookObservabilityLog.shared.record(
                level: "ERROR",
                category: "ui.chat",
                message: "Chat generation failed: \(error.localizedDescription)",
                requestID: reqID,
                metadata: ["conversation_id": convID, "model": modelID]
            )
            if selectedConversationID == convID {
                currentMessages = (try? loadAndHydrateMessages(conversationID: convID)) ?? currentMessages
            }
        }

        isGenerating = false
        generatingConversationID = nil
        activeRequestID = nil
        isEndingThinking = false
        streamStatusLine = nil
        streamingThinking = ""
        streamingContent = ""
        if selectedConversationID == convID && showPromptPreview {
            schedulePromptPreviewRefresh()
        }
        await onGenerationCompleted?()
    }

    public func endThinkingNow() {
        guard let reqID = activeRequestID, !isEndingThinking else { return }
        isEndingThinking = true
        _ = registry.requestReasoningEnd(requestID: reqID)
        if let remote = remoteBackendProvider() {
            Task { try? await remote.endReasoning(id: reqID) }
        }
    }

    public func cancelGeneration() {
        guard let reqID = activeRequestID else { return }
        _ = registry.abort(requestID: reqID)
        if let remote = remoteBackendProvider() {
            Task { try? await remote.abortRequest(id: reqID) }
        }
    }
}
