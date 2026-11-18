import Foundation
import Observation
import HeylookCore
import HeylookEngines
import HeylookServer

public enum NotebookPromptChipState: Equatable, Sendable {
    case empty
    case custom
    case presetUnmodified(name: String)
    case presetModified(name: String)

    public var chipLabel: String {
        switch self {
        case .empty:
            return "System prompt"
        case .custom:
            return "System prompt (custom)"
        case .presetUnmodified(let name):
            return "Prompt: \(name)"
        case .presetModified(let name):
            return "Prompt: \(name) (modified)"
        }
    }

    public var isActiveOrModified: Bool {
        switch self {
        case .empty: return false
        case .custom, .presetUnmodified, .presetModified: return true
        }
    }
}

public struct PresetModelPreviewEntry: Identifiable, Sendable, Equatable {
    public let id: String
    public let label: String
    public let presetValueText: String?
    public let modelDefaultText: String?
    public let modelDefaultSource: String?
    public let effectiveValueText: String
    public let unsupportedReason: String?

    public var isOverriddenByPreset: Bool { presetValueText != nil }
    public var isUnsupported: Bool { unsupportedReason != nil }
}

public struct PresetModelPreviewSummary: Sendable, Equatable {
    public let modelID: String
    public let thinkingDefault: Bool
    public let capabilities: [String]
    public let entries: [PresetModelPreviewEntry]
    public let unsupportedCount: Int
    public let isRemoteDerived: Bool
}

/// Domain controller for Notebooks (inline cursor continuation, summarize, selection analysis,
/// auto-save, Chat export) and Global Presets CRUD/filtering.
///
/// Strictly reads and updates its dedicated `notebookSettings: DocumentSettingsController` instance
/// so Chat and Notebook document states never collide even across mid-stream surface switches.
@MainActor
@Observable
public final class NotebookAndPresetsController {
    public let store: SQLiteStore
    public let scheduler: ComputeScheduler
    public let registry: RequestRegistry
    public let notebookSettings: DocumentSettingsController

    // MARK: - Notebook State

    public var notebooks: [NotebookRecord] = []
    public var selectedNotebookID: String?
    public var notebookTitle: String = "Untitled Notebook" {
        didSet {
            guard !isApplyingProgrammaticState else { return }
            noteNotebookUserEdited()
        }
    }
    public var notebookBody: String = ""
    public var notebookSelectedRange: NSRange = NSRange(location: 0, length: 0)
    public var isNotebookGenerating: Bool = false
    public private(set) var activeNotebookRequestID: String?
    public private(set) var activeNotebookStreamingID: String?
    public private(set) var isNotebookDirty: Bool = false

    /// Notebook system prompt delegates directly to `notebookSettings.systemPrompt`.
    public var notebookSystemPrompt: String {
        get { notebookSettings.systemPrompt }
        set {
            guard notebookSettings.systemPrompt != newValue else { return }
            notebookSettings.systemPrompt = newValue
            if !isApplyingProgrammaticState {
                isNotebookDirty = true
            }
        }
    }

    // MARK: - Presets Catalog State

    public var presets: [PresetRecord] = [] {
        didSet {
            recomputeFilteredPresets()
            if let selID = selectedPresetRecordID, !presets.contains(where: { $0.id == selID }) {
                selectedPresetRecordID = presets.first?.id
            } else if selectedPresetRecordID == nil {
                selectedPresetRecordID = presets.first?.id
            }
            onPresetsUpdated?()
        }
    }
    public private(set) var filteredPresets: [PresetRecord] = []
    public var presetSearchQuery: String = "" {
        didSet { recomputeFilteredPresets() }
    }
    public var presetScopeFilter: String? = nil {
        didSet { recomputeFilteredPresets() }
    }
    public var selectedPresetRecordID: String? = nil
    public private(set) var presetConflictServerRecord: PresetRecord?
    public var presetConcurrencyConflictMessage: String?
    public private(set) var cachedRemotePublicRows: [String: PublicModelWireRow] = [:]

    // MARK: - Shell Dependencies & Callbacks

    @ObservationIgnored private var isApplyingProgrammaticState: Bool = false
    @ObservationIgnored private var notebookAutoSaveTask: Task<Void, Never>?
    @ObservationIgnored private var pendingNotebookSaveSnapshot: (docID: String?, title: String, body: String, modelID: String?)?
    @ObservationIgnored private var remoteBackendProvider: () -> RemoteWorkbenchBackend? = { nil }
    @ObservationIgnored private var activeModelIDProvider: () -> String = { "" }
    @ObservationIgnored private var activeModelProvider: () -> EngineDescription? = { nil }
    @ObservationIgnored private var generationModelsProvider: () -> [EngineDescription] = { [] }
    @ObservationIgnored private var activeEngineContractProvider: () async -> EngineContract? = { nil }

    @ObservationIgnored public var onStatusMessage: (String?) -> Void = { _ in }
    @ObservationIgnored public var onErrorMessage: (String?) -> Void = { _ in }
    @ObservationIgnored public var onSelectModelID: (String) -> Void = { _ in }
    @ObservationIgnored public var onSendToChatRequested: (ConversationRecord) -> Void = { _ in }
    @ObservationIgnored public var onGenerationCompleted: (() async -> Void)?
    @ObservationIgnored public var onPresetsUpdated: (() -> Void)?

    public init(
        store: SQLiteStore,
        scheduler: ComputeScheduler,
        registry: RequestRegistry,
        notebookSettings: DocumentSettingsController
    ) {
        self.store = store
        self.scheduler = scheduler
        self.registry = registry
        self.notebookSettings = notebookSettings
    }

    public func configureDependencies(
        remoteBackend: @escaping () -> RemoteWorkbenchBackend?,
        activeModelID: @escaping () -> String,
        activeModel: @escaping () -> EngineDescription?,
        generationModels: @escaping () -> [EngineDescription],
        activeEngineContract: @escaping () async -> EngineContract?
    ) {
        self.remoteBackendProvider = remoteBackend
        self.activeModelIDProvider = activeModelID
        self.activeModelProvider = activeModel
        self.generationModelsProvider = generationModels
        self.activeEngineContractProvider = activeEngineContract
    }

    // MARK: - Store & Remote Refresh

    public func seedDefaultPresetsIfNeeded() {
        presets = (try? store.listPresets()) ?? []
    }

    public func refreshFromStore() {
        presets = (try? store.listPresets()) ?? presets
        notebooks = (try? store.listNotebooks()) ?? notebooks

        if selectedNotebookID == nil {
            let preferredID = notebookSettings.readLastDocumentID()
            let initial = preferredID.flatMap { id in notebooks.first(where: { $0.id == id }) } ?? notebooks.first
            if let initial {
                hydrateSelectedNotebook(initial)
            } else {
                hydrateEmptyNotebookState()
            }
        }
    }

    public func syncWithRemoteBackend(_ remote: RemoteWorkbenchBackend) async {
        do {
            let remotePresetsResp = try await remote.listPresets()
            let mappedPresets: [PresetRecord] = remotePresetsResp.presets.map { dto in
                let trimmedSys = dto.systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                return PresetRecord(
                    id: dto.id,
                    name: dto.name,
                    scope: trimmedSys.isEmpty ? "sampler" : "full",
                    settings: dto.params.settings,
                    systemPrompt: trimmedSys.isEmpty ? nil : dto.systemPrompt,
                    description: "",
                    updatedUnixMs: WireTimestamp.unixMs(fromISO8601: dto.updatedAt) ?? SQLiteStore.nowMillis(),
                    createdUnixMs: WireTimestamp.unixMs(fromISO8601: dto.createdAt)
                )
            }
            for preset in mappedPresets {
                try? store.upsertPreset(preset)
            }
            presets = mappedPresets
        } catch {
            // Keep local presets if peer preset list fails.
        }

        do {
            let remoteNbList = try await remote.listNotebooks()
            var loaded: [NotebookRecord] = []
            for summary in remoteNbList.notebooks {
                if let detail = try? await remote.getNotebook(id: summary.id) {
                    loaded.append(
                        NotebookRecord(
                            id: detail.id,
                            title: detail.title,
                            body: detail.content,
                            systemPrompt: detail.systemPrompt ?? "",
                            modelID: detail.modelID,
                            params: detail.params.settings == SamplerSettings() ? nil : detail.params.settings,
                            appliedPresetID: detail.appliedPresetID,
                            updatedUnixMs: WireTimestamp.unixMs(fromISO8601: detail.updatedAt) ?? SQLiteStore.nowMillis(),
                            createdUnixMs: WireTimestamp.unixMs(fromISO8601: detail.createdAt)
                        )
                    )
                }
            }
            notebooks = loaded
            let preferredID = notebookSettings.readLastDocumentID()
            if !loaded.isEmpty {
                let initial = preferredID.flatMap { id in loaded.first(where: { $0.id == id }) } ?? loaded.first
                if let initial {
                    hydrateSelectedNotebook(initial)
                }
            } else if selectedNotebookID != nil {
                hydrateEmptyNotebookState()
            }
        } catch {
            onStatusMessage("Remote notebook list failed: \(error.localizedDescription)")
        }

        let publicModels = await remote.listPublicModels()
        if !publicModels.isEmpty {
            cachedRemotePublicRows = Dictionary(uniqueKeysWithValues: publicModels.map { ($0.id, $0) })
        }
    }

    public func handleStoreMutationEvent(_ event: StoreMutationEvent) {
        switch event {
        case .notebookUpserted, .notebookDeleted:
            notebooks = (try? store.listNotebooks()) ?? notebooks
        case .presetsChanged:
            presets = (try? store.listPresets()) ?? presets
        case .dataCleared:
            selectedNotebookID = nil
            hydrateEmptyNotebookState()
            refreshFromStore()
        default:
            break
        }
    }

    private func hydrateSelectedNotebook(_ nb: NotebookRecord) {
        isApplyingProgrammaticState = true
        selectedNotebookID = nb.id
        notebookTitle = nb.title
        notebookBody = nb.body
        notebookSelectedRange = NSRange(location: (nb.body as NSString).length, length: 0)
        isNotebookDirty = false
        isApplyingProgrammaticState = false

        notebookSettings.saveLastDocumentID(nb.id)
        let targetModel = nb.modelID.flatMap { id in
            generationModelsProvider().first(where: { $0.modelID == id })
        } ?? activeModelProvider()

        notebookSettings.hydrateDocument(
            id: nb.id,
            systemPrompt: nb.systemPrompt,
            params: nb.params,
            appliedPresetID: nb.appliedPresetID,
            activeModel: targetModel,
            engineContract: nil
        )
    }

    private func hydrateEmptyNotebookState() {
        isApplyingProgrammaticState = true
        selectedNotebookID = nil
        notebookTitle = "Untitled Notebook"
        notebookBody = ""
        notebookSelectedRange = NSRange(location: 0, length: 0)
        isNotebookDirty = false
        isApplyingProgrammaticState = false
        notebookSettings.saveLastDocumentID(nil)
        notebookSettings.hydrateDocument(
            id: nil,
            systemPrompt: "",
            params: nil,
            appliedPresetID: nil,
            activeModel: activeModelProvider(),
            engineContract: nil
        )
    }

    // MARK: - Notebook Editing, Auto-Save & CRUD

    public func noteNotebookUserEdited() {
        guard !isApplyingProgrammaticState else { return }
        isNotebookDirty = true
        let capturedDocID = selectedNotebookID
        let capturedTitle = notebookTitle
        let capturedBody = notebookBody
        let capturedModel = activeModelIDProvider()
        pendingNotebookSaveSnapshot = (
            docID: capturedDocID,
            title: capturedTitle,
            body: capturedBody,
            modelID: capturedModel.isEmpty ? nil : capturedModel
        )
        notebookAutoSaveTask?.cancel()
        notebookAutoSaveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled else { return }
            self?.commitPendingNotebookSnapshotIfMatching(
                docID: capturedDocID,
                title: capturedTitle,
                body: capturedBody,
                modelID: capturedModel.isEmpty ? nil : capturedModel
            )
        }
    }

    public func flushNotebookSaveIfDirty() {
        notebookAutoSaveTask?.cancel()
        notebookAutoSaveTask = nil
        notebookSettings.flushPendingWrites()
        if let pending = pendingNotebookSaveSnapshot {
            pendingNotebookSaveSnapshot = nil
            commitNotebookSnapshot(
                targetID: pending.docID,
                title: pending.title,
                body: pending.body,
                modelID: pending.modelID,
                syncActiveSettings: pending.docID == nil || pending.docID == selectedNotebookID
            )
        } else if isNotebookDirty {
            saveNotebook()
        }
    }

    public func createNewNotebook() {
        if isNotebookGenerating {
            cancelNotebookGeneration()
            if activeNotebookStreamingID == selectedNotebookID {
                saveNotebook()
            }
        }
        flushNotebookSaveIfDirty()

        // Inheritance per presetForNewDoc(): stamped preset if a notebook is open, or pre-doc draft otherwise.
        let inherited = notebookSettings.presetForNewDoc()
        let selectedID = activeModelIDProvider()
        let model = selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID

        do {
            let nb = NotebookRecord(
                title: "Untitled Notebook",
                body: "",
                systemPrompt: inherited.systemPrompt,
                modelID: model.isEmpty ? nil : model,
                params: inherited.params,
                appliedPresetID: inherited.appliedPresetID
            )
            try store.upsertNotebook(nb)
            notebooks = try store.listNotebooks()
            hydrateSelectedNotebook(nb)

            if let remote = remoteBackendProvider() {
                Task {
                    _ = try? await remote.createNotebook(
                        title: nb.title,
                        content: nb.body,
                        systemPrompt: nb.systemPrompt.isEmpty ? nil : nb.systemPrompt,
                        modelID: nb.modelID,
                        params: nb.params,
                        appliedPresetID: nb.appliedPresetID
                    )
                }
            }
        } catch {
            onStatusMessage("Could not create notebook: \(error.localizedDescription)")
        }
    }

    public func selectNotebook(_ id: String) {
        guard id != selectedNotebookID else { return }
        if isNotebookGenerating {
            cancelNotebookGeneration()
            if activeNotebookStreamingID == selectedNotebookID {
                saveNotebook()
            }
        }
        flushNotebookSaveIfDirty()

        let fresh = (try? store.getNotebook(id: id)) ?? notebooks.first(where: { $0.id == id })
        if let nb = fresh {
            hydrateSelectedNotebook(nb)
            if let modelID = nb.modelID, !modelID.isEmpty,
               generationModelsProvider().contains(where: { $0.modelID == modelID }) {
                onSelectModelID(modelID)
            }
        }
    }

    public func deleteNotebook(_ id: String) {
        if activeNotebookStreamingID == id || (selectedNotebookID == id && isNotebookGenerating) {
            cancelNotebookGeneration()
        }
        if selectedNotebookID == id {
            notebookAutoSaveTask?.cancel()
            isNotebookDirty = false
        }
        _ = try? store.deleteNotebook(id: id)
        notebooks = (try? store.listNotebooks()) ?? []
        if let remote = remoteBackendProvider() {
            Task { try? await remote.deleteNotebook(id: id) }
        }
        if selectedNotebookID == id {
            selectedNotebookID = nil
            if let first = notebooks.first {
                selectNotebook(first.id)
            } else {
                hydrateEmptyNotebookState()
            }
        }
    }

    public func saveNotebook() {
        notebookAutoSaveTask?.cancel()
        notebookAutoSaveTask = nil
        pendingNotebookSaveSnapshot = nil
        notebookSettings.flushPendingWrites()
        let selectedID = activeModelIDProvider()
        let model = selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID
        commitNotebookSnapshot(
            targetID: selectedNotebookID,
            title: notebookTitle,
            body: notebookBody,
            modelID: model.isEmpty ? nil : model,
            syncActiveSettings: true
        )
    }

    private func commitPendingNotebookSnapshotIfMatching(
        docID: String?,
        title: String,
        body: String,
        modelID: String?
    ) {
        pendingNotebookSaveSnapshot = nil
        commitNotebookSnapshot(
            targetID: docID,
            title: title,
            body: body,
            modelID: modelID,
            syncActiveSettings: false
        )
    }

    private func commitNotebookSnapshot(
        targetID: String?,
        title: String,
        body: String,
        modelID: String?,
        syncActiveSettings: Bool
    ) {
        do {
            let resolvedID = targetID ?? UUID().uuidString
            let existing = targetID.flatMap { try? store.getNotebook(id: $0) }
            let isStillSelected = (targetID == nil && selectedNotebookID == nil) || (resolvedID == selectedNotebookID)

            // Field-scoped autosave (`notebook.js:doSave`): existing notebooks only update
            // title/content/model_id on body debounce so prompt/params writes never clobber.
            let resolvedSystemPrompt: String
            let resolvedParams: SamplerSettings?
            let resolvedPresetID: String?
            if let existing, !syncActiveSettings {
                resolvedSystemPrompt = existing.systemPrompt
                resolvedParams = existing.params
                resolvedPresetID = existing.appliedPresetID
            } else {
                resolvedSystemPrompt = isStillSelected ? notebookSettings.systemPrompt : (existing?.systemPrompt ?? "")
                resolvedParams = isStillSelected ? notebookSettings.currentManualParamsForPersistence : existing?.params
                resolvedPresetID = isStillSelected ? notebookSettings.appliedPresetID : existing?.appliedPresetID
            }

            let resolvedModel = (modelID?.isEmpty == false) ? modelID : existing?.modelID
            let nb = NotebookRecord(
                id: resolvedID,
                title: title,
                body: body,
                systemPrompt: resolvedSystemPrompt,
                modelID: resolvedModel,
                params: resolvedParams,
                appliedPresetID: resolvedPresetID
            )
            try store.upsertNotebook(nb)
            if isStillSelected {
                if selectedNotebookID == nil {
                    selectedNotebookID = nb.id
                    notebookSettings.saveLastDocumentID(nb.id)
                    let targetModel = nb.modelID.flatMap { id in
                        generationModelsProvider().first(where: { $0.modelID == id })
                    } ?? activeModelProvider()
                    notebookSettings.hydrateDocument(
                        id: nb.id,
                        systemPrompt: nb.systemPrompt,
                        params: nb.params,
                        appliedPresetID: nb.appliedPresetID,
                        activeModel: targetModel,
                        engineContract: nil
                    )
                } else {
                    notebookSettings.saveLastDocumentID(nb.id)
                }
                if pendingNotebookSaveSnapshot == nil {
                    isNotebookDirty = false
                }
            }
            notebooks = try store.listNotebooks()

            if let remote = remoteBackendProvider() {
                Task {
                    if existing != nil && !syncActiveSettings {
                        _ = try? await remote.updateNotebook(
                            id: nb.id,
                            title: nb.title,
                            content: nb.body,
                            modelID: nb.modelID
                        )
                    } else {
                        _ = try? await remote.updateNotebook(
                            id: nb.id,
                            title: nb.title,
                            content: nb.body,
                            systemPrompt: nb.systemPrompt,
                            modelID: nb.modelID,
                            params: resolvedParams,
                            appliedPresetID: resolvedPresetID
                        )
                    }
                }
            }
        } catch {
            onStatusMessage("Notebook save failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Inline Notebook Generation & Export

    public func cancelNotebookGeneration() {
        guard let reqID = activeNotebookRequestID else { return }
        _ = registry.abort(requestID: reqID)
        if let remote = remoteBackendProvider() {
            Task { try? await remote.abortRequest(id: reqID) }
        }
    }

    public var hasNotebookSelection: Bool {
        notebookSelectedRange.length > 0
    }

    public func continueNotebookGeneration() async {
        guard !isNotebookGenerating else { return }
        let selectedID = activeModelIDProvider()
        let modelID = selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID
        guard !modelID.isEmpty else {
            onStatusMessage("No generation model selected for notebook continuation.")
            return
        }

        let nsBody = notebookBody as NSString
        let clamped = Self.clampUTF16Range(notebookSelectedRange, length: nsBody.length)
        let insertionPoint = clamped.location + clamped.length
        let head = nsBody.substring(to: insertionPoint)
        let tail = nsBody.substring(from: insertionPoint)

        // Allow generating into an empty notebook (or before any text) using upstream default prompt (`notebook.js:543`).
        let trimmedHead = head.trimmingCharacters(in: .whitespacesAndNewlines)
        let promptContent = trimmedHead.isEmpty ? "Continue writing." : head

        let trimmedNbSys = notebookSettings.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        await streamNotebookInsertion(
            modelID: modelID,
            systemOverride: trimmedNbSys.isEmpty ? nil : notebookSettings.systemPrompt,
            promptContent: promptContent,
            head: head,
            prefixInjection: "",
            tail: tail
        )
    }

    public func summarizeNotebook() async {
        guard !isNotebookGenerating else { return }
        let selectedID = activeModelIDProvider()
        let modelID = selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID
        guard !modelID.isEmpty else {
            onStatusMessage("No generation model selected for notebook summary.")
            return
        }
        let body = notebookBody
        guard !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        await streamNotebookInsertion(
            modelID: modelID,
            systemOverride: "You are a concise technical writer. Summarize the key points in clean bullet form.",
            promptContent: body,
            head: body,
            prefixInjection: "\n\n## Summary\n",
            tail: ""
        )
    }

    public func analyzeNotebookSelection() async {
        guard !isNotebookGenerating else { return }
        let selectedID = activeModelIDProvider()
        let modelID = selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID
        guard !modelID.isEmpty else {
            onStatusMessage("No generation model selected for notebook analysis.")
            return
        }

        let nsBody = notebookBody as NSString
        let clamped = Self.clampUTF16Range(notebookSelectedRange, length: nsBody.length)
        guard clamped.length > 0 else {
            onStatusMessage("Select a text span in the notebook to analyze.")
            return
        }

        let selectedSpan = nsBody.substring(with: clamped)
        let insertionPoint = clamped.location + clamped.length
        let head = nsBody.substring(to: insertionPoint)
        let tail = nsBody.substring(from: insertionPoint)

        await streamNotebookInsertion(
            modelID: modelID,
            systemOverride: "Critique the key assumptions, edge cases, and concrete improvements for the selected passage.",
            promptContent: selectedSpan,
            head: head,
            prefixInjection: "\n\n## Selection Analysis\n",
            tail: tail
        )
    }

    private func streamNotebookInsertion(
        modelID: String,
        systemOverride: String?,
        promptContent: String,
        head: String,
        prefixInjection: String,
        tail: String
    ) async {
        notebookSettings.flushPendingWrites()
        if selectedNotebookID == nil {
            saveNotebook()
        }
        guard let targetNotebookID = selectedNotebookID else {
            onStatusMessage("Could not persist active notebook before streaming.")
            return
        }

        let initialExisting = try? store.getNotebook(id: targetNotebookID)
        let targetSystemPrompt = notebookSettings.systemPrompt
        let targetParams = notebookSettings.currentManualParamsForPersistence ?? initialExisting?.params
        let targetAppliedPresetID = notebookSettings.appliedPresetID ?? initialExisting?.appliedPresetID

        // Capture sampling exclusively from `notebookSettings` before any async yield!
        let resolvedEffective = notebookSettings.currentResolvedSamplers.effective
        let sparseOverlay = notebookSettings.requestSamplerOverlay
        let supportsThinking = notebookSettings.activeModelSupportsThinking
        let thinkingEnabled = notebookSettings.effectiveThinkingEnabled
        let depthOptions = notebookSettings.activeModelDepthOptions
        let depthValue = depthOptions.isEmpty ? nil : notebookSettings.effectiveThinkingDepth

        let reqID = "nb_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(10))"
        activeNotebookRequestID = reqID
        activeNotebookStreamingID = targetNotebookID
        isNotebookGenerating = true
        onStatusMessage(nil)

        let fixedPrefix = head + prefixInjection
        let fixedPrefixUTF16 = (fixedPrefix as NSString).length
        var workingBody = fixedPrefix + tail
        if !prefixInjection.isEmpty && selectedNotebookID == targetNotebookID {
            isApplyingProgrammaticState = true
            notebookBody = workingBody
            notebookSelectedRange = NSRange(location: fixedPrefixUTF16, length: 0)
            isApplyingProgrammaticState = false
        }

        do {
            let provider = try await scheduler.provider(for: modelID)
            let isRemote = provider is RemoteHeylookProvider
            let effective = isRemote ? sparseOverlay : resolvedEffective

            let req = MessagesRequest(
                model: modelID,
                messages: [MessagesRequest.MessageTurn(role: "user", content: promptContent)],
                system: systemOverride,
                maxTokens: effective.maxTokens ?? (isRemote ? nil : 1024),
                temperature: effective.temperature,
                topP: effective.topP,
                topK: effective.topK,
                minP: effective.minP,
                repetitionPenalty: effective.repetitionPenalty,
                repetitionContextSize: effective.repetitionContextSize,
                presencePenalty: effective.presencePenalty,
                seed: effective.seed,
                stream: true,
                thinking: isRemote ? effective.thinking : (supportsThinking ? thinkingEnabled : nil),
                thinkingDepth: isRemote ? effective.thinkingDepth : depthValue,
                thinkingBudget: isRemote ? effective.thinkingBudget : (supportsThinking ? effective.thinkingBudget : nil),
                stopSequences: effective.stopSequences
            )
            registry.register(requestID: reqID, conversationID: nil, modelID: modelID)

            try await scheduler.withComputeLease(for: modelID) { [registry] in
                defer {
                    registry.finish(requestID: reqID)
                }
                var streamedDelta = ""
                let stream = await provider.generate(
                    requestID: reqID,
                    request: req,
                    effectiveSampling: effective,
                    registry: registry
                )
                for try await chunk in stream where !chunk.text.isEmpty {
                    streamedDelta.append(chunk.text)
                    let combined = fixedPrefix + streamedDelta + tail
                    let newCursor = fixedPrefixUTF16 + (streamedDelta as NSString).length
                    await MainActor.run {
                        workingBody = combined
                        if self.selectedNotebookID == targetNotebookID {
                            self.isApplyingProgrammaticState = true
                            self.notebookBody = combined
                            self.notebookSelectedRange = NSRange(location: newCursor, length: 0)
                            self.isApplyingProgrammaticState = false
                        }
                    }
                }
            }
        } catch {
            onStatusMessage("Notebook generation error: \(error.localizedDescription)")
        }

        if activeNotebookRequestID == reqID {
            activeNotebookRequestID = nil
            activeNotebookStreamingID = nil
            isNotebookGenerating = false
        }

        if let latestExisting = try? store.getNotebook(id: targetNotebookID) {
            let persisted = NotebookRecord(
                id: targetNotebookID,
                title: selectedNotebookID == targetNotebookID ? notebookTitle : latestExisting.title,
                body: workingBody,
                systemPrompt: selectedNotebookID == targetNotebookID ? targetSystemPrompt : (latestExisting.systemPrompt.isEmpty ? targetSystemPrompt : latestExisting.systemPrompt),
                modelID: modelID,
                params: latestExisting.params ?? targetParams,
                appliedPresetID: latestExisting.appliedPresetID ?? targetAppliedPresetID
            )
            try? store.upsertNotebook(persisted)
            notebooks = (try? store.listNotebooks()) ?? notebooks
            if selectedNotebookID == targetNotebookID {
                isNotebookDirty = false
            }
        }
        await onGenerationCompleted?()
    }

    private static func clampUTF16Range(_ range: NSRange, length: Int) -> NSRange {
        let loc = min(max(0, range.location), length)
        let rem = max(0, length - loc)
        let len = min(max(0, range.length), rem)
        return NSRange(location: loc, length: len)
    }

    public func saveNotebookToConversations() {
        flushNotebookSaveIfDirty()
        let content = notebookBody.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else {
            onStatusMessage("Write some content in the Notebook before sending to Chat.")
            return
        }
        do {
            let selectedID = activeModelIDProvider()
            let model = selectedID.isEmpty ? (generationModelsProvider().first?.modelID ?? "") : selectedID
            let conv = ConversationRecord(
                title: "Notebook: \(notebookTitle)",
                systemPrompt: notebookSettings.systemPrompt,
                defaultModelID: model.isEmpty ? nil : model,
                params: notebookSettings.currentManualParamsForPersistence,
                appliedPresetID: notebookSettings.appliedPresetID
            )
            try store.upsertConversation(conv)
            try store.appendMessage(
                MessageRecord(
                    conversationID: conv.id,
                    position: 0,
                    role: "user",
                    content: notebookBody,
                    modelID: model.isEmpty ? nil : model
                )
            )
            onSendToChatRequested(conv)
        } catch {
            onStatusMessage("Could not export notebook: \(error.localizedDescription)")
        }
    }

    // MARK: - Presets Catalog Filtering & CRUD

    private func recomputeFilteredPresets() {
        let q = presetSearchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        filteredPresets = presets.filter { preset in
            if let scope = presetScopeFilter, !scope.isEmpty, preset.scope != scope {
                return false
            }
            guard !q.isEmpty else { return true }
            return preset.name.localizedCaseInsensitiveContains(q)
                || preset.description.localizedCaseInsensitiveContains(q)
                || (preset.systemPrompt?.localizedCaseInsensitiveContains(q) ?? false)
        }
    }

    @discardableResult
    public func createNewPresetRecord(
        name: String,
        scope: String = "sampler",
        description: String = "",
        systemPrompt: String? = nil,
        settings: SamplerSettings = SamplerSettings()
    ) -> PresetRecord? {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let cleanSys = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        let record = PresetRecord(
            name: trimmed,
            scope: scope,
            settings: settings,
            systemPrompt: (scope == "full" && !(cleanSys ?? "").isEmpty) ? systemPrompt : nil,
            description: description.trimmingCharacters(in: .whitespacesAndNewlines)
        )
        do {
            let created = try store.createPreset(record)
            presets = try store.listPresets()
            selectedPresetRecordID = created.id
            if let remote = remoteBackendProvider() {
                Task { _ = try? await remote.createPreset(name: created.name, systemPrompt: created.systemPrompt, params: created.settings) }
            }
            onErrorMessage(nil)
            onStatusMessage("Created preset \"\(created.name)\".")
            return created
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Preset creation failed: \(error.localizedDescription)")
            return nil
        }
    }

    @discardableResult
    public func updatePresetRecord(_ updated: PresetRecord) -> PresetRecord? {
        updatePresetRecord(updated, expectedUpdatedUnixMs: updated.updatedUnixMs, forceOverwrite: false)
    }

    @discardableResult
    public func updatePresetRecord(
        _ updated: PresetRecord,
        expectedUpdatedUnixMs: Int64?,
        forceOverwrite: Bool = false
    ) -> PresetRecord? {
        do {
            if !forceOverwrite, let expectedUpdatedUnixMs, let fresh = try? store.getPreset(id: updated.id),
               fresh.updatedUnixMs != expectedUpdatedUnixMs {
                presetConflictServerRecord = fresh
                presetConcurrencyConflictMessage =
                    "Preset \"\(fresh.name)\" changed in another window or background update. Review the latest copy or choose Overwrite Anyway."
                onErrorMessage(presetConcurrencyConflictMessage)
                onStatusMessage(presetConcurrencyConflictMessage)
                return nil
            }
            var stamped = updated
            stamped.updatedUnixMs = SQLiteStore.nowMillis()
            let saved = try store.updatePreset(id: stamped.id, stamped)
            presetConflictServerRecord = nil
            presetConcurrencyConflictMessage = nil
            presets = try store.listPresets()
            selectedPresetRecordID = saved.id
            if let remote = remoteBackendProvider() {
                let clearPrompt: String?? = saved.systemPrompt == nil ? .some(nil) : .some(saved.systemPrompt)
                Task { _ = try? await remote.updatePreset(id: saved.id, name: saved.name, systemPrompt: clearPrompt, params: saved.settings) }
            }
            onErrorMessage(nil)
            onStatusMessage("Saved preset \"\(saved.name)\".")
            return saved
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Preset update failed: \(error.localizedDescription)")
            return nil
        }
    }

    /// Remote-aware preset update enforcing optimistic concurrency (`updated_at` / `updatedUnixMs`)
    /// across both local SQLite and connected remote Heylook server before writing.
    @discardableResult
    public func savePresetRecordCheckingConcurrency(
        _ updated: PresetRecord,
        expectedUpdatedUnixMs: Int64?,
        forceOverwrite: Bool = false
    ) async -> PresetRecord? {
        if !forceOverwrite, let expectedUpdatedUnixMs, let remote = remoteBackendProvider() {
            if let remoteDTO = try? await remote.getPreset(id: updated.id),
               let remoteUpdatedMs = WireTimestamp.unixMs(fromISO8601: remoteDTO.updatedAt),
               remoteUpdatedMs != expectedUpdatedUnixMs {
                let trimmedSys = remoteDTO.systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let remoteRecord = PresetRecord(
                    id: remoteDTO.id,
                    name: remoteDTO.name,
                    scope: trimmedSys.isEmpty ? "sampler" : "full",
                    settings: remoteDTO.params.settings,
                    systemPrompt: trimmedSys.isEmpty ? nil : remoteDTO.systemPrompt,
                    description: updated.description,
                    updatedUnixMs: remoteUpdatedMs,
                    createdUnixMs: WireTimestamp.unixMs(fromISO8601: remoteDTO.createdAt)
                )
                presetConflictServerRecord = remoteRecord
                presetConcurrencyConflictMessage =
                    "Preset \"\(remoteDTO.name)\" was updated on the server while you were editing. Reload or choose Overwrite Anyway."
                onErrorMessage(presetConcurrencyConflictMessage)
                onStatusMessage(presetConcurrencyConflictMessage)
                return nil
            }
        }
        return updatePresetRecord(updated, expectedUpdatedUnixMs: expectedUpdatedUnixMs, forceOverwrite: forceOverwrite)
    }

    public func clearPresetConflict() {
        presetConflictServerRecord = nil
        presetConcurrencyConflictMessage = nil
    }

    @discardableResult
    public func duplicatePresetRecord(_ source: PresetRecord) -> PresetRecord? {
        var candidateName = "\(source.name) Copy"
        var suffix = 2
        while presets.contains(where: { $0.name.caseInsensitiveCompare(candidateName) == .orderedSame }) {
            candidateName = "\(source.name) Copy \(suffix)"
            suffix += 1
        }
        return createNewPresetRecord(
            name: candidateName,
            scope: source.scope,
            description: source.description,
            systemPrompt: source.systemPrompt,
            settings: source.settings
        )
    }

    public func deletePresetByID(_ id: String) {
        let target = presets.first(where: { $0.id == id })
        do {
            _ = try store.deletePreset(id: id)
            presets = try store.listPresets()
            if selectedPresetRecordID == id {
                selectedPresetRecordID = presets.first?.id
            }
            if let remote = remoteBackendProvider() {
                Task { try? await remote.deletePreset(id: id) }
            }
            onErrorMessage(nil)
            onStatusMessage("Deleted preset \"\(target?.name ?? id)\".")
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Preset delete failed: \(error.localizedDescription)")
        }
    }

    public func persistNotebookParamsCallback(docID: String, params: SamplerSettings?, appliedPresetID: String?) {
        guard var nb = try? store.getNotebook(id: docID) else { return }
        guard nb.params != params || nb.appliedPresetID != appliedPresetID else { return }
        nb.params = params
        nb.appliedPresetID = appliedPresetID
        nb.updatedUnixMs = SQLiteStore.nowMillis()
        try? store.upsertNotebook(nb)
        notebooks = (try? store.listNotebooks()) ?? notebooks
        if let remote = remoteBackendProvider() {
            Task {
                _ = try? await remote.updateNotebook(
                    id: docID,
                    params: params,
                    appliedPresetID: appliedPresetID
                )
            }
        }
    }

    public func persistNotebookSystemPromptCallback(docID: String, systemPrompt: String) {
        guard var nb = try? store.getNotebook(id: docID) else { return }
        guard nb.systemPrompt != systemPrompt else { return }
        nb.systemPrompt = systemPrompt
        nb.updatedUnixMs = SQLiteStore.nowMillis()
        try? store.upsertNotebook(nb)
        notebooks = (try? store.listNotebooks()) ?? notebooks
        if let remote = remoteBackendProvider() {
            Task { _ = try? await remote.updateNotebook(id: docID, systemPrompt: systemPrompt) }
        }
    }

    // MARK: - Notebook Model & 4-State System Prompt Chip

    public var availableGenerationModels: [EngineDescription] {
        generationModelsProvider()
    }

    public var activeNotebookModelID: String {
        if let nbID = selectedNotebookID,
           let nb = notebooks.first(where: { $0.id == nbID }),
           let modelID = nb.modelID,
           !modelID.isEmpty {
            return modelID
        }
        return activeModelIDProvider()
    }

    public func selectNotebookModel(_ modelID: String) {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        onSelectModelID(trimmed)
        let modelDesc = generationModelsProvider().first(where: { $0.modelID == trimmed })
        notebookSettings.reconcileSettings(for: modelDesc, engineContract: nil, persistIfDropped: true)

        guard let docID = selectedNotebookID,
              var nb = try? store.getNotebook(id: docID) else {
            noteNotebookUserEdited()
            return
        }
        guard nb.modelID != trimmed else { return }
        nb.modelID = trimmed
        nb.updatedUnixMs = SQLiteStore.nowMillis()
        try? store.upsertNotebook(nb)
        notebooks = (try? store.listNotebooks()) ?? notebooks
        if let remote = remoteBackendProvider() {
            Task { _ = try? await remote.updateNotebook(id: docID, modelID: trimmed) }
        }
    }

    /// 4-state prompt chip derivation matching upstream `preset-bar.js:promptState()`.
    public var notebookPromptChipState: NotebookPromptChipState {
        let livePrompt = notebookSettings.systemPrompt
        let trimmedLive = livePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if let stamped = notebookSettings.stampedPreset,
           let presetSys = stamped.systemPrompt,
           !presetSys.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return livePrompt == presetSys
                ? .presetUnmodified(name: stamped.name)
                : .presetModified(name: stamped.name)
        }
        return trimmedLive.isEmpty ? .empty : .custom
    }

    // MARK: - Preset Preview Against Model

    public func refreshRemotePublicModelsForPreview() async {
        guard let remote = remoteBackendProvider() else { return }
        let publicModels = await remote.listPublicModels()
        if !publicModels.isEmpty {
            cachedRemotePublicRows = Dictionary(uniqueKeysWithValues: publicModels.map { ($0.id, $0) })
        }
    }

    public func resolvePresetPreview(
        for modelID: String?,
        draftSettings: SamplerSettings
    ) -> PresetModelPreviewSummary? {
        let models = generationModelsProvider()
        guard let targetID = modelID?.isEmpty == false ? modelID : (activeModelIDProvider().isEmpty ? models.first?.modelID : activeModelIDProvider()) else {
            return nil
        }
        let desc = models.first(where: { $0.modelID == targetID })
        let row: PublicModelWireRow
        let isRemoteDerived: Bool
        if let desc, desc.provider == .remote, let cachedRemote = cachedRemotePublicRows[desc.modelID] {
            row = cachedRemote
            isRemoteDerived = true
        } else if let desc {
            row = HeylookConfig.buildPublicWireRow(for: desc)
            isRemoteDerived = false
        } else if let cachedRemote = cachedRemotePublicRows[targetID] {
            row = cachedRemote
            isRemoteDerived = true
        } else {
            return nil
        }

        var entries: [PresetModelPreviewEntry] = []
        var unsupportedCount = 0
        for key in SamplerCascade.SamplerFieldKey.allCases {
            let wireKey = key.canonicalWireKey
            let presetVal = Self.jsonValue(in: draftSettings, for: key)
            let defaultVal = row.samplerDefaults[wireKey]
            let source = row.samplerSources[wireKey]
            let effectiveVal = presetVal ?? defaultVal
            let unsupported = Self.unsupportedPresetReason(
                key: key,
                presetValue: presetVal,
                row: row,
                modelDescription: desc
            )
            if unsupported != nil {
                unsupportedCount += 1
            }
            entries.append(
                PresetModelPreviewEntry(
                    id: wireKey,
                    label: Self.label(for: key),
                    presetValueText: presetVal.map(Self.formatJSONValue),
                    modelDefaultText: defaultVal.map(Self.formatJSONValue),
                    modelDefaultSource: source,
                    effectiveValueText: effectiveVal.map(Self.formatJSONValue) ?? "unset",
                    unsupportedReason: unsupported
                )
            )
        }

        return PresetModelPreviewSummary(
            modelID: row.id,
            thinkingDefault: row.thinkingDefault,
            capabilities: row.capabilities,
            entries: entries,
            unsupportedCount: unsupportedCount,
            isRemoteDerived: isRemoteDerived
        )
    }

    private static func jsonValue(in settings: SamplerSettings, for key: SamplerCascade.SamplerFieldKey) -> JSONValue? {
        switch key {
        case .temperature: return settings.temperature.map(JSONValue.double)
        case .topP: return settings.topP.map(JSONValue.double)
        case .topK: return settings.topK.map(JSONValue.int)
        case .minP: return settings.minP.map(JSONValue.double)
        case .repetitionPenalty: return settings.repetitionPenalty.map(JSONValue.double)
        case .repetitionContextSize: return settings.repetitionContextSize.map(JSONValue.int)
        case .presencePenalty: return settings.presencePenalty.map(JSONValue.double)
        case .maxTokens: return settings.maxTokens.map(JSONValue.int)
        case .seed: return settings.seed.map(JSONValue.int)
        case .thinking: return settings.thinking.map(JSONValue.bool)
        case .thinkingDepth: return settings.thinkingDepth.map(JSONValue.string)
        case .thinkingBudget: return settings.thinkingBudget.map(JSONValue.int)
        case .stopSequences:
            guard let seq = settings.stopSequences, !seq.isEmpty else { return nil }
            return .array(seq.map(JSONValue.string))
        }
    }

    private static func label(for key: SamplerCascade.SamplerFieldKey) -> String {
        switch key {
        case .temperature: return "temperature"
        case .topP: return "top_p"
        case .topK: return "top_k"
        case .minP: return "min_p"
        case .repetitionPenalty: return "repetition_penalty"
        case .repetitionContextSize: return "repetition_context_size"
        case .presencePenalty: return "presence_penalty"
        case .maxTokens: return "max_tokens"
        case .seed: return "seed"
        case .thinking: return "enable_thinking"
        case .thinkingDepth: return "thinking_depth"
        case .thinkingBudget: return "thinking_budget"
        case .stopSequences: return "stop_sequences"
        }
    }

    private static func formatJSONValue(_ value: JSONValue) -> String {
        switch value {
        case .null: return "null"
        case .bool(let v): return v ? "true" : "false"
        case .int(let v): return String(v)
        case .double(let v): return String(format: "%.3g", v)
        case .string(let v): return "\"\(v)\""
        case .array(let arr): return "[\(arr.map(formatJSONValue).joined(separator: ", "))]"
        case .object: return "{...}"
        }
    }

    private static func unsupportedPresetReason(
        key: SamplerCascade.SamplerFieldKey,
        presetValue: JSONValue?,
        row: PublicModelWireRow,
        modelDescription: EngineDescription?
    ) -> String? {
        guard let presetValue else { return nil }
        let wireKey = key.canonicalWireKey
        if let supportedFields = row.engine.decoding.requestFields.value,
           !supportedFields.contains(wireKey) && !supportedFields.contains(key.rawValue) {
            return "Not in model decoding.request_fields"
        }

        let supportsThinking = row.capabilities.contains("thinking")
            || (modelDescription?.supportsThinking.value ?? false)
        let controls = modelDescription?.thinkingControls
        let thinkingFacts = row.engine.thinking

        switch key {
        case .thinking:
            if !supportsThinking {
                return "Model does not support thinking"
            }
            let canDisable = controls?.canDisableThinking ?? true
            if !canDisable, case .bool(false) = presetValue {
                return "Thinking is always-on for this model (cannot disable)"
            }
            return nil

        case .thinkingDepth:
            if !supportsThinking {
                return "Model does not support thinking"
            }
            let supportedDepths = controls?.depthValues ?? thinkingFacts?.depth?.values ?? []
            if supportedDepths.isEmpty {
                return "Model has no depth / reasoning_effort control"
            }
            if case .string(let depth) = presetValue, !supportedDepths.contains(depth) {
                return "Supported levels: \(supportedDepths.joined(separator: ", "))"
            }
            return nil

        case .thinkingBudget:
            if !supportsThinking {
                return "Model does not support thinking"
            }
            let budgetEnforced = (thinkingFacts?.budget?.enforced ?? false)
                || row.capabilities.contains("thinking_budget")
                || (controls?.budgetEnforced ?? false)
            if !budgetEnforced {
                return "thinking_budget is not enforced on this model"
            }
            return nil

        default:
            return nil
        }
    }
}
