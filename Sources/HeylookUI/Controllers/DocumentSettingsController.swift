import Foundation
import Observation
import HeylookCore

/// Identifies which document domain a `DocumentSettingsController` instance owns.
public enum DocumentSettingsScope: String, Sendable {
    case conversation
    case notebook
}

/// Armed prompt-replacement request when applying a preset with a non-empty `systemPrompt`
/// onto a document that already has a different non-empty `systemPrompt`.
public struct PendingPromptReplacement: Identifiable, Sendable, Equatable {
    public let id: String
    public let preset: PresetRecord
    public let existingPrompt: String
    public let incomingPrompt: String

    public init(preset: PresetRecord, existingPrompt: String, incomingPrompt: String) {
        self.id = preset.id
        self.preset = preset
        self.existingPrompt = existingPrompt
        self.incomingPrompt = incomingPrompt
    }
}

/// Drift summary comparing a document's live prompt and sparse `params` against its stamped `PresetRecord`.
public struct DocumentPresetDriftReport: Sendable, Equatable {
    public let presetName: String?
    public let promptChanged: Bool
    public let changedFieldLabels: [String]

    public var hasChanges: Bool {
        promptChanged || !changedFieldLabels.isEmpty
    }

    public var provenanceSummary: String {
        guard let presetName else { return "No preset" }
        guard hasChanges else { return "From preset \(presetName)" }
        var halves: [String] = []
        if promptChanged {
            halves.append("prompt")
        }
        if !changedFieldLabels.isEmpty {
            let countWord = Self.countWords(changedFieldLabels.count)
            let noun = changedFieldLabels.count == 1 ? "knob" : "knobs"
            halves.append("\(countWord) \(noun) (\(changedFieldLabels.joined(separator: ", ")))")
        }
        return "\(presetName), modified: \(halves.joined(separator: " and "))"
    }

    private static func countWords(_ count: Int) -> String {
        let words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
        return count < words.count ? words[count] : "\(count)"
    }
}

/// Per-document scoped `@Observable` controller managing sparse sampler overrides, system prompt
/// drafts/debounced persistence, preset provenance ("Preset is a Copy" + "Null-Means-Cascade" +
/// "Override-Box Rule"), capability reconciliation, and `UserDefaults` session restore.
///
/// The workbench shell instantiates **separate** instances for `chatSettings` and `notebookSettings`
/// so switching surfaces never overwrites shared sampler overrides or collides mid-stream.
@MainActor
@Observable
public final class DocumentSettingsController {
    public static let chatDraftPromptDefaultsKey = "heylook.draftPrompt"
    public static let legacyChatDraftPromptDefaultsKey = "heylook.chat.draft-prompt"
    public static let lastConversationDefaultsKey = "heylook.chat.last-conversation"
    public static let lastNotebookDefaultsKey = "heylook.notebook.last-notebook"

    public let scopeKind: DocumentSettingsScope
    public let store: SQLiteStore
    private let userDefaults: UserDefaults

    // MARK: - Active Document Binding & State

    public private(set) var activeDocumentID: String?

    public var systemPrompt: String = "" {
        didSet {
            guard !isApplyingProgrammaticState else { return }
            recomputeDriftState()
            if scopeKind == .conversation && activeDocumentID == nil {
                persistPreDocDraftPrompt(systemPrompt)
            } else {
                scheduleSystemPromptSave()
            }
            onSchedulePromptPreview?()
        }
    }

    public var appliedPresetID: String? = nil {
        didSet {
            guard !isApplyingProgrammaticState else { return }
            recomputeDriftState()
        }
    }

    public private(set) var stampedPresetTimestamp: Int64?
    public var isPresetModified: Bool = false
    public var pendingPromptReplacement: PendingPromptReplacement?
    public var reconcileDisclosureMessage: String?

    // Sparse per-document overrides (`nil` = cascade to model / sidecar / vendor / built-in defaults)
    public var temperatureOverride: Double? = nil { didSet { handleSamplerKnobEdited() } }
    public var topPOverride: Double? = nil { didSet { handleSamplerKnobEdited() } }
    public var topKOverride: Int? = nil { didSet { handleSamplerKnobEdited() } }
    public var minPOverride: Double? = nil { didSet { handleSamplerKnobEdited() } }
    public var repetitionPenaltyOverride: Double? = nil { didSet { handleSamplerKnobEdited() } }
    public var repetitionContextSizeOverride: Int? = nil { didSet { handleSamplerKnobEdited() } }
    public var presencePenaltyOverride: Double? = nil { didSet { handleSamplerKnobEdited() } }
    public var maxTokensOverride: Int? = nil { didSet { handleSamplerKnobEdited() } }
    public var seedOverride: Int? = nil { didSet { handleSamplerKnobEdited() } }
    public var stopSequencesOverride: [String]? = nil { didSet { handleSamplerKnobEdited() } }
    public var thinkingEnabledOverride: Bool? = nil { didSet { handleSamplerKnobEdited() } }
    public var thinkingDepthOverride: String? = nil { didSet { handleSamplerKnobEdited() } }
    public var thinkingBudgetOverride: Int? = nil { didSet { handleSamplerKnobEdited() } }

    // MARK: - Wiring Callbacks (Configured by Shell Coordinator)

    @ObservationIgnored public var activeModelProvider: () -> EngineDescription? = { nil }
    @ObservationIgnored public var activeContractProvider: () -> EngineContract? = { nil }
    @ObservationIgnored public var livePresetsProvider: () -> [PresetRecord] = { [] }
    @ObservationIgnored public var onPresetsChanged: ([PresetRecord]) -> Void = { _ in }
    @ObservationIgnored public var onPersistDocumentParams: (_ documentID: String, _ params: SamplerSettings?, _ appliedPresetID: String?) -> Void = { _, _, _ in }
    @ObservationIgnored public var onPersistDocumentSystemPrompt: (_ documentID: String, _ systemPrompt: String) -> Void = { _, _ in }
    @ObservationIgnored public var onSchedulePromptPreview: (() -> Void)?
    @ObservationIgnored public var onStatusMessage: (String) -> Void = { _ in }
    @ObservationIgnored public var onErrorMessage: (String) -> Void = { _ in }
    @ObservationIgnored public var onRequestDeletePresetConfirmation: (PresetRecord) -> Void = { _ in }
    @ObservationIgnored public var remoteCreatePresetHandler: ((_ record: PresetRecord) async throws -> PresetRecord)?
    @ObservationIgnored public var remoteUpdatePresetHandler: ((_ record: PresetRecord) async throws -> PresetRecord)?

    // MARK: - Debounced Write Captures (Captured at Schedule Time)

    @ObservationIgnored private var isApplyingProgrammaticState: Bool = false
    @ObservationIgnored private var pendingParamsSaveTask: Task<Void, Never>?
    @ObservationIgnored private var pendingParamsSnapshot: (docID: String, params: SamplerSettings?, appliedPresetID: String?)?
    @ObservationIgnored private var pendingPromptSaveTask: Task<Void, Never>?
    @ObservationIgnored private var pendingPromptSnapshot: (docID: String, systemPrompt: String)?

    public init(
        scopeKind: DocumentSettingsScope,
        store: SQLiteStore,
        userDefaults: UserDefaults = .standard
    ) {
        self.scopeKind = scopeKind
        self.store = store
        self.userDefaults = userDefaults
        if scopeKind == .conversation {
            self.systemPrompt = Self.loadInitialChatDraftPrompt(from: userDefaults)
        }
    }

    // MARK: - Session Restore Keys

    public var lastDocumentDefaultsKey: String {
        switch scopeKind {
        case .conversation: return Self.lastConversationDefaultsKey
        case .notebook: return Self.lastNotebookDefaultsKey
        }
    }

    public func readLastDocumentID() -> String? {
        let raw = userDefaults.string(forKey: lastDocumentDefaultsKey)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (raw?.isEmpty == false) ? raw : nil
    }

    public func saveLastDocumentID(_ id: String?) {
        if let id, !id.isEmpty {
            userDefaults.set(id, forKey: lastDocumentDefaultsKey)
        } else {
            userDefaults.removeObject(forKey: lastDocumentDefaultsKey)
        }
    }

    public func clearPreDocDraftPrompt() {
        guard scopeKind == .conversation else { return }
        userDefaults.removeObject(forKey: Self.chatDraftPromptDefaultsKey)
        userDefaults.removeObject(forKey: Self.legacyChatDraftPromptDefaultsKey)
    }

    private func persistPreDocDraftPrompt(_ value: String) {
        guard scopeKind == .conversation else { return }
        if value.isEmpty {
            userDefaults.removeObject(forKey: Self.chatDraftPromptDefaultsKey)
        } else {
            userDefaults.set(value, forKey: Self.chatDraftPromptDefaultsKey)
        }
    }

    private static func loadInitialChatDraftPrompt(from defaults: UserDefaults) -> String {
        if let current = defaults.string(forKey: chatDraftPromptDefaultsKey), !current.isEmpty {
            return current
        }
        if let legacy = defaults.string(forKey: legacyChatDraftPromptDefaultsKey), !legacy.isEmpty {
            defaults.set(legacy, forKey: chatDraftPromptDefaultsKey)
            defaults.removeObject(forKey: legacyChatDraftPromptDefaultsKey)
            return legacy
        }
        return ""
    }

    // MARK: - Sparse Bag & Model Capability Filtering

    public var unmaskedManualSamplerOverlay: SamplerSettings {
        SamplerSettings(
            temperature: temperatureOverride,
            topP: topPOverride,
            topK: topKOverride,
            minP: minPOverride,
            repetitionPenalty: repetitionPenaltyOverride,
            repetitionContextSize: repetitionContextSizeOverride,
            presencePenalty: presencePenaltyOverride,
            maxTokens: maxTokensOverride,
            seed: seedOverride,
            thinking: thinkingEnabledOverride,
            thinkingDepth: thinkingDepthOverride,
            thinkingBudget: thinkingBudgetOverride,
            stopSequences: stopSequencesOverride
        )
    }

    public var hasAnyManualSamplerOverride: Bool {
        temperatureOverride != nil
            || topPOverride != nil
            || topKOverride != nil
            || minPOverride != nil
            || repetitionPenaltyOverride != nil
            || repetitionContextSizeOverride != nil
            || presencePenaltyOverride != nil
            || maxTokensOverride != nil
            || seedOverride != nil
            || stopSequencesOverride != nil
            || thinkingEnabledOverride != nil
            || thinkingDepthOverride != nil
            || thinkingBudgetOverride != nil
    }

    public var currentManualParamsForPersistence: SamplerSettings? {
        hasAnyManualSamplerOverride ? unmaskedManualSamplerOverlay : nil
    }

    public var stampedPreset: PresetRecord? {
        guard let appliedPresetID else { return nil }
        return livePresetsProvider().first(where: { $0.id == appliedPresetID })
    }

    /// Resolves display name from the document's `appliedPresetID` stamp against live presets.
    /// Self-heals to `nil` when the stamped preset has been deleted.
    public var selectedPresetName: String? {
        stampedPreset?.name
    }

    public var activeModelSupportsThinking: Bool {
        if let controls = activeModelProvider()?.thinkingControls {
            return controls.supportsThinking
        }
        return activeModelProvider()?.supportsThinking.value ?? false
    }

    public var activeModelCanDisableThinking: Bool {
        if let controls = activeModelProvider()?.thinkingControls {
            return controls.canDisableThinking
        }
        return activeModelSupportsThinking
    }

    public var activeModelDepthOptions: [String] {
        activeModelProvider()?.thinkingControls?.depthValues ?? []
    }

    /// Sparse sampler overlay trimmed to what the current active model accepts (`Null-Means-Cascade`).
    public var requestSamplerOverlay: SamplerSettings {
        filteredSettings(unmaskedManualSamplerOverlay, for: activeModelProvider(), contract: activeContractProvider())
    }

    /// Resolves the 5-layer cascade strictly from the document's own copied overrides (`preset: nil`),
    /// honoring **Preset is a Copy** so later edits to a stored preset never mutate open documents.
    public var currentResolvedSamplers: SamplerCascade.ResolvedSampling {
        let modelDesc = activeModelProvider()
        return HeylookConfig.resolveEffectiveSampling(
            modelPath: modelDesc?.provider == .remote ? nil : modelDesc?.modelPath,
            preset: nil,
            request: requestSamplerOverlay,
            thinkingCapable: modelDesc?.supportsThinking.value
        )
    }

    public var effectiveThinkingEnabled: Bool {
        guard activeModelSupportsThinking else { return false }
        if !activeModelCanDisableThinking { return true }
        if let explicit = currentResolvedSamplers.effective.thinking {
            return explicit
        }
        return activeModelProvider()?.thinkingControls?.defaultEnabled ?? true
    }

    public var effectiveThinkingDepth: String {
        if let explicit = currentResolvedSamplers.effective.thinkingDepth,
           activeModelDepthOptions.contains(explicit) {
            return explicit
        }
        return activeModelDepthOptions.first ?? "medium"
    }

    // MARK: - Document Lifecycle, Hydration & Inheritance

    /// Loads an existing or blank document into this controller without scheduling redundant writes.
    public func hydrateDocument(
        id: String?,
        systemPrompt: String,
        params: SamplerSettings?,
        appliedPresetID: String?,
        activeModel: EngineDescription? = nil,
        engineContract: EngineContract? = nil
    ) {
        flushPendingWrites()

        isApplyingProgrammaticState = true
        self.activeDocumentID = id
        self.pendingPromptReplacement = nil
        self.reconcileDisclosureMessage = nil
        self.systemPrompt = systemPrompt
        self.appliedPresetID = appliedPresetID
        if let appliedPresetID, let preset = livePresetsProvider().first(where: { $0.id == appliedPresetID }) {
            self.stampedPresetTimestamp = preset.updatedUnixMs
        } else {
            self.stampedPresetTimestamp = nil
        }
        setOverridesFromSettings(params)
        isApplyingProgrammaticState = false

        let resolvedModel = activeModel ?? activeModelProvider()
        let resolvedContract = engineContract ?? activeContractProvider()
        _ = reconcileSettings(for: resolvedModel, engineContract: resolvedContract, persistIfDropped: false)
        recomputeDriftState()
        onSchedulePromptPreview?()
    }

    /// Computes initial settings for a new document (`preset-bar.js:presetForNewDoc` + draft carry-forward):
    /// - When a document IS open: returns that document's stamped preset as stored (`appliedPresetID`,
    ///   `systemPrompt`, sparse copy of `settings`), or blank if unstamped — never leaking drifted edits.
    /// - When NO document is open: promotes the current pre-doc drawer draft into the new document
    ///   and clears `UserDefaults` `"heylook.draftPrompt"`.
    public func presetForNewDoc() -> (appliedPresetID: String?, systemPrompt: String, params: SamplerSettings?) {
        if activeDocumentID != nil {
            guard let stamped = stampedPreset else {
                return (nil, "", nil)
            }
            let prompt = stamped.systemPrompt ?? ""
            let sparse = filteredSettings(stamped.settings, for: activeModelProvider(), contract: activeContractProvider())
            let params: SamplerSettings? = Self.isEmptySettings(sparse) ? nil : sparse
            return (stamped.id, prompt, params)
        } else {
            let carryPresetID = stampedPreset?.id
            let carryPrompt = systemPrompt
            let carryParams = currentManualParamsForPersistence
            clearPreDocDraftPrompt()
            return (carryPresetID, carryPrompt, carryParams)
        }
    }

    // MARK: - Model Capability Reconciliation

    /// Drops settings from this document's overrides that `model` (and optional `engineContract`) cannot
    /// honor, setting `reconcileDisclosureMessage` when any key was removed.
    @discardableResult
    public func reconcileSettings(
        for model: EngineDescription?,
        engineContract: EngineContract? = nil,
        persistIfDropped: Bool = true
    ) -> [String] {
        guard let model else { return [] }
        let dropped = Self.inapplicableDescriptions(for: unmaskedManualSamplerOverlay, model: model, contract: engineContract)
        guard !dropped.isEmpty else { return [] }

        isApplyingProgrammaticState = true
        for item in dropped {
            clearOverride(for: item.key)
        }
        isApplyingProgrammaticState = false

        let summary = dropped.map { "\($0.label) \"\($0.valueText)\" (\($0.reason))" }.joined(separator: "; ")
        reconcileDisclosureMessage = "Removed settings this model can't use: \(summary)."
        onStatusMessage(reconcileDisclosureMessage ?? "")
        recomputeDriftState()

        if persistIfDropped, let docID = activeDocumentID {
            persistParamsSnapshotNow(docID: docID)
        }
        onSchedulePromptPreview?()
        return dropped.map(\.label)
    }

    // MARK: - Preset Operations (Copy, Null-Means-Cascade, Override-Box Rule)

    /// Applies `preset` as a sparse **copy** onto this document.
    ///
    /// Implements the **Override-Box Rule**:
    /// - Empty/nil `preset.systemPrompt` is `"settings only"` and preserves the document's existing prompt.
    /// - Non-empty `preset.systemPrompt` differing from an existing non-empty prompt arms
    ///   `pendingPromptReplacement` unless `forceReplacePrompt` is true.
    @discardableResult
    public func applyPreset(_ preset: PresetRecord, forceReplacePrompt: Bool = false) -> Bool {
        let incomingPrompt = preset.systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let currentPrompt = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)

        if !forceReplacePrompt,
           !incomingPrompt.isEmpty,
           !currentPrompt.isEmpty,
           preset.systemPrompt != systemPrompt {
            pendingPromptReplacement = PendingPromptReplacement(
                preset: preset,
                existingPrompt: systemPrompt,
                incomingPrompt: preset.systemPrompt ?? ""
            )
            return false
        }

        applyPresetConfirmed(preset, replacePrompt: !incomingPrompt.isEmpty)
        return true
    }

    public func confirmPendingPromptReplacement(replacePrompt: Bool) {
        guard let pending = pendingPromptReplacement else { return }
        pendingPromptReplacement = nil
        applyPresetConfirmed(pending.preset, replacePrompt: replacePrompt)
    }

    public func cancelPendingPromptReplacement() {
        pendingPromptReplacement = nil
    }

    private func applyPresetConfirmed(_ preset: PresetRecord, replacePrompt: Bool) {
        pendingParamsSaveTask?.cancel()
        pendingParamsSaveTask = nil
        pendingParamsSnapshot = nil

        isApplyingProgrammaticState = true
        // Preset is a Copy: copy preset's sparse overrides directly onto this document.
        setOverridesFromSettings(preset.settings)
        appliedPresetID = preset.id
        stampedPresetTimestamp = preset.updatedUnixMs

        if replacePrompt, let sp = preset.systemPrompt, !sp.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            systemPrompt = sp
            if scopeKind == .conversation && activeDocumentID == nil {
                persistPreDocDraftPrompt(sp)
            }
        }
        isApplyingProgrammaticState = false

        _ = reconcileSettings(for: activeModelProvider(), engineContract: activeContractProvider(), persistIfDropped: false)
        recomputeDriftState()

        if let docID = activeDocumentID {
            if replacePrompt {
                pendingPromptSaveTask?.cancel()
                pendingPromptSaveTask = nil
                pendingPromptSnapshot = nil
                onPersistDocumentSystemPrompt(docID, systemPrompt)
            }
            persistParamsSnapshotNow(docID: docID)
        }
        onSchedulePromptPreview?()
    }

    /// Detaches any preset stamp AND clears sparse manual overrides (`Model Defaults (No Preset)`).
    public func clearPresetSelection() {
        isApplyingProgrammaticState = true
        appliedPresetID = nil
        stampedPresetTimestamp = nil
        isPresetModified = false
        setOverridesFromSettings(nil)
        isApplyingProgrammaticState = false

        if let docID = activeDocumentID {
            persistParamsSnapshotNow(docID: docID)
        }
        onSchedulePromptPreview?()
    }

    /// Clears only manual sampler knobs (and detaches preset drift if any).
    public func resetManualSamplerOverrides() {
        isApplyingProgrammaticState = true
        setOverridesFromSettings(nil)
        isApplyingProgrammaticState = false
        recomputeDriftState()

        if let docID = activeDocumentID {
            persistParamsSnapshotNow(docID: docID)
        }
        onSchedulePromptPreview?()
    }

    /// Saves ONLY the document's sparse non-nil overrides (`Null-Means-Cascade`) as a new preset.
    @discardableResult
    public func saveAsPreset(name: String, scope: String? = nil, description: String = "") -> PresetRecord? {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let existing = livePresetsProvider()
        if existing.contains(where: { $0.name.caseInsensitiveCompare(trimmed) == .orderedSame }) {
            let msg = "A preset named \"\(trimmed)\" already exists."
            onErrorMessage(msg)
            onStatusMessage(msg)
            return nil
        }

        let trimmedSys = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedScope = scope ?? (trimmedSys.isEmpty ? "sampler" : "full")
        // Null-Means-Cascade: persist ONLY sparse non-nil overrides accepted by the active model.
        let sparseSettings = requestSamplerOverlay
        let record = PresetRecord(
            name: trimmed,
            scope: resolvedScope,
            settings: sparseSettings,
            systemPrompt: resolvedScope == "full" ? (trimmedSys.isEmpty ? nil : systemPrompt) : nil,
            description: description.trimmingCharacters(in: .whitespacesAndNewlines)
        )

        do {
            let created = try store.createPreset(record)
            let updatedList = try store.listPresets()
            onPresetsChanged(updatedList)
            if let remoteCreatePresetHandler {
                Task { _ = try? await remoteCreatePresetHandler(created) }
            }

            isApplyingProgrammaticState = true
            appliedPresetID = created.id
            stampedPresetTimestamp = created.updatedUnixMs
            isPresetModified = false
            isApplyingProgrammaticState = false

            if let docID = activeDocumentID {
                persistParamsSnapshotNow(docID: docID)
            }
            onStatusMessage("Saved preset \"\(created.name)\".")
            return created
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Preset save failed: \(error.localizedDescription)")
            return nil
        }
    }

    /// Overwrites the document's stamped preset using ONLY sparse non-nil overrides (`Null-Means-Cascade`),
    /// guarding against stale updates when the stored preset changed since this drawer read it.
    @discardableResult
    public func updateActivePreset() -> PresetRecord? {
        guard let existing = stampedPreset else { return nil }

        if let fresh = try? store.getPreset(id: existing.id),
           let loadedStamp = stampedPresetTimestamp,
           fresh.updatedUnixMs != loadedStamp {
            let msg = "Preset \"\(existing.name)\" changed since this document loaded it; re-apply or save as new."
            onErrorMessage(msg)
            onStatusMessage(msg)
            return nil
        }

        let trimmedSys = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let nextScope = (existing.scope == "full" || !trimmedSys.isEmpty) ? "full" : "sampler"
        // Null-Means-Cascade: save ONLY pinned sparse overrides, NEVER resolved effective defaults.
        let updated = PresetRecord(
            id: existing.id,
            name: existing.name,
            scope: nextScope,
            settings: requestSamplerOverlay,
            systemPrompt: trimmedSys.isEmpty ? nil : systemPrompt,
            description: existing.description
        )

        do {
            let saved = try store.updatePreset(id: existing.id, updated)
            let updatedList = try store.listPresets()
            onPresetsChanged(updatedList)
            if let remoteUpdatePresetHandler {
                Task { _ = try? await remoteUpdatePresetHandler(saved) }
            }

            isApplyingProgrammaticState = true
            appliedPresetID = saved.id
            stampedPresetTimestamp = saved.updatedUnixMs
            isPresetModified = false
            isApplyingProgrammaticState = false

            if let docID = activeDocumentID {
                persistParamsSnapshotNow(docID: docID)
            }
            onStatusMessage("Updated preset \"\(saved.name)\".")
            return saved
        } catch {
            onErrorMessage(error.localizedDescription)
            onStatusMessage("Preset update failed: \(error.localizedDescription)")
            return nil
        }
    }

    /// Arms explicit confirmation before deleting the active preset — never deletes silently in 1 click.
    public func requestDeleteActivePreset() {
        guard let existing = stampedPreset else { return }
        onRequestDeletePresetConfirmation(existing)
    }

    // MARK: - Override-Box & Field-by-Field Drift Comparison

    public func matchesState(of preset: PresetRecord) -> Bool {
        !driftReport(against: preset).hasChanges
    }

    public func driftReport() -> DocumentPresetDriftReport {
        guard let preset = stampedPreset else {
            return DocumentPresetDriftReport(presetName: nil, promptChanged: false, changedFieldLabels: [])
        }
        return driftReport(against: preset)
    }

    public func driftReport(against preset: PresetRecord) -> DocumentPresetDriftReport {
        // Override-Box Rule: empty/nil preset prompt means "settings only" and is ignored in drift checks.
        let presetPromptTrimmed = preset.systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let promptChanged = !presetPromptTrimmed.isEmpty && (preset.systemPrompt != systemPrompt)

        let model = activeModelProvider()
        let contract = activeContractProvider()
        let current = unmaskedManualSamplerOverlay
        let saved = filteredSettings(preset.settings, for: model, contract: contract)

        var changed: [String] = []
        if current.temperature != saved.temperature { changed.append("temperature") }
        if current.topP != saved.topP { changed.append("top p") }
        if current.topK != saved.topK { changed.append("top k") }
        if current.minP != saved.minP { changed.append("min p") }
        if current.repetitionPenalty != saved.repetitionPenalty { changed.append("repetition penalty") }
        if current.repetitionContextSize != saved.repetitionContextSize { changed.append("repeat window") }
        if current.presencePenalty != saved.presencePenalty { changed.append("presence penalty") }
        if current.maxTokens != saved.maxTokens { changed.append("max tokens") }
        if current.seed != saved.seed { changed.append("seed") }
        if current.stopSequences != saved.stopSequences { changed.append("stop sequences") }
        if current.thinking != saved.thinking { changed.append("thinking") }
        if current.thinkingDepth != saved.thinkingDepth { changed.append("thinking depth") }
        if current.thinkingBudget != saved.thinkingBudget { changed.append("thinking budget") }

        return DocumentPresetDriftReport(
            presetName: preset.name,
            promptChanged: promptChanged,
            changedFieldLabels: changed
        )
    }

    public func recomputeDriftState() {
        guard let preset = stampedPreset else {
            isPresetModified = false
            return
        }
        isPresetModified = !matchesState(of: preset)
    }

    // MARK: - Debounced Writes (Capturing Document ID + Snapshot at Schedule Time)

    private func handleSamplerKnobEdited() {
        guard !isApplyingProgrammaticState else { return }
        recomputeDriftState()
        scheduleParamsSave()
        onSchedulePromptPreview?()
    }

    private func scheduleParamsSave() {
        guard let docID = activeDocumentID else { return }
        // If another document still owes a debounced params write, flush it before capturing the new doc!
        if let existing = pendingParamsSnapshot, existing.docID != docID {
            flushPendingParamsSave()
        }
        pendingParamsSnapshot = (docID: docID, params: currentManualParamsForPersistence, appliedPresetID: appliedPresetID)
        pendingParamsSaveTask?.cancel()
        pendingParamsSaveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled else { return }
            self?.flushPendingParamsSave()
        }
    }

    private func scheduleSystemPromptSave() {
        guard let docID = activeDocumentID else { return }
        if let existing = pendingPromptSnapshot, existing.docID != docID {
            flushPendingPromptSave()
        }
        pendingPromptSnapshot = (docID: docID, systemPrompt: systemPrompt)
        pendingPromptSaveTask?.cancel()
        pendingPromptSaveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled else { return }
            self?.flushPendingPromptSave()
        }
    }

    public func flushPendingWrites() {
        flushPendingPromptSave()
        flushPendingParamsSave()
    }

    public func flushPendingPromptSave() {
        pendingPromptSaveTask?.cancel()
        pendingPromptSaveTask = nil
        guard let snapshot = pendingPromptSnapshot else { return }
        pendingPromptSnapshot = nil
        onPersistDocumentSystemPrompt(snapshot.docID, snapshot.systemPrompt)
    }

    public func flushPendingParamsSave() {
        pendingParamsSaveTask?.cancel()
        pendingParamsSaveTask = nil
        guard let snapshot = pendingParamsSnapshot else { return }
        pendingParamsSnapshot = nil
        onPersistDocumentParams(snapshot.docID, snapshot.params, snapshot.appliedPresetID)
    }

    private func persistParamsSnapshotNow(docID: String) {
        pendingParamsSaveTask?.cancel()
        pendingParamsSaveTask = nil
        pendingParamsSnapshot = nil
        onPersistDocumentParams(docID, currentManualParamsForPersistence, appliedPresetID)
    }

    // MARK: - Internal Field Helpers

    private func setOverridesFromSettings(_ settings: SamplerSettings?) {
        temperatureOverride = settings?.temperature
        topPOverride = settings?.topP
        topKOverride = settings?.topK
        minPOverride = settings?.minP
        repetitionPenaltyOverride = settings?.repetitionPenalty
        repetitionContextSizeOverride = settings?.repetitionContextSize
        presencePenaltyOverride = settings?.presencePenalty
        maxTokensOverride = settings?.maxTokens
        seedOverride = settings?.seed
        stopSequencesOverride = settings?.stopSequences
        thinkingEnabledOverride = settings?.thinking
        thinkingDepthOverride = settings?.thinkingDepth
        thinkingBudgetOverride = settings?.thinkingBudget
    }

    private func clearOverride(for key: SamplerCascade.SamplerFieldKey) {
        switch key {
        case .temperature: temperatureOverride = nil
        case .topP: topPOverride = nil
        case .topK: topKOverride = nil
        case .minP: minPOverride = nil
        case .repetitionPenalty: repetitionPenaltyOverride = nil
        case .repetitionContextSize: repetitionContextSizeOverride = nil
        case .presencePenalty: presencePenaltyOverride = nil
        case .maxTokens: maxTokensOverride = nil
        case .seed: seedOverride = nil
        case .stopSequences: stopSequencesOverride = nil
        case .thinking: thinkingEnabledOverride = nil
        case .thinkingDepth: thinkingDepthOverride = nil
        case .thinkingBudget: thinkingBudgetOverride = nil
        }
    }

    private struct DroppedItem {
        let key: SamplerCascade.SamplerFieldKey
        let label: String
        let valueText: String
        let reason: String
    }

    private func filteredSettings(
        _ raw: SamplerSettings,
        for model: EngineDescription?,
        contract: EngineContract?
    ) -> SamplerSettings {
        guard let model else { return raw }
        var copy = raw
        for item in Self.inapplicableDescriptions(for: raw, model: model, contract: contract) {
            switch item.key {
            case .temperature: copy.temperature = nil
            case .topP: copy.topP = nil
            case .topK: copy.topK = nil
            case .minP: copy.minP = nil
            case .repetitionPenalty: copy.repetitionPenalty = nil
            case .repetitionContextSize: copy.repetitionContextSize = nil
            case .presencePenalty: copy.presencePenalty = nil
            case .maxTokens: copy.maxTokens = nil
            case .seed: copy.seed = nil
            case .stopSequences: copy.stopSequences = nil
            case .thinking: copy.thinking = nil
            case .thinkingDepth: copy.thinkingDepth = nil
            case .thinkingBudget: copy.thinkingBudget = nil
            }
        }
        return copy
    }

    private static func isEmptySettings(_ s: SamplerSettings) -> Bool {
        s == SamplerSettings()
    }

    private static func inapplicableDescriptions(
        for settings: SamplerSettings,
        model: EngineDescription,
        contract: EngineContract?
    ) -> [DroppedItem] {
        var dropped: [DroppedItem] = []
        let supportsThinking = model.thinkingControls?.supportsThinking ?? model.supportsThinking.value
        let allowedFields: Set<String>? = contract.flatMap { c in
            guard let fields = c.decoding.requestFields.value, !fields.isEmpty else { return nil }
            return Set(fields)
        }

        func engineSupports(_ key: SamplerCascade.SamplerFieldKey) -> Bool {
            guard let allowedFields else { return true }
            return !Set(key.acceptedAliases).isDisjoint(with: allowedFields)
        }

        if let v = settings.temperature, !engineSupports(.temperature) {
            dropped.append(.init(key: .temperature, label: "Temperature", valueText: "\(v)", reason: "not supported by \(model.modelID)"))
        }
        if let v = settings.topP, !engineSupports(.topP) {
            dropped.append(.init(key: .topP, label: "Top P", valueText: "\(v)", reason: "not supported by \(model.modelID)"))
        }
        if let v = settings.topK, !engineSupports(.topK) {
            dropped.append(.init(key: .topK, label: "Top K", valueText: "\(v)", reason: "not supported by \(model.modelID)"))
        }
        if let v = settings.minP, !engineSupports(.minP) {
            dropped.append(.init(key: .minP, label: "Min P", valueText: "\(v)", reason: "not supported by \(model.modelID)"))
        }
        if let v = settings.repetitionPenalty, !engineSupports(.repetitionPenalty) {
            dropped.append(.init(key: .repetitionPenalty, label: "Repetition Penalty", valueText: "\(v)", reason: "not supported by \(model.modelID)"))
        }
        if let v = settings.repetitionContextSize, !engineSupports(.repetitionContextSize) {
            dropped.append(.init(key: .repetitionContextSize, label: "Repeat Window", valueText: "\(v)", reason: "not supported by \(model.modelID)"))
        }
        if let v = settings.presencePenalty, !engineSupports(.presencePenalty) {
            dropped.append(.init(key: .presencePenalty, label: "Presence Penalty", valueText: "\(v)", reason: "not supported by \(model.modelID)"))
        }
        if let v = settings.maxTokens, !engineSupports(.maxTokens) {
            dropped.append(.init(key: .maxTokens, label: "Max Tokens", valueText: "\(v)", reason: "not supported by \(model.modelID)"))
        }
        if let v = settings.seed, !engineSupports(.seed) {
            dropped.append(.init(key: .seed, label: "Seed", valueText: "\(v)", reason: "not supported by \(model.modelID)"))
        }
        if let v = settings.stopSequences, !v.isEmpty, !engineSupports(.stopSequences) {
            dropped.append(.init(key: .stopSequences, label: "Stop Sequences", valueText: v.joined(separator: ", "), reason: "not supported by \(model.modelID)"))
        }

        if let v = settings.thinking {
            let canDisable = model.thinkingControls?.canDisableThinking ?? supportsThinking
            if !supportsThinking {
                dropped.append(.init(key: .thinking, label: "Thinking", valueText: v ? "on" : "off", reason: "this model has no thinking switch"))
            } else if !v && !canDisable {
                dropped.append(.init(key: .thinking, label: "Thinking", valueText: "off", reason: "thinking cannot be disabled on \(model.modelID)"))
            }
        }

        if let v = settings.thinkingDepth, !v.isEmpty {
            if let controls = model.thinkingControls {
                if let err = controls.checkDepth(v) {
                    dropped.append(.init(key: .thinkingDepth, label: "Thinking Depth", valueText: v, reason: err))
                }
            } else if !supportsThinking {
                dropped.append(.init(key: .thinkingDepth, label: "Thinking Depth", valueText: v, reason: "this model's template has no thinking depth"))
            }
        }

        if let v = settings.thinkingBudget {
            if !supportsThinking {
                dropped.append(.init(key: .thinkingBudget, label: "Thinking Budget", valueText: "\(v)", reason: "this model has no thinking budget control"))
            } else if model.thinkingControls?.budgetEnforced == false || contract?.thinking?.budget?.enforced == false {
                dropped.append(.init(key: .thinkingBudget, label: "Thinking Budget", valueText: "\(v)", reason: "this engine cannot enforce a thinking cap"))
            }
        }

        return dropped
    }
}
