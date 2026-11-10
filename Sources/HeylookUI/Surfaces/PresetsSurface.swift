import Observation
import SwiftUI
import HeylookCore

public struct PresetsSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    @State private var draftID: String?
    @State private var draftName: String = ""
    @State private var draftScope: String = "sampler"
    @State private var draftDescription: String = ""
    @State private var draftSystemPrompt: String = ""

    @State private var draftTemperature: Double?
    @State private var draftTopP: Double?
    @State private var draftTopK: Int?
    @State private var draftMinP: Double?
    @State private var draftRepetitionPenalty: Double?
    @State private var draftRepetitionContextSize: Int?
    @State private var draftPresencePenalty: Double?
    @State private var draftMaxTokens: Int?
    @State private var draftSeed: Int?
    @State private var draftThinkingEnabled: Bool?
    @State private var draftThinkingDepth: String?
    @State private var draftThinkingBudget: Int?
    @State private var draftStopSequencesText: String = ""
    @State private var isCreatingNew: Bool = false

    public init(viewModel: HeylookWorkbenchViewModel) {
        self.viewModel = viewModel
    }

    private var selectedPreset: PresetRecord? {
        guard let selID = viewModel.selectedPresetRecordID else { return nil }
        return viewModel.presets.first(where: { $0.id == selID })
    }

    public var body: some View {
        VStack(spacing: 0) {
            PresetsHeaderBar(
                viewModel: viewModel,
                onNewPreset: beginCreateNewPreset
            )
            Divider()

            if horizontalSizeClass == .compact {
                VStack(spacing: 0) {
                    PresetsCompactPickerBar(
                        viewModel: viewModel,
                        isCreatingNew: isCreatingNew,
                        onSelectPreset: { preset in
                            isCreatingNew = false
                            viewModel.selectedPresetRecordID = preset.id
                            loadDraft(from: preset)
                        }
                    )
                    Divider()
                    editorPane
                }
            } else {
                HStack(spacing: 0) {
                    PresetsMasterListPane(
                        viewModel: viewModel,
                        isCreatingNew: isCreatingNew,
                        onSelectPreset: { preset in
                            isCreatingNew = false
                            viewModel.selectedPresetRecordID = preset.id
                            loadDraft(from: preset)
                        },
                        onNewPreset: beginCreateNewPreset
                    )
                    .frame(width: 280)

                    Divider()

                    editorPane
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .onAppear {
            syncInitialSelection()
        }
        .onChange(of: viewModel.selectedPresetRecordID) { _, newID in
            guard !isCreatingNew else { return }
            if let newID, let preset = viewModel.presets.first(where: { $0.id == newID }), draftID != preset.id {
                loadDraft(from: preset)
            }
        }
    }

    @ViewBuilder
    private var editorPane: some View {
        if isCreatingNew || selectedPreset != nil {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    HStack(alignment: .center, spacing: 10) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(isCreatingNew ? "New Preset" : (draftName.isEmpty ? "Edit Preset" : draftName))
                                .font(.title3.weight(.semibold))
                            Text(
                                isCreatingNew
                                    ? "Create a reusable sampler or full (sampler + system prompt) preset."
                                    : "Preset ID: \(draftID ?? "—")"
                            )
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                        }
                        Spacer()

                        if !isCreatingNew, let active = selectedPreset {
                            Button {
                                viewModel.applyPreset(active)
                            } label: {
                                Label(
                                    viewModel.selectedPresetName == active.name ? "Applied" : "Apply to Chat / Notebook",
                                    systemImage: viewModel.selectedPresetName == active.name ? "checkmark.circle.fill" : "play.circle"
                                )
                            }
                            .buttonStyle(.glass)
                            .controlSize(.small)

                            Button {
                                if let copy = viewModel.duplicatePresetRecord(active) {
                                    isCreatingNew = false
                                    loadDraft(from: copy)
                                }
                            } label: {
                                Label("Duplicate", systemImage: "plus.square.on.square")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)

                            Button(role: .destructive) {
                                viewModel.pendingConfirmation = .deletePreset(id: active.id, name: active.name)
                            } label: {
                                Label("Delete…", systemImage: "trash")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }

                        Button {
                            saveDraftPreset()
                        } label: {
                            Label(isCreatingNew ? "Create Preset" : "Save Changes", systemImage: "checkmark")
                        }
                        .buttonStyle(.glassProminent)
                        .controlSize(.small)
                        .disabled(draftName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }

                    GroupBox("Identity & Scope") {
                        VStack(alignment: .leading, spacing: 10) {
                            LabeledContent("Preset Name") {
                                TextField("e.g. Precise Code Review", text: $draftName)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(maxWidth: 340)
                            }

                            LabeledContent("Preset Scope") {
                                Picker("Scope", selection: $draftScope) {
                                    Text("Sampler Only").tag("sampler")
                                    Text("Full (Sampler + System Prompt)").tag("full")
                                }
                                .pickerStyle(.segmented)
                                .labelsHidden()
                                .frame(maxWidth: 340)
                            }

                            LabeledContent("Description") {
                                TextField("Optional notes shown in the Presets list…", text: $draftDescription)
                                    .textFieldStyle(.roundedBorder)
                            }
                        }
                        .padding(6)
                    }

                    if draftScope == "full" {
                        GroupBox("System Prompt (Full Scope)") {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Applied automatically when this preset is selected in Chat or Notebook.")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)

                                TextEditor(text: $draftSystemPrompt)
                                    .font(.system(.callout, design: .monospaced))
                                    .frame(minHeight: 96, maxHeight: 180)
                                    .padding(4)
                                    .background(
                                        RoundedRectangle(cornerRadius: 8)
                                            .fill(Color.secondary.opacity(0.08))
                                    )
                            }
                            .padding(6)
                        }
                    }

                    GroupBox {
                        VStack(alignment: .leading, spacing: 10) {
                            PresetOptionalDoubleSliderRow(
                                label: "Temperature",
                                value: $draftTemperature,
                                fallback: 0.7,
                                range: 0.0...2.0,
                                step: 0.05,
                                format: "%.2f"
                            )
                            PresetOptionalDoubleSliderRow(
                                label: "Top-P",
                                value: $draftTopP,
                                fallback: 0.95,
                                range: 0.05...1.0,
                                step: 0.05,
                                format: "%.2f"
                            )
                            PresetOptionalDoubleSliderRow(
                                label: "Min-P",
                                value: $draftMinP,
                                fallback: 0.05,
                                range: 0.0...0.5,
                                step: 0.01,
                                format: "%.2f"
                            )
                            PresetOptionalDoubleSliderRow(
                                label: "Repetition Penalty",
                                value: $draftRepetitionPenalty,
                                fallback: 1.05,
                                range: 1.0...1.5,
                                step: 0.01,
                                format: "%.2f"
                            )
                            PresetOptionalDoubleSliderRow(
                                label: "Presence Penalty",
                                value: $draftPresencePenalty,
                                fallback: 0.0,
                                range: -2.0...2.0,
                                step: 0.1,
                                format: "%.2f"
                            )

                            Divider()

                            PresetOptionalIntFieldRow(label: "Top-K", value: $draftTopK, placeholder: "Inherit (e.g. 40)")
                            PresetOptionalIntFieldRow(label: "Repetition Window", value: $draftRepetitionContextSize, placeholder: "Inherit (e.g. 64)")
                            PresetOptionalIntFieldRow(label: "Max Output Tokens", value: $draftMaxTokens, placeholder: "Inherit (e.g. 2048)")
                            PresetOptionalIntFieldRow(label: "Deterministic Seed", value: $draftSeed, placeholder: "Random")

                            Divider()

                            HStack {
                                Text("Thinking Override")
                                    .font(.caption.weight(.medium))
                                Spacer()
                                Picker(
                                    "Thinking",
                                    selection: Binding(
                                        get: {
                                            guard let val = draftThinkingEnabled else { return "inherit" }
                                            return val ? "on" : "off"
                                        },
                                        set: { choice in
                                            switch choice {
                                            case "on": draftThinkingEnabled = true
                                            case "off": draftThinkingEnabled = false
                                            default: draftThinkingEnabled = nil
                                            }
                                        }
                                    )
                                ) {
                                    Text("Inherit").tag("inherit")
                                    Text("Enable").tag("on")
                                    Text("Disable").tag("off")
                                }
                                .pickerStyle(.segmented)
                                .labelsHidden()
                                .frame(maxWidth: 220)
                            }

                            HStack {
                                Text("Thinking Depth")
                                    .font(.caption.weight(.medium))
                                Spacer()
                                TextField(
                                    "Inherit (e.g. minimal / low / medium / high)",
                                    text: Binding(
                                        get: { draftThinkingDepth ?? "" },
                                        set: { raw in
                                            let t = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                                            draftThinkingDepth = t.isEmpty ? nil : t
                                        }
                                    )
                                )
                                .textFieldStyle(.roundedBorder)
                                .font(.caption.monospaced())
                                .frame(maxWidth: 260)
                            }

                            PresetOptionalIntFieldRow(
                                label: "Thinking Budget (tokens)",
                                value: $draftThinkingBudget,
                                placeholder: "Unlimited"
                            )

                            HStack {
                                Text("Stop Sequences (comma-sep)")
                                    .font(.caption.weight(.medium))
                                Spacer()
                                TextField("Inherit (none)", text: $draftStopSequencesText)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.caption.monospaced())
                                    .frame(maxWidth: 260)
                            }
                        }
                        .padding(6)
                    } label: {
                        HStack {
                            Text("Sampler Cascade Overrides (Unset = Inherit Model / Vendor Default)")
                            Spacer()
                            Button("Clear All Overrides") {
                                clearDraftSamplerFields()
                            }
                            .buttonStyle(.borderless)
                            .font(.caption2)
                        }
                    }
                }
                .padding(18)
            }
        } else {
            ContentUnavailableView(
                "No Preset Selected",
                systemImage: "slider.horizontal.3",
                description: Text("Select a preset from the list or create a new sampler / full preset.")
            )
        }
    }

    private func syncInitialSelection() {
        if viewModel.selectedPresetRecordID == nil, let first = viewModel.presets.first {
            viewModel.selectedPresetRecordID = first.id
        }
        if let active = selectedPreset {
            loadDraft(from: active)
        } else if viewModel.presets.isEmpty {
            beginCreateNewPreset()
        }
    }

    private func beginCreateNewPreset() {
        isCreatingNew = true
        draftID = nil
        draftName = "Custom Preset"
        draftScope = "sampler"
        draftDescription = ""
        draftSystemPrompt = viewModel.draftSystemPrompt
        let activeOverlay = viewModel.unmaskedManualSamplerOverlay
        draftTemperature = activeOverlay.temperature
        draftTopP = activeOverlay.topP
        draftTopK = activeOverlay.topK
        draftMinP = activeOverlay.minP
        draftRepetitionPenalty = activeOverlay.repetitionPenalty
        draftRepetitionContextSize = activeOverlay.repetitionContextSize
        draftPresencePenalty = activeOverlay.presencePenalty
        draftMaxTokens = activeOverlay.maxTokens
        draftSeed = activeOverlay.seed
        draftThinkingEnabled = activeOverlay.thinking
        draftThinkingDepth = activeOverlay.thinkingDepth
        draftThinkingBudget = activeOverlay.thinkingBudget
        draftStopSequencesText = (activeOverlay.stopSequences ?? []).joined(separator: ", ")
    }

    private func loadDraft(from preset: PresetRecord) {
        draftID = preset.id
        draftName = preset.name
        draftScope = preset.scope
        draftDescription = preset.description
        draftSystemPrompt = preset.systemPrompt ?? ""
        let s = preset.settings
        draftTemperature = s.temperature
        draftTopP = s.topP
        draftTopK = s.topK
        draftMinP = s.minP
        draftRepetitionPenalty = s.repetitionPenalty
        draftRepetitionContextSize = s.repetitionContextSize
        draftPresencePenalty = s.presencePenalty
        draftMaxTokens = s.maxTokens
        draftSeed = s.seed
        draftThinkingEnabled = s.thinking
        draftThinkingDepth = s.thinkingDepth
        draftThinkingBudget = s.thinkingBudget
        draftStopSequencesText = (s.stopSequences ?? []).joined(separator: ", ")
    }

    private func clearDraftSamplerFields() {
        draftTemperature = nil
        draftTopP = nil
        draftTopK = nil
        draftMinP = nil
        draftRepetitionPenalty = nil
        draftRepetitionContextSize = nil
        draftPresencePenalty = nil
        draftMaxTokens = nil
        draftSeed = nil
        draftThinkingEnabled = nil
        draftThinkingDepth = nil
        draftThinkingBudget = nil
        draftStopSequencesText = ""
    }

    private func buildDraftSettings() -> SamplerSettings {
        let stops = draftStopSequencesText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return SamplerSettings(
            temperature: draftTemperature,
            topP: draftTopP,
            topK: draftTopK,
            minP: draftMinP,
            repetitionPenalty: draftRepetitionPenalty,
            repetitionContextSize: draftRepetitionContextSize,
            presencePenalty: draftPresencePenalty,
            maxTokens: draftMaxTokens,
            seed: draftSeed,
            thinking: draftThinkingEnabled,
            thinkingDepth: draftThinkingDepth,
            thinkingBudget: draftThinkingBudget,
            stopSequences: stops.isEmpty ? nil : stops
        )
    }

    private func saveDraftPreset() {
        let settings = buildDraftSettings()
        let trimmedSys = draftSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if isCreatingNew {
            if let created = viewModel.createNewPresetRecord(
                name: draftName,
                scope: draftScope,
                description: draftDescription,
                systemPrompt: draftScope == "full" ? (trimmedSys.isEmpty ? nil : draftSystemPrompt) : nil,
                settings: settings
            ) {
                isCreatingNew = false
                loadDraft(from: created)
            }
        } else if let id = draftID {
            let record = PresetRecord(
                id: id,
                name: draftName.trimmingCharacters(in: .whitespacesAndNewlines),
                scope: draftScope,
                settings: settings,
                systemPrompt: draftScope == "full" ? (trimmedSys.isEmpty ? nil : draftSystemPrompt) : nil,
                description: draftDescription.trimmingCharacters(in: .whitespacesAndNewlines)
            )
            if let saved = viewModel.updatePresetRecord(record) {
                loadDraft(from: saved)
            }
        }
    }
}

// MARK: - Subviews

private struct PresetsHeaderBar: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let onNewPreset: () -> Void

    var body: some View {
        GlassEffectContainer(spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: "slider.horizontal.3")
                    .foregroundStyle(.teal)
                Text("Saved Presets")
                    .font(.headline)

                Text("\(viewModel.presets.count)")
                    .font(.caption2.monospacedDigit())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.teal.opacity(0.16)))
                    .foregroundStyle(.teal)

                if let activeName = viewModel.selectedPresetName {
                    Text("Active: \(activeName)\(viewModel.isPresetModified ? " • modified" : "")")
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Capsule().fill(Color.accentColor.opacity(0.14)))
                        .foregroundStyle(Color.accentColor)
                }

                Spacer()

                Button(action: onNewPreset) {
                    Label("New Preset", systemImage: "plus")
                }
                .buttonStyle(.glassProminent)
                .controlSize(.small)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }
}

private struct PresetsCompactPickerBar: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let isCreatingNew: Bool
    let onSelectPreset: (PresetRecord) -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(viewModel.filteredPresets) { preset in
                    let selected = !isCreatingNew && viewModel.selectedPresetRecordID == preset.id
                    Button {
                        onSelectPreset(preset)
                    } label: {
                        Text(preset.name)
                            .font(.caption.weight(selected ? .semibold : .regular))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(
                                Capsule().fill(selected ? Color.accentColor.opacity(0.2) : Color.secondary.opacity(0.12))
                            )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
        }
    }
}

private struct PresetsMasterListPane: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let isCreatingNew: Bool
    let onSelectPreset: (PresetRecord) -> Void
    let onNewPreset: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            VStack(spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextField("Search presets…", text: $viewModel.presetSearchQuery)
                        .textFieldStyle(.plain)
                        .font(.caption)
                    if !viewModel.presetSearchQuery.isEmpty {
                        Button {
                            viewModel.presetSearchQuery = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .frame(minWidth: 28, minHeight: 28)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 7))

                Picker(
                    "Scope",
                    selection: Binding(
                        get: { viewModel.presetScopeFilter ?? "all" },
                        set: { val in
                            viewModel.presetScopeFilter = val == "all" ? nil : val
                        }
                    )
                ) {
                    Text("All").tag("all")
                    Text("Sampler").tag("sampler")
                    Text("Full").tag("full")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
            }
            .padding(.horizontal, 10)
            .padding(.top, 10)

            List {
                ForEach(viewModel.filteredPresets) { preset in
                    let isSelected = !isCreatingNew && viewModel.selectedPresetRecordID == preset.id
                    let isApplied = viewModel.selectedPresetName == preset.name
                    Button {
                        onSelectPreset(preset)
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 6) {
                                Text(preset.name)
                                    .font(.subheadline.weight(isSelected ? .semibold : .medium))
                                    .lineLimit(1)
                                Spacer(minLength: 0)
                                if isApplied {
                                    Image(systemName: "checkmark.circle.fill")
                                        .font(.caption)
                                        .foregroundStyle(Color.accentColor)
                                }
                                Text(preset.scope.uppercased())
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 1)
                                    .background(
                                        Capsule().fill(
                                            preset.scope == "full"
                                                ? Color.purple.opacity(0.16)
                                                : Color.teal.opacity(0.16)
                                        )
                                    )
                                    .foregroundStyle(preset.scope == "full" ? .purple : .teal)
                            }
                            if !preset.description.isEmpty {
                                Text(preset.description)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                        }
                        .padding(.vertical, 3)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .listRowBackground(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(isSelected ? Color.accentColor.opacity(0.16) : Color.clear)
                    )
                    .contextMenu {
                        Button {
                            viewModel.applyPreset(preset)
                        } label: {
                            Label("Apply Preset", systemImage: "play.circle")
                                .labelStyle(.titleAndIcon)
                        }

                        Button {
                            if let copy = viewModel.duplicatePresetRecord(preset) {
                                onSelectPreset(copy)
                            }
                        } label: {
                            Label("Duplicate Preset", systemImage: "plus.square.on.square")
                                .labelStyle(.titleAndIcon)
                        }

                        Divider()

                        Button(role: .destructive) {
                            viewModel.pendingConfirmation = .deletePreset(id: preset.id, name: preset.name)
                        } label: {
                            Label("Delete Preset…", systemImage: "trash")
                                .labelStyle(.titleAndIcon)
                        }
                    }
                }
            }
            .listStyle(.plain)
        }
    }
}

private struct PresetOptionalDoubleSliderRow: View {
    let label: String
    @Binding var value: Double?
    let fallback: Double
    let range: ClosedRange<Double>
    let step: Double
    let format: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(label)
                    .font(.caption.weight(.medium))
                Spacer()
                if let val = value {
                    Text(String(format: format, val))
                        .font(.caption.monospaced().weight(.semibold))
                    Button {
                        value = nil
                    } label: {
                        Image(systemName: "arrow.uturn.backward.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(minWidth: 28, minHeight: 28)
                    }
                    .buttonStyle(.plain)
                    .help("Reset to Inherit")
                } else {
                    Text("Inherit")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                    Button("Set") {
                        value = fallback
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
            }

            if value != nil {
                Slider(
                    value: Binding(
                        get: { value ?? fallback },
                        set: { value = $0 }
                    ),
                    in: range,
                    step: step
                )
            }
        }
    }
}

private struct PresetOptionalIntFieldRow: View {
    let label: String
    @Binding var value: Int?
    let placeholder: String

    var body: some View {
        HStack {
            Text(label)
                .font(.caption.weight(.medium))
            Spacer()
            TextField(
                placeholder,
                text: Binding(
                    get: { value.map(String.init) ?? "" },
                    set: { raw in
                        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                        value = trimmed.isEmpty ? nil : Int(trimmed)
                    }
                )
            )
            .textFieldStyle(.roundedBorder)
            .font(.caption.monospaced())
            .frame(width: 170)

            if value != nil {
                Button {
                    value = nil
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(minWidth: 28, minHeight: 28)
                }
                .buttonStyle(.plain)
                .help("Clear override")
            }
        }
    }
}
