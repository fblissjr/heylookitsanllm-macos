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

// MARK: - 1. Chat Workbench Surface

struct ChatWorkbenchSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @FocusState private var isPromptFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Top Control Bar (no truncated preset buttons)
            HStack(spacing: 10) {
                Menu {
                    Section("Sampler Presets") {
                        ForEach(viewModel.presets) { preset in
                            Button {
                                viewModel.applyPreset(preset)
                            } label: {
                                Label(
                                    preset.name,
                                    systemImage: viewModel.selectedPresetName == preset.name ? "checkmark" : "circle"
                                )
                                .labelStyle(.titleAndIcon)
                            }
                        }
                    }
                    Divider()
                    Button("Customize Samplers…") {
                        viewModel.showInspector = true
                    }
                    Button("Reset Overrides") {
                        viewModel.resetSamplerOverrides()
                    }
                    .disabled(viewModel.userOverriddenSamplerKeys.isEmpty)
                } label: {
                    Label(viewModel.selectedPresetName, systemImage: "dial.medium")
                }
                .controlSize(.small)
                .fixedSize()

                Divider().frame(height: 16)

                Toggle(isOn: $viewModel.thinkingEnabled) {
                    Label("Thinking", systemImage: "brain")
                }
                .toggleStyle(.button)
                .controlSize(.small)
                .disabled(viewModel.activeModelDescription?.supportsThinking.value == false)
                .onChange(of: viewModel.thinkingEnabled) { _, _ in
                    viewModel.userOverriddenSamplerKeys.insert("thinking")
                }

                if let controls = viewModel.activeModelDescription?.thinkingControls,
                   !controls.depthValues.isEmpty {
                    Picker("Depth", selection: $viewModel.thinkingDepth) {
                        ForEach(controls.depthValues, id: \.self) { level in
                            Text(level).tag(level)
                        }
                    }
                    .pickerStyle(.segmented)
                    .controlSize(.small)
                    .frame(maxWidth: 200)
                    .onChange(of: viewModel.thinkingDepth) { _, _ in
                        viewModel.userOverriddenSamplerKeys.insert("thinking_depth")
                    }
                }

                Spacer()

                Toggle(isOn: $viewModel.showPromptPreview) {
                    Label("Prompt Preview", systemImage: "curlybraces")
                }
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
                    HStack {
                        Text("Live Rendered Template Preview")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer()
                        if let tmplSource = viewModel.activeModelDescription?.templateSource.value {
                            Text(tmplSource)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }
                    ScrollView {
                        Text(viewModel.renderedPromptPreview())
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(maxHeight: 140)
                }
                .padding(12)
                .background(Color.secondary.opacity(0.08))
                Divider()
            }

            // Message Transcript or Empty State
            if viewModel.currentMessages.isEmpty && !viewModel.isGenerating {
                ChatEmptyStateView(
                    viewModel: viewModel,
                    onSelectStarterPrompt: { prompt in
                        viewModel.draftPrompt = prompt
                        isPromptFocused = true
                    }
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        ForEach(viewModel.currentMessages) { msg in
                            MessageBubbleView(
                                message: msg,
                                isEditing: viewModel.editingMessageID == msg.id,
                                editingContent: $viewModel.editingContent,
                                onBeginEdit: {
                                    viewModel.editingMessageID = msg.id
                                    viewModel.editingContent = msg.content
                                },
                                onSaveEdit: {
                                    Task { await viewModel.saveAndContinueEditedMessage(msg) }
                                },
                                onCancelEdit: {
                                    viewModel.editingMessageID = nil
                                }
                            )
                        }

                        if viewModel.isGenerating {
                            StreamingAssistantView(viewModel: viewModel)
                        }
                    }
                    .padding(16)
                }
                .swipeActionsContainer()
                .defaultScrollAnchor(.bottom, for: .sizeChanges)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            Divider()

            // Prompt Composer Bar
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

                GlassEffectContainer(spacing: 10) {
                    HStack(alignment: .bottom, spacing: 10) {
                        #if os(macOS)
                        Button {
                            pickChatAttachments()
                        } label: {
                            Image(systemName: "paperclip")
                                .font(.system(size: 14, weight: .semibold))
                        }
                        .buttonStyle(.glass)
                        .help("Attach image or audio files")
                        #endif

                        Button {
                            Task { await viewModel.toggleVoiceRecording() }
                        } label: {
                            Image(systemName: viewModel.isRecordingVoice ? "stop.circle.fill" : "mic.fill")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(viewModel.isRecordingVoice ? .red : .primary)
                        }
                        .buttonStyle(.glass)
                        .disabled(viewModel.isGenerating)
                        .help(viewModel.isRecordingVoice ? "Stop recording & transcribe on ANE" : "Record voice turn (16 kHz ANE transcription)")

                        TextField("Ask anything or drop images/audio (⌘↩ to send)…", text: $viewModel.draftPrompt, axis: .vertical)
                            .lineLimit(1...6)
                            .textFieldStyle(.roundedBorder)
                            .focused($isPromptFocused)
                            .defaultFocus($isPromptFocused, true)
                            .onSubmit {
                                Task { await viewModel.sendChatPrompt() }
                            }

                        if viewModel.isGenerating {
                            Button("Stop", role: .destructive) {
                                Task { await viewModel.cancelGeneration() }
                            }
                            .buttonStyle(.glassProminent)
                            .keyboardShortcut(.escape, modifiers: [])
                            .help("Cancel generation (Esc)")
                        } else {
                            Button("Send") {
                                Task { await viewModel.sendChatPrompt() }
                            }
                            .buttonStyle(.glassProminent)
                            .keyboardShortcut(.return, modifiers: [.command])
                            .disabled(
                                viewModel.draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                    && viewModel.draftAttachments.isEmpty
                            )
                            .help("Send message (⌘↩)")
                        }
                    }
                }
            }
            .padding(12)
            .glassEffect(.regular, in: .rect(cornerRadius: 16))
            .padding(.horizontal, 12)
            .padding(.bottom, 10)
            .padding(.top, 6)
            .onDrop(of: [.fileURL], isTargeted: nil) { providers in
                handleDroppedChatFiles(providers)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            isPromptFocused = true
        }
        .onChange(of: viewModel.selectedConversationID) { _, _ in
            isPromptFocused = true
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

private struct ChatEmptyStateView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let onSelectStarterPrompt: (String) -> Void

    var body: some View {
        ContentUnavailableView {
            Label(
                viewModel.activeModelDescription?.modelID ?? "Heylook Chat Workbench",
                systemImage: "bubble.left.and.bubble.right.fill"
            )
        } description: {
            if let model = viewModel.activeModelDescription {
                VStack(spacing: 8) {
                    Text("Provider: \(model.provider.displayName) • Compute: \(model.computePlacement.value.label) • Context: \(model.contextSize.value) tokens")
                        .font(.caption.monospaced())
                    HStack(spacing: 6) {
                        if model.supportsThinking.value {
                            Text("Thinking (\(model.reasoningFormat.value))")
                                .font(.caption2.monospaced())
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(Color.purple.opacity(0.14), in: Capsule())
                        }
                        if model.supportsVision.value {
                            Text("Vision Enabled")
                                .font(.caption2.monospaced())
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(Color.blue.opacity(0.14), in: Capsule())
                        }
                        if model.supportsAudio.value {
                            Text("Audio Enabled")
                                .font(.caption2.monospaced())
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(Color.green.opacity(0.14), in: Capsule())
                        }
                    }
                }
            } else {
                Text("Select a configured model in the toolbar or load the sample demo workspace to start chatting.")
            }
        } actions: {
            GlassEffectContainer(spacing: 10) {
                HStack(spacing: 8) {
                    Button("Explain GPU + ANE Co-Residency") {
                        onSelectStarterPrompt("Explain how ComputeScheduler co-residents MLX GPU generation with Apple Neural Engine embedding towers.")
                    }
                    .buttonStyle(.glass)
                    .controlSize(.small)

                    Button("Write a Swift 6 Actor") {
                        onSelectStarterPrompt("Write a thread-safe Swift 6 actor that manages a bounded FIFO request queue.")
                    }
                    .buttonStyle(.glass)
                    .controlSize(.small)

                    if viewModel.registeredModels.isEmpty {
                        Button("Load Sample Demo Workspace") {
                            Task { await viewModel.loadSampleDemoWorkspace() }
                        }
                        .buttonStyle(.glassProminent)
                        .controlSize(.small)
                    }
                }
            }
        }
    }
}

struct MessageBubbleView: View {
    let message: MessageRecord
    let isEditing: Bool
    @Binding var editingContent: String
    let onBeginEdit: () -> Void
    let onSaveEdit: () -> Void
    let onCancelEdit: () -> Void

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
                    onBeginEdit()
                } label: {
                    Image(systemName: "pencil")
                        .font(.caption)
                        .padding(4)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .help("Edit message and regenerate from here")
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

            if isEditing {
                VStack(alignment: .leading, spacing: 8) {
                    TextEditor(text: $editingContent)
                        .font(.body)
                        .frame(minHeight: 80)
                    HStack {
                        Button("Save & Continue") {
                            onSaveEdit()
                        }
                        .buttonStyle(.glassProminent)
                        .controlSize(.small)

                        Button("Cancel") {
                            onCancelEdit()
                        }
                        .buttonStyle(.glass)
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
        .swipeActions(edge: .trailing) {
            Button {
                onBeginEdit()
            } label: {
                Label("Edit", systemImage: "pencil")
            }
            .tint(.blue)
        }
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
                .buttonStyle(.glass)
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

// MARK: - Sampler Cascade & System Prompt Inspector / Sheet

struct LoadAndSamplerInspectorView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        let resolved = viewModel.currentResolvedSamplers
        TabView {
            Tab("Samplers", systemImage: "slider.horizontal.3") {
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Sampler Cascade")
                                    .font(.headline)
                                Text("builtin → model → preset → request")
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Reset") {
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
                            Slider(value: $viewModel.boundTemperature, in: 0...2)

                            HStack {
                                Text("Top-P: \(String(format: "%.2f", viewModel.topPOverride))")
                                    .font(.subheadline.monospacedDigit())
                                Spacer()
                                SamplerProvenancePill(
                                    source: resolved.provenanceByField[.topP] ?? .derived,
                                    isRequestOverride: viewModel.userOverriddenSamplerKeys.contains("top_p")
                                )
                            }
                            Slider(value: $viewModel.boundTopP, in: 0.05...1.0)

                            HStack {
                                Stepper(
                                    "Top-K: \(viewModel.topKOverride)",
                                    value: $viewModel.boundTopK,
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
                            Slider(value: $viewModel.boundMinP, in: 0.0...0.5)

                            HStack {
                                Stepper(
                                    "Max Tokens: \(viewModel.maxTokensOverride)",
                                    value: $viewModel.boundMaxTokens,
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
                                    "Thinking Budget: \(viewModel.thinkingBudgetOverride)",
                                    value: $viewModel.boundThinkingBudget,
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
                    }
                    .padding(16)
                }
            }

            Tab("Prompt", systemImage: "text.quote") {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("System Prompt Override")
                            .font(.headline)
                        Text("Applied to the active conversation turn before chat template rendering.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        TextField("Optional system prompt override…", text: $viewModel.draftSystemPrompt, axis: .vertical)
                            .lineLimit(5...12)
                            .textFieldStyle(.roundedBorder)
                    }
                    .padding(16)
                }
            }
        }
    }
}

struct LoadAndSamplerPanelSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            LoadAndSamplerInspectorView(viewModel: viewModel)
            Divider()
            HStack {
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
            .padding(16)
        }
        .frame(minWidth: 480, minHeight: 520)
    }
}

struct SamplerProvenancePill: View {
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
