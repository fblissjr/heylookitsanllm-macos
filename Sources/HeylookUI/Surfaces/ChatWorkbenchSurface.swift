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
