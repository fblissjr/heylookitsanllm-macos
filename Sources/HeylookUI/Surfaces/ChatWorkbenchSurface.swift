import CoreGraphics
import Foundation
import ImageIO
import SwiftUI
import UniformTypeIdentifiers
import HeylookCore
import HeylookEngines

#if os(macOS)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

// MARK: - Surface 1: Multimodal Chat & Reasoning Workbench

public struct ChatWorkbenchSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var showSystemPromptSheet: Bool = false
    @State private var followingTail: Bool = true
    @State private var showAttachmentImporter: Bool = false
    @FocusState private var isComposerFocused: Bool

    public init(viewModel: HeylookWorkbenchViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        VStack(spacing: 0) {
            ChatEngineReadoutBar(
                viewModel: viewModel,
                showSystemPromptSheet: $showSystemPromptSheet
            )

            if let switchWarning = viewModel.modelSwitchWarningMessage {
                ModelSwitchLossBanner(
                    message: switchWarning,
                    onConfirm: { viewModel.confirmPendingModelSwitch() },
                    onCancel: { viewModel.cancelPendingModelSwitch() }
                )
            }

            Divider()

            if viewModel.showPromptPreview {
                ChatPromptPreviewDrawer(viewModel: viewModel)
                Divider()
            }

            ZStack(alignment: .bottom) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 16) {
                            if viewModel.messages.isEmpty && !viewModel.isGenerating {
                                ChatEmptyWelcomeCard(viewModel: viewModel)
                                    .padding(.top, 36)
                            }

                            let lastAssistantID = viewModel.messages.last(where: { $0.role == "assistant" })?.id
                            ForEach(viewModel.messages) { msg in
                                MessageBubbleView(
                                    viewModel: viewModel,
                                    message: msg,
                                    isLatestAssistant: msg.id == lastAssistantID
                                )
                                .id(msg.id)
                            }

                            if viewModel.isGenerating
                                && (viewModel.generatingConversationID == nil
                                    || viewModel.generatingConversationID == viewModel.selectedConversationID) {
                                StreamingAssistantView(viewModel: viewModel)
                                    .id("streaming-assistant")
                            }

                            Color.clear
                                .frame(height: 1)
                                .id("chat-bottom")
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 16)
                    }
                    .onScrollGeometryChange(for: Bool.self) { geo in
                        let distanceFromBottom = geo.contentSize.height - geo.contentOffset.y - geo.containerSize.height
                        return distanceFromBottom < 100
                    } action: { _, isNearBottom in
                        followingTail = isNearBottom
                    }
                    .onChange(of: viewModel.messages.count) { _, _ in
                        followingTail = true
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo("chat-bottom", anchor: .bottom)
                        }
                    }
                    .onChange(of: viewModel.selectedConversationID) { _, _ in
                        followingTail = true
                        proxy.scrollTo("chat-bottom", anchor: .bottom)
                        isComposerFocused = true
                    }
                    .onChange(of: viewModel.streamingContent) { _, _ in
                        guard followingTail else { return }
                        proxy.scrollTo("chat-bottom", anchor: .bottom)
                    }
                    .onChange(of: viewModel.streamingThinking) { _, _ in
                        guard followingTail else { return }
                        proxy.scrollTo("chat-bottom", anchor: .bottom)
                    }
                    .overlay(alignment: .bottom) {
                        if !followingTail && viewModel.isGenerating {
                            Button {
                                followingTail = true
                                withAnimation(.easeOut(duration: 0.2)) {
                                    proxy.scrollTo("chat-bottom", anchor: .bottom)
                                }
                            } label: {
                                HStack(spacing: 6) {
                                    Image(systemName: "arrow.down.circle.fill")
                                    Text("Jump to latest")
                                        .font(.caption.weight(.semibold))
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(.regularMaterial, in: Capsule())
                                .overlay {
                                    Capsule().strokeBorder(Color.accentColor.opacity(0.35), lineWidth: 1)
                                }
                                .shadow(color: .black.opacity(0.12), radius: 6, y: 2)
                            }
                            .buttonStyle(.plain)
                            .padding(.bottom, 12)
                        }
                    }
                }
            }

            Divider()

            ChatComposerBar(
                viewModel: viewModel,
                showAttachmentImporter: $showAttachmentImporter,
                isComposerFocused: $isComposerFocused
            )
        }
        .fileImporter(
            isPresented: $showAttachmentImporter,
            allowedContentTypes: [.image, .audio],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                for url in urls {
                    viewModel.attachChatMediaFile(url: url)
                }
            }
        }
        .sheet(isPresented: $showSystemPromptSheet) {
            SystemPromptEditorSheet(viewModel: viewModel)
        }
        .onAppear {
            isComposerFocused = true
        }
    }
}

private struct ModelSwitchLossBanner: View {
    let message: String
    let onConfirm: () -> Void
    let onCancel: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text(message)
                .font(.caption)
                .foregroundStyle(.primary)
            Spacer()
            Button("Cancel", action: onCancel)
                .buttonStyle(.bordered)
                .controlSize(.small)
            Button("Switch anyway", action: onConfirm)
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .controlSize(.small)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.orange.opacity(0.12))
    }
}

private struct ChatEngineReadoutBar: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Binding var showSystemPromptSheet: Bool

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                if let active = viewModel.activeModelDescription {
                    HStack(spacing: 5) {
                        Circle()
                            .fill(active.isLoaded ? Color.green : Color.secondary.opacity(0.5))
                            .frame(width: 7, height: 7)
                        Text(active.provider.displayName)
                            .font(.caption.weight(.semibold))
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(Color.accentColor.opacity(0.14)))

                    if !active.isLoaded {
                        let isLoading = viewModel.loadingModelIDs.contains(active.modelID)
                        Button {
                            Task { await viewModel.toggleModelLoaded(active) }
                        } label: {
                            HStack(spacing: 4) {
                                if isLoading {
                                    ProgressView()
                                        .controlSize(.mini)
                                } else {
                                    Image(systemName: "bolt.circle")
                                        .font(.caption2)
                                }
                                Text(isLoading ? "Loading…" : "Load")
                                    .font(.caption2.weight(.semibold))
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Capsule().fill(Color.green.opacity(0.15)))
                            .foregroundStyle(.green)
                        }
                        .buttonStyle(.plain)
                        .disabled(isLoading)
                    }

                    ProvenancePill(
                        label: active.computePlacement.value.label,
                        source: active.computePlacement.provenance
                    )
                    ProvenancePill(
                        label: "ctx \(active.contextSize.value / 1024)k",
                        source: active.contextSize.provenance
                    )

                    if let drafter = active.speculativeDrafter.value {
                        ProvenancePill(
                            label: "MTP: \(drafter)",
                            source: active.speculativeDrafter.provenance
                        )
                    }
                } else {
                    Text("No active model selected")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                // Active Preset / Defaults Pill
                Menu {
                    Button {
                        viewModel.clearPresetSelection()
                    } label: {
                        HStack {
                            Text("Model Defaults (No Preset)")
                            if viewModel.selectedPresetName == nil {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                    Divider()
                    ForEach(viewModel.presets) { preset in
                        Button {
                            viewModel.applyPreset(preset)
                        } label: {
                            HStack {
                                Text(preset.name)
                                if viewModel.selectedPresetName == preset.name {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                    Divider()
                    Button {
                        viewModel.activeSurface = .presets
                    } label: {
                        Label("Manage Presets…", systemImage: "slider.horizontal.3")
                            .labelStyle(.titleAndIcon)
                    }
                } label: {
                    let presetLabel: String = {
                        if let name = viewModel.selectedPresetName {
                            return viewModel.isPresetModified ? "\(name) (modified)" : name
                        }
                        return viewModel.hasAnyManualSamplerOverride ? "Custom Samplers" : "Model Defaults"
                    }()
                    HStack(spacing: 4) {
                        Image(systemName: "dial.medium")
                            .font(.system(size: 10))
                        Text(presetLabel)
                            .font(.system(size: 11, weight: .medium))
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(
                        Capsule().fill(
                            viewModel.selectedPresetName != nil
                                ? Color.indigo.opacity(0.15)
                                : Color.secondary.opacity(0.12)
                        )
                    )
                }
                .menuStyle(.borderlessButton)
                .fixedSize()

                if viewModel.responseFormatMode != .text {
                    HStack(spacing: 4) {
                        Image(systemName: "curlybraces")
                            .font(.system(size: 10))
                        Text(viewModel.responseFormatMode.label)
                            .font(.system(size: 11, weight: .semibold))
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(Color.teal.opacity(0.15)))
                    .foregroundStyle(.teal)
                }

                Spacer(minLength: 12)

                // Dynamic Thinking Control derived from ModelTemplateInfo
                if viewModel.activeModelSupportsThinking {
                    let depths = viewModel.activeModelDepthOptions
                    if !depths.isEmpty {
                        Picker(
                            "Depth",
                            selection: Binding(
                                get: { viewModel.effectiveThinkingDepth },
                                set: { viewModel.thinkingDepthOverride = $0 }
                            )
                        ) {
                            ForEach(depths, id: \.self) { depth in
                                Text(depth.capitalized).tag(depth)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 210)
                    } else if viewModel.activeModelCanDisableThinking {
                        Toggle(
                            isOn: Binding(
                                get: { viewModel.effectiveThinkingEnabled },
                                set: { viewModel.thinkingEnabledOverride = $0 }
                            )
                        ) {
                            Label("Thinking", systemImage: "brain")
                                .font(.caption)
                        }
                        .toggleStyle(.button)
                        .controlSize(.small)
                    } else {
                        Label("Always-Think", systemImage: "brain.head.profile")
                            .font(.caption2)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 3)
                            .background(Capsule().fill(Color.purple.opacity(0.14)))
                            .foregroundStyle(.purple)
                    }
                }

                let hasSys = !viewModel.draftSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                Button {
                    showSystemPromptSheet = true
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: hasSys ? "terminal.fill" : "terminal")
                        Text(hasSys ? "System ●" : "System")
                    }
                    .font(.caption)
                }
                .buttonStyle(.bordered)
                .tint(hasSys ? .accentColor : nil)
                .controlSize(.small)

                Button {
                    viewModel.showPromptPreview.toggle()
                } label: {
                    Label("Prompt", systemImage: viewModel.showPromptPreview ? "eye.fill" : "eye")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .tint(viewModel.showPromptPreview ? .accentColor : nil)
                .controlSize(.small)
                .help("Inspect exact Jinja-rendered prompt & special tokens")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .background(Color.secondary.opacity(0.04))
    }
}

private struct ChatPromptPreviewDrawer: View {
    let viewModel: HeylookWorkbenchViewModel

    private static let specialTokens: [String] = [
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>",
        "<|thought|>", "<|/thought|>", "<|channel>", "<channel|>",
        "<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|eot|>", "<|eom|>",
        "<think>", "</think>", "<thought>", "</thought>"
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label("RENDERED PROMPT PREVIEW", systemImage: "curlybraces.square")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.secondary)
                if viewModel.isRenderingPromptPreview {
                    ProgressView()
                        .controlSize(.mini)
                }
                if !viewModel.promptPreviewSource.isEmpty {
                    Text("• \(viewModel.promptPreviewSource)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                if !viewModel.promptPreviewMarkers.isEmpty {
                    Text("\(viewModel.promptPreviewMarkers.count) special markers •")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.purple)
                }
                Text("~\(max(0, viewModel.promptPreviewText.utf8.count / 4)) est. chars/4 tokens")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            if !viewModel.promptPreviewMarkers.isEmpty {
                let uniqueTokens = Array(Set(viewModel.promptPreviewMarkers.map(\.token))).sorted()
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 5) {
                        ForEach(uniqueTokens, id: \.self) { token in
                            Text(token)
                                .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Capsule().fill(Color.purple.opacity(0.14)))
                                .foregroundStyle(.purple)
                        }
                    }
                }
            }

            ScrollView {
                Text(highlightSpecialTokens(viewModel.promptPreviewText, markers: viewModel.promptPreviewMarkers))
                    .font(.system(size: 11.5, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .frame(maxHeight: 170)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.secondary.opacity(0.08))
            )
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.secondary.opacity(0.03))
    }

    private func highlightSpecialTokens(_ raw: String, markers: [PromptMarkerSpan]) -> AttributedString {
        guard !raw.isEmpty, raw.utf8.count <= 32_000 else {
            return AttributedString(raw)
        }
        var attr = AttributedString(raw)
        if !markers.isEmpty {
            let charCount = raw.count
            for marker in markers where marker.start >= 0 && marker.end > marker.start && marker.end <= charCount {
                if let sIdx = raw.index(raw.startIndex, offsetBy: marker.start, limitedBy: raw.endIndex),
                   let eIdx = raw.index(raw.startIndex, offsetBy: marker.end, limitedBy: raw.endIndex),
                   let attrStart = AttributedString.Index(sIdx, within: attr),
                   let attrEnd = AttributedString.Index(eIdx, within: attr),
                   attrStart < attrEnd {
                    attr[attrStart..<attrEnd].foregroundColor = .purple
                    attr[attrStart..<attrEnd].backgroundColor = Color.purple.opacity(0.14)
                    attr[attrStart..<attrEnd].inlinePresentationIntent = .stronglyEmphasized
                }
            }
            return attr
        }
        for token in Self.specialTokens {
            var searchRange = attr.startIndex..<attr.endIndex
            while let found = attr[searchRange].range(of: token) {
                attr[found].foregroundColor = .purple
                attr[found].backgroundColor = Color.purple.opacity(0.14)
                attr[found].inlinePresentationIntent = .stronglyEmphasized
                searchRange = found.upperBound..<attr.endIndex
            }
        }
        return attr
    }
}

private struct ChatEmptyWelcomeCard: View {
    let viewModel: HeylookWorkbenchViewModel

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "sparkles.tv")
                .font(.system(size: 40))
                .foregroundStyle(.tint)

            Text("Heylook Personal Inference Workbench")
                .font(.title3.weight(.semibold))

            Text("Every token, reasoning span, KV cache hit, and ANE/GPU placement is visible and inspectable.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 520)

            HStack(spacing: 10) {
                QuickPromptChip("Explain Swift 6 actor isolation with code examples") { prompt in
                    viewModel.draftPrompt = prompt
                }
                QuickPromptChip("Compare Matryoshka 256d vs 768d cosine search") { prompt in
                    viewModel.draftPrompt = prompt
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(28)
    }
}

private struct QuickPromptChip: View {
    let title: String
    let action: (String) -> Void

    init(_ title: String, action: @escaping (String) -> Void) {
        self.title = title
        self.action = action
    }

    var body: some View {
        Button {
            action(title)
        } label: {
            Text(title)
                .font(.caption)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(
                    Capsule()
                        .fill(Color.secondary.opacity(0.1))
                )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Unified Collapsible Thinking Trace Card

private struct ThinkingTraceCardView: View {
    let thinkingText: String
    let isStreaming: Bool
    let canEndThinking: Bool
    let isEndingThinking: Bool
    var onEndThinking: (() -> Void)?
    @Binding var isExpanded: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        isExpanded.toggle()
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                            .font(.caption2.weight(.semibold))
                            .frame(width: 10)
                        if isStreaming {
                            ProgressView()
                                .controlSize(.mini)
                        } else {
                            Image(systemName: "brain")
                                .font(.caption)
                        }
                        Text(isStreaming ? "Thinking…" : "Reasoning Trace (\(thinkingText.count) chars)")
                            .font(.caption.weight(.medium))
                    }
                    .foregroundStyle(.purple)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                Spacer()

                if canEndThinking {
                    Button {
                        onEndThinking?()
                    } label: {
                        HStack(spacing: 4) {
                            if isEndingThinking {
                                ProgressView()
                                    .controlSize(.mini)
                            }
                            Text(isEndingThinking ? "Ending…" : "End thinking")
                                .font(.caption2.weight(.semibold))
                        }
                    }
                    .buttonStyle(.bordered)
                    .tint(.purple)
                    .controlSize(.mini)
                    .disabled(isEndingThinking)
                }
            }

            if isExpanded {
                Text(thinkingText)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.purple.opacity(0.06))
                    )
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.purple.opacity(0.05))
        )
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(Color.purple.opacity(0.18), lineWidth: 1)
        }
    }
}

// MARK: - Message Bubble & Turn Actions

private struct MessageBubbleView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let message: MessageRecord
    let isLatestAssistant: Bool

    @State private var showThinking: Bool = false
    @State private var hasInitializedThinkingDisclosure: Bool = false
    @State private var showRawSource: Bool = false
    @State private var didCopyTurn: Bool = false

    private var isUser: Bool { message.role == "user" }
    private var isEditing: Bool { viewModel.editingMessageID == message.id }
    private var turnAttachments: [StagedChatAttachment] {
        viewModel.messageAttachmentsByID[message.id] ?? []
    }

    var body: some View {
        VStack(alignment: isUser ? .trailing : .leading, spacing: 6) {
            // Header
            HStack(spacing: 6) {
                Text(isUser ? "You" : (message.modelID ?? "Assistant"))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                if let stop = message.stopReason, !stop.isEmpty, stop != StopReason.endTurn.rawValue {
                    Text(stop)
                        .font(.system(size: 9, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.orange.opacity(0.15)))
                        .foregroundStyle(.orange)
                }
            }

            // Collapsible Reasoning Trace
            if !isEditing, let thinking = message.thinking, !thinking.isEmpty {
                ThinkingTraceCardView(
                    thinkingText: thinking,
                    isStreaming: false,
                    canEndThinking: false,
                    isEndingThinking: false,
                    onEndThinking: nil,
                    isExpanded: $showThinking
                )
                .frame(maxWidth: 760, alignment: .leading)
                .onAppear {
                    if !hasInitializedThinkingDisclosure {
                        hasInitializedThinkingDisclosure = true
                        // Keep expanded automatically when the turn produced no visible content!
                        showThinking = message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    }
                }
            }

            // In-Session Multimodal Attachments Gallery
            if !turnAttachments.isEmpty {
                HStack(spacing: 8) {
                    ForEach(turnAttachments) { att in
                        TurnAttachmentPreviewCard(attachment: att)
                    }
                }
            }

            if let omittedNote = viewModel.omittedMediaNote(for: message.id) {
                HStack(spacing: 4) {
                    Image(systemName: "eye.slash")
                        .font(.system(size: 10))
                    Text(omittedNote)
                        .font(.system(size: 10.5, weight: .medium))
                }
                .foregroundStyle(.orange)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Capsule().fill(Color.orange.opacity(0.12)))
            }

            // Message Body or Non-Destructive Multi-Verb Editor
            if isEditing {
                VStack(alignment: .leading, spacing: 8) {
                    if !isUser && (message.thinking != nil || viewModel.activeModelSupportsThinking) {
                        VStack(alignment: .leading, spacing: 4) {
                            Label("Thinking Trace (optional)", systemImage: "brain")
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.purple)
                            TextEditor(text: $viewModel.editingThinking)
                                .font(.system(size: 12, design: .monospaced))
                                .frame(minHeight: 64, maxHeight: 150)
                                .padding(6)
                                .background(
                                    RoundedRectangle(cornerRadius: 8)
                                        .strokeBorder(Color.purple.opacity(0.35), lineWidth: 1)
                                )
                        }
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        if !isUser {
                            Text("Assistant Response")
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.secondary)
                        }
                        TextEditor(text: $viewModel.editingContent)
                            .font(.body)
                            .frame(minHeight: 84, maxHeight: 260)
                            .padding(6)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .strokeBorder(Color.accentColor.opacity(0.4), lineWidth: 1)
                            )
                    }

                    HStack(spacing: 8) {
                        Button("Cancel") {
                            viewModel.cancelEditingMessage()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)

                        Spacer()

                        Button("Save") {
                            viewModel.saveEditedMessageInPlace(message)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Save edits in place without deleting subsequent turns")

                        Button("Save & Continue") {
                            Task { await viewModel.saveAndContinueEditedMessage(message) }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(viewModel.isGenerating)
                        .help(isUser ? "Truncate later turns and generate reply" : "Prefill this assistant text and continue generating")

                        Button("Save & Regenerate") {
                            Task { await viewModel.saveAndRegenerateEditedMessage(message) }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(viewModel.isGenerating)
                    }
                }
                .frame(maxWidth: 720)
            } else if !message.content.isEmpty || isUser {
                Group {
                    if showRawSource || isUser {
                        Text(message.content)
                            .font(showRawSource ? .system(size: 12.5, design: .monospaced) : .body)
                            .textSelection(.enabled)
                    } else {
                        MarkdownMessageView(text: message.content, isStreaming: false)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(isUser ? Color.accentColor.opacity(0.14) : Color.secondary.opacity(0.08))
                )
                .frame(maxWidth: 760, alignment: isUser ? .trailing : .leading)
            }

            // Footer Telemetry + Rich Turn Actions
            HStack(spacing: 10) {
                if let tps = message.generationTPS, tps > 0 {
                    Text(String(format: "%.1f tok/s", tps))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                if let count = message.tokenCount, count > 0 {
                    Text("\(count) tok")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
                if let ttft = message.ttftMs, ttft > 0 {
                    Text(String(format: "TTFT %.0fms", ttft))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
                if let placement = message.computePlacement {
                    Text(placement.rawValue)
                        .font(.system(size: 9.5, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
                if let cache = message.cacheReport {
                    Text(cache.summaryBadge)
                        .font(.system(size: 10, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.green.opacity(0.14)))
                        .foregroundStyle(.green)
                }
                if let spec = message.speculativeReport {
                    Text(spec.summaryBadge)
                        .font(.system(size: 10, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.blue.opacity(0.14)))
                        .foregroundStyle(.blue)
                }

                // Turn Action Buttons
                if !isEditing {
                    HStack(spacing: 8) {
                        Button {
                            copyMessageText(message.content)
                            didCopyTurn = true
                            Task { @MainActor in
                                try? await Task.sleep(for: .milliseconds(1400))
                                didCopyTurn = false
                            }
                        } label: {
                            Label(didCopyTurn ? "Copied" : "Copy", systemImage: didCopyTurn ? "checkmark" : "doc.on.doc")
                                .font(.caption2)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(didCopyTurn ? .green : .secondary)

                        Button {
                            viewModel.beginEditingMessage(message)
                        } label: {
                            Label("Edit", systemImage: "pencil")
                                .font(.caption2)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)

                        Button {
                            viewModel.cloneConversation(
                                id: message.conversationID,
                                upToPosition: message.position
                            )
                        } label: {
                            Label("Fork", systemImage: "arrow.branch")
                                .font(.caption2)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .help("Branch a cloned conversation up to this turn")

                        if !isUser {
                            Button {
                                showRawSource.toggle()
                            } label: {
                                Label(showRawSource ? "Rendered" : "Raw", systemImage: showRawSource ? "richtext.page" : "curlybraces")
                                    .font(.caption2)
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)

                            Button {
                                Task { await viewModel.regenerateAssistantMessage(message) }
                            } label: {
                                Label("Regenerate", systemImage: "arrow.clockwise")
                                    .font(.caption2)
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)
                            .disabled(viewModel.isGenerating)

                            if isLatestAssistant {
                                Button {
                                    Task { await viewModel.continueAssistantMessage(message) }
                                } label: {
                                    Label("Continue", systemImage: "text.append")
                                        .font(.caption2)
                                }
                                .buttonStyle(.plain)
                                .foregroundStyle(.secondary)
                                .disabled(viewModel.isGenerating)
                                .help("Prefill this response and continue generating")
                            }
                        }

                        Button {
                            viewModel.pendingConfirmation = .deleteMessage(
                                id: message.id,
                                conversationID: message.conversationID
                            )
                        } label: {
                            Image(systemName: "trash")
                                .font(.caption2)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .disabled(viewModel.isGenerating)
                        .help("Delete this turn")
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
    }

    private func copyMessageText(_ string: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(string, forType: .string)
        #elseif canImport(UIKit)
        UIPasteboard.general.string = string
        #endif
    }
}

private struct TurnAttachmentPreviewCard: View {
    let attachment: StagedChatAttachment

    var body: some View {
        if attachment.kind == .image, let cgImage = decodeCGImage(from: attachment.effectiveData) {
            VStack(alignment: .leading, spacing: 3) {
                Image(decorative: cgImage, scale: 1.0)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 112, height: 84)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                HStack(spacing: 4) {
                    Text(attachment.filename)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let dims = attachment.dimensionsLabel {
                        Text("• \(dims)")
                    }
                }
                .font(.system(size: 9.5, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(maxWidth: 112, alignment: .leading)
            }
            .padding(5)
            .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(0.08)))
        } else {
            HStack(spacing: 6) {
                Image(systemName: attachment.kind == .audio ? "waveform" : "photo")
                    .font(.caption)
                    .foregroundStyle(.tint)
                Text(attachment.filename)
                    .font(.caption2)
                    .lineLimit(1)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Capsule().fill(Color.secondary.opacity(0.12)))
        }
    }

    private func decodeCGImage(from data: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}

// MARK: - Active Streaming Assistant Bubble

private struct StreamingAssistantView: View {
    let viewModel: HeylookWorkbenchViewModel
    @State private var isThinkingExpanded: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)
                Text(viewModel.selectedModelID)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                if viewModel.streamingTPS > 0 {
                    Text(String(format: "%.1f tok/s • %d tok", viewModel.streamingTPS, viewModel.streamingTokens))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }

            if let statusLine = viewModel.streamStatusLine,
               viewModel.streamingThinking.isEmpty && viewModel.streamingContent.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "hourglass")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(statusLine)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.07)))
            }

            if !viewModel.streamingThinking.isEmpty {
                let canEnd = viewModel.isGenerating
                    && viewModel.streamingContent.isEmpty
                    && viewModel.activeModelSupportsThinking
                ThinkingTraceCardView(
                    thinkingText: viewModel.streamingThinking,
                    isStreaming: viewModel.streamingContent.isEmpty,
                    canEndThinking: canEnd,
                    isEndingThinking: viewModel.isEndingThinking,
                    onEndThinking: { viewModel.endThinkingNow() },
                    isExpanded: $isThinkingExpanded
                )
                .frame(maxWidth: 760, alignment: .leading)
            }

            if !viewModel.streamingContent.isEmpty {
                MarkdownMessageView(text: viewModel.streamingContent, isStreaming: true)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.secondary.opacity(0.08))
                    )
                    .frame(maxWidth: 760, alignment: .leading)
            }
        }
    }
}

// MARK: - Multimodal Chat Composer Bar

private struct ChatComposerBar: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Binding var showAttachmentImporter: Bool
    var isComposerFocused: FocusState<Bool>.Binding

    var body: some View {
        let capabilityWarning = viewModel.chatAttachmentCapabilityWarning
        let schemaError = viewModel.responseFormatValidationError
        let canSend = (!viewModel.draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !viewModel.stagedAttachments.isEmpty)
            && capabilityWarning == nil
            && schemaError == nil
            && !viewModel.isGenerating

        VStack(alignment: .leading, spacing: 6) {
            if !viewModel.stagedAttachments.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(viewModel.stagedAttachments) { att in
                            StagedComposerAttachmentPill(
                                attachment: att,
                                imagePlan: viewModel.estimateStagedImagePlan(for: att),
                                onToggleShrink: { viewModel.toggleAttachmentShrink(id: att.id) },
                                onRemove: { viewModel.removeDraftAttachment(id: att.id) }
                            )
                        }

                        let totalImgTokens = viewModel.totalStagedImageTokens
                        if totalImgTokens > 0 {
                            let cap = viewModel.stagedAttachments
                                .compactMap { viewModel.estimateStagedImagePlan(for: $0)?.tokenCap }
                                .first ?? 16384
                            Text("Σ ~\(totalImgTokens) / \(cap) img tok")
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .background(
                                    Capsule().fill(
                                        totalImgTokens > cap
                                            ? Color.orange.opacity(0.18)
                                            : Color.secondary.opacity(0.12)
                                    )
                                )
                                .foregroundStyle(totalImgTokens > cap ? .orange : .secondary)
                        }
                    }
                }
            }

            if viewModel.lastDroppedMediaCount > 0 {
                HStack(spacing: 6) {
                    Image(systemName: "eye.slash.fill")
                        .foregroundStyle(.orange)
                    Text("\(viewModel.lastDroppedMediaCount) historical attachment(s) omitted in last turn due to single-turn media capability.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            if let warning = capabilityWarning {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(warning)
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(.orange)
                }
            }

            if let schemaError {
                HStack(spacing: 6) {
                    Image(systemName: "curlybraces.square.fill")
                        .foregroundStyle(.red)
                    Text("Structured Output: \(schemaError)")
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(.red)
                }
            }

            if viewModel.isRecordingVoice {
                HStack(spacing: 6) {
                    Image(systemName: "waveform.circle.fill")
                        .foregroundStyle(.red)
                    Text("Recording 16 kHz mono — \(viewModel.activeASRBackendLabel)")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.red)
                    Spacer()
                    Button("Stop") {
                        viewModel.toggleVoiceRecording()
                    }
                    .buttonStyle(.borderless)
                    .font(.caption2.weight(.semibold))
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(
                    RoundedRectangle(cornerRadius: 7)
                        .fill(Color.red.opacity(0.10))
                )
            } else if let voiceResult = viewModel.lastVoiceTranscription, !voiceResult.text.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "waveform.badge.magnifyingglass")
                        .foregroundStyle(.teal)
                    Text(
                        String(
                            format: "ASR: %@ • %.1fs • %@ (%d mel)",
                            voiceResult.modelID,
                            voiceResult.durationSeconds,
                            voiceResult.computePlacement.rawValue.uppercased(),
                            voiceResult.melFrames
                        )
                    )
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        viewModel.lastVoiceTranscription = nil
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Dismiss voice transcription summary")
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 7)
                        .fill(Color.teal.opacity(0.08))
                )
            }

            HStack(alignment: .bottom, spacing: 10) {
                Button {
                    showAttachmentImporter = true
                } label: {
                    Image(systemName: "paperclip")
                        .font(.system(size: 15, weight: .medium))
                        .padding(7)
                }
                .buttonStyle(.bordered)
                .help("Attach Image or Audio File")

                #if os(macOS)
                Button {
                    if !viewModel.pasteAttachmentFromClipboard() {
                        viewModel.statusBanner = "Clipboard does not contain an image or media file."
                    }
                } label: {
                    Image(systemName: "doc.on.clipboard")
                        .font(.system(size: 14, weight: .medium))
                        .padding(7)
                }
                .buttonStyle(.bordered)
                .help("Paste Screenshot or Image from Clipboard (⌘⇧V)")
                .keyboardShortcut("v", modifiers: [.command, .shift])
                #endif

                if viewModel.isVoiceASRConfigured {
                    Button {
                        #if os(macOS)
                        let optionHeld = NSEvent.modifierFlags.contains(.option)
                        viewModel.toggleVoiceRecording(autoSendOnStop: optionHeld ? true : nil)
                        #else
                        viewModel.toggleVoiceRecording()
                        #endif
                    } label: {
                        Image(systemName: viewModel.isRecordingVoice ? "waveform.circle.fill" : (viewModel.autoSendVoiceTurn ? "mic.badge.plus" : "mic"))
                            .font(.system(size: 15, weight: .medium))
                            .foregroundStyle(viewModel.isRecordingVoice ? .red : .primary)
                            .padding(7)
                    }
                    .buttonStyle(.bordered)
                    .contextMenu {
                        Toggle("Hands-Free Auto-Send on Stop", isOn: $viewModel.autoSendVoiceTurn)
                        if !viewModel.availableASRModels.isEmpty {
                            Divider()
                            Button {
                                viewModel.selectedASRModelID = nil
                            } label: {
                                Label(
                                    "Automatic (\(viewModel.activeASRBackendLabel))",
                                    systemImage: viewModel.selectedASRModelID == nil ? "checkmark" : "waveform"
                                )
                            }
                            ForEach(viewModel.availableASRModels, id: \.modelID) { asrModel in
                                Button {
                                    viewModel.selectedASRModelID = asrModel.modelID
                                } label: {
                                    Label(
                                        asrModel.modelID,
                                        systemImage: viewModel.selectedASRModelID == asrModel.modelID ? "checkmark" : "cpu"
                                    )
                                }
                            }
                        }
                    }
                    .help(
                        viewModel.isRecordingVoice
                            ? "Stop Recording & Transcribe"
                            : "Dictate Voice Input — \(viewModel.activeASRBackendLabel) (⌥-click or right-click for Hands-Free Auto-Send)"
                    )
                }

                TextField("Message \(viewModel.selectedModelID.isEmpty ? "model" : viewModel.selectedModelID)… (Shift+Return for newline, ⌘Return or Return to send)", text: $viewModel.draftPrompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .focused(isComposerFocused)
                    .lineLimit(1...8)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 9)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color.secondary.opacity(0.08))
                    )
                    .onSubmit {
                        guard canSend else { return }
                        Task { await viewModel.sendChatPrompt() }
                    }

                if viewModel.isGenerating {
                    Button(role: .destructive) {
                        viewModel.cancelGeneration()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .help("Cancel active generation (Esc)")
                } else {
                    Button {
                        Task { await viewModel.sendChatPrompt() }
                    } label: {
                        Label("Send", systemImage: "arrow.up.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canSend)
                    .keyboardShortcut(.return, modifiers: [.command])
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }
}

private struct StagedComposerAttachmentPill: View {
    let attachment: StagedChatAttachment
    let imagePlan: StagedImagePlanEstimate?
    let onToggleShrink: () -> Void
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 7) {
            if attachment.kind == .image, let cgImage = decodeThumb(attachment.effectiveData) {
                Image(decorative: cgImage, scale: 1.0)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 28, height: 28)
                    .clipShape(RoundedRectangle(cornerRadius: 5))
            } else {
                Image(systemName: attachment.kind == .image ? "photo" : "waveform")
                    .font(.caption)
                    .foregroundStyle(Color.accentColor)
            }

            VStack(alignment: .leading, spacing: 1) {
                Text(attachment.filename)
                    .font(.caption2.weight(.medium))
                    .lineLimit(1)
                if let plan = imagePlan {
                    Text("~\(plan.estimatedTokens) tok • \(plan.resizedWidth)×\(plan.resizedHeight)")
                        .font(.system(size: 9, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.secondary)
                } else if let dims = attachment.dimensionsLabel {
                    Text(dims)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }

            if attachment.canToggleShrink {
                Button(action: onToggleShrink) {
                    Text(attachment.useShrunk2048 ? "≤2048px" : "Full res")
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            Capsule().fill(
                                attachment.useShrunk2048
                                    ? Color.green.opacity(0.18)
                                    : Color.orange.opacity(0.18)
                            )
                        )
                        .foregroundStyle(attachment.useShrunk2048 ? .green : .orange)
                }
                .buttonStyle(.plain)
                .help("Toggle between ≤2048px max-edge shrink and original full resolution")
            }

            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.accentColor.opacity(0.1))
        )
    }

    private func decodeThumb(_ data: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}

private struct SystemPromptEditorSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("System Prompt (persisted per conversation; included in Jinja prompt preview)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if !viewModel.draftSystemPrompt.isEmpty {
                        Button("Clear") {
                            viewModel.draftSystemPrompt = ""
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    }
                }

                TextEditor(text: $viewModel.draftSystemPrompt)
                    .font(.system(size: 13, design: .monospaced))
                    .frame(minHeight: 180)
                    .padding(6)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.secondary.opacity(0.08))
                    )
            }
            .padding(16)
            .navigationTitle("System Prompt")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        viewModel.flushSystemPromptSave()
                        dismiss()
                    }
                }
            }
        }
        .frame(minWidth: 480, minHeight: 300)
    }
}

// MARK: - Complete Optional 5-Layer Sampler Cascade & Presets Inspector

public struct LoadAndSamplerInspectorView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var newPresetName: String = ""
    @State private var showSaveAsField: Bool = false

    public init(viewModel: HeylookWorkbenchViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        let resolved = viewModel.currentResolvedSamplers

        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // 1. Two-Verb Preset Manager
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("SAMPLER PRESET")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer()
                        if viewModel.isPresetModified {
                            Text("modified")
                                .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 1)
                                .background(Capsule().fill(Color.orange.opacity(0.16)))
                                .foregroundStyle(.orange)
                        }
                        Button("Manage…") {
                            viewModel.activeSurface = .presets
                        }
                        .font(.caption2)
                        .buttonStyle(.plain)
                        .foregroundStyle(Color.accentColor)
                    }

                    Picker(
                        "Preset",
                        selection: Binding(
                            get: { viewModel.selectedPresetName ?? "__none__" },
                            set: { newVal in
                                if newVal == "__none__" {
                                    viewModel.clearPresetSelection()
                                } else if let p = viewModel.presets.first(where: { $0.name == newVal }) {
                                    viewModel.applyPreset(p)
                                }
                            }
                        )
                    ) {
                        Text("Model Defaults (No Preset)").tag("__none__")
                        ForEach(viewModel.presets) { preset in
                            Text(preset.name).tag(preset.name)
                        }
                    }
                    .labelsHidden()

                    HStack(spacing: 6) {
                        if let activeName = viewModel.selectedPresetName {
                            Button {
                                viewModel.updateActivePreset()
                            } label: {
                                Label("Update \"\(activeName)\"", systemImage: "square.and.arrow.down")
                                    .lineLimit(1)
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                            .disabled(!viewModel.isPresetModified)
                        }

                        Button {
                            withAnimation(.easeInOut(duration: 0.15)) {
                                showSaveAsField.toggle()
                            }
                        } label: {
                            Label("Save as…", systemImage: "plus.square.on.square")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)

                        if viewModel.selectedPresetName != nil {
                            Button(role: .destructive) {
                                viewModel.deleteActivePreset()
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .help("Delete selected preset")
                        }
                    }

                    if showSaveAsField {
                        HStack(spacing: 6) {
                            TextField("Preset name…", text: $newPresetName)
                                .textFieldStyle(.roundedBorder)
                                .font(.caption)
                                .onSubmit {
                                    commitSaveAs()
                                }
                            Button("Save") {
                                commitSaveAs()
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                            .disabled(newPresetName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                    }
                }

                Divider()

                // 2. Core Entropy & Nucleus Cascade Controls
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("SAMPLING CASCADE")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer()
                        if viewModel.hasAnyManualSamplerOverride {
                            Button("Reset All") {
                                viewModel.resetManualSamplerOverrides()
                            }
                            .font(.caption2)
                            .buttonStyle(.plain)
                            .foregroundStyle(Color.accentColor)
                        }
                    }

                    OptionalDoubleSliderControl(
                        title: "Temperature",
                        overrideValue: $viewModel.temperatureOverride,
                        effectiveValue: resolved.effective.temperature ?? 0.7,
                        range: 0.0...2.0,
                        step: 0.05,
                        format: "%.2f",
                        provenance: resolved.provenanceByField[.temperature] ?? .derived
                    )

                    OptionalDoubleSliderControl(
                        title: "Top P",
                        overrideValue: $viewModel.topPOverride,
                        effectiveValue: resolved.effective.topP ?? 0.95,
                        range: 0.05...1.0,
                        step: 0.01,
                        format: "%.2f",
                        provenance: resolved.provenanceByField[.topP] ?? .derived
                    )

                    OptionalDoubleSliderControl(
                        title: "Min P",
                        overrideValue: $viewModel.minPOverride,
                        effectiveValue: resolved.effective.minP ?? 0.0,
                        range: 0.0...0.5,
                        step: 0.01,
                        format: "%.2f",
                        provenance: resolved.provenanceByField[.minP] ?? .derived
                    )

                    OptionalIntStepperControl(
                        title: "Top K",
                        overrideValue: $viewModel.topKOverride,
                        effectiveValue: resolved.effective.topK ?? 40,
                        range: 0...200,
                        step: 5,
                        provenance: resolved.provenanceByField[.topK] ?? .derived
                    )

                    OptionalIntStepperControl(
                        title: "Max Output Tokens",
                        overrideValue: $viewModel.maxTokensOverride,
                        effectiveValue: resolved.effective.maxTokens ?? 2048,
                        range: 64...32768,
                        step: 256,
                        provenance: resolved.provenanceByField[.maxTokens] ?? .derived
                    )
                }

                Divider()

                // 3. Penalties, Seed, Stop Sequences & Thinking Budget
                VStack(alignment: .leading, spacing: 12) {
                    Text("PENALTIES, THINKING & DETERMINISM")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)

                    OptionalDoubleSliderControl(
                        title: "Repetition Penalty",
                        overrideValue: $viewModel.repetitionPenaltyOverride,
                        effectiveValue: resolved.effective.repetitionPenalty ?? 1.0,
                        range: 1.0...1.5,
                        step: 0.02,
                        format: "%.2f",
                        provenance: resolved.provenanceByField[.repetitionPenalty] ?? .derived
                    )

                    OptionalIntStepperControl(
                        title: "Repetition Window",
                        overrideValue: $viewModel.repetitionContextSizeOverride,
                        effectiveValue: resolved.effective.repetitionContextSize ?? 64,
                        range: 0...512,
                        step: 16,
                        provenance: resolved.provenanceByField[.repetitionContextSize] ?? .derived
                    )

                    OptionalDoubleSliderControl(
                        title: "Presence Penalty",
                        overrideValue: $viewModel.presencePenaltyOverride,
                        effectiveValue: resolved.effective.presencePenalty ?? 0.0,
                        range: 0.0...2.0,
                        step: 0.05,
                        format: "%.2f",
                        provenance: resolved.provenanceByField[.presencePenalty] ?? .derived
                    )

                    if viewModel.activeModelSupportsThinking {
                        OptionalNullableIntFieldControl(
                            title: "Thinking Token Budget",
                            placeholder: "Unlimited",
                            overrideValue: $viewModel.thinkingBudgetOverride,
                            effectiveValue: resolved.effective.thinkingBudget,
                            provenance: resolved.provenanceByField[.thinkingBudget] ?? .derived
                        )
                    }

                    OptionalNullableIntFieldControl(
                        title: "Random Seed",
                        placeholder: "Random",
                        overrideValue: $viewModel.seedOverride,
                        effectiveValue: resolved.effective.seed,
                        provenance: resolved.provenanceByField[.seed] ?? .derived
                    )

                    StopSequencesEditorControl(
                        overrideValue: $viewModel.stopSequencesOverride,
                        effectiveValue: resolved.effective.stopSequences ?? [],
                        provenance: resolved.provenanceByField[.stopSequences] ?? .derived
                    )
                }

                Divider()

                // 4. Context-Aware System Prompt (Conversation vs Notebook)
                VStack(alignment: .leading, spacing: 6) {
                    let isNotebook = viewModel.activeSurface == .notebook
                    let promptBinding: Binding<String> = isNotebook
                        ? $viewModel.notebookSystemPrompt
                        : $viewModel.draftSystemPrompt
                    HStack {
                        Text(isNotebook ? "NOTEBOOK SYSTEM PROMPT" : "CONVERSATION SYSTEM PROMPT")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer()
                        if !promptBinding.wrappedValue.isEmpty {
                            Button("Clear") {
                                promptBinding.wrappedValue = ""
                            }
                            .font(.caption2)
                            .buttonStyle(.plain)
                            .foregroundStyle(Color.accentColor)
                        }
                    }

                    TextEditor(text: promptBinding)
                        .font(.system(size: 11.5, design: .monospaced))
                        .frame(minHeight: 68, maxHeight: 130)
                        .padding(5)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.secondary.opacity(0.08))
                        )
                }

                Divider()

                // 5. Structured Output (`response_format`: text / json_object / json_schema)
                VStack(alignment: .leading, spacing: 8) {
                    Text("STRUCTURED OUTPUT (RESPONSE FORMAT)")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)

                    Picker("Format", selection: $viewModel.responseFormatMode) {
                        ForEach(StructuredOutputMode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)

                    if viewModel.responseFormatMode == .jsonSchema {
                        HStack(spacing: 8) {
                            TextField("Schema name", text: $viewModel.responseFormatSchemaName)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 11, design: .monospaced))
                            Toggle("Strict", isOn: $viewModel.responseFormatStrict)
                                .font(.caption)
                                .toggleStyle(.switch)
                                .controlSize(.mini)
                        }

                        TextEditor(text: $viewModel.responseFormatSchemaJSON)
                            .font(.system(size: 11, design: .monospaced))
                            .frame(minHeight: 92, maxHeight: 160)
                            .padding(5)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color.secondary.opacity(0.08))
                            )

                        if let err = viewModel.responseFormatValidationError {
                            Label(err, systemImage: "exclamationmark.triangle.fill")
                                .font(.caption2)
                                .foregroundStyle(.red)
                        } else {
                            Label("Valid JSON Schema object", systemImage: "checkmark.circle.fill")
                                .font(.caption2)
                                .foregroundStyle(.green)
                        }
                    }
                }

                // 6. Active Model Provenance Card + Live RAM Fit
                if let active = viewModel.activeModelDescription {
                    Divider()

                    let fit = viewModel.assessModelRAMFit(active)
                    VStack(alignment: .leading, spacing: 8) {
                        Text("ACTIVE ENGINE PROVENANCE")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        HStack {
                            Text("RAM Fit")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Spacer()
                            let fitLabel = !fit.fitsInMemory
                                ? "EXCEEDS"
                                : (fit.remainingHeadroomGB < 2.0 ? "TIGHT" : "FITS")
                            let fitColor: Color = !fit.fitsInMemory
                                ? .red
                                : (fit.remainingHeadroomGB < 2.0 ? .orange : .green)
                            Text(String(format: "%@ (%.1f GB headroom)", fitLabel, fit.remainingHeadroomGB))
                                .font(.system(size: 10.5, weight: .semibold, design: .monospaced))
                                .foregroundStyle(fitColor)
                        }

                        ProvenanceKeyValueRow(key: "Architecture", field: active.architecture)
                        ProvenanceKeyValueRow(key: "Quantization", field: active.quantization)
                        ProvenanceKeyValueRow(
                            key: "Placement",
                            valueText: active.computePlacement.value.label,
                            source: active.computePlacement.provenance,
                            effect: active.computePlacement.effect
                        )
                        ProvenanceKeyValueRow(
                            key: "Context Size",
                            valueText: "\(active.contextSize.value) tok",
                            source: active.contextSize.provenance,
                            effect: active.contextSize.effect
                        )
                        ProvenanceKeyValueRow(key: "Reasoning Parser", field: active.reasoningFormat)
                        ProvenanceKeyValueRow(key: "Template Source", field: active.templateSource)
                    }
                }
            }
            .padding(16)
        }
    }

    private func commitSaveAs() {
        let trimmed = newPresetName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        viewModel.saveCurrentAsPreset(name: trimmed)
        newPresetName = ""
        showSaveAsField = false
    }
}

// MARK: - Optional Sampler Cascade Control Subviews

private struct OptionalDoubleSliderControl: View {
    let title: String
    @Binding var overrideValue: Double?
    let effectiveValue: Double
    let range: ClosedRange<Double>
    let step: Double
    let format: String
    let provenance: ProvenanceSource

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(title)
                    .font(.caption)
                Spacer()
                if overrideValue != nil {
                    Button {
                        overrideValue = nil
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Reset to model/preset default")
                }
                ProvenanceBadgeView(source: provenance)
                Text(String(format: format, effectiveValue))
                    .font(.system(size: 11, weight: overrideValue != nil ? .bold : .semibold, design: .monospaced))
                    .foregroundStyle(overrideValue != nil ? Color.accentColor : .primary)
            }
            Slider(
                value: Binding(
                    get: { overrideValue ?? effectiveValue },
                    set: { overrideValue = $0 }
                ),
                in: range,
                step: step
            )
            .controlSize(.small)
        }
    }
}

private struct OptionalIntStepperControl: View {
    let title: String
    @Binding var overrideValue: Int?
    let effectiveValue: Int
    let range: ClosedRange<Int>
    let step: Int
    let provenance: ProvenanceSource

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(title)
                    .font(.caption)
                Spacer()
                if overrideValue != nil {
                    Button {
                        overrideValue = nil
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Reset to model/preset default")
                }
                ProvenanceBadgeView(source: provenance)
            }
            Stepper(
                value: Binding(
                    get: { overrideValue ?? effectiveValue },
                    set: { overrideValue = $0 }
                ),
                in: range,
                step: step
            ) {
                Text("\(effectiveValue)")
                    .font(.system(size: 11, weight: overrideValue != nil ? .bold : .semibold, design: .monospaced))
                    .foregroundStyle(overrideValue != nil ? Color.accentColor : .primary)
            }
            .controlSize(.small)
        }
    }
}

private struct OptionalNullableIntFieldControl: View {
    let title: String
    let placeholder: String
    @Binding var overrideValue: Int?
    let effectiveValue: Int?
    let provenance: ProvenanceSource

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(title)
                    .font(.caption)
                Spacer()
                if overrideValue != nil {
                    Button {
                        overrideValue = nil
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Reset to model/preset default")
                }
                ProvenanceBadgeView(source: provenance)
            }
            TextField(
                effectiveValue.map { "\($0)" } ?? placeholder,
                text: Binding(
                    get: {
                        if let v = overrideValue ?? effectiveValue {
                            return "\(v)"
                        }
                        return ""
                    },
                    set: { str in
                        let trimmed = str.trimmingCharacters(in: .whitespaces)
                        if trimmed.isEmpty {
                            overrideValue = nil
                        } else if let parsed = Int(trimmed) {
                            overrideValue = parsed
                        }
                    }
                )
            )
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 11, design: .monospaced))
        }
    }
}

private struct StopSequencesEditorControl: View {
    @Binding var overrideValue: [String]?
    let effectiveValue: [String]
    let provenance: ProvenanceSource
    @State private var newStopSeq: String = ""

    var body: some View {
        let activeList = overrideValue ?? effectiveValue
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                Text("Stop Sequences")
                    .font(.caption)
                Spacer()
                if overrideValue != nil {
                    Button {
                        overrideValue = nil
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Reset stop sequences")
                }
                ProvenanceBadgeView(source: provenance)
            }

            if !activeList.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 5) {
                        ForEach(activeList, id: \.self) { seq in
                            HStack(spacing: 4) {
                                Text(seq.debugDescription)
                                    .font(.system(size: 10, design: .monospaced))
                                Button {
                                    var copy = activeList
                                    copy.removeAll { $0 == seq }
                                    overrideValue = copy
                                } label: {
                                    Image(systemName: "xmark")
                                        .font(.system(size: 8, weight: .bold))
                                }
                                .buttonStyle(.plain)
                            }
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.secondary.opacity(0.14)))
                        }
                    }
                }
            }

            HStack(spacing: 6) {
                TextField("Add stop string…", text: $newStopSeq)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11, design: .monospaced))
                    .onSubmit { addSequence() }
                Button("Add") { addSequence() }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .disabled(newStopSeq.isEmpty)
            }
        }
    }

    private func addSequence() {
        guard !newStopSeq.isEmpty else { return }
        var copy = overrideValue ?? effectiveValue
        if !copy.contains(newStopSeq) {
            copy.append(newStopSeq)
            overrideValue = copy
        }
        newStopSeq = ""
    }
}

// MARK: - Shared Provenance Visual Badges (Forward-Compatible With Fine-Grained Provenance Sources)

public struct SamplerProvenancePill: View {
    public let source: ProvenanceSource
    public var isRequestOverride: Bool = false

    public init(source: ProvenanceSource, isRequestOverride: Bool = false) {
        self.source = source
        self.isRequestOverride = isRequestOverride
    }

    public var body: some View {
        if isRequestOverride {
            Text("request")
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(Capsule().fill(Color.purple.opacity(0.16)))
                .foregroundStyle(.purple)
        } else {
            ProvenanceBadgeView(source: source)
        }
    }
}

public struct ProvenanceBadgeView: View {
    public let source: ProvenanceSource

    public init(source: ProvenanceSource) {
        self.source = source
    }

    private var color: Color {
        switch source.rawValue {
        case "builtin": return .secondary
        case "vendor", "derived": return .teal
        case "model_config", "configured": return .blue
        case "preset": return .indigo
        case "request": return .purple
        case "observed": return .green
        case "observed_cached": return .mint
        case "unknown": return .orange
        default: return .secondary
        }
    }

    public var body: some View {
        Text(source.rawValue)
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(Capsule().fill(color.opacity(0.16)))
            .foregroundStyle(color)
    }
}

public struct ProvenancePill: View {
    public let label: String
    public let source: ProvenanceSource

    public init(label: String, source: ProvenanceSource) {
        self.label = label
        self.source = source
    }

    public var body: some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.system(size: 11, weight: .medium))
            ProvenanceBadgeView(source: source)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(
            Capsule()
                .fill(Color.secondary.opacity(0.1))
        )
    }
}

public struct ProvenanceKeyValueRow: View {
    let key: String
    let valueText: String
    let source: ProvenanceSource
    let effect: EffectClass

    public init(key: String, field: ProvenanceField<String>) {
        self.key = key
        self.valueText = field.value
        self.source = field.provenance
        self.effect = field.effect
    }

    public init(key: String, valueText: String, source: ProvenanceSource, effect: EffectClass) {
        self.key = key
        self.valueText = valueText
        self.source = source
        self.effect = effect
    }

    public var body: some View {
        HStack {
            Text(key)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Spacer()
            Text(valueText)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .lineLimit(1)
            ProvenanceBadgeView(source: source)
        }
    }
}
