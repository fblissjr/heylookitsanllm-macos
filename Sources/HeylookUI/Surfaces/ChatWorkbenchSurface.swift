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
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast
    @State private var showSystemPromptSheet: Bool = false
    @State private var showCompactControlsSheet: Bool = false
    @State private var followingTail: Bool = true
    @State private var showAttachmentImporter: Bool = false
    @State private var isDropTargeted: Bool = false
    @FocusState private var isComposerFocused: Bool

    public init(viewModel: HeylookWorkbenchViewModel) {
        self.viewModel = viewModel
    }

    private var isCompact: Bool { horizontalSizeClass == .compact }

    public var body: some View {
        let visibleMessages = viewModel.chat.visibleConversationMessages
        let contrast = AdaptiveContrastStyle(colorSchemeContrast)

        VStack(spacing: 0) {
            ChatEngineReadoutBar(
                viewModel: viewModel,
                showSystemPromptSheet: $showSystemPromptSheet,
                showCompactControlsSheet: $showCompactControlsSheet
            )

            if viewModel.showLoadPanel {
                Divider()
                ModelLoadPanelView(viewModel: viewModel)
            }

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
                            if visibleMessages.isEmpty
                                && !viewModel.chat.isActivelyStreamingSelectedConversation
                                && viewModel.chat.backgroundGeneratingConversationID != viewModel.selectedConversationID {
                                ChatEmptyWelcomeCard(viewModel: viewModel)
                                    .padding(.top, 36)
                            }

                            let lastAssistantID = visibleMessages.last(where: { $0.role == "assistant" })?.id
                            ForEach(visibleMessages) { msg in
                                MessageBubbleView(
                                    viewModel: viewModel,
                                    message: msg,
                                    isLatestAssistant: msg.id == lastAssistantID
                                )
                                .id(msg.id)
                            }

                            if viewModel.chat.isActivelyStreamingSelectedConversation {
                                StreamingAssistantView(viewModel: viewModel)
                                    .id("streaming-assistant")
                            } else if let bgConvID = viewModel.chat.backgroundGeneratingConversationID,
                                      bgConvID == viewModel.selectedConversationID {
                                BackgroundGeneratingBannerView(viewModel: viewModel)
                                    .id("streaming-background")
                            }

                            Color.clear
                                .frame(height: 1)
                                .id("chat-bottom")
                        }
                        .padding(.horizontal, isCompact ? 12 : 20)
                        .padding(.vertical, 16)
                    }
                    .onScrollGeometryChange(for: Bool.self) { geo in
                        let distanceFromBottom = geo.contentSize.height - geo.contentOffset.y - geo.containerSize.height
                        return distanceFromBottom < 100
                    } action: { _, isNearBottom in
                        followingTail = isNearBottom
                    }
                    .onChange(of: visibleMessages.count) { _, _ in
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
                        if !followingTail && viewModel.chat.isSelectedConversationGenerating {
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
                                .frame(minHeight: isCompact ? 44 : 28)
                                .background(.regularMaterial, in: Capsule())
                                .overlay {
                                    Capsule().strokeBorder(Color.accentColor.opacity(contrast.isIncreased ? 0.65 : 0.35), lineWidth: contrast.borderWidth)
                                }
                                .shadow(color: .black.opacity(0.12), radius: 6, y: 2)
                            }
                            .buttonStyle(.plain)
                            .padding(.bottom, 12)
                        }
                    }
                }

                if isDropTargeted {
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                        .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                        .overlay {
                            Label("Drop image or audio files to stage", systemImage: "arrow.down.doc.fill")
                                .font(.callout.weight(.semibold))
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(.regularMaterial, in: Capsule())
                        }
                        .padding(16)
                        .allowsHitTesting(false)
                }
            }
            .dropDestination(for: URL.self) { items, _ in
                guard !items.isEmpty else { return false }
                viewModel.attachChatMedia(from: items)
                return true
            } isTargeted: { targeted in
                isDropTargeted = targeted
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
                viewModel.attachChatMedia(from: urls)
            }
        }
        .sheet(isPresented: $showSystemPromptSheet) {
            SystemPromptEditorSheet(viewModel: viewModel)
        }
        .sheet(isPresented: $showCompactControlsSheet) {
            CompactChatControlsSheet(
                viewModel: viewModel,
                showSystemPromptSheet: $showSystemPromptSheet
            )
        }
        .onAppear {
            isComposerFocused = true
        }
    }
}

private struct BackgroundGeneratingBannerView: View {
    let viewModel: HeylookWorkbenchViewModel

    var body: some View {
        HStack(spacing: 10) {
            ProgressView()
                .controlSize(.small)
            VStack(alignment: .leading, spacing: 2) {
                Text("Generation in progress…")
                    .font(.caption.weight(.semibold))
                Text("Streaming continues in the background and will sync automatically when complete.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Stop", role: .destructive) {
                viewModel.cancelGeneration()
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
            .controlSize(.small)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.accentColor.opacity(0.08)))
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
    @Binding var showCompactControlsSheet: Bool
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast

    private var contrast: AdaptiveContrastStyle {
        AdaptiveContrastStyle(colorSchemeContrast)
    }

    private var groupedModelsByProvider: [(provider: EngineProviderKind, models: [EngineDescription])] {
        let order: [EngineProviderKind] = [.mlx, .coreai, .gguf, .remote]
        let grouped = Dictionary(grouping: viewModel.generationModels, by: \.provider)
        return order.compactMap { kind in
            guard let list = grouped[kind], !list.isEmpty else { return nil }
            return (kind, list)
        }
    }

    private var systemPromptChipDisplay: (label: String, isEmpty: Bool, isModified: Bool, tooltip: String) {
        let trimmed = viewModel.draftSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let settings = viewModel.chatSettings
        let stamped = settings.stampedPreset
        let presetPrompt = stamped?.systemPrompt ?? ""
        let presetOwnsPrompt = !presetPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let label: String
        let isModified: Bool
        if trimmed.isEmpty {
            if let stamped, presetOwnsPrompt {
                label = "\(stamped.name) (modified)"
                isModified = true
            } else {
                label = "No system prompt"
                isModified = false
            }
        } else if let stamped, presetOwnsPrompt {
            if viewModel.draftSystemPrompt == presetPrompt {
                label = "System prompt: \(stamped.name)"
                isModified = false
            } else {
                label = "\(stamped.name) (modified)"
                isModified = true
            }
        } else {
            label = "System prompt: custom"
            isModified = false
        }
        let tooltip: String
        if trimmed.isEmpty {
            tooltip = "Click to set a system prompt"
        } else if trimmed.count > 300 {
            tooltip = String(trimmed.prefix(300)) + "…"
        } else {
            tooltip = trimmed
        }
        return (label, trimmed.isEmpty, isModified, tooltip)
    }

    var body: some View {
        if horizontalSizeClass == .compact {
            compactPhoneBar
        } else {
            desktopReadoutBar
        }
    }

    private var compactPhoneBar: some View {
        let active = viewModel.activeModelDescription
        let resolved = viewModel.currentResolvedSamplers
        let ctxK = max(1, (viewModel.chat.activeModelRunningContext ?? active?.contextSize.value ?? 4096) / 1024)
        let tempText = resolved.effective.temperature.map { String(format: "T %.2g", $0) } ?? "T def"
        let presetPart = viewModel.selectedPresetName.map { viewModel.isPresetModified ? "\($0)*" : $0 } ?? "Defaults"
        let sysPart = systemPromptChipDisplay.isEmpty ? "No sys" : "Sys"
        let summaryParts = [
            active?.modelID ?? (viewModel.selectedModelID.isEmpty ? "No model" : viewModel.selectedModelID),
            "\(ctxK)k",
            tempText,
            presetPart,
            sysPart
        ]

        return HStack(spacing: 8) {
            Button {
                showCompactControlsSheet = true
            } label: {
                HStack(spacing: 8) {
                    Circle()
                        .fill(statusDotColor(for: active))
                        .frame(width: 8, height: 8)
                    Text(summaryParts.joined(separator: " · "))
                        .font(.caption.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 4)
                    Image(systemName: "slider.horizontal.3")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .frame(minHeight: 44)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Chat model and sampler controls")
        }
        .background(Color.secondary.opacity(contrast.isIncreased ? 0.10 : 0.04))
    }

    private var desktopReadoutBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                modelSelectorMenu

                if let active = viewModel.activeModelDescription {
                    loadPanelButton(for: active)

                    let ctxK = max(1, (viewModel.chat.activeModelRunningContext ?? active.contextSize.value) / 1024)
                    ProvenancePill(
                        label: "ctx \(ctxK)k",
                        source: active.contextSize.provenance
                    )

                    if let drafter = active.speculativeDrafter.value {
                        ProvenancePill(
                            label: "MTP: \(drafter)",
                            source: active.speculativeDrafter.provenance
                        )
                    }
                }

                appliedPresetChip
                responseFormatBadge

                Spacer(minLength: 12)

                systemPromptChip
                promptPreviewToggleButton
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .background(Color.secondary.opacity(contrast.isIncreased ? 0.08 : 0.04))
    }

    @ViewBuilder
    private var modelSelectorMenu: some View {
        Menu {
            if groupedModelsByProvider.isEmpty {
                Text("No generation models configured")
            } else {
                ForEach(groupedModelsByProvider, id: \.provider) { section in
                    Section(section.provider.displayName) {
                        ForEach(section.models, id: \.modelID) { model in
                            let isLoading = viewModel.loadingModelIDs.contains(model.modelID)
                            let bullet = isLoading ? "◐" : (model.isLoaded ? "●" : "○")
                            let selectedMark = (model.modelID == viewModel.selectedModelID) ? " ✓" : ""
                            Button {
                                viewModel.selectGenerationModel(model.modelID)
                            } label: {
                                Text("\(bullet) \(model.modelID)\(selectedMark)")
                            }
                        }
                    }
                }
            }
        } label: {
            let active = viewModel.activeModelDescription
            let isLoading = active.map { viewModel.loadingModelIDs.contains($0.modelID) } ?? false
            HStack(spacing: 6) {
                Text(isLoading ? "◐" : (active?.isLoaded == true ? "●" : "○"))
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(statusDotColor(for: active))
                Text(active?.modelID ?? (viewModel.selectedModelID.isEmpty ? "Select model…" : viewModel.selectedModelID))
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                if let active {
                    Text(active.provider.displayName)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(Capsule().fill(Color.accentColor.opacity(contrast.chipFillOpacity)))
            .overlay {
                Capsule().strokeBorder(contrast.borderColor(.accentColor), lineWidth: contrast.borderWidth)
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    @ViewBuilder
    private func loadPanelButton(for active: EngineDescription) -> some View {
        let needsReload = viewModel.chat.canReloadActiveModel
        let isLoading = viewModel.chat.isLoadingPanelModel || viewModel.loadingModelIDs.contains(active.modelID)
        let labelText = isLoading ? "Loading…" : (!active.isLoaded ? "Load" : (needsReload ? "Reload" : "Loaded"))
        let fillColor: Color = !active.isLoaded
            ? Color.green.opacity(contrast.chipFillOpacity)
            : (needsReload ? Color.orange.opacity(contrast.chipFillOpacity) : Color.secondary.opacity(contrast.chipFillOpacity))
        let fgColor: Color = !active.isLoaded ? .green : (needsReload ? .orange : .primary)

        Button {
            withAnimation(.easeInOut(duration: 0.15)) {
                viewModel.showLoadPanel.toggle()
            }
            if viewModel.showLoadPanel {
                Task { await viewModel.chat.refreshActiveAdminModel(resetDrafts: true) }
            }
        } label: {
            HStack(spacing: 4) {
                if isLoading {
                    ProgressView()
                        .controlSize(.mini)
                } else {
                    Image(systemName: active.isLoaded ? (needsReload ? "arrow.triangle.2.circlepath.circle.fill" : "slider.horizontal.below.rectangle") : "bolt.circle")
                        .font(.caption2)
                }
                Text(labelText)
                    .font(.caption2.weight(.semibold))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Capsule().fill(fillColor))
            .foregroundStyle(fgColor)
        }
        .buttonStyle(.plain)
        .help("Configure context window / flash attention and Load or Reload model")
    }

    @ViewBuilder
    private var appliedPresetChip: some View {
        if let presetName = viewModel.selectedPresetName {
            Button {
                viewModel.showInspector = true
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "dial.medium")
                        .font(.system(size: 10))
                    Text(viewModel.isPresetModified ? "\(presetName) (modified)" : presetName)
                        .font(.system(size: 11, weight: .medium))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(
                    Capsule().fill(
                        viewModel.isPresetModified
                            ? Color.orange.opacity(contrast.chipFillOpacity)
                            : Color.indigo.opacity(contrast.chipFillOpacity)
                    )
                )
                .foregroundStyle(viewModel.isPresetModified ? .orange : .indigo)
            }
            .buttonStyle(.plain)
            .help("Active preset (click to view preset drift and inspector)")
        }
    }

    @ViewBuilder
    private var responseFormatBadge: some View {
        if viewModel.responseFormatMode != .text {
            HStack(spacing: 4) {
                Image(systemName: "curlybraces")
                    .font(.system(size: 10))
                Text(viewModel.responseFormatMode.label)
                    .font(.system(size: 11, weight: .semibold))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Capsule().fill(Color.teal.opacity(contrast.chipFillOpacity)))
            .foregroundStyle(.teal)
        }
    }

    @ViewBuilder
    private var systemPromptChip: some View {
        let sysChip = systemPromptChipDisplay
        Button {
            showSystemPromptSheet = true
        } label: {
            HStack(spacing: 5) {
                Image(systemName: sysChip.isEmpty ? "text.badge.plus" : "terminal.fill")
                    .font(.caption2)
                Text(sysChip.label)
                    .font(.caption.weight(sysChip.isEmpty ? .regular : .medium))
                    .lineLimit(1)
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .foregroundStyle(
                sysChip.isEmpty
                    ? AnyShapeStyle(.secondary)
                    : AnyShapeStyle(sysChip.isModified ? Color.orange : Color.accentColor)
            )
            .background(
                Capsule().fill(
                    sysChip.isEmpty
                        ? Color.clear
                        : (sysChip.isModified ? Color.orange.opacity(contrast.chipFillOpacity) : Color.accentColor.opacity(contrast.chipFillOpacity))
                )
            )
            .overlay {
                if sysChip.isEmpty {
                    Capsule()
                        .strokeBorder(contrast.borderColor(.secondary), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                } else {
                    Capsule()
                        .strokeBorder(contrast.borderColor(sysChip.isModified ? .orange : .accentColor), lineWidth: contrast.borderWidth)
                }
            }
        }
        .buttonStyle(.plain)
        .help(sysChip.tooltip)
    }

    private var promptPreviewToggleButton: some View {
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

    private func statusDotColor(for active: EngineDescription?) -> Color {
        guard let active else { return .secondary }
        if viewModel.loadingModelIDs.contains(active.modelID) || viewModel.chat.isLoadingPanelModel {
            return .orange
        }
        return active.isLoaded ? .green : .secondary.opacity(0.6)
    }
}

private struct ModelLoadPanelView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast
    @State private var useCustomContext: Bool = false
    @State private var customContextText: String = ""

    private var contrast: AdaptiveContrastStyle {
        AdaptiveContrastStyle(colorSchemeContrast)
    }

    var body: some View {
        let active = viewModel.activeModelDescription
        let isLoaded = viewModel.chat.activeAdminRow?.loaded ?? active?.isLoaded ?? false
        let isLoading = viewModel.chat.isLoadingPanelModel || (active.map { viewModel.loadingModelIDs.contains($0.modelID) } ?? false)
        let ceiling = max(512, viewModel.chat.activeModelContextCeiling)
        let steps = viewModel.chat.activeModelContextStepChoices

        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Label(active?.modelID ?? "Model Load Panel", systemImage: "cpu")
                    .font(.caption.weight(.semibold))
                if let active {
                    Text(active.provider.displayName)
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.secondary.opacity(contrast.chipFillOpacity)))
                }
                if let running = viewModel.chat.activeModelRunningContext {
                    Text("Running ctx \(running)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                if !viewModel.chat.activeModelStaleReloadFields.isEmpty {
                    Text("Reload needed: \(viewModel.chat.activeModelStaleReloadFields.joined(separator: ", "))")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.orange)
                }
                Spacer()
                Button("Close") {
                    viewModel.showLoadPanel = false
                }
                .buttonStyle(.borderless)
                .font(.caption)
            }

            if viewModel.chat.isActiveModelGGUF {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Context Size (`ctx_size`)")
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.secondary)
                        Picker(
                            "Context Size",
                            selection: Binding<String>(
                                get: {
                                    if useCustomContext { return "__custom__" }
                                    return viewModel.chat.loadPanelCtxSize.map(String.init) ?? "__auto__"
                                },
                                set: { selection in
                                    if selection == "__auto__" {
                                        useCustomContext = false
                                        viewModel.chat.loadPanelCtxSize = nil
                                    } else if selection == "__custom__" {
                                        useCustomContext = true
                                        customContextText = viewModel.chat.loadPanelCtxSize.map(String.init) ?? "\(ceiling)"
                                    } else if let n = Int(selection) {
                                        useCustomContext = false
                                        viewModel.chat.loadPanelCtxSize = min(max(512, n), ceiling)
                                    }
                                }
                            )
                        ) {
                            Text("Auto (max \(ceiling))").tag("__auto__")
                            ForEach(steps, id: \.self) { step in
                                let label = step >= 1024 && step.isMultiple(of: 1024) ? "\(step / 1024)k (\(step))" : "\(step)"
                                Text(label).tag("\(step)")
                            }
                            Text("Custom…").tag("__custom__")
                        }
                        .labelsHidden()
                        .frame(maxWidth: 220)
                    }

                    if useCustomContext {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Custom (`512…\(ceiling)`)")
                                .font(.caption2.weight(.medium))
                                .foregroundStyle(.secondary)
                            TextField("Tokens", text: $customContextText)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                                .frame(width: 110)
                                .onChange(of: customContextText) { _, newValue in
                                    if let n = Int(newValue.trimmingCharacters(in: .whitespaces)) {
                                        viewModel.chat.loadPanelCtxSize = min(max(512, n), ceiling)
                                    }
                                }
                        }
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Flash Attention (`flash_attn`)")
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.secondary)
                        Picker(
                            "Flash Attention",
                            selection: Binding<String>(
                                get: { viewModel.chat.loadPanelFlashAttn ?? "__default__" },
                                set: { viewModel.chat.loadPanelFlashAttn = ($0 == "__default__") ? nil : $0 }
                            )
                        ) {
                            Text(viewModel.chat.activeModelFlashAttnDefaultLabel).tag("__default__")
                            Text("on").tag("on")
                            Text("off").tag("off")
                            Text("auto").tag("auto")
                        }
                        .labelsHidden()
                        .frame(maxWidth: 160)
                    }

                    Spacer()
                }
            }

            HStack(spacing: 10) {
                if !isLoaded {
                    Button {
                        Task { await viewModel.chat.performLoadOrReloadActiveModel(warm: true) }
                    } label: {
                        HStack(spacing: 6) {
                            if isLoading { ProgressView().controlSize(.mini) }
                            Text(isLoading ? "Loading & Warming…" : "Load")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .controlSize(.small)
                    .disabled(isLoading || active == nil)
                } else {
                    Button {
                        Task { await viewModel.chat.performLoadOrReloadActiveModel(warm: true, reloadAnyway: true) }
                    } label: {
                        HStack(spacing: 6) {
                            if isLoading { ProgressView().controlSize(.mini) }
                            Text(isLoading ? "Reloading…" : (viewModel.chat.canReloadActiveModel ? "Reload (Apply Changes)" : "Reload"))
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(viewModel.chat.canReloadActiveModel ? .orange : .accentColor)
                    .controlSize(.small)
                    .disabled(isLoading)

                    if let active {
                        Button("Unload") {
                            Task {
                                await viewModel.toggleModelLoaded(active)
                                await viewModel.chat.refreshActiveAdminModel(resetDrafts: true)
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(isLoading)
                    }
                }

                if let status = viewModel.chat.loadPanelStatusMessage {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(viewModel.chat.loadPanelStatusIsError ? .red : .green)
                        .lineLimit(2)
                }
                Spacer()
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color.secondary.opacity(contrast.isIncreased ? 0.08 : 0.04))
    }
}

private struct CompactChatControlsSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Binding var showSystemPromptSheet: Bool
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("MODEL")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Picker(
                            "Model",
                            selection: Binding(
                                get: { viewModel.selectedModelID },
                                set: { viewModel.selectGenerationModel($0) }
                            )
                        ) {
                            ForEach(viewModel.generationModels, id: \.modelID) { model in
                                let dot = model.isLoaded ? "●" : "○"
                                Text("\(dot) \(model.modelID) (\(model.provider.displayName))").tag(model.modelID)
                            }
                        }
                        .labelsHidden()
                    }

                    ModelLoadPanelView(viewModel: viewModel)
                        .clipShape(RoundedRectangle(cornerRadius: 10))

                    HStack(spacing: 10) {
                        Button {
                            dismiss()
                            showSystemPromptSheet = true
                        } label: {
                            Label("Edit System Prompt", systemImage: "terminal")
                                .frame(maxWidth: .infinity, minHeight: 44)
                        }
                        .buttonStyle(.bordered)

                        Button {
                            viewModel.showPromptPreview.toggle()
                            dismiss()
                        } label: {
                            Label(viewModel.showPromptPreview ? "Hide Preview" : "Prompt Preview", systemImage: "eye")
                                .frame(maxWidth: .infinity, minHeight: 44)
                        }
                        .buttonStyle(.bordered)
                    }

                    Divider()

                    LoadAndSamplerInspectorView(viewModel: viewModel)
                }
                .padding(16)
            }
            .navigationTitle("Chat Controls")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

private struct ChatPromptPreviewDrawer: View {
    let viewModel: HeylookWorkbenchViewModel

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
                Text("\(viewModel.promptPreviewText.count) chars")
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
        guard !raw.isEmpty, raw.utf8.count <= 32_000, !markers.isEmpty else {
            return AttributedString(raw)
        }
        var attr = AttributedString(raw)
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
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast
    let thinkingText: String
    let isStreaming: Bool
    let canEndThinking: Bool
    let isEndingThinking: Bool
    var onEndThinking: (() -> Void)?
    @Binding var isExpanded: Bool

    private var contrast: AdaptiveContrastStyle {
        AdaptiveContrastStyle(colorSchemeContrast)
    }

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
                            .fill(Color.purple.opacity(contrast.isIncreased ? 0.12 : 0.06))
                    )
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.purple.opacity(contrast.isIncreased ? 0.10 : 0.05))
        )
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(contrast.borderColor(.purple), lineWidth: contrast.borderWidth)
        }
    }
}

// MARK: - Message Bubble & Turn Actions

private struct MessageBubbleView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let message: MessageRecord
    let isLatestAssistant: Bool
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast

    @State private var showThinking: Bool = false
    @State private var hasInitializedThinkingDisclosure: Bool = false
    @State private var showRawSource: Bool = false
    @State private var didCopyTurn: Bool = false
    @State private var showEditorMediaImporter: Bool = false

    private var contrast: AdaptiveContrastStyle {
        AdaptiveContrastStyle(colorSchemeContrast)
    }

    private var isUser: Bool { message.role == "user" }
    private var isEditing: Bool { viewModel.editingMessageID == message.id }
    private var turnAttachments: [StagedChatAttachment] {
        viewModel.messageAttachmentsByID[message.id] ?? []
    }

    private var peakMemoryGB: Double? {
        message.stats?.peakMemoryGB ?? message.peakMemoryGB
    }

    private var cacheReport: CacheReport? {
        message.cacheReport ?? message.stats?.cache
    }

    private var specReport: SpeculativeReport? {
        message.speculativeReport ?? message.stats?.speculative
    }

    var body: some View {
        let isGenerating = viewModel.chat.isSelectedConversationGenerating
        let canContinueUserTurn = viewModel.chat.canContinueEditedUserTurn
        let continueDisabledForGGUF = isUser && !canContinueUserTurn

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
                        showThinking = message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    }
                }
            }

            // In-Session Multimodal Attachments Gallery
            if !isEditing && !turnAttachments.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(turnAttachments) { att in
                            TurnAttachmentPreviewCard(attachment: att)
                        }
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
                .background(Capsule().fill(Color.orange.opacity(contrast.bannerFillOpacity)))
                .overlay {
                    Capsule().strokeBorder(contrast.subtleBorderColor(.orange), lineWidth: contrast.borderWidth)
                }
            }

            // Message Body or Non-Destructive Multi-Verb Editor
            if isEditing {
                let canEditTurnMedia = viewModel.chat.canEditMedia(for: message)
                let editList = viewModel.chat.editingAttachments

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
                                        .strokeBorder(contrast.borderColor(.purple), lineWidth: max(1, contrast.borderWidth))
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
                                    .strokeBorder(contrast.borderColor(.accentColor), lineWidth: max(1, contrast.borderWidth))
                            )
                    }

                    // Editable / Reorderable Turn Attachments
                    if !editList.isEmpty || canEditTurnMedia {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack(spacing: 8) {
                                Label("Turn Attachments (\(editList.count))", systemImage: "photo.on.rectangle")
                                    .font(.caption2.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                if canEditTurnMedia {
                                    Button {
                                        showEditorMediaImporter = true
                                    } label: {
                                        Label("Add media", systemImage: "plus.circle")
                                            .font(.caption2.weight(.medium))
                                    }
                                    .buttonStyle(.bordered)
                                    .controlSize(.mini)
                                }
                            }

                            if !editList.isEmpty {
                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 8) {
                                        ForEach(Array(editList.enumerated()), id: \.element.id) { idx, item in
                                            HStack(spacing: 6) {
                                                if item.kind == .image, let raw = item.previewData,
                                                   let cg = EditorThumbnailDecoder.decode(from: raw) {
                                                    Image(decorative: cg, scale: 1.0)
                                                        .resizable()
                                                        .scaledToFill()
                                                        .frame(width: 28, height: 28)
                                                        .clipShape(RoundedRectangle(cornerRadius: 4))
                                                } else {
                                                    Image(systemName: item.kind == .image ? "photo" : "waveform")
                                                        .font(.caption)
                                                        .foregroundStyle(.secondary)
                                                }
                                                Text(item.filename)
                                                    .font(.caption2)
                                                    .lineLimit(1)
                                                    .frame(maxWidth: 110, alignment: .leading)
                                                if editList.count > 1 {
                                                    Button {
                                                        viewModel.chat.moveEditingAttachment(id: item.id, delta: -1)
                                                    } label: {
                                                        Image(systemName: "chevron.left")
                                                            .font(.system(size: 9, weight: .bold))
                                                    }
                                                    .buttonStyle(.plain)
                                                    .disabled(idx == 0)

                                                    Button {
                                                        viewModel.chat.moveEditingAttachment(id: item.id, delta: 1)
                                                    } label: {
                                                        Image(systemName: "chevron.right")
                                                            .font(.system(size: 9, weight: .bold))
                                                    }
                                                    .buttonStyle(.plain)
                                                    .disabled(idx == editList.count - 1)
                                                }
                                                Button {
                                                    viewModel.chat.removeEditingAttachment(id: item.id)
                                                } label: {
                                                    Image(systemName: "xmark.circle.fill")
                                                        .font(.caption2)
                                                        .foregroundStyle(.secondary)
                                                }
                                                .buttonStyle(.plain)
                                            }
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 5)
                                            .background(
                                                RoundedRectangle(cornerRadius: 7)
                                                    .fill(Color.secondary.opacity(contrast.chipFillOpacity))
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // In-Editor Prompt Preview Output
                    if viewModel.chat.isRenderingEditorPreview
                        || viewModel.chat.editorPreviewResponse != nil
                        || viewModel.chat.editorPreviewError != nil {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 6) {
                                Text("EDITOR PROMPT PREVIEW")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(.secondary)
                                if viewModel.chat.isRenderingEditorPreview {
                                    ProgressView().controlSize(.mini)
                                }
                                if let resp = viewModel.chat.editorPreviewResponse {
                                    Text("• \(ChatController.formatPromptPreviewSummary(resp))")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                            }
                            if let err = viewModel.chat.editorPreviewError {
                                Text(err)
                                    .font(.caption2)
                                    .foregroundStyle(.orange)
                            } else if let resp = viewModel.chat.editorPreviewResponse {
                                let spans = ChatController.computeMarkerSpans(in: resp.prompt, markers: resp.markers)
                                ScrollView {
                                    Text(PromptSyntaxHighlightHelper.highlight(resp.prompt, markers: spans))
                                        .font(.system(size: 11, design: .monospaced))
                                        .textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .padding(6)
                                }
                                .frame(maxHeight: 130)
                                .background(
                                    RoundedRectangle(cornerRadius: 6)
                                        .fill(Color.secondary.opacity(contrast.chipFillOpacity))
                                )
                            }
                        }
                    }

                    HStack(spacing: 8) {
                        Button("Cancel") {
                            viewModel.cancelEditingMessage()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)

                        Button {
                            Task { await viewModel.chat.previewEditorPrompt(for: message) }
                        } label: {
                            Label("Preview prompt", systemImage: "eye")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(viewModel.chat.isRenderingEditorPreview)

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
                        .disabled(isGenerating || continueDisabledForGGUF)
                        .help(
                            continueDisabledForGGUF
                                ? "GGUF llama-server cannot continue from a user turn — use Save & Regenerate"
                                : (isUser ? "Keep up to this user turn and generate a new assistant reply" : "Prefill this edited assistant response and continue generating")
                        )

                        Button("Save & Regenerate") {
                            Task { await viewModel.saveAndRegenerateEditedMessage(message) }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(isGenerating)
                    }
                }
                .frame(maxWidth: 720)
                .fileImporter(
                    isPresented: $showEditorMediaImporter,
                    allowedContentTypes: [.image, .audio],
                    allowsMultipleSelection: true
                ) { result in
                    if case .success(let urls) = result {
                        viewModel.chat.addMediaToEditingMessage(from: urls, message: message)
                    }
                }
            } else if !message.content.isEmpty || isUser {
                Group {
                    if showRawSource || isUser {
                        Text(message.content.isEmpty ? " " : message.content)
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
                        .fill(isUser ? Color.accentColor.opacity(contrast.chipFillOpacity) : Color.secondary.opacity(contrast.isIncreased ? 0.14 : 0.08))
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(contrast.subtleBorderColor(isUser ? .accentColor : .secondary), lineWidth: contrast.borderWidth)
                }
                .frame(maxWidth: 760, alignment: isUser ? .trailing : .leading)
            }

            // Footer Telemetry + Rich Turn Actions
            HStack(spacing: 8) {
                if !isUser, let msgModel = message.modelID, !msgModel.isEmpty {
                    Text(msgModel)
                        .font(.system(size: 9.5, weight: .medium, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.secondary.opacity(contrast.chipFillOpacity)))
                        .foregroundStyle(.secondary)
                }
                if let tps = message.generationTPS ?? message.stats?.generationTPS, tps > 0 {
                    Text(String(format: "%.1f tok/s", tps))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                if let count = message.tokenCount ?? message.stats?.outputTokens, count > 0 {
                    Text("\(count) tok")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
                if let ttft = message.ttftMs ?? message.stats?.ttftMs, ttft > 0 {
                    Text(String(format: "TTFT %.0fms", ttft))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
                if let peak = peakMemoryGB, peak > 0 {
                    Text(String(format: "%.2f GB peak", peak))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
                if let placement = message.computePlacement {
                    Text(placement.rawValue)
                        .font(.system(size: 9.5, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
                if let cache = cacheReport {
                    let isReused = cache.outcome == .reused
                    Text(cache.summaryBadge)
                        .font(.system(size: 10, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Capsule().fill((isReused ? Color.green : Color.secondary).opacity(contrast.chipFillOpacity)))
                        .foregroundStyle(isReused ? .green : .secondary)
                        .help(cache.reason ?? cache.humanReadableCause ?? cache.outcome.rawValue)
                }
                if let spec = specReport {
                    let pct = Int(((spec.draftedTokens > 0 ? spec.draftShare : spec.acceptanceRate) * 100.0).rounded())
                    let specText = spec.draftedTokens > 0
                        ? "draft \(pct)% (\(spec.acceptedTokens)/\(spec.draftedTokens) accepted)"
                        : spec.summaryBadge
                    Text(specText)
                        .font(.system(size: 10, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.blue.opacity(contrast.chipFillOpacity)))
                        .foregroundStyle(.blue)
                        .help("Drafter: \(spec.drafterID) • \(spec.emittedTokens) emitted")
                }

                // Turn Action Buttons
                if !isEditing {
                    let minBtnHeight: CGFloat = horizontalSizeClass == .compact ? 36 : 20
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
                                .frame(minHeight: minBtnHeight)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(didCopyTurn ? .green : .secondary)

                        Button {
                            viewModel.beginEditingMessage(message)
                        } label: {
                            Label("Edit", systemImage: "pencil")
                                .font(.caption2)
                                .frame(minHeight: minBtnHeight)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .disabled(isGenerating)

                        Button {
                            viewModel.cloneConversation(
                                id: message.conversationID,
                                upToPosition: message.position
                            )
                        } label: {
                            Label("Fork", systemImage: "arrow.branch")
                                .font(.caption2)
                                .frame(minHeight: minBtnHeight)
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
                                    .frame(minHeight: minBtnHeight)
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)

                            Button {
                                Task { await viewModel.regenerateAssistantMessage(message) }
                            } label: {
                                Label("Regenerate", systemImage: "arrow.clockwise")
                                    .font(.caption2)
                                    .frame(minHeight: minBtnHeight)
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)
                            .disabled(isGenerating)

                            if isLatestAssistant {
                                Button {
                                    Task { await viewModel.continueAssistantMessage(message) }
                                } label: {
                                    Label("Continue", systemImage: "text.append")
                                        .font(.caption2)
                                        .frame(minHeight: minBtnHeight)
                                }
                                .buttonStyle(.plain)
                                .foregroundStyle(.secondary)
                                .disabled(isGenerating)
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
                                .frame(minWidth: minBtnHeight, minHeight: minBtnHeight)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .disabled(isGenerating)
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

private enum EditorThumbnailDecoder {
    static func decode(from data: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}

private enum PromptSyntaxHighlightHelper {
    static func highlight(_ raw: String, markers: [PromptMarkerSpan]) -> AttributedString {
        guard !raw.isEmpty, raw.utf8.count <= 32_000, !markers.isEmpty else {
            return AttributedString(raw)
        }
        var attr = AttributedString(raw)
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
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast
    @State private var isThinkingExpanded: Bool = true

    private var contrast: AdaptiveContrastStyle {
        AdaptiveContrastStyle(colorSchemeContrast)
    }

    var body: some View {
        let prefillProcessed = viewModel.chat.prefillProcessed
        let prefillTotal = viewModel.chat.prefillTotal

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

            if prefillTotal > 0 && viewModel.streamingThinking.isEmpty && viewModel.streamingContent.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: Double(min(prefillProcessed, prefillTotal)), total: Double(prefillTotal))
                        .progressViewStyle(.linear)
                    Text("Prefilling \(prefillProcessed) / \(prefillTotal) tokens")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .frame(maxWidth: 420, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.secondary.opacity(contrast.isIncreased ? 0.14 : 0.07))
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(contrast.subtleBorderColor(.secondary), lineWidth: contrast.borderWidth)
                }
            } else if let statusLine = viewModel.streamStatusLine,
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
                .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(contrast.isIncreased ? 0.14 : 0.07)))
            }

            if !viewModel.streamingThinking.isEmpty {
                let canEnd = viewModel.isGenerating
                    && viewModel.streamingContent.isEmpty
                    && viewModel.chat.activeModelCanEndThinking
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
                            .fill(Color.secondary.opacity(contrast.isIncreased ? 0.14 : 0.08))
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
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast

    private var isCompact: Bool { horizontalSizeClass == .compact }
    private var contrast: AdaptiveContrastStyle { AdaptiveContrastStyle(colorSchemeContrast) }

    var body: some View {
        let capabilityWarning = viewModel.chatAttachmentCapabilityWarning
        let schemaError = viewModel.responseFormatValidationError
        let hasPendingSwitch = viewModel.pendingModelSwitchTargetID != nil
        let isGenerating = viewModel.isGenerating || viewModel.chat.isSelectedConversationGenerating
        let canSend = (!viewModel.draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !viewModel.stagedAttachments.isEmpty)
            && (capabilityWarning == nil || hasPendingSwitch)
            && schemaError == nil
            && !isGenerating
        let touchSize: CGFloat = isCompact ? 44 : 30

        VStack(alignment: .leading, spacing: 6) {
            if !viewModel.stagedAttachments.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(viewModel.stagedAttachments) { att in
                            StagedComposerAttachmentPill(
                                attachment: att,
                                serverPlan: viewModel.chat.stagedImagePlansByID[att.id],
                                onToggleShrink: { viewModel.toggleAttachmentShrink(id: att.id) },
                                onRemove: { viewModel.removeDraftAttachment(id: att.id) }
                            )
                        }

                        let totalImgTokens = viewModel.totalStagedImageTokens
                        if viewModel.chat.hasExactTotalStagedImageTokens && totalImgTokens > 0 {
                            Text("Σ \(totalImgTokens) img tok")
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .background(
                                    Capsule().fill(Color.secondary.opacity(contrast.chipFillOpacity))
                                )
                                .foregroundStyle(.secondary)
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

            HStack(alignment: .bottom, spacing: 8) {
                Button {
                    showAttachmentImporter = true
                } label: {
                    Image(systemName: "paperclip")
                        .font(.system(size: 15, weight: .medium))
                        .frame(minWidth: touchSize - 10, minHeight: touchSize - 10)
                }
                .buttonStyle(.bordered)
                .help("Attach Image or Audio File (max 8 images / 2 audio)")

                if viewModel.activeModelSupportsThinking && viewModel.activeModelCanDisableThinking {
                    let thinkOn = viewModel.effectiveThinkingEnabled
                    Button {
                        if thinkOn {
                            viewModel.thinkingEnabledOverride = false
                            viewModel.thinkingDepthOverride = nil
                        } else {
                            viewModel.thinkingEnabledOverride = true
                        }
                    } label: {
                        Image(systemName: thinkOn ? "brain.filled.head.profile" : "brain.head.profile")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundStyle(thinkOn ? .purple : .secondary)
                            .frame(minWidth: touchSize - 10, minHeight: touchSize - 10)
                    }
                    .buttonStyle(.bordered)
                    .tint(thinkOn ? .purple : nil)
                    .help(thinkOn ? "Thinking enabled (click to turn Off)" : "Thinking Off (click to enable)")
                }

                #if os(macOS)
                Button {
                    if !viewModel.pasteAttachmentFromClipboard() {
                        viewModel.statusBanner = "Clipboard does not contain a compatible image or media file."
                    }
                } label: {
                    Image(systemName: "doc.on.clipboard")
                        .font(.system(size: 14, weight: .medium))
                        .frame(minWidth: touchSize - 10, minHeight: touchSize - 10)
                }
                .buttonStyle(.bordered)
                .help("Paste Screenshot or Image from Clipboard (⌘⇧V)")
                .keyboardShortcut("v", modifiers: [.command, .shift])
                #elseif canImport(UIKit)
                if UIPasteboard.general.hasImages {
                    Button {
                        if !viewModel.chat.pasteAttachmentFromIOSClipboard() {
                            viewModel.statusBanner = "Clipboard does not contain a compatible image."
                        }
                    } label: {
                        Image(systemName: "doc.on.clipboard")
                            .font(.system(size: 14, weight: .medium))
                            .frame(minWidth: touchSize - 10, minHeight: touchSize - 10)
                    }
                    .buttonStyle(.bordered)
                    .help("Paste image from clipboard")
                }
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
                            .frame(minWidth: touchSize - 10, minHeight: touchSize - 10)
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
                            .fill(Color.secondary.opacity(contrast.isIncreased ? 0.14 : 0.08))
                    )
                    .overlay {
                        RoundedRectangle(cornerRadius: 10)
                            .strokeBorder(contrast.subtleBorderColor(.secondary), lineWidth: contrast.borderWidth)
                    }
                    .onSubmit {
                        guard canSend else { return }
                        Task { await viewModel.sendChatPrompt() }
                    }

                if isGenerating {
                    Button(role: .destructive) {
                        viewModel.cancelGeneration()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                            .frame(minHeight: isCompact ? 34 : nil)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .help("Cancel active generation (Esc)")
                } else {
                    Button {
                        Task { await viewModel.sendChatPrompt() }
                    } label: {
                        Label("Send", systemImage: "arrow.up.circle.fill")
                            .frame(minHeight: isCompact ? 34 : nil)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canSend)
                    .keyboardShortcut(.return, modifiers: [.command])
                }
            }
        }
        .padding(.horizontal, isCompact ? 12 : 16)
        .padding(.vertical, 10)
    }
}

private struct StagedComposerAttachmentPill: View {
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast
    let attachment: StagedChatAttachment
    let serverPlan: StagedImagePlanState?
    let onToggleShrink: () -> Void
    let onRemove: () -> Void

    private var contrast: AdaptiveContrastStyle { AdaptiveContrastStyle(colorSchemeContrast) }

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
                if let plan = serverPlan {
                    if plan.isExactServerPlan {
                        Text("\(plan.tokens) tok • \(plan.currentWidth)×\(plan.currentHeight)")
                            .font(.system(size: 9, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.secondary)
                    } else if let note = plan.note {
                        Text(note)
                            .font(.system(size: 9, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                } else if let dims = attachment.dimensionsLabel {
                    Text(dims)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }

            if let actionLabel = serverPlan?.actionLabel {
                Button(action: onToggleShrink) {
                    Text(actionLabel)
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            Capsule().fill(
                                (actionLabel == "Full res" ? Color.green : Color.orange).opacity(contrast.chipFillOpacity)
                            )
                        )
                        .foregroundStyle(actionLabel == "Full res" ? .green : .orange)
                }
                .buttonStyle(.plain)
                .help("Resize client-side to model's exact recommended target resolution or restore original")
            } else if attachment.canToggleShrink {
                Button(action: onToggleShrink) {
                    Text(attachment.useShrunk2048 ? "≤2048px" : "Full res")
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            Capsule().fill(
                                (attachment.useShrunk2048 ? Color.green : Color.orange).opacity(contrast.chipFillOpacity)
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
                .fill(Color.accentColor.opacity(contrast.chipFillOpacity))
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
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @FocusState private var isEditorFocused: Bool
    @State private var isClearArmed: Bool = false

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("System Prompt (persisted per conversation; included in Jinja prompt preview)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if !viewModel.draftSystemPrompt.isEmpty {
                        Button(isClearArmed ? "Clear prompt?" : "Clear", role: isClearArmed ? .destructive : nil) {
                            if isClearArmed {
                                viewModel.draftSystemPrompt = ""
                                isClearArmed = false
                            } else {
                                isClearArmed = true
                            }
                        }
                        .buttonStyle(.bordered)
                        .tint(isClearArmed ? .red : nil)
                        .controlSize(.mini)
                    }
                }

                TextEditor(text: $viewModel.draftSystemPrompt)
                    .font(.system(size: 13, design: .monospaced))
                    .focused($isEditorFocused)
                    .frame(minHeight: 180)
                    .padding(6)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.secondary.opacity(0.08))
                    )
                    .onChange(of: viewModel.draftSystemPrompt) { _, _ in
                        isClearArmed = false
                    }
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
        .frame(
            minWidth: horizontalSizeClass == .compact ? nil : 480,
            minHeight: horizontalSizeClass == .compact ? nil : 300
        )
        .onAppear {
            isEditorFocused = true
        }
    }
}

// MARK: - Complete Optional 5-Layer Sampler Cascade & Presets Inspector

public struct LoadAndSamplerInspectorView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast
    @State private var newPresetName: String = ""
    @State private var showSaveAsField: Bool = false
    @State private var isOverwriteArmed: Bool = false
    @State private var isCustomDepthActive: Bool = false

    public init(viewModel: HeylookWorkbenchViewModel) {
        self.viewModel = viewModel
    }

    private var contrast: AdaptiveContrastStyle {
        AdaptiveContrastStyle(colorSchemeContrast)
    }

    public var body: some View {
        @Bindable var settings = viewModel.activeDocumentSettings
        let resolved = settings.currentResolvedSamplers
        let drift = settings.driftReport()
        let activeModel = viewModel.activeModelDescription
        let adminThinking = viewModel.chat.activeAdminRow?.engine.thinking
        let controls = activeModel?.thinkingControls

        let supportsThinking = settings.activeModelSupportsThinking
        let canDisableThinking = settings.activeModelCanDisableThinking
        let defaultEnabled = controls?.defaultEnabled ?? viewModel.chat.activeAdminRow?.thinkingDefault ?? true
        let rawDepthValues = adminThinking?.depth?.values ?? controls?.depthValues ?? []
        let depthOffSet = Set(adminThinking?.depth?.off ?? controls?.depthOffValues ?? [])
        let depthChoices = rawDepthValues.filter { !depthOffSet.contains($0) }
        let depthAliases = adminThinking?.depth?.aliases ?? controls?.depthAliases ?? [:]
        let depthDefault = adminThinking?.depth?.defaultValue ?? controls?.depthDefault
        let depthUnknown = adminThinking?.depth?.unknown ?? controls?.depthUnknown
        let depthChangesPrefix = (adminThinking?.depth?.changesPrefix ?? controls?.depthChangesPrefix) == true
        let rawBudgetEnforced = adminThinking?.budget?.enforced ?? controls?.budgetEnforced
        let budgetEnforced = viewModel.chat.activeModelCanEndThinking && rawBudgetEnforced != false

        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let disclosure = settings.reconcileDisclosureMessage {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "info.circle.fill")
                            .foregroundStyle(.orange)
                            .font(.caption)
                        Text(disclosure)
                            .font(.caption2)
                            .foregroundStyle(.primary)
                        Spacer()
                        Button {
                            settings.reconcileDisclosureMessage = nil
                        } label: {
                            Image(systemName: "xmark")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(8)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.orange.opacity(contrast.bannerFillOpacity))
                    )
                    .overlay {
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(contrast.borderColor(.orange), lineWidth: contrast.borderWidth)
                    }
                }

                if let replacement = settings.pendingPromptReplacement {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Replace system prompt with \"\(replacement.preset.name)\"?")
                            .font(.caption.weight(.semibold))
                        Text("The current document has an edited system prompt.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        HStack(spacing: 8) {
                            Button("Cancel") {
                                settings.cancelPendingPromptReplacement()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                            Button("Keep Prompt") {
                                settings.confirmPendingPromptReplacement(replacePrompt: false)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                            Button("Replace") {
                                settings.confirmPendingPromptReplacement(replacePrompt: true)
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.mini)
                        }
                    }
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.accentColor.opacity(contrast.bannerFillOpacity))
                    )
                    .overlay {
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(contrast.borderColor(.accentColor), lineWidth: contrast.borderWidth)
                    }
                }

                // 1. Two-Verb Preset Manager (Apply / Save / Armed Overwrite / Confirmed Delete)
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("SAMPLER PRESET")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer()
                        if settings.isPresetModified {
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
                        "Apply preset",
                        selection: Binding(
                            get: { settings.selectedPresetName ?? "__none__" },
                            set: { newVal in
                                isOverwriteArmed = false
                                if newVal == "__none__" {
                                    settings.clearPresetSelection()
                                } else if let p = viewModel.presets.first(where: { $0.name == newVal }) {
                                    _ = settings.applyPreset(p)
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

                    if !drift.provenanceSummary.isEmpty {
                        Text(drift.provenanceSummary)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }

                    HStack(spacing: 6) {
                        if let activeName = settings.selectedPresetName {
                            Button {
                                if isOverwriteArmed {
                                    _ = settings.updateActivePreset()
                                    isOverwriteArmed = false
                                } else {
                                    isOverwriteArmed = true
                                }
                            } label: {
                                Label(
                                    isOverwriteArmed ? "Confirm overwrite?" : "Overwrite \"\(activeName)\"",
                                    systemImage: "square.and.arrow.down"
                                )
                                .lineLimit(1)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(isOverwriteArmed ? .orange : .accentColor)
                            .controlSize(.small)
                            .disabled(!drift.hasChanges)
                        }

                        Button {
                            withAnimation(.easeInOut(duration: 0.15)) {
                                showSaveAsField.toggle()
                                isOverwriteArmed = false
                            }
                        } label: {
                            Label("Save…", systemImage: "plus.square.on.square")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)

                        if settings.selectedPresetName != nil {
                            Button(role: .destructive) {
                                settings.requestDeleteActivePreset()
                            } label: {
                                Text("Delete…")
                                    .font(.caption2.weight(.medium))
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .help("Delete selected preset (requires confirmation)")
                        }
                    }

                    if showSaveAsField {
                        HStack(spacing: 6) {
                            TextField("Preset name…", text: $newPresetName)
                                .textFieldStyle(.roundedBorder)
                                .font(.caption)
                                .onSubmit {
                                    commitSaveAs(settings: settings)
                                }
                            Button("Save") {
                                commitSaveAs(settings: settings)
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                            .disabled(newPresetName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                    }
                }

                Divider()

                // 2. Unified Thinking Control (single switch + depth + verbatim + budget)
                if supportsThinking {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("THINKING")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        let currentChoiceTag: String = {
                            if settings.thinkingEnabledOverride == false {
                                return "__off__"
                            }
                            if let depth = settings.thinkingDepthOverride, !depth.isEmpty {
                                if depthChoices.contains(depth) {
                                    return "level:\(depth)"
                                }
                                if depthUnknown == "verbatim" {
                                    return "__verbatim__"
                                }
                            }
                            if settings.thinkingEnabledOverride == true && depthChoices.isEmpty && !defaultEnabled {
                                return "__on__"
                            }
                            return "__default__"
                        }()

                        Picker(
                            "Thinking mode",
                            selection: Binding<String>(
                                get: { currentChoiceTag },
                                set: { tag in
                                    isOverwriteArmed = false
                                    if tag == "__default__" {
                                        isCustomDepthActive = false
                                        settings.thinkingEnabledOverride = nil
                                        settings.thinkingDepthOverride = nil
                                    } else if tag == "__off__" {
                                        isCustomDepthActive = false
                                        settings.thinkingEnabledOverride = false
                                        settings.thinkingDepthOverride = nil
                                    } else if tag == "__on__" {
                                        isCustomDepthActive = false
                                        settings.thinkingEnabledOverride = true
                                        settings.thinkingDepthOverride = nil
                                    } else if tag == "__verbatim__" {
                                        isCustomDepthActive = true
                                        settings.thinkingEnabledOverride = true
                                        if settings.thinkingDepthOverride == nil {
                                            settings.thinkingDepthOverride = depthChoices.first ?? depthDefault ?? ""
                                        }
                                    } else if tag.hasPrefix("level:") {
                                        isCustomDepthActive = false
                                        let lvl = String(tag.dropFirst("level:".count))
                                        settings.thinkingEnabledOverride = true
                                        settings.thinkingDepthOverride = lvl
                                    }
                                }
                            )
                        ) {
                            let defDetail = depthDefault.map { ", \($0)" } ?? ""
                            Text("Model default (\(defaultEnabled ? "on" : "off")\(defDetail))").tag("__default__")
                            if canDisableThinking {
                                Text("Off").tag("__off__")
                            }
                            if depthChoices.isEmpty {
                                if !defaultEnabled {
                                    Text("On").tag("__on__")
                                }
                            } else {
                                ForEach(depthChoices, id: \.self) { depth in
                                    let aliasNote = depthAliases.first(where: { $0.value == depth })?.key
                                    let label = aliasNote.map { "\(depth.capitalized) (\($0))" } ?? depth.capitalized
                                    Text(label).tag("level:\(depth)")
                                }
                            }
                            if depthUnknown == "verbatim" {
                                Text("Custom verbatim level…").tag("__verbatim__")
                            }
                        }
                        .labelsHidden()

                        if depthUnknown == "verbatim" && (isCustomDepthActive || currentChoiceTag == "__verbatim__") {
                            VStack(alignment: .leading, spacing: 4) {
                                TextField(
                                    "Custom level",
                                    text: Binding(
                                        get: { settings.thinkingDepthOverride ?? "" },
                                        set: { raw in
                                            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                                            settings.thinkingEnabledOverride = true
                                            settings.thinkingDepthOverride = trimmed.isEmpty ? nil : trimmed
                                        }
                                    )
                                )
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 11.5, design: .monospaced))

                                if !depthChoices.isEmpty {
                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 4) {
                                            ForEach(depthChoices, id: \.self) { suggestion in
                                                Button(suggestion) {
                                                    settings.thinkingEnabledOverride = true
                                                    settings.thinkingDepthOverride = suggestion
                                                }
                                                .buttonStyle(.bordered)
                                                .controlSize(.mini)
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if depthChangesPrefix && settings.thinkingDepthOverride != nil {
                            Text("Selected level alters prompt prefix (KV cache resets on change).")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }

                        // Indented thinking_budget_tokens hidden when thinking is Off or !budgetEnforced.
                        let isThinkingEffectiveOn = settings.effectiveThinkingEnabled
                        if isThinkingEffectiveOn && budgetEnforced {
                            OptionalNullableIntFieldControl(
                                title: "Thinking Token Budget",
                                placeholder: "Unlimited",
                                overrideValue: $settings.thinkingBudgetOverride,
                                effectiveValue: resolved.effective.thinkingBudget,
                                provenance: resolved.provenanceByField[.thinkingBudget] ?? .derived
                            )
                            .padding(.leading, 14)
                            if rawBudgetEnforced == nil {
                                Text("May not be enforced on this engine.")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .padding(.leading, 14)
                            }
                        } else if isThinkingEffectiveOn, let reason = viewModel.chat.activeModelBudgetDisabledReason {
                            Text("Token budget unavailable: \(reason)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .padding(.leading, 14)
                        }
                    }

                    Divider()
                }

                // 3. Core Entropy & Nucleus Cascade Controls
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("SAMPLING CASCADE")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer()
                        if settings.hasAnyManualSamplerOverride {
                            Button("Reset All") {
                                isOverwriteArmed = false
                                settings.resetManualSamplerOverrides()
                            }
                            .font(.caption2)
                            .buttonStyle(.plain)
                            .foregroundStyle(Color.accentColor)
                        }
                    }

                    OptionalDoubleSliderControl(
                        title: "Temperature",
                        overrideValue: $settings.temperatureOverride,
                        effectiveValue: resolved.effective.temperature ?? 0.7,
                        range: 0.0...2.0,
                        step: 0.05,
                        format: "%.2f",
                        provenance: resolved.provenanceByField[.temperature] ?? .derived
                    )

                    OptionalDoubleSliderControl(
                        title: "Top P",
                        overrideValue: $settings.topPOverride,
                        effectiveValue: resolved.effective.topP ?? 0.95,
                        range: 0.05...1.0,
                        step: 0.01,
                        format: "%.2f",
                        provenance: resolved.provenanceByField[.topP] ?? .derived
                    )

                    OptionalDoubleSliderControl(
                        title: "Min P",
                        overrideValue: $settings.minPOverride,
                        effectiveValue: resolved.effective.minP ?? 0.0,
                        range: 0.0...0.5,
                        step: 0.01,
                        format: "%.2f",
                        provenance: resolved.provenanceByField[.minP] ?? .derived
                    )

                    OptionalIntStepperControl(
                        title: "Top K",
                        overrideValue: $settings.topKOverride,
                        effectiveValue: resolved.effective.topK ?? 40,
                        range: 0...200,
                        step: 5,
                        provenance: resolved.provenanceByField[.topK] ?? .derived
                    )

                    OptionalIntStepperControl(
                        title: "Max Output Tokens",
                        overrideValue: $settings.maxTokensOverride,
                        effectiveValue: resolved.effective.maxTokens ?? 2048,
                        range: 64...32768,
                        step: 256,
                        provenance: resolved.provenanceByField[.maxTokens] ?? .derived
                    )
                }

                Divider()

                // 4. Penalties, Seed & Stop Sequences
                VStack(alignment: .leading, spacing: 12) {
                    Text("PENALTIES & DETERMINISM")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)

                    OptionalDoubleSliderControl(
                        title: "Repetition Penalty",
                        overrideValue: $settings.repetitionPenaltyOverride,
                        effectiveValue: resolved.effective.repetitionPenalty ?? 1.0,
                        range: 1.0...1.5,
                        step: 0.02,
                        format: "%.2f",
                        provenance: resolved.provenanceByField[.repetitionPenalty] ?? .derived
                    )

                    OptionalIntStepperControl(
                        title: "Repetition Window",
                        overrideValue: $settings.repetitionContextSizeOverride,
                        effectiveValue: resolved.effective.repetitionContextSize ?? 64,
                        range: 0...512,
                        step: 16,
                        provenance: resolved.provenanceByField[.repetitionContextSize] ?? .derived
                    )

                    OptionalDoubleSliderControl(
                        title: "Presence Penalty",
                        overrideValue: $settings.presencePenaltyOverride,
                        effectiveValue: resolved.effective.presencePenalty ?? 0.0,
                        range: 0.0...2.0,
                        step: 0.05,
                        format: "%.2f",
                        provenance: resolved.provenanceByField[.presencePenalty] ?? .derived
                    )

                    OptionalNullableIntFieldControl(
                        title: "Random Seed",
                        placeholder: "Random",
                        overrideValue: $settings.seedOverride,
                        effectiveValue: resolved.effective.seed,
                        provenance: resolved.provenanceByField[.seed] ?? .derived
                    )

                    StopSequencesEditorControl(
                        overrideValue: $settings.stopSequencesOverride,
                        effectiveValue: resolved.effective.stopSequences ?? [],
                        provenance: resolved.provenanceByField[.stopSequences] ?? .derived
                    )
                }

                Divider()

                // 5. Context-Aware System Prompt (Conversation vs Notebook via activeDocumentSettings)
                VStack(alignment: .leading, spacing: 6) {
                    let isNotebook = viewModel.activeSurface == .notebook
                    let promptBinding: Binding<String> = isNotebook
                        ? $viewModel.notebookSystemPrompt
                        : $settings.systemPrompt
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
                                .fill(Color.secondary.opacity(contrast.isIncreased ? 0.14 : 0.08))
                        )
                        .overlay {
                            RoundedRectangle(cornerRadius: 8)
                                .strokeBorder(contrast.subtleBorderColor(.secondary), lineWidth: contrast.borderWidth)
                        }
                }

                Divider()

                // 6. Structured Output (`response_format`: text / json_object / json_schema)
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
                                    .fill(Color.secondary.opacity(contrast.isIncreased ? 0.14 : 0.08))
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

                // 7. Active Model Provenance Card + Live RAM Fit
                if let active = activeModel {
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

    private func commitSaveAs(settings: DocumentSettingsController) {
        let trimmed = newPresetName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        _ = settings.saveAsPreset(name: trimmed, scope: viewModel.activeSurface.rawValue)
        newPresetName = ""
        showSaveAsField = false
        isOverwriteArmed = false
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
