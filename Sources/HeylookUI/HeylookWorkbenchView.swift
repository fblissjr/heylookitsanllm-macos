import Foundation
import Observation
import SwiftUI
import UniformTypeIdentifiers
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

public enum HeylookSurface: String, CaseIterable, Identifiable, Sendable {
    case chat = "Chat"
    case notebook = "Notebook"
    case presets = "Presets"
    case embeddings = "Embeddings & ANE"
    case models = "Models"
    case perf = "Perf & Server"

    public var id: String { rawValue }

    public var shortTitle: String {
        switch self {
        case .chat: return "Chat"
        case .notebook: return "Notebook"
        case .presets: return "Presets"
        case .embeddings: return "Embeddings"
        case .models: return "Models"
        case .perf: return "Perf & Server"
        }
    }

    public var icon: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        case .notebook: return "doc.text"
        case .presets: return "slider.horizontal.3"
        case .embeddings: return "sparkles.rectangle.stack"
        case .models: return "cpu"
        case .perf: return "gauge.with.dots.needle.67percent"
        }
    }

    public var tintColor: Color {
        switch self {
        case .chat: return .blue
        case .notebook: return .orange
        case .presets: return .teal
        case .embeddings: return .purple
        case .models: return .mint
        case .perf: return .indigo
        }
    }
}

// MARK: - Root Multiplatform Workbench View

public struct HeylookRootView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var columnVisibility: NavigationSplitViewVisibility = .all
    @State private var showShortcutsSheet: Bool = false

    public init(viewModel: HeylookWorkbenchViewModel) {
        self.viewModel = viewModel
    }

    private var activeModelSelectionBinding: Binding<String> {
        Binding(
            get: { viewModel.selectedModelID },
            set: { newID in
                viewModel.selectGenerationModel(newID)
            }
        )
    }

    public var body: some View {
        let genModels = viewModel.generationModels
        NavigationSplitView(columnVisibility: $columnVisibility) {
            HeylookSidebarView(viewModel: viewModel)
                .navigationTitle("Heylook")
                .navigationSplitViewColumnWidth(min: 250, ideal: 280, max: 340)
        } detail: {
            VStack(spacing: 0) {
                if let banner = viewModel.statusBanner, !banner.isEmpty {
                    StatusBannerBar(banner: banner) {
                        viewModel.statusBanner = nil
                    }
                    Divider()
                }

                if viewModel.registeredModels.isEmpty {
                    EmptyModelsDiscoveryBanner(
                        onRescan: { Task { await viewModel.rescanConfiguredModels() } },
                        onLoadDemo: { Task { await viewModel.loadSampleDemoWorkspace() } }
                    )
                    Divider()
                }

                Group {
                    switch viewModel.activeSurface {
                    case .chat:
                        ChatWorkbenchSurface(viewModel: viewModel)
                    case .notebook:
                        NotebookSurface(viewModel: viewModel)
                    case .presets:
                        PresetsSurface(viewModel: viewModel)
                    case .embeddings:
                        EmbeddingsStudioSurface(viewModel: viewModel)
                    case .models:
                        ModelsInspectorSurface(viewModel: viewModel)
                    case .perf:
                        PerfServerSurface(viewModel: viewModel)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .modifier(ToolbarContainerBehavior27_3Modifier())
            .inspector(isPresented: $viewModel.showInspector) {
                Group {
                    switch viewModel.activeSurface {
                    case .embeddings:
                        EmbeddingsInspectorView(
                            hit: viewModel.selectedHit,
                            selectedDimension: viewModel.selectedDimension,
                            onSelectDimension: { dim in
                                viewModel.selectedDimension = dim
                                Task { await viewModel.runEmbeddingSearch() }
                            },
                            onSendToCompare: { hit in
                                viewModel.sendHitToPairwiseCompare(hit)
                            },
                            onPinToLab: { hit in
                                Task { await viewModel.pinHitToRetrievalLab(hit) }
                            }
                        )
                    case .chat, .notebook, .presets, .models, .perf:
                        LoadAndSamplerInspectorView(viewModel: viewModel)
                    }
                }
                .inspectorColumnWidth(min: 310, ideal: 350, max: 440)
            }
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    Picker("Active Model", selection: activeModelSelectionBinding) {
                        if genModels.isEmpty {
                            Text("No Generation Models").tag("")
                        }
                        if !viewModel.selectedModelID.isEmpty,
                           !genModels.contains(where: { $0.modelID == viewModel.selectedModelID }) {
                            Text("○ \(viewModel.selectedModelID)").tag(viewModel.selectedModelID)
                        }
                        ForEach(genModels) { model in
                            let dot = viewModel.loadingModelIDs.contains(model.modelID)
                                ? "◐"
                                : (model.isLoaded ? "●" : "○")
                            Text("\(dot) \(model.modelID)").tag(model.modelID)
                        }
                    }
                    .frame(minWidth: 190, maxWidth: 320)
                    .help("Select active generation model (● loaded in Unified Memory, ○ unloaded)")
                }
                .visibilityPriority(.high)

                ToolbarSpacer(.fixed)

                ToolbarItemGroup(placement: .automatic) {
                    Button {
                        showShortcutsSheet = true
                    } label: {
                        Label("Shortcuts", systemImage: "questionmark.circle")
                    }
                    .keyboardShortcut("/", modifiers: [.command])
                    .help("Keyboard Shortcuts (⌘/)")

                    Button {
                        viewModel.showInspector.toggle()
                    } label: {
                        Label("Inspector", systemImage: "sidebar.right")
                    }
                    .keyboardShortcut("i", modifiers: [.command, .option])
                    .help("Toggle Right Inspector — Presets, Samplers & System Prompt (⌥⌘I)")
                }
                .visibilityPriority(.high)

                #if os(iOS)
                ToolbarOverflowMenu {
                    Button {
                        Task { await viewModel.rescanConfiguredModels() }
                    } label: {
                        Label("Re-scan Models", systemImage: "arrow.clockwise")
                    }

                    Button {
                        Task { await viewModel.loadSampleDemoWorkspace() }
                    } label: {
                        Label("Load Demo Workspace", systemImage: "sparkles")
                    }

                    Button {
                        Task { await viewModel.toggleServer() }
                    } label: {
                        Label(
                            viewModel.isServerRunning ? "Stop HTTP Server" : "Start HTTP Server",
                            systemImage: viewModel.isServerRunning ? "stop.circle" : "play.circle"
                        )
                    }
                }
                #endif
            }
            .sheet(isPresented: $showShortcutsSheet) {
                KeyboardShortcutsSheet()
            }
            .confirmationDialog(
                viewModel.pendingConfirmation?.title ?? "Confirm Action",
                item: $viewModel.pendingConfirmation
            ) { action in
                Button(action.confirmLabel, role: .destructive) {
                    viewModel.confirmDestructiveAction(action)
                }
                Button("Cancel", role: .cancel) {}
            } message: { action in
                Text(action.message)
            }
        }
        .onKeyPress(.escape) {
            handleLayeredEscape()
        }
    }

    private func handleLayeredEscape() -> KeyPress.Result {
        if viewModel.isGenerating {
            viewModel.cancelGeneration()
            return .handled
        }
        if viewModel.isNotebookGenerating {
            viewModel.cancelNotebookGeneration()
            return .handled
        }
        if viewModel.pendingModelSwitchTargetID != nil {
            viewModel.cancelPendingModelSwitch()
            return .handled
        }
        if viewModel.editingMessageID != nil {
            viewModel.cancelEditingMessage()
            return .handled
        }
        if viewModel.showPromptPreview {
            viewModel.showPromptPreview = false
            return .handled
        }
        if viewModel.showInspector {
            viewModel.showInspector = false
            return .handled
        }
        return .ignored
    }
}

private struct ToolbarContainerBehavior27_3Modifier: ViewModifier {
    func body(content: Content) -> some View {
        if #available(macOS 27.3, iOS 27.3, *) {
            content.toolbarContainerBehavior(.automatic)
        } else {
            content
        }
    }
}

private struct StatusBannerBar: View {
    let banner: String
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "info.circle.fill")
                .foregroundStyle(.blue)
            Text(banner)
                .font(.caption.weight(.medium))
                .lineLimit(2)
                .textSelection(.enabled)
            Spacer()
            Button(action: onDismiss) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 28, minHeight: 28)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Dismiss status banner")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(Color.blue.opacity(0.1))
    }
}

private struct EmptyModelsDiscoveryBanner: View {
    let onRescan: () -> Void
    let onLoadDemo: () -> Void

    var body: some View {
        GlassEffectContainer(spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: "externaldrive.badge.plus")
                    .foregroundStyle(.orange)
                Text("No local models discovered in heylook.toml scan_folders.")
                    .font(.caption.weight(.medium))
                Spacer()
                Button("Re-scan Folders", action: onRescan)
                    .buttonStyle(.glass)
                    .controlSize(.small)

                Button("Load Sample Demo Workspace", action: onLoadDemo)
                    .buttonStyle(.glassProminent)
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(Color.orange.opacity(0.1))
    }
}

// MARK: - Unified Sidebar with Contextual Surface Sections

private struct HeylookSidebarView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.appearsActive) private var appearsActive
    @State private var isDropTargeted = false

    var body: some View {
        List {
            Section("Workbench") {
                ForEach(HeylookSurface.allCases) { surface in
                    Button {
                        viewModel.activeSurface = surface
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: surface.icon)
                                .frame(width: 18)
                                .foregroundStyle(viewModel.activeSurface == surface ? Color.accentColor : surface.tintColor)
                            Text(surface.shortTitle)
                                .fontWeight(viewModel.activeSurface == surface ? .semibold : .regular)
                                .foregroundStyle(.primary)
                            Spacer()
                            if isSurfaceBusy(surface) {
                                ProgressView()
                                    .controlSize(.mini)
                            }
                            if let badge = badgeText(for: surface) {
                                Text(badge)
                                    .font(.caption2.monospacedDigit())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 1)
                                    .background(
                                        Capsule().fill(
                                            viewModel.activeSurface == surface
                                                ? Color.accentColor.opacity(0.18)
                                                : Color.secondary.opacity(0.12)
                                        )
                                    )
                                    .foregroundStyle(viewModel.activeSurface == surface ? Color.accentColor : .secondary)
                            }
                        }
                        .padding(.vertical, 3)
                        .padding(.horizontal, 6)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .listRowBackground(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(viewModel.activeSurface == surface ? Color.accentColor.opacity(0.14) : Color.clear)
                    )
                }
            }

            switch viewModel.activeSurface {
            case .chat:
                ChatConversationsSidebarSection(viewModel: viewModel)
            case .notebook:
                NotebooksSidebarSection(viewModel: viewModel)
            case .presets:
                PresetsSidebarSection(viewModel: viewModel)
            case .embeddings:
                EmbeddingsSidebarSections(viewModel: viewModel)
            case .models:
                ModelsFilterSidebarSection(viewModel: viewModel)
            case .perf:
                PerfQuickActionsSidebarSection(viewModel: viewModel)
            }
        }
        .listStyle(.sidebar)
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            guard viewModel.activeSurface == .embeddings else { return false }
            for provider in providers {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in
                        viewModel.startIndexing([url])
                    }
                }
            }
            return true
        }
        .safeAreaInset(edge: .bottom) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Circle()
                        .fill(viewModel.isServerRunning ? Color.green : Color.secondary.opacity(0.5))
                        .frame(width: 8, height: 8)
                    Text(viewModel.isServerRunning ? "Server :\(viewModel.serverPort)" : "Server Offline")
                        .font(.caption.monospaced())
                    Spacer()
                    Button(viewModel.isServerRunning ? "Stop" : "Serve") {
                        Task { await viewModel.toggleServer() }
                    }
                    .buttonStyle(.glass)
                    .controlSize(.small)
                }
                Text("\(viewModel.schedulerMetrics.loadedGPUModels.count) GPU • \(viewModel.schedulerMetrics.loadedANEModels.count) ANE")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(10)
            .glassEffect(.regular, in: .rect(cornerRadius: 12))
            .padding(8)
            .opacity(appearsActive ? 1.0 : 0.6)
        }
    }

    private func isSurfaceBusy(_ surface: HeylookSurface) -> Bool {
        switch surface {
        case .chat: return viewModel.isGenerating
        case .notebook: return viewModel.isNotebookGenerating
        case .presets: return false
        case .embeddings: return viewModel.isIndexing
        case .models: return !viewModel.loadingModelIDs.isEmpty
        case .perf: return false
        }
    }

    private func badgeText(for surface: HeylookSurface) -> String? {
        switch surface {
        case .chat:
            return viewModel.conversations.isEmpty ? nil : "\(viewModel.conversations.count)"
        case .notebook:
            return viewModel.notebooks.isEmpty ? nil : "\(viewModel.notebooks.count)"
        case .presets:
            return viewModel.presets.isEmpty ? nil : "\(viewModel.presets.count)"
        case .embeddings:
            return viewModel.artifactCount > 0 ? "\(viewModel.artifactCount)" : nil
        case .models:
            return viewModel.registeredModels.isEmpty ? nil : "\(viewModel.registeredModels.count)"
        case .perf:
            return viewModel.isServerRunning ? "LIVE" : nil
        }
    }
}

// MARK: - Extracted Contextual Sidebar Sections

private struct ChatConversationsSidebarSection: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var renamingConversation: ConversationRecord?
    @State private var renameDraftTitle: String = ""

    var body: some View {
        Section {
            Button {
                viewModel.createNewConversation()
            } label: {
                Label("New Conversation", systemImage: "plus.bubble")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Color.accentColor)
            }
            .buttonStyle(.plain)

            if viewModel.conversations.count > 3 || !viewModel.conversationSearchQuery.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    TextField("Filter conversations…", text: $viewModel.conversationSearchQuery)
                        .textFieldStyle(.plain)
                        .font(.caption)
                    if !viewModel.conversationSearchQuery.isEmpty {
                        Button {
                            viewModel.conversationSearchQuery = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 4)
                .background(Color.secondary.opacity(0.09), in: RoundedRectangle(cornerRadius: 6))
            }

            ForEach(viewModel.filteredConversations) { conv in
                let isSelected = viewModel.selectedConversationID == conv.id
                let isStreamingHere = viewModel.isGenerating && viewModel.generatingConversationID == conv.id
                Button {
                    viewModel.selectConversation(conv.id)
                } label: {
                    HStack(spacing: 6) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(conv.title)
                                .font(.subheadline.weight(isSelected ? .semibold : .regular))
                                .lineLimit(1)
                            HStack(spacing: 4) {
                                if let modelID = conv.defaultModelID, !modelID.isEmpty {
                                    Text(modelID)
                                        .font(.caption2.monospaced())
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                                if conv.appliedPresetID != nil || conv.params != nil {
                                    Image(systemName: "dial.medium")
                                        .font(.system(size: 9))
                                        .foregroundStyle(.indigo)
                                }
                            }
                        }
                        Spacer(minLength: 0)
                        if isStreamingHere {
                            ProgressView()
                                .controlSize(.mini)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 2)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .listRowBackground(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(isSelected ? Color.accentColor.opacity(0.16) : Color.clear)
                )
                .contextMenu {
                    Button {
                        renameDraftTitle = conv.title
                        renamingConversation = conv
                    } label: {
                        Label("Rename Conversation…", systemImage: "pencil")
                            .labelStyle(.titleAndIcon)
                    }

                    Button {
                        viewModel.cloneConversation(id: conv.id)
                    } label: {
                        Label("Clone Conversation", systemImage: "doc.on.doc")
                            .labelStyle(.titleAndIcon)
                    }

                    Divider()

                    Button(role: .destructive) {
                        viewModel.pendingConfirmation = .deleteConversation(id: conv.id, title: conv.title)
                    } label: {
                        Label("Delete Conversation…", systemImage: "trash")
                            .labelStyle(.titleAndIcon)
                    }
                }
                .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                    Button(role: .destructive) {
                        viewModel.pendingConfirmation = .deleteConversation(id: conv.id, title: conv.title)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
        } header: {
            HStack {
                Text("Conversations")
                Spacer()
                Button {
                    viewModel.createNewConversation()
                } label: {
                    Image(systemName: "square.and.pencil")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .help("New Conversation (⌘N)")
            }
        }
        .alert(
            "Rename Conversation",
            isPresented: Binding(
                get: { renamingConversation != nil },
                set: { if !$0 { renamingConversation = nil } }
            ),
            presenting: renamingConversation
        ) { conv in
            TextField("Conversation title", text: $renameDraftTitle)
            Button("Save") {
                viewModel.renameConversation(id: conv.id, newTitle: renameDraftTitle)
                renamingConversation = nil
            }
            Button("Cancel", role: .cancel) {
                renamingConversation = nil
            }
        }
    }
}

private struct NotebooksSidebarSection: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        Section {
            Button {
                viewModel.createNewNotebook()
            } label: {
                Label("New Notebook", systemImage: "doc.badge.plus")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Color.accentColor)
            }
            .buttonStyle(.plain)

            ForEach(viewModel.notebooks) { nb in
                let isSelected = viewModel.selectedNotebookID == nb.id
                Button {
                    viewModel.selectNotebook(nb.id)
                } label: {
                    HStack(spacing: 6) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(nb.title)
                                .font(.subheadline.weight(isSelected ? .semibold : .regular))
                                .lineLimit(1)
                            if let modelID = nb.modelID, !modelID.isEmpty {
                                Text(modelID)
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                        }
                        Spacer(minLength: 0)
                        if isSelected && viewModel.isNotebookGenerating {
                            ProgressView()
                                .controlSize(.mini)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 2)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .listRowBackground(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(isSelected ? Color.accentColor.opacity(0.16) : Color.clear)
                )
                .contextMenu {
                    Button(role: .destructive) {
                        viewModel.pendingConfirmation = .deleteNotebook(id: nb.id, title: nb.title)
                    } label: {
                        Label("Delete Notebook…", systemImage: "trash")
                    }
                }
                .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                    Button(role: .destructive) {
                        viewModel.pendingConfirmation = .deleteNotebook(id: nb.id, title: nb.title)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
        } header: {
            HStack {
                Text("Saved Notebooks")
                Spacer()
                Button {
                    viewModel.createNewNotebook()
                } label: {
                    Image(systemName: "plus")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .help("New Notebook (⇧⌘N)")
            }
        }
    }
}

private struct PresetsSidebarSection: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        Section("Active Preset State") {
            HStack {
                Image(systemName: "dial.medium")
                    .foregroundStyle(.teal)
                Text(viewModel.selectedPresetName ?? "Model Defaults")
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                Spacer()
                if viewModel.selectedPresetName != nil || viewModel.hasAnyManualSamplerOverride {
                    Button("Reset") {
                        viewModel.clearPresetSelection()
                    }
                    .font(.caption2)
                    .buttonStyle(.borderless)
                }
            }
        }

        Section("Scope Filter") {
            Button {
                viewModel.presetScopeFilter = nil
            } label: {
                HStack {
                    Text("All Presets")
                        .fontWeight(viewModel.presetScopeFilter == nil ? .semibold : .regular)
                    Spacer()
                    Text("\(viewModel.presets.count)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Button {
                viewModel.presetScopeFilter = "sampler"
            } label: {
                HStack {
                    Text("Sampler Only")
                        .fontWeight(viewModel.presetScopeFilter == "sampler" ? .semibold : .regular)
                    Spacer()
                    Text("\(viewModel.presets.filter { $0.scope == "sampler" }.count)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Button {
                viewModel.presetScopeFilter = "full"
            } label: {
                HStack {
                    Text("Full (+ System Prompt)")
                        .fontWeight(viewModel.presetScopeFilter == "full" ? .semibold : .regular)
                    Spacer()
                    Text("\(viewModel.presets.filter { $0.scope == "full" }.count)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
    }
}

private struct ModelsFilterSidebarSection: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        Section("Provider Filter") {
            Button {
                viewModel.selectedProviderFilter = nil
            } label: {
                HStack {
                    Label("All Providers", systemImage: "square.stack.3d.up")
                        .fontWeight(viewModel.selectedProviderFilter == nil ? .semibold : .regular)
                    Spacer()
                    Text("\(viewModel.registeredModels.count)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            ForEach(EngineProviderKind.allCases, id: \.self) { provider in
                let count = viewModel.modelCountsByProvider[provider] ?? 0
                Button {
                    viewModel.selectedProviderFilter = provider
                } label: {
                    HStack {
                        Text(provider.displayName)
                            .fontWeight(viewModel.selectedProviderFilter == provider ? .semibold : .regular)
                        Spacer()
                        Text("\(count)")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }

        Section("Model Actions") {
            Button {
                Task { await viewModel.rescanConfiguredModels() }
            } label: {
                Label("Re-scan heylook.toml", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.plain)

            Button {
                viewModel.convertingSourceModel = nil
                viewModel.showMLXConverterSheet = true
            } label: {
                Label("Quantize / Convert MLX…", systemImage: "cube.transparent")
            }
            .buttonStyle(.plain)

            Button(role: .destructive) {
                if viewModel.schedulerMetrics.loadedGPUModels.isEmpty && viewModel.schedulerMetrics.loadedANEModels.isEmpty {
                    Task { await viewModel.unloadAllModels() }
                } else {
                    viewModel.pendingConfirmation = .unloadAllModels
                }
            } label: {
                Label("Unload All Models", systemImage: "eject.circle")
            }
            .buttonStyle(.plain)
        }
    }
}

private struct PerfQuickActionsSidebarSection: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        Section("Telemetry & Server") {
            Button {
                Task { await viewModel.refreshMetrics() }
            } label: {
                Label("Refresh Telemetry", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.plain)

            Button {
                Task { await viewModel.toggleServer() }
            } label: {
                Label(
                    viewModel.isServerRunning ? "Stop HTTP Server" : "Start HTTP Server",
                    systemImage: viewModel.isServerRunning ? "stop.circle" : "play.circle"
                )
            }
            .buttonStyle(.plain)

            Button(role: .destructive) {
                if viewModel.schedulerMetrics.loadedGPUModels.isEmpty && viewModel.schedulerMetrics.loadedANEModels.isEmpty {
                    Task { await viewModel.unloadAllModels() }
                } else {
                    viewModel.pendingConfirmation = .unloadAllModels
                }
            } label: {
                Label("Unload All & Reclaim Memory", systemImage: "memorychip")
            }
            .buttonStyle(.plain)
        }
    }
}

// MARK: - Keyboard Shortcuts Cheat Sheet

private struct KeyboardShortcutsSheet: View {
    @Environment(\.dismiss) private var dismiss

    private let shortcuts: [(category: String, items: [(keys: String, action: String)])] = [
        (
            "Navigation & Surfaces",
            [
                ("⌘1 – ⌘6", "Switch between Chat, Notebook, Presets, Embeddings, Models, and Perf"),
                ("⌥⌘I", "Toggle Right Inspector (Presets, Samplers & System Prompt)"),
                ("⌘/", "Show Keyboard Shortcuts")
            ]
        ),
        (
            "Chat & Notebook Generation",
            [
                ("⌘N", "New Conversation"),
                ("⇧⌘N", "New Notebook"),
                ("⌘↩", "Send Chat Prompt / Continue Notebook at Cursor"),
                ("⇧⌘V", "Paste Image from Clipboard into Chat Composer"),
                ("Esc", "Stop Active Generation → Close Inline Edit → Close Drawers")
            ]
        ),
        (
            "Window & Lifecycle",
            [
                ("⌥⌘N", "Open Workbench Window"),
                ("⌘S", "Save Active Notebook Snapshot")
            ]
        )
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("Keyboard Shortcuts", systemImage: "keyboard")
                    .font(.headline)
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .keyboardShortcut(.defaultAction)
            }

            Divider()

            ForEach(shortcuts, id: \.category) { group in
                VStack(alignment: .leading, spacing: 6) {
                    Text(group.category)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    ForEach(group.items, id: \.keys) { item in
                        HStack {
                            Text(item.action)
                                .font(.caption)
                            Spacer()
                            Text(item.keys)
                                .font(.caption.monospaced().weight(.semibold))
                                .padding(.horizontal, 7)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.14), in: RoundedRectangle(cornerRadius: 5))
                        }
                    }
                }
            }
        }
        .padding(20)
        .frame(minWidth: 440)
    }
}
