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
    case embeddings = "Embeddings & ANE"
    case models = "Models"
    case perf = "Perf & Server"

    public var id: String { rawValue }

    public var shortTitle: String {
        switch self {
        case .chat: return "Chat"
        case .notebook: return "Notebook"
        case .embeddings: return "Embeddings"
        case .models: return "Models"
        case .perf: return "Perf & Server"
        }
    }

    public var icon: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        case .notebook: return "doc.text"
        case .embeddings: return "sparkles.rectangle.stack"
        case .models: return "cpu"
        case .perf: return "gauge.with.dots.needle.67percent"
        }
    }

    public var tintColor: Color {
        switch self {
        case .chat: return .blue
        case .notebook: return .orange
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

    public init(viewModel: HeylookWorkbenchViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            HeylookSidebarView(viewModel: viewModel)
                .navigationTitle("Heylook")
                .navigationSplitViewColumnWidth(min: 250, ideal: 280, max: 340)
        } detail: {
            VStack(spacing: 0) {
                if let banner = viewModel.statusBanner, !banner.isEmpty {
                    HStack(spacing: 10) {
                        Image(systemName: "info.circle.fill")
                            .foregroundStyle(.blue)
                        Text(banner)
                            .font(.caption.weight(.medium))
                            .lineLimit(2)
                            .textSelection(.enabled)
                        Spacer()
                        Button {
                            viewModel.statusBanner = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color.blue.opacity(0.1))
                    Divider()
                }

                if viewModel.registeredModels.isEmpty {
                    GlassEffectContainer(spacing: 10) {
                        HStack(spacing: 10) {
                            Image(systemName: "externaldrive.badge.plus")
                                .foregroundStyle(.orange)
                            Text("No local models discovered in heylook.toml scan_folders.")
                                .font(.caption.weight(.medium))
                            Spacer()
                            Button("Re-scan Folders") {
                                Task { await viewModel.rescanConfiguredModels() }
                            }
                            .buttonStyle(.glass)
                            .controlSize(.small)

                            Button("Load Sample Demo Workspace") {
                                Task { await viewModel.loadSampleDemoWorkspace() }
                            }
                            .buttonStyle(.glassProminent)
                            .controlSize(.small)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color.orange.opacity(0.1))
                    Divider()
                }

                Group {
                    switch viewModel.activeSurface {
                    case .chat:
                        ChatWorkbenchSurface(viewModel: viewModel)
                    case .notebook:
                        NotebookSurface(viewModel: viewModel)
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
                    case .chat, .notebook, .models, .perf:
                        LoadAndSamplerInspectorView(viewModel: viewModel)
                    }
                }
                .inspectorColumnWidth(min: 300, ideal: 340, max: 430)
            }
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    Picker("Active Model", selection: $viewModel.selectedModelID) {
                        if viewModel.registeredModels.isEmpty {
                            Text("No Models Loaded").tag("")
                        }
                        ForEach(viewModel.registeredModels) { model in
                            Text(model.modelID).tag(model.modelID)
                        }
                    }
                    .frame(minWidth: 180, maxWidth: 300)
                    .help("Select active model for generation")
                }
                .visibilityPriority(.high)

                ToolbarSpacer(.fixed)

                ToolbarItemGroup(placement: .automatic) {
                    Button {
                        viewModel.showLoadPanel.toggle()
                    } label: {
                        Label("Samplers", systemImage: "slider.horizontal.3")
                    }
                    .help("Open Sampler Cascade & System Prompt Sheet")

                    Button {
                        viewModel.showInspector.toggle()
                    } label: {
                        Label("Inspector", systemImage: "sidebar.right")
                    }
                    .help("Toggle Right Inspector")
                }
                .visibilityPriority(.high)
            }
            .sheet(isPresented: $viewModel.showLoadPanel) {
                LoadAndSamplerPanelSheet(viewModel: viewModel)
            }
        }
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

    private func badgeText(for surface: HeylookSurface) -> String? {
        switch surface {
        case .chat:
            return viewModel.conversations.isEmpty ? nil : "\(viewModel.conversations.count)"
        case .notebook:
            return viewModel.notebooks.isEmpty ? nil : "\(viewModel.notebooks.count)"
        case .embeddings:
            return viewModel.artifactCount > 0 ? "\(viewModel.artifactCount)" : nil
        case .models:
            return viewModel.registeredModels.isEmpty ? nil : "\(viewModel.registeredModels.count)"
        case .perf:
            return viewModel.isServerRunning ? "LIVE" : nil
        }
    }
}

// MARK: - Extracted Contextual Sidebar Sections (swiftui-specialist / structure.md)

private struct ChatConversationsSidebarSection: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

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

            ForEach(viewModel.conversations) { conv in
                let isSelected = viewModel.selectedConversationID == conv.id
                Button {
                    viewModel.selectConversation(conv.id)
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(conv.title)
                            .font(.subheadline.weight(isSelected ? .semibold : .regular))
                            .lineLimit(1)
                        if let modelID = conv.defaultModelID, !modelID.isEmpty {
                            Text(modelID)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
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
                    Button("Delete Conversation", role: .destructive) {
                        viewModel.deleteConversation(conv.id)
                    }
                }
                .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                    Button(role: .destructive) {
                        viewModel.deleteConversation(conv.id)
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
                    Button("Delete Notebook", role: .destructive) {
                        viewModel.deleteNotebook(nb.id)
                    }
                }
                .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                    Button(role: .destructive) {
                        viewModel.deleteNotebook(nb.id)
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
                Task { await viewModel.unloadAllModels() }
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
                Task { await viewModel.unloadAllModels() }
            } label: {
                Label("Unload All & Reclaim Memory", systemImage: "memorychip")
            }
            .buttonStyle(.plain)
        }
    }
}
