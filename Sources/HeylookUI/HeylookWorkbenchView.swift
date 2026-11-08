import Foundation
import Observation
import SwiftUI
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

    public var icon: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        case .notebook: return "doc.text"
        case .embeddings: return "sparkles.rectangle.stack"
        case .models: return "cpu"
        case .perf: return "gauge.with.dots.needle.67percent"
        }
    }
}

// MARK: - Root Multiplatform Workbench View

public struct HeylookRootView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    public init(viewModel: HeylookWorkbenchViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        NavigationSplitView {
            List(HeylookSurface.allCases, selection: $viewModel.activeSurface) { surface in
                NavigationLink(value: surface) {
                    Label(surface.rawValue, systemImage: surface.icon)
                }
            }
            .navigationTitle("Heylook")
            .safeAreaInset(edge: .bottom) {
                VStack(alignment: .leading, spacing: 6) {
                    Divider()
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
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    }
                    Text("\(viewModel.schedulerMetrics.loadedGPUModels.count) GPU • \(viewModel.schedulerMetrics.loadedANEModels.count) ANE")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(10)
            }
        } detail: {
            VStack(spacing: 0) {
                if viewModel.registeredModels.isEmpty {
                    HStack(spacing: 10) {
                        Image(systemName: "externaldrive.badge.plus")
                            .foregroundStyle(.orange)
                        Text("No local models discovered in heylook.toml scan_folders.")
                            .font(.caption.weight(.medium))
                        Spacer()
                        Button("Re-scan Folders") {
                            Task { await viewModel.rescanConfiguredModels() }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)

                        Button("Load Sample Demo Workspace") {
                            Task { await viewModel.loadSampleDemoWorkspace() }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.mini)
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
            }
            .toolbar {
                ToolbarItemGroup(placement: .automatic) {
                    Picker("Active Model", selection: $viewModel.selectedModelID) {
                        ForEach(viewModel.registeredModels) { model in
                            Text(model.modelID).tag(model.modelID)
                        }
                    }
                    .frame(maxWidth: 280)

                    Button {
                        viewModel.showLoadPanel.toggle()
                    } label: {
                        Label("Load & Samplers", systemImage: "slider.horizontal.3")
                    }
                }
            }
            .sheet(isPresented: $viewModel.showLoadPanel) {
                LoadAndSamplerPanelSheet(viewModel: viewModel)
            }
        }
    }
}
