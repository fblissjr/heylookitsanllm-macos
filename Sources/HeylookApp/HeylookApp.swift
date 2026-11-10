import Observation
import SwiftUI
import UniformTypeIdentifiers
import HeylookCore
import HeylookUI

#if os(macOS)
import AppKit

@MainActor
final class HeylookAppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate()
        for window in NSApplication.shared.windows {
            window.styleMask.insert(.resizable)
            window.makeKeyAndOrderFront(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }
}
#endif

@main
struct HeylookApp: App {
    #if os(macOS)
    @NSApplicationDelegateAdaptor(HeylookAppDelegate.self) private var appDelegate
    #endif
    @State private var viewModel = HeylookWorkbenchViewModel()

    init() {
        #if os(macOS)
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate()
        #endif
    }

    var body: some Scene {
        WindowGroup("Heylook Workbench", id: "workbench") {
            HeylookRootView(viewModel: viewModel)
                .frame(minWidth: 980, minHeight: 640)
                #if os(macOS)
                .onAppear {
                    NSApplication.shared.setActivationPolicy(.regular)
                    NSApplication.shared.activate()
                    for window in NSApplication.shared.windows {
                        window.styleMask.insert(.resizable)
                        window.makeKeyAndOrderFront(nil)
                    }
                }
                #endif
        }
        #if os(macOS)
        .defaultSize(width: 1360, height: 860)
        .windowResizability(.contentMinSize)
        .commands {
            SidebarCommands()
            InspectorCommands()
            CommandGroup(replacing: .newItem) {
                HeylookNewItemCommandsView(viewModel: viewModel)
            }
        }
        #endif

        #if os(macOS)
        Settings {
            HeylookSettingsView(viewModel: viewModel)
        }

        MenuBarExtra(
            "Heylook",
            systemImage: viewModel.isServerRunning ? "bolt.horizontal.circle.fill" : "bolt.horizontal.circle"
        ) {
            HeylookMenuBarMenuContent(viewModel: viewModel)
        }
        #endif
    }
}

#if os(macOS)
private struct HeylookNewItemCommandsView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button {
            openAndActivateWorkbench()
            viewModel.activeSurface = .chat
            viewModel.createNewConversation()
        } label: {
            Label("New Conversation", systemImage: "plus.bubble")
                .labelStyle(.titleAndIcon)
        }
        .keyboardShortcut("n", modifiers: [.command])

        Button {
            openAndActivateWorkbench()
            viewModel.activeSurface = .notebook
            viewModel.createNewNotebook()
        } label: {
            Label("New Notebook", systemImage: "doc.badge.plus")
                .labelStyle(.titleAndIcon)
        }
        .keyboardShortcut("n", modifiers: [.command, .shift])

        Button {
            openAndActivateWorkbench()
        } label: {
            Label("Open Workbench Window", systemImage: "macwindow")
                .labelStyle(.titleAndIcon)
        }
        .keyboardShortcut("n", modifiers: [.command, .option])

        Divider()

        ForEach(Array(HeylookSurface.allCases.enumerated()), id: \.element.id) { idx, surface in
            Button {
                openAndActivateWorkbench()
                viewModel.activeSurface = surface
            } label: {
                Label("Show \(surface.rawValue)", systemImage: surface.icon)
                    .labelStyle(.titleAndIcon)
            }
            .keyboardShortcut(KeyEquivalent(Character("\(idx + 1)")), modifiers: [.command])
        }
    }

    private func openAndActivateWorkbench() {
        if NSApplication.shared.windows.allSatisfy({ !$0.isVisible }) {
            openWindow(id: "workbench")
        }
        NSApplication.shared.activate()
    }
}

private struct HeylookMenuBarMenuContent: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Heylook Inference & ANE Server")
                .font(.headline)

            Text(
                viewModel.isServerRunning
                    ? "Online • 127.0.0.1:\(viewModel.serverPort)"
                    : "Server Stopped"
            )
            .font(.caption.monospaced())

            Text(
                String(
                    format: "Memory: %d MB Metal • %d MB RSS • %.1f GB Sys RAM • %.0f%% CPU",
                    viewModel.memorySnapshot.metalCurrentAllocatedMB,
                    viewModel.memorySnapshot.processResidentMB,
                    viewModel.memorySnapshot.ramUsedGB,
                    viewModel.memorySnapshot.cpuPercent
                )
            )
            .font(.caption2.monospaced())
            .foregroundStyle(.secondary)

            let gpuList = viewModel.schedulerMetrics.loadedGPUModels
            let aneList = viewModel.schedulerMetrics.loadedANEModels
            if gpuList.isEmpty && aneList.isEmpty {
                Text("Residency: 0 GPU • 0 ANE models loaded")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                if !gpuList.isEmpty {
                    Text("GPU (\(gpuList.count)): \(gpuList.joined(separator: ", "))")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                if !aneList.isEmpty {
                    Text("ANE (\(aneList.count)): \(aneList.joined(separator: ", "))")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }

            Divider()

            Button {
                openWindow(id: "workbench")
                NSApplication.shared.activate()
            } label: {
                Label("Open Workbench", systemImage: "macwindow")
                    .labelStyle(.titleAndIcon)
            }

            Button {
                Task { await viewModel.toggleServer() }
            } label: {
                Label(
                    viewModel.isServerRunning ? "Stop Server" : "Start Server",
                    systemImage: viewModel.isServerRunning ? "stop.circle" : "play.circle"
                )
                .labelStyle(.titleAndIcon)
            }

            Button {
                Task { await viewModel.unloadAllModels() }
            } label: {
                Label("Unload All Models", systemImage: "eject.circle")
                    .labelStyle(.titleAndIcon)
            }

            Divider()

            Button {
                Task {
                    await viewModel.shutdown()
                    NSApplication.shared.terminate(nil)
                }
            } label: {
                Label("Quit Heylook", systemImage: "power")
                    .labelStyle(.titleAndIcon)
            }
        }
        .padding(8)
    }
}

struct HeylookSettingsView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var showScanFolderImporter: Bool = false
    @State private var showClearDataConfirmation: Bool = false

    var body: some View {
        TabView {
            Tab("Server & Network", systemImage: "network") {
                Form {
                    Section("Embedded HTTP & SSE Server") {
                        LabeledContent("Server Status") {
                            HStack(spacing: 8) {
                                Circle()
                                    .fill(viewModel.isServerRunning ? Color.green : Color.secondary)
                                    .frame(width: 8, height: 8)
                                Text(viewModel.isServerRunning ? "Online (127.0.0.1:\(viewModel.serverPort))" : "Offline")
                                    .font(.caption.monospaced())
                                Button(viewModel.isServerRunning ? "Stop" : "Start") {
                                    Task { await viewModel.toggleServer() }
                                }
                                .buttonStyle(.borderedProminent)
                                .controlSize(.small)
                            }
                        }
                        TextField("Remote Peer URL", text: $viewModel.remoteServerURLString)
                            .font(.caption.monospaced())
                    }
                }
                .formStyle(.grouped)
            }

            Tab("Models & Scan Folders", systemImage: "folder.badge.gearshape") {
                Form {
                    Section {
                        if HeylookConfig.isModelConfigReadOnly {
                            Label("Read-only mode active (HEYLOOK_READONLY_MODELS=1)", systemImage: "lock.fill")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }

                        if viewModel.config.scanFolders.isEmpty {
                            Text("No scan_folders configured in heylook.toml.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else {
                            ForEach(viewModel.config.scanFolders, id: \.self) { folder in
                                HStack {
                                    Image(systemName: "folder.fill")
                                        .foregroundStyle(.secondary)
                                    Text(folder)
                                        .font(.caption.monospaced())
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                    Spacer()
                                    Button(role: .destructive) {
                                        Task { await viewModel.removeModelScanFolder(folder) }
                                    } label: {
                                        Image(systemName: "minus.circle.fill")
                                            .foregroundStyle(.red)
                                            .frame(minWidth: 28, minHeight: 28)
                                            .contentShape(Rectangle())
                                    }
                                    .buttonStyle(.plain)
                                    .disabled(HeylookConfig.isModelConfigReadOnly)
                                    .help("Remove folder from heylook.toml")
                                }
                            }
                        }

                        HStack(spacing: 10) {
                            Button {
                                showScanFolderImporter = true
                            } label: {
                                Label("Add Scan Folder…", systemImage: "folder.badge.plus")
                            }
                            .disabled(HeylookConfig.isModelConfigReadOnly)

                            Button {
                                Task { await viewModel.rescanConfiguredModels() }
                            } label: {
                                Label("Re-scan Models", systemImage: "arrow.clockwise")
                            }
                        }
                    } header: {
                        Text("heylook.toml [scan] Folders")
                    }
                }
                .formStyle(.grouped)
                .fileImporter(
                    isPresented: $showScanFolderImporter,
                    allowedContentTypes: [.folder],
                    allowsMultipleSelection: true
                ) { result in
                    if case .success(let urls) = result {
                        Task {
                            for url in urls {
                                await viewModel.addModelScanFolder(url)
                            }
                        }
                    }
                }
            }

            Tab("Embeddings & ANE", systemImage: "sparkles.rectangle.stack") {
                Form {
                    Section("Default Checkpoint & Server") {
                        TextField("Embedding Checkpoint Path", text: $viewModel.checkpointPath)
                            .font(.caption.monospaced())
                        TextField("Embedding Server URL", text: $viewModel.embeddingServerURLString)
                            .font(.caption.monospaced())
                        Toggle("Include Video Soundtrack (16 kHz Audio)", isOn: $viewModel.loadAudioFromVideo)
                    }
                }
                .formStyle(.grouped)
            }

            Tab("Data & Reset", systemImage: "externaldrive.badge.minus") {
                Form {
                    Section("SQLite Store Maintenance") {
                        LabeledContent("Conversations & Notebooks") {
                            Text("\(viewModel.conversations.count) conversations • \(viewModel.notebooks.count) notebooks • \(viewModel.presets.count) presets")
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                        }
                        Button("Clear All Conversations & Notebooks…", role: .destructive) {
                            showClearDataConfirmation = true
                        }
                    }
                }
                .formStyle(.grouped)
                .confirmationDialog(
                    "Clear All Conversations & Notebooks?",
                    isPresented: $showClearDataConfirmation
                ) {
                    Button("Clear All Data", role: .destructive) {
                        viewModel.clearAllStoreData()
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("Permanently delete all conversations, messages, media blobs, turn stats, and notebooks? Saved presets and indexed embeddings are preserved.")
                }
            }
        }
        .frame(minWidth: 560, minHeight: 360)
    }
}
#endif
