import Observation
import SwiftUI
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
        WindowGroup("Heylook Workbench") {
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
            CommandGroup(after: .newItem) {
                Button {
                    viewModel.activeSurface = .chat
                    viewModel.createNewConversation()
                } label: {
                    Label("New Conversation", systemImage: "plus.bubble")
                        .labelStyle(.titleAndIcon)
                }
                .keyboardShortcut("n", modifiers: [.command])

                Button {
                    viewModel.activeSurface = .notebook
                    viewModel.createNewNotebook()
                } label: {
                    Label("New Notebook", systemImage: "doc.badge.plus")
                        .labelStyle(.titleAndIcon)
                }
                .keyboardShortcut("n", modifiers: [.command, .shift])

                Divider()

                ForEach(HeylookSurface.allCases.enumerated(), id: \.element.id) { idx, surface in
                    Button {
                        viewModel.activeSurface = surface
                    } label: {
                        Label("Show \(surface.rawValue)", systemImage: surface.icon)
                            .labelStyle(.titleAndIcon)
                    }
                    .keyboardShortcut(KeyEquivalent(Character("\(idx + 1)")), modifiers: [.command])
                }
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
            VStack(alignment: .leading, spacing: 8) {
                Text("Heylook Inference & ANE Server")
                    .font(.headline)
                Text(
                    viewModel.isServerRunning
                        ? "Online • 127.0.0.1:\(viewModel.serverPort)"
                        : "Server Stopped"
                )
                .font(.caption.monospaced())

                Divider()

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
        #endif
    }
}

#if os(macOS)
struct HeylookSettingsView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

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
        }
        .frame(minWidth: 520, minHeight: 320)
    }
}
#endif
