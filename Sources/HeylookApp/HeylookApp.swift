import SwiftUI
import HeylookUI

@main
struct HeylookApp: App {
    @State private var viewModel = HeylookWorkbenchViewModel()

    var body: some Scene {
        WindowGroup("Heylook Workbench") {
            HeylookRootView(viewModel: viewModel)
                .frame(minWidth: 960, minHeight: 640)
        }

        #if os(macOS)
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

                Button(viewModel.isServerRunning ? "Stop Server" : "Start Server") {
                    Task { await viewModel.toggleServer() }
                }

                Divider()

                Button("Quit Heylook") {
                    NSApplication.shared.terminate(nil)
                }
            }
            .padding(8)
        }
        #endif
    }
}
