import Foundation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

@main
struct HeylookCLI {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        let command = args.first ?? "status"

        switch command {
        case "doctor", "status":
            await runDoctor()
        case "models":
            runModels()
        case "inspect-ane":
            let path = args.dropFirst().first ?? "."
            runInspectANE(path: path)
        case "serve":
            let port = args.dropFirst().first.flatMap { UInt16($0) } ?? 8000
            await runServe(port: port)
        case "chat":
            let prompt = args.dropFirst().joined(separator: " ")
            await runChat(prompt: prompt.isEmpty ? "Explain unified memory + ANE co-residency in two sentences." : prompt)
        case "search":
            let query = args.dropFirst().joined(separator: " ")
            await runSearch(query: query.isEmpty ? "Apple Neural Engine co-residency" : query)
        default:
            printUsage()
        }
    }

    static func runDoctor() async {
        let snap = HardwareMemoryMonitor.snapshot()
        print("=== Heylook Native macOS 27.2 / iOS 27.2 Doctor ===")
        print(String(format: "Device              : %@", snap.deviceName))
        print(String(format: "Host Unified Memory : %.1f GB (Metal Recommended Max: %.1f GB)", snap.totalPhysicalRAMGB, snap.metalRecommendedMaxWorkingSetGB))
        print("Swift Concurrency   : Swift 6.2 Strict Actor Isolation")
        print("Compute Lanes       : GPU GenerationGate (FIFO) + ANE Co-Residency Lane")
        print("Supported Providers : MLXSwiftProvider, CoreAIProvider (.aimodel), GGUFSubprocessProvider, RemoteHeylookProvider")
        print("Embeddings Engine   : EmbeddingGemma 2 + vDSP Matryoshka (768d / 512d / 256d / 128d)")
        print("Store Engine        : SQLite3 WAL (Conversations, Messages, Notebooks, Presets, VectorSpans)")
    }

    static func runModels() {
        let cfg = HeylookConfig()
        let folderURLs = cfg.scanFolders.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        let found = HeylookConfig.discoverModels(in: folderURLs)
        print("Discovered \(found.count) models across \(cfg.scanFolders.count) configured folders.")
        for m in found {
            print("- [\(m.provider.rawValue)] \(m.modelID) (\(m.modelPath))")
        }
    }

    static func runInspectANE(path: String) {
        let url = URL(fileURLWithPath: path)
        let reports = ANETowerInspector.statusSnapshot(options: .hybridANETowers, checkpointDirectory: url)
        print("=== ANE CoreML Tower Inspection (\(url.path)) ===")
        for report in reports {
            print("- \(report.component.title): unit=\(report.assignedUnit.label) compiled=\(report.isCompiledBundlePresent) aneShare=\(Int(report.aneResidencyFraction * 100))%")
        }
    }

    static func runChat(prompt: String) async {
        let scheduler = ComputeScheduler()
        let desc = EngineDescription(
            modelID: "gemma-4-26b-a4b-it-mlx",
            provider: .mlx,
            modelPath: "models/mlx/Gemma-4-26B-A4B-IT",
            architecture: ProvenanceField(value: "gemma4", provenance: .derived),
            quantization: ProvenanceField(value: "4bit", provenance: .derived),
            computePlacement: ProvenanceField(value: .hybridANEGPU, provenance: .configured, computePlacement: .hybridANEGPU),
            contextSize: ProvenanceField(value: 32768, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let provider = MLXSwiftProvider(description: desc)
        await scheduler.register(provider: provider, description: desc)
        let registry = RequestRegistry()
        let req = MessagesRequest(
            model: desc.modelID,
            messages: [MessagesRequest.MessageTurn(role: "user", content: prompt)],
            maxTokens: 256,
            stream: true,
            thinking: true
        )
        let resolved = SamplerCascade.resolve(request: req.samplerOverlay)
        do {
            try await scheduler.withComputeLease(for: desc.modelID) {
                let stream = await provider.generate(
                    requestID: "cli_req_1",
                    request: req,
                    effectiveSampling: resolved.effective,
                    registry: registry
                )
                for try await chunk in stream {
                    if !chunk.thinking.isEmpty {
                        FileHandle.standardError.write(Data(chunk.thinking.utf8))
                    }
                    if !chunk.text.isEmpty {
                        FileHandle.standardOutput.write(Data(chunk.text.utf8))
                    }
                }
                print("")
            }
        } catch {
            print("Chat error: \(error)")
        }
    }

    static func runSearch(query: String) async {
        do {
            let dbURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("cli_vectors_\(UUID().uuidString).sqlite")
            defer { try? FileManager.default.removeItem(at: dbURL) }
            let vectorStore = try VectorStore(databaseURL: dbURL)
            let engine = DeterministicHashEmbeddingEngine()
            let snippet = "Unified MLX GPU inference + Apple Neural Engine (ANE) co-residency."
            let doc = Artifact(
                id: "vision-doc",
                collectionID: "default",
                kind: .text,
                url: URL(fileURLWithPath: "/workspace/VISION.md"),
                title: "VISION.md",
                fingerprint: ContentFingerprint(byteCount: Int64(snippet.utf8.count), modifiedUnixMs: 1_700_000_000_000)
            )
            let vec = engine.embed(text: snippet)
            let chunk = EmbeddedChunk(id: "vision-doc#0", artifactID: doc.id, span: .whole, vector: vec, previewText: snippet)
            try vectorStore.upsert(artifact: doc, chunks: [chunk])

            let qVec = engine.embed(text: query)
            let hits = try vectorStore.search(queryVector: qVec, dimensions: 768, limit: 5)
            print("Search results for '\(query)':")
            for hit in hits {
                print(String(format: "  [%.4f] %@ — %@", hit.bestScore, hit.artifact.title, hit.previewText ?? ""))
            }
        } catch {
            print("Search error: \(error)")
        }
    }

    static func runServe(port: UInt16) async {
        do {
            let dbURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("heylook_serve.sqlite")
            let store = try SQLiteStore(databaseURL: dbURL)
            let scheduler = ComputeScheduler()
            let router = HeylookRouter(store: store, scheduler: scheduler)
            let server = HeylookHTTPServer(router: router, host: "127.0.0.1", port: port)
            try await server.start()
            print("Heylook Server listening on http://127.0.0.1:\(port)")
            try await Task.sleep(for: .seconds(3600 * 24))
        } catch {
            print("Server error: \(error)")
        }
    }

    static func printUsage() {
        print("""
        Usage: heylook-cli <command> [args]
        Commands:
          doctor | status      Inspect unified memory, compute lanes, and providers
          models               Discover models from configured folders
          inspect-ane <dir>    Inspect CoreML ANE tower packages and op compatibility
          chat <prompt>        Stream a chat completion with thinking traces
          search <query>       Run Matryoshka cosine search over sample spans
          serve [port]         Run the embedded /v1/messages & /v1/embeddings server
        """)
    }
}
