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
            let path = args.dropFirst().first
            await runInspectANE(path: path)
        case "inspect-checkpoint":
            let path = args.dropFirst().first
            runInspectCheckpoint(path: path)
        case "index":
            await runIndex(args: Array(args.dropFirst()))
        case "search":
            await runSearch(args: Array(args.dropFirst()))
        case "watched":
            runWatched()
        case "serve":
            let port = args.dropFirst().first.flatMap { UInt16($0) } ?? 8000
            await runServe(port: port)
        case "chat":
            await runChat(args: Array(args.dropFirst()))
        default:
            printUsage()
        }
    }

    static func defaultVectorStoreURL() -> URL {
        if let envPath = ProcessInfo.processInfo.environment["HEYLOOK_DB_PATH"], !envPath.isEmpty {
            return URL(fileURLWithPath: envPath)
        }
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent("heylook", isDirectory: true)
            .appendingPathComponent("vectors.sqlite")
    }

    static func resolveVectorStoreURL(from args: [String] = []) -> URL {
        if let dbIdx = args.firstIndex(of: "--db"), dbIdx + 1 < args.count {
            return URL(fileURLWithPath: args[dbIdx + 1])
        }
        return defaultVectorStoreURL()
    }

    static func parseComputeOptions(from args: [String]) -> ModelComputeOptions {
        guard let flagIdx = args.firstIndex(of: "--compute"), flagIdx + 1 < args.count else {
            return .hybridANETowers
        }
        switch args[flagIdx + 1].lowercased() {
        case "all_gpu", "gpu", "mlx":
            return .allMLXGPU
        case "all_ane", "ane":
            return .allNeuralEngine
        default:
            return .hybridANETowers
        }
    }

    static func makeEmbeddingEngine(computeOptions: ModelComputeOptions) -> any EmbeddingEngine {
        return NativeMLXEmbeddingGemma2.makeBestAvailableEngine(computeOptions: computeOptions)
    }

    static func runDoctor() async {
        let snap = HardwareMemoryMonitor.snapshot()
        let storeURL = defaultVectorStoreURL()
        print("=== Heylook Native macOS 27.2 / iOS 27.2 Doctor ===")
        print(String(format: "Device              : %@", snap.deviceName))
        print(String(format: "Host Unified Memory : %.1f GB (Metal Recommended Max: %.1f GB)", snap.totalPhysicalRAMGB, snap.metalRecommendedMaxWorkingSetGB))
        print("Swift Concurrency   : Swift 6.2 Strict Actor Isolation")
        print("Compute Lanes       : GPU GenerationGate (FIFO) + ANE Co-Residency Lane")
        print("Supported Providers : MLXSwiftProvider, CoreAIProvider (.aimodel), GGUFSubprocessProvider, RemoteHeylookProvider")
        print("Embeddings Engine   : EmbeddingGemma 2 + vDSP Matryoshka (768d / 512d / 256d / 128d)")
        print("Store Engine        : SQLite3 WAL (Conversations, Messages, Notebooks, Presets, VectorSpans)")
        if let store = try? VectorStore(databaseURL: storeURL),
           let stats = try? store.stats() {
            let watched = (try? store.listWatchedFolders()) ?? []
            print("Vector Store        : \(storeURL.path) (\(stats.artifactCount) artifacts, \(stats.chunkCount) chunks, \(watched.count) watched folders)")
            for kind in ArtifactKind.allCases {
                let c = stats.countsByKind[kind] ?? 0
                if c > 0 {
                    print("  - \(kind.displayName.padding(toLength: 16, withPad: " ", startingAt: 0)): \(c)")
                }
            }
        }
    }

    static func runModels() {
        let cfg = HeylookConfig.loadDefault()
        let folderURLs = cfg.scanFolders.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        let found = HeylookConfig.discoverModels(in: folderURLs)
        print("Discovered \(found.count) models across \(cfg.scanFolders.count) configured folders (\(cfg.explicitModels.count) explicit entries in heylook.toml).")
        for m in found {
            let mtpBadge = m.tags.contains("mtp") ? " [MTP]" : ""
            print("- [\(m.provider.rawValue)] \(m.modelID)\(mtpBadge) ctx=\(m.contextSize.value) template=\(m.templateSource.value) (\(m.modelPath))")
        }
    }

    static func runInspectANE(path: String?) async {
        let dir = path.map { URL(fileURLWithPath: $0) } ?? NativeMLXEmbeddingGemma2.defaultCheckpointURL
        let options = ModelComputeOptions.hybridANETowers
        let reports = await ANETowerInspector.inspect(checkpointDirectory: dir, options: options)
        let split = options.memorySplitMB
        print("=== ANE CoreML Tower Inspection (\(dir.path)) ===")
        print("Preset       : \(options.shortSummaryBadge)")
        print("Memory Split : ANE ~\(split.aneMB) MB | GPU ~\(split.gpuMB) MB | CPU ~\(split.cpuMB) MB")
        for report in reports {
            print("- \(report.component.title): unit=\(report.assignedUnit.label) compiled=\(report.isCompiledBundlePresent) aneShare=\(Int((report.aneResidencyFraction * 100).rounded()))% (\(report.component.architectureSummary))")
        }
    }

    static func runInspectCheckpoint(path: String?) {
        let dir = path.map { URL(fileURLWithPath: $0) } ?? NativeMLXEmbeddingGemma2.defaultCheckpointURL
        do {
            let info = try NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: dir)
            print("EmbeddingGemma 2 Checkpoint Inspection:")
            print("  Directory         : \(dir.path)")
            print("  Model Type        : \(info.modelType)")
            print("  Hidden Size       : \(info.hiddenSize)")
            print("  Embedding Dim     : \(info.embeddingDim)")
            print("  Hidden Layers     : \(info.numHiddenLayers)")
            print("  Max Input Tokens  : \(info.maxInputTokens)")
            print("  Video Window      : \(Int(info.maxWindowSeconds))s (\(info.videoMaxFrames) frames @ \(info.videoFPS) fps, <=\(info.videoMaxSoftTokens) soft tokens/frame)")
            print("  Image Soft Tokens : <=\(info.imageMaxSoftTokens)")
        } catch {
            print("Failed to inspect checkpoint at \(dir.path): \(error.localizedDescription)")
        }
    }

    static func runIndex(args: [String]) async {
        guard let targetPath = args.first, !targetPath.hasPrefix("--") else {
            print("Error: missing path to index. Usage: heylook-cli index <path> [--sampling fps|content|off] [--audio-from-video] [--compute all_gpu|hybrid_ane|all_ane]")
            return
        }
        var sampling: VideoWindowSampling = .content
        if let sIdx = args.firstIndex(of: "--sampling"), sIdx + 1 < args.count {
            let val = args[sIdx + 1].lowercased()
            if val == "fps" { sampling = .fps }
            else if val == "off" { sampling = .off }
        }
        let includeAudio = args.contains("--audio-from-video") || !args.contains("--no-audio-from-video")
        let computeOptions = parseComputeOptions(from: args)

        do {
            let store = try VectorStore(databaseURL: resolveVectorStoreURL(from: args))
            let engine = makeEmbeddingEngine(computeOptions: computeOptions)
            let coordinator = IndexCoordinator(engine: engine, store: store)
            let url = URL(fileURLWithPath: targetPath)
            print("Indexing \(url.path) using \(engine.modelID) (sampling: \(sampling.label), soundtrack: \(includeAudio))...")
            let result = try await coordinator.indexURLs(
                [url],
                videoWindowSampling: sampling,
                loadAudioFromVideo: includeAudio
            ) { progress in
                print("[\(progress.processed)/\(progress.total)] \(progress.currentTitle)")
            }
            let stats = try store.stats()
            print("Done! Indexed \(result.indexed) artifacts (\(result.skipped.count) skipped). Total in store: \(stats.artifactCount) artifacts (\(stats.chunkCount) chunks).")
            for kind in ArtifactKind.allCases {
                let count = stats.countsByKind[kind] ?? 0
                if count > 0 {
                    print("  \(kind.displayName): \(count)")
                }
            }
        } catch {
            print("Index error: \(error)")
        }
    }

    static func runSearch(args: [String]) async {
        let queryParts = args.filter { !$0.hasPrefix("--") }
        let query = queryParts.first ?? "Apple Neural Engine co-residency"
        var dim = 768
        if let dimFlagIdx = args.firstIndex(of: "--dim"), dimFlagIdx + 1 < args.count,
           let parsedDim = Int(args[dimFlagIdx + 1]) {
            dim = parsedDim
        }
        var taskOverride: EmbeddingTask? = nil
        if let taskIdx = args.firstIndex(of: "--task"), taskIdx + 1 < args.count {
            let val = args[taskIdx + 1].lowercased()
            switch val {
            case "similarity", "sts": taskOverride = .sts
            case "code": taskOverride = .codeRetrieval
            case "qa": taskOverride = .questionAnswering
            case "fact": taskOverride = .factVerification
            default: taskOverride = .query
            }
        }
        var mediaURL: URL? = nil
        if let mediaIdx = args.firstIndex(of: "--media"), mediaIdx + 1 < args.count {
            mediaURL = URL(fileURLWithPath: args[mediaIdx + 1])
        }
        let computeOptions = parseComputeOptions(from: args)

        do {
            let storeURL = resolveVectorStoreURL(from: args)
            let store = try VectorStore(databaseURL: storeURL)
            let engine = makeEmbeddingEngine(computeOptions: computeOptions)
            let coordinator = IndexCoordinator(engine: engine, store: store)

            let stats = try store.stats()
            if stats.artifactCount == 0 {
                // Seed a sample span if empty so `heylook-cli search` always demonstrates retrieval out of the box
                let snippet = "Unified MLX GPU inference + Apple Neural Engine (ANE) co-residency."
                let doc = Artifact(
                    id: "vision-doc",
                    collectionID: "default",
                    kind: .text,
                    url: URL(fileURLWithPath: "VISION.md"),
                    title: "VISION.md",
                    fingerprint: ContentFingerprint(byteCount: Int64(snippet.utf8.count), modifiedUnixMs: 1_700_000_000_000)
                )
                let outputs = try await engine.embedBatch(
                    [EmbeddingInput(parts: [.text(snippet)], task: .document, title: doc.title)],
                    dimensions: 768
                )
                if let vec = outputs.first?.vector {
                    let chunk = EmbeddedChunk(id: "vision-doc#0", artifactID: doc.id, span: .whole, vector: vec, previewText: snippet)
                    try store.upsert(artifact: doc, chunks: [chunk])
                }
            }

            let hits = try await coordinator.search(
                queryText: query,
                exampleFileURL: mediaURL,
                dimensions: dim,
                taskOverride: taskOverride,
                limit: 15
            )
            print("Top \(hits.count) results for \"\(query)\" (\(dim)d via \(engine.modelID)):")
            for (idx, hit) in hits.enumerated() {
                let badge = hit.bestSpan.displayBadge.map { " [\($0)]" } ?? ""
                let feat = " {\(hit.bestScoredFeature.title)}"
                let multi = hit.hasMultiFeatureMatch ? " (+Audio&Vision)" : ""
                print(String(format: "%2d. [%.3f] [%@]%@%@%@ %@ — %@", idx + 1, hit.bestScore, hit.artifact.kind.rawValue, badge, feat, multi, hit.artifact.title, hit.artifact.url.path))
                if let mScores = hit.spanMatches.first?.scoresByDimension, !mScores.isEmpty {
                    let formatted = [128, 256, 512, 768].compactMap { d in
                        mScores[d].map { String(format: "%dd:%.3f", d, $0) }
                    }.joined(separator: " | ")
                    print("    Matryoshka: \(formatted)")
                }
            }
        } catch {
            print("Search error: \(error)")
        }
    }

    static func runWatched() {
        do {
            let store = try VectorStore(databaseURL: defaultVectorStoreURL())
            let folders = try store.listWatchedFolders()
            if folders.isEmpty {
                print("No watched folders configured.")
            } else {
                print("Watched folders (\(folders.count)):")
                for f in folders {
                    print("  • \(f.path) (sampling: \(f.windowSampling.label))")
                }
            }
        } catch {
            print("Error listing watched folders: \(error)")
        }
    }

    static func registerConfiguredModels(into scheduler: ComputeScheduler) async -> [EngineDescription] {
        let cfg = HeylookConfig.loadDefault()
        let folderURLs = cfg.scanFolders.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        let found = HeylookConfig.discoverModels(in: folderURLs)
        for desc in found {
            let provider: any InferenceProvider
            switch desc.provider {
            case .mlx:
                provider = MLXSwiftProvider(description: desc)
            case .gguf:
                provider = GGUFSubprocessProvider(description: desc)
            case .coreai:
                provider = CoreAIProvider(description: desc)
            case .remote:
                provider = RemoteHeylookProvider(baseURL: URL(string: "http://127.0.0.1:8000")!, description: desc)
            }
            await scheduler.register(provider: provider, description: desc)
        }
        return found
    }

    static func runChat(args: [String]) async {
        var requestedModel: String? = ProcessInfo.processInfo.environment["HEYLOOK_MODEL"]
        var maxTokens = 256
        var thinking = true
        var promptParts: [String] = []

        var i = 0
        while i < args.count {
            let arg = args[i]
            if arg == "--model" || arg == "-m", i + 1 < args.count {
                requestedModel = args[i + 1]
                i += 2
            } else if arg == "--max-tokens", i + 1 < args.count, let n = Int(args[i + 1]) {
                maxTokens = n
                i += 2
            } else if arg == "--no-thinking" {
                thinking = false
                i += 1
            } else if arg == "--thinking" {
                thinking = true
                i += 1
            } else {
                promptParts.append(arg)
                i += 1
            }
        }
        let prompt = promptParts.isEmpty
            ? "Explain unified memory + ANE co-residency in two sentences."
            : promptParts.joined(separator: " ")

        let scheduler = ComputeScheduler()
        let discovered = await registerConfiguredModels(into: scheduler)
        let cfg = HeylookConfig.loadDefault()

        let selectedDesc: EngineDescription
        if let reqModel = requestedModel, !reqModel.isEmpty {
            if let exact = discovered.first(where: { $0.modelID.caseInsensitiveCompare(reqModel) == .orderedSame })
                ?? discovered.first(where: { $0.modelID.localizedCaseInsensitiveContains(reqModel) }) {
                selectedDesc = exact
            } else if FileManager.default.fileExists(atPath: reqModel) {
                let dirURL = URL(fileURLWithPath: reqModel)
                let adHoc = HeylookConfig.discoverModels(in: [dirURL]).first ?? EngineDescription(
                    modelID: dirURL.lastPathComponent,
                    provider: .mlx,
                    modelPath: dirURL.path,
                    architecture: ProvenanceField(value: "mlx", provenance: .derived),
                    quantization: ProvenanceField(value: "bf16", provenance: .derived),
                    computePlacement: ProvenanceField(value: .mlxGPU, provenance: .derived),
                    contextSize: ProvenanceField(value: 32768, provenance: .derived),
                    flashAttention: ProvenanceField(value: true, provenance: .derived),
                    supportsVision: ProvenanceField(value: false, provenance: .derived),
                    supportsThinking: ProvenanceField(value: true, provenance: .derived)
                )
                let prov = MLXSwiftProvider(description: adHoc)
                await scheduler.register(provider: prov, description: adHoc)
                selectedDesc = adHoc
            } else {
                print("Model '\(reqModel)' not found among \(discovered.count) discovered models.")
                return
            }
        } else if let defMatch = discovered.first(where: { $0.modelID == cfg.defaultModel })
            ?? discovered.first(where: { $0.provider == .mlx })
            ?? discovered.first {
            selectedDesc = defMatch
        } else {
            let fallback = EngineDescription(
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
            await scheduler.register(provider: MLXSwiftProvider(description: fallback), description: fallback)
            selectedDesc = fallback
        }

        let registry = RequestRegistry()
        let req = MessagesRequest(
            model: selectedDesc.modelID,
            messages: [MessagesRequest.MessageTurn(role: "user", content: prompt)],
            maxTokens: maxTokens,
            stream: true,
            thinking: thinking
        )
        let vendorSamplers = HeylookConfig.loadVendorSampling(in: URL(fileURLWithPath: selectedDesc.modelPath))
        let resolved = SamplerCascade.resolve(vendor: vendorSamplers, request: req.samplerOverlay)
        do {
            let provider = try await scheduler.provider(for: selectedDesc.modelID)
            let lastChunk = try await scheduler.withComputeLease(for: selectedDesc.modelID) { () throws -> GenerationChunk? in
                let stream = await provider.generate(
                    requestID: "cli_req_1",
                    request: req,
                    effectiveSampling: resolved.effective,
                    registry: registry
                )
                var latest: GenerationChunk?
                for try await chunk in stream {
                    latest = chunk
                    if !chunk.thinking.isEmpty {
                        FileHandle.standardError.write(Data(chunk.thinking.utf8))
                    }
                    if !chunk.text.isEmpty {
                        FileHandle.standardOutput.write(Data(chunk.text.utf8))
                    }
                }
                print("")
                return latest
            }
            if let final = lastChunk {
                let cacheStr = final.cacheReport.map { "\($0.outcome.rawValue)(\($0.cachedTokens)/\($0.promptTokens))" } ?? "n/a"
                let statsLine = String(
                    format: "[%@] %.1f tok/s | prompt: %d tok (%.1f tok/s) | output: %d tok | cache: %@ | compute: %@\n",
                    selectedDesc.modelID,
                    final.generationTPS,
                    final.promptTokens,
                    final.promptTPS,
                    final.tokenCount,
                    cacheStr,
                    final.activeComputePlacement.label
                )
                FileHandle.standardError.write(Data(statsLine.utf8))
            }
        } catch {
            print("Chat error: \(error)")
        }
    }

    static func runServe(port: UInt16) async {
        do {
            let dbURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("heylook_serve.sqlite")
            let store = try SQLiteStore(databaseURL: dbURL)
            let scheduler = ComputeScheduler()
            let registered = await registerConfiguredModels(into: scheduler)
            let router = HeylookRouter(store: store, scheduler: scheduler)
            let server = HeylookHTTPServer(router: router, host: "127.0.0.1", port: port)
            try await server.start()
            print("Heylook Server listening on http://127.0.0.1:\(port) (\(registered.count) models registered)")
            try await Task.sleep(for: .seconds(3600 * 24))
        } catch {
            print("Server error: \(error)")
        }
    }

    static func printUsage() {
        print("""
        Usage: heylook-cli <command> [args]
        Commands:
          doctor | status                               Inspect unified memory, compute lanes, and vector store
          models                                        Discover LLM/VLM models from configured folders
          index <path> [--sampling fps|content|off] [--no-audio-from-video] [--compute all_gpu|hybrid_ane|all_ane]
                                                        Index files or folders into the multimodal vector store
          search <query> [--dim 768|512|256|128] [--task query|similarity|code|qa|fact] [--media <path>]
                                                        Search across indexed spans with Matryoshka cosine attribution
          watched                                       List FSEvents-watched folders in the vector store
          inspect-ane [dir]                             Inspect CoreML ANE tower packages and op compatibility
          inspect-checkpoint [dir]                      Inspect local EmbeddingGemma-2-2B safetensors checkpoint
          chat <prompt>                                 Stream a chat completion with thinking traces
          serve [port]                                  Run the embedded /v1/messages & /v1/embeddings server
        """)
    }
}
