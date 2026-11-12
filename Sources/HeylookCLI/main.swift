import Darwin
import Dispatch
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
            let port = args.dropFirst().first.flatMap { UInt16($0) }
            await runServe(portOverride: port)
        case "chat":
            await runChat(args: Array(args.dropFirst()))
        case "transcribe":
            await runTranscribe(args: Array(args.dropFirst()))
        case "eval-lab":
            await runEvalLab(args: Array(args.dropFirst()))
        case "convert-mlx":
            await runConvertMLX(args: Array(args.dropFirst()))
        default:
            printUsage()
        }
    }

    static func defaultVectorStoreURL() -> URL {
        HeylookConfig.defaultVectorStoreURL
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
        NativeMLXEmbeddingGemma2.makeBestAvailableEngine(
            computeOptions: computeOptions,
            allowSyntheticFallback: false
        )
    }

    static func runDoctor() async {
        let snap = HardwareMemoryMonitor.snapshot()
        let storeURL = defaultVectorStoreURL()
        print("=== Heylook Native macOS 27.2 / iOS 27.2 Doctor ===")
        print(String(format: "Device              : %@", snap.deviceName))
        print(
            String(
                format: "Host Unified Memory : %.1f GB (Used: %.1f GB, Available: %.1f GB, Metal Max: %.1f GB)",
                snap.totalPhysicalRAMGB,
                snap.ramUsedGB,
                snap.ramAvailableGB,
                snap.metalRecommendedMaxWorkingSetGB
            )
        )
        print("Swift Concurrency   : Swift 6.2 Strict Actor Isolation")
        print("Compute Lanes       : GPU GenerationGate (FIFO) + ANE Co-Residency Lane")
        print("Supported Providers : MLXSwiftProvider, CoreAIProvider (.aimodel), GGUFSubprocessProvider, RemoteHeylookProvider")
        let doctorScheduler = ComputeScheduler()
        _ = await registerConfiguredModels(into: doctorScheduler)
        let asrSummary = await doctorScheduler.voiceASRSummary()
        let asrStatusText = asrSummary.available
            ? "Available (default=\(asrSummary.defaultModel ?? "none"), backend=\(asrSummary.backend), models=[\(asrSummary.models.joined(separator: ", "))])"
            : "Unavailable"
        print("Voice ASR Engine    : \(asrStatusText)")
        print("Embeddings Engine   : EmbeddingGemma 2 + vDSP Matryoshka (768d / 512d / 256d / 128d)")
        print("Store Engine        : SQLite3 WAL (Conversations, Messages, MediaBlobs, Stats, Notebooks, Presets, VectorSpans)")
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
        let found = cfg.discoverConfiguredModels()
        print("Discovered \(found.count) models across \(cfg.scanFolders.count) configured folders (\(cfg.explicitModels.count) explicit entries in heylook.toml).")
        for m in found {
            let mtpBadge = m.tags.contains("mtp") ? " [MTP]" : ""
            let asrBadge = m.isSpeechRecognizer ? " [ASR]" : ""
            let visionBadge = m.supportsVision.value ? " [vision]" : ""
            let audioBadge = m.supportsAudio.value ? " [audio]" : ""
            let depthBadge: String
            if let depthVar = m.thinkingControls?.depthVariable {
                depthBadge = " depth=\(depthVar)(\(m.thinkingControls?.depthValues.joined(separator: "|") ?? ""))"
            } else {
                depthBadge = ""
            }
            print("- [\(m.provider.rawValue)] \(m.modelID)\(mtpBadge)\(asrBadge)\(visionBadge)\(audioBadge) ctx=\(m.contextSize.value) mem=\(m.estimatedMemoryMB)MB reasoning=\(m.reasoningFormat.value)\(depthBadge) template=\(m.templateSource.value) (\(m.modelPath))")
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
            exit(1)
        }
        var sampling: VideoWindowSampling = .fps
        if let sIdx = args.firstIndex(of: "--sampling"), sIdx + 1 < args.count {
            let val = args[sIdx + 1].lowercased()
            if val == "fps" { sampling = .fps }
            else if val == "content" { sampling = .content }
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
            print("Index error: \(error.localizedDescription)")
            exit(1)
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
                print("Vector store at \(storeURL.path) is empty (0 indexed artifacts). Run 'heylook-cli index <path>' first.")
                return
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
            print("Search error: \(error.localizedDescription)")
            exit(1)
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
        let found = cfg.discoverConfiguredModels()
        await scheduler.registerDiscoveredModels(found, allowSyntheticFallback: false)
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
                let prov = ProviderFactory.makeProvider(for: adHoc, allowSyntheticFallback: false)
                await scheduler.register(provider: prov, description: adHoc)
                selectedDesc = adHoc
            } else {
                print("Model '\(reqModel)' not found among \(discovered.count) discovered models.")
                exit(1)
            }
        } else if let defMatch = discovered.first(where: { $0.modelID == cfg.defaultModel })
            ?? discovered.first(where: { $0.provider == .mlx })
            ?? discovered.first {
            selectedDesc = defMatch
        } else {
            print("No models discovered in configured scan_folders (\(cfg.scanFolders.joined(separator: ", "))). Pass '--model <path>' or configure heylook.toml.")
            exit(1)
        }

        let registry = RequestRegistry()
        let req = MessagesRequest(
            model: selectedDesc.modelID,
            messages: [MessagesRequest.MessageTurn(role: "user", content: prompt)],
            maxTokens: maxTokens,
            stream: true,
            thinking: thinking
        )
        let resolved = HeylookConfig.resolveEffectiveSampling(
            modelPath: selectedDesc.modelPath,
            request: req.samplerOverlay
        )
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
            print("Chat error: \(error.localizedDescription)")
            await scheduler.unloadAllModels()
            exit(1)
        }
        await scheduler.unloadAllModels()
    }

    static func runServe(portOverride: UInt16?) async {
        do {
            let cfg = HeylookConfig.loadDefault()
            let port = portOverride ?? UInt16(cfg.serverPort)
            let dbURL = HeylookConfig.defaultStoreURL
            let store = try SQLiteStore(databaseURL: dbURL)
            let scheduler = ComputeScheduler(
                maxLoadedGPUModels: cfg.maxLoadedModels,
                maxQueueDepth: cfg.maxQueueDepth
            )
            let registered = await registerConfiguredModels(into: scheduler)
            let embeddingEngine = makeEmbeddingEngine(computeOptions: .hybridANETowers)
            let router = HeylookRouter(
                store: store,
                scheduler: scheduler,
                embeddingEngine: embeddingEngine,
                bindHost: cfg.serverHost,
                allowedHosts: cfg.allowedHosts,
                adminToken: cfg.adminToken ?? ProcessInfo.processInfo.environment["HEYLOOK_ADMIN_TOKEN"],
                allowSyntheticFallback: false
            )
            let server = HeylookHTTPServer(router: router, host: cfg.serverHost, port: port)
            try await server.start()
            print("Heylook Server listening on http://\(cfg.serverHost):\(port) (\(registered.count) models registered, store: \(dbURL.path))")

            // Block indefinitely on SIGINT / SIGTERM via DispatchSourceSignal (no 24-hour exit bug!)
            await waitForShutdownSignal()
            print("\nShutting down Heylook Server gracefully...")
            await server.stop()
            SubprocessLifecycleRegistry.shared.terminateAll()
        } catch {
            print("Server error: \(error.localizedDescription)")
            exit(1)
        }
    }

    private static func waitForShutdownSignal() async {
        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)

        final class ResumeOnce: @unchecked Sendable {
            private let lock = NSLock()
            private var resumed = false
            private var sources: [any DispatchSourceSignal] = []

            func retain(_ s: [any DispatchSourceSignal]) {
                lock.lock()
                sources = s
                lock.unlock()
            }

            func finish(_ cont: CheckedContinuation<Void, Never>) {
                lock.lock()
                guard !resumed else {
                    lock.unlock()
                    return
                }
                resumed = true
                let active = sources
                sources.removeAll()
                lock.unlock()
                for src in active {
                    src.cancel()
                }
                cont.resume()
            }
        }

        let guardBox = ResumeOnce()
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global(qos: .utility))
            let sigtermSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global(qos: .utility))
            guardBox.retain([sigintSource, sigtermSource])
            sigintSource.setEventHandler {
                guardBox.finish(cont)
            }
            sigtermSource.setEventHandler {
                guardBox.finish(cont)
            }
            sigintSource.resume()
            sigtermSource.resume()
        }
    }

    static func runEvalLab(args: [String]) async {
        let syntheticSelfTest = args.contains("--synthetic-self-test")
        print("=== Heylook Evaluation Lab (chain_probe + vlm_parity_probe + matryoshka_recall_probe) ===")
        do {
            let report = try await EvalLabSuite.runAll(allowSyntheticSelfTest: syntheticSelfTest)
            for probe in report.probes {
                let badge = probe.passed ? "PASS" : "FAIL"
                print("- [\(badge)] \(probe.name.padding(toLength: 24, withPad: " ", startingAt: 0)) : \(probe.detail)")
            }
            if !report.matryoshkaBenchmarks.isEmpty {
                print("--- Matryoshka Representation Learning (MRL) Recall@1 / Recall@3 / MRR ---")
                for bench in report.matryoshkaBenchmarks {
                    print(
                        String(
                            format: "  %4dd | Recall@1=%.2f | Recall@3=%.2f | MRR=%.2f",
                            bench.dimensions,
                            bench.recallAt1,
                            bench.recallAt3,
                            bench.mrr
                        )
                    )
                }
            }
            if !report.allPassed {
                exit(1)
            }
        } catch {
            print("Eval lab error: \(error.localizedDescription)")
            exit(1)
        }
    }

    static func runConvertMLX(args: [String]) async {
        let positional = args.filter { !$0.hasPrefix("--") }
        guard positional.count >= 2 else {
            print("Usage: heylook-cli convert-mlx <src-dir> <dst-dir> [--bits 4|8] [--group-size 32|64] [--mode affine|mxfp4|mxfp8|nvfp4] [--calibration standard|q4_zero] [--overwrite] [--dry-run]")
            exit(1)
        }
        let srcURL = URL(fileURLWithPath: positional[0])
        let dstURL = URL(fileURLWithPath: positional[1])

        var bits: Int? = nil
        if let bIdx = args.firstIndex(of: "--bits"), bIdx + 1 < args.count {
            bits = Int(args[bIdx + 1])
        }
        var groupSize: Int? = nil
        if let gIdx = args.firstIndex(of: "--group-size"), gIdx + 1 < args.count {
            groupSize = Int(args[gIdx + 1])
        }
        var mode: MLXQuantizationMode = .affine
        if let mIdx = args.firstIndex(of: "--mode"), mIdx + 1 < args.count,
           let parsed = MLXQuantizationMode(rawValue: args[mIdx + 1].lowercased()) {
            mode = parsed
        }
        var calibration: MLXQuantizationCalibration = .standard
        if let cIdx = args.firstIndex(of: "--calibration"), cIdx + 1 < args.count,
           let parsed = MLXQuantizationCalibration(rawValue: args[cIdx + 1].lowercased()) {
            calibration = parsed
        }
        let overwrite = args.contains("--overwrite")
        let dryRun = args.contains("--dry-run")

        let spec = MLXConversionSpec(
            bits: bits,
            groupSize: groupSize,
            mode: mode,
            calibration: calibration,
            overwriteExistingOutput: overwrite
        )

        do {
            let plan = try MLXModelConverter.planConversion(from: srcURL, to: dstURL, spec: spec)
            print("=== Pure-Swift MLX Model Conversion Plan ===")
            print("  Source      : \(plan.sourceDirectory.path)")
            print("  Output      : \(plan.outputDirectory.path)")
            print("  Model Type  : \(plan.modelType) (\(plan.isVisionModel ? "VLM" : "LLM"), reasoning=\(plan.reasoningFormat))")
            print("  Quantization: \(plan.resolvedBits)-bit \(plan.mode.rawValue) (group_size=\(plan.resolvedGroupSize), calibration=\(plan.calibration.rawValue))")
            print("  Safetensors : \(plan.safetensorsFiles.joined(separator: ", "))")
            print("  Sidecars    : \(plan.sidecarFiles.joined(separator: ", "))")

            if dryRun {
                return
            }

            let (_, writtenURLs) = try await MLXModelConverter.convert(
                from: srcURL,
                to: dstURL,
                spec: spec
            ) { status in
                print("  \(status)")
            }
            print("Conversion complete! Wrote \(writtenURLs.count) shard(s) to \(plan.outputDirectory.path)")
        } catch {
            print("MLX conversion error: \(error.localizedDescription)")
            exit(1)
        }
    }

    static func runTranscribe(args: [String]) async {
        var audioPath: String?
        var requestedModel: String?
        var languageHint = "en"

        var i = 0
        while i < args.count {
            let arg = args[i]
            if (arg == "--model" || arg == "-m"), i + 1 < args.count {
                requestedModel = args[i + 1]
                i += 2
            } else if (arg == "--lang" || arg == "-l"), i + 1 < args.count {
                languageHint = args[i + 1]
                i += 2
            } else if !arg.hasPrefix("-") && audioPath == nil {
                audioPath = arg
                i += 1
            } else {
                i += 1
            }
        }

        guard let audioPath, !audioPath.isEmpty else {
            print("Usage: heylook-cli transcribe <audio-path> [--model <id_or_path>] [--lang <locale>]")
            exit(1)
        }

        let audioURL = URL(fileURLWithPath: (audioPath as NSString).expandingTildeInPath)
        guard let wavData = try? Data(contentsOf: audioURL) else {
            print("Error: unable to read audio file at \(audioURL.path)")
            exit(1)
        }

        let scheduler = ComputeScheduler()
        _ = await registerConfiguredModels(into: scheduler)

        var preferredModelID = requestedModel
        if let reqModel = requestedModel, !reqModel.isEmpty {
            let expanded = (reqModel as NSString).expandingTildeInPath
            if FileManager.default.fileExists(atPath: expanded) {
                let dirURL = URL(fileURLWithPath: expanded)
                let arch = HeylookConfig.inspectSpeechRecognizerArchitecture(in: dirURL) ?? "speech_recognizer"
                let adHocDesc = EngineDescription(
                    modelID: CoreAIBundleManifest.canonicalBundleID(for: dirURL),
                    provider: .coreai,
                    modelPath: dirURL.path,
                    architecture: ProvenanceField(value: arch, provenance: .derived, effect: .reload),
                    quantization: ProvenanceField(value: "int4", provenance: .derived, effect: .reload),
                    computePlacement: ProvenanceField(value: .ane, provenance: .derived, effect: .reload, computePlacement: .ane),
                    supportsVision: ProvenanceField(value: false, provenance: .derived, effect: .reload),
                    supportsAudio: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                    supportsThinking: ProvenanceField(value: false, provenance: .derived, effect: .perRequest),
                    reasoningFormat: ProvenanceField(value: "none", provenance: .derived, effect: .reload),
                    templateSource: ProvenanceField(value: "speech_recognizer", provenance: .derived, effect: .reload),
                    tags: ["asr", "speech"]
                )
                let prov = CoreAIProvider(description: adHocDesc, allowSyntheticFallback: false)
                await scheduler.register(provider: prov, description: adHocDesc)
                preferredModelID = adHocDesc.modelID
            }
        }

        do {
            let result = try await scheduler.transcribeVoice(
                wavData: wavData,
                languageHint: languageHint,
                preferredASRModelID: preferredModelID,
                allowSyntheticFallback: false
            )
            print("=== Heylook ANE Speech Transcription ===")
            print(String(format: "  Model     : %@ (backend=%@, placement=%@)", result.modelID, result.backend, result.computePlacement.rawValue))
            print(String(format: "  Audio     : %.2fs @ %d Hz (rms=%.4f, mel_frames=%d)", result.durationSeconds, result.sampleRateHz, result.rmsEnergy, result.melFrames))
            print("  Transcript: \(result.text.isEmpty ? "(no speech detected)" : result.text)")
        } catch {
            print("Transcription error: \(error.localizedDescription)")
            exit(1)
        }
    }

    static func printUsage() {
        print("""
        Usage: heylook-cli <command> [args]
        Commands:
          doctor | status                               Inspect unified memory, compute lanes, ASR, and vector store
          models                                        Discover LLM/VLM/ASR models from configured folders
          index <path> [--sampling fps|content|off] [--no-audio-from-video] [--compute all_gpu|hybrid_ane|all_ane]
                                                        Index files or folders into the multimodal vector store
          search <query> [--dim 768|512|256|128] [--task query|similarity|code|qa|fact] [--media <path>]
                                                        Search across indexed spans with Matryoshka cosine attribution
          watched                                       List FSEvents-watched folders in the vector store
          inspect-ane [dir]                             Inspect CoreML ANE tower packages and op compatibility
          inspect-checkpoint [dir]                      Inspect local EmbeddingGemma-2-2B safetensors checkpoint
          convert-mlx <src-dir> <dst-dir> [--bits 4|8] [--group-size 32|64] [--mode affine|mxfp4|mxfp8|nvfp4] [--calibration standard|q4_zero] [--overwrite] [--dry-run]
                                                        Quantize & shard a safetensors LLM/VLM checkpoint in pure Swift
          transcribe <audio-path> [--model <id_or_path>] [--lang <locale>]
                                                        Transcribe WAV/M4A/FLAC speech on the Apple Neural Engine
          eval-lab [--synthetic-self-test]              Run multi-turn KV chain, VLM parity, and MRL Recall@K probes
          chat <prompt>                                 Stream a chat completion with thinking traces
          serve [port]                                  Run the embedded /v1/messages & /v1/embeddings server
        """)
    }
}
