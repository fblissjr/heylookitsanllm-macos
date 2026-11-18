import Darwin
import Foundation
import Testing
@testable import HeylookCore
@testable import HeylookEmbeddings
@testable import HeylookEngines
@testable import HeylookServer

@Suite("Multimodal Embeddings, Tokenizer, Model Specialization, GGUF Spawn & RAM Fit Suite")
struct EmbeddingsAndSpecializationTests {

    @Test("POST /v1/embeddings enforces model routing (422/404/400/409), prompt_name validation, GGUF windows refusal, and single/multi windows.seconds planning")
    func testEmbeddingsRoutesValidationWindowsAndCancellation() async throws {
        let embEngine = TestDeterministicEmbeddingEngine(
            modelID: "custom-emb-mlx",
            knownPrompts: ["Retrieval-query", "Retrieval-document", "STS"],
            maxWindowSeconds: 32.0,
            simulatedVideoDurationSeconds: 10.5
        )
        let harness = try TestRouterHarness(embeddingEngine: embEngine)

        // Register a non-embedding chat model + a GGUF embedding model backed by a real minimal GGUF header (for GGUF `windows` 400 test)
        _ = await harness.registerStub(id: "chat-only-mlx", provider: .mlx, supportsEmbeddings: false)
        let ggufEmbDir = harness.sandbox.modelsDir.appendingPathComponent("embed-gguf", isDirectory: true)
        try FileManager.default.createDirectory(at: ggufEmbDir, withIntermediateDirectories: true)
        try GGUFHeaderReader.buildMinimalGGUF(
            version: 3,
            tensorCount: 4,
            stringMetadata: ["general.architecture": "gemma-embedding"],
            uint32Metadata: ["gemma-embedding.embedding_length": 768, "gemma-embedding.context_length": 8192]
        ).write(to: ggufEmbDir.appendingPathComponent("embed.gguf"))
        _ = await harness.registerStub(
            id: "embed-gguf",
            provider: .gguf,
            modelPath: ggufEmbDir.path,
            supportsVision: true,
            supportsEmbeddings: true,
            isLoaded: false
        )

        // 1. Missing or empty `model` -> 422; unknown `model` -> 404; chat model -> 400
        let missingModel = try await harness.sendJSON("POST", "/v1/embeddings", object: ["input": "hello"])
        #expect(missingModel.status == 422)

        let emptyModel = try await harness.sendJSON("POST", "/v1/embeddings", object: ["model": "   ", "input": "hello"])
        #expect(emptyModel.status == 422)

        let unknownModel = try await harness.sendJSON("POST", "/v1/embeddings", object: ["model": "does-not-exist", "input": "hello"])
        #expect(unknownModel.status == 404)

        let chatModel = try await harness.sendJSON("POST", "/v1/embeddings", object: ["model": "chat-only-mlx", "input": "hello"])
        #expect(chatModel.status == 400)

        // 2. Unknown `prompt_name` -> 400
        let badPrompt = try await harness.sendJSON(
            "POST",
            "/v1/embeddings",
            object: [
                "model": "custom-emb-mlx",
                "input": [["content": [["type": "text", "text": "hello"]], "prompt_name": "InvalidTaskName"]],
            ]
        )
        #expect(badPrompt.status == 400)

        // 3. GGUF embedding model rejects video `windows` cleanly with 400
        let dummyVideoB64 = Data("FAKE_MP4_BYTES_FOR_VALIDATION".utf8).base64EncodedString()
        let ggufWindows = try await harness.sendJSON(
            "POST",
            "/v1/embeddings",
            object: [
                "model": "embed-gguf",
                "input": [[
                    "content": [[
                        "type": "video",
                        "source": ["type": "base64", "media_type": "video/mp4", "data": dummyVideoB64],
                        "windows": ["seconds": 8.0, "sampling": "fps"],
                    ]],
                ]],
            ]
        )
        #expect(ggufWindows.status == 400)

        // 4. Short <= 16s clip (`10.5s`) returns a single window (1 window!) rather than dropping sub-16s clips
        let shortClipResp = try await harness.sendJSON(
            "POST",
            "/v1/embeddings",
            object: [
                "model": "custom-emb-mlx",
                "dimensions": 256,
                "input": [[
                    "prompt_name": "Retrieval-document",
                    "content": [[
                        "type": "video",
                        "source": ["type": "base64", "media_type": "video/mp4", "data": dummyVideoB64],
                        "windows": ["seconds": 16.0, "sampling": "fps"],
                    ]],
                ]],
            ]
        )
        #expect(shortClipResp.status == 200)
        let shortJSON = try TestFixtures.decodeJSONObject(shortClipResp.body)
        #expect(shortJSON["dimensions"] as? Int == 256)
        let row0 = try #require((shortJSON["data"] as? [[String: Any]])?.first)
        let windows0 = try #require(row0["windows"] as? [[String: Any]])
        #expect(windows0.count == 1)
        #expect(windows0[0]["start"] as? Double == 0.0)
        #expect(windows0[0]["end"] as? Double == 10.5)

        // 5. WindowPlanner consecutive multi-window ranges (`45s` @ `16s` -> 3 non-overlapping windows)
        //    plus <= 16s short clip single-window (`12s` @ `32s` -> 1 window)
        let shortTimes = WindowPlanner.uniformSampleTimes(duration: 12.0, fps: 1.0, maxFrames: 32)
        let singlePlanned = WindowPlanner.planWindows(times: shortTimes, duration: 12.0, windowSeconds: 32.0)
        #expect(singlePlanned.count == 1)
        #expect(singlePlanned[0].start == 0.0)
        #expect(singlePlanned[0].end == 12.0)

        let longRanges = WindowPlanner.consecutiveRanges(duration: 45.0, windowSeconds: 16.0)
        #expect(longRanges.count == 3)
        #expect(longRanges[0].start == 0.0 && longRanges[0].end == 16.0)
        #expect(longRanges[1].start == 16.0 && longRanges[1].end == 32.0)
        #expect(longRanges[2].start == 32.0 && longRanges[2].end == 45.0)

        // 6. Active /v1/embeddings request cancelled via DELETE /v1/requests/{id} returns 409 cancelled
        let slowEmb = TestDeterministicEmbeddingEngine(
            modelID: "slow-emb-mlx",
            perItemDelayNanoseconds: 400_000_000
        )
        let cancelHarness = try TestRouterHarness(embeddingEngine: slowEmb)
        let reqID = "req_emb_cancel_probe"
        let embTask = Task {
            try await cancelHarness.sendJSON(
                "POST",
                "/v1/embeddings",
                headers: ["X-Request-ID": reqID],
                object: ["model": "slow-emb-mlx", "input": "cancel me"]
            )
        }
        var registered = false
        for _ in 0..<40 {
            if !cancelHarness.registry.activeRequests().isEmpty {
                registered = true
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        #expect(registered)
        let delResp = await cancelHarness.send("DELETE", "/v1/requests/\(reqID)")
        #expect(delResp.status == 200)
        let embResultResp = try await embTask.value
        #expect(embResultResp.status == 409)
    }

    @Test("CheckpointPieceTokenizer Viterbi Unigram, Matryoshka VectorStore model_id invalidation, and IndexCoordinator corpus lifecycle")
    func testTokenizerMatryoshkaVectorStoreAndIndexCoordinator() async throws {
        let sandbox = try TestTempSandbox(prefix: "heylook_vec")
        defer { sandbox.cleanup() }

        // 1. Viterbi Unigram CheckpointPieceTokenizer
        let tokenizerJSON = """
        {
          "model": {
            "type": "Unigram",
            "vocab": [
              ["<pad>", 0.0],
              ["<eos>", 0.0],
              ["<bos>", 0.0],
              ["<unk>", 0.0],
              ["▁apple", -2.1],
              ["▁neural", -2.3],
              ["▁engine", -2.2],
              ["▁", -4.0],
              ["a", -6.0],
              ["p", -6.0],
              ["l", -6.0],
              ["e", -6.0]
            ]
          }
        }
        """
        try tokenizerJSON.write(
            to: sandbox.modelsDir.appendingPathComponent("tokenizer.json"),
            atomically: true,
            encoding: .utf8
        )
        let pieceTok = try #require(CheckpointPieceTokenizer.load(from: sandbox.modelsDir))
        #expect(pieceTok.bosTokenID == 2)
        #expect(pieceTok.eosTokenID == 1)
        #expect(pieceTok.encode(" apple neural engine") == [4, 5, 6])

        // 2. Matryoshka L2 projections (`768/512/256/128`) & VectorStore `model_id` invalidation
        let store = try VectorStore(databaseURL: sandbox.vectorDBURL)
        let engineA = TestDeterministicEmbeddingEngine(modelID: "emb-model-a")
        let vec = engineA.embed(text: "Apple Neural Engine Matryoshka slices")
        for dim in [768, 512, 256, 128] {
            let proj = Matryoshka.project(vec, dimensions: dim)
            #expect(proj.count == dim)
            let norm = sqrt(proj.reduce(0) { $0 + $1 * $1 })
            #expect(abs(norm - 1.0) < 1e-4)
        }

        let corpusDir = sandbox.rootURL.appendingPathComponent("corpus", isDirectory: true)
        try FileManager.default.createDirectory(at: corpusDir, withIntermediateDirectories: true)
        let docURL = corpusDir.appendingPathComponent("note.md")
        try "Apple Neural Engine co-residency keeps vision towers hot.".write(to: docURL, atomically: true, encoding: .utf8)
        let coordinator = IndexCoordinator(engine: engineA, store: store)
        let res1 = try await coordinator.indexURLs([corpusDir])
        #expect(res1.indexed == 1)

        // Second pass skips unchanged file under the same modelID
        let res2 = try await coordinator.indexURLs([corpusDir])
        #expect(res2.indexed == 0)

        // Model ID change invalidates stored vectors cleanly via `isUpToDate(..., modelID:)`
        let fp = ContentFingerprint(byteCount: 54, modifiedUnixMs: 1_700_000_000_000)
        let art = Artifact(id: "art-model-check", collectionID: "default", kind: .text, url: docURL, title: "note", fingerprint: fp)
        try store.upsert(artifact: art, chunks: [EmbeddedChunk(id: "art-model-check#0", artifactID: art.id, span: .pages(startPage: 1, endPage: 1), vector: vec)], modelID: "emb-model-a")
        #expect(try store.isUpToDate(artifactID: art.id, fingerprint: fp, modelID: "emb-model-a") == true)
        #expect(try store.isUpToDate(artifactID: art.id, fingerprint: fp, modelID: "emb-model-b") == false)

        // Search works at 256d and deleted file is pruned on next index pass
        let hits = try await coordinator.search(queryText: "Neural Engine co-residency", dimensions: 256, taskOverride: .query, limit: 5)
        #expect(!hits.isEmpty)

        try FileManager.default.removeItem(at: docURL)
        _ = try await coordinator.indexURLs([corpusDir])
        let statsAfterDelete = try store.stats()
        #expect(statsAfterDelete.countsByKind[.text] == nil)
    }

    @Test("Gemma4Specialization validates gemma4_assistant hidden_size, 48px vision geometry, Leviathan acceptance-residual exactness, and GGUF flags/clamping")
    func testGemma4SpecializationGeometryLeviathanAndGGUFSpawn() throws {
        let gemmaSpec = Gemma4Specialization()
        func makeSig(
            modelType: String?,
            hiddenSize: Int?,
            patchSize: Int? = nil,
            poolingKernelSize: Int? = nil,
            maxSoftTokens: Int? = nil,
            draftModelType: String? = nil,
            draftBackboneHiddenSize: Int? = nil
        ) -> CheckpointSignature {
            CheckpointSignature(
                modelType: modelType,
                hiddenSize: hiddenSize,
                slidingWindow: nil,
                layerTypes: [],
                hasSlidingWindowAttention: false,
                hasLinearOrRecurrentState: false,
                patchSize: patchSize,
                poolingKernelSize: poolingKernelSize,
                maxSoftTokens: maxSoftTokens,
                draftModelType: draftModelType,
                draftBackboneHiddenSize: draftBackboneHiddenSize,
                draftMaxContext: nil,
                hasMMProj: false,
                hasAudioEncoder: false
            )
        }

        // 1. gemma4_assistant `draft-mtp` + backbone_hidden_size match vs mismatch
        let matchingSig = makeSig(
            modelType: "gemma4",
            hiddenSize: 2048,
            patchSize: 16,
            poolingKernelSize: 3,
            maxSoftTokens: 280,
            draftModelType: "gemma4_assistant",
            draftBackboneHiddenSize: 2048
        )
        #expect(gemmaSpec.matches(matchingSig) == true)
        #expect(gemmaSpec.validatesMTPDrafterCompatibility(matchingSig) == true)
        #expect(gemmaSpec.allowsMTPAfterMultimodalPrefill(matchingSig) == true)

        let geom = try #require(gemmaSpec.visionGeometry(for: matchingSig))
        #expect(geom.alignmentMultiplePx == 48)
        #expect(geom.visionSoftTokensPerImage == 280)

        let mismatchedHidden = makeSig(
            modelType: "gemma4",
            hiddenSize: 2048,
            draftModelType: "gemma4_assistant",
            draftBackboneHiddenSize: 4096
        )
        #expect(gemmaSpec.validatesMTPDrafterCompatibility(mismatchedHidden) == false)

        let wrongDrafterType = makeSig(
            modelType: "gemma4",
            hiddenSize: 2048,
            draftModelType: "llama_assistant",
            draftBackboneHiddenSize: 2048
        )
        #expect(gemmaSpec.validatesMTPDrafterCompatibility(wrongDrafterType) == false)

        // 2. Leviathan lossless speculative rejection sampling exact distribution invariant:
        //    For any target distribution p and draft distribution q,
        //    P_emit(k) = q(k) * min(1, p(k)/q(k)) + (1 - sum_j q(j)*min(1, p(j)/q(j))) * max(0, p(k) - q(k)) / sum_m max(0, p(m) - q(m))
        //    MUST equal p(k) identically across every vocabulary outcome!
        let p: [Double] = [0.55, 0.25, 0.15, 0.05]
        let q: [Double] = [0.30, 0.40, 0.10, 0.20]
        let acceptMass = zip(p, q).reduce(0.0) { acc, pair in
            acc + pair.1 * min(1.0, pair.0 / pair.1)
        }
        let residual = zip(p, q).map { max(0.0, $0 - $1) }
        let resSum = residual.reduce(0.0, +)
        #expect(abs((1.0 - acceptMass) - resSum) < 1e-12)
        for k in p.indices {
            let pEmit = q[k] * min(1.0, p[k] / q[k]) + (1.0 - acceptMass) * (residual[k] / resSum)
            #expect(abs(pEmit - p[k]) < 1e-12)
        }

        // 3. SWA/Gemma4 GGUF spawn omits `--cache-reuse`, sets `--ctx-checkpoints 32 --checkpoint-min-step 256 --reasoning-preserve`,
        //    and clamps `--image-max-tokens <= ubatch` for non-causal mmproj projectors.
        let sandbox = try TestTempSandbox(prefix: "heylook_g4_gguf")
        defer { sandbox.cleanup() }
        let modelDir = sandbox.modelsDir.appendingPathComponent("gemma4-gguf-vlm", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let modelGGUF = GGUFHeaderReader.buildMinimalGGUF(
            version: 3,
            tensorCount: 32,
            stringMetadata: ["general.architecture": "gemma4"],
            uint32Metadata: [
                "gemma4.context_length": 32768,
                "gemma4.attention.sliding_window": 1024,
            ]
        )
        let modelPath = modelDir.appendingPathComponent("Gemma4-VLM-Q4_K_M.gguf")
        try modelGGUF.write(to: modelPath)

        let mmprojGGUF = GGUFHeaderReader.buildMinimalGGUF(
            version: 3,
            tensorCount: 16,
            stringMetadata: ["clip.projector_type": "gemma4"]
        )
        let mmprojPath = modelDir.appendingPathComponent("mmproj-Gemma4-f16.gguf")
        try mmprojGGUF.write(to: mmprojPath)

        try "image_max_tokens = 4096\n".write(
            to: modelDir.appendingPathComponent("model.heylook.toml"),
            atomically: true,
            encoding: .utf8
        )

        let vlmDesc = EngineDescription(
            modelID: "gemma4-gguf-vlm",
            provider: .gguf,
            modelPath: modelDir.path,
            contextSize: ProvenanceField(value: 32768, provenance: .derived),
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let spawn = GGUFSpawnConfig.autoSized(for: vlmDesc, headroomGB: 64.0)
        GGUFPortPool.shared.releasePort(for: vlmDesc.modelID)

        #expect(spawn.cacheReuse == nil)
        #expect(spawn.ctxCheckpoints == 32)
        #expect(spawn.checkpointMinStep == 256)
        #expect(spawn.reasoningPreserve == true)
        #expect((spawn.imageMaxTokens ?? Int.max) <= (spawn.ubatchSize ?? 512))

        let argv = GGUFSubprocessProvider.buildLlamaServerArgs(config: spawn)
        #expect(!argv.contains("--cache-reuse"))
        #expect(argv.contains("--reasoning-preserve"))
        if let ckptIdx = argv.firstIndex(of: "--ctx-checkpoints"), ckptIdx + 1 < argv.count {
            #expect(argv[ckptIdx + 1] == "32")
        } else {
            Issue.record("Missing --ctx-checkpoints 32 in GGUF argv: \(argv)")
        }
        if let stepIdx = argv.firstIndex(of: "--checkpoint-min-step"), stepIdx + 1 < argv.count {
            #expect(argv[stepIdx + 1] == "256")
        } else {
            Issue.record("Missing --checkpoint-min-step 256 in GGUF argv: \(argv)")
        }
    }

    @Test("GGUFPortPool skips OS-bound sockets, SubprocessLifecycleRegistry rejects fast exits, and RAMFitCalculator distinguishes MLX hard fail vs GGUF warn")
    func testPortPoolFastExitAndRAMFitCalculator() async throws {
        // 1. OS socket bind probe
        let sockFD = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        #expect(sockFD >= 0)
        defer { if sockFD >= 0 { Darwin.close(sockFD) } }
        var reuse: Int32 = 1
        _ = setsockopt(sockFD, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        let busyPort = 28390
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(busyPort).bigEndian
        addr.sin_addr.s_addr = CFSwapInt32HostToBig(INADDR_LOOPBACK)
        let rc = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.bind(sockFD, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if rc == 0 {
            _ = Darwin.listen(sockFD, 2)
            #expect(GGUFPortPool.isPortAvailable(busyPort) == false)
            let localPool = GGUFPortPool(basePort: busyPort)
            let chosen = localPool.acquirePort(for: "probe-gguf")
            #expect(chosen > busyPort)
            localPool.releaseAll()
        }

        // 2. Fast-exiting subprocess rejected cleanly
        #if os(macOS)
        let sandbox = try TestTempSandbox(prefix: "heylook_subproc")
        defer { sandbox.cleanup() }
        let fakeModel = sandbox.modelsDir.appendingPathComponent("tiny.gguf")
        try Data("GGUF".utf8).write(to: fakeModel)
        let fastExitCfg = GGUFSpawnConfig(binaryPath: "/usr/bin/false", modelPath: fakeModel.path, port: 28992)
        let ggufProvider = GGUFSubprocessProvider(
            description: EngineDescription(modelID: "fast-exit", provider: .gguf, modelPath: fakeModel.path),
            spawnConfig: fastExitCfg
        )
        do {
            _ = try await ggufProvider.launchSubprocess()
            Issue.record("Expected fast-exiting child process to throw ProviderExecutionError")
        } catch let err as ProviderExecutionError {
            switch err {
            case .runtimeUnavailable:
                break
            default:
                Issue.record("Expected runtimeUnavailable, got \(err)")
            }
        }
        #endif

        // 3. RAMFitCalculator constants & MLX hard `fail` vs GGUF `warn` working-set boundary + `sysctlSuggestMB`
        #expect(RAMFitCalculator.thinHeadroomGB == 24.0)
        let fitSandbox = try TestTempSandbox(prefix: "heylook_ramfit")
        defer { fitSandbox.cleanup() }
        let mlxWeightsDir = fitSandbox.modelsDir.appendingPathComponent("big-mlx", isDirectory: true)
        try FileManager.default.createDirectory(at: mlxWeightsDir, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 1024).write(to: mlxWeightsDir.appendingPathComponent("weights.safetensors"))

        // Use a small workingSetGB snapshot vs real 25 MB weights so MLX hard-fails working set while GGUF only warns
        let largeSparseURL = mlxWeightsDir.appendingPathComponent("model.safetensors")
        let twentyFiveMB = Data(repeating: 0xAA, count: 25 * 1024 * 1024)
        try twentyFiveMB.write(to: largeSparseURL)

        let tightSnapshot = RAMFitMemorySnapshot(
            totalGB: 16.0,
            reclaimableGB: 10.0,
            workingSetGB: 0.01, // smaller than 25 MB (~0.023 GB), triggering working-set threshold with plenty of total reclaimable RAM
            maxBufferGB: 8.0,
            wiredLimitMB: 100
        )

        let mlxDesc = EngineDescription(modelID: "tight-mlx", provider: .mlx, modelPath: mlxWeightsDir.path)
        let mlxFit = RAMFitCalculator.assess(for: mlxDesc, snapshot: tightSnapshot)
        #expect(mlxFit.hardWorkingSet == true)
        #expect(mlxFit.workingSetDecision == .fail)
        #expect(mlxFit.fits == false)
        #expect(mlxFit.decision == .fail)
        #expect(mlxFit.sysctlSuggestMB != nil)

        let ggufFile = fitSandbox.modelsDir.appendingPathComponent("tight.gguf")
        try twentyFiveMB.write(to: ggufFile)
        let ggufDesc = EngineDescription(modelID: "tight-gguf", provider: .gguf, modelPath: ggufFile.path)
        let ggufFit = RAMFitCalculator.assess(for: ggufDesc, snapshot: tightSnapshot)
        #expect(ggufFit.hardWorkingSet == false)
        #expect(ggufFit.workingSetDecision == .warn)
        #expect(ggufFit.decision == .thin) // < 24.0 GB thinHeadroomGB
    }

    @Test("ModelComputeOptions calculates ANE vs GPU memory splits and ANECoreMLRuntime computes 98x128 audio log-mel spectrograms")
    func testModelComputeOptionsAndAudioLogMelSpectrogram() {
        var opts = ModelComputeOptions.hybridANETowers
        let splitHybrid = opts.memorySplitMB
        #expect(splitHybrid.aneMB == 335 + 611)
        #expect(splitHybrid.gpuMB == 542)

        opts = .allNeuralEngine
        let splitAllANE = opts.memorySplitMB
        #expect(splitAllANE.aneMB == 335 + 611 + 542)
        #expect(splitAllANE.gpuMB == 0)

        let samples = (0..<16_000).map { i in sin(Float(i) * 2 * .pi * 440.0 / 16_000.0) }
        let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: samples)
        #expect(mel != nil)
        #expect(mel?.frameCount == 98)
        #expect(mel?.features.count == 98 * 128)
    }
}
