import Foundation
import Testing
@testable import HeylookCore
@testable import HeylookEmbeddings
@testable import HeylookEngines
@testable import HeylookServer

@Suite("Heylook Core, Embeddings, Engines & Server Verification Suite")
struct HeylookTests {

    @Test("SamplerCascade resolves 4-layer precedence and records provenance")
    func testSamplerCascadePrecedence() {
        let builtin = SamplerSettings(temperature: 0.7, topP: 0.95, topK: 40, maxTokens: 1024)
        let sidecar = SamplerSettings(temperature: 0.4, minP: 0.05)
        let preset = SamplerSettings(topP: 0.85, repetitionPenalty: 1.1)
        let request = SamplerSettings(temperature: 0.2, maxTokens: 512, seed: 42)

        let resolved = SamplerCascade.resolve(
            builtin: builtin,
            modelSidecar: sidecar,
            preset: preset,
            request: request
        )

        #expect(resolved.effective.temperature == 0.2)
        #expect(resolved.provenanceByField[.temperature] == .configured)
        #expect(resolved.effective.topP == 0.85)
        #expect(resolved.provenanceByField[.topP] == .configured)
        #expect(resolved.effective.minP == 0.05)
        #expect(resolved.provenanceByField[.minP] == .configured)
        #expect(resolved.effective.topK == 40)
        #expect(resolved.provenanceByField[.topK] == .derived)
        #expect(resolved.effective.maxTokens == 512)
        #expect(resolved.effective.seed == 42)
    }

    @Test("ReasoningParser streams <think> tags, strips specials, and supports mid-stream forceEndReasoning")
    func testReasoningParserStreamingAndForceEnd() {
        var parser = ReasoningParser(startsInThinking: false)
        var scrubber = StripSpecials()

        let d1 = parser.feed("<think>Analyzing unified memory ")
        #expect(d1.thinking == "Analyzing unified memory ")
        #expect(d1.text.isEmpty)

        // Mid-stream user clicks "End thinking"
        let forced = parser.forceEndReasoning()
        #expect(parser.reasoningEndedForced)
        #expect(forced.text.isEmpty)

        // Subsequent tokens route straight to visible content even before </think> arrives, and </think> + specials are stripped
        let d2 = parser.feed("still thinking</think>Final visible answer<|im_end|>")
        let cleanText = scrubber.feed(d2.text) + scrubber.finalize()
        #expect(d2.thinking.isEmpty)
        #expect(cleanText == "still thinkingFinal visible answer")
    }

    @Test("HeylookConfig and ModelSidecarConfig round-trip TOML cleanly")
    func testConfigRoundTrip() {
        let original = HeylookConfig(
            defaultModel: "gemma-4-26b-a4b-it-mlx",
            maxLoadedModels: 3,
            maxQueueDepth: 2,
            scanFolders: ["models/mlx", "models/gguf"],
            serverHost: "127.0.0.1",
            serverPort: 8088
        )
        let serialized = original.toTOML()
        let parsed = HeylookConfig.parseTOML(serialized)
        #expect(parsed == original)

        var sidecar = ModelSidecarConfig()
        sidecar.id = "custom-gemma"
        sidecar.provider = .mlx
        sidecar.pinned = true
        sidecar.computePlacement = .hybridANEGPU
        sidecar.visionComputePlacement = .ane
        sidecar.audioComputePlacement = .ane
        sidecar.tags = ["chat", "vision", "thinking"]
        let sidecarTOML = sidecar.toTOML()
        let parsedSidecar = ModelSidecarConfig.parseTOML(sidecarTOML)
        #expect(parsedSidecar == sidecar)
    }

    @Test("SQLiteStore persists conversations, messages, Save & Continue truncation, and notebooks in WAL mode")
    func testSQLiteStoreOperations() throws {
        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_test_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: dbURL) }
        let store = try SQLiteStore(databaseURL: dbURL)

        let presets = try store.listPresets()
        #expect(presets.count >= 3)

        let conv = ConversationRecord(
            title: "Test Session",
            systemPrompt: "Be concise.",
            defaultModelID: "gemma-4-26b-a4b-it-mlx"
        )
        try store.upsertConversation(conv)

        let m0 = try store.appendMessage(MessageRecord(conversationID: conv.id, position: 0, role: "user", content: "Turn 1"))
        let m1 = try store.appendMessage(MessageRecord(conversationID: conv.id, position: 1, role: "assistant", content: "Reply 1", thinking: "Step 1"))
        let m2 = try store.appendMessage(MessageRecord(conversationID: conv.id, position: 2, role: "user", content: "Turn 2"))
        #expect(m0.position == 0)
        #expect(m1.position == 1)
        #expect(m2.position == 2)

        // Edit Turn 1 and truncate Turn 2 ("Save & Continue")
        var editedM0 = m0
        editedM0.content = "Turn 1 Edited"
        try store.appendMessage(editedM0)
        let deleted = try store.deleteMessages(conversationID: conv.id, fromPosition: 1)
        #expect(deleted == 2)

        let remaining = try store.listMessages(conversationID: conv.id)
        #expect(remaining.count == 1)
        #expect(remaining.first?.content == "Turn 1 Edited")

        let nb = NotebookRecord(title: "Draft Note", body: "Hello ANE", modelID: "qwen3.5")
        try store.upsertNotebook(nb)
        let notebooks = try store.listNotebooks()
        #expect(notebooks.contains(where: { $0.id == nb.id && $0.title == "Draft Note" }))
    }

    @Test("Matryoshka vDSP truncation, WindowPlanner, and VectorStore parent-artifact deduplication")
    func testEmbeddingsAndVectorStore() throws {
        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_vec_test_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: dbURL) }
        let store = try VectorStore(databaseURL: dbURL)
        let engine = DeterministicHashEmbeddingEngine()

        let art1 = Artifact(
            id: "art-1",
            collectionID: "default",
            kind: .text,
            url: URL(fileURLWithPath: "/tmp/doc1.md"),
            title: "ANE CoreML Vision Tower",
            fingerprint: ContentFingerprint(byteCount: 120, modifiedUnixMs: 1_700_000_000_000)
        )
        let v1a = engine.embed(text: "Apple Neural Engine CoreML vision tower")
        let v1b = engine.embed(text: "Apple Neural Engine CoreML vision tower span 2")
        let chunk1a = EmbeddedChunk(id: "art-1#0", artifactID: art1.id, span: .pages(startPage: 1, endPage: 1), vector: v1a, previewText: "ANE Vision")
        let chunk1b = EmbeddedChunk(id: "art-1#1", artifactID: art1.id, span: .pages(startPage: 2, endPage: 2), vector: v1b, previewText: "ANE Vision 2")
        try store.upsert(artifact: art1, chunks: [chunk1a, chunk1b])

        // Query at 256d Matryoshka slice
        let hits = try store.search(queryVector: v1a, dimensions: 256, limit: 5)
        #expect(hits.count == 1) // Deduplicated by parent artifact ID
        #expect(hits.first?.artifact.id == art1.id)
        #expect((hits.first?.bestScore ?? 0) > 0.99)
        #expect(hits.first?.spanMatches.count == 2)

        // Verify WindowPlanner content-novelty frame selection
        let thumbs: [[Float]] = (0..<12).map { idx in
            idx == 5 ? [Float](repeating: 1.0, count: 16) : [Float](repeating: 0.1, count: 16)
        }
        let times = (0..<12).map { Double($0) * 0.25 }
        let picked = WindowPlanner.pickContent(thumbnails: thumbs, times: times, lo: 0.0, hi: 3.0, fps: 1.0, maxFrames: 4)
        #expect(picked.contains(5))
    }

    @Test("ModelComputeOptions calculates ANE vs GPU memory split and audio log-mel spectrogram")
    func testModelComputeOptionsAndAudioMel() {
        var options = ModelComputeOptions.hybridANETowers
        let split1 = options.memorySplitMB
        #expect(split1.aneMB == 335 + 611)
        #expect(split1.gpuMB == 542)
        #expect(options.preset == .hybridANETowers)

        options = .allNeuralEngine
        let split2 = options.memorySplitMB
        #expect(split2.aneMB == 335 + 611 + 542)
        #expect(split2.gpuMB == 0)
        #expect(options.preset == .allNeuralEngine)

        // Verify 1 second of 16 kHz sine wave produces 98 frames x 128 mel bins
        let samples = (0..<16_000).map { i in sin(Float(i) * 2 * .pi * 440.0 / 16_000.0) }
        let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: samples)
        #expect(mel != nil)
        #expect(mel?.frameCount == 98)
        #expect(mel?.features.count == 98 * 128)

        let mel2D = ANECoreMLRuntime.computeAudioLogMel(samples: samples, sampleRate: 16_000, nMels: 128)
        #expect(mel2D.count == 98)
        #expect(mel2D.first?.count == 128)
    }

    @Test("IndexCoordinator indexes folders, deduplicates unchanged files, prunes deleted files, and searches spans")
    func testIndexCoordinatorPipelineAndHeylookBatchPlanner() async throws {
        let fm = FileManager.default
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_index_test_\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempDir) }

        let dbURL = tempDir.appendingPathComponent("vectors.sqlite")
        let corpusDir = tempDir.appendingPathComponent("corpus", isDirectory: true)
        try fm.createDirectory(at: corpusDir, withIntermediateDirectories: true)

        let doc1URL = corpusDir.appendingPathComponent("ane_architecture.md")
        let doc2URL = corpusDir.appendingPathComponent("ComputeScheduler.swift")
        try "Apple Neural Engine co-residency keeps vision and audio towers hot in 192 GB unified RAM.".write(to: doc1URL, atomically: true, encoding: .utf8)
        try "public actor ComputeScheduler { public func withComputeLease() async throws {} }".write(to: doc2URL, atomically: true, encoding: .utf8)

        let store = try VectorStore(databaseURL: dbURL)
        let engine = DeterministicHashEmbeddingEngine()
        let coordinator = IndexCoordinator(engine: engine, store: store)

        let res1 = try await coordinator.indexURLs([corpusDir])
        #expect(res1.indexed == 2)

        // Second pass skips unchanged files via ContentFingerprint
        let res2 = try await coordinator.indexURLs([corpusDir])
        #expect(res2.indexed == 0)

        let stats1 = try store.stats()
        #expect(stats1.countsByKind[.text] == 1)
        #expect(stats1.countsByKind[.code] == 1)

        let watched = try store.listWatchedFolders()
        #expect(watched.contains(where: { $0.path == corpusDir.standardizedFileURL.path }))

        // Search across indexed spans
        let hits = try await coordinator.search(
            queryText: "Apple Neural Engine co-residency",
            dimensions: 256,
            taskOverride: .query,
            limit: 5
        )
        #expect(hits.count == 2)
        #expect(hits.first?.spanMatches.first?.scoresByDimension[256] != nil)

        // Delete one file and re-index to verify pruning of missing artifacts
        try fm.removeItem(at: doc2URL)
        _ = try await coordinator.indexURLs([corpusDir])
        let stats2 = try store.stats()
        #expect(stats2.countsByKind[.text] == 1)
        #expect((stats2.countsByKind[.code] ?? 0) == 0)

        // Verify HeylookMLXEngine adaptive sub-batching and maskedMeanPoolAndNormalize
        let batchItems = (0..<10).map { i in
            EmbeddingInput(
                parts: [
                    .text("item \(i)"),
                    .media(kind: .image, mimeType: "image/png", data: Data(repeating: 0xAA, count: 600_000))
                ],
                task: .document
            )
        }
        let batches = HeylookMLXEngine.planBatches(batchItems, maxItems: 4, maxRequestMediaBytes: 2_000_000)
        #expect(batches.count == 5) // Each batch holds at most 2 items (1.2 MB <= 1.5 MB raw budget)

        let pooled = NativeMLXEmbeddingGemma2.maskedMeanPoolAndNormalize(
            projectedTokens: [
                [1.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [99.0, 99.0, 99.0, 99.0]
            ],
            attentionMask: [1, 1, 0],
            dimensions: 2
        )
        #expect(pooled.count == 2)
        #expect(abs(pooled[0] - 1.0) < 1e-4)
        #expect(abs(pooled[1] - 0.0) < 1e-4)
    }

    @Test("GGUFHeaderReader parses valid GGUF v3 binary header")
    func testGGUFHeaderReader() {
        var data = Data("GGUF".utf8)
        var version: UInt32 = 3
        var tensors: UInt64 = 291
        var kvCount: UInt64 = 42
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &tensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &kvCount) { data.append(contentsOf: $0) }

        let parsed = GGUFHeaderReader.readHeader(from: data)
        #expect(parsed == GGUFHeaderReader.HeaderSummary(version: 3, tensorCount: 291, metadataKVCount: 42))
    }

    @Test("ComputeScheduler enforces Dual-Lane ANE Co-Residency without evicting GPU LLMs")
    func testComputeSchedulerDualLaneAndLRU() async throws {
        let scheduler = ComputeScheduler(maxLoadedGPUModels: 1, maxQueueDepth: 1)

        func makeDesc(id: String, provider: EngineProviderKind, placement: ComputePlacement) -> EngineDescription {
            EngineDescription(
                modelID: id,
                provider: provider,
                modelPath: "/tmp/\(id)",
                architecture: ProvenanceField(value: "test", provenance: .derived),
                quantization: ProvenanceField(value: "4bit", provenance: .derived),
                computePlacement: ProvenanceField(value: placement, provenance: .configured, computePlacement: placement),
                contextSize: ProvenanceField(value: 8192, provenance: .derived),
                flashAttention: ProvenanceField(value: true, provenance: .derived),
                supportsVision: ProvenanceField(value: false, provenance: .derived),
                supportsThinking: ProvenanceField(value: true, provenance: .derived)
            )
        }

        let aneDesc = makeDesc(id: "ane-embed", provider: .coreai, placement: .ane)
        let gpu1Desc = makeDesc(id: "gpu-llm-1", provider: .mlx, placement: .mlxGPU)
        let gpu2Desc = makeDesc(id: "gpu-llm-2", provider: .mlx, placement: .mlxGPU)

        await scheduler.register(provider: CoreAIProvider(description: aneDesc), description: aneDesc)
        await scheduler.register(provider: MLXSwiftProvider(description: gpu1Desc), description: gpu1Desc)
        await scheduler.register(provider: MLXSwiftProvider(description: gpu2Desc), description: gpu2Desc)

        _ = try await scheduler.activateModel("ane-embed")
        _ = try await scheduler.activateModel("gpu-llm-1")
        _ = try await scheduler.activateModel("gpu-llm-2")

        let metrics = await scheduler.metrics()
        // ANE model resides on the ANE lane and is never evicted by GPU model rotation;
        // gpu-llm-1 is evicted when gpu-llm-2 loads under maxLoadedGPUModels = 1.
        #expect(metrics.loadedANEModels.contains("ane-embed"))
        #expect(metrics.loadedGPUModels == ["gpu-llm-2"])
    }

    @Test("HeylookRouter handles /v1/models, /v1/messages, and disconnect-resilient /v1/conversations/{id}/generate")
    func testRouterEndpointsAndDisconnectResilientGenerate() async throws {
        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_router_test_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: dbURL) }
        let store = try SQLiteStore(databaseURL: dbURL)
        let scheduler = ComputeScheduler()

        let desc = EngineDescription(
            modelID: "gemma-4-26b-a4b-it-mlx",
            provider: .mlx,
            modelPath: "/tmp/gemma4",
            architecture: ProvenanceField(value: "gemma4", provenance: .derived),
            quantization: ProvenanceField(value: "4bit", provenance: .derived),
            computePlacement: ProvenanceField(value: .hybridANEGPU, provenance: .configured, computePlacement: .hybridANEGPU),
            contextSize: ProvenanceField(value: 32768, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        await scheduler.register(provider: MLXSwiftProvider(description: desc), description: desc)

        let router = HeylookRouter(store: store, scheduler: scheduler, bindHost: "127.0.0.1")

        // 1. Host header loopback protection
        let forbiddenResp = await router.handle(
            method: "GET",
            path: "/v1/models",
            headers: ["Host": "evil.example.com"]
        )
        #expect(forbiddenResp.status == 403)

        // 2. Non-streaming POST /v1/messages
        let msgReq = MessagesRequest(
            model: "gemma-4-26b-a4b-it-mlx",
            messages: [MessagesRequest.MessageTurn(role: "user", content: "Hello native workbench")],
            maxTokens: 128,
            stream: false,
            thinking: true
        )
        let body = try JSONEncoder().encode(msgReq)
        let msgResp = await router.handle(
            method: "POST",
            path: "/v1/messages",
            headers: ["Host": "127.0.0.1:8000"],
            body: body
        )
        #expect(msgResp.status == 200)
        let decodedMsg = try JSONDecoder().decode(MessagesResponse.self, from: msgResp.body)
        #expect(decodedMsg.role == "assistant")
        #expect(!decodedMsg.content.isEmpty)

        // 3. Disconnect-resilient POST /v1/conversations/{id}/generate
        let convID = "conv-mobile-sync"
        let genReq = ConversationGenerateRequest(
            model: "gemma-4-26b-a4b-it-mlx",
            content: "Run on Mac Studio while iPhone locks"
        )
        let genBody = try JSONEncoder().encode(genReq)
        let genResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/\(convID)/generate",
            headers: ["Host": "127.0.0.1:8000"],
            body: genBody
        )
        #expect(genResp.status == 200)
        let sseTranscript = String(decoding: genResp.body, as: UTF8.self)
        #expect(sseTranscript.contains("event: heylook_saved"))

        // Verify both user and assistant messages were durably persisted to SQLite WAL
        let persisted = try store.listMessages(conversationID: convID)
        #expect(persisted.count == 2)
        #expect(persisted.first?.role == "user")
        #expect(persisted.last?.role == "assistant")
    }

    @Test("PrefixCacheTracker reports cold, reused, system_changed, and history_diverged cache outcomes")
    func testPrefixCacheTrackerOutcomes() {
        var tracker = PrefixCacheTracker()
        let u1 = MessagesRequest.MessageTurn(role: "user", content: "Explain Apple Silicon unified memory bandwidth.")
        let a1 = MessagesRequest.MessageTurn(role: "assistant", content: "M2 Ultra delivers 800 GB/s across CPU, GPU, and ANE.")
        let u2 = MessagesRequest.MessageTurn(role: "user", content: "How does ANE co-residency avoid GPU eviction?")

        // Turn 1: Cold cache miss
        let r1 = tracker.evaluate(system: "You are a systems architect.", messages: [u1])
        #expect(r1.outcome == .miss)
        #expect(r1.cause == "cold")
        #expect(r1.cachedTokens == 0)
        #expect(r1.processedTokens == r1.promptTokens)
        tracker.recordAssistantReply(a1.content)

        // Turn 2: Prefix reuse across [u1, a1] + system prompt
        let r2 = tracker.evaluate(system: "You are a systems architect.", messages: [u1, a1, u2])
        #expect(r2.outcome == .reused)
        #expect(r2.cause == nil)
        #expect(r2.cachedTokens > 0)
        #expect(r2.processedTokens < r2.promptTokens)

        // Turn 3: System prompt changed -> system_changed miss
        let r3 = tracker.evaluate(system: "Different system prompt.", messages: [u1, a1, u2])
        #expect(r3.outcome == .miss)
        #expect(r3.cause == "system_changed")
        #expect(r3.cachedTokens == 0)

        // Turn 4: Branch edit on Turn 1 -> history_diverged miss
        let u1Edited = MessagesRequest.MessageTurn(role: "user", content: "Completely different opening question.")
        let r4 = tracker.evaluate(system: "Different system prompt.", messages: [u1Edited, a1, u2])
        #expect(r4.outcome == .miss)
        #expect(r4.cause == "history_diverged")
        #expect(r4.cachedTokens == 0)
    }

    @Test("CoreAIBundleManifest inspects .aimodel metadata.json and CoreAIProvider emits SpeculativeReport")
    func testCoreAIBundleManifestAndSpeculativeTelemetry() async throws {
        let fm = FileManager.default
        let bundleDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwen3_ane_\(UUID().uuidString).aimodel", isDirectory: true)
        try fm.createDirectory(at: bundleDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: bundleDir) }

        let metaJSON = """
        {
          "name": "Qwen3-4B-ANE",
          "kind": "vlm",
          "vocabSize": 151936,
          "maxContextLength": 16384,
          "prefillChunkSize": 128,
          "components": {
            "main": {},
            "vision": {},
            "drafter": {}
          }
        }
        """
        try metaJSON.write(to: bundleDir.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)

        let manifest = try CoreAIBundleManifest.inspect(at: bundleDir)
        #expect(manifest.name == "Qwen3-4B-ANE")
        #expect(manifest.kind == "vlm")
        #expect(manifest.maxContextLength == 16384)
        #expect(manifest.prefillChunkSize == 128)
        #expect(manifest.hasVisionComponent)
        #expect(manifest.hasDrafterComponent)

        let baseDesc = EngineDescription(
            modelID: "qwen3-4b-ane",
            provider: .coreai,
            modelPath: bundleDir.path,
            architecture: ProvenanceField(value: "qwen3", provenance: .derived),
            quantization: ProvenanceField(value: "w4a16", provenance: .derived),
            computePlacement: ProvenanceField(value: .ane, provenance: .configured, computePlacement: .ane),
            contextSize: ProvenanceField(value: 8192, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: false, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let provider = CoreAIProvider(description: baseDesc)
        let enriched = await provider.describe()
        #expect(enriched.contextSize.value == 16384)
        #expect(enriched.contextSize.provenance == .observed)
        #expect(enriched.supportsVision.value == true)

        let registry = RequestRegistry()
        let stream = await provider.generate(
            requestID: "req-coreai-spec",
            request: MessagesRequest(
                model: "qwen3-4b-ane",
                messages: [MessagesRequest.MessageTurn(role: "user", content: "Test speculative ANE")]
            ),
            effectiveSampling: .builtinDefaults,
            registry: registry
        )
        var lastChunk: GenerationChunk?
        for try await chunk in stream {
            lastChunk = chunk
        }
        #expect(lastChunk?.speculativeReport != nil)
        #expect((lastChunk?.speculativeReport?.acceptedTokens ?? 0) > 0)
    }

    @Test("GGUFSubprocessProvider builds llama-server CLI flags, sanitizes child env, and parses SSE timings")
    func testGGUFSubprocessProviderContract() {
        let cfg = GGUFSpawnConfig(
            binaryPath: "/usr/local/bin/llama-server",
            modelPath: "/models/Qwen3-30B-A3B-Q4_K_M.gguf",
            host: "127.0.0.1",
            port: 18081,
            nGPULayers: 999,
            ctxSize: 32768,
            flashAttention: true,
            batchSize: 2048,
            ubatchSize: 512,
            mmprojPath: "/models/mmproj-f16.gguf",
            imageMaxTokens: 1024,
            chatTemplateFile: "/models/template.jinja",
            draftModelPath: "/models/Qwen3-0.6B-Q8_0.gguf",
            specType: "ngram-mod",
            specDraftMax: 16,
            specDraftPMin: 0.75,
            cacheTypeK: "q8_0",
            cacheTypeV: "q8_0",
            cacheRAMMiB: 4096
        )

        let args = GGUFSubprocessProvider.buildLlamaServerArgs(config: cfg)
        #expect(args.contains("-np"))
        #expect(args.contains("--no-webui"))
        #expect(args.contains("--mmproj"))
        #expect(args.contains("--image-max-tokens"))
        #expect(args.contains("-md"))
        #expect(args.contains("--spec-type"))
        #expect(args.contains("-ctk"))
        #expect(args.contains("-ctv"))
        #expect(args.contains("-cram"))
        #expect(args.suffix(2) == ["-lv", "4"])

        let sanitizedEnv = GGUFSubprocessProvider.childEnvironment(from: [
            "PATH": "/usr/bin",
            "LLAMA_ARG_LOG_FILE": "/tmp/should_be_removed.log",
        ])
        #expect(sanitizedEnv["LLAMA_ARG_LOG_FILE"] == nil)
        #expect(sanitizedEnv["GGML_METAL_RESIDENCY_KEEP_ALIVE_S"] == "604800")

        let sseLine = """
        data: {"choices":[{"delta":{"content":"Hello","reasoning_content":"step 1"},"finish_reason":null}],"timings":{"prompt_n":48,"cache_n":32,"predicted_n":12,"prompt_per_second":820.5,"predicted_per_second":94.2,"draft_n":16,"draft_n_accepted":14}}
        """
        let delta = GGUFSubprocessProvider.parseOpenAISSEDataLine(sseLine)
        #expect(delta?.content == "Hello")
        #expect(delta?.reasoningContent == "step 1")
        #expect(delta?.promptTokens == 48)
        #expect(delta?.cachedTokens == 32)
        #expect(delta?.predictedTokens == 12)
        #expect(delta?.draftedTokens == 16)
        #expect(delta?.acceptedTokens == 14)
        #expect(GGUFSubprocessProvider.parseOpenAISSEDataLine("data: [DONE]") == nil)

        // Verify RemoteHeylookProvider SSE frame parser
        let remoteDelta = RemoteHeylookProvider.parseAnthropicSSEEvent(
            event: "content_block_delta",
            data: #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Native Swift"}}"#
        )
        #expect(remoteDelta?.textDelta == "Native Swift")
    }

    @Test("HeylookConfig parses Python [scan] folders, [[models]], 4-rung GGUF template ladder, drafters, and 192 GB auto-sizing")
    func testPythonHeylookTOMLAndGGUFTemplateLadderAndDrafter() throws {
        let pythonTOML = """
        default_model = "Qwen3.8-27B"
        max_loaded_models = 4

        [scan]
        folders = [
          "models/mlx",
          "models/gguf",
        ]

        [[models]]
        id = "Qwen3.8-27B"
        provider = "gguf"
        model_path = "models/gguf/Qwen3.8-27B.gguf"
        ctx_size = 32768
        """
        let parsed = HeylookConfig.parseTOML(pythonTOML)
        #expect(parsed.defaultModel == "Qwen3.8-27B")
        #expect(parsed.scanFolders == ["models/mlx", "models/gguf"])
        #expect(parsed.explicitModels.count == 1)
        #expect(parsed.explicitModels.first?.id == "Qwen3.8-27B")
        #expect(parsed.explicitModels.first?.provider == .gguf)
        #expect(parsed.explicitModels.first?.sidecar.ctxSize == 32768)

        let fm = FileManager.default
        let modelDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("gguf_ladder_\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: modelDir.appendingPathComponent("MTP"), withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: modelDir) }

        // Rung 4: Embedded fallback when no jinja files exist
        let rung4 = HeylookConfig.resolveGGUFTemplate(in: modelDir, sidecar: nil)
        #expect(rung4.source.value == "gguf_embedded")
        #expect(rung4.source.provenance == .derived)

        // Rung 3: chat_template.jinja sidecar
        try "{{ messages }}".write(to: modelDir.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
        let rung3 = HeylookConfig.resolveGGUFTemplate(in: modelDir, sidecar: nil)
        #expect(rung3.source.value == "chat_template.jinja")
        #expect(rung3.source.provenance == .derived)

        // Rung 2: chat_template.heylook.jinja operator override beats chat_template.jinja
        try "{{ operator }}".write(to: modelDir.appendingPathComponent("chat_template.heylook.jinja"), atomically: true, encoding: .utf8)
        let rung2 = HeylookConfig.resolveGGUFTemplate(in: modelDir, sidecar: nil)
        #expect(rung2.source.value == "chat_template.heylook.jinja")
        #expect(rung2.source.provenance == .configured)

        // Drafter discovery in MTP/ subfolder + unset override
        let drafterFile = modelDir.appendingPathComponent("MTP/mtp-qwen3.gguf")
        try Data("GGUF".utf8).write(to: drafterFile)
        let foundDrafter = HeylookConfig.discoverDrafter(in: modelDir, provider: .gguf, sidecar: nil)
        #expect(foundDrafter.draftModelPath == drafterFile.standardizedFileURL.path)

        var unsetSidecar = ModelSidecarConfig()
        unsetSidecar.unsetKeys = ["draft_model_path", "spec_type"]
        let suppressedDrafter = HeylookConfig.discoverDrafter(in: modelDir, provider: .gguf, sidecar: unsetSidecar)
        #expect(suppressedDrafter.draftModelPath == nil)

        // 192 GB RAM headroom auto-sizing (-ub 2048 and -cram >= 8192 MiB)
        let desc = EngineDescription(
            modelID: "qwen3.8-27b-gguf",
            provider: .gguf,
            modelPath: modelDir.path,
            architecture: ProvenanceField(value: "qwen3", provenance: .derived),
            quantization: ProvenanceField(value: "Q8_0", provenance: .derived),
            computePlacement: ProvenanceField(value: .ggufSubprocess, provenance: .derived),
            contextSize: ProvenanceField(value: 32768, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: false, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived),
            estimatedMemoryMB: 28_000
        )
        let sized = GGUFSpawnConfig.autoSized(for: desc, headroomGB: 96.0)
        #expect(sized.ubatchSize == 2048)
        #expect(sized.cacheRAMMiB == 32768)
        #expect(sized.draftModelPath == drafterFile.standardizedFileURL.path)
    }

    @Test("HeylookRouter serves GET /v1/capabilities, POST /v1/embeddings, streaming POST /v1/messages SSE, and multi-frame HTTP Content-Length")
    func testStreamingMessagesEmbeddingsAndCapabilitiesRoutes() async throws {
        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase2a_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: dbURL) }
        let store = try SQLiteStore(databaseURL: dbURL)
        let scheduler = ComputeScheduler()
        let desc = EngineDescription(
            modelID: "qwen3.5-mlx",
            provider: .mlx,
            modelPath: "/tmp/qwen3.5",
            architecture: ProvenanceField(value: "qwen3_5", provenance: .derived),
            quantization: ProvenanceField(value: "8bit", provenance: .derived),
            computePlacement: ProvenanceField(value: .mlxGPU, provenance: .derived),
            contextSize: ProvenanceField(value: 32768, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        await scheduler.register(provider: MLXSwiftProvider(description: desc), description: desc)
        let router = HeylookRouter(store: store, scheduler: scheduler)

        // 1. GET /v1/capabilities
        let capResp = await router.handle(method: "GET", path: "/v1/capabilities", headers: ["Host": "127.0.0.1:8000"])
        #expect(capResp.status == 200)
        let capJSON = try JSONSerialization.jsonObject(with: capResp.body) as? [String: Any]
        #expect((capJSON?["max_request_media_bytes"] as? Int) == 24 * 1024 * 1024)

        // 2. POST /v1/embeddings with Matryoshka 256d truncation
        let embPayload = try JSONSerialization.data(withJSONObject: [
            "model": "embedding-gemma2",
            "input": ["Apple Neural Engine vision tower", "Metal GPU decode lane"],
            "dimensions": 256,
            "task": "query",
        ])
        let embResp = await router.handle(
            method: "POST",
            path: "/v1/embeddings",
            headers: ["Host": "127.0.0.1:8000"],
            body: embPayload
        )
        #expect(embResp.status == 200)
        let embJSON = try JSONSerialization.jsonObject(with: embResp.body) as? [String: Any]
        let embData = embJSON?["data"] as? [[String: Any]]
        #expect(embData?.count == 2)
        let firstVec = embData?.first?["embedding"] as? [Double]
        #expect(firstVec?.count == 256)

        // 3. Streaming POST /v1/messages (stream: true)
        let streamReq = MessagesRequest(
            model: "qwen3.5-mlx",
            messages: [MessagesRequest.MessageTurn(role: "user", content: "Stream SSE tokens")],
            stream: true,
            thinking: true
        )
        let streamResp = await router.handle(
            method: "POST",
            path: "/v1/messages",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(streamReq)
        )
        #expect(streamResp.status == 200)
        #expect(streamResp.headers["Content-Type"] == "text/event-stream")
        let sseText = String(decoding: streamResp.body, as: UTF8.self)
        #expect(sseText.contains("event: message_start"))
        #expect(sseText.contains("event: content_block_delta"))
        #expect(sseText.contains("event: message_delta"))
        #expect(sseText.contains("event: message_stop"))

        // 4. Multi-frame HTTP Content-Length calculation
        let rawHTTP = Data("POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nContent-Length: 1024\r\n\r\nhello".utf8)
        let expectedBytes = HeylookHTTPServer.expectedTotalBytes(for: rawHTTP)
        #expect(expectedBytes == rawHTTP.count - 5 + 1024)
    }

    @Test("HeylookConfig discovers per-quant GGUF variant folders, pairs sibling MTP/assistant drafters, and skips auxiliary CoreML sub-bundles")
    func testDiscoverModelsVariantFoldersAndDrafterFiltering() throws {
        let rootDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase2b_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: rootDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootDir) }

        // 1. Per-quant GGUF variant folder with parent template + parent dspark drafter
        let ggufRepo = rootDir.appendingPathComponent("unsloth_Qwen3.8-Flash-Next-GGUF", isDirectory: true)
        let quantDir = ggufRepo.appendingPathComponent("UD-Q6_K_XL", isDirectory: true)
        let dsparkDir = ggufRepo.appendingPathComponent("dspark", isDirectory: true)
        try FileManager.default.createDirectory(at: quantDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: dsparkDir, withIntermediateDirectories: true)

        let shardFile = quantDir.appendingPathComponent("Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00003.gguf")
        try Data("GGUF".utf8).write(to: shardFile)
        try "{{ messages }}".write(
            to: ggufRepo.appendingPathComponent("chat_template.heylook.jinja"),
            atomically: true,
            encoding: .utf8
        )
        let dsparkFile = dsparkDir.appendingPathComponent("dspark-qwen3.8-draft.gguf")
        try Data("GGUF".utf8).write(to: dsparkFile)

        #expect(HeylookConfig.ggufModelStem(from: shardFile.lastPathComponent) == "Qwen3.8-Flash-Next-UD-Q6_K_XL")
        #expect(HeylookConfig.isGGUFVariantDirectory(quantDir, primaryGGUFFile: shardFile) == true)

        // 2. MLX target folder + sibling -assistant drafter folder + nested coreml/ ANE bundle
        let mlxTarget = rootDir.appendingPathComponent("meta-models_Muse-Glimmer-30B-8bit", isDirectory: true)
        let mlxAssistant = rootDir.appendingPathComponent("meta-models_Muse-Glimmer-30B-assistant", isDirectory: true)
        let nestedCoreML = mlxTarget
            .appendingPathComponent("coreml", isDirectory: true)
            .appendingPathComponent("VisionTower.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: mlxTarget, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: mlxAssistant, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: nestedCoreML, withIntermediateDirectories: true)

        try Data("weights".utf8).write(to: mlxTarget.appendingPathComponent("model.safetensors"))
        try Data("weights".utf8).write(to: mlxAssistant.appendingPathComponent("model.safetensors"))

        // Verify ANETowerInspector.isReadableCoreMLBundle sentinel check
        #expect(ANETowerInspector.isReadableCoreMLBundle(at: nestedCoreML) == false)
        try Data("coreml".utf8).write(to: nestedCoreML.appendingPathComponent("coremldata.bin"))
        #expect(ANETowerInspector.isReadableCoreMLBundle(at: nestedCoreML) == true)

        let discovered = HeylookConfig.discoverModels(in: [rootDir])
        let ids = discovered.map(\.modelID)
        #expect(ids == ["meta-models_Muse-Glimmer-30B-8bit", "Qwen3.8-Flash-Next-UD-Q6_K_XL"])

        let ggufModel = try #require(discovered.first(where: { $0.modelID == "Qwen3.8-Flash-Next-UD-Q6_K_XL" }))
        #expect(ggufModel.templateSource.value == "chat_template.heylook.jinja")
        #expect(ggufModel.tags.contains("mtp"))
        let ggufDrafter = HeylookConfig.discoverDrafter(in: quantDir, provider: .gguf, sidecar: nil)
        #expect(ggufDrafter.draftModelPath == dsparkFile.standardizedFileURL.path)

        let mlxModel = try #require(discovered.first(where: { $0.modelID == "meta-models_Muse-Glimmer-30B-8bit" }))
        #expect(mlxModel.tags.contains("mtp"))
        let mlxDrafter = HeylookConfig.discoverDrafter(in: mlxTarget, provider: .mlx, sidecar: nil)
        #expect(mlxDrafter.draftModelPath == mlxAssistant.standardizedFileURL.path)
        #expect(mlxDrafter.specType == "draft-model")
    }

    @Test("Bonjour _heylook._tcp endpoint resolution, multimodal MessageTurn block encoding, and HTTP 422 capability gating")
    func testBonjourMultimodalTurnAndCapabilityGate() async throws {
        // 1. Bonjour _heylook._tcp service name -> .local baseURL resolution
        let peer = DiscoveredHeylookServer.fromBonjourService(
            name: "Mac Studio M2 Ultra",
            type: "_heylook._tcp",
            domain: "local.",
            defaultPort: 8000
        )
        #expect(peer.baseURLString == "http://Mac-Studio-M2-Ultra.local:8000")
        #expect(peer.isBonjour == true)

        // 2. Multimodal MessageTurn round-trip (Anthropic image/audio content block array)
        let imgFile = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("test_img_\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: imgFile) }
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: imgFile)

        let attachment = try MessagesRequest.MediaAttachment.fromFileURL(imgFile)
        #expect(attachment.kind == "image")
        #expect(attachment.mediaType == "image/png")

        let turn = MessagesRequest.MessageTurn(
            role: "user",
            content: "What is in this diagram?",
            attachments: [attachment]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let encodedTurn = try encoder.encode(turn)
        let jsonString = String(decoding: encodedTurn, as: UTF8.self)
        #expect(jsonString.contains("\"type\":\"image\""))
        #expect(jsonString.contains("\"media_type\":\"image/png\""))

        let decodedTurn = try JSONDecoder().decode(MessagesRequest.MessageTurn.self, from: encodedTurn)
        #expect(decodedTurn.content == "What is in this diagram?")
        #expect(decodedTurn.attachments.count == 1)
        #expect(decodedTurn.attachments.first?.mediaType == "image/png")

        // 3. HTTP 422 capability gating on text-only model vs 200 on vision-capable model
        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase3_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: dbURL) }
        let store = try SQLiteStore(databaseURL: dbURL)
        let scheduler = ComputeScheduler()

        let textOnlyDesc = EngineDescription(
            modelID: "qwen3-text-only",
            provider: .mlx,
            modelPath: "/tmp/qwen3-text",
            architecture: ProvenanceField(value: "qwen3", provenance: .derived),
            quantization: ProvenanceField(value: "8bit", provenance: .derived),
            computePlacement: ProvenanceField(value: .mlxGPU, provenance: .derived),
            contextSize: ProvenanceField(value: 32768, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: false, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let visionDesc = EngineDescription(
            modelID: "gemma4-vision",
            provider: .mlx,
            modelPath: "/tmp/gemma4-vision",
            architecture: ProvenanceField(value: "gemma4", provenance: .derived),
            quantization: ProvenanceField(value: "4bit", provenance: .derived),
            computePlacement: ProvenanceField(value: .hybridANEGPU, provenance: .derived),
            contextSize: ProvenanceField(value: 32768, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        await scheduler.register(provider: MLXSwiftProvider(description: textOnlyDesc), description: textOnlyDesc)
        await scheduler.register(provider: MLXSwiftProvider(description: visionDesc), description: visionDesc)
        let router = HeylookRouter(store: store, scheduler: scheduler)

        let badReq = MessagesRequest(model: "qwen3-text-only", messages: [turn])
        let badResp = await router.handle(
            method: "POST",
            path: "/v1/messages",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(badReq)
        )
        #expect(badResp.status == 422)

        let okReq = MessagesRequest(model: "gemma4-vision", messages: [turn])
        let okResp = await router.handle(
            method: "POST",
            path: "/v1/messages",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(okReq)
        )
        #expect(okResp.status == 200)
    }

    @Test("GGUF --mmproj auto-discovery, OpenAI vision/audio message serialization, and /v1/admin/models/{id} sidecar & load routes")
    func testMMProjOpenAIVisionSerializationAndAdminModelRoutes() async throws {
        // 1. GGUF --mmproj auto-discovery & buildLlamaServerArgs
        let rootDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase4_\(UUID().uuidString)", isDirectory: true)
        let modelDir = rootDir.appendingPathComponent("DeepSeek-V4-Vision-GGUF", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootDir) }

        let primaryGGUF = modelDir.appendingPathComponent("DeepSeek-V4-Vision-Q8_0.gguf")
        let mmprojGGUF = modelDir.appendingPathComponent("mmproj-DeepSeek-V4-F16.gguf")
        try Data("GGUF".utf8).write(to: primaryGGUF)
        try Data("GGUF".utf8).write(to: mmprojGGUF)

        let discovered = HeylookConfig.discoverModels(in: [rootDir])
            .first(where: { $0.modelPath == modelDir.standardizedFileURL.path })
        let vlmDesc = try #require(discovered)
        #expect(vlmDesc.supportsVision.value == true)
        #expect(vlmDesc.tags.contains("vision"))

        let spawnCfg = GGUFSpawnConfig.autoSized(for: vlmDesc, headroomGB: 64.0)
        #expect(spawnCfg.mmprojPath == mmprojGGUF.standardizedFileURL.path)
        let cliArgs = GGUFSubprocessProvider.buildLlamaServerArgs(config: spawnCfg)
        #expect(cliArgs.contains("--mmproj"))
        #expect(cliArgs.contains(mmprojGGUF.standardizedFileURL.path))

        // 2. OpenAI vision + audio message serialization for llama-server
        let imgAtt = MessagesRequest.MediaAttachment(
            kind: "image",
            mediaType: "image/png",
            dataBase64: "iVBORw==",
            filename: "chart.png"
        )
        let audioAtt = MessagesRequest.MediaAttachment(
            kind: "audio",
            mediaType: "audio/wav",
            dataBase64: "UklGRg==",
            filename: "clip.wav"
        )
        let req = MessagesRequest(
            model: vlmDesc.modelID,
            messages: [
                MessagesRequest.MessageTurn(
                    role: "user",
                    content: "Analyze image and audio",
                    attachments: [imgAtt, audioAtt]
                )
            ],
            system: "Be concise"
        )
        let openAIMessages = GGUFSubprocessProvider.buildOpenAIChatMessages(from: req)
        #expect(openAIMessages.count == 2)
        let userParts = try #require(openAIMessages.last?["content"] as? [[String: Any]])
        #expect(userParts.count == 3)
        #expect((userParts[0]["type"] as? String) == "image_url")
        #expect((userParts[1]["type"] as? String) == "input_audio")
        #expect((userParts[2]["type"] as? String) == "text")

        // 3. /v1/admin/models/{id} PATCH sidecar + load/unload routes
        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase4_db_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: dbURL) }
        let store = try SQLiteStore(databaseURL: dbURL)
        let scheduler = ComputeScheduler()
        await scheduler.register(provider: GGUFSubprocessProvider(description: vlmDesc), description: vlmDesc)
        let router = HeylookRouter(store: store, scheduler: scheduler)

        let sidecarPatch = ModelSidecarConfig(
            id: vlmDesc.modelID,
            provider: .gguf,
            pinned: true,
            ctxSize: 65536,
            flashAttn: true,
            supportsVision: true,
            supportsThinking: true,
            sampling: SamplerSettings(temperature: 0.4, thinkingBudget: 8192),
            tags: ["vision", "pinned-vlm"]
        )
        let patchResp = await router.handle(
            method: "PATCH",
            path: "/v1/admin/models/\(vlmDesc.modelID)",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(sidecarPatch)
        )
        #expect(patchResp.status == 200)
        let updatedDesc = try JSONDecoder().decode(EngineDescription.self, from: patchResp.body)
        #expect(updatedDesc.isPinned == true)
        #expect(updatedDesc.contextSize.value == 65536)
        #expect(updatedDesc.contextSize.provenance == .configured)
        #expect(updatedDesc.tags.contains("pinned-vlm"))
        #expect(FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("model.heylook.toml").path))

        let loadResp = await router.handle(
            method: "POST",
            path: "/v1/admin/models/\(vlmDesc.modelID)/load",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(loadResp.status == 200)
        let loadedDesc = try JSONDecoder().decode(EngineDescription.self, from: loadResp.body)
        #expect(loadedDesc.isLoaded == true)

        let unloadResp = await router.handle(
            method: "POST",
            path: "/v1/admin/models/\(vlmDesc.modelID)/unload",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(unloadResp.status == 200)
        let unloadedDesc = try JSONDecoder().decode(EngineDescription.self, from: unloadResp.body)
        #expect(unloadedDesc.isLoaded == false)
    }
}
