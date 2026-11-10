import CryptoKit
import Darwin
import Foundation
import SQLite3
import Testing
@testable import HeylookCore
@testable import HeylookEmbeddings
@testable import HeylookEngines
@testable import HeylookServer
@testable import HeylookUI

@Suite("Heylook Core, Embeddings, Engines & Server Verification Suite", .serialized)
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

        let aliasSidecar = ModelSidecarConfig.parseTOML("context_size = 32768\n")
        #expect(aliasSidecar.ctxSize == 32768)

        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let exampleURL = repoRoot.appendingPathComponent("heylook.toml.example")
        let exampleText = try? String(contentsOf: exampleURL, encoding: .utf8)
        #expect(exampleText != nil)
        if let exampleText {
            let parsedExample = HeylookConfig.parseTOML(exampleText)
            #expect(parsedExample.serverHost == "127.0.0.1")
            #expect(parsedExample.serverPort == 8000)
            #expect(parsedExample.scanFolders == ["~/.heylook/models"])
        }
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
        await scheduler.register(provider: MLXSwiftProvider(description: desc, allowSyntheticFallback: true), description: desc)

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
        let provider = CoreAIProvider(description: baseDesc, allowSyntheticFallback: true)
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
        await scheduler.register(provider: MLXSwiftProvider(description: desc, allowSyntheticFallback: true), description: desc)
        let strictRouter = HeylookRouter(store: store, scheduler: scheduler)
        let router = HeylookRouter(
            store: store,
            scheduler: scheduler,
            embeddingEngine: DeterministicHashEmbeddingEngine()
        )

        // Truthful default: router without compiled .mlmodelc towers returns 503 on /v1/embeddings
        let strictEmbResp = await strictRouter.handle(
            method: "POST",
            path: "/v1/embeddings",
            headers: ["Host": "127.0.0.1:8000"],
            body: Data(#"{"model":"embedding-gemma2","input":"test"}"#.utf8)
        )
        #expect(strictEmbResp.status == 503)

        // 1. GET /v1/capabilities
        let capResp = await router.handle(method: "GET", path: "/v1/capabilities", headers: ["Host": "127.0.0.1:8000"])
        #expect(capResp.status == 200)
        let capJSON = try JSONSerialization.jsonObject(with: capResp.body) as? [String: Any]
        #expect((capJSON?["max_request_media_bytes"] as? Int) == EmbeddingRequestLimits.maxRequestMediaBytes)
        let embLimits = capJSON?["embedding_limits"] as? [String: Any]
        #expect((embLimits?["items"] as? Int) == EmbeddingRequestLimits.maxItems)
        #expect((embLimits?["blocks_per_item"] as? Int) == EmbeddingRequestLimits.maxBlocksPerItem)

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
        await scheduler.register(provider: MLXSwiftProvider(description: textOnlyDesc, allowSyntheticFallback: true), description: textOnlyDesc)
        await scheduler.register(provider: MLXSwiftProvider(description: visionDesc, allowSyntheticFallback: true), description: visionDesc)
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
        let syntheticSpawn = GGUFSpawnConfig(
            binaryPath: "/nonexistent/llama-server",
            modelPath: primaryGGUF.path
        )
        await scheduler.register(
            provider: GGUFSubprocessProvider(
                description: vlmDesc,
                spawnConfig: syntheticSpawn,
                allowSyntheticFallback: true
            ),
            description: vlmDesc
        )
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

    @Test("Vendor generation_config.json sampling layer, multi-turn reasoning_content forwarding, chat_template.heylook.jinja routes, and /image-plan")
    func testVendorSamplingReasoningContentAndTemplateRoutes() async throws {
        let rootDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase5_\(UUID().uuidString)", isDirectory: true)
        let modelDir = rootDir.appendingPathComponent("Qwen3.5-Vendor-MLX", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootDir) }

        // 1. Vendor generation_config.json sampling layer (builtin < vendor < modelSidecar < preset < request)
        let genCfgJSON = """
        {
          "temperature": 0.6,
          "top_p": 0.92,
          "top_k": 30
        }
        """
        try genCfgJSON.write(
            to: modelDir.appendingPathComponent("generation_config.json"),
            atomically: true,
            encoding: .utf8
        )
        let vendorSamplers = try #require(HeylookConfig.loadVendorSampling(in: modelDir))
        #expect(vendorSamplers.temperature == 0.6)
        #expect(vendorSamplers.topP == 0.92)
        #expect(vendorSamplers.topK == 30)

        let resolvedVendor = SamplerCascade.resolve(
            vendor: vendorSamplers,
            modelSidecar: SamplerSettings(topP: 0.85),
            preset: nil,
            request: nil
        )
        #expect(resolvedVendor.effective.temperature == 0.6)
        #expect(resolvedVendor.provenanceByField[.temperature] == .derived)
        #expect(resolvedVendor.effective.topP == 0.85)
        #expect(resolvedVendor.provenanceByField[.topP] == .configured)
        #expect(resolvedVendor.effective.topK == 30)
        #expect(resolvedVendor.provenanceByField[.topK] == .derived)

        // 2. Multi-turn reasoning_content forwarding to llama-server
        let multiTurnReq = MessagesRequest(
            model: "qwen3.5-gguf",
            messages: [
                MessagesRequest.MessageTurn(role: "user", content: "What is 6 * 7?"),
                MessagesRequest.MessageTurn(
                    role: "assistant",
                    content: "42.",
                    thinking: "Multiply 6 by 7 to get 42."
                ),
                MessagesRequest.MessageTurn(role: "user", content: "Double it.")
            ]
        )
        let openAIMsgs = GGUFSubprocessProvider.buildOpenAIChatMessages(from: multiTurnReq)
        #expect(openAIMsgs.count == 3)
        #expect((openAIMsgs[1]["role"] as? String) == "assistant")
        #expect((openAIMsgs[1]["content"] as? String) == "42.")
        #expect((openAIMsgs[1]["reasoning_content"] as? String) == "Multiply 6 by 7 to get 42.")

        // 3. chat_template.heylook.jinja GET/PUT/DELETE without mutating vendor chat_template.jinja
        let vendorTemplateURL = modelDir.appendingPathComponent("chat_template.jinja")
        try "VENDOR_TEMPLATE".write(to: vendorTemplateURL, atomically: true, encoding: .utf8)

        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase5_db_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: dbURL) }
        let store = try SQLiteStore(databaseURL: dbURL)
        let scheduler = ComputeScheduler()
        let desc = EngineDescription(
            modelID: "qwen3.5-vendor-mlx",
            provider: .mlx,
            modelPath: modelDir.path,
            architecture: ProvenanceField(value: "qwen3_5", provenance: .derived),
            quantization: ProvenanceField(value: "8bit", provenance: .derived),
            computePlacement: ProvenanceField(value: .mlxGPU, provenance: .derived),
            contextSize: ProvenanceField(value: 32768, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived),
            templateSource: ProvenanceField(value: "chat_template.jinja", provenance: .derived)
        )
        await scheduler.register(provider: MLXSwiftProvider(description: desc, allowSyntheticFallback: true), description: desc)
        let strictRouter = HeylookRouter(store: store, scheduler: scheduler)
        let router = HeylookRouter(
            store: store,
            scheduler: scheduler,
            embeddingEngine: DeterministicHashEmbeddingEngine()
        )

        // Truthful default: router without compiled .mlmodelc towers returns 503 on /v1/embeddings
        let strictEmbResp = await strictRouter.handle(
            method: "POST",
            path: "/v1/embeddings",
            headers: ["Host": "127.0.0.1:8000"],
            body: Data(#"{"model":"embedding-gemma2","input":"test"}"#.utf8)
        )
        #expect(strictEmbResp.status == 503)

        let putResp = await router.handle(
            method: "PUT",
            path: "/v1/admin/models/qwen3.5-vendor-mlx/chat-template",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONSerialization.data(withJSONObject: ["template": "OPERATOR_OVERRIDE_TEMPLATE"])
        )
        #expect(putResp.status == 200)
        let overrideURL = modelDir.appendingPathComponent("chat_template.heylook.jinja")
        #expect(FileManager.default.fileExists(atPath: overrideURL.path))
        #expect(try String(contentsOf: vendorTemplateURL, encoding: .utf8) == "VENDOR_TEMPLATE")
        let descAfterPut = await scheduler.modelDescription(for: "qwen3.5-vendor-mlx")
        #expect(descAfterPut?.templateSource.value == "chat_template.heylook.jinja")
        #expect(descAfterPut?.templateSource.provenance == .configured)

        let getResp = await router.handle(
            method: "GET",
            path: "/v1/admin/models/qwen3.5-vendor-mlx/chat-template",
            headers: ["Host": "127.0.0.1:8000"]
        )
        let getJSON = try JSONSerialization.jsonObject(with: getResp.body) as? [String: Any]
        #expect((getJSON?["has_override"] as? Bool) == true)
        #expect((getJSON?["source"] as? String) == "chat_template.heylook.jinja")
        #expect((getJSON?["template"] as? String) == "OPERATOR_OVERRIDE_TEMPLATE")

        let delResp = await router.handle(
            method: "DELETE",
            path: "/v1/admin/models/qwen3.5-vendor-mlx/chat-template",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(delResp.status == 200)
        #expect(!FileManager.default.fileExists(atPath: overrideURL.path))
        #expect(try String(contentsOf: vendorTemplateURL, encoding: .utf8) == "VENDOR_TEMPLATE")
        let descAfterDelete = await scheduler.modelDescription(for: "qwen3.5-vendor-mlx")
        #expect(descAfterDelete?.templateSource.value == "chat_template.jinja")
        #expect(descAfterDelete?.templateSource.provenance == .derived)

        // 4. POST /v1/models/{id}/image-plan (409 when unloaded, 200 when loaded) & POST /v1/conversations/{id}/prompt
        let unloadedPlanResp = await router.handle(
            method: "POST",
            path: "/v1/models/qwen3.5-vendor-mlx/image-plan",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(unloadedPlanResp.status == 409)

        _ = await router.handle(
            method: "POST",
            path: "/v1/admin/models/qwen3.5-vendor-mlx/load",
            headers: ["Host": "127.0.0.1:8000"]
        )
        let loadedPlanResp = await router.handle(
            method: "POST",
            path: "/v1/models/qwen3.5-vendor-mlx/image-plan",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(loadedPlanResp.status == 200)
        let planJSON = try JSONSerialization.jsonObject(with: loadedPlanResp.body) as? [String: Any]
        #expect((planJSON?["tokens_per_image"] as? Int) == 280)

        let conv = ConversationRecord(id: "conv_p5", title: "Prompt Inspector Test", defaultModelID: "qwen3.5-vendor-mlx")
        try store.upsertConversation(conv)
        _ = try store.appendMessage(MessageRecord(conversationID: "conv_p5", position: 0, role: "user", content: "Inspect prompt"))
        let promptResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/conv_p5/prompt",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(promptResp.status == 200)
        let promptJSON = try JSONSerialization.jsonObject(with: promptResp.body) as? [String: Any]
        #expect((promptJSON?["message_count"] as? Int) == 1)
    }

    @Test("Provider unload(), LRU eviction subprocess termination, POST /v1/admin/models/unload-all, and strict ProviderExecutionError")
    func testProviderUnloadSubprocessLifecycleAndStrictErrors() async throws {
        let scheduler = ComputeScheduler(maxLoadedGPUModels: 1, maxQueueDepth: 2)

        let ggufDesc = EngineDescription(
            modelID: "qwen3-gguf-live",
            provider: .gguf,
            modelPath: "/tmp/nonexistent-qwen3.gguf",
            architecture: ProvenanceField(value: "qwen3", provenance: .derived),
            quantization: ProvenanceField(value: "Q8_0", provenance: .derived),
            computePlacement: ProvenanceField(value: .ggufSubprocess, provenance: .derived),
            contextSize: ProvenanceField(value: 8192, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: false, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let sleepSpawn = GGUFSpawnConfig(
            binaryPath: "/bin/sleep",
            modelPath: "/tmp/nonexistent-qwen3.gguf"
        )
        let ggufProvider = GGUFSubprocessProvider(description: ggufDesc, spawnConfig: sleepSpawn, allowSyntheticFallback: true)

        let mlxDesc = EngineDescription(
            modelID: "gemma4-mlx-evictor",
            provider: .mlx,
            modelPath: "/tmp/nonexistent-gemma4",
            architecture: ProvenanceField(value: "gemma4", provenance: .derived),
            quantization: ProvenanceField(value: "4bit", provenance: .derived),
            computePlacement: ProvenanceField(value: .mlxGPU, provenance: .derived),
            contextSize: ProvenanceField(value: 8192, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: false, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let mlxProvider = MLXSwiftProvider(description: mlxDesc)

        await scheduler.register(provider: ggufProvider, description: ggufDesc)
        await scheduler.register(provider: mlxProvider, description: mlxDesc)

        // 1. Activate GGUF model, warm its PrefixCacheTracker, and spawn a real child process (/bin/sleep 60)
        _ = try await scheduler.activateModel("qwen3-gguf-live")
        let registry = RequestRegistry()
        let warmStream = await ggufProvider.generate(
            requestID: "req_warm_gguf",
            request: MessagesRequest(
                model: "qwen3-gguf-live",
                messages: [MessagesRequest.MessageTurn(role: "user", content: "Warm prefix cache")]
            ),
            effectiveSampling: .builtinDefaults,
            registry: registry
        )
        for try await _ in warmStream {}
        #expect(await ggufProvider.isPrefixCacheWarm() == true)

        #if os(macOS)
        let childPID = try await ggufProvider.launchSubprocess(customArguments: ["60"], checkHealth: false)
        #expect(childPID > 0)
        #expect(SubprocessLifecycleRegistry.shared.isRunning(for: "qwen3-gguf-live") == true)
        #expect(kill(childPID, 0) == 0)
        #endif

        // 2. Activating second GPU model under maxLoadedGPUModels = 1 evicts qwen3-gguf-live,
        //    resets its PrefixCacheTracker, and terminates its OS child process!
        _ = try await scheduler.activateModel("gemma4-mlx-evictor")
        #expect(await ggufProvider.totalUnloadCount() == 1)
        #expect(await ggufProvider.isPrefixCacheWarm() == false)
        #if os(macOS)
        #expect(SubprocessLifecycleRegistry.shared.isRunning(for: "qwen3-gguf-live") == false)
        #expect(SubprocessLifecycleRegistry.shared.activeSubprocessCount == 0)
        #expect(kill(childPID, 0) != 0)
        #endif

        // 3. POST /v1/admin/models/unload-all unloads all remaining models
        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase7_unload_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: dbURL) }
        let store = try SQLiteStore(databaseURL: dbURL)
        let router = HeylookRouter(store: store, scheduler: scheduler)

        let unloadAllResp = await router.handle(
            method: "POST",
            path: "/v1/admin/models/unload-all",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(unloadAllResp.status == 200)
        let metricsAfterUnloadAll = await scheduler.metrics()
        #expect(metricsAfterUnloadAll.loadedGPUModels.isEmpty)
        #expect(metricsAfterUnloadAll.loadedANEModels.isEmpty)
        #expect(await mlxProvider.totalUnloadCount() == 1)

        // 4. Strict provider mode (allowSyntheticFallback: false) throws ProviderExecutionError
        let strictDesc = EngineDescription(
            modelID: "strict-missing-mlx",
            provider: .mlx,
            modelPath: "/tmp/missing_checkpoint_\(UUID().uuidString)",
            architecture: ProvenanceField(value: "qwen3", provenance: .derived),
            quantization: ProvenanceField(value: "4bit", provenance: .derived),
            computePlacement: ProvenanceField(value: .mlxGPU, provenance: .derived),
            contextSize: ProvenanceField(value: 8192, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: false, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let strictProvider = MLXSwiftProvider(description: strictDesc, allowSyntheticFallback: false)
        await scheduler.register(provider: strictProvider, description: strictDesc)

        let strictReq = MessagesRequest(
            model: "strict-missing-mlx",
            messages: [MessagesRequest.MessageTurn(role: "user", content: "Should fail loudly")]
        )
        let strictResp = await router.handle(
            method: "POST",
            path: "/v1/messages",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(strictReq)
        )
        #expect(strictResp.status == 500)
        let errText = String(decoding: strictResp.body, as: UTF8.self)
        #expect(errText.contains("checkpoint_not_found"))
    }

    @Test("Incremental SSE frame callback streaming and model-derived /image-plan and /prompt routes")
    func testIncrementalSSEStreamingAndDerivedImagePlanAndPromptRoutes() async throws {
        let rootDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase7_derived_\(UUID().uuidString)", isDirectory: true)
        let gemmaDir = rootDir.appendingPathComponent("Gemma-4-Vision-IT", isDirectory: true)
        try FileManager.default.createDirectory(at: gemmaDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootDir) }

        // Write config.json with custom vision_config (image_size 896, patch_size 14, spatial_merge_size 2 -> (64*64)/(2*2) = 1024 tokens)
        let configJSON = """
        {
          "model_type": "gemma4",
          "vision_config": {
            "image_size": 896,
            "patch_size": 14,
            "spatial_merge_size": 2
          }
        }
        """
        try configJSON.write(to: gemmaDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "<start_of_turn>user\n{{ messages }}<end_of_turn>".write(
            to: gemmaDir.appendingPathComponent("chat_template.jinja"),
            atomically: true,
            encoding: .utf8
        )

        #expect(HeylookConfig.estimateImageTokenBudget(in: gemmaDir) == 1024)

        let dbURL = rootDir.appendingPathComponent("phase7.sqlite")
        let store = try SQLiteStore(databaseURL: dbURL)
        let scheduler = ComputeScheduler()
        let gemmaDesc = EngineDescription(
            modelID: "gemma-4-vision-it",
            provider: .mlx,
            modelPath: gemmaDir.path,
            architecture: ProvenanceField(value: "gemma4", provenance: .derived),
            quantization: ProvenanceField(value: "4bit", provenance: .derived),
            computePlacement: ProvenanceField(value: .mlxGPU, provenance: .derived),
            isLoaded: true,
            contextSize: ProvenanceField(value: 32768, provenance: .derived),
            flashAttention: ProvenanceField(value: true, provenance: .derived),
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived),
            templateSource: ProvenanceField(value: "chat_template.jinja", provenance: .derived)
        )
        await scheduler.register(provider: MLXSwiftProvider(description: gemmaDesc, allowSyntheticFallback: true), description: gemmaDesc)
        let router = HeylookRouter(store: store, scheduler: scheduler)

        // 1. Model-derived /v1/models/{id}/image-plan returns 1024 from vision_config
        let planResp = await router.handle(
            method: "POST",
            path: "/v1/models/gemma-4-vision-it/image-plan",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(planResp.status == 200)
        let planJSON = try JSONSerialization.jsonObject(with: planResp.body) as? [String: Any]
        #expect((planJSON?["tokens_per_image"] as? Int) == 1024)

        // 2. Model-derived /v1/conversations/{id}/prompt formats Gemma <start_of_turn> markers
        let conv = ConversationRecord(
            id: "conv_gemma",
            title: "Gemma Prompt Test",
            systemPrompt: "Answer concisely.",
            defaultModelID: "gemma-4-vision-it"
        )
        try store.upsertConversation(conv)
        _ = try store.appendMessage(
            MessageRecord(
                conversationID: "conv_gemma",
                position: 0,
                role: "user",
                content: "Describe the vision tower."
            )
        )
        let promptResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/conv_gemma/prompt",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(promptResp.status == 200)
        let promptJSON = try JSONSerialization.jsonObject(with: promptResp.body) as? [String: Any]
        let renderedPrompt = (promptJSON?["prompt"] as? String) ?? ""
        #expect(renderedPrompt.contains("<start_of_turn>user"))
        #expect(renderedPrompt.contains("<end_of_turn>"))
        #expect((promptJSON?["template_source"] as? String) == "chat_template.jinja")

        // 3. Incremental SSE frame streaming via onSSEFrame
        actor FrameCollector {
            var frames: [String] = []
            func append(_ frame: String) {
                frames.append(frame)
            }
        }
        let collector = FrameCollector()
        let streamReq = MessagesRequest(
            model: "gemma-4-vision-it",
            messages: [MessagesRequest.MessageTurn(role: "user", content: "Stream incremental SSE")],
            stream: true,
            thinking: true
        )
        let sseResp = await router.handle(
            method: "POST",
            path: "/v1/messages",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(streamReq),
            onSSEFrame: { frame in
                await collector.append(frame)
            }
        )
        #expect(sseResp.status == 200)
        let collected = await collector.frames
        #expect(collected.count >= 4)
        #expect(collected.first?.hasPrefix("event: message_start") == true)
        #expect(collected.contains(where: { $0.hasPrefix("event: content_block_delta") }))
        #expect(collected.contains(where: { $0.hasPrefix("event: message_delta") }))
        #expect(collected.last?.hasPrefix("event: message_stop") == true)
    }

    @Test("MLX multi-turn KV cache session slicing, exact tokenCounter, and speculative drafter telemetry")
    func testMLXMultiTurnKVCacheSlicingExactTokenCounterAndSpeculativeDrafter() async throws {
        let rootDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase8_mlx_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: rootDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootDir) }

        // Write model.safetensors.index.json with MTP weights so discoverDrafter auto-detects draft-mtp
        let indexJSON = """
        {
          "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
            "mtp.layers.0.head.weight": "model-00001.safetensors"
          }
        }
        """
        try indexJSON.write(
            to: rootDir.appendingPathComponent("model.safetensors.index.json"),
            atomically: true,
            encoding: .utf8
        )

        let desc = EngineDescription(
            modelID: "qwen3.5-mtp-mlx",
            provider: .mlx,
            modelPath: rootDir.path
        )
        let provider = MLXSwiftProvider(
            description: desc,
            allowSyntheticFallback: true,
            tokenCounter: { text in max(2, text.count) }
        )

        let initialKV = await provider.kvSessionState()
        #expect(initialKV.speculativeMode == "draft-mtp")
        #expect(initialKV.speculativeDrafterPath != nil)

        let reg = RequestRegistry()
        var turn1Final = GenerationChunk()
        for try await chunk in await provider.generate(
            requestID: "p8_t1",
            request: MessagesRequest(
                model: "qwen3.5-mtp-mlx",
                messages: [.init(role: "user", content: "First turn message")],
                system: "System v1"
            ),
            effectiveSampling: .builtinDefaults,
            registry: reg
        ) {
            turn1Final = chunk
        }
        #expect(turn1Final.cacheReport?.outcome == .miss)
        #expect(turn1Final.cacheReport?.cause == "cold")
        // Truthful runtime: never fabricate SpeculativeReport when no real drafter executed
        #expect(turn1Final.speculativeReport == nil)

        var turn2Final = GenerationChunk()
        for try await chunk in await provider.generate(
            requestID: "p8_t2",
            request: MessagesRequest(
                model: "qwen3.5-mtp-mlx",
                messages: [
                    .init(role: "user", content: "First turn message"),
                    .init(role: "assistant", content: "[qwen3.5-mtp-mlx] Processed: First turn message"),
                    .init(role: "user", content: "Second turn follow up"),
                ],
                system: "System v1"
            ),
            effectiveSampling: .builtinDefaults,
            registry: reg
        ) {
            turn2Final = chunk
        }
        #expect(turn2Final.cacheReport?.outcome == .reused)
        let kvAfterTurn2 = await provider.kvSessionState()
        #expect(kvAfterTurn2.activeTurnCount == 4)
        #expect(kvAfterTurn2.sliceTruncations == 0)

        // Branch edit on turn 2: reuses turn 1 + assistant 1 prefix and slices trailing KV entries
        var branchFinal = GenerationChunk()
        for try await chunk in await provider.generate(
            requestID: "p8_t2_branch",
            request: MessagesRequest(
                model: "qwen3.5-mtp-mlx",
                messages: [
                    .init(role: "user", content: "First turn message"),
                    .init(role: "assistant", content: "[qwen3.5-mtp-mlx] Processed: First turn message"),
                    .init(role: "user", content: "Edited second turn branch"),
                ],
                system: "System v1"
            ),
            effectiveSampling: .builtinDefaults,
            registry: reg
        ) {
            branchFinal = chunk
        }
        #expect(branchFinal.cacheReport?.outcome == .reused)
        let kvAfterBranch = await provider.kvSessionState()
        #expect(kvAfterBranch.sliceTruncations == 1)

        await provider.unload()
        let kvAfterUnload = await provider.kvSessionState()
        #expect(kvAfterUnload.activeTurnCount == 0)
        #expect(kvAfterUnload.cachedTokenCount == 0)
        #expect(kvAfterUnload.sliceTruncations == 0)
    }

    @Test("GGUF header KV metadata reader, /props provenance parser, and dynamic loopback port pool")
    func testGGUFHeaderMetadataReaderPropsParserAndDynamicPortPool() async throws {
        let rootDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase9_gguf_\(UUID().uuidString)", isDirectory: true)
        let modelDir = rootDir.appendingPathComponent("Gemma-4-GGUF", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootDir) }

        let ggufBytes = GGUFHeaderReader.buildMinimalGGUF(
            version: 3,
            tensorCount: 42,
            stringMetadata: [
                "general.architecture": "gemma4",
                "tokenizer.chat_template": "<start_of_turn>user\n{{ content }}<end_of_turn>",
            ],
            uint32Metadata: [
                "gemma4.context_length": 65536,
            ]
        )
        let ggufURL = modelDir.appendingPathComponent("Gemma-4-26B-Q8_0.gguf")
        try ggufBytes.write(to: ggufURL)

        let parsedHeader = GGUFHeaderReader.readFileHeader(at: ggufURL)
        #expect(parsedHeader?.version == 3)
        #expect(parsedHeader?.tensorCount == 42)
        #expect(parsedHeader?.architecture == "gemma4")
        #expect(parsedHeader?.contextLength == 65536)
        #expect(parsedHeader?.embeddedChatTemplate?.contains("<start_of_turn>") == true)

        let discovered = HeylookConfig.discoverModels(in: [rootDir])
        #expect(discovered.count == 1)
        #expect(discovered.first?.architecture.value == "gemma4")
        #expect(discovered.first?.contextSize.value == 65536)

        let preview = HeylookConfig.renderPromptPreview(
            in: modelDir,
            system: "Be concise.",
            messages: [(role: "user", content: "Hello GGUF", thinking: nil)]
        )
        #expect(preview.source == "gguf_embedded")
        #expect(preview.prompt.contains("<start_of_turn>user"))

        // Dynamic port pool assigns non-colliding ports and releases on unload
        let descA = EngineDescription(modelID: "gguf-model-a", provider: .gguf, modelPath: ggufURL.path)
        let descB = EngineDescription(modelID: "gguf-model-b", provider: .gguf, modelPath: ggufURL.path)
        let provA = GGUFSubprocessProvider(description: descA)
        let provB = GGUFSubprocessProvider(description: descB)
        #expect(provA.spawnConfig.port != provB.spawnConfig.port)
        #expect(provA.spawnConfig.port >= 18080)
        #expect(provB.spawnConfig.port >= 18080)

        // /props JSON parser and observed provenance upgrade
        let propsPayload = Data("""
        {
          "default_generation_settings": {
            "n_ctx": 131072
          },
          "chat_template": "<start_of_turn>user...",
          "total_slots": 1
        }
        """.utf8)
        let props = GGUFSubprocessProvider.parsePropsJSON(propsPayload)
        #expect(props?.contextSize == 131072)
        #expect(props?.totalSlots == 1)
        if let props {
            await provA.applyObservedProps(props)
        }
        let updatedA = await provA.describe()
        #expect(updatedA.contextSize.value == 131072)
        #expect(updatedA.contextSize.provenance == .observed)
        #expect(updatedA.templateSource.value == "llama-server:/props")
        #expect(updatedA.templateSource.provenance == .observed)

        await provA.unload()
        await provB.unload()
        #expect(GGUFPortPool.shared.assignedPort(for: "gguf-model-a") == nil)
        #expect(GGUFPortPool.shared.assignedPort(for: "gguf-model-b") == nil)
    }

    @Test("ANE speech transcription, disconnect-resilient voice-generate handoff, and EvalLabSuite")
    func testANESpeechTranscriptionVoiceGenerateHandoffAndEvalLab() async throws {
        // Synthesize a 0.25s 16 kHz mono 16-bit PCM WAV buffer with an inline TEXT: marker
        let sampleRate: UInt32 = 16_000
        let sampleCount = 4_000
        var pcmData = Data(capacity: sampleCount * 2)
        for idx in 0..<sampleCount {
            let val = Int16(sin(Double(idx) * 0.1) * 8000.0).littleEndian
            withUnsafeBytes(of: val) { pcmData.append(contentsOf: $0) }
        }
        var wav = Data("RIFF".utf8)
        var chunkSize = UInt32(36 + pcmData.count).littleEndian
        withUnsafeBytes(of: &chunkSize) { wav.append(contentsOf: $0) }
        wav.append(Data("WAVEfmt ".utf8))
        var fmtLen = UInt32(16).littleEndian
        var audioFmt = UInt16(1).littleEndian
        var channels = UInt16(1).littleEndian
        var srLE = sampleRate.littleEndian
        var byteRate = (sampleRate * 2).littleEndian
        var blockAlign = UInt16(2).littleEndian
        var bitsPerSample = UInt16(16).littleEndian
        withUnsafeBytes(of: &fmtLen) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &audioFmt) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &channels) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &srLE) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &byteRate) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &blockAlign) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &bitsPerSample) { wav.append(contentsOf: $0) }
        wav.append(Data("data".utf8))
        var dataLen = UInt32(pcmData.count).littleEndian
        withUnsafeBytes(of: &dataLen) { wav.append(contentsOf: $0) }
        wav.append(pcmData)
        wav.append(Data("TEXT:Summarize dual-lane ANE scheduling".utf8))

        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_phase10_voice_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let store = try SQLiteStore(databaseURL: tempDir.appendingPathComponent("voice.sqlite"))
        let scheduler = ComputeScheduler()
        let desc = EngineDescription(
            modelID: "voice-chat-mlx",
            provider: .mlx,
            modelPath: tempDir.path,
            isLoaded: true
        )
        await scheduler.register(provider: MLXSwiftProvider(description: desc, allowSyntheticFallback: true), description: desc)
        let strictVoiceRouter = HeylookRouter(store: store, scheduler: scheduler)
        let router = HeylookRouter(store: store, scheduler: scheduler, allowSyntheticFallback: true)

        let voiceReq = ConversationVoiceGenerateRequest(
            model: "voice-chat-mlx",
            audioBase64: wav.base64EncodedString(),
            languageHint: "en"
        )
        // Truthful default: even if WAV contains TEXT: bytes, production router (allowSyntheticFallback: false) rejects without real ASR
        let strictVoiceResp = await strictVoiceRouter.handle(
            method: "POST",
            path: "/v1/conversations/conv_voice_strict/voice-generate",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(voiceReq)
        )
        #expect(strictVoiceResp.status == 503)

        let resp = await router.handle(
            method: "POST",
            path: "/v1/conversations/conv_voice_1/voice-generate",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(voiceReq)
        )
        #expect(resp.status == 200)
        let bodyText = String(decoding: resp.body, as: UTF8.self)
        #expect(bodyText.contains("event: heylook_voice_transcribed"))
        #expect(bodyText.contains("Summarize dual-lane ANE scheduling"))
        #expect(bodyText.contains("event: heylook_saved"))

        let savedMsgs = try store.listMessages(conversationID: "conv_voice_1")
        #expect(savedMsgs.count == 2)
        #expect(savedMsgs.first?.content == "Summarize dual-lane ANE scheduling")
        #expect(savedMsgs.last?.role == "assistant")

        // Truthful default: EvalLabSuite.runAll() fails honestly when no real weights/towers are discovered
        let strictEvalReport = try await EvalLabSuite.runAll(allowSyntheticSelfTest: false)
        #expect(strictEvalReport.allPassed == false)

        // Offline pipeline contract self-test (allowSyntheticSelfTest: true)
        let evalReport = try await EvalLabSuite.runAll(allowSyntheticSelfTest: true)
        #expect(evalReport.allPassed == true)
        #expect(evalReport.probes.count == 3)
        #expect(evalReport.matryoshkaBenchmarks.count == 4)
    }

    @Test("Explicit [[models]] discovery, canonical 5-layer SamplerCascade on HTTP routes, and unified store paths")
    func testExplicitModelsDiscoveryCanonicalSamplerCascadeAndUnifiedStoragePaths() async throws {
        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_audit_\(UUID().uuidString)", isDirectory: true)
        let scanDir = tempRoot.appendingPathComponent("scanned", isDirectory: true)
        let scannedModelDir = scanDir.appendingPathComponent("Scanned-MLX-Model", isDirectory: true)
        let explicitModelDir = tempRoot.appendingPathComponent("Explicit-External-Model", isDirectory: true)
        try FileManager.default.createDirectory(at: scannedModelDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: explicitModelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        try Data("stub".utf8).write(to: scannedModelDir.appendingPathComponent("model.safetensors"))
        try """
        {"model_type": "qwen3_5", "max_position_embeddings": 16384}
        """.write(to: scannedModelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        // Write vendor generation_config.json + model.heylook.toml in explicitModelDir
        try """
        {"temperature": 0.55, "top_p": 0.91, "top_k": 25}
        """.write(to: explicitModelDir.appendingPathComponent("generation_config.json"), atomically: true, encoding: .utf8)
        try """
        thinking = false
        max_tokens = 512
        """.write(to: explicitModelDir.appendingPathComponent("model.heylook.toml"), atomically: true, encoding: .utf8)

        let tomlText = """
        default_model = "custom-explicit-mlx"
        scan_folders = ["\(scanDir.path)"]

        [[models]]
        id = "custom-explicit-mlx"
        model_path = "\(explicitModelDir.path)"
        provider = "mlx"
        ctx_size = 65536
        pinned = true
        thinking_capable = true
        """
        let cfg = HeylookConfig.parseTOML(tomlText)
        let discovered = cfg.discoverConfiguredModels()
        #expect(discovered.count == 2)
        let explicitDesc = try #require(discovered.first(where: { $0.modelID == "custom-explicit-mlx" }))
        #expect(explicitDesc.contextSize.value == 65536)
        #expect(explicitDesc.contextSize.provenance == .configured)
        #expect(explicitDesc.isPinned == true)

        // Verify 5-layer SamplerCascade resolution via HeylookConfig.resolveEffectiveSampling
        let resolved = HeylookConfig.resolveEffectiveSampling(
            modelPath: explicitDesc.modelPath,
            preset: SamplerSettings(topK: 50),
            request: SamplerSettings(temperature: 0.2)
        )
        #expect(resolved.effective.temperature == 0.2)
        #expect(resolved.provenanceByField[.temperature] == .configured)
        #expect(resolved.effective.topP == 0.91)
        #expect(resolved.provenanceByField[.topP] == .derived)
        #expect(resolved.effective.topK == 50)
        #expect(resolved.provenanceByField[.topK] == .configured)
        #expect(resolved.effective.maxTokens == 512)
        #expect(resolved.provenanceByField[.maxTokens] == .configured)
        #expect(resolved.effective.thinking == false)

        // Verify HeylookRouter honors model.heylook.toml `thinking = false` when request omits `thinking`
        let store = try SQLiteStore(databaseURL: tempRoot.appendingPathComponent("audit.sqlite"))
        let scheduler = ComputeScheduler()
        await scheduler.register(provider: MLXSwiftProvider(description: explicitDesc, allowSyntheticFallback: true), description: explicitDesc)
        let router = HeylookRouter(store: store, scheduler: scheduler)

        let msgReq = MessagesRequest(
            model: "custom-explicit-mlx",
            messages: [MessagesRequest.MessageTurn(role: "user", content: "Check sidecar thinking override")]
        )
        let resp = await router.handle(
            method: "POST",
            path: "/v1/messages",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(msgReq)
        )
        #expect(resp.status == 200)
        let decoded = try JSONDecoder().decode(MessagesResponse.self, from: resp.body)
        // Because `thinking = false` in model.heylook.toml and request.thinking == nil, no thinking block is emitted!
        let hasThinkingBlock = decoded.content.contains { block in
            block.type == "thinking" || (block.thinking?.isEmpty == false)
        }
        #expect(hasThinkingBlock == false)

        // Verify unified defaultStoreURL and defaultVectorStoreURL share defaultAppSupportDirectory
        #expect(HeylookConfig.defaultStoreURL.deletingLastPathComponent() == HeylookConfig.defaultAppSupportDirectory)
        #expect(HeylookConfig.defaultVectorStoreURL.deletingLastPathComponent() == HeylookConfig.defaultAppSupportDirectory)
    }

    @Test("File-derived ModelTemplateInfo, 4-protocol ReasoningParser, CoreAI 0.2 manifest, and GGUF BOOL/FLOAT32/SSM metadata")
    func testModelTemplateInfoMultiProtocolReasoningParserCoreAISchema02AndGGUFExtendedMetadata() async throws {
        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_model_families_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        // 1. Qwen 3.5 / 3.8 template (`<think>` prefill + `enable_thinking`)
        let qwenInfo = ModelTemplateInfo.analyze(
            templateText: """
            {% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}{% endif %}
            """,
            specialTokens: ["<|im_end|>"]
        )
        #expect(qwenInfo.reasoningFormat == .tagPair(open: "<think>", close: "</think>", prefillsThinking: true))
        #expect(qwenInfo.thinkingControls.supportsThinking == true)
        #expect(qwenInfo.thinkingControls.switchVariable == "enable_thinking")
        #expect(qwenInfo.strippableSpecialTokens.contains("<|im_end|>") == true)
        #expect(qwenInfo.strippableSpecialTokens.contains("<think>") == false)

        // 2. Gemma 4 template + tokenizer.json (`<|channel>thought\n...<channel|>` + `<turn|>`)
        let gemma4Dir = tempRoot.appendingPathComponent("Gemma-4-31B-IT", isDirectory: true)
        try FileManager.default.createDirectory(at: gemma4Dir, withIntermediateDirectories: true)
        try """
        {
          "added_tokens": [
            {"id": 100, "content": "<|channel>", "special": true},
            {"id": 101, "content": "<channel|>", "special": true},
            {"id": 102, "content": "<turn|>", "special": true}
          ]
        }
        """.write(to: gemma4Dir.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try """
        {% if enable_thinking is defined and enable_thinking %}<|channel>thought\n{% endif %}{{ content }}<turn|>
        """.write(to: gemma4Dir.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)

        let gemma4Info = ModelTemplateInfo.inspect(modelDirectory: gemma4Dir)
        #expect(gemma4Info.reasoningFormat == .gemmaChannel(openPrefix: "<|channel>", closeTag: "<channel|>"))
        #expect(gemma4Info.thinkingControls.supportsThinking == true)
        #expect(gemma4Info.thinkingControls.defaultEnabled == false)
        #expect(gemma4Info.strippableSpecialTokens.contains("<turn|>") == true)
        #expect(gemma4Info.strippableSpecialTokens.contains("<|channel>") == false)

        // Stream Gemma 4 labeled channel across chunk boundaries
        var gemmaParser = ReasoningParser(format: gemma4Info.reasoningFormat, thinkingEnabled: true)
        var gemmaScrubber = StripSpecials(tokens: Array(gemma4Info.strippableSpecialTokens))
        let gemmaChunks = ["<|chan", "nel>thought\n", "Step 1: check ", "Gemma 4<chan", "nel|>Final ", "answer<tu", "rn|>"]
        var gemmaThinking = ""
        var gemmaText = ""
        for piece in gemmaChunks {
            let d = gemmaParser.feed(piece)
            gemmaThinking += d.thinking
            gemmaText += gemmaScrubber.feed(d.text)
        }
        let gemmaTail = gemmaParser.finalize()
        gemmaThinking += gemmaTail.thinking
        gemmaText += gemmaScrubber.feed(gemmaTail.text) + gemmaScrubber.finalize()
        #expect(gemmaThinking == "Step 1: check Gemma 4")
        #expect(gemmaText == "Final answer")

        // 3. OpenAI Harmony channel (`<|channel|>analysis<|message|>...` + `reasoning_effort`)
        let harmonyInfo = ModelTemplateInfo.analyze(
            templateText: """
            {% if reasoning_effort in ["low", "medium", "high"] %}<|start|>assistant<|channel|>analysis<|message|>{% endif %}
            """
        )
        #expect(harmonyInfo.reasoningFormat == .harmonyChannel)
        #expect(harmonyInfo.thinkingControls.depthVariable == "reasoning_effort")
        #expect(harmonyInfo.thinkingControls.depthValues == ["low", "medium", "high"])

        var harmonyParser = ReasoningParser(format: .harmonyChannel)
        let harmonyStream = [
            "<|channel|>analy",
            "sis<|message|>Weighing ",
            "tradeoffs<|end|><|start|>assistant<|channel|>fin",
            "al<|message|>Harmony reply<|end|>"
        ]
        var harmonyThinking = ""
        var harmonyText = ""
        for piece in harmonyStream {
            let d = harmonyParser.feed(piece)
            harmonyThinking += d.thinking
            harmonyText += d.text
        }
        let harmonyTail = harmonyParser.finalize()
        harmonyThinking += harmonyTail.thinking
        harmonyText += harmonyTail.text
        #expect(harmonyThinking == "Weighing tradeoffs")
        #expect(harmonyText == "Harmony reply")

        // 4. Muse-Glimmer Recipient channel (`to=self<|message|>...<|eom|>` + `reasoning_strength`)
        let recipientInfo = ModelTemplateInfo.analyze(
            templateText: """
            {% if reasoning_strength in ['none', 'low', 'medium', 'high', 'max'] %}to=self<|message|>...<|eom|>to=user<|message|>...<|eot|>{% endif %}
            """
        )
        #expect(recipientInfo.reasoningFormat == .recipientChannel)
        #expect(recipientInfo.thinkingControls.depthVariable == "reasoning_strength")
        #expect(recipientInfo.thinkingControls.depthValues == ["none", "low", "medium", "high", "max"])

        var recipientParser = ReasoningParser(format: .recipientChannel)
        let recipientStream = [
            "to=se",
            "lf<|message|>Drafting ",
            "plan<|eom|><|start|>assistant to=us",
            "er<|message|>Visible Muse response<|eot|>"
        ]
        var recipientThinking = ""
        var recipientText = ""
        for piece in recipientStream {
            let d = recipientParser.feed(piece)
            recipientThinking += d.thinking
            recipientText += d.text
        }
        let recipientTail = recipientParser.finalize()
        recipientThinking += recipientTail.thinking
        recipientText += recipientTail.text
        #expect(recipientThinking == "Drafting plan")
        #expect(recipientText == "Visible Muse response")

        // 5. Non-thinking template (e.g. Llama 3.2) -> supportsThinking == false
        let plainInfo = ModelTemplateInfo.analyze(
            templateText: "<|begin_of_text|>{% for m in messages %}{{ m.content }}<|eot_id|>{% endfor %}"
        )
        #expect(plainInfo.reasoningFormat == .none)
        #expect(plainInfo.thinkingControls.supportsThinking == false)

        // 6. CoreAI Schema 0.2 nested `.aimodel/metadata.json` inspection
        let coreaiBundle = tempRoot.appendingPathComponent("Qwen3_5_VLM.aimodel", isDirectory: true)
        try FileManager.default.createDirectory(at: coreaiBundle, withIntermediateDirectories: true)
        try """
        {
          "schema_version": "0.2",
          "model_type": "vlm",
          "tokenizer": "Qwen/Qwen3.5-4B",
          "assets": {
            "main": "main.mlmodelc",
            "vision": "vision.mlmodelc",
            "drafter": "dflash.mlmodelc",
            "tokenizer_dir": "tokenizer"
          },
          "language": {
            "max_context_length": 32768,
            "prefill_chunk_size": 128,
            "vocab_size": 151936
          },
          "vision": {
            "backbone": "qwen3_5_vit",
            "token_merge": 2
          },
          "speculative": {
            "drafter_kind": "dflash",
            "num_prediction_heads": 4
          }
        }
        """.write(to: coreaiBundle.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)

        let manifest = try CoreAIBundleManifest.inspect(at: coreaiBundle)
        #expect(manifest.schemaVersion == "0.2")
        #expect(manifest.name == "Qwen3.5-4B")
        #expect(manifest.kind == "vlm")
        #expect(manifest.maxContextLength == 32768)
        #expect(manifest.prefillChunkSize == 128)
        #expect(manifest.hasVisionComponent == true)
        #expect(manifest.visionBackbone == "qwen3_5_vit")
        #expect(manifest.hasDrafterComponent == true)
        #expect(manifest.drafterKind == "dflash")

        // 7. GGUF BOOL (`clip.has_audio_encoder`), FLOAT32 (`general.sampling.temp`), and SSM (`*.ssm.conv_kernel`)
        let ggufDir = tempRoot.appendingPathComponent("Qwen3.5-Omni-GGUF", isDirectory: true)
        try FileManager.default.createDirectory(at: ggufDir, withIntermediateDirectories: true)
        let mainGGUF = GGUFHeaderReader.buildMinimalGGUF(
            stringMetadata: [
                "general.architecture": "qwen35",
                "tokenizer.chat_template": "{% if enable_thinking %}<think>\n{% endif %}",
            ],
            uint32Metadata: [
                "qwen35.context_length": 65536,
                "qwen35.ssm.conv_kernel": 4,
                "qwen35.nextn_predict_layers": 1,
                "general.sampling.top_k": 20,
            ],
            float32Metadata: [
                "general.sampling.temp": 0.6,
                "general.sampling.top_p": 0.95,
            ]
        )
        let mmprojGGUF = GGUFHeaderReader.buildMinimalGGUF(
            boolMetadata: [
                "clip.has_vision_encoder": true,
                "clip.has_audio_encoder": true,
            ]
        )
        let mainGGUFURL = ggufDir.appendingPathComponent("Qwen3.5-9B-Q8_0.gguf")
        try mainGGUF.write(to: mainGGUFURL)
        try mmprojGGUF.write(to: ggufDir.appendingPathComponent("mmproj-Qwen3.5-9B-f16.gguf"))

        let mainHeader = try #require(GGUFHeaderReader.readFileHeader(at: mainGGUFURL))
        #expect(mainHeader.isHybridRecurrent == true)
        #expect(mainHeader.hasBuiltInMTP == true)
        #expect(abs((mainHeader.vendorSampling?.temperature ?? 0) - 0.6) < 1e-4)
        #expect(mainHeader.vendorSampling?.topK == 20)

        let discoveredGGUF = try #require(HeylookConfig.discoverModels(in: [ggufDir]).first)
        #expect(discoveredGGUF.supportsVision.value == true)
        #expect(discoveredGGUF.supportsAudio.value == true)
        #expect(discoveredGGUF.supportsThinking.value == true)
        #expect(discoveredGGUF.tags.contains("mtp") == true)
        #expect(discoveredGGUF.tags.contains("audio") == true)

        // Verify GGUFSpawnConfig.autoSized enables `--cache-ram 8192` and `--spec-type draft-mtp` even with 0 GB extra headroom
        let spawn = GGUFSpawnConfig.autoSized(for: discoveredGGUF, headroomGB: 2.0)
        #expect(spawn.cacheRAMMiB == 8192)
        #expect(spawn.specType == "draft-mtp")
        GGUFPortPool.shared.releasePort(for: discoveredGGUF.modelID)

        // 8. MessagesRequest decodes both `thinking_depth` and `reasoning_effort` into `samplerOverlay.thinkingDepth`
        let effortJSON = Data("""
        {
          "model": "gpt-oss-20b",
          "messages": [{"role": "user", "content": "Solve this"}],
          "reasoning_effort": "high"
        }
        """.utf8)
        let decodedReq = try JSONDecoder().decode(MessagesRequest.self, from: effortJSON)
        #expect(decodedReq.thinkingDepth == "high")
        #expect(decodedReq.samplerOverlay.thinkingDepth == "high")

        // 9. CoreAIProvider reports `dflash` speculative mode from Schema 0.2 manifest
        let coreaiProvider = CoreAIProvider(
            description: EngineDescription(
                modelID: "qwen3.5-coreai",
                provider: .coreai,
                modelPath: coreaiBundle.path
            ),
            allowSyntheticFallback: true
        )
        var coreaiFinal = GenerationChunk()
        for try await chunk in await coreaiProvider.generate(
            requestID: "req_coreai_02",
            request: decodedReq,
            effectiveSampling: decodedReq.samplerOverlay,
            registry: RequestRegistry()
        ) {
            coreaiFinal = chunk
        }
        #expect(coreaiFinal.speculativeReport?.mode == "dflash")

        // 10. File-derived vision detection (`ModelTemplateInfo.hasVisionConfiguration`) and 4-protocol `renderPromptPreview`
        #expect(ModelTemplateInfo.hasVisionConfiguration(in: coreaiBundle) == true)

        let harmonyDir = tempRoot.appendingPathComponent("gpt-oss-harmony", isDirectory: true)
        try FileManager.default.createDirectory(at: harmonyDir, withIntermediateDirectories: true)
        #expect(ModelTemplateInfo.hasVisionConfiguration(in: harmonyDir) == false)
        let harmonyTemplate = """
        {%- for message in messages %}
        {%- if message['role'] == 'system' %}
        {{- '<|start|>system<|message|>' + message['content'] + '\\n# reasoning_effort = ' + (reasoning_effort | default('medium')) + '<|end|>' }}
        {%- else %}
        {{- '<|start|>' + message['role'] + '<|message|>' + message['content'] + '<|end|>' }}
        {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
        {%- if reasoning_effort in ['low', 'medium', 'high'] %}
        {{- '<|start|>assistant<|channel|>analysis<|message|>' }}
        {%- else %}
        {{- '<|start|>assistant<|channel|>final<|message|>' }}
        {%- endif %}
        {%- endif %}
        """
        try harmonyTemplate.write(
            to: harmonyDir.appendingPathComponent("chat_template.jinja"),
            atomically: true,
            encoding: .utf8
        )
        let harmonyPreview = HeylookConfig.renderPromptPreview(
            in: harmonyDir,
            system: "Be concise.",
            messages: [(role: "user", content: "Test Harmony", thinking: nil)],
            thinkingEnabled: true,
            thinkingDepth: "high"
        )
        #expect(harmonyPreview.source == "chat_template.jinja")
        #expect(harmonyPreview.prompt.contains("# reasoning_effort = high"))
        #expect(harmonyPreview.prompt.contains("<|start|>user<|message|>Test Harmony<|end|>"))
        #expect(harmonyPreview.prompt.contains("<|channel|>analysis<|message|>"))

        // 11. GGUFSubprocessProvider.applyObservedProps upgrades `reasoningFormat` and `thinkingControls` to `.observed`
        let ggufProv = GGUFSubprocessProvider(description: discoveredGGUF)
        let harmonyProps = GGUFSubprocessProvider.GGUFServerProps(
            contextSize: 131072,
            chatTemplate: "{% if reasoning_effort in ['low', 'medium', 'high'] %}<|channel|>analysis<|message|>...<|end|>{% endif %}",
            totalSlots: 1
        )
        await ggufProv.applyObservedProps(harmonyProps)
        let observedGGUFDesc = await ggufProv.describe()
        #expect(observedGGUFDesc.reasoningFormat.value == "harmony")
        #expect(observedGGUFDesc.reasoningFormat.provenance == .observed)
        #expect(observedGGUFDesc.thinkingControls?.depthVariable == "reasoning_effort")
        await ggufProv.unload()

        // 12. Pure-Swift MLXModelConverter preflight validation & conversion planning (`affine`, `mxfp8`, `q4Zero`)
        let unquantDir = tempRoot.appendingPathComponent("qwen3_8_bf16", isDirectory: true)
        try FileManager.default.createDirectory(at: unquantDir, withIntermediateDirectories: true)
        try """
        {
          "model_type": "qwen3_8",
          "vision_config": { "hidden_size": 1152 },
          "text_config": { "max_position_embeddings": 131072 }
        }
        """.write(to: unquantDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{% if enable_thinking %}<think>{% endif %}".write(
            to: unquantDir.appendingPathComponent("chat_template.jinja"),
            atomically: true,
            encoding: .utf8
        )
        try Data([0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]).write(
            to: unquantDir.appendingPathComponent("model.safetensors")
        )

        let out4BitDir = tempRoot.appendingPathComponent("qwen3_8_4bit", isDirectory: true)
        let plan4Bit = try MLXModelConverter.planConversion(
            from: unquantDir,
            to: out4BitDir,
            spec: MLXConversionSpec(bits: nil, groupSize: nil, mode: .affine, calibration: .q4Zero)
        )
        #expect(plan4Bit.modelType == "qwen3_8")
        #expect(plan4Bit.isVisionModel == true)
        #expect(plan4Bit.resolvedBits == 4)
        #expect(plan4Bit.resolvedGroupSize == 32)
        #expect(plan4Bit.calibration == .q4Zero)
        #expect(plan4Bit.safetensorsFiles == ["model.safetensors"])
        #expect(plan4Bit.sidecarFiles.contains("config.json"))
        #expect(plan4Bit.sidecarFiles.contains("chat_template.jinja"))

        let planMXFP8 = try MLXModelConverter.planConversion(
            from: unquantDir,
            to: out4BitDir,
            spec: MLXConversionSpec(mode: .mxfp8)
        )
        #expect(planMXFP8.resolvedBits == 8)
        #expect(planMXFP8.resolvedGroupSize == 32)

        // Reject incompatible q4Zero with bits=8
        #expect(throws: MLXModelConversionError.self) {
            _ = try MLXModelConverter.planConversion(
                from: unquantDir,
                to: out4BitDir,
                spec: MLXConversionSpec(bits: 8, groupSize: 64, mode: .affine, calibration: .q4Zero)
            )
        }
        // Reject overlapping source/output directories
        #expect(throws: MLXModelConversionError.self) {
            _ = try MLXModelConverter.planConversion(
                from: unquantDir,
                to: unquantDir.appendingPathComponent("sub"),
                spec: MLXConversionSpec()
            )
        }

        // 13. POST /v1/conversations/{id}/prompt forwards effective `thinking_enabled` & `thinking_depth`
        let promptDbURL = tempRoot.appendingPathComponent("prompt_preview.sqlite")
        let promptStore = try SQLiteStore(databaseURL: promptDbURL)
        let promptScheduler = ComputeScheduler()
        let harmonyDesc = EngineDescription(
            modelID: "gpt-oss-harmony",
            provider: .mlx,
            modelPath: harmonyDir.path,
            reasoningFormat: ProvenanceField(value: "harmony", provenance: .derived),
            thinkingControls: ThinkingControls(
                supportsThinking: true,
                canDisableThinking: true,
                defaultEnabled: true,
                depthVariable: "reasoning_effort",
                depthValues: ["low", "medium", "high"]
            )
        )
        await promptScheduler.register(provider: MLXSwiftProvider(description: harmonyDesc), description: harmonyDesc)
        let promptRouter = HeylookRouter(store: promptStore, scheduler: promptScheduler)
        let harmonyConv = ConversationRecord(
            id: "conv_harmony_prompt",
            title: "Harmony Prompt Inspector",
            systemPrompt: "Be precise.",
            defaultModelID: "gpt-oss-harmony"
        )
        try promptStore.upsertConversation(harmonyConv)
        _ = try promptStore.appendMessage(
            MessageRecord(conversationID: "conv_harmony_prompt", position: 0, role: "user", content: "Inspect reasoning effort")
        )

        let highEffortPromptResp = await promptRouter.handle(
            method: "POST",
            path: "/v1/conversations/conv_harmony_prompt/prompt",
            headers: ["Host": "127.0.0.1:8000", "Content-Type": "application/json"],
            body: try JSONSerialization.data(withJSONObject: ["reasoning_effort": "high"])
        )
        #expect(highEffortPromptResp.status == 200)
        let highEffortJSON = try JSONSerialization.jsonObject(with: highEffortPromptResp.body) as? [String: Any]
        #expect((highEffortJSON?["thinking_enabled"] as? Bool) == true)
        #expect((highEffortJSON?["thinking_depth"] as? String) == "high")
        let renderedHighEffortPrompt = (highEffortJSON?["prompt"] as? String) ?? ""
        #expect(renderedHighEffortPrompt.contains("# reasoning_effort = high"))
    }

    // MARK: - 27. Structured /v1/embeddings Wire, EmbeddingRequestLimits & MLXModelConverter Preflight

    @Test("Structured /v1/embeddings wire with video windows, EmbeddingRequestLimits 422 validation, and MLXModelConverter preflight")
    func testStructuredEmbeddingsWireAndMLXModelConverterPreflight() async throws {
        let dbURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_emb_wire_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: dbURL) }
        let store = try SQLiteStore(databaseURL: dbURL)
        let scheduler = ComputeScheduler()
        let router = HeylookRouter(
            store: store,
            scheduler: scheduler,
            embeddingEngine: DeterministicHashEmbeddingEngine()
        )

        // 1. Structured HeylookMLXEngine wire payload with prompt_name and windowed video block
        let fakeVideoB64 = Data("fake_mp4_payload_bytes".utf8).base64EncodedString()
        let structuredBody = try JSONSerialization.data(withJSONObject: [
            "model": "embeddinggemma-2-MLX-BF16",
            "dimensions": 512,
            "input": [
                [
                    "content": [
                        ["type": "text", "text": "title: architecture.md | text: Dual-lane GPU and ANE scheduler"]
                    ]
                ],
                [
                    "content": [
                        [
                            "type": "video",
                            "source": [
                                "type": "base64",
                                "media_type": "video/mp4",
                                "data": fakeVideoB64,
                            ],
                            "windows": ["sampling": "fps"],
                        ]
                    ],
                    "prompt_name": "query",
                ],
            ],
        ])
        let embResp = await router.handle(
            method: "POST",
            path: "/v1/embeddings",
            headers: ["Host": "127.0.0.1:8000", "Content-Type": "application/json"],
            body: structuredBody
        )
        #expect(embResp.status == 200)
        let embJSON = try JSONSerialization.jsonObject(with: embResp.body) as? [String: Any]
        #expect((embJSON?["dimensions"] as? Int) == 512)
        let rows = embJSON?["data"] as? [[String: Any]]
        #expect(rows?.count == 2)
        let videoRow = rows?[1]
        #expect((videoRow?["prompt_name"] as? String) == "query")
        let windows = videoRow?["windows"] as? [[String: Any]]
        #expect(windows?.count == 1)
        let firstWindow = windows?.first
        #expect((firstWindow?["start"] as? Double) == 0.0)
        #expect((firstWindow?["end"] as? Double) == 16.0)
        let winEmbedding = firstWindow?["embedding"] as? [Double]
        #expect(winEmbedding?.count == 512)

        // 2. EmbeddingRequestLimits HTTP 422 validation (windowed video cannot mix with extra blocks)
        let invalidWindowBody = try JSONSerialization.data(withJSONObject: [
            "model": "embeddinggemma-2-MLX-BF16",
            "input": [
                [
                    "content": [
                        ["type": "text", "text": "extra block"],
                        [
                            "type": "video",
                            "source": ["type": "base64", "media_type": "video/mp4", "data": fakeVideoB64],
                            "windows": ["sampling": "fps"],
                        ],
                    ]
                ]
            ],
        ])
        let invalidWindowResp = await router.handle(
            method: "POST",
            path: "/v1/embeddings",
            headers: ["Host": "127.0.0.1:8000", "Content-Type": "application/json"],
            body: invalidWindowBody
        )
        #expect(invalidWindowResp.status == 422)

        // 3. MLXModelConverter.planConversion preflight validation
        let tmpRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_mlx_convert_\(UUID().uuidString)")
        let srcDir = tmpRoot.appendingPathComponent("Qwen3.5-BF16")
        let dstDir = tmpRoot.appendingPathComponent("Qwen3.5-MLX-4bit")
        defer { try? FileManager.default.removeItem(at: tmpRoot) }
        try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)

        try """
        {
          "model_type": "qwen3_5",
          "vision_config": { "hidden_size": 1152 }
        }
        """.write(to: srcDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data("safetensors_header".utf8).write(to: srcDir.appendingPathComponent("model-00001-of-00001.safetensors"))
        try """
        {% if enable_thinking %}<think>{% endif %}
        """.write(to: srcDir.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)

        let plan = try MLXModelConverter.planConversion(
            from: srcDir,
            to: dstDir,
            spec: MLXConversionSpec(bits: 4, groupSize: 64, mode: .affine, calibration: .standard)
        )
        #expect(plan.modelType == "qwen3_5")
        #expect(plan.isVisionModel == true)
        #expect(plan.resolvedBits == 4)
        #expect(plan.resolvedGroupSize == 64)
        #expect(plan.safetensorsFiles == ["model-00001-of-00001.safetensors"])
        #expect(plan.sidecarFiles.contains("config.json"))
        #expect(plan.sidecarFiles.contains("chat_template.jinja"))

        // Verify q4_zero calibration rejects groupSize=64
        #expect(throws: MLXModelConversionError.self) {
            _ = try MLXModelConverter.planConversion(
                from: srcDir,
                to: dstDir,
                spec: MLXConversionSpec(bits: 4, groupSize: 64, mode: .affine, calibration: .q4Zero)
            )
        }
    }

    // MARK: - 28. Truthful Runtime Defaults, Sidecar Preservation, Non-Causal Vision Clamping & Thinking Depth Validation

    @Test("Truthful runtime defaults, sidecar update preservation, GGUF non-causal vision ubatch clamping, SSE error frame propagation, and HTTP 400 thinking_depth validation")
    func testTruthfulRuntimeDefaultsSidecarPreservationNonCausalClampingAndThinkingDepthValidation() async throws {
        // 1. DeterministicHashEmbeddingEngine and HybridLocalEmbeddingEngine without .mlmodelc bundles report honest modelID
        let hashEngine = DeterministicHashEmbeddingEngine()
        #expect(hashEngine.modelID == "deterministic-hash-768d")

        let missingEmbDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("missing_emb_\(UUID().uuidString)", isDirectory: true)
        let hybridFallback = HybridLocalEmbeddingEngine(checkpointURL: missingEmbDir, computeOptions: .hybridANETowers)
        #expect(hybridFallback.hasCompiledANETowers == false)
        #expect(hybridFallback.modelID == missingEmbDir.lastPathComponent)
        #expect(hybridFallback.modelID != "deterministic-hash-768d")
        await #expect(throws: EmbeddingEngineError.self) {
            _ = try await hybridFallback.embedBatch([.text("truthful check", task: .query)], dimensions: 768)
        }

        // 2. ProviderFactory routes .remote to RemoteHeylookProvider and defaults allowSyntheticFallback to false
        let remoteDesc = EngineDescription(
            modelID: "remote-studio-model",
            provider: .remote,
            modelPath: "http://127.0.0.1:8000"
        )
        let remoteProvider = ProviderFactory.makeProvider(for: remoteDesc)
        #expect(remoteProvider is RemoteHeylookProvider)

        // CoreAIProvider.transcribeAudio without TEXT: marker or .aimodel bundle fails loudly by default
        let strictSpeechProvider = CoreAIProvider(
            description: EngineDescription(
                modelID: "ane-speech-strict",
                provider: .coreai,
                modelPath: missingEmbDir.path
            )
        )
        await #expect(throws: ProviderExecutionError.self) {
            _ = try await strictSpeechProvider.transcribeAudio(wavData: Data(repeating: 0, count: 128))
        }

        // 3. ComputeScheduler.applySidecarUpdate preserves reasoningFormat and thinkingControls
        let scheduler = ComputeScheduler()
        let initialHarmonyDesc = EngineDescription(
            modelID: "gpt-oss-20b-mlx",
            provider: .mlx,
            modelPath: missingEmbDir.path,
            supportsThinking: ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
            reasoningFormat: ProvenanceField(value: "harmony", provenance: .derived, effect: .perRequest),
            thinkingControls: ThinkingControls(
                supportsThinking: true,
                canDisableThinking: true,
                defaultEnabled: true,
                depthVariable: "reasoning_effort",
                depthValues: ["low", "medium", "high"]
            ),
            templateSource: ProvenanceField(value: "chat_template.jinja", provenance: .derived, effect: .reload)
        )
        await scheduler.register(
            provider: MLXSwiftProvider(description: initialHarmonyDesc, allowSyntheticFallback: true),
            description: initialHarmonyDesc
        )
        let sidecarPatch = ModelSidecarConfig(
            id: "gpt-oss-20b-mlx",
            provider: .mlx,
            pinned: true,
            ctxSize: 65536,
            supportsThinking: true
        )
        let updatedHarmony = try await scheduler.applySidecarUpdate(sidecarPatch, for: "gpt-oss-20b-mlx")
        #expect(updatedHarmony.reasoningFormat.value == "harmony")
        #expect(updatedHarmony.thinkingControls?.depthVariable == "reasoning_effort")
        #expect(updatedHarmony.thinkingControls?.depthValues == ["low", "medium", "high"])
        #expect(updatedHarmony.contextSize.value == 65536)

        // 4. GGUFHeaderReader detects non-causal clip.projector_type ("gemma4v") and GGUFSpawnConfig clamps imageMaxTokens <= ubatchSize
        let tempGGUFDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_gemma4v_gguf_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempGGUFDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempGGUFDir) }

        let mainGGUF = GGUFHeaderReader.buildMinimalGGUF(
            stringMetadata: ["general.architecture": "gemma4"],
            uint32Metadata: ["gemma4.context_length": 32768]
        )
        let mmprojGGUF = GGUFHeaderReader.buildMinimalGGUF(
            stringMetadata: ["clip.projector_type": "gemma4v"],
            boolMetadata: ["clip.has_vision_encoder": true]
        )
        let mainURL = tempGGUFDir.appendingPathComponent("gemma-4-26b-Q4_K_M.gguf")
        let mmprojURL = tempGGUFDir.appendingPathComponent("mmproj-gemma-4-f16.gguf")
        try mainGGUF.write(to: mainURL)
        try mmprojGGUF.write(to: mmprojURL)

        let mmprojHeader = try #require(GGUFHeaderReader.readFileHeader(at: mmprojURL))
        #expect(mmprojHeader.clipProjectorType == "gemma4v")
        #expect(mmprojHeader.hasNonCausalVisionProjector == true)

        var clampSidecar = ModelSidecarConfig()
        clampSidecar.imageMaxTokens = 2048
        try clampSidecar.toTOML().write(
            to: tempGGUFDir.appendingPathComponent("model.heylook.toml"),
            atomically: true,
            encoding: .utf8
        )
        let gemmaGGUFDesc = EngineDescription(
            modelID: "gemma4v-gguf",
            provider: .gguf,
            modelPath: tempGGUFDir.path,
            supportsVision: ProvenanceField(value: true, provenance: .derived)
        )
        // With low headroom (2 GB), ubatchSize is 512, so imageMaxTokens (2048) must be clamped to 512 for non-causal gemma4v
        let clampedSpawn = GGUFSpawnConfig.autoSized(for: gemmaGGUFDesc, headroomGB: 2.0)
        #expect(clampedSpawn.ubatchSize == 512)
        #expect(clampedSpawn.imageMaxTokens == 512)
        GGUFPortPool.shared.releasePort(for: gemmaGGUFDesc.modelID)

        // 5. GGUFSubprocessProvider.parseOpenAISSEDataLine surfaces mid-stream error frames
        let errFrame = #"data: {"error":{"message":"KV cache slot exhausted","type":"server_error"}}"#
        let parsedErrDelta = GGUFSubprocessProvider.parseOpenAISSEDataLine(errFrame)
        #expect(parsedErrDelta?.errorMessage == "KV cache slot exhausted")

        // 6. HeylookRouter validates thinking_depth / reasoning_effort and returns HTTP 400 on invalid level
        let dbURL = tempGGUFDir.appendingPathComponent("depth_validation.sqlite")
        let store = try SQLiteStore(databaseURL: dbURL)
        let router = HeylookRouter(store: store, scheduler: scheduler)

        let badDepthReq = MessagesRequest(
            model: "gpt-oss-20b-mlx",
            messages: [.init(role: "user", content: "Test invalid depth")],
            thinkingDepth: "ultra_extreme"
        )
        let badDepthResp = await router.handle(
            method: "POST",
            path: "/v1/messages",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(badDepthReq)
        )
        #expect(badDepthResp.status == 400)
        let badDepthBody = String(decoding: badDepthResp.body, as: UTF8.self)
        #expect(badDepthBody.contains("thinking_depth"))

        let okDepthReq = MessagesRequest(
            model: "gpt-oss-20b-mlx",
            messages: [.init(role: "user", content: "Test valid depth")],
            thinkingDepth: "high"
        )
        let okDepthResp = await router.handle(
            method: "POST",
            path: "/v1/messages",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(okDepthReq)
        )
        #expect(okDepthResp.status == 200)
    }

    // MARK: - 29. HeylookWorkbenchViewModel Cached @Observable Projections & DestructiveActionPrompt

    @MainActor
    @Test("HeylookWorkbenchViewModel caches activeModelDescription, filteredModels, modelCountsByProvider, sortedActiveGGUFPorts, and DestructiveActionPrompt")
    func testWorkbenchViewModelCachedObservableProjectionsAndDestructivePrompt() throws {
        let vm = HeylookWorkbenchViewModel(autoBootstrap: false)
        #expect(vm.activeModelDescription == nil)
        #expect(vm.filteredModels.isEmpty)
        #expect(vm.sortedActiveGGUFPorts.isEmpty)
        #expect(vm.pendingConfirmation == nil)

        let mlxDesc = EngineDescription(
            modelID: "qwen3.5-0.8b-mlx",
            provider: .mlx,
            modelPath: "/tmp/qwen3.5-0.8b",
            supportsVision: ProvenanceField(value: false, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let ggufDesc = EngineDescription(
            modelID: "gemma4-26b-gguf",
            provider: .gguf,
            modelPath: "/tmp/gemma4-26b.gguf",
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )

        vm.registeredModels = [mlxDesc, ggufDesc]
        #expect(vm.modelCountsByProvider[.mlx] == 1)
        #expect(vm.modelCountsByProvider[.gguf] == 1)
        #expect(vm.filteredModels.map(\.modelID) == ["qwen3.5-0.8b-mlx", "gemma4-26b-gguf"])

        vm.selectedModelID = "gemma4-26b-gguf"
        #expect(vm.activeModelDescription?.modelID == "gemma4-26b-gguf")
        #expect(vm.activeModelDescription?.supportsVision.value == true)

        vm.selectedProviderFilter = .mlx
        #expect(vm.filteredModels.map(\.modelID) == ["qwen3.5-0.8b-mlx"])

        vm.selectedProviderFilter = nil
        #expect(vm.filteredModels.count == 2)

        vm.activeGGUFPorts = ["z-model": 18082, "a-model": 18080, "m-model": 18081]
        #expect(vm.sortedActiveGGUFPorts == ["a-model::18080", "m-model::18081", "z-model::18082"])

        vm.pendingConfirmation = .clearEmbeddingIndex
        #expect(vm.pendingConfirmation?.id == "clearEmbeddingIndex")
        #expect(vm.pendingConfirmation?.title == "Clear Embedding Index?")
        #expect(vm.pendingConfirmation?.confirmLabel == "Clear Index")

        vm.pendingConfirmation = .unloadAllModels
        #expect(vm.pendingConfirmation?.id == "unloadAllModels")
        #expect(vm.pendingConfirmation?.title == "Unload All Resident Models?")
        #expect(vm.pendingConfirmation?.confirmLabel == "Unload All Models")

        // Verify resolveModelDirectoryURL handles both directories and .gguf file paths
        let resolvedGGUFDir = HeylookConfig.resolveModelDirectoryURL(from: "/tmp/models/gguf/Qwen3.5-8B.gguf")
        #expect(resolvedGGUFDir.path == "/tmp/models/gguf")
        let resolvedMLXDir = HeylookConfig.resolveModelDirectoryURL(from: "/tmp/models/mlx/Qwen3.5-8B-MLX")
        #expect(resolvedMLXDir.path == "/tmp/models/mlx/Qwen3.5-8B-MLX")
    }

    // MARK: - 30. Real Jinja2 AST Evaluation, Syntax Validation, Prefix-Stability Lint & HF Download ETag Check

    @Test("HeylookJinjaRenderer evaluates loops, namespaces, macros, slices, syntax errors, prefix stability, and HF modifiedSinceDownload")
    func testHeylookJinjaRendererASTLoopsNamespacesMacrosSyntaxAndPrefixStability() async throws {
        // 1. Loops (`loop.index0`, `loop.last`), `namespace` cross-scope mutation, `macro`, `tojson`, and Python `[::-1]` slice
        let complexTemplate = """
        {%- macro render_tool(tool) -%}{{ tool.name }}:{{ tool.schema | tojson }}{%- endmacro -%}
        {%- set ns = namespace(last_user_idx=-1, count=0) -%}
        {%- for m in messages -%}
        {%- set ns.count = ns.count + 1 -%}
        {%- if m.role == 'user' -%}{%- set ns.last_user_idx = loop.index0 -%}{%- endif -%}
        {%- endfor -%}
        count={{ ns.count }},last_user={{ ns.last_user_idx }},rev={{ 'abc'[-1::-1] }}
        {%- if tools %}
        tools=[{%- for t in tools -%}{{ render_tool(t) }}{%- if not loop.last -%}|{%- endif -%}{%- endfor -%}]
        {%- endif %}
        {%- for m in messages %}
        <{{ m.role }}>{{ m.content | trim }}</{{ m.role }}>
        {%- endfor %}
        """
        let rendered = try HeylookJinjaRenderer.render(
            template: complexTemplate,
            context: [
                "messages": .array([
                    .object(["role": .string("system"), "content": .string(" Sys ")]),
                    .object(["role": .string("user"), "content": .string(" First ")]),
                    .object(["role": .string("assistant"), "content": .string(" Reply ")]),
                    .object(["role": .string("user"), "content": .string(" Second ")]),
                ]),
                "tools": .array([
                    .object(["name": .string("search"), "schema": .object(["q": .string("str")])]),
                ]),
            ]
        )
        #expect(rendered.contains("count=4,last_user=3,rev=cba"))
        #expect(rendered.contains(#"tools=[search:{"q": "str"}]"#))
        #expect(rendered.contains("<user>Second</user>"))

        // 2. raise_exception surfaces JinjaTemplateError.templateRaisedException
        #expect(throws: JinjaTemplateError.self) {
            _ = try HeylookJinjaRenderer.render(
                template: "{{ raise_exception('System role must come first') }}",
                context: [:]
            )
        }

        // 3. Syntax validation rejects empty, unclosed `{{`, and mismatched `{% if %}` blocks
        #expect(throws: JinjaTemplateError.self) {
            try HeylookJinjaRenderer.validateSyntax("   ")
        }
        #expect(throws: JinjaTemplateError.self) {
            try HeylookJinjaRenderer.validateSyntax("{% if enable_thinking %}<think>")
        }
        #expect(throws: JinjaTemplateError.self) {
            try HeylookJinjaRenderer.validateSyntax("{{ unclosed_expr ")
        }

        // 4. Prefix-stability lint flags `strftime_now` and Qwen3-style prior-reasoning-stripping templates
        let unstableDateTpl = "Today: {{ strftime_now('%Y-%m-%d %H:%M:%S') }}\n{% for m in messages %}{{ m.content }}{% endfor %}"
        let dateCheck = HeylookJinjaRenderer.checkPrefixStability(template: unstableDateTpl)
        #expect(dateCheck.stable == false)
        #expect(dateCheck.note != nil)

        let qwenStripPriorThinkingTpl = """
        {%- set ns = namespace(last_query_index = messages|length - 1) -%}
        {%- for m in messages -%}
        {%- if m.role == 'user' -%}{%- set ns.last_query_index = loop.index0 -%}{%- endif -%}
        {%- endfor -%}
        {%- for m in messages -%}
        {%- if m.role == 'assistant' and loop.index0 > ns.last_query_index and m.reasoning_content -%}
        <think>{{ m.reasoning_content }}</think>{{ m.content }}
        {%- elif m.role == 'assistant' and loop.index0 == messages|length - 1 and m.reasoning_content -%}
        <think>{{ m.reasoning_content }}</think>{{ m.content }}
        {%- else -%}
        <{{ m.role }}>{{ m.content }}</{{ m.role }}>
        {%- endif -%}
        {%- endfor -%}
        """
        let qwenStability = HeylookJinjaRenderer.checkPrefixStability(template: qwenStripPriorThinkingTpl)
        #expect(qwenStability.stable == false)

        let stableChatML = HeylookJinjaRenderer.defaultChatMLTemplate
        let stableCheck = HeylookJinjaRenderer.checkPrefixStability(template: stableChatML)
        #expect(stableCheck.stable == true)

        // 5. HF `.cache/huggingface/download/<file>.metadata` git-blob SHA-1 modifiedSinceDownload + HTTP 422/404 template routes
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_jinja_hf_\(UUID().uuidString)", isDirectory: true)
        let metaDir = tmpDir
            .appendingPathComponent(".cache", isDirectory: true)
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("download", isDirectory: true)
        try FileManager.default.createDirectory(at: metaDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let originalTpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        let tplURL = tmpDir.appendingPathComponent("chat_template.jinja")
        let tplBytes = Data(originalTpl.utf8)
        try tplBytes.write(to: tplURL)

        var gitBlob = Data("blob \(tplBytes.count)\0".utf8)
        gitBlob.append(tplBytes)
        let sha1Hex = Insecure.SHA1.hash(data: gitBlob).map { String(format: "%02x", $0) }.joined()
        try "0123456789abcdef\n\(sha1Hex)\n1700000000.0\n".write(
            to: metaDir.appendingPathComponent("chat_template.jinja.metadata"),
            atomically: true,
            encoding: .utf8
        )

        let cleanState = HeylookConfig.inspectChatTemplate(in: tmpDir, fallbackSource: "chat_template.jinja")
        #expect(cleanState.modifiedSinceDownload == false)
        #expect(cleanState.origin == "vendor_sidecar")

        try (originalTpl + "\n{# locally edited #}").write(to: tplURL, atomically: true, encoding: .utf8)
        let dirtyState = HeylookConfig.inspectChatTemplate(in: tmpDir, fallbackSource: "chat_template.jinja")
        #expect(dirtyState.modifiedSinceDownload == true)

        // Verify PUT /v1/admin/models/{id}/chat-template returns 422 on invalid Jinja2 and DELETE returns 404 when no override exists
        let store = try SQLiteStore(databaseURL: tmpDir.appendingPathComponent("tpl.sqlite"))
        let scheduler = ComputeScheduler()
        let desc = EngineDescription(modelID: "tpl-check-mlx", provider: .mlx, modelPath: tmpDir.path)
        await scheduler.register(provider: MLXSwiftProvider(description: desc, allowSyntheticFallback: true), description: desc)
        let router = HeylookRouter(store: store, scheduler: scheduler)

        let delMissing = await router.handle(
            method: "DELETE",
            path: "/v1/admin/models/tpl-check-mlx/chat-template",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(delMissing.status == 404)

        let putInvalid = await router.handle(
            method: "PUT",
            path: "/v1/admin/models/tpl-check-mlx/chat-template",
            headers: ["Host": "127.0.0.1:8000", "Content-Type": "application/json"],
            body: Data(#"{"template":"{% if broken "}"#.utf8)
        )
        #expect(putInvalid.status == 422)
    }

    // MARK: - 31. GGUF Large Vocab Header Reader, OS-Bound Port Pool Probe, RAM Fit & Fast Child Exit

    @Test("GGUFHeaderReader parses past large vocab arrays, GGUFPortPool skips OS-bound sockets, GGUFSpawnConfig assesses RAM fit & media-blind sidecar, and launchSubprocess rejects fast exit")
    func testGGUFHeaderLargeVocabArrayPortPoolOSBindProbeRAMFitAndFastExit() async throws {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_gguf_p2_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // 1. Build a GGUF v3 file with a 10,000-entry `tokenizer.ggml.tokens` array (>140 KB, exceeding legacy 64 KB & 4,096-elem caps)
        let largeVocab = (0..<10_000).map { "tok_piece_\(String(format: "%05d", $0))" }
        let ggufData = GGUFHeaderReader.buildMinimalGGUF(
            version: 3,
            tensorCount: 128,
            stringMetadata: [
                "general.architecture": "qwen3_5",
                "tokenizer.chat_template": "{% for m in messages %}{% if m.content is not string %}<__media__>{% endif %}{{ m.content }}{% endfor %}",
            ],
            stringArrayMetadata: ["a.tokenizer.ggml.tokens": largeVocab],
            uint32Metadata: [
                "qwen3_5.context_length": 131072,
                "qwen3_5.ssm.conv_kernel": 4,
            ]
        )
        #expect(ggufData.count > 120_000)
        let mainGGUF = tmpDir.appendingPathComponent("Qwen3.5-VL-Q4_K_M.gguf")
        try ggufData.write(to: mainGGUF)

        let header = try #require(GGUFHeaderReader.readFileHeader(at: mainGGUF))
        #expect(header.architecture == "qwen3_5")
        #expect(header.contextLength == 131072)
        #expect(header.isHybridRecurrent == true)
        #expect(header.embeddedChatTemplate?.contains("<__media__>") == true)

        // 2. Bind an actual OS TCP socket on loopback and verify `GGUFPortPool` detects and skips it
        let sockFD = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        #expect(sockFD >= 0)
        defer { if sockFD >= 0 { Darwin.close(sockFD) } }

        var reuse: Int32 = 1
        _ = setsockopt(sockFD, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        let testBasePort = 28190
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(UInt16(testBasePort).bigEndian)
        addr.sin_addr.s_addr = CFSwapInt32HostToBig(INADDR_LOOPBACK)
        let bindRC = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.bind(sockFD, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if bindRC == 0 {
            _ = Darwin.listen(sockFD, 4)
            #expect(GGUFPortPool.isPortAvailable(testBasePort) == false)
            let localPool = GGUFPortPool(basePort: testBasePort)
            let acquired = localPool.acquirePort(for: "probe-model")
            #expect(acquired > testBasePort)
            localPool.releaseAll()
        }

        // 3. Media-blind `chat_template.jinja` sidecar fallback to `gguf_embedded` when `--mmproj` is present + RAM-fit drafter drop
        let mmprojURL = tmpDir.appendingPathComponent("mmproj-Qwen3.5-f16.gguf")
        try Data("GGUF".utf8).write(to: mmprojURL)
        let drafterURL = tmpDir.appendingPathComponent("mtp-qwen3.5-draft.gguf")
        try Data(repeating: 0xAB, count: 1024).write(to: drafterURL)

        // Write a text-only vendor sidecar (`chat_template.jinja`) that has NO image/media handling
        try "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}".write(
            to: tmpDir.appendingPathComponent("chat_template.jinja"),
            atomically: true,
            encoding: .utf8
        )

        let vlmDesc = EngineDescription(
            modelID: "qwen3.5-vl-gguf",
            provider: .gguf,
            modelPath: tmpDir.path,
            contextSize: ProvenanceField(value: 32768, provenance: .derived),
            supportsVision: ProvenanceField(value: true, provenance: .derived)
        )
        // Tight headroom (1.0 GB): drops external drafter AND skips media-blind `chat_template.jinja` in favor of embedded GGUF template
        let tightSpawn = GGUFSpawnConfig.autoSized(for: vlmDesc, headroomGB: 1.0)
        #expect(tightSpawn.mmprojPath == mmprojURL.standardizedFileURL.path)
        #expect(tightSpawn.draftModelPath == nil)
        #expect(tightSpawn.chatTemplateFile == nil) // fell back to `gguf_embedded`!
        #expect(tightSpawn.cacheReuse == nil)
        #expect(tightSpawn.ctxCheckpoints == 8)
        #expect(tightSpawn.cacheRAMMiB == 8192)
        GGUFPortPool.shared.releasePort(for: vlmDesc.modelID)

        // Plenty of headroom (48.0 GB): keeps external drafter
        let roomySpawn = GGUFSpawnConfig.autoSized(for: vlmDesc, headroomGB: 48.0)
        #expect(roomySpawn.draftModelPath == drafterURL.standardizedFileURL.path)
        GGUFPortPool.shared.releasePort(for: vlmDesc.modelID)

        // 4. `launchSubprocess` against `/usr/bin/false` fails fast with `ProviderExecutionError.runtimeUnavailable`
        #if os(macOS)
        let failingSpawn = GGUFSpawnConfig(
            binaryPath: "/usr/bin/false",
            modelPath: mainGGUF.path,
            port: 28991
        )
        let failingProvider = GGUFSubprocessProvider(description: vlmDesc, spawnConfig: failingSpawn)
        await #expect(throws: ProviderExecutionError.self) {
            _ = try await failingProvider.launchSubprocess(customArguments: [], checkHealth: true)
        }
        #expect(await failingProvider.describe().isLoaded == false)
        await failingProvider.unload()
        #endif
    }

    // MARK: - 32. StopSequenceFilter Multi-Chunk Holdback, ThinkingBudget & Immediate Cancellation

    @Test("StopSequenceFilter holds back split stop sequences, StreamingPipeline enforces thinkingBudget, and RequestRegistry cancels immediately")
    func testStopSequenceFilterMultiChunkHoldbackImmediateAbortAndThinkingBudget() async throws {
        // 1. Multi-chunk stop sequence holdback: never leaks stop string or trailing text
        var filter = StopSequenceFilter(stopSequences: ["<|end|>", "STOP_NOW"])
        var collected = ""
        let r1 = filter.feed("Hello ")
        collected += r1.emitted
        #expect(r1.matchedStop == nil)

        let r2 = filter.feed("world <|e")
        collected += r2.emitted
        #expect(collected == "Hello world ")
        #expect(r2.matchedStop == nil)

        let r3 = filter.feed("nd|>EXTRA_GARBAGE")
        collected += r3.emitted
        #expect(r3.matchedStop == "<|end|>")
        #expect(collected == "Hello world ")
        #expect(filter.flush().isEmpty)

        // Partial false-alarm prefix flushes cleanly at EOS
        var safeFilter = StopSequenceFilter(stopSequences: ["<|end|>"])
        var safeOut = safeFilter.feed("Engine <|en").emitted
        safeOut += safeFilter.feed("gine").emitted
        safeOut += safeFilter.flush()
        #expect(safeOut == "Engine <|engine")

        // 2. StreamingPipeline thinkingBudget forces reasoning end + suppresses late `.splitReasoning` + stops on `stopSequences`
        actor HookFlag {
            var didFire = false
            func mark() { didFire = true }
        }
        let hookFlag = HookFlag()
        let reg = RequestRegistry()
        reg.register(id: "req_budget_stop")
        let budgetStream = StreamingPipeline.transformEventStream(
            requestID: "req_budget_stop",
            cacheReport: CacheReport(promptTokens: 16, cachedTokens: 8, processedTokens: 8, outcome: .reused),
            thinkingEnabled: true,
            stopSequences: ["</answer>"],
            thinkingBudget: 2,
            computePlacement: .mlxGPU,
            registry: reg,
            onReasoningEndHook: { await hookFlag.mark() }
        ) { cont in
            cont.yield(.splitReasoning("step1 "))
            cont.yield(.splitReasoning("step2 "))
            cont.yield(.splitReasoning("step3_suppressed "))
            cont.yield(.splitContent("Final output</ans"))
            cont.yield(.splitContent("wer>trailing_ignored"))
        }

        var visible = ""
        var thinking = ""
        var finalChunk = GenerationChunk()
        for try await c in budgetStream {
            visible += c.text
            thinking += c.thinking
            finalChunk = c
        }
        reg.unregister(id: "req_budget_stop")

        #expect(await hookFlag.didFire == true)
        #expect(thinking == "step1 step2 ")
        #expect(!thinking.contains("step3_suppressed"))
        #expect(visible == "Final output")
        #expect(finalChunk.stopReason == .stopSequence)
        #expect(finalChunk.thinkingTokens == 2)

        // 3. Immediate cancellation via `RequestRegistry.cancel(id:)` wakes a blocked 30s prefill producer in < 500ms
        let cancelReg = RequestRegistry()
        cancelReg.register(id: "req_block_prefill")
        let startWall = ContinuousClock.now
        let slowStream = StreamingPipeline.transformEventStream(
            requestID: "req_block_prefill",
            cacheReport: CacheReport(promptTokens: 32, cachedTokens: 0, processedTokens: 32, outcome: .miss, cause: "cold"),
            computePlacement: .mlxGPU,
            registry: cancelReg
        ) { _ in
            try await Task.sleep(for: .seconds(30))
        }

        Task {
            try? await Task.sleep(for: .milliseconds(30))
            _ = cancelReg.cancel(id: "req_block_prefill")
        }

        var cancelChunk = GenerationChunk()
        for try await c in slowStream {
            cancelChunk = c
        }
        cancelReg.unregister(id: "req_block_prefill")
        let elapsed = ContinuousClock.now - startWall
        #expect(elapsed < .seconds(2))
        #expect(cancelChunk.stopReason == .cancelled)
    }

    // MARK: - 33. Real 48 kHz Stereo WAV Resampling to 16 kHz Mono & Unigram Viterbi Piece Tokenizer

    @Test("ArtifactProcessor resamples 48 kHz stereo WAV to 16 kHz mono and CheckpointPieceTokenizer segments Unigram pieces via Viterbi")
    func testRealStereo48kHzWAVResamplingTo16kHzMonoAndCheckpointPieceTokenizer() async throws {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_audio_tok_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // 1. Synthesize a 0.5-second 48,000 Hz 2-channel stereo 16-bit PCM WAV file (24,000 stereo frames = 96,000 PCM bytes)
        let srcSampleRate: UInt32 = 48_000
        let channels: UInt16 = 2
        let frameCount = 24_000 // 0.50s
        var pcmBytes = Data(capacity: frameCount * Int(channels) * 2)
        for frame in 0..<frameCount {
            let t = Double(frame) / Double(srcSampleRate)
            let left = Int16((sin(2.0 * .pi * 440.0 * t) * 14_000.0).rounded()).littleEndian
            let right = Int16((sin(2.0 * .pi * 880.0 * t) * 14_000.0).rounded()).littleEndian
            withUnsafeBytes(of: left) { pcmBytes.append(contentsOf: $0) }
            withUnsafeBytes(of: right) { pcmBytes.append(contentsOf: $0) }
        }

        var wav = Data("RIFF".utf8)
        var chunkSize = UInt32(36 + pcmBytes.count).littleEndian
        withUnsafeBytes(of: &chunkSize) { wav.append(contentsOf: $0) }
        wav.append(Data("WAVEfmt ".utf8))
        var fmtLen = UInt32(16).littleEndian
        var audioFmt = UInt16(1).littleEndian
        var chLE = channels.littleEndian
        var srLE = srcSampleRate.littleEndian
        var byteRate = (srcSampleRate * UInt32(channels) * 2).littleEndian
        var blockAlign = (channels * 2).littleEndian
        var bitsPerSample = UInt16(16).littleEndian
        withUnsafeBytes(of: &fmtLen) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &audioFmt) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &chLE) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &srLE) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &byteRate) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &blockAlign) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &bitsPerSample) { wav.append(contentsOf: $0) }
        wav.append(Data("data".utf8))
        var dataLen = UInt32(pcmBytes.count).littleEndian
        withUnsafeBytes(of: &dataLen) { wav.append(contentsOf: $0) }
        wav.append(pcmBytes)

        let stereoURL = tmpDir.appendingPathComponent("stereo_48k.wav")
        try wav.write(to: stereoURL)

        let mono16k = try await ArtifactProcessor.decodeMonoPCM(for: stereoURL, sampleRate: 16_000)
        // 0.50s @ 16,000 Hz = ~8,000 mono samples (whereas un-resampled 48k stereo would have been 48,000 samples!)
        #expect(abs(mono16k.count - 8_000) <= 64)
        let peakAbs = mono16k.map { abs($0) }.max() ?? 0
        #expect(peakAbs > 0.05)

        // 2. Real Unigram SentencePiece `tokenizer.json` Viterbi segmentation
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
            to: tmpDir.appendingPathComponent("tokenizer.json"),
            atomically: true,
            encoding: .utf8
        )
        let pieceTok = try #require(CheckpointPieceTokenizer.load(from: tmpDir))
        #expect(pieceTok.bosTokenID == 2)
        #expect(pieceTok.eosTokenID == 1)
        let encoded = pieceTok.encode(" apple neural engine")
        #expect(encoded == [4, 5, 6])
    }

    // MARK: - 34. SQLiteStore Clean Schema Recreation, Content-Addressed Media Blobs GC & Preset Seed-Once

    @Test("SQLiteStore recreates outdated tables cleanly, deduplicates & GCs SHA-256 media_blobs, and seeds presets once")
    func testSQLiteStoreCleanSchemaRecreationMediaBlobsGCAndPresetSeedOnce() async throws {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_store_p3_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let dbURL = tmpDir.appendingPathComponent("legacy_to_clean.sqlite")

        // 1. Pre-create an outdated `conversations` table missing `default_model_id` / `params_json`
        var rawDB: OpaquePointer?
        #expect(sqlite3_open(dbURL.path, &rawDB) == SQLITE_OK)
        if let rawDB {
            _ = sqlite3_exec(rawDB, "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL);", nil, nil, nil)
            sqlite3_close(rawDB)
        }

        // Opening SQLiteStore drops & recreates outdated schema cleanly (no migration sprawl, zero crashes)
        do {
            let store = try SQLiteStore(databaseURL: dbURL)
            try store.upsertConversation(
                ConversationRecord(
                    id: "conv_media",
                    title: "Media Dedup Test",
                    systemPrompt: "Sys",
                    defaultModelID: "vlm-1",
                    params: SamplerSettings(temperature: 0.3)
                )
            )
            let loadedConv = try #require(try store.getConversation(id: "conv_media"))
            #expect(loadedConv.defaultModelID == "vlm-1")
            #expect(loadedConv.params?.temperature == 0.3)

            // 2. Append two messages referencing the exact same PNG bytes -> 1 deduplicated `media_blobs` row
            let pngBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0xAA, 0xBB, 0xCC])
            let att = MessagesRequest.MediaAttachment(
                kind: "image",
                mediaType: "image/png",
                dataBase64: pngBytes.base64EncodedString(),
                filename: "test.png"
            )
            _ = try store.appendMessage(
                MessageRecord(
                    id: "m1",
                    conversationID: "conv_media",
                    position: 0,
                    role: "user",
                    content: "First image",
                    contentBlocks: [.media(att), .text("First image")]
                )
            )
            _ = try store.appendMessage(
                MessageRecord(
                    id: "m2",
                    conversationID: "conv_media",
                    position: 1,
                    role: "user",
                    content: "Same image again",
                    contentBlocks: [.media(att), .text("Same image again")]
                )
            )
            #expect(try store.mediaBlobCount() == 1)

            // Stored JSON externalized base64 (`dataBase64 == ""`, `mediaID != nil`)
            let unhydrated = try store.listMessages(conversationID: "conv_media", hydrateMedia: false)
            var extractedMediaID: String?
            if let firstBlock = unhydrated.first?.contentBlocks?.first,
               case .media(let storedAtt) = firstBlock {
                #expect(storedAtt.dataBase64.isEmpty)
                extractedMediaID = storedAtt.mediaID
            }
            let mediaID = try #require(extractedMediaID)

            // Hydrated read restores exact base64 bytes
            let hydrated = try store.listMessages(conversationID: "conv_media", hydrateMedia: true)
            if let firstBlock = hydrated.first?.contentBlocks?.first,
               case .media(let hydAtt) = firstBlock {
                #expect(Data(base64Encoded: hydAtt.dataBase64) == pngBytes)
            } else {
                Issue.record("Expected hydrated media attachment")
            }

            // Serve binary blob via GET /v1/conversations/{id}/media/{media_id}
            let router = HeylookRouter(store: store, scheduler: ComputeScheduler())
            let mediaResp = await router.handle(
                method: "GET",
                path: "/v1/conversations/conv_media/media/\(mediaID)",
                headers: ["Host": "127.0.0.1:8000"]
            )
            #expect(mediaResp.status == 200)
            #expect(mediaResp.headers["Content-Type"] == "image/png")
            #expect(mediaResp.body == pngBytes)

            // Deleting the conversation garbage-collects orphaned `media_blobs`
            try store.deleteConversation(id: "conv_media")
            #expect(try store.mediaBlobCount() == 0)

            // 3. Presets seed-once guard + HTTP 409 conflict on duplicate name
            let initialPresets = try store.listPresets()
            #expect(!initialPresets.isEmpty)
            let dupPreset = PresetRecord(name: initialPresets[0].name, settings: SamplerSettings(temperature: 0.5))
            let conflictResp = await router.handle(
                method: "POST",
                path: "/v1/presets",
                headers: ["Host": "127.0.0.1:8000", "Content-Type": "application/json"],
                body: try JSONEncoder().encode(dupPreset)
            )
            #expect(conflictResp.status == 409)

            // Delete all presets and reopen store -> stays empty (never re-seeds!)
            for p in initialPresets {
                try store.deletePreset(id: p.id)
            }
            #expect(try store.listPresets().isEmpty)
        }

        let reopenedStore = try SQLiteStore(databaseURL: dbURL)
        #expect(try reopenedStore.listPresets().isEmpty)
    }

    // MARK: - 35. ConversationGenerationCoordinator Modes, Dropped Media, 409 Lock & Performance Profile

    @Test("ConversationGenerationCoordinator supports append/regenerate/continue, dropped_media, 409 lock, Anthropic SSE blocks, and /v1/performance/profile")
    func testConversationGenerationCoordinatorModesDroppedMediaLockAndAnthropicSSE() async throws {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_saga_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let store = try SQLiteStore(databaseURL: tmpDir.appendingPathComponent("saga.sqlite"))
        let scheduler = ComputeScheduler()
        let vlmDesc = EngineDescription(
            modelID: "vlm-saga",
            provider: .mlx,
            modelPath: tmpDir.path,
            isLoaded: true,
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let textOnlyDesc = EngineDescription(
            modelID: "txt-saga",
            provider: .mlx,
            modelPath: tmpDir.path,
            isLoaded: true,
            supportsVision: ProvenanceField(value: false, provenance: .derived),
            supportsThinking: ProvenanceField(value: false, provenance: .derived)
        )
        await scheduler.register(provider: MLXSwiftProvider(description: vlmDesc, allowSyntheticFallback: true), description: vlmDesc)
        await scheduler.register(provider: MLXSwiftProvider(description: textOnlyDesc, allowSyntheticFallback: true), description: textOnlyDesc)
        let registry = RequestRegistry()
        let router = HeylookRouter(store: store, scheduler: scheduler, registry: registry)

        // 1. Append mode with image attachment + full Anthropic SSE framing (`content_block_start` / `content_block_stop` / `heylook_saved`)
        let imgPart = MessagesRequest.MessageTurn.ContentPart.media(
            .init(kind: "image", mediaType: "image/png", dataBase64: Data([0x89, 0x50, 0x4E, 0x47]).base64EncodedString())
        )
        let appendReq = ConversationGenerateRequest(
            model: "vlm-saga",
            content: "Inspect chart",
            mode: .append,
            userParts: [imgPart, .text("Inspect chart")],
            thinking: true
        )
        let appendResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/conv_saga/generate",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(appendReq)
        )
        #expect(appendResp.status == 200)
        let appendSSE = String(decoding: appendResp.body, as: UTF8.self)
        #expect(appendSSE.contains("event: message_start"))
        #expect(appendSSE.contains("event: content_block_start"))
        #expect(appendSSE.contains("event: content_block_delta"))
        #expect(appendSSE.contains("event: content_block_stop"))
        #expect(appendSSE.contains("event: message_delta"))
        #expect(appendSSE.contains("event: message_stop"))
        #expect(appendSSE.contains("event: heylook_saved"))
        #expect(appendSSE.contains("\"dropped_media\":0"))

        // 2. Regenerate mode switching from VLM to text-only model strips history image & reports `dropped_media: 1`
        let regenReq = ConversationGenerateRequest(
            model: "txt-saga",
            mode: .regenerate
        )
        let regenResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/conv_saga/generate",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(regenReq)
        )
        #expect(regenResp.status == 200)
        let regenSSE = String(decoding: regenResp.body, as: UTF8.self)
        #expect(regenSSE.contains("\"mode\":\"regenerate\""))
        #expect(regenSSE.contains("\"dropped_media\":1"))

        let msgsAfterRegen = try store.listMessages(conversationID: "conv_saga")
        #expect(msgsAfterRegen.count == 2)
        let beforeContinueLen = msgsAfterRegen[1].content.count

        // 3. Continue mode appends to existing assistant message in-place at position 1
        let contReq = ConversationGenerateRequest(
            model: "txt-saga",
            mode: .continue,
            messageID: msgsAfterRegen[1].id
        )
        let contResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/conv_saga/generate",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(contReq)
        )
        #expect(contResp.status == 200)
        let msgsAfterContinue = try store.listMessages(conversationID: "conv_saga")
        #expect(msgsAfterContinue.count == 2)
        #expect(msgsAfterContinue[1].id == msgsAfterRegen[1].id)
        #expect(msgsAfterContinue[1].content.count > beforeContinueLen)

        // 4. Per-conversation concurrent generation lock returns HTTP 409 (`generation_in_progress`)
        _ = registry.tryAcquireConversationLock(conversationID: "conv_saga", requestID: "req_active_lock", modelID: "txt-saga")
        let lockedResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/conv_saga/generate",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(regenReq)
        )
        #expect(lockedResp.status == 409)
        registry.finish(requestID: "req_active_lock")

        // 5. Append a second turn and verify GET /v1/performance/profile/24h aggregates persisted `message_stats`
        let followUpReq = ConversationGenerateRequest(
            model: "txt-saga",
            content: "Summarize findings",
            mode: .append
        )
        let followUpResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/conv_saga/generate",
            headers: ["Host": "127.0.0.1:8000"],
            body: try JSONEncoder().encode(followUpReq)
        )
        #expect(followUpResp.status == 200)

        let perfResp = await router.handle(
            method: "GET",
            path: "/v1/performance/profile/24h",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(perfResp.status == 200)
        let perfJSON = try JSONSerialization.jsonObject(with: perfResp.body) as? [String: Any]
        #expect(((perfJSON?["sample_count"] as? Int) ?? 0) == 2)
    }

    // MARK: - 36. Security Gates: DNS-Rebinding Host Header on 0.0.0.0, Admin Token, Read-Only Config & X-Request-ID

    @Test("HeylookRouter enforces DNS-rebinding Host checks on 0.0.0.0, X-Heylook-Admin-Token 401/403, read-only config 409, and X-Request-ID 422")
    func testRouterHostHeaderSecurityAdminTokenReadOnlyConfigAndRequestIDValidation() async throws {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_sec_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let store = try SQLiteStore(databaseURL: tmpDir.appendingPathComponent("sec.sqlite"))
        let scheduler = ComputeScheduler()

        // 1. `bindHost: "0.0.0.0"` MUST reject arbitrary DNS names (`Host: evil.attacker.example` -> 403) while allowing LAN IPv4 / loopback / .local
        let lanRouter = HeylookRouter(
            store: store,
            scheduler: scheduler,
            bindHost: "0.0.0.0",
            allowedHosts: ["studio.tailnet.ts.net"],
            adminToken: "topsecret_admin_token"
        )

        let evilResp = await lanRouter.handle(
            method: "GET",
            path: "/v1/models",
            headers: ["Host": "evil.attacker.example:8000"]
        )
        #expect(evilResp.status == 403)

        let lanIPResp = await lanRouter.handle(
            method: "GET",
            path: "/v1/models",
            headers: ["Host": "192.168.1.50:8000"]
        )
        #expect(lanIPResp.status == 200)

        let allowlistResp = await lanRouter.handle(
            method: "GET",
            path: "/v1/models",
            headers: ["Host": "studio.tailnet.ts.net:8000"]
        )
        #expect(allowlistResp.status == 200)

        // 2. `X-Heylook-Admin-Token` enforces 401 (missing), 403 (mismatch), and 200 (valid) on `/v1/admin/*` and `/v1/data/clear`
        let noAuthResp = await lanRouter.handle(
            method: "GET",
            path: "/v1/admin/config",
            headers: ["Host": "127.0.0.1:8000"]
        )
        #expect(noAuthResp.status == 401)

        let badAuthResp = await lanRouter.handle(
            method: "GET",
            path: "/v1/admin/config",
            headers: ["Host": "127.0.0.1:8000", "X-Heylook-Admin-Token": "wrong_token"]
        )
        #expect(badAuthResp.status == 403)

        let okAuthResp = await lanRouter.handle(
            method: "GET",
            path: "/v1/admin/config",
            headers: ["Host": "127.0.0.1:8000", "X-Heylook-Admin-Token": "topsecret_admin_token"]
        )
        #expect(okAuthResp.status == 200)

        // 3. `X-Request-ID` validation rejects illegal characters with 422 and echoes valid IDs
        let badReqIDResp = await lanRouter.handle(
            method: "GET",
            path: "/v1/models",
            headers: ["Host": "127.0.0.1:8000", "X-Request-ID": "bad id with spaces!"]
        )
        #expect(badReqIDResp.status == 422)

        let validReqIDResp = await lanRouter.handle(
            method: "GET",
            path: "/v1/models",
            headers: ["Host": "127.0.0.1:8000", "X-Request-ID": "trace_req-2026.01:ok"]
        )
        #expect(validReqIDResp.status == 200)
        #expect(validReqIDResp.headers["X-Request-ID"] == "trace_req-2026.01:ok")

        // 4. `isReadOnlyConfig: true` blocks config/sidecar writes with 409
        let roRouter = HeylookRouter(store: store, scheduler: scheduler, isReadOnlyConfig: true)
        let roPatch = await roRouter.handle(
            method: "PATCH",
            path: "/v1/admin/config",
            headers: ["Host": "127.0.0.1:8000"],
            body: Data(#"{"max_loaded_models":2}"#.utf8)
        )
        #expect(roPatch.status == 409)

        // 5. HTTP status reason phrases
        #expect(HeylookHTTPResponse.statusReasonPhrase(for: 201) == "Created")
        #expect(HeylookHTTPResponse.statusReasonPhrase(for: 409) == "Conflict")
        #expect(HeylookHTTPResponse.statusReasonPhrase(for: 422) == "Unprocessable Entity")
        #expect(HeylookHTTPResponse.statusReasonPhrase(for: 503) == "Service Unavailable")
    }

    // MARK: - 37. Full CRUD Route Parity: Clone, Message Compaction, Notebooks, Presets, Admin Fit/Validate & Telemetry

    @Test("HeylookRouter serves conversation clone, single-message delete compaction, notebooks CRUD+generate, presets CRUD, admin validate/fit, cache/data clear, and telemetry logs")
    func testConversationsNotebooksPresetsAndAdminRoutesFullCRUDParity() async throws {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_crud_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let store = try SQLiteStore(databaseURL: tmpDir.appendingPathComponent("crud.sqlite"))
        let scheduler = ComputeScheduler()
        let desc = EngineDescription(
            modelID: "crud-mlx",
            provider: .mlx,
            modelPath: tmpDir.path,
            isLoaded: true
        )
        await scheduler.register(provider: MLXSwiftProvider(description: desc, allowSyntheticFallback: true), description: desc)
        let router = HeylookRouter(store: store, scheduler: scheduler, allowSyntheticFallback: true)
        let hostHdr = ["Host": "127.0.0.1:8000", "Content-Type": "application/json"]

        // 1. Conversation messages + single-message DELETE position compaction + POST /clone
        _ = await router.handle(
            method: "POST",
            path: "/v1/conversations",
            headers: hostHdr,
            body: try JSONEncoder().encode(ConversationRecord(id: "c_orig", title: "Original"))
        )
        for idx in 0..<3 {
            _ = await router.handle(
                method: "POST",
                path: "/v1/conversations/c_orig/messages",
                headers: hostHdr,
                body: try JSONEncoder().encode(
                    MessageRecord(id: "msg_\(idx)", conversationID: "c_orig", position: idx, role: "user", content: "Turn \(idx)")
                )
            )
        }
        // Delete middle message `msg_1` -> remaining positions compact to `[0, 1]`
        let delMiddle = await router.handle(
            method: "DELETE",
            path: "/v1/conversations/c_orig/messages/msg_1",
            headers: hostHdr
        )
        #expect(delMiddle.status == 200)
        let compacted = try store.listMessages(conversationID: "c_orig")
        #expect(compacted.map(\.id) == ["msg_0", "msg_2"])
        #expect(compacted.map(\.position) == [0, 1])

        // Clone conversation
        let cloneResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/c_orig/clone",
            headers: hostHdr,
            body: Data(#"{"title":"Branch Clone"}"#.utf8)
        )
        #expect(cloneResp.status == 201)
        let clonedConv = try JSONDecoder().decode(ConversationRecord.self, from: cloneResp.body)
        #expect(clonedConv.title == "Branch Clone")
        #expect(try store.listMessages(conversationID: clonedConv.id).count == 2)

        // 2. Notebooks CRUD + `/generate`
        let nbCreate = await router.handle(
            method: "POST",
            path: "/v1/notebooks",
            headers: hostHdr,
            body: try JSONEncoder().encode(NotebookRecord(id: "nb_1", title: "Draft", body: "Outline kernel architecture", modelID: "crud-mlx"))
        )
        #expect(nbCreate.status == 201)

        let nbPatch = await router.handle(
            method: "PATCH",
            path: "/v1/notebooks/nb_1",
            headers: hostHdr,
            body: Data(#"{"title":"Updated Draft"}"#.utf8)
        )
        #expect(nbPatch.status == 200)
        let updatedNB = try JSONDecoder().decode(NotebookRecord.self, from: nbPatch.body)
        #expect(updatedNB.title == "Updated Draft")

        let nbGen = await router.handle(
            method: "POST",
            path: "/v1/notebooks/nb_1/generate",
            headers: hostHdr,
            body: Data("{}".utf8)
        )
        #expect(nbGen.status == 200)

        let nbDel = await router.handle(method: "DELETE", path: "/v1/notebooks/nb_1", headers: hostHdr)
        #expect(nbDel.status == 200)
        let nbGet404 = await router.handle(method: "GET", path: "/v1/notebooks/nb_1", headers: hostHdr)
        #expect(nbGet404.status == 404)

        // 3. Presets individual GET / PATCH / DELETE
        let presetCreate = await router.handle(
            method: "POST",
            path: "/v1/presets",
            headers: hostHdr,
            body: try JSONEncoder().encode(PresetRecord(id: "p_custom", name: "Precise Deterministic", settings: SamplerSettings(temperature: 0.0)))
        )
        #expect(presetCreate.status == 201)
        let presetPatch = await router.handle(
            method: "PATCH",
            path: "/v1/presets/p_custom",
            headers: hostHdr,
            body: Data(#"{"description":"Zero-temp deterministic"}"#.utf8)
        )
        #expect(presetPatch.status == 200)
        #expect(try store.getPreset(id: "p_custom")?.description == "Zero-temp deterministic")
        let presetDel = await router.handle(method: "DELETE", path: "/v1/presets/p_custom", headers: hostHdr)
        #expect(presetDel.status == 200)

        // 4. Admin validate, fit, model-options, cache/clear, telemetry/events, admin/config/logs, and data/clear
        let fitResp = await router.handle(
            method: "POST",
            path: "/v1/admin/models/fit",
            headers: hostHdr,
            body: Data(#"{"model_id":"crud-mlx","ctx_size":16384}"#.utf8)
        )
        #expect(fitResp.status == 200)

        let optsResp = await router.handle(method: "GET", path: "/v1/admin/models/model-options", headers: hostHdr)
        #expect(optsResp.status == 200)

        let cacheClearResp = await router.handle(method: "POST", path: "/v1/cache/clear", headers: hostHdr)
        #expect(cacheClearResp.status == 200)

        let telemResp = await router.handle(
            method: "POST",
            path: "/v1/telemetry/events",
            headers: hostHdr,
            body: Data(#"{"event":"ui_surface_open"}"#.utf8)
        )
        #expect(telemResp.status == 202)

        let logsResp = await router.handle(method: "GET", path: "/v1/admin/config/logs", headers: hostHdr)
        #expect(logsResp.status == 200)

        let clearDataResp = await router.handle(method: "POST", path: "/v1/data/clear", headers: hostHdr)
        #expect(clearDataResp.status == 200)
        #expect(try store.listConversations().isEmpty)
    }

    // MARK: - 38. Multimodal Message PATCH Block Sync, Schema PATCH/Generate Fields, 400 Depth on Generate, 6h Profile, SSE X-Request-ID & Case-Insensitive Presets

    @Test("HeylookRouter syncs multimodal contentBlocks on PATCH, preserves Wave 1/2 PATCH & notebook generate fields, validates 400 thinking_depth on conversation generate, filters 6h profile, emits fallback SSE X-Request-ID, and enforces NOCASE presets")
    func testMultimodalPatchSyncSchemaFieldsGenerateDepth6hProfileSSERequestIDAndNocasePresets() async throws {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_wave1_rev_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let dbURL = tmpDir.appendingPathComponent("rev.sqlite")
        let store = try SQLiteStore(databaseURL: dbURL)
        let scheduler = ComputeScheduler()
        let desc = EngineDescription(
            modelID: "harmony-vlm",
            provider: .mlx,
            modelPath: tmpDir.path,
            isLoaded: true,
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived),
            thinkingControls: ThinkingControls(
                supportsThinking: true,
                canDisableThinking: true,
                defaultEnabled: true,
                depthVariable: "reasoning_effort",
                depthValues: ["low", "medium", "high"]
            )
        )
        await scheduler.register(provider: MLXSwiftProvider(description: desc, allowSyntheticFallback: true), description: desc)
        let router = HeylookRouter(store: store, scheduler: scheduler, allowSyntheticFallback: true)
        let hostHdr = ["Host": "127.0.0.1:8000", "Content-Type": "application/json"]

        // 1. PATCH /v1/conversations/{id}/messages/{msgID} updates `.text(...)` inside multimodal `contentBlocks` while preserving `.media(...)`
        try store.upsertConversation(ConversationRecord(id: "conv_mm_patch", title: "MM Edit", defaultModelID: "harmony-vlm"))
        let pngAtt = MessagesRequest.MediaAttachment(
            kind: "image",
            mediaType: "image/png",
            dataBase64: Data([0x89, 0x50, 0x4E, 0x47, 0x01, 0x02]).base64EncodedString(),
            filename: "diagram.png"
        )
        let savedMM = try store.appendMessage(
            MessageRecord(
                id: "m_mm_0",
                conversationID: "conv_mm_patch",
                position: 0,
                role: "user",
                content: "Old pre-edit text",
                contentBlocks: [.media(pngAtt), .text("Old pre-edit text")]
            )
        )
        let patchMsgResp = await router.handle(
            method: "PATCH",
            path: "/v1/conversations/conv_mm_patch/messages/\(savedMM.id)",
            headers: hostHdr,
            body: Data(#"{"content":"Updated post-edit text"}"#.utf8)
        )
        #expect(patchMsgResp.status == 200)
        let reloadedMM = try #require(try store.getMessage(id: savedMM.id, hydrateMedia: true))
        #expect(reloadedMM.content == "Updated post-edit text")
        #expect(reloadedMM.contentBlocks?.count == 2)
        if let blocks = reloadedMM.contentBlocks {
            if case .media(let keptAtt) = blocks[0] {
                #expect(keptAtt.kind == "image")
                #expect(!keptAtt.dataBase64.isEmpty)
            } else {
                Issue.record("Expected preserved .media block at index 0")
            }
            #expect(blocks[1] == .text("Updated post-edit text"))
        }
        let promptInspect = await router.handle(
            method: "GET",
            path: "/v1/conversations/conv_mm_patch/prompt",
            queryItems: [URLQueryItem(name: "model", value: "harmony-vlm")],
            headers: hostHdr
        )
        #expect(promptInspect.status == 200)
        let promptBody = String(decoding: promptInspect.body, as: UTF8.self)
        #expect(promptBody.contains("Updated post-edit text"))
        #expect(!promptBody.contains("Old pre-edit text"))

        // 2 & 6. Case-insensitive preset uniqueness (`"creative"` vs `"Creative"` -> 409) + PATCH /v1/presets/{id} updating & clearing `system_prompt` and `scope`
        let p1 = PresetRecord(
            id: "preset_creative_1",
            name: "creative",
            scope: "sampler",
            settings: SamplerSettings(temperature: 0.9),
            systemPrompt: "Initial system prompt"
        )
        let p1Resp = await router.handle(method: "POST", path: "/v1/presets", headers: hostHdr, body: try JSONEncoder().encode(p1))
        #expect(p1Resp.status == 201)

        let pDupCase = PresetRecord(
            id: "preset_creative_2",
            name: "Creative",
            scope: "sampler",
            settings: SamplerSettings(temperature: 0.8)
        )
        let pDupResp = await router.handle(method: "POST", path: "/v1/presets", headers: hostHdr, body: try JSONEncoder().encode(pDupCase))
        #expect(pDupResp.status == 409)

        let pUpdateScopeAndSys = await router.handle(
            method: "PATCH",
            path: "/v1/presets/preset_creative_1",
            headers: hostHdr,
            body: Data(#"{"scope":"full","system_prompt":"Revised system prompt"}"#.utf8)
        )
        #expect(pUpdateScopeAndSys.status == 200)
        let afterSet = try #require(try store.getPreset(id: "preset_creative_1"))
        #expect(afterSet.scope == "full")
        #expect(afterSet.systemPrompt == "Revised system prompt")

        let pClearSys = await router.handle(
            method: "PATCH",
            path: "/v1/presets/preset_creative_1",
            headers: hostHdr,
            body: Data(#"{"system_prompt":""}"#.utf8)
        )
        #expect(pClearSys.status == 200)
        #expect(try store.getPreset(id: "preset_creative_1")?.systemPrompt == nil)

        // Conversation PATCH `params` + `applied_preset_id` (and clearing)
        let convPatchSet = await router.handle(
            method: "PATCH",
            path: "/v1/conversations/conv_mm_patch",
            headers: hostHdr,
            body: Data(#"{"params":{"temperature":0.25,"top_k":16},"applied_preset_id":"preset_creative_1"}"#.utf8)
        )
        #expect(convPatchSet.status == 200)
        let patchedConv = try #require(try store.getConversation(id: "conv_mm_patch"))
        #expect(patchedConv.params?.temperature == 0.25)
        #expect(patchedConv.params?.topK == 16)
        #expect(patchedConv.appliedPresetID == "preset_creative_1")

        let convPatchClear = await router.handle(
            method: "PATCH",
            path: "/v1/conversations/conv_mm_patch",
            headers: hostHdr,
            body: Data(#"{"params":null,"applied_preset_id":""}"#.utf8)
        )
        #expect(convPatchClear.status == 200)
        let clearedConv = try #require(try store.getConversation(id: "conv_mm_patch"))
        #expect(clearedConv.params == nil)
        #expect(clearedConv.appliedPresetID == nil)

        // Notebook PATCH `params` + `applied_preset_id` AND `POST /v1/notebooks/{id}/generate` forwarding `nb.params` (verified via thinking_depth validation)
        _ = await router.handle(
            method: "POST",
            path: "/v1/notebooks",
            headers: hostHdr,
            body: try JSONEncoder().encode(NotebookRecord(id: "nb_params", title: "NB", body: "Draft", modelID: "harmony-vlm"))
        )
        let nbPatch = await router.handle(
            method: "PATCH",
            path: "/v1/notebooks/nb_params",
            headers: hostHdr,
            body: Data(#"{"params":{"temperature":0.12,"thinking_depth":"ultra_invalid"},"applied_preset_id":"preset_creative_1"}"#.utf8)
        )
        #expect(nbPatch.status == 200)
        let patchedNB = try #require(try store.getNotebook(id: "nb_params"))
        #expect(patchedNB.params?.temperature == 0.12)
        #expect(patchedNB.appliedPresetID == "preset_creative_1")

        // Calling `/v1/notebooks/nb_params/generate` forwards `nb.params.thinkingDepth == "ultra_invalid"` into `MessagesRequest` -> HTTP 400!
        let nbBadDepthGen = await router.handle(
            method: "POST",
            path: "/v1/notebooks/nb_params/generate",
            headers: hostHdr,
            body: Data("{}".utf8)
        )
        #expect(nbBadDepthGen.status == 400)

        // Overriding `thinking_depth` in the `/generate` body to `"high"` succeeds (200)
        let nbValidOverrideGen = await router.handle(
            method: "POST",
            path: "/v1/notebooks/nb_params/generate",
            headers: hostHdr,
            body: Data(#"{"thinking_depth":"high"}"#.utf8)
        )
        #expect(nbValidOverrideGen.status == 200)

        // 3. POST /v1/conversations/{id}/generate validates `thinking_depth` with HTTP 400 BEFORE mutating the store
        let badConvGen = ConversationGenerateRequest(
            model: "harmony-vlm",
            content: "Should not persist on 400",
            mode: .append,
            thinkingDepth: "bogus_depth"
        )
        let badConvResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/conv_never_created/generate",
            headers: hostHdr,
            body: try JSONEncoder().encode(badConvGen)
        )
        #expect(badConvResp.status == 400)
        #expect(try store.getConversation(id: "conv_never_created") == nil)

        // 4. GET /v1/performance/profile/6h includes 1h-old stats and excludes 12h-old stats
        let nowMs = Int64(Date().timeIntervalSince1970 * 1000)
        _ = try store.appendMessage(
            MessageRecord(id: "m_stat_old", conversationID: "conv_mm_patch", position: 1, role: "assistant", content: "Old")
        )
        _ = try store.appendMessage(
            MessageRecord(id: "m_stat_recent", conversationID: "conv_mm_patch", position: 2, role: "assistant", content: "Recent")
        )
        try store.upsertMessageStats(
            MessageStatsRecord(
                messageID: "m_stat_old",
                conversationID: "conv_mm_patch",
                modelID: "harmony-vlm",
                promptTokens: 20,
                outputTokens: 10,
                generationTPS: 50.0,
                createdUnixMs: nowMs - 12 * 3_600_000
            )
        )
        try store.upsertMessageStats(
            MessageStatsRecord(
                messageID: "m_stat_recent",
                conversationID: "conv_mm_patch",
                modelID: "harmony-vlm",
                promptTokens: 30,
                outputTokens: 15,
                generationTPS: 80.0,
                createdUnixMs: nowMs - 1 * 3_600_000
            )
        )
        let prof6hResp = await router.handle(method: "GET", path: "/v1/performance/profile/6h", headers: hostHdr)
        #expect(prof6hResp.status == 200)
        let prof6hObj = try JSONSerialization.jsonObject(with: prof6hResp.body) as? [String: Any]
        #expect((prof6hObj?["sample_count"] as? Int) == 1)

        let prof24hResp = await router.handle(method: "GET", path: "/v1/performance/profile/24h", headers: hostHdr)
        let prof24hObj = try JSONSerialization.jsonObject(with: prof24hResp.body) as? [String: Any]
        #expect((prof24hObj?["sample_count"] as? Int) == 2)

        // 5. Live chunked SSE wire dispatcher (`dispatchRawHTTPBuffer`) emits generated `X-Request-ID: req_...` when client omits header
        actor WireCapture {
            var data = Data()
            var didClose = false
            func append(_ d: Data) { data.append(d) }
            func markClosed() { didClose = true }
        }
        let wireCapture = WireCapture()
        let streamPayload = #"{"model":"harmony-vlm","messages":[{"role":"user","content":"Ping"}],"stream":true}"#
        let rawHTTP = "POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nContent-Type: application/json\r\nContent-Length: \(streamPayload.utf8.count)\r\n\r\n\(streamPayload)"
        await HeylookHTTPServer.dispatchRawHTTPBuffer(
            Data(rawHTTP.utf8),
            router: router,
            sendBytes: { chunk in await wireCapture.append(chunk) },
            closeConnection: { Task { await wireCapture.markClosed() } }
        )
        let capturedWire = String(decoding: await wireCapture.data, as: UTF8.self)
        #expect(capturedWire.contains("HTTP/1.1 200 OK\r\n"))
        #expect(capturedWire.contains("Transfer-Encoding: chunked\r\n"))
        #expect(capturedWire.contains("X-Request-ID: req_"))
        #expect(capturedWire.hasSuffix("0\r\n\r\n"))

        // 7. `model.heylook.toml` `chat_template` sidecar key sets `origin == "sidecar"` in `inspectChatTemplate`
        let sidecarTplDir = tmpDir.appendingPathComponent("sidecar_tpl_model", isDirectory: true)
        try FileManager.default.createDirectory(at: sidecarTplDir, withIntermediateDirectories: true)
        var scTpl = ModelSidecarConfig()
        scTpl.chatTemplate = "{% for m in messages %}[SC:{{ m.role }}]{{ m.content }}{% endfor %}"
        try scTpl.toTOML().write(to: sidecarTplDir.appendingPathComponent("model.heylook.toml"), atomically: true, encoding: .utf8)
        let scState = HeylookConfig.inspectChatTemplate(in: sidecarTplDir)
        #expect(scState.origin == "sidecar")
        let scRendered = try HeylookConfig.renderPromptDetailed(
            in: sidecarTplDir,
            system: nil,
            turns: [.init(role: "user", content: "Hello")]
        )
        #expect(scRendered.prompt == "[SC:user]Hello")
    }

    @Test("Wave 2 public seams: non-destructive [scan] folders TOML editing, ChatTemplateInspection + validateSyntax, assessRAMFit, and HeylookObservabilityLog")
    func testWave2ConfigTemplateRAMFitAndObservabilitySeams() throws {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_wave2_seams_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // 1. Non-destructive `[scan] folders` round-trip preserving `[[models]]`, `admin_token`, and `allowed_hosts`
        let cfgURL = tmpDir.appendingPathComponent("heylook.toml")
        let initialTOML = """
        default_model = "qwen3.5-9b-mlx"
        max_loaded_models = 4
        max_queue_depth = 3
        admin_token = "test_fixture_token"
        allowed_hosts = ["example.local", "127.0.0.1"]

        [scan]
        folders = ["~/.heylook/models/primary"]

        [[models]]
        id = "qwen3.5-9b-mlx"
        path = "~/.heylook/models/primary/Qwen3.5-9B-MLX-4bit"
        provider = "mlx"
        pinned = true
        ctx_size = 32768
        tags = ["chat", "thinking"]

        [[models]]
        id = "gemma4-gguf"
        path = "~/.heylook/models/gguf/gemma-4.gguf"
        provider = "gguf"
        image_max_tokens = 560
        """
        try initialTOML.write(to: cfgURL, atomically: true, encoding: .utf8)

        let afterAdd = try HeylookConfig.addScanFolder("~/.heylook/models/secondary", configURL: cfgURL)
        #expect(afterAdd.scanFolders == ["~/.heylook/models/primary", "~/.heylook/models/secondary"])
        // Duplicate normalized path with trailing slash is ignored
        let afterDup = try HeylookConfig.addScanFolder("~/.heylook/models/secondary/", configURL: cfgURL)
        #expect(afterDup.scanFolders == ["~/.heylook/models/primary", "~/.heylook/models/secondary"])

        let reparsedAfterAdd = HeylookConfig.parseTOML(try String(contentsOf: cfgURL, encoding: .utf8))
        #expect(reparsedAfterAdd.defaultModel == "qwen3.5-9b-mlx")
        #expect(reparsedAfterAdd.maxLoadedModels == 4)
        #expect(reparsedAfterAdd.maxQueueDepth == 3)
        #expect(reparsedAfterAdd.adminToken == "test_fixture_token")
        #expect(reparsedAfterAdd.allowedHosts == ["example.local", "127.0.0.1"])
        #expect(reparsedAfterAdd.explicitModels.count == 2)
        #expect(reparsedAfterAdd.explicitModels[0].sidecar.pinned == true)
        #expect(reparsedAfterAdd.explicitModels[0].sidecar.ctxSize == 32768)
        #expect(reparsedAfterAdd.explicitModels[0].sidecar.tags == ["chat", "thinking"])
        #expect(reparsedAfterAdd.explicitModels[1].sidecar.imageMaxTokens == 560)

        let afterRemove = try HeylookConfig.removeScanFolder("~/.heylook/models/primary", configURL: cfgURL)
        #expect(afterRemove.scanFolders == ["~/.heylook/models/secondary"])
        let reparsedAfterRemove = HeylookConfig.parseTOML(try String(contentsOf: cfgURL, encoding: .utf8))
        #expect(reparsedAfterRemove.scanFolders == ["~/.heylook/models/secondary"])
        #expect(reparsedAfterRemove.explicitModels.count == 2)
        #expect(reparsedAfterRemove.adminToken == "test_fixture_token")

        // 2. `ChatTemplateInspection` (`origin`, `writable`, `inertReason`, `stale`, `prefixStable`, `modifiedSinceDownload`, `detectedCapabilities`) + `validateSyntax`
        let tplModelDir = tmpDir.appendingPathComponent("tpl_inspect_model", isDirectory: true)
        let hfMetaDir = tplModelDir
            .appendingPathComponent(".cache", isDirectory: true)
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("download", isDirectory: true)
        try FileManager.default.createDirectory(at: hfMetaDir, withIntermediateDirectories: true)

        let vendorTpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.content is string %}{{ m.content }}{% else %}<|vision_start|>{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt and enable_thinking %}<think>{% endif %}"
        let vendorURL = tplModelDir.appendingPathComponent("chat_template.jinja")
        try vendorTpl.write(to: vendorURL, atomically: true, encoding: .utf8)

        let vendorBytes = Data(vendorTpl.utf8)
        var gitBlob = Data("blob \(vendorBytes.count)\0".utf8)
        gitBlob.append(vendorBytes)
        let vendorGitSHA1 = Insecure.SHA1.hash(data: gitBlob).map { String(format: "%02x", $0) }.joined()
        try "https://huggingface.co/example/resolve/main/chat_template.jinja\n\(vendorGitSHA1)\n"
            .write(to: hfMetaDir.appendingPathComponent("chat_template.jinja.metadata"), atomically: true, encoding: .utf8)

        let cleanInspect: ChatTemplateInspection = HeylookConfig.inspectChatTemplate(in: tplModelDir)
        #expect(cleanInspect.origin == "vendor_sidecar")
        #expect(cleanInspect.writable == true)
        #expect(cleanInspect.modifiedSinceDownload == false)
        #expect(cleanInspect.prefixStable == true)
        #expect(cleanInspect.detectedCapabilities.supportsThinking == true)
        #expect(cleanInspect.detectedCapabilities.supportsVision == true)
        #expect(cleanInspect.stale == false)

        // Mutate vendor file on disk -> modifiedSinceDownload flips to true, stale flips when loadedTemplateText differs
        try (vendorTpl + "\n{# local edit #}").write(to: vendorURL, atomically: true, encoding: .utf8)
        let dirtyInspect = HeylookConfig.inspectChatTemplate(in: tplModelDir, loadedTemplateText: vendorTpl)
        #expect(dirtyInspect.modifiedSinceDownload == true)
        #expect(dirtyInspect.stale == true)

        // Sidecar `chat_template_path` sets `inertReason` when operator override is also present
        let customURL = tplModelDir.appendingPathComponent("custom.jinja")
        try "{% for m in messages %}{{ strftime_now('%H:%M') }} {{ m.content }}{% endfor %}".write(to: customURL, atomically: true, encoding: .utf8)
        var scInert = ModelSidecarConfig()
        scInert.chatTemplatePath = "custom.jinja"
        try HeylookConfig.writeSidecar(scInert, in: tplModelDir)
        try vendorTpl.write(to: tplModelDir.appendingPathComponent("chat_template.heylook.jinja"), atomically: true, encoding: .utf8)
        let inertInspect = HeylookConfig.inspectChatTemplate(in: tplModelDir)
        #expect(inertInspect.inertReason != nil)
        #expect(inertInspect.prefixStable == false)
        #expect(inertInspect.prefixStabilityNote != nil)

        try HeylookJinjaRenderer.validateSyntax(vendorTpl)
        #expect(throws: JinjaTemplateError.self) {
            try HeylookJinjaRenderer.validateSyntax("   ")
        }
        #expect(throws: JinjaTemplateError.self) {
            try HeylookJinjaRenderer.validateSyntax("{% if enable_thinking %}{{ message.content }")
        }
        #expect(throws: JinjaTemplateError.self) {
            try HeylookJinjaRenderer.validateSyntax("{% if enable_thinking %}hi{% endfor %}")
        }

        // 3. `GGUFSpawnConfig.assessRAMFit` & `HeylookConfig.estimateModelMemoryMB`
        let ggufDir = tmpDir.appendingPathComponent("gguf_fit_model", isDirectory: true)
        try FileManager.default.createDirectory(at: ggufDir, withIntermediateDirectories: true)
        let mainGGUF = ggufDir.appendingPathComponent("model-q4_k_m.gguf")
        let drafterGGUF = ggufDir.appendingPathComponent("mtp-drafter.gguf")
        let mmprojGGUF = ggufDir.appendingPathComponent("mmproj-f16.gguf")
        var mainPayload = GGUFHeaderReader.buildMinimalGGUF(
            stringMetadata: ["general.architecture": "qwen35"],
            uint32Metadata: [
                "qwen35.block_count": 32,
                "qwen35.attention.head_count_kv": 8,
                "qwen35.embedding_length": 4096,
            ]
        )
        mainPayload.append(Data(count: 4 * 1_048_576))
        try mainPayload.write(to: mainGGUF)
        try Data(count: 3 * 1_048_576).write(to: drafterGGUF)
        try Data(count: 2 * 1_048_576).write(to: mmprojGGUF)

        let estMB = HeylookConfig.estimateModelMemoryMB(
            in: ggufDir,
            primaryGGUFURL: mainGGUF,
            drafterPath: drafterGGUF.path,
            mmprojPath: mmprojGGUF.path
        )
        #expect(estMB >= 9)

        let ggufDesc = EngineDescription(
            modelID: "qwen35-fit",
            provider: .gguf,
            modelPath: mainGGUF.path,
            contextSize: .init(value: 32768, provenance: .derived, effect: .reload),
            estimatedMemoryMB: 12 * 1024 // 12 GB main model
        )
        let generousFit = GGUFSpawnConfig.assessRAMFit(for: ggufDesc, ctxSize: 8192, headroomOverrideGB: 64.0)
        #expect(generousFit.fitsInMemory == true)
        #expect(generousFit.droppedDrafterForMemory == false)

        let tinyFit = GGUFSpawnConfig.assessRAMFit(for: ggufDesc, ctxSize: 32768, headroomOverrideGB: 1.0)
        #expect(tinyFit.fitsInMemory == false)
        #expect(tinyFit.droppedDrafterForMemory == true)
        #expect(tinyFit.recommendedCtxSize < 32768)

        // 4. `HeylookObservabilityLog` level filtering, retention pruning, category/search query, and clear
        let logFile = tmpDir.appendingPathComponent("events_wave2.jsonl")
        let obsLog = HeylookObservabilityLog(maxEntries: 32, level: "WARN", logFileURL: logFile)
        #expect(obsLog.level == "WARN")
        obsLog.record(level: "DEBUG", category: "mlx", message: "debug ignored")
        obsLog.record(level: "INFO", category: "router", message: "info ignored")
        obsLog.record(level: "WARN", category: "mlx", message: "kv_trim fallback triggered", requestID: "req_w1")
        obsLog.record(level: "ERROR", category: "gguf", message: "drafter rejected", requestID: "req_e1")
        #expect(obsLog.recentEvents.count == 2)
        #expect(FileManager.default.fileExists(atPath: logFile.path))

        obsLog.level = "DEBUG"
        for idx in 0..<24 {
            obsLog.record(
                level: idx % 2 == 0 ? "DEBUG" : "INFO",
                category: idx % 3 == 0 ? "mlx" : "router",
                message: "event_\(idx) kv_trim=\(idx % 3 == 0)"
            )
        }
        obsLog.retentionLimit = 16
        #expect(obsLog.retentionLimit == 16)
        #expect(obsLog.recentEvents.count == 16)

        let mlxFiltered = obsLog.recentEntries(limit: 16, level: "INFO", category: "mlx", search: "kv_trim=true")
        #expect(!mlxFiltered.isEmpty)
        #expect(mlxFiltered.allSatisfy { $0.category == "mlx" && $0.level != "DEBUG" })

        obsLog.clear()
        #expect(obsLog.recentEvents.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: logFile.path))
    }

    @Test("Wave 2 store & lock integration: cloneConversation(upToPosition:), clearAllData(), queryPerformanceProfile(1h/6h/24h/7d), mutationStream(), and shared RequestRegistry 409 lock")
    func testWave2StoreForkProfileClearAndSharedLockIntegration() async throws {
        let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_wave2_store_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let store = try SQLiteStore(databaseURL: tmpDir.appendingPathComponent("store.sqlite"))
        let scheduler = ComputeScheduler(maxLoadedGPUModels: 2, maxQueueDepth: 2)
        let desc = EngineDescription(
            modelID: "wave2-mlx",
            provider: .mlx,
            modelPath: tmpDir.path,
            supportsVision: .init(value: true, provenance: .derived, effect: .reload),
            supportsThinking: .init(value: true, provenance: .derived, effect: .reload)
        )
        await scheduler.register(
            provider: MLXSwiftProvider(description: desc, allowSyntheticFallback: true),
            description: desc
        )
        let sharedRegistry = RequestRegistry()
        let router = HeylookRouter(
            store: store,
            scheduler: scheduler,
            registry: sharedRegistry,
            allowSyntheticFallback: true
        )

        actor MutationRecorder {
            var events: [StoreMutationEvent] = []
            func append(_ e: StoreMutationEvent) { events.append(e) }
            func snapshot() -> [StoreMutationEvent] { events }
        }
        let recorder = MutationRecorder()
        let stream = store.mutationStream()
        let listenerTask = Task {
            for await ev in stream {
                await recorder.append(ev)
            }
        }
        defer { listenerTask.cancel() }

        // 1. Seed 4-turn multimodal conversation & fork at position 1 (`upToPosition: 1`)
        let conv = ConversationRecord(id: "conv_fork_src", title: "Root Thread", systemPrompt: "System A", defaultModelID: "wave2-mlx")
        try store.upsertConversation(conv)
        let imgBase64 = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]).base64EncodedString()
        let mmBlocks: [MessagesRequest.MessageTurn.ContentPart] = [
            .media(.init(kind: "image", mediaType: "image/png", dataBase64: imgBase64)),
            .text("Describe diagram"),
        ]
        try store.appendMessage(MessageRecord(id: "m0", conversationID: conv.id, position: 0, role: "user", content: "Describe diagram", contentBlocks: mmBlocks))
        try store.appendMessage(MessageRecord(id: "m1", conversationID: conv.id, position: 1, role: "assistant", content: "Diagram shows Dual-Lane ANE.", thinking: "Inspecting nodes"))
        try store.appendMessage(MessageRecord(id: "m2", conversationID: conv.id, position: 2, role: "user", content: "Turn 2 question"))
        try store.appendMessage(MessageRecord(id: "m3", conversationID: conv.id, position: 3, role: "assistant", content: "Turn 2 answer"))

        let forked = try store.cloneConversation(id: conv.id, upToPosition: 1, newTitle: "Fork @ Turn 1")
        #expect(forked.id != conv.id)
        #expect(forked.title == "Fork @ Turn 1")
        let forkedMsgs = try store.listMessages(conversationID: forked.id, hydrateMedia: true)
        #expect(forkedMsgs.map(\.position) == [0, 1])
        #expect(forkedMsgs.map(\.content) == ["Describe diagram", "Diagram shows Dual-Lane ANE."])
        let hydratedMedia = forkedMsgs[0].contentBlocks?.compactMap { part -> String? in
            if case .media(let att) = part { return att.dataBase64 }
            return nil
        }.first
        #expect(hydratedMedia == imgBase64)
        #expect(try store.listMessages(conversationID: conv.id).count == 4)

        // 2. `queryPerformanceProfile` across `1h` / `6h` / `24h` / `7d` + per-model breakdown
        let nowMs = SQLiteStore.nowMillis()
        let statRows: [MessageStatsRecord] = [
            MessageStatsRecord(messageID: "m0", conversationID: conv.id, modelID: "wave2-mlx", promptTokens: 100, outputTokens: 50, cachedTokens: 60, promptTPS: 400, generationTPS: 80, ttftMs: 120, createdUnixMs: nowMs - 30 * 60_000),
            MessageStatsRecord(messageID: "m1", conversationID: conv.id, modelID: "wave2-mlx", promptTokens: 200, outputTokens: 100, cachedTokens: 150, promptTPS: 500, generationTPS: 100, ttftMs: 100, createdUnixMs: nowMs - 3 * 3_600_000),
            MessageStatsRecord(messageID: "m2", conversationID: conv.id, modelID: "gemma4-gguf", promptTokens: 80, outputTokens: 40, cachedTokens: 20, promptTPS: 250, generationTPS: 60, ttftMs: 180, createdUnixMs: nowMs - 18 * 3_600_000),
            MessageStatsRecord(messageID: "m3", conversationID: conv.id, modelID: "gemma4-gguf", promptTokens: 120, outputTokens: 60, cachedTokens: 40, promptTPS: 300, generationTPS: 70, ttftMs: 160, createdUnixMs: nowMs - 3 * 86_400_000),
        ]
        for row in statRows {
            try store.upsertMessageStats(row)
        }
        #expect(try store.queryPerformanceProfile(window: "1h", nowUnixMs: nowMs).sampleCount == 1)
        #expect(try store.queryPerformanceProfile(window: "6h", nowUnixMs: nowMs).sampleCount == 2)
        #expect(try store.queryPerformanceProfile(window: "24h", nowUnixMs: nowMs).sampleCount == 3)
        let p7d = try store.queryPerformanceProfile(window: "7d", nowUnixMs: nowMs)
        #expect(p7d.sampleCount == 4)
        #expect(p7d.totalOutputTokens == 250)
        #expect(p7d.totalCachedTokens == 270)
        #expect(p7d.byModel.count == 2)
        #expect(p7d.byModel.first(where: { $0.modelID == "wave2-mlx" })?.averageGenerationTPS == 90.0)

        let hostHdr = ["Host": "127.0.0.1:8000"]
        let routeProf = await router.handle(method: "GET", path: "/v1/performance/profile/7d", headers: hostHdr)
        #expect(routeProf.status == 200)
        let routeProfObj = try JSONSerialization.jsonObject(with: routeProf.body) as? [String: Any]
        #expect((routeProfObj?["by_model"] as? [[String: Any]])?.count == 2)

        // 3. Shared `RequestRegistry` lock contention (`tryAcquireConversationLock` -> HTTP `409` -> `releaseConversationLock` -> HTTP `200`)
        let lockExec = sharedRegistry.tryAcquireConversationLock(
            conversationID: conv.id,
            requestID: "ui_chat_req_1",
            modelID: "wave2-mlx"
        )
        #expect(lockExec != nil)
        #expect(sharedRegistry.isConversationGenerating(conv.id) == true)

        let genBody = try JSONSerialization.data(withJSONObject: ["model": "wave2-mlx", "content": "Concurrent HTTP turn"])
        let conflictResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/\(conv.id)/generate",
            headers: hostHdr,
            body: genBody
        )
        #expect(conflictResp.status == 409)

        sharedRegistry.releaseConversationLock(conversationID: conv.id, requestID: "ui_chat_req_1", executionID: lockExec)
        #expect(sharedRegistry.isConversationGenerating(conv.id) == false)

        let okResp = await router.handle(
            method: "POST",
            path: "/v1/conversations/\(conv.id)/generate",
            headers: hostHdr,
            body: genBody
        )
        #expect(okResp.status == 200)

        // 4. `clearAllData()` wipes conversations, messages, stats, media_blobs, and notebooks while preserving presets
        _ = try store.createPreset(PresetRecord(id: "p_custom_w2", name: "Wave2 Custom", settings: SamplerSettings(temperature: 0.3)))
        try store.upsertNotebook(NotebookRecord(id: "nb_w2", title: "Scratch", body: "Draft"))
        let presetCountBefore = try store.listPresets().count
        try store.clearAllData()

        #expect(try store.listConversations().isEmpty)
        #expect(try store.listMessages(conversationID: conv.id).isEmpty)
        #expect(try store.listMessageStats(sinceUnixMs: 0).isEmpty)
        #expect(try store.listNotebooks().isEmpty)
        #expect(try store.listPresets().count == presetCountBefore)

        try await Task.sleep(for: .milliseconds(20))
        let recordedEvents = await recorder.snapshot()
        #expect(recordedEvents.contains(.dataCleared))
    }
}
