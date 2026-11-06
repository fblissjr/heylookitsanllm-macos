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

        options = .allNeuralEngine
        let split2 = options.memorySplitMB
        #expect(split2.aneMB == 335 + 611 + 542)
        #expect(split2.gpuMB == 0)

        // Verify 1 second of 16 kHz sine wave produces 98 frames x 128 mel bins
        let samples = (0..<16_000).map { i in sin(Float(i) * 2 * .pi * 440.0 / 16_000.0) }
        let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: samples)
        #expect(mel != nil)
        #expect(mel?.frameCount == 98)
        #expect(mel?.features.count == 98 * 128)
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
}
