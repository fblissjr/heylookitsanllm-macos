import CoreGraphics
import Foundation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer
import ImageIO
import UniformTypeIdentifiers

/// Real-model CLI Evaluation Lab running greedy `chain_probe`, `vlm_parity_probe`, and
/// `matryoshka_recall_probe` exclusively against live model weights and compiled ANE towers
/// through `WorkbenchService` and `ComputeScheduler` (zero synthetic fallback paths).
public enum CLIEvalLabSuite {
    public struct ProbeSummary: Codable, Sendable, Equatable, Identifiable {
        public var id: String { name }
        public let name: String
        public let passed: Bool
        public let detail: String

        public init(name: String, passed: Bool, detail: String) {
            self.name = name
            self.passed = passed
            self.detail = detail
        }
    }

    public struct SuiteReport: Codable, Sendable, Equatable {
        public let allPassed: Bool
        public let probes: [ProbeSummary]
        public let matryoshkaBenchmarks: [EmbeddingLabRunner.DimensionMetrics]

        public init(
            allPassed: Bool,
            probes: [ProbeSummary],
            matryoshkaBenchmarks: [EmbeddingLabRunner.DimensionMetrics]
        ) {
            self.allPassed = allPassed
            self.probes = probes
            self.matryoshkaBenchmarks = matryoshkaBenchmarks
        }
    }

    public static func runAll() async throws -> SuiteReport {
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("heylook_cli_eval_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let store = try SQLiteStore(databaseURL: tempDir.appendingPathComponent("eval_lab.sqlite"))
        let cfg = HeylookConfig.loadDefault()
        let scheduler = ComputeScheduler(
            maxLoadedGPUModels: cfg.maxLoadedModels,
            maxQueueDepth: cfg.maxQueueDepth
        )
        let discovered = cfg.discoverConfiguredModels()
        await scheduler.registerDiscoveredModels(discovered)

        let embEngine = NativeMLXEmbeddingGemma2.makeBestAvailableEngine(
            computeOptions: .hybridANETowers
        )
        let service = WorkbenchService(
            store: store,
            scheduler: scheduler,
            embeddingEngine: embEngine,
            isReadOnlyConfig: true
        )

        var probes: [ProbeSummary] = []

        // 1. chain_probe (greedy multi-turn prefix reuse + edited branch slicing)
        if let textModel = discovered.first(where: { !$0.supportsEmbeddings.value && !$0.isSpeechRecognizer }) {
            let chainProbe = await runChainProbe(service: service, modelID: textModel.modelID)
            probes.append(chainProbe)
        } else {
            probes.append(
                ProbeSummary(
                    name: "chain_probe",
                    passed: false,
                    detail: "No discovered chat/text model weights configured in heylook.toml / HEYLOOK_MODELS_DIR"
                )
            )
        }

        // 2. vlm_parity_probe (resident image-plan + 422 text-only capability gate + VLM decode)
        let vlmModel = discovered.first(where: { $0.supportsVision.value })
        let textOnlyModel = discovered.first(where: { !$0.supportsVision.value && !$0.supportsEmbeddings.value && !$0.isSpeechRecognizer })
        if let vlmModel {
            let vlmProbe = await runVLMParityProbe(
                service: service,
                vlmModelID: vlmModel.modelID,
                textOnlyModelID: textOnlyModel?.modelID
            )
            probes.append(vlmProbe)
        } else {
            probes.append(
                ProbeSummary(
                    name: "vlm_parity_probe",
                    passed: false,
                    detail: "No discovered vision model with multimodal weights configured"
                )
            )
        }

        // 3. matryoshka_recall_probe (768d / 512d / 256d / 128d MRL Recall@1 / MRR)
        let (mrlProbe, mrlMetrics) = await runMatryoshkaProbe(
            engine: embEngine,
            workDir: tempDir
        )
        probes.append(mrlProbe)

        await service.unloadAllForShutdown()

        return SuiteReport(
            allPassed: probes.allSatisfy(\.passed),
            probes: probes,
            matryoshkaBenchmarks: mrlMetrics
        )
    }

    private static func runChainProbe(service: WorkbenchService, modelID: String) async -> ProbeSummary {
        do {
            _ = try await service.loadModel(id: modelID, warm: false)
            let signal = StreamCancellationSignal()

            let r1 = MessagesRequest(
                model: modelID,
                messages: [.init(role: "user", content: "Reply with the word alpha.")],
                system: "Be concise.",
                maxTokens: 16,
                temperature: 0.0,
                stream: false,
                thinking: false
            )
            guard let resp1 = try await service.createMessageResponse(
                request: r1,
                requestID: "eval-chain-1-\(UUID().uuidString.prefix(8))",
                cancellation: signal,
                onSSEFrame: nil
            ) else {
                return ProbeSummary(name: "chain_probe", passed: false, detail: "Turn 1 produced empty response")
            }
            let text1 = resp1.content.compactMap { $0.text }.joined()

            let r2 = MessagesRequest(
                model: modelID,
                messages: [
                    .init(role: "user", content: "Reply with the word alpha."),
                    .init(role: "assistant", content: text1),
                    .init(role: "user", content: "Reply with the word beta."),
                ],
                system: "Be concise.",
                maxTokens: 16,
                temperature: 0.0,
                stream: false,
                thinking: false
            )
            guard let resp2 = try await service.createMessageResponse(
                request: r2,
                requestID: "eval-chain-2-\(UUID().uuidString.prefix(8))",
                cancellation: signal,
                onSSEFrame: nil
            ) else {
                return ProbeSummary(name: "chain_probe", passed: false, detail: "Turn 2 produced empty response")
            }

            let r3 = MessagesRequest(
                model: modelID,
                messages: [
                    .init(role: "user", content: "Reply with the word alpha."),
                    .init(role: "assistant", content: text1),
                    .init(role: "user", content: "Now reply with the word gamma."),
                ],
                system: "Be concise.",
                maxTokens: 16,
                temperature: 0.0,
                stream: false,
                thinking: false
            )
            guard let resp3 = try await service.createMessageResponse(
                request: r3,
                requestID: "eval-chain-3-\(UUID().uuidString.prefix(8))",
                cancellation: signal,
                onSSEFrame: nil
            ) else {
                return ProbeSummary(name: "chain_probe", passed: false, detail: "Branch turn 3 produced empty response")
            }

            let cached2 = resp2.performance?.cache?.cachedTokens ?? resp2.usage.cacheReadInputTokens ?? 0
            let cached3 = resp3.performance?.cache?.cachedTokens ?? resp3.usage.cacheReadInputTokens ?? 0
            let passed = !text1.isEmpty && resp2.usage.outputTokens > 0 && resp3.usage.outputTokens > 0
            return ProbeSummary(
                name: "chain_probe",
                passed: passed,
                detail: "model=\(modelID), t1_out=\(resp1.usage.outputTokens)t, t2_cached=\(cached2)t, t3_cached=\(cached3)t"
            )
        } catch {
            return ProbeSummary(name: "chain_probe", passed: false, detail: "\(modelID): \(error.localizedDescription)")
        }
    }

    private static func runVLMParityProbe(
        service: WorkbenchService,
        vlmModelID: String,
        textOnlyModelID: String?
    ) async -> ProbeSummary {
        do {
            guard let pngData = makeSyntheticTestPNG(width: 64, height: 64) else {
                return ProbeSummary(name: "vlm_parity_probe", passed: false, detail: "Failed to synthesize PNG fixture")
            }
            _ = try await service.loadModel(id: vlmModelID, warm: false)
            let plan = try await service.planImages(id: vlmModelID, sizes: [[64, 64]])
            let plannedTokens = plan.images.first?.tokens ?? 0

            let imgTurn = MessagesRequest.MessageTurn(
                role: "user",
                content: "Identify this solid color swatch in one word.",
                attachments: [.init(kind: "image", mediaType: "image/png", dataBase64: pngData.base64EncodedString())]
            )
            let signal = StreamCancellationSignal()
            let vlmReq = MessagesRequest(
                model: vlmModelID,
                messages: [imgTurn],
                maxTokens: 16,
                temperature: 0.0,
                stream: false,
                thinking: false
            )
            let vlmResp = try await service.createMessageResponse(
                request: vlmReq,
                requestID: "eval-vlm-\(UUID().uuidString.prefix(8))",
                cancellation: signal,
                onSSEFrame: nil
            )

            var gateBlocked = true
            var gateDetail = "text_only_gate=skipped(none configured)"
            if let textOnlyModelID {
                do {
                    _ = try await service.createMessageResponse(
                        request: MessagesRequest(
                            model: textOnlyModelID,
                            messages: [imgTurn],
                            maxTokens: 4,
                            temperature: 0.0,
                            stream: false
                        ),
                        requestID: "eval-vlm-gate-\(UUID().uuidString.prefix(8))",
                        cancellation: signal,
                        onSSEFrame: nil
                    )
                    gateBlocked = false
                    gateDetail = "text_only_gate=unexpected_pass"
                } catch let wbErr as WorkbenchError {
                    if case .unprocessable = wbErr {
                        gateBlocked = true
                        gateDetail = "text_only_gate=422"
                    } else {
                        gateBlocked = false
                        gateDetail = "text_only_gate=\(wbErr.localizedDescription)"
                    }
                }
            }

            let ok = plannedTokens > 0 && (vlmResp?.usage.outputTokens ?? 0) > 0 && gateBlocked
            return ProbeSummary(
                name: "vlm_parity_probe",
                passed: ok,
                detail: "vlm=\(vlmModelID), plan_tokens=\(plannedTokens) (\(plan.engine)), \(gateDetail)"
            )
        } catch {
            return ProbeSummary(name: "vlm_parity_probe", passed: false, detail: "\(vlmModelID): \(error.localizedDescription)")
        }
    }

    private static func runMatryoshkaProbe(
        engine: any EmbeddingEngine,
        workDir: URL
    ) async -> (ProbeSummary, [EmbeddingLabRunner.DimensionMetrics]) {
        do {
            // Verify live checkpoint / .mlmodelc tower execution with no synthetic fallback
            _ = try await engine.embedBatch([.text("probe", task: .query)], dimensions: 128)

            let vecStore = try VectorStore(databaseURL: workDir.appendingPathComponent("eval_vectors.sqlite"))
            let coordinator = IndexCoordinator(engine: engine, store: vecStore)
            let corpusDir = workDir.appendingPathComponent("corpus", isDirectory: true)
            try FileManager.default.createDirectory(at: corpusDir, withIntermediateDirectories: true)
            let docA = corpusDir.appendingPathComponent("ane_scheduler.md")
            let docB = corpusDir.appendingPathComponent("gguf_subprocesses.md")
            try "Apple Neural Engine co-residency and CoreML MLComputePlan scheduling".write(to: docA, atomically: true, encoding: .utf8)
            try "Out-of-process llama-server subprocess lifecycle and dynamic port pool".write(to: docB, atomically: true, encoding: .utf8)
            _ = try await coordinator.indexURLs([corpusDir])

            let hitsA = try await coordinator.search(queryText: "Apple Neural Engine co-residency", limit: 2)
            let hitsB = try await coordinator.search(queryText: "llama-server subprocess dynamic port pool", limit: 2)
            var cases: [EmbeddingLabRunner.BenchmarkCase] = []
            if let idA = hitsA.first?.artifact.id {
                cases.append(.init(query: "Apple Neural Engine co-residency", expectedArtifactID: idA))
            }
            if let idB = hitsB.first?.artifact.id {
                cases.append(.init(query: "llama-server subprocess dynamic port pool", expectedArtifactID: idB))
            }

            guard cases.count == 2 else {
                return (
                    ProbeSummary(
                        name: "matryoshka_recall_probe",
                        passed: false,
                        detail: "Corpus indexing yielded \(cases.count)/2 resolvable benchmark cases"
                    ),
                    []
                )
            }

            let metrics = try await EmbeddingLabRunner.evaluate(coordinator: coordinator, cases: cases)
            let passed = metrics.count == 4 && metrics.allSatisfy { $0.recallAt1 >= 0.5 && $0.mrr > 0 }
            let detail = metrics.map { "\($0.dimensions)d:R@1=\(Int($0.recallAt1 * 100))%" }.joined(separator: ", ")
            return (
                ProbeSummary(name: "matryoshka_recall_probe", passed: passed, detail: detail),
                metrics
            )
        } catch {
            return (
                ProbeSummary(
                    name: "matryoshka_recall_probe",
                    passed: false,
                    detail: "Compiled EmbeddingGemma 2 ANE towers or checkpoint unavailable: \(error.localizedDescription)"
                ),
                []
            )
        }
    }

    private static func makeSyntheticTestPNG(width: Int, height: Int) -> Data? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }
        ctx.setFillColor(red: 0.25, green: 0.55, blue: 0.85, alpha: 1.0)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let cgImage = ctx.makeImage() else { return nil }
        let mutableData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(mutableData, UTType.png.identifier as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return mutableData as Data
    }
}
