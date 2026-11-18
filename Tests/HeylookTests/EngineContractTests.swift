import Foundation
import Testing
@testable import HeylookCore
@testable import HeylookEmbeddings
@testable import HeylookEngines
@testable import HeylookServer

@Suite("Engine Contract, Model Discovery & Admin Options Suite")
struct EngineContractTests {

    private static func assertNoHostPaths(in node: Any, route: String, context: String) {
        if let s = node as? String {
            let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
            #expect(!trimmed.hasPrefix("/Users/"), "\(route) \(context) leaked /Users path: \(s)")
            #expect(!trimmed.hasPrefix("/home/"), "\(route) \(context) leaked /home path: \(s)")
            #expect(!trimmed.hasPrefix("/var/"), "\(route) \(context) leaked /var path: \(s)")
            #expect(!trimmed.hasPrefix("/tmp/"), "\(route) \(context) leaked /tmp path: \(s)")
            #expect(!trimmed.hasPrefix("~"), "\(route) \(context) leaked tilde path: \(s)")
            if let eqIdx = trimmed.firstIndex(of: "=") {
                let rhs = String(trimmed[trimmed.index(after: eqIdx)...])
                #expect(!rhs.hasPrefix("/Users/") && !rhs.hasPrefix("/var/") && !rhs.hasPrefix("/tmp/") && !rhs.hasPrefix("~"),
                        "\(route) \(context) leaked path RHS in '\(s)'")
            }
        } else if let dict = node as? [String: Any] {
            for (k, v) in dict {
                #expect(k != "modelPath" && k != "model_path", "\(route) exposed raw path field '\(k)'")
                assertNoHostPaths(in: v, route: route, context: "\(context).\(k)")
            }
        } else if let arr = node as? [Any] {
            for (i, item) in arr.enumerated() {
                assertNoHostPaths(in: item, route: route, context: "\(context)[\(i)]")
            }
        }
    }

    private static func assertFactShape(_ factAny: Any?, context: String) throws {
        let fact = try #require(factAny as? [String: Any], "Missing Fact object at \(context)")
        #expect(Set(fact.keys) == Set(["value", "provenance", "source"]), "Unexpected Fact keys at \(context): \(fact.keys)")
        let provenance = try #require(fact["provenance"] as? String)
        if provenance == "unknown" || provenance == "not_applicable" {
            #expect(fact["value"] is NSNull, "Expected null value for \(provenance) fact at \(context)")
        }
    }

    @Test("GET /v1/models and GET /v1/admin/models obey EngineContract, zero-path-leak, and Fact<T> invariants")
    func testPublicAndAdminModelContractInvariants() async throws {
        let harness = try TestRouterHarness()
        let mlxDir = harness.sandbox.modelsDir.appendingPathComponent("org/qwen3-stub", isDirectory: true)
        let ggufDir = harness.sandbox.modelsDir.appendingPathComponent("gemma-gguf", isDirectory: true)
        let coreaiDir = harness.sandbox.modelsDir.appendingPathComponent("qwen-coreai", isDirectory: true)
        try FileManager.default.createDirectory(at: mlxDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: ggufDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: coreaiDir, withIntermediateDirectories: true)

        try """
        {
          "architectures": ["Qwen3ForCausalLM"],
          "bos_token": "<|im_start|>",
          "eos_token": "<|im_end|>"
        }
        """.write(to: mlxDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let mlxStub = await harness.registerStub(
            id: "org/qwen3-stub",
            provider: .mlx,
            modelPath: mlxDir.path,
            supportsVision: false,
            supportsThinking: true,
            isLoaded: false
        )
        await mlxStub.setObservedState(ObservedContractState(runningContextLength: 16384))
        _ = await harness.registerStub(
            id: "gemma-gguf",
            provider: .gguf,
            modelPath: ggufDir.path,
            supportsVision: true,
            supportsThinking: true,
            isLoaded: false
        )
        _ = await harness.registerStub(
            id: "qwen-coreai",
            provider: .coreai,
            modelPath: coreaiDir.path,
            supportsThinking: false,
            isLoaded: false
        )

        func inspectRows(phase: String) async throws -> [String: [String: Any]] {
            let pubResp = await harness.send("GET", "/v1/models")
            #expect(pubResp.status == 200)
            let pubJSON = try TestFixtures.decodeJSONObject(pubResp.body)
            #expect(pubJSON["object"] as? String == "list")
            let pubRows = try #require(pubJSON["data"] as? [[String: Any]])
            Self.assertNoHostPaths(in: pubJSON, route: "GET /v1/models", context: phase)

            let adminResp = await harness.send("GET", "/v1/admin/models")
            #expect(adminResp.status == 200)
            let adminJSON = try TestFixtures.decodeJSONObject(adminResp.body)
            let adminRows = try #require(adminJSON["models"] as? [[String: Any]])
            #expect(adminJSON["total"] as? Int == adminRows.count)
            #expect(adminRows.count == pubRows.count)

            var enginesByID: [String: [String: Any]] = [:]
            for row in pubRows {
                let id = try #require(row["id"] as? String)
                #expect(row["object"] as? String == "model")
                #expect(row["owned_by"] as? String == "user")
                #expect(row["capabilities"] is [String])
                #expect(row["sampler_defaults"] is [String: Any])
                #expect(row["sampler_sources"] is [String: String])

                let samplerDefaults = try #require(row["sampler_defaults"] as? [String: Any])
                #expect(row["thinking_default"] as? Bool == samplerDefaults["enable_thinking"] as? Bool)

                let engine = try #require(row["engine"] as? [String: Any])
                enginesByID[id] = engine
                try Self.assertFactShape(engine["runtime"], context: "\(phase):\(id).runtime")
                let ctx = try #require(engine["context"] as? [String: Any])
                try Self.assertFactShape(ctx["length"], context: "\(phase):\(id).context.length")
                try Self.assertFactShape(ctx["running"], context: "\(phase):\(id).context.running")
                let tpl = try #require(engine["template"] as? [String: Any])
                for leaf in ["origin", "path", "sha256", "running_sha256", "caps"] {
                    try Self.assertFactShape(tpl[leaf], context: "\(phase):\(id).template.\(leaf)")
                }

                let cache = try #require(engine["cache"] as? [String: Any])
                for (k, v) in cache {
                    try Self.assertFactShape(v, context: "\(phase):\(id).cache.\(k)")
                }

                let settings = try #require(engine["settings"] as? [String: [String: Any]])
                for (name, setting) in settings {
                    let reason = setting["reason"] as? String ?? ""
                    #expect(!reason.isEmpty, "SettingFact '\(name)' missing reason")
                    let prov = setting["provenance"] as? String ?? ""
                    if !(setting["configured"] is NSNull) && setting["configured"] != nil {
                        #expect(prov == "configured", "Configured setting '\(name)' must carry configured provenance")
                    }
                    if prov == "unknown" || prov == "not_applicable" {
                        #expect(setting["value"] is NSNull, "Null value expected for \(prov) setting '\(name)'")
                    }
                }
            }
            return enginesByID
        }

        let before = try await inspectRows(phase: "unloaded")
        let beforeMLXSettings = Set((before["org/qwen3-stub"]?["settings"] as? [String: Any])?.keys ?? [:].keys)
        let beforeMLXCache = Set((before["org/qwen3-stub"]?["cache"] as? [String: Any])?.keys ?? [:].keys)

        // Load one model via public route: observed running context joins without changing key sets
        let loadResp = await harness.send("POST", "/v1/models/org/qwen3-stub/load")
        #expect(loadResp.status == 200)

        let after = try await inspectRows(phase: "loaded")
        let afterMLXSettings = Set((after["org/qwen3-stub"]?["settings"] as? [String: Any])?.keys ?? [:].keys)
        let afterMLXCache = Set((after["org/qwen3-stub"]?["cache"] as? [String: Any])?.keys ?? [:].keys)
        #expect(beforeMLXSettings == afterMLXSettings)
        #expect(beforeMLXCache == afterMLXCache)

        let runningFact = (after["org/qwen3-stub"]?["context"] as? [String: Any])?["running"] as? [String: Any]
        #expect(runningFact?["provenance"] as? String == "observed")
        #expect(runningFact?["value"] as? Int == 16384)
    }

    @Test("GET /v1/admin/models reports discovered vs config sources, strips flat [[models]] metadata, and computes stale_reload_fields")
    func testAdminModelsSourcesExplicitConfigAndStaleReloadFields() async throws {
        let harness = try TestRouterHarness()
        let configDir = harness.sandbox.modelsDir.appendingPathComponent("cfg-model", isDirectory: true)
        let discDir = harness.sandbox.modelsDir.appendingPathComponent("disc-model", isDirectory: true)
        try FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: discDir, withIntermediateDirectories: true)

        try """
        [
          "config.json"
        ]
        """.write(to: configDir.appendingPathComponent("dummy.txt"), atomically: true, encoding: .utf8)

        let sidecarTOML = """
        ctx_size = 16384
        temperature = 0.6
        """
        try sidecarTOML.write(to: configDir.appendingPathComponent("model.heylook.toml"), atomically: true, encoding: .utf8)

        // Parse an explicit [[models]] block with nested config and verify `toExplicitJSONDictionary()` omits flat id/provider
        let parsedTOML = """
        [[models]]
        id = "cfg-model"
        provider = "mlx"
        enabled = true
        tags = ["local"]

        [models.config]
        model_path = "\(configDir.path)"
        ctx_size = 16384
        temperature = 0.6
        """
        let cfg = HeylookConfig.parseTOML(parsedTOML)
        #expect(cfg.explicitModels.count == 1)
        let modelRow = cfg.explicitModels[0]
        let configSidecar = HeylookConfig.loadSidecar(in: configDir)
        let explicitDict = configSidecar?.toExplicitJSONDictionary() ?? [:]
        #expect(explicitDict["ctx_size"] == .int(16384))
        #expect(explicitDict["temperature"] == .double(0.6))
        #expect(explicitDict["id"] == nil)
        #expect(explicitDict["provider"] == nil)
        #expect(modelRow.id == "cfg-model")

        let cfgDesc = EngineDescription(
            modelID: "cfg-model",
            provider: .mlx,
            modelPath: configDir.path,
            isLoaded: false,
            contextSize: ProvenanceField(value: 16384, provenance: .configured, effect: .reload)
        )
        let discDesc = EngineDescription(
            modelID: "disc-model",
            provider: .mlx,
            modelPath: discDir.path,
            isLoaded: false,
            contextSize: ProvenanceField(value: 8192, provenance: .derived, effect: .reload)
        )
        let cfgStub = TestStubInferenceProvider(description: cfgDesc)
        let discStub = TestStubInferenceProvider(description: discDesc)
        await harness.scheduler.register(provider: cfgStub, description: cfgDesc)
        await harness.scheduler.register(provider: discStub, description: discDesc)
        let preLoad = await harness.send("POST", "/v1/models/cfg-model/load")
        #expect(preLoad.status == 200)

        let listResp = await harness.send("GET", "/v1/admin/models")
        #expect(listResp.status == 200)
        let listObj = try TestFixtures.decodeJSONObject(listResp.body)
        let models = try #require(listObj["models"] as? [[String: Any]])
        let byID = Dictionary(uniqueKeysWithValues: models.compactMap { row -> (String, [String: Any])? in
            guard let id = row["id"] as? String else { return nil }
            return (id, row)
        })

        #expect(byID["disc-model"]?["source"] as? String == "discovered")
        #expect((byID["disc-model"]?["config"] as? [String: Any])?.isEmpty == true)

        #expect(byID["cfg-model"]?["source"] as? String == "config")
        let cfgBlock = try #require(byID["cfg-model"]?["config"] as? [String: Any])
        #expect(cfgBlock["ctx_size"] as? Int == 16384)
        #expect(cfgBlock["id"] == nil)
        #expect(cfgBlock["provider"] == nil)
        #expect((byID["cfg-model"]?["stale_reload_fields"] as? [String])?.isEmpty == true)

        // Patch reload-required field -> stale_reload_fields records ["ctx_size"]
        let patchResp = try await harness.sendJSON(
            "PATCH",
            "/v1/admin/models/cfg-model",
            object: ["config": ["ctx_size": 32768]]
        )
        #expect(patchResp.status == 200)

        let getOneResp = await harness.send("GET", "/v1/admin/models/cfg-model")
        #expect(getOneResp.status == 200)
        let oneRow = try TestFixtures.decodeJSONObject(getOneResp.body)
        #expect(oneRow["stale_reload_fields"] as? [String] == ["ctx_size"])

        // Once snapshot matches updated config (post-reload state), stale_reload_fields clears cleanly to []
        let syncedStale = ProviderConfigSchema.staleReloadFields(
            provider: .mlx,
            loadedSnapshot: ["ctx_size": .int(32768)],
            currentConfig: ["ctx_size": .int(32768)]
        )
        #expect(syncedStale.isEmpty)

        // Reloading a stub directory without safetensors weights truthfully fails instead of fabricating success
        let reloadResp = try await harness.sendJSON("POST", "/v1/admin/models/cfg-model/reload", object: [String: Any]())
        #expect(reloadResp.status == 500)
    }

    @Test("GET /v1/admin/model-options satisfies schema completeness, GGUF bounds, and 13-key SamplerFieldKey parity")
    func testAdminModelOptionsCompletenessAndTwinParity() async throws {
        let harness = try TestRouterHarness()
        let resp = await harness.send("GET", "/v1/admin/model-options")
        #expect(resp.status == 200)
        let root = try TestFixtures.decodeJSONObject(resp.body)

        let samplerFields = try #require(root["sampler_fields"] as? [String])
        #expect(samplerFields == SamplerCascade.SamplerFieldKey.allCases.map(\.rawValue))
        #expect(samplerFields.count == 13)
        let expectedWireSamplerKeys = Set(SamplerCascade.SamplerFieldKey.allCases.map(\.canonicalWireKey))
        #expect(expectedWireSamplerKeys.contains("enable_thinking"))

        let providers = try #require(root["providers"] as? [String: [String: Any]])
        let expectedEffects = Set(ConfigFieldEffect.allCases.map(\.rawValue))
        let renderableTypes: Set<String> = ["integer", "number", "string", "boolean", "array", "object"]

        for providerName in ["mlx", "gguf", "coreai"] {
            let pObj = try #require(providers[providerName], "Missing provider '\(providerName)' in /v1/admin/model-options")
            let fields = try #require(pObj["fields"] as? [[String: Any]])
            #expect(!fields.isEmpty)
            let byName = Dictionary(uniqueKeysWithValues: fields.compactMap { f -> (String, [String: Any])? in
                guard let name = f["name"] as? String else { return nil }
                return (name, f)
            })
            #expect(expectedWireSamplerKeys.isSubset(of: Set(byName.keys)), "\(providerName) missing sampler wire keys")
            // `model_path` is strictly hidden + file-only (never offered in visible config controls)
            #expect(byName["model_path"]?["ui"] as? String == "hidden")
            #expect(byName["model_path"]?["file_only"] as? Bool == true)
            for (name, field) in byName {
                #expect(field["anyOf"] == nil, "\(providerName).\(name) leaked anyOf")
                let eff = try #require(field["effect"] as? String)
                #expect(expectedEffects.contains(eff), "\(providerName).\(name) invalid effect '\(eff)'")
                let fType = try #require(field["type"] as? String)
                #expect(renderableTypes.contains(fType), "\(providerName).\(name) invalid type '\(fType)'")
                if eff == ConfigFieldEffect.loadTimeOnly.rawValue {
                    let reason = field["reason"] as? String ?? ""
                    #expect(!reason.isEmpty, "\(providerName).\(name) load_time_only missing reason")
                }
            }
        }

        let ggufFields = Dictionary(
            uniqueKeysWithValues: (providers["gguf"]?["fields"] as? [[String: Any]] ?? []).compactMap { f -> (String, [String: Any])? in
                guard let name = f["name"] as? String else { return nil }
                return (name, f)
            }
        )
        #expect((ggufFields["ctx_size"]?["minimum"] as? NSNumber)?.intValue == 512)
        #expect((ggufFields["spec_draft_p_min"]?["minimum"] as? NSNumber)?.doubleValue == 0.0)
        #expect((ggufFields["spec_draft_p_min"]?["maximum"] as? NSNumber)?.doubleValue == 1.0)
        #expect((ggufFields["n_cpu_moe"]?["minimum"] as? NSNumber)?.intValue == 0)
        #expect(ggufFields["cpu_moe"]?["shape"] as? String == "flag")
        #expect(ggufFields["cpu_moe"]?["type"] as? String == "boolean")
    }

    @Test("PATCH /v1/admin/models refuses isFileOnly fields with 422, reload/status/unload enforce 400 & 409 busy guards")
    func testAdminPatchFileOnlyGuardStatusReloadAndUnloadGuards() async throws {
        let harness = try TestRouterHarness()
        let modelDir = harness.sandbox.modelsDir.appendingPathComponent("org/dotted.model-Q4", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let sidecarURL = modelDir.appendingPathComponent("model.heylook.toml")
        let initialSidecar = "ctx_size = 8192\n"
        try initialSidecar.write(to: sidecarURL, atomically: true, encoding: .utf8)

        _ = await harness.registerStub(
            id: "org/dotted.model-Q4",
            provider: .gguf,
            modelPath: modelDir.path,
            isLoaded: true,
            chunks: [
                GenerationChunk(text: "chunk1", tokenCount: 1, contentTokens: 1, promptTokens: 2),
                GenerationChunk(text: "chunk2", tokenCount: 1, contentTokens: 1, promptTokens: 2, stopReason: .endTurn),
            ],
            chunkDelayNanoseconds: 250_000_000 // 250ms so we can deterministically probe in-flight 409 guards
        )

        // 1. isFileOnly fields must be refused with HTTP 422 and zero sidecar mutation
        for fileOnlyField in ["server_binary", "extra_args", "rpc", "chat_template_path"] {
            let patchVal: Any = fileOnlyField == "extra_args" ? ["--flag"] : "/tmp/custom"
            let patchResp = try await harness.sendJSON(
                "PATCH",
                "/v1/admin/models/org/dotted.model-Q4",
                object: ["config": [fileOnlyField: patchVal]]
            )
            #expect(patchResp.status == 422, "Expected 422 refusing isFileOnly field '\(fileOnlyField)'")
        }
        #expect((try String(contentsOf: sidecarURL, encoding: .utf8)) == initialSidecar)

        // 2. Status route works for slashed/dotted IDs without greedy *id shadowing
        let statusResp = await harness.send("GET", "/v1/admin/models/org/dotted.model-Q4/status")
        #expect(statusResp.status == 200)
        let statusObj = try TestFixtures.decodeJSONObject(statusResp.body)
        #expect(statusObj["loaded"] as? Bool == true)
        #expect(statusObj["requests_active"] as? Int == 0)

        // 3. Reload with non-load setting fails with 400
        let badReload = try await harness.sendJSON(
            "POST",
            "/v1/admin/models/org/dotted.model-Q4/reload",
            object: ["temperature": 0.5]
        )
        #expect(badReload.status == 400)

        // 4. Concurrent generation triggers 409 generation_in_progress on unload & reload
        let genTask = Task {
            try await harness.sendJSON(
                "POST",
                "/v1/messages",
                object: [
                    "model": "org/dotted.model-Q4",
                    "messages": [["role": "user", "content": "hold busy"]],
                    "stream": false,
                ]
            )
        }
        // Poll active request registration deterministically instead of blind timing assumption
        var activeObserved = false
        for _ in 0..<40 {
            if !harness.registry.activeRequests().isEmpty {
                activeObserved = true
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        #expect(activeObserved)

        let unloadBusy = await harness.send("POST", "/v1/admin/models/org/dotted.model-Q4/unload")
        #expect(unloadBusy.status == 409)
        let reloadBusy = await harness.send("POST", "/v1/admin/models/org/dotted.model-Q4/reload")
        #expect(reloadBusy.status == 409)

        let genResp = try await genTask.value
        #expect(genResp.status == 200)

        // 5. After completion, unload succeeds and status flips to loaded: false, then warm load restores loaded: true
        let unloadIdle = await harness.send("POST", "/v1/admin/models/org/dotted.model-Q4/unload")
        #expect(unloadIdle.status == 200)
        let statusAfterUnload = try TestFixtures.decodeJSONObject(
            await harness.send("GET", "/v1/admin/models/org/dotted.model-Q4/status").body
        )
        #expect(statusAfterUnload["loaded"] as? Bool == false)

        let warmLoad = await harness.send(
            "POST",
            "/v1/models/org/dotted.model-Q4/load",
            queryItems: [URLQueryItem(name: "warm", value: "true")]
        )
        #expect(warmLoad.status == 200)
        let warmObj = try TestFixtures.decodeJSONObject(warmLoad.body)
        #expect(warmObj["warmed"] as? Bool == true)

        // 6. Null PATCH deletes explicit sidecar keys cleanly and deletes empty model.heylook.toml
        let clearResp = try await harness.sendJSON(
            "PATCH",
            "/v1/admin/models/org/dotted.model-Q4",
            object: ["config": ["ctx_size": NSNull()]]
        )
        #expect(clearResp.status == 200)
        #expect(!FileManager.default.fileExists(atPath: sidecarURL.path))
    }

    @Test("ComputeScheduler enforces Dual-Lane ANE Co-Residency without evicting GPU LLMs and LRU evicts excess GPU slots")
    func testComputeSchedulerDualLaneAndLRUEviction() async throws {
        let harness = try TestRouterHarness(maxLoadedModels: 1)
        let llm1 = await harness.registerStub(id: "llm-1", provider: .mlx, isLoaded: true)
        let llm2 = await harness.registerStub(id: "llm-2", provider: .mlx, isLoaded: false)
        let asrDesc = EngineDescription(
            modelID: "whisper-ane",
            provider: .coreai,
            modelPath: harness.sandbox.modelsDir.appendingPathComponent("whisper").path,
            computePlacement: ProvenanceField(value: .ane, provenance: .observed, effect: .reload, computePlacement: .ane),
            isLoaded: false,
            supportsAudio: ProvenanceField(value: true, provenance: .observed, effect: .reload),
            tags: ["asr"]
        )
        let asrStub = TestStubInferenceProvider(description: asrDesc)
        await harness.scheduler.register(provider: asrStub, description: asrDesc)

        // Activating an ANE lane model must NEVER evict the loaded GPU LLM (`maxLoadedModels == 1`)
        _ = try await harness.scheduler.activateModel("whisper-ane", warm: false)
        let llm1UnloadDuringANE = await llm1.unloadCount
        #expect(llm1UnloadDuringANE == 0)

        // Activating a second GPU LLM evicts `llm-1` cleanly
        _ = try await harness.scheduler.activateModel("llm-2", warm: false)
        let llm1UnloadAfterGPU = await llm1.unloadCount
        let asrUnloadAfterGPU = await asrStub.unloadCount
        #expect(llm1UnloadAfterGPU == 1)
        #expect(asrUnloadAfterGPU == 0)
        _ = await llm2.preloadCount
    }

    @Test("CoreAIBundleManifest inspects .aimodel bundles cleanly and DiscoveredHeylookServer maps _heylook._tcp Bonjour services")
    func testCoreAIBundleManifestAndBonjourEndpointMapping() throws {
        let sandbox = try TestTempSandbox(prefix: "heylook_coreai_bonjour")
        defer { sandbox.cleanup() }

        let speechBundle = sandbox.modelsDir.appendingPathComponent("parakeet-tdt-0.6b-v3.aimodel", isDirectory: true)
        try FileManager.default.createDirectory(at: speechBundle, withIntermediateDirectories: true)
        let manifestObj: [String: Any] = [
            "format_version": 2,
            "kind": "speech_recognizer",
            "config": ["architecture": "parakeet_tdt", "sample_rate": 16_000],
        ]
        try JSONSerialization.data(withJSONObject: manifestObj)
            .write(to: speechBundle.appendingPathComponent("metadata.json"))

        let manifest = try CoreAIBundleManifest.inspect(at: speechBundle)
        #expect(manifest.kind == "speech_recognizer")
        #expect(manifest.isSpeechRecognizer == true)
        #expect(CoreAIBundleManifest.canonicalBundleID(for: speechBundle) == "parakeet-tdt-0.6b-v3")

        let bonjour = DiscoveredHeylookServer.fromBonjourService(name: "Studio Mac", domain: "local.", defaultPort: 8000)
        #expect(bonjour.isBonjour == true)
        #expect(bonjour.baseURLString == "http://Studio-Mac.local:8000")
    }
}
