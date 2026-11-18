import CryptoKit
import Foundation
import Testing
@testable import HeylookCore
@testable import HeylookEngines
@testable import HeylookServer

@Suite("Sampler Cascade, Render-Probed Templates, TOMLCodec & GGUF Metadata Suite")
struct SamplerAndTemplateTests {

    @Test("SamplerCascade enforces global floor, strict 3-key vendor allowlist, 5-layer precedence, and SettingFact auto baseline")
    func testSamplerCascadeFloorVendorKeysPrecedenceAndAutoBaseline() throws {
        // 1. Canonical floor (`1.0 / 0.95 / 16384`, KNOBS_OFF `top_k: 0 / min_p: 0.0 / repetition_penalty: 1.0`, `thinking == thinkingCapable`)
        let capTrueSummary = SamplerCascade.resolveModelDefaults(thinkingCapable: true)
        #expect(capTrueSummary.thinkingDefault == true)
        #expect(capTrueSummary.samplerDefaults["temperature"] == .double(1.0))
        #expect(capTrueSummary.samplerDefaults["top_p"] == .double(0.95))
        #expect(capTrueSummary.samplerDefaults["max_tokens"] == .int(16_384))
        #expect(capTrueSummary.samplerDefaults["top_k"] == .int(SamplerCascade.knobsOffTopK))
        #expect(capTrueSummary.samplerDefaults["enable_thinking"] == .bool(true))
        #expect(capTrueSummary.samplerSources["temperature"] == "default")
        #expect(capTrueSummary.samplerSources["enable_thinking"] == nil)

        let capFalseSummary = SamplerCascade.resolveModelDefaults(thinkingCapable: false)
        #expect(capFalseSummary.thinkingDefault == false)
        #expect(capFalseSummary.samplerDefaults["enable_thinking"] == .bool(false))

        // 2. Vendor allowlist strictly contains [temperature, top_p, top_k]; vendor min_p / max_tokens / thinking stay at builtin KNOBS_OFF floor
        #expect(SamplerCascade.vendorSamplingKeys == ["temperature", "top_p", "top_k"])
        let noisyVendor = SamplerSettings(
            temperature: 0.65,
            topP: 0.9,
            topK: 32,
            minP: 0.12,
            repetitionPenalty: 1.4,
            maxTokens: 2048,
            thinking: false
        )
        let vendorResolved = SamplerCascade.resolve(
            builtin: .builtinDefaults,
            vendor: noisyVendor,
            modelSidecar: nil,
            preset: nil,
            request: nil,
            thinkingCapable: true
        )
        #expect(vendorResolved.effective.temperature == 0.65)
        #expect(vendorResolved.effective.topP == 0.9)
        #expect(vendorResolved.effective.topK == 32)
        #expect(vendorResolved.effective.minP == SamplerCascade.knobsOffMinP)
        #expect(vendorResolved.effective.repetitionPenalty == SamplerCascade.knobsOffRepetitionPenalty)
        #expect(vendorResolved.effective.maxTokens == 16_384)
        #expect(vendorResolved.effective.thinking == true)
        #expect(vendorResolved.sourceByField[.temperature] == .vendor)
        #expect(vendorResolved.provenanceByField[.temperature] == .derived)
        #expect(vendorResolved.sourceByField[.minP] == .builtin)
        #expect(vendorResolved.sourceByField[.repetitionPenalty] == .builtin)

        // 3. 5-layer cascade: request > preset > modelSidecar > vendor > builtin
        let sidecar = SamplerSettings(temperature: 0.4, minP: 0.05)
        let preset = SamplerSettings(topP: 0.85, repetitionPenalty: 1.1)
        let request = SamplerSettings(temperature: 0.2, maxTokens: 512, seed: 42)
        let layered = SamplerCascade.resolve(
            builtin: .builtinDefaults,
            vendor: noisyVendor,
            modelSidecar: sidecar,
            preset: preset,
            request: request,
            thinkingCapable: true
        )
        #expect(layered.effective.temperature == 0.2)
        #expect(layered.sourceByField[.temperature] == .request)
        #expect(layered.provenanceByField[.temperature] == .configured)
        #expect(layered.effective.topP == 0.85)
        #expect(layered.sourceByField[.topP] == .preset)
        #expect(layered.effective.minP == 0.05)
        #expect(layered.sourceByField[.minP] == .modelConfig)
        #expect(layered.effective.topK == 32)
        #expect(layered.sourceByField[.topK] == .vendor)
        #expect(layered.effective.maxTokens == 512)
        #expect(layered.effective.seed == 42)

        // 4. SettingFact `auto` retains unstored baseline even when sidecar overrides `temperature`
        let sidecarOnlySummary = SamplerCascade.resolveModelDefaults(
            vendor: noisyVendor,
            modelSidecar: SamplerSettings(temperature: 0.35),
            thinkingCapable: true
        )
        let unstoredSummary = SamplerCascade.resolveModelDefaults(
            vendor: noisyVendor,
            modelSidecar: nil,
            thinkingCapable: true
        )
        let settingFacts = ProviderConfigSchema.buildEngineSettings(
            provider: .mlx,
            effectiveConfig: ["temperature": .double(0.35)],
            explicitKeys: ["temperature"],
            derivedBaseline: ["temperature": .double(0.65)],
            engineDefaultName: "mlx",
            samplerSummary: sidecarOnlySummary,
            unstoredSamplerSummary: unstoredSummary
        )
        let tempFact = try #require(settingFacts["temperature"])
        #expect(tempFact.value == .double(0.35))
        #expect(tempFact.configured == .double(0.35))
        #expect(tempFact.auto == .double(0.65))
        #expect(tempFact.provenance == .configured)
    }

    @Test("StrictSamplerParams preserves sparse overrides, uses enable_thinking wire key, and rejects unknown or out-of-range fields")
    func testStrictSamplerParamsSparseAndValidation() throws {
        let sparse = StrictSamplerParams(settings: SamplerSettings(temperature: 0.55, thinking: false))
        let encoded = try JSONEncoder().encode(sparse)
        let obj = try TestFixtures.decodeJSONObject(encoded)
        #expect(obj["temperature"] as? Double == 0.55)
        #expect(obj["enable_thinking"] as? Bool == false)
        #expect(obj["thinking"] == nil)
        #expect(obj["max_tokens"] == nil)
        #expect(obj["top_p"] == nil)

        let roundTripped = try JSONDecoder().decode(StrictSamplerParams.self, from: encoded)
        #expect(roundTripped == sparse)

        // Empty sparse bag encodes to {}
        let emptyEncoded = try JSONEncoder().encode(StrictSamplerParams())
        #expect(String(data: emptyEncoded, encoding: .utf8) == "{}")

        // Reject unknown key and out-of-bounds values
        #expect(throws: Error.self) {
            _ = try JSONDecoder().decode(StrictSamplerParams.self, from: Data(#"{"bogus_knob": 1}"#.utf8))
        }
        #expect(throws: Error.self) {
            _ = try JSONDecoder().decode(StrictSamplerParams.self, from: Data(#"{"temperature": 2.5}"#.utf8))
        }
        #expect(throws: Error.self) {
            _ = try JSONDecoder().decode(StrictSamplerParams.self, from: Data(#"{"max_tokens": 0}"#.utf8))
        }
        #expect(throws: Error.self) {
            _ = try JSONDecoder().decode(StrictSamplerParams.self, from: Data(#"{"top_k": -1}"#.utf8))
        }
    }

    @Test("HeylookJinjaRenderer evaluates AST constructs, validates syntax, detects prefix instability, and runs 4-shape refusal probes")
    func testHeylookJinjaRendererASTStabilityAnd4ShapeProbes() throws {
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

        #expect(throws: JinjaTemplateError.self) {
            try HeylookJinjaRenderer.validateSyntax("   ")
        }
        #expect(throws: JinjaTemplateError.self) {
            try HeylookJinjaRenderer.validateSyntax("{% if enable_thinking %}<think>")
        }
        #expect(throws: JinjaTemplateError.self) {
            try HeylookJinjaRenderer.validateSyntax("{{ unclosed_expr ")
        }

        // Prefix stability catches timestamps & prior-turn thinking stripping while clearing standard ChatML
        let dateTemplate = "Now: {{ strftime_now('%Y-%m-%d %H:%M') }}\n{% for m in messages %}{{ m.content }}{% endfor %}"
        #expect(HeylookJinjaRenderer.checkPrefixStability(template: dateTemplate).stable == false)

        let qwenStripPrior = """
        {%- set ns = namespace(last_query_index = messages|length - 1) -%}
        {%- for m in messages -%}
        {%- if m.role == 'user' -%}{%- set ns.last_query_index = loop.index0 -%}{%- endif -%}
        {%- endfor -%}
        {%- for m in messages -%}
        {%- if m.role == 'assistant' and loop.index0 == messages|length - 1 and m.reasoning_content -%}
        <think>{{ m.reasoning_content }}</think>{{ m.content }}
        {%- else -%}
        <{{ m.role }}>{{ m.content }}</{{ m.role }}>
        {%- endif -%}
        {%- endfor -%}
        """
        #expect(HeylookJinjaRenderer.checkPrefixStability(template: qwenStripPrior).stable == false)
        #expect(HeylookJinjaRenderer.checkPrefixStability(template: HeylookJinjaRenderer.defaultChatMLTemplate).stable == true)

        // 4-shape refusal probe catches system-rejection and non-alternating role errors cleanly
        let strictTemplate = """
        {%- for m in messages -%}
        {%- if m.role == 'system' -%}{{ raise_exception('System role not supported') }}{%- endif -%}
        <{{ m.role }}>{{ m.content }}</{{ m.role }}>
        {%- endfor -%}
        """
        let refused = HeylookJinjaRenderer.probeRefusedShapes(template: strictTemplate)
        #expect(refused.contains(where: { $0.hasPrefix("system+user:") }))
        #expect(refused.contains(where: { $0.hasPrefix("system+system+user:") }))
        #expect(!refused.contains(where: { $0.hasPrefix("user+assistant:") }))
        #expect(HeylookJinjaRenderer.probeRefusedShapes(template: HeylookJinjaRenderer.defaultChatMLTemplate).isEmpty)
    }

    @Test("Render-probed ThinkingControls inspects values, aliases, default, unknown, changes_prefix, and off without model-ID heuristics")
    func testRenderProbedThinkingControlsWithoutModelNameHeuristics() throws {
        // Synthetic generic name proves behavior derives strictly from template rendering, never model ID substrings.
        let depthTemplate = """
        {%- if reasoning_effort is not defined -%}
        {%- set _eff = 'medium' -%}
        {%- elif reasoning_effort == 'none' -%}
        {%- set _eff = 'off' -%}
        {%- elif reasoning_effort == 'minimal' -%}
        {%- set _eff = 'low' -%}
        {%- elif reasoning_effort in ['low', 'medium', 'high'] -%}
        {%- set _eff = reasoning_effort -%}
        {%- else -%}
        {%- set _eff = 'fallback_default' -%}
        {%- endif -%}
        {%- for m in messages -%}
        {%- if loop.first and m.role == 'system' -%}
        <|sys|>effort={{ _eff }}|{{ m.content }}<|end|>
        {%- else -%}
        <|{{ m.role }}|>{{ m.content }}<|end|>
        {%- endif -%}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
        <|assistant|>{%- if _eff != 'off' -%}<think>{%- endif -%}
        {%- endif -%}
        """

        let info = ModelTemplateInfo.analyze(
            templateText: depthTemplate,
            specialTokens: ["<think>", "</think>", "<|end|>"],
            templateSource: "chat_template.jinja",
            modelID: "custom-model-alpha"
        )
        let ctrl = info.thinkingControls
        #expect(ctrl.supportsThinking == true)
        #expect(ctrl.depthVariable == "reasoning_effort")
        #expect(ctrl.depthDefault == "medium")
        #expect(ctrl.depthUnknown == "fallback")
        #expect(ctrl.depthChangesPrefix == true)
        #expect(ctrl.depthValues.contains("low"))
        #expect(ctrl.depthValues.contains("medium"))
        #expect(ctrl.depthValues.contains("high"))
        #expect(ctrl.depthAliases["minimal"] == "low")
        #expect(ctrl.depthOffValues.contains("none") || ctrl.depthValues.contains("none"))
        #expect(ctrl.canDisableThinking == true)

        #expect(ctrl.checkDepth("medium") == nil)
        #expect(ctrl.checkDepth("bogus-effort") != nil)

        // Model without any thinking markers/variables and named "qwen3-thinker-pro" must NOT hallucinate thinking via substring
        let plainChatML = "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}"
        let noThinkInfo = ModelTemplateInfo.analyze(
            templateText: plainChatML,
            specialTokens: [],
            modelID: "qwen3-thinker-pro"
        )
        #expect(noThinkInfo.thinkingControls.supportsThinking == false)
    }

    @Test("ChatTemplateState ladder, HF SHA-1 modifiedSinceDownload, unloaded stale=null, and PUT/DELETE HTTP contract")
    func testChatTemplateLadderSHA1StalenessAndHTTPRoutes() async throws {
        let harness = try TestRouterHarness()
        let modelDir = harness.sandbox.modelsDir.appendingPathComponent("tpl-model", isDirectory: true)
        let hfMetaDir = modelDir
            .appendingPathComponent(".cache", isDirectory: true)
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("download", isDirectory: true)
        try FileManager.default.createDirectory(at: hfMetaDir, withIntermediateDirectories: true)

        let vendorTpl = """
        {%- for m in messages -%}
        <|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n
        {%- endfor -%}
        {%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}
        """
        let vendorURL = modelDir.appendingPathComponent("chat_template.jinja")
        let tplBytes = Data(vendorTpl.utf8)
        try tplBytes.write(to: vendorURL)

        var gitBlob = Data("blob \(tplBytes.count)\0".utf8)
        gitBlob.append(tplBytes)
        let sha1Hex = Insecure.SHA1.hash(data: gitBlob).map { String(format: "%02x", $0) }.joined()
        try "0123456789abcdef\n\(sha1Hex)\n1700000000.0\n".write(
            to: hfMetaDir.appendingPathComponent("chat_template.jinja.metadata"),
            atomically: true,
            encoding: .utf8
        )
        let cleanState = HeylookConfig.inspectChatTemplate(in: modelDir, fallbackSource: "chat_template.jinja")
        #expect(cleanState.modifiedSinceDownload == false)
        #expect(cleanState.observedStale == nil)
        #expect(cleanState.origin == "vendor_sidecar")

        // Edit file on disk -> HF git-blob SHA-1 mismatch flips modifiedSinceDownload to true
        try (vendorTpl + "\n{# modified #}").write(to: vendorURL, atomically: true, encoding: .utf8)
        let dirtyState = HeylookConfig.inspectChatTemplate(in: modelDir, fallbackSource: "chat_template.jinja")
        #expect(dirtyState.modifiedSinceDownload == true)

        // Register model UNLOADED -> GET /chat-template returns stale: null and sources ladder
        _ = await harness.registerStub(id: "tpl-model", provider: .mlx, modelPath: modelDir.path, isLoaded: false)
        let getUnloaded = await harness.send("GET", "/v1/admin/models/tpl-model/chat-template")
        #expect(getUnloaded.status == 200)
        let unloadedJSON = try TestFixtures.decodeJSONObject(getUnloaded.body)
        #expect(unloadedJSON["stale"] is NSNull)
        #expect(unloadedJSON["override_present"] as? Bool == false)

        // DELETE when no override exists -> 404
        let delMissing = await harness.send("DELETE", "/v1/admin/models/tpl-model/chat-template")
        #expect(delMissing.status == 404)

        // PUT bad jinja -> 400 and leaves zero chat_template.heylook.jinja override file on disk
        let putBad = try await harness.sendJSON(
            "PUT",
            "/v1/admin/models/tpl-model/chat-template",
            object: ["template": "{% if broken "]
        )
        #expect(putBad.status == 400)
        let overrideURL = modelDir.appendingPathComponent("chat_template.heylook.jinja")
        #expect(!FileManager.default.fileExists(atPath: overrideURL.path))

        // Load model, then PUT valid override -> stale flips true, DELETE removes override cleanly (200)
        _ = await harness.send("POST", "/v1/models/tpl-model/load")
        let validOverride = """
        {%- for m in messages -%}[{{ m.role }}]{{ m.content }}{%- endfor -%}
        {%- if add_generation_prompt -%}[assistant]{%- endif -%}
        """
        let putValid = try await harness.sendJSON(
            "PUT",
            "/v1/admin/models/tpl-model/chat-template",
            object: ["template": validOverride]
        )
        #expect(putValid.status == 200)
        let putJSON = try TestFixtures.decodeJSONObject(putValid.body)
        #expect(putJSON["override_present"] as? Bool == true)
        #expect(putJSON["stale"] as? Bool == true)
        let sources = try #require(putJSON["sources"] as? [[String: Any]])
        #expect(sources.count >= 2)
        #expect(sources.first?["source"] as? String == "heylook override")
        #expect(FileManager.default.fileExists(atPath: overrideURL.path))

        let delOK = await harness.send("DELETE", "/v1/admin/models/tpl-model/chat-template")
        #expect(delOK.status == 200)
        #expect(!FileManager.default.fileExists(atPath: overrideURL.path))
    }

    @Test("TOMLCodec preserves comments, handles arrays containing commas, edits [scan] & [settings], and deletes empty sidecars")
    func testTOMLCodecCommentsCommasSectionsAndSidecarDeletion() throws {
        let initialTOML = """
        # Operator workbench configuration — preserve this header!
        [server]
        host = "127.0.0.1" # local bind
        port = 8000

        [settings]
        max_loaded_models = 2

        [scan]
        # Model roots to watch
        folders = ["models/a,b", "models/c"]
        auto_discover = true

        [[models]]
        id = "alpha-mlx"
        provider = "mlx"

        [models.config]
        model_path = "models/alpha"
        temperature = 0.7
        tags = ["fast,small", "chat"]
        """

        let parsed = HeylookConfig.parseTOML(initialTOML)
        #expect(parsed.serverHost == "127.0.0.1")
        #expect(parsed.serverPort == 8000)
        #expect(parsed.maxLoadedModels == 2)
        #expect(parsed.scanFolders == ["models/a,b", "models/c"])
        #expect(parsed.explicitModels.count == 1)
        #expect(parsed.explicitModels[0].id == "alpha-mlx")
        #expect(parsed.explicitModels[0].sidecar.tags == ["fast,small", "chat"])

        // Edit [scan] folders containing commas + [settings] and assert existing comments & [[models]] survive intact
        let updatedScan = TOMLCodec.updateScanConfig(in: initialTOML, folders: ["models/x,y", "models/z"])
        let updatedAll = TOMLCodec.updateSettings(in: updatedScan, updates: ["max_loaded_models": .int(4)])
        #expect(updatedAll.contains("# Operator workbench configuration — preserve this header!"))
        #expect(updatedAll.contains("# Model roots to watch"))
        #expect(updatedAll.contains("# local bind"))
        #expect(updatedAll.contains("id = \"alpha-mlx\""))
        let reparsed = HeylookConfig.parseTOML(updatedAll)
        #expect(reparsed.scanFolders == ["models/x,y", "models/z"])
        #expect(reparsed.maxLoadedModels == 4)
        #expect(reparsed.explicitModels[0].sidecar.tags == ["fast,small", "chat"])

        // Sidecar patch deleting final explicit key deletes `model.heylook.toml` from disk
        let sandbox = try TestTempSandbox(prefix: "heylook_toml")
        defer { sandbox.cleanup() }
        let sidecarURL = sandbox.rootURL.appendingPathComponent("model.heylook.toml")
        #expect(try TOMLCodec.applyModelSidecarPatch(at: sidecarURL, updates: ["ctx_size": .int(16384)]) == true)
        #expect(FileManager.default.fileExists(atPath: sidecarURL.path))
        #expect(try TOMLCodec.applyModelSidecarPatch(at: sidecarURL, updates: ["ctx_size": .null]) == false)
        #expect(!FileManager.default.fileExists(atPath: sidecarURL.path))
    }

    @Test("GGUFHeaderReader streams past large vocab arrays, detects SWA sliding_window, and pairs -assistant drafters as draft-mtp")
    func testGGUFHeaderReaderStreamingSWABoolFloatAndAssistantDrafter() throws {
        let sandbox = try TestTempSandbox(prefix: "heylook_gguf")
        defer { sandbox.cleanup() }

        // 1. Streaming GGUF v3 header past 10k-entry vocab array + BOOL/FLOAT32/SWA metadata
        let largeVocab = (0..<10_000).map { "vocab_\(String(format: "%05d", $0))" }
        let swaGGUF = GGUFHeaderReader.buildMinimalGGUF(
            version: 3,
            tensorCount: 64,
            stringMetadata: [
                "general.architecture": "gemma4",
                "general.name": "Gemma4-SWA",
            ],
            stringArrayMetadata: ["a.tokenizer.ggml.tokens": largeVocab],
            uint32Metadata: [
                "gemma4.context_length": 65536,
                "gemma4.attention.sliding_window": 1024,
            ]
        )
        let swaFile = sandbox.modelsDir.appendingPathComponent("Gemma4-SWA-Q4_K_M.gguf")
        try swaGGUF.write(to: swaFile)
        let header = try #require(GGUFHeaderReader.readFileHeader(at: swaFile))
        #expect(header.architecture == "gemma4")
        #expect(header.contextLength == 65536)
        #expect(header.hasSlidingWindowAttention == true)
        #expect(header.memoryKind == "sliding_window")

        // 2. Sibling `-assistant` drafter directory with `model_type: "gemma4_assistant"` pairs as `"draft-mtp"`
        let targetDir = sandbox.modelsDir.appendingPathComponent("gemma-4-e2b-it-4bit", isDirectory: true)
        let assistantDir = sandbox.modelsDir.appendingPathComponent("gemma-4-e2b-it-assistant", isDirectory: true)
        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: assistantDir, withIntermediateDirectories: true)

        try Data("{}".utf8).write(to: targetDir.appendingPathComponent("model.safetensors"))
        try #"{"model_type":"gemma4","hidden_size":2048,"vocab_size":256000}"#.write(
            to: targetDir.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )
        try Data("{}".utf8).write(to: assistantDir.appendingPathComponent("model.safetensors"))
        try #"{"model_type":"gemma4_assistant","backbone_hidden_size":2048,"vocab_size":256000}"#.write(
            to: assistantDir.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )

        let cfg = HeylookConfig(scanFolders: [sandbox.modelsDir.path])
        let discovered = cfg.discoverConfiguredModels()
        let targetDesc = try #require(discovered.first(where: { $0.modelID == "gemma-4-e2b-it-4bit" }))
        #expect(targetDesc.speculativeDrafter.value?.hasSuffix("gemma-4-e2b-it-assistant") == true)
        // Auxiliary assistant folder must NOT clutter primary chat model list
        #expect(!discovered.contains(where: { $0.modelID == "gemma-4-e2b-it-assistant" }))

        let drafterPair = HeylookConfig.discoverDrafter(in: targetDir, provider: .mlx, sidecar: nil)
        #expect(drafterPair.specType == "draft-mtp")
    }
}
