import Foundation

/// Per-folder `model.heylook.toml` sidecar configuration.
/// Compatible with Python `heylookitsanllm`'s override-only model discovery and extended with
/// Apple Silicon compute placement (`compute_unit`, per-tower ANE overrides, `pinned`).
public struct ModelSidecarConfig: Codable, Sendable, Equatable {
    public var id: String?
    public var provider: EngineProviderKind?
    public var pinned: Bool?
    public var computePlacement: ComputePlacement?
    public var visionComputePlacement: ComputePlacement?
    public var audioComputePlacement: ComputePlacement?
    public var textComputePlacement: ComputePlacement?
    public var ctxSize: Int?
    public var flashAttn: Bool?
    public var supportsVision: Bool?
    public var supportsThinking: Bool?
    public var draftModel: String?
    public var specType: String?
    public var chatTemplatePath: String?
    public var useSidecarChatTemplate: Bool?
    public var unsetKeys: [String]
    public var sampling: SamplerSettings
    public var tags: [String]

    public init(
        id: String? = nil,
        provider: EngineProviderKind? = nil,
        pinned: Bool? = nil,
        computePlacement: ComputePlacement? = nil,
        visionComputePlacement: ComputePlacement? = nil,
        audioComputePlacement: ComputePlacement? = nil,
        textComputePlacement: ComputePlacement? = nil,
        ctxSize: Int? = nil,
        flashAttn: Bool? = nil,
        supportsVision: Bool? = nil,
        supportsThinking: Bool? = nil,
        draftModel: String? = nil,
        specType: String? = nil,
        chatTemplatePath: String? = nil,
        useSidecarChatTemplate: Bool? = nil,
        unsetKeys: [String] = [],
        sampling: SamplerSettings = SamplerSettings(),
        tags: [String] = []
    ) {
        self.id = id
        self.provider = provider
        self.pinned = pinned
        self.computePlacement = computePlacement
        self.visionComputePlacement = visionComputePlacement
        self.audioComputePlacement = audioComputePlacement
        self.textComputePlacement = textComputePlacement
        self.ctxSize = ctxSize
        self.flashAttn = flashAttn
        self.supportsVision = supportsVision
        self.supportsThinking = supportsThinking
        self.draftModel = draftModel
        self.specType = specType
        self.chatTemplatePath = chatTemplatePath
        self.useSidecarChatTemplate = useSidecarChatTemplate
        self.unsetKeys = unsetKeys
        self.sampling = sampling
        self.tags = tags
    }

    /// Parses a `model.heylook.toml` file content into a `ModelSidecarConfig`.
    public static func parseTOML(_ text: String) -> ModelSidecarConfig {
        let kv = SimpleTOML.parseFlatKeyValues(text)
        return fromKeyValues(kv)
    }

    static func fromKeyValues(_ kv: [String: SimpleTOML.Value]) -> ModelSidecarConfig {
        var config = ModelSidecarConfig()
        config.id = kv["id"]?.stringValue
        if let p = kv["provider"]?.stringValue.lowercased() {
            config.provider = EngineProviderKind(rawValue: p)
        }
        config.pinned = kv["pinned"]?.boolValue
        if let cu = kv["compute_unit"]?.stringValue {
            config.computePlacement = ComputePlacement(rawValue: cu)
        }
        if let vcu = kv["vision_compute_unit"]?.stringValue {
            config.visionComputePlacement = ComputePlacement(rawValue: vcu)
        }
        if let acu = kv["audio_compute_unit"]?.stringValue {
            config.audioComputePlacement = ComputePlacement(rawValue: acu)
        }
        if let tcu = kv["text_compute_unit"]?.stringValue {
            config.textComputePlacement = ComputePlacement(rawValue: tcu)
        }
        config.ctxSize = kv["ctx_size"]?.intValue ?? kv["cache_size"]?.intValue
        config.flashAttn = kv["flash_attn"]?.boolValue
        config.supportsVision = kv["vision"]?.boolValue
        config.supportsThinking = kv["thinking_capable"]?.boolValue ?? kv["supports_thinking"]?.boolValue
        config.draftModel = kv["draft_model"]?.stringValue ?? kv["draft_model_path"]?.stringValue
        config.specType = kv["spec_type"]?.stringValue
        config.chatTemplatePath = kv["chat_template_path"]?.stringValue
        config.useSidecarChatTemplate = kv["use_sidecar_chat_template"]?.boolValue
        config.unsetKeys = kv["unset"]?.stringArrayValue ?? []
        config.tags = kv["tags"]?.stringArrayValue ?? []

        config.sampling = SamplerSettings(
            temperature: kv["temperature"]?.doubleValue,
            topP: kv["top_p"]?.doubleValue,
            topK: kv["top_k"]?.intValue,
            minP: kv["min_p"]?.doubleValue,
            repetitionPenalty: kv["repetition_penalty"]?.doubleValue,
            repetitionContextSize: kv["repetition_context_size"]?.intValue,
            presencePenalty: kv["presence_penalty"]?.doubleValue,
            maxTokens: kv["max_tokens"]?.intValue,
            seed: kv["seed"]?.intValue,
            thinking: kv["thinking"]?.boolValue ?? kv["enable_thinking"]?.boolValue,
            thinkingDepth: kv["thinking_depth"]?.stringValue,
            thinkingBudget: kv["thinking_budget"]?.intValue
        )
        return config
    }

    /// Serializes this sidecar into clean `model.heylook.toml` format.
    public func toTOML() -> String {
        var lines: [String] = ["# model.heylook.toml — Per-model override sidecar"]
        if let id { lines.append("id = \"\(id)\"") }
        if let provider { lines.append("provider = \"\(provider.rawValue)\"") }
        if let pinned { lines.append("pinned = \(pinned)") }
        if let computePlacement { lines.append("compute_unit = \"\(computePlacement.rawValue)\"") }
        if let visionComputePlacement { lines.append("vision_compute_unit = \"\(visionComputePlacement.rawValue)\"") }
        if let audioComputePlacement { lines.append("audio_compute_unit = \"\(audioComputePlacement.rawValue)\"") }
        if let textComputePlacement { lines.append("text_compute_unit = \"\(textComputePlacement.rawValue)\"") }
        if let ctxSize { lines.append("ctx_size = \(ctxSize)") }
        if let flashAttn { lines.append("flash_attn = \(flashAttn)") }
        if let supportsVision { lines.append("vision = \(supportsVision)") }
        if let supportsThinking { lines.append("thinking_capable = \(supportsThinking)") }
        if let draftModel { lines.append("draft_model = \"\(draftModel)\"") }
        if let specType { lines.append("spec_type = \"\(specType)\"") }
        if let chatTemplatePath { lines.append("chat_template_path = \"\(chatTemplatePath)\"") }
        if let useSidecarChatTemplate { lines.append("use_sidecar_chat_template = \(useSidecarChatTemplate)") }
        if !unsetKeys.isEmpty {
            let quoted = unsetKeys.map { "\"\($0)\"" }.joined(separator: ", ")
            lines.append("unset = [\(quoted)]")
        }
        if !tags.isEmpty {
            let quoted = tags.map { "\"\($0)\"" }.joined(separator: ", ")
            lines.append("tags = [\(quoted)]")
        }
        if let v = sampling.temperature { lines.append("temperature = \(v)") }
        if let v = sampling.topP { lines.append("top_p = \(v)") }
        if let v = sampling.topK { lines.append("top_k = \(v)") }
        if let v = sampling.minP { lines.append("min_p = \(v)") }
        if let v = sampling.repetitionPenalty { lines.append("repetition_penalty = \(v)") }
        if let v = sampling.repetitionContextSize { lines.append("repetition_context_size = \(v)") }
        if let v = sampling.presencePenalty { lines.append("presence_penalty = \(v)") }
        if let v = sampling.maxTokens { lines.append("max_tokens = \(v)") }
        if let v = sampling.thinking { lines.append("thinking = \(v)") }
        if let v = sampling.thinkingDepth { lines.append("thinking_depth = \"\(v)\"") }
        if let v = sampling.thinkingBudget { lines.append("thinking_budget = \(v)") }
        return lines.joined(separator: "\n") + "\n"
    }
}

/// Root `heylook.toml` server & workbench configuration.
/// Supports both flat top-level keys and Python `heylookitsanllm`'s `[scan] folders = [...]`
/// and `[[models]]` table sections.
public struct HeylookConfig: Codable, Sendable, Equatable {
    public struct ExplicitModelEntry: Codable, Sendable, Equatable {
        public let id: String
        public let path: String
        public let provider: EngineProviderKind
        public let sidecar: ModelSidecarConfig

        public init(
            id: String,
            path: String,
            provider: EngineProviderKind = .mlx,
            sidecar: ModelSidecarConfig = ModelSidecarConfig()
        ) {
            self.id = id
            self.path = path
            self.provider = provider
            self.sidecar = sidecar
        }
    }

    public var defaultModel: String?
    public var maxLoadedModels: Int
    public var maxQueueDepth: Int
    public var scanFolders: [String]
    public var allowedHosts: [String]
    public var serverHost: String
    public var serverPort: Int
    public var adminToken: String?
    public var explicitModels: [ExplicitModelEntry]

    public init(
        defaultModel: String? = nil,
        maxLoadedModels: Int = 3,
        maxQueueDepth: Int = 2,
        scanFolders: [String] = [],
        allowedHosts: [String] = [],
        serverHost: String = "127.0.0.1",
        serverPort: Int = 8000,
        adminToken: String? = nil,
        explicitModels: [ExplicitModelEntry] = []
    ) {
        self.defaultModel = defaultModel
        self.maxLoadedModels = max(1, maxLoadedModels)
        self.maxQueueDepth = max(1, maxQueueDepth)
        self.scanFolders = scanFolders
        self.allowedHosts = allowedHosts
        self.serverHost = serverHost
        self.serverPort = serverPort
        self.adminToken = adminToken
        self.explicitModels = explicitModels
    }

    /// Loads `heylook.toml` from `HEYLOOK_CONFIG`, `./heylook.toml`,
    /// or `~/.heylook/heylook.toml`, merging `HEYLOOK_MODELS_DIR` if present.
    public static func loadDefault() -> HeylookConfig {
        let env = ProcessInfo.processInfo.environment
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let home = fm.homeDirectoryForCurrentUser

        var candidates: [URL] = []
        if let custom = env["HEYLOOK_CONFIG"], !custom.isEmpty {
            candidates.append(URL(fileURLWithPath: (custom as NSString).expandingTildeInPath))
        }
        candidates.append(cwd.appendingPathComponent("heylook.toml"))
        candidates.append(home.appendingPathComponent(".heylook/heylook.toml"))

        var config = HeylookConfig()
        for url in candidates {
            if let text = try? String(contentsOf: url, encoding: .utf8) {
                config = parseTOML(text)
                break
            }
        }
        if let modelsDir = env["HEYLOOK_MODELS_DIR"], !modelsDir.isEmpty, !config.scanFolders.contains(modelsDir) {
            config.scanFolders.append(modelsDir)
        }
        return config
    }

    public static func parseTOML(_ text: String) -> HeylookConfig {
        let doc = SimpleTOML.parseDocument(text)
        let root = doc.root
        let scanSec = doc.sections["scan"] ?? [:]
        let serverSec = doc.sections["server"] ?? [:]

        let folders = root["scan_folders"]?.stringArrayValue
            ?? scanSec["folders"]?.stringArrayValue
            ?? root["folders"]?.stringArrayValue
            ?? []

        var explicit: [ExplicitModelEntry] = []
        if let modelTables = doc.arrayOfTables["models"] {
            for table in modelTables {
                let rawPath = table["model_path"]?.stringValue ?? table["path"]?.stringValue ?? ""
                guard !rawPath.isEmpty else { continue }
                let id = table["id"]?.stringValue ?? URL(fileURLWithPath: rawPath).lastPathComponent
                let provRaw = table["provider"]?.stringValue.lowercased() ?? ""
                let prov = EngineProviderKind(rawValue: provRaw)
                    ?? (rawPath.lowercased().hasSuffix(".gguf") ? .gguf : .mlx)
                let sidecar = ModelSidecarConfig.fromKeyValues(table)
                explicit.append(ExplicitModelEntry(id: id, path: rawPath, provider: prov, sidecar: sidecar))
            }
        }

        return HeylookConfig(
            defaultModel: root["default_model"]?.stringValue,
            maxLoadedModels: root["max_loaded_models"]?.intValue ?? 3,
            maxQueueDepth: root["max_queue_depth"]?.intValue ?? 2,
            scanFolders: folders,
            allowedHosts: root["allowed_hosts"]?.stringArrayValue ?? [],
            serverHost: root["host"]?.stringValue ?? serverSec["host"]?.stringValue ?? "127.0.0.1",
            serverPort: root["port"]?.intValue ?? serverSec["port"]?.intValue ?? 8000,
            adminToken: root["admin_token"]?.stringValue ?? serverSec["admin_token"]?.stringValue,
            explicitModels: explicit
        )
    }

    public func toTOML() -> String {
        var lines: [String] = ["# heylook.toml — Global server & model discovery configuration"]
        if let defaultModel { lines.append("default_model = \"\(defaultModel)\"") }
        lines.append("max_loaded_models = \(maxLoadedModels)")
        lines.append("max_queue_depth = \(maxQueueDepth)")
        lines.append("host = \"\(serverHost)\"")
        lines.append("port = \(serverPort)")
        if !scanFolders.isEmpty {
            let quoted = scanFolders.map { "\"\($0)\"" }.joined(separator: ", ")
            lines.append("scan_folders = [\(quoted)]")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Resolves the 4-rung GGUF chat template ladder (`gguf.md`):
    /// 1. Explicit `chat_template_path` (`configured`)
    /// 2. Operator override `chat_template.heylook.jinja` in the model dir (`configured`)
    /// 3. Sidecar `chat_template.jinja` in the model dir unless `use_sidecar_chat_template = false` (`derived`)
    /// 4. Embedded GGUF header template (`"gguf_embedded"`, `derived`).
    public static func resolveGGUFTemplate(
        in candidateDir: URL,
        sidecar: ModelSidecarConfig?
    ) -> (source: ProvenanceField<String>, resolvedFilePath: String?) {
        let fm = FileManager.default
        if let explicitPath = sidecar?.chatTemplatePath, !explicitPath.isEmpty {
            let url = explicitPath.hasPrefix("/")
                ? URL(fileURLWithPath: explicitPath)
                : candidateDir.appendingPathComponent(explicitPath)
            return (
                ProvenanceField(value: url.lastPathComponent, provenance: .configured, effect: .reload),
                url.standardizedFileURL.path
            )
        }
        let operatorOverride = candidateDir.appendingPathComponent("chat_template.heylook.jinja")
        if fm.fileExists(atPath: operatorOverride.path) {
            return (
                ProvenanceField(value: "chat_template.heylook.jinja", provenance: .configured, effect: .reload),
                operatorOverride.standardizedFileURL.path
            )
        }
        let allowSidecar = sidecar?.useSidecarChatTemplate ?? true
        let sidecarJinja = candidateDir.appendingPathComponent("chat_template.jinja")
        if allowSidecar && fm.fileExists(atPath: sidecarJinja.path) {
            return (
                ProvenanceField(value: "chat_template.jinja", provenance: .derived, effect: .reload),
                sidecarJinja.standardizedFileURL.path
            )
        }
        return (
            ProvenanceField(value: "gguf_embedded", provenance: .derived, effect: .reload),
            nil
        )
    }

    /// Discovers a paired speculative drafter or built-in MTP head for a model directory (`gguf.md` & `mlx.md`).
    /// Respects `unset = ["draft_model_path", "spec_type"]` in `model.heylook.toml`.
    public static func discoverDrafter(
        in candidateDir: URL,
        provider: EngineProviderKind,
        sidecar: ModelSidecarConfig?
    ) -> (draftModelPath: String?, specType: String?) {
        let unset = Set(sidecar?.unsetKeys ?? [])
        if unset.contains("draft_model_path") || unset.contains("draft_model") || unset.contains("spec_type") {
            return (nil, nil)
        }
        if let explicitDraft = sidecar?.draftModel, !explicitDraft.isEmpty {
            return (explicitDraft, sidecar?.specType)
        }
        if let explicitSpec = sidecar?.specType, !explicitSpec.isEmpty {
            return (nil, explicitSpec)
        }

        let fm = FileManager.default
        let drafterPrefixes = ["mtp-", "dflash-", "dspark-", "eagle3-"]

        func findDrafterFile(in folder: URL) -> String? {
            guard let items = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil) else {
                return nil
            }
            for item in items.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                let lower = item.lastPathComponent.lowercased()
                if lower.hasSuffix(".gguf") && drafterPrefixes.contains(where: { lower.hasPrefix($0) }) {
                    return item.standardizedFileURL.path
                }
            }
            return nil
        }

        if provider == .gguf {
            if let rootDrafter = findDrafterFile(in: candidateDir) {
                return (rootDrafter, nil)
            }
            for subName in ["MTP", "dspark", "dflash", "eagle3"] {
                let subDir = candidateDir.appendingPathComponent(subName, isDirectory: true)
                if let subDrafter = findDrafterFile(in: subDir) {
                    return (subDrafter, nil)
                }
            }
            // Also check parent directory if candidateDir is a per-quant variant subfolder
            let parentDir = candidateDir.deletingLastPathComponent()
            if let parentDrafter = findDrafterFile(in: parentDir) {
                return (parentDrafter, nil)
            }
            for subName in ["MTP", "dspark", "dflash", "eagle3"] {
                let parentSub = parentDir.appendingPathComponent(subName, isDirectory: true)
                if let subDrafter = findDrafterFile(in: parentSub) {
                    return (subDrafter, nil)
                }
            }
        } else if provider == .mlx {
            let indexURL = candidateDir.appendingPathComponent("model.safetensors.index.json")
            if let data = try? Data(contentsOf: indexURL),
               let text = String(data: data, encoding: .utf8),
               text.contains("\"mtp.") || text.contains(".mtp.") {
                return (candidateDir.standardizedFileURL.path, "draft-mtp")
            }
            // Check for nested or sibling *-mtp / *-MTP-* drafter folder
            if let children = try? fm.contentsOfDirectory(at: candidateDir, includingPropertiesForKeys: [.isDirectoryKey]) {
                if let nestedMTP = children.first(where: { $0.lastPathComponent.lowercased().contains("-mtp") }) {
                    return (nestedMTP.standardizedFileURL.path, "draft-mtp")
                }
            }
            let parentDir = candidateDir.deletingLastPathComponent()
            let baseLower = candidateDir.lastPathComponent.lowercased()
            if let siblings = try? fm.contentsOfDirectory(at: parentDir, includingPropertiesForKeys: [.isDirectoryKey]) {
                for sibling in siblings.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                    let sLower = sibling.lastPathComponent.lowercased()
                    guard sLower != baseLower else { continue }
                    let stem: String?
                    if let range = sLower.range(of: "-mtp") {
                        stem = String(sLower[..<range.lowerBound])
                    } else if sLower.hasSuffix("-assistant") {
                        stem = String(sLower.dropLast("-assistant".count))
                    } else {
                        stem = nil
                    }
                    if let stem, !stem.isEmpty, baseLower.hasPrefix(stem) {
                        let spec = sLower.hasSuffix("-assistant") ? "draft-model" : "draft-mtp"
                        return (sibling.standardizedFileURL.path, spec)
                    }
                }
            }
        }
        return (nil, nil)
    }

    /// Loads vendor sampling defaults (`temperature`, `top_p`, `top_k`) from `generation_config.json`
    /// in a model directory (`engine-contract.md` §Sampling).
    public static func loadVendorSampling(in candidateDir: URL) -> SamplerSettings? {
        guard let json = loadJSONDictionary(at: candidateDir.appendingPathComponent("generation_config.json")) else {
            return nil
        }
        let temp = (json["temperature"] as? NSNumber)?.doubleValue
        let topP = (json["top_p"] as? NSNumber)?.doubleValue
        let topK = (json["top_k"] as? NSNumber)?.intValue
        guard temp != nil || topP != nil || topK != nil else {
            return nil
        }
        return SamplerSettings(temperature: temp, topP: topP, topK: topK)
    }

    /// Derives the per-image vision soft-token budget from `processor_config.json` or `config.json`
    /// (`vision_config`), falling back to `280` when unspecified.
    public static func estimateImageTokenBudget(in candidateDir: URL) -> Int {
        if let procJSON = loadJSONDictionary(at: candidateDir.appendingPathComponent("processor_config.json")) {
            if let ip = procJSON["image_processor"] as? [String: Any],
               let maxSoft = (ip["max_soft_tokens"] as? NSNumber)?.intValue,
               maxSoft > 0 {
                return maxSoft
            }
            if let maxPixels = (procJSON["max_pixels"] as? NSNumber)?.intValue,
               let patchSize = (procJSON["patch_size"] as? NSNumber)?.intValue,
               patchSize > 0 {
                let mergeSize = max(1, (procJSON["merge_size"] as? NSNumber)?.intValue ?? 2)
                let denom = patchSize * patchSize * mergeSize * mergeSize
                if denom > 0 {
                    return max(1, maxPixels / denom)
                }
            }
        }
        if let configJSON = loadJSONDictionary(at: candidateDir.appendingPathComponent("config.json")),
           let visionCfg = configJSON["vision_config"] as? [String: Any] {
            if let defaultLen = (visionCfg["default_output_length"] as? NSNumber)?.intValue,
               defaultLen > 0 {
                return defaultLen
            }
            if let imageSize = (visionCfg["image_size"] as? NSNumber)?.intValue,
               let patchSize = (visionCfg["patch_size"] as? NSNumber)?.intValue,
               patchSize > 0 {
                let mergeSize = max(
                    1,
                    (visionCfg["spatial_merge_size"] as? NSNumber)?.intValue
                        ?? (visionCfg["pooling_kernel_size"] as? NSNumber)?.intValue
                        ?? (configJSON["mm_tokens_per_image"] as? NSNumber).map { _ in 1 }
                        ?? 1
                )
                if let mmTokens = (configJSON["mm_tokens_per_image"] as? NSNumber)?.intValue, mmTokens > 0 {
                    return mmTokens
                }
                let grid = imageSize / patchSize
                return max(1, (grid * grid) / (mergeSize * mergeSize))
            }
        }
        return 280
    }

    public struct ChatTemplateState: Codable, Sendable, Equatable {
        public let source: String
        public let hasOverride: Bool
        public let template: String

        public init(source: String, hasOverride: Bool, template: String) {
            self.source = source
            self.hasOverride = hasOverride
            self.template = template
        }
    }

    /// Inspects the active chat template in `candidateDir` across the 4-rung ladder:
    /// `chat_template.heylook.jinja` (operator override) -> `chat_template.jinja` (vendor ship) ->
    /// `tokenizer_config.json` -> `.gguf` header `tokenizer.chat_template` (`gguf_embedded`).
    public static func inspectChatTemplate(
        in candidateDir: URL,
        fallbackSource: String = "default_chatml"
    ) -> ChatTemplateState {
        let fm = FileManager.default
        let overrideURL = candidateDir.appendingPathComponent("chat_template.heylook.jinja")
        let sidecarURL = candidateDir.appendingPathComponent("chat_template.jinja")
        let tokConfigURL = candidateDir.appendingPathComponent("tokenizer_config.json")

        if fm.fileExists(atPath: overrideURL.path),
           let t = try? String(contentsOf: overrideURL, encoding: .utf8) {
            return ChatTemplateState(
                source: "chat_template.heylook.jinja",
                hasOverride: true,
                template: t
            )
        }
        if fm.fileExists(atPath: sidecarURL.path),
           let t = try? String(contentsOf: sidecarURL, encoding: .utf8) {
            return ChatTemplateState(
                source: "chat_template.jinja",
                hasOverride: false,
                template: t
            )
        }
        if let tokJSON = loadJSONDictionary(at: tokConfigURL),
           let t = tokJSON["chat_template"] as? String {
            return ChatTemplateState(
                source: "tokenizer_config.json",
                hasOverride: false,
                template: t
            )
        }
        if let ggufURL = firstGGUFFile(in: candidateDir),
           let ggufHeader = GGUFHeaderReader.readFileHeader(at: ggufURL),
           let embedded = ggufHeader.embeddedChatTemplate,
           !embedded.isEmpty {
            return ChatTemplateState(
                source: "gguf_embedded",
                hasOverride: false,
                template: embedded
            )
        }
        return ChatTemplateState(
            source: fallbackSource,
            hasOverride: false,
            template: ""
        )
    }

    /// Writes `chat_template.heylook.jinja` inside `candidateDir` without mutating vendor files.
    @discardableResult
    public static func writeChatTemplateOverride(
        _ template: String,
        in candidateDir: URL
    ) throws -> ChatTemplateState {
        try FileManager.default.createDirectory(at: candidateDir, withIntermediateDirectories: true)
        let overrideURL = candidateDir.appendingPathComponent("chat_template.heylook.jinja")
        try template.write(to: overrideURL, atomically: true, encoding: .utf8)
        return ChatTemplateState(
            source: "chat_template.heylook.jinja",
            hasOverride: true,
            template: template
        )
    }

    /// Removes `chat_template.heylook.jinja` from `candidateDir` and returns the reverted template state.
    @discardableResult
    public static func deleteChatTemplateOverride(
        in candidateDir: URL,
        fallbackSource: String = "gguf_embedded"
    ) -> ChatTemplateState {
        let overrideURL = candidateDir.appendingPathComponent("chat_template.heylook.jinja")
        if FileManager.default.fileExists(atPath: overrideURL.path) {
            try? FileManager.default.removeItem(at: overrideURL)
        }
        return inspectChatTemplate(in: candidateDir, fallbackSource: fallbackSource)
    }

    /// Renders a model-derived prompt preview by inspecting the model directory's active template
    /// (`chat_template.heylook.jinja`, `chat_template.jinja`, `tokenizer_config.json`, or `gguf_embedded`).
    public static func renderPromptPreview(
        in candidateDir: URL?,
        system: String?,
        messages: [(role: String, content: String, thinking: String?)],
        thinkingEnabled: Bool = false
    ) -> (prompt: String, source: String) {
        var source = "default_chatml"
        var templateText = ""
        var modelType = ""

        if let candidateDir {
            let state = inspectChatTemplate(in: candidateDir, fallbackSource: "default_chatml")
            templateText = state.template
            source = state.source
            let configURL = candidateDir.appendingPathComponent("config.json")
            if let cfgJSON = loadJSONDictionary(at: configURL),
               let mt = cfgJSON["model_type"] as? String {
                modelType = mt.lowercased()
            } else if let ggufURL = firstGGUFFile(in: candidateDir),
                      let ggufHeader = GGUFHeaderReader.readFileHeader(at: ggufURL),
                      let arch = ggufHeader.architecture {
                modelType = arch.lowercased()
            }
        }

        let usesGemmaTurnMarkers = templateText.contains("<start_of_turn>")
            || modelType.contains("gemma")

        var lines: [String] = []
        if usesGemmaTurnMarkers {
            if let sys = system, !sys.isEmpty {
                lines.append("<start_of_turn>system\n\(sys)<end_of_turn>")
            }
            for msg in messages {
                let gemmaRole = msg.role == "assistant" ? "model" : msg.role
                if let thinking = msg.thinking, !thinking.isEmpty {
                    lines.append("<start_of_turn>\(gemmaRole)\n<thought>\n\(thinking)\n</thought>\n\(msg.content)<end_of_turn>")
                } else {
                    lines.append("<start_of_turn>\(gemmaRole)\n\(msg.content)<end_of_turn>")
                }
            }
            lines.append("<start_of_turn>model")
            if thinkingEnabled {
                lines.append("<thought>")
            }
        } else {
            if let sys = system, !sys.isEmpty {
                lines.append("<|im_start|>system\n\(sys)<|im_end|>")
            }
            for msg in messages {
                if let thinking = msg.thinking, !thinking.isEmpty {
                    lines.append("<|im_start|>\(msg.role)\n<think>\n\(thinking)\n</think>\n\(msg.content)<|im_end|>")
                } else {
                    lines.append("<|im_start|>\(msg.role)\n\(msg.content)<|im_end|>")
                }
            }
            lines.append("<|im_start|>assistant")
            if thinkingEnabled {
                lines.append("<think>")
            }
        }
        return (lines.joined(separator: "\n"), source)
    }

    /// Discovers a `mmproj*.gguf` multimodal vision projector in `candidateDir` or its parent repo directory.
    public static func discoverMMProj(in candidateDir: URL) -> String? {
        let fm = FileManager.default
        for folder in [candidateDir, candidateDir.deletingLastPathComponent()] {
            guard let items = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil) else {
                continue
            }
            for item in items.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                let lower = item.lastPathComponent.lowercased()
                if lower.hasSuffix(".gguf") && lower.contains("mmproj") {
                    return item.standardizedFileURL.path
                }
            }
        }
        return nil
    }

    /// Strips `-00001-of-00005` shard suffix and `.gguf` extension from a GGUF filename (`model_importer.py`).
    public static func ggufModelStem(from filename: String) -> String {
        var name = filename
        if name.lowercased().hasSuffix(".gguf") {
            name = String(name.dropLast(5))
        }
        if let range = name.range(of: #"-\d{5}-of-\d{5}$"#, options: .regularExpression) {
            name.removeSubrange(range)
        }
        return name
    }

    /// True when `candidateDir` is a per-quant variant subfolder (e.g. `UD-Q6_K_XL/` or `Q8_0/`)
    /// whose name is a proper substring of the primary `.gguf` file stem (`model_importer.py`).
    public static func isGGUFVariantDirectory(_ candidateDir: URL, primaryGGUFFile: URL) -> Bool {
        let stem = ggufModelStem(from: primaryGGUFFile.lastPathComponent).lowercased()
        let dirName = candidateDir.lastPathComponent.lowercased()
        return dirName != stem && stem.contains(dirName)
    }

    /// Scans `scanFolders` for `.safetensors` (MLX), `.aimodel` / `.mlmodelc` (CoreAI / ANE),
    /// and `.gguf` (llama-server) model directories, merging any `model.heylook.toml` sidecar.
    public static func discoverModels(in folders: [URL]) -> [EngineDescription] {
        var discovered: [EngineDescription] = []
        let fm = FileManager.default
        let drafterSubfolderNames: Set<String> = ["mtp", "dspark", "dflash", "eagle3", "coreml", "ane"]

        for root in folders {
            let expandedRoot = URL(fileURLWithPath: (root.path as NSString).expandingTildeInPath).resolvingSymlinksInPath()
            guard fm.fileExists(atPath: expandedRoot.path) else { continue }
            guard let enumerator = fm.enumerator(
                at: expandedRoot,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else { continue }

            var visitedDirs = Set<String>()
            for case let url as URL in enumerator {
                let ext = url.pathExtension.lowercased()
                let candidateDir: URL
                let detectedProvider: EngineProviderKind
                let defaultPlacement: ComputePlacement
                var primaryGGUFURL: URL?

                if ext == "safetensors" {
                    candidateDir = url.deletingLastPathComponent()
                    detectedProvider = .mlx
                    defaultPlacement = .mlxGPU
                } else if ext == "aimodel" || ext == "mlmodelc" {
                    let parentDir = url.deletingLastPathComponent()
                    // Skip auxiliary ANE tower sub-bundles inside an MLX checkpoint's `coreml/` or `ane/` folder
                    if drafterSubfolderNames.contains(parentDir.lastPathComponent.lowercased()) {
                        continue
                    }
                    candidateDir = parentDir
                    detectedProvider = .coreai
                    defaultPlacement = .ane
                } else if ext == "gguf" {
                    let lowerName = url.lastPathComponent.lowercased()
                    if lowerName.contains("mmproj")
                        || lowerName.hasPrefix("mtp-")
                        || lowerName.hasPrefix("dflash-")
                        || lowerName.hasPrefix("dspark-")
                        || lowerName.hasPrefix("eagle3-")
                        || lowerName.contains("-dspark-") {
                        continue
                    }
                    let parentDir = url.deletingLastPathComponent()
                    if drafterSubfolderNames.contains(parentDir.lastPathComponent.lowercased()) {
                        continue
                    }
                    candidateDir = parentDir
                    detectedProvider = .gguf
                    defaultPlacement = .ggufSubprocess
                    primaryGGUFURL = url
                } else {
                    continue
                }

                let dirPath = candidateDir.standardizedFileURL.path
                guard !visitedDirs.contains(dirPath) else { continue }
                visitedDirs.insert(dirPath)

                let dirLower = candidateDir.lastPathComponent.lowercased()
                if drafterSubfolderNames.contains(dirLower)
                    || dirLower.hasSuffix("-assistant")
                    || dirLower.hasSuffix("-mtp")
                    || dirLower.contains("-mtp-")
                    || dirLower.contains("-dspark-") {
                    continue
                }

                let configJSON = loadJSONDictionary(at: candidateDir.appendingPathComponent("config.json"))
                if let archs = configJSON?["architectures"] as? [String],
                   archs.contains(where: { $0.localizedCaseInsensitiveContains("assistant") || $0 == "D1Model" }) {
                    continue
                }
                if configJSON?["mtp_num_layers"] != nil && configJSON?["num_hidden_layers"] == nil {
                    continue
                }

                let isVariant = (detectedProvider == .gguf && primaryGGUFURL != nil)
                    ? isGGUFVariantDirectory(candidateDir, primaryGGUFFile: primaryGGUFURL!)
                    : false
                let repoRootDir = isVariant ? candidateDir.deletingLastPathComponent() : candidateDir

                let sidecarURL = fm.fileExists(atPath: candidateDir.appendingPathComponent("model.heylook.toml").path)
                    ? candidateDir.appendingPathComponent("model.heylook.toml")
                    : repoRootDir.appendingPathComponent("model.heylook.toml")
                let sidecar: ModelSidecarConfig? = (try? String(contentsOf: sidecarURL, encoding: .utf8))
                    .map { ModelSidecarConfig.parseTOML($0) }

                let defaultID: String
                if isVariant, let primaryGGUFURL {
                    defaultID = ggufModelStem(from: primaryGGUFURL.lastPathComponent)
                } else {
                    defaultID = candidateDir.lastPathComponent
                }
                let modelID = sidecar?.id ?? defaultID
                let provider = sidecar?.provider ?? detectedProvider
                let placementValue = sidecar?.computePlacement ?? defaultPlacement
                let placementProv: ProvenanceSource = sidecar?.computePlacement != nil ? .configured : .derived

                let ggufHeader = primaryGGUFURL.flatMap { GGUFHeaderReader.readFileHeader(at: $0) }
                let archString = (configJSON?["model_type"] as? String)
                    ?? (configJSON?["architectures"] as? [String])?.first
                    ?? ggufHeader?.architecture
                    ?? provider.rawValue
                let isEmbedding = modelID.localizedCaseInsensitiveContains("embed")
                    || archString.localizedCaseInsensitiveContains("embedding")
                let mmprojPath = provider == .gguf ? discoverMMProj(in: candidateDir) : nil
                let hasVision = sidecar?.supportsVision
                    ?? (configJSON?["vision_config"] != nil
                        || mmprojPath != nil
                        || modelID.localizedCaseInsensitiveContains("vl")
                        || modelID.localizedCaseInsensitiveContains("vision")
                        || isEmbedding)
                let hasAudio = configJSON?["audio_config"] != nil || isEmbedding
                let ctxValue = sidecar?.ctxSize
                    ?? (configJSON?["max_position_embeddings"] as? Int)
                    ?? ggufHeader?.contextLength
                    ?? 8192

                let templateField: ProvenanceField<String>
                if provider == .gguf {
                    let ownTemplate = resolveGGUFTemplate(in: candidateDir, sidecar: sidecar)
                    if ownTemplate.source.value == "gguf_embedded" && isVariant {
                        templateField = resolveGGUFTemplate(in: repoRootDir, sidecar: sidecar).source
                    } else {
                        templateField = ownTemplate.source
                    }
                } else {
                    templateField = ProvenanceField(value: "tokenizer_config.json", provenance: .derived, effect: .reload)
                }

                var tags = sidecar?.tags ?? []
                let drafter = discoverDrafter(in: candidateDir, provider: provider, sidecar: sidecar)
                if drafter.draftModelPath != nil || drafter.specType != nil {
                    if !tags.contains("mtp") {
                        tags.append("mtp")
                    }
                }
                if hasVision && !tags.contains("vision") {
                    tags.append("vision")
                }

                discovered.append(
                    EngineDescription(
                        modelID: modelID,
                        provider: provider,
                        modelPath: dirPath,
                        architecture: ProvenanceField(value: archString, provenance: .derived, effect: .reload),
                        quantization: ProvenanceField(
                            value: inferQuantization(from: modelID, configJSON: configJSON),
                            provenance: .derived,
                            effect: .reload
                        ),
                        computePlacement: ProvenanceField(
                            value: placementValue,
                            provenance: placementProv,
                            effect: .reload,
                            computePlacement: placementValue
                        ),
                        isPinned: sidecar?.pinned ?? false,
                        isLoaded: false,
                        contextSize: ProvenanceField(
                            value: ctxValue,
                            provenance: sidecar?.ctxSize != nil ? .configured : .derived,
                            effect: .reload
                        ),
                        flashAttention: ProvenanceField(
                            value: sidecar?.flashAttn ?? true,
                            provenance: sidecar?.flashAttn != nil ? .configured : .derived,
                            effect: .reload
                        ),
                        supportsVision: ProvenanceField(
                            value: hasVision,
                            provenance: sidecar?.supportsVision != nil ? .configured : .derived,
                            effect: .reload
                        ),
                        supportsAudio: ProvenanceField(
                            value: hasAudio,
                            provenance: .derived,
                            effect: .reload
                        ),
                        supportsEmbeddings: ProvenanceField(
                            value: isEmbedding,
                            provenance: .derived,
                            effect: .reload
                        ),
                        supportsThinking: ProvenanceField(
                            value: sidecar?.supportsThinking ?? (!isEmbedding),
                            provenance: sidecar?.supportsThinking != nil ? .configured : .derived,
                            effect: .perRequest
                        ),
                        thinkingBudget: ProvenanceField(
                            value: sidecar?.sampling.thinkingBudget,
                            provenance: sidecar?.sampling.thinkingBudget != nil ? .configured : .notApplicable,
                            effect: .perRequest
                        ),
                        templateSource: templateField,
                        tags: tags
                    )
                )
            }
        }
        return discovered.sorted { $0.modelID.localizedStandardCompare($1.modelID) == .orderedAscending }
    }

    private static func loadJSONDictionary(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj
    }

    private static func firstGGUFFile(in candidateDir: URL) -> URL? {
        if candidateDir.pathExtension.lowercased() == "gguf" {
            return candidateDir
        }
        guard let items = try? FileManager.default.contentsOfDirectory(at: candidateDir, includingPropertiesForKeys: nil) else {
            return nil
        }
        return items
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            .first(where: { $0.pathExtension.lowercased() == "gguf" && !$0.lastPathComponent.lowercased().contains("mmproj") })
    }

    private static func inferQuantization(from modelID: String, configJSON: [String: Any]?) -> String {
        if let qCfg = configJSON?["quantization"] as? [String: Any],
           let bits = qCfg["bits"] as? Int {
            return "\(bits)-bit"
        }
        let lower = modelID.lowercased()
        if lower.contains("bf16") { return "bf16" }
        if lower.contains("fp16") { return "fp16" }
        if lower.contains("8bit") || lower.contains("q8") { return "8-bit" }
        if lower.contains("4bit") || lower.contains("q4") { return "4-bit" }
        return "native"
    }
}

/// Zero-dependency TOML parser supporting flat keys, `[section]` tables, `[[array_of_tables]]`,
/// and multi-line array values (`folders = [\n  "...",\n]`) for `heylook.toml` and `model.heylook.toml`.
enum SimpleTOML {
    struct Value {
        let raw: String

        var stringValue: String {
            var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if (s.hasPrefix("\"") && s.hasSuffix("\"")) || (s.hasPrefix("'") && s.hasSuffix("'")), s.count >= 2 {
                s.removeFirst()
                s.removeLast()
            }
            return s
        }

        var intValue: Int? { Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)) }
        var doubleValue: Double? { Double(raw.trimmingCharacters(in: .whitespacesAndNewlines)) }
        var boolValue: Bool? {
            switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "true": return true
            case "false": return false
            default: return nil
            }
        }

        var stringArrayValue: [String]? {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.hasPrefix("[") && trimmed.hasSuffix("]") else { return nil }
            let inner = trimmed.dropFirst().dropLast().trimmingCharacters(in: .whitespacesAndNewlines)
            if inner.isEmpty { return [] }
            return inner
                .split(separator: ",")
                .map { Value(raw: String($0)).stringValue }
                .filter { !$0.isEmpty }
        }
    }

    struct Document {
        var root: [String: Value] = [:]
        var sections: [String: [String: Value]] = [:]
        var arrayOfTables: [String: [[String: Value]]] = [:]
    }

    static func parseFlatKeyValues(_ text: String) -> [String: Value] {
        let doc = parseDocument(text)
        var merged = doc.root
        for (_, sec) in doc.sections {
            for (k, v) in sec where merged[k] == nil {
                merged[k] = v
            }
        }
        return merged
    }

    static func parseDocument(_ text: String) -> Document {
        var doc = Document()
        enum Target {
            case root
            case section(String)
            case arrayItem(String, Int)
        }
        var currentTarget: Target = .root
        var pendingKey: String?
        var pendingArrayBuffer = ""

        func store(key: String, value: Value) {
            switch currentTarget {
            case .root:
                doc.root[key] = value
            case .section(let name):
                var sec = doc.sections[name] ?? [:]
                sec[key] = value
                doc.sections[name] = sec
            case .arrayItem(let name, let idx):
                doc.arrayOfTables[name]?[idx][key] = value
            }
        }

        for rawLine in text.components(separatedBy: .newlines) {
            let line = stripInlineComment(from: rawLine)
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }

            if let key = pendingKey {
                pendingArrayBuffer.append(" " + trimmed)
                if trimmed.hasSuffix("]") {
                    store(key: key, value: Value(raw: pendingArrayBuffer))
                    pendingKey = nil
                    pendingArrayBuffer = ""
                }
                continue
            }

            if trimmed.hasPrefix("[[") && trimmed.hasSuffix("]]") && trimmed.count > 4 {
                let tableName = String(trimmed.dropFirst(2).dropLast(2)).trimmingCharacters(in: .whitespaces)
                var list = doc.arrayOfTables[tableName] ?? []
                list.append([:])
                doc.arrayOfTables[tableName] = list
                currentTarget = .arrayItem(tableName, list.count - 1)
                continue
            }

            if trimmed.hasPrefix("[") && trimmed.hasSuffix("]") && trimmed.count > 2 {
                let secName = String(trimmed.dropFirst().dropLast()).trimmingCharacters(in: .whitespaces)
                currentTarget = .section(secName)
                continue
            }

            let parts = trimmed.split(separator: "=", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let key = parts[0].trimmingCharacters(in: .whitespaces)
            let val = parts[1].trimmingCharacters(in: .whitespaces)
            if val.hasPrefix("[") && !val.hasSuffix("]") {
                pendingKey = key
                pendingArrayBuffer = val
            } else {
                store(key: key, value: Value(raw: val))
            }
        }
        return doc
    }

    private static func stripInlineComment(from line: String) -> String {
        var inDoubleQuotes = false
        var inSingleQuotes = false
        var result = ""
        for ch in line {
            if ch == "\"" && !inSingleQuotes {
                inDoubleQuotes.toggle()
            } else if ch == "'" && !inDoubleQuotes {
                inSingleQuotes.toggle()
            } else if ch == "#" && !inDoubleQuotes && !inSingleQuotes {
                break
            }
            result.append(ch)
        }
        return result
    }
}

/// Pure-Swift GGUF v2/v3 binary header reader that inspects magic bytes, version,
/// tensor/KV counts, and string/integer metadata without spawning a C++ process.
public enum GGUFHeaderReader {
    public struct HeaderSummary: Sendable, Equatable {
        public let version: UInt32
        public let tensorCount: UInt64
        public let metadataKVCount: UInt64
        public let stringMetadata: [String: String]
        public let uintMetadata: [String: UInt64]

        public init(
            version: UInt32,
            tensorCount: UInt64,
            metadataKVCount: UInt64,
            stringMetadata: [String: String] = [:],
            uintMetadata: [String: UInt64] = [:]
        ) {
            self.version = version
            self.tensorCount = tensorCount
            self.metadataKVCount = metadataKVCount
            self.stringMetadata = stringMetadata
            self.uintMetadata = uintMetadata
        }

        public var architecture: String? {
            stringMetadata["general.architecture"]
        }

        public var embeddedChatTemplate: String? {
            stringMetadata["tokenizer.chat_template"]
        }

        public var contextLength: Int? {
            if let arch = architecture, let val = uintMetadata["\(arch).context_length"] {
                return Int(val)
            }
            if let entry = uintMetadata.first(where: { $0.key.hasSuffix(".context_length") }) {
                return Int(entry.value)
            }
            return nil
        }
    }

    public static func readFileHeader(at url: URL, maxBytes: Int = 262_144) -> HeaderSummary? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: maxBytes) else { return nil }
        return readHeader(from: data)
    }

    public static func readHeader(from data: Data) -> HeaderSummary? {
        guard data.count >= 24 else { return nil }
        let magic = String(decoding: data.prefix(4), as: UTF8.self)
        guard magic == "GGUF" else { return nil }

        let version = data.subdata(in: 4..<8).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian }
        guard version == 2 || version == 3 else { return nil }
        let tensorCount = data.subdata(in: 8..<16).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        let kvCount = data.subdata(in: 16..<24).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }

        var offset = 24
        var strings: [String: String] = [:]
        var uints: [String: UInt64] = [:]

        func readUInt32() -> UInt32? {
            guard offset + 4 <= data.count else { return nil }
            let val = data.subdata(in: offset..<(offset + 4)).withUnsafeBytes {
                $0.loadUnaligned(as: UInt32.self).littleEndian
            }
            offset += 4
            return val
        }

        func readUInt64() -> UInt64? {
            guard offset + 8 <= data.count else { return nil }
            let val = data.subdata(in: offset..<(offset + 8)).withUnsafeBytes {
                $0.loadUnaligned(as: UInt64.self).littleEndian
            }
            offset += 8
            return val
        }

        func readString() -> String? {
            guard let len = readUInt64() else { return nil }
            guard len <= 1_048_576, offset + Int(len) <= data.count else { return nil }
            let strData = data.subdata(in: offset..<(offset + Int(len)))
            offset += Int(len)
            return String(decoding: strData, as: UTF8.self)
        }

        func skipOrReadValue(type: UInt32, key: String?) -> Bool {
            switch type {
            case 0: // UINT8
                guard offset + 1 <= data.count else { return false }
                let b = data[offset]
                offset += 1
                if let key { uints[key] = UInt64(b) }
                return true
            case 1, 7: // INT8, BOOL
                guard offset + 1 <= data.count else { return false }
                offset += 1
                return true
            case 2: // UINT16
                guard offset + 2 <= data.count else { return false }
                let v = data.subdata(in: offset..<(offset + 2)).withUnsafeBytes {
                    $0.loadUnaligned(as: UInt16.self).littleEndian
                }
                offset += 2
                if let key { uints[key] = UInt64(v) }
                return true
            case 3: // INT16
                guard offset + 2 <= data.count else { return false }
                offset += 2
                return true
            case 4: // UINT32
                guard let v = readUInt32() else { return false }
                if let key { uints[key] = UInt64(v) }
                return true
            case 5, 6: // INT32, FLOAT32
                guard offset + 4 <= data.count else { return false }
                offset += 4
                return true
            case 8: // STRING
                guard let str = readString() else { return false }
                if let key { strings[key] = str }
                return true
            case 9: // ARRAY
                guard let elemType = readUInt32(), let count = readUInt64(), count <= 65536 else { return false }
                for _ in 0..<count {
                    if !skipOrReadValue(type: elemType, key: nil) { return false }
                }
                return true
            case 10: // UINT64
                guard let v = readUInt64() else { return false }
                if let key { uints[key] = v }
                return true
            case 11, 12: // INT64, FLOAT64
                guard offset + 8 <= data.count else { return false }
                offset += 8
                return true
            default:
                return false
            }
        }

        let maxEntries = min(kvCount, 2048)
        for _ in 0..<maxEntries {
            guard offset < data.count else { break }
            guard let key = readString(), let valType = readUInt32() else { break }
            if !skipOrReadValue(type: valType, key: key) {
                break
            }
        }

        return HeaderSummary(
            version: version,
            tensorCount: tensorCount,
            metadataKVCount: kvCount,
            stringMetadata: strings,
            uintMetadata: uints
        )
    }

    /// Builds a valid GGUF v2/v3 binary header with embedded string and uint32 KV entries.
    public static func buildMinimalGGUF(
        version: UInt32 = 3,
        tensorCount: UInt64 = 0,
        stringMetadata: [String: String] = [:],
        uint32Metadata: [String: UInt32] = [:]
    ) -> Data {
        var data = Data("GGUF".utf8)
        var verLE = version.littleEndian
        var tcLE = tensorCount.littleEndian
        let totalKV = UInt64(stringMetadata.count + uint32Metadata.count)
        var kvLE = totalKV.littleEndian

        withUnsafeBytes(of: &verLE) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &tcLE) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &kvLE) { data.append(contentsOf: $0) }

        func appendString(_ s: String) {
            let utf8 = Array(s.utf8)
            var lenLE = UInt64(utf8.count).littleEndian
            withUnsafeBytes(of: &lenLE) { data.append(contentsOf: $0) }
            data.append(contentsOf: utf8)
        }

        func appendUInt32(_ v: UInt32) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }

        for (k, v) in stringMetadata.sorted(by: { $0.key < $1.key }) {
            appendString(k)
            appendUInt32(8) // STRING
            appendString(v)
        }
        for (k, v) in uint32Metadata.sorted(by: { $0.key < $1.key }) {
            appendString(k)
            appendUInt32(4) // UINT32
            appendUInt32(v)
        }
        return data
    }
}
