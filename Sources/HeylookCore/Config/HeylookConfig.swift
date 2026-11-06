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
        self.sampling = sampling
        self.tags = tags
    }

    /// Parses a `model.heylook.toml` file content into a `ModelSidecarConfig`.
    public static func parseTOML(_ text: String) -> ModelSidecarConfig {
        let kv = SimpleTOML.parseFlatKeyValues(text)
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
        config.supportsThinking = kv["thinking_capable"]?.boolValue
        config.draftModel = kv["draft_model"]?.stringValue
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
public struct HeylookConfig: Codable, Sendable, Equatable {
    public var defaultModel: String?
    public var maxLoadedModels: Int
    public var maxQueueDepth: Int
    public var scanFolders: [String]
    public var serverHost: String
    public var serverPort: Int
    public var adminToken: String?

    public init(
        defaultModel: String? = nil,
        maxLoadedModels: Int = 3,
        maxQueueDepth: Int = 2,
        scanFolders: [String] = [],
        serverHost: String = "127.0.0.1",
        serverPort: Int = 8000,
        adminToken: String? = nil
    ) {
        self.defaultModel = defaultModel
        self.maxLoadedModels = max(1, maxLoadedModels)
        self.maxQueueDepth = max(1, maxQueueDepth)
        self.scanFolders = scanFolders
        self.serverHost = serverHost
        self.serverPort = serverPort
        self.adminToken = adminToken
    }

    public static func parseTOML(_ text: String) -> HeylookConfig {
        let kv = SimpleTOML.parseFlatKeyValues(text)
        return HeylookConfig(
            defaultModel: kv["default_model"]?.stringValue,
            maxLoadedModels: kv["max_loaded_models"]?.intValue ?? 3,
            maxQueueDepth: kv["max_queue_depth"]?.intValue ?? 2,
            scanFolders: kv["scan_folders"]?.stringArrayValue ?? [],
            serverHost: kv["host"]?.stringValue ?? "127.0.0.1",
            serverPort: kv["port"]?.intValue ?? 8000,
            adminToken: kv["admin_token"]?.stringValue
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

    /// Scans `scanFolders` for `.safetensors` (MLX), `.aimodel` / `.mlmodelc` (CoreAI / ANE),
    /// and `.gguf` (llama-server) model directories, merging any `model.heylook.toml` sidecar.
    public static func discoverModels(in folders: [URL]) -> [EngineDescription] {
        var discovered: [EngineDescription] = []
        let fm = FileManager.default

        for root in folders {
            let expandedRoot = URL(fileURLWithPath: (root.path as NSString).expandingTildeInPath)
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

                if ext == "safetensors" {
                    candidateDir = url.deletingLastPathComponent()
                    detectedProvider = .mlx
                    defaultPlacement = .mlxGPU
                } else if ext == "aimodel" || ext == "mlmodelc" {
                    candidateDir = url.deletingLastPathComponent()
                    detectedProvider = .coreai
                    defaultPlacement = .ane
                } else if ext == "gguf" {
                    candidateDir = url.deletingLastPathComponent()
                    detectedProvider = .gguf
                    defaultPlacement = .ggufSubprocess
                } else {
                    continue
                }

                let dirPath = candidateDir.standardizedFileURL.path
                guard !visitedDirs.contains(dirPath) else { continue }
                visitedDirs.insert(dirPath)

                let sidecarURL = candidateDir.appendingPathComponent("model.heylook.toml")
                let sidecar: ModelSidecarConfig? = (try? String(contentsOf: sidecarURL, encoding: .utf8))
                    .map { ModelSidecarConfig.parseTOML($0) }

                let modelID = sidecar?.id ?? candidateDir.lastPathComponent
                let provider = sidecar?.provider ?? detectedProvider
                let placementValue = sidecar?.computePlacement ?? defaultPlacement
                let placementProv: ProvenanceSource = sidecar?.computePlacement != nil ? .configured : .derived

                let configJSON = loadJSONDictionary(at: candidateDir.appendingPathComponent("config.json"))
                let archString = (configJSON?["model_type"] as? String)
                    ?? (configJSON?["architectures"] as? [String])?.first
                    ?? provider.rawValue
                let isEmbedding = modelID.localizedCaseInsensitiveContains("embed")
                    || archString.localizedCaseInsensitiveContains("embedding")
                let hasVision = sidecar?.supportsVision
                    ?? (configJSON?["vision_config"] != nil || modelID.localizedCaseInsensitiveContains("vl") || isEmbedding)
                let hasAudio = configJSON?["audio_config"] != nil || isEmbedding
                let ctxValue = sidecar?.ctxSize
                    ?? (configJSON?["max_position_embeddings"] as? Int)
                    ?? 8192

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
                        tags: sidecar?.tags ?? []
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

/// Zero-dependency flat TOML parser supporting strings, ints, floats, bools, and string arrays
/// for `heylook.toml` and `model.heylook.toml`.
enum SimpleTOML {
    struct Value {
        let raw: String

        var stringValue: String {
            var s = raw.trimmingCharacters(in: .whitespaces)
            if (s.hasPrefix("\"") && s.hasSuffix("\"")) || (s.hasPrefix("'") && s.hasSuffix("'")), s.count >= 2 {
                s.removeFirst()
                s.removeLast()
            }
            return s
        }

        var intValue: Int? { Int(raw.trimmingCharacters(in: .whitespaces)) }
        var doubleValue: Double? { Double(raw.trimmingCharacters(in: .whitespaces)) }
        var boolValue: Bool? {
            switch raw.trimmingCharacters(in: .whitespaces).lowercased() {
            case "true": return true
            case "false": return false
            default: return nil
            }
        }

        var stringArrayValue: [String]? {
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("[") && trimmed.hasSuffix("]") else { return nil }
            let inner = trimmed.dropFirst().dropLast()
            if inner.trimmingCharacters(in: .whitespaces).isEmpty { return [] }
            return inner.split(separator: ",").map { Value(raw: String($0)).stringValue }
        }
    }

    static func parseFlatKeyValues(_ text: String) -> [String: Value] {
        var result: [String: Value] = [:]
        for rawLine in text.components(separatedBy: .newlines) {
            let line = rawLine.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? ""
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("[") { continue }
            let parts = trimmed.split(separator: "=", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let key = parts[0].trimmingCharacters(in: .whitespaces)
            let val = parts[1].trimmingCharacters(in: .whitespaces)
            result[key] = Value(raw: val)
        }
        return result
    }
}
