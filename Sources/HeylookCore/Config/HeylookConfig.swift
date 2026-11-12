import CryptoKit
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
    public var imageMaxTokens: Int?
    public var draftModel: String?
    public var specType: String?
    public var chatTemplate: String?
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
        imageMaxTokens: Int? = nil,
        draftModel: String? = nil,
        specType: String? = nil,
        chatTemplate: String? = nil,
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
        self.imageMaxTokens = imageMaxTokens
        self.draftModel = draftModel
        self.specType = specType
        self.chatTemplate = chatTemplate
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
        config.ctxSize = kv["ctx_size"]?.intValue
            ?? kv["context_size"]?.intValue
            ?? kv["cache_size"]?.intValue
        config.flashAttn = kv["flash_attn"]?.boolValue
        config.supportsVision = kv["vision"]?.boolValue
        config.supportsThinking = kv["thinking_capable"]?.boolValue ?? kv["supports_thinking"]?.boolValue
        config.imageMaxTokens = kv["image_max_tokens"]?.intValue
        config.draftModel = kv["draft_model"]?.stringValue ?? kv["draft_model_path"]?.stringValue
        config.specType = kv["spec_type"]?.stringValue
        config.chatTemplate = kv["chat_template"]?.stringValue
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
            thinkingDepth: kv["thinking_depth"]?.stringValue
                ?? kv["reasoning_effort"]?.stringValue
                ?? kv["reasoning_strength"]?.stringValue,
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
        if let imageMaxTokens { lines.append("image_max_tokens = \(imageMaxTokens)") }
        if let draftModel { lines.append("draft_model = \"\(draftModel)\"") }
        if let specType { lines.append("spec_type = \"\(specType)\"") }
        if let chatTemplate { lines.append("chat_template = \"\(chatTemplate)\"") }
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

    /// Canonical application support directory (`~/Library/Application Support/Heylook`),
    /// shared between `HeylookApp`, `HeylookUI`, `HeylookServer`, and `heylook-cli`.
    public static var defaultAppSupportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let dir = base.appendingPathComponent("Heylook", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Canonical SQLite WAL store path for conversations, messages, notebooks, and presets.
    /// Honors `HEYLOOK_STORE_DB_PATH` or `HEYLOOK_DB_PATH` (when ending in `heylook.sqlite`).
    public static var defaultStoreURL: URL {
        let env = ProcessInfo.processInfo.environment
        if let custom = env["HEYLOOK_STORE_DB_PATH"], !custom.isEmpty {
            return URL(fileURLWithPath: (custom as NSString).expandingTildeInPath)
        }
        if let custom = env["HEYLOOK_DB_PATH"], !custom.isEmpty, custom.hasSuffix("heylook.sqlite") {
            return URL(fileURLWithPath: (custom as NSString).expandingTildeInPath)
        }
        return defaultAppSupportDirectory.appendingPathComponent("heylook.sqlite")
    }

    /// Canonical SQLite WAL vector store path for `HeylookEmbeddings` (`VectorStore`).
    /// Honors `HEYLOOK_VECTOR_DB_PATH` or `HEYLOOK_DB_PATH`.
    public static var defaultVectorStoreURL: URL {
        let env = ProcessInfo.processInfo.environment
        if let custom = env["HEYLOOK_VECTOR_DB_PATH"], !custom.isEmpty {
            return URL(fileURLWithPath: (custom as NSString).expandingTildeInPath)
        }
        if let custom = env["HEYLOOK_DB_PATH"], !custom.isEmpty {
            return URL(fileURLWithPath: (custom as NSString).expandingTildeInPath)
        }
        return defaultAppSupportDirectory.appendingPathComponent("heylook_vectors.sqlite")
    }

    /// Resolves a model path (which may point to a directory or a `.gguf` file) to its parent directory URL.
    public static func resolveModelDirectoryURL(from modelPath: String) -> URL {
        let rawURL = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath)
        if rawURL.pathExtension.lowercased() == "gguf" {
            return rawURL.deletingLastPathComponent()
        }
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: rawURL.path, isDirectory: &isDir), !isDir.boolValue {
            return rawURL.deletingLastPathComponent()
        }
        return rawURL
    }

    /// Resolves the 5-layer `SamplerCascade` (`builtin` -> `vendor generation_config.json` ->
    /// `model.heylook.toml` -> `preset` -> `request`) from a model directory path.
    public static func resolveEffectiveSampling(
        modelPath: String?,
        preset: SamplerSettings? = nil,
        request: SamplerSettings? = nil
    ) -> SamplerCascade.ResolvedSampling {
        var vendorSettings: SamplerSettings?
        var modelSidecarSettings: SamplerSettings?
        if let modelPath, !modelPath.isEmpty {
            let dirURL = resolveModelDirectoryURL(from: modelPath)
            vendorSettings = loadVendorSampling(in: dirURL)
            let sidecarURL = dirURL.appendingPathComponent("model.heylook.toml")
            if let text = try? String(contentsOf: sidecarURL, encoding: .utf8) {
                modelSidecarSettings = ModelSidecarConfig.parseTOML(text).sampling
            }
        }
        return SamplerCascade.resolve(
            vendor: vendorSettings,
            modelSidecar: modelSidecarSettings,
            preset: preset,
            request: request
        )
    }

    /// Loads `heylook.toml` from `HEYLOOK_CONFIG`, `./heylook.toml`, the package source root
    /// (when launched from Xcode DerivedData), `~/Library/Application Support/Heylook/heylook.toml`,
    /// or `~/.heylook/heylook.toml`, merging `HEYLOOK_MODELS_DIR` if present.
    public static func loadDefault() -> HeylookConfig {
        let env = ProcessInfo.processInfo.environment
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        #if os(macOS)
        let home = fm.homeDirectoryForCurrentUser
        #else
        let home = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        #endif
        let sourceRepoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        var candidates: [URL] = []
        if let custom = env["HEYLOOK_CONFIG"], !custom.isEmpty {
            candidates.append(URL(fileURLWithPath: (custom as NSString).expandingTildeInPath))
        }
        candidates.append(cwd.appendingPathComponent("heylook.toml"))
        candidates.append(sourceRepoRoot.appendingPathComponent("heylook.toml"))
        candidates.append(defaultAppSupportDirectory.appendingPathComponent("heylook.toml"))
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

    /// True when `HEYLOOK_READONLY_MODEL_CONFIG` is enabled (`1` or `true`), blocking disk writes to model configs and overrides.
    public static var isModelConfigReadOnly: Bool {
        guard let raw = ProcessInfo.processInfo.environment["HEYLOOK_READONLY_MODEL_CONFIG"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }

    /// Resolves the active `heylook.toml` file URL used for reading and persisting operator configuration updates.
    public static func activeConfigURL() -> URL {
        let env = ProcessInfo.processInfo.environment
        if let custom = env["HEYLOOK_CONFIG"], !custom.isEmpty {
            return URL(fileURLWithPath: (custom as NSString).expandingTildeInPath)
        }
        let cwdToml = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("heylook.toml")
        if FileManager.default.fileExists(atPath: cwdToml.path) {
            return cwdToml
        }
        return defaultAppSupportDirectory.appendingPathComponent("heylook.toml")
    }

    /// Loads `model.heylook.toml` from `candidateDir` when present.
    public static func loadSidecar(in candidateDir: URL) -> ModelSidecarConfig? {
        let dir = resolveModelDirectoryURL(from: candidateDir.path)
        let sidecarURL = dir.appendingPathComponent("model.heylook.toml")
        guard let text = try? String(contentsOf: sidecarURL, encoding: .utf8) else {
            return nil
        }
        return ModelSidecarConfig.parseTOML(text)
    }

    /// Writes `model.heylook.toml` atomically inside `candidateDir`.
    public static func writeSidecar(_ sidecar: ModelSidecarConfig, in candidateDir: URL) throws {
        let dir = resolveModelDirectoryURL(from: candidateDir.path)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let sidecarURL = dir.appendingPathComponent("model.heylook.toml")
        try sidecar.toTOML().write(to: sidecarURL, atomically: true, encoding: .utf8)
    }

    /// Convenience static discovery entry point matching `loadDefault().discoverConfiguredModels()`.
    public static func discoverConfiguredModels() -> [EngineDescription] {
        loadDefault().discoverConfiguredModels()
    }

    /// Persists this `HeylookConfig` atomically to `configURL` (or `activeConfigURL()` when omitted),
    /// preserving `default_model`, `admin_token`, `allowed_hosts`, `scan_folders`, and all `[[models]]` entries.
    public func save(to configURL: URL? = nil) throws {
        let targetURL = configURL ?? Self.activeConfigURL()
        try FileManager.default.createDirectory(
            at: targetURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try toTOML().write(to: targetURL, atomically: true, encoding: .utf8)
    }

    private static func normalizeScanFolderPath(_ rawPath: String) -> String {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        if trimmed == "/" || trimmed.hasPrefix("~") {
            return trimmed.count > 1 && trimmed.hasSuffix("/") ? String(trimmed.dropLast()) : trimmed
        }
        return URL(fileURLWithPath: trimmed).standardizedFileURL.path
    }

    /// Appends `folderPath` to `scanFolders` if non-empty and not already present (comparing normalized paths).
    @discardableResult
    public mutating func addScanFolder(_ folderPath: String) -> Bool {
        let trimmed = folderPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        let normalizedNew = Self.normalizeScanFolderPath(trimmed)
        let alreadyExists = scanFolders.contains { existing in
            existing == trimmed || Self.normalizeScanFolderPath(existing) == normalizedNew
        }
        guard !alreadyExists else { return false }
        scanFolders.append(trimmed)
        return true
    }

    /// Removes `folderPath` from `scanFolders` (matching either exact text or normalized filesystem path).
    @discardableResult
    public mutating func removeScanFolder(_ folderPath: String) -> Bool {
        let trimmed = folderPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        let normalizedTarget = Self.normalizeScanFolderPath(trimmed)
        let before = scanFolders.count
        scanFolders.removeAll { existing in
            existing == trimmed || Self.normalizeScanFolderPath(existing) == normalizedTarget
        }
        return scanFolders.count != before
    }

    /// Non-destructively adds a watched/scan folder to `heylook.toml` on disk, preserving `[[models]]`,
    /// `admin_token`, `allowed_hosts`, and server limits.
    @discardableResult
    public static func addScanFolder(_ folderPath: String, configURL: URL? = nil) throws -> HeylookConfig {
        let targetURL = configURL ?? activeConfigURL()
        var cfg: HeylookConfig
        if let existingText = try? String(contentsOf: targetURL, encoding: .utf8) {
            cfg = parseTOML(existingText)
        } else {
            cfg = configURL == nil ? loadDefault() : HeylookConfig()
        }
        _ = cfg.addScanFolder(folderPath)
        try cfg.save(to: targetURL)
        return cfg
    }

    /// Non-destructively removes a watched/scan folder from `heylook.toml` on disk, preserving `[[models]]`,
    /// `admin_token`, `allowed_hosts`, and server limits.
    @discardableResult
    public static func removeScanFolder(_ folderPath: String, configURL: URL? = nil) throws -> HeylookConfig {
        let targetURL = configURL ?? activeConfigURL()
        var cfg: HeylookConfig
        if let existingText = try? String(contentsOf: targetURL, encoding: .utf8) {
            cfg = parseTOML(existingText)
        } else {
            cfg = configURL == nil ? loadDefault() : HeylookConfig()
        }
        _ = cfg.removeScanFolder(folderPath)
        try cfg.save(to: targetURL)
        return cfg
    }

    public func toTOML() -> String {
        var lines: [String] = ["# heylook.toml — Global server & model discovery configuration"]
        if let defaultModel { lines.append("default_model = \"\(defaultModel)\"") }
        lines.append("max_loaded_models = \(maxLoadedModels)")
        lines.append("max_queue_depth = \(maxQueueDepth)")
        lines.append("host = \"\(serverHost)\"")
        lines.append("port = \(serverPort)")
        if let adminToken, !adminToken.isEmpty {
            lines.append("admin_token = \"\(adminToken)\"")
        }
        if !allowedHosts.isEmpty {
            let quoted = allowedHosts.map { "\"\($0)\"" }.joined(separator: ", ")
            lines.append("allowed_hosts = [\(quoted)]")
        }
        if !scanFolders.isEmpty {
            let quoted = scanFolders.map { "\"\($0)\"" }.joined(separator: ", ")
            lines.append("scan_folders = [\(quoted)]")
        }
        for entry in explicitModels {
            lines.append("")
            lines.append("[[models]]")
            lines.append("id = \"\(entry.id)\"")
            lines.append("path = \"\(entry.path)\"")
            lines.append("provider = \"\(entry.provider.rawValue)\"")
            for extra in entry.sidecar.toTOML().components(separatedBy: .newlines) {
                let t = extra.trimmingCharacters(in: .whitespaces)
                if t.isEmpty || t.hasPrefix("#") || t.hasPrefix("id =") || t.hasPrefix("provider =") {
                    continue
                }
                lines.append(t)
            }
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

    /// Loads vendor sampling defaults (`temperature`, `top_p`, `top_k`, `min_p`) from `generation_config.json`
    /// or `.gguf` header `general.sampling.*` metadata in a model directory.
    public static func loadVendorSampling(in candidateDir: URL) -> SamplerSettings? {
        if let json = loadJSONDictionary(at: candidateDir.appendingPathComponent("generation_config.json")) {
            let temp = (json["temperature"] as? NSNumber)?.doubleValue
            let topP = (json["top_p"] as? NSNumber)?.doubleValue
            let topK = (json["top_k"] as? NSNumber)?.intValue
            let minP = (json["min_p"] as? NSNumber)?.doubleValue
            if temp != nil || topP != nil || topK != nil || minP != nil {
                return SamplerSettings(temperature: temp, topP: topP, topK: topK, minP: minP)
            }
        }
        if let ggufURL = firstGGUFFile(in: candidateDir),
           let header = GGUFHeaderReader.readFileHeader(at: ggufURL),
           let ggufSampling = header.vendorSampling {
            return ggufSampling
        }
        return nil
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

    public typealias ChatTemplateInspection = ChatTemplateState

    public struct ChatTemplateState: Codable, Sendable, Equatable {
        public let source: String
        public let hasOverride: Bool
        public let template: String
        public let origin: String
        public let writable: Bool
        public let inertReason: String?
        public let stale: Bool
        public let prefixStable: Bool
        public let prefixNote: String?
        public let modifiedSinceDownload: Bool?
        public let caps: TemplateCapabilities

        public var overridePresent: Bool { hasOverride }
        public var detectedCapabilities: TemplateCapabilities { caps }
        public var prefixStabilityNote: String? { prefixNote }

        enum CodingKeys: String, CodingKey {
            case source
            case hasOverride = "has_override"
            case overridePresent = "override_present"
            case template
            case origin
            case writable
            case inertReason = "inert_reason"
            case stale
            case prefixStable = "prefix_stable"
            case prefixNote = "prefix_note"
            case modifiedSinceDownload = "modified_since_download"
            case caps
        }

        public init(
            source: String,
            hasOverride: Bool,
            template: String,
            origin: String? = nil,
            writable: Bool = !HeylookConfig.isModelConfigReadOnly,
            inertReason: String? = nil,
            stale: Bool = false,
            prefixStable: Bool? = nil,
            prefixNote: String? = nil,
            modifiedSinceDownload: Bool? = nil,
            caps: TemplateCapabilities? = nil
        ) {
            self.source = source
            self.hasOverride = hasOverride
            self.template = template
            self.origin = origin ?? (hasOverride ? "operator_override" : (source == "gguf_embedded" ? "gguf_header" : "vendor_file"))
            self.writable = writable
            self.inertReason = inertReason
            self.stale = stale
            if let prefixStable {
                self.prefixStable = prefixStable
                self.prefixNote = prefixNote
            } else {
                let stability = HeylookJinjaRenderer.checkPrefixStability(template: template)
                self.prefixStable = stability.stable
                self.prefixNote = prefixNote ?? stability.note
            }
            self.modifiedSinceDownload = modifiedSinceDownload
            self.caps = caps ?? HeylookJinjaRenderer.detectCapabilities(template: template)
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let source = try c.decode(String.self, forKey: .source)
            let hasOverride = try c.decodeIfPresent(Bool.self, forKey: .hasOverride)
                ?? c.decodeIfPresent(Bool.self, forKey: .overridePresent)
                ?? false
            let template = try c.decode(String.self, forKey: .template)
            let origin = try c.decodeIfPresent(String.self, forKey: .origin)
            let writable = try c.decodeIfPresent(Bool.self, forKey: .writable) ?? !HeylookConfig.isModelConfigReadOnly
            let inertReason = try c.decodeIfPresent(String.self, forKey: .inertReason)
            let stale = try c.decodeIfPresent(Bool.self, forKey: .stale) ?? false
            let prefixStable = try c.decodeIfPresent(Bool.self, forKey: .prefixStable)
            let prefixNote = try c.decodeIfPresent(String.self, forKey: .prefixNote)
            let modified = try c.decodeIfPresent(Bool.self, forKey: .modifiedSinceDownload)
            let caps = try c.decodeIfPresent(TemplateCapabilities.self, forKey: .caps)
            self.init(
                source: source,
                hasOverride: hasOverride,
                template: template,
                origin: origin,
                writable: writable,
                inertReason: inertReason,
                stale: stale,
                prefixStable: prefixStable,
                prefixNote: prefixNote,
                modifiedSinceDownload: modified,
                caps: caps
            )
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(source, forKey: .source)
            try c.encode(hasOverride, forKey: .hasOverride)
            try c.encode(hasOverride, forKey: .overridePresent)
            try c.encode(template, forKey: .template)
            try c.encode(origin, forKey: .origin)
            try c.encode(writable, forKey: .writable)
            try c.encode(inertReason, forKey: .inertReason)
            try c.encode(stale, forKey: .stale)
            try c.encode(prefixStable, forKey: .prefixStable)
            try c.encode(prefixNote, forKey: .prefixNote)
            try c.encode(modifiedSinceDownload, forKey: .modifiedSinceDownload)
            try c.encode(caps, forKey: .caps)
        }
    }

    /// Checks HuggingFace Hub `.cache/huggingface/download/<filename>.metadata` git-blob SHA-1 / LFS SHA-256 ETag
    /// to detect whether a vendor file on disk was edited after download (`chat_template_files.py`).
    /// Returns `nil` when no `.metadata` file exists, `false` when clean, and `true` when modified.
    public static func checkModifiedSinceDownload(fileURL: URL, repoRoot: URL? = nil) -> Bool? {
        let fm = FileManager.default
        guard let fileData = try? Data(contentsOf: fileURL) else { return nil }
        let fileName = fileURL.lastPathComponent
        let dir = fileURL.deletingLastPathComponent()
        var candidateRoots = [dir]
        if let repoRoot, repoRoot.standardizedFileURL.path != dir.standardizedFileURL.path {
            candidateRoots.append(repoRoot)
        } else {
            candidateRoots.append(dir.deletingLastPathComponent())
        }

        for root in candidateRoots {
            let metaURL = root
                .appendingPathComponent(".cache", isDirectory: true)
                .appendingPathComponent("huggingface", isDirectory: true)
                .appendingPathComponent("download", isDirectory: true)
                .appendingPathComponent("\(fileName).metadata")
            guard fm.fileExists(atPath: metaURL.path),
                  let metaText = try? String(contentsOf: metaURL, encoding: .utf8)
            else { continue }

            let lines = metaText
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            guard lines.count >= 2 else { continue }
            let rawETag = lines[1].trimmingCharacters(in: CharacterSet(charactersIn: "\"'")).lowercased()
            if rawETag.count == 40, rawETag.allSatisfy({ $0.isHexDigit }) {
                var gitBlob = Data("blob \(fileData.count)\0".utf8)
                gitBlob.append(fileData)
                let digest = Insecure.SHA1.hash(data: gitBlob)
                    .map { String(format: "%02x", $0) }
                    .joined()
                return digest != rawETag
            } else if rawETag.count == 64, rawETag.allSatisfy({ $0.isHexDigit }) {
                let digest = SHA256.hash(data: fileData)
                    .map { String(format: "%02x", $0) }
                    .joined()
                return digest != rawETag
            }
        }
        return nil
    }

    /// Inspects the active chat template in `candidateDir` across the 4-rung ladder:
    /// `chat_template.heylook.jinja` (operator override) -> `chat_template.jinja` (vendor ship) ->
    /// `tokenizer_config.json` -> `.gguf` header `tokenizer.chat_template` (`gguf_embedded`).
    public static func inspectChatTemplate(
        in candidateDir: URL,
        fallbackSource: String = "default_chatml",
        loadedTemplateText: String? = nil
    ) -> ChatTemplateState {
        let dir = candidateDir.pathExtension.lowercased() == "gguf"
            ? candidateDir.deletingLastPathComponent()
            : candidateDir
        let fm = FileManager.default
        let sidecar = loadSidecar(in: dir)
        let overrideURL = dir.appendingPathComponent("chat_template.heylook.jinja")
        let sidecarURL = dir.appendingPathComponent("chat_template.jinja")
        let tokConfigURL = dir.appendingPathComponent("tokenizer_config.json")
        let writable = !isModelConfigReadOnly

        let inertReason: String?
        if let explicitPath = sidecar?.chatTemplatePath, !explicitPath.isEmpty {
            inertReason = "Explicit chat_template_path ('\(explicitPath)') in model.heylook.toml overrides chat_template.heylook.jinja."
        } else {
            inertReason = nil
        }

        func isStale(against diskTemplate: String) -> Bool {
            guard let loadedTemplateText else { return false }
            return loadedTemplateText != diskTemplate
        }

        if let inlineOrRef = sidecar?.chatTemplate, !inlineOrRef.isEmpty {
            let candidateFile = inlineOrRef.hasPrefix("/")
                ? URL(fileURLWithPath: inlineOrRef)
                : dir.appendingPathComponent(inlineOrRef)
            if fm.fileExists(atPath: candidateFile.path),
               let t = try? String(contentsOf: candidateFile, encoding: .utf8) {
                return ChatTemplateState(
                    source: candidateFile.lastPathComponent,
                    hasOverride: fm.fileExists(atPath: overrideURL.path),
                    template: t,
                    origin: "sidecar",
                    writable: writable,
                    inertReason: inertReason,
                    stale: isStale(against: t),
                    modifiedSinceDownload: checkModifiedSinceDownload(fileURL: candidateFile, repoRoot: dir)
                )
            }
            return ChatTemplateState(
                source: "model.heylook.toml",
                hasOverride: fm.fileExists(atPath: overrideURL.path),
                template: inlineOrRef,
                origin: "sidecar",
                writable: writable,
                inertReason: inertReason,
                stale: isStale(against: inlineOrRef),
                modifiedSinceDownload: nil
            )
        }

        if let explicitPath = sidecar?.chatTemplatePath, !explicitPath.isEmpty {
            let explicitURL = explicitPath.hasPrefix("/")
                ? URL(fileURLWithPath: explicitPath)
                : dir.appendingPathComponent(explicitPath)
            if let t = try? String(contentsOf: explicitURL, encoding: .utf8) {
                return ChatTemplateState(
                    source: explicitURL.lastPathComponent,
                    hasOverride: fm.fileExists(atPath: overrideURL.path),
                    template: t,
                    origin: "explicit_path",
                    writable: writable,
                    inertReason: inertReason,
                    stale: isStale(against: t),
                    modifiedSinceDownload: checkModifiedSinceDownload(fileURL: explicitURL, repoRoot: dir)
                )
            }
        }

        if fm.fileExists(atPath: overrideURL.path),
           let t = try? String(contentsOf: overrideURL, encoding: .utf8) {
            return ChatTemplateState(
                source: "chat_template.heylook.jinja",
                hasOverride: true,
                template: t,
                origin: "operator_override",
                writable: writable,
                inertReason: inertReason,
                stale: isStale(against: t),
                modifiedSinceDownload: nil
            )
        }
        let allowSidecar = sidecar?.useSidecarChatTemplate ?? true
        if allowSidecar,
           fm.fileExists(atPath: sidecarURL.path),
           let t = try? String(contentsOf: sidecarURL, encoding: .utf8) {
            return ChatTemplateState(
                source: "chat_template.jinja",
                hasOverride: false,
                template: t,
                origin: "vendor_sidecar",
                writable: writable,
                inertReason: inertReason,
                stale: isStale(against: t),
                modifiedSinceDownload: checkModifiedSinceDownload(fileURL: sidecarURL, repoRoot: dir)
            )
        }
        if let tokJSON = loadJSONDictionary(at: tokConfigURL),
           let t = tokJSON["chat_template"] as? String {
            return ChatTemplateState(
                source: "tokenizer_config.json",
                hasOverride: false,
                template: t,
                origin: "tokenizer_config",
                writable: writable,
                inertReason: inertReason,
                stale: isStale(against: t),
                modifiedSinceDownload: checkModifiedSinceDownload(fileURL: tokConfigURL, repoRoot: dir)
            )
        }
        if let ggufURL = firstGGUFFile(in: candidateDir),
           let ggufHeader = GGUFHeaderReader.readFileHeader(at: ggufURL),
           let embedded = ggufHeader.embeddedChatTemplate,
           !embedded.isEmpty {
            return ChatTemplateState(
                source: "gguf_embedded",
                hasOverride: false,
                template: embedded,
                origin: "gguf_header",
                writable: writable,
                inertReason: inertReason,
                stale: isStale(against: embedded),
                modifiedSinceDownload: nil
            )
        }
        return ChatTemplateState(
            source: fallbackSource,
            hasOverride: false,
            template: "",
            origin: "fallback",
            writable: writable,
            inertReason: inertReason,
            stale: false,
            modifiedSinceDownload: nil
        )
    }

    /// True when `chat_template.heylook.jinja` exists inside `candidateDir`.
    public static func hasChatTemplateOverride(in candidateDir: URL) -> Bool {
        let dir = candidateDir.pathExtension.lowercased() == "gguf"
            ? candidateDir.deletingLastPathComponent()
            : candidateDir
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("chat_template.heylook.jinja").path)
    }

    /// Validates Jinja2 syntax and writes `chat_template.heylook.jinja` inside `candidateDir` without mutating vendor files.
    @discardableResult
    public static func writeChatTemplateOverride(
        _ template: String,
        in candidateDir: URL
    ) throws -> ChatTemplateState {
        try HeylookJinjaRenderer.validateSyntax(template)
        let dir = candidateDir.pathExtension.lowercased() == "gguf"
            ? candidateDir.deletingLastPathComponent()
            : candidateDir
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let overrideURL = dir.appendingPathComponent("chat_template.heylook.jinja")
        try template.write(to: overrideURL, atomically: true, encoding: .utf8)
        return inspectChatTemplate(in: candidateDir)
    }

    /// Removes `chat_template.heylook.jinja` from `candidateDir` and returns the reverted template state.
    @discardableResult
    public static func deleteChatTemplateOverride(
        in candidateDir: URL,
        fallbackSource: String = "gguf_embedded"
    ) -> ChatTemplateState {
        let dir = candidateDir.pathExtension.lowercased() == "gguf"
            ? candidateDir.deletingLastPathComponent()
            : candidateDir
        let overrideURL = dir.appendingPathComponent("chat_template.heylook.jinja")
        if FileManager.default.fileExists(atPath: overrideURL.path) {
            try? FileManager.default.removeItem(at: overrideURL)
        }
        return inspectChatTemplate(in: candidateDir, fallbackSource: fallbackSource)
    }

    /// Evaluates the model's real Jinja2 chat template against `turns` and returns the rendered prompt,
    /// special-token markers, continuation flag, and dropped/unrendered media counts.
    public static func renderPromptDetailed(
        in candidateDir: URL?,
        system: String?,
        turns: [MessagesRequest.MessageTurn],
        thinkingEnabled: Bool = false,
        thinkingDepth: String? = nil,
        continuation: Bool? = nil,
        modelSupportsVision: Bool? = nil,
        modelSupportsAudio: Bool? = nil
    ) throws -> PromptRenderResult {
        var source = "default_chatml"
        var templateText = ""
        var templateInfo = ModelTemplateInfo.analyze(templateText: "")
        var bosToken = ""
        var eosToken = ""
        var effectiveVision = modelSupportsVision ?? true
        var effectiveAudio = modelSupportsAudio ?? true

        if let candidateDir {
            let dir = resolveModelDirectoryURL(from: candidateDir.path)
            let state = inspectChatTemplate(in: dir, fallbackSource: "default_chatml")
            templateText = state.template
            source = state.source
            templateInfo = ModelTemplateInfo.inspect(modelDirectory: dir)

            if let tokCfg = loadJSONDictionary(at: dir.appendingPathComponent("tokenizer_config.json")) {
                if let b = tokCfg["bos_token"] as? String { bosToken = b }
                else if let bDict = tokCfg["bos_token"] as? [String: Any], let c = bDict["content"] as? String { bosToken = c }
                if let e = tokCfg["eos_token"] as? String { eosToken = e }
                else if let eDict = tokCfg["eos_token"] as? [String: Any], let c = eDict["content"] as? String { eosToken = c }
            }
            if modelSupportsVision == nil {
                let mmproj = discoverMMProj(in: dir)
                effectiveVision = ModelTemplateInfo.hasVisionConfiguration(in: dir) || mmproj != nil
            }
            if modelSupportsAudio == nil {
                let cfg = loadJSONDictionary(at: dir.appendingPathComponent("config.json"))
                let mmprojHeader = discoverMMProj(in: dir).flatMap { GGUFHeaderReader.readFileHeader(at: URL(fileURLWithPath: $0)) }
                effectiveAudio = (cfg?["audio_config"] != nil) || (mmprojHeader?.hasAudioEncoder == true)
            }
        }

        let effectiveTemplate = templateText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? HeylookJinjaRenderer.defaultChatMLTemplate
            : templateText
        let caps = HeylookJinjaRenderer.detectCapabilities(template: effectiveTemplate)

        let isContinuation = continuation ?? (turns.last?.role == "assistant")
        var droppedMedia = 0
        var unrenderedMedia = 0
        var jinjaMessages: [JinjaValue] = []

        if let sys = system, !sys.isEmpty, turns.first?.role != "system" {
            jinjaMessages.append(.object([
                "role": .string("system"),
                "content": .string(sys),
            ]))
        }

        for turn in turns {
            let orderedParts = turn.parts
            let hasMediaParts = orderedParts.contains {
                if case .media = $0 { return true }
                return false
            }

            var msgObj: [String: JinjaValue] = [
                "role": .string(turn.role),
            ]
            if let thinking = turn.thinking, !thinking.isEmpty {
                msgObj["thinking"] = .string(thinking)
                msgObj["reasoning_content"] = .string(thinking)
            }

            if hasMediaParts {
                var keptParts: [JinjaValue] = []
                for part in orderedParts {
                    switch part {
                    case .text(let text):
                        if !text.isEmpty {
                            keptParts.append(.object([
                                "type": .string("text"),
                                "text": .string(text),
                            ]))
                        }
                    case .thinking(let th):
                        if !th.isEmpty && msgObj["thinking"] == nil {
                            msgObj["thinking"] = .string(th)
                            msgObj["reasoning_content"] = .string(th)
                        }
                    case .media(let media):
                        if media.kind == "image" {
                            if !effectiveVision {
                                droppedMedia += 1
                            } else {
                                if !caps.supportsVision {
                                    unrenderedMedia += 1
                                }
                                keptParts.append(.object(["type": .string("image")]))
                            }
                        } else if media.kind == "audio" {
                            if !effectiveAudio {
                                droppedMedia += 1
                            } else {
                                if !caps.supportsAudio {
                                    unrenderedMedia += 1
                                }
                                keptParts.append(.object(["type": .string("audio")]))
                            }
                        }
                    }
                }
                if keptParts.isEmpty || (!caps.requiresTypedContent && !caps.supportsVision && !caps.supportsAudio) {
                    msgObj["content"] = .string(turn.content)
                } else {
                    msgObj["content"] = .array(keptParts)
                }
            } else {
                msgObj["content"] = .string(turn.content)
            }
            jinjaMessages.append(.object(msgObj))
        }

        var context: [String: JinjaValue] = [
            "messages": .array(jinjaMessages),
            "add_generation_prompt": .bool(!isContinuation),
            "enable_thinking": .bool(thinkingEnabled),
            "bos_token": .string(bosToken),
            "eos_token": .string(eosToken),
            "content": .string(turns.last?.content ?? ""),
        ]
        if let depth = thinkingDepth, !depth.isEmpty {
            if let depthVar = templateInfo.thinkingControls.depthVariable {
                context[depthVar] = .string(depth)
            }
            context["reasoning_effort"] = .string(depth)
        }

        var rendered = try HeylookJinjaRenderer.render(template: effectiveTemplate, context: context)
        if isContinuation {
            let turnEndCandidates = [
                eosToken,
                "<|im_end|>\n", "<|im_end|>",
                "<end_of_turn>\n", "<end_of_turn>",
                "<turn|>\n", "<turn|>",
                "<|eot_id|>", "<|end|>", "<|eot|>",
            ].filter { !$0.isEmpty }
            for suffix in turnEndCandidates {
                if rendered.hasSuffix(suffix) {
                    rendered.removeLast(suffix.count)
                    break
                }
            }
        }

        let markers = HeylookJinjaRenderer.extractMarkers(
            in: rendered,
            specialTokens: templateInfo.specialTokens
        )
        return PromptRenderResult(
            prompt: rendered,
            source: source,
            markers: markers,
            continuation: isContinuation,
            droppedMedia: droppedMedia,
            unrenderedMedia: unrenderedMedia
        )
    }

    /// Renders a model-derived prompt preview by evaluating the model directory's active Jinja2 template
    /// (`chat_template.heylook.jinja`, `chat_template.jinja`, `tokenizer_config.json`, or `gguf_embedded`).
    public static func renderPromptPreview(
        in candidateDir: URL?,
        system: String?,
        messages: [(role: String, content: String, thinking: String?)],
        thinkingEnabled: Bool = false,
        thinkingDepth: String? = nil
    ) -> (prompt: String, source: String) {
        let turns = messages.map {
            MessagesRequest.MessageTurn(role: $0.role, content: $0.content, thinking: $0.thinking)
        }
        do {
            let result = try renderPromptDetailed(
                in: candidateDir,
                system: system,
                turns: turns,
                thinkingEnabled: thinkingEnabled,
                thinkingDepth: thinkingDepth,
                continuation: false
            )
            return (result.prompt, result.source)
        } catch {
            return ("/* Jinja template error: \(error.localizedDescription) */", "template_error")
        }
    }

    /// Convenience overload accepting a `modelPath` string directly.
    public static func renderPromptPreview(
        modelPath: String,
        provider: EngineProviderKind = .mlx,
        turns: [(role: String, content: String, thinking: String?)],
        thinkingEnabled: Bool = false,
        thinkingDepth: String? = nil
    ) -> (prompt: String, source: String) {
        let dirURL = modelPath.isEmpty ? nil : resolveModelDirectoryURL(from: modelPath)
        return renderPromptPreview(
            in: dirURL,
            system: nil,
            messages: turns,
            thinkingEnabled: thinkingEnabled,
            thinkingDepth: thinkingDepth
        )
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

    /// Discovers all models configured in this `HeylookConfig`, combining `scanFolders`
    /// and explicit `[[models]]` entries (`explicitModels` take precedence on matching `modelID` or `modelPath`).
    public func discoverConfiguredModels() -> [EngineDescription] {
        let folderURLs = scanFolders.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        return Self.discoverModels(in: folderURLs, explicitModels: explicitModels)
    }

    /// Scans `scanFolders` for `.safetensors` (MLX), `.aimodel` / `.mlmodelc` (CoreAI / ANE),
    /// and `.gguf` (llama-server) model directories, merging any `model.heylook.toml` sidecar
    /// as well as explicit `[[models]]` entries from `heylook.toml`.
    public static func discoverModels(
        in folders: [URL],
        explicitModels: [ExplicitModelEntry] = []
    ) -> [EngineDescription] {
        var byID: [String: EngineDescription] = [:]
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
                    let urlLower = url.lastPathComponent.lowercased()
                    // Skip auxiliary ANE tower sub-bundles inside an MLX checkpoint's `coreml/` or `ane/` folder
                    if drafterSubfolderNames.contains(parentDir.lastPathComponent.lowercased())
                        || drafterSubfolderNames.contains(urlLower) {
                        enumerator.skipDescendants()
                        continue
                    }
                    guard let resolvedRoot = resolveCoreAIBundleRoot(for: url, scanRoot: expandedRoot) else {
                        continue
                    }
                    if resolvedRoot.standardizedFileURL.path == url.standardizedFileURL.path {
                        enumerator.skipDescendants()
                    }
                    candidateDir = resolvedRoot
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

                let desc = buildEngineDescription(
                    for: candidateDir,
                    detectedProvider: detectedProvider,
                    defaultPlacement: defaultPlacement,
                    primaryGGUFURL: primaryGGUFURL,
                    configJSON: configJSON,
                    explicitID: nil,
                    explicitSidecar: nil
                )
                byID[desc.modelID] = desc
            }
        }

        for entry in explicitModels {
            let expandedURL = URL(fileURLWithPath: (entry.path as NSString).expandingTildeInPath).standardizedFileURL
            let ext = expandedURL.pathExtension.lowercased()
            let candidateDir = ext == "gguf" ? expandedURL.deletingLastPathComponent() : expandedURL
            let primaryGGUF = ext == "gguf" ? expandedURL : firstGGUFFile(in: candidateDir)
            let defaultPlacement: ComputePlacement
            switch entry.provider {
            case .mlx: defaultPlacement = .mlxGPU
            case .coreai: defaultPlacement = .ane
            case .gguf: defaultPlacement = .ggufSubprocess
            case .remote: defaultPlacement = .remote
            }
            let configJSON = loadJSONDictionary(at: candidateDir.appendingPathComponent("config.json"))
            let desc = buildEngineDescription(
                for: candidateDir,
                detectedProvider: entry.provider,
                defaultPlacement: defaultPlacement,
                primaryGGUFURL: primaryGGUF,
                configJSON: configJSON,
                explicitID: entry.id,
                explicitSidecar: entry.sidecar
            )
            // Remove any earlier folder-scanned entry that points at the same directory path
            if let existingSamePath = byID.first(where: { $0.value.modelPath == desc.modelPath })?.key {
                byID.removeValue(forKey: existingSamePath)
            }
            byID[desc.modelID] = desc
        }

        return byID.values.sorted { $0.modelID.localizedStandardCompare($1.modelID) == .orderedAscending }
    }

    /// Resolves the canonical root directory for a discovered `.aimodel` or `.mlmodelc` entry without
    /// ever collapsing a top-level `<model>.aimodel` directory into its parent `scanRoot`.
    private static func resolveCoreAIBundleRoot(for url: URL, scanRoot: URL) -> URL? {
        let fm = FileManager.default
        let standardizedURL = url.standardizedFileURL
        let standardizedScanRoot = scanRoot.standardizedFileURL
        let parentDir = standardizedURL.deletingLastPathComponent()

        var isDir: ObjCBool = false
        let existsAsDir = fm.fileExists(atPath: standardizedURL.path, isDirectory: &isDir) && isDir.boolValue
        if existsAsDir {
            let hasManifest = fm.fileExists(atPath: standardizedURL.appendingPathComponent("metadata.json").path)
            let hasWhisperAssets = fm.fileExists(atPath: standardizedURL.appendingPathComponent("encoder.aimodel").path)
                && fm.fileExists(atPath: standardizedURL.appendingPathComponent("decoder.aimodel").path)
            if hasManifest || hasWhisperAssets || parentDir.path == standardizedScanRoot.path {
                return standardizedURL
            }
        }
        guard parentDir.path != standardizedScanRoot.path else {
            return existsAsDir ? standardizedURL : nil
        }
        return parentDir
    }

    /// Inspects a Core AI bundle directory for `speech_recognizer` metadata (`parakeet_tdt` / `whisper`)
    /// or legacy Whisper `encoder.aimodel` + `decoder.aimodel` assets.
    public static func inspectSpeechRecognizerArchitecture(in bundleDir: URL) -> String? {
        let fm = FileManager.default
        if let meta = loadJSONDictionary(at: bundleDir.appendingPathComponent("metadata.json")),
           let kind = (meta["kind"] as? String)?.lowercased(),
           kind == "speech_recognizer" {
            if let cfg = meta["config"] as? [String: Any],
               let arch = cfg["architecture"] as? String,
               !arch.isEmpty {
                return arch.lowercased()
            }
            return "whisper"
        }
        let hasWhisperEncoder = fm.fileExists(atPath: bundleDir.appendingPathComponent("encoder.aimodel").path)
        let hasWhisperDecoder = fm.fileExists(atPath: bundleDir.appendingPathComponent("decoder.aimodel").path)
        if hasWhisperEncoder && hasWhisperDecoder {
            return "whisper"
        }
        return nil
    }

    private static func buildEngineDescription(
        for candidateDir: URL,
        detectedProvider: EngineProviderKind,
        defaultPlacement: ComputePlacement,
        primaryGGUFURL: URL?,
        configJSON: [String: Any]?,
        explicitID: String?,
        explicitSidecar: ModelSidecarConfig?
    ) -> EngineDescription {
        let fm = FileManager.default
        let dirPath = candidateDir.standardizedFileURL.path
        let isVariant = (detectedProvider == .gguf && primaryGGUFURL != nil)
            ? isGGUFVariantDirectory(candidateDir, primaryGGUFFile: primaryGGUFURL!)
            : false
        let repoRootDir = isVariant ? candidateDir.deletingLastPathComponent() : candidateDir

        let sidecarURL = fm.fileExists(atPath: candidateDir.appendingPathComponent("model.heylook.toml").path)
            ? candidateDir.appendingPathComponent("model.heylook.toml")
            : repoRootDir.appendingPathComponent("model.heylook.toml")
        let fileSidecar: ModelSidecarConfig? = (try? String(contentsOf: sidecarURL, encoding: .utf8))
            .map { ModelSidecarConfig.parseTOML($0) }
        let sidecar = explicitSidecar ?? fileSidecar

        let defaultID: String
        if isVariant, let primaryGGUFURL {
            defaultID = ggufModelStem(from: primaryGGUFURL.lastPathComponent)
        } else {
            defaultID = candidateDir.lastPathComponent
        }
        let modelID = explicitID ?? sidecar?.id ?? defaultID
        let provider = sidecar?.provider ?? detectedProvider
        let placementValue = sidecar?.computePlacement ?? defaultPlacement
        let placementProv: ProvenanceSource = sidecar?.computePlacement != nil ? .configured : .derived

        let ggufHeader = primaryGGUFURL.flatMap { GGUFHeaderReader.readFileHeader(at: $0) }
        let speechBundleArch = provider == .coreai ? inspectSpeechRecognizerArchitecture(in: candidateDir) : nil
        let isSpeechBundle = speechBundleArch != nil
        let coreAIMeta = provider == .coreai ? loadJSONDictionary(at: candidateDir.appendingPathComponent("metadata.json")) : nil
        let coreAILangArch = ((coreAIMeta?["language"] as? [String: Any])?["architecture"] as? String)
            ?? (coreAIMeta?["model_type"] as? String)
        let archString = speechBundleArch
            ?? (configJSON?["model_type"] as? String)
            ?? (configJSON?["architectures"] as? [String])?.first
            ?? coreAILangArch
            ?? ggufHeader?.architecture
            ?? provider.rawValue
        let isEmbedding = !isSpeechBundle && (
            modelID.localizedCaseInsensitiveContains("embed")
            || archString.localizedCaseInsensitiveContains("embedding")
            || (coreAIMeta?["kind"] as? String)?.lowercased() == "embedding"
        )
        let mmprojPath = provider == .gguf ? discoverMMProj(in: candidateDir) : nil
        let mmprojHeader = mmprojPath.flatMap { GGUFHeaderReader.readFileHeader(at: URL(fileURLWithPath: $0)) }
        let rawHasVision = !isSpeechBundle && (
            ModelTemplateInfo.hasVisionConfiguration(in: candidateDir, configJSON: configJSON)
            || mmprojHeader?.hasVisionEncoder == true
            || mmprojPath != nil
            || isEmbedding
        )
        let hasVision = sidecar?.supportsVision ?? rawHasVision
        let hasAudio = isSpeechBundle
            || configJSON?["audio_config"] != nil
            || mmprojHeader?.hasAudioEncoder == true
            || isEmbedding
        let coreAIMaxCtx = (coreAIMeta?["maxContextLength"] as? Int)
            ?? (coreAIMeta?["max_context_length"] as? Int)
            ?? ((coreAIMeta?["language"] as? [String: Any])?["max_context_length"] as? Int)
        let ctxValue = sidecar?.ctxSize
            ?? (configJSON?["max_position_embeddings"] as? Int)
            ?? ((configJSON?["text_config"] as? [String: Any])?["max_position_embeddings"] as? Int)
            ?? coreAIMaxCtx
            ?? ggufHeader?.contextLength
            ?? (isSpeechBundle ? 448 : 8192)

        let templateInfo = ModelTemplateInfo.inspect(
            modelDirectory: candidateDir,
            modelID: modelID
        )
        let templateField: ProvenanceField<String>
        if provider == .gguf {
            let ownTemplate = resolveGGUFTemplate(in: candidateDir, sidecar: sidecar)
            if ownTemplate.source.value == "gguf_embedded" && isVariant {
                templateField = resolveGGUFTemplate(in: repoRootDir, sidecar: sidecar).source
            } else {
                templateField = ownTemplate.source
            }
        } else if isSpeechBundle {
            templateField = ProvenanceField(
                value: "speech_recognizer",
                provenance: .notApplicable,
                effect: .reload
            )
        } else {
            let resolvedSource = templateInfo.templateSource == "default_chatml"
                ? "tokenizer_config.json"
                : templateInfo.templateSource
            templateField = ProvenanceField(
                value: resolvedSource,
                provenance: resolvedSource == "chat_template.heylook.jinja" ? .configured : .derived,
                effect: .reload
            )
        }
        let derivedSupportsThinking: Bool
        if isEmbedding || isSpeechBundle {
            derivedSupportsThinking = false
        } else if templateInfo.templateSource != "default_chatml" {
            derivedSupportsThinking = templateInfo.thinkingControls.supportsThinking
        } else {
            derivedSupportsThinking = true
        }

        var tags = sidecar?.tags ?? []
        if isSpeechBundle {
            if !tags.contains("asr") { tags.append("asr") }
            if !tags.contains("speech") { tags.append("speech") }
        }
        let drafter = discoverDrafter(in: candidateDir, provider: provider, sidecar: sidecar)
        if drafter.draftModelPath != nil || drafter.specType != nil || ggufHeader?.hasBuiltInMTP == true {
            if !tags.contains("mtp") {
                tags.append("mtp")
            }
        }
        if hasVision && !tags.contains("vision") {
            tags.append("vision")
        }
        if hasAudio && !isEmbedding && !tags.contains("audio") {
            tags.append("audio")
        }

        let drafterVal: String? = drafter.draftModelPath
            ?? drafter.specType
            ?? (ggufHeader?.hasBuiltInMTP == true ? "draft-mtp" : nil)
        let drafterProv: ProvenanceSource
        if sidecar?.draftModel != nil {
            drafterProv = .configured
        } else if drafterVal != nil {
            drafterProv = .derived
        } else {
            drafterProv = .notApplicable
        }
        let estimatedMB = estimateModelMemoryMB(
            in: candidateDir,
            primaryGGUFURL: primaryGGUFURL,
            drafterPath: drafter.draftModelPath,
            mmprojPath: mmprojPath
        )
        let quantValue = inferQuantization(from: modelID, configJSON: configJSON)
        let archField = ProvenanceField(value: archString, provenance: .derived, effect: .reload)
        let quantField = ProvenanceField(value: quantValue, provenance: .derived, effect: .reload)
        let placementField = ProvenanceField(
            value: placementValue,
            provenance: placementProv,
            effect: .reload,
            computePlacement: placementValue
        )
        let ctxField = ProvenanceField(
            value: ctxValue,
            provenance: sidecar?.ctxSize != nil ? .configured : .derived,
            effect: .reload
        )
        let faField = ProvenanceField(
            value: sidecar?.flashAttn ?? true,
            provenance: sidecar?.flashAttn != nil ? .configured : .derived,
            effect: .reload
        )
        let visionField = ProvenanceField(
            value: hasVision,
            provenance: sidecar?.supportsVision != nil ? .configured : .derived,
            effect: .reload
        )
        let audioField = ProvenanceField(value: hasAudio, provenance: .derived, effect: .reload)
        let embField = ProvenanceField(value: isEmbedding, provenance: .derived, effect: .reload)
        let thinkField = ProvenanceField(
            value: sidecar?.supportsThinking ?? derivedSupportsThinking,
            provenance: sidecar?.supportsThinking != nil ? .configured : .derived,
            effect: .perRequest
        )
        let budgetField = ProvenanceField(
            value: sidecar?.sampling.thinkingBudget,
            provenance: sidecar?.sampling.thinkingBudget != nil ? .configured : .notApplicable,
            effect: .perRequest
        )
        let reasoningField = ProvenanceField(
            value: templateInfo.reasoningFormat.wireIdentifier,
            provenance: .derived,
            effect: .reload
        )
        let specDrafterField = ProvenanceField(
            value: drafterVal,
            provenance: drafterProv,
            effect: .reload
        )

        return EngineDescription(
            modelID: modelID,
            provider: provider,
            modelPath: dirPath,
            architecture: archField,
            quantization: quantField,
            computePlacement: placementField,
            isPinned: sidecar?.pinned ?? false,
            isLoaded: false,
            contextSize: ctxField,
            flashAttention: faField,
            supportsVision: visionField,
            supportsAudio: audioField,
            supportsEmbeddings: embField,
            supportsThinking: thinkField,
            thinkingBudget: budgetField,
            reasoningFormat: reasoningField,
            thinkingControls: templateInfo.thinkingControls,
            templateSource: templateField,
            speculativeDrafter: specDrafterField,
            estimatedMemoryMB: estimatedMB,
            tags: tags
        )
    }

    /// Computes total weight file footprint in MB from `.safetensors`, `.gguf`, `.aimodel`, `.mlmodelc`, and `.bin` files on disk.
    public static func estimateModelMemoryMB(
        in candidateDir: URL,
        primaryGGUFURL: URL? = nil,
        drafterPath: String? = nil,
        mmprojPath: String? = nil
    ) -> Int {
        let fm = FileManager.default
        let weightExtensions: Set<String> = ["safetensors", "gguf", "bin", "mlx"]
        var totalBytes: Int64 = 0
        var visited = Set<String>()

        func addFileSize(_ url: URL) {
            let std = url.standardizedFileURL.path
            guard !visited.contains(std) else { return }
            visited.insert(std)
            if let attrs = try? fm.attributesOfItem(atPath: std),
               let sz = (attrs[.size] as? NSNumber)?.int64Value,
               sz > 0 {
                totalBytes += sz
            }
        }

        if let primaryGGUFURL {
            let stem = ggufModelStem(from: primaryGGUFURL.lastPathComponent).lowercased()
            let folder = primaryGGUFURL.deletingLastPathComponent()
            if let items = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.fileSizeKey]) {
                for item in items where item.pathExtension.lowercased() == "gguf" {
                    let itemStem = ggufModelStem(from: item.lastPathComponent).lowercased()
                    if itemStem == stem || item.standardizedFileURL.path == primaryGGUFURL.standardizedFileURL.path {
                        addFileSize(item)
                    }
                }
            } else {
                addFileSize(primaryGGUFURL)
            }
        } else if let enumerator = fm.enumerator(
            at: candidateDir,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) {
            for case let fileURL as URL in enumerator {
                let ext = fileURL.pathExtension.lowercased()
                let pathLower = fileURL.path.lowercased()
                if weightExtensions.contains(ext)
                    || pathLower.contains(".aimodel/")
                    || pathLower.contains(".mlmodelc/") {
                    addFileSize(fileURL)
                }
            }
        }

        if let mmprojPath, !mmprojPath.isEmpty {
            addFileSize(URL(fileURLWithPath: mmprojPath))
        }
        if let drafterPath, !drafterPath.isEmpty, drafterPath != candidateDir.standardizedFileURL.path {
            let dURL = URL(fileURLWithPath: drafterPath)
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: dURL.path, isDirectory: &isDir) {
                if isDir.boolValue {
                    if let items = try? fm.contentsOfDirectory(at: dURL, includingPropertiesForKeys: nil) {
                        for item in items where weightExtensions.contains(item.pathExtension.lowercased()) {
                            addFileSize(item)
                        }
                    }
                } else {
                    addFileSize(dURL)
                }
            }
        }

        guard totalBytes > 0 else { return 0 }
        return max(1, Int(totalBytes / 1_048_576))
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
            .first(where: { url in
                let lower = url.lastPathComponent.lowercased()
                return url.pathExtension.lowercased() == "gguf"
                    && !lower.contains("mmproj")
                    && !lower.hasPrefix("draft")
                    && !lower.contains("-draft")
                    && !lower.hasPrefix("mtp-")
                    && !lower.contains("-mtp")
                    && !lower.hasPrefix("eagle")
                    && !lower.contains("-dspark-")
            })
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
/// tensor/KV counts, and string/integer/boolean/float metadata without spawning a C++ process.
public enum GGUFHeaderReader {
    public struct HeaderSummary: Sendable, Equatable {
        public let version: UInt32
        public let tensorCount: UInt64
        public let metadataKVCount: UInt64
        public let stringMetadata: [String: String]
        public let uintMetadata: [String: UInt64]
        public let boolMetadata: [String: Bool]
        public let doubleMetadata: [String: Double]

        public init(
            version: UInt32,
            tensorCount: UInt64,
            metadataKVCount: UInt64,
            stringMetadata: [String: String] = [:],
            uintMetadata: [String: UInt64] = [:],
            boolMetadata: [String: Bool] = [:],
            doubleMetadata: [String: Double] = [:]
        ) {
            self.version = version
            self.tensorCount = tensorCount
            self.metadataKVCount = metadataKVCount
            self.stringMetadata = stringMetadata
            self.uintMetadata = uintMetadata
            self.boolMetadata = boolMetadata
            self.doubleMetadata = doubleMetadata
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

        public var hasVisionEncoder: Bool {
            boolMetadata["clip.has_vision_encoder"] == true
        }

        public var hasAudioEncoder: Bool {
            boolMetadata["clip.has_audio_encoder"] == true
        }

        public var clipProjectorType: String? {
            stringMetadata["clip.projector_type"]
        }

        /// True when `clip.projector_type` is a non-causal vision projector (`gemma4v`, `qwen2vl`,
        /// `qwen2.5vl`, `qwen3vl`, `pixtral`, `internvl`) where `llama-server` requires
        /// `--image-max-tokens <= n_ubatch` so `llama_context::decode` does not abort.
        public var hasNonCausalVisionProjector: Bool {
            guard let proj = clipProjectorType?.lowercased() else { return false }
            let nonCausal = ["gemma4", "qwen2vl", "qwen2.5vl", "qwen3vl", "qwen3_5", "pixtral", "internvl", "minicpmv"]
            return nonCausal.contains(where: { proj.contains($0) })
        }

        /// True when the GGUF header declares a hybrid-recurrent SSM architecture (`*.ssm.conv_kernel`,
        /// `*.ssm.inner_size`, or `*.ssm.state_size`), requiring `--cache-ram 8192` in `llama-server`.
        public var isHybridRecurrent: Bool {
            uintMetadata.keys.contains { key in
                key.hasSuffix(".ssm.conv_kernel")
                    || key.hasSuffix(".ssm.inner_size")
                    || key.hasSuffix(".ssm.state_size")
            }
        }

        /// True when the GGUF header declares built-in Multi-Token Prediction layers (`*.nextn_predict_layers > 0`).
        public var hasBuiltInMTP: Bool {
            uintMetadata.contains { key, val in
                key.hasSuffix(".nextn_predict_layers") && val > 0
            }
        }

        /// Extracts vendor sampling recommendations (`general.sampling.*`) embedded in the GGUF header.
        public var vendorSampling: SamplerSettings? {
            let temp = doubleMetadata["general.sampling.temp"]
            let topP = doubleMetadata["general.sampling.top_p"]
            let minP = doubleMetadata["general.sampling.min_p"]
            let topK = uintMetadata["general.sampling.top_k"].map(Int.init)
            let repPen = doubleMetadata["general.sampling.penalty_repeat"]
            guard temp != nil || topP != nil || minP != nil || topK != nil || repPen != nil else {
                return nil
            }
            return SamplerSettings(
                temperature: temp,
                topP: topP,
                topK: topK,
                minP: minP,
                repetitionPenalty: repPen
            )
        }
    }

    public static func readFileHeader(at url: URL, maxBytes: Int = 67_108_864) -> HeaderSummary? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: maxBytes) else { return nil }
        return readHeader(from: data)
    }

    public static func readHeader(from data: Data) -> HeaderSummary? {
        guard data.count >= 24 else { return nil }
        let magic = String(decoding: data.prefix(4), as: UTF8.self)
        guard magic == "GGUF" else { return nil }

        return data.withUnsafeBytes { rawBuf -> HeaderSummary? in
            let totalBytes = rawBuf.count
            guard let base = rawBuf.baseAddress else { return nil }

            func loadLE<T: FixedWidthInteger>(_ type: T.Type, at pos: Int) -> T {
                base.advanced(by: pos).loadUnaligned(as: T.self).littleEndian
            }

            let version = loadLE(UInt32.self, at: 4)
            guard version == 2 || version == 3 else { return nil }
            let tensorCount = loadLE(UInt64.self, at: 8)
            let kvCount = loadLE(UInt64.self, at: 16)

            var offset = 24
            var strings: [String: String] = [:]
            var uints: [String: UInt64] = [:]
            var bools: [String: Bool] = [:]
            var doubles: [String: Double] = [:]

            func readUInt32() -> UInt32? {
                guard offset + 4 <= totalBytes else { return nil }
                let val = loadLE(UInt32.self, at: offset)
                offset += 4
                return val
            }

            func readUInt64() -> UInt64? {
                guard offset + 8 <= totalBytes else { return nil }
                let val = loadLE(UInt64.self, at: offset)
                offset += 8
                return val
            }

            func readString() -> String? {
                guard let len = readUInt64() else { return nil }
                guard len <= 4_194_304, offset + Int(len) <= totalBytes else { return nil }
                let slice = UnsafeRawBufferPointer(rebasing: rawBuf[offset..<(offset + Int(len))])
                offset += Int(len)
                return String(decoding: slice, as: UTF8.self)
            }

            func skipString() -> Bool {
                guard let len = readUInt64() else { return false }
                guard len <= 4_194_304, offset + Int(len) <= totalBytes else { return false }
                offset += Int(len)
                return true
            }

            func fixedElementSize(for type: UInt32) -> Int? {
                switch type {
                case 0, 1, 7: return 1
                case 2, 3: return 2
                case 4, 5, 6: return 4
                case 10, 11, 12: return 8
                default: return nil
                }
            }

            func skipOrReadValue(type: UInt32, key: String?) -> Bool {
                switch type {
                case 0: // UINT8
                    guard offset + 1 <= totalBytes else { return false }
                    let b = rawBuf[offset]
                    offset += 1
                    if let key { uints[key] = UInt64(b) }
                    return true
                case 1: // INT8
                    guard offset + 1 <= totalBytes else { return false }
                    offset += 1
                    return true
                case 7: // BOOL
                    guard offset + 1 <= totalBytes else { return false }
                    let b = rawBuf[offset]
                    offset += 1
                    if let key { bools[key] = (b != 0) }
                    return true
                case 2: // UINT16
                    guard offset + 2 <= totalBytes else { return false }
                    let v = loadLE(UInt16.self, at: offset)
                    offset += 2
                    if let key { uints[key] = UInt64(v) }
                    return true
                case 3: // INT16
                    guard offset + 2 <= totalBytes else { return false }
                    offset += 2
                    return true
                case 4: // UINT32
                    guard let v = readUInt32() else { return false }
                    if let key { uints[key] = UInt64(v) }
                    return true
                case 5: // INT32
                    guard offset + 4 <= totalBytes else { return false }
                    offset += 4
                    return true
                case 6: // FLOAT32
                    guard let bits = readUInt32() else { return false }
                    if let key {
                        doubles[key] = Double(Float(bitPattern: bits))
                    }
                    return true
                case 8: // STRING
                    if let key {
                        guard let str = readString() else { return false }
                        strings[key] = str
                        return true
                    } else {
                        return skipString()
                    }
                case 9: // ARRAY (supports 256k+ vocab arrays without String allocations)
                    guard let elemType = readUInt32(), let count = readUInt64(), count <= 4_000_000 else {
                        return false
                    }
                    if let elemSize = fixedElementSize(for: elemType) {
                        let byteSpan = Int(count) * elemSize
                        guard byteSpan >= 0, offset + byteSpan <= totalBytes else { return false }
                        offset += byteSpan
                        return true
                    }
                    if elemType == 8 {
                        for _ in 0..<count {
                            if !skipString() { return false }
                        }
                        return true
                    }
                    for _ in 0..<count {
                        if !skipOrReadValue(type: elemType, key: nil) { return false }
                    }
                    return true
                case 10: // UINT64
                    guard let v = readUInt64() else { return false }
                    if let key { uints[key] = v }
                    return true
                case 11: // INT64
                    guard offset + 8 <= totalBytes else { return false }
                    offset += 8
                    return true
                case 12: // FLOAT64
                    guard let bits = readUInt64() else { return false }
                    if let key {
                        doubles[key] = Double(bitPattern: bits)
                    }
                    return true
                default:
                    return false
                }
            }

            let maxEntries = min(kvCount, 1_000_000)
            for _ in 0..<maxEntries {
                guard offset < totalBytes else { break }
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
                uintMetadata: uints,
                boolMetadata: bools,
                doubleMetadata: doubles
            )
        }
    }

    /// Builds a valid GGUF v2/v3 binary header with embedded string, string-array, uint32, bool, and float32 KV entries.
    public static func buildMinimalGGUF(
        version: UInt32 = 3,
        tensorCount: UInt64 = 0,
        stringMetadata: [String: String] = [:],
        stringArrayMetadata: [String: [String]] = [:],
        uint32Metadata: [String: UInt32] = [:],
        boolMetadata: [String: Bool] = [:],
        float32Metadata: [String: Float] = [:]
    ) -> Data {
        var data = Data("GGUF".utf8)
        var verLE = version.littleEndian
        var tcLE = tensorCount.littleEndian
        let totalKV = UInt64(
            stringArrayMetadata.count
                + stringMetadata.count
                + uint32Metadata.count
                + boolMetadata.count
                + float32Metadata.count
        )
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

        func appendUInt64(_ v: UInt64) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }

        // Emit array metadata FIRST (matching llama.cpp / HF GGUF layout where `tokenizer.ggml.tokens` precedes later KV entries)
        for (k, arr) in stringArrayMetadata.sorted(by: { $0.key < $1.key }) {
            appendString(k)
            appendUInt32(9) // ARRAY
            appendUInt32(8) // elemType = STRING
            appendUInt64(UInt64(arr.count))
            for elem in arr {
                appendString(elem)
            }
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
        for (k, v) in boolMetadata.sorted(by: { $0.key < $1.key }) {
            appendString(k)
            appendUInt32(7) // BOOL
            data.append(v ? 1 : 0)
        }
        for (k, v) in float32Metadata.sorted(by: { $0.key < $1.key }) {
            appendString(k)
            appendUInt32(6) // FLOAT32
            appendUInt32(v.bitPattern)
        }
        return data
    }
}

/// Top-level convenience alias for `HeylookConfig.ChatTemplateState`.
public typealias ChatTemplateInspection = HeylookConfig.ChatTemplateState

/// Structured JSONL observability & telemetry event log ring (`api-and-store.md`).
public final class HeylookObservabilityLog: @unchecked Sendable {
    public struct Entry: Codable, Sendable, Equatable, Identifiable {
        public var id: String { "\(timestampMs)-\(category)-\(requestID ?? "")-\(message.hashValue)" }
        public let timestampMs: Int64
        public let level: String
        public let category: String
        public let message: String
        public let requestID: String?
        public let metadata: [String: String]

        enum CodingKeys: String, CodingKey {
            case timestampMs = "timestamp_ms"
            case level
            case category
            case message
            case requestID = "request_id"
            case metadata
        }

        public init(
            timestampMs: Int64 = SQLiteStore.nowMillis(),
            level: String = "INFO",
            category: String = "server",
            message: String,
            requestID: String? = nil,
            metadata: [String: String] = [:]
        ) {
            self.timestampMs = timestampMs
            self.level = HeylookObservabilityLog.normalizeLevel(level)
            self.category = category
            self.message = message
            self.requestID = requestID
            self.metadata = metadata
        }
    }

    public static let shared = HeylookObservabilityLog()

    private let lock = NSLock()
    private var ring: [Entry] = []
    private var maxEntriesStorage: Int
    private var minimumLevelStorage: String
    public let logFileURL: URL?

    public static func normalizeLevel(_ raw: String) -> String {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() {
        case "DEBUG", "TRACE":
            return "DEBUG"
        case "WARN", "WARNING":
            return "WARN"
        case "ERROR", "ERR", "CRITICAL", "FATAL":
            return "ERROR"
        default:
            return "INFO"
        }
    }

    public static func severityRank(for level: String) -> Int {
        switch normalizeLevel(level) {
        case "DEBUG": return 0
        case "INFO": return 1
        case "WARN": return 2
        case "ERROR": return 3
        default: return 1
        }
    }

    /// Active minimum severity threshold (`DEBUG`, `INFO`, `WARN`, `ERROR`).
    public var level: String {
        get {
            lock.lock()
            defer { lock.unlock() }
            return minimumLevelStorage
        }
        set {
            setLevel(newValue)
        }
    }

    /// Maximum number of structured events retained in the in-memory ring buffer.
    public var retentionLimit: Int {
        get {
            lock.lock()
            defer { lock.unlock() }
            return maxEntriesStorage
        }
        set {
            setRetentionLimit(newValue)
        }
    }

    /// Snapshot of all currently retained structured events in chronological order.
    public var recentEvents: [Entry] {
        lock.lock()
        defer { lock.unlock() }
        return ring
    }

    public init(maxEntries: Int = 1000, level: String = "INFO", logFileURL: URL? = nil) {
        self.maxEntriesStorage = max(16, maxEntries)
        self.minimumLevelStorage = Self.normalizeLevel(level)
        if let logFileURL {
            self.logFileURL = logFileURL
        } else if let envPath = ProcessInfo.processInfo.environment["HEYLOOK_LOG_PATH"], !envPath.isEmpty {
            self.logFileURL = URL(fileURLWithPath: (envPath as NSString).expandingTildeInPath)
        } else {
            self.logFileURL = HeylookConfig.defaultAppSupportDirectory.appendingPathComponent("heylook_events.jsonl")
        }
    }

    public func setLevel(_ newLevel: String) {
        let normalized = Self.normalizeLevel(newLevel)
        lock.lock()
        minimumLevelStorage = normalized
        lock.unlock()
    }

    public func setRetentionLimit(_ limit: Int) {
        let clamped = max(16, min(50_000, limit))
        lock.lock()
        maxEntriesStorage = clamped
        if ring.count > clamped {
            ring.removeFirst(ring.count - clamped)
        }
        lock.unlock()
    }

    public func record(
        level: String = "INFO",
        category: String = "server",
        message: String,
        requestID: String? = nil,
        metadata: [String: String] = [:]
    ) {
        let normalized = Self.normalizeLevel(level)
        let incomingRank = Self.severityRank(for: normalized)

        lock.lock()
        let activeMinRank = Self.severityRank(for: minimumLevelStorage)
        guard incomingRank >= activeMinRank else {
            lock.unlock()
            return
        }
        let entry = Entry(
            level: normalized,
            category: category,
            message: message,
            requestID: requestID,
            metadata: metadata
        )
        ring.append(entry)
        if ring.count > maxEntriesStorage {
            ring.removeFirst(ring.count - maxEntriesStorage)
        }
        let targetURL = logFileURL
        lock.unlock()

        if let targetURL,
           let encoded = try? JSONEncoder().encode(entry),
           var line = String(data: encoded, encoding: .utf8)?.data(using: .utf8) {
            line.append(0x0A)
            try? FileManager.default.createDirectory(
                at: targetURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if FileManager.default.fileExists(atPath: targetURL.path),
               let handle = try? FileHandle(forWritingTo: targetURL) {
                defer { try? handle.close() }
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: line)
            } else {
                try? line.write(to: targetURL, options: .atomic)
            }
        }
    }

    public func recentEntries(
        limit: Int = 200,
        level: String? = nil,
        category: String? = nil,
        search: String? = nil
    ) -> [Entry] {
        lock.lock()
        let snapshot = ring
        let cap = maxEntriesStorage
        lock.unlock()

        let minRank = level.map { Self.severityRank(for: $0) }
        let normalizedCat = category?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let query = search?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        let filtered = snapshot.filter { entry in
            if let minRank, Self.severityRank(for: entry.level) < minRank {
                return false
            }
            if let normalizedCat, !normalizedCat.isEmpty, entry.category.lowercased() != normalizedCat {
                return false
            }
            if let query, !query.isEmpty {
                let matchesMsg = entry.message.lowercased().contains(query)
                let matchesCat = entry.category.lowercased().contains(query)
                let matchesReq = entry.requestID?.lowercased().contains(query) == true
                let matchesMeta = entry.metadata.contains {
                    $0.key.lowercased().contains(query) || $0.value.lowercased().contains(query)
                }
                if !matchesMsg && !matchesCat && !matchesReq && !matchesMeta {
                    return false
                }
            }
            return true
        }
        let clamped = max(1, min(limit, cap))
        return Array(filtered.suffix(clamped))
    }

    public func clear() {
        lock.lock()
        ring.removeAll()
        let targetURL = logFileURL
        lock.unlock()
        if let targetURL {
            try? FileManager.default.removeItem(at: targetURL)
        }
    }
}
