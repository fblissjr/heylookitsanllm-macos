import Foundation

/// Path sanitization helpers ensuring LAN-facing wire responses (`/v1/models` and `EngineContract`)
/// never leak absolute or tilde-prefixed host filesystem paths.
public enum EngineSanitization {
    /// Returns `true` when `raw` looks like an absolute or home-relative filesystem path.
    public static func isPathLike(_ raw: String) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.hasPrefix("/") || trimmed.hasPrefix("~")
    }

    /// Reduces a path string to its basename (`lastPathComponent`), leaving non-path strings intact.
    /// Matches `contract.py:public_value`.
    public static func publicPathString(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isPathLike(trimmed) else {
            // Also sanitize flag=value tokens such as `--chat-template-file=/abs/path/t.jinja`.
            if let eqIdx = trimmed.firstIndex(of: "=") {
                let rhs = String(trimmed[trimmed.index(after: eqIdx)...])
                if isPathLike(rhs) {
                    let lhs = String(trimmed[...eqIdx])
                    return lhs + basename(of: rhs)
                }
            }
            return raw
        }
        return basename(of: trimmed)
    }

    private static func basename(of path: String) -> String {
        if path == "/" || path == "~" { return path }
        let name = URL(fileURLWithPath: path).lastPathComponent
        return name.isEmpty ? path : name
    }

    /// Recursively replaces any path-like `.string` in a `JSONValue` with its basename.
    public static func publicValue(_ value: JSONValue) -> JSONValue {
        switch value {
        case .null, .bool, .int, .double:
            return value
        case .string(let s):
            return .string(publicPathString(s))
        case .array(let items):
            return .array(items.map(publicValue))
        case .object(let map):
            var out: [String: JSONValue] = [:]
            out.reserveCapacity(map.count)
            for (k, v) in map {
                out[k] = publicValue(v)
            }
            return .object(out)
        }
    }

    /// Optional convenience overload.
    public static func publicValue(_ value: JSONValue?) -> JSONValue? {
        value.map(publicValue)
    }

    /// Optional string helper returning only the basename when path-like.
    public static func publicString(_ value: String?) -> String? {
        value.map(publicPathString)
    }
}

/// Fine-grained effect class matching `heylook_llm/config.py:EFFECT_CLASSES` and `engine.js:EFFECT_GROUPS`.
public enum ConfigFieldEffect: String, Codable, CaseIterable, Sendable, Hashable {
    case perRequest = "per_request"
    case appliesLive = "applies_live"
    case descriptive = "descriptive"
    case requiresReload = "requires_reload"
    case loadTimeOnly = "load_time_only"
    case identity = "identity"

    /// Bridges to the coarse Wave 1 `EffectClass` used by existing UI components.
    public var legacyEffectClass: EffectClass {
        switch self {
        case .perRequest, .appliesLive, .descriptive:
            return .perRequest
        case .requiresReload, .loadTimeOnly:
            return .reload
        case .identity:
            return .restart
        }
    }
}

/// Truthful provenance leaf: `{value, provenance, source}`.
/// When `provenance` is `.unknown` or `.notApplicable`, `value` is forced to `nil` and explicitly
/// encoded as JSON `null` so schema consumers always see `"value": null` (`providers/contract.py:Fact`).
public struct Fact<Value: Codable & Sendable & Equatable>: Codable, Sendable, Equatable {
    public let value: Value?
    public let provenance: ProvenanceSource
    public let source: String?

    enum CodingKeys: String, CodingKey {
        case value
        case provenance
        case source
    }

    public init(
        value: Value? = nil,
        provenance: ProvenanceSource = .unknown,
        source: String? = nil
    ) {
        if provenance == .unknown || provenance == .notApplicable {
            self.value = nil
        } else {
            self.value = value
        }
        self.provenance = provenance
        self.source = source.map(EngineSanitization.publicPathString)
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let provenance = try c.decodeIfPresent(ProvenanceSource.self, forKey: .provenance) ?? .unknown
        let decodedValue = try c.decodeIfPresent(Value.self, forKey: .value)
        let source = try c.decodeIfPresent(String.self, forKey: .source)
        self.init(value: decodedValue, provenance: provenance, source: source)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        if let value {
            try c.encode(value, forKey: .value)
        } else {
            try c.encodeNil(forKey: .value)
        }
        try c.encode(provenance, forKey: .provenance)
        if let source {
            try c.encode(source, forKey: .source)
        } else {
            try c.encodeNil(forKey: .source)
        }
    }
}

/// One configuration setting or load decision in `EngineContract.settings`.
/// Matches `providers/contract.py:Setting` (`{value, configured, auto, reason, provenance, effect}`).
public struct SettingFact: Codable, Sendable, Equatable {
    public let value: JSONValue?
    public let configured: JSONValue?
    public let auto: JSONValue?
    public let reason: String
    public let provenance: ProvenanceSource
    public let effect: ConfigFieldEffect?

    enum CodingKeys: String, CodingKey {
        case value
        case configured
        case auto
        case reason
        case provenance
        case effect
    }

    public init(
        value: JSONValue? = nil,
        configured: JSONValue? = nil,
        auto: JSONValue? = nil,
        reason: String,
        provenance: ProvenanceSource,
        effect: ConfigFieldEffect? = nil
    ) {
        self.value = EngineSanitization.publicValue(value)
        self.configured = EngineSanitization.publicValue(configured)
        self.auto = EngineSanitization.publicValue(auto)
        self.reason = reason
        self.provenance = provenance
        self.effect = effect
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.value = EngineSanitization.publicValue(try c.decodeIfPresent(JSONValue.self, forKey: .value))
        self.configured = EngineSanitization.publicValue(try c.decodeIfPresent(JSONValue.self, forKey: .configured))
        self.auto = EngineSanitization.publicValue(try c.decodeIfPresent(JSONValue.self, forKey: .auto))
        self.reason = try c.decode(String.self, forKey: .reason)
        self.provenance = try c.decode(ProvenanceSource.self, forKey: .provenance)
        self.effect = try c.decodeIfPresent(ConfigFieldEffect.self, forKey: .effect)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        if let value { try c.encode(value, forKey: .value) } else { try c.encodeNil(forKey: .value) }
        if let configured { try c.encode(configured, forKey: .configured) } else { try c.encodeNil(forKey: .configured) }
        if let auto { try c.encode(auto, forKey: .auto) } else { try c.encodeNil(forKey: .auto) }
        try c.encode(reason, forKey: .reason)
        try c.encode(provenance, forKey: .provenance)
        if let effect { try c.encode(effect, forKey: .effect) } else { try c.encodeNil(forKey: .effect) }
    }

    /// Returns a copy overriding `value`, `auto`, and `reason`.
    public func updating(value: JSONValue?, auto: JSONValue?, reason: String) -> SettingFact {
        SettingFact(
            value: value,
            configured: configured,
            auto: auto,
            reason: reason,
            provenance: provenance,
            effect: effect
        )
    }
}

/// Context ceiling and running-allocation facts (`providers/contract.py:ContextFacts`).
public struct EngineContextFacts: Codable, Sendable, Equatable {
    public let length: Fact<Int>
    public let running: Fact<Int>

    public init(length: Fact<Int>, running: Fact<Int>) {
        self.length = length
        self.running = running
    }
}

/// Chat template facts (`providers/contract.py:TemplateFacts`).
public struct EngineTemplateFacts: Codable, Sendable, Equatable {
    public let origin: Fact<String>
    public let path: Fact<String>
    public let sha256: Fact<String>
    public let runningSha256: Fact<String>
    public let prefixStable: Fact<Bool>
    public let caps: Fact<[String: Bool]>

    enum CodingKeys: String, CodingKey {
        case origin
        case path
        case sha256
        case runningSha256 = "running_sha256"
        case prefixStable = "prefix_stable"
        case caps
    }

    public init(
        origin: Fact<String>,
        path: Fact<String>,
        sha256: Fact<String>,
        runningSha256: Fact<String>,
        prefixStable: Fact<Bool>,
        caps: Fact<[String: Bool]>
    ) {
        self.origin = origin
        self.path = path
        self.sha256 = sha256
        self.runningSha256 = runningSha256
        self.prefixStable = prefixStable
        self.caps = caps
    }
}

/// Depth sub-object rendered from a model's active Jinja chat template (`thinking_controls.py`).
public struct EngineThinkingDepthFacts: Codable, Sendable, Equatable {
    public let variable: String
    public let values: [String]
    public let aliases: [String: String]
    public let defaultValue: String?
    public let unknown: String?
    public let changesPrefix: Bool?
    public let off: [String]

    enum CodingKeys: String, CodingKey {
        case variable
        case values
        case aliases
        case defaultValue = "default"
        case unknown
        case changesPrefix = "changes_prefix"
        case off
    }

    public init(
        variable: String,
        values: [String],
        aliases: [String: String] = [:],
        defaultValue: String? = nil,
        unknown: String? = nil,
        changesPrefix: Bool? = nil,
        off: [String] = []
    ) {
        self.variable = variable
        self.values = values
        self.aliases = aliases
        self.defaultValue = defaultValue
        self.unknown = unknown
        self.changesPrefix = changesPrefix
        self.off = off
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(variable, forKey: .variable)
        try c.encode(values, forKey: .values)
        try c.encode(aliases, forKey: .aliases)
        if let defaultValue { try c.encode(defaultValue, forKey: .defaultValue) } else { try c.encodeNil(forKey: .defaultValue) }
        if let unknown { try c.encode(unknown, forKey: .unknown) } else { try c.encodeNil(forKey: .unknown) }
        if let changesPrefix { try c.encode(changesPrefix, forKey: .changesPrefix) } else { try c.encodeNil(forKey: .changesPrefix) }
        try c.encode(off, forKey: .off)
    }
}

/// Thinking token budget enforcement metadata in `EngineThinkingFacts`.
public struct EngineThinkingBudgetFacts: Codable, Sendable, Equatable {
    public let enforced: Bool
    public let reason: String

    public init(enforced: Bool, reason: String) {
        self.enforced = enforced
        self.reason = reason
    }
}

/// In-force template thinking controls (`providers/contract.py:EngineDescription.thinking`).
public struct EngineThinkingFacts: Codable, Sendable, Equatable {
    public let switchVariable: String?
    public let depth: EngineThinkingDepthFacts?
    public let budget: EngineThinkingBudgetFacts?
    public let template: String?

    enum CodingKeys: String, CodingKey {
        case switchVariable = "switch"
        case depth
        case budget
        case template
    }

    public init(
        switchVariable: String? = nil,
        depth: EngineThinkingDepthFacts? = nil,
        budget: EngineThinkingBudgetFacts? = nil,
        template: String? = nil
    ) {
        self.switchVariable = switchVariable
        self.depth = depth
        self.budget = budget
        self.template = template.map(EngineSanitization.publicPathString)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        if let switchVariable { try c.encode(switchVariable, forKey: .switchVariable) } else { try c.encodeNil(forKey: .switchVariable) }
        if let depth { try c.encode(depth, forKey: .depth) } else { try c.encodeNil(forKey: .depth) }
        if let budget { try c.encode(budget, forKey: .budget) } else { try c.encodeNil(forKey: .budget) }
        if let template { try c.encode(template, forKey: .template) } else { try c.encodeNil(forKey: .template) }
    }
}

/// Speculative decoding / drafter facts (`drafter`, `type`, `in_force`).
public struct EngineSpeculativeFacts: Codable, Sendable, Equatable {
    public let drafter: Fact<String>
    public let type: Fact<String>
    public let inForce: Fact<Bool>

    enum CodingKeys: String, CodingKey {
        case drafter
        case type
        case inForce = "in_force"
    }

    public init(
        drafter: Fact<String>,
        type: Fact<String>,
        inForce: Fact<Bool> = Fact(provenance: .unknown, source: "not loaded")
    ) {
        self.drafter = drafter
        self.type = type
        self.inForce = inForce
    }
}

/// Decoding mode and accepted sampler field list (`mode`, `request_fields`).
public struct EngineDecodingFacts: Codable, Sendable, Equatable {
    public let mode: Fact<String>
    public let requestFields: Fact<[String]>

    enum CodingKeys: String, CodingKey {
        case mode
        case requestFields = "request_fields"
    }

    public init(
        mode: Fact<String>,
        requestFields: Fact<[String]> = Fact(provenance: .unknown, source: "not loaded")
    ) {
        self.mode = mode
        self.requestFields = requestFields
    }
}

/// Video windowing contract reported on `embedding.video_windows`.
public struct EngineEmbeddingVideoWindows: Codable, Sendable, Equatable {
    public let maxSeconds: Double
    public let sampling: [String]
    public let defaultSampling: String

    enum CodingKeys: String, CodingKey {
        case maxSeconds = "max_seconds"
        case sampling
        case defaultSampling = "default_sampling"
    }

    public init(maxSeconds: Double, sampling: [String] = ["fps", "content"], defaultSampling: String = "fps") {
        self.maxSeconds = maxSeconds
        self.sampling = sampling
        self.defaultSampling = defaultSampling
    }
}

/// Request ceiling contract reported on `embedding.limits`.
public struct EngineEmbeddingLimits: Codable, Sendable, Equatable {
    public let maxBatchSize: Int
    public let maxTotalTokens: Int
    public let maxImagesPerItem: Int
    public let maxAudioSeconds: Double

    enum CodingKeys: String, CodingKey {
        case maxBatchSize = "max_batch_size"
        case maxTotalTokens = "max_total_tokens"
        case maxImagesPerItem = "max_images_per_item"
        case maxAudioSeconds = "max_audio_seconds"
    }

    public init(
        maxBatchSize: Int = 64,
        maxTotalTokens: Int = 8192,
        maxImagesPerItem: Int = 8,
        maxAudioSeconds: Double = 30.0
    ) {
        self.maxBatchSize = maxBatchSize
        self.maxTotalTokens = maxTotalTokens
        self.maxImagesPerItem = maxImagesPerItem
        self.maxAudioSeconds = maxAudioSeconds
    }
}

/// Embedding operation metadata (`embedding_models.py:embedding_info`). `nil` for generation models.
public struct EngineEmbeddingFacts: Codable, Sendable, Equatable {
    public let modalities: [String]
    public let dimensions: [Int]
    public let nativeDimensions: Int
    public let maxInputTokens: Int
    public let prompts: [String: String]
    public let normalization: String
    public let videoAudio: String?
    public let videoWindows: EngineEmbeddingVideoWindows?
    public let limits: EngineEmbeddingLimits?
    public let provenance: String
    public let loadProfile: [String: Int]?
    public let computePlacement: String?

    enum CodingKeys: String, CodingKey {
        case modalities
        case dimensions
        case nativeDimensions = "native_dimensions"
        case maxInputTokens = "max_input_tokens"
        case prompts
        case normalization
        case videoAudio = "video_audio"
        case videoWindows = "video_windows"
        case limits
        case provenance
        case loadProfile = "load_profile"
        case computePlacement = "compute_placement"
    }

    public init(
        modalities: [String],
        dimensions: [Int] = [128, 256, 512, 768],
        nativeDimensions: Int = 768,
        maxInputTokens: Int = 8192,
        prompts: [String: String] = [:],
        normalization: String = "l2",
        videoAudio: String? = "separate_audio_block",
        videoWindows: EngineEmbeddingVideoWindows? = nil,
        limits: EngineEmbeddingLimits? = nil,
        provenance: String = "model_files_and_architecture_contract",
        loadProfile: [String: Int]? = nil,
        computePlacement: String? = nil
    ) {
        self.modalities = modalities
        self.dimensions = dimensions
        self.nativeDimensions = nativeDimensions
        self.maxInputTokens = maxInputTokens
        self.prompts = prompts
        self.normalization = normalization
        self.videoAudio = videoAudio
        self.videoWindows = videoWindows
        self.limits = limits
        self.provenance = provenance
        self.loadProfile = loadProfile
        self.computePlacement = computePlacement
    }
}

/// Full engine provenance contract matching `heylookitsanllm/providers/contract.py:EngineDescription`.
public struct EngineContract: Codable, Sendable, Equatable {
    public let runtime: Fact<String>
    public let context: EngineContextFacts
    public let template: EngineTemplateFacts
    public let settings: [String: SettingFact]
    public let cache: [String: Fact<JSONValue>]
    public let thinking: EngineThinkingFacts?
    public let speculative: EngineSpeculativeFacts
    public let decoding: EngineDecodingFacts
    public let embedding: EngineEmbeddingFacts?
    public let image: JSONValue?
    public let steering: JSONValue?

    enum CodingKeys: String, CodingKey {
        case runtime
        case context
        case template
        case settings
        case cache
        case thinking
        case speculative
        case decoding
        case embedding
        case image
        case steering
    }

    public init(
        runtime: Fact<String>,
        context: EngineContextFacts,
        template: EngineTemplateFacts,
        settings: [String: SettingFact] = [:],
        cache: [String: Fact<JSONValue>] = [:],
        thinking: EngineThinkingFacts? = nil,
        speculative: EngineSpeculativeFacts,
        decoding: EngineDecodingFacts,
        embedding: EngineEmbeddingFacts? = nil,
        image: JSONValue? = nil,
        steering: JSONValue? = nil
    ) {
        self.runtime = runtime
        self.context = context
        self.template = template
        self.settings = settings
        self.cache = cache
        self.thinking = thinking
        self.speculative = speculative
        self.decoding = decoding
        self.embedding = embedding
        self.image = EngineSanitization.publicValue(image)
        self.steering = EngineSanitization.publicValue(steering)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(runtime, forKey: .runtime)
        try c.encode(context, forKey: .context)
        try c.encode(template, forKey: .template)
        try c.encode(settings, forKey: .settings)
        try c.encode(cache, forKey: .cache)
        if let thinking { try c.encode(thinking, forKey: .thinking) } else { try c.encodeNil(forKey: .thinking) }
        try c.encode(speculative, forKey: .speculative)
        try c.encode(decoding, forKey: .decoding)
        if let embedding { try c.encode(embedding, forKey: .embedding) } else { try c.encodeNil(forKey: .embedding) }
        if let image { try c.encode(image, forKey: .image) } else { try c.encodeNil(forKey: .image) }
        if let steering { try c.encode(steering, forKey: .steering) } else { try c.encodeNil(forKey: .steering) }
    }
}

/// Public wire row for `GET /v1/models` (`capabilities.derived_model_facts` + `router.py`).
/// Never contains `modelPath` or any absolute/tilde host path so LAN clients see zero filesystem paths.
public struct PublicModelWireRow: Codable, Sendable, Equatable {
    public let id: String
    public let object: String
    public let ownedBy: String
    public let provider: EngineProviderKind
    public let modalities: [String]
    public let capabilities: [String]
    public let thinkingDefault: Bool
    public let samplerDefaults: [String: JSONValue]
    public let samplerSources: [String: String]
    public let engine: EngineContract

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case ownedBy = "owned_by"
        case provider
        case modalities
        case capabilities
        case thinkingDefault = "thinking_default"
        case samplerDefaults = "sampler_defaults"
        case samplerSources = "sampler_sources"
        case engine
    }

    public init(
        id: String,
        object: String = "model",
        ownedBy: String = "user",
        provider: EngineProviderKind,
        modalities: [String],
        capabilities: [String],
        thinkingDefault: Bool,
        samplerDefaults: [String: JSONValue],
        samplerSources: [String: String],
        engine: EngineContract
    ) {
        self.id = id
        self.object = object
        self.ownedBy = ownedBy
        self.provider = provider
        self.modalities = modalities
        self.capabilities = capabilities
        self.thinkingDefault = thinkingDefault
        self.samplerDefaults = samplerDefaults
        self.samplerSources = samplerSources
        self.engine = engine
    }
}

/// Admin wire row for `GET /v1/admin/models` (`admin_api.py:AdminModelResponse`).
/// `config` contains ONLY explicitly set keys (`exclude_unset`).
public struct AdminModelWireRow: Codable, Sendable, Equatable {
    public enum ConfigOrigin: String, Codable, Sendable, Equatable {
        case config
        case discovered
    }

    public let id: String
    public let provider: EngineProviderKind
    public let description: String?
    public let tags: [String]
    public let capabilities: [String]
    public let config: [String: JSONValue]
    public let loaded: Bool
    public let source: ConfigOrigin
    public let staleReloadFields: [String]
    public let engine: EngineContract
    public let thinkingDefault: Bool
    public let samplerDefaults: [String: JSONValue]
    public let samplerSources: [String: String]

    enum CodingKeys: String, CodingKey {
        case id
        case provider
        case description
        case tags
        case capabilities
        case config
        case loaded
        case source
        case staleReloadFields = "stale_reload_fields"
        case engine
        case thinkingDefault = "thinking_default"
        case samplerDefaults = "sampler_defaults"
        case samplerSources = "sampler_sources"
    }

    public init(
        id: String,
        provider: EngineProviderKind,
        description: String? = nil,
        tags: [String] = [],
        capabilities: [String] = [],
        config: [String: JSONValue] = [:],
        loaded: Bool = false,
        source: ConfigOrigin = .discovered,
        staleReloadFields: [String] = [],
        engine: EngineContract,
        thinkingDefault: Bool,
        samplerDefaults: [String: JSONValue],
        samplerSources: [String: String]
    ) {
        self.id = id
        self.provider = provider
        self.description = description
        self.tags = tags
        self.capabilities = capabilities
        self.config = config
        self.loaded = loaded
        self.source = source
        self.staleReloadFields = staleReloadFields
        self.engine = engine
        self.thinkingDefault = thinkingDefault
        self.samplerDefaults = samplerDefaults
        self.samplerSources = samplerSources
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(provider, forKey: .provider)
        if let description { try c.encode(description, forKey: .description) } else { try c.encodeNil(forKey: .description) }
        try c.encode(tags, forKey: .tags)
        try c.encode(capabilities, forKey: .capabilities)
        try c.encode(config, forKey: .config)
        try c.encode(loaded, forKey: .loaded)
        try c.encode(source, forKey: .source)
        try c.encode(staleReloadFields, forKey: .staleReloadFields)
        try c.encode(engine, forKey: .engine)
        try c.encode(thinkingDefault, forKey: .thinkingDefault)
        try c.encode(samplerDefaults, forKey: .samplerDefaults)
        try c.encode(samplerSources, forKey: .samplerSources)
    }
}
