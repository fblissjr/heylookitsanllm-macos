import Foundation

/// Primitive JSON Schema field types exposed by `/v1/admin/model-options`.
public enum ConfigFieldValueType: String, Codable, Sendable, Equatable {
    case string
    case integer
    case number
    case boolean
    case array
    case object
}

/// Frontend editor visibility level matching `model-config.js`.
public enum ConfigUIVisibility: String, Codable, Sendable, Equatable {
    case normal
    case advanced
    case hidden
}

/// Declarative descriptor for a single provider configuration field.
/// Single source of truth driving `/v1/admin/model-options`, `PATCH` validation,
/// `/reload` load-setting validation, `stale_reload_fields`, and `EngineContract.settings`.
public struct ConfigFieldDescriptor: Codable, Sendable, Equatable {
    public let name: String
    public let type: ConfigFieldValueType
    public let defaultValue: JSONValue?
    public let minimum: Double?
    public let maximum: Double?
    public let exclusiveMinimum: Double?
    public let exclusiveMaximum: Double?
    public let enumValues: [String]?
    public let itemType: ConfigFieldValueType?
    public let effect: ConfigFieldEffect
    public let arg: String?
    public let ui: ConfigUIVisibility
    public let shape: String?
    public let reason: String?
    public let isFileOnly: Bool
    public let isLoadSetting: Bool
    public let required: Bool
    public let description: String

    enum CodingKeys: String, CodingKey {
        case name
        case type
        case defaultValue = "default"
        case minimum
        case maximum
        case exclusiveMinimum
        case exclusiveMaximum
        case enumValues = "enum"
        case items
        case effect
        case arg
        case ui
        case shape
        case reason
        case isFileOnly = "file_only"
        case isLoadSetting = "load_setting"
        case required
        case description
    }

    private struct ItemsSchema: Codable, Sendable, Equatable {
        let type: ConfigFieldValueType
    }

    public init(
        name: String,
        type: ConfigFieldValueType,
        defaultValue: JSONValue? = nil,
        minimum: Double? = nil,
        maximum: Double? = nil,
        exclusiveMinimum: Double? = nil,
        exclusiveMaximum: Double? = nil,
        enumValues: [String]? = nil,
        itemType: ConfigFieldValueType? = nil,
        effect: ConfigFieldEffect,
        arg: String? = nil,
        ui: ConfigUIVisibility = .normal,
        shape: String? = nil,
        reason: String? = nil,
        isFileOnly: Bool = false,
        isLoadSetting: Bool = false,
        required: Bool = false,
        description: String
    ) {
        self.name = name
        self.type = type
        self.defaultValue = defaultValue
        self.minimum = minimum
        self.maximum = maximum
        self.exclusiveMinimum = exclusiveMinimum
        self.exclusiveMaximum = exclusiveMaximum
        self.enumValues = enumValues
        self.itemType = itemType
        self.effect = effect
        self.arg = arg
        self.ui = ui
        self.shape = shape
        self.reason = reason
        self.isFileOnly = isFileOnly
        self.isLoadSetting = isLoadSetting
        self.required = required
        self.description = description
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try c.decode(String.self, forKey: .name)
        self.type = try c.decode(ConfigFieldValueType.self, forKey: .type)
        self.defaultValue = try c.decodeIfPresent(JSONValue.self, forKey: .defaultValue)
        self.minimum = try c.decodeIfPresent(Double.self, forKey: .minimum)
        self.maximum = try c.decodeIfPresent(Double.self, forKey: .maximum)
        self.exclusiveMinimum = try c.decodeIfPresent(Double.self, forKey: .exclusiveMinimum)
        self.exclusiveMaximum = try c.decodeIfPresent(Double.self, forKey: .exclusiveMaximum)
        self.enumValues = try c.decodeIfPresent([String].self, forKey: .enumValues)
        self.itemType = try c.decodeIfPresent(ItemsSchema.self, forKey: .items)?.type
        self.effect = try c.decode(ConfigFieldEffect.self, forKey: .effect)
        self.arg = try c.decodeIfPresent(String.self, forKey: .arg)
        self.ui = try c.decodeIfPresent(ConfigUIVisibility.self, forKey: .ui) ?? .normal
        self.shape = try c.decodeIfPresent(String.self, forKey: .shape)
        self.reason = try c.decodeIfPresent(String.self, forKey: .reason)
        self.isFileOnly = try c.decodeIfPresent(Bool.self, forKey: .isFileOnly) ?? false
        self.isLoadSetting = try c.decodeIfPresent(Bool.self, forKey: .isLoadSetting) ?? false
        self.required = try c.decodeIfPresent(Bool.self, forKey: .required) ?? false
        self.description = try c.decode(String.self, forKey: .description)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(name, forKey: .name)
        try c.encode(effect, forKey: .effect)
        try c.encode(type, forKey: .type)
        if let defaultValue {
            try c.encode(defaultValue, forKey: .defaultValue)
        } else {
            try c.encodeNil(forKey: .defaultValue)
        }
        try c.encode(required, forKey: .required)
        try c.encodeIfPresent(minimum, forKey: .minimum)
        try c.encodeIfPresent(maximum, forKey: .maximum)
        try c.encodeIfPresent(exclusiveMinimum, forKey: .exclusiveMinimum)
        try c.encodeIfPresent(exclusiveMaximum, forKey: .exclusiveMaximum)
        try c.encodeIfPresent(enumValues, forKey: .enumValues)
        if let itemType {
            try c.encode(ItemsSchema(type: itemType), forKey: .items)
        }
        try c.encode(description, forKey: .description)
        try c.encodeIfPresent(arg, forKey: .arg)
        if ui != .normal {
            try c.encode(ui, forKey: .ui)
        }
        try c.encodeIfPresent(shape, forKey: .shape)
        try c.encodeIfPresent(reason, forKey: .reason)
        if isFileOnly {
            try c.encode(true, forKey: .isFileOnly)
        }
        if isLoadSetting {
            try c.encode(true, forKey: .isLoadSetting)
        }
    }
}

/// Complete schema and derived field sets for a single provider (`mlx`, `gguf`, `coreai`, `remote`).
public struct ProviderSchema: Sendable, Equatable {
    public let provider: EngineProviderKind
    public let fields: [ConfigFieldDescriptor]
    public let aliases: [String: String]
    private let fieldsByName: [String: ConfigFieldDescriptor]

    public init(
        provider: EngineProviderKind,
        fields: [ConfigFieldDescriptor],
        aliases: [String: String] = [:]
    ) {
        self.provider = provider
        self.fields = fields.sorted { $0.name < $1.name }
        self.aliases = aliases
        var map: [String: ConfigFieldDescriptor] = [:]
        map.reserveCapacity(fields.count)
        for field in fields {
            map[field.name] = field
        }
        self.fieldsByName = map
    }

    /// Resolves canonical or alias field name to its descriptor.
    public func descriptor(named rawName: String) -> ConfigFieldDescriptor? {
        if let exact = fieldsByName[rawName] {
            return exact
        }
        if let canonical = aliases[rawName] {
            return fieldsByName[canonical]
        }
        return nil
    }

    /// Maps an input key or alias to the schema's canonical field name, if known.
    public func canonicalName(for rawName: String) -> String? {
        descriptor(named: rawName)?.name
    }

    /// Editable non-identity fields (`effect != .identity`), matching `config.configurable_fields`.
    public var configurableFields: Set<String> {
        Set(fields.filter { $0.effect != .identity }.map(\.name))
    }

    /// Fields whose modification requires reloading a loaded model (`requires_reload` + `identity`).
    public var reloadRequiredFields: Set<String> {
        Set(fields.filter { $0.effect == .requiresReload || $0.effect == .identity }.map(\.name))
    }

    /// Load-panel settings accepted by `POST /v1/admin/models/{id}/reload`.
    public var loadSettingFields: Set<String> {
        Set(fields.filter(\.isLoadSetting).map(\.name))
    }

    /// File-only fields refused on HTTP `PATCH`.
    public var fileOnlyFields: Set<String> {
        Set(fields.filter(\.isFileOnly).map(\.name))
    }
}

/// Validated `PATCH /v1/admin/models/{id}` result.
public struct PatchValidationResult: Sendable, Equatable {
    public let canonicalPatch: [String: JSONValue]
    public let reloadRequiredFields: [String]

    public init(canonicalPatch: [String: JSONValue], reloadRequiredFields: [String]) {
        self.canonicalPatch = canonicalPatch
        self.reloadRequiredFields = reloadRequiredFields
    }
}

/// Validation failure carrying the exact HTTP status (`422` for PATCH schema/range errors,
/// `400` for `/reload` unknown load setting errors) and diagnostic message.
public struct SchemaValidationError: Error, LocalizedError, Sendable, Equatable {
    public let statusCode: Int
    public let field: String?
    public let message: String

    public var errorDescription: String? { message }

    public init(statusCode: Int, field: String? = nil, message: String) {
        self.statusCode = statusCode
        self.field = field
        self.message = message
    }
}

/// Declarative per-provider configuration registry and single-source validators.
public enum ProviderConfigSchema {
    private static let llamaCacheTypes = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
    private static let llamaFlashAttnValues = ["on", "off", "auto"]
    private static let llamaLoadModes = ["none", "mmap", "mlock", "mmap+mlock", "dio"]

    /// Canonical field names for all 13 sampler parameters derived straight from `SamplerCascade.SamplerFieldKey`.
    public static var canonicalSamplerFieldNames: [String] {
        SamplerCascade.SamplerFieldKey.allCases.map(\.rawValue)
    }

    /// Canonical API/wire field names for all 13 sampler parameters (`enable_thinking` for `.thinking`).
    public static var wireSamplerFieldNames: [String] {
        SamplerCascade.SamplerFieldKey.allCases.map(\.canonicalWireKey)
    }

    /// Derives all 13 sampler `ConfigFieldDescriptor` entries directly from `SamplerCascade.SamplerFieldKey.allCases`
    /// so there is never a hand-copied list of sampler keys.
    public static func samplerDescriptors(for provider: EngineProviderKind) -> [ConfigFieldDescriptor] {
        let vendorNote = (provider == .gguf)
            ? "its GGUF header general.sampling.*, else heylook's floor"
            : "this model's generation_config.json, else heylook's floor"

        return SamplerCascade.SamplerFieldKey.allCases.map { key in
            switch key {
            case .temperature:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .number,
                    defaultValue: nil,
                    minimum: 0.0,
                    maximum: 2.0,
                    effect: .perRequest,
                    description: "Per-model default sampling temperature. Unset = \(vendorNote)."
                )
            case .topP:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .number,
                    defaultValue: nil,
                    minimum: 0.0,
                    maximum: 1.0,
                    effect: .perRequest,
                    description: "Per-model default nucleus-sampling cutoff. Unset = \(vendorNote)."
                )
            case .topK:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .integer,
                    defaultValue: nil,
                    minimum: 0,
                    effect: .perRequest,
                    description: "Per-model default top-k cutoff; 0 disables it. Unset = \(vendorNote)."
                )
            case .minP:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .number,
                    defaultValue: nil,
                    minimum: 0.0,
                    maximum: 1.0,
                    effect: .perRequest,
                    description: "Per-model default min-p cutoff (0.0 disables it). Not a vendor-layer key."
                )
            case .repetitionPenalty:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .number,
                    defaultValue: nil,
                    minimum: 0.1,
                    maximum: 2.0,
                    effect: .perRequest,
                    description: "Per-model repetition penalty over generated tokens (1.0 = off)."
                )
            case .repetitionContextSize:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .integer,
                    defaultValue: nil,
                    minimum: 1,
                    effect: .perRequest,
                    ui: provider == .gguf ? .hidden : .normal,
                    description: "Token window inspected for repetition penalty (default 64)."
                )
            case .presencePenalty:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .number,
                    defaultValue: nil,
                    minimum: 0.0,
                    maximum: 2.0,
                    effect: .perRequest,
                    description: "Per-model presence penalty (0.0 = off)."
                )
            case .maxTokens:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .integer,
                    defaultValue: nil,
                    exclusiveMinimum: 0,
                    effect: .perRequest,
                    description: "Per-model default cap on generated tokens. Unset = global floor (16384)."
                )
            case .seed:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .integer,
                    defaultValue: nil,
                    effect: .perRequest,
                    ui: .advanced,
                    description: "Default RNG seed for reproducible sampling; unset = non-deterministic."
                )
            case .thinking:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .boolean,
                    defaultValue: nil,
                    effect: .perRequest,
                    description: "Per-model thinking default. Unset = follow model thinking capability; a bool pins it either way."
                )
            case .thinkingDepth:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .string,
                    defaultValue: nil,
                    effect: .perRequest,
                    description: "Per-model default thinking depth in this model's own template spelling."
                )
            case .thinkingBudget:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .integer,
                    defaultValue: nil,
                    minimum: 1,
                    effect: .perRequest,
                    description: "Default hard token cap on the thinking block before closing thinking."
                )
            case .stopSequences:
                return ConfigFieldDescriptor(
                    name: key.canonicalWireKey,
                    type: .array,
                    defaultValue: .array([]),
                    itemType: .string,
                    effect: .perRequest,
                    ui: .advanced,
                    description: "Default stop strings halting generation for this model."
                )
            }
        }
    }

    private static var commonSamplerAliases: [String: String] {
        var map: [String: String] = [:]
        for key in SamplerCascade.SamplerFieldKey.allCases {
            for alias in key.acceptedAliases where alias != key.canonicalWireKey {
                map[alias] = key.canonicalWireKey
            }
        }
        return map
    }

    /// Builds the schema for `provider` from declarative descriptors and derived sampler keys.
    public static func schema(for provider: EngineProviderKind) -> ProviderSchema {
        var fields = samplerDescriptors(for: provider)
        var aliases = commonSamplerAliases

        // Shared administrative metadata fields editable per model
        fields.append(contentsOf: [
            ConfigFieldDescriptor(
                name: "pinned",
                type: .boolean,
                defaultValue: .bool(false),
                effect: .appliesLive,
                description: "Keep this model exempt from idle eviction and LRU unloading."
            ),
            ConfigFieldDescriptor(
                name: "unload_after_idle_seconds",
                type: .integer,
                defaultValue: nil,
                minimum: 0,
                effect: .appliesLive,
                description: "Seconds idle before unloading; 0 disables idle unload, unset uses global server setting."
            ),
            ConfigFieldDescriptor(
                name: "tags",
                type: .array,
                defaultValue: .array([]),
                itemType: .string,
                effect: .descriptive,
                description: "User-defined metadata tags for filtering and UI organization."
            ),
            ConfigFieldDescriptor(
                name: "draft_model_path",
                type: .string,
                defaultValue: nil,
                effect: .requiresReload,
                arg: provider == .gguf ? "-md" : nil,
                description: "Sidecar speculative drafter weights/folder path."
            ),
        ])
        aliases["draft_model"] = "draft_model_path"

        switch provider {
        case .mlx:
            aliases["context_length"] = "ctx_size"
            aliases["context_size"] = "ctx_size"
            aliases["cache_size"] = "ctx_size"
            aliases["vision"] = "supports_vision"

            fields.append(contentsOf: [
                ConfigFieldDescriptor(
                    name: "model_path",
                    type: .string,
                    effect: .identity,
                    ui: .hidden,
                    isFileOnly: true,
                    required: true,
                    description: "Model directory holding weights, config.json, and tokenizer."
                ),
                ConfigFieldDescriptor(
                    name: "modalities",
                    type: .array,
                    defaultValue: nil,
                    itemType: .string,
                    effect: .requiresReload,
                    description: "Author-declared capability set (e.g. [\"text\", \"vision\"]). Decides VLM vs text path at load."
                ),
                ConfigFieldDescriptor(
                    name: "supports_vision",
                    type: .boolean,
                    defaultValue: nil,
                    effect: .requiresReload,
                    ui: .hidden,
                    description: "Legacy mirror of vision capability; edit modalities instead."
                ),
                ConfigFieldDescriptor(
                    name: "ctx_size",
                    type: .integer,
                    defaultValue: nil,
                    exclusiveMinimum: 0,
                    effect: .requiresReload,
                    isLoadSetting: true,
                    description: "Override context window ceiling when config.json omits YaRN/scaling length."
                ),
                ConfigFieldDescriptor(
                    name: "kv_bits",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 2,
                    maximum: 8,
                    effect: .requiresReload,
                    ui: .advanced,
                    description: "KV cache quantization bit width for MLX Swift (4 or 8; unset = full precision)."
                ),
                ConfigFieldDescriptor(
                    name: "kv_group_size",
                    type: .integer,
                    defaultValue: .int(64),
                    minimum: 16,
                    effect: .requiresReload,
                    ui: .advanced,
                    description: "KV quantization group size when kv_bits is enabled (default 64)."
                ),
                ConfigFieldDescriptor(
                    name: "image_max_tokens",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 64,
                    effect: .perRequest,
                    description: "Soft visual token budget passed into VLM processor resizing."
                ),
                ConfigFieldDescriptor(
                    name: "prefill_step_size",
                    type: .integer,
                    defaultValue: nil,
                    exclusiveMinimum: 0,
                    effect: .perRequest,
                    ui: .advanced,
                    description: "Prompt tokens per MLX prefill forward chunk."
                ),
                ConfigFieldDescriptor(
                    name: "draft_block_size",
                    type: .integer,
                    defaultValue: nil,
                    exclusiveMinimum: 0,
                    effect: .perRequest,
                    description: "Draft tokens proposed per speculation round."
                ),
                ConfigFieldDescriptor(
                    name: "draft_max_context",
                    type: .integer,
                    defaultValue: nil,
                    exclusiveMinimum: 0,
                    effect: .perRequest,
                    description: "Maximum prompt length at which speculative decoding remains active."
                ),
                ConfigFieldDescriptor(
                    name: "max_queue_depth",
                    type: .integer,
                    defaultValue: .int(8),
                    minimum: 1,
                    effect: .loadTimeOnly,
                    reason: "process-wide: the first provider created wins, so reloading this model cannot change it",
                    description: "Process-wide in-flight plus queued generation cap before 503 backpressure."
                ),
                ConfigFieldDescriptor(
                    name: "chat_template_source",
                    type: .string,
                    defaultValue: nil,
                    effect: .requiresReload,
                    description: "Template source selection policy: auto, jinja, tokenizer_config, chat_template_json, or file path."
                ),
                ConfigFieldDescriptor(
                    name: "pipeline_ratios",
                    type: .array,
                    defaultValue: nil,
                    itemType: .number,
                    effect: .requiresReload,
                    ui: .advanced,
                    description: "Per-rank decoder layer split ratios for distributed MLX pipeline parallelism (e.g. [0.8, 0.2])."
                ),
                ConfigFieldDescriptor(
                    name: "pipeline_peers",
                    type: .array,
                    defaultValue: nil,
                    itemType: .string,
                    effect: .requiresReload,
                    ui: .hidden,
                    isFileOnly: true,
                    description: "Peer endpoints in rank order for multi-Mac MLX pipeline execution (file-only)."
                ),
            ])

        case .gguf:
            aliases["context_size"] = "ctx_size"
            aliases["context_length"] = "ctx_size"
            aliases["vision"] = "supports_vision"
            aliases["thinking_capable"] = "supports_thinking"

            fields.append(contentsOf: [
                ConfigFieldDescriptor(
                    name: "model_path",
                    type: .string,
                    effect: .identity,
                    ui: .hidden,
                    isFileOnly: true,
                    required: true,
                    description: "Path to the primary .gguf weights file."
                ),
                ConfigFieldDescriptor(
                    name: "mmproj_path",
                    type: .string,
                    defaultValue: nil,
                    effect: .requiresReload,
                    arg: "--mmproj",
                    ui: .hidden,
                    isFileOnly: true,
                    description: "Path to the multimodal projector GGUF sidecar (file-only)."
                ),
                ConfigFieldDescriptor(
                    name: "chat_template_path",
                    type: .string,
                    defaultValue: nil,
                    effect: .requiresReload,
                    arg: "--chat-template-file",
                    ui: .hidden,
                    isFileOnly: true,
                    description: "Explicit Jinja template file passed to llama-server at spawn (file-only)."
                ),
                ConfigFieldDescriptor(
                    name: "use_sidecar_chat_template",
                    type: .boolean,
                    defaultValue: .bool(true),
                    effect: .requiresReload,
                    description: "Use adjacent chat_template.jinja over embedded GGUF template when chat_template_path is unset."
                ),
                ConfigFieldDescriptor(
                    name: "spec_type",
                    type: .string,
                    defaultValue: nil,
                    effect: .requiresReload,
                    arg: "--spec-type",
                    description: "Speculative draft mode passed to llama-server (e.g. draft-mtp)."
                ),
                ConfigFieldDescriptor(
                    name: "spec_draft_n_max",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 1,
                    maximum: 16,
                    effect: .requiresReload,
                    arg: "--spec-draft-n-max",
                    description: "Maximum draft tokens per speculation round."
                ),
                ConfigFieldDescriptor(
                    name: "spec_draft_p_min",
                    type: .number,
                    defaultValue: nil,
                    minimum: 0.0,
                    maximum: 1.0,
                    effect: .requiresReload,
                    arg: "--spec-draft-p-min",
                    description: "Minimum probability threshold for drafted tokens."
                ),
                ConfigFieldDescriptor(
                    name: "spec_draft_n_min",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 0,
                    effect: .requiresReload,
                    arg: "--spec-draft-n-min",
                    description: "Minimum draft tokens retained per speculation round."
                ),
                ConfigFieldDescriptor(
                    name: "draft_max_context",
                    type: .integer,
                    defaultValue: nil,
                    exclusiveMinimum: 0,
                    effect: .perRequest,
                    description: "Maximum prompt length at which speculative drafting runs."
                ),
                ConfigFieldDescriptor(
                    name: "ctx_size",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 512,
                    effect: .requiresReload,
                    arg: "--ctx-size",
                    isLoadSetting: true,
                    description: "Context slot allocated by llama-server at spawn; unset uses model training context."
                ),
                ConfigFieldDescriptor(
                    name: "flash_attn",
                    type: .string,
                    defaultValue: nil,
                    enumValues: llamaFlashAttnValues,
                    effect: .requiresReload,
                    arg: "-fa",
                    isLoadSetting: true,
                    description: "Flash attention setting for llama-server (unset = on, on/off/auto)."
                ),
                ConfigFieldDescriptor(
                    name: "n_cpu_moe",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 0,
                    effect: .requiresReload,
                    arg: "-ncmoe",
                    ui: .advanced,
                    description: "Number of MoE layers whose expert tensors stay on CPU."
                ),
                ConfigFieldDescriptor(
                    name: "cpu_moe",
                    type: .boolean,
                    defaultValue: nil,
                    effect: .requiresReload,
                    arg: "-cmoe",
                    ui: .advanced,
                    shape: "flag",
                    description: "Move all MoE expert tensors to CPU."
                ),
                ConfigFieldDescriptor(
                    name: "override_tensor",
                    type: .string,
                    defaultValue: nil,
                    effect: .requiresReload,
                    arg: "-ot",
                    ui: .advanced,
                    description: "Raw tensor-buffer placement override pattern."
                ),
                ConfigFieldDescriptor(
                    name: "n_cpu_moe_draft",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 0,
                    effect: .requiresReload,
                    arg: "-ncmoed",
                    ui: .advanced,
                    description: "MoE expert layers pinned to CPU for the drafter."
                ),
                ConfigFieldDescriptor(
                    name: "cpu_moe_draft",
                    type: .boolean,
                    defaultValue: nil,
                    effect: .requiresReload,
                    arg: "-cmoed",
                    ui: .advanced,
                    shape: "flag",
                    description: "Move all drafter MoE expert tensors to CPU."
                ),
                ConfigFieldDescriptor(
                    name: "cache_type_k",
                    type: .string,
                    defaultValue: nil,
                    enumValues: llamaCacheTypes,
                    effect: .requiresReload,
                    arg: "-ctk",
                    ui: .advanced,
                    description: "Quantization format for the K half of the KV cache (unset = f16)."
                ),
                ConfigFieldDescriptor(
                    name: "cache_type_v",
                    type: .string,
                    defaultValue: nil,
                    enumValues: llamaCacheTypes,
                    effect: .requiresReload,
                    arg: "-ctv",
                    ui: .advanced,
                    description: "Quantization format for the V half of the KV cache (requires flash attention when quantized)."
                ),
                ConfigFieldDescriptor(
                    name: "n_gpu_layers",
                    type: .integer,
                    defaultValue: .int(999),
                    minimum: 0,
                    effect: .requiresReload,
                    arg: "-ngl",
                    description: "Number of layers offloaded to Metal GPU (999 = all layers)."
                ),
                ConfigFieldDescriptor(
                    name: "rpc",
                    type: .string,
                    defaultValue: nil,
                    effect: .requiresReload,
                    arg: "--rpc",
                    ui: .hidden,
                    isFileOnly: true,
                    description: "Comma-separated ggml-rpc worker endpoints (file-only)."
                ),
                ConfigFieldDescriptor(
                    name: "tensor_split",
                    type: .string,
                    defaultValue: nil,
                    effect: .requiresReload,
                    arg: "--tensor-split",
                    ui: .advanced,
                    description: "Layer proportion split across RPC and local Metal devices."
                ),
                ConfigFieldDescriptor(
                    name: "n_ubatch",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 32,
                    effect: .requiresReload,
                    arg: "-ub",
                    ui: .advanced,
                    description: "Physical prompt-processing micro-batch size."
                ),
                ConfigFieldDescriptor(
                    name: "n_batch",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 32,
                    effect: .requiresReload,
                    arg: "-b",
                    ui: .advanced,
                    description: "Logical prompt-processing batch ceiling (default 2048)."
                ),
                ConfigFieldDescriptor(
                    name: "n_gpu_layers_draft",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 0,
                    effect: .requiresReload,
                    arg: "-ngld",
                    description: "GPU layers offloaded for the speculative drafter."
                ),
                ConfigFieldDescriptor(
                    name: "sleep_idle_seconds",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 1,
                    effect: .requiresReload,
                    arg: "--sleep-idle-seconds",
                    description: "llama-server idle sleep duration freeing weights/KV while keeping the subprocess alive."
                ),
                ConfigFieldDescriptor(
                    name: "cache_ram_mb",
                    type: .integer,
                    defaultValue: nil,
                    minimum: -1,
                    effect: .requiresReload,
                    arg: "-cram",
                    description: "Host-RAM prompt cache ceiling in MiB (-1 = unlimited, 0 = disabled, unset = auto)."
                ),
                ConfigFieldDescriptor(
                    name: "ctx_checkpoints",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 0,
                    effect: .requiresReload,
                    arg: "--ctx-checkpoints",
                    ui: .advanced,
                    description: "Maximum context checkpoints per slot for hybrid/SWA architectures."
                ),
                ConfigFieldDescriptor(
                    name: "checkpoint_min_step",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 0,
                    effect: .requiresReload,
                    arg: "--checkpoint-min-step",
                    ui: .advanced,
                    description: "Minimum token spacing between checkpoints (auto = 256 on checkpointed models)."
                ),
                ConfigFieldDescriptor(
                    name: "cache_reuse",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 0,
                    effect: .requiresReload,
                    arg: "--cache-reuse",
                    ui: .advanced,
                    description: "Minimum chunk size for KV shifting reuse on full-attention text models."
                ),
                ConfigFieldDescriptor(
                    name: "load_mode",
                    type: .string,
                    defaultValue: nil,
                    enumValues: llamaLoadModes,
                    effect: .requiresReload,
                    arg: "-lm",
                    description: "Weight memory mapping mode (none, mmap, mlock, mmap+mlock, dio)."
                ),
                ConfigFieldDescriptor(
                    name: "server_binary",
                    type: .string,
                    defaultValue: nil,
                    effect: .requiresReload,
                    ui: .hidden,
                    isFileOnly: true,
                    description: "Override llama-server executable path for this model (file-only)."
                ),
                ConfigFieldDescriptor(
                    name: "host",
                    type: .string,
                    defaultValue: .string("127.0.0.1"),
                    effect: .requiresReload,
                    ui: .hidden,
                    description: "Loopback bind interface for llama-server subprocess."
                ),
                ConfigFieldDescriptor(
                    name: "port",
                    type: .integer,
                    defaultValue: .int(0),
                    minimum: 0,
                    effect: .requiresReload,
                    ui: .hidden,
                    description: "Port for llama-server subprocess (0 = auto free port)."
                ),
                ConfigFieldDescriptor(
                    name: "startup_timeout_s",
                    type: .number,
                    defaultValue: .double(300.0),
                    exclusiveMinimum: 0,
                    effect: .requiresReload,
                    ui: .hidden,
                    description: "Seconds to wait for subprocess readiness before failing load."
                ),
                ConfigFieldDescriptor(
                    name: "extra_args",
                    type: .array,
                    defaultValue: .array([]),
                    itemType: .string,
                    effect: .requiresReload,
                    ui: .hidden,
                    isFileOnly: true,
                    description: "Raw passthrough CLI flags appended to llama-server invocation (file-only)."
                ),
                ConfigFieldDescriptor(
                    name: "supports_thinking",
                    type: .boolean,
                    defaultValue: nil,
                    effect: .descriptive,
                    description: "Descriptive capability override used only when no template can be inspected."
                ),
                ConfigFieldDescriptor(
                    name: "supports_vision",
                    type: .boolean,
                    defaultValue: nil,
                    effect: .descriptive,
                    ui: .hidden,
                    description: "Descriptive vision flag; vision on GGUF requires mmproj_path."
                ),
                ConfigFieldDescriptor(
                    name: "modalities",
                    type: .array,
                    defaultValue: nil,
                    itemType: .string,
                    effect: .descriptive,
                    description: "Descriptive capability list advertised on /v1/models."
                ),
                ConfigFieldDescriptor(
                    name: "image_max_tokens",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 64,
                    effect: .perRequest,
                    ui: .advanced,
                    description: "Advised per-image token ceiling for vision context planning."
                ),
            ])

        case .coreai:
            let placementValues = ComputePlacement.allCases.map(\.rawValue)
            aliases["context_length"] = "ctx_size"
            aliases["vision"] = "supports_vision"
            aliases["thinking_capable"] = "supports_thinking"

            fields.append(contentsOf: [
                ConfigFieldDescriptor(
                    name: "model_path",
                    type: .string,
                    effect: .identity,
                    ui: .hidden,
                    isFileOnly: true,
                    required: true,
                    description: "Path to the .aimodel or .mlmodelc bundle."
                ),
                ConfigFieldDescriptor(
                    name: "compute_unit",
                    type: .string,
                    defaultValue: nil,
                    enumValues: placementValues,
                    effect: .requiresReload,
                    isLoadSetting: true,
                    description: "Target CoreML compute unit placement (ane, hybrid_ane_gpu, cpu_and_gpu, cpu_only)."
                ),
                ConfigFieldDescriptor(
                    name: "vision_compute_unit",
                    type: .string,
                    defaultValue: nil,
                    enumValues: placementValues,
                    effect: .requiresReload,
                    ui: .advanced,
                    description: "Compute unit override for vision tower execution."
                ),
                ConfigFieldDescriptor(
                    name: "audio_compute_unit",
                    type: .string,
                    defaultValue: nil,
                    enumValues: placementValues,
                    effect: .requiresReload,
                    ui: .advanced,
                    description: "Compute unit override for audio tower execution."
                ),
                ConfigFieldDescriptor(
                    name: "text_compute_unit",
                    type: .string,
                    defaultValue: nil,
                    enumValues: placementValues,
                    effect: .requiresReload,
                    ui: .advanced,
                    description: "Compute unit override for text tower execution."
                ),
                ConfigFieldDescriptor(
                    name: "ctx_size",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 128,
                    effect: .requiresReload,
                    isLoadSetting: true,
                    description: "Configured sequence window for Core AI model state allocation."
                ),
                ConfigFieldDescriptor(
                    name: "image_max_tokens",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 64,
                    effect: .perRequest,
                    ui: .advanced,
                    description: "Visual token cap for vision preprocessing."
                ),
                ConfigFieldDescriptor(
                    name: "modalities",
                    type: .array,
                    defaultValue: nil,
                    itemType: .string,
                    effect: .descriptive,
                    description: "Declared modality list advertised on /v1/models."
                ),
                ConfigFieldDescriptor(
                    name: "supports_vision",
                    type: .boolean,
                    defaultValue: nil,
                    effect: .descriptive,
                    ui: .hidden,
                    description: "Descriptive vision support flag."
                ),
                ConfigFieldDescriptor(
                    name: "supports_thinking",
                    type: .boolean,
                    defaultValue: nil,
                    effect: .descriptive,
                    description: "Descriptive thinking support flag."
                ),
            ])

        case .remote:
            aliases["context_length"] = "ctx_size"
            fields.append(contentsOf: [
                ConfigFieldDescriptor(
                    name: "model_path",
                    type: .string,
                    effect: .identity,
                    ui: .hidden,
                    isFileOnly: true,
                    required: true,
                    description: "Remote endpoint base URL or remote model identifier."
                ),
                ConfigFieldDescriptor(
                    name: "ctx_size",
                    type: .integer,
                    defaultValue: nil,
                    minimum: 128,
                    effect: .descriptive,
                    description: "Reported remote model context ceiling."
                ),
            ])
        }

        return ProviderSchema(provider: provider, fields: fields, aliases: aliases)
    }

    /// Generates the full `/v1/admin/model-options` payload matching `admin_api.py:get_model_options`.
    public static func modelOptionsJSON(includeCoreAI: Bool = true) -> [String: Any] {
        var providersMap: [String: Any] = [:]
        let targets: [EngineProviderKind] = includeCoreAI ? [.mlx, .gguf, .coreai] : [.mlx, .gguf]
        let encoder = JSONEncoder()
        for kind in targets {
            let s = schema(for: kind)
            if let encoded = try? encoder.encode(s.fields),
               let jsonArr = try? JSONSerialization.jsonObject(with: encoded) {
                providersMap[kind.rawValue] = ["fields": jsonArr]
            }
        }
        return [
            "providers": providersMap,
            "sampler_fields": canonicalSamplerFieldNames,
            "compute_placements": ComputePlacement.allCases.map(\.rawValue),
        ]
    }

    /// Validates an HTTP `PATCH /v1/admin/models/{id}` `config` delta against `provider`'s schema.
    /// Rejects unknown keys, `isFileOnly` keys, and invalid types/ranges with status `422`.
    public static func validatePatch(
        provider: EngineProviderKind,
        patch: [String: JSONValue]
    ) -> Result<PatchValidationResult, SchemaValidationError> {
        let s = schema(for: provider)
        var canonical: [String: JSONValue] = [:]
        var reloadFields: [String] = []
        let reloadSet = s.reloadRequiredFields

        for (rawKey, value) in patch.sorted(by: { $0.key < $1.key }) {
            guard let descriptor = s.descriptor(named: rawKey) else {
                return .failure(SchemaValidationError(
                    statusCode: 422,
                    field: rawKey,
                    message: "Unknown config key '\(rawKey)' for provider '\(provider.rawValue)'."
                ))
            }
            if descriptor.isFileOnly {
                return .failure(SchemaValidationError(
                    statusCode: 422,
                    field: descriptor.name,
                    message: "'\(descriptor.name)' is file-only and must be edited in model.heylook.toml or heylook.toml, never over HTTP PATCH."
                ))
            }
            if let error = validateValue(value, against: descriptor) {
                return .failure(SchemaValidationError(
                    statusCode: 422,
                    field: descriptor.name,
                    message: error
                ))
            }
            canonical[descriptor.name] = value
            if reloadSet.contains(descriptor.name) {
                reloadFields.append(descriptor.name)
            }
        }

        return .success(PatchValidationResult(
            canonicalPatch: canonical,
            reloadRequiredFields: reloadFields.sorted()
        ))
    }

    /// Validates and canonicalizes the optional JSON `settings` body of `POST /v1/admin/models/{id}/reload`.
    /// Rejects keys that are not `load_setting` fields for `provider` with status `400`.
    public static func parseReloadSettings(
        provider: EngineProviderKind,
        modelID: String,
        settings: [String: JSONValue]?
    ) -> Result<[String: JSONValue], SchemaValidationError> {
        guard let settings, !settings.isEmpty else {
            return .success([:])
        }
        let s = schema(for: provider)
        let offered = s.loadSettingFields
        var unknown: [String] = []
        var canonical: [String: JSONValue] = [:]

        for (rawKey, value) in settings.sorted(by: { $0.key < $1.key }) {
            guard let descriptor = s.descriptor(named: rawKey), offered.contains(descriptor.name) else {
                unknown.append(rawKey)
                continue
            }
            if let error = validateValue(value, against: descriptor) {
                return .failure(SchemaValidationError(
                    statusCode: 400,
                    field: descriptor.name,
                    message: error
                ))
            }
            canonical[descriptor.name] = value
        }

        if !unknown.isEmpty {
            let offeredStr = offered.sorted().joined(separator: ", ")
            return .failure(SchemaValidationError(
                statusCode: 400,
                field: unknown.first,
                message: "\(unknown.joined(separator: ", ")): not a load setting of '\(modelID)' (provider '\(provider.rawValue)'); its load settings are \(offeredStr.isEmpty ? "none" : offeredStr)"
            ))
        }

        return .success(canonical)
    }

    /// Computes `stale_reload_fields` by diffing `reloadRequiredFields` between the snapshot
    /// the running provider was constructed with and the current saved config, plus `"chat_template"`
    /// when `chatTemplateStale == true`. Returns `[]` when `loadedSnapshot` is `nil` (unloaded).
    public static func staleReloadFields(
        provider: EngineProviderKind,
        loadedSnapshot: [String: JSONValue]?,
        currentConfig: [String: JSONValue],
        chatTemplateStale: Bool? = nil,
        loadedProvider: EngineProviderKind? = nil
    ) -> [String] {
        guard let loadedSnapshot else { return [] }
        if let loadedProvider, loadedProvider != provider {
            return ["provider"]
        }
        let s = schema(for: provider)
        var stale: [String] = []
        for key in s.reloadRequiredFields.sorted() {
            let snapVal = loadedSnapshot[key] ?? .null
            let freshVal = currentConfig[key] ?? .null
            if snapVal != freshVal {
                stale.append(key)
            }
        }
        if chatTemplateStale == true {
            stale.append("chat_template")
        }
        return stale
    }

    /// Builds `EngineContract.settings` from the provider's declarative schema without hand-listing keys
    /// (`contract.py:config_settings` + `capabilities.py:_engine_with_sampler_answers`).
    public static func buildEngineSettings(
        provider: EngineProviderKind,
        effectiveConfig: [String: JSONValue],
        explicitKeys: Set<String>,
        derivedBaseline: [String: JSONValue] = [:],
        writtenStoreName: String? = nil,
        engineDefaultName: String,
        samplerSummary: SamplerCascade.ModelSamplerSummary? = nil,
        unstoredSamplerSummary: SamplerCascade.ModelSamplerSummary? = nil
    ) -> [String: SettingFact] {
        let s = schema(for: provider)
        var canonicalExplicit: Set<String> = []
        for key in explicitKeys {
            if let canonical = s.canonicalName(for: key) {
                canonicalExplicit.insert(canonical)
            }
        }

        let storeLabel = writtenStoreName ?? "model.heylook.toml"
        var out: [String: SettingFact] = [:]

        for name in s.configurableFields.sorted() {
            guard let descriptor = s.descriptor(named: name) else { continue }
            let defaultVal = descriptor.defaultValue
            let currentVal = effectiveConfig[name] ?? defaultVal
            let baselineVal = derivedBaseline[name] ?? defaultVal

            let fact: SettingFact
            if canonicalExplicit.contains(name), !jsonValuesEquivalent(currentVal, baselineVal) {
                fact = SettingFact(
                    value: currentVal,
                    configured: currentVal,
                    auto: baselineVal,
                    reason: "set in this model's \(storeLabel)",
                    provenance: .configured,
                    effect: descriptor.effect
                )
            } else if canonicalExplicit.contains(name) {
                fact = SettingFact(
                    value: currentVal,
                    configured: nil,
                    auto: baselineVal,
                    reason: "stored in this model's \(storeLabel), same as derived",
                    provenance: .derived,
                    effect: descriptor.effect
                )
            } else if !jsonValuesEquivalent(currentVal, defaultVal) {
                fact = SettingFact(
                    value: currentVal,
                    configured: nil,
                    auto: currentVal,
                    reason: "derived from the model's own files",
                    provenance: .derived,
                    effect: descriptor.effect
                )
            } else {
                let reason = (defaultVal == nil)
                    ? "not set; \(engineDefaultName) decides"
                    : "not set; heylook's default"
                fact = SettingFact(
                    value: defaultVal,
                    configured: nil,
                    auto: defaultVal,
                    reason: reason,
                    provenance: .derived,
                    effect: descriptor.effect
                )
            }
            out[name] = fact
        }

        if let summary = samplerSummary {
            let cascadeReason = "the sampling cascade: this model's stored value, else its own vendor defaults, else heylook's floor"
            for (key, resolvedVal) in summary.samplerDefaults {
                guard let existing = out[key] else { continue }
                let storedHere = (existing.provenance == .configured)
                let autoVal: JSONValue? = storedHere
                    ? (unstoredSamplerSummary != nil ? unstoredSamplerSummary?.samplerDefaults[key] : (existing.auto ?? resolvedVal))
                    : resolvedVal
                out[key] = existing.updating(
                    value: resolvedVal,
                    auto: autoVal,
                    reason: storedHere ? existing.reason : cascadeReason
                )
            }
        }

        return out
    }

    private static func jsonValuesEquivalent(_ a: JSONValue?, _ b: JSONValue?) -> Bool {
        switch (a, b) {
        case (nil, nil), (.some(.null), nil), (nil, .some(.null)):
            return true
        case (.some(.int(let x)), .some(.double(let y))):
            return Double(x) == y
        case (.some(.double(let x)), .some(.int(let y))):
            return x == Double(y)
        case (.some(let lhs), .some(let rhs)):
            return lhs == rhs
        default:
            return false
        }
    }

    private static func validateValue(_ value: JSONValue, against d: ConfigFieldDescriptor) -> String? {
        if value == .null {
            return nil
        }

        // Special pipeline_ratios rule matching config.py:_validate_pipeline_ratios
        if d.name == "pipeline_ratios" {
            guard case .array(let items) = value else {
                return "pipeline_ratios must be a list of positive numbers (e.g. [0.8, 0.2])"
            }
            if items.isEmpty {
                return nil
            }
            if items.count < 2 {
                return "pipeline_ratios requires at least 2 rank weights (e.g. [0.8, 0.2])"
            }
            for item in items {
                guard let num = numericValue(item), num > 0 else {
                    return "pipeline_ratios entries must be > 0"
                }
            }
            return nil
        }

        if d.name == "thinking_depth" {
            guard case .string(let s) = value else {
                return "thinking_depth must be a string"
            }
            if s.isEmpty || s.count > 32 {
                return "thinking_depth must be 1..32 characters"
            }
            let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
            if s.unicodeScalars.contains(where: { !allowed.contains($0) }) {
                return "thinking_depth contains invalid characters; only [A-Za-z0-9_-] allowed"
            }
            return nil
        }

        switch d.type {
        case .string:
            guard case .string(let s) = value else {
                return "'\(d.name)' must be a string"
            }
            if let allowed = d.enumValues, !allowed.contains(s) {
                return "'\(d.name)' must be one of: \(allowed.joined(separator: ", "))"
            }
            return nil

        case .boolean:
            guard case .bool = value else {
                return "'\(d.name)' must be a boolean"
            }
            return nil

        case .integer:
            guard case .int(let i) = value else {
                return "'\(d.name)' must be an integer"
            }
            return validateRange(Double(i), against: d)

        case .number:
            guard let num = numericValue(value) else {
                return "'\(d.name)' must be a number"
            }
            return validateRange(num, against: d)

        case .array:
            guard case .array(let items) = value else {
                return "'\(d.name)' must be an array"
            }
            if let itemType = d.itemType {
                for (idx, item) in items.enumerated() {
                    switch itemType {
                    case .string:
                        guard case .string = item else { return "'\(d.name)[\(idx)]' must be a string" }
                    case .number:
                        guard numericValue(item) != nil else { return "'\(d.name)[\(idx)]' must be a number" }
                    case .integer:
                        guard case .int = item else { return "'\(d.name)[\(idx)]' must be an integer" }
                    case .boolean:
                        guard case .bool = item else { return "'\(d.name)[\(idx)]' must be a boolean" }
                    case .array, .object:
                        break
                    }
                }
            }
            return nil

        case .object:
            guard case .object = value else {
                return "'\(d.name)' must be an object"
            }
            return nil
        }
    }

    private static func numericValue(_ value: JSONValue) -> Double? {
        switch value {
        case .int(let i): return Double(i)
        case .double(let d) where d.isFinite: return d
        default: return nil
        }
    }

    private static func validateRange(_ number: Double, against d: ConfigFieldDescriptor) -> String? {
        if let min = d.minimum, number < min {
            return "'\(d.name)' must be >= \(formatBound(min))"
        }
        if let max = d.maximum, number > max {
            return "'\(d.name)' must be <= \(formatBound(max))"
        }
        if let exMin = d.exclusiveMinimum, number <= exMin {
            return "'\(d.name)' must be > \(formatBound(exMin))"
        }
        if let exMax = d.exclusiveMaximum, number >= exMax {
            return "'\(d.name)' must be < \(formatBound(exMax))"
        }
        return nil
    }

    private static func formatBound(_ v: Double) -> String {
        v.rounded() == v ? String(Int(v)) : String(v)
    }
}
