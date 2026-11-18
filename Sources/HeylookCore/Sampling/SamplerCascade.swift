import Foundation

/// Canonical sampling and reasoning parameters shared across Chat, Notebook, Presets,
/// `model.heylook.toml` sidecars, and `/v1/messages` requests.
public struct SamplerSettings: Codable, Sendable, Equatable, Hashable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?
    public var repetitionContextSize: Int?
    public var presencePenalty: Double?
    public var maxTokens: Int?
    public var seed: Int?
    public var thinking: Bool?
    public var thinkingDepth: String?
    public var thinkingBudget: Int?
    public var stopSequences: [String]?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        repetitionContextSize: Int? = nil,
        presencePenalty: Double? = nil,
        maxTokens: Int? = nil,
        seed: Int? = nil,
        thinking: Bool? = nil,
        thinkingDepth: String? = nil,
        thinkingBudget: Int? = nil,
        stopSequences: [String]? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.maxTokens = maxTokens
        self.seed = seed
        self.thinking = thinking
        self.thinkingDepth = thinkingDepth
        self.thinkingBudget = thinkingBudget
        self.stopSequences = stopSequences
    }

    private enum CodingKeys: String, CodingKey {
        case temperature
        case topP = "top_p"
        case topPCamel = "topP"
        case topK = "top_k"
        case topKCamel = "topK"
        case minP = "min_p"
        case minPCamel = "minP"
        case repetitionPenalty = "repetition_penalty"
        case repetitionPenaltyCamel = "repetitionPenalty"
        case repetitionContextSize = "repetition_context_size"
        case repetitionContextSizeCamel = "repetitionContextSize"
        case presencePenalty = "presence_penalty"
        case presencePenaltyCamel = "presencePenalty"
        case maxTokens = "max_tokens"
        case maxTokensCamel = "maxTokens"
        case seed
        case thinking
        case enableThinking = "enable_thinking"
        case thinkingDepth = "thinking_depth"
        case thinkingDepthCamel = "thinkingDepth"
        case reasoningEffort = "reasoning_effort"
        case thinkingBudget = "thinking_budget"
        case thinkingBudgetCamel = "thinkingBudget"
        case stopSequences = "stop_sequences"
        case stopSequencesCamel = "stopSequences"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.temperature = try c.decodeIfPresent(Double.self, forKey: .temperature)
        self.topP = try c.decodeIfPresent(Double.self, forKey: .topP)
            ?? (try c.decodeIfPresent(Double.self, forKey: .topPCamel))
        self.topK = try c.decodeIfPresent(Int.self, forKey: .topK)
            ?? (try c.decodeIfPresent(Int.self, forKey: .topKCamel))
        self.minP = try c.decodeIfPresent(Double.self, forKey: .minP)
            ?? (try c.decodeIfPresent(Double.self, forKey: .minPCamel))
        self.repetitionPenalty = try c.decodeIfPresent(Double.self, forKey: .repetitionPenalty)
            ?? (try c.decodeIfPresent(Double.self, forKey: .repetitionPenaltyCamel))
        self.repetitionContextSize = try c.decodeIfPresent(Int.self, forKey: .repetitionContextSize)
            ?? (try c.decodeIfPresent(Int.self, forKey: .repetitionContextSizeCamel))
        self.presencePenalty = try c.decodeIfPresent(Double.self, forKey: .presencePenalty)
            ?? (try c.decodeIfPresent(Double.self, forKey: .presencePenaltyCamel))
        self.maxTokens = try c.decodeIfPresent(Int.self, forKey: .maxTokens)
            ?? (try c.decodeIfPresent(Int.self, forKey: .maxTokensCamel))
        self.seed = try c.decodeIfPresent(Int.self, forKey: .seed)
        self.thinking = try c.decodeIfPresent(Bool.self, forKey: .thinking)
            ?? (try c.decodeIfPresent(Bool.self, forKey: .enableThinking))
        self.thinkingDepth = try c.decodeIfPresent(String.self, forKey: .thinkingDepth)
            ?? (try c.decodeIfPresent(String.self, forKey: .thinkingDepthCamel))
            ?? (try c.decodeIfPresent(String.self, forKey: .reasoningEffort))
        self.thinkingBudget = try c.decodeIfPresent(Int.self, forKey: .thinkingBudget)
            ?? (try c.decodeIfPresent(Int.self, forKey: .thinkingBudgetCamel))
        self.stopSequences = try c.decodeIfPresent([String].self, forKey: .stopSequences)
            ?? (try c.decodeIfPresent([String].self, forKey: .stopSequencesCamel))
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(temperature, forKey: .temperature)
        try c.encodeIfPresent(topP, forKey: .topP)
        try c.encodeIfPresent(topK, forKey: .topK)
        try c.encodeIfPresent(minP, forKey: .minP)
        try c.encodeIfPresent(repetitionPenalty, forKey: .repetitionPenalty)
        try c.encodeIfPresent(repetitionContextSize, forKey: .repetitionContextSize)
        try c.encodeIfPresent(presencePenalty, forKey: .presencePenalty)
        try c.encodeIfPresent(maxTokens, forKey: .maxTokens)
        try c.encodeIfPresent(seed, forKey: .seed)
        try c.encodeIfPresent(thinking, forKey: .thinking)
        try c.encodeIfPresent(thinkingDepth, forKey: .thinkingDepth)
        try c.encodeIfPresent(thinkingBudget, forKey: .thinkingBudget)
        try c.encodeIfPresent(stopSequences, forKey: .stopSequences)
    }

    /// Built-in baseline defaults when no model sidecar, preset, or request override is set.
    /// Aligned with `heylookitsanllm/src/heylook_llm/samplers.py` (`GLOBAL_SAMPLER_FLOOR`).
    public static let builtinDefaults = SamplerSettings(
        temperature: SamplerCascade.fallbackTemperature,
        topP: SamplerCascade.fallbackTopP,
        topK: SamplerCascade.knobsOffTopK,
        minP: SamplerCascade.knobsOffMinP,
        repetitionPenalty: SamplerCascade.knobsOffRepetitionPenalty,
        repetitionContextSize: SamplerCascade.defaultRepetitionContextSize,
        presencePenalty: SamplerCascade.knobsOffPresencePenalty,
        maxTokens: SamplerCascade.defaultMaxTokens,
        seed: nil,
        thinking: nil,
        thinkingDepth: nil,
        thinkingBudget: nil,
        stopSequences: []
    )
}

/// Deterministic 5-layer sampling cascade (`builtin` -> `vendor` -> `model_config` -> `preset` -> `request`).
/// Derived from `SamplerFieldKey.allCases` so adding a parameter automatically participates in
/// cascade resolution and provenance tracking (`VISION.md` Principle 2: Derive, don't copy).
public enum SamplerCascade {
    /// Global temperature fallback when vendor, model, preset, and request are all silent (`samplers.py`).
    public static let fallbackTemperature: Double = 1.0
    /// Global top_p fallback (`samplers.py`).
    public static let fallbackTopP: Double = 0.95
    /// Safety stop on `max_tokens` (`samplers.py:DEFAULT_MAX_TOKENS`).
    public static let defaultMaxTokens: Int = 16384
    /// Default repetition window size.
    public static let defaultRepetitionContextSize: Int = 64

    /// Off-switches neutralizing engine-specific defaults (`samplers.py:KNOBS_OFF`).
    public static let knobsOffTopK: Int = 0
    public static let knobsOffMinP: Double = 0.0
    public static let knobsOffRepetitionPenalty: Double = 1.0
    public static let knobsOffPresencePenalty: Double = 0.0

    /// Strict allowlist of keys read from `generation_config.json` / GGUF `general.sampling.*` (`samplers.py:VENDOR_SAMPLING_KEYS`).
    public static let vendorSamplingKeys: [String] = ["temperature", "top_p", "top_k"]
    public static let vendorSamplingKeySet: Set<String> = Set(vendorSamplingKeys)

    /// Canonical field names on `/v1/models` `sampler_defaults` / `sampler_sources` (`ModelSamplerSummary`).
    public struct ModelSamplerSummary: Sendable, Equatable {
        public let thinkingDefault: Bool
        public let samplerDefaults: [String: JSONValue]
        public let samplerSources: [String: String]
        public let resolved: ResolvedSampling

        public init(
            thinkingDefault: Bool,
            samplerDefaults: [String: JSONValue],
            samplerSources: [String: String],
            resolved: ResolvedSampling
        ) {
            self.thinkingDefault = thinkingDefault
            self.samplerDefaults = samplerDefaults
            self.samplerSources = samplerSources
            self.resolved = resolved
        }
    }

    /// Fine-grained 5-layer origin of a resolved sampler field.
    public enum SamplerLayerSource: String, CaseIterable, Codable, Sendable {
        case builtin
        case vendor
        case modelConfig = "model_config"
        case preset
        case request

        public var provenanceSource: ProvenanceSource {
            switch self {
            case .builtin, .vendor:
                return .derived
            case .modelConfig, .preset, .request:
                return .configured
            }
        }

        /// Coarse 3-way origin string reported in `GET /v1/models` `sampler_sources` (`"model"` | `"vendor"` | `"default"`).
        public var publicSamplerSource: String {
            switch self {
            case .modelConfig, .preset, .request:
                return "model"
            case .vendor:
                return "vendor"
            case .builtin:
                return "default"
            }
        }
    }

    public enum SamplerFieldKey: String, CaseIterable, Codable, Sendable {
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case repetitionContextSize = "repetition_context_size"
        case presencePenalty = "presence_penalty"
        case maxTokens = "max_tokens"
        case seed
        case thinking
        case thinkingDepth = "thinking_depth"
        case thinkingBudget = "thinking_budget"
        case stopSequences = "stop_sequences"

        /// Canonical wire key used in `sampler_defaults` (maps `.thinking` -> `"enable_thinking"`).
        public var canonicalWireKey: String {
            switch self {
            case .thinking: return "enable_thinking"
            default: return rawValue
            }
        }

        /// All known config/wire spelling aliases accepted for this sampler parameter.
        public var acceptedAliases: [String] {
            switch self {
            case .thinking: return ["enable_thinking", "thinking"]
            case .thinkingDepth: return ["thinking_depth", "reasoning_effort", "reasoning_strength"]
            case .thinkingBudget: return ["thinking_budget", "thinking_budget_tokens"]
            default: return [rawValue]
            }
        }
    }

    public struct ResolvedSampling: Sendable, Equatable {
        public let effective: SamplerSettings
        public let provenanceByField: [SamplerFieldKey: ProvenanceSource]
        public let sourceByField: [SamplerFieldKey: SamplerLayerSource]

        public init(
            effective: SamplerSettings,
            provenanceByField: [SamplerFieldKey: ProvenanceSource],
            sourceByField: [SamplerFieldKey: SamplerLayerSource] = [:]
        ) {
            self.effective = effective
            self.provenanceByField = provenanceByField
            if sourceByField.isEmpty {
                var synthesized: [SamplerFieldKey: SamplerLayerSource] = [:]
                for (k, v) in provenanceByField {
                    synthesized[k] = (v == .configured) ? .modelConfig : .builtin
                }
                self.sourceByField = synthesized
            } else {
                self.sourceByField = sourceByField
            }
        }
    }

    /// Resolves effective sampling parameters across all 5 cascade layers (`builtin` -> `vendor` -> `modelSidecar` -> `preset` -> `request`).
    /// When `thinkingCapable` is supplied, `thinking` falls back to `thinkingCapable` (unless overridden by `modelSidecar`, `preset`, or `request`, ignoring `vendor`).
    public static func resolve(
        builtin: SamplerSettings = .builtinDefaults,
        vendor: SamplerSettings? = nil,
        modelSidecar: SamplerSettings? = nil,
        preset: SamplerSettings? = nil,
        request: SamplerSettings? = nil,
        thinkingCapable: Bool? = nil
    ) -> ResolvedSampling {
        var resolved = builtin
        if let thinkingCapable {
            resolved.thinking = thinkingCapable
        }
        var provenance: [SamplerFieldKey: ProvenanceSource] = [:]
        var layerSource: [SamplerFieldKey: SamplerLayerSource] = [:]
        for key in SamplerFieldKey.allCases {
            provenance[key] = .derived
            layerSource[key] = .builtin
        }

        func applyLayer(_ layer: SamplerSettings?, source: SamplerLayerSource, allowVendorThinking: Bool = true) {
            guard let layer else { return }
            let prov = source.provenanceSource
            if let v = layer.temperature { resolved.temperature = v; provenance[.temperature] = prov; layerSource[.temperature] = source }
            if let v = layer.topP { resolved.topP = v; provenance[.topP] = prov; layerSource[.topP] = source }
            if let v = layer.topK { resolved.topK = v; provenance[.topK] = prov; layerSource[.topK] = source }
            if source != .vendor {
                if let v = layer.minP { resolved.minP = v; provenance[.minP] = prov; layerSource[.minP] = source }
                if let v = layer.repetitionPenalty { resolved.repetitionPenalty = v; provenance[.repetitionPenalty] = prov; layerSource[.repetitionPenalty] = source }
                if let v = layer.repetitionContextSize { resolved.repetitionContextSize = v; provenance[.repetitionContextSize] = prov; layerSource[.repetitionContextSize] = source }
                if let v = layer.presencePenalty { resolved.presencePenalty = v; provenance[.presencePenalty] = prov; layerSource[.presencePenalty] = source }
                if let v = layer.maxTokens { resolved.maxTokens = v; provenance[.maxTokens] = prov; layerSource[.maxTokens] = source }
                if let v = layer.seed { resolved.seed = v; provenance[.seed] = prov; layerSource[.seed] = source }
                if allowVendorThinking, let v = layer.thinking { resolved.thinking = v; provenance[.thinking] = prov; layerSource[.thinking] = source }
                if let v = layer.thinkingDepth { resolved.thinkingDepth = v; provenance[.thinkingDepth] = prov; layerSource[.thinkingDepth] = source }
                if let v = layer.thinkingBudget { resolved.thinkingBudget = v; provenance[.thinkingBudget] = prov; layerSource[.thinkingBudget] = source }
                if let v = layer.stopSequences { resolved.stopSequences = v; provenance[.stopSequences] = prov; layerSource[.stopSequences] = source }
            }
        }

        applyLayer(vendor, source: .vendor, allowVendorThinking: false)
        applyLayer(modelSidecar, source: .modelConfig)
        applyLayer(preset, source: .preset)
        applyLayer(request, source: .request)

        return ResolvedSampling(
            effective: resolved,
            provenanceByField: provenance,
            sourceByField: layerSource
        )
    }

    /// Computes `thinking_default`, `sampler_defaults`, and `sampler_sources` in ONE cascade pass
    /// so `sampler_defaults.enable_thinking == thinking_default` by construction (`capabilities.py:derived_model_facts`).
    public static func resolveModelDefaults(
        vendor: SamplerSettings? = nil,
        modelSidecar: SamplerSettings? = nil,
        thinkingCapable: Bool
    ) -> ModelSamplerSummary {
        let resolved = resolve(
            builtin: .builtinDefaults,
            vendor: vendor,
            modelSidecar: modelSidecar,
            preset: nil,
            request: nil,
            thinkingCapable: thinkingCapable
        )
        let eff = resolved.effective
        let thinkingDefault = eff.thinking ?? thinkingCapable

        var defaults: [String: JSONValue] = [:]
        var sources: [String: String] = [:]

        for key in SamplerFieldKey.allCases {
            let wireKey = key.canonicalWireKey
            let src = (resolved.sourceByField[key] ?? .builtin).publicSamplerSource
            switch key {
            case .temperature:
                if let v = eff.temperature { defaults[wireKey] = .double(v); sources[wireKey] = src }
            case .topP:
                if let v = eff.topP { defaults[wireKey] = .double(v); sources[wireKey] = src }
            case .topK:
                if let v = eff.topK { defaults[wireKey] = .int(v); sources[wireKey] = src }
            case .minP:
                if let v = eff.minP { defaults[wireKey] = .double(v); sources[wireKey] = src }
            case .repetitionPenalty:
                if let v = eff.repetitionPenalty { defaults[wireKey] = .double(v); sources[wireKey] = src }
            case .repetitionContextSize:
                if let v = eff.repetitionContextSize { defaults[wireKey] = .int(v); sources[wireKey] = src }
            case .presencePenalty:
                if let v = eff.presencePenalty { defaults[wireKey] = .double(v); sources[wireKey] = src }
            case .maxTokens:
                if let v = eff.maxTokens { defaults[wireKey] = .int(v); sources[wireKey] = src }
            case .seed:
                if let v = eff.seed { defaults[wireKey] = .int(v); sources[wireKey] = src }
            case .thinking:
                defaults[wireKey] = .bool(thinkingDefault)
                // `enable_thinking` default provenance is reported via `thinking_default`; omit from `sampler_sources` matching `capabilities.py`.
            case .thinkingDepth:
                if let v = eff.thinkingDepth { defaults[wireKey] = .string(v); sources[wireKey] = src }
            case .thinkingBudget:
                if let v = eff.thinkingBudget { defaults[wireKey] = .int(v); sources[wireKey] = src }
            case .stopSequences:
                if let v = eff.stopSequences, !v.isEmpty {
                    defaults[wireKey] = .array(v.map { .string($0) })
                    sources[wireKey] = src
                }
            }
        }

        return ModelSamplerSummary(
            thinkingDefault: thinkingDefault,
            samplerDefaults: defaults,
            samplerSources: sources,
            resolved: resolved
        )
    }
}
