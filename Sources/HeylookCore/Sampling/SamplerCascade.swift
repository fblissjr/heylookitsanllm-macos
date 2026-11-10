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
    public static let builtinDefaults = SamplerSettings(
        temperature: 0.7,
        topP: 0.95,
        topK: 40,
        minP: 0.0,
        repetitionPenalty: 1.0,
        repetitionContextSize: 64,
        presencePenalty: 0.0,
        maxTokens: 2048,
        seed: nil,
        thinking: true,
        thinkingDepth: nil,
        thinkingBudget: nil,
        stopSequences: []
    )
}

/// Deterministic 5-layer sampling cascade (`builtin` -> `vendor` -> `model_config` -> `preset` -> `request`).
/// Derived from `SamplerFieldKey.allCases` so adding a parameter automatically participates in
/// cascade resolution and provenance tracking (`VISION.md` Principle 2: Derive, don't copy).
public enum SamplerCascade {

    /// Fine-grained 5-layer origin of a resolved sampler field (`GET /v1/models` `sampler_sources`).
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

    public static func resolve(
        builtin: SamplerSettings = .builtinDefaults,
        vendor: SamplerSettings? = nil,
        modelSidecar: SamplerSettings? = nil,
        preset: SamplerSettings? = nil,
        request: SamplerSettings? = nil
    ) -> ResolvedSampling {
        var resolved = builtin
        var provenance: [SamplerFieldKey: ProvenanceSource] = [:]
        var layerSource: [SamplerFieldKey: SamplerLayerSource] = [:]
        for key in SamplerFieldKey.allCases {
            provenance[key] = .derived
            layerSource[key] = .builtin
        }

        func applyLayer(_ layer: SamplerSettings?, source: SamplerLayerSource) {
            guard let layer else { return }
            let prov = source.provenanceSource
            if let v = layer.temperature { resolved.temperature = v; provenance[.temperature] = prov; layerSource[.temperature] = source }
            if let v = layer.topP { resolved.topP = v; provenance[.topP] = prov; layerSource[.topP] = source }
            if let v = layer.topK { resolved.topK = v; provenance[.topK] = prov; layerSource[.topK] = source }
            if let v = layer.minP { resolved.minP = v; provenance[.minP] = prov; layerSource[.minP] = source }
            if let v = layer.repetitionPenalty { resolved.repetitionPenalty = v; provenance[.repetitionPenalty] = prov; layerSource[.repetitionPenalty] = source }
            if let v = layer.repetitionContextSize { resolved.repetitionContextSize = v; provenance[.repetitionContextSize] = prov; layerSource[.repetitionContextSize] = source }
            if let v = layer.presencePenalty { resolved.presencePenalty = v; provenance[.presencePenalty] = prov; layerSource[.presencePenalty] = source }
            if let v = layer.maxTokens { resolved.maxTokens = v; provenance[.maxTokens] = prov; layerSource[.maxTokens] = source }
            if let v = layer.seed { resolved.seed = v; provenance[.seed] = prov; layerSource[.seed] = source }
            if let v = layer.thinking { resolved.thinking = v; provenance[.thinking] = prov; layerSource[.thinking] = source }
            if let v = layer.thinkingDepth { resolved.thinkingDepth = v; provenance[.thinkingDepth] = prov; layerSource[.thinkingDepth] = source }
            if let v = layer.thinkingBudget { resolved.thinkingBudget = v; provenance[.thinkingBudget] = prov; layerSource[.thinkingBudget] = source }
            if let v = layer.stopSequences { resolved.stopSequences = v; provenance[.stopSequences] = prov; layerSource[.stopSequences] = source }
        }

        applyLayer(vendor, source: .vendor)
        applyLayer(modelSidecar, source: .modelConfig)
        applyLayer(preset, source: .preset)
        applyLayer(request, source: .request)

        return ResolvedSampling(
            effective: resolved,
            provenanceByField: provenance,
            sourceByField: layerSource
        )
    }
}
