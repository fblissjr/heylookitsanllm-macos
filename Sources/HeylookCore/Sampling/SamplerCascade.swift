import Foundation

/// Canonical sampling and reasoning parameters shared across Chat, Notebook, Presets,
/// `model.heylook.toml` sidecars, and `/v1/messages` requests.
public struct SamplerSettings: Codable, Sendable, Equatable {
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

/// Deterministic 4-layer sampling cascade (`builtin` -> `modelSidecar` -> `preset` -> `request`).
/// Derived from `SamplerFieldKey.allCases` so adding a parameter automatically participates in
/// cascade resolution and provenance tracking (`VISION.md` Principle 2: Derive, don't copy).
public enum SamplerCascade {

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

        public init(effective: SamplerSettings, provenanceByField: [SamplerFieldKey: ProvenanceSource]) {
            self.effective = effective
            self.provenanceByField = provenanceByField
        }
    }

    public static func resolve(
        builtin: SamplerSettings = .builtinDefaults,
        modelSidecar: SamplerSettings? = nil,
        preset: SamplerSettings? = nil,
        request: SamplerSettings? = nil
    ) -> ResolvedSampling {
        var resolved = builtin
        var provenance: [SamplerFieldKey: ProvenanceSource] = [:]
        for key in SamplerFieldKey.allCases {
            provenance[key] = .derived
        }

        func applyLayer(_ layer: SamplerSettings?, source: ProvenanceSource) {
            guard let layer else { return }
            if let v = layer.temperature { resolved.temperature = v; provenance[.temperature] = source }
            if let v = layer.topP { resolved.topP = v; provenance[.topP] = source }
            if let v = layer.topK { resolved.topK = v; provenance[.topK] = source }
            if let v = layer.minP { resolved.minP = v; provenance[.minP] = source }
            if let v = layer.repetitionPenalty { resolved.repetitionPenalty = v; provenance[.repetitionPenalty] = source }
            if let v = layer.repetitionContextSize { resolved.repetitionContextSize = v; provenance[.repetitionContextSize] = source }
            if let v = layer.presencePenalty { resolved.presencePenalty = v; provenance[.presencePenalty] = source }
            if let v = layer.maxTokens { resolved.maxTokens = v; provenance[.maxTokens] = source }
            if let v = layer.seed { resolved.seed = v; provenance[.seed] = source }
            if let v = layer.thinking { resolved.thinking = v; provenance[.thinking] = source }
            if let v = layer.thinkingDepth { resolved.thinkingDepth = v; provenance[.thinkingDepth] = source }
            if let v = layer.thinkingBudget { resolved.thinkingBudget = v; provenance[.thinkingBudget] = source }
            if let v = layer.stopSequences { resolved.stopSequences = v; provenance[.stopSequences] = source }
        }

        applyLayer(modelSidecar, source: .configured)
        applyLayer(preset, source: .configured)
        applyLayer(request, source: .configured)

        return ResolvedSampling(effective: resolved, provenanceByField: provenance)
    }
}
