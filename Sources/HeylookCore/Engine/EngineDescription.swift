import Foundation

/// Provenance of a model or engine setting (`VISION.md` Principle 1).
public enum ProvenanceSource: String, Codable, CaseIterable, Sendable {
    case derived
    case configured
    case observed
    case observedCached = "observed_cached"
    case unknown
    case notApplicable = "not_applicable"
}

/// Effect class of changing a setting (`per_request`, `reload`, or `restart`).
public enum EffectClass: String, Codable, CaseIterable, Sendable {
    case perRequest = "per_request"
    case reload
    case restart
}

/// Active hardware compute placement on Apple Silicon.
public enum ComputePlacement: String, Codable, CaseIterable, Sendable, Hashable {
    case mlxGPU = "mlx_gpu"
    case ane = "ane"
    case hybridANEGPU = "hybrid_ane_gpu"
    case cpuAndGPU = "cpu_and_gpu"
    case cpuOnly = "cpu_only"
    case ggufSubprocess = "gguf_subprocess"
    case remote = "remote"

    public var label: String {
        switch self {
        case .mlxGPU: return "Metal GPU (MLX)"
        case .ane: return "Apple Neural Engine (ANE)"
        case .hybridANEGPU: return "Hybrid ANE Towers + Metal GPU"
        case .cpuAndGPU: return "CoreML GPU + CPU"
        case .cpuOnly: return "CPU Only"
        case .ggufSubprocess: return "llama-server Subprocess (Metal)"
        case .remote: return "Remote Heylook Server"
        }
    }

    /// True when this placement executes on the Apple Neural Engine without contending
    /// for the primary Metal GPU generation gate.
    public var runsOnNeuralEngineLane: Bool {
        self == .ane
    }
}

/// Engine provider family.
public enum EngineProviderKind: String, Codable, CaseIterable, Sendable {
    case mlx
    case coreai
    case gguf
    case remote

    public var displayName: String {
        switch self {
        case .mlx: return "MLX Swift"
        case .coreai: return "Core AI / CoreML"
        case .gguf: return "GGUF (llama-server)"
        case .remote: return "Remote Heylook"
        }
    }
}

/// Wraps a configuration or capability value with its provenance, effect class, and compute placement.
public struct ProvenanceField<Value: Codable & Sendable & Equatable>: Codable, Sendable, Equatable {
    public let value: Value
    public let provenance: ProvenanceSource
    public let effect: EffectClass
    public let computePlacement: ComputePlacement?

    public init(
        value: Value,
        provenance: ProvenanceSource,
        effect: EffectClass = .perRequest,
        computePlacement: ComputePlacement? = nil
    ) {
        self.value = value
        self.provenance = provenance
        self.effect = effect
        self.computePlacement = computePlacement
    }
}

/// Prompt/KV cache reuse outcome reported per request.
public enum CacheOutcome: String, Codable, Sendable, Equatable, Hashable {
    case reused
    case miss
    case ineligible
}

/// Per-request prompt cache reuse report (`performance.cache` on the Messages wire).
public struct CacheReport: Codable, Sendable, Equatable, Hashable {
    /// Canonical machine-readable cache cause tokens shared with the frontend and API spec.
    public static let knownCauses: [String] = [
        "cold",
        "memory",
        "new_image_set",
        "no_common_prefix",
        "probable_template_diverged",
        "probable_budget_skipped",
        "probable_evicted",
    ]

    public let promptTokens: Int
    public let cachedTokens: Int
    public let processedTokens: Int
    public let outcome: CacheOutcome
    public let cause: String?
    public let reason: String?

    public init(
        promptTokens: Int,
        cachedTokens: Int,
        processedTokens: Int,
        outcome: CacheOutcome,
        cause: String? = nil,
        reason: String? = nil
    ) {
        self.promptTokens = promptTokens
        self.cachedTokens = cachedTokens
        self.processedTokens = processedTokens
        self.outcome = outcome
        self.cause = cause
        self.reason = reason
    }

    /// Converts a machine cache cause token into human-readable words matching `frontend/js/utils.js`.
    /// For example, `"probable_template_diverged"` becomes `"probably template diverged"`.
    public static func causeWords(_ cause: String) -> String {
        let trimmed = cause.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        if trimmed.hasPrefix("probable_") {
            let suffix = String(trimmed.dropFirst("probable_".count)).replacingOccurrences(of: "_", with: " ")
            return "probably \(suffix)"
        }
        return trimmed.replacingOccurrences(of: "_", with: " ")
    }

    /// Human-readable form of `cause`, or `nil` when `cause` is absent/blank.
    public var humanReadableCause: String? {
        guard let cause else { return nil }
        let words = Self.causeWords(cause)
        return words.isEmpty ? nil : words
    }

    public var summaryBadge: String {
        let why = humanReadableCause
        switch outcome {
        case .reused:
            if let why {
                return "cache \(cachedTokens)/\(promptTokens) (\(why))"
            }
            return "cache \(cachedTokens)/\(promptTokens)"
        case .miss:
            if let why {
                return "cache miss: \(why)"
            }
            return "cache miss"
        case .ineligible:
            if let why {
                return "cache not used: \(why)"
            }
            return "cache not used"
        }
    }
}

/// Speculative decoding / Multi-Token Prediction (MTP) telemetry report.
public struct SpeculativeReport: Codable, Sendable, Equatable, Hashable {
    public let drafterID: String
    public let mode: String
    public let draftedTokens: Int
    public let acceptedTokens: Int
    public let emittedTokens: Int

    enum CodingKeys: String, CodingKey {
        case drafterID
        case mode
        case draftedTokens
        case acceptedTokens
        case emittedTokens
        // Alternate wire spellings for flexible decoding
        case drafterIdSnake = "drafter_id"
        case drafted
        case accepted
        case emitted
    }

    public init(
        drafterID: String,
        mode: String = "mtp",
        draftedTokens: Int,
        acceptedTokens: Int,
        emittedTokens: Int = 0
    ) {
        self.drafterID = drafterID
        self.mode = mode
        self.draftedTokens = draftedTokens
        self.acceptedTokens = acceptedTokens
        self.emittedTokens = emittedTokens
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.drafterID = try c.decodeIfPresent(String.self, forKey: .drafterID)
            ?? c.decodeIfPresent(String.self, forKey: .drafterIdSnake)
            ?? ""
        self.mode = try c.decodeIfPresent(String.self, forKey: .mode) ?? "mtp"
        self.draftedTokens = try c.decodeIfPresent(Int.self, forKey: .draftedTokens)
            ?? c.decodeIfPresent(Int.self, forKey: .drafted)
            ?? 0
        self.acceptedTokens = try c.decodeIfPresent(Int.self, forKey: .acceptedTokens)
            ?? c.decodeIfPresent(Int.self, forKey: .accepted)
            ?? 0
        self.emittedTokens = try c.decodeIfPresent(Int.self, forKey: .emittedTokens)
            ?? c.decodeIfPresent(Int.self, forKey: .emitted)
            ?? 0
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(drafterID, forKey: .drafterID)
        try c.encode(mode, forKey: .mode)
        try c.encode(draftedTokens, forKey: .draftedTokens)
        try c.encode(acceptedTokens, forKey: .acceptedTokens)
        try c.encode(emittedTokens, forKey: .emittedTokens)
    }

    /// Fraction of drafted tokens that were accepted (`acceptedTokens / draftedTokens`), or `0` when none drafted.
    public var acceptanceRate: Double {
        guard draftedTokens > 0 else { return 0 }
        return Double(acceptedTokens) / Double(draftedTokens)
    }

    /// Fraction of emitted tokens sourced from accepted drafts (`acceptedTokens / emittedTokens`), or `0` when none emitted.
    public var draftShare: Double {
        guard emittedTokens > 0 else { return 0 }
        return Double(acceptedTokens) / Double(emittedTokens)
    }

    /// Nullable `acceptance_rate` for wire serialization matching `schema/responses.py`.
    public var optionalAcceptanceRate: Double? {
        guard draftedTokens > 0 else { return nil }
        return Double(acceptedTokens) / Double(draftedTokens)
    }

    /// Nullable `draft_share` for wire serialization matching `schema/responses.py`.
    public var optionalDraftShare: Double? {
        guard emittedTokens > 0 else { return nil }
        return Double(acceptedTokens) / Double(emittedTokens)
    }

    /// Legacy alias for `acceptanceRate` preserved for Wave 1 caller compatibility.
    public var acceptanceShare: Double {
        acceptanceRate
    }

    public var summaryBadge: String {
        if draftedTokens > 0 {
            let pct = Int((acceptanceRate * 100.0).rounded())
            return "\(mode) \(acceptedTokens)/\(draftedTokens) (\(pct)%)"
        }
        if emittedTokens > 0 {
            let pct = Int((draftShare * 100.0).rounded())
            return "\(pct)% from draft (\(acceptedTokens)/\(emittedTokens))"
        }
        return "\(mode) \(acceptedTokens)/0 (0%)"
    }
}

/// Canonical Anthropic Messages stop reasons plus `cancelled`.
public enum StopReason: String, Codable, Sendable, Equatable, Hashable {
    case endTurn = "end_turn"
    case maxTokens = "max_tokens"
    case stopSequence = "stop_sequence"
    case cancelled = "cancelled"
}

/// Unified streaming chunk emitted by all `InferenceProvider` implementations.
public struct GenerationChunk: Codable, Sendable, Equatable {
    public let text: String
    public let thinking: String
    public let tokenCount: Int
    public let thinkingTokens: Int
    public let contentTokens: Int
    public let promptTokens: Int
    public let generationTPS: Double
    public let promptTPS: Double
    public let ttftMs: Double?
    public let thinkingDurationMs: Double?
    public let contentDurationMs: Double?
    public let generationDurationMs: Double?
    public let peakMemoryGB: Double
    public let stopReason: StopReason?
    public let cacheReport: CacheReport?
    public let speculativeReport: SpeculativeReport?
    public let activeComputePlacement: ComputePlacement

    public init(
        text: String = "",
        thinking: String = "",
        tokenCount: Int = 0,
        thinkingTokens: Int = 0,
        contentTokens: Int = 0,
        promptTokens: Int = 0,
        generationTPS: Double = 0,
        promptTPS: Double = 0,
        ttftMs: Double? = nil,
        thinkingDurationMs: Double? = nil,
        contentDurationMs: Double? = nil,
        generationDurationMs: Double? = nil,
        peakMemoryGB: Double = 0,
        stopReason: StopReason? = nil,
        cacheReport: CacheReport? = nil,
        speculativeReport: SpeculativeReport? = nil,
        activeComputePlacement: ComputePlacement = .mlxGPU
    ) {
        self.text = text
        self.thinking = thinking
        self.tokenCount = tokenCount
        self.thinkingTokens = thinkingTokens
        self.contentTokens = contentTokens
        self.promptTokens = promptTokens
        self.generationTPS = generationTPS
        self.promptTPS = promptTPS
        self.ttftMs = ttftMs
        self.thinkingDurationMs = thinkingDurationMs
        self.contentDurationMs = contentDurationMs
        self.generationDurationMs = generationDurationMs
        self.peakMemoryGB = peakMemoryGB
        self.stopReason = stopReason
        self.cacheReport = cacheReport
        self.speculativeReport = speculativeReport
        self.activeComputePlacement = activeComputePlacement
    }
}

/// Full visibility and provenance descriptor for a registered model (`GET /v1/models`).
public struct EngineDescription: Codable, Identifiable, Sendable, Equatable {
    public var id: String { modelID }
    public let modelID: String
    public let provider: EngineProviderKind
    public let modelPath: String
    public let architecture: ProvenanceField<String>
    public let quantization: ProvenanceField<String>
    public let computePlacement: ProvenanceField<ComputePlacement>
    public var isPinned: Bool
    public var isLoaded: Bool
    public let contextSize: ProvenanceField<Int>
    public let flashAttention: ProvenanceField<Bool>
    public var supportsVision: ProvenanceField<Bool>
    public let supportsAudio: ProvenanceField<Bool>
    public let supportsEmbeddings: ProvenanceField<Bool>
    public let supportsThinking: ProvenanceField<Bool>
    public let thinkingBudget: ProvenanceField<Int?>
    public let reasoningFormat: ProvenanceField<String>
    public let thinkingControls: ThinkingControls?
    public let templateSource: ProvenanceField<String>
    public let speculativeDrafter: ProvenanceField<String?>
    public let estimatedMemoryMB: Int
    public let tags: [String]

    /// True when this entry represents a dedicated ASR speech recognizer (`CoreAISpeech` Whisper / Parakeet TDT
    /// or Apple `SpeechTranscriber`) rather than an autoregressive text-chat model.
    public var isSpeechRecognizer: Bool {
        let arch = architecture.value.lowercased()
        if arch == "whisper"
            || arch == "parakeet_tdt"
            || arch == "speech_recognizer"
            || arch == "apple_speech_transcriber" {
            return true
        }
        return tags.contains { ["asr", "speech_recognizer"].contains($0.lowercased()) }
    }

    public init(
        modelID: String,
        provider: EngineProviderKind,
        modelPath: String,
        architecture: ProvenanceField<String> = ProvenanceField(value: "llm", provenance: .derived, effect: .reload),
        quantization: ProvenanceField<String> = ProvenanceField(value: "native", provenance: .derived, effect: .reload),
        computePlacement: ProvenanceField<ComputePlacement> = ProvenanceField(value: .mlxGPU, provenance: .derived, effect: .reload, computePlacement: .mlxGPU),
        isPinned: Bool = false,
        isLoaded: Bool = false,
        contextSize: ProvenanceField<Int> = ProvenanceField(value: 8192, provenance: .derived, effect: .reload),
        flashAttention: ProvenanceField<Bool> = ProvenanceField(value: true, provenance: .derived, effect: .reload),
        supportsVision: ProvenanceField<Bool> = ProvenanceField(value: false, provenance: .derived, effect: .reload),
        supportsAudio: ProvenanceField<Bool> = ProvenanceField(value: false, provenance: .derived, effect: .reload),
        supportsEmbeddings: ProvenanceField<Bool> = ProvenanceField(value: false, provenance: .derived, effect: .reload),
        supportsThinking: ProvenanceField<Bool> = ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
        thinkingBudget: ProvenanceField<Int?> = ProvenanceField(value: nil, provenance: .notApplicable, effect: .perRequest),
        reasoningFormat: ProvenanceField<String> = ProvenanceField(value: "tag_pair", provenance: .derived, effect: .reload),
        thinkingControls: ThinkingControls? = nil,
        templateSource: ProvenanceField<String> = ProvenanceField(value: "tokenizer_config.json", provenance: .derived, effect: .reload),
        speculativeDrafter: ProvenanceField<String?> = ProvenanceField(value: nil, provenance: .notApplicable, effect: .reload),
        estimatedMemoryMB: Int = 0,
        tags: [String] = []
    ) {
        self.modelID = modelID
        self.provider = provider
        self.modelPath = modelPath
        self.architecture = architecture
        self.quantization = quantization
        self.computePlacement = computePlacement
        self.isPinned = isPinned
        self.isLoaded = isLoaded
        self.contextSize = contextSize
        self.flashAttention = flashAttention
        self.supportsVision = supportsVision
        self.supportsAudio = supportsAudio
        self.supportsEmbeddings = supportsEmbeddings
        self.supportsThinking = supportsThinking
        self.thinkingBudget = thinkingBudget
        self.reasoningFormat = reasoningFormat
        self.thinkingControls = thinkingControls
        self.templateSource = templateSource
        self.speculativeDrafter = speculativeDrafter
        self.estimatedMemoryMB = estimatedMemoryMB
        self.tags = tags
    }
}
