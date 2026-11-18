import Foundation
import HeylookCore

// MARK: - Task-Local Progress Bridge

/// Delivers live prefill progress callbacks `(processedTokens, totalTokens)` from any `InferenceProvider`
/// without requiring every legacy caller to switch away from `generate(requestID:request:effectiveSampling:registry:)`.
public enum PrefillProgressBridge {
    @TaskLocal
    public static var current: (@Sendable (Int, Int) async -> Void)?
}

// MARK: - Runtime Adapter Contract Extensions

public struct ImagePlanRow: Sendable, Codable, Equatable {
    public let size: [Int]
    public let tokens: Int
    public let target: [Int]?

    public init(size: [Int], tokens: Int, target: [Int]?) {
        self.size = size
        self.tokens = tokens
        self.target = target
    }
}

public struct ImagePlanResult: Sendable, Codable, Equatable {
    public let engine: String
    public let images: [ImagePlanRow]
    public let source: String

    public init(engine: String, images: [ImagePlanRow], source: String) {
        self.engine = engine
        self.images = images
        self.source = source
    }
}

public struct PromptPreviewResult: Sendable, Codable, Equatable {
    public let prompt: String
    public let tokenCount: Int?
    public let tokens: [Int]?
    public let templateSource: String
    public let chatTemplateCaps: [String: Bool]?

    enum CodingKeys: String, CodingKey {
        case prompt
        case tokenCount = "token_count"
        case tokens
        case templateSource = "template_source"
        case chatTemplateCaps = "chat_template_caps"
    }

    public init(
        prompt: String,
        tokenCount: Int?,
        tokens: [Int]? = nil,
        templateSource: String,
        chatTemplateCaps: [String: Bool]? = nil
    ) {
        self.prompt = prompt
        self.tokenCount = tokenCount
        self.tokens = tokens
        self.templateSource = templateSource
        self.chatTemplateCaps = chatTemplateCaps
    }
}

/// Live runtime facts observed from a resident provider, fed directly into `HeylookConfig.buildEngineContract(...)`.
public struct ObservedContractState: Sendable, Equatable {
    public let runningContextLength: Int?
    public let loadedTemplateText: String?
    public let runtimeCaps: [String: Bool]?
    public let speculativeInForce: Bool?
    public let loadedSnapshot: [String: JSONValue]?
    public let isPrefixCacheWarm: Bool
    public let contextUsed: Int?

    public init(
        runningContextLength: Int? = nil,
        loadedTemplateText: String? = nil,
        runtimeCaps: [String: Bool]? = nil,
        speculativeInForce: Bool? = nil,
        loadedSnapshot: [String: JSONValue]? = nil,
        isPrefixCacheWarm: Bool = false,
        contextUsed: Int? = nil
    ) {
        self.runningContextLength = runningContextLength
        self.loadedTemplateText = loadedTemplateText
        self.runtimeCaps = runtimeCaps
        self.speculativeInForce = speculativeInForce
        self.loadedSnapshot = loadedSnapshot
        self.isPrefixCacheWarm = isPrefixCacheWarm
        self.contextUsed = contextUsed
    }
}

public extension InferenceProvider {
    /// Plans image token counts and target resized dimensions through the live resident processor / server.
    func planImages(sizes: [[Int]]) async throws -> ImagePlanResult {
        try ModelSpecializationRegistry.validateImagePlanSizes(sizes)
        let desc = await describe()
        guard desc.supportsVision.value else {
            throw ProviderExecutionError.invalidRequest(
                detail: "Model '\(modelID)' does not support image input."
            )
        }
        throw ProviderExecutionError.runtimeUnavailable(
            modelID: modelID,
            detail: "Provider '\(providerKind.rawValue)' for '\(modelID)' does not expose live image-plan inspection."
        )
    }

    /// Renders `request` through the live runtime template + tokenizer without running decode.
    func renderPromptPreview(request: MessagesRequest) async throws -> PromptPreviewResult {
        let desc = await describe()
        let modelURL = URL(fileURLWithPath: desc.modelPath)
        let modelDir = HeylookConfig.resolveModelDirectoryURL(from: modelURL.path)
        let enableThinking = ModelSpecializationRegistry.resolveThinkingEnabled(
            requestThinking: request.thinking,
            sidecarThinking: nil,
            modelCapable: desc.supportsThinking.value
        )
        let rendered = try HeylookConfig.renderPromptDetailed(
            in: FileManager.default.fileExists(atPath: modelDir.path) ? modelDir : nil,
            system: request.system,
            turns: request.messages,
            thinkingEnabled: enableThinking,
            thinkingDepth: request.thinkingDepth
        )
        return PromptPreviewResult(
            prompt: rendered.prompt,
            tokenCount: nil,
            tokens: nil,
            templateSource: rendered.source,
            chatTemplateCaps: nil
        )
    }

    /// Live contract snapshot used by `EngineContract` / `/v1/models` truth projections.
    func observedContractState() async -> ObservedContractState {
        ObservedContractState()
    }

    /// Executes a truthful 1-token warmup using the live runtime engine (never synthetic tokens).
    func warmUp1Token() async throws {
        let desc = await describe()
        guard desc.isLoaded else {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Cannot warm up '\(modelID)': model is not loaded."
            )
        }
        var sampling = SamplerSettings()
        sampling.temperature = 0
        sampling.maxTokens = 1
        let warmupReq = MessagesRequest(
            model: modelID,
            system: nil,
            messages: [.init(role: "user", content: "hi")],
            maxTokens: 1,
            temperature: 0,
            stopSequences: nil,
            stream: false,
            thinking: false
        )
        let stream = await generate(
            requestID: "warmup-\(UUID().uuidString)",
            request: warmupReq,
            effectiveSampling: sampling,
            registry: .shared
        )
        for try await _ in stream {}
    }
}

// MARK: - Checkpoint Signature (Architecture + Geometry Inspection, Never Folder Strings)

/// Inspected architectural fingerprint derived strictly from checkpoint files (`config.json` `model_type`,
/// hidden dimensions, vision pooling geometry, drafter metadata, and GGUF KV header flags).
public struct CheckpointSignature: Sendable, Equatable {
    public let modelType: String?
    public let hiddenSize: Int?
    public let slidingWindow: Int?
    public let layerTypes: [String]
    public let hasSlidingWindowAttention: Bool
    public let hasLinearOrRecurrentState: Bool
    public let patchSize: Int?
    public let poolingKernelSize: Int?
    public let maxSoftTokens: Int?
    public let draftModelType: String?
    public let draftBackboneHiddenSize: Int?
    public let draftMaxContext: Int?
    public let hasMMProj: Bool
    public let hasAudioEncoder: Bool

    public var isGemma4Family: Bool {
        guard let mt = modelType?.lowercased() else { return false }
        return mt == "gemma4" || mt == "gemma4_text" || mt == "gemma4_unified"
    }

    public var isHybridRecurrentOrSWA: Bool {
        hasLinearOrRecurrentState || hasSlidingWindowAttention
    }

    public static func inspect(
        modelDirectoryOrFile url: URL,
        drafterDirectory: URL? = nil,
        mmprojPath: String? = nil
    ) -> CheckpointSignature {
        let fm = FileManager.default
        let resolvedDir = HeylookConfig.resolveModelDirectoryURL(from: url.path)

        var modelType: String?
        var hiddenSize: Int?
        var slidingWindow: Int?
        var layerTypes: [String] = []
        var hasSWA = false
        var hasLinear = false
        var patchSize: Int?
        var poolingKernelSize: Int?
        var maxSoftTokens: Int?
        var hasAudio = false

        let ggufURL: URL? = url.pathExtension.lowercased() == "gguf"
            ? url
            : ((try? fm.contentsOfDirectory(at: resolvedDir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]))?
                .first { $0.pathExtension.lowercased() == "gguf" && !$0.lastPathComponent.lowercased().contains("mmproj") })

        if let ggufURL, let header = GGUFHeaderReader.readFileHeader(at: ggufURL) {
            modelType = header.architecture
            hasSWA = header.hasSlidingWindowAttention
            hasLinear = header.isHybridRecurrent
            hasAudio = header.hasAudioEncoder
        }

        if let cfgData = try? Data(contentsOf: resolvedDir.appendingPathComponent("config.json")),
           let raw = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any] {
            if let mt = raw["model_type"] as? String {
                modelType = mt
            }
            let textCfg = (raw["text_config"] as? [String: Any]) ?? raw
            if let hs = textCfg["hidden_size"] as? Int {
                hiddenSize = hs
            } else if let hs = raw["hidden_size"] as? Int {
                hiddenSize = hs
            }
            if let sw = textCfg["sliding_window"] as? Int, sw > 0 {
                slidingWindow = sw
                hasSWA = true
            }
            if let lts = textCfg["layer_types"] as? [String] {
                layerTypes = lts
                let lowered = lts.map { $0.lowercased() }
                if lowered.contains(where: { $0.contains("sliding") }) {
                    hasSWA = true
                }
                if lowered.contains(where: { $0.contains("linear") || $0.contains("mamba") || $0.contains("ssm") || $0.contains("delta") }) {
                    hasLinear = true
                }
            }
            if let visionCfg = raw["vision_config"] as? [String: Any] {
                patchSize = visionCfg["patch_size"] as? Int
                poolingKernelSize = visionCfg["pooling_kernel_size"] as? Int
                maxSoftTokens = (raw["vision_soft_tokens_per_image"] as? Int)
                    ?? (visionCfg["default_output_length"] as? Int)
            }
            if raw["audio_config"] != nil {
                hasAudio = true
            }
        }

        if let procData = try? Data(contentsOf: resolvedDir.appendingPathComponent("preprocessor_config.json")),
           let rawProc = try? JSONSerialization.jsonObject(with: procData) as? [String: Any] {
            let imgProc = (rawProc["image_processor"] as? [String: Any]) ?? rawProc
            if patchSize == nil { patchSize = imgProc["patch_size"] as? Int }
            if poolingKernelSize == nil { poolingKernelSize = imgProc["pooling_kernel_size"] as? Int }
            if maxSoftTokens == nil {
                maxSoftTokens = (imgProc["max_soft_tokens"] as? Int)
                    ?? (imgProc["num_soft_tokens"] as? Int)
                    ?? (rawProc["max_soft_tokens"] as? Int)
                    ?? (rawProc["image_seq_length"] as? Int)
            }
        }

        var draftModelType: String?
        var draftBackboneHiddenSize: Int?
        var draftMaxContext: Int?
        if let drafterDirectory,
           let dData = try? Data(contentsOf: drafterDirectory.appendingPathComponent("config.json")),
           let dRaw = try? JSONSerialization.jsonObject(with: dData) as? [String: Any] {
            draftModelType = dRaw["model_type"] as? String
            draftBackboneHiddenSize = dRaw["backbone_hidden_size"] as? Int
            let draftTextCfg = (dRaw["text_config"] as? [String: Any]) ?? dRaw
            draftMaxContext = (dRaw["draft_max_context"] as? Int)
                ?? (dRaw["max_context_length"] as? Int)
                ?? (draftTextCfg["max_position_embeddings"] as? Int)
        }

        let resolvedMMProj = mmprojPath ?? HeylookConfig.discoverMMProj(in: resolvedDir)
        if let resolvedMMProj,
           let mmHeader = GGUFHeaderReader.readFileHeader(at: URL(fileURLWithPath: resolvedMMProj)) {
            hasAudio = hasAudio || mmHeader.hasAudioEncoder
        }

        return CheckpointSignature(
            modelType: modelType,
            hiddenSize: hiddenSize,
            slidingWindow: slidingWindow,
            layerTypes: layerTypes,
            hasSlidingWindowAttention: hasSWA,
            hasLinearOrRecurrentState: hasLinear,
            patchSize: patchSize,
            poolingKernelSize: poolingKernelSize,
            maxSoftTokens: maxSoftTokens,
            draftModelType: draftModelType,
            draftBackboneHiddenSize: draftBackboneHiddenSize,
            draftMaxContext: draftMaxContext,
            hasMMProj: resolvedMMProj != nil,
            hasAudioEncoder: hasAudio
        )
    }
}

// MARK: - Composable Model Specialization Protocol & Implementations

public struct APCCheckpointPolicy: Sendable, Equatable {
    public let checkpointIntervalTokens: Int
    public let nearEndCaptures: Int
    public let maxEntries: Int
    public let snapshottingAtTurnAndSystemBoundaries: Bool

    public static let standard = APCCheckpointPolicy(
        checkpointIntervalTokens: 64,
        nearEndCaptures: 3,
        maxEntries: 16,
        snapshottingAtTurnAndSystemBoundaries: true
    )
}

public struct GGUFSpecializationFlags: Sendable, Equatable {
    public let disableCacheReuse: Bool
    public let ctxCheckpoints: Int?
    public let checkpointMinStep: Int?
    public let reasoningPreserve: Bool
}

public struct VisionGeometrySpec: Sendable, Equatable {
    public let alignmentMultiplePx: Int
    public let visionSoftTokensPerImage: Int
}

public protocol ModelSpecialization: Sendable {
    var name: String { get }
    func matches(_ signature: CheckpointSignature) -> Bool
    func apcCheckpointPolicy(for signature: CheckpointSignature) -> APCCheckpointPolicy
    func ggufFlags(for signature: CheckpointSignature) -> GGUFSpecializationFlags
    func visionGeometry(for signature: CheckpointSignature) -> VisionGeometrySpec?
    func validatesMTPDrafterCompatibility(_ signature: CheckpointSignature) -> Bool
    func allowsMTPAfterMultimodalPrefill(_ signature: CheckpointSignature) -> Bool
}

public extension ModelSpecialization {
    func apcCheckpointPolicy(for signature: CheckpointSignature) -> APCCheckpointPolicy {
        .standard
    }
    func ggufFlags(for signature: CheckpointSignature) -> GGUFSpecializationFlags {
        let needsCheckpointPolicy = signature.hasMMProj || signature.isHybridRecurrentOrSWA
        return GGUFSpecializationFlags(
            disableCacheReuse: needsCheckpointPolicy,
            ctxCheckpoints: needsCheckpointPolicy ? 32 : nil,
            checkpointMinStep: needsCheckpointPolicy ? 256 : nil,
            reasoningPreserve: true
        )
    }
    func visionGeometry(for signature: CheckpointSignature) -> VisionGeometrySpec? { nil }
    func validatesMTPDrafterCompatibility(_ signature: CheckpointSignature) -> Bool { false }
    func allowsMTPAfterMultimodalPrefill(_ signature: CheckpointSignature) -> Bool { false }
}

/// Gemma 4 specialization:
/// - Matches `gemma4` / `gemma4_text` / `gemma4_unified` strictly on inspected `model_type`
/// - Pairs `gemma4_assistant` where `backbone_hidden_size == target.hidden_size` (`draft-mtp`)
/// - Keeps MTP enabled after multimodal prefill (uses single-token prime step + Leviathan verification)
/// - Supplies exact `48px` (`16 * 3`) alignment + `vision_soft_tokens_per_image` geometry
/// - Disables `--cache-reuse` on GGUF and enables `--ctx-checkpoints 32 --checkpoint-min-step 256`
public struct Gemma4Specialization: ModelSpecialization {
    public let name = "gemma4"

    public init() {}

    public func matches(_ signature: CheckpointSignature) -> Bool {
        signature.isGemma4Family
    }

    public func visionGeometry(for signature: CheckpointSignature) -> VisionGeometrySpec? {
        let patch = max(1, signature.patchSize ?? 16)
        let pool = max(1, signature.poolingKernelSize ?? 3)
        let softTokens = max(1, signature.maxSoftTokens ?? 280)
        return VisionGeometrySpec(
            alignmentMultiplePx: patch * pool,
            visionSoftTokensPerImage: softTokens
        )
    }

    public func validatesMTPDrafterCompatibility(_ signature: CheckpointSignature) -> Bool {
        guard signature.isGemma4Family,
              signature.draftModelType?.lowercased() == "gemma4_assistant" else {
            return false
        }
        if let targetHidden = signature.hiddenSize,
           let draftBackboneHidden = signature.draftBackboneHiddenSize {
            return targetHidden == draftBackboneHidden
        }
        return true
    }

    public func allowsMTPAfterMultimodalPrefill(_ signature: CheckpointSignature) -> Bool {
        signature.isGemma4Family
    }

    public func ggufFlags(for signature: CheckpointSignature) -> GGUFSpecializationFlags {
        GGUFSpecializationFlags(
            disableCacheReuse: true,
            ctxCheckpoints: 32,
            checkpointMinStep: 256,
            reasoningPreserve: true
        )
    }
}

/// Specialization for hybrid linear/SSM/Mamba & sliding-window attention checkpoints (e.g., Qwen 3.5 hybrid, LFM2, Gemma 3/Mistral SWA).
/// Enforces boundary KV snapshot capture so multi-turn thinking conversations reuse turn/system checkpoints
/// cleanly without calling unsupported `trim(_:)` on non-trimmable recurrent states.
public struct HybridRecurrentSWASpecialization: ModelSpecialization {
    public let name = "hybrid_recurrent_swa"

    public init() {}

    public func matches(_ signature: CheckpointSignature) -> Bool {
        !signature.isGemma4Family && signature.isHybridRecurrentOrSWA
    }

    public func ggufFlags(for signature: CheckpointSignature) -> GGUFSpecializationFlags {
        GGUFSpecializationFlags(
            disableCacheReuse: true,
            ctxCheckpoints: 32,
            checkpointMinStep: 256,
            reasoningPreserve: true
        )
    }
}

/// Default general-purpose specialization for standard dense/MoE attention checkpoints.
public struct StandardAttentionSpecialization: ModelSpecialization {
    public let name = "standard"

    public init() {}

    public func matches(_ signature: CheckpointSignature) -> Bool {
        true
    }
}

public enum ModelSpecializationRegistry {
    public static let maxImagePlanSizes = 16
    public static let maxImagePlanSidePx = 16_384

    public static let all: [any ModelSpecialization] = [
        Gemma4Specialization(),
        HybridRecurrentSWASpecialization(),
        StandardAttentionSpecialization(),
    ]

    public static func resolve(for signature: CheckpointSignature) -> any ModelSpecialization {
        all.first { $0.matches(signature) } ?? StandardAttentionSpecialization()
    }

    public static func resolveThinkingEnabled(
        requestThinking: Bool?,
        sidecarThinking: Bool?,
        modelCapable: Bool
    ) -> Bool {
        guard modelCapable else { return false }
        return requestThinking ?? sidecarThinking ?? true
    }

    public static func validateImagePlanSizes(_ sizes: [[Int]]) throws {
        guard !sizes.isEmpty && sizes.count <= maxImagePlanSizes else {
            throw ProviderExecutionError.invalidRequest(
                detail: "Provide 1 to \(maxImagePlanSizes) [width, height] image size pairs."
            )
        }
        for pair in sizes {
            guard pair.count == 2 else {
                throw ProviderExecutionError.invalidRequest(
                    detail: "Each size entry must contain exactly [width, height]."
                )
            }
            let w = pair[0]
            let h = pair[1]
            guard (1...maxImagePlanSidePx).contains(w) && (1...maxImagePlanSidePx).contains(h) else {
                throw ProviderExecutionError.invalidRequest(
                    detail: "Image dimensions \(w)x\(h) are outside valid range 1...\(maxImagePlanSidePx)."
                )
            }
        }
    }
}
