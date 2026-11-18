import Foundation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines

/// Request payload for `POST /v1/conversations/{id}/voice-generate`.
public struct ConversationVoiceGenerateRequest: Codable, Sendable, Equatable {
    public var model: String
    public var audioBase64: String
    public var languageHint: String?
    public var asrModel: String?
    public var sampling: SamplerSettings?

    enum CodingKeys: String, CodingKey {
        case model
        case audioBase64 = "audio_base64"
        case languageHint = "language_hint"
        case asrModel = "asr_model"
        case sampling
    }

    public init(
        model: String,
        audioBase64: String,
        languageHint: String? = "en",
        asrModel: String? = nil,
        sampling: SamplerSettings? = nil
    ) {
        self.model = model
        self.audioBase64 = audioBase64
        self.languageHint = languageHint
        self.asrModel = asrModel
        self.sampling = sampling
    }
}

/// Request payload for `POST /v1/audio/transcriptions`.
public struct AudioTranscriptionRequest: Codable, Sendable, Equatable {
    public var audioBase64: String
    public var languageHint: String?
    public var model: String?

    enum CodingKeys: String, CodingKey {
        case audioBase64 = "audio_base64"
        case languageHint = "language_hint"
        case model
    }

    public init(
        audioBase64: String,
        languageHint: String? = "en",
        model: String? = nil
    ) {
        self.audioBase64 = audioBase64
        self.languageHint = languageHint
        self.model = model
    }
}

public struct ModelLoadResult: Codable, Sendable, Equatable {
    public let status: String
    public let modelID: String
    public let warmed: Bool?
    public let warmMs: Int?
    public let warmError: String?

    enum CodingKeys: String, CodingKey {
        case status
        case modelID = "model_id"
        case warmed
        case warmMs = "warm_ms"
        case warmError = "warm_error"
    }

    public init(
        status: String = "loaded",
        modelID: String,
        warmed: Bool? = nil,
        warmMs: Int? = nil,
        warmError: String? = nil
    ) {
        self.status = status
        self.modelID = modelID
        self.warmed = warmed
        self.warmMs = warmMs
        self.warmError = warmError
    }
}

public struct ModelUnloadResult: Codable, Sendable, Equatable {
    public let status: String
    public let modelID: String

    enum CodingKeys: String, CodingKey {
        case status
        case modelID = "model_id"
    }

    public init(status: String, modelID: String) {
        self.status = status
        self.modelID = modelID
    }
}

public struct ImagePlanRequestPayload: Codable, Sendable, Equatable {
    public let sizes: [[Int]]

    public init(sizes: [[Int]]) {
        self.sizes = sizes
    }
}

public struct ImagePlanWireResponse: Codable, Sendable, Equatable {
    public let modelID: String
    public let engine: String
    public let images: [ImagePlanRow]
    public let source: String

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case engine
        case images
        case source
    }

    public init(modelID: String, engine: String, images: [ImagePlanRow], source: String) {
        self.modelID = modelID
        self.engine = engine
        self.images = images
        self.source = source
    }
}

public struct PromptPreviewOverrides: Codable, Sendable, Equatable {
    public var model: String?
    public var system: String?
    public var thinking: ThinkingParameter?
    public var thinkingDepth: String?
    public var thinkingBudget: Int?

    enum CodingKeys: String, CodingKey {
        case model
        case system
        case thinking
        case thinkingDepth = "thinking_depth"
        case reasoningEffort = "reasoning_effort"
        case thinkingBudget = "thinking_budget"
        case enableThinking = "enable_thinking"
    }

    public init(
        model: String? = nil,
        system: String? = nil,
        thinking: ThinkingParameter? = nil,
        thinkingDepth: String? = nil,
        thinkingBudget: Int? = nil
    ) {
        self.model = model
        self.system = system
        self.thinking = thinking
        self.thinkingDepth = thinkingDepth
        self.thinkingBudget = thinkingBudget
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.model = try c.decodeIfPresent(String.self, forKey: .model)
        self.system = try c.decodeIfPresent(String.self, forKey: .system)
        self.thinking = try c.decodeIfPresent(ThinkingParameter.self, forKey: .thinking)
            ?? c.decodeIfPresent(ThinkingParameter.self, forKey: .enableThinking)
        self.thinkingDepth = try c.decodeIfPresent(String.self, forKey: .thinkingDepth)
            ?? c.decodeIfPresent(String.self, forKey: .reasoningEffort)
        self.thinkingBudget = try c.decodeIfPresent(Int.self, forKey: .thinkingBudget)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(model, forKey: .model)
        try c.encodeIfPresent(system, forKey: .system)
        try c.encodeIfPresent(thinking, forKey: .thinking)
        try c.encodeIfPresent(thinkingDepth, forKey: .thinkingDepth)
        try c.encodeIfPresent(thinkingBudget, forKey: .thinkingBudget)
    }
}

public struct PromptPreviewRequestPayload: Codable, Sendable, Equatable {
    public var mode: ConversationGenerateMode
    public var messageID: String?
    public var userContent: String?
    public var userParts: [MessagesRequest.MessageTurn.ContentPart]
    public var overrides: PromptPreviewOverrides?
    public var edits: [String: String]

    enum CodingKeys: String, CodingKey {
        case mode
        case messageID = "message_id"
        case userContent = "user_content"
        case content
        case overrides
        case edits
    }

    public init(
        mode: ConversationGenerateMode = .append,
        messageID: String? = nil,
        userContent: String? = nil,
        userParts: [MessagesRequest.MessageTurn.ContentPart] = [],
        overrides: PromptPreviewOverrides? = nil,
        edits: [String: String] = [:]
    ) {
        self.mode = mode
        self.messageID = messageID
        self.userContent = userContent
        self.userParts = userParts
        self.overrides = overrides
        self.edits = edits
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.mode = try c.decodeIfPresent(ConversationGenerateMode.self, forKey: .mode) ?? .append
        self.messageID = try c.decodeIfPresent(String.self, forKey: .messageID)
        self.overrides = try c.decodeIfPresent(PromptPreviewOverrides.self, forKey: .overrides)
        self.edits = try c.decodeIfPresent([String: String].self, forKey: .edits) ?? [:]

        var textValue: String? = nil
        var partsValue: [MessagesRequest.MessageTurn.ContentPart] = []
        if c.contains(.userContent) {
            if let s = try? c.decode(String.self, forKey: .userContent) {
                textValue = s
            } else if let parts = try? c.decode([MessagesRequest.MessageTurn.ContentPart].self, forKey: .userContent) {
                partsValue = parts
                textValue = parts.compactMap { part in
                    if case .text(let t) = part { return t }
                    return nil
                }.joined(separator: "\n")
            }
        } else if c.contains(.content) {
            if let s = try? c.decode(String.self, forKey: .content) {
                textValue = s
            } else if let parts = try? c.decode([MessagesRequest.MessageTurn.ContentPart].self, forKey: .content) {
                partsValue = parts
                textValue = parts.compactMap { part in
                    if case .text(let t) = part { return t }
                    return nil
                }.joined(separator: "\n")
            }
        }
        self.userContent = textValue
        self.userParts = partsValue
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(mode, forKey: .mode)
        try c.encodeIfPresent(messageID, forKey: .messageID)
        if !userParts.isEmpty {
            try c.encode(userParts, forKey: .userContent)
        } else {
            try c.encodeIfPresent(userContent, forKey: .userContent)
        }
        try c.encodeIfPresent(overrides, forKey: .overrides)
        if !edits.isEmpty {
            try c.encode(edits, forKey: .edits)
        }
    }
}

public struct PromptPreviewWireResponse: Codable, Sendable, Equatable {
    public let modelID: String
    public let prompt: String
    public let source: String
    public let markers: [String]
    public let continuation: String?
    public let droppedMedia: DroppedMediaCounts
    public let unrenderedMedia: Int
    public let tokenCount: Int?

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case prompt
        case source
        case markers
        case continuation
        case droppedMedia = "dropped_media"
        case unrenderedMedia = "unrendered_media"
        case tokenCount = "token_count"
    }

    public init(
        modelID: String,
        prompt: String,
        source: String,
        markers: [String],
        continuation: String?,
        droppedMedia: DroppedMediaCounts,
        unrenderedMedia: Int,
        tokenCount: Int? = nil
    ) {
        self.modelID = modelID
        self.prompt = prompt
        self.source = source
        self.markers = markers
        self.continuation = continuation
        self.droppedMedia = droppedMedia
        self.unrenderedMedia = unrenderedMedia
        self.tokenCount = tokenCount
    }
}

public struct ModelConfigPatchPayload: Codable, Sendable, Equatable {
    public let description: String?
    public let tags: [String]?
    public let capabilities: [String]?
    public let config: [String: JSONValue]?

    public init(
        description: String? = nil,
        tags: [String]? = nil,
        capabilities: [String]? = nil,
        config: [String: JSONValue]? = nil
    ) {
        self.description = description
        self.tags = tags
        self.capabilities = capabilities
        self.config = config
    }

    public init(from decoder: Decoder) throws {
        let allowed: Set<String> = ["description", "tags", "capabilities", "config"]
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        let extra = raw.keys.filter { !allowed.contains($0) }.sorted()
        if !extra.isEmpty {
            throw WireValidationError.unprocessable(
                "Unknown field(s) in model update body: \(extra.joined(separator: ", ")). Config keys must live under {\"config\": {...}}."
            )
        }
        let data = try JSONEncoder().encode(raw)
        enum CodingKeys: String, CodingKey { case description, tags, capabilities, config }
        struct Helper: Decodable {
            let description: String?
            let tags: [String]?
            let capabilities: [String]?
            let config: [String: JSONValue]?
        }
        let parsed = try JSONDecoder().decode(Helper.self, from: data)
        self.description = parsed.description
        self.tags = parsed.tags
        self.capabilities = parsed.capabilities
        self.config = parsed.config
    }
}

public struct AdminModelPatchResult: Codable, Sendable, Equatable {
    public let model: AdminModelWireRow
    public let reloadRequiredFields: [String]
    public let warning: String?

    enum CodingKeys: String, CodingKey {
        case model
        case reloadRequiredFields = "reload_required_fields"
        case warning
    }

    public init(model: AdminModelWireRow, reloadRequiredFields: [String], warning: String? = nil) {
        self.model = model
        self.reloadRequiredFields = reloadRequiredFields
        self.warning = warning
    }
}

public struct ScanConfigPayload: Codable, Sendable, Equatable {
    public let folders: [String]
    public let scanIntervalSeconds: Int
    public let modelsServed: Int
    public let warning: String?

    enum CodingKeys: String, CodingKey {
        case folders
        case scanIntervalSeconds = "scan_interval_seconds"
        case modelsServed = "models_served"
        case warning
    }

    public init(
        folders: [String],
        scanIntervalSeconds: Int = 900,
        modelsServed: Int = 0,
        warning: String? = nil
    ) {
        self.folders = folders
        self.scanIntervalSeconds = scanIntervalSeconds
        self.modelsServed = modelsServed
        self.warning = warning
    }
}

public struct EmbeddingWireRequest: Sendable, Equatable {
    public struct Item: Sendable, Equatable {
        public let content: [MessagesRequest.MessageTurn.ContentPart]
        public let promptName: String?
        public let hasExplicitPromptName: Bool
        public let videoWindowsByBlockIndex: [Int: VideoWindowOptions]
        public let hasInvalidMediaMIME: String?
    }

    public let model: String
    public let items: [Item]
    public let dimensions: Int?
}

public struct EmbeddingWireResponse: Codable, Sendable, Equatable {
    public struct WindowPayload: Codable, Sendable, Equatable {
        public let start: Double
        public let end: Double
        public let frameTimes: [Double]
        public let embedding: [Float]

        enum CodingKeys: String, CodingKey {
            case start
            case end
            case frameTimes = "frame_times"
            case embedding
        }
    }

    public struct VectorRow: Codable, Sendable, Equatable {
        public let index: Int
        public let embedding: [Float]
        public let promptName: String?
        public let windows: [WindowPayload]?

        enum CodingKeys: String, CodingKey {
            case index
            case embedding
            case promptName = "prompt_name"
            case windows
        }
    }

    public let object: String
    public let model: String
    public let dimensions: Int
    public let data: [VectorRow]
    public let inputTokens: Int
    public let requestID: String
    public let elapsedMs: Double

    enum CodingKeys: String, CodingKey {
        case object
        case model
        case dimensions
        case data
        case inputTokens = "input_tokens"
        case requestID = "request_id"
        case elapsedMs = "elapsed_ms"
    }
}

/// Canonical backend orchestration contract shared by the embedded HTTP/SSE server and
/// local/remote SwiftUI workbench views.
public protocol WorkbenchBackend: Sendable {
    func listPublicModels() async -> [PublicModelWireRow]
    func listAdminModels() async -> [AdminModelWireRow]
    func adminModel(id: String) async throws -> AdminModelWireRow
    func loadModel(id: String, warm: Bool) async throws -> ModelLoadResult
    func reloadModel(id: String, loadSettings: [String: JSONValue]?, warm: Bool) async throws -> ModelLoadResult
    func unloadModel(id: String) async throws -> ModelUnloadResult
    func modelStatus(id: String) async -> [String: JSONValue]
    func assessModelFit(id: String, configOverrides: [String: JSONValue], headroomGB: Double) async throws -> RAMFitEvaluator.FitReport
    func planImages(id: String, sizes: [[Int]]) async throws -> ImagePlanWireResponse
    func previewPrompt(conversationID: String, request: PromptPreviewRequestPayload) async throws -> PromptPreviewWireResponse
    func generateInConversation(
        conversationID: String,
        request: ConversationGenerateRequest,
        requestID: String,
        detachOnSubscriberExit: Bool,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> HeylookSavedEvent
    func createMessageResponse(
        request: MessagesRequest,
        requestID: String,
        cancellation: StreamCancellationSignal,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> MessagesResponse?
    func embed(
        request: EmbeddingWireRequest,
        requestID: String
    ) async throws -> EmbeddingWireResponse
    func getChatTemplate(id: String) async throws -> HeylookConfig.ChatTemplateState
    func setChatTemplate(id: String, template: String) async throws -> HeylookConfig.ChatTemplateState
    func deleteChatTemplate(id: String) async throws -> HeylookConfig.ChatTemplateState
}
