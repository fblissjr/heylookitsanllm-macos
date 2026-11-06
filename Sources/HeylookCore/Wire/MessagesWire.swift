import Foundation

/// Anthropic Messages-conformant `/v1/messages` request payload.
public struct MessagesRequest: Codable, Sendable, Equatable {
    public struct MessageTurn: Codable, Sendable, Equatable {
        public let role: String
        public let content: String

        public init(role: String, content: String) {
            self.role = role
            self.content = content
        }
    }

    public let model: String
    public let messages: [MessageTurn]
    public var system: String?
    public var maxTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var stream: Bool?
    public var thinking: Bool?
    public var thinkingBudget: Int?
    public var stopSequences: [String]?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case system
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case stream
        case thinking
        case thinkingBudget = "thinking_budget"
        case stopSequences = "stop_sequences"
    }

    public init(
        model: String,
        messages: [MessageTurn],
        system: String? = nil,
        maxTokens: Int? = 2048,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        stream: Bool? = false,
        thinking: Bool? = nil,
        thinkingBudget: Int? = nil,
        stopSequences: [String]? = nil
    ) {
        self.model = model
        self.messages = messages
        self.system = system
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.stream = stream
        self.thinking = thinking
        self.thinkingBudget = thinkingBudget
        self.stopSequences = stopSequences
    }

    public var samplerOverlay: SamplerSettings {
        SamplerSettings(
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            maxTokens: maxTokens,
            thinking: thinking,
            thinkingBudget: thinkingBudget,
            stopSequences: stopSequences
        )
    }
}

/// Anthropic Messages-conformant `/v1/messages` response payload (with `heylook` `performance` extension).
public struct MessagesResponse: Codable, Sendable, Equatable {
    public struct ContentBlock: Codable, Sendable, Equatable {
        public let type: String
        public let text: String?
        public let thinking: String?

        public init(type: String, text: String? = nil, thinking: String? = nil) {
            self.type = type
            self.text = text
            self.thinking = thinking
        }

        public static func text(_ value: String) -> ContentBlock {
            ContentBlock(type: "text", text: value)
        }

        public static func thinking(_ value: String) -> ContentBlock {
            ContentBlock(type: "thinking", thinking: value)
        }
    }

    public struct Usage: Codable, Sendable, Equatable {
        public let inputTokens: Int
        public let outputTokens: Int
        public let cacheReadInputTokens: Int?

        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
            case cacheReadInputTokens = "cache_read_input_tokens"
        }

        public init(inputTokens: Int, outputTokens: Int, cacheReadInputTokens: Int? = nil) {
            self.inputTokens = inputTokens
            self.outputTokens = outputTokens
            self.cacheReadInputTokens = cacheReadInputTokens
        }
    }

    public struct PerformanceExtension: Codable, Sendable, Equatable {
        public let generationTPS: Double
        public let promptTPS: Double
        public let computePlacement: ComputePlacement
        public let cache: CacheReport?
        public let speculative: SpeculativeReport?

        enum CodingKeys: String, CodingKey {
            case generationTPS = "generation_tps"
            case promptTPS = "prompt_tps"
            case computePlacement = "compute_placement"
            case cache
            case speculative
        }

        public init(
            generationTPS: Double,
            promptTPS: Double,
            computePlacement: ComputePlacement,
            cache: CacheReport? = nil,
            speculative: SpeculativeReport? = nil
        ) {
            self.generationTPS = generationTPS
            self.promptTPS = promptTPS
            self.computePlacement = computePlacement
            self.cache = cache
            self.speculative = speculative
        }
    }

    public let id: String
    public let type: String
    public let role: String
    public let model: String
    public let content: [ContentBlock]
    public let stopReason: StopReason
    public let usage: Usage
    public let performance: PerformanceExtension?

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case role
        case model
        case content
        case stopReason = "stop_reason"
        case usage
        case performance
    }

    public init(
        id: String = "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
        model: String,
        content: [ContentBlock],
        stopReason: StopReason = .endTurn,
        usage: Usage,
        performance: PerformanceExtension? = nil
    ) {
        self.id = id
        self.type = "message"
        self.role = "assistant"
        self.model = model
        self.content = content
        self.stopReason = stopReason
        self.usage = usage
        self.performance = performance
    }
}

/// Request payload for `POST /v1/conversations/{id}/generate`.
public struct ConversationGenerateRequest: Codable, Sendable, Equatable {
    public let model: String
    public let content: String
    public var presetName: String?
    public var sampling: SamplerSettings?

    enum CodingKeys: String, CodingKey {
        case model
        case content
        case presetName = "preset"
        case sampling
    }

    public init(
        model: String,
        content: String,
        presetName: String? = nil,
        sampling: SamplerSettings? = nil
    ) {
        self.model = model
        self.content = content
        self.presetName = presetName
        self.sampling = sampling
    }
}

/// Final `event: heylook_saved` payload emitted on `/v1/conversations/{id}/generate` once the
/// assistant turn is durably written to `SQLiteStore`.
public struct HeylookSavedEvent: Codable, Sendable, Equatable {
    public let conversationID: String
    public let userMessageID: String
    public let assistantMessageID: String
    public let position: Int
    public let stopReason: StopReason

    enum CodingKeys: String, CodingKey {
        case conversationID = "conversation_id"
        case userMessageID = "user_message_id"
        case assistantMessageID = "assistant_message_id"
        case position
        case stopReason = "stop_reason"
    }

    public init(
        conversationID: String,
        userMessageID: String,
        assistantMessageID: String,
        position: Int,
        stopReason: StopReason
    ) {
        self.conversationID = conversationID
        self.userMessageID = userMessageID
        self.assistantMessageID = assistantMessageID
        self.position = position
        self.stopReason = stopReason
    }
}

/// Process-global registry of active generation requests supporting immediate cancellation
/// (`DELETE /v1/requests/{id}`) and mid-stream `reasoning-end` (`POST /v1/requests/{id}/reasoning-end`).
public actor RequestRegistry {
    public struct ActiveRequest: Sendable {
        public let requestID: String
        public let conversationID: String?
        public let modelID: String
        public let startedAt: Date
        public var isAborted: Bool
        public var isReasoningEndRequested: Bool
    }

    private var active: [String: ActiveRequest] = [:]
    private var conversationToRequestID: [String: String] = [:]

    public init() {}

    public func register(requestID: String, conversationID: String? = nil, modelID: String) {
        let entry = ActiveRequest(
            requestID: requestID,
            conversationID: conversationID,
            modelID: modelID,
            startedAt: Date(),
            isAborted: false,
            isReasoningEndRequested: false
        )
        active[requestID] = entry
        if let conversationID {
            conversationToRequestID[conversationID] = requestID
        }
    }

    public func finish(requestID: String) {
        if let entry = active.removeValue(forKey: requestID),
           let convID = entry.conversationID,
           conversationToRequestID[convID] == requestID {
            conversationToRequestID.removeValue(forKey: convID)
        }
    }

    @discardableResult
    public func abort(requestID: String) -> Bool {
        guard var entry = active[requestID] else { return false }
        entry.isAborted = true
        active[requestID] = entry
        return true
    }

    @discardableResult
    public func requestReasoningEnd(requestID: String) -> Bool {
        guard var entry = active[requestID] else { return false }
        entry.isReasoningEndRequested = true
        active[requestID] = entry
        return true
    }

    @discardableResult
    public func requestReasoningEnd(forConversationID conversationID: String) -> Bool {
        guard let reqID = conversationToRequestID[conversationID] else { return false }
        return requestReasoningEnd(requestID: reqID)
    }

    public func state(for requestID: String) -> (isAborted: Bool, isReasoningEndRequested: Bool) {
        guard let entry = active[requestID] else { return (false, false) }
        return (entry.isAborted, entry.isReasoningEndRequested)
    }

    public func activeRequests() -> [ActiveRequest] {
        Array(active.values).sorted { $0.startedAt < $1.startedAt }
    }
}
