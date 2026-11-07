import Foundation

/// Anthropic Messages-conformant `/v1/messages` request payload.
public struct MessagesRequest: Codable, Sendable, Equatable {
    /// Multimodal attachment block (`image` or `audio`) carried inside a `MessageTurn`.
    public struct MediaAttachment: Codable, Sendable, Equatable, Identifiable {
        public let id: String
        /// `"image"` or `"audio"`
        public let kind: String
        /// MIME type (`"image/jpeg"`, `"image/png"`, `"audio/wav"`, etc.)
        public let mediaType: String
        /// Base64-encoded payload bytes
        public let dataBase64: String
        /// Display filename for UI pills
        public let filename: String?

        /// Estimated soft-token budget on vision/audio towers (`<=280` for image, `750` for 30s audio).
        public var estimatedTokens: Int {
            kind == "audio" ? 750 : 280
        }

        public init(
            id: String = UUID().uuidString,
            kind: String,
            mediaType: String,
            dataBase64: String,
            filename: String? = nil
        ) {
            self.id = id
            self.kind = kind
            self.mediaType = mediaType
            self.dataBase64 = dataBase64
            self.filename = filename
        }

        /// Loads a local image or audio file into a `MediaAttachment`.
        public static func fromFileURL(_ url: URL) throws -> MediaAttachment {
            let data = try Data(contentsOf: url)
            let ext = url.pathExtension.lowercased()
            let (kind, mime): (String, String)
            switch ext {
            case "png":
                (kind, mime) = ("image", "image/png")
            case "webp":
                (kind, mime) = ("image", "image/webp")
            case "gif":
                (kind, mime) = ("image", "image/gif")
            case "jpg", "jpeg", "heic":
                (kind, mime) = ("image", "image/jpeg")
            case "wav":
                (kind, mime) = ("audio", "audio/wav")
            case "mp3":
                (kind, mime) = ("audio", "audio/mp3")
            case "m4a", "aac":
                (kind, mime) = ("audio", "audio/mp4")
            case "flac":
                (kind, mime) = ("audio", "audio/flac")
            default:
                (kind, mime) = ("image", "image/jpeg")
            }
            return MediaAttachment(
                kind: kind,
                mediaType: mime,
                dataBase64: data.base64EncodedString(),
                filename: url.lastPathComponent
            )
        }
    }

    public struct MessageTurn: Codable, Sendable, Equatable {
        public let role: String
        public let content: String
        public var thinking: String?
        public var attachments: [MediaAttachment]

        public init(
            role: String,
            content: String,
            thinking: String? = nil,
            attachments: [MediaAttachment] = []
        ) {
            self.role = role
            self.content = content
            self.thinking = thinking
            self.attachments = attachments
        }

        private enum CodingKeys: String, CodingKey {
            case role
            case content
            case thinking
            case reasoningContent = "reasoning_content"
        }

        private struct WireContentBlock: Codable {
            struct Source: Codable {
                let type: String
                let media_type: String?
                let data: String?
            }
            let type: String
            let text: String?
            let thinking: String?
            let source: Source?
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.role = try container.decode(String.self, forKey: .role)
            let explicitThinking = (try? container.decodeIfPresent(String.self, forKey: .thinking))
                ?? (try? container.decodeIfPresent(String.self, forKey: .reasoningContent))
            if let plainText = try? container.decode(String.self, forKey: .content) {
                self.content = plainText
                self.thinking = explicitThinking
                self.attachments = []
            } else if let blocks = try? container.decode([WireContentBlock].self, forKey: .content) {
                var textParts: [String] = []
                var thinkingParts: [String] = []
                var media: [MediaAttachment] = []
                for block in blocks {
                    if block.type == "text", let t = block.text {
                        textParts.append(t)
                    } else if block.type == "thinking", let th = block.thinking ?? block.text {
                        thinkingParts.append(th)
                    } else if (block.type == "image" || block.type == "audio"),
                              let src = block.source,
                              let b64 = src.data {
                        media.append(
                            MediaAttachment(
                                kind: block.type,
                                mediaType: src.media_type ?? (block.type == "audio" ? "audio/wav" : "image/jpeg"),
                                dataBase64: b64
                            )
                        )
                    }
                }
                self.content = textParts.joined(separator: "\n")
                self.thinking = explicitThinking ?? (thinkingParts.isEmpty ? nil : thinkingParts.joined(separator: "\n"))
                self.attachments = media
            } else {
                self.content = ""
                self.thinking = explicitThinking
                self.attachments = []
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(role, forKey: .role)
            if let thinking, !thinking.isEmpty {
                try container.encode(thinking, forKey: .thinking)
            }
            if attachments.isEmpty {
                try container.encode(content, forKey: .content)
            } else {
                var blocks: [WireContentBlock] = attachments.map { att in
                    WireContentBlock(
                        type: att.kind,
                        text: nil,
                        thinking: nil,
                        source: .init(type: "base64", media_type: att.mediaType, data: att.dataBase64)
                    )
                }
                if !content.isEmpty {
                    blocks.append(WireContentBlock(type: "text", text: content, thinking: nil, source: nil))
                }
                try container.encode(blocks, forKey: .content)
            }
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
    public var repetitionPenalty: Double?
    public var repetitionContextSize: Int?
    public var presencePenalty: Double?
    public var seed: Int?
    public var stream: Bool?
    public var thinking: Bool?
    public var thinkingDepth: String?
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
        case repetitionPenalty = "repetition_penalty"
        case repetitionContextSize = "repetition_context_size"
        case presencePenalty = "presence_penalty"
        case seed
        case stream
        case thinking
        case thinkingDepth = "thinking_depth"
        case reasoningEffort = "reasoning_effort"
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
        repetitionPenalty: Double? = nil,
        repetitionContextSize: Int? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        stream: Bool? = false,
        thinking: Bool? = nil,
        thinkingDepth: String? = nil,
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
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.stream = stream
        self.thinking = thinking
        self.thinkingDepth = thinkingDepth
        self.thinkingBudget = thinkingBudget
        self.stopSequences = stopSequences
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.model = try container.decode(String.self, forKey: .model)
        self.messages = try container.decode([MessageTurn].self, forKey: .messages)
        self.system = try container.decodeIfPresent(String.self, forKey: .system)
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        self.temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        self.topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        self.topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        self.minP = try container.decodeIfPresent(Double.self, forKey: .minP)
        self.repetitionPenalty = try container.decodeIfPresent(Double.self, forKey: .repetitionPenalty)
        self.repetitionContextSize = try container.decodeIfPresent(Int.self, forKey: .repetitionContextSize)
        self.presencePenalty = try container.decodeIfPresent(Double.self, forKey: .presencePenalty)
        self.seed = try container.decodeIfPresent(Int.self, forKey: .seed)
        self.stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        self.thinking = try container.decodeIfPresent(Bool.self, forKey: .thinking)
        self.thinkingDepth = try container.decodeIfPresent(String.self, forKey: .thinkingDepth)
            ?? container.decodeIfPresent(String.self, forKey: .reasoningEffort)
        self.thinkingBudget = try container.decodeIfPresent(Int.self, forKey: .thinkingBudget)
        self.stopSequences = try container.decodeIfPresent([String].self, forKey: .stopSequences)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(model, forKey: .model)
        try container.encode(messages, forKey: .messages)
        try container.encodeIfPresent(system, forKey: .system)
        try container.encodeIfPresent(maxTokens, forKey: .maxTokens)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(topP, forKey: .topP)
        try container.encodeIfPresent(topK, forKey: .topK)
        try container.encodeIfPresent(minP, forKey: .minP)
        try container.encodeIfPresent(repetitionPenalty, forKey: .repetitionPenalty)
        try container.encodeIfPresent(repetitionContextSize, forKey: .repetitionContextSize)
        try container.encodeIfPresent(presencePenalty, forKey: .presencePenalty)
        try container.encodeIfPresent(seed, forKey: .seed)
        try container.encodeIfPresent(stream, forKey: .stream)
        try container.encodeIfPresent(thinking, forKey: .thinking)
        try container.encodeIfPresent(thinkingDepth, forKey: .thinkingDepth)
        try container.encodeIfPresent(thinkingBudget, forKey: .thinkingBudget)
        try container.encodeIfPresent(stopSequences, forKey: .stopSequences)
    }

    public var samplerOverlay: SamplerSettings {
        SamplerSettings(
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize,
            presencePenalty: presencePenalty,
            maxTokens: maxTokens,
            seed: seed,
            thinking: thinking,
            thinkingDepth: thinkingDepth,
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
