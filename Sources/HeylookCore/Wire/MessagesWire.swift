import Foundation

/// Arbitrary JSON value used for `response_format` JSON schemas and structured wire payloads.
public indirect enum JSONValue: Codable, Sendable, Equatable, Hashable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let i = try? container.decode(Int.self) {
            self = .int(i)
        } else if let d = try? container.decode(Double.self) {
            self = .double(d)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let arr = try? container.decode([JSONValue].self) {
            self = .array(arr)
        } else if let obj = try? container.decode([String: JSONValue].self) {
            self = .object(obj)
        } else {
            self = .null
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let b):
            try container.encode(b)
        case .int(let i):
            try container.encode(i)
        case .double(let d):
            try container.encode(d)
        case .string(let s):
            try container.encode(s)
        case .array(let arr):
            try container.encode(arr)
        case .object(let obj):
            try container.encode(obj)
        }
    }

    public func toFoundationObject() -> Any {
        switch self {
        case .null:
            return NSNull()
        case .bool(let b):
            return b
        case .int(let i):
            return i
        case .double(let d):
            return d
        case .string(let s):
            return s
        case .array(let arr):
            return arr.map { $0.toFoundationObject() }
        case .object(let obj):
            var out: [String: Any] = [:]
            for (k, v) in obj {
                out[k] = v.toFoundationObject()
            }
            return out
        }
    }
}

/// Structured output constraint (`response_format` on `/v1/messages`).
public struct ResponseFormat: Codable, Sendable, Equatable, Hashable {
    public struct JSONSchemaSpec: Codable, Sendable, Equatable, Hashable {
        public var name: String?
        public var description: String?
        public var schema: JSONValue?
        public var strict: Bool?

        public init(
            name: String? = nil,
            description: String? = nil,
            schema: JSONValue? = nil,
            strict: Bool? = nil
        ) {
            self.name = name
            self.description = description
            self.schema = schema
            self.strict = strict
        }
    }

    /// `"text"`, `"json_object"`, or `"json_schema"`.
    public var type: String
    public var jsonSchema: JSONSchemaSpec?
    public var schema: JSONValue?

    enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
        case schema
    }

    public init(
        type: String,
        jsonSchema: JSONSchemaSpec? = nil,
        schema: JSONValue? = nil
    ) {
        self.type = type
        self.jsonSchema = jsonSchema
        self.schema = schema
    }

    public func toOpenAIObject() -> [String: Any] {
        var out: [String: Any] = ["type": type]
        if let jsonSchema {
            var specObj: [String: Any] = [:]
            if let name = jsonSchema.name { specObj["name"] = name }
            if let desc = jsonSchema.description { specObj["description"] = desc }
            if let strict = jsonSchema.strict { specObj["strict"] = strict }
            if let s = jsonSchema.schema { specObj["schema"] = s.toFoundationObject() }
            out["json_schema"] = specObj
        } else if let schema {
            out["schema"] = schema.toFoundationObject()
        }
        return out
    }
}

/// Anthropic Messages-conformant `/v1/messages` request payload.
public struct MessagesRequest: Codable, Sendable, Equatable {
    public enum MediaKind: String, Codable, Sendable, Equatable, Hashable {
        case image
        case audio
    }

    /// Multimodal attachment block (`image` or `audio`) carried inside a `MessageTurn`.
    public struct MediaAttachment: Codable, Sendable, Equatable, Hashable, Identifiable {
        public let id: String
        /// `"image"` or `"audio"`
        public let kind: String
        /// MIME type (`"image/jpeg"`, `"image/png"`, `"audio/wav"`, etc.)
        public let mediaType: String
        /// Base64-encoded payload bytes (or empty when referenced via `mediaID` in `SQLiteStore`)
        public let dataBase64: String
        /// Display filename for UI pills
        public let filename: String?
        /// Optional content-addressed SHA-256 media blob ID in `SQLiteStore.media_blobs`
        public var mediaID: String?

        /// Estimated soft-token budget on vision/audio towers (`<=280` for image, `750` for 30s audio).
        public var estimatedTokens: Int {
            kind == "audio" ? 750 : 280
        }

        public init(
            id: String = UUID().uuidString,
            kind: String,
            mediaType: String,
            dataBase64: String,
            filename: String? = nil,
            mediaID: String? = nil
        ) {
            self.id = id
            self.kind = kind
            self.mediaType = mediaType
            self.dataBase64 = dataBase64
            self.filename = filename
            self.mediaID = mediaID
        }

        public init(
            id: String = UUID().uuidString,
            kind: MediaKind,
            mimeType: String,
            base64Data: String,
            filename: String? = nil,
            mediaID: String? = nil
        ) {
            self.id = id
            self.kind = kind.rawValue
            self.mediaType = mimeType
            self.dataBase64 = base64Data
            self.filename = filename
            self.mediaID = mediaID
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

    public struct MessageTurn: Codable, Sendable, Equatable, Hashable {
        /// Ordered block inside a multimodal turn, preserving exact `[text, image, text]` interleaving.
        public enum ContentPart: Codable, Sendable, Equatable, Hashable {
            case text(String)
            case thinking(String)
            case media(MediaAttachment)

            private enum CodingKeys: String, CodingKey {
                case type
                case text
                case thinking
                case source
            }

            private struct SourcePayload: Codable {
                let type: String
                let media_type: String?
                let data: String?
                let media_id: String?
                let filename: String?
            }

            public init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                let type = try c.decode(String.self, forKey: .type)
                switch type {
                case "thinking":
                    let th = (try? c.decodeIfPresent(String.self, forKey: .thinking))
                        ?? (try? c.decodeIfPresent(String.self, forKey: .text))
                        ?? ""
                    self = .thinking(th)
                case "image", "audio":
                    let src = try? c.decodeIfPresent(SourcePayload.self, forKey: .source)
                    let mime = src?.media_type ?? (type == "audio" ? "audio/wav" : "image/jpeg")
                    let b64 = src?.data ?? ""
                    let mediaID = src?.media_id
                    self = .media(
                        MediaAttachment(
                            id: mediaID ?? UUID().uuidString,
                            kind: type,
                            mediaType: mime,
                            dataBase64: b64,
                            filename: src?.filename,
                            mediaID: mediaID
                        )
                    )
                default:
                    let txt = (try? c.decodeIfPresent(String.self, forKey: .text)) ?? ""
                    self = .text(txt)
                }
            }

            public func encode(to encoder: Encoder) throws {
                var c = encoder.container(keyedBy: CodingKeys.self)
                switch self {
                case .text(let t):
                    try c.encode("text", forKey: .type)
                    try c.encode(t, forKey: .text)
                case .thinking(let th):
                    try c.encode("thinking", forKey: .type)
                    try c.encode(th, forKey: .thinking)
                case .media(let att):
                    try c.encode(att.kind, forKey: .type)
                    if let mediaID = att.mediaID, att.dataBase64.isEmpty {
                        try c.encode(
                            SourcePayload(
                                type: "media_ref",
                                media_type: att.mediaType,
                                data: nil,
                                media_id: mediaID,
                                filename: att.filename
                            ),
                            forKey: .source
                        )
                    } else {
                        try c.encode(
                            SourcePayload(
                                type: "base64",
                                media_type: att.mediaType,
                                data: att.dataBase64,
                                media_id: att.mediaID,
                                filename: att.filename
                            ),
                            forKey: .source
                        )
                    }
                }
            }
        }

        public let role: String
        public let content: String
        public var thinking: String?
        public var attachments: [MediaAttachment]
        /// Exact ordered content parts (`[text, image, text]`), preserving multimodal block order.
        public var parts: [ContentPart]

        public init(
            role: String,
            content: String,
            thinking: String? = nil,
            attachments: [MediaAttachment] = [],
            parts: [ContentPart]? = nil
        ) {
            self.role = role
            self.content = content
            self.thinking = thinking
            self.attachments = attachments
            if let parts {
                self.parts = parts
            } else {
                var built: [ContentPart] = []
                if let thinking, !thinking.isEmpty {
                    built.append(.thinking(thinking))
                }
                for att in attachments {
                    built.append(.media(att))
                }
                if !content.isEmpty || built.isEmpty {
                    built.append(.text(content))
                }
                self.parts = built
            }
        }

        private enum CodingKeys: String, CodingKey {
            case role
            case content
            case thinking
            case reasoningContent = "reasoning_content"
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
                var built: [ContentPart] = []
                if let explicitThinking, !explicitThinking.isEmpty {
                    built.append(.thinking(explicitThinking))
                }
                built.append(.text(plainText))
                self.parts = built
            } else if let decodedParts = try? container.decode([ContentPart].self, forKey: .content) {
                var textParts: [String] = []
                var thinkingParts: [String] = []
                var media: [MediaAttachment] = []
                for part in decodedParts {
                    switch part {
                    case .text(let t):
                        textParts.append(t)
                    case .thinking(let th):
                        thinkingParts.append(th)
                    case .media(let att):
                        media.append(att)
                    }
                }
                self.content = textParts.joined(separator: "\n")
                let mergedThinking = explicitThinking ?? (thinkingParts.isEmpty ? nil : thinkingParts.joined(separator: "\n"))
                self.thinking = mergedThinking
                self.attachments = media
                if let explicitThinking, !explicitThinking.isEmpty, thinkingParts.isEmpty {
                    self.parts = [.thinking(explicitThinking)] + decodedParts
                } else {
                    self.parts = decodedParts
                }
            } else {
                self.content = ""
                self.thinking = explicitThinking
                self.attachments = []
                self.parts = explicitThinking.map { [.thinking($0), .text("")] } ?? [.text("")]
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
                let nonThinkingParts = parts.filter {
                    if case .thinking = $0 { return false }
                    return true
                }
                if !nonThinkingParts.isEmpty {
                    try container.encode(nonThinkingParts, forKey: .content)
                } else {
                    var fallback: [ContentPart] = attachments.map { .media($0) }
                    if !content.isEmpty {
                        fallback.append(.text(content))
                    }
                    try container.encode(fallback, forKey: .content)
                }
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
    public var responseFormat: ResponseFormat?

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
        case responseFormat = "response_format"
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
        stopSequences: [String]? = nil,
        responseFormat: ResponseFormat? = nil
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
        self.responseFormat = responseFormat
    }

    public init(
        model: String,
        system: String?,
        messages: [MessageTurn],
        maxTokens: Int? = 2048,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        repetitionContextSize: Int? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        stopSequences: [String]?,
        stream: Bool? = false,
        thinking: Bool? = nil,
        thinkingDepth: String? = nil,
        thinkingBudget: Int? = nil,
        responseFormat: ResponseFormat? = nil
    ) {
        self.init(
            model: model,
            messages: messages,
            system: system,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize,
            presencePenalty: presencePenalty,
            seed: seed,
            stream: stream,
            thinking: thinking,
            thinkingDepth: thinkingDepth,
            thinkingBudget: thinkingBudget,
            stopSequences: stopSequences,
            responseFormat: responseFormat
        )
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
        self.responseFormat = try container.decodeIfPresent(ResponseFormat.self, forKey: .responseFormat)
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
        try container.encodeIfPresent(responseFormat, forKey: .responseFormat)
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
            ContentBlock(type: "thinking", text: value, thinking: value)
        }
    }

    public struct Usage: Codable, Sendable, Equatable {
        public let inputTokens: Int
        public let outputTokens: Int
        public let cacheReadInputTokens: Int?
        public let thinkingTokens: Int?
        public let contentTokens: Int?

        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
            case cacheReadInputTokens = "cache_read_input_tokens"
            case thinkingTokens = "thinking_tokens"
            case contentTokens = "content_tokens"
        }

        public init(
            inputTokens: Int,
            outputTokens: Int,
            cacheReadInputTokens: Int? = nil,
            thinkingTokens: Int? = nil,
            contentTokens: Int? = nil
        ) {
            self.inputTokens = inputTokens
            self.outputTokens = outputTokens
            self.cacheReadInputTokens = cacheReadInputTokens
            self.thinkingTokens = thinkingTokens
            self.contentTokens = contentTokens
        }
    }

    public struct PerformanceExtension: Codable, Sendable, Equatable {
        public let generationTPS: Double
        public let promptTPS: Double
        public let tokensPerSecond: Double
        public let ttftMs: Double?
        public let thinkingDurationMs: Double?
        public let contentDurationMs: Double?
        public let generationDurationMs: Double?
        public let requestDurationMs: Double?
        public let computePlacement: ComputePlacement
        public let cache: CacheReport?
        public let speculative: SpeculativeReport?

        enum CodingKeys: String, CodingKey {
            case generationTPS = "generation_tps"
            case promptTPS = "prompt_tps"
            case tokensPerSecond = "tokens_per_second"
            case ttftMs = "ttft_ms"
            case thinkingDurationMs = "thinking_duration_ms"
            case contentDurationMs = "content_duration_ms"
            case generationDurationMs = "generation_duration_ms"
            case requestDurationMs = "request_duration_ms"
            case computePlacement = "compute_placement"
            case cache
            case speculative
        }

        public init(
            generationTPS: Double,
            promptTPS: Double,
            tokensPerSecond: Double? = nil,
            ttftMs: Double? = nil,
            thinkingDurationMs: Double? = nil,
            contentDurationMs: Double? = nil,
            generationDurationMs: Double? = nil,
            requestDurationMs: Double? = nil,
            computePlacement: ComputePlacement,
            cache: CacheReport? = nil,
            speculative: SpeculativeReport? = nil
        ) {
            self.generationTPS = generationTPS
            self.promptTPS = promptTPS
            self.tokensPerSecond = tokensPerSecond ?? generationTPS
            self.ttftMs = ttftMs
            self.thinkingDurationMs = thinkingDurationMs
            self.contentDurationMs = contentDurationMs
            self.generationDurationMs = generationDurationMs
            self.requestDurationMs = requestDurationMs
            self.computePlacement = computePlacement
            self.cache = cache
            self.speculative = speculative
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let genTPS = (try? c.decodeIfPresent(Double.self, forKey: .generationTPS))
                ?? (try? c.decodeIfPresent(Double.self, forKey: .tokensPerSecond))
                ?? 0
            self.generationTPS = genTPS
            self.promptTPS = (try? c.decodeIfPresent(Double.self, forKey: .promptTPS)) ?? 0
            self.tokensPerSecond = (try? c.decodeIfPresent(Double.self, forKey: .tokensPerSecond)) ?? genTPS
            self.ttftMs = try? c.decodeIfPresent(Double.self, forKey: .ttftMs)
            self.thinkingDurationMs = try? c.decodeIfPresent(Double.self, forKey: .thinkingDurationMs)
            self.contentDurationMs = try? c.decodeIfPresent(Double.self, forKey: .contentDurationMs)
            self.generationDurationMs = try? c.decodeIfPresent(Double.self, forKey: .generationDurationMs)
            self.requestDurationMs = try? c.decodeIfPresent(Double.self, forKey: .requestDurationMs)
            self.computePlacement = (try? c.decodeIfPresent(ComputePlacement.self, forKey: .computePlacement)) ?? .mlxGPU
            self.cache = try? c.decodeIfPresent(CacheReport.self, forKey: .cache)
            self.speculative = try? c.decodeIfPresent(SpeculativeReport.self, forKey: .speculative)
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

/// Saga mode for `POST /v1/conversations/{id}/generate` (`frontend_v3_spec.md` §4.3).
public enum ConversationGenerateMode: String, Codable, Sendable, Equatable {
    case append
    case regenerate
    case `continue` = "continue"
}

/// Request payload for `POST /v1/conversations/{id}/generate`.
public struct ConversationGenerateRequest: Codable, Sendable, Equatable {
    public let model: String
    public var mode: ConversationGenerateMode
    public var messageID: String?
    public let content: String
    public var userParts: [MessagesRequest.MessageTurn.ContentPart]
    public var system: String?
    public var thinking: Bool?
    public var thinkingDepth: String?
    public var thinkingBudget: Int?
    public var presetName: String?
    public var sampling: SamplerSettings?
    public var responseFormat: ResponseFormat?

    enum CodingKeys: String, CodingKey {
        case model
        case mode
        case messageID = "message_id"
        case content
        case userContent = "user_content"
        case system
        case thinking
        case thinkingDepth = "thinking_depth"
        case reasoningEffort = "reasoning_effort"
        case thinkingBudget = "thinking_budget"
        case presetName = "preset"
        case sampling
        case overrides
        case responseFormat = "response_format"
    }

    public init(
        model: String,
        content: String = "",
        mode: ConversationGenerateMode = .append,
        messageID: String? = nil,
        userParts: [MessagesRequest.MessageTurn.ContentPart]? = nil,
        system: String? = nil,
        thinking: Bool? = nil,
        thinkingDepth: String? = nil,
        thinkingBudget: Int? = nil,
        presetName: String? = nil,
        sampling: SamplerSettings? = nil,
        responseFormat: ResponseFormat? = nil
    ) {
        self.model = model
        self.mode = mode
        self.messageID = messageID
        self.content = content
        self.userParts = userParts ?? (content.isEmpty ? [] : [.text(content)])
        self.system = system
        self.thinking = thinking
        self.thinkingDepth = thinkingDepth
        self.thinkingBudget = thinkingBudget
        self.presetName = presetName
        self.sampling = sampling
        self.responseFormat = responseFormat
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.model = try c.decode(String.self, forKey: .model)
        self.mode = (try? c.decodeIfPresent(ConversationGenerateMode.self, forKey: .mode)) ?? .append
        self.messageID = try? c.decodeIfPresent(String.self, forKey: .messageID)
        self.system = try? c.decodeIfPresent(String.self, forKey: .system)
        self.thinking = try? c.decodeIfPresent(Bool.self, forKey: .thinking)
        self.thinkingDepth = (try? c.decodeIfPresent(String.self, forKey: .thinkingDepth))
            ?? (try? c.decodeIfPresent(String.self, forKey: .reasoningEffort))
        self.thinkingBudget = try? c.decodeIfPresent(Int.self, forKey: .thinkingBudget)
        self.presetName = try? c.decodeIfPresent(String.self, forKey: .presetName)
        self.sampling = (try? c.decodeIfPresent(SamplerSettings.self, forKey: .overrides))
            ?? (try? c.decodeIfPresent(SamplerSettings.self, forKey: .sampling))
        self.responseFormat = try? c.decodeIfPresent(ResponseFormat.self, forKey: .responseFormat)

        if let parts = try? c.decode([MessagesRequest.MessageTurn.ContentPart].self, forKey: .userContent) {
            self.userParts = parts
            self.content = parts.compactMap {
                if case .text(let t) = $0 { return t }
                return nil
            }.joined(separator: "\n")
        } else if let userStr = try? c.decode(String.self, forKey: .userContent) {
            self.content = userStr
            self.userParts = [.text(userStr)]
        } else if let legacyStr = try? c.decode(String.self, forKey: .content) {
            self.content = legacyStr
            self.userParts = [.text(legacyStr)]
        } else {
            self.content = ""
            self.userParts = []
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(model, forKey: .model)
        try c.encode(mode, forKey: .mode)
        try c.encodeIfPresent(messageID, forKey: .messageID)
        if userParts.contains(where: { if case .media = $0 { return true }; return false }) {
            try c.encode(userParts, forKey: .userContent)
        } else {
            try c.encode(content, forKey: .content)
        }
        try c.encodeIfPresent(system, forKey: .system)
        try c.encodeIfPresent(thinking, forKey: .thinking)
        try c.encodeIfPresent(thinkingDepth, forKey: .thinkingDepth)
        try c.encodeIfPresent(thinkingBudget, forKey: .thinkingBudget)
        try c.encodeIfPresent(presetName, forKey: .presetName)
        try c.encodeIfPresent(sampling, forKey: .sampling)
        try c.encodeIfPresent(responseFormat, forKey: .responseFormat)
    }
}

/// Final `event: heylook_saved` payload emitted on `/v1/conversations/{id}/generate` once the
/// assistant turn is durably written to `SQLiteStore`.
public struct HeylookSavedEvent: Codable, Sendable, Equatable {
    public let conversationID: String
    public let userMessageID: String?
    public let assistantMessageID: String
    public let position: Int
    public let stopReason: StopReason
    public let mode: ConversationGenerateMode
    public let droppedMedia: Int

    enum CodingKeys: String, CodingKey {
        case conversationID = "conversation_id"
        case userMessageID = "user_message_id"
        case assistantMessageID = "assistant_message_id"
        case position
        case stopReason = "stop_reason"
        case mode
        case droppedMedia = "dropped_media"
    }

    public init(
        conversationID: String,
        userMessageID: String?,
        assistantMessageID: String,
        position: Int,
        stopReason: StopReason,
        mode: ConversationGenerateMode = .append,
        droppedMedia: Int = 0
    ) {
        self.conversationID = conversationID
        self.userMessageID = userMessageID
        self.assistantMessageID = assistantMessageID
        self.position = position
        self.stopReason = stopReason
        self.mode = mode
        self.droppedMedia = droppedMedia
    }
}

/// Process-global registry of active generation requests supporting immediate cancellation
/// (`DELETE /v1/requests/{id}`), conversation-level lock arbitration (`409 generation_in_progress`),
/// and mid-stream `reasoning-end` (`POST /v1/requests/{id}/reasoning-end`).
public actor RequestRegistry {
    public struct ActiveRequest: Sendable {
        public let executionID: UUID
        public let requestID: String
        public let conversationID: String?
        public let modelID: String
        public let startedAt: Date
        public var isAborted: Bool
        public var isReasoningEndRequested: Bool
    }

    private final class StateBox: @unchecked Sendable {
        let lock = NSLock()
        var entriesByRequestID: [String: [ActiveRequest]] = [:]
        var cancelHandlersByExecution: [UUID: [@Sendable () -> Void]] = [:]
        var conversationLocks: [String: (requestID: String, executionID: UUID)] = [:]
    }

    public static let shared = RequestRegistry()

    private let box = StateBox()

    public init() {}

    /// Validates client-supplied `X-Request-ID` (`^[A-Za-z0-9._:-]{1,128}$`).
    public static func isValidClientRequestID(_ id: String) -> Bool {
        guard !id.isEmpty, id.utf8.count <= 128 else { return false }
        for byte in id.utf8 {
            let isAlphaNum = (byte >= 0x30 && byte <= 0x39)
                || (byte >= 0x41 && byte <= 0x5A)
                || (byte >= 0x61 && byte <= 0x7A)
            let isPunct = byte == 0x2E || byte == 0x5F || byte == 0x3A || byte == 0x2D
            if !isAlphaNum && !isPunct {
                return false
            }
        }
        return true
    }

    @discardableResult
    public nonisolated func register(requestID: String, conversationID: String? = nil, modelID: String) -> UUID {
        box.lock.lock()
        defer { box.lock.unlock() }
        let execID = UUID()
        let entry = ActiveRequest(
            executionID: execID,
            requestID: requestID,
            conversationID: conversationID,
            modelID: modelID,
            startedAt: Date(),
            isAborted: false,
            isReasoningEndRequested: false
        )
        box.entriesByRequestID[requestID, default: []].append(entry)
        if let conversationID, box.conversationLocks[conversationID] == nil {
            box.conversationLocks[conversationID] = (requestID: requestID, executionID: execID)
        }
        return execID
    }

    /// Synchronous registration shorthand used by UI `@MainActor` tasks.
    @discardableResult
    public nonisolated func register(id: String, modelID: String = "ui") -> UUID {
        register(requestID: id, conversationID: nil, modelID: modelID)
    }

    public nonisolated func unregister(id: String) {
        finish(requestID: id)
    }

    @discardableResult
    public nonisolated func cancel(id: String) -> Bool {
        abort(requestID: id)
    }

    @discardableResult
    public nonisolated func endThinking(id: String) -> Bool {
        requestReasoningEnd(requestID: id)
    }

    /// Atomically acquires a per-conversation generation lock or returns `nil` (`409 generation_in_progress`).
    public nonisolated func tryAcquireConversationLock(conversationID: String, requestID: String, modelID: String) -> UUID? {
        box.lock.lock()
        defer { box.lock.unlock() }
        if box.conversationLocks[conversationID] != nil {
            return nil
        }
        let execID = UUID()
        let entry = ActiveRequest(
            executionID: execID,
            requestID: requestID,
            conversationID: conversationID,
            modelID: modelID,
            startedAt: Date(),
            isAborted: false,
            isReasoningEndRequested: false
        )
        box.entriesByRequestID[requestID, default: []].append(entry)
        box.conversationLocks[conversationID] = (requestID: requestID, executionID: execID)
        return execID
    }

    public nonisolated func isConversationGenerating(_ conversationID: String) -> Bool {
        box.lock.lock()
        defer { box.lock.unlock() }
        return box.conversationLocks[conversationID] != nil
    }

    public nonisolated func activeRequestID(forConversationID conversationID: String) -> String? {
        box.lock.lock()
        defer { box.lock.unlock() }
        return box.conversationLocks[conversationID]?.requestID
    }

    /// Releases a per-conversation generation lock acquired via `tryAcquireConversationLock` and cleans up its request entry.
    public nonisolated func releaseConversationLock(
        conversationID: String,
        requestID: String? = nil,
        executionID: UUID? = nil
    ) {
        let resolved: (requestID: String, executionID: UUID)?
        box.lock.lock()
        if let locked = box.conversationLocks[conversationID] {
            let matchesReq = requestID == nil || locked.requestID == requestID
            let matchesExec = executionID == nil || locked.executionID == executionID
            if matchesReq && matchesExec {
                box.conversationLocks.removeValue(forKey: conversationID)
            }
            resolved = locked
        } else {
            resolved = nil
        }
        box.lock.unlock()

        if let targetReqID = requestID ?? resolved?.requestID {
            finish(requestID: targetReqID, executionID: executionID ?? resolved?.executionID)
        }
    }

    /// Registers an immediate cancellation callback (e.g., cancelling an MLX/CoreAI decode `Task`
    /// or `URLSession` stream) that fires synchronously if the request is or becomes aborted.
    public nonisolated func addCancelHandler(for requestID: String, executionID: UUID? = nil, handler: @escaping @Sendable () -> Void) {
        var fireImmediately = false
        box.lock.lock()
        if let list = box.entriesByRequestID[requestID], !list.isEmpty {
            let targetExecs = executionID.map { [$0] } ?? list.map(\.executionID)
            let alreadyAborted = list.contains { targetExecs.contains($0.executionID) && $0.isAborted }
            if alreadyAborted {
                fireImmediately = true
            } else {
                for execID in targetExecs {
                    box.cancelHandlersByExecution[execID, default: []].append(handler)
                }
            }
        }
        box.lock.unlock()
        if fireImmediately {
            handler()
        }
    }

    public nonisolated func finish(requestID: String, executionID: UUID? = nil) {
        box.lock.lock()
        defer { box.lock.unlock() }
        guard var list = box.entriesByRequestID[requestID] else { return }
        let removed: [ActiveRequest]
        if let executionID {
            removed = list.filter { $0.executionID == executionID }
            list.removeAll { $0.executionID == executionID }
        } else if !list.isEmpty {
            removed = [list.removeFirst()]
        } else {
            removed = []
        }
        if list.isEmpty {
            box.entriesByRequestID.removeValue(forKey: requestID)
        } else {
            box.entriesByRequestID[requestID] = list
        }
        for item in removed {
            box.cancelHandlersByExecution.removeValue(forKey: item.executionID)
            if let convID = item.conversationID,
               box.conversationLocks[convID]?.executionID == item.executionID {
                box.conversationLocks.removeValue(forKey: convID)
            }
        }
    }

    @discardableResult
    public nonisolated func abort(requestID: String) -> Bool {
        var handlersToRun: [@Sendable () -> Void] = []
        box.lock.lock()
        guard var list = box.entriesByRequestID[requestID], !list.isEmpty else {
            box.lock.unlock()
            return false
        }
        for idx in list.indices {
            list[idx].isAborted = true
            if let hs = box.cancelHandlersByExecution.removeValue(forKey: list[idx].executionID) {
                handlersToRun.append(contentsOf: hs)
            }
        }
        box.entriesByRequestID[requestID] = list
        box.lock.unlock()
        for h in handlersToRun {
            h()
        }
        return true
    }

    @discardableResult
    public nonisolated func abort(forConversationID conversationID: String) -> Bool {
        box.lock.lock()
        let reqID = box.conversationLocks[conversationID]?.requestID
        box.lock.unlock()
        guard let reqID else { return false }
        return abort(requestID: reqID)
    }

    @discardableResult
    public nonisolated func requestReasoningEnd(requestID: String) -> Bool {
        box.lock.lock()
        defer { box.lock.unlock() }
        guard var list = box.entriesByRequestID[requestID], !list.isEmpty else { return false }
        for idx in list.indices {
            list[idx].isReasoningEndRequested = true
        }
        box.entriesByRequestID[requestID] = list
        return true
    }

    @discardableResult
    public nonisolated func requestReasoningEnd(forConversationID conversationID: String) -> Bool {
        box.lock.lock()
        let reqID = box.conversationLocks[conversationID]?.requestID
        box.lock.unlock()
        guard let reqID else { return false }
        return requestReasoningEnd(requestID: reqID)
    }

    public nonisolated func state(for requestID: String) -> (isAborted: Bool, isReasoningEndRequested: Bool) {
        box.lock.lock()
        defer { box.lock.unlock() }
        guard let list = box.entriesByRequestID[requestID], !list.isEmpty else { return (false, false) }
        let aborted = list.contains(where: \.isAborted)
        let reasoningEnd = list.contains(where: \.isReasoningEndRequested)
        return (aborted, reasoningEnd)
    }

    public nonisolated func activeRequests() -> [ActiveRequest] {
        box.lock.lock()
        defer { box.lock.unlock() }
        return box.entriesByRequestID.values.flatMap { $0 }.sorted { $0.startedAt < $1.startedAt }
    }
}
