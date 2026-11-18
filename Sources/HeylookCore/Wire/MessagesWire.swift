import Foundation

/// Structured wire-boundary validation errors mapped to HTTP `400` or `422` by `HeylookRouter`.
public enum WireValidationError: LocalizedError, Sendable, Equatable {
    case badRequest(String)
    case unprocessable(String)

    public var statusCode: Int {
        switch self {
        case .badRequest: return 400
        case .unprocessable: return 422
        }
    }

    public var errorType: String {
        switch self {
        case .badRequest: return "invalid_request_error"
        case .unprocessable: return "validation_error"
        }
    }

    public var errorDescription: String? {
        switch self {
        case .badRequest(let detail), .unprocessable(let detail):
            return detail
        }
    }
}

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

    public static func fromFoundationObject(_ raw: Any) -> JSONValue? {
        switch raw {
        case is NSNull:
            return .null
        case let b as Bool:
            return .bool(b)
        case let num as NSNumber:
            if CFGetTypeID(num) == CFBooleanGetTypeID() {
                return .bool(num.boolValue)
            }
            let objCType = String(cString: num.objCType)
            if objCType == "f" || objCType == "d" {
                return .double(num.doubleValue)
            }
            return .int(num.intValue)
        case let s as String:
            return .string(s)
        case let arr as [Any]:
            return .array(arr.compactMap { fromFoundationObject($0) })
        case let dict as [String: Any]:
            var out: [String: JSONValue] = [:]
            for (k, v) in dict {
                if let val = fromFoundationObject(v) {
                    out[k] = val
                }
            }
            return .object(out)
        default:
            return nil
        }
    }

    var isTruthy: Bool {
        switch self {
        case .null: return false
        case .bool(let b): return b
        case .int(let i): return i != 0
        case .double(let d): return d != 0.0
        case .string(let s): return !s.isEmpty
        case .array(let a): return !a.isEmpty
        case .object(let o): return !o.isEmpty
        }
    }
}

struct WireAnyCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init(_ key: String) {
        self.stringValue = key
        self.intValue = nil
    }

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}

/// Anthropic's object form for `thinking`: `{"type": "enabled"|"disabled", "budget_tokens": Int >= 1}`.
public struct ThinkingConfig: Codable, Sendable, Equatable, Hashable {
    public enum ThinkingType: String, Codable, Sendable, Equatable, Hashable {
        case enabled
        case disabled
    }

    public var type: ThinkingType?
    public var budgetTokens: Int?

    enum CodingKeys: String, CodingKey {
        case type
        case budgetTokens = "budget_tokens"
    }

    public init(type: ThinkingType? = nil, budgetTokens: Int? = nil) {
        self.type = type
        self.budgetTokens = budgetTokens
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let rawType = try container.decodeIfPresent(String.self, forKey: .type) {
            guard let parsed = ThinkingType(rawValue: rawType) else {
                throw WireValidationError.unprocessable(
                    "Invalid thinking.type '\(rawType)': expected 'enabled' or 'disabled'"
                )
            }
            self.type = parsed
        } else {
            self.type = nil
        }

        if container.contains(.budgetTokens) {
            let val = try container.decodeIfPresent(JSONValue.self, forKey: .budgetTokens)
            switch val {
            case .none, .null:
                self.budgetTokens = nil
            case .int(let budget):
                guard budget >= 1 else {
                    throw WireValidationError.unprocessable(
                        "thinking.budget_tokens must be >= 1 (got \(budget))"
                    )
                }
                self.budgetTokens = budget
            default:
                throw WireValidationError.unprocessable("thinking.budget_tokens must be an integer >= 1")
            }
        } else {
            self.budgetTokens = nil
        }
    }
}

/// Polymorphic `thinking` wire parameter accepting either `Bool` or `{type?, budget_tokens?}`.
public enum ThinkingParameter: Codable, Sendable, Equatable, Hashable, ExpressibleByBooleanLiteral {
    case bool(Bool)
    case config(ThinkingConfig)

    public init(booleanLiteral value: Bool) {
        self = .bool(value)
    }

    public var enabledBool: Bool? {
        switch self {
        case .bool(let b):
            return b
        case .config(let cfg):
            guard let t = cfg.type else { return nil }
            return t == .enabled
        }
    }

    public var budgetTokens: Int? {
        switch self {
        case .bool:
            return nil
        case .config(let cfg):
            return cfg.budgetTokens
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let b = try? container.decode(Bool.self) {
            self = .bool(b)
            return
        }
        if let cfg = try? container.decode(ThinkingConfig.self) {
            self = .config(cfg)
            return
        }
        throw WireValidationError.unprocessable(
            "thinking must be a boolean or an object {\"type\": \"enabled\"|\"disabled\", \"budget_tokens\": Int >= 1}"
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .bool(let b):
            try container.encode(b)
        case .config(let cfg):
            try container.encode(cfg)
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

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guard let decodedType = try? container.decode(String.self, forKey: .type),
              ["text", "json_object", "json_schema"].contains(decodedType) else {
            throw WireValidationError.unprocessable(
                "response_format.type must be 'text', 'json_object', or 'json_schema'"
            )
        }
        self.type = decodedType
        self.jsonSchema = try container.decodeIfPresent(JSONSchemaSpec.self, forKey: .jsonSchema)
        self.schema = try container.decodeIfPresent(JSONValue.self, forKey: .schema)

        if decodedType == "json_schema" {
            let nested = self.jsonSchema?.schema
            let hasNestedObject: Bool
            if case .object = nested {
                hasNestedObject = true
            } else {
                hasNestedObject = false
            }
            if !hasNestedObject {
                throw WireValidationError.unprocessable(
                    "response_format type \"json_schema\" needs non-null json_schema.schema"
                )
            }
        }
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

public extension StopReason {
    /// Canonical Anthropic Messages wire value (`.cancelled` maps to `"max_tokens"`, matching `converters.py`).
    var anthropicWireValue: String {
        switch self {
        case .endTurn: return "end_turn"
        case .maxTokens, .cancelled: return "max_tokens"
        case .stopSequence: return "stop_sequence"
        }
    }

    /// Parses either Anthropic wire stop reasons or OpenAI finish reasons (`stop`, `length`, `stop_sequence`).
    static func fromWireOrFinishReason(_ raw: String?) -> StopReason {
        switch raw {
        case "length", "max_tokens":
            return .maxTokens
        case "stop_sequence":
            return .stopSequence
        case "cancelled":
            return .cancelled
        default:
            return .endTurn
        }
    }
}

/// Anthropic Messages-conformant `/v1/messages` request payload.
public struct MessagesRequest: Codable, Sendable, Equatable {
    public static let maxStopSequences = 16
    public static let maxStopSequenceChars = 256

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
        /// Base64-encoded payload bytes (or empty when referenced via `url` / `mediaID` in `SQLiteStore`)
        public let dataBase64: String
        /// Display filename for UI pills
        public let filename: String?
        /// Optional content-addressed 64-hex SHA-256 media blob ID in `SQLiteStore.media_blobs`
        public var mediaID: String?
        /// Optional URL (external `https://...` or local `/v1/conversations/{id}/media/{sha256}`)
        public var url: String?

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
            mediaID: String? = nil,
            url: String? = nil
        ) {
            self.id = id
            self.kind = kind
            self.mediaType = mediaType
            self.dataBase64 = dataBase64
            self.filename = filename
            self.mediaID = mediaID
            self.url = url
        }

        public init(
            id: String = UUID().uuidString,
            kind: MediaKind,
            mimeType: String,
            base64Data: String,
            filename: String? = nil,
            mediaID: String? = nil,
            url: String? = nil
        ) {
            self.id = id
            self.kind = kind.rawValue
            self.mediaType = mimeType
            self.dataBase64 = base64Data
            self.filename = filename
            self.mediaID = mediaID
            self.url = url
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
                case sourceType = "source_type"
                case mediaType = "media_type"
                case data
                case url
                case mediaID = "media_id"
                case filename
            }

            private struct SourcePayload: Codable {
                let type: String?
                let media_type: String?
                let data: String?
                let url: String?
                let media_id: String?
                let filename: String?
            }

            public init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                guard let type = try? c.decode(String.self, forKey: .type) else {
                    throw WireValidationError.unprocessable("Each content block must have a string 'type'")
                }
                switch type {
                case "text":
                    guard let txt = try? c.decode(String.self, forKey: .text) else {
                        throw WireValidationError.unprocessable("Text block requires a string 'text'")
                    }
                    self = .text(txt)

                case "thinking":
                    let explicitThinking = try c.decodeIfPresent(String.self, forKey: .thinking)
                    let fallbackText = try c.decodeIfPresent(String.self, forKey: .text)
                    let resolved = (explicitThinking?.isEmpty == false ? explicitThinking : nil)
                        ?? (fallbackText?.isEmpty == false ? fallbackText : nil)
                    guard let th = resolved else {
                        throw WireValidationError.unprocessable("ThinkingBlock requires non-empty 'thinking' or 'text'")
                    }
                    self = .thinking(th)

                case "image", "audio":
                    let flatSourceType = try c.decodeIfPresent(String.self, forKey: .sourceType)
                    let flatMediaType = try c.decodeIfPresent(String.self, forKey: .mediaType)
                    let flatData = try c.decodeIfPresent(String.self, forKey: .data)
                    let flatURL = try c.decodeIfPresent(String.self, forKey: .url)
                    let flatMediaID = try c.decodeIfPresent(String.self, forKey: .mediaID)
                    let flatFilename = try c.decodeIfPresent(String.self, forKey: .filename)

                    let nestedSource: SourcePayload?
                    if c.contains(.source) {
                        let rawSrc = try c.decodeIfPresent(JSONValue.self, forKey: .source)
                        if let rawSrc, rawSrc != .null {
                            guard case .object = rawSrc else {
                                throw WireValidationError.unprocessable("\(type) block 'source' must be an object")
                            }
                            nestedSource = try c.decode(SourcePayload.self, forKey: .source)
                        } else {
                            nestedSource = nil
                        }
                    } else {
                        nestedSource = nil
                    }

                    // Per-field null fallback (`content_blocks.py:_flatten_source`):
                    // if flat `source_type` is declared and disagrees with nested `source.type`, flat wins.
                    let useNested = nestedSource != nil
                        && (flatSourceType == nil || flatSourceType == nestedSource?.type)
                    let effectiveSourceType = flatSourceType ?? (useNested ? nestedSource?.type : nil)
                    let effectiveMediaType = flatMediaType ?? (useNested ? nestedSource?.media_type : nil)
                    let effectiveData = flatData ?? (useNested ? nestedSource?.data : nil)
                    let effectiveURL = flatURL ?? (useNested ? nestedSource?.url : nil)
                    let effectiveMediaID = flatMediaID ?? (useNested ? nestedSource?.media_id : nil)
                    let effectiveFilename = flatFilename ?? (useNested ? nestedSource?.filename : nil)

                    guard let sourceKind = effectiveSourceType, !sourceKind.isEmpty else {
                        throw WireValidationError.unprocessable(
                            "\(type) block requires 'source_type', or a 'source' object carrying 'type'"
                        )
                    }
                    switch sourceKind {
                    case "base64":
                        guard let b64 = effectiveData, !b64.isEmpty else {
                            throw WireValidationError.unprocessable(
                                "\(type) block declares source_type='base64' but carries no 'data'"
                            )
                        }
                        let mime = effectiveMediaType ?? (type == "audio" ? "audio/wav" : "image/jpeg")
                        self = .media(
                            MediaAttachment(
                                id: effectiveMediaID ?? UUID().uuidString,
                                kind: type,
                                mediaType: mime,
                                dataBase64: b64,
                                filename: effectiveFilename,
                                mediaID: effectiveMediaID,
                                url: effectiveURL
                            )
                        )
                    case "url":
                        guard let resolvedURL = effectiveURL, !resolvedURL.isEmpty else {
                            throw WireValidationError.unprocessable(
                                "\(type) block declares source_type='url' but carries no 'url'"
                            )
                        }
                        let mime = effectiveMediaType ?? (type == "audio" ? "audio/wav" : "image/jpeg")
                        self = .media(
                            MediaAttachment(
                                id: effectiveMediaID ?? UUID().uuidString,
                                kind: type,
                                mediaType: mime,
                                dataBase64: "",
                                filename: effectiveFilename,
                                mediaID: effectiveMediaID,
                                url: resolvedURL
                            )
                        )
                    case "media_ref":
                        // Legacy internal SQLite row format
                        guard let mID = effectiveMediaID, !mID.isEmpty else {
                            throw WireValidationError.unprocessable(
                                "\(type) block declares source_type='media_ref' without 'media_id'"
                            )
                        }
                        let mime = effectiveMediaType ?? (type == "audio" ? "audio/wav" : "image/jpeg")
                        self = .media(
                            MediaAttachment(
                                id: mID,
                                kind: type,
                                mediaType: mime,
                                dataBase64: "",
                                filename: effectiveFilename,
                                mediaID: mID,
                                url: effectiveURL
                            )
                        )
                    default:
                        throw WireValidationError.unprocessable(
                            "\(type) block source type must be 'base64' or 'url' (got '\(sourceKind)')"
                        )
                    }

                default:
                    throw WireValidationError.unprocessable(
                        "Unsupported content block type '\(type)': expected 'text', 'image', 'audio', or 'thinking'"
                    )
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
                    try c.encode(th, forKey: .text)
                case .media(let att):
                    try c.encode(att.kind, forKey: .type)
                    if att.dataBase64.isEmpty, (att.url != nil || att.mediaID != nil) {
                        let resolvedURL = att.url
                        let sourceType = resolvedURL != nil ? "url" : "media_ref"
                        try c.encode(
                            SourcePayload(
                                type: sourceType,
                                media_type: att.mediaType,
                                data: nil,
                                url: resolvedURL,
                                media_id: att.mediaID,
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
                                url: nil,
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
            if let parts {
                self.parts = parts
                if attachments.isEmpty {
                    self.attachments = parts.compactMap {
                        if case .media(let att) = $0 { return att }
                        return nil
                    }
                } else {
                    self.attachments = attachments
                }
            } else {
                self.attachments = attachments
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
            guard let decodedRole = try? container.decode(String.self, forKey: .role),
                  decodedRole == "user" || decodedRole == "assistant" else {
                throw WireValidationError.unprocessable("Message 'role' must be 'user' or 'assistant'")
            }
            self.role = decodedRole

            let explicitThinking = (try? container.decodeIfPresent(String.self, forKey: .thinking))
                ?? (try? container.decodeIfPresent(String.self, forKey: .reasoningContent))

            guard container.contains(.content) else {
                throw WireValidationError.unprocessable("Message 'content' is required")
            }

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
            } else {
                // Decode as content block array — propagate WireValidationError rather than swallowing!
                let rawContent = try container.decode(JSONValue.self, forKey: .content)
                guard case .array = rawContent else {
                    throw WireValidationError.unprocessable(
                        "Message 'content' must be a string or an array of content blocks"
                    )
                }
                let decodedParts = try container.decode([ContentPart].self, forKey: .content)
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
                // Matches Python `Message.text_content`: single-space join across text blocks.
                self.content = textParts.joined(separator: " ")
                let mergedThinking = explicitThinking ?? (thinkingParts.isEmpty ? nil : thinkingParts.joined(separator: "\n"))
                self.thinking = mergedThinking
                self.attachments = media
                if let explicitThinking, !explicitThinking.isEmpty, thinkingParts.isEmpty {
                    self.parts = [.thinking(explicitThinking)] + decodedParts
                } else {
                    self.parts = decodedParts
                }
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

    /// Optional model identifier as transmitted on the wire (`nil` if omitted or `null`).
    public var requestedModel: String?

    /// Non-optional accessor preserved for backward compatibility with existing Wave 1 callers.
    public var model: String {
        get { requestedModel ?? "" }
        set { requestedModel = newValue }
    }

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

    /// Polymorphic `thinking` wire field (`Bool` or `ThinkingConfig` object).
    public var thinkingConfig: ThinkingParameter?

    /// Boolean thinking toggle preserved for downstream providers and Wave 1 callers.
    public var thinking: Bool? {
        get { thinkingConfig?.enabledBool }
        set {
            if let newValue {
                if case .config(var cfg) = thinkingConfig {
                    cfg.type = newValue ? .enabled : .disabled
                    thinkingConfig = .config(cfg)
                } else {
                    thinkingConfig = .bool(newValue)
                }
            } else {
                if case .config(var cfg) = thinkingConfig, cfg.budgetTokens != nil {
                    cfg.type = nil
                    thinkingConfig = .config(cfg)
                } else {
                    thinkingConfig = nil
                }
            }
        }
    }

    public var thinkingDepth: String?
    private var explicitThinkingBudget: Int?
    public var thinkingBudget: Int? {
        get { explicitThinkingBudget ?? thinkingConfig?.budgetTokens }
        set { explicitThinkingBudget = newValue }
    }

    public var stopSequences: [String]?
    public var responseFormat: ResponseFormat?
    public var metadata: [String: String]?

    /// Validates and returns a non-empty model ID or throws `WireValidationError.badRequest` (`400`).
    public func requireModel() throws -> String {
        guard let raw = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            throw WireValidationError.badRequest("Request must name a non-empty 'model'")
        }
        return raw
    }

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
        case metadata
    }

    public init(
        model: String? = nil,
        messages: [MessageTurn],
        system: String? = nil,
        maxTokens: Int? = nil,
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
        thinkingConfig: ThinkingParameter? = nil,
        thinkingDepth: String? = nil,
        thinkingBudget: Int? = nil,
        stopSequences: [String]? = nil,
        responseFormat: ResponseFormat? = nil,
        metadata: [String: String]? = nil
    ) {
        self.requestedModel = model
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
        self.thinkingConfig = thinkingConfig ?? thinking.map { .bool($0) }
        self.thinkingDepth = thinkingDepth
        self.explicitThinkingBudget = thinkingBudget
        self.stopSequences = stopSequences
        self.responseFormat = responseFormat
        self.metadata = metadata
    }

    public init(
        model: String?,
        system: String?,
        messages: [MessageTurn],
        maxTokens: Int? = nil,
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
        responseFormat: ResponseFormat? = nil,
        metadata: [String: String]? = nil
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
            thinkingConfig: nil,
            thinkingDepth: thinkingDepth,
            thinkingBudget: thinkingBudget,
            stopSequences: stopSequences,
            responseFormat: responseFormat,
            metadata: metadata
        )
    }

    private static func validateRetiredAndMisspelledKeys(in raw: KeyedDecodingContainer<WireAnyCodingKey>) throws {
        let logprobsGone = ["logprobs", "top_logprobs"].filter { raw.contains(WireAnyCodingKey($0)) }
        if !logprobsGone.isEmpty {
            throw WireValidationError.unprocessable(
                "\(logprobsGone.joined(separator: ", ")) is no longer supported: logprobs were removed with the token explorer"
            )
        }
        if raw.contains(WireAnyCodingKey("show_special_tokens")) {
            let val = (try? raw.decodeIfPresent(JSONValue.self, forKey: WireAnyCodingKey("show_special_tokens"))) ?? .null
            if val.isTruthy {
                throw WireValidationError.unprocessable(
                    "show_special_tokens=true is no longer supported: declared specials are now always stripped, so sending false (or omitting it) is what this server does"
                )
            }
        }
        let renamed: [(bad: String, replacement: String)] = [
            ("enable_thinking", "thinking"),
            ("max_new_tokens", "max_tokens"),
            ("system_prompt", "system"),
        ]
        let misspelt = renamed.filter { raw.contains(WireAnyCodingKey($0.bad)) }
        if !misspelt.isEmpty {
            let detail = misspelt.map {
                "`\($0.bad)` is not a field on this API -- send `\($0.replacement)` instead"
            }.joined(separator: "; ")
            throw WireValidationError.unprocessable(detail)
        }
        if raw.contains(WireAnyCodingKey("chat_template_kwargs")) {
            throw WireValidationError.unprocessable(
                "`chat_template_kwargs` is not a field on this API -- send `thinking` and/or `reasoning_effort` as top-level fields instead"
            )
        }
        let unbuilt = ["tools", "tool_choice"].filter { raw.contains(WireAnyCodingKey($0)) }
        if !unbuilt.isEmpty {
            throw WireValidationError.unprocessable(
                "\(unbuilt.joined(separator: ", ")) is not supported on this server yet: tool use is not built. Send the request without it"
            )
        }
        if raw.contains(WireAnyCodingKey("preset")) || raw.contains(WireAnyCodingKey("sampler")) {
            throw WireValidationError.unprocessable(
                "named sampler bundles (`preset` / `sampler`) were removed in v2.0.30 -- send the sampler fields themselves (temperature, top_p, ...)"
            )
        }
        if raw.contains(WireAnyCodingKey("vision_tokens")) {
            throw WireValidationError.unprocessable(
                "vision_tokens was removed in v2.0.64: cap image cost by the pixels you send, and ask POST /v1/models/{id}/image-plan what an image of a given size costs this model"
            )
        }
    }

    private static func decodeOptionalDouble(
        _ container: KeyedDecodingContainer<CodingKeys>,
        forKey key: CodingKeys,
        range: ClosedRange<Double>
    ) throws -> Double? {
        guard container.contains(key) else { return nil }
        let raw = try container.decode(JSONValue.self, forKey: key)
        switch raw {
        case .null:
            return nil
        case .double(let d):
            guard d.isFinite && range.contains(d) else {
                throw WireValidationError.unprocessable(
                    "\(key.rawValue) must be within \(range.lowerBound)...\(range.upperBound) (got \(d))"
                )
            }
            return d
        case .int(let i):
            let d = Double(i)
            guard range.contains(d) else {
                throw WireValidationError.unprocessable(
                    "\(key.rawValue) must be within \(range.lowerBound)...\(range.upperBound) (got \(i))"
                )
            }
            return d
        default:
            throw WireValidationError.unprocessable("\(key.rawValue) must be a number")
        }
    }

    private static func decodeOptionalInt(
        _ container: KeyedDecodingContainer<CodingKeys>,
        forKey key: CodingKeys,
        minValue: Int? = nil
    ) throws -> Int? {
        guard container.contains(key) else { return nil }
        let raw = try container.decode(JSONValue.self, forKey: key)
        switch raw {
        case .null:
            return nil
        case .int(let i):
            if let minValue, i < minValue {
                throw WireValidationError.unprocessable(
                    "\(key.rawValue) must be >= \(minValue) (got \(i))"
                )
            }
            return i
        default:
            throw WireValidationError.unprocessable("\(key.rawValue) must be an integer")
        }
    }

    public init(from decoder: Decoder) throws {
        let raw = try decoder.container(keyedBy: WireAnyCodingKey.self)
        try Self.validateRetiredAndMisspelledKeys(in: raw)

        let container = try decoder.container(keyedBy: CodingKeys.self)

        if container.contains(.model) {
            let rawModel = try container.decode(JSONValue.self, forKey: .model)
            switch rawModel {
            case .null:
                self.requestedModel = nil
            case .string(let s):
                self.requestedModel = s
            default:
                throw WireValidationError.unprocessable("model must be a string or null")
            }
        } else {
            self.requestedModel = nil
        }

        guard container.contains(.messages) else {
            throw WireValidationError.unprocessable("messages field is required and cannot be empty")
        }
        let rawMsgs = try container.decode(JSONValue.self, forKey: .messages)
        guard case .array = rawMsgs else {
            throw WireValidationError.unprocessable("messages must be a non-empty array")
        }
        let decodedMessages = try container.decode([MessageTurn].self, forKey: .messages)
        guard !decodedMessages.isEmpty else {
            throw WireValidationError.unprocessable("Messages list cannot be empty")
        }
        self.messages = decodedMessages

        self.system = try container.decodeIfPresent(String.self, forKey: .system)
        self.maxTokens = try Self.decodeOptionalInt(container, forKey: .maxTokens, minValue: 1)
        self.temperature = try Self.decodeOptionalDouble(container, forKey: .temperature, range: 0.0...2.0)
        self.topP = try Self.decodeOptionalDouble(container, forKey: .topP, range: 0.0...1.0)
        self.topK = try Self.decodeOptionalInt(container, forKey: .topK, minValue: 0)
        self.minP = try Self.decodeOptionalDouble(container, forKey: .minP, range: 0.0...1.0)
        self.repetitionPenalty = try Self.decodeOptionalDouble(container, forKey: .repetitionPenalty, range: 0.1...2.0)
        self.repetitionContextSize = try Self.decodeOptionalInt(container, forKey: .repetitionContextSize, minValue: 1)
        self.presencePenalty = try Self.decodeOptionalDouble(container, forKey: .presencePenalty, range: 0.0...2.0)
        self.seed = try Self.decodeOptionalInt(container, forKey: .seed)

        if container.contains(.stream) {
            let rawStream = try container.decode(JSONValue.self, forKey: .stream)
            switch rawStream {
            case .null:
                self.stream = nil
            case .bool(let b):
                self.stream = b
            default:
                throw WireValidationError.unprocessable("stream must be a boolean")
            }
        } else {
            self.stream = nil
        }

        if container.contains(.thinking) {
            let rawThinking = try container.decode(JSONValue.self, forKey: .thinking)
            if rawThinking == .null {
                self.thinkingConfig = nil
            } else {
                self.thinkingConfig = try container.decode(ThinkingParameter.self, forKey: .thinking)
            }
        } else {
            self.thinkingConfig = nil
        }

        self.thinkingDepth = try container.decodeIfPresent(String.self, forKey: .thinkingDepth)
            ?? container.decodeIfPresent(String.self, forKey: .reasoningEffort)
        self.explicitThinkingBudget = try Self.decodeOptionalInt(container, forKey: .thinkingBudget, minValue: 1)

        if container.contains(.stopSequences) {
            let rawStops = try container.decode(JSONValue.self, forKey: .stopSequences)
            switch rawStops {
            case .null:
                self.stopSequences = nil
            case .array(let items):
                guard items.count >= 1 && items.count <= Self.maxStopSequences else {
                    throw WireValidationError.unprocessable(
                        "stop_sequences must contain 1...\(Self.maxStopSequences) items (got \(items.count))"
                    )
                }
                var parsedStops: [String] = []
                parsedStops.reserveCapacity(items.count)
                for item in items {
                    guard case .string(let s) = item else {
                        throw WireValidationError.unprocessable("Each stop_sequences item must be a string")
                    }
                    guard !s.isEmpty && s.count <= Self.maxStopSequenceChars else {
                        throw WireValidationError.unprocessable(
                            "Each stop_sequences item must be 1...\(Self.maxStopSequenceChars) characters"
                        )
                    }
                    parsedStops.append(s)
                }
                self.stopSequences = parsedStops
            default:
                throw WireValidationError.unprocessable("stop_sequences must be an array of strings")
            }
        } else {
            self.stopSequences = nil
        }

        if container.contains(.responseFormat) {
            let rawRF = try container.decode(JSONValue.self, forKey: .responseFormat)
            self.responseFormat = (rawRF == .null)
                ? nil
                : try container.decode(ResponseFormat.self, forKey: .responseFormat)
        } else {
            self.responseFormat = nil
        }

        if container.contains(.metadata) {
            let rawMeta = try container.decode(JSONValue.self, forKey: .metadata)
            switch rawMeta {
            case .null:
                self.metadata = nil
            case .object(let dict):
                var parsed: [String: String] = [:]
                for (k, v) in dict {
                    guard case .string(let str) = v else {
                        throw WireValidationError.unprocessable("metadata values must be strings (key '\(k)')")
                    }
                    parsed[k] = str
                }
                self.metadata = parsed
            default:
                throw WireValidationError.unprocessable("metadata must be an object mapping strings to strings")
            }
        } else {
            self.metadata = nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(requestedModel, forKey: .model)
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
        try container.encodeIfPresent(thinkingConfig, forKey: .thinking)
        try container.encodeIfPresent(thinkingDepth, forKey: .thinkingDepth)
        try container.encodeIfPresent(explicitThinkingBudget, forKey: .thinkingBudget)
        try container.encodeIfPresent(stopSequences, forKey: .stopSequences)
        try container.encodeIfPresent(responseFormat, forKey: .responseFormat)
        try container.encodeIfPresent(metadata, forKey: .metadata)
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

        enum CodingKeys: String, CodingKey {
            case type
            case text
            case thinking
        }

        public init(type: String, text: String? = nil, thinking: String? = nil) {
            self.type = type
            self.text = text
            self.thinking = thinking
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let decodedType = try c.decode(String.self, forKey: .type)
            self.type = decodedType
            let rawText = try c.decodeIfPresent(String.self, forKey: .text)
            let rawThinking = try c.decodeIfPresent(String.self, forKey: .thinking)
            if decodedType == "thinking" {
                let resolved = rawThinking ?? rawText ?? ""
                self.thinking = resolved
                self.text = rawText ?? resolved
            } else {
                self.text = rawText
                self.thinking = rawThinking
            }
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

    /// Unified performance telemetry block (`perf_collector.build_performance`), carrying only real measurements.
    /// Zero rates, memory, or queue wait mean unmeasured and serialize as omitted/`nil`.
    public struct PerformanceExtension: Codable, Sendable, Equatable {
        public let generationTPS: Double?
        public let promptTPS: Double?
        public let tokensPerSecond: Double?
        public let peakMemoryGB: Double?
        public let queueWaitMs: Double?
        public let ttftMs: Double?
        public let thinkingDurationMs: Double?
        public let contentDurationMs: Double?
        public let generationDurationMs: Double?
        public let requestDurationMs: Double?
        public let computePlacement: ComputePlacement?
        public let cache: CacheReport?
        public let speculative: SpeculativeReport?

        enum CodingKeys: String, CodingKey {
            case generationTPS = "generation_tps"
            case promptTPS = "prompt_tps"
            case tokensPerSecond = "tokens_per_second"
            case peakMemoryGB = "peak_memory_gb"
            case queueWaitMs = "queue_wait_ms"
            case ttftMs = "ttft_ms"
            case thinkingDurationMs = "thinking_duration_ms"
            case contentDurationMs = "content_duration_ms"
            case generationDurationMs = "generation_duration_ms"
            case requestDurationMs = "request_duration_ms"
            case computePlacement = "compute_placement"
            case cache
            case speculative
        }

        private static func measuredPositive(_ value: Double?) -> Double? {
            guard let value, value.isFinite, value > 0 else { return nil }
            return value
        }

        public init(
            generationTPS: Double? = nil,
            promptTPS: Double? = nil,
            tokensPerSecond: Double? = nil,
            peakMemoryGB: Double? = nil,
            queueWaitMs: Double? = nil,
            ttftMs: Double? = nil,
            thinkingDurationMs: Double? = nil,
            contentDurationMs: Double? = nil,
            generationDurationMs: Double? = nil,
            requestDurationMs: Double? = nil,
            computePlacement: ComputePlacement? = nil,
            cache: CacheReport? = nil,
            speculative: SpeculativeReport? = nil
        ) {
            let genTPS = Self.measuredPositive(generationTPS)
            let pTPS = Self.measuredPositive(promptTPS)
            let tps = Self.measuredPositive(tokensPerSecond) ?? genTPS
            let peakMem = Self.measuredPositive(peakMemoryGB)
            let qWait = Self.measuredPositive(queueWaitMs)

            self.generationTPS = genTPS
            self.promptTPS = pTPS
            self.tokensPerSecond = tps
            self.peakMemoryGB = peakMem
            self.queueWaitMs = qWait
            self.ttftMs = ttftMs
            self.thinkingDurationMs = thinkingDurationMs
            self.contentDurationMs = contentDurationMs
            if let generationDurationMs, let qWait {
                self.generationDurationMs = max(0.0, generationDurationMs - qWait)
            } else {
                self.generationDurationMs = generationDurationMs
            }
            self.requestDurationMs = requestDurationMs
            self.computePlacement = computePlacement
            self.cache = cache
            self.speculative = speculative
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let genTPS = Self.measuredPositive(
                (try? c.decodeIfPresent(Double.self, forKey: .generationTPS))
                    ?? (try? c.decodeIfPresent(Double.self, forKey: .tokensPerSecond))
            )
            self.generationTPS = genTPS
            self.promptTPS = Self.measuredPositive(try? c.decodeIfPresent(Double.self, forKey: .promptTPS))
            self.tokensPerSecond = Self.measuredPositive(try? c.decodeIfPresent(Double.self, forKey: .tokensPerSecond)) ?? genTPS
            self.peakMemoryGB = Self.measuredPositive(try? c.decodeIfPresent(Double.self, forKey: .peakMemoryGB))
            self.queueWaitMs = Self.measuredPositive(try? c.decodeIfPresent(Double.self, forKey: .queueWaitMs))
            self.ttftMs = try? c.decodeIfPresent(Double.self, forKey: .ttftMs)
            self.thinkingDurationMs = try? c.decodeIfPresent(Double.self, forKey: .thinkingDurationMs)
            self.contentDurationMs = try? c.decodeIfPresent(Double.self, forKey: .contentDurationMs)
            self.generationDurationMs = try? c.decodeIfPresent(Double.self, forKey: .generationDurationMs)
            self.requestDurationMs = try? c.decodeIfPresent(Double.self, forKey: .requestDurationMs)
            self.computePlacement = try? c.decodeIfPresent(ComputePlacement.self, forKey: .computePlacement)
            self.cache = try? c.decodeIfPresent(CacheReport.self, forKey: .cache)
            self.speculative = try? c.decodeIfPresent(SpeculativeReport.self, forKey: .speculative)
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encodeIfPresent(promptTPS, forKey: .promptTPS)
            try c.encodeIfPresent(generationTPS, forKey: .generationTPS)
            try c.encodeIfPresent(peakMemoryGB, forKey: .peakMemoryGB)
            try c.encodeIfPresent(queueWaitMs, forKey: .queueWaitMs)
            if let thinkingDurationMs {
                try c.encode(Int(thinkingDurationMs.rounded()), forKey: .thinkingDurationMs)
            }
            if let contentDurationMs {
                try c.encode(Int(contentDurationMs.rounded()), forKey: .contentDurationMs)
            }
            if let generationDurationMs {
                try c.encode(Int(generationDurationMs.rounded()), forKey: .generationDurationMs)
            }
            if let requestDurationMs {
                try c.encode(Int(requestDurationMs.rounded()), forKey: .requestDurationMs)
            }
            try c.encodeIfPresent(ttftMs, forKey: .ttftMs)
            try c.encodeIfPresent(computePlacement, forKey: .computePlacement)
            try c.encodeIfPresent(cache, forKey: .cache)
            try c.encodeIfPresent(speculative, forKey: .speculative)
        }
    }

    public let id: String
    public let type: String
    public let role: String
    public let model: String
    public let content: [ContentBlock]
    public let stopReason: StopReason
    public let stopSequence: String?
    public let usage: Usage
    public let performance: PerformanceExtension?
    public let metadata: [String: String]?

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case role
        case model
        case content
        case stopReason = "stop_reason"
        case stopSequence = "stop_sequence"
        case usage
        case performance
        case metadata
    }

    public init(
        id: String = "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
        model: String,
        content: [ContentBlock],
        stopReason: StopReason = .endTurn,
        stopSequence: String? = nil,
        usage: Usage,
        performance: PerformanceExtension? = nil,
        metadata: [String: String]? = nil
    ) {
        self.id = id
        self.type = "message"
        self.role = "assistant"
        self.model = model
        self.content = content
        self.stopReason = stopReason
        self.stopSequence = stopSequence
        self.usage = usage
        self.performance = performance
        self.metadata = metadata
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.type = (try? c.decodeIfPresent(String.self, forKey: .type)) ?? "message"
        self.role = (try? c.decodeIfPresent(String.self, forKey: .role)) ?? "assistant"
        self.model = try c.decode(String.self, forKey: .model)
        self.content = try c.decode([ContentBlock].self, forKey: .content)
        let rawStop = try? c.decodeIfPresent(String.self, forKey: .stopReason)
        self.stopReason = StopReason.fromWireOrFinishReason(rawStop)
        self.stopSequence = try c.decodeIfPresent(String.self, forKey: .stopSequence)
        self.usage = try c.decode(Usage.self, forKey: .usage)
        self.performance = try c.decodeIfPresent(PerformanceExtension.self, forKey: .performance)
        self.metadata = try c.decodeIfPresent([String: String].self, forKey: .metadata)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(type, forKey: .type)
        try c.encode(role, forKey: .role)
        try c.encode(model, forKey: .model)
        try c.encode(content, forKey: .content)
        try c.encode(stopReason.anthropicWireValue, forKey: .stopReason)
        try c.encode(stopSequence, forKey: .stopSequence)
        try c.encode(usage, forKey: .usage)
        try c.encodeIfPresent(performance, forKey: .performance)
        try c.encodeIfPresent(metadata, forKey: .metadata)
    }
}

// MARK: - Canonical Anthropic & Heylook SSE Events

public struct MessageStartSSEEvent: Codable, Sendable, Equatable {
    public struct MessageShell: Codable, Sendable, Equatable {
        public let id: String
        public let type: String
        public let role: String
        public let model: String
        public let content: [MessagesResponse.ContentBlock]
        public let usage: MessagesResponse.Usage

        public init(id: String, model: String, inputTokens: Int = 0) {
            self.id = id
            self.type = "message"
            self.role = "assistant"
            self.model = model
            self.content = []
            self.usage = MessagesResponse.Usage(inputTokens: inputTokens, outputTokens: 0)
        }
    }

    public let type: String
    public let message: MessageShell

    public init(id: String, model: String, inputTokens: Int = 0) {
        self.type = "message_start"
        self.message = MessageShell(id: id, model: model, inputTokens: inputTokens)
    }
}

public struct ContentBlockStartSSEEvent: Codable, Sendable, Equatable {
    public struct ContentBlockInfo: Codable, Sendable, Equatable {
        public let type: String
        public let text: String?
        public let thinking: String?

        public init(type: String, text: String? = nil, thinking: String? = nil) {
            self.type = type
            self.text = text
            self.thinking = thinking
        }

        public static func thinking() -> ContentBlockInfo {
            ContentBlockInfo(type: "thinking", text: "", thinking: "")
        }

        public static func text() -> ContentBlockInfo {
            ContentBlockInfo(type: "text", text: "", thinking: nil)
        }
    }

    public let type: String
    public let index: Int
    public let contentBlock: ContentBlockInfo

    enum CodingKeys: String, CodingKey {
        case type
        case index
        case contentBlock = "content_block"
    }

    public init(index: Int, contentBlock: ContentBlockInfo) {
        self.type = "content_block_start"
        self.index = index
        self.contentBlock = contentBlock
    }
}

public struct ContentBlockDeltaSSEEvent: Codable, Sendable, Equatable {
    public struct DeltaPayload: Codable, Sendable, Equatable {
        public let type: String
        public let text: String?
        public let thinking: String?

        public init(type: String, text: String? = nil, thinking: String? = nil) {
            self.type = type
            self.text = text
            self.thinking = thinking
        }

        public static func textDelta(_ value: String) -> DeltaPayload {
            DeltaPayload(type: "text_delta", text: value, thinking: nil)
        }

        /// Emits both `thinking` (Anthropic) and `text` (v3 compatibility) on `thinking_delta`.
        public static func thinkingDelta(_ value: String) -> DeltaPayload {
            DeltaPayload(type: "thinking_delta", text: value, thinking: value)
        }
    }

    public let type: String
    public let index: Int
    public let delta: DeltaPayload

    public init(index: Int, delta: DeltaPayload) {
        self.type = "content_block_delta"
        self.index = index
        self.delta = delta
    }
}

public struct ContentBlockStopSSEEvent: Codable, Sendable, Equatable {
    public let type: String
    public let index: Int

    public init(index: Int) {
        self.type = "content_block_stop"
        self.index = index
    }
}

/// Inner `delta` payload of `event: message_delta` carrying Anthropic wire `stop_reason` and optional `stop_sequence`.
public struct MessageDelta: Codable, Sendable, Equatable {
    public let stopReason: StopReason
    public let stopSequence: String?

    enum CodingKeys: String, CodingKey {
        case stopReason = "stop_reason"
        case stopSequence = "stop_sequence"
    }

    public init(stopReason: StopReason, stopSequence: String? = nil) {
        self.stopReason = stopReason
        self.stopSequence = stopSequence
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let rawStop = try? c.decodeIfPresent(String.self, forKey: .stopReason)
        self.stopReason = StopReason.fromWireOrFinishReason(rawStop)
        self.stopSequence = try c.decodeIfPresent(String.self, forKey: .stopSequence)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(stopReason.anthropicWireValue, forKey: .stopReason)
        try c.encodeIfPresent(stopSequence, forKey: .stopSequence)
    }
}

public struct MessageDeltaSSEEvent: Codable, Sendable, Equatable {
    public let type: String
    public let delta: MessageDelta
    public let usage: MessagesResponse.Usage?

    public init(delta: MessageDelta, usage: MessagesResponse.Usage? = nil) {
        self.type = "message_delta"
        self.delta = delta
        self.usage = usage
    }
}

/// Terminal `event: message_stop` carrying the unified `performance` telemetry object.
public struct MessageStopSSEEvent: Codable, Sendable, Equatable {
    public let type: String
    public let performance: MessagesResponse.PerformanceExtension?

    public init(performance: MessagesResponse.PerformanceExtension? = nil) {
        self.type = "message_stop"
        self.performance = performance
    }
}

/// Prefill progress frame emitted during long prompt evaluations (`event: heylook_progress`).
public struct HeylookProgressSSEEvent: Codable, Sendable, Equatable {
    public struct PrefillProgress: Codable, Sendable, Equatable {
        public let processed: Int
        public let total: Int

        public init(processed: Int, total: Int) {
            self.processed = processed
            self.total = total
        }
    }

    public let type: String
    public let prefill: PrefillProgress

    public init(processed: Int, total: Int) {
        self.type = "heylook_progress"
        self.prefill = PrefillProgress(processed: processed, total: total)
    }

    public func sseFrame() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let payload = (try? String(data: encoder.encode(self), encoding: .utf8)) ?? "{}"
        return "event: heylook_progress\ndata: \(payload)\n\n"
    }
}

public struct PingSSEEvent: Codable, Sendable, Equatable {
    public let type: String

    public init() {
        self.type = "ping"
    }
}

public struct StreamErrorSSEEvent: Codable, Sendable, Equatable {
    public struct ErrorBody: Codable, Sendable, Equatable {
        public let type: String
        public let message: String

        public init(type: String, message: String) {
            self.type = type
            self.message = message
        }
    }

    public let type: String
    public let error: ErrorBody

    public init(errorType: String = "api_error", message: String) {
        self.type = "error"
        self.error = ErrorBody(type: errorType, message: message)
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
    public var requestedModel: String?
    public var model: String {
        get { requestedModel ?? "" }
        set { requestedModel = newValue }
    }
    public var mode: ConversationGenerateMode
    public var messageID: String?
    public let content: String
    public var userParts: [MessagesRequest.MessageTurn.ContentPart]
    public var system: String?
    public var thinkingConfig: ThinkingParameter?
    public var thinking: Bool? {
        get { thinkingConfig?.enabledBool }
        set {
            if let newValue {
                thinkingConfig = .bool(newValue)
            } else {
                thinkingConfig = nil
            }
        }
    }
    public var thinkingDepth: String?
    public var thinkingBudget: Int?
    public var presetName: String?
    public var sampling: SamplerSettings?
    public var stopSequences: [String]?
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
        case stopSequences = "stop_sequences"
        case responseFormat = "response_format"
    }

    public init(
        model: String? = nil,
        content: String = "",
        mode: ConversationGenerateMode = .append,
        messageID: String? = nil,
        userParts: [MessagesRequest.MessageTurn.ContentPart]? = nil,
        system: String? = nil,
        thinking: Bool? = nil,
        thinkingConfig: ThinkingParameter? = nil,
        thinkingDepth: String? = nil,
        thinkingBudget: Int? = nil,
        presetName: String? = nil,
        sampling: SamplerSettings? = nil,
        stopSequences: [String]? = nil,
        responseFormat: ResponseFormat? = nil
    ) {
        self.requestedModel = model
        self.mode = mode
        self.messageID = messageID
        self.content = content
        self.userParts = userParts ?? (content.isEmpty ? [] : [.text(content)])
        self.system = system
        self.thinkingConfig = thinkingConfig ?? thinking.map { .bool($0) }
        self.thinkingDepth = thinkingDepth
        self.thinkingBudget = thinkingBudget ?? self.thinkingConfig?.budgetTokens
        self.presetName = presetName
        self.sampling = sampling
        self.stopSequences = stopSequences
        self.responseFormat = responseFormat
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        var overrideModel: String?
        if c.contains(.overrides),
           let ovObj = try? c.decodeIfPresent([String: JSONValue].self, forKey: .overrides),
           case .string(let ovModel) = ovObj["model"] {
            overrideModel = ovModel
        }
        self.requestedModel = (try? c.decodeIfPresent(String.self, forKey: .model)) ?? overrideModel
        self.mode = (try? c.decodeIfPresent(ConversationGenerateMode.self, forKey: .mode)) ?? .append
        self.messageID = try? c.decodeIfPresent(String.self, forKey: .messageID)
        self.system = try? c.decodeIfPresent(String.self, forKey: .system)
        self.thinkingConfig = try? c.decodeIfPresent(ThinkingParameter.self, forKey: .thinking)
        self.thinkingDepth = (try? c.decodeIfPresent(String.self, forKey: .thinkingDepth))
            ?? (try? c.decodeIfPresent(String.self, forKey: .reasoningEffort))
        self.thinkingBudget = (try? c.decodeIfPresent(Int.self, forKey: .thinkingBudget))
            ?? self.thinkingConfig?.budgetTokens
        self.presetName = try? c.decodeIfPresent(String.self, forKey: .presetName)
        self.sampling = (try? c.decodeIfPresent(SamplerSettings.self, forKey: .overrides))
            ?? (try? c.decodeIfPresent(SamplerSettings.self, forKey: .sampling))
        self.stopSequences = try? c.decodeIfPresent([String].self, forKey: .stopSequences)
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
        try c.encodeIfPresent(requestedModel, forKey: .model)
        try c.encode(mode, forKey: .mode)
        try c.encodeIfPresent(messageID, forKey: .messageID)
        if userParts.contains(where: { if case .media = $0 { return true }; return false }) {
            try c.encode(userParts, forKey: .userContent)
        } else {
            try c.encode(content, forKey: .content)
        }
        try c.encodeIfPresent(system, forKey: .system)
        try c.encodeIfPresent(thinkingConfig, forKey: .thinking)
        try c.encodeIfPresent(thinkingDepth, forKey: .thinkingDepth)
        try c.encodeIfPresent(thinkingBudget, forKey: .thinkingBudget)
        try c.encodeIfPresent(presetName, forKey: .presetName)
        try c.encodeIfPresent(sampling, forKey: .sampling)
        try c.encodeIfPresent(stopSequences, forKey: .stopSequences)
        try c.encodeIfPresent(responseFormat, forKey: .responseFormat)
    }
}

/// Counts of media attachments omitted at wire serialization due to target model capabilities.
public struct DroppedMediaCounts: Codable, Sendable, Equatable, Hashable {
    public let images: Int
    public let audio: Int

    public init(images: Int = 0, audio: Int = 0) {
        self.images = images
        self.audio = audio
    }

    public var total: Int { images + audio }
}

/// Final `event: heylook_saved` payload emitted on `/v1/conversations/{id}/generate` once the
/// assistant turn is durably written to `SQLiteStore`.
public struct HeylookSavedEvent: Codable, Sendable, Equatable {
    public let type: String
    public let conversationID: String
    public let mode: ConversationGenerateMode
    public let endReason: String
    public let messages: [WireMessageDTO]
    public let droppedMedia: DroppedMediaCounts
    public let timing: MessagesResponse.PerformanceExtension
    public let error: String?

    enum CodingKeys: String, CodingKey {
        case type
        case conversationID = "conversation_id"
        case mode
        case endReason = "end_reason"
        case messages
        case droppedMedia = "dropped_media"
        case timing
        case error
    }

    public init(
        conversationID: String,
        mode: ConversationGenerateMode,
        endReason: String,
        messages: [WireMessageDTO],
        droppedMedia: DroppedMediaCounts = DroppedMediaCounts(),
        timing: MessagesResponse.PerformanceExtension = MessagesResponse.PerformanceExtension(),
        error: String? = nil
    ) {
        self.type = "heylook_saved"
        self.conversationID = conversationID
        self.mode = mode
        self.endReason = endReason
        self.messages = messages
        self.droppedMedia = droppedMedia
        self.timing = timing
        self.error = error
    }

    /// Wave-1 compatibility initializer for existing callers passing individual IDs & `StopReason`.
    public init(
        conversationID: String,
        userMessageID: String?,
        assistantMessageID: String,
        position: Int,
        stopReason: StopReason,
        mode: ConversationGenerateMode = .append,
        droppedMedia: Int = 0
    ) {
        self.type = "heylook_saved"
        self.conversationID = conversationID
        self.mode = mode
        self.endReason = (stopReason == .cancelled) ? "aborted" : "complete"
        self.messages = []
        self.droppedMedia = DroppedMediaCounts(images: droppedMedia, audio: 0)
        self.timing = MessagesResponse.PerformanceExtension()
        self.error = nil
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
