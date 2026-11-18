import Foundation

/// UTC ISO-8601 timestamp helpers bridging `SQLiteStore`'s Unix millisecond epochs to Python wire strings.
public enum WireTimestamp {
    public static func iso8601(fromUnixMs ms: Int64) -> String {
        let fmt = ISO8601DateFormatter()
        fmt.timeZone = TimeZone(secondsFromGMT: 0)
        fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fmt.string(from: Date(timeIntervalSince1970: TimeInterval(ms) / 1000.0))
    }

    public static func unixMs(fromISO8601 raw: String) -> Int64? {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = fmt.date(from: raw) {
            return Int64(d.timeIntervalSince1970 * 1000.0)
        }
        fmt.formatOptions = [.withInternetDateTime]
        if let d = fmt.date(from: raw) {
            return Int64(d.timeIntervalSince1970 * 1000.0)
        }
        return nil
    }
}

/// Strictly validated `params` bag (`schema/sampler_params.py`) used by conversation, notebook,
/// and preset wire DTOs. Keys derive directly from `SamplerCascade.SamplerFieldKey` (plus internal
/// wire aliases `enable_thinking`, `reasoning_effort`, `thinking_budget_tokens`). Unknown keys,
/// invalid types, and out-of-range numbers fail loudly with `WireValidationError.unprocessable` (`422`).
public struct StrictSamplerParams: Codable, Sendable, Equatable, Hashable {
    /// Canonical allowed keys derived from `SamplerCascade.SamplerFieldKey.allCases` and its `acceptedAliases`.
    public static let allowedKeys: Set<String> = Set(
        SamplerCascade.SamplerFieldKey.allCases.flatMap(\.acceptedAliases)
    )

    public var settings: SamplerSettings

    public init(settings: SamplerSettings = SamplerSettings()) {
        self.settings = settings
    }

    public init(from decoder: Decoder) throws {
        let rawValue = try JSONValue(from: decoder)
        guard case .object(let dict) = rawValue else {
            throw WireValidationError.unprocessable("'params' must be a JSON object")
        }

        var unknown: [String] = []
        for key in dict.keys.sorted() where !Self.allowedKeys.contains(key) {
            unknown.append("unknown key '\(key)'")
        }
        if !unknown.isEmpty {
            throw WireValidationError.unprocessable(unknown.joined(separator: "; "))
        }

        var parsed = SamplerSettings()
        var problems: [String] = []

        func readDouble(_ key: String, range: ClosedRange<Double>) -> Double? {
            guard let val = dict[key], val != .null else { return nil }
            switch val {
            case .double(let d) where d.isFinite && range.contains(d):
                return d
            case .int(let i) where range.contains(Double(i)):
                return Double(i)
            case .double(let d):
                problems.append("\(key): must be within \(range.lowerBound)...\(range.upperBound) (got \(d))")
                return nil
            case .int(let i):
                problems.append("\(key): must be within \(range.lowerBound)...\(range.upperBound) (got \(i))")
                return nil
            default:
                problems.append("\(key): must be a number")
                return nil
            }
        }

        func readInt(_ key: String, minValue: Int? = nil) -> Int? {
            guard let val = dict[key], val != .null else { return nil }
            switch val {
            case .int(let i):
                if let minValue, i < minValue {
                    problems.append("\(key): must be >= \(minValue) (got \(i))")
                    return nil
                }
                return i
            default:
                problems.append("\(key): must be an integer")
                return nil
            }
        }

        func readBool(_ key: String) -> Bool? {
            guard let val = dict[key], val != .null else { return nil }
            guard case .bool(let b) = val else {
                problems.append("\(key): must be a boolean")
                return nil
            }
            return b
        }

        func readString(_ key: String) -> String? {
            guard let val = dict[key], val != .null else { return nil }
            guard case .string(let s) = val else {
                problems.append("\(key): must be a string")
                return nil
            }
            return s
        }

        parsed.temperature = readDouble("temperature", range: 0.0...2.0)
        parsed.topP = readDouble("top_p", range: 0.0...1.0)
        parsed.topK = readInt("top_k", minValue: 0)
        parsed.minP = readDouble("min_p", range: 0.0...1.0)
        parsed.repetitionPenalty = readDouble("repetition_penalty", range: 0.1...2.0)
        parsed.repetitionContextSize = readInt("repetition_context_size", minValue: 1)
        parsed.presencePenalty = readDouble("presence_penalty", range: 0.0...2.0)
        parsed.maxTokens = readInt("max_tokens", minValue: 1)
        parsed.seed = readInt("seed")
        parsed.thinking = readBool("thinking") ?? readBool("enable_thinking")
        parsed.thinkingDepth = readString("thinking_depth") ?? readString("reasoning_effort") ?? readString("reasoning_strength")
        parsed.thinkingBudget = readInt("thinking_budget", minValue: 1) ?? readInt("thinking_budget_tokens", minValue: 1)

        if let rawStops = dict["stop_sequences"], rawStops != .null {
            guard case .array(let items) = rawStops else {
                problems.append("stop_sequences: must be an array of strings")
                throw WireValidationError.unprocessable(problems.joined(separator: "; "))
            }
            if items.count > MessagesRequest.maxStopSequences {
                problems.append("stop_sequences: at most \(MessagesRequest.maxStopSequences) items allowed")
            } else {
                var stops: [String] = []
                for item in items {
                    guard case .string(let s) = item else {
                        problems.append("stop_sequences: items must be strings")
                        break
                    }
                    guard !s.isEmpty && s.count <= MessagesRequest.maxStopSequenceChars else {
                        problems.append("stop_sequences: each string must be 1...\(MessagesRequest.maxStopSequenceChars) chars")
                        break
                    }
                    stops.append(s)
                }
                if stops.count == items.count {
                    parsed.stopSequences = stops
                }
            }
        }

        if !problems.isEmpty {
            throw WireValidationError.unprocessable(problems.joined(separator: "; "))
        }
        self.settings = parsed
    }

    private struct WireCodingKey: CodingKey {
        let stringValue: String
        let intValue: Int? = nil

        init(stringValue: String) {
            self.stringValue = stringValue
        }

        init?(intValue: Int) {
            return nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        // Encode via `SamplerCascade.SamplerFieldKey.canonicalWireKey` (`thinking` -> `"enable_thinking"`,
        // omitting nulls -> `{}` when empty) to match Python `SamplerParams` / `preset_api.py`.
        var container = encoder.container(keyedBy: WireCodingKey.self)
        for key in SamplerCascade.SamplerFieldKey.allCases {
            let codingKey = WireCodingKey(stringValue: key.canonicalWireKey)
            switch key {
            case .temperature: try container.encodeIfPresent(settings.temperature, forKey: codingKey)
            case .topP: try container.encodeIfPresent(settings.topP, forKey: codingKey)
            case .topK: try container.encodeIfPresent(settings.topK, forKey: codingKey)
            case .minP: try container.encodeIfPresent(settings.minP, forKey: codingKey)
            case .repetitionPenalty: try container.encodeIfPresent(settings.repetitionPenalty, forKey: codingKey)
            case .repetitionContextSize: try container.encodeIfPresent(settings.repetitionContextSize, forKey: codingKey)
            case .presencePenalty: try container.encodeIfPresent(settings.presencePenalty, forKey: codingKey)
            case .maxTokens: try container.encodeIfPresent(settings.maxTokens, forKey: codingKey)
            case .seed: try container.encodeIfPresent(settings.seed, forKey: codingKey)
            case .thinking: try container.encodeIfPresent(settings.thinking, forKey: codingKey)
            case .thinkingDepth: try container.encodeIfPresent(settings.thinkingDepth, forKey: codingKey)
            case .thinkingBudget: try container.encodeIfPresent(settings.thinkingBudget, forKey: codingKey)
            case .stopSequences: try container.encodeIfPresent(settings.stopSequences, forKey: codingKey)
            }
        }
    }
}

// MARK: - Conversation & Message Wire DTOs

/// Canonical snake_case message wire representation (`db._message_row_to_dict`).
public struct WireMessageDTO: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let role: String
    public let content: String
    public let contentBlocks: [MessagesRequest.MessageTurn.ContentPart]
    public let thinking: String?
    public let modelID: String?
    public let position: Int
    public let createdAt: String
    public let updatedAt: String
    public let stats: MessageRecordStats?

    enum CodingKeys: String, CodingKey {
        case id
        case role
        case content
        case contentBlocks = "content_blocks"
        case thinking
        case modelID = "model_id"
        case position
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case stats
    }

    public init(from record: MessageRecord) {
        self.id = record.id
        self.role = record.role
        self.content = record.content
        self.contentBlocks = record.contentBlocks ?? [.text(record.content)]
        self.thinking = record.thinking
        self.modelID = record.modelID
        self.position = record.position
        self.createdAt = WireTimestamp.iso8601(fromUnixMs: record.createdUnixMs)
        self.updatedAt = WireTimestamp.iso8601(fromUnixMs: record.updatedUnixMs ?? record.createdUnixMs)
        self.stats = record.stats
    }

    public init(
        id: String,
        role: String,
        content: String,
        contentBlocks: [MessagesRequest.MessageTurn.ContentPart],
        thinking: String? = nil,
        modelID: String? = nil,
        position: Int,
        createdAt: String,
        updatedAt: String,
        stats: MessageRecordStats? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.contentBlocks = contentBlocks
        self.thinking = thinking
        self.modelID = modelID
        self.position = position
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.stats = stats
    }
}

/// Conversation summary item returned by `GET /v1/conversations` (omits `system_prompt` and `params`).
public struct ConversationSummaryDTO: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let title: String
    public let modelID: String?
    public let appliedPresetID: String?
    public let createdAt: String
    public let updatedAt: String
    public let generating: Bool

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case modelID = "model_id"
        case appliedPresetID = "applied_preset_id"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case generating
    }

    public init(from record: ConversationRecord, generating: Bool = false) {
        self.id = record.id
        self.title = record.title
        self.modelID = record.defaultModelID
        self.appliedPresetID = record.appliedPresetID
        self.createdAt = WireTimestamp.iso8601(fromUnixMs: record.createdUnixMs)
        self.updatedAt = WireTimestamp.iso8601(fromUnixMs: record.updatedUnixMs)
        self.generating = generating
    }
}

/// Envelope returned by `GET /v1/conversations`: `{"conversations": [...], "total": N}`.
public struct ConversationListResponseDTO: Codable, Sendable, Equatable {
    public let conversations: [ConversationSummaryDTO]
    public let total: Int

    public init(conversations: [ConversationSummaryDTO]) {
        self.conversations = conversations
        self.total = conversations.count
    }
}

/// Flat conversation detail returned by `GET/POST/PUT/PATCH /v1/conversations/{id}` and `/clone`.
public struct ConversationDetailDTO: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let title: String
    public let modelID: String?
    public let systemPrompt: String?
    public let params: StrictSamplerParams
    public let appliedPresetID: String?
    public let createdAt: String
    public let updatedAt: String
    public let generating: Bool
    public let messages: [WireMessageDTO]

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case modelID = "model_id"
        case systemPrompt = "system_prompt"
        case params
        case appliedPresetID = "applied_preset_id"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case generating
        case messages
    }

    public init(
        from record: ConversationRecord,
        messages: [MessageRecord] = [],
        generating: Bool = false
    ) {
        self.id = record.id
        self.title = record.title
        self.modelID = record.defaultModelID
        self.systemPrompt = record.systemPrompt.isEmpty ? nil : record.systemPrompt
        self.params = StrictSamplerParams(settings: record.params ?? SamplerSettings())
        self.appliedPresetID = record.appliedPresetID
        self.createdAt = WireTimestamp.iso8601(fromUnixMs: record.createdUnixMs)
        self.updatedAt = WireTimestamp.iso8601(fromUnixMs: record.updatedUnixMs)
        self.generating = generating
        self.messages = messages.map(WireMessageDTO.init(from:))
    }
}

public struct ConversationCreateRequestDTO: Decodable, Sendable, Equatable {
    public let title: String
    public let modelID: String?
    public let systemPrompt: String?
    public let params: StrictSamplerParams
    public let appliedPresetID: String?

    enum CodingKeys: String, CodingKey {
        case title
        case modelID = "model_id"
        case systemPrompt = "system_prompt"
        case params
        case appliedPresetID = "applied_preset_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.title = (try c.decodeIfPresent(String.self, forKey: .title)) ?? "New Conversation"
        self.modelID = try c.decodeIfPresent(String.self, forKey: .modelID)
        self.systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        if c.contains(.params) {
            let raw = try c.decode(JSONValue.self, forKey: .params)
            self.params = raw == .null ? StrictSamplerParams() : try c.decode(StrictSamplerParams.self, forKey: .params)
        } else {
            self.params = StrictSamplerParams()
        }
        self.appliedPresetID = try c.decodeIfPresent(String.self, forKey: .appliedPresetID)
    }
}

public struct ConversationUpdateRequestDTO: Decodable, Sendable, Equatable {
    public let presentKeys: Set<String>
    public let title: String?
    public let modelID: String?
    public let systemPrompt: String?
    public let params: StrictSamplerParams?
    public let appliedPresetID: String?

    enum CodingKeys: String, CodingKey {
        case title
        case modelID = "model_id"
        case systemPrompt = "system_prompt"
        case params
        case appliedPresetID = "applied_preset_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.presentKeys = Set(c.allKeys.map(\.rawValue))
        self.title = try c.decodeIfPresent(String.self, forKey: .title)
        self.modelID = try c.decodeIfPresent(String.self, forKey: .modelID)
        self.systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        if c.contains(.params) {
            let raw = try c.decode(JSONValue.self, forKey: .params)
            self.params = raw == .null ? nil : try c.decode(StrictSamplerParams.self, forKey: .params)
        } else {
            self.params = nil
        }
        self.appliedPresetID = try c.decodeIfPresent(String.self, forKey: .appliedPresetID)
    }
}

public struct ConversationCloneRequestDTO: Decodable, Sendable, Equatable {
    public let title: String?
    public let upToPosition: Int?

    enum CodingKeys: String, CodingKey {
        case title
        case upToPosition = "up_to_position"
    }
}

/// Shared decoder helper for message create/update `content` fields; joins `.text` blocks with `" "`
/// to match `MessagesWire.swift` and Python `Message.text_content`.
private func decodeWireMessageContent<K: CodingKey>(
    from container: KeyedDecodingContainer<K>,
    key: K,
    nullDefaultsToEmpty: Bool
) throws -> (content: String?, blocks: [MessagesRequest.MessageTurn.ContentPart]?) {
    guard container.contains(key) else {
        return nullDefaultsToEmpty ? ("", [.text("")]) : (nil, nil)
    }
    let raw = try container.decode(JSONValue.self, forKey: key)
    switch raw {
    case .null:
        return nullDefaultsToEmpty ? ("", [.text("")]) : (nil, nil)
    case .string(let s):
        return (s, [.text(s)])
    case .array:
        let parts = try container.decode([MessagesRequest.MessageTurn.ContentPart].self, forKey: key)
        let joined = parts.compactMap {
            if case .text(let t) = $0 { return t }
            return nil
        }.joined(separator: " ")
        return (joined, parts)
    default:
        throw WireValidationError.unprocessable("Message 'content' must be a string or array of content blocks")
    }
}

public struct MessageCreateRequestDTO: Decodable, Sendable, Equatable {
    public let role: String
    public let content: String
    public let contentBlocks: [MessagesRequest.MessageTurn.ContentPart]
    public let thinking: String?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case thinking
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard let r = try c.decodeIfPresent(String.self, forKey: .role), !r.isEmpty else {
            throw WireValidationError.unprocessable("Message 'role' must be a non-empty string")
        }
        self.role = r
        self.thinking = try c.decodeIfPresent(String.self, forKey: .thinking)

        let decoded = try decodeWireMessageContent(from: c, key: .content, nullDefaultsToEmpty: true)
        self.content = decoded.content ?? ""
        self.contentBlocks = decoded.blocks ?? [.text("")]
    }
}

public struct MessageUpdateRequestDTO: Decodable, Sendable, Equatable {
    public let presentKeys: Set<String>
    public let content: String?
    public let contentBlocks: [MessagesRequest.MessageTurn.ContentPart]?
    public let thinking: String?
    public let truncateAfter: Bool

    enum CodingKeys: String, CodingKey {
        case content
        case thinking
        case truncateAfter = "truncate_after"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.presentKeys = Set(c.allKeys.map(\.rawValue))
        self.thinking = try c.decodeIfPresent(String.self, forKey: .thinking)
        self.truncateAfter = (try c.decodeIfPresent(Bool.self, forKey: .truncateAfter)) ?? false

        let decoded = try decodeWireMessageContent(from: c, key: .content, nullDefaultsToEmpty: false)
        self.content = decoded.content
        self.contentBlocks = decoded.blocks
    }
}

// MARK: - Notebook Wire DTOs

/// Notebook list item returned by `GET /v1/notebooks` (omits `content`, `params`, and `applied_preset_id`).
public struct NotebookSummaryDTO: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let title: String
    public let systemPrompt: String?
    public let modelID: String?
    public let createdAt: String
    public let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case systemPrompt = "system_prompt"
        case modelID = "model_id"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }

    public init(from record: NotebookRecord) {
        self.id = record.id
        self.title = record.title
        self.systemPrompt = record.systemPrompt.isEmpty ? nil : record.systemPrompt
        self.modelID = record.modelID
        self.createdAt = WireTimestamp.iso8601(fromUnixMs: record.createdUnixMs)
        self.updatedAt = WireTimestamp.iso8601(fromUnixMs: record.updatedUnixMs)
    }
}

/// Envelope returned by `GET /v1/notebooks`: `{"notebooks": [...], "total": N}`.
public struct NotebookListResponseDTO: Codable, Sendable, Equatable {
    public let notebooks: [NotebookSummaryDTO]
    public let total: Int

    public init(notebooks: [NotebookSummaryDTO]) {
        self.notebooks = notebooks
        self.total = notebooks.count
    }
}

/// Full notebook payload using `content` (matching `notebook_api.py` / `db.py`, not `body`).
public struct NotebookDetailDTO: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let title: String
    public let content: String
    public let systemPrompt: String?
    public let modelID: String?
    public let params: StrictSamplerParams
    public let appliedPresetID: String?
    public let createdAt: String
    public let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case content
        case systemPrompt = "system_prompt"
        case modelID = "model_id"
        case params
        case appliedPresetID = "applied_preset_id"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }

    public init(from record: NotebookRecord) {
        self.id = record.id
        self.title = record.title
        self.content = record.body
        self.systemPrompt = record.systemPrompt.isEmpty ? nil : record.systemPrompt
        self.modelID = record.modelID
        self.params = StrictSamplerParams(settings: record.params ?? SamplerSettings())
        self.appliedPresetID = record.appliedPresetID
        self.createdAt = WireTimestamp.iso8601(fromUnixMs: record.createdUnixMs)
        self.updatedAt = WireTimestamp.iso8601(fromUnixMs: record.updatedUnixMs)
    }
}

public struct NotebookCreateRequestDTO: Decodable, Sendable, Equatable {
    public let title: String
    public let content: String
    public let systemPrompt: String?
    public let modelID: String?
    public let params: StrictSamplerParams
    public let appliedPresetID: String?

    enum CodingKeys: String, CodingKey {
        case title
        case content
        case systemPrompt = "system_prompt"
        case modelID = "model_id"
        case params
        case appliedPresetID = "applied_preset_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.title = (try c.decodeIfPresent(String.self, forKey: .title)) ?? "Untitled"
        self.content = (try c.decodeIfPresent(String.self, forKey: .content)) ?? ""
        self.systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        self.modelID = try c.decodeIfPresent(String.self, forKey: .modelID)
        if c.contains(.params) {
            let raw = try c.decode(JSONValue.self, forKey: .params)
            self.params = raw == .null ? StrictSamplerParams() : try c.decode(StrictSamplerParams.self, forKey: .params)
        } else {
            self.params = StrictSamplerParams()
        }
        self.appliedPresetID = try c.decodeIfPresent(String.self, forKey: .appliedPresetID)
    }
}

public struct NotebookUpdateRequestDTO: Decodable, Sendable, Equatable {
    public let presentKeys: Set<String>
    public let title: String?
    public let content: String?
    public let systemPrompt: String?
    public let modelID: String?
    public let params: StrictSamplerParams?
    public let appliedPresetID: String?

    enum CodingKeys: String, CodingKey {
        case title
        case content
        case systemPrompt = "system_prompt"
        case modelID = "model_id"
        case params
        case appliedPresetID = "applied_preset_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.presentKeys = Set(c.allKeys.map(\.rawValue))
        self.title = try c.decodeIfPresent(String.self, forKey: .title)
        self.content = try c.decodeIfPresent(String.self, forKey: .content)
        self.systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        self.modelID = try c.decodeIfPresent(String.self, forKey: .modelID)
        if c.contains(.params) {
            let raw = try c.decode(JSONValue.self, forKey: .params)
            self.params = raw == .null ? nil : try c.decode(StrictSamplerParams.self, forKey: .params)
        } else {
            self.params = nil
        }
        self.appliedPresetID = try c.decodeIfPresent(String.self, forKey: .appliedPresetID)
    }
}

// MARK: - Preset Wire DTOs

/// Canonical snake_case preset wire payload using `params` (not `settings`).
public struct PresetDTO: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let name: String
    public let systemPrompt: String?
    public let params: StrictSamplerParams
    public let createdAt: String
    public let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case systemPrompt = "system_prompt"
        case params
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }

    public init(from record: PresetRecord) {
        self.id = record.id
        self.name = record.name
        self.systemPrompt = record.systemPrompt
        self.params = StrictSamplerParams(settings: record.settings)
        self.createdAt = WireTimestamp.iso8601(fromUnixMs: record.createdUnixMs)
        self.updatedAt = WireTimestamp.iso8601(fromUnixMs: record.updatedUnixMs)
    }
}

/// Envelope returned by `GET /v1/presets`: `{"presets": [...], "total": N}`.
public struct PresetListResponseDTO: Codable, Sendable, Equatable {
    public let presets: [PresetDTO]
    public let total: Int

    public init(presets: [PresetDTO]) {
        self.presets = presets
        self.total = presets.count
    }
}

public struct PresetCreateRequestDTO: Decodable, Sendable, Equatable {
    public let name: String
    public let systemPrompt: String?
    public let params: StrictSamplerParams

    enum CodingKeys: String, CodingKey {
        case name
        case systemPrompt = "system_prompt"
        case params
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard let rawName = try c.decodeIfPresent(String.self, forKey: .name) else {
            throw WireValidationError.unprocessable("Preset 'name' is required")
        }
        let trimmed = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw WireValidationError.unprocessable("Preset 'name' must be a non-empty string")
        }
        self.name = trimmed
        self.systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        if c.contains(.params) {
            let raw = try c.decode(JSONValue.self, forKey: .params)
            self.params = raw == .null ? StrictSamplerParams() : try c.decode(StrictSamplerParams.self, forKey: .params)
        } else {
            self.params = StrictSamplerParams()
        }
    }
}

public struct PresetUpdateRequestDTO: Decodable, Sendable, Equatable {
    public let presentKeys: Set<String>
    public let name: String?
    public let systemPrompt: String?
    public let params: StrictSamplerParams?

    enum CodingKeys: String, CodingKey {
        case name
        case systemPrompt = "system_prompt"
        case params
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let keys = Set(c.allKeys.map(\.rawValue))
        guard !keys.isEmpty else {
            throw WireValidationError.badRequest("No fields to update")
        }
        self.presentKeys = keys
        if c.contains(.name) {
            guard let rawName = try c.decodeIfPresent(String.self, forKey: .name) else {
                throw WireValidationError.unprocessable("Preset 'name' must be a non-empty string")
            }
            let trimmed = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                throw WireValidationError.unprocessable("Preset 'name' must be a non-empty string")
            }
            self.name = trimmed
        } else {
            self.name = nil
        }
        self.systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        if c.contains(.params) {
            let raw = try c.decode(JSONValue.self, forKey: .params)
            self.params = raw == .null ? nil : try c.decode(StrictSamplerParams.self, forKey: .params)
        } else {
            self.params = nil
        }
    }
}

// MARK: - Observability Logs Wire DTO

public struct ObservabilityLogsResponseDTO: Codable, Sendable, Equatable {
    public let stream: String
    public let level: String
    public let records: [[String: JSONValue]]

    public init(stream: ObservabilityStream, level: String, records: [[String: JSONValue]]) {
        self.stream = stream.rawValue
        self.level = level
        self.records = records
    }
}
