import CryptoKit
import Foundation
import SQLite3

private let SQLITE_TRANSIENT_CORE = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

public enum SQLiteStoreError: LocalizedError, Sendable, Equatable {
    case openFailed(String)
    case sqlFailed(String)
    case notFound(String)
    case conflict(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let msg): return "Cannot open Heylook SQLite store: \(msg)"
        case .sqlFailed(let msg): return "SQLite error: \(msg)"
        case .notFound(let msg): return "Record not found: \(msg)"
        case .conflict(let msg): return "Conflict: \(msg)"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        }
    }
}

/// Live store mutation notification broadcast to observers (e.g. SwiftUI `@Observable` ViewModels)
/// so external HTTP `/v1/*` mutations stay synchronized with in-process UI state.
public enum StoreMutationEvent: Sendable, Equatable {
    case conversationUpserted(String)
    case conversationDeleted(String)
    case messagesChanged(String)
    case notebookUpserted(String)
    case notebookDeleted(String)
    case presetsChanged
    case dataCleared
}

public struct ConversationRecord: Codable, Identifiable, Sendable, Equatable, Hashable {
    public let id: String
    public var title: String
    public var systemPrompt: String
    public var defaultModelID: String?
    public var params: SamplerSettings?
    public var appliedPresetID: String?
    public let createdUnixMs: Int64
    public var updatedUnixMs: Int64

    public init(
        id: String = UUID().uuidString,
        title: String = "New Conversation",
        systemPrompt: String = "",
        defaultModelID: String? = nil,
        params: SamplerSettings? = nil,
        appliedPresetID: String? = nil,
        createdUnixMs: Int64 = SQLiteStore.nowMillis(),
        updatedUnixMs: Int64 = SQLiteStore.nowMillis()
    ) {
        self.id = id
        self.title = title
        self.systemPrompt = systemPrompt
        self.defaultModelID = defaultModelID
        self.params = params
        self.appliedPresetID = appliedPresetID
        self.createdUnixMs = createdUnixMs
        self.updatedUnixMs = updatedUnixMs
    }
}

/// Per-assistant-message telemetry joined onto `MessageRecord` reads (`message_stats`).
public struct MessageRecordStats: Codable, Sendable, Equatable, Hashable {
    public var outputTokens: Int?
    public var generationTPS: Double?
    public var promptTPS: Double?
    public var ttftMs: Double?
    public var peakMemoryGB: Double?
    public var queueWaitMs: Double?
    public var thinkingDurationMs: Double?
    public var contentDurationMs: Double?
    public var generationDurationMs: Double?
    public var requestDurationMs: Double?
    public var cache: CacheReport?
    public var speculative: SpeculativeReport?

    enum CodingKeys: String, CodingKey {
        case outputTokens = "output_tokens"
        case generationTPS = "generation_tps"
        case promptTPS = "prompt_tps"
        case ttftMs = "ttft_ms"
        case peakMemoryGB = "peak_memory_gb"
        case queueWaitMs = "queue_wait_ms"
        case thinkingDurationMs = "thinking_duration_ms"
        case contentDurationMs = "content_duration_ms"
        case generationDurationMs = "generation_duration_ms"
        case requestDurationMs = "request_duration_ms"
        case cache
        case speculative
    }

    public init(
        outputTokens: Int? = nil,
        generationTPS: Double? = nil,
        promptTPS: Double? = nil,
        ttftMs: Double? = nil,
        peakMemoryGB: Double? = nil,
        queueWaitMs: Double? = nil,
        thinkingDurationMs: Double? = nil,
        contentDurationMs: Double? = nil,
        generationDurationMs: Double? = nil,
        requestDurationMs: Double? = nil,
        cache: CacheReport? = nil,
        speculative: SpeculativeReport? = nil
    ) {
        self.outputTokens = outputTokens
        self.generationTPS = generationTPS
        self.promptTPS = promptTPS
        self.ttftMs = ttftMs
        self.peakMemoryGB = peakMemoryGB
        self.queueWaitMs = queueWaitMs
        self.thinkingDurationMs = thinkingDurationMs
        self.contentDurationMs = contentDurationMs
        self.generationDurationMs = generationDurationMs
        self.requestDurationMs = requestDurationMs
        self.cache = cache
        self.speculative = speculative
    }
}

public struct MessageRecord: Codable, Identifiable, Sendable, Equatable, Hashable {
    public let id: String
    public let conversationID: String
    public var position: Int
    public var role: String
    public var content: String
    public var thinking: String?
    public var modelID: String?
    public var stopReason: String?
    public var tokenCount: Int?
    public var generationTPS: Double?
    public var ttftMs: Double?
    public var cacheReport: CacheReport?
    public var speculativeReport: SpeculativeReport?
    public var computePlacement: ComputePlacement?
    public var contentBlocks: [MessagesRequest.MessageTurn.ContentPart]?
    public let createdUnixMs: Int64
    public var updatedUnixMs: Int64?
    public var promptTokens: Int?
    public var promptTPS: Double?
    public var peakMemoryGB: Double?
    public var queueWaitMs: Double?
    public var stats: MessageRecordStats?

    public var stopReasonEnum: StopReason? {
        stopReason.flatMap(StopReason.init(rawValue:))
    }

    public init(
        id: String = UUID().uuidString,
        conversationID: String,
        position: Int,
        role: String,
        content: String,
        thinking: String? = nil,
        modelID: String? = nil,
        stopReason: String? = nil,
        tokenCount: Int? = nil,
        generationTPS: Double? = nil,
        ttftMs: Double? = nil,
        cacheReport: CacheReport? = nil,
        speculativeReport: SpeculativeReport? = nil,
        computePlacement: ComputePlacement? = nil,
        contentBlocks: [MessagesRequest.MessageTurn.ContentPart]? = nil,
        createdUnixMs: Int64 = SQLiteStore.nowMillis(),
        updatedUnixMs: Int64? = nil,
        promptTokens: Int? = nil,
        promptTPS: Double? = nil,
        peakMemoryGB: Double? = nil,
        queueWaitMs: Double? = nil,
        stats: MessageRecordStats? = nil
    ) {
        self.id = id
        self.conversationID = conversationID
        self.position = position
        self.role = role
        self.content = content
        self.thinking = thinking
        self.modelID = modelID
        self.stopReason = stopReason
        self.tokenCount = tokenCount
        self.generationTPS = generationTPS
        self.ttftMs = ttftMs
        self.cacheReport = cacheReport
        self.speculativeReport = speculativeReport
        self.computePlacement = computePlacement
        self.contentBlocks = contentBlocks
        self.createdUnixMs = createdUnixMs
        self.updatedUnixMs = updatedUnixMs
        self.promptTokens = promptTokens
        self.promptTPS = promptTPS
        self.peakMemoryGB = peakMemoryGB
        self.queueWaitMs = queueWaitMs
        self.stats = stats
    }

    public init(
        id: String = UUID().uuidString,
        conversationID: String,
        position: Int,
        role: String,
        content: String,
        thinking: String? = nil,
        modelID: String? = nil,
        stopReason: StopReason,
        cacheReport: CacheReport? = nil,
        speculativeReport: SpeculativeReport? = nil,
        generationTPS: Double? = nil,
        computePlacement: ComputePlacement? = nil,
        tokenCount: Int? = nil,
        ttftMs: Double? = nil,
        contentBlocks: [MessagesRequest.MessageTurn.ContentPart]? = nil,
        createdUnixMs: Int64 = SQLiteStore.nowMillis()
    ) {
        self.init(
            id: id,
            conversationID: conversationID,
            position: position,
            role: role,
            content: content,
            thinking: thinking,
            modelID: modelID,
            stopReason: stopReason.rawValue,
            tokenCount: tokenCount,
            generationTPS: generationTPS,
            ttftMs: ttftMs,
            cacheReport: cacheReport,
            speculativeReport: speculativeReport,
            computePlacement: computePlacement,
            contentBlocks: contentBlocks,
            createdUnixMs: createdUnixMs
        )
    }
}

public struct MediaBlobRecord: Sendable, Equatable {
    public let conversationID: String
    public let mediaID: String
    public let mediaType: String
    public let sha256: String
    public let byteSize: Int
    public let data: Data
    public let createdUnixMs: Int64

    public init(
        conversationID: String = "",
        mediaID: String,
        mediaType: String,
        sha256: String,
        byteSize: Int,
        data: Data,
        createdUnixMs: Int64 = SQLiteStore.nowMillis()
    ) {
        self.conversationID = conversationID
        self.mediaID = mediaID
        self.mediaType = mediaType
        self.sha256 = sha256
        self.byteSize = byteSize
        self.data = data
        self.createdUnixMs = createdUnixMs
    }
}

public struct MessageStatsRecord: Codable, Sendable, Equatable {
    public let messageID: String
    public let conversationID: String
    public let modelID: String
    public let promptTokens: Int
    public let outputTokens: Int
    public let thinkingTokens: Int
    public let contentTokens: Int
    public let cachedTokens: Int
    public let promptTPS: Double
    public let generationTPS: Double
    public let ttftMs: Double?
    public let peakMemoryGB: Double?
    public let queueWaitMs: Double?
    public let thinkingDurationMs: Double?
    public let contentDurationMs: Double?
    public let generationDurationMs: Double?
    public let requestDurationMs: Double?
    public let modelLoadMs: Double?
    public let success: Bool
    public let cache: CacheReport?
    public let speculative: SpeculativeReport?
    public let specEmittedTokens: Int?
    public let createdUnixMs: Int64

    public init(
        messageID: String,
        conversationID: String,
        modelID: String,
        promptTokens: Int,
        outputTokens: Int,
        thinkingTokens: Int = 0,
        contentTokens: Int = 0,
        cachedTokens: Int = 0,
        promptTPS: Double = 0,
        generationTPS: Double = 0,
        ttftMs: Double? = nil,
        thinkingDurationMs: Double? = nil,
        contentDurationMs: Double? = nil,
        generationDurationMs: Double? = nil,
        requestDurationMs: Double? = nil,
        createdUnixMs: Int64 = SQLiteStore.nowMillis(),
        peakMemoryGB: Double? = nil,
        queueWaitMs: Double? = nil,
        cache: CacheReport? = nil,
        speculative: SpeculativeReport? = nil,
        specEmittedTokens: Int? = nil,
        modelLoadMs: Double? = nil,
        success: Bool = true
    ) {
        self.messageID = messageID
        self.conversationID = conversationID
        self.modelID = modelID
        self.promptTokens = promptTokens
        self.outputTokens = outputTokens
        self.thinkingTokens = thinkingTokens
        self.contentTokens = contentTokens
        self.cachedTokens = cachedTokens
        self.promptTPS = promptTPS
        self.generationTPS = generationTPS
        self.ttftMs = ttftMs
        self.peakMemoryGB = peakMemoryGB
        self.queueWaitMs = queueWaitMs
        self.thinkingDurationMs = thinkingDurationMs
        self.contentDurationMs = contentDurationMs
        self.generationDurationMs = generationDurationMs
        self.requestDurationMs = requestDurationMs
        self.modelLoadMs = modelLoadMs
        self.success = success
        self.cache = cache
        self.speculative = speculative
        self.specEmittedTokens = specEmittedTokens
        self.createdUnixMs = createdUnixMs
    }
}

public struct NotebookRecord: Codable, Identifiable, Sendable, Equatable, Hashable {
    public let id: String
    public var title: String
    public var body: String
    public var systemPrompt: String
    public var modelID: String?
    public var params: SamplerSettings?
    public var appliedPresetID: String?
    public let createdUnixMs: Int64
    public var updatedUnixMs: Int64

    public init(
        id: String = UUID().uuidString,
        title: String = "Untitled Notebook",
        body: String = "",
        systemPrompt: String = "",
        modelID: String? = nil,
        params: SamplerSettings? = nil,
        appliedPresetID: String? = nil,
        updatedUnixMs: Int64 = SQLiteStore.nowMillis(),
        createdUnixMs: Int64? = nil
    ) {
        self.id = id
        self.title = title
        self.body = body
        self.systemPrompt = systemPrompt
        self.modelID = modelID
        self.params = params
        self.appliedPresetID = appliedPresetID
        self.createdUnixMs = createdUnixMs ?? updatedUnixMs
        self.updatedUnixMs = updatedUnixMs
    }
}

public struct PresetRecord: Codable, Identifiable, Sendable, Equatable {
    public let id: String
    public var name: String
    public var description: String
    public var scope: String
    public var settings: SamplerSettings
    public var systemPrompt: String?
    public let createdUnixMs: Int64
    public var updatedUnixMs: Int64

    public init(
        id: String = UUID().uuidString,
        name: String,
        scope: String = "sampler",
        settings: SamplerSettings,
        systemPrompt: String? = nil,
        description: String = "",
        updatedUnixMs: Int64 = SQLiteStore.nowMillis(),
        createdUnixMs: Int64? = nil
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.scope = scope
        self.settings = settings
        self.systemPrompt = systemPrompt
        self.createdUnixMs = createdUnixMs ?? updatedUnixMs
        self.updatedUnixMs = updatedUnixMs
    }

    public init(
        id: String = UUID().uuidString,
        name: String,
        description: String,
        scope: String = "sampler",
        systemPrompt: String? = nil,
        settings: SamplerSettings,
        updatedUnixMs: Int64 = SQLiteStore.nowMillis(),
        createdUnixMs: Int64? = nil
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.scope = scope
        self.settings = settings
        self.systemPrompt = systemPrompt
        self.createdUnixMs = createdUnixMs ?? updatedUnixMs
        self.updatedUnixMs = updatedUnixMs
    }
}

/// Zero-dependency SQLite3 (`WAL` mode) store for conversations, messages, media blobs,
/// message stats, notebooks, and presets. Works identically on `macOS 27.2+` and `iOS 27.2+`.
public final class SQLiteStore: @unchecked Sendable {
    public static let validProfileRanges: Set<String> = ["1h", "6h", "24h", "7d"]

    private var db: OpaquePointer?
    private let lock = NSLock()
    private let observerLock = NSLock()
    private var mutationObservers: [UUID: AsyncStream<StoreMutationEvent>.Continuation] = [:]
    public let databaseURL: URL

    public static func nowMillis() -> Int64 {
        Int64(Date().timeIntervalSince1970 * 1000)
    }

    /// Constrains MIME type to the block's media family (`image/*` or `audio/*`), degrading off-family claims
    /// (such as `text/html`) to `application/octet-stream`.
    public static func sanitizeMediaType(_ raw: String, kind: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        let prefix = "\(kind.lowercased())/"
        if lower.hasPrefix(prefix), trimmed.count > prefix.count {
            return trimmed
        }
        return "application/octet-stream"
    }

    public static func mediaServeURL(conversationID: String, mediaID: String) -> String {
        "/v1/conversations/\(conversationID)/media/\(mediaID)"
    }

    public static func extractLocalMediaID(from urlString: String?, conversationID: String) -> String? {
        guard let urlString, !urlString.isEmpty else { return nil }
        let prefix = "/v1/conversations/\(conversationID)/media/"
        if urlString.hasPrefix(prefix) {
            let tail = String(urlString.dropFirst(prefix.count))
            if !tail.isEmpty, !tail.contains("/") {
                return tail
            }
        }
        return nil
    }

    public convenience init(path: String) throws {
        let expanded = (path as NSString).expandingTildeInPath
        try self.init(databaseURL: URL(fileURLWithPath: expanded))
    }

    public init(databaseURL: URL) throws {
        self.databaseURL = databaseURL
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var handle: OpaquePointer?
        if sqlite3_open(databaseURL.path, &handle) != SQLITE_OK {
            let msg = handle.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            sqlite3_close(handle)
            throw SQLiteStoreError.openFailed(msg)
        }
        self.db = handle
        try execute("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA foreign_keys = ON;
        """)
        try ensureCurrentSchemaUnlocked()
    }

    deinit {
        observerLock.lock()
        let continuations = Array(mutationObservers.values)
        mutationObservers.removeAll()
        observerLock.unlock()
        for c in continuations {
            c.finish()
        }
        if let db { sqlite3_close(db) }
    }

    /// Subscribes to live `StoreMutationEvent` notifications emitted after any store mutation.
    public func mutationStream() -> AsyncStream<StoreMutationEvent> {
        let id = UUID()
        return AsyncStream { continuation in
            self.observerLock.lock()
            self.mutationObservers[id] = continuation
            self.observerLock.unlock()
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                self.observerLock.lock()
                self.mutationObservers.removeValue(forKey: id)
                self.observerLock.unlock()
            }
        }
    }

    private func emit(_ event: StoreMutationEvent) {
        observerLock.lock()
        let continuations = Array(mutationObservers.values)
        observerLock.unlock()
        for c in continuations {
            c.yield(event)
        }
    }

    private func inTransactionUnlocked<T>(_ body: () throws -> T) throws -> T {
        try execute("BEGIN IMMEDIATE TRANSACTION;")
        do {
            let result = try body()
            try execute("COMMIT;")
            return result
        } catch {
            try? execute("ROLLBACK;")
            throw error
        }
    }

    // MARK: - Clean Table Recreation (No Migration Code)

    private func ensureCurrentSchemaUnlocked() throws {
        let requiredColumnsByTable: [(table: String, required: Set<String>)] = [
            ("store_meta", ["key", "value"]),
            ("conversations", ["id", "title", "system_prompt", "default_model_id", "params_json", "applied_preset_id", "created_ms", "updated_ms"]),
            ("messages", ["id", "conversation_id", "position", "role", "content", "thinking", "model_id", "stop_reason", "token_count", "gen_tps", "ttft_ms", "cache_json", "speculative_json", "compute_placement", "content_blocks_json", "created_ms", "updated_ms"]),
            ("media_blobs", ["conversation_id", "media_id", "media_type", "sha256", "byte_size", "data", "created_ms"]),
            ("message_stats", ["message_id", "conversation_id", "model_id", "prompt_tokens", "output_tokens", "thinking_tokens", "content_tokens", "cached_tokens", "prompt_tps", "generation_tps", "ttft_ms", "peak_memory_gb", "queue_wait_ms", "thinking_duration_ms", "content_duration_ms", "generation_duration_ms", "request_duration_ms", "model_load_ms", "success", "cache_json", "speculative_json", "spec_emitted_tokens", "created_ms"]),
            ("notebooks", ["id", "title", "body", "system_prompt", "model_id", "params_json", "applied_preset_id", "created_ms", "updated_ms"]),
            ("presets", ["id", "name", "description", "scope", "settings_json", "system_prompt", "created_ms", "updated_ms"]),
        ]

        var needsRecreate = false
        for (table, required) in requiredColumnsByTable {
            let existing = tableColumnsUnlocked(table)
            if !existing.isEmpty && !required.isSubset(of: existing) {
                needsRecreate = true
                break
            }
        }
        if !needsRecreate, let presetsSQL = tableCreateSQLUnlocked("presets"),
           !presetsSQL.localizedCaseInsensitiveContains("COLLATE NOCASE") {
            needsRecreate = true
        }

        if needsRecreate {
            try execute("""
            PRAGMA foreign_keys = OFF;
            DROP TABLE IF EXISTS message_stats;
            DROP TABLE IF EXISTS messages;
            DROP TABLE IF EXISTS conversations;
            DROP TABLE IF EXISTS media_blobs;
            DROP TABLE IF EXISTS notebooks;
            DROP TABLE IF EXISTS presets;
            DROP TABLE IF EXISTS store_meta;
            PRAGMA foreign_keys = ON;
            """)
        }

        try execute("""
        CREATE TABLE IF NOT EXISTS store_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            system_prompt TEXT NOT NULL DEFAULT '',
            default_model_id TEXT,
            params_json TEXT,
            applied_preset_id TEXT,
            created_ms INTEGER NOT NULL,
            updated_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            thinking TEXT,
            model_id TEXT,
            stop_reason TEXT,
            token_count INTEGER,
            gen_tps REAL,
            ttft_ms REAL,
            cache_json TEXT,
            speculative_json TEXT,
            compute_placement TEXT,
            content_blocks_json TEXT,
            created_ms INTEGER NOT NULL,
            updated_ms INTEGER NOT NULL,
            UNIQUE(conversation_id, position)
        );

        CREATE TABLE IF NOT EXISTS media_blobs (
            conversation_id TEXT NOT NULL,
            media_id TEXT NOT NULL,
            media_type TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            data BLOB NOT NULL,
            created_ms INTEGER NOT NULL,
            PRIMARY KEY (conversation_id, media_id)
        );

        CREATE TABLE IF NOT EXISTS message_stats (
            message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            thinking_tokens INTEGER NOT NULL DEFAULT 0,
            content_tokens INTEGER NOT NULL DEFAULT 0,
            cached_tokens INTEGER NOT NULL DEFAULT 0,
            prompt_tps REAL NOT NULL DEFAULT 0,
            generation_tps REAL NOT NULL DEFAULT 0,
            ttft_ms REAL,
            peak_memory_gb REAL,
            queue_wait_ms REAL,
            thinking_duration_ms REAL,
            content_duration_ms REAL,
            generation_duration_ms REAL,
            request_duration_ms REAL,
            model_load_ms REAL,
            success INTEGER NOT NULL DEFAULT 1,
            cache_json TEXT,
            speculative_json TEXT,
            spec_emitted_tokens INTEGER,
            created_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS notebooks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            system_prompt TEXT NOT NULL DEFAULT '',
            model_id TEXT,
            params_json TEXT,
            applied_preset_id TEXT,
            created_ms INTEGER NOT NULL,
            updated_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS presets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            description TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            system_prompt TEXT,
            created_ms INTEGER NOT NULL,
            updated_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conv_pos ON messages(conversation_id, position);
        CREATE INDEX IF NOT EXISTS idx_message_stats_created ON message_stats(created_ms);
        INSERT OR IGNORE INTO store_meta (key, value) VALUES ('schema_version', '7');
        """)
    }

    private func tableCreateSQLUnlocked(_ table: String) -> String? {
        var stmt: OpaquePointer?
        let sql = "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1;"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, table, -1, SQLITE_TRANSIENT_CORE)
        guard sqlite3_step(stmt) == SQLITE_ROW, let cStr = sqlite3_column_text(stmt, 0) else {
            return nil
        }
        return String(cString: cStr)
    }

    private func tableColumnsUnlocked(_ table: String) -> Set<String> {
        var stmt: OpaquePointer?
        var cols = Set<String>()
        guard sqlite3_prepare_v2(db, "PRAGMA table_info(\(table));", -1, &stmt, nil) == SQLITE_OK else {
            return cols
        }
        defer { sqlite3_finalize(stmt) }
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let cStr = sqlite3_column_text(stmt, 1) {
                cols.insert(String(cString: cStr))
            }
        }
        return cols
    }

    // MARK: - Conversations

    public func upsertConversation(_ conv: ConversationRecord) throws {
        lock.lock()
        do {
            try upsertConversationUnlocked(conv)
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        emit(.conversationUpserted(conv.id))
    }

    private func upsertConversationUnlocked(_ conv: ConversationRecord) throws {
        let sql = """
        INSERT INTO conversations (
            id, title, system_prompt, default_model_id, params_json, applied_preset_id, created_ms, updated_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            system_prompt = excluded.system_prompt,
            default_model_id = excluded.default_model_id,
            params_json = excluded.params_json,
            applied_preset_id = excluded.applied_preset_id,
            updated_ms = excluded.updated_ms;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, conv.id, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, conv.title, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 3, conv.systemPrompt, -1, SQLITE_TRANSIENT_CORE)
        bindOptionalText(stmt, index: 4, value: conv.defaultModelID)
        bindOptionalEncodable(stmt, index: 5, value: conv.params)
        bindOptionalText(stmt, index: 6, value: conv.appliedPresetID)
        sqlite3_bind_int64(stmt, 7, conv.createdUnixMs)
        sqlite3_bind_int64(stmt, 8, conv.updatedUnixMs)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    public func listConversations() throws -> [ConversationRecord] {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        SELECT id, title, system_prompt, default_model_id, params_json, applied_preset_id, created_ms, updated_ms
        FROM conversations
        ORDER BY updated_ms DESC;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        let decoder = JSONDecoder()
        var out: [ConversationRecord] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append(readConversationRow(stmt, decoder: decoder))
        }
        return out
    }

    public func getConversation(id: String) throws -> ConversationRecord? {
        lock.lock()
        defer { lock.unlock() }
        return try getConversationUnlocked(id: id)
    }

    /// Returns a conversation together with its ordered lightweight messages (`hydrateMedia: false`, joined `stats`).
    public func getConversationDetail(id: String) throws -> (conversation: ConversationRecord, messages: [MessageRecord])? {
        lock.lock()
        defer { lock.unlock() }
        guard let conv = try getConversationUnlocked(id: id) else { return nil }
        let msgs = try listMessagesUnlocked(conversationID: id, hydrateMedia: false)
        return (conv, msgs)
    }

    private func getConversationUnlocked(id: String) throws -> ConversationRecord? {
        let sql = """
        SELECT id, title, system_prompt, default_model_id, params_json, applied_preset_id, created_ms, updated_ms
        FROM conversations
        WHERE id = ?
        LIMIT 1;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
        if sqlite3_step(stmt) == SQLITE_ROW {
            return readConversationRow(stmt, decoder: JSONDecoder())
        }
        return nil
    }

    @discardableResult
    public func deleteConversation(id: String) throws -> Bool {
        lock.lock()
        let removed: Bool
        do {
            removed = try inTransactionUnlocked {
                var existsStmt: OpaquePointer?
                guard sqlite3_prepare_v2(db, "SELECT 1 FROM conversations WHERE id = ? LIMIT 1;", -1, &existsStmt, nil) == SQLITE_OK else {
                    throw lastError()
                }
                sqlite3_bind_text(existsStmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
                let exists = sqlite3_step(existsStmt) == SQLITE_ROW
                sqlite3_finalize(existsStmt)
                guard exists else { return false }

                try executeBoundDelete("DELETE FROM message_stats WHERE conversation_id = ?;", param: id)
                try executeBoundDelete("DELETE FROM messages WHERE conversation_id = ?;", param: id)
                try executeBoundDelete("DELETE FROM media_blobs WHERE conversation_id = ?;", param: id)
                try executeBoundDelete("DELETE FROM conversations WHERE id = ?;", param: id)
                return true
            }
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        if removed {
            emit(.conversationDeleted(id))
        }
        return removed
    }

    private func executeBoundDelete(_ sql: String, param: String) throws {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, param, -1, SQLITE_TRANSIENT_CORE)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    /// Clones a conversation (optionally truncating at `upToPosition` inclusive) and returns
    /// both the cloned `ConversationRecord` and its cloned `MessageRecord` rows with rewritten `/media/` URLs.
    public func cloneConversationWithMessages(
        id: String,
        upToPosition: Int? = nil,
        newTitle: String? = nil
    ) throws -> (conversation: ConversationRecord, messages: [MessageRecord]) {
        lock.lock()
        let cloned: (conversation: ConversationRecord, messages: [MessageRecord])
        do {
            cloned = try inTransactionUnlocked {
                guard let source = try getConversationUnlocked(id: id) else {
                    throw SQLiteStoreError.notFound("Conversation '\(id)' not found")
                }
                let now = Self.nowMillis()
                let trimmedTitle = newTitle?.trimmingCharacters(in: .whitespacesAndNewlines)
                let clonedTitle = (trimmedTitle?.isEmpty == false) ? trimmedTitle! : "Copy of \(source.title)"
                let newConv = ConversationRecord(
                    id: UUID().uuidString,
                    title: clonedTitle,
                    systemPrompt: source.systemPrompt,
                    defaultModelID: source.defaultModelID,
                    params: source.params,
                    appliedPresetID: source.appliedPresetID,
                    createdUnixMs: now,
                    updatedUnixMs: now
                )
                try upsertConversationUnlocked(newConv)

                // Copy scoped `media_blobs` rows from source conversation to new conversation ID.
                let copyBlobsSQL = """
                INSERT OR IGNORE INTO media_blobs (
                    conversation_id, media_id, media_type, sha256, byte_size, data, created_ms
                )
                SELECT ?, media_id, media_type, sha256, byte_size, data, ?
                FROM media_blobs
                WHERE conversation_id = ?;
                """
                var blobStmt: OpaquePointer?
                guard sqlite3_prepare_v2(db, copyBlobsSQL, -1, &blobStmt, nil) == SQLITE_OK else { throw lastError() }
                sqlite3_bind_text(blobStmt, 1, newConv.id, -1, SQLITE_TRANSIENT_CORE)
                sqlite3_bind_int64(blobStmt, 2, now)
                sqlite3_bind_text(blobStmt, 3, id, -1, SQLITE_TRANSIENT_CORE)
                guard sqlite3_step(blobStmt) == SQLITE_DONE else {
                    sqlite3_finalize(blobStmt)
                    throw lastError()
                }
                sqlite3_finalize(blobStmt)

                let oldPrefix = "/v1/conversations/\(id)/media/"
                let newPrefix = "/v1/conversations/\(newConv.id)/media/"
                let sourceMsgs = try listMessagesUnlocked(conversationID: id, hydrateMedia: false)
                var copiedMessages: [MessageRecord] = []
                var nextPosition = 0
                for msg in sourceMsgs {
                    if let maxPos = upToPosition, msg.position > maxPos {
                        continue
                    }
                    let rewrittenBlocks = msg.contentBlocks.map { blocks in
                        blocks.map { part -> MessagesRequest.MessageTurn.ContentPart in
                            guard case .media(var att) = part else { return part }
                            if let oldURL = att.url, oldURL.contains(oldPrefix) {
                                att.url = oldURL.replacingOccurrences(of: oldPrefix, with: newPrefix)
                            }
                            return .media(att)
                        }
                    }
                    let copyMsg = MessageRecord(
                        id: UUID().uuidString,
                        conversationID: newConv.id,
                        position: nextPosition,
                        role: msg.role,
                        content: msg.content,
                        thinking: msg.thinking,
                        modelID: msg.modelID,
                        stopReason: msg.stopReason,
                        tokenCount: msg.tokenCount,
                        generationTPS: msg.generationTPS,
                        ttftMs: msg.ttftMs,
                        cacheReport: msg.cacheReport,
                        speculativeReport: msg.speculativeReport,
                        computePlacement: msg.computePlacement,
                        contentBlocks: rewrittenBlocks,
                        createdUnixMs: now + Int64(nextPosition),
                        updatedUnixMs: now + Int64(nextPosition)
                    )
                    let inserted = try appendMessageUnlocked(copyMsg)
                    copiedMessages.append(inserted)
                    nextPosition += 1
                }
                if upToPosition != nil {
                    try gcOrphanMediaBlobsUnlocked(conversationID: newConv.id)
                }
                return (newConv, copiedMessages)
            }
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        emit(.conversationUpserted(cloned.conversation.id))
        emit(.messagesChanged(cloned.conversation.id))
        return cloned
    }

    /// Clones a conversation (optionally truncating at `upToPosition` inclusive) into a new independent record.
    public func cloneConversation(
        id: String,
        upToPosition: Int? = nil,
        newTitle: String? = nil
    ) throws -> ConversationRecord {
        try cloneConversationWithMessages(id: id, upToPosition: upToPosition, newTitle: newTitle).conversation
    }

    // MARK: - Messages & Media Blobs

    public func nextMessagePosition(conversationID: String) throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        return try nextMessagePositionUnlocked(conversationID: conversationID)
    }

    private func nextMessagePositionUnlocked(conversationID: String) throws -> Int {
        let sql = "SELECT COALESCE(MAX(position) + 1, 0) FROM messages WHERE conversation_id = ?;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
        if sqlite3_step(stmt) == SQLITE_ROW {
            return Int(sqlite3_column_int64(stmt, 0))
        }
        return 0
    }

    @discardableResult
    public func appendMessage(_ message: MessageRecord) throws -> MessageRecord {
        lock.lock()
        let stored: MessageRecord
        do {
            stored = try inTransactionUnlocked {
                try appendMessageUnlocked(message)
            }
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        emit(.messagesChanged(stored.conversationID))
        return stored
    }

    @discardableResult
    public func updateMessage(_ message: MessageRecord) throws -> MessageRecord {
        try appendMessage(message)
    }

    private func appendMessageUnlocked(_ message: MessageRecord) throws -> MessageRecord {
        var normalized = message
        if let blocks = message.contentBlocks {
            normalized.contentBlocks = try externalizeMediaInBlocksUnlocked(
                blocks,
                conversationID: normalized.conversationID
            )
        }
        let updatedMs = normalized.updatedUnixMs ?? Self.nowMillis()
        normalized.updatedUnixMs = updatedMs

        // Ensure `UNIQUE(conversation_id, position)` never collides if caller overwrites a position with a new ID
        let dedupPosSQL = "DELETE FROM messages WHERE conversation_id = ? AND position = ? AND id != ?;"
        var dedupStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, dedupPosSQL, -1, &dedupStmt, nil) == SQLITE_OK {
            sqlite3_bind_text(dedupStmt, 1, normalized.conversationID, -1, SQLITE_TRANSIENT_CORE)
            sqlite3_bind_int64(dedupStmt, 2, Int64(normalized.position))
            sqlite3_bind_text(dedupStmt, 3, normalized.id, -1, SQLITE_TRANSIENT_CORE)
            _ = sqlite3_step(dedupStmt)
            sqlite3_finalize(dedupStmt)
        }

        let sql = """
        INSERT INTO messages (
            id, conversation_id, position, role, content, thinking, model_id,
            stop_reason, token_count, gen_tps, ttft_ms, cache_json, speculative_json,
            compute_placement, content_blocks_json, created_ms, updated_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            position = excluded.position,
            role = excluded.role,
            content = excluded.content,
            thinking = excluded.thinking,
            model_id = excluded.model_id,
            stop_reason = excluded.stop_reason,
            token_count = excluded.token_count,
            gen_tps = excluded.gen_tps,
            ttft_ms = excluded.ttft_ms,
            cache_json = excluded.cache_json,
            speculative_json = excluded.speculative_json,
            compute_placement = excluded.compute_placement,
            content_blocks_json = COALESCE(excluded.content_blocks_json, messages.content_blocks_json),
            updated_ms = excluded.updated_ms;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, normalized.id, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, normalized.conversationID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_int64(stmt, 3, Int64(normalized.position))
        sqlite3_bind_text(stmt, 4, normalized.role, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 5, normalized.content, -1, SQLITE_TRANSIENT_CORE)
        bindOptionalText(stmt, index: 6, value: normalized.thinking)
        bindOptionalText(stmt, index: 7, value: normalized.modelID)
        bindOptionalText(stmt, index: 8, value: normalized.stopReason)
        bindOptionalInt(stmt, index: 9, value: normalized.tokenCount)
        bindOptionalDouble(stmt, index: 10, value: normalized.generationTPS)
        bindOptionalDouble(stmt, index: 11, value: normalized.ttftMs)
        bindOptionalEncodable(stmt, index: 12, value: normalized.cacheReport)
        bindOptionalEncodable(stmt, index: 13, value: normalized.speculativeReport)
        bindOptionalText(stmt, index: 14, value: normalized.computePlacement?.rawValue)
        bindOptionalEncodable(stmt, index: 15, value: normalized.contentBlocks)
        sqlite3_bind_int64(stmt, 16, normalized.createdUnixMs)
        sqlite3_bind_int64(stmt, 17, updatedMs)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }

        try touchConversationUnlocked(conversationID: normalized.conversationID, nowMs: updatedMs)
        if message.contentBlocks != nil {
            try gcOrphanMediaBlobsUnlocked(conversationID: normalized.conversationID)
        }
        return (try getMessageUnlocked(id: normalized.id, hydrateMedia: false)) ?? normalized
    }

    private func touchConversationUnlocked(conversationID: String, nowMs: Int64 = SQLiteStore.nowMillis()) throws {
        let touchConv = "UPDATE conversations SET updated_ms = ? WHERE id = ?;"
        var uStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, touchConv, -1, &uStmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(uStmt) }
        sqlite3_bind_int64(uStmt, 1, nowMs)
        sqlite3_bind_text(uStmt, 2, conversationID, -1, SQLITE_TRANSIENT_CORE)
        guard sqlite3_step(uStmt) == SQLITE_DONE else { throw lastError() }
    }

    private static let messageSelectColumns = """
    m.id, m.conversation_id, m.position, m.role, m.content, m.thinking, m.model_id,
    m.stop_reason, m.token_count, m.gen_tps, m.ttft_ms, m.cache_json, m.speculative_json,
    m.compute_placement, m.content_blocks_json, m.created_ms, m.updated_ms,
    s.message_id, s.prompt_tokens, s.output_tokens, s.prompt_tps, s.generation_tps,
    s.ttft_ms, s.peak_memory_gb, s.queue_wait_ms, s.thinking_duration_ms,
    s.content_duration_ms, s.generation_duration_ms, s.request_duration_ms,
    s.cache_json, s.speculative_json
    """

    public func getMessage(id: String, hydrateMedia: Bool = false) throws -> MessageRecord? {
        lock.lock()
        defer { lock.unlock() }
        return try getMessageUnlocked(id: id, hydrateMedia: hydrateMedia)
    }

    private func getMessageUnlocked(id: String, hydrateMedia: Bool) throws -> MessageRecord? {
        let sql = """
        SELECT \(Self.messageSelectColumns)
        FROM messages m
        LEFT JOIN message_stats s ON s.message_id = m.id
        WHERE m.id = ?
        LIMIT 1;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
        if sqlite3_step(stmt) == SQLITE_ROW {
            var msg = readMessageRow(stmt, decoder: JSONDecoder())
            if hydrateMedia, let blocks = msg.contentBlocks {
                msg.contentBlocks = hydrateMediaInBlocksUnlocked(blocks, conversationID: msg.conversationID)
            }
            return msg
        }
        return nil
    }

    public func listMessages(conversationID: String, hydrateMedia: Bool = false) throws -> [MessageRecord] {
        lock.lock()
        defer { lock.unlock() }
        return try listMessagesUnlocked(conversationID: conversationID, hydrateMedia: hydrateMedia)
    }

    private func listMessagesUnlocked(conversationID: String, hydrateMedia: Bool) throws -> [MessageRecord] {
        let sql = """
        SELECT \(Self.messageSelectColumns)
        FROM messages m
        LEFT JOIN message_stats s ON s.message_id = m.id
        WHERE m.conversation_id = ?
        ORDER BY m.position ASC, m.created_ms ASC;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
        var out: [MessageRecord] = []
        let decoder = JSONDecoder()
        while sqlite3_step(stmt) == SQLITE_ROW {
            var row = readMessageRow(stmt, decoder: decoder)
            if hydrateMedia, let blocks = row.contentBlocks {
                row.contentBlocks = hydrateMediaInBlocksUnlocked(blocks, conversationID: conversationID)
            }
            out.append(row)
        }
        return out
    }

    /// Resolves local `/v1/conversations/{convID}/media/{media_id}` URL blocks in `messages` back into
    /// raw base64 payloads for inference providers. Throws `.notFound` if a referenced blob is missing.
    public func hydrateMessagesForProvider(
        conversationID: String,
        messages: [MessageRecord]
    ) throws -> [MessageRecord] {
        lock.lock()
        defer { lock.unlock() }
        return try messages.map { msg in
            guard let blocks = msg.contentBlocks else { return msg }
            var hydrated = msg
            hydrated.contentBlocks = try hydrateMediaInBlocksStrictUnlocked(
                blocks,
                conversationID: conversationID
            )
            return hydrated
        }
    }

    /// Fetches messages for `conversationID` and hydrates all local media URLs back into raw bytes.
    public func hydrateMessagesForProvider(conversationID: String) throws -> [MessageRecord] {
        lock.lock()
        defer { lock.unlock() }
        let lightweight = try listMessagesUnlocked(conversationID: conversationID, hydrateMedia: false)
        return try lightweight.map { msg in
            guard let blocks = msg.contentBlocks else { return msg }
            var hydrated = msg
            hydrated.contentBlocks = try hydrateMediaInBlocksStrictUnlocked(
                blocks,
                conversationID: conversationID
            )
            return hydrated
        }
    }

    /// Deletes all messages strictly after `afterPosition` (`position > afterPosition`) and returns deleted count.
    @discardableResult
    public func deleteMessagesAfter(conversationID: String, afterPosition: Int) throws -> Int {
        lock.lock()
        let removed: Int
        do {
            removed = try inTransactionUnlocked {
                let sql = "DELETE FROM messages WHERE conversation_id = ? AND position > ?;"
                var stmt: OpaquePointer?
                guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
                defer { sqlite3_finalize(stmt) }
                sqlite3_bind_text(stmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
                sqlite3_bind_int64(stmt, 2, Int64(afterPosition))
                guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
                let count = Int(sqlite3_changes(db))
                if count > 0 {
                    try touchConversationUnlocked(conversationID: conversationID)
                    try gcOrphanMediaBlobsUnlocked(conversationID: conversationID)
                }
                return count
            }
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        if removed > 0 {
            emit(.messagesChanged(conversationID))
        }
        return removed
    }

    /// Truncates messages at or after `fromPosition` (`position >= fromPosition`).
    @discardableResult
    public func deleteMessages(conversationID: String, fromPosition: Int) throws -> Int {
        try deleteMessagesAfter(conversationID: conversationID, afterPosition: fromPosition - 1)
    }

    /// Deletes a single message row WITHOUT compacting subsequent positions (gaps are valid).
    @discardableResult
    public func deleteMessage(conversationID: String, messageID: String) throws -> Bool {
        lock.lock()
        let didDelete: Bool
        do {
            didDelete = try inTransactionUnlocked {
                let delSQL = "DELETE FROM messages WHERE id = ? AND conversation_id = ?;"
                var delStmt: OpaquePointer?
                guard sqlite3_prepare_v2(db, delSQL, -1, &delStmt, nil) == SQLITE_OK else { throw lastError() }
                defer { sqlite3_finalize(delStmt) }
                sqlite3_bind_text(delStmt, 1, messageID, -1, SQLITE_TRANSIENT_CORE)
                sqlite3_bind_text(delStmt, 2, conversationID, -1, SQLITE_TRANSIENT_CORE)
                guard sqlite3_step(delStmt) == SQLITE_DONE else { throw lastError() }
                let deleted = sqlite3_changes(db) > 0
                if deleted {
                    try touchConversationUnlocked(conversationID: conversationID)
                    try gcOrphanMediaBlobsUnlocked(conversationID: conversationID)
                }
                return deleted
            }
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        if didDelete {
            emit(.messagesChanged(conversationID))
        }
        return didDelete
    }

    // MARK: - Atomic Generation Tail Commits

    /// Atomically deletes `position >= fromPosition`, inserts `newMessage`, and persists optional `stats`
    /// inside a single SQLite transaction (`regenerate` / `append` commit invariant).
    /// Returns `nil` if the conversation does not exist.
    @discardableResult
    public func replaceTailWithMessage(
        conversationID: String,
        fromPosition: Int,
        newMessage: MessageRecord,
        stats: MessageStatsRecord? = nil
    ) throws -> MessageRecord? {
        lock.lock()
        let stored: MessageRecord?
        do {
            stored = try inTransactionUnlocked {
                guard try getConversationUnlocked(id: conversationID) != nil else {
                    return nil
                }
                let delSQL = "DELETE FROM messages WHERE conversation_id = ? AND position >= ?;"
                var delStmt: OpaquePointer?
                guard sqlite3_prepare_v2(db, delSQL, -1, &delStmt, nil) == SQLITE_OK else { throw lastError() }
                sqlite3_bind_text(delStmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
                sqlite3_bind_int64(delStmt, 2, Int64(fromPosition))
                guard sqlite3_step(delStmt) == SQLITE_DONE else {
                    sqlite3_finalize(delStmt)
                    throw lastError()
                }
                sqlite3_finalize(delStmt)

                let nextPos = try nextMessagePositionUnlocked(conversationID: conversationID)
                let msgToInsert = MessageRecord(
                    id: newMessage.id,
                    conversationID: conversationID,
                    position: nextPos,
                    role: newMessage.role,
                    content: newMessage.content,
                    thinking: newMessage.thinking,
                    modelID: newMessage.modelID,
                    stopReason: newMessage.stopReason,
                    tokenCount: newMessage.tokenCount,
                    generationTPS: newMessage.generationTPS,
                    ttftMs: newMessage.ttftMs,
                    cacheReport: newMessage.cacheReport,
                    speculativeReport: newMessage.speculativeReport,
                    computePlacement: newMessage.computePlacement,
                    contentBlocks: newMessage.contentBlocks,
                    createdUnixMs: newMessage.createdUnixMs,
                    updatedUnixMs: Self.nowMillis()
                )
                let savedRow = try appendMessageUnlocked(msgToInsert)

                if let stats {
                    let boundStats = MessageStatsRecord(
                        messageID: savedRow.id,
                        conversationID: conversationID,
                        modelID: stats.modelID,
                        promptTokens: stats.promptTokens,
                        outputTokens: stats.outputTokens,
                        thinkingTokens: stats.thinkingTokens,
                        contentTokens: stats.contentTokens,
                        cachedTokens: stats.cachedTokens,
                        promptTPS: stats.promptTPS,
                        generationTPS: stats.generationTPS,
                        ttftMs: stats.ttftMs,
                        thinkingDurationMs: stats.thinkingDurationMs,
                        contentDurationMs: stats.contentDurationMs,
                        generationDurationMs: stats.generationDurationMs,
                        requestDurationMs: stats.requestDurationMs,
                        createdUnixMs: stats.createdUnixMs,
                        peakMemoryGB: stats.peakMemoryGB,
                        queueWaitMs: stats.queueWaitMs,
                        cache: stats.cache,
                        speculative: stats.speculative,
                        specEmittedTokens: stats.specEmittedTokens,
                        modelLoadMs: stats.modelLoadMs,
                        success: stats.success
                    )
                    try upsertMessageStatsUnlocked(boundStats)
                }

                try gcOrphanMediaBlobsUnlocked(conversationID: conversationID)
                return try getMessageUnlocked(id: savedRow.id, hydrateMedia: false)
            }
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        if stored != nil {
            emit(.messagesChanged(conversationID))
        }
        return stored
    }

    /// Convenience overload matching Python `replace_tail_with_message(after_position:)`.
    @discardableResult
    public func replaceTailWithMessage(
        conversationID: String,
        afterPosition: Int,
        newMessage: MessageRecord,
        stats: MessageStatsRecord? = nil
    ) throws -> MessageRecord? {
        try replaceTailWithMessage(
            conversationID: conversationID,
            fromPosition: afterPosition + 1,
            newMessage: newMessage,
            stats: stats
        )
    }

    /// Atomically deletes messages strictly after the anchor (`position > anchor.position`) and merges
    /// continuation output onto `anchorMessageID` while preserving the anchor's original `modelID`.
    /// Passing `updatedThinking == nil` preserves the anchor's prior `thinking`.
    @discardableResult
    public func replaceTailWithUpdate(
        conversationID: String,
        anchorMessageID: String,
        updatedContent: String,
        updatedThinking: String? = nil,
        updatedContentBlocks: [MessagesRequest.MessageTurn.ContentPart]? = nil,
        stopReason: String? = nil,
        tokenCount: Int? = nil,
        generationTPS: Double? = nil,
        ttftMs: Double? = nil,
        cacheReport: CacheReport? = nil,
        speculativeReport: SpeculativeReport? = nil,
        computePlacement: ComputePlacement? = nil,
        stats: MessageStatsRecord? = nil
    ) throws -> MessageRecord? {
        lock.lock()
        let updated: MessageRecord?
        do {
            updated = try inTransactionUnlocked {
                guard let anchor = try getMessageUnlocked(id: anchorMessageID, hydrateMedia: false),
                      anchor.conversationID == conversationID else {
                    return nil
                }
                let delSQL = "DELETE FROM messages WHERE conversation_id = ? AND position > ?;"
                var delStmt: OpaquePointer?
                guard sqlite3_prepare_v2(db, delSQL, -1, &delStmt, nil) == SQLITE_OK else { throw lastError() }
                sqlite3_bind_text(delStmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
                sqlite3_bind_int64(delStmt, 2, Int64(anchor.position))
                guard sqlite3_step(delStmt) == SQLITE_DONE else {
                    sqlite3_finalize(delStmt)
                    throw lastError()
                }
                sqlite3_finalize(delStmt)

                let finalThinking = updatedThinking ?? anchor.thinking
                let finalBlocks: [MessagesRequest.MessageTurn.ContentPart]?
                if let explicitBlocks = updatedContentBlocks {
                    finalBlocks = try externalizeMediaInBlocksUnlocked(
                        explicitBlocks,
                        conversationID: conversationID
                    )
                } else if let existingBlocks = anchor.contentBlocks {
                    var rebuilt: [MessagesRequest.MessageTurn.ContentPart] = []
                    if let th = finalThinking, !th.isEmpty {
                        rebuilt.append(.thinking(th))
                    }
                    for part in existingBlocks {
                        if case .media = part {
                            rebuilt.append(part)
                        }
                    }
                    rebuilt.append(.text(updatedContent))
                    finalBlocks = rebuilt
                } else {
                    finalBlocks = nil
                }

                let nowMs = Self.nowMillis()
                // Deliberately DO NOT update `model_id`: the original author stamp on the anchor stands.
                let updateSQL = """
                UPDATE messages
                SET content = ?,
                    thinking = ?,
                    stop_reason = ?,
                    token_count = ?,
                    gen_tps = ?,
                    ttft_ms = ?,
                    cache_json = ?,
                    speculative_json = ?,
                    compute_placement = ?,
                    content_blocks_json = ?,
                    updated_ms = ?
                WHERE id = ? AND conversation_id = ?;
                """
                var uStmt: OpaquePointer?
                guard sqlite3_prepare_v2(db, updateSQL, -1, &uStmt, nil) == SQLITE_OK else { throw lastError() }
                defer { sqlite3_finalize(uStmt) }
                sqlite3_bind_text(uStmt, 1, updatedContent, -1, SQLITE_TRANSIENT_CORE)
                bindOptionalText(uStmt, index: 2, value: finalThinking)
                bindOptionalText(uStmt, index: 3, value: stopReason ?? anchor.stopReason)
                bindOptionalInt(uStmt, index: 4, value: tokenCount ?? anchor.tokenCount)
                bindOptionalDouble(uStmt, index: 5, value: generationTPS ?? anchor.generationTPS)
                bindOptionalDouble(uStmt, index: 6, value: ttftMs ?? anchor.ttftMs)
                bindOptionalEncodable(uStmt, index: 7, value: cacheReport ?? anchor.cacheReport)
                bindOptionalEncodable(uStmt, index: 8, value: speculativeReport ?? anchor.speculativeReport)
                bindOptionalText(uStmt, index: 9, value: (computePlacement ?? anchor.computePlacement)?.rawValue)
                bindOptionalEncodable(uStmt, index: 10, value: finalBlocks)
                sqlite3_bind_int64(uStmt, 11, nowMs)
                sqlite3_bind_text(uStmt, 12, anchorMessageID, -1, SQLITE_TRANSIENT_CORE)
                sqlite3_bind_text(uStmt, 13, conversationID, -1, SQLITE_TRANSIENT_CORE)
                guard sqlite3_step(uStmt) == SQLITE_DONE else { throw lastError() }

                if let stats {
                    let boundStats = MessageStatsRecord(
                        messageID: anchorMessageID,
                        conversationID: conversationID,
                        modelID: stats.modelID,
                        promptTokens: stats.promptTokens,
                        outputTokens: stats.outputTokens,
                        thinkingTokens: stats.thinkingTokens,
                        contentTokens: stats.contentTokens,
                        cachedTokens: stats.cachedTokens,
                        promptTPS: stats.promptTPS,
                        generationTPS: stats.generationTPS,
                        ttftMs: stats.ttftMs,
                        thinkingDurationMs: stats.thinkingDurationMs,
                        contentDurationMs: stats.contentDurationMs,
                        generationDurationMs: stats.generationDurationMs,
                        requestDurationMs: stats.requestDurationMs,
                        createdUnixMs: stats.createdUnixMs,
                        peakMemoryGB: stats.peakMemoryGB,
                        queueWaitMs: stats.queueWaitMs,
                        cache: stats.cache,
                        speculative: stats.speculative,
                        specEmittedTokens: stats.specEmittedTokens,
                        modelLoadMs: stats.modelLoadMs,
                        success: stats.success
                    )
                    try upsertMessageStatsUnlocked(boundStats)
                }

                try touchConversationUnlocked(conversationID: conversationID, nowMs: nowMs)
                try gcOrphanMediaBlobsUnlocked(conversationID: conversationID)
                return try getMessageUnlocked(id: anchorMessageID, hydrateMedia: false)
            }
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        if updated != nil {
            emit(.messagesChanged(conversationID))
        }
        return updated
    }

    // MARK: - Content-Addressed Media Blobs

    @discardableResult
    public func storeMediaBlob(
        conversationID: String,
        data: Data,
        mediaType: String,
        kind: String? = nil
    ) throws -> String {
        lock.lock()
        defer { lock.unlock() }
        return try storeMediaBlobUnlocked(
            conversationID: conversationID,
            data: data,
            mediaType: mediaType,
            kind: kind
        ).mediaID
    }

    @discardableResult
    public func storeMediaBlob(data: Data, mediaType: String) throws -> String {
        try storeMediaBlob(conversationID: "_default", data: data, mediaType: mediaType, kind: nil)
    }

    private func storeMediaBlobUnlocked(
        conversationID: String,
        data: Data,
        mediaType: String,
        kind: String?
    ) throws -> (mediaID: String, sanitizedType: String) {
        let digest = SHA256.hash(data: data)
        let shaHex = digest.map { String(format: "%02x", $0) }.joined()
        let resolvedKind: String
        if let kind, !kind.isEmpty {
            resolvedKind = kind
        } else {
            let lower = mediaType.lowercased()
            resolvedKind = lower.hasPrefix("audio/") ? "audio" : "image"
        }
        let sanitizedType = Self.sanitizeMediaType(mediaType, kind: resolvedKind)
        let sql = """
        INSERT INTO media_blobs (conversation_id, media_id, media_type, sha256, byte_size, data, created_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id, media_id) DO NOTHING;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, shaHex, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 3, sanitizedType, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 4, shaHex, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_int64(stmt, 5, Int64(data.count))
        _ = data.withUnsafeBytes { rawPtr in
            sqlite3_bind_blob(stmt, 6, rawPtr.baseAddress, Int32(data.count), SQLITE_TRANSIENT_CORE)
        }
        sqlite3_bind_int64(stmt, 7, Self.nowMillis())
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
        return (shaHex, sanitizedType)
    }

    public func getMediaBlob(conversationID: String, mediaID: String) throws -> MediaBlobRecord? {
        lock.lock()
        defer { lock.unlock() }
        return getMediaBlobUnlocked(conversationID: conversationID, mediaID: mediaID)
    }

    public func getMediaBlobs(conversationID: String, mediaIDs: [String]) throws -> [String: MediaBlobRecord] {
        lock.lock()
        defer { lock.unlock() }
        var out: [String: MediaBlobRecord] = [:]
        for id in Set(mediaIDs) {
            if let blob = getMediaBlobUnlocked(conversationID: conversationID, mediaID: id) {
                out[id] = blob
            }
        }
        return out
    }

    public func getMediaBlob(id: String) throws -> MediaBlobRecord? {
        lock.lock()
        defer { lock.unlock() }
        return getMediaBlobAnyConversationUnlocked(id: id)
    }

    private func getMediaBlobUnlocked(conversationID: String, mediaID: String) -> MediaBlobRecord? {
        let sql = """
        SELECT conversation_id, media_id, media_type, sha256, byte_size, data, created_ms
        FROM media_blobs
        WHERE conversation_id = ? AND media_id = ?
        LIMIT 1;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, mediaID, -1, SQLITE_TRANSIENT_CORE)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return readMediaBlobRow(stmt)
    }

    private func getMediaBlobAnyConversationUnlocked(id: String) -> MediaBlobRecord? {
        let sql = """
        SELECT conversation_id, media_id, media_type, sha256, byte_size, data, created_ms
        FROM media_blobs
        WHERE media_id = ?
        LIMIT 1;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return readMediaBlobRow(stmt)
    }

    private func readMediaBlobRow(_ stmt: OpaquePointer?) -> MediaBlobRecord {
        let convID = String(cString: sqlite3_column_text(stmt, 0))
        let mediaID = String(cString: sqlite3_column_text(stmt, 1))
        let mediaType = String(cString: sqlite3_column_text(stmt, 2))
        let sha = String(cString: sqlite3_column_text(stmt, 3))
        let byteSize = Int(sqlite3_column_int64(stmt, 4))
        let blobBytes = Int(sqlite3_column_bytes(stmt, 5))
        let blobData: Data
        if blobBytes > 0, let raw = sqlite3_column_blob(stmt, 5) {
            blobData = Data(bytes: raw, count: blobBytes)
        } else {
            blobData = Data()
        }
        let createdMs = sqlite3_column_int64(stmt, 6)
        return MediaBlobRecord(
            conversationID: convID,
            mediaID: mediaID,
            mediaType: mediaType,
            sha256: sha,
            byteSize: byteSize,
            data: blobData,
            createdUnixMs: createdMs
        )
    }

    private func mediaBlobExistsUnlocked(conversationID: String, mediaID: String) -> Bool {
        let sql = "SELECT 1 FROM media_blobs WHERE conversation_id = ? AND media_id = ? LIMIT 1;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return false }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, mediaID, -1, SQLITE_TRANSIENT_CORE)
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    public func mediaBlobCount() throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM media_blobs;", -1, &stmt, nil) == SQLITE_OK else {
            throw lastError()
        }
        defer { sqlite3_finalize(stmt) }
        return sqlite3_step(stmt) == SQLITE_ROW ? Int(sqlite3_column_int64(stmt, 0)) : 0
    }

    private func externalizeMediaInBlocksUnlocked(
        _ blocks: [MessagesRequest.MessageTurn.ContentPart],
        conversationID: String
    ) throws -> [MessagesRequest.MessageTurn.ContentPart] {
        var out: [MessagesRequest.MessageTurn.ContentPart] = []
        out.reserveCapacity(blocks.count)
        for part in blocks {
            switch part {
            case .media(var att):
                if !att.dataBase64.isEmpty {
                    guard let decoded = Data(base64Encoded: att.dataBase64) else {
                        throw SQLiteStoreError.invalidInput("\(att.kind) block carries invalid base64 data")
                    }
                    if !decoded.isEmpty {
                        let stored = try storeMediaBlobUnlocked(
                            conversationID: conversationID,
                            data: decoded,
                            mediaType: att.mediaType,
                            kind: att.kind
                        )
                        att = MessagesRequest.MediaAttachment(
                            id: stored.mediaID,
                            kind: att.kind,
                            mediaType: stored.sanitizedType,
                            dataBase64: "",
                            filename: att.filename,
                            mediaID: stored.mediaID,
                            url: Self.mediaServeURL(conversationID: conversationID, mediaID: stored.mediaID)
                        )
                    }
                } else if let mID = att.mediaID {
                    // Only honor `media_id` if the blob actually belongs to THIS conversation.
                    if mediaBlobExistsUnlocked(conversationID: conversationID, mediaID: mID) {
                        att = MessagesRequest.MediaAttachment(
                            id: att.id,
                            kind: att.kind,
                            mediaType: Self.sanitizeMediaType(att.mediaType, kind: att.kind),
                            dataBase64: "",
                            filename: att.filename,
                            mediaID: mID,
                            url: att.url ?? Self.mediaServeURL(conversationID: conversationID, mediaID: mID)
                        )
                    } else {
                        att = MessagesRequest.MediaAttachment(
                            id: att.id == mID ? UUID().uuidString : att.id,
                            kind: att.kind,
                            mediaType: att.mediaType,
                            dataBase64: "",
                            filename: att.filename,
                            mediaID: nil,
                            url: att.url
                        )
                    }
                }
                out.append(.media(att))
            default:
                out.append(part)
            }
        }
        return out
    }

    private func hydrateMediaInBlocksUnlocked(
        _ blocks: [MessagesRequest.MessageTurn.ContentPart],
        conversationID: String
    ) -> [MessagesRequest.MessageTurn.ContentPart] {
        blocks.map { part in
            switch part {
            case .media(let att):
                let candidateID = att.mediaID ?? Self.extractLocalMediaID(from: att.url, conversationID: conversationID)
                if att.dataBase64.isEmpty,
                   let mID = candidateID,
                   let blob = getMediaBlobUnlocked(conversationID: conversationID, mediaID: mID) {
                    return .media(
                        MessagesRequest.MediaAttachment(
                            id: att.id,
                            kind: att.kind,
                            mediaType: blob.mediaType,
                            dataBase64: blob.data.base64EncodedString(),
                            filename: att.filename,
                            mediaID: mID,
                            url: att.url ?? Self.mediaServeURL(conversationID: conversationID, mediaID: mID)
                        )
                    )
                }
                return part
            default:
                return part
            }
        }
    }

    private func hydrateMediaInBlocksStrictUnlocked(
        _ blocks: [MessagesRequest.MessageTurn.ContentPart],
        conversationID: String
    ) throws -> [MessagesRequest.MessageTurn.ContentPart] {
        var out: [MessagesRequest.MessageTurn.ContentPart] = []
        out.reserveCapacity(blocks.count)
        for part in blocks {
            switch part {
            case .media(let att):
                if att.dataBase64.isEmpty,
                   let mID = att.mediaID ?? Self.extractLocalMediaID(from: att.url, conversationID: conversationID) {
                    guard let blob = getMediaBlobUnlocked(conversationID: conversationID, mediaID: mID) else {
                        throw SQLiteStoreError.notFound(
                            "Media blob '\(mID)' referenced by conversation '\(conversationID)' is missing"
                        )
                    }
                    out.append(
                        .media(
                            MessagesRequest.MediaAttachment(
                                id: att.id,
                                kind: att.kind,
                                mediaType: blob.mediaType,
                                dataBase64: blob.data.base64EncodedString(),
                                filename: att.filename,
                                mediaID: mID,
                                url: att.url ?? Self.mediaServeURL(conversationID: conversationID, mediaID: mID)
                            )
                        )
                    )
                } else {
                    out.append(part)
                }
            default:
                out.append(part)
            }
        }
        return out
    }

    /// Garbage-collects unreferenced `media_blobs` and `message_stats` rows after message/conversation deletion.
    private func gcOrphanMediaBlobsUnlocked(conversationID: String? = nil) throws {
        if let conversationID {
            let gcBlobsSQL = """
            DELETE FROM media_blobs
            WHERE conversation_id = ?
              AND NOT EXISTS (
                SELECT 1 FROM messages m
                WHERE m.conversation_id = media_blobs.conversation_id
                  AND m.content_blocks_json IS NOT NULL
                  AND m.content_blocks_json LIKE '%' || media_blobs.media_id || '%'
              );
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, gcBlobsSQL, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
            sqlite3_bind_text(stmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                sqlite3_finalize(stmt)
                throw lastError()
            }
            sqlite3_finalize(stmt)

            let gcStatsSQL = """
            DELETE FROM message_stats
            WHERE conversation_id = ?
              AND NOT EXISTS (
                SELECT 1 FROM messages m WHERE m.id = message_stats.message_id
              );
            """
            var sStmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, gcStatsSQL, -1, &sStmt, nil) == SQLITE_OK else { throw lastError() }
            sqlite3_bind_text(sStmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
            guard sqlite3_step(sStmt) == SQLITE_DONE else {
                sqlite3_finalize(sStmt)
                throw lastError()
            }
            sqlite3_finalize(sStmt)
        } else {
            try execute("""
            DELETE FROM media_blobs
            WHERE NOT EXISTS (
                SELECT 1 FROM messages m
                WHERE m.conversation_id = media_blobs.conversation_id
                  AND m.content_blocks_json IS NOT NULL
                  AND m.content_blocks_json LIKE '%' || media_blobs.media_id || '%'
            );
            DELETE FROM message_stats
            WHERE NOT EXISTS (
                SELECT 1 FROM messages m WHERE m.id = message_stats.message_id
            );
            """)
        }
    }

    // MARK: - Message Stats & Performance Profile

    public func upsertMessageStats(_ stats: MessageStatsRecord) throws {
        lock.lock()
        defer { lock.unlock() }
        try upsertMessageStatsUnlocked(stats)
    }

    private func upsertMessageStatsUnlocked(_ stats: MessageStatsRecord) throws {
        // Guard: if the message was deleted meanwhile, keep no orphan row.
        var existsStmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT 1 FROM messages WHERE id = ? AND conversation_id = ? LIMIT 1;", -1, &existsStmt, nil) == SQLITE_OK else {
            throw lastError()
        }
        sqlite3_bind_text(existsStmt, 1, stats.messageID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(existsStmt, 2, stats.conversationID, -1, SQLITE_TRANSIENT_CORE)
        let exists = sqlite3_step(existsStmt) == SQLITE_ROW
        sqlite3_finalize(existsStmt)
        guard exists else { return }

        // Strip human prose `reason` from stored cache JSON (matching Python `perf_collector.message_stats`).
        let storedCache: CacheReport? = stats.cache.map {
            CacheReport(
                promptTokens: $0.promptTokens,
                cachedTokens: $0.cachedTokens,
                processedTokens: $0.processedTokens,
                outcome: $0.outcome,
                cause: $0.cause,
                reason: nil
            )
        }

        let sql = """
        INSERT INTO message_stats (
            message_id, conversation_id, model_id, prompt_tokens, output_tokens,
            thinking_tokens, content_tokens, cached_tokens, prompt_tps, generation_tps,
            ttft_ms, peak_memory_gb, queue_wait_ms, thinking_duration_ms,
            content_duration_ms, generation_duration_ms, request_duration_ms,
            model_load_ms, success, cache_json, speculative_json, spec_emitted_tokens, created_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            model_id = excluded.model_id,
            prompt_tokens = excluded.prompt_tokens,
            output_tokens = excluded.output_tokens,
            thinking_tokens = excluded.thinking_tokens,
            content_tokens = excluded.content_tokens,
            cached_tokens = excluded.cached_tokens,
            prompt_tps = excluded.prompt_tps,
            generation_tps = excluded.generation_tps,
            ttft_ms = excluded.ttft_ms,
            peak_memory_gb = excluded.peak_memory_gb,
            queue_wait_ms = excluded.queue_wait_ms,
            thinking_duration_ms = excluded.thinking_duration_ms,
            content_duration_ms = excluded.content_duration_ms,
            generation_duration_ms = excluded.generation_duration_ms,
            request_duration_ms = excluded.request_duration_ms,
            model_load_ms = excluded.model_load_ms,
            success = excluded.success,
            cache_json = excluded.cache_json,
            speculative_json = excluded.speculative_json,
            spec_emitted_tokens = excluded.spec_emitted_tokens;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, stats.messageID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, stats.conversationID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 3, stats.modelID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_int64(stmt, 4, Int64(stats.promptTokens))
        sqlite3_bind_int64(stmt, 5, Int64(stats.outputTokens))
        sqlite3_bind_int64(stmt, 6, Int64(stats.thinkingTokens))
        sqlite3_bind_int64(stmt, 7, Int64(stats.contentTokens))
        sqlite3_bind_int64(stmt, 8, Int64(stats.cachedTokens))
        sqlite3_bind_double(stmt, 9, stats.promptTPS)
        sqlite3_bind_double(stmt, 10, stats.generationTPS)
        bindOptionalDouble(stmt, index: 11, value: stats.ttftMs)
        bindOptionalDouble(stmt, index: 12, value: stats.peakMemoryGB)
        bindOptionalDouble(stmt, index: 13, value: stats.queueWaitMs)
        bindOptionalDouble(stmt, index: 14, value: stats.thinkingDurationMs)
        bindOptionalDouble(stmt, index: 15, value: stats.contentDurationMs)
        bindOptionalDouble(stmt, index: 16, value: stats.generationDurationMs)
        bindOptionalDouble(stmt, index: 17, value: stats.requestDurationMs)
        bindOptionalDouble(stmt, index: 18, value: stats.modelLoadMs)
        sqlite3_bind_int64(stmt, 19, stats.success ? 1 : 0)
        bindOptionalEncodable(stmt, index: 20, value: storedCache)
        bindOptionalEncodable(stmt, index: 21, value: stats.speculative)
        let resolvedEmitted = stats.specEmittedTokens ?? stats.speculative.flatMap { $0.emittedTokens > 0 ? $0.emittedTokens : nil }
        bindOptionalInt(stmt, index: 22, value: resolvedEmitted)
        sqlite3_bind_int64(stmt, 23, stats.createdUnixMs)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    private struct SpeculativeDecodedSnapshot: Decodable {
        let draftedTokens: Int?
        let drafted: Int?
        let acceptedTokens: Int?
        let accepted: Int?
        let emittedTokens: Int?
        let emitted: Int?

        var resolvedDrafted: Int { draftedTokens ?? drafted ?? 0 }
        var resolvedAccepted: Int { acceptedTokens ?? accepted ?? 0 }
        var resolvedEmitted: Int? { emittedTokens ?? emitted }
    }

    public func listMessageStats(sinceUnixMs: Int64) throws -> [MessageStatsRecord] {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        SELECT message_id, conversation_id, model_id, prompt_tokens, output_tokens,
               thinking_tokens, content_tokens, cached_tokens, prompt_tps, generation_tps,
               ttft_ms, peak_memory_gb, queue_wait_ms, thinking_duration_ms,
               content_duration_ms, generation_duration_ms, request_duration_ms,
               model_load_ms, success, cache_json, speculative_json, spec_emitted_tokens, created_ms
        FROM message_stats
        WHERE created_ms >= ?
        ORDER BY created_ms ASC;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int64(stmt, 1, sinceUnixMs)
        let decoder = JSONDecoder()
        var out: [MessageStatsRecord] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let cacheObj = optionalColumnText(stmt, 19)
                .flatMap { try? decoder.decode(CacheReport.self, from: Data($0.utf8)) }
            let specRaw = optionalColumnText(stmt, 20)
            let specObj = specRaw.flatMap { try? decoder.decode(SpeculativeReport.self, from: Data($0.utf8)) }
            let specEmittedFromJSON = specRaw
                .flatMap { try? decoder.decode(SpeculativeDecodedSnapshot.self, from: Data($0.utf8)) }?
                .resolvedEmitted
            let explicitEmitted = optionalColumnInt(stmt, 21)

            out.append(
                MessageStatsRecord(
                    messageID: String(cString: sqlite3_column_text(stmt, 0)),
                    conversationID: String(cString: sqlite3_column_text(stmt, 1)),
                    modelID: String(cString: sqlite3_column_text(stmt, 2)),
                    promptTokens: Int(sqlite3_column_int64(stmt, 3)),
                    outputTokens: Int(sqlite3_column_int64(stmt, 4)),
                    thinkingTokens: Int(sqlite3_column_int64(stmt, 5)),
                    contentTokens: Int(sqlite3_column_int64(stmt, 6)),
                    cachedTokens: Int(sqlite3_column_int64(stmt, 7)),
                    promptTPS: sqlite3_column_double(stmt, 8),
                    generationTPS: sqlite3_column_double(stmt, 9),
                    ttftMs: optionalColumnDouble(stmt, 10),
                    thinkingDurationMs: optionalColumnDouble(stmt, 13),
                    contentDurationMs: optionalColumnDouble(stmt, 14),
                    generationDurationMs: optionalColumnDouble(stmt, 15),
                    requestDurationMs: optionalColumnDouble(stmt, 16),
                    createdUnixMs: sqlite3_column_int64(stmt, 22),
                    peakMemoryGB: optionalColumnDouble(stmt, 11),
                    queueWaitMs: optionalColumnDouble(stmt, 12),
                    cache: cacheObj,
                    speculative: specObj,
                    specEmittedTokens: explicitEmitted ?? specEmittedFromJSON,
                    modelLoadMs: optionalColumnDouble(stmt, 17),
                    success: sqlite3_column_int64(stmt, 18) != 0
                )
            )
        }
        return out
    }

    public struct ModelPerformanceSummary: Codable, Sendable, Equatable, Identifiable {
        public var id: String { modelID }
        public let modelID: String
        public let sampleCount: Int
        public let averageGenerationTPS: Double
        public let averagePromptTPS: Double
        public let averageTTFTMs: Double
        public let totalOutputTokens: Int
        public let totalPromptTokens: Int
        public let totalCachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case sampleCount = "sample_count"
            case averageGenerationTPS = "average_generation_tps"
            case averagePromptTPS = "average_prompt_tps"
            case averageTTFTMs = "average_ttft_ms"
            case totalOutputTokens = "total_output_tokens"
            case totalPromptTokens = "total_prompt_tokens"
            case totalCachedTokens = "total_cached_tokens"
        }

        public init(
            modelID: String,
            sampleCount: Int,
            averageGenerationTPS: Double,
            averagePromptTPS: Double,
            averageTTFTMs: Double,
            totalOutputTokens: Int,
            totalPromptTokens: Int,
            totalCachedTokens: Int
        ) {
            self.modelID = modelID
            self.sampleCount = sampleCount
            self.averageGenerationTPS = averageGenerationTPS
            self.averagePromptTPS = averagePromptTPS
            self.averageTTFTMs = averageTTFTMs
            self.totalOutputTokens = totalOutputTokens
            self.totalPromptTokens = totalPromptTokens
            self.totalCachedTokens = totalCachedTokens
        }
    }

    public struct TimingBreakdownEntry: Codable, Sendable, Equatable {
        public let operation: String
        public let avgTimeMs: Double
        public let count: Int
        public let percentage: Double

        enum CodingKeys: String, CodingKey {
            case operation
            case avgTimeMs = "avg_time_ms"
            case count
            case percentage
        }

        public init(operation: String, avgTimeMs: Double, count: Int, percentage: Double) {
            self.operation = operation
            self.avgTimeMs = avgTimeMs
            self.count = count
            self.percentage = percentage
        }
    }

    public struct ModelBottleneckEntry: Codable, Sendable, Equatable {
        public struct Breakdown: Codable, Sendable, Equatable {
            public let queueWait: Double
            public let modelLoad: Double
            public let tokenGeneration: Double

            enum CodingKeys: String, CodingKey {
                case queueWait = "queue_wait"
                case modelLoad = "model_load"
                case tokenGeneration = "token_generation"
            }

            public init(queueWait: Double, modelLoad: Double, tokenGeneration: Double) {
                self.queueWait = queueWait
                self.modelLoad = modelLoad
                self.tokenGeneration = tokenGeneration
            }
        }

        public let model: String
        public let avgTotalMs: Double
        public let breakdown: Breakdown
        public let requestCount: Int

        enum CodingKeys: String, CodingKey {
            case model
            case avgTotalMs = "avg_total_ms"
            case breakdown
            case requestCount = "request_count"
        }

        public init(model: String, avgTotalMs: Double, breakdown: Breakdown, requestCount: Int) {
            self.model = model
            self.avgTotalMs = avgTotalMs
            self.breakdown = breakdown
            self.requestCount = requestCount
        }
    }

    public struct ModelCacheProfile: Codable, Sendable, Equatable {
        public let model: String
        public let requests: Int
        public let promptTokens: Int
        public let cachedTokens: Int
        public let cacheShare: Double?
        public let outcomes: [String: Int]
        public let causes: [String: Int]

        enum CodingKeys: String, CodingKey {
            case model
            case requests
            case promptTokens = "prompt_tokens"
            case cachedTokens = "cached_tokens"
            case cacheShare = "cache_share"
            case outcomes
            case causes
        }

        public init(
            model: String,
            requests: Int,
            promptTokens: Int,
            cachedTokens: Int,
            cacheShare: Double?,
            outcomes: [String: Int],
            causes: [String: Int]
        ) {
            self.model = model
            self.requests = requests
            self.promptTokens = promptTokens
            self.cachedTokens = cachedTokens
            self.cacheShare = cacheShare
            self.outcomes = outcomes
            self.causes = causes
        }
    }

    public struct HourlyTrendEntry: Codable, Sendable, Equatable {
        public let hour: String
        public let responseTimeMs: Double
        public let tokensPerSecond: Double
        public let requests: Int
        public let errors: Int
        public let responseTimeChange: Double
        public let tpsChange: Double
        public let cacheShare: Double?
        public let draftAcceptance: Double?
        public let draftShare: Double?

        enum CodingKeys: String, CodingKey {
            case hour
            case responseTimeMs = "response_time_ms"
            case tokensPerSecond = "tokens_per_second"
            case requests
            case errors
            case responseTimeChange = "response_time_change"
            case tpsChange = "tps_change"
            case cacheShare = "cache_share"
            case draftAcceptance = "draft_acceptance"
            case draftShare = "draft_share"
        }

        public init(
            hour: String,
            responseTimeMs: Double,
            tokensPerSecond: Double,
            requests: Int,
            errors: Int,
            responseTimeChange: Double,
            tpsChange: Double,
            cacheShare: Double?,
            draftAcceptance: Double?,
            draftShare: Double?
        ) {
            self.hour = hour
            self.responseTimeMs = responseTimeMs
            self.tokensPerSecond = tokensPerSecond
            self.requests = requests
            self.errors = errors
            self.responseTimeChange = responseTimeChange
            self.tpsChange = tpsChange
            self.cacheShare = cacheShare
            self.draftAcceptance = draftAcceptance
            self.draftShare = draftShare
        }
    }

    public struct PerformanceProfileSummary: Codable, Sendable, Equatable {
        public let timeRange: String
        public let sinceUnixMs: Int64
        public let sampleCount: Int
        public let averageGenerationTPS: Double
        public let averagePromptTPS: Double
        public let averageTTFTMs: Double
        public let totalOutputTokens: Int
        public let totalPromptTokens: Int
        public let totalCachedTokens: Int
        public let byModel: [ModelPerformanceSummary]
        public let recent: [MessageStatsRecord]
        public let timingBreakdown: [TimingBreakdownEntry]
        public let bottlenecks: [ModelBottleneckEntry]
        public let cache: [ModelCacheProfile]
        public let trends: [HourlyTrendEntry]

        enum CodingKeys: String, CodingKey {
            case timeRange = "time_range"
            case sinceUnixMs = "since_unix_ms"
            case sampleCount = "sample_count"
            case averageGenerationTPS = "average_generation_tps"
            case averagePromptTPS = "average_prompt_tps"
            case averageTTFTMs = "average_ttft_ms"
            case totalOutputTokens = "total_output_tokens"
            case totalPromptTokens = "total_prompt_tokens"
            case totalCachedTokens = "total_cached_tokens"
            case byModel = "by_model"
            case recent
            case timingBreakdown = "timing_breakdown"
            case bottlenecks
            case cache
            case trends
        }

        public init(
            timeRange: String = "24h",
            sinceUnixMs: Int64,
            sampleCount: Int,
            averageGenerationTPS: Double,
            averagePromptTPS: Double,
            averageTTFTMs: Double,
            totalOutputTokens: Int,
            totalPromptTokens: Int,
            totalCachedTokens: Int,
            byModel: [ModelPerformanceSummary],
            recent: [MessageStatsRecord],
            timingBreakdown: [TimingBreakdownEntry] = [],
            bottlenecks: [ModelBottleneckEntry] = [],
            cache: [ModelCacheProfile] = [],
            trends: [HourlyTrendEntry] = []
        ) {
            self.timeRange = timeRange
            self.sinceUnixMs = sinceUnixMs
            self.sampleCount = sampleCount
            self.averageGenerationTPS = averageGenerationTPS
            self.averagePromptTPS = averagePromptTPS
            self.averageTTFTMs = averageTTFTMs
            self.totalOutputTokens = totalOutputTokens
            self.totalPromptTokens = totalPromptTokens
            self.totalCachedTokens = totalCachedTokens
            self.byModel = byModel
            self.recent = recent
            self.timingBreakdown = timingBreakdown
            self.bottlenecks = bottlenecks
            self.cache = cache
            self.trends = trends
        }
    }

    public static func validateProfileRange(_ range: String) throws -> String {
        let normalized = range.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard validProfileRanges.contains(normalized) else {
            throw SQLiteStoreError.invalidInput(
                "Invalid time_range '\(range)'. Must be one of: 1h, 6h, 24h, 7d"
            )
        }
        return normalized
    }

    public static func sinceUnixMs(forWindow window: String, nowUnixMs: Int64 = SQLiteStore.nowMillis()) -> Int64 {
        switch window.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1h", "hour":
            return max(0, nowUnixMs - 3_600_000)
        case "6h":
            return max(0, nowUnixMs - 6 * 3_600_000)
        case "24h", "1d", "day":
            return max(0, nowUnixMs - 86_400_000)
        case "7d", "week":
            return max(0, nowUnixMs - 7 * 86_400_000)
        case "30d", "month":
            return max(0, nowUnixMs - 30 * 86_400_000)
        default:
            return 0
        }
    }

    private static func roundDecimals(_ value: Double, places: Int) -> Double {
        let factor = pow(10.0, Double(places))
        return (value * factor).rounded() / factor
    }

    private static func utcHourKey(fromUnixMs ms: Int64) -> String {
        let seconds = TimeInterval(ms) / 1000.0
        let hourFloor = floor(seconds / 3600.0) * 3600.0
        let date = Date(timeIntervalSince1970: hourFloor)
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.timeZone = TimeZone(secondsFromGMT: 0)
        fmt.dateFormat = "yyyy-MM-dd'T'HH:00:00+00:00"
        return fmt.string(from: date)
    }

    private func buildProfileSummary(
        rangeLabel: String,
        sinceUnixMs: Int64,
        recentLimit: Int
    ) throws -> PerformanceProfileSummary {
        let allStats = try listMessageStats(sinceUnixMs: sinceUnixMs)
        let stats = Array(allStats.suffix(1000))
        let avgGenTPS = stats.isEmpty ? 0.0 : (stats.map(\.generationTPS).reduce(0, +) / Double(stats.count))
        let avgPromptTPS = stats.isEmpty ? 0.0 : (stats.map(\.promptTPS).reduce(0, +) / Double(stats.count))
        let ttfts = stats.compactMap(\.ttftMs)
        let avgTTFTMs = ttfts.isEmpty ? 0.0 : (ttfts.reduce(0, +) / Double(ttfts.count))
        let totalOutTokens = stats.map(\.outputTokens).reduce(0, +)
        let totalPromptTokens = stats.map(\.promptTokens).reduce(0, +)
        let totalCachedTokens = stats.map(\.cachedTokens).reduce(0, +)

        let grouped = Dictionary(grouping: stats, by: \.modelID)
        let byModel: [ModelPerformanceSummary] = grouped.map { modelID, rows in
            let mGen = rows.isEmpty ? 0.0 : (rows.map(\.generationTPS).reduce(0, +) / Double(rows.count))
            let mPrompt = rows.isEmpty ? 0.0 : (rows.map(\.promptTPS).reduce(0, +) / Double(rows.count))
            let mTTFTs = rows.compactMap(\.ttftMs)
            let mTTFT = mTTFTs.isEmpty ? 0.0 : (mTTFTs.reduce(0, +) / Double(mTTFTs.count))
            return ModelPerformanceSummary(
                modelID: modelID,
                sampleCount: rows.count,
                averageGenerationTPS: mGen,
                averagePromptTPS: mPrompt,
                averageTTFTMs: mTTFT,
                totalOutputTokens: rows.map(\.outputTokens).reduce(0, +),
                totalPromptTokens: rows.map(\.promptTokens).reduce(0, +),
                totalCachedTokens: rows.map(\.cachedTokens).reduce(0, +)
            )
        }
        .sorted { lhs, rhs in
            if lhs.sampleCount != rhs.sampleCount { return lhs.sampleCount > rhs.sampleCount }
            return lhs.modelID < rhs.modelID
        }

        // `timing_breakdown` matching `perf_collector._timing_breakdown`
        let timingBreakdown: [TimingBreakdownEntry]
        if stats.isEmpty {
            timingBreakdown = ["model_load", "token_generation", "other"].map {
                TimingBreakdownEntry(operation: $0, avgTimeMs: 0.0, count: 0, percentage: 0.0)
            }
        } else {
            let n = Double(stats.count)
            let sumLoad = stats.map { $0.modelLoadMs ?? 0.0 }.reduce(0, +)
            let sumGen = stats.map { $0.generationDurationMs ?? 0.0 }.reduce(0, +)
            let sumTotal = stats.map { row -> Double in
                row.requestDurationMs ?? ((row.modelLoadMs ?? 0.0) + (row.generationDurationMs ?? 0.0) + (row.queueWaitMs ?? 0.0))
            }.reduce(0, +)
            let sumOther = max(0.0, sumTotal - (sumLoad + sumGen))
            let grandTotal = sumTotal > 0 ? sumTotal : 1.0
            timingBreakdown = [
                TimingBreakdownEntry(
                    operation: "model_load",
                    avgTimeMs: Self.roundDecimals(sumLoad / n, places: 1),
                    count: stats.count,
                    percentage: Self.roundDecimals(sumLoad / grandTotal, places: 4)
                ),
                TimingBreakdownEntry(
                    operation: "token_generation",
                    avgTimeMs: Self.roundDecimals(sumGen / n, places: 1),
                    count: stats.count,
                    percentage: Self.roundDecimals(sumGen / grandTotal, places: 4)
                ),
                TimingBreakdownEntry(
                    operation: "other",
                    avgTimeMs: Self.roundDecimals(sumOther / n, places: 1),
                    count: stats.count,
                    percentage: Self.roundDecimals(sumOther / grandTotal, places: 4)
                ),
            ]
        }

        // `bottlenecks` matching `perf_collector._bottlenecks`
        let bottlenecks: [ModelBottleneckEntry] = grouped.map { modelID, rows in
            let n = Double(rows.count)
            let totalMs = rows.map { $0.requestDurationMs ?? ($0.generationDurationMs ?? 0.0) }.reduce(0, +)
            let qMs = rows.map { $0.queueWaitMs ?? 0.0 }.reduce(0, +)
            let lMs = rows.map { $0.modelLoadMs ?? 0.0 }.reduce(0, +)
            let gMs = rows.map { $0.generationDurationMs ?? 0.0 }.reduce(0, +)
            return ModelBottleneckEntry(
                model: modelID,
                avgTotalMs: Self.roundDecimals(totalMs / n, places: 1),
                breakdown: .init(
                    queueWait: Self.roundDecimals(qMs / n, places: 1),
                    modelLoad: Self.roundDecimals(lMs / n, places: 1),
                    tokenGeneration: Self.roundDecimals(gMs / n, places: 1)
                ),
                requestCount: rows.count
            )
        }.sorted { lhs, rhs in
            if lhs.avgTotalMs != rhs.avgTotalMs { return lhs.avgTotalMs > rhs.avgTotalMs }
            return lhs.model < rhs.model
        }

        // `cache` per-model matching `perf_collector._cache_by_model`
        var cacheRowsByModel: [String: [MessageStatsRecord]] = [:]
        for row in stats where row.success && row.cache != nil {
            cacheRowsByModel[row.modelID, default: []].append(row)
        }
        let cacheProfiles: [ModelCacheProfile] = cacheRowsByModel.map { modelID, reports in
            let promptSum = reports.compactMap { $0.cache?.promptTokens }.reduce(0, +)
            let cachedSum = reports.compactMap { $0.cache?.cachedTokens }.reduce(0, +)
            var outcomes: [String: Int] = [:]
            var causes: [String: Int] = [:]
            for r in reports {
                if let c = r.cache {
                    outcomes[c.outcome.rawValue, default: 0] += 1
                    if let cause = c.cause, !cause.isEmpty {
                        causes[cause, default: 0] += 1
                    }
                }
            }
            let share: Double? = promptSum > 0
                ? Self.roundDecimals(Double(cachedSum) / Double(promptSum), places: 3)
                : nil
            return ModelCacheProfile(
                model: modelID,
                requests: reports.count,
                promptTokens: promptSum,
                cachedTokens: cachedSum,
                cacheShare: share,
                outcomes: outcomes,
                causes: causes
            )
        }.sorted { lhs, rhs in
            if lhs.promptTokens != rhs.promptTokens { return lhs.promptTokens > rhs.promptTokens }
            return lhs.model < rhs.model
        }

        // `trends` matching `perf_collector._trends`, capped to the last 8 hourly buckets
        let buckets = Dictionary(grouping: stats) { Self.utcHourKey(fromUnixMs: $0.createdUnixMs) }
        let sortedHours = buckets.keys.sorted()
        var trendList: [HourlyTrendEntry] = []
        trendList.reserveCapacity(sortedHours.count)
        var prevRT: Double?
        var prevTPS: Double?
        for hour in sortedHours {
            let hourEvents = buckets[hour] ?? []
            let totalReqs = hourEvents.count
            let successes = hourEvents.filter(\.success)
            let okCount = successes.count
            let errors = totalReqs - okCount

            let avgRT: Double
            let avgTPS: Double
            if okCount > 0 {
                let sumRT = successes.map { $0.requestDurationMs ?? ($0.generationDurationMs ?? 0.0) }.reduce(0, +)
                let sumTPS = successes.map(\.generationTPS).reduce(0, +)
                avgRT = sumRT / Double(okCount)
                avgTPS = sumTPS / Double(okCount)
            } else {
                avgRT = 0.0
                avgTPS = 0.0
            }

            let rtChange = (prevRT != nil && prevRT! > 0)
                ? Self.roundDecimals((avgRT - prevRT!) / prevRT!, places: 4)
                : 0.0
            let tpsChange = (prevTPS != nil && prevTPS! > 0)
                ? Self.roundDecimals((avgTPS - prevTPS!) / prevTPS!, places: 4)
                : 0.0

            let cachePairs = successes.compactMap { row -> (cached: Int, prompt: Int)? in
                if let c = row.cache, c.promptTokens > 0 {
                    return (c.cachedTokens, c.promptTokens)
                }
                if row.promptTokens > 0 && row.cachedTokens > 0 {
                    return (row.cachedTokens, row.promptTokens)
                }
                return nil
            }
            let cachePromptTotal = cachePairs.map(\.prompt).reduce(0, +)
            let cacheShare: Double? = cachePromptTotal > 0
                ? Self.roundDecimals(Double(cachePairs.map(\.cached).reduce(0, +)) / Double(cachePromptTotal), places: 3)
                : nil

            let draftAccPairs = successes.compactMap { row -> (accepted: Int, drafted: Int)? in
                guard let spec = row.speculative, spec.draftedTokens > 0 else { return nil }
                return (spec.acceptedTokens, spec.draftedTokens)
            }
            let draftedTotal = draftAccPairs.map(\.drafted).reduce(0, +)
            let draftAcceptance: Double? = draftedTotal > 0
                ? Self.roundDecimals(Double(draftAccPairs.map(\.accepted).reduce(0, +)) / Double(draftedTotal), places: 3)
                : nil

            let draftSharePairs = successes.compactMap { row -> (accepted: Int, emitted: Int)? in
                guard let spec = row.speculative else { return nil }
                let emitted = row.specEmittedTokens ?? (spec.emittedTokens > 0 ? spec.emittedTokens : nil)
                guard let emitted, emitted > 0 else { return nil }
                return (spec.acceptedTokens, emitted)
            }
            let emittedTotal = draftSharePairs.map(\.emitted).reduce(0, +)
            let draftShare: Double? = emittedTotal > 0
                ? Self.roundDecimals(Double(draftSharePairs.map(\.accepted).reduce(0, +)) / Double(emittedTotal), places: 3)
                : nil

            trendList.append(
                HourlyTrendEntry(
                    hour: hour,
                    responseTimeMs: Self.roundDecimals(avgRT, places: 1),
                    tokensPerSecond: Self.roundDecimals(avgTPS, places: 1),
                    requests: totalReqs,
                    errors: errors,
                    responseTimeChange: rtChange,
                    tpsChange: tpsChange,
                    cacheShare: cacheShare,
                    draftAcceptance: draftAcceptance,
                    draftShare: draftShare
                )
            )
            prevRT = avgRT
            prevTPS = avgTPS
        }
        let lastEightTrends = Array(trendList.suffix(8))

        let recentSlice = Array(stats.suffix(max(1, recentLimit)).reversed())
        return PerformanceProfileSummary(
            timeRange: rangeLabel,
            sinceUnixMs: sinceUnixMs,
            sampleCount: stats.count,
            averageGenerationTPS: avgGenTPS,
            averagePromptTPS: avgPromptTPS,
            averageTTFTMs: avgTTFTMs,
            totalOutputTokens: totalOutTokens,
            totalPromptTokens: totalPromptTokens,
            totalCachedTokens: totalCachedTokens,
            byModel: byModel,
            recent: recentSlice,
            timingBreakdown: timingBreakdown,
            bottlenecks: bottlenecks,
            cache: cacheProfiles,
            trends: lastEightTrends
        )
    }

    /// Aggregates persisted `message_stats` since `sinceUnixMs`, including per-model breakdowns and recent turns.
    public func queryPerformanceProfile(sinceUnixMs: Int64 = 0, recentLimit: Int = 50) throws -> PerformanceProfileSummary {
        try buildProfileSummary(rangeLabel: "custom", sinceUnixMs: sinceUnixMs, recentLimit: recentLimit)
    }

    /// Strictly validates `range` (`"1h"`, `"6h"`, `"24h"`, `"7d"`) and builds the full performance profile.
    public func queryPerformanceProfile(
        range: String,
        nowUnixMs: Int64 = SQLiteStore.nowMillis(),
        recentLimit: Int = 50
    ) throws -> PerformanceProfileSummary {
        let validated = try Self.validateProfileRange(range)
        let sinceMs = Self.sinceUnixMs(forWindow: validated, nowUnixMs: nowUnixMs)
        return try buildProfileSummary(rangeLabel: validated, sinceUnixMs: sinceMs, recentLimit: recentLimit)
    }

    /// Bridged window overload strictly validating `"1h"`, `"6h"`, `"24h"`, `"7d"`.
    public func queryPerformanceProfile(
        window: String,
        nowUnixMs: Int64 = SQLiteStore.nowMillis(),
        recentLimit: Int = 50
    ) throws -> PerformanceProfileSummary {
        try queryPerformanceProfile(range: window, nowUnixMs: nowUnixMs, recentLimit: recentLimit)
    }

    // MARK: - Notebooks

    public func upsertNotebook(_ notebook: NotebookRecord) throws {
        lock.lock()
        do {
            let sql = """
            INSERT INTO notebooks (id, title, body, system_prompt, model_id, params_json, applied_preset_id, created_ms, updated_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                body = excluded.body,
                system_prompt = excluded.system_prompt,
                model_id = excluded.model_id,
                params_json = excluded.params_json,
                applied_preset_id = excluded.applied_preset_id,
                updated_ms = excluded.updated_ms;
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, notebook.id, -1, SQLITE_TRANSIENT_CORE)
            sqlite3_bind_text(stmt, 2, notebook.title, -1, SQLITE_TRANSIENT_CORE)
            sqlite3_bind_text(stmt, 3, notebook.body, -1, SQLITE_TRANSIENT_CORE)
            sqlite3_bind_text(stmt, 4, notebook.systemPrompt, -1, SQLITE_TRANSIENT_CORE)
            bindOptionalText(stmt, index: 5, value: notebook.modelID)
            bindOptionalEncodable(stmt, index: 6, value: notebook.params)
            bindOptionalText(stmt, index: 7, value: notebook.appliedPresetID)
            sqlite3_bind_int64(stmt, 8, notebook.createdUnixMs)
            sqlite3_bind_int64(stmt, 9, notebook.updatedUnixMs)
            guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        emit(.notebookUpserted(notebook.id))
    }

    public func getNotebook(id: String) throws -> NotebookRecord? {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        SELECT id, title, body, system_prompt, model_id, params_json, applied_preset_id, created_ms, updated_ms
        FROM notebooks
        WHERE id = ?
        LIMIT 1;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
        if sqlite3_step(stmt) == SQLITE_ROW {
            return readNotebookRow(stmt, decoder: JSONDecoder())
        }
        return nil
    }

    public func listNotebooks() throws -> [NotebookRecord] {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        SELECT id, title, body, system_prompt, model_id, params_json, applied_preset_id, created_ms, updated_ms
        FROM notebooks
        ORDER BY updated_ms DESC;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        let decoder = JSONDecoder()
        var out: [NotebookRecord] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append(readNotebookRow(stmt, decoder: decoder))
        }
        return out
    }

    @discardableResult
    public func deleteNotebook(id: String) throws -> Bool {
        lock.lock()
        let removed: Bool
        do {
            let sql = "DELETE FROM notebooks WHERE id = ?;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
            guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
            removed = sqlite3_changes(db) > 0
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        if removed {
            emit(.notebookDeleted(id))
        }
        return removed
    }

    // MARK: - Presets (No Default Seeding)

    public func upsertPreset(_ preset: PresetRecord) throws {
        lock.lock()
        do {
            try upsertPresetUnlocked(preset)
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        emit(.presetsChanged)
    }

    /// Creates a new preset, throwing `.conflict` (`HTTP 409`) if `name` already exists.
    public func createPreset(_ preset: PresetRecord) throws -> PresetRecord {
        let trimmedName = preset.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else {
            throw SQLiteStoreError.invalidInput("Preset 'name' must be a non-empty string")
        }
        let cleanPreset = PresetRecord(
            id: preset.id,
            name: trimmedName,
            description: preset.description,
            scope: preset.scope,
            systemPrompt: preset.systemPrompt,
            settings: preset.settings,
            updatedUnixMs: preset.updatedUnixMs,
            createdUnixMs: preset.createdUnixMs
        )
        lock.lock()
        do {
            if try getPresetByNameUnlocked(cleanPreset.name) != nil {
                throw SQLiteStoreError.conflict("Preset named '\(cleanPreset.name)' already exists")
            }
            try insertPresetStrictUnlocked(cleanPreset)
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        emit(.presetsChanged)
        return cleanPreset
    }

    public func updatePreset(id: String, _ updated: PresetRecord) throws -> PresetRecord {
        let trimmedName = updated.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else {
            throw SQLiteStoreError.invalidInput("Preset 'name' must be a non-empty string")
        }
        lock.lock()
        let finalPreset: PresetRecord
        do {
            guard let existingRow = try getPresetByIDUnlocked(id) else {
                throw SQLiteStoreError.notFound("Preset '\(id)' not found")
            }
            if let existingByName = try getPresetByNameUnlocked(trimmedName), existingByName.id != id {
                throw SQLiteStoreError.conflict("Preset named '\(trimmedName)' already exists")
            }
            finalPreset = PresetRecord(
                id: id,
                name: trimmedName,
                description: updated.description,
                scope: updated.scope,
                systemPrompt: updated.systemPrompt,
                settings: updated.settings,
                updatedUnixMs: updated.updatedUnixMs,
                createdUnixMs: existingRow.createdUnixMs
            )
            let data = try JSONEncoder().encode(finalPreset.settings)
            let json = String(data: data, encoding: .utf8) ?? "{}"
            let sql = """
            UPDATE presets
            SET name = ?, description = ?, scope = ?, settings_json = ?, system_prompt = ?, updated_ms = ?
            WHERE id = ?;
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, finalPreset.name, -1, SQLITE_TRANSIENT_CORE)
            sqlite3_bind_text(stmt, 2, finalPreset.description, -1, SQLITE_TRANSIENT_CORE)
            sqlite3_bind_text(stmt, 3, finalPreset.scope, -1, SQLITE_TRANSIENT_CORE)
            sqlite3_bind_text(stmt, 4, json, -1, SQLITE_TRANSIENT_CORE)
            bindOptionalText(stmt, index: 5, value: finalPreset.systemPrompt)
            sqlite3_bind_int64(stmt, 6, finalPreset.updatedUnixMs)
            sqlite3_bind_text(stmt, 7, id, -1, SQLITE_TRANSIENT_CORE)
            guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        emit(.presetsChanged)
        return finalPreset
    }

    public func getPreset(id: String) throws -> PresetRecord? {
        lock.lock()
        defer { lock.unlock() }
        return try getPresetByIDUnlocked(id)
    }

    @discardableResult
    public func deletePreset(id: String) throws -> Bool {
        lock.lock()
        let removed: Bool
        do {
            let sql = "DELETE FROM presets WHERE id = ?;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
            guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
            removed = sqlite3_changes(db) > 0
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        if removed {
            emit(.presetsChanged)
        }
        return removed
    }

    @discardableResult
    public func deletePreset(name: String) throws -> Bool {
        lock.lock()
        let removed: Bool
        do {
            let sql = "DELETE FROM presets WHERE name = ? COLLATE NOCASE;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, name, -1, SQLITE_TRANSIENT_CORE)
            guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
            removed = sqlite3_changes(db) > 0
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        if removed {
            emit(.presetsChanged)
        }
        return removed
    }

    public func listPresets() throws -> [PresetRecord] {
        lock.lock()
        defer { lock.unlock() }
        let sql = "SELECT id, name, description, scope, settings_json, system_prompt, created_ms, updated_ms FROM presets ORDER BY name ASC;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        var out: [PresetRecord] = []
        let decoder = JSONDecoder()
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append(readPresetRow(stmt, decoder: decoder))
        }
        return out
    }

    // MARK: - Maintenance (`POST /v1/data/clear`)

    /// Clears conversations, messages, message_stats, media_blobs, and notebooks while preserving presets (`api.py:308`).
    public func clearAllData() throws {
        lock.lock()
        do {
            try execute("""
            DELETE FROM message_stats;
            DELETE FROM messages;
            DELETE FROM conversations;
            DELETE FROM media_blobs;
            DELETE FROM notebooks;
            """)
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        emit(.dataCleared)
    }

    // MARK: - Private Row Helpers

    private func getPresetByIDUnlocked(_ id: String) throws -> PresetRecord? {
        let sql = "SELECT id, name, description, scope, settings_json, system_prompt, created_ms, updated_ms FROM presets WHERE id = ? LIMIT 1;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
        if sqlite3_step(stmt) == SQLITE_ROW {
            return readPresetRow(stmt, decoder: JSONDecoder())
        }
        return nil
    }

    private func getPresetByNameUnlocked(_ name: String) throws -> PresetRecord? {
        let sql = "SELECT id, name, description, scope, settings_json, system_prompt, created_ms, updated_ms FROM presets WHERE name = ? COLLATE NOCASE LIMIT 1;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, name, -1, SQLITE_TRANSIENT_CORE)
        if sqlite3_step(stmt) == SQLITE_ROW {
            return readPresetRow(stmt, decoder: JSONDecoder())
        }
        return nil
    }

    private func insertPresetStrictUnlocked(_ preset: PresetRecord) throws {
        let data = try JSONEncoder().encode(preset.settings)
        let json = String(data: data, encoding: .utf8) ?? "{}"
        let sql = """
        INSERT INTO presets (id, name, description, scope, settings_json, system_prompt, created_ms, updated_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, preset.id, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, preset.name, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 3, preset.description, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 4, preset.scope, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 5, json, -1, SQLITE_TRANSIENT_CORE)
        bindOptionalText(stmt, index: 6, value: preset.systemPrompt)
        sqlite3_bind_int64(stmt, 7, preset.createdUnixMs)
        sqlite3_bind_int64(stmt, 8, preset.updatedUnixMs)
        let rc = sqlite3_step(stmt)
        if rc == SQLITE_CONSTRAINT {
            throw SQLiteStoreError.conflict("Preset '\(preset.name)' already exists")
        }
        guard rc == SQLITE_DONE else { throw lastError() }
    }

    private func upsertPresetUnlocked(_ preset: PresetRecord) throws {
        let data = try JSONEncoder().encode(preset.settings)
        let json = String(data: data, encoding: .utf8) ?? "{}"
        let sql = """
        INSERT INTO presets (id, name, description, scope, settings_json, system_prompt, created_ms, updated_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description = excluded.description,
            scope = excluded.scope,
            settings_json = excluded.settings_json,
            system_prompt = excluded.system_prompt,
            updated_ms = excluded.updated_ms;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, preset.id, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, preset.name, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 3, preset.description, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 4, preset.scope, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 5, json, -1, SQLITE_TRANSIENT_CORE)
        bindOptionalText(stmt, index: 6, value: preset.systemPrompt)
        sqlite3_bind_int64(stmt, 7, preset.createdUnixMs)
        sqlite3_bind_int64(stmt, 8, preset.updatedUnixMs)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    private func readConversationRow(_ stmt: OpaquePointer?, decoder: JSONDecoder) -> ConversationRecord {
        let params = optionalColumnText(stmt, 4)
            .flatMap { try? decoder.decode(SamplerSettings.self, from: Data($0.utf8)) }
        return ConversationRecord(
            id: String(cString: sqlite3_column_text(stmt, 0)),
            title: String(cString: sqlite3_column_text(stmt, 1)),
            systemPrompt: String(cString: sqlite3_column_text(stmt, 2)),
            defaultModelID: optionalColumnText(stmt, 3),
            params: params,
            appliedPresetID: optionalColumnText(stmt, 5),
            createdUnixMs: sqlite3_column_int64(stmt, 6),
            updatedUnixMs: sqlite3_column_int64(stmt, 7)
        )
    }

    private func readMessageRow(_ stmt: OpaquePointer?, decoder: JSONDecoder) -> MessageRecord {
        let id = String(cString: sqlite3_column_text(stmt, 0))
        let cID = String(cString: sqlite3_column_text(stmt, 1))
        let pos = Int(sqlite3_column_int64(stmt, 2))
        let role = String(cString: sqlite3_column_text(stmt, 3))
        let content = String(cString: sqlite3_column_text(stmt, 4))
        let thinking = optionalColumnText(stmt, 5)
        let modelID = optionalColumnText(stmt, 6)
        let stopReason = optionalColumnText(stmt, 7)
        let msgTokenCount = optionalColumnInt(stmt, 8)
        let msgGenTPS = optionalColumnDouble(stmt, 9)
        let msgTTFTMs = optionalColumnDouble(stmt, 10)
        let msgCache = optionalColumnText(stmt, 11)
            .flatMap { try? decoder.decode(CacheReport.self, from: Data($0.utf8)) }
        let msgSpec = optionalColumnText(stmt, 12)
            .flatMap { try? decoder.decode(SpeculativeReport.self, from: Data($0.utf8)) }
        let placement = optionalColumnText(stmt, 13).flatMap(ComputePlacement.init(rawValue:))
        let blocks = optionalColumnText(stmt, 14)
            .flatMap { try? decoder.decode([MessagesRequest.MessageTurn.ContentPart].self, from: Data($0.utf8)) }
        let createdMs = sqlite3_column_int64(stmt, 15)
        let updatedMs = optionalColumnInt64(stmt, 16) ?? createdMs

        let hasJoinedStats = optionalColumnText(stmt, 17) != nil
        var promptTokens: Int?
        var promptTPS: Double?
        var peakMemoryGB: Double?
        var queueWaitMs: Double?
        var joinedOutputTokens: Int?
        var joinedGenTPS: Double?
        var joinedTTFT: Double?
        var joinedCache: CacheReport?
        var joinedSpec: SpeculativeReport?
        var statsObj: MessageRecordStats?

        if hasJoinedStats {
            let sPromptTok = optionalColumnInt(stmt, 18)
            let sOutTok = optionalColumnInt(stmt, 19)
            let sPromptTPS = optionalColumnDouble(stmt, 20)
            let sGenTPS = optionalColumnDouble(stmt, 21)
            let sTTFT = optionalColumnDouble(stmt, 22)
            let sPeakMem = optionalColumnDouble(stmt, 23)
            let sQueueWait = optionalColumnDouble(stmt, 24)
            let sThinkDur = optionalColumnDouble(stmt, 25)
            let sContentDur = optionalColumnDouble(stmt, 26)
            let sGenDur = optionalColumnDouble(stmt, 27)
            let sReqDur = optionalColumnDouble(stmt, 28)
            let sCache = optionalColumnText(stmt, 29)
                .flatMap { try? decoder.decode(CacheReport.self, from: Data($0.utf8)) }
            let sSpec = optionalColumnText(stmt, 30)
                .flatMap { try? decoder.decode(SpeculativeReport.self, from: Data($0.utf8)) }

            promptTokens = sPromptTok.flatMap { $0 > 0 ? $0 : nil }
            joinedOutputTokens = sOutTok.flatMap { $0 > 0 ? $0 : nil }
            promptTPS = sPromptTPS.flatMap { $0 > 0 ? $0 : nil }
            joinedGenTPS = sGenTPS.flatMap { $0 > 0 ? $0 : nil }
            joinedTTFT = sTTFT
            peakMemoryGB = sPeakMem.flatMap { $0 > 0 ? $0 : nil }
            queueWaitMs = sQueueWait.flatMap { $0 > 0 ? $0 : nil }
            joinedCache = sCache
            joinedSpec = sSpec

            statsObj = MessageRecordStats(
                outputTokens: joinedOutputTokens ?? msgTokenCount,
                generationTPS: joinedGenTPS ?? msgGenTPS,
                promptTPS: promptTPS,
                ttftMs: joinedTTFT ?? msgTTFTMs,
                peakMemoryGB: peakMemoryGB,
                queueWaitMs: queueWaitMs,
                thinkingDurationMs: sThinkDur,
                contentDurationMs: sContentDur,
                generationDurationMs: sGenDur,
                requestDurationMs: sReqDur,
                cache: joinedCache ?? msgCache,
                speculative: joinedSpec ?? msgSpec
            )
        }

        return MessageRecord(
            id: id,
            conversationID: cID,
            position: pos,
            role: role,
            content: content,
            thinking: thinking,
            modelID: modelID,
            stopReason: stopReason,
            tokenCount: msgTokenCount ?? joinedOutputTokens,
            generationTPS: msgGenTPS ?? joinedGenTPS,
            ttftMs: msgTTFTMs ?? joinedTTFT,
            cacheReport: msgCache ?? joinedCache,
            speculativeReport: msgSpec ?? joinedSpec,
            computePlacement: placement,
            contentBlocks: blocks,
            createdUnixMs: createdMs,
            updatedUnixMs: updatedMs,
            promptTokens: promptTokens,
            promptTPS: promptTPS,
            peakMemoryGB: peakMemoryGB,
            queueWaitMs: queueWaitMs,
            stats: statsObj
        )
    }

    private func readNotebookRow(_ stmt: OpaquePointer?, decoder: JSONDecoder) -> NotebookRecord {
        let params = optionalColumnText(stmt, 5)
            .flatMap { try? decoder.decode(SamplerSettings.self, from: Data($0.utf8)) }
        let createdMs = sqlite3_column_int64(stmt, 7)
        let updatedMs = sqlite3_column_int64(stmt, 8)
        return NotebookRecord(
            id: String(cString: sqlite3_column_text(stmt, 0)),
            title: String(cString: sqlite3_column_text(stmt, 1)),
            body: String(cString: sqlite3_column_text(stmt, 2)),
            systemPrompt: String(cString: sqlite3_column_text(stmt, 3)),
            modelID: optionalColumnText(stmt, 4),
            params: params,
            appliedPresetID: optionalColumnText(stmt, 6),
            updatedUnixMs: updatedMs,
            createdUnixMs: createdMs
        )
    }

    private func readPresetRow(_ stmt: OpaquePointer?, decoder: JSONDecoder) -> PresetRecord {
        let id = String(cString: sqlite3_column_text(stmt, 0))
        let name = String(cString: sqlite3_column_text(stmt, 1))
        let description = String(cString: sqlite3_column_text(stmt, 2))
        let scope = String(cString: sqlite3_column_text(stmt, 3))
        let settingsJSON = String(cString: sqlite3_column_text(stmt, 4))
        let settings = (try? decoder.decode(SamplerSettings.self, from: Data(settingsJSON.utf8))) ?? SamplerSettings()
        let sysPrompt = optionalColumnText(stmt, 5)
        let created = sqlite3_column_int64(stmt, 6)
        let updated = sqlite3_column_int64(stmt, 7)
        return PresetRecord(
            id: id,
            name: name,
            description: description,
            scope: scope,
            systemPrompt: sysPrompt,
            settings: settings,
            updatedUnixMs: updated,
            createdUnixMs: created
        )
    }

    private func bindOptionalText(_ stmt: OpaquePointer?, index: Int32, value: String?) {
        if let value {
            sqlite3_bind_text(stmt, index, value, -1, SQLITE_TRANSIENT_CORE)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func bindOptionalInt(_ stmt: OpaquePointer?, index: Int32, value: Int?) {
        if let value {
            sqlite3_bind_int64(stmt, index, Int64(value))
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func bindOptionalDouble(_ stmt: OpaquePointer?, index: Int32, value: Double?) {
        if let value {
            sqlite3_bind_double(stmt, index, value)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func bindOptionalEncodable<T: Encodable>(_ stmt: OpaquePointer?, index: Int32, value: T?) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        if let value,
           let data = try? encoder.encode(value),
           let json = String(data: data, encoding: .utf8) {
            sqlite3_bind_text(stmt, index, json, -1, SQLITE_TRANSIENT_CORE)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private func optionalColumnText(_ stmt: OpaquePointer?, _ col: Int32) -> String? {
        guard sqlite3_column_type(stmt, col) != SQLITE_NULL,
              let ptr = sqlite3_column_text(stmt, col)
        else { return nil }
        return String(cString: ptr)
    }

    private func optionalColumnInt(_ stmt: OpaquePointer?, _ col: Int32) -> Int? {
        guard sqlite3_column_type(stmt, col) != SQLITE_NULL else { return nil }
        return Int(sqlite3_column_int64(stmt, col))
    }

    private func optionalColumnInt64(_ stmt: OpaquePointer?, _ col: Int32) -> Int64? {
        guard sqlite3_column_type(stmt, col) != SQLITE_NULL else { return nil }
        return sqlite3_column_int64(stmt, col)
    }

    private func optionalColumnDouble(_ stmt: OpaquePointer?, _ col: Int32) -> Double? {
        guard sqlite3_column_type(stmt, col) != SQLITE_NULL else { return nil }
        return sqlite3_column_double(stmt, col)
    }

    private func execute(_ sql: String) throws {
        var errMsg: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &errMsg) != SQLITE_OK {
            let msg = errMsg.map { String(cString: $0) } ?? "SQLite error"
            sqlite3_free(errMsg)
            throw SQLiteStoreError.sqlFailed(msg)
        }
    }

    private func lastError() -> SQLiteStoreError {
        let msg = db.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "SQLite error"
        return .sqlFailed(msg)
    }
}
