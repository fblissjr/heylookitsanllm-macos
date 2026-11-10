import CryptoKit
import Foundation
import SQLite3

private let SQLITE_TRANSIENT_CORE = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

public enum SQLiteStoreError: LocalizedError, Sendable, Equatable {
    case openFailed(String)
    case sqlFailed(String)
    case notFound(String)
    case conflict(String)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let msg): return "Cannot open Heylook SQLite store: \(msg)"
        case .sqlFailed(let msg): return "SQLite error: \(msg)"
        case .notFound(let msg): return "Record not found: \(msg)"
        case .conflict(let msg): return "Conflict: \(msg)"
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
        createdUnixMs: Int64 = SQLiteStore.nowMillis()
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
    public let mediaID: String
    public let mediaType: String
    public let sha256: String
    public let byteSize: Int
    public let data: Data
    public let createdUnixMs: Int64

    public init(
        mediaID: String,
        mediaType: String,
        sha256: String,
        byteSize: Int,
        data: Data,
        createdUnixMs: Int64 = SQLiteStore.nowMillis()
    ) {
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
    public let thinkingDurationMs: Double?
    public let contentDurationMs: Double?
    public let generationDurationMs: Double?
    public let requestDurationMs: Double?
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
        createdUnixMs: Int64 = SQLiteStore.nowMillis()
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
        self.thinkingDurationMs = thinkingDurationMs
        self.contentDurationMs = contentDurationMs
        self.generationDurationMs = generationDurationMs
        self.requestDurationMs = requestDurationMs
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
    public var updatedUnixMs: Int64

    public init(
        id: String = UUID().uuidString,
        title: String = "Untitled Notebook",
        body: String = "",
        systemPrompt: String = "",
        modelID: String? = nil,
        params: SamplerSettings? = nil,
        appliedPresetID: String? = nil,
        updatedUnixMs: Int64 = SQLiteStore.nowMillis()
    ) {
        self.id = id
        self.title = title
        self.body = body
        self.systemPrompt = systemPrompt
        self.modelID = modelID
        self.params = params
        self.appliedPresetID = appliedPresetID
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
    public var updatedUnixMs: Int64

    public init(
        id: String = UUID().uuidString,
        name: String,
        scope: String = "sampler",
        settings: SamplerSettings,
        systemPrompt: String? = nil,
        description: String = "",
        updatedUnixMs: Int64 = SQLiteStore.nowMillis()
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.scope = scope
        self.settings = settings
        self.systemPrompt = systemPrompt
        self.updatedUnixMs = updatedUnixMs
    }

    public init(
        id: String = UUID().uuidString,
        name: String,
        description: String,
        scope: String = "sampler",
        systemPrompt: String? = nil,
        settings: SamplerSettings,
        updatedUnixMs: Int64 = SQLiteStore.nowMillis()
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.scope = scope
        self.settings = settings
        self.systemPrompt = systemPrompt
        self.updatedUnixMs = updatedUnixMs
    }
}

/// Zero-dependency SQLite3 (`WAL` mode) store for conversations, messages, media blobs,
/// message stats, notebooks, and presets. Works identically on `macOS 27.2+` and `iOS 27.2+`.
public final class SQLiteStore: @unchecked Sendable {
    private var db: OpaquePointer?
    private let lock = NSLock()
    private let observerLock = NSLock()
    private var mutationObservers: [UUID: AsyncStream<StoreMutationEvent>.Continuation] = [:]
    public let databaseURL: URL

    public static func nowMillis() -> Int64 {
        Int64(Date().timeIntervalSince1970 * 1000)
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
        try seedDefaultPresetsOnceUnlocked()
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

    // MARK: - Clean Table Recreation (No Migration Code)

    private func ensureCurrentSchemaUnlocked() throws {
        let requiredColumnsByTable: [(table: String, required: Set<String>)] = [
            ("store_meta", ["key", "value"]),
            ("conversations", ["id", "title", "system_prompt", "default_model_id", "params_json", "applied_preset_id", "created_ms", "updated_ms"]),
            ("messages", ["id", "conversation_id", "position", "role", "content", "thinking", "model_id", "stop_reason", "token_count", "gen_tps", "ttft_ms", "cache_json", "speculative_json", "compute_placement", "content_blocks_json", "created_ms"]),
            ("media_blobs", ["media_id", "media_type", "sha256", "byte_size", "data", "created_ms"]),
            ("message_stats", ["message_id", "conversation_id", "model_id", "prompt_tokens", "output_tokens", "thinking_tokens", "content_tokens", "cached_tokens", "prompt_tps", "generation_tps", "ttft_ms", "thinking_duration_ms", "content_duration_ms", "generation_duration_ms", "request_duration_ms", "created_ms"]),
            ("notebooks", ["id", "title", "body", "system_prompt", "model_id", "params_json", "applied_preset_id", "updated_ms"]),
            ("presets", ["id", "name", "description", "scope", "settings_json", "system_prompt", "updated_ms"]),
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
            UNIQUE(conversation_id, position)
        );

        CREATE TABLE IF NOT EXISTS media_blobs (
            media_id TEXT PRIMARY KEY,
            media_type TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            byte_size INTEGER NOT NULL,
            data BLOB NOT NULL,
            created_ms INTEGER NOT NULL
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
            thinking_duration_ms REAL,
            content_duration_ms REAL,
            generation_duration_ms REAL,
            request_duration_ms REAL,
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
            updated_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS presets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            description TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            system_prompt TEXT,
            updated_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conv_pos ON messages(conversation_id, position);
        CREATE INDEX IF NOT EXISTS idx_message_stats_created ON message_stats(created_ms);
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
            let sql = "DELETE FROM conversations WHERE id = ?;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
            guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
            removed = sqlite3_changes(db) > 0
            if removed {
                try gcOrphanMediaBlobsUnlocked()
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

    /// Clones a conversation (optionally truncating at `upToPosition` inclusive) into a new independent record.
    public func cloneConversation(
        id: String,
        upToPosition: Int? = nil,
        newTitle: String? = nil
    ) throws -> ConversationRecord {
        lock.lock()
        let cloned: ConversationRecord
        do {
            guard let source = try getConversationUnlocked(id: id) else {
                throw SQLiteStoreError.notFound("Conversation '\(id)' not found")
            }
            let now = Self.nowMillis()
            cloned = ConversationRecord(
                id: UUID().uuidString,
                title: newTitle ?? "\(source.title) (Copy)",
                systemPrompt: source.systemPrompt,
                defaultModelID: source.defaultModelID,
                params: source.params,
                appliedPresetID: source.appliedPresetID,
                createdUnixMs: now,
                updatedUnixMs: now
            )
            try upsertConversationUnlocked(cloned)
            let sourceMsgs = try listMessagesUnlocked(conversationID: id, hydrateMedia: false)
            var nextPosition = 0
            for msg in sourceMsgs {
                if let maxPos = upToPosition, msg.position > maxPos {
                    continue
                }
                let copyMsg = MessageRecord(
                    id: UUID().uuidString,
                    conversationID: cloned.id,
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
                    contentBlocks: msg.contentBlocks,
                    createdUnixMs: now + Int64(nextPosition)
                )
                _ = try appendMessageUnlocked(copyMsg)
                nextPosition += 1
            }
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        emit(.conversationUpserted(cloned.id))
        emit(.messagesChanged(cloned.id))
        return cloned
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
            stored = try appendMessageUnlocked(message)
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
            normalized.contentBlocks = try externalizeMediaInBlocksUnlocked(blocks)
        }

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
            compute_placement, content_blocks_json, created_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            content_blocks_json = COALESCE(excluded.content_blocks_json, messages.content_blocks_json);
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
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }

        let touchConv = "UPDATE conversations SET updated_ms = ? WHERE id = ?;"
        var uStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, touchConv, -1, &uStmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(uStmt) }
            sqlite3_bind_int64(uStmt, 1, Self.nowMillis())
            sqlite3_bind_text(uStmt, 2, normalized.conversationID, -1, SQLITE_TRANSIENT_CORE)
            _ = sqlite3_step(uStmt)
        }
        return normalized
    }

    public func getMessage(id: String, hydrateMedia: Bool = false) throws -> MessageRecord? {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        SELECT id, conversation_id, position, role, content, thinking, model_id,
               stop_reason, token_count, gen_tps, ttft_ms, cache_json, speculative_json,
               compute_placement, content_blocks_json, created_ms
        FROM messages
        WHERE id = ?
        LIMIT 1;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
        if sqlite3_step(stmt) == SQLITE_ROW {
            var msg = readMessageRow(stmt, decoder: JSONDecoder())
            if hydrateMedia, let blocks = msg.contentBlocks {
                msg.contentBlocks = hydrateMediaInBlocksUnlocked(blocks)
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
        SELECT id, conversation_id, position, role, content, thinking, model_id,
               stop_reason, token_count, gen_tps, ttft_ms, cache_json, speculative_json,
               compute_placement, content_blocks_json, created_ms
        FROM messages
        WHERE conversation_id = ?
        ORDER BY position ASC, created_ms ASC;
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
                row.contentBlocks = hydrateMediaInBlocksUnlocked(blocks)
            }
            out.append(row)
        }
        return out
    }

    /// Truncates messages at or after `fromPosition` (`DELETE /v1/conversations/{id}/messages?from=N`).
    @discardableResult
    public func deleteMessages(conversationID: String, fromPosition: Int) throws -> Int {
        lock.lock()
        let removed: Int
        do {
            let sql = "DELETE FROM messages WHERE conversation_id = ? AND position >= ?;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
            sqlite3_bind_int64(stmt, 2, Int64(fromPosition))
            guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
            removed = Int(sqlite3_changes(db))
            if removed > 0 {
                try gcOrphanMediaBlobsUnlocked()
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

    /// Deletes a single message row and re-compacts subsequent message positions contiguously.
    @discardableResult
    public func deleteMessage(conversationID: String, messageID: String) throws -> Bool {
        lock.lock()
        let didDelete: Bool
        do {
            let all = try listMessagesUnlocked(conversationID: conversationID, hydrateMedia: false)
            guard let target = all.first(where: { $0.id == messageID }) else {
                lock.unlock()
                return false
            }
            let delSQL = "DELETE FROM messages WHERE id = ? AND conversation_id = ?;"
            var delStmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, delSQL, -1, &delStmt, nil) == SQLITE_OK else { throw lastError() }
            sqlite3_bind_text(delStmt, 1, messageID, -1, SQLITE_TRANSIENT_CORE)
            sqlite3_bind_text(delStmt, 2, conversationID, -1, SQLITE_TRANSIENT_CORE)
            guard sqlite3_step(delStmt) == SQLITE_DONE else {
                sqlite3_finalize(delStmt)
                throw lastError()
            }
            sqlite3_finalize(delStmt)
            didDelete = sqlite3_changes(db) > 0

            // Shift trailing positions down one by one in ascending order to satisfy UNIQUE(conversation_id, position)
            let trailing = all.filter { $0.position > target.position }
            for item in trailing {
                let shiftSQL = "UPDATE messages SET position = ? WHERE id = ?;"
                var shiftStmt: OpaquePointer?
                guard sqlite3_prepare_v2(db, shiftSQL, -1, &shiftStmt, nil) == SQLITE_OK else { throw lastError() }
                sqlite3_bind_int64(shiftStmt, 1, Int64(item.position - 1))
                sqlite3_bind_text(shiftStmt, 2, item.id, -1, SQLITE_TRANSIENT_CORE)
                guard sqlite3_step(shiftStmt) == SQLITE_DONE else {
                    sqlite3_finalize(shiftStmt)
                    throw lastError()
                }
                sqlite3_finalize(shiftStmt)
            }
            try gcOrphanMediaBlobsUnlocked()
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

    // MARK: - Content-Addressed Media Blobs

    @discardableResult
    public func storeMediaBlob(data: Data, mediaType: String) throws -> String {
        lock.lock()
        defer { lock.unlock() }
        return try storeMediaBlobUnlocked(data: data, mediaType: mediaType)
    }

    private func storeMediaBlobUnlocked(data: Data, mediaType: String) throws -> String {
        let digest = SHA256.hash(data: data)
        let shaHex = digest.map { String(format: "%02x", $0) }.joined()
        let mediaID = "media_\(shaHex.prefix(24))"
        let sql = """
        INSERT INTO media_blobs (media_id, media_type, sha256, byte_size, data, created_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(media_id) DO NOTHING;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, mediaID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, mediaType, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 3, shaHex, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_int64(stmt, 4, Int64(data.count))
        _ = data.withUnsafeBytes { rawPtr in
            sqlite3_bind_blob(stmt, 5, rawPtr.baseAddress, Int32(data.count), SQLITE_TRANSIENT_CORE)
        }
        sqlite3_bind_int64(stmt, 6, Self.nowMillis())
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
        return mediaID
    }

    public func getMediaBlob(id: String) throws -> MediaBlobRecord? {
        lock.lock()
        defer { lock.unlock() }
        return getMediaBlobUnlocked(id: id)
    }

    private func getMediaBlobUnlocked(id: String) -> MediaBlobRecord? {
        let sql = "SELECT media_id, media_type, sha256, byte_size, data, created_ms FROM media_blobs WHERE media_id = ? LIMIT 1;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        let mediaID = String(cString: sqlite3_column_text(stmt, 0))
        let mediaType = String(cString: sqlite3_column_text(stmt, 1))
        let sha = String(cString: sqlite3_column_text(stmt, 2))
        let byteSize = Int(sqlite3_column_int64(stmt, 3))
        let blobBytes = Int(sqlite3_column_bytes(stmt, 4))
        let blobData: Data
        if blobBytes > 0, let raw = sqlite3_column_blob(stmt, 4) {
            blobData = Data(bytes: raw, count: blobBytes)
        } else {
            blobData = Data()
        }
        let createdMs = sqlite3_column_int64(stmt, 5)
        return MediaBlobRecord(
            mediaID: mediaID,
            mediaType: mediaType,
            sha256: sha,
            byteSize: byteSize,
            data: blobData,
            createdUnixMs: createdMs
        )
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
        _ blocks: [MessagesRequest.MessageTurn.ContentPart]
    ) throws -> [MessagesRequest.MessageTurn.ContentPart] {
        var out: [MessagesRequest.MessageTurn.ContentPart] = []
        out.reserveCapacity(blocks.count)
        for part in blocks {
            switch part {
            case .media(var att):
                if !att.dataBase64.isEmpty, let decoded = Data(base64Encoded: att.dataBase64), !decoded.isEmpty {
                    let mediaID = try storeMediaBlobUnlocked(data: decoded, mediaType: att.mediaType)
                    att = MessagesRequest.MediaAttachment(
                        id: att.id,
                        kind: att.kind,
                        mediaType: att.mediaType,
                        dataBase64: "",
                        filename: att.filename,
                        mediaID: mediaID
                    )
                }
                out.append(.media(att))
            default:
                out.append(part)
            }
        }
        return out
    }

    private func hydrateMediaInBlocksUnlocked(
        _ blocks: [MessagesRequest.MessageTurn.ContentPart]
    ) -> [MessagesRequest.MessageTurn.ContentPart] {
        blocks.map { part in
            switch part {
            case .media(let att):
                if att.dataBase64.isEmpty, let mID = att.mediaID, let blob = getMediaBlobUnlocked(id: mID) {
                    return .media(
                        MessagesRequest.MediaAttachment(
                            id: att.id,
                            kind: att.kind,
                            mediaType: blob.mediaType,
                            dataBase64: blob.data.base64EncodedString(),
                            filename: att.filename,
                            mediaID: mID
                        )
                    )
                }
                return part
            default:
                return part
            }
        }
    }

    /// Garbage-collects unreferenced `media_blobs` rows after message/conversation deletion.
    private func gcOrphanMediaBlobsUnlocked() throws {
        var referenced = Set<String>()
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT content_blocks_json FROM messages WHERE content_blocks_json IS NOT NULL;", -1, &stmt, nil) == SQLITE_OK {
            let decoder = JSONDecoder()
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let json = optionalColumnText(stmt, 0),
                   let blocks = try? decoder.decode([MessagesRequest.MessageTurn.ContentPart].self, from: Data(json.utf8)) {
                    for b in blocks {
                        if case .media(let att) = b, let mID = att.mediaID {
                            referenced.insert(mID)
                        }
                    }
                }
            }
            sqlite3_finalize(stmt)
        }
        var existingIDs: [String] = []
        var bStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT media_id FROM media_blobs;", -1, &bStmt, nil) == SQLITE_OK {
            while sqlite3_step(bStmt) == SQLITE_ROW {
                if let id = optionalColumnText(bStmt, 0) {
                    existingIDs.append(id)
                }
            }
            sqlite3_finalize(bStmt)
        }
        for id in existingIDs where !referenced.contains(id) {
            var dStmt: OpaquePointer?
            if sqlite3_prepare_v2(db, "DELETE FROM media_blobs WHERE media_id = ?;", -1, &dStmt, nil) == SQLITE_OK {
                sqlite3_bind_text(dStmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
                _ = sqlite3_step(dStmt)
                sqlite3_finalize(dStmt)
            }
        }
    }

    // MARK: - Message Stats & Performance Profile

    public func upsertMessageStats(_ stats: MessageStatsRecord) throws {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        INSERT INTO message_stats (
            message_id, conversation_id, model_id, prompt_tokens, output_tokens,
            thinking_tokens, content_tokens, cached_tokens, prompt_tps, generation_tps,
            ttft_ms, thinking_duration_ms, content_duration_ms, generation_duration_ms,
            request_duration_ms, created_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            prompt_tokens = excluded.prompt_tokens,
            output_tokens = excluded.output_tokens,
            thinking_tokens = excluded.thinking_tokens,
            content_tokens = excluded.content_tokens,
            cached_tokens = excluded.cached_tokens,
            prompt_tps = excluded.prompt_tps,
            generation_tps = excluded.generation_tps,
            ttft_ms = excluded.ttft_ms,
            thinking_duration_ms = excluded.thinking_duration_ms,
            content_duration_ms = excluded.content_duration_ms,
            generation_duration_ms = excluded.generation_duration_ms,
            request_duration_ms = excluded.request_duration_ms;
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
        bindOptionalDouble(stmt, index: 12, value: stats.thinkingDurationMs)
        bindOptionalDouble(stmt, index: 13, value: stats.contentDurationMs)
        bindOptionalDouble(stmt, index: 14, value: stats.generationDurationMs)
        bindOptionalDouble(stmt, index: 15, value: stats.requestDurationMs)
        sqlite3_bind_int64(stmt, 16, stats.createdUnixMs)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    public func listMessageStats(sinceUnixMs: Int64) throws -> [MessageStatsRecord] {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        SELECT message_id, conversation_id, model_id, prompt_tokens, output_tokens,
               thinking_tokens, content_tokens, cached_tokens, prompt_tps, generation_tps,
               ttft_ms, thinking_duration_ms, content_duration_ms, generation_duration_ms,
               request_duration_ms, created_ms
        FROM message_stats
        WHERE created_ms >= ?
        ORDER BY created_ms ASC;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int64(stmt, 1, sinceUnixMs)
        var out: [MessageStatsRecord] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
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
                    thinkingDurationMs: optionalColumnDouble(stmt, 11),
                    contentDurationMs: optionalColumnDouble(stmt, 12),
                    generationDurationMs: optionalColumnDouble(stmt, 13),
                    requestDurationMs: optionalColumnDouble(stmt, 14),
                    createdUnixMs: sqlite3_column_int64(stmt, 15)
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

    public struct PerformanceProfileSummary: Codable, Sendable, Equatable {
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

        enum CodingKeys: String, CodingKey {
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
        }

        public init(
            sinceUnixMs: Int64,
            sampleCount: Int,
            averageGenerationTPS: Double,
            averagePromptTPS: Double,
            averageTTFTMs: Double,
            totalOutputTokens: Int,
            totalPromptTokens: Int,
            totalCachedTokens: Int,
            byModel: [ModelPerformanceSummary],
            recent: [MessageStatsRecord]
        ) {
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
        }
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

    /// Aggregates persisted `message_stats` since `sinceUnixMs`, including per-model breakdowns and recent turns.
    public func queryPerformanceProfile(sinceUnixMs: Int64 = 0, recentLimit: Int = 50) throws -> PerformanceProfileSummary {
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

        let recentSlice = Array(stats.suffix(max(1, recentLimit)).reversed())
        return PerformanceProfileSummary(
            sinceUnixMs: sinceUnixMs,
            sampleCount: stats.count,
            averageGenerationTPS: avgGenTPS,
            averagePromptTPS: avgPromptTPS,
            averageTTFTMs: avgTTFTMs,
            totalOutputTokens: totalOutTokens,
            totalPromptTokens: totalPromptTokens,
            totalCachedTokens: totalCachedTokens,
            byModel: byModel,
            recent: recentSlice
        )
    }

    /// Convenience window overload accepting `"1h"`, `"6h"`, `"24h"`, `"7d"`, `"30d"`, or `"all"`.
    public func queryPerformanceProfile(
        window: String,
        nowUnixMs: Int64 = SQLiteStore.nowMillis(),
        recentLimit: Int = 50
    ) throws -> PerformanceProfileSummary {
        let sinceMs = Self.sinceUnixMs(forWindow: window, nowUnixMs: nowUnixMs)
        return try queryPerformanceProfile(sinceUnixMs: sinceMs, recentLimit: recentLimit)
    }

    // MARK: - Notebooks

    public func upsertNotebook(_ notebook: NotebookRecord) throws {
        lock.lock()
        do {
            let sql = """
            INSERT INTO notebooks (id, title, body, system_prompt, model_id, params_json, applied_preset_id, updated_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            sqlite3_bind_int64(stmt, 8, notebook.updatedUnixMs)
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
        SELECT id, title, body, system_prompt, model_id, params_json, applied_preset_id, updated_ms
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
        SELECT id, title, body, system_prompt, model_id, params_json, applied_preset_id, updated_ms
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

    // MARK: - Presets

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
        lock.lock()
        do {
            if try getPresetByNameUnlocked(preset.name) != nil {
                throw SQLiteStoreError.conflict("Preset named '\(preset.name)' already exists")
            }
            try insertPresetStrictUnlocked(preset)
            lock.unlock()
        } catch {
            lock.unlock()
            throw error
        }
        emit(.presetsChanged)
        return preset
    }

    public func updatePreset(id: String, _ updated: PresetRecord) throws -> PresetRecord {
        lock.lock()
        let finalPreset: PresetRecord
        do {
            guard try getPresetByIDUnlocked(id) != nil else {
                throw SQLiteStoreError.notFound("Preset '\(id)' not found")
            }
            if let existingByName = try getPresetByNameUnlocked(updated.name), existingByName.id != id {
                throw SQLiteStoreError.conflict("Preset named '\(updated.name)' already exists")
            }
            finalPreset = PresetRecord(
                id: id,
                name: updated.name,
                description: updated.description,
                scope: updated.scope,
                systemPrompt: updated.systemPrompt,
                settings: updated.settings,
                updatedUnixMs: updated.updatedUnixMs
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
        let sql = "SELECT id, name, description, scope, settings_json, system_prompt, updated_ms FROM presets ORDER BY name ASC;"
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

    // MARK: - Private Preset Seeding & Helpers

    private func seedDefaultPresetsOnceUnlocked() throws {
        var metaStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT value FROM store_meta WHERE key = 'presets_seeded' LIMIT 1;", -1, &metaStmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(metaStmt) }
            if sqlite3_step(metaStmt) == SQLITE_ROW {
                return
            }
        }

        var count = 0
        var cStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM presets;", -1, &cStmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(cStmt) }
            if sqlite3_step(cStmt) == SQLITE_ROW {
                count = Int(sqlite3_column_int64(cStmt, 0))
            }
        }

        if count == 0 {
            let defaults: [PresetRecord] = [
                PresetRecord(
                    id: "preset-balanced",
                    name: "Balanced Daily Driver",
                    description: "General coding & reasoning defaults",
                    settings: SamplerSettings(temperature: 0.7, topP: 0.95, topK: 40, maxTokens: 4096, thinking: true)
                ),
                PresetRecord(
                    id: "preset-coding-deterministic",
                    name: "Precise Coding",
                    description: "Low-variance generation for code & patches",
                    settings: SamplerSettings(temperature: 0.15, topP: 0.9, topK: 20, maxTokens: 8192, thinking: true)
                ),
                PresetRecord(
                    id: "preset-fast-direct",
                    name: "Fast Direct (Thinking Off)",
                    description: "Direct replies with reasoning disabled",
                    settings: SamplerSettings(temperature: 0.5, topP: 0.9, topK: 40, maxTokens: 2048, thinking: false)
                ),
            ]
            for p in defaults {
                try upsertPresetUnlocked(p)
            }
        }

        try execute("INSERT OR REPLACE INTO store_meta (key, value) VALUES ('presets_seeded', '1');")
    }

    private func getPresetByIDUnlocked(_ id: String) throws -> PresetRecord? {
        let sql = "SELECT id, name, description, scope, settings_json, system_prompt, updated_ms FROM presets WHERE id = ? LIMIT 1;"
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
        let sql = "SELECT id, name, description, scope, settings_json, system_prompt, updated_ms FROM presets WHERE name = ? COLLATE NOCASE LIMIT 1;"
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
        INSERT INTO presets (id, name, description, scope, settings_json, system_prompt, updated_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?);
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
        sqlite3_bind_int64(stmt, 7, preset.updatedUnixMs)
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
        INSERT INTO presets (id, name, description, scope, settings_json, system_prompt, updated_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
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
        sqlite3_bind_int64(stmt, 7, preset.updatedUnixMs)
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
        let tokenCount = optionalColumnInt(stmt, 8)
        let genTPS = optionalColumnDouble(stmt, 9)
        let ttftMs = optionalColumnDouble(stmt, 10)
        let cacheReport = optionalColumnText(stmt, 11)
            .flatMap { try? decoder.decode(CacheReport.self, from: Data($0.utf8)) }
        let speculativeReport = optionalColumnText(stmt, 12)
            .flatMap { try? decoder.decode(SpeculativeReport.self, from: Data($0.utf8)) }
        let placement = optionalColumnText(stmt, 13).flatMap(ComputePlacement.init(rawValue:))
        let blocks = optionalColumnText(stmt, 14)
            .flatMap { try? decoder.decode([MessagesRequest.MessageTurn.ContentPart].self, from: Data($0.utf8)) }
        let createdMs = sqlite3_column_int64(stmt, 15)
        return MessageRecord(
            id: id,
            conversationID: cID,
            position: pos,
            role: role,
            content: content,
            thinking: thinking,
            modelID: modelID,
            stopReason: stopReason,
            tokenCount: tokenCount,
            generationTPS: genTPS,
            ttftMs: ttftMs,
            cacheReport: cacheReport,
            speculativeReport: speculativeReport,
            computePlacement: placement,
            contentBlocks: blocks,
            createdUnixMs: createdMs
        )
    }

    private func readNotebookRow(_ stmt: OpaquePointer?, decoder: JSONDecoder) -> NotebookRecord {
        let params = optionalColumnText(stmt, 5)
            .flatMap { try? decoder.decode(SamplerSettings.self, from: Data($0.utf8)) }
        return NotebookRecord(
            id: String(cString: sqlite3_column_text(stmt, 0)),
            title: String(cString: sqlite3_column_text(stmt, 1)),
            body: String(cString: sqlite3_column_text(stmt, 2)),
            systemPrompt: String(cString: sqlite3_column_text(stmt, 3)),
            modelID: optionalColumnText(stmt, 4),
            params: params,
            appliedPresetID: optionalColumnText(stmt, 6),
            updatedUnixMs: sqlite3_column_int64(stmt, 7)
        )
    }

    private func readPresetRow(_ stmt: OpaquePointer?, decoder: JSONDecoder) -> PresetRecord {
        let id = String(cString: sqlite3_column_text(stmt, 0))
        let name = String(cString: sqlite3_column_text(stmt, 1))
        let description = String(cString: sqlite3_column_text(stmt, 2))
        let scope = String(cString: sqlite3_column_text(stmt, 3))
        let settingsJSON = String(cString: sqlite3_column_text(stmt, 4))
        let settings = (try? decoder.decode(SamplerSettings.self, from: Data(settingsJSON.utf8))) ?? .builtinDefaults
        let sysPrompt = optionalColumnText(stmt, 5)
        let updated = sqlite3_column_int64(stmt, 6)
        return PresetRecord(
            id: id,
            name: name,
            description: description,
            scope: scope,
            systemPrompt: sysPrompt,
            settings: settings,
            updatedUnixMs: updated
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
        if let value,
           let data = try? JSONEncoder().encode(value),
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
