import Foundation
import SQLite3

private let SQLITE_TRANSIENT_CORE = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

public enum SQLiteStoreError: LocalizedError, Sendable {
    case openFailed(String)
    case sqlFailed(String)
    case notFound(String)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let msg): return "Cannot open Heylook SQLite store: \(msg)"
        case .sqlFailed(let msg): return "SQLite error: \(msg)"
        case .notFound(let msg): return "Record not found: \(msg)"
        }
    }
}

public struct ConversationRecord: Codable, Identifiable, Sendable, Equatable, Hashable {
    public let id: String
    public var title: String
    public var systemPrompt: String
    public var defaultModelID: String?
    public let createdUnixMs: Int64
    public var updatedUnixMs: Int64

    public init(
        id: String = UUID().uuidString,
        title: String = "New Conversation",
        systemPrompt: String = "",
        defaultModelID: String? = nil,
        createdUnixMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000),
        updatedUnixMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)
    ) {
        self.id = id
        self.title = title
        self.systemPrompt = systemPrompt
        self.defaultModelID = defaultModelID
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
    public var stopReason: StopReason?
    public var cacheReport: CacheReport?
    public var speculativeReport: SpeculativeReport?
    public var generationTPS: Double?
    public var computePlacement: ComputePlacement?
    public let createdUnixMs: Int64

    public init(
        id: String = UUID().uuidString,
        conversationID: String,
        position: Int,
        role: String,
        content: String,
        thinking: String? = nil,
        modelID: String? = nil,
        stopReason: StopReason? = nil,
        cacheReport: CacheReport? = nil,
        speculativeReport: SpeculativeReport? = nil,
        generationTPS: Double? = nil,
        computePlacement: ComputePlacement? = nil,
        createdUnixMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)
    ) {
        self.id = id
        self.conversationID = conversationID
        self.position = position
        self.role = role
        self.content = content
        self.thinking = thinking
        self.modelID = modelID
        self.stopReason = stopReason
        self.cacheReport = cacheReport
        self.speculativeReport = speculativeReport
        self.generationTPS = generationTPS
        self.computePlacement = computePlacement
        self.createdUnixMs = createdUnixMs
    }
}

public struct NotebookRecord: Codable, Identifiable, Sendable, Equatable, Hashable {
    public let id: String
    public var title: String
    public var body: String
    public var modelID: String?
    public var updatedUnixMs: Int64

    public init(
        id: String = UUID().uuidString,
        title: String = "Untitled Notebook",
        body: String = "",
        modelID: String? = nil,
        updatedUnixMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)
    ) {
        self.id = id
        self.title = title
        self.body = body
        self.modelID = modelID
        self.updatedUnixMs = updatedUnixMs
    }
}

public struct PresetRecord: Codable, Identifiable, Sendable, Equatable {
    public let id: String
    public var name: String
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
        updatedUnixMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)
    ) {
        self.id = id
        self.name = name
        self.scope = scope
        self.settings = settings
        self.systemPrompt = systemPrompt
        self.updatedUnixMs = updatedUnixMs
    }
}

/// Zero-dependency SQLite3 (`WAL` mode) store for conversations, messages, notebooks, and presets.
/// Works identically on `macOS 27.2+` and `iOS 27.2+`.
public final class SQLiteStore: @unchecked Sendable {
    private var db: OpaquePointer?
    private let lock = NSLock()
    public let databaseURL: URL

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

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            system_prompt TEXT NOT NULL DEFAULT '',
            default_model_id TEXT,
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
            cache_json TEXT,
            speculative_json TEXT,
            gen_tps REAL,
            compute_placement TEXT,
            created_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS notebooks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            model_id TEXT,
            updated_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS presets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            scope TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            system_prompt TEXT,
            updated_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conv_pos ON messages(conversation_id, position);
        """)
        try seedDefaultPresetsIfEmpty()
    }

    deinit {
        if let db { sqlite3_close(db) }
    }

    // MARK: - Conversations & Messages

    public func upsertConversation(_ conv: ConversationRecord) throws {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        INSERT INTO conversations (id, title, system_prompt, default_model_id, created_ms, updated_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            system_prompt = excluded.system_prompt,
            default_model_id = excluded.default_model_id,
            updated_ms = excluded.updated_ms;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, conv.id, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, conv.title, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 3, conv.systemPrompt, -1, SQLITE_TRANSIENT_CORE)
        if let model = conv.defaultModelID {
            sqlite3_bind_text(stmt, 4, model, -1, SQLITE_TRANSIENT_CORE)
        } else {
            sqlite3_bind_null(stmt, 4)
        }
        sqlite3_bind_int64(stmt, 5, conv.createdUnixMs)
        sqlite3_bind_int64(stmt, 6, conv.updatedUnixMs)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    public func listConversations() throws -> [ConversationRecord] {
        lock.lock()
        defer { lock.unlock() }
        let sql = "SELECT id, title, system_prompt, default_model_id, created_ms, updated_ms FROM conversations ORDER BY updated_ms DESC;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        var out: [ConversationRecord] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append(readConversationRow(stmt))
        }
        return out
    }

    public func getConversation(id: String) throws -> ConversationRecord? {
        lock.lock()
        defer { lock.unlock() }
        let sql = "SELECT id, title, system_prompt, default_model_id, created_ms, updated_ms FROM conversations WHERE id = ? LIMIT 1;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
        if sqlite3_step(stmt) == SQLITE_ROW {
            return readConversationRow(stmt)
        }
        return nil
    }

    public func deleteConversation(id: String) throws {
        lock.lock()
        defer { lock.unlock() }
        let sql = "DELETE FROM conversations WHERE id = ?;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, SQLITE_TRANSIENT_CORE)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    @discardableResult
    public func appendMessage(_ message: MessageRecord) throws -> MessageRecord {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        INSERT INTO messages (
            id, conversation_id, position, role, content, thinking, model_id,
            stop_reason, cache_json, speculative_json, gen_tps, compute_placement, created_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            content = excluded.content,
            thinking = excluded.thinking,
            stop_reason = excluded.stop_reason,
            cache_json = excluded.cache_json,
            speculative_json = excluded.speculative_json,
            gen_tps = excluded.gen_tps,
            compute_placement = excluded.compute_placement;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, message.id, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, message.conversationID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_int64(stmt, 3, Int64(message.position))
        sqlite3_bind_text(stmt, 4, message.role, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 5, message.content, -1, SQLITE_TRANSIENT_CORE)
        bindOptionalText(stmt, index: 6, value: message.thinking)
        bindOptionalText(stmt, index: 7, value: message.modelID)
        bindOptionalText(stmt, index: 8, value: message.stopReason?.rawValue)
        if let cache = message.cacheReport,
           let data = try? JSONEncoder().encode(cache),
           let json = String(data: data, encoding: .utf8) {
            sqlite3_bind_text(stmt, 9, json, -1, SQLITE_TRANSIENT_CORE)
        } else {
            sqlite3_bind_null(stmt, 9)
        }
        if let spec = message.speculativeReport,
           let data = try? JSONEncoder().encode(spec),
           let json = String(data: data, encoding: .utf8) {
            sqlite3_bind_text(stmt, 10, json, -1, SQLITE_TRANSIENT_CORE)
        } else {
            sqlite3_bind_null(stmt, 10)
        }
        if let tps = message.generationTPS {
            sqlite3_bind_double(stmt, 11, tps)
        } else {
            sqlite3_bind_null(stmt, 11)
        }
        bindOptionalText(stmt, index: 12, value: message.computePlacement?.rawValue)
        sqlite3_bind_int64(stmt, 13, message.createdUnixMs)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }

        let touchConv = "UPDATE conversations SET updated_ms = ? WHERE id = ?;"
        var uStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, touchConv, -1, &uStmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(uStmt) }
            sqlite3_bind_int64(uStmt, 1, message.createdUnixMs)
            sqlite3_bind_text(uStmt, 2, message.conversationID, -1, SQLITE_TRANSIENT_CORE)
            _ = sqlite3_step(uStmt)
        }
        return message
    }

    public func listMessages(conversationID: String) throws -> [MessageRecord] {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        SELECT id, conversation_id, position, role, content, thinking, model_id,
               stop_reason, cache_json, speculative_json, gen_tps, compute_placement, created_ms
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
            let id = String(cString: sqlite3_column_text(stmt, 0))
            let cID = String(cString: sqlite3_column_text(stmt, 1))
            let pos = Int(sqlite3_column_int64(stmt, 2))
            let role = String(cString: sqlite3_column_text(stmt, 3))
            let content = String(cString: sqlite3_column_text(stmt, 4))
            let thinking = optionalColumnText(stmt, 5)
            let modelID = optionalColumnText(stmt, 6)
            let stopReason = optionalColumnText(stmt, 7).flatMap(StopReason.init(rawValue:))
            let cacheReport = optionalColumnText(stmt, 8)
                .flatMap { try? decoder.decode(CacheReport.self, from: Data($0.utf8)) }
            let speculativeReport = optionalColumnText(stmt, 9)
                .flatMap { try? decoder.decode(SpeculativeReport.self, from: Data($0.utf8)) }
            let genTPS: Double? = sqlite3_column_type(stmt, 10) == SQLITE_NULL ? nil : sqlite3_column_double(stmt, 10)
            let placement = optionalColumnText(stmt, 11).flatMap(ComputePlacement.init(rawValue:))
            let createdMs = sqlite3_column_int64(stmt, 12)
            out.append(
                MessageRecord(
                    id: id,
                    conversationID: cID,
                    position: pos,
                    role: role,
                    content: content,
                    thinking: thinking,
                    modelID: modelID,
                    stopReason: stopReason,
                    cacheReport: cacheReport,
                    speculativeReport: speculativeReport,
                    generationTPS: genTPS,
                    computePlacement: placement,
                    createdUnixMs: createdMs
                )
            )
        }
        return out
    }

    /// Truncates messages at or after `fromPosition` (`DELETE /v1/conversations/{id}/messages?from=N`),
    /// enabling power-user Message Edit -> **Save & Continue**.
    @discardableResult
    public func deleteMessages(conversationID: String, fromPosition: Int) throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        let sql = "DELETE FROM messages WHERE conversation_id = ? AND position >= ?;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, conversationID, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_int64(stmt, 2, Int64(fromPosition))
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
        return Int(sqlite3_changes(db))
    }

    // MARK: - Notebooks

    public func upsertNotebook(_ notebook: NotebookRecord) throws {
        lock.lock()
        defer { lock.unlock() }
        let sql = """
        INSERT INTO notebooks (id, title, body, model_id, updated_ms)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            body = excluded.body,
            model_id = excluded.model_id,
            updated_ms = excluded.updated_ms;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, notebook.id, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 2, notebook.title, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 3, notebook.body, -1, SQLITE_TRANSIENT_CORE)
        bindOptionalText(stmt, index: 4, value: notebook.modelID)
        sqlite3_bind_int64(stmt, 5, notebook.updatedUnixMs)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    public func listNotebooks() throws -> [NotebookRecord] {
        lock.lock()
        defer { lock.unlock() }
        let sql = "SELECT id, title, body, model_id, updated_ms FROM notebooks ORDER BY updated_ms DESC;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        var out: [NotebookRecord] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append(
                NotebookRecord(
                    id: String(cString: sqlite3_column_text(stmt, 0)),
                    title: String(cString: sqlite3_column_text(stmt, 1)),
                    body: String(cString: sqlite3_column_text(stmt, 2)),
                    modelID: optionalColumnText(stmt, 3),
                    updatedUnixMs: sqlite3_column_int64(stmt, 4)
                )
            )
        }
        return out
    }

    // MARK: - Presets

    public func upsertPreset(_ preset: PresetRecord) throws {
        lock.lock()
        defer { lock.unlock() }
        try upsertPresetUnlocked(preset)
    }

    public func listPresets() throws -> [PresetRecord] {
        lock.lock()
        defer { lock.unlock() }
        let sql = "SELECT id, name, scope, settings_json, system_prompt, updated_ms FROM presets ORDER BY name ASC;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        var out: [PresetRecord] = []
        let decoder = JSONDecoder()
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = String(cString: sqlite3_column_text(stmt, 0))
            let name = String(cString: sqlite3_column_text(stmt, 1))
            let scope = String(cString: sqlite3_column_text(stmt, 2))
            let settingsJSON = String(cString: sqlite3_column_text(stmt, 3))
            let settings = (try? decoder.decode(SamplerSettings.self, from: Data(settingsJSON.utf8))) ?? .builtinDefaults
            let sysPrompt = optionalColumnText(stmt, 4)
            let updated = sqlite3_column_int64(stmt, 5)
            out.append(
                PresetRecord(
                    id: id,
                    name: name,
                    scope: scope,
                    settings: settings,
                    systemPrompt: sysPrompt,
                    updatedUnixMs: updated
                )
            )
        }
        return out
    }

    private func seedDefaultPresetsIfEmpty() throws {
        lock.lock()
        defer { lock.unlock() }
        var stmt: OpaquePointer?
        var count = 0
        if sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM presets;", -1, &stmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(stmt) }
            if sqlite3_step(stmt) == SQLITE_ROW {
                count = Int(sqlite3_column_int64(stmt, 0))
            }
        }
        guard count == 0 else { return }

        let defaults: [PresetRecord] = [
            PresetRecord(
                id: "preset-balanced",
                name: "Balanced Daily Driver",
                settings: SamplerSettings(temperature: 0.7, topP: 0.95, topK: 40, maxTokens: 4096, thinking: true)
            ),
            PresetRecord(
                id: "preset-coding-deterministic",
                name: "Precise Coding",
                settings: SamplerSettings(temperature: 0.15, topP: 0.9, topK: 20, maxTokens: 8192, thinking: true)
            ),
            PresetRecord(
                id: "preset-fast-direct",
                name: "Fast Direct (Thinking Off)",
                settings: SamplerSettings(temperature: 0.5, topP: 0.9, topK: 40, maxTokens: 2048, thinking: false)
            ),
        ]
        for p in defaults {
            try upsertPresetUnlocked(p)
        }
    }

    private func upsertPresetUnlocked(_ preset: PresetRecord) throws {
        let data = try JSONEncoder().encode(preset.settings)
        let json = String(data: data, encoding: .utf8) ?? "{}"
        let sql = """
        INSERT INTO presets (id, name, scope, settings_json, system_prompt, updated_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
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
        sqlite3_bind_text(stmt, 3, preset.scope, -1, SQLITE_TRANSIENT_CORE)
        sqlite3_bind_text(stmt, 4, json, -1, SQLITE_TRANSIENT_CORE)
        bindOptionalText(stmt, index: 5, value: preset.systemPrompt)
        sqlite3_bind_int64(stmt, 6, preset.updatedUnixMs)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    private func readConversationRow(_ stmt: OpaquePointer?) -> ConversationRecord {
        ConversationRecord(
            id: String(cString: sqlite3_column_text(stmt, 0)),
            title: String(cString: sqlite3_column_text(stmt, 1)),
            systemPrompt: String(cString: sqlite3_column_text(stmt, 2)),
            defaultModelID: optionalColumnText(stmt, 3),
            createdUnixMs: sqlite3_column_int64(stmt, 4),
            updatedUnixMs: sqlite3_column_int64(stmt, 5)
        )
    }

    private func bindOptionalText(_ stmt: OpaquePointer?, index: Int32, value: String?) {
        if let value {
            sqlite3_bind_text(stmt, index, value, -1, SQLITE_TRANSIENT_CORE)
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
