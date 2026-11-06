import Accelerate
import Foundation
import SQLite3

private let SQLITE_TRANSIENT_VEC = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

public enum VectorStoreError: LocalizedError, Sendable {
    case openFailed(String)
    case sqlFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .openFailed(msg): return "Cannot open vector SQLite store: \(msg)"
        case let .sqlFailed(msg): return "Vector SQLite error: \(msg)"
        }
    }
}

/// Persistent SQLite3 (`WAL`) + Accelerate (`vDSP`) Matryoshka vector store.
public final class VectorStore: @unchecked Sendable {
    private var db: OpaquePointer?
    private let lock = NSLock()

    public init(databaseURL: URL) throws {
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var handle: OpaquePointer?
        if sqlite3_open(databaseURL.path, &handle) != SQLITE_OK {
            let msg = handle.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            sqlite3_close(handle)
            throw VectorStoreError.openFailed(msg)
        }
        self.db = handle
        try execute("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            collection_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            byte_count INTEGER NOT NULL,
            modified_ms INTEGER NOT NULL,
            window_sampling TEXT NOT NULL DEFAULT '',
            model_id TEXT NOT NULL DEFAULT '',
            attributes_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
            span_json TEXT NOT NULL,
            vector_blob BLOB NOT NULL,
            thumbnail_jpeg BLOB,
            preview_text TEXT
        );

        CREATE TABLE IF NOT EXISTS watched_folders (
            path TEXT PRIMARY KEY,
            collection_id TEXT NOT NULL,
            window_sampling TEXT NOT NULL DEFAULT 'fps',
            added_ms INTEGER NOT NULL,
            last_indexed_ms INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_artifact ON chunks(artifact_id);
        CREATE INDEX IF NOT EXISTS idx_artifacts_collection ON artifacts(collection_id);
        """)
    }

    deinit {
        if let db { sqlite3_close(db) }
    }

    public func isUpToDate(
        artifactID: String,
        fingerprint: ContentFingerprint,
        windowSampling: VideoWindowSampling = .off,
        modelID: String = ""
    ) throws -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let sql = "SELECT byte_count, modified_ms, window_sampling, model_id FROM artifacts WHERE id = ? LIMIT 1;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, artifactID, -1, SQLITE_TRANSIENT_VEC)
        if sqlite3_step(stmt) == SQLITE_ROW {
            let bytes = sqlite3_column_int64(stmt, 0)
            let mtime = sqlite3_column_int64(stmt, 1)
            let storedSampling = sqlite3_column_text(stmt, 2).map { String(cString: $0) } ?? ""
            let storedModel = sqlite3_column_text(stmt, 3).map { String(cString: $0) } ?? ""
            let modelMatches = modelID.isEmpty || storedModel.isEmpty || storedModel == modelID
            return bytes == fingerprint.byteCount
                && mtime == fingerprint.modifiedUnixMs
                && storedSampling == windowSampling.rawValue
                && modelMatches
        }
        return false
    }

    public func upsert(
        artifact: Artifact,
        chunks: [EmbeddedChunk],
        windowSampling: VideoWindowSampling = .off,
        modelID: String = ""
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        try execute("BEGIN IMMEDIATE TRANSACTION;")
        do {
            try deleteChunksAndArtifactUnlocked(artifactID: artifact.id)
            let attrData = try JSONEncoder().encode(artifact.attributes)
            let attrString = String(data: attrData, encoding: .utf8) ?? "{}"

            let insertArtifact = """
            INSERT INTO artifacts (
                id, collection_id, kind, url, title, byte_count, modified_ms, window_sampling, model_id, attributes_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            var aStmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, insertArtifact, -1, &aStmt, nil) == SQLITE_OK else { throw lastError() }
            defer { sqlite3_finalize(aStmt) }
            sqlite3_bind_text(aStmt, 1, artifact.id, -1, SQLITE_TRANSIENT_VEC)
            sqlite3_bind_text(aStmt, 2, artifact.collectionID, -1, SQLITE_TRANSIENT_VEC)
            sqlite3_bind_text(aStmt, 3, artifact.kind.rawValue, -1, SQLITE_TRANSIENT_VEC)
            sqlite3_bind_text(aStmt, 4, artifact.url.absoluteString, -1, SQLITE_TRANSIENT_VEC)
            sqlite3_bind_text(aStmt, 5, artifact.title, -1, SQLITE_TRANSIENT_VEC)
            sqlite3_bind_int64(aStmt, 6, artifact.fingerprint.byteCount)
            sqlite3_bind_int64(aStmt, 7, artifact.fingerprint.modifiedUnixMs)
            sqlite3_bind_text(aStmt, 8, windowSampling.rawValue, -1, SQLITE_TRANSIENT_VEC)
            sqlite3_bind_text(aStmt, 9, modelID, -1, SQLITE_TRANSIENT_VEC)
            sqlite3_bind_text(aStmt, 10, attrString, -1, SQLITE_TRANSIENT_VEC)
            guard sqlite3_step(aStmt) == SQLITE_DONE else { throw lastError() }

            let insertChunk = """
            INSERT INTO chunks (id, artifact_id, span_json, vector_blob, thumbnail_jpeg, preview_text)
            VALUES (?, ?, ?, ?, ?, ?);
            """
            for chunk in chunks {
                var cStmt: OpaquePointer?
                guard sqlite3_prepare_v2(db, insertChunk, -1, &cStmt, nil) == SQLITE_OK else { throw lastError() }
                defer { sqlite3_finalize(cStmt) }
                let spanData = try JSONEncoder().encode(chunk.span)
                let spanJSON = String(data: spanData, encoding: .utf8) ?? "{}"
                let vecData = chunk.vector.withUnsafeBufferPointer { Data(buffer: $0) }

                sqlite3_bind_text(cStmt, 1, chunk.id, -1, SQLITE_TRANSIENT_VEC)
                sqlite3_bind_text(cStmt, 2, chunk.artifactID, -1, SQLITE_TRANSIENT_VEC)
                sqlite3_bind_text(cStmt, 3, spanJSON, -1, SQLITE_TRANSIENT_VEC)
                _ = vecData.withUnsafeBytes { raw in
                    sqlite3_bind_blob(cStmt, 4, raw.baseAddress, Int32(vecData.count), SQLITE_TRANSIENT_VEC)
                }
                if let thumb = chunk.thumbnailJPEG {
                    _ = thumb.withUnsafeBytes { raw in
                        sqlite3_bind_blob(cStmt, 5, raw.baseAddress, Int32(thumb.count), SQLITE_TRANSIENT_VEC)
                    }
                } else {
                    sqlite3_bind_null(cStmt, 5)
                }
                if let preview = chunk.previewText {
                    sqlite3_bind_text(cStmt, 6, preview, -1, SQLITE_TRANSIENT_VEC)
                } else {
                    sqlite3_bind_null(cStmt, 6)
                }
                guard sqlite3_step(cStmt) == SQLITE_DONE else { throw lastError() }
            }
            try execute("COMMIT;")
        } catch {
            _ = try? execute("ROLLBACK;")
            throw error
        }
    }

    public func search(
        queryVector: [Float],
        dimensions: Int = Matryoshka.gemma2NativeDimensions,
        kinds: Set<ArtifactKind>? = nil,
        collectionID: String? = nil,
        limit: Int = 24
    ) throws -> [SearchHit] {
        lock.lock()
        defer { lock.unlock() }

        let projectedQuery = Matryoshka.project(queryVector, dimensions: dimensions)
        var projectedByDim = [Int: [Float]]()
        for dim in Matryoshka.gemma2Dimensions {
            projectedByDim[dim] = (dim == dimensions)
                ? projectedQuery
                : Matryoshka.project(queryVector, dimensions: dim)
        }

        let sql = """
        SELECT
            a.id, a.collection_id, a.kind, a.url, a.title, a.byte_count, a.modified_ms, a.attributes_json,
            c.id, c.span_json, c.vector_blob, c.thumbnail_jpeg, c.preview_text,
            a.window_sampling, a.model_id
        FROM chunks c
        JOIN artifacts a ON a.id = c.artifact_id;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }

        struct GroupedAccumulator {
            let artifact: Artifact
            var bestScore: Float
            var bestSpan: ArtifactSpan
            var spanMatches: [SearchHit.SpanMatch]
            var thumbnailJPEG: Data?
            var previewText: String?
        }

        var groups = [String: GroupedAccumulator]()
        let decoder = JSONDecoder()

        while sqlite3_step(stmt) == SQLITE_ROW {
            let aID = String(cString: sqlite3_column_text(stmt, 0))
            let colID = String(cString: sqlite3_column_text(stmt, 1))
            if let filterCol = collectionID, filterCol != colID { continue }

            let kindRaw = String(cString: sqlite3_column_text(stmt, 2))
            guard let kind = ArtifactKind(rawValue: kindRaw) else { continue }
            if let filterKinds = kinds, !filterKinds.contains(kind) { continue }

            let urlStr = String(cString: sqlite3_column_text(stmt, 3))
            let title = String(cString: sqlite3_column_text(stmt, 4))
            let bytes = sqlite3_column_int64(stmt, 5)
            let mtime = sqlite3_column_int64(stmt, 6)
            let attrStr = String(cString: sqlite3_column_text(stmt, 7))

            let chunkID = String(cString: sqlite3_column_text(stmt, 8))
            let spanStr = String(cString: sqlite3_column_text(stmt, 9))
            let span = (try? decoder.decode(ArtifactSpan.self, from: Data(spanStr.utf8))) ?? .whole

            let blobPtr = sqlite3_column_blob(stmt, 10)
            let blobBytes = Int(sqlite3_column_bytes(stmt, 10))
            let floatCount = blobBytes / MemoryLayout<Float>.size
            guard let blobPtr, floatCount > 0 else { continue }
            let floatPtr = blobPtr.bindMemory(to: Float.self, capacity: floatCount)
            let rawVector = Array(UnsafeBufferPointer(start: floatPtr, count: floatCount))

            var scoresByDim = [Int: Float]()
            for dim in Matryoshka.gemma2Dimensions {
                if let pq = projectedByDim[dim] {
                    scoresByDim[dim] = Matryoshka.cosine(pq, rawVector, dimensions: dim)
                }
            }
            let score = scoresByDim[dimensions] ?? Matryoshka.cosine(projectedQuery, rawVector, dimensions: dimensions)

            var thumb: Data?
            if let tPtr = sqlite3_column_blob(stmt, 11) {
                let tLen = Int(sqlite3_column_bytes(stmt, 11))
                if tLen > 0 { thumb = Data(bytes: tPtr, count: tLen) }
            }
            let preview: String? = sqlite3_column_type(stmt, 12) == SQLITE_NULL ? nil : String(cString: sqlite3_column_text(stmt, 12))
            let windowSamplingRaw: String = sqlite3_column_type(stmt, 13) == SQLITE_NULL ? "" : String(cString: sqlite3_column_text(stmt, 13))
            let modelID: String = sqlite3_column_type(stmt, 14) == SQLITE_NULL ? "" : String(cString: sqlite3_column_text(stmt, 14))

            let match = SearchHit.SpanMatch(
                chunkID: chunkID,
                span: span,
                score: score,
                previewText: preview,
                scoresByDimension: scoresByDim
            )

            if var existing = groups[aID] {
                existing.spanMatches.append(match)
                if score > existing.bestScore || (abs(score - existing.bestScore) < 1e-4 && existing.bestSpan == .whole && span != .whole) {
                    existing.bestScore = score
                    existing.bestSpan = span
                    if let preview { existing.previewText = preview }
                }
                if existing.thumbnailJPEG == nil, let thumb {
                    existing.thumbnailJPEG = thumb
                }
                groups[aID] = existing
            } else {
                let attrs = (try? decoder.decode([String: String].self, from: Data(attrStr.utf8))) ?? [:]
                let artifact = Artifact(
                    id: aID,
                    collectionID: colID,
                    kind: kind,
                    url: URL(string: urlStr) ?? URL(fileURLWithPath: urlStr),
                    title: title,
                    fingerprint: ContentFingerprint(byteCount: bytes, modifiedUnixMs: mtime),
                    windowSamplingRaw: windowSamplingRaw,
                    modelID: modelID,
                    attributes: attrs
                )
                groups[aID] = GroupedAccumulator(
                    artifact: artifact,
                    bestScore: score,
                    bestSpan: span,
                    spanMatches: [match],
                    thumbnailJPEG: thumb,
                    previewText: preview
                )
            }
        }

        return groups.values
            .sorted { $0.bestScore > $1.bestScore }
            .prefix(limit)
            .map { group in
                SearchHit(
                    artifact: group.artifact,
                    bestScore: group.bestScore,
                    bestSpan: group.bestSpan,
                    spanMatches: group.spanMatches.sorted { $0.score > $1.score },
                    thumbnailJPEG: group.thumbnailJPEG,
                    previewText: group.previewText
                )
            }
    }

    public struct StoreStats: Sendable, Equatable {
        public let artifactCount: Int
        public let chunkCount: Int
        public let countsByKind: [ArtifactKind: Int]
    }

    public func stats() throws -> StoreStats {
        lock.lock()
        defer { lock.unlock() }
        var byKind = [ArtifactKind: Int]()
        var totalArtifacts = 0
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT kind, COUNT(*) FROM artifacts GROUP BY kind;", -1, &stmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(stmt) }
            while sqlite3_step(stmt) == SQLITE_ROW {
                let kStr = String(cString: sqlite3_column_text(stmt, 0))
                let count = Int(sqlite3_column_int64(stmt, 1))
                totalArtifacts += count
                if let k = ArtifactKind(rawValue: kStr) {
                    byKind[k] = count
                }
            }
        }
        var totalChunks = 0
        if sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM chunks;", -1, &stmt, nil) == SQLITE_OK {
            defer { sqlite3_finalize(stmt) }
            if sqlite3_step(stmt) == SQLITE_ROW {
                totalChunks = Int(sqlite3_column_int64(stmt, 0))
            }
        }
        return StoreStats(artifactCount: totalArtifacts, chunkCount: totalChunks, countsByKind: byKind)
    }

    public struct WatchedFolder: Sendable, Equatable, Identifiable, Hashable {
        public var id: String { path }
        public let path: String
        public let collectionID: String
        public let windowSampling: VideoWindowSampling
        public let addedUnixMs: Int64
        public let lastIndexedUnixMs: Int64

        public var url: URL { URL(fileURLWithPath: path) }
        public var title: String { url.lastPathComponent }

        public init(
            path: String,
            collectionID: String,
            windowSampling: VideoWindowSampling = .fps,
            addedUnixMs: Int64,
            lastIndexedUnixMs: Int64
        ) {
            self.path = path
            self.collectionID = collectionID
            self.windowSampling = windowSampling
            self.addedUnixMs = addedUnixMs
            self.lastIndexedUnixMs = lastIndexedUnixMs
        }
    }

    public func addWatchedFolder(
        _ url: URL,
        collectionID: String = "default",
        windowSampling: VideoWindowSampling = .fps
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        let standardized = url.standardizedFileURL.path
        let nowMs = Int64(Date().timeIntervalSince1970 * 1000)
        let sql = """
        INSERT INTO watched_folders (path, collection_id, window_sampling, added_ms, last_indexed_ms)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            window_sampling = excluded.window_sampling,
            last_indexed_ms = excluded.last_indexed_ms;
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, standardized, -1, SQLITE_TRANSIENT_VEC)
        sqlite3_bind_text(stmt, 2, collectionID, -1, SQLITE_TRANSIENT_VEC)
        sqlite3_bind_text(stmt, 3, windowSampling.rawValue, -1, SQLITE_TRANSIENT_VEC)
        sqlite3_bind_int64(stmt, 4, nowMs)
        sqlite3_bind_int64(stmt, 5, nowMs)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    public func removeWatchedFolder(_ url: URL) throws {
        try removeWatchedFolder(path: url.standardizedFileURL.path)
    }

    public func removeWatchedFolder(path: String) throws {
        lock.lock()
        defer { lock.unlock() }
        let sql = "DELETE FROM watched_folders WHERE path = ?;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, path, -1, SQLITE_TRANSIENT_VEC)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
    }

    public func clearAll() throws {
        lock.lock()
        defer { lock.unlock() }
        try execute("DELETE FROM chunks; DELETE FROM artifacts;")
    }

    public func listWatchedFolders() throws -> [WatchedFolder] {
        lock.lock()
        defer { lock.unlock() }
        let sql = "SELECT path, collection_id, window_sampling, added_ms, last_indexed_ms FROM watched_folders ORDER BY added_ms ASC;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(stmt) }
        var out = [WatchedFolder]()
        while sqlite3_step(stmt) == SQLITE_ROW {
            let p = String(cString: sqlite3_column_text(stmt, 0))
            let c = String(cString: sqlite3_column_text(stmt, 1))
            let sRaw = sqlite3_column_type(stmt, 2) == SQLITE_NULL ? "fps" : String(cString: sqlite3_column_text(stmt, 2))
            let s = VideoWindowSampling(rawValue: sRaw) ?? .fps
            let added = sqlite3_column_int64(stmt, 3)
            let lastIdx = sqlite3_column_int64(stmt, 4)
            out.append(WatchedFolder(path: p, collectionID: c, windowSampling: s, addedUnixMs: added, lastIndexedUnixMs: lastIdx))
        }
        return out
    }

    @discardableResult
    public func pruneMissingArtifacts(
        under rootDirectory: URL,
        collectionID: String = "default"
    ) throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        let rawRoot = rootDirectory.standardizedFileURL.path
        let resolvedRoot = rootDirectory.resolvingSymlinksInPath().standardizedFileURL.path
        let prefixes = Set([
            rawRoot.hasSuffix("/") ? rawRoot : rawRoot + "/",
            resolvedRoot.hasSuffix("/") ? resolvedRoot : resolvedRoot + "/",
        ])
        let sql = "SELECT id, url FROM artifacts WHERE collection_id = ?;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
        var staleIDs = [String]()
        sqlite3_bind_text(stmt, 1, collectionID, -1, SQLITE_TRANSIENT_VEC)
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = String(cString: sqlite3_column_text(stmt, 0))
            let urlStr = String(cString: sqlite3_column_text(stmt, 1))
            let fileURL = URL(string: urlStr) ?? URL(fileURLWithPath: urlStr)
            let filePath = fileURL.standardizedFileURL.path
            if prefixes.contains(where: { filePath.hasPrefix($0) }) && !FileManager.default.fileExists(atPath: filePath) {
                staleIDs.append(id)
            }
        }
        sqlite3_finalize(stmt)

        guard !staleIDs.isEmpty else { return 0 }
        try execute("BEGIN IMMEDIATE TRANSACTION;")
        do {
            for id in staleIDs {
                try deleteChunksAndArtifactUnlocked(artifactID: id)
            }
            try execute("COMMIT;")
            return staleIDs.count
        } catch {
            _ = try? execute("ROLLBACK;")
            throw error
        }
    }

    private func deleteChunksAndArtifactUnlocked(artifactID: String) throws {
        for sql in ["DELETE FROM chunks WHERE artifact_id = ?;", "DELETE FROM artifacts WHERE id = ?;"] {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { throw lastError() }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, artifactID, -1, SQLITE_TRANSIENT_VEC)
            guard sqlite3_step(stmt) == SQLITE_DONE else { throw lastError() }
        }
    }

    private func execute(_ sql: String) throws {
        var errMsg: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &errMsg) != SQLITE_OK {
            let msg = errMsg.map { String(cString: $0) } ?? "SQLite error"
            sqlite3_free(errMsg)
            throw VectorStoreError.sqlFailed(msg)
        }
    }

    private func lastError() -> VectorStoreError {
        let msg = db.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "SQLite error"
        return .sqlFailed(msg)
    }
}

/// Coordinates multimodal indexing and search across folders and files.
public final class IndexCoordinator: Sendable {
    public let engine: any EmbeddingEngine
    public let store: VectorStore

    public init(engine: any EmbeddingEngine, store: VectorStore) {
        self.engine = engine
        self.store = store
    }

    public struct IndexProgress: Sendable {
        public let processed: Int
        public let total: Int
        public let currentTitle: String
        public let skipped: [String]

        public init(processed: Int, total: Int, currentTitle: String, skipped: [String]) {
            self.processed = processed
            self.total = total
            self.currentTitle = currentTitle
            self.skipped = skipped
        }
    }

    /// Recursively scans `urls` (files or directories), skips unchanged files via `ContentFingerprint`
    /// + `(windowSampling, modelID)`, embeds new or updated media, text, code, and PDF artifacts
    /// at native 768d, and stores them in `VectorStore`.
    public func indexURLs(
        _ urls: [URL],
        collectionID: String = "default",
        videoWindowSampling: VideoWindowSampling = .fps,
        loadAudioFromVideo: Bool = false,
        includeNonMedia: Bool = true,
        onProgress: (@Sendable (IndexProgress) -> Void)? = nil
    ) async throws -> (indexed: Int, skipped: [String]) {
        for root in urls {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: root.path, isDirectory: &isDir), isDir.boolValue {
                try? store.addWatchedFolder(
                    root,
                    collectionID: collectionID,
                    windowSampling: videoWindowSampling
                )
                _ = try? store.pruneMissingArtifacts(under: root, collectionID: collectionID)
            }
        }

        let candidates = discoverFiles(in: urls, includeNonMedia: includeNonMedia)
        var skipped = [String]()
        var indexed = 0

        for (idx, fileURL) in candidates.enumerated() {
            try Task.checkCancellation()
            guard let kind = ArtifactKind.detect(from: fileURL) else { continue }
            guard let fingerprint = Self.fingerprint(for: fileURL) else {
                skipped.append("\(fileURL.lastPathComponent): unreadable file attributes")
                continue
            }

            let effectiveSampling: VideoWindowSampling =
                (kind == .video || kind == .audio) ? videoWindowSampling : .off
            let effectiveModelID: String =
                (kind == .video && loadAudioFromVideo) ? "\(engine.modelID)+audio" : engine.modelID

            let artifactID = "\(collectionID)|\(fileURL.standardizedFileURL.path)"
            if (try? store.isUpToDate(
                artifactID: artifactID,
                fingerprint: fingerprint,
                windowSampling: effectiveSampling,
                modelID: effectiveModelID
            )) == true {
                onProgress?(IndexProgress(
                    processed: idx + 1,
                    total: candidates.count,
                    currentTitle: fileURL.lastPathComponent,
                    skipped: skipped
                ))
                continue
            }

            onProgress?(IndexProgress(
                processed: idx,
                total: candidates.count,
                currentTitle: fileURL.lastPathComponent,
                skipped: skipped
            ))

            do {
                let artifact = Artifact(
                    id: artifactID,
                    collectionID: collectionID,
                    kind: kind,
                    url: fileURL.standardizedFileURL,
                    title: fileURL.lastPathComponent,
                    fingerprint: fingerprint,
                    windowSamplingRaw: effectiveSampling.rawValue,
                    modelID: effectiveModelID
                )
                let chunks = try await buildChunks(
                    for: artifact,
                    videoWindowSampling: videoWindowSampling,
                    loadAudioFromVideo: loadAudioFromVideo
                )
                guard !chunks.isEmpty else {
                    skipped.append("\(fileURL.lastPathComponent): no embeddable content")
                    continue
                }
                try store.upsert(
                    artifact: artifact,
                    chunks: chunks,
                    windowSampling: effectiveSampling,
                    modelID: effectiveModelID
                )
                indexed += 1
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                skipped.append("\(fileURL.lastPathComponent): \(error.localizedDescription)")
            }

            onProgress?(IndexProgress(
                processed: idx + 1,
                total: candidates.count,
                currentTitle: fileURL.lastPathComponent,
                skipped: skipped
            ))
        }

        return (indexed, skipped)
    }

    public func search(
        queryText: String = "",
        exampleFileURL: URL? = nil,
        dimensions: Int = Matryoshka.gemma2NativeDimensions,
        kinds: Set<ArtifactKind>? = nil,
        taskOverride: EmbeddingTask? = nil,
        collectionID: String? = nil,
        limit: Int = 24
    ) async throws -> [SearchHit] {
        var parts = [EmbeddingPart]()
        let trimmed = queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            parts.append(.text(trimmed))
        }
        if let fileURL = exampleFileURL, let kind = ArtifactKind.detect(from: fileURL) {
            switch kind {
            case .image, .video, .audio:
                parts.append(.mediaURL(kind: kind, url: fileURL, windowSampling: .off))
            case .document:
                let plans = try ArtifactProcessor.chunkDocument(at: fileURL)
                if let firstText = plans.first(where: { !$0.text.isEmpty })?.text {
                    parts.append(.text(String(firstText.prefix(8000))))
                } else if let firstThumb = plans.first(where: { $0.thumbnailJPEG != nil })?.thumbnailJPEG {
                    parts.append(.media(kind: .image, mimeType: "image/jpeg", data: firstThumb))
                }
            case .text, .code:
                let text = try String(contentsOf: fileURL, encoding: .utf8)
                parts.append(.text(String(text.prefix(8000))))
            }
        }
        guard !parts.isEmpty else { return [] }

        let queryTask: EmbeddingTask = taskOverride ?? ((kinds == [.code]) ? .codeRetrieval : .query)
        let outputs = try await engine.embedBatch(
            [EmbeddingInput(parts: parts, task: queryTask)],
            dimensions: engine.nativeDimensions
        )
        guard let queryVec = outputs.first?.vector else { return [] }
        return try store.search(
            queryVector: queryVec,
            dimensions: dimensions,
            kinds: kinds,
            collectionID: collectionID,
            limit: limit
        )
    }

    private func buildChunks(
        for artifact: Artifact,
        videoWindowSampling: VideoWindowSampling,
        loadAudioFromVideo: Bool
    ) async throws -> [EmbeddedChunk] {
        switch artifact.kind {
        case .image:
            let input = EmbeddingInput(
                parts: [.mediaURL(kind: .image, url: artifact.url, windowSampling: .off)],
                task: .document
            )
            async let thumbTask = ArtifactProcessor.makeThumbnailJPEG(for: artifact.url, kind: .image)
            let outputs = try await engine.embedBatch([input], dimensions: engine.nativeDimensions)
            let thumb = await thumbTask
            guard let first = outputs.first else { return [] }
            return [
                EmbeddedChunk(
                    id: "\(artifact.id)#whole",
                    artifactID: artifact.id,
                    span: .whole,
                    vector: first.vector,
                    thumbnailJPEG: thumb,
                    previewText: nil
                )
            ]

        case .video:
            var wholeParts: [EmbeddingPart] = [
                .mediaURL(kind: .video, url: artifact.url, windowSampling: videoWindowSampling)
            ]
            var pcmSamples: [Float] = []
            if loadAudioFromVideo {
                pcmSamples = (try? await ArtifactProcessor.decodeMonoPCM(for: artifact.url)) ?? []
                if let wholeWAV = ArtifactProcessor.encodeAudioSliceWAV(
                    samples: pcmSamples,
                    sampleRate: ArtifactProcessor.audioSampleRate,
                    startSeconds: 0.0,
                    endSeconds: ArtifactProcessor.audioWindowSeconds
                ) {
                    wholeParts.append(.text(" "))
                    wholeParts.append(
                        .media(kind: .audio, mimeType: "audio/wav", data: wholeWAV, windowSampling: .off)
                    )
                }
            }

            async let thumbTask = ArtifactProcessor.makeThumbnailJPEG(for: artifact.url, kind: .video)
            let outputs = try await engine.embedBatch(
                [EmbeddingInput(parts: wholeParts, task: .document)],
                dimensions: engine.nativeDimensions
            )
            let thumb = await thumbTask
            guard let first = outputs.first else { return [] }

            var chunks = [
                EmbeddedChunk(
                    id: "\(artifact.id)#whole",
                    artifactID: artifact.id,
                    span: .whole,
                    vector: first.vector,
                    thumbnailJPEG: thumb,
                    previewText: nil
                )
            ]
            if first.windows.count > 1 {
                for w in first.windows {
                    let repTime = w.frameTimes.isEmpty
                        ? (w.start + w.end) / 2.0
                        : w.frameTimes[w.frameTimes.count / 2]
                    let winThumb = await ArtifactProcessor.makeThumbnailJPEG(
                        for: artifact.url,
                        kind: .video,
                        atSeconds: repTime
                    ) ?? thumb
                    chunks.append(
                        EmbeddedChunk(
                            id: "\(artifact.id)#t=\(w.start)-\(w.end)",
                            artifactID: artifact.id,
                            span: .temporal(start: w.start, end: w.end, frameTimes: w.frameTimes),
                            vector: w.vector,
                            thumbnailJPEG: winThumb,
                            previewText: nil
                        )
                    )
                }

                if loadAudioFromVideo && !pcmSamples.isEmpty {
                    var alignedSlices: [(win: EmbeddedOutput.WindowVector, wav: Data)] = []
                    for w in first.windows {
                        if let wav = ArtifactProcessor.encodeAudioSliceWAV(
                            samples: pcmSamples,
                            sampleRate: ArtifactProcessor.audioSampleRate,
                            startSeconds: w.start,
                            endSeconds: min(w.end, w.start + ArtifactProcessor.audioWindowSeconds)
                        ) {
                            alignedSlices.append((w, wav))
                        }
                    }
                    if !alignedSlices.isEmpty {
                        let audioInputs = alignedSlices.map { pair in
                            EmbeddingInput(
                                parts: [.media(kind: .audio, mimeType: "audio/wav", data: pair.wav, windowSampling: .off)],
                                task: .document
                            )
                        }
                        let audioOutputs = try await engine.embedBatch(audioInputs, dimensions: engine.nativeDimensions)
                        for (pair, out) in zip(alignedSlices, audioOutputs) {
                            let w = pair.win
                            chunks.append(
                                EmbeddedChunk(
                                    id: "\(artifact.id)#audio-t=\(w.start)-\(w.end)",
                                    artifactID: artifact.id,
                                    span: .temporal(start: w.start, end: w.end, frameTimes: w.frameTimes),
                                    vector: out.vector,
                                    thumbnailJPEG: thumb,
                                    previewText: "Audio track (\(ArtifactSpan.formatSeconds(w.start))-\(ArtifactSpan.formatSeconds(w.end)))"
                                )
                            )
                        }
                    }
                }
            }
            return chunks

        case .audio:
            let extracted = try? await ArtifactProcessor.extractAudioWindows(for: artifact.url)
            let wholePart: EmbeddingPart
            if let wholeWAV = extracted?.wholeWAV {
                wholePart = .media(kind: .audio, mimeType: "audio/wav", data: wholeWAV, windowSampling: .off)
            } else {
                wholePart = .mediaURL(kind: .audio, url: artifact.url, windowSampling: .off)
            }
            let wholeInput = EmbeddingInput(parts: [wholePart], task: .document)
            let outputs = try await engine.embedBatch([wholeInput], dimensions: engine.nativeDimensions)
            guard let first = outputs.first else { return [] }

            var chunks = [
                EmbeddedChunk(
                    id: "\(artifact.id)#whole",
                    artifactID: artifact.id,
                    span: .whole,
                    vector: first.vector,
                    thumbnailJPEG: nil,
                    previewText: nil
                )
            ]
            if videoWindowSampling != .off,
               let extracted,
               extracted.windows.count > 1 {
                let winInputs = extracted.windows.map { slice in
                    EmbeddingInput(
                        parts: [.media(kind: .audio, mimeType: "audio/wav", data: slice.wavData, windowSampling: .off)],
                        task: .document
                    )
                }
                let winOutputs = try await engine.embedBatch(winInputs, dimensions: engine.nativeDimensions)
                for (slice, out) in zip(extracted.windows, winOutputs) {
                    chunks.append(
                        EmbeddedChunk(
                            id: "\(artifact.id)#t=\(slice.start)-\(slice.end)",
                            artifactID: artifact.id,
                            span: .temporal(start: slice.start, end: slice.end, frameTimes: []),
                            vector: out.vector,
                            thumbnailJPEG: nil,
                            previewText: nil
                        )
                    )
                }
            }
            return chunks

        case .document:
            let plans = try ArtifactProcessor.chunkDocument(at: artifact.url)
            guard !plans.isEmpty else { return [] }
            let inputs: [EmbeddingInput] = plans.compactMap { plan in
                if !plan.text.isEmpty {
                    return EmbeddingInput.text(plan.text, task: .document, title: artifact.title)
                } else if let pageJPEG = plan.thumbnailJPEG {
                    return EmbeddingInput(
                        parts: [.media(kind: .image, mimeType: "image/jpeg", data: pageJPEG)],
                        task: .document
                    )
                }
                return nil
            }
            guard inputs.count == plans.count else { return [] }
            let outputs = try await engine.embedBatch(inputs, dimensions: engine.nativeDimensions)
            let fallbackThumb = plans.first(where: { $0.thumbnailJPEG != nil })?.thumbnailJPEG
            return zip(plans, outputs).enumerated().map { idx, pair in
                let (plan, out) = pair
                return EmbeddedChunk(
                    id: "\(artifact.id)#doc-\(idx)",
                    artifactID: artifact.id,
                    span: plan.span,
                    vector: out.vector,
                    thumbnailJPEG: plan.thumbnailJPEG ?? fallbackThumb,
                    previewText: plan.text.isEmpty ? nil : String(plan.text.prefix(280))
                )
            }

        case .text, .code:
            let text = try String(contentsOf: artifact.url, encoding: .utf8)
            let textChunks = ArtifactProcessor.chunkTextDocument(text)
            guard !textChunks.isEmpty else { return [] }
            let inputs = textChunks.map { EmbeddingInput.text($0.text, task: .document, title: artifact.title) }
            let outputs = try await engine.embedBatch(inputs, dimensions: engine.nativeDimensions)
            return zip(textChunks, outputs).enumerated().map { idx, pair in
                let (spanAndText, out) = pair
                return EmbeddedChunk(
                    id: "\(artifact.id)#chunk-\(idx)",
                    artifactID: artifact.id,
                    span: spanAndText.span,
                    vector: out.vector,
                    thumbnailJPEG: nil,
                    previewText: String(spanAndText.text.prefix(280))
                )
            }
        }
    }

    private func discoverFiles(in urls: [URL], includeNonMedia: Bool) -> [URL] {
        var result = [URL]()
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.isRegularFileKey, .isHiddenKey]

        for root in urls {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: root.path, isDirectory: &isDir) else { continue }
            if !isDir.boolValue {
                if let kind = ArtifactKind.detect(from: root), (kind.isMedia || includeNonMedia) {
                    result.append(root)
                }
                continue
            }
            guard let enumerator = fm.enumerator(
                at: root,
                includingPropertiesForKeys: keys,
                options: [.skipsHiddenFiles, .skipsPackageDescendants]
            ) else { continue }
            for case let fileURL as URL in enumerator {
                guard let vals = try? fileURL.resourceValues(forKeys: Set(keys)),
                      vals.isRegularFile == true,
                      let kind = ArtifactKind.detect(from: fileURL),
                      (kind.isMedia || includeNonMedia)
                else { continue }
                result.append(fileURL)
            }
        }
        return result.sorted { $0.path < $1.path }
    }

    public static func fingerprint(for url: URL) -> ContentFingerprint? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? NSNumber,
              let mtime = attrs[.modificationDate] as? Date
        else { return nil }
        return ContentFingerprint(
            byteCount: size.int64Value,
            modifiedUnixMs: Int64(mtime.timeIntervalSince1970 * 1000)
        )
    }
}

/// Built-in Retrieval Evaluation Lab (`Recall@1`, `Recall@3`, `MRR`) across Matryoshka dimensions.
public enum EmbeddingLabRunner {
    public struct BenchmarkCase: Sendable {
        public let query: String
        public let expectedArtifactID: String

        public init(query: String, expectedArtifactID: String) {
            self.query = query
            self.expectedArtifactID = expectedArtifactID
        }
    }

    public struct DimensionMetrics: Sendable, Equatable {
        public let dimensions: Int
        public let recallAt1: Double
        public let recallAt3: Double
        public let mrr: Double

        public init(dimensions: Int, recallAt1: Double, recallAt3: Double, mrr: Double) {
            self.dimensions = dimensions
            self.recallAt1 = recallAt1
            self.recallAt3 = recallAt3
            self.mrr = mrr
        }
    }

    public static func evaluate(
        coordinator: IndexCoordinator,
        cases: [BenchmarkCase],
        dimensions: [Int] = Matryoshka.gemma2Dimensions
    ) async throws -> [DimensionMetrics] {
        guard !cases.isEmpty else { return [] }
        var results: [DimensionMetrics] = []
        for dim in dimensions {
            var hitsAt1 = 0
            var hitsAt3 = 0
            var rrSum = 0.0
            for item in cases {
                let hits = try await coordinator.search(queryText: item.query, dimensions: dim, limit: 10)
                if let rankIdx = hits.firstIndex(where: { $0.artifact.id == item.expectedArtifactID }) {
                    let rank = rankIdx + 1
                    if rank == 1 { hitsAt1 += 1 }
                    if rank <= 3 { hitsAt3 += 1 }
                    rrSum += 1.0 / Double(rank)
                }
            }
            let n = Double(cases.count)
            results.append(
                DimensionMetrics(
                    dimensions: dim,
                    recallAt1: Double(hitsAt1) / n,
                    recallAt3: Double(hitsAt3) / n,
                    mrr: rrSum / n
                )
            )
        }
        return results
    }
}
