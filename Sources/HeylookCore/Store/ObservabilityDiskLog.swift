import Foundation

/// Observability JSONL stream selector matching Python `observability.py` and `/v1/admin/config/logs`.
public enum ObservabilityStream: String, Codable, Sendable, CaseIterable {
    case events
    case metrics

    public var filename: String { "\(rawValue).jsonl" }

    /// Strictly parses `"events"` or `"metrics"`, throwing `WireValidationError.unprocessable` (`422`) otherwise.
    public static func parse(_ raw: String) throws -> ObservabilityStream {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let stream = ObservabilityStream(rawValue: normalized) else {
            throw WireValidationError.unprocessable(
                "Invalid observability stream '\(raw)'. Expected 'events' or 'metrics'."
            )
        }
        return stream
    }
}

/// Ordered verbosity gate for disk JSONL persistence (`off < minimal < standard < debug`).
public enum ObservabilityVerbosity: String, Codable, Sendable, CaseIterable, Comparable {
    case off
    case minimal
    case standard
    case debug

    public var rank: Int {
        switch self {
        case .off: return 0
        case .minimal: return 1
        case .standard: return 2
        case .debug: return 3
        }
    }

    public static func < (lhs: ObservabilityVerbosity, rhs: ObservabilityVerbosity) -> Bool {
        lhs.rank < rhs.rank
    }

    public static func parse(_ raw: String) -> ObservabilityVerbosity {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ObservabilityVerbosity(rawValue: normalized) ?? .off
    }
}

/// Maps diagnostic/client severity tokens (`error`/`warn`/`info`/`debug`) onto `ObservabilityVerbosity`.
public enum ObservabilitySeverity: String, Codable, Sendable, CaseIterable {
    case error
    case warn
    case info
    case debug

    public var minimumVerbosity: ObservabilityVerbosity {
        switch self {
        case .error, .warn: return .minimal
        case .info: return .standard
        case .debug: return .debug
        }
    }

    public static func minimumVerbosity(forSeverity raw: String) -> ObservabilityVerbosity {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalized {
        case "error", "err", "critical", "fatal", "warn", "warning":
            return .minimal
        case "info":
            return .standard
        case "debug", "trace":
            return .debug
        default:
            return .standard
        }
    }
}

/// Operational `[settings]` schema (`settings.py`), validated on write with unknown-key rejection (`422`).
public struct OperationalSettings: Codable, Sendable, Equatable {
    public var observabilityLevel: ObservabilityVerbosity
    public var observabilityRetentionDays: Int
    public var mlxCacheLimitGB: Double?

    enum CodingKeys: String, CodingKey {
        case observabilityLevel = "observability_level"
        case observabilityRetentionDays = "observability_retention_days"
        case mlxCacheLimitGB = "mlx_cache_limit_gb"
    }

    public init(
        observabilityLevel: ObservabilityVerbosity = .off,
        observabilityRetentionDays: Int = 30,
        mlxCacheLimitGB: Double? = nil
    ) {
        self.observabilityLevel = observabilityLevel
        self.observabilityRetentionDays = max(0, observabilityRetentionDays)
        self.mlxCacheLimitGB = mlxCacheLimitGB
    }

    public init(from decoder: Decoder) throws {
        let raw = try decoder.container(keyedBy: WireAnyCodingKey.self)
        let allowed = Set([
            CodingKeys.observabilityLevel.rawValue,
            CodingKeys.observabilityRetentionDays.rawValue,
            CodingKeys.mlxCacheLimitGB.rawValue,
        ])
        for key in raw.allKeys where !allowed.contains(key.stringValue) {
            throw WireValidationError.unprocessable("Unknown operational setting '\(key.stringValue)'")
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        if container.contains(.observabilityLevel) {
            let val = try container.decode(JSONValue.self, forKey: .observabilityLevel)
            switch val {
            case .null:
                self.observabilityLevel = .off
            case .string(let s):
                guard let parsed = ObservabilityVerbosity(rawValue: s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()) else {
                    throw WireValidationError.unprocessable(
                        "observability_level must be 'off', 'minimal', 'standard', or 'debug'"
                    )
                }
                self.observabilityLevel = parsed
            default:
                throw WireValidationError.unprocessable("observability_level must be a string")
            }
        } else {
            self.observabilityLevel = .off
        }

        if container.contains(.observabilityRetentionDays) {
            let val = try container.decode(JSONValue.self, forKey: .observabilityRetentionDays)
            switch val {
            case .null:
                self.observabilityRetentionDays = 30
            case .int(let days) where days >= 0:
                self.observabilityRetentionDays = days
            default:
                throw WireValidationError.unprocessable("observability_retention_days must be an integer >= 0")
            }
        } else {
            self.observabilityRetentionDays = 30
        }

        if container.contains(.mlxCacheLimitGB) {
            let val = try container.decode(JSONValue.self, forKey: .mlxCacheLimitGB)
            switch val {
            case .null:
                self.mlxCacheLimitGB = nil
            case .double(let gb) where gb.isFinite && gb > 0:
                self.mlxCacheLimitGB = gb
            case .int(let gb) where gb > 0:
                self.mlxCacheLimitGB = Double(gb)
            default:
                throw WireValidationError.unprocessable("mlx_cache_limit_gb must be a positive number or null")
            }
        } else {
            self.mlxCacheLimitGB = nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(observabilityLevel.rawValue, forKey: .observabilityLevel)
        try c.encode(observabilityRetentionDays, forKey: .observabilityRetentionDays)
        try c.encode(mlxCacheLimitGB, forKey: .mlxCacheLimitGB)
    }
}

/// Level-gated, best-effort JSONL telemetry & event disk writer/reader (`observability.py` + `config_api.py`).
/// Maintains two distinct JSONL streams (`events.jsonl` and `metrics.jsonl`) with size rotation (50 MB),
/// retention pruning, and newest-first tail reading for `/v1/admin/config/logs`.
public final class ObservabilityDiskLog: @unchecked Sendable {
    public static let shared = ObservabilityDiskLog()
    public static let maxStreamBytes: Int = 50 * 1024 * 1024
    public static let rotateIntervalSeconds: TimeInterval = 3600.0
    public static let maxTailLimit: Int = 500

    private let lock = NSLock()
    private var verbosity: ObservabilityVerbosity
    private var logDirectory: URL
    private var retentionDays: Int
    private var lastRotateEpoch: TimeInterval = 0

    public static func defaultLogsDirectory() -> URL {
        if let envDir = ProcessInfo.processInfo.environment["HEYLOOK_LOGS_DIR"],
           !envDir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let expanded = (envDir as NSString).expandingTildeInPath
            return URL(fileURLWithPath: expanded, isDirectory: true)
        }
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return appSupport
            .appendingPathComponent("heylookitsanllm", isDirectory: true)
            .appendingPathComponent("logs", isDirectory: true)
    }

    public init(
        level: String = "off",
        logDirectory: URL? = nil,
        retentionDays: Int = 30
    ) {
        self.verbosity = ObservabilityVerbosity.parse(level)
        self.logDirectory = logDirectory ?? Self.defaultLogsDirectory()
        self.retentionDays = max(0, retentionDays)
    }

    /// Configures active verbosity, log directory, and retention policy. Unknown levels degrade to `.off`.
    public func configure(
        level: String,
        logDirectory: URL? = nil,
        retentionDays: Int = 30
    ) {
        lock.lock()
        defer { lock.unlock() }
        self.verbosity = ObservabilityVerbosity.parse(level)
        if let logDirectory {
            self.logDirectory = logDirectory
        }
        self.retentionDays = max(0, retentionDays)
    }

    /// Applies operational `[settings]` parsed from `heylook.toml` or `/v1/admin/config`.
    public func apply(settings: OperationalSettings) {
        configure(
            level: settings.observabilityLevel.rawValue,
            logDirectory: nil,
            retentionDays: settings.observabilityRetentionDays
        )
    }

    public func currentLevel() -> String {
        lock.lock()
        defer { lock.unlock() }
        return verbosity.rawValue
    }

    public func currentRetentionDays() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return retentionDays
    }

    public func currentLogDirectory() -> URL {
        lock.lock()
        defer { lock.unlock() }
        return logDirectory
    }

    public func isEnabled(forMinLevel minLevel: String) -> Bool {
        lock.lock()
        let active = verbosity
        lock.unlock()
        guard let gate = ObservabilityVerbosity(rawValue: minLevel.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()),
              gate != .off else {
            return false
        }
        return active >= gate
    }

    /// Appends one structured JSONL record to `events.jsonl` or `metrics.jsonl` if enabled.
    /// Reserved keys (`ts`, `iso`, `type`, `source`) are applied last so caller fields cannot overwrite them.
    /// Best-effort: never throws.
    public func recordEvent(
        _ eventType: String,
        tier: ObservabilityStream,
        minLevel: String = "standard",
        source: String = "backend",
        fields: [String: JSONValue] = [:]
    ) {
        lock.lock()
        let active = verbosity
        let dir = logDirectory
        lock.unlock()

        guard let gate = ObservabilityVerbosity(rawValue: minLevel.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()),
              gate != .off,
              active >= gate else {
            return
        }

        let now = Date()
        let ts = now.timeIntervalSince1970
        var record = fields
        record["ts"] = .double(ts)
        record["iso"] = .string(Self.localISO8601Millis(now))
        record["type"] = .string(eventType)
        record["source"] = .string(source)

        guard let encoded = try? JSONEncoder().encode(record) else { return }
        var line = encoded
        line.append(0x0A)

        let targetURL = dir.appendingPathComponent(tier.filename)
        lock.lock()
        defer { lock.unlock() }
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: targetURL.path),
               let handle = try? FileHandle(forWritingTo: targetURL) {
                defer { try? handle.close() }
                _ = try handle.seekToEnd()
                try handle.write(contentsOf: line)
            } else {
                try line.write(to: targetURL, options: .atomic)
            }
        } catch {
            // Best-effort invariant: disk telemetry errors never fail inference.
        }
    }

    /// Convenience helper matching Python `diagnostic_logger.diag_event`.
    public func recordDiagEvent(
        _ eventType: String,
        requestID: String? = nil,
        severity: String = "info",
        fields: [String: JSONValue] = [:]
    ) {
        var payload = fields
        if let requestID {
            payload["request_id"] = .string(requestID)
        }
        payload["level"] = .string(severity)
        let minVerbosity = ObservabilitySeverity.minimumVerbosity(forSeverity: severity)
        recordEvent(
            eventType,
            tier: .events,
            minLevel: minVerbosity.rawValue,
            source: "backend",
            fields: payload
        )
    }

    /// Reads up to `limit` (clamped `1...500`) JSONL records from the tail of `stream`, **newest first**.
    public func readTail(stream: ObservabilityStream, limit: Int = 50) -> [[String: JSONValue]] {
        let clampedLimit = max(1, min(Self.maxTailLimit, limit))
        lock.lock()
        let targetURL = logDirectory.appendingPathComponent(stream.filename)
        lock.unlock()

        guard FileManager.default.fileExists(atPath: targetURL.path),
              let data = try? Data(contentsOf: targetURL, options: .mappedIfSafe),
              !data.isEmpty else {
            return []
        }

        let decoder = JSONDecoder()
        var ring: [Data] = []
        ring.reserveCapacity(min(clampedLimit, 128))

        // Scan newline-delimited slices and keep only the trailing `clampedLimit` non-empty lines.
        var lineStart = data.startIndex
        for idx in data.indices {
            if data[idx] == 0x0A {
                if idx > lineStart {
                    let slice = data[lineStart..<idx]
                    if slice.contains(where: { $0 != 0x20 && $0 != 0x09 && $0 != 0x0D }) {
                        if ring.count == clampedLimit {
                            ring.removeFirst()
                        }
                        ring.append(Data(slice))
                    }
                }
                lineStart = data.index(after: idx)
            }
        }
        if lineStart < data.endIndex {
            let tailSlice = data[lineStart..<data.endIndex]
            if tailSlice.contains(where: { $0 != 0x20 && $0 != 0x09 && $0 != 0x0D }) {
                if ring.count == clampedLimit {
                    ring.removeFirst()
                }
                ring.append(Data(tailSlice))
            }
        }

        var records: [[String: JSONValue]] = []
        records.reserveCapacity(ring.count)
        for lineData in ring.reversed() {
            if let obj = try? decoder.decode([String: JSONValue].self, from: lineData) {
                records.append(obj)
            }
        }
        return records
    }

    /// Rolls active `<stream>.jsonl` above `maxBytes` to `<stream>.<epoch>.jsonl` and prunes expired archives.
    public func rotateStreams(
        retentionDays overrideDays: Int? = nil,
        maxBytes: Int = ObservabilityDiskLog.maxStreamBytes,
        now: Date = Date()
    ) {
        lock.lock()
        let dir = logDirectory
        let keepDays = max(0, overrideDays ?? retentionDays)
        lock.unlock()

        let fm = FileManager.default
        let nowEpoch = now.timeIntervalSince1970
        let cutoff = now.addingTimeInterval(-TimeInterval(keepDays) * 86_400.0)

        for stream in ObservabilityStream.allCases {
            let activeURL = dir.appendingPathComponent(stream.filename)
            let stem = stream.rawValue

            if let attrs = try? fm.attributesOfItem(atPath: activeURL.path),
               let sizeNum = attrs[.size] as? NSNumber,
               sizeNum.intValue > maxBytes {
                var archiveURL = dir.appendingPathComponent("\(stem).\(Int(nowEpoch)).jsonl")
                var suffix = 1
                while fm.fileExists(atPath: archiveURL.path) {
                    archiveURL = dir.appendingPathComponent("\(stem).\(Int(nowEpoch)).\(suffix).jsonl")
                    suffix += 1
                }
                try? fm.moveItem(at: activeURL, to: archiveURL)
            }

            if keepDays > 0,
               let items = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey]) {
                let prefix = "\(stem)."
                for item in items {
                    let name = item.lastPathComponent
                    guard name != stream.filename,
                          name.hasPrefix(prefix),
                          name.hasSuffix(".jsonl") else {
                        continue
                    }
                    if let vals = try? item.resourceValues(forKeys: [.contentModificationDateKey]),
                       let mtime = vals.contentModificationDate,
                       mtime < cutoff {
                        try? fm.removeItem(at: item)
                    }
                }
            }
        }
    }

    /// Throttled hourly rotation sweep.
    @discardableResult
    public func maybeRotate(now: Date = Date()) -> Bool {
        lock.lock()
        let nowEpoch = now.timeIntervalSince1970
        if nowEpoch - lastRotateEpoch < Self.rotateIntervalSeconds {
            lock.unlock()
            return false
        }
        lastRotateEpoch = nowEpoch
        lock.unlock()

        rotateStreams(now: now)
        return true
    }

    /// Removes active JSONL log file(s) from disk.
    public func clear(stream: ObservabilityStream? = nil) {
        lock.lock()
        let dir = logDirectory
        lock.unlock()

        let targets = stream.map { [$0] } ?? ObservabilityStream.allCases
        for s in targets {
            try? FileManager.default.removeItem(at: dir.appendingPathComponent(s.filename))
        }
    }

    private static func localISO8601Millis(_ date: Date) -> String {
        let fmt = ISO8601DateFormatter()
        fmt.timeZone = .current
        fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fmt.string(from: date)
    }
}
