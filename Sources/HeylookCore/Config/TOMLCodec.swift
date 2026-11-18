import Foundation

/// Typed value produced by `TOMLCodec` recursive-descent parsing.
public indirect enum TOMLValue: Sendable, Equatable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([TOMLValue])
    case table([String: TOMLValue])

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    public var intValue: Int? {
        switch self {
        case .int(let i): return i
        default: return nil
        }
    }

    public var doubleValue: Double? {
        switch self {
        case .double(let d): return d
        case .int(let i): return Double(i)
        default: return nil
        }
    }

    public var boolValue: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }

    public var arrayValue: [TOMLValue]? {
        if case .array(let items) = self { return items }
        return nil
    }

    public var stringArrayValue: [String]? {
        guard case .array(let items) = self else { return nil }
        return items.compactMap(\.stringValue)
    }

    public var doubleArrayValue: [Double]? {
        guard case .array(let items) = self else { return nil }
        var out: [Double] = []
        out.reserveCapacity(items.count)
        for item in items {
            guard let d = item.doubleValue else { return nil }
            out.append(d)
        }
        return out
    }

    public var tableValue: [String: TOMLValue]? {
        if case .table(let dict) = self { return dict }
        return nil
    }

    /// Converts to canonical `JSONValue`.
    public func toJSONValue() -> JSONValue {
        switch self {
        case .string(let s): return .string(s)
        case .int(let i): return .int(i)
        case .double(let d): return .double(d)
        case .bool(let b): return .bool(b)
        case .array(let items): return .array(items.map { $0.toJSONValue() })
        case .table(let map):
            var out: [String: JSONValue] = [:]
            out.reserveCapacity(map.count)
            for (k, v) in map {
                out[k] = v.toJSONValue()
            }
            return .object(out)
        }
    }

    /// Converts a non-null `JSONValue` to a `TOMLValue`. Returns `nil` for `.null`.
    public static func fromJSONValue(_ json: JSONValue) -> TOMLValue? {
        switch json {
        case .null:
            return nil
        case .string(let s):
            return .string(s)
        case .int(let i):
            return .int(i)
        case .double(let d):
            return .double(d)
        case .bool(let b):
            return .bool(b)
        case .array(let items):
            var out: [TOMLValue] = []
            out.reserveCapacity(items.count)
            for item in items {
                guard let converted = fromJSONValue(item) else {
                    return nil
                }
                out.append(converted)
            }
            return .array(out)
        case .object(let map):
            var out: [String: TOMLValue] = [:]
            out.reserveCapacity(map.count)
            for (k, v) in map {
                if let converted = fromJSONValue(v) {
                    out[k] = converted
                }
            }
            return .table(out)
        }
    }
}

/// Errors thrown by `TOMLCodec` on malformed TOML input.
public enum TOMLParseError: LocalizedError, Sendable, Equatable {
    case syntaxError(line: Int, message: String)
    case duplicateKey(key: String)
    case invalidTableRedefinition(path: String)

    public var errorDescription: String? {
        switch self {
        case .syntaxError(let line, let message):
            return "TOML syntax error at line \(line): \(message)"
        case .duplicateKey(let key):
            return "Duplicate TOML key '\(key)'"
        case .invalidTableRedefinition(let path):
            return "Invalid TOML table redefinition for '\(path)'"
        }
    }
}

/// Spec-accurate recursive-descent TOML parser and comment-preserving surgical editor.
/// Supports bare/quoted/dotted keys, `[table]` / `[models.config]` subtables, `[[array_of_tables]]`,
/// inline tables `{ ... }`, basic/literal strings (including escapes and commas inside strings),
/// multiline `"""..."""` / `'''...'''` strings, multiline arrays, comments inside arrays, ints/floats/booleans.
public enum TOMLCodec {
    /// Parses full TOML text into nested `[String: TOMLValue]`. Throws on syntax errors.
    public static func parse(_ text: String) throws -> [String: TOMLValue] {
        var parser = RecursiveTOMLParser(text)
        return try parser.parseRoot()
    }

    /// Best-effort non-throwing variant used when loading optional user files cleanly.
    public static func parseBestEffort(_ text: String) -> [String: TOMLValue] {
        (try? parse(text)) ?? [:]
    }

    /// Formats a `TOMLValue` as valid TOML source.
    public static func formatValue(_ value: TOMLValue) -> String {
        switch value {
        case .string(let s):
            return formatString(s)
        case .int(let i):
            return String(i)
        case .double(let d):
            if d.isNaN { return "nan" }
            if d == .infinity { return "inf" }
            if d == -.infinity { return "-inf" }
            let formatted = String(d)
            if formatted.contains(".") || formatted.contains("e") || formatted.contains("E") {
                return formatted
            }
            return formatted + ".0"
        case .bool(let b):
            return b ? "true" : "false"
        case .array(let items):
            if items.isEmpty { return "[]" }
            let formattedItems = items.map { formatValue($0) }
            let singleLine = "[\(formattedItems.joined(separator: ", "))]"
            if singleLine.count <= 80 && !formattedItems.contains(where: { $0.contains("\n") }) {
                return singleLine
            }
            let body = formattedItems.map { "  \($0)," }.joined(separator: "\n")
            return "[\n\(body)\n]"
        case .table(let dict):
            let pairs = dict.keys.sorted().compactMap { key -> String? in
                guard let v = dict[key] else { return nil }
                return "\(formatKey(key)) = \(formatValue(v))"
            }
            return "{ \(pairs.joined(separator: ", ")) }"
        }
    }

    /// Formats a key as bare if `[A-Za-z0-9_-]+`, otherwise double-quoted.
    public static func formatKey(_ key: String) -> String {
        let isBare = !key.isEmpty && key.unicodeScalars.allSatisfy {
            ("A"..."Z").contains($0) || ("a"..."z").contains($0) || ("0"..."9").contains($0) || $0 == "_" || $0 == "-"
        }
        return isBare ? key : formatString(key)
    }

    /// Formats a string as a TOML basic double-quoted string with proper escaping.
    public static func formatString(_ s: String) -> String {
        var out = "\""
        out.reserveCapacity(s.count + 4)
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out.append("\\\"")
            case "\\": out.append("\\\\")
            case "\u{0008}": out.append("\\b")
            case "\u{000C}": out.append("\\f")
            case "\n": out.append("\\n")
            case "\r": out.append("\\r")
            case "\t": out.append("\\t")
            default:
                if scalar.value < 0x20 || scalar.value == 0x7F {
                    out.append(String(format: "\\u%04X", scalar.value))
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out.append("\"")
        return out
    }

    // MARK: - Surgical Comment-Preserving Editors

    /// Surgically sets or removes (`value == nil`) a key inside `[section]` (or root when `section == nil`),
    /// preserving all user `#` comments, untouched keys, and untouched sections/tables.
    /// Properly spans multiline arrays and multiline strings so partial trailing lines are never left orphaned.
    public static func updateKey(
        in text: String,
        section: String?,
        key: String,
        value: TOMLValue?
    ) -> String {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        var lines = normalized.isEmpty ? [] : normalized.components(separatedBy: "\n")
        if lines.last == "" && normalized.hasSuffix("\n") {
            lines.removeLast()
        }

        let scanner = StatementScanner(lines: lines)
        let targetSectionRange = scanner.findSectionRange(named: section)

        if let targetSectionRange {
            // Find key inside [targetSectionRange.start ..< targetSectionRange.end]
            if let existingStmt = scanner.findKeyStatement(named: key, in: targetSectionRange) {
                if let value {
                    let formattedAssignment = "\(formatKey(key)) = \(formatValue(value))"
                    var replacementLines = formattedAssignment.components(separatedBy: "\n")
                    // Preserve inline comment when replacing single-line statement with single-line value
                    if existingStmt.startLine == existingStmt.endLine && replacementLines.count == 1,
                       let inlineComment = existingStmt.inlineComment {
                        replacementLines[0] += inlineComment
                    }
                    lines.replaceSubrange(existingStmt.startLine...existingStmt.endLine, with: replacementLines)
                } else {
                    lines.removeSubrange(existingStmt.startLine...existingStmt.endLine)
                }
                return joinLines(lines)
            }

            // Key does not exist in existing section
            guard let value else {
                return joinLines(lines)
            }
            let formattedAssignment = "\(formatKey(key)) = \(formatValue(value))"
            let assignmentLines = formattedAssignment.components(separatedBy: "\n")
            let insertAt = targetSectionRange.lastNonBlankContentLine + 1
            lines.insert(contentsOf: assignmentLines, at: insertAt)
            return joinLines(lines)
        }

        // Section does not exist yet
        guard let value else {
            return joinLines(lines)
        }
        let formattedAssignment = "\(formatKey(key)) = \(formatValue(value))"
        let newBlock: [String]
        if let section {
            newBlock = ["[\(section)]"] + formattedAssignment.components(separatedBy: "\n")
        } else {
            newBlock = formattedAssignment.components(separatedBy: "\n")
        }

        if let firstModelsIdx = scanner.firstArrayOfTablesLineIndex() {
            // Keep `[scan]` / `[settings]` ahead of `[[models]]` tables
            var insertIdx = firstModelsIdx
            while insertIdx > 0 && lines[insertIdx - 1].trimmingCharacters(in: .whitespaces).hasPrefix("#") {
                // Keep comment header attached to first [[models]] entry
                insertIdx -= 1
            }
            var toInsert = newBlock
            if insertIdx > 0 && !lines[insertIdx - 1].trimmingCharacters(in: .whitespaces).isEmpty {
                toInsert.insert("", at: 0)
            }
            if insertIdx < lines.count && !lines[insertIdx].trimmingCharacters(in: .whitespaces).isEmpty {
                toInsert.append("")
            }
            lines.insert(contentsOf: toInsert, at: insertIdx)
        } else {
            if !lines.isEmpty && !lines[lines.count - 1].trimmingCharacters(in: .whitespaces).isEmpty {
                lines.append("")
            }
            lines.append(contentsOf: newBlock)
        }
        return joinLines(lines)
    }

    /// Surgically updates `[scan]` in `heylook.toml`, preserving all `#` comments, `[settings]`, and `[[models]]`.
    public static func updateScanConfig(
        in text: String,
        folders: [String]? = nil,
        autoDiscover: Bool? = nil
    ) -> String {
        var result = text
        if let folders {
            result = updateKey(in: result, section: "scan", key: "folders", value: .array(folders.map { .string($0) }))
        }
        if let autoDiscover {
            result = updateKey(in: result, section: "scan", key: "auto_discover", value: .bool(autoDiscover))
        }
        return result
    }

    /// Surgically updates keys under `[settings]` in `heylook.toml`, preserving comments and all other sections.
    /// Passing `.null` in `updates` removes that key from `[settings]`.
    public static func updateSettings(
        in text: String,
        updates: [String: JSONValue]
    ) -> String {
        var result = text
        for key in updates.keys.sorted() {
            guard let jsonVal = updates[key] else { continue }
            let tomlVal = TOMLValue.fromJSONValue(jsonVal)
            result = updateKey(in: result, section: "settings", key: key, value: tomlVal)
        }
        return result
    }

    /// Surgically edits `model.heylook.toml` text for a key delta map (`JSONValue.null` removes the key).
    /// Returns `nil` if applying the delta leaves zero keys in the sidecar (caller should delete the file).
    public static func updateModelSidecar(
        existingText: String?,
        updates: [String: JSONValue]
    ) -> String? {
        var result = existingText ?? "# model.heylook.toml\n"
        for key in updates.keys.sorted() {
            guard let jsonVal = updates[key] else { continue }
            let tomlVal = TOMLValue.fromJSONValue(jsonVal)
            result = updateKey(in: result, section: nil, key: key, value: tomlVal)
        }
        let parsed = parseBestEffort(result)
        if parsed.isEmpty {
            return nil
        }
        return result
    }

    /// Applies `updateModelSidecar` on disk at `sidecarURL`, deleting `sidecarURL` when empty (`nil`).
    /// Returns `true` if the sidecar exists on disk after the update, `false` if deleted.
    @discardableResult
    public static func applyModelSidecarPatch(
        at sidecarURL: URL,
        updates: [String: JSONValue]
    ) throws -> Bool {
        let fm = FileManager.default
        let existing = try? String(contentsOf: sidecarURL, encoding: .utf8)
        guard let nextText = updateModelSidecar(existingText: existing, updates: updates) else {
            if fm.fileExists(atPath: sidecarURL.path) {
                try fm.removeItem(at: sidecarURL)
            }
            return false
        }
        try fm.createDirectory(at: sidecarURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try nextText.write(to: sidecarURL, atomically: true, encoding: .utf8)
        return true
    }

    private static func joinLines(_ lines: [String]) -> String {
        guard !lines.isEmpty else { return "" }
        return lines.joined(separator: "\n") + "\n"
    }
}

// MARK: - Recursive-Descent TOML Parser

private struct RecursiveTOMLParser {
    private let chars: [UnicodeScalar]
    private var idx: Int = 0
    private var line: Int = 1

    private var root: [String: TOMLValue] = [:]
    private enum CurrentTarget {
        case root
        case table([String])
        case arrayTableItem(arrayPath: [String], index: Int, subPath: [String])
    }
    private var currentTarget: CurrentTarget = .root
    private var knownArrayOfTables: Set<[String]> = []

    init(_ source: String) {
        self.chars = Array(source.unicodeScalars)
    }

    private var isEOF: Bool { idx >= chars.count }

    private func peek(_ offset: Int = 0) -> UnicodeScalar? {
        let pos = idx + offset
        return pos < chars.count ? chars[pos] : nil
    }

    private mutating func advance() -> UnicodeScalar? {
        guard idx < chars.count else { return nil }
        let c = chars[idx]
        idx += 1
        if c == "\n" {
            line += 1
        }
        return c
    }

    private mutating func skipHorizontalSpace() {
        while let c = peek(), c == " " || c == "\t" || c == "\r" {
            _ = advance()
        }
    }

    private mutating func skipSpaceAndComments(allowNewlines: Bool) {
        while let c = peek() {
            if c == " " || c == "\t" || c == "\r" {
                _ = advance()
            } else if allowNewlines && c == "\n" {
                _ = advance()
            } else if c == "#" {
                while let cc = peek(), cc != "\n" {
                    _ = advance()
                }
            } else {
                break
            }
        }
    }

    private mutating func expectEndOfLineOrEOF() throws {
        skipHorizontalSpace()
        if peek() == "#" {
            while let c = peek(), c != "\n" {
                _ = advance()
            }
        }
        if let c = peek() {
            guard c == "\n" else {
                throw TOMLParseError.syntaxError(line: line, message: "Unexpected trailing token '\(c)'")
            }
            _ = advance()
        }
    }

    mutating func parseRoot() throws -> [String: TOMLValue] {
        while true {
            skipSpaceAndComments(allowNewlines: true)
            if isEOF { break }

            if peek() == "[" {
                if peek(1) == "[" {
                    _ = advance()
                    _ = advance()
                    skipHorizontalSpace()
                    let path = try parseKeyPath()
                    skipHorizontalSpace()
                    guard peek() == "]" && peek(1) == "]" else {
                        throw TOMLParseError.syntaxError(line: line, message: "Expected ']]' closing array-of-tables header")
                    }
                    _ = advance()
                    _ = advance()
                    try expectEndOfLineOrEOF()
                    let newIndex = try appendArrayTableItem(at: path)
                    currentTarget = .arrayTableItem(arrayPath: path, index: newIndex, subPath: [])
                } else {
                    _ = advance()
                    skipHorizontalSpace()
                    let path = try parseKeyPath()
                    skipHorizontalSpace()
                    guard peek() == "]" else {
                        throw TOMLParseError.syntaxError(line: line, message: "Expected ']' closing table header")
                    }
                    _ = advance()
                    try expectEndOfLineOrEOF()
                    currentTarget = try openTableHeader(at: path)
                }
            } else {
                let keyPath = try parseKeyPath()
                skipHorizontalSpace()
                guard peek() == "=" else {
                    throw TOMLParseError.syntaxError(line: line, message: "Expected '=' after key '\(keyPath.joined(separator: "."))'")
                }
                _ = advance()
                skipHorizontalSpace()
                let value = try parseValue()
                try expectEndOfLineOrEOF()
                try insertValue(value, atRelativeKeyPath: keyPath)
            }
        }
        return root
    }

    private mutating func parseKeyPath() throws -> [String] {
        var segments: [String] = []
        while true {
            skipHorizontalSpace()
            segments.append(try parseSingleKey())
            skipHorizontalSpace()
            if peek() == "." {
                _ = advance()
                continue
            }
            break
        }
        return segments
    }

    private mutating func parseSingleKey() throws -> String {
        guard let c = peek() else {
            throw TOMLParseError.syntaxError(line: line, message: "Unexpected EOF waiting for key")
        }
        if c == "\"" {
            return try parseBasicString(allowMultiline: false)
        }
        if c == "'" {
            return try parseLiteralString(allowMultiline: false)
        }
        var key = ""
        while let ch = peek(), Self.isBareKeyChar(ch) {
            key.unicodeScalars.append(ch)
            _ = advance()
        }
        guard !key.isEmpty else {
            throw TOMLParseError.syntaxError(line: line, message: "Invalid key starting with '\(c)'")
        }
        return key
    }

    private static func isBareKeyChar(_ c: UnicodeScalar) -> Bool {
        ("A"..."Z").contains(c) || ("a"..."z").contains(c) || ("0"..."9").contains(c) || c == "_" || c == "-"
    }

    private mutating func parseValue() throws -> TOMLValue {
        skipHorizontalSpace()
        guard let c = peek() else {
            throw TOMLParseError.syntaxError(line: line, message: "Unexpected EOF waiting for value")
        }
        if c == "\"" {
            return .string(try parseBasicString(allowMultiline: true))
        }
        if c == "'" {
            return .string(try parseLiteralString(allowMultiline: true))
        }
        if c == "[" {
            return try parseArray()
        }
        if c == "{" {
            return try parseInlineTable()
        }
        return try parseScalar()
    }

    private mutating func parseBasicString(allowMultiline: Bool) throws -> String {
        let startLine = line
        _ = advance() // consume opening "
        if allowMultiline && peek() == "\"" && peek(1) == "\"" {
            _ = advance()
            _ = advance()
            if peek() == "\r" && peek(1) == "\n" {
                _ = advance(); _ = advance()
            } else if peek() == "\n" {
                _ = advance()
            }
            var out = ""
            while let c = peek() {
                if c == "\"" && peek(1) == "\"" && peek(2) == "\"" {
                    _ = advance(); _ = advance(); _ = advance()
                    // Up to 2 additional quotes allowed before closing """
                    var extra = 0
                    while peek() == "\"" && extra < 2 {
                        out.append("\"")
                        _ = advance()
                        extra += 1
                    }
                    return out
                }
                if c == "\\" {
                    _ = advance()
                    if let next = peek(), next == "\n" || next == "\r" || next == " " || next == "\t" {
                        // Line-ending backslash trims all whitespace including newlines
                        var consumedNewline = false
                        while let ws = peek(), ws == " " || ws == "\t" || ws == "\r" || ws == "\n" {
                            if ws == "\n" { consumedNewline = true }
                            _ = advance()
                        }
                        if consumedNewline { continue }
                        throw TOMLParseError.syntaxError(line: line, message: "Invalid escape after '\\'")
                    }
                    try appendEscapeSequence(into: &out)
                } else {
                    if let ch = advance() { out.unicodeScalars.append(ch) }
                }
            }
            throw TOMLParseError.syntaxError(line: startLine, message: "Unterminated multiline basic string")
        }

        var out = ""
        while let c = peek() {
            if c == "\"" {
                _ = advance()
                return out
            }
            if c == "\n" {
                throw TOMLParseError.syntaxError(line: startLine, message: "Newline inside single-line basic string")
            }
            if c == "\\" {
                _ = advance()
                try appendEscapeSequence(into: &out)
            } else {
                if let ch = advance() { out.unicodeScalars.append(ch) }
            }
        }
        throw TOMLParseError.syntaxError(line: startLine, message: "Unterminated basic string")
    }

    private mutating func appendEscapeSequence(into out: inout String) throws {
        guard let esc = advance() else {
            throw TOMLParseError.syntaxError(line: line, message: "Unexpected EOF in string escape")
        }
        switch esc {
        case "\"": out.append("\"")
        case "\\": out.append("\\")
        case "b": out.append("\u{0008}")
        case "f": out.append("\u{000C}")
        case "n": out.append("\n")
        case "r": out.append("\r")
        case "t": out.append("\t")
        case "u": out.unicodeScalars.append(try parseUnicodeEscape(digits: 4))
        case "U": out.unicodeScalars.append(try parseUnicodeEscape(digits: 8))
        default:
            throw TOMLParseError.syntaxError(line: line, message: "Unsupported string escape '\\\(esc)'")
        }
    }

    private mutating func parseUnicodeEscape(digits: Int) throws -> UnicodeScalar {
        var hex = ""
        for _ in 0..<digits {
            guard let ch = advance() else {
                throw TOMLParseError.syntaxError(line: line, message: "Incomplete unicode escape")
            }
            hex.unicodeScalars.append(ch)
        }
        guard let code = UInt32(hex, radix: 16), let scalar = UnicodeScalar(code) else {
            throw TOMLParseError.syntaxError(line: line, message: "Invalid unicode escape '\\u\(hex)'")
        }
        return scalar
    }

    private mutating func parseLiteralString(allowMultiline: Bool) throws -> String {
        let startLine = line
        _ = advance() // consume opening '
        if allowMultiline && peek() == "'" && peek(1) == "'" {
            _ = advance()
            _ = advance()
            if peek() == "\r" && peek(1) == "\n" {
                _ = advance(); _ = advance()
            } else if peek() == "\n" {
                _ = advance()
            }
            var out = ""
            while let c = peek() {
                if c == "'" && peek(1) == "'" && peek(2) == "'" {
                    _ = advance(); _ = advance(); _ = advance()
                    var extra = 0
                    while peek() == "'" && extra < 2 {
                        out.append("'")
                        _ = advance()
                        extra += 1
                    }
                    return out
                }
                if let ch = advance() { out.unicodeScalars.append(ch) }
            }
            throw TOMLParseError.syntaxError(line: startLine, message: "Unterminated multiline literal string")
        }

        var out = ""
        while let c = peek() {
            if c == "'" {
                _ = advance()
                return out
            }
            if c == "\n" {
                throw TOMLParseError.syntaxError(line: startLine, message: "Newline inside single-line literal string")
            }
            if let ch = advance() { out.unicodeScalars.append(ch) }
        }
        throw TOMLParseError.syntaxError(line: startLine, message: "Unterminated literal string")
    }

    private mutating func parseArray() throws -> TOMLValue {
        let startLine = line
        _ = advance() // [
        var items: [TOMLValue] = []
        while true {
            skipSpaceAndComments(allowNewlines: true)
            if peek() == "]" {
                _ = advance()
                return .array(items)
            }
            if isEOF {
                throw TOMLParseError.syntaxError(line: startLine, message: "Unterminated array")
            }
            items.append(try parseValue())
            skipSpaceAndComments(allowNewlines: true)
            if peek() == "," {
                _ = advance()
                continue
            }
            if peek() == "]" {
                _ = advance()
                return .array(items)
            }
            throw TOMLParseError.syntaxError(line: line, message: "Expected ',' or ']' inside array")
        }
    }

    private mutating func parseInlineTable() throws -> TOMLValue {
        let startLine = line
        _ = advance() // {
        var table: [String: TOMLValue] = [:]
        while true {
            skipHorizontalSpace()
            if peek() == "}" {
                _ = advance()
                return .table(table)
            }
            if isEOF || peek() == "\n" {
                throw TOMLParseError.syntaxError(line: startLine, message: "Unterminated inline table")
            }
            let keyPath = try parseKeyPath()
            skipHorizontalSpace()
            guard peek() == "=" else {
                throw TOMLParseError.syntaxError(line: line, message: "Expected '=' inside inline table")
            }
            _ = advance()
            skipHorizontalSpace()
            let val = try parseValue()
            try Self.setPath(keyPath, value: val, in: &table)
            skipHorizontalSpace()
            if peek() == "," {
                _ = advance()
                skipHorizontalSpace()
                if peek() == "}" { // tolerate trailing comma defensively
                    _ = advance()
                    return .table(table)
                }
                continue
            }
            if peek() == "}" {
                _ = advance()
                return .table(table)
            }
            throw TOMLParseError.syntaxError(line: line, message: "Expected ',' or '}' inside inline table")
        }
    }

    private mutating func parseScalar() throws -> TOMLValue {
        var raw = ""
        while let c = peek(), c != " " && c != "\t" && c != "\r" && c != "\n" && c != "#" && c != "," && c != "]" && c != "}" {
            raw.unicodeScalars.append(c)
            _ = advance()
        }
        guard !raw.isEmpty else {
            throw TOMLParseError.syntaxError(line: line, message: "Expected TOML value")
        }

        if raw == "true" { return .bool(true) }
        if raw == "false" { return .bool(false) }
        if raw == "inf" || raw == "+inf" { return .double(.infinity) }
        if raw == "-inf" { return .double(-.infinity) }
        if raw == "nan" || raw == "+nan" || raw == "-nan" { return .double(.nan) }

        let clean = raw.replacingOccurrences(of: "_", with: "")
        if clean.hasPrefix("0x"), let v = Int(clean.dropFirst(2), radix: 16) {
            return .int(v)
        }
        if clean.hasPrefix("0o"), let v = Int(clean.dropFirst(2), radix: 8) {
            return .int(v)
        }
        if clean.hasPrefix("0b"), let v = Int(clean.dropFirst(2), radix: 2) {
            return .int(v)
        }
        if !clean.contains("."), !clean.contains("e"), !clean.contains("E"), let i = Int(clean) {
            return .int(i)
        }
        if let d = Double(clean) {
            return .double(d)
        }
        // Fallback for ISO dates/times (RFC 3339) -> preserve as string
        if raw.contains("-") || raw.contains(":") {
            return .string(raw)
        }
        throw TOMLParseError.syntaxError(line: line, message: "Invalid TOML value '\(raw)'")
    }

    // MARK: - Table Insertion Helpers

    private mutating func openTableHeader(at path: [String]) throws -> CurrentTarget {
        // Check if path starts with a previously opened array-of-tables (e.g. `[[models]]` then `[models.config]`).
        for prefixLen in stride(from: path.count - 1, through: 1, by: -1) {
            let prefix = Array(path.prefix(prefixLen))
            if knownArrayOfTables.contains(prefix) {
                let tail = Array(path.dropFirst(prefixLen))
                guard let count = arrayLength(at: prefix), count > 0 else {
                    throw TOMLParseError.invalidTableRedefinition(path: path.joined(separator: "."))
                }
                try ensureSubtable(tail, inArrayAt: prefix, index: count - 1)
                return .arrayTableItem(arrayPath: prefix, index: count - 1, subPath: tail)
            }
        }
        try Self.ensureTable(at: path, in: &root)
        return .table(path)
    }

    private mutating func appendArrayTableItem(at path: [String]) throws -> Int {
        knownArrayOfTables.insert(path)
        return try Self.appendEmptyTable(at: path, in: &root)
    }

    private func arrayLength(at path: [String]) -> Int? {
        var current: [String: TOMLValue] = root
        for idx in 0..<(path.count - 1) {
            guard case .table(let next) = current[path[idx]] else { return nil }
            current = next
        }
        guard let lastKey = path.last, case .array(let items) = current[lastKey] else { return nil }
        return items.count
    }

    private mutating func ensureSubtable(_ subPath: [String], inArrayAt arrayPath: [String], index: Int) throws {
        guard !subPath.isEmpty else { return }
        try mutateArrayItem(at: arrayPath, index: index) { item in
            try Self.ensureTable(at: subPath, in: &item)
        }
    }

    private mutating func insertValue(_ value: TOMLValue, atRelativeKeyPath keyPath: [String]) throws {
        switch currentTarget {
        case .root:
            try Self.setPath(keyPath, value: value, in: &root)
        case .table(let tablePath):
            try Self.setPath(tablePath + keyPath, value: value, in: &root)
        case .arrayTableItem(let arrayPath, let index, let subPath):
            try mutateArrayItem(at: arrayPath, index: index) { item in
                try Self.setPath(subPath + keyPath, value: value, in: &item)
            }
        }
    }

    private static func appendEmptyTable(at path: [String], in dict: inout [String: TOMLValue]) throws -> Int {
        guard let head = path.first else {
            throw TOMLParseError.invalidTableRedefinition(path: "")
        }
        if path.count == 1 {
            var arr: [TOMLValue]
            if let existing = dict[head] {
                guard case .array(let items) = existing else {
                    throw TOMLParseError.invalidTableRedefinition(path: head)
                }
                arr = items
            } else {
                arr = []
            }
            arr.append(.table([:]))
            dict[head] = .array(arr)
            return arr.count - 1
        } else {
            var sub: [String: TOMLValue]
            if let existing = dict[head] {
                guard case .table(let t) = existing else {
                    throw TOMLParseError.invalidTableRedefinition(path: head)
                }
                sub = t
            } else {
                sub = [:]
            }
            let newIdx = try appendEmptyTable(at: Array(path.dropFirst()), in: &sub)
            dict[head] = .table(sub)
            return newIdx
        }
    }

    private mutating func mutateArrayItem(
        at arrayPath: [String],
        index: Int,
        _ body: (inout [String: TOMLValue]) throws -> Void
    ) throws {
        func recurse(_ remaining: ArraySlice<String>, in dict: inout [String: TOMLValue]) throws {
            guard let head = remaining.first else { return }
            if remaining.count == 1 {
                guard case .array(var items) = dict[head], index < items.count, case .table(var itemDict) = items[index] else {
                    throw TOMLParseError.invalidTableRedefinition(path: arrayPath.joined(separator: "."))
                }
                try body(&itemDict)
                items[index] = .table(itemDict)
                dict[head] = .array(items)
            } else {
                guard case .table(var sub) = dict[head] else {
                    throw TOMLParseError.invalidTableRedefinition(path: arrayPath.joined(separator: "."))
                }
                try recurse(remaining.dropFirst(), in: &sub)
                dict[head] = .table(sub)
            }
        }
        try recurse(arrayPath[...], in: &root)
    }

    private static func ensureTable(at path: [String], in dict: inout [String: TOMLValue]) throws {
        guard let head = path.first else { return }
        if path.count == 1 {
            if let existing = dict[head] {
                guard case .table = existing else {
                    throw TOMLParseError.invalidTableRedefinition(path: head)
                }
            } else {
                dict[head] = .table([:])
            }
        } else {
            var sub: [String: TOMLValue]
            if let existing = dict[head] {
                guard case .table(let t) = existing else {
                    throw TOMLParseError.invalidTableRedefinition(path: head)
                }
                sub = t
            } else {
                sub = [:]
            }
            try ensureTable(at: Array(path.dropFirst()), in: &sub)
            dict[head] = .table(sub)
        }
    }

    private static func setPath(_ path: [String], value: TOMLValue, in dict: inout [String: TOMLValue]) throws {
        guard let head = path.first else { return }
        if path.count == 1 {
            if dict[head] != nil {
                throw TOMLParseError.duplicateKey(key: head)
            }
            dict[head] = value
        } else {
            var sub: [String: TOMLValue]
            if let existing = dict[head] {
                guard case .table(let t) = existing else {
                    throw TOMLParseError.duplicateKey(key: head)
                }
                sub = t
            } else {
                sub = [:]
            }
            try setPath(Array(path.dropFirst()), value: value, in: &sub)
            dict[head] = .table(sub)
        }
    }
}

// MARK: - Line-Aware Surgical Scanner

private struct StatementScanner {
    struct SectionRange {
        let startLine: Int // first line INSIDE section (after header)
        let endLine: Int   // exclusive
        let lastNonBlankContentLine: Int // index in `lines` to append after
    }

    struct KeyStatement {
        let startLine: Int
        let endLine: Int
        let inlineComment: String?
    }

    let lines: [String]

    func findSectionRange(named section: String?) -> SectionRange? {
        if section == nil {
            // Root section: from line 0 until the first [table] or [[array]] header
            var end = lines.count
            for (idx, line) in lines.enumerated() {
                let t = line.trimmingCharacters(in: .whitespaces)
                if t.hasPrefix("[") && !t.hasPrefix("#") && isHeaderLine(t) {
                    end = idx
                    break
                }
            }
            let lastContent = findLastContentLine(in: 0..<end, defaultAnchor: -1)
            return SectionRange(startLine: 0, endLine: end, lastNonBlankContentLine: lastContent)
        }

        guard let target = section else { return nil }
        var startIdx: Int?
        var endIdx = lines.count
        for (idx, raw) in lines.enumerated() {
            let trimmed = stripHeaderComment(raw).trimmingCharacters(in: .whitespaces)
            if trimmed == "[\(target)]" {
                startIdx = idx + 1
                continue
            }
            if startIdx != nil, isHeaderLine(trimmed) {
                endIdx = idx
                break
            }
        }
        guard let start = startIdx else { return nil }
        let lastContent = findLastContentLine(in: start..<endIdx, defaultAnchor: start - 1)
        return SectionRange(startLine: start, endLine: endIdx, lastNonBlankContentLine: lastContent)
    }

    func firstArrayOfTablesLineIndex() -> Int? {
        for (idx, raw) in lines.enumerated() {
            let t = stripHeaderComment(raw).trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("[[") && t.hasSuffix("]]") {
                return idx
            }
        }
        return nil
    }

    func findKeyStatement(named targetKey: String, in range: SectionRange) -> KeyStatement? {
        var i = range.startLine
        while i < range.endLine {
            let raw = lines[i]
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                i += 1
                continue
            }
            if isHeaderLine(trimmed) {
                break
            }
            if let (lhsKey, rhsStartOffset) = parseKeyPrefix(in: raw) {
                let endLine = findStatementEndLine(startingAt: i, rhsOffset: rhsStartOffset, limit: range.endLine)
                if lhsKey == targetKey {
                    let inlineComment: String? = (i == endLine) ? extractInlineComment(from: raw) : nil
                    return KeyStatement(startLine: i, endLine: endLine, inlineComment: inlineComment)
                }
                i = endLine + 1
            } else {
                i += 1
            }
        }
        return nil
    }

    private func findLastContentLine(in range: Range<Int>, defaultAnchor: Int) -> Int {
        var last = defaultAnchor
        for idx in range {
            let t = lines[idx].trimmingCharacters(in: .whitespaces)
            if !t.isEmpty && !t.hasPrefix("#") {
                last = idx
            }
        }
        // Wait: if trailing comments sit just above the next section header, don't append AFTER comments meant for next section;
        // `last` already points at the final actual `key = value` line inside the section!
        return last
    }

    private func isHeaderLine(_ trimmed: String) -> Bool {
        let bare = stripHeaderComment(trimmed).trimmingCharacters(in: .whitespaces)
        return (bare.hasPrefix("[[") && bare.hasSuffix("]]"))
            || (bare.hasPrefix("[") && bare.hasSuffix("]") && !bare.contains("="))
    }

    private func stripHeaderComment(_ line: String) -> String {
        var inBasic = false
        var inLiteral = false
        var escaped = false
        var out = ""
        for c in line {
            if escaped { out.append(c); escaped = false; continue }
            if c == "\\" && inBasic { out.append(c); escaped = true; continue }
            if c == "\"" && !inLiteral { inBasic.toggle(); out.append(c); continue }
            if c == "'" && !inBasic { inLiteral.toggle(); out.append(c); continue }
            if c == "#" && !inBasic && !inLiteral { break }
            out.append(c)
        }
        return out
    }

    private func parseKeyPrefix(in line: String) -> (String, String.Index)? {
        var idx = line.startIndex
        while idx < line.endIndex && (line[idx] == " " || line[idx] == "\t") {
            idx = line.index(after: idx)
        }
        guard idx < line.endIndex else { return nil }
        var key = ""
        if line[idx] == "\"" {
            idx = line.index(after: idx)
            var esc = false
            while idx < line.endIndex {
                let c = line[idx]
                idx = line.index(after: idx)
                if esc { key.append(c); esc = false; continue }
                if c == "\\" { esc = true; continue }
                if c == "\"" { break }
                key.append(c)
            }
        } else if line[idx] == "'" {
            idx = line.index(after: idx)
            while idx < line.endIndex {
                let c = line[idx]
                idx = line.index(after: idx)
                if c == "'" { break }
                key.append(c)
            }
        } else {
            while idx < line.endIndex && (line[idx].isLetter || line[idx].isNumber || line[idx] == "_" || line[idx] == "-") {
                key.append(line[idx])
                idx = line.index(after: idx)
            }
        }
        guard !key.isEmpty else { return nil }
        while idx < line.endIndex && (line[idx] == " " || line[idx] == "\t") {
            idx = line.index(after: idx)
        }
        guard idx < line.endIndex && line[idx] == "=" else { return nil }
        idx = line.index(after: idx)
        return (key, idx)
    }

    private func findStatementEndLine(startingAt startLine: Int, rhsOffset: String.Index, limit: Int) -> Int {
        var bracketDepth = 0
        var braceDepth = 0
        var inMultilineBasic = false
        var inMultilineLiteral = false

        for lineIdx in startLine..<limit {
            let raw = lines[lineIdx]
            var idx = (lineIdx == startLine) ? rhsOffset : raw.startIndex
            var inSingleBasic = false
            var inSingleLiteral = false
            var escaped = false

            while idx < raw.endIndex {
                if inMultilineBasic {
                    if escaped {
                        escaped = false
                        idx = raw.index(after: idx)
                        continue
                    }
                    if raw[idx] == "\\" {
                        escaped = true
                        idx = raw.index(after: idx)
                        continue
                    }
                    if raw[idx...].hasPrefix("\"\"\"") {
                        inMultilineBasic = false
                        idx = raw.index(idx, offsetBy: 3)
                        continue
                    }
                    idx = raw.index(after: idx)
                    continue
                }
                if inMultilineLiteral {
                    if raw[idx...].hasPrefix("'''") {
                        inMultilineLiteral = false
                        idx = raw.index(idx, offsetBy: 3)
                        continue
                    }
                    idx = raw.index(after: idx)
                    continue
                }
                if inSingleBasic {
                    if escaped { escaped = false; idx = raw.index(after: idx); continue }
                    if raw[idx] == "\\" { escaped = true; idx = raw.index(after: idx); continue }
                    if raw[idx] == "\"" { inSingleBasic = false }
                    idx = raw.index(after: idx)
                    continue
                }
                if inSingleLiteral {
                    if raw[idx] == "'" { inSingleLiteral = false }
                    idx = raw.index(after: idx)
                    continue
                }

                let c = raw[idx]
                if c == "#" { break }
                if c == "\"" {
                    if raw[idx...].hasPrefix("\"\"\"") {
                        inMultilineBasic = true
                        idx = raw.index(idx, offsetBy: 3)
                        continue
                    }
                    inSingleBasic = true
                } else if c == "'" {
                    if raw[idx...].hasPrefix("'''") {
                        inMultilineLiteral = true
                        idx = raw.index(idx, offsetBy: 3)
                        continue
                    }
                    inSingleLiteral = true
                } else if c == "[" {
                    bracketDepth += 1
                } else if c == "]" {
                    bracketDepth = max(0, bracketDepth - 1)
                } else if c == "{" {
                    braceDepth += 1
                } else if c == "}" {
                    braceDepth = max(0, braceDepth - 1)
                }
                idx = raw.index(after: idx)
            }

            if !inMultilineBasic && !inMultilineLiteral && bracketDepth == 0 && braceDepth == 0 {
                return lineIdx
            }
        }
        return max(startLine, limit - 1)
    }

    private func extractInlineComment(from line: String) -> String? {
        var inBasic = false
        var inLiteral = false
        var escaped = false
        var idx = line.startIndex
        var lastNonSpaceBeforeComment = line.startIndex
        while idx < line.endIndex {
            let c = line[idx]
            if escaped { escaped = false; idx = line.index(after: idx); lastNonSpaceBeforeComment = idx; continue }
            if c == "\\" && inBasic { escaped = true; idx = line.index(after: idx); continue }
            if c == "\"" && !inLiteral { inBasic.toggle(); idx = line.index(after: idx); lastNonSpaceBeforeComment = idx; continue }
            if c == "'" && !inBasic { inLiteral.toggle(); idx = line.index(after: idx); lastNonSpaceBeforeComment = idx; continue }
            if c == "#" && !inBasic && !inLiteral {
                let wsSlice = String(line[lastNonSpaceBeforeComment..<idx])
                let spacing = wsSlice.isEmpty ? "  " : wsSlice
                return spacing + String(line[idx...])
            }
            if c != " " && c != "\t" {
                lastNonSpaceBeforeComment = line.index(after: idx)
            }
            idx = line.index(after: idx)
        }
        return nil
    }
}
