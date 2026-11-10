import Foundation
import SwiftUI

#if os(macOS)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

// MARK: - Structured GFM Markdown Block AST & Safe-Boundary Streaming Renderer

/// Renders GitHub-Flavored Markdown (headings, fenced code blocks with one-click Copy,
/// pipe tables with column alignment, blockquotes, ordered/unordered/task lists, horizontal rules,
/// and inline formatting) while preserving literal `<angle-bracket>` text (such as `Vec<T>` or XML tags)
/// and throttling active streaming tail layout to ~60ms with stable prefix block commits.
public struct MarkdownMessageView: View {
    public let text: String
    public var isStreaming: Bool = false

    @State private var committedCharCount: Int = 0
    @State private var committedBlocks: [MarkdownBlock] = []
    @State private var activeTailBlocks: [MarkdownBlock] = []
    @State private var throttleTask: Task<Void, Never>?
    @State private var lastThrottleTick: ContinuousClock.Instant = .now - .milliseconds(100)

    public init(text: String, isStreaming: Bool = false) {
        self.text = text
        self.isStreaming = isStreaming
    }

    public var body: some View {
        Group {
            if isStreaming {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(committedBlocks) { block in
                        MarkdownBlockView(block: block)
                    }
                    ForEach(activeTailBlocks) { block in
                        MarkdownBlockView(block: block)
                    }
                }
                .onAppear {
                    synchronizeStreamingNow(fullText: text)
                }
                .onChange(of: text) { _, newValue in
                    scheduleStreamingUpdate(fullText: newValue)
                }
                .onDisappear {
                    throttleTask?.cancel()
                    throttleTask = nil
                }
            } else {
                StaticMarkdownBlocksView(text: text)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func scheduleStreamingUpdate(fullText: String) {
        let elapsed = ContinuousClock.now - lastThrottleTick
        if elapsed >= .milliseconds(60) {
            throttleTask?.cancel()
            throttleTask = nil
            synchronizeStreamingNow(fullText: fullText)
            return
        }

        guard throttleTask == nil else { return }
        let remaining = Duration.milliseconds(60) - elapsed
        throttleTask = Task { @MainActor in
            try? await Task.sleep(for: remaining)
            guard !Task.isCancelled else { return }
            throttleTask = nil
            synchronizeStreamingNow(fullText: self.text)
        }
    }

    private func synchronizeStreamingNow(fullText: String) {
        lastThrottleTick = .now
        if fullText.count < committedCharCount {
            // Stream restarted or reset
            committedCharCount = 0
            committedBlocks = []
        }

        let splitUTF16Offset = MarkdownParser.lastSafeBlockBoundaryUTF16Offset(in: fullText)
        let nsText = fullText as NSString
        let totalUTF16 = nsText.length
        let clampedOffset = min(max(0, splitUTF16Offset), totalUTF16)

        let prefixString = clampedOffset > 0 ? nsText.substring(to: clampedOffset) : ""
        if prefixString.count != committedCharCount {
            committedBlocks = MarkdownParser.parseBlocks(prefixString, idOffset: 0)
            committedCharCount = prefixString.count
        }

        let tailString = clampedOffset < totalUTF16 ? nsText.substring(from: clampedOffset) : ""
        activeTailBlocks = MarkdownParser.parseBlocks(tailString, idOffset: 1_000_000)
    }
}

private struct StaticMarkdownBlocksView: View, Equatable {
    let text: String

    nonisolated static func == (lhs: StaticMarkdownBlocksView, rhs: StaticMarkdownBlocksView) -> Bool {
        lhs.text == rhs.text
    }

    var body: some View {
        let blocks = MarkdownParser.parseBlocks(text, idOffset: 0)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(blocks) { block in
                MarkdownBlockView(block: block)
            }
        }
        .textSelection(.enabled)
    }
}

// MARK: - AST Models

enum MarkdownTableAlignment: Sendable, Equatable {
    case leading
    case center
    case trailing

    var horizontalAlignment: HorizontalAlignment {
        switch self {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }

    var frameAlignment: Alignment {
        switch self {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }
}

struct MarkdownListItem: Identifiable, Sendable, Equatable {
    let id: Int
    let indentLevel: Int
    let marker: Marker
    let text: String

    enum Marker: Sendable, Equatable {
        case bullet
        case ordered(Int)
        case task(checked: Bool)
    }
}

struct MarkdownBlock: Identifiable, Sendable, Equatable {
    let id: Int
    let kind: Kind

    enum Kind: Sendable, Equatable {
        case heading(level: Int, text: String)
        case paragraph(String)
        case codeBlock(language: String?, code: String, isClosed: Bool)
        case blockquote(String)
        case list([MarkdownListItem])
        case table(headers: [String], alignments: [MarkdownTableAlignment], rows: [[String]])
        case thematicBreak
    }
}

// MARK: - Block & Inline GFM Parser

enum MarkdownParser {
    /// Finds the UTF-16 offset right after the last safe `\n\n` boundary that is NOT inside an
    /// unclosed fenced code block and NOT inside a contiguous pipe table or list block.
    static func lastSafeBlockBoundaryUTF16Offset(in text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        let lines = text.components(separatedBy: "\n")
        var inFence = false
        var fenceChar: Character = "`"
        var fenceLen = 0
        var utf16Cursor = 0
        var lastSafeUTF16 = 0
        var previousLineWasBlank = false

        for (idx, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let fence = parseFenceDelimiter(trimmed) {
                if !inFence {
                    inFence = true
                    fenceChar = fence.char
                    fenceLen = fence.length
                } else if fence.char == fenceChar && fence.length >= fenceLen && fence.info.isEmpty {
                    inFence = false
                }
            }

            let lineUTF16 = (line as NSString).length
            let newlineUTF16 = idx < lines.count - 1 ? 1 : 0
            utf16Cursor += lineUTF16 + newlineUTF16

            if !inFence && trimmed.isEmpty {
                if idx > 0 {
                    lastSafeUTF16 = utf16Cursor
                }
                previousLineWasBlank = true
            } else {
                _ = previousLineWasBlank
                previousLineWasBlank = false
            }
        }

        return lastSafeUTF16
    }

    static func parseBlocks(_ rawText: String, idOffset: Int) -> [MarkdownBlock] {
        let normalized = rawText.replacingOccurrences(of: "\r\n", with: "\n")
        guard !normalized.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return []
        }

        let lines = normalized.components(separatedBy: "\n")
        var blocks: [MarkdownBlock] = []
        var nextID = idOffset
        var i = 0

        func emit(_ kind: MarkdownBlock.Kind) {
            blocks.append(MarkdownBlock(id: nextID, kind: kind))
            nextID += 1
        }

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.isEmpty {
                i += 1
                continue
            }

            // 1. Fenced Code Block (``` or ~~~)
            if let fence = parseFenceDelimiter(trimmed) {
                let lang = fence.info.isEmpty ? nil : fence.info
                var codeLines: [String] = []
                var closed = false
                i += 1
                while i < lines.count {
                    let candidate = lines[i]
                    let candidateTrimmed = candidate.trimmingCharacters(in: .whitespaces)
                    if let closeFence = parseFenceDelimiter(candidateTrimmed),
                       closeFence.char == fence.char,
                       closeFence.length >= fence.length,
                       closeFence.info.isEmpty {
                        closed = true
                        i += 1
                        break
                    }
                    codeLines.append(candidate)
                    i += 1
                }
                emit(.codeBlock(language: lang, code: codeLines.joined(separator: "\n"), isClosed: closed))
                continue
            }

            // 2. Thematic Break (---, ***, ___)
            if isThematicBreak(trimmed) {
                emit(.thematicBreak)
                i += 1
                continue
            }

            // 3. ATX Heading (# .. ######)
            if let heading = parseATXHeading(trimmed) {
                emit(.heading(level: heading.level, text: heading.text))
                i += 1
                continue
            }

            // 4. GFM Pipe Table (Header + Separator row)
            if i + 1 < lines.count,
               line.contains("|"),
               let alignments = parseTableSeparatorRow(lines[i + 1]) {
                let headers = splitTableRow(line)
                if !headers.isEmpty && headers.count == alignments.count {
                    var rows: [[String]] = []
                    i += 2
                    while i < lines.count {
                        let rowLine = lines[i]
                        let rowTrimmed = rowLine.trimmingCharacters(in: .whitespaces)
                        if rowTrimmed.isEmpty || !rowLine.contains("|") || parseFenceDelimiter(rowTrimmed) != nil {
                            break
                        }
                        var cells = splitTableRow(rowLine)
                        if cells.count < headers.count {
                            cells.append(contentsOf: Array(repeating: "", count: headers.count - cells.count))
                        } else if cells.count > headers.count {
                            cells = Array(cells.prefix(headers.count))
                        }
                        rows.append(cells)
                        i += 1
                    }
                    emit(.table(headers: headers, alignments: alignments, rows: rows))
                    continue
                }
            }

            // 5. Blockquote (> ...)
            if trimmed.hasPrefix(">") {
                var quoteLines: [String] = []
                while i < lines.count {
                    let qTrimmed = lines[i].trimmingCharacters(in: .whitespaces)
                    if qTrimmed.hasPrefix(">") {
                        var stripped = String(qTrimmed.dropFirst())
                        if stripped.hasPrefix(" ") {
                            stripped.removeFirst()
                        }
                        quoteLines.append(stripped)
                        i += 1
                    } else {
                        break
                    }
                }
                emit(.blockquote(quoteLines.joined(separator: "\n")))
                continue
            }

            // 6. Lists (Unordered, Ordered, Task)
            if parseListLine(line, itemID: 0) != nil {
                var items: [MarkdownListItem] = []
                var itemIndex = 0
                while i < lines.count {
                    let current = lines[i]
                    let currentTrimmed = current.trimmingCharacters(in: .whitespaces)
                    if currentTrimmed.isEmpty {
                        // Lookahead: if next non-empty line is another list item, continue list; else end
                        if i + 1 < lines.count, parseListLine(lines[i + 1], itemID: 0) != nil {
                            i += 1
                            continue
                        }
                        break
                    }
                    if let parsedItem = parseListLine(current, itemID: itemIndex) {
                        items.append(parsedItem)
                        itemIndex += 1
                        i += 1
                    } else if !items.isEmpty && (current.hasPrefix("  ") || current.hasPrefix("\t")) {
                        // Continuation line of previous list item
                        let last = items.removeLast()
                        items.append(
                            MarkdownListItem(
                                id: last.id,
                                indentLevel: last.indentLevel,
                                marker: last.marker,
                                text: last.text + "\n" + currentTrimmed
                            )
                        )
                        i += 1
                    } else {
                        break
                    }
                }
                emit(.list(items))
                continue
            }

            // 7. Paragraph (accumulate until blank line or block starter)
            var paraLines: [String] = [line]
            i += 1
            while i < lines.count {
                let nextLine = lines[i]
                let nextTrimmed = nextLine.trimmingCharacters(in: .whitespaces)
                if nextTrimmed.isEmpty
                    || parseFenceDelimiter(nextTrimmed) != nil
                    || isThematicBreak(nextTrimmed)
                    || parseATXHeading(nextTrimmed) != nil
                    || nextTrimmed.hasPrefix(">")
                    || parseListLine(nextLine, itemID: 0) != nil {
                    break
                }
                if i + 1 < lines.count,
                   nextLine.contains("|"),
                   let aligns = parseTableSeparatorRow(lines[i + 1]),
                   splitTableRow(nextLine).count == aligns.count {
                    break
                }
                paraLines.append(nextLine)
                i += 1
            }
            emit(.paragraph(paraLines.joined(separator: "\n")))
        }

        return blocks
    }

    private static func parseFenceDelimiter(_ trimmed: String) -> (char: Character, length: Int, info: String)? {
        guard let first = trimmed.first, first == "`" || first == "~" else { return nil }
        var count = 0
        for ch in trimmed {
            if ch == first { count += 1 } else { break }
        }
        guard count >= 3 else { return nil }
        let rest = String(trimmed.dropFirst(count)).trimmingCharacters(in: .whitespaces)
        // Backtick fence info string cannot contain backticks per CommonMark
        if first == "`" && rest.contains("`") { return nil }
        let langToken = rest.split(separator: " ").first.map(String.init) ?? ""
        return (first, count, langToken)
    }

    private static func isThematicBreak(_ trimmed: String) -> Bool {
        guard trimmed.count >= 3 else { return false }
        let compact = trimmed.filter { !$0.isWhitespace }
        guard compact.count >= 3, let ch = compact.first, ch == "-" || ch == "*" || ch == "_" else {
            return false
        }
        return compact.allSatisfy { $0 == ch }
    }

    private static func parseATXHeading(_ trimmed: String) -> (level: Int, text: String)? {
        guard trimmed.hasPrefix("#") else { return nil }
        var level = 0
        for ch in trimmed {
            if ch == "#" { level += 1 } else { break }
        }
        guard (1...6).contains(level) else { return nil }
        let remainder = trimmed.dropFirst(level)
        guard remainder.isEmpty || remainder.first == " " || remainder.first == "\t" else {
            return nil
        }
        let headingText = remainder.trimmingCharacters(in: .whitespaces)
        return (level, headingText)
    }

    private static func splitTableRow(_ line: String) -> [String] {
        var s = line.trimmingCharacters(in: .whitespaces)
        if s.hasPrefix("|") { s.removeFirst() }
        if s.hasSuffix("|") && !s.hasSuffix("\\|") { s.removeLast() }

        var cells: [String] = []
        var current = ""
        var escaped = false
        var inCode = false
        for ch in s {
            if escaped {
                current.append(ch)
                escaped = false
                continue
            }
            if ch == "\\" {
                escaped = true
                continue
            }
            if ch == "`" {
                inCode.toggle()
                current.append(ch)
                continue
            }
            if ch == "|" && !inCode {
                cells.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            } else {
                current.append(ch)
            }
        }
        cells.append(current.trimmingCharacters(in: .whitespaces))
        return cells
    }

    private static func parseTableSeparatorRow(_ line: String) -> [MarkdownTableAlignment]? {
        guard line.contains("|") && line.contains("-") else { return nil }
        let cells = splitTableRow(line)
        guard !cells.isEmpty else { return nil }

        var alignments: [MarkdownTableAlignment] = []
        for cell in cells {
            let trimmed = cell.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { return nil }
            let startsColon = trimmed.hasPrefix(":")
            let endsColon = trimmed.hasSuffix(":")
            var core = trimmed
            if startsColon { core.removeFirst() }
            if endsColon && !core.isEmpty { core.removeLast() }
            guard core.count >= 3 && core.allSatisfy({ $0 == "-" }) else { return nil }

            if startsColon && endsColon {
                alignments.append(.center)
            } else if endsColon {
                alignments.append(.trailing)
            } else {
                alignments.append(.leading)
            }
        }
        return alignments
    }

    private static func parseListLine(_ line: String, itemID: Int) -> MarkdownListItem? {
        var leadingSpaces = 0
        for ch in line {
            if ch == " " { leadingSpaces += 1 }
            else if ch == "\t" { leadingSpaces += 4 }
            else { break }
        }
        let indentLevel = min(4, leadingSpaces / 2)
        let trimmed = line.trimmingCharacters(in: .whitespaces)

        // Unordered / Task list: -, *, +
        if trimmed.count >= 2,
           let first = trimmed.first,
           (first == "-" || first == "*" || first == "+") {
            let afterBullet = trimmed.dropFirst()
            if afterBullet.first == " " || afterBullet.first == "\t" {
                let body = String(afterBullet.dropFirst()).trimmingCharacters(in: .whitespaces)
                if body.hasPrefix("[ ] ") || body == "[ ]" {
                    let taskBody = String(body.dropFirst(min(4, body.count)))
                    return MarkdownListItem(id: itemID, indentLevel: indentLevel, marker: .task(checked: false), text: taskBody)
                } else if body.hasPrefix("[x] ") || body.hasPrefix("[X] ") || body == "[x]" || body == "[X]" {
                    let taskBody = String(body.dropFirst(min(4, body.count)))
                    return MarkdownListItem(id: itemID, indentLevel: indentLevel, marker: .task(checked: true), text: taskBody)
                }
                return MarkdownListItem(id: itemID, indentLevel: indentLevel, marker: .bullet, text: body)
            }
        }

        // Ordered list: 1. or 1)
        var digits = ""
        var idx = trimmed.startIndex
        while idx < trimmed.endIndex, trimmed[idx].isNumber, digits.count < 9 {
            digits.append(trimmed[idx])
            idx = trimmed.index(after: idx)
        }
        if !digits.isEmpty, idx < trimmed.endIndex, (trimmed[idx] == "." || trimmed[idx] == ")") {
            let afterDelimiter = trimmed.index(after: idx)
            if afterDelimiter < trimmed.endIndex && (trimmed[afterDelimiter] == " " || trimmed[afterDelimiter] == "\t") {
                let body = String(trimmed[trimmed.index(after: afterDelimiter)...]).trimmingCharacters(in: .whitespaces)
                let num = Int(digits) ?? 1
                return MarkdownListItem(id: itemID, indentLevel: indentLevel, marker: .ordered(num), text: body)
            }
        }

        return nil
    }

    // MARK: - Inline GFM Parser (Preserves Raw `<tag>` Text & Soft Line Breaks)

    static func parseInline(_ source: String, baseFont: Font = .body) -> AttributedString {
        var result = AttributedString()
        let chars = Array(source)
        let n = chars.count
        var i = 0
        var plainBuffer = ""

        func flushPlain() {
            guard !plainBuffer.isEmpty else { return }
            var chunk = AttributedString(plainBuffer)
            chunk.font = baseFont
            result.append(chunk)
            plainBuffer.removeAll(keepingCapacity: true)
        }

        while i < n {
            let ch = chars[i]

            // Backslash escape for markdown punctuation
            if ch == "\\" && i + 1 < n {
                let next = chars[i + 1]
                if "\\`*_{}[]()#+-.!|~".contains(next) {
                    plainBuffer.append(next)
                    i += 2
                    continue
                }
            }

            // 1. Inline code span (`...` or ``...``)
            if ch == "`" {
                var tickCount = 0
                var j = i
                while j < n && chars[j] == "`" {
                    tickCount += 1
                    j += 1
                }
                if let closeIndex = findClosingTicks(chars, from: j, count: tickCount) {
                    flushPlain()
                    var codeContent = String(chars[j..<closeIndex])
                    if codeContent.count >= 2 && codeContent.hasPrefix(" ") && codeContent.hasSuffix(" ") {
                        codeContent.removeFirst()
                        codeContent.removeLast()
                    }
                    var codeAttr = AttributedString(codeContent)
                    codeAttr.font = .system(.callout, design: .monospaced)
                    codeAttr.backgroundColor = Color.secondary.opacity(0.16)
                    result.append(codeAttr)
                    i = closeIndex + tickCount
                    continue
                } else {
                    // Unclosed ticks — treat literally
                    for _ in 0..<tickCount { plainBuffer.append("`") }
                    i = j
                    continue
                }
            }

            // 2. Link [label](url)
            if ch == "[", let link = parseInlineLink(chars, from: i) {
                flushPlain()
                var linkAttr = parseInline(link.label, baseFont: baseFont)
                if let url = link.safeURL {
                    linkAttr.link = url
                    linkAttr.foregroundColor = .accentColor
                    linkAttr.underlineStyle = .single
                }
                result.append(linkAttr)
                i = link.nextIndex
                continue
            }

            // 3. Bold + Italic (***...***)
            if i + 2 < n && chars[i] == "*" && chars[i + 1] == "*" && chars[i + 2] == "*",
               let close = findDelimiter(chars, delimiter: ["*", "*", "*"], from: i + 3),
               close > i + 3 {
                flushPlain()
                var inner = parseInline(String(chars[(i + 3)..<close]), baseFont: baseFont.bold().italic())
                inner.inlinePresentationIntent = [.stronglyEmphasized, .emphasized]
                result.append(inner)
                i = close + 3
                continue
            }

            // 4. Bold (**...** or __...__)
            if i + 1 < n && ((chars[i] == "*" && chars[i + 1] == "*") || (chars[i] == "_" && chars[i + 1] == "_")) {
                let delim = chars[i]
                let prevIsWord = i > 0 && (chars[i - 1].isLetter || chars[i - 1].isNumber)
                if delim == "*" || !prevIsWord,
                   let close = findDelimiter(chars, delimiter: [delim, delim], from: i + 2),
                   close > i + 2 {
                    let afterCloseIdx = close + 2
                    let nextIsWord = afterCloseIdx < n && (chars[afterCloseIdx].isLetter || chars[afterCloseIdx].isNumber)
                    if delim == "*" || !nextIsWord {
                        flushPlain()
                        var inner = parseInline(String(chars[(i + 2)..<close]), baseFont: baseFont.bold())
                        inner.inlinePresentationIntent = .stronglyEmphasized
                        result.append(inner)
                        i = afterCloseIdx
                        continue
                    }
                }
            }

            // 5. Strikethrough (~~...~~)
            if i + 1 < n && chars[i] == "~" && chars[i + 1] == "~",
               let close = findDelimiter(chars, delimiter: ["~", "~"], from: i + 2),
               close > i + 2 {
                flushPlain()
                var inner = parseInline(String(chars[(i + 2)..<close]), baseFont: baseFont)
                inner.strikethroughStyle = .single
                result.append(inner)
                i = close + 2
                continue
            }

            // 6. Italic (*...* or word-boundary _..._)
            if ch == "*" || ch == "_" {
                let prevIsWord = i > 0 && (chars[i - 1].isLetter || chars[i - 1].isNumber)
                let nextNotSpace = i + 1 < n && !chars[i + 1].isWhitespace && chars[i + 1] != ch
                if nextNotSpace && (ch == "*" || !prevIsWord),
                   let close = findSingleItalicClosing(chars, delimiter: ch, from: i + 1) {
                    let afterClose = close + 1
                    let nextIsWord = afterClose < n && (chars[afterClose].isLetter || chars[afterClose].isNumber)
                    if ch == "*" || !nextIsWord {
                        flushPlain()
                        var inner = parseInline(String(chars[(i + 1)..<close]), baseFont: baseFont.italic())
                        inner.inlinePresentationIntent = .emphasized
                        result.append(inner)
                        i = afterClose
                        continue
                    }
                }
            }

            plainBuffer.append(ch)
            i += 1
        }

        flushPlain()
        return result
    }

    private static func findClosingTicks(_ chars: [Character], from start: Int, count: Int) -> Int? {
        var i = start
        let n = chars.count
        while i < n {
            if chars[i] == "`" {
                var run = 0
                let runStart = i
                while i < n && chars[i] == "`" {
                    run += 1
                    i += 1
                }
                if run == count {
                    return runStart
                }
            } else {
                i += 1
            }
        }
        return nil
    }

    private static func findDelimiter(_ chars: [Character], delimiter: [Character], from start: Int) -> Int? {
        let dLen = delimiter.count
        guard chars.count >= start + dLen else { return nil }
        var i = start
        while i <= chars.count - dLen {
            if chars[i] == "\\" {
                i += 2
                continue
            }
            if chars[i] == "`" {
                // Skip inline code span inside emphasis
                var ticks = 0
                var j = i
                while j < chars.count && chars[j] == "`" { ticks += 1; j += 1 }
                if let endTicks = findClosingTicks(chars, from: j, count: ticks) {
                    i = endTicks + ticks
                    continue
                }
            }
            var matched = true
            for k in 0..<dLen where chars[i + k] != delimiter[k] {
                matched = false
                break
            }
            if matched {
                return i
            }
            i += 1
        }
        return nil
    }

    private static func findSingleItalicClosing(_ chars: [Character], delimiter: Character, from start: Int) -> Int? {
        var i = start
        let n = chars.count
        while i < n {
            if chars[i] == "\\" {
                i += 2
                continue
            }
            if chars[i] == "\n" {
                return nil
            }
            if chars[i] == delimiter {
                // Avoid matching a double delimiter
                if i + 1 < n && chars[i + 1] == delimiter {
                    i += 2
                    continue
                }
                if i > start && !chars[i - 1].isWhitespace {
                    return i
                }
            }
            i += 1
        }
        return nil
    }

    private static func parseInlineLink(_ chars: [Character], from start: Int) -> (label: String, safeURL: URL?, nextIndex: Int)? {
        let n = chars.count
        var depth = 0
        var i = start
        var closeBracket: Int?
        while i < n {
            if chars[i] == "\\" {
                i += 2
                continue
            }
            if chars[i] == "[" {
                depth += 1
            } else if chars[i] == "]" {
                depth -= 1
                if depth == 0 {
                    closeBracket = i
                    break
                }
            } else if chars[i] == "\n" {
                return nil
            }
            i += 1
        }
        guard let cb = closeBracket, cb + 1 < n, chars[cb + 1] == "(" else { return nil }

        var parenDepth = 0
        var j = cb + 1
        var closeParen: Int?
        while j < n {
            if chars[j] == "\\" {
                j += 2
                continue
            }
            if chars[j] == "(" {
                parenDepth += 1
            } else if chars[j] == ")" {
                parenDepth -= 1
                if parenDepth == 0 {
                    closeParen = j
                    break
                }
            } else if chars[j] == "\n" {
                return nil
            }
            j += 1
        }
        guard let cp = closeParen else { return nil }

        let label = String(chars[(start + 1)..<cb])
        let rawTarget = String(chars[(cb + 2)..<cp]).trimmingCharacters(in: .whitespaces)
        let urlToken = rawTarget.split(separator: " ").first.map(String.init) ?? rawTarget
        var safeURL: URL?
        if let parsed = URL(string: urlToken),
           let scheme = parsed.scheme?.lowercased(),
           ["http", "https", "mailto"].contains(scheme) {
            safeURL = parsed
        }
        return (label, safeURL, cp + 1)
    }
}

// MARK: - Block Views

private struct MarkdownBlockView: View {
    let block: MarkdownBlock

    var body: some View {
        switch block.kind {
        case .heading(let level, let text):
            Text(MarkdownParser.parseInline(text, baseFont: headingFont(for: level)))
                .padding(.top, level <= 2 ? 4 : 2)
                .fixedSize(horizontal: false, vertical: true)

        case .paragraph(let text):
            Text(MarkdownParser.parseInline(text, baseFont: .body))
                .lineSpacing(2)
                .fixedSize(horizontal: false, vertical: true)

        case .codeBlock(let language, let code, _):
            MarkdownCodeBlockView(language: language, code: code)

        case .blockquote(let text):
            HStack(alignment: .top, spacing: 10) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.accentColor.opacity(0.65))
                    .frame(width: 3)
                Text(MarkdownParser.parseInline(text, baseFont: .body))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.vertical, 4)
            .padding(.leading, 2)

        case .list(let items):
            VStack(alignment: .leading, spacing: 5) {
                ForEach(items) { item in
                    HStack(alignment: .firstTextBaseline, spacing: 7) {
                        listMarkerView(item.marker)
                        Text(MarkdownParser.parseInline(item.text, baseFont: .body))
                            .lineSpacing(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.leading, CGFloat(item.indentLevel) * 16)
                }
            }

        case .table(let headers, let alignments, let rows):
            MarkdownTableView(headers: headers, alignments: alignments, rows: rows)

        case .thematicBreak:
            Divider()
                .padding(.vertical, 4)
        }
    }

    private func headingFont(for level: Int) -> Font {
        switch level {
        case 1: return .system(size: 20, weight: .bold)
        case 2: return .system(size: 17, weight: .bold)
        case 3: return .system(size: 15, weight: .semibold)
        case 4: return .system(size: 14, weight: .semibold)
        default: return .system(size: 13, weight: .semibold)
        }
    }

    @ViewBuilder
    private func listMarkerView(_ marker: MarkdownListItem.Marker) -> some View {
        switch marker {
        case .bullet:
            Text("•")
                .font(.body.weight(.bold))
                .foregroundStyle(.secondary)
                .frame(minWidth: 14, alignment: .trailing)
        case .ordered(let number):
            Text("\(number).")
                .font(.callout.monospacedDigit().weight(.medium))
                .foregroundStyle(.secondary)
                .frame(minWidth: 20, alignment: .trailing)
        case .task(let checked):
            Image(systemName: checked ? "checkmark.square.fill" : "square")
                .font(.callout)
                .foregroundStyle(checked ? Color.accentColor : .secondary)
                .frame(minWidth: 16, alignment: .trailing)
        }
    }
}

private struct MarkdownCodeBlockView: View {
    let language: String?
    let code: String
    @State private var didCopy: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Text((language?.isEmpty == false ? language! : "code").lowercased())
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.secondary)

                Spacer()

                Button {
                    copyCodeToClipboard(code)
                    didCopy = true
                    Task { @MainActor in
                        try? await Task.sleep(for: .milliseconds(1500))
                        didCopy = false
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: didCopy ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 10, weight: .semibold))
                        Text(didCopy ? "Copied" : "Copy")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(didCopy ? .green : .secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Copy code block")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.secondary.opacity(0.14))

            Divider()
                .opacity(0.5)

            ScrollView(.horizontal, showsIndicators: true) {
                Text(SyntaxHighlighter.highlight(code: code, language: language))
                    .font(.system(size: 12.5, design: .monospaced))
                    .lineSpacing(2)
                    .textSelection(.enabled)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.secondary.opacity(0.08))
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(Color.secondary.opacity(0.18), lineWidth: 1)
        }
    }

    private func copyCodeToClipboard(_ string: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(string, forType: .string)
        #elseif canImport(UIKit)
        UIPasteboard.general.string = string
        #endif
    }
}

private struct MarkdownTableView: View {
    let headers: [String]
    let alignments: [MarkdownTableAlignment]
    let rows: [[String]]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: true) {
            Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(headers.indices, id: \.self) { col in
                        let align = col < alignments.count ? alignments[col] : .leading
                        Text(MarkdownParser.parseInline(headers[col], baseFont: .callout.weight(.semibold)))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .frame(maxWidth: .infinity, alignment: align.frameAlignment)
                            .background(Color.secondary.opacity(0.14))
                    }
                }

                Divider()

                ForEach(rows.indices, id: \.self) { rowIdx in
                    let row = rows[rowIdx]
                    GridRow {
                        ForEach(headers.indices, id: \.self) { col in
                            let cellText = col < row.count ? row[col] : ""
                            let align = col < alignments.count ? alignments[col] : .leading
                            Text(MarkdownParser.parseInline(cellText, baseFont: .callout))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .frame(maxWidth: .infinity, alignment: align.frameAlignment)
                                .background(rowIdx.isMultiple(of: 2) ? Color.clear : Color.secondary.opacity(0.04))
                        }
                    }
                    if rowIdx < rows.count - 1 {
                        Divider().opacity(0.45)
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.secondary.opacity(0.22), lineWidth: 1)
            }
        }
    }
}

// MARK: - Lightweight Zero-Dependency Code Highlighter

private enum SyntaxHighlighter {
    private static let commonKeywords: Set<String> = [
        "func", "let", "var", "struct", "class", "actor", "enum", "protocol", "extension",
        "import", "return", "if", "else", "guard", "switch", "case", "default", "for", "while",
        "in", "try", "await", "async", "throws", "throw", "defer", "public", "private", "internal",
        "static", "final", "self", "nil", "true", "false", "None", "True", "False", "def", "from",
        "as", "with", "yield", "lambda", "pass", "break", "continue", "const", "function", "export",
        "interface", "type", "impl", "fn", "pub", "mut", "match", "use", "mod", "crate", "where"
    ]

    static func highlight(code: String, language: String?) -> AttributedString {
        // Keep highlighting fast and bounded; very large blocks render cleanly unhighlighted
        guard code.utf8.count <= 24_000 else {
            return AttributedString(code)
        }

        let lang = (language ?? "").lowercased()
        if lang == "text" || lang == "plaintext" || lang == "markdown" || lang == "md" {
            return AttributedString(code)
        }

        let usesHashComment = ["py", "python", "sh", "bash", "zsh", "toml", "yaml", "yml", "rb", "ruby", "r"].contains(lang)
        let usesSlashComment = lang.isEmpty || [
            "swift", "js", "javascript", "ts", "typescript", "jsx", "tsx", "c", "cpp", "c++",
            "h", "hpp", "objc", "rust", "rs", "go", "java", "kotlin", "kt", "jsonc", "metal"
        ].contains(lang)

        var output = AttributedString()
        let lines = code.components(separatedBy: "\n")
        for (lineIdx, line) in lines.enumerated() {
            output.append(highlightLine(line, hashComments: usesHashComment, slashComments: usesSlashComment))
            if lineIdx < lines.count - 1 {
                output.append(AttributedString("\n"))
            }
        }
        return output
    }

    private static func highlightLine(_ line: String, hashComments: Bool, slashComments: Bool) -> AttributedString {
        var out = AttributedString()
        let chars = Array(line)
        let n = chars.count
        var i = 0

        while i < n {
            let ch = chars[i]

            // Line comments
            if (hashComments && ch == "#") || (slashComments && ch == "/" && i + 1 < n && chars[i + 1] == "/") {
                var comment = AttributedString(String(chars[i..<n]))
                comment.foregroundColor = .secondary
                out.append(comment)
                break
            }

            // String literals
            if ch == "\"" || ch == "'" {
                let quote = ch
                var j = i + 1
                while j < n {
                    if chars[j] == "\\" {
                        j += 2
                        continue
                    }
                    if chars[j] == quote {
                        j += 1
                        break
                    }
                    j += 1
                }
                var strAttr = AttributedString(String(chars[i..<min(j, n)]))
                strAttr.foregroundColor = .orange
                out.append(strAttr)
                i = min(j, n)
                continue
            }

            // Identifiers / Keywords / Types
            if ch.isLetter || ch == "_" || ch == "@" {
                var j = i + 1
                while j < n && (chars[j].isLetter || chars[j].isNumber || chars[j] == "_") {
                    j += 1
                }
                let token = String(chars[i..<j])
                var attr = AttributedString(token)
                if token.hasPrefix("@") || commonKeywords.contains(token) {
                    attr.foregroundColor = .pink
                } else if let first = token.first, first.isUppercase {
                    attr.foregroundColor = .teal
                }
                out.append(attr)
                i = j
                continue
            }

            // Numeric literals
            if ch.isNumber {
                var j = i + 1
                while j < n && (chars[j].isNumber || chars[j] == "." || chars[j] == "_" || chars[j] == "x" || chars[j] == "X" || chars[j].isHexDigit) {
                    j += 1
                }
                var numAttr = AttributedString(String(chars[i..<j]))
                numAttr.foregroundColor = .purple
                out.append(numAttr)
                i = j
                continue
            }

            // Punctuation / Whitespace run
            var j = i + 1
            while j < n {
                let c = chars[j]
                if c.isLetter || c.isNumber || c == "_" || c == "@" || c == "\"" || c == "'" || c == "#" || c == "/" {
                    break
                }
                j += 1
            }
            out.append(AttributedString(String(chars[i..<j])))
            i = j
        }

        return out
    }
}
