import Foundation

/// Streaming parser that separates `<think>...</think>` (or `<|thought|>...</|thought|>`)
/// reasoning blocks from visible assistant text across arbitrary token chunk boundaries,
/// and supports live mid-stream `forceEndReasoning()` (`POST .../reasoning-end`).
public struct ReasoningParser: Sendable {
    public struct Delta: Sendable, Equatable {
        public let text: String
        public let thinking: String

        public init(text: String = "", thinking: String = "") {
            self.text = text
            self.thinking = thinking
        }
    }

    private enum State: Sendable {
        case initial
        case inThinking(closeTag: String)
        case inContent
    }

    private static let tagPairs: [(open: String, close: String)] = [
        ("<think>", "</think>"),
        ("<|thought|>", "<|/thought|>"),
    ]

    private var state: State
    private var buffer: String = ""
    public private(set) var reasoningEndedForced: Bool = false

    /// If `startsInThinking` is true (e.g., templates that end generation prompt with `<think>\n`),
    /// the parser starts directly inside `.inThinking(closeTag: "</think>")`.
    public init(startsInThinking: Bool = false) {
        self.state = startsInThinking ? .inThinking(closeTag: "</think>") : .initial
    }

    /// Feeds a raw decoded token string and returns the separated `(text, thinking)` delta.
    public mutating func feed(_ chunk: String) -> (text: String, thinking: String) {
        guard !chunk.isEmpty else { return ("", "") }
        buffer.append(chunk)
        return drain(flushing: false)
    }

    /// Forces reasoning to end immediately (`POST /v1/requests/{id}/reasoning-end`),
    /// flushing any buffered thinking and switching all subsequent output to visible `text`.
    public mutating func forceEndReasoning() -> (text: String, thinking: String) {
        reasoningEndedForced = true
        switch state {
        case .inThinking:
            let flushedThinking = buffer
            buffer = ""
            state = .inContent
            return (text: "", thinking: flushedThinking)
        case .initial:
            let flushedText = buffer
            buffer = ""
            state = .inContent
            return (text: flushedText, thinking: "")
        case .inContent:
            return ("", "")
        }
    }

    /// Flushes any held partial tag prefix at end-of-stream.
    public mutating func finalize() -> (text: String, thinking: String) {
        return drain(flushing: true)
    }

    private mutating func drain(flushing: Bool) -> (text: String, thinking: String) {
        var emittedText = ""
        var emittedThinking = ""

        loop: while !buffer.isEmpty {
            switch state {
            case .initial:
                let trimmedLeading = buffer.drop(while: \.isWhitespace)
                if trimmedLeading.isEmpty {
                    if flushing {
                        emittedText.append(buffer)
                        buffer = ""
                    }
                    break loop
                }
                let leadingStr = String(trimmedLeading)
                for pair in Self.tagPairs {
                    if leadingStr.hasPrefix(pair.open) {
                        if let range = buffer.range(of: pair.open) {
                            buffer.removeSubrange(buffer.startIndex..<range.upperBound)
                            state = .inThinking(closeTag: pair.close)
                            continue loop
                        }
                    }
                    if !flushing && pair.open.hasPrefix(leadingStr) {
                        // Wait for more characters to disambiguate `<think>` prefix
                        break loop
                    }
                }
                state = .inContent

            case .inThinking(let closeTag):
                if let range = buffer.range(of: closeTag) {
                    emittedThinking.append(contentsOf: buffer[..<range.lowerBound])
                    var remainder = String(buffer[range.upperBound...])
                    if remainder.hasPrefix("\n") {
                        remainder.removeFirst()
                    }
                    buffer = remainder
                    state = .inContent
                    continue loop
                }
                if flushing {
                    emittedThinking.append(buffer)
                    buffer = ""
                    break loop
                }
                let holdBack = Self.longestSuffixMatchingPrefix(of: closeTag, in: buffer)
                let emitCount = buffer.count - holdBack
                if emitCount > 0 {
                    let idx = buffer.index(buffer.startIndex, offsetBy: emitCount)
                    emittedThinking.append(contentsOf: buffer[..<idx])
                    buffer.removeSubrange(..<idx)
                }
                break loop

            case .inContent:
                // Strip a stray `</think>` if `forceEndReasoning()` was triggered before the model emitted its own close tag
                if reasoningEndedForced, let range = buffer.range(of: "</think>") {
                    emittedText.append(contentsOf: buffer[..<range.lowerBound])
                    buffer.removeSubrange(..<range.upperBound)
                    continue loop
                }
                emittedText.append(buffer)
                buffer = ""
                break loop
            }
        }

        return (text: emittedText, thinking: emittedThinking)
    }

    private static func longestSuffixMatchingPrefix(of tag: String, in text: String) -> Int {
        let maxCheck = min(tag.count - 1, text.count)
        guard maxCheck > 0 else { return 0 }
        for len in stride(from: maxCheck, through: 1, by: -1) {
            let suffix = text.suffix(len)
            if tag.hasPrefix(suffix) {
                return len
            }
        }
        return 0
    }
}

/// Streaming special-token scrubber (`StripSpecials`) that strips leaked template control tokens
/// across chunk boundaries while leaving normal user/assistant prose intact.
public struct StripSpecials: Sendable {
    public static let defaultSpecialTokens: [String] = [
        "<|im_end|>",
        "<|im_start|>",
        "<|eot_id|>",
        "<end_of_turn>",
        "<|endoftext|>",
        "<|end|>",
    ]

    private let tokens: [String]
    private var buffer: String = ""

    public init(tokens: [String] = StripSpecials.defaultSpecialTokens) {
        self.tokens = tokens.filter { !$0.isEmpty }
    }

    public mutating func feed(_ chunk: String) -> String {
        guard !chunk.isEmpty else { return "" }
        buffer.append(chunk)
        return drain(flushing: false)
    }

    public mutating func finalize() -> String {
        return drain(flushing: true)
    }

    private mutating func drain(flushing: Bool) -> String {
        for tok in tokens {
            while let range = buffer.range(of: tok) {
                buffer.removeSubrange(range)
            }
        }
        if flushing {
            let out = buffer
            buffer = ""
            return out
        }
        var maxHold = 0
        for tok in tokens {
            let maxCheck = min(tok.count - 1, buffer.count)
            guard maxCheck > maxHold else { continue }
            for len in stride(from: maxCheck, through: maxHold + 1, by: -1) {
                if tok.hasPrefix(buffer.suffix(len)) {
                    maxHold = len
                    break
                }
            }
        }
        let emitCount = buffer.count - maxHold
        guard emitCount > 0 else { return "" }
        let idx = buffer.index(buffer.startIndex, offsetBy: emitCount)
        let emitted = String(buffer[..<idx])
        buffer.removeSubrange(..<idx)
        return emitted
    }
}
