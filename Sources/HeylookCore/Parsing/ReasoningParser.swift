import Foundation

/// Multi-protocol streaming parser that separates reasoning spans from visible assistant text
/// across arbitrary token chunk boundaries, and supports live mid-stream `forceEndReasoning()`
/// (`POST /v1/requests/{id}/reasoning-end`).
///
/// Supports all 4 structural reasoning protocols derived by `ModelTemplateInfo`:
/// 1. `.tagPair` (`<think>...</think>`, `<|thought|>...<|/thought|>`, `<thought>...</thought>`)
/// 2. `.gemmaChannel` (`<|channel>thought\n...<channel|>`)
/// 3. `.harmonyChannel` (`<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>`)
/// 4. `.recipientChannel` (`to=self<|message|>...<|eom|><|start|>assistant to=user<|message|>...<|eot|>`)
public struct ReasoningParser: Sendable {
    public struct Delta: Sendable, Equatable {
        public let text: String
        public let thinking: String

        public init(text: String = "", thinking: String = "") {
            self.text = text
            self.thinking = thinking
        }
    }

    private enum Mode: Sendable {
        case tagPair(openTag: String, closeTag: String)
        case gemmaChannel(openPrefix: String, closeTag: String)
        case harmonyChannel
        case recipientChannel
        case passthrough
    }

    private enum State: Sendable {
        case initial
        case inThinking(closeTag: String)
        case inGemmaLabel(openPrefix: String, closeTag: String)
        case inGemmaChannel(isThought: Bool, closeTag: String)
        case inContent
    }

    private static let tagPairs: [(open: String, close: String)] = [
        ("<think>", "</think>"),
        ("<|thought|>", "<|/thought|>"),
        ("<thought>", "</thought>"),
    ]

    private let mode: Mode
    private var state: State
    private var buffer: String = ""
    public private(set) var reasoningEndedForced: Bool = false

    /// Initializes a parser for a specific `ReasoningFormat` and `thinkingEnabled` state.
    public init(
        format: ReasoningFormat,
        thinkingEnabled: Bool = true,
        startsInThinkingOverride: Bool? = nil
    ) {
        switch format {
        case .tagPair(let open, let close, let prefills):
            self.mode = .tagPair(openTag: open, closeTag: close)
            let starts = startsInThinkingOverride ?? (thinkingEnabled && prefills)
            self.state = starts ? .inThinking(closeTag: close) : .initial
        case .gemmaChannel(let openPrefix, let closeTag):
            self.mode = .gemmaChannel(openPrefix: openPrefix, closeTag: closeTag)
            if startsInThinkingOverride == true {
                self.state = .inGemmaChannel(isThought: true, closeTag: closeTag)
            } else {
                self.state = .initial
            }
        case .harmonyChannel:
            self.mode = .harmonyChannel
            self.state = .initial
        case .recipientChannel:
            self.mode = .recipientChannel
            self.state = .initial
        case .none:
            self.mode = .passthrough
            self.state = .inContent
        }
    }

    /// If `startsInThinking` is true (e.g., templates that end generation prompt with `<think>\n`),
    /// the parser starts directly inside `.inThinking(closeTag: "</think>")`.
    public init(startsInThinking: Bool = false) {
        self.mode = .tagPair(openTag: "<think>", closeTag: "</think>")
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
        case .inThinking, .inGemmaLabel:
            let flushedThinking = buffer
            buffer = ""
            state = .inContent
            return (text: "", thinking: flushedThinking)
        case .inGemmaChannel(let isThought, _):
            let flushed = buffer
            buffer = ""
            state = .inContent
            return isThought ? (text: "", thinking: flushed) : (text: flushed, thinking: "")
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
        switch mode {
        case .passthrough:
            let out = buffer
            buffer = ""
            return (text: out, thinking: "")
        case .tagPair(let openTag, let closeTag):
            return drainTagPair(openTag: openTag, closeTag: closeTag, flushing: flushing)
        case .gemmaChannel(let openPrefix, let closeTag):
            return drainGemmaChannel(openPrefix: openPrefix, closeTag: closeTag, flushing: flushing)
        case .harmonyChannel:
            return drainHarmonyChannel(flushing: flushing)
        case .recipientChannel:
            return drainRecipientChannel(flushing: flushing)
        }
    }

    // MARK: - 1. Tag-Pair State Machine (`<think>...</think>`)

    private mutating func drainTagPair(
        openTag: String,
        closeTag: String,
        flushing: Bool
    ) -> (text: String, thinking: String) {
        var emittedText = ""
        var emittedThinking = ""
        let candidatePairs = [(open: openTag, close: closeTag)] + Self.tagPairs.filter { $0.open != openTag }

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
                for pair in candidatePairs {
                    if leadingStr.hasPrefix(pair.open) {
                        if let range = buffer.range(of: pair.open) {
                            buffer.removeSubrange(buffer.startIndex..<range.upperBound)
                            if buffer.hasPrefix("\n") {
                                buffer.removeFirst()
                            }
                            state = .inThinking(closeTag: pair.close)
                            continue loop
                        }
                    }
                    if !flushing && pair.open.hasPrefix(leadingStr) {
                        break loop
                    }
                }
                state = .inContent

            case .inThinking(let activeClose):
                if let range = buffer.range(of: activeClose) {
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
                let holdBack = Self.longestSuffixMatchingPrefix(of: activeClose, in: buffer)
                let emitCount = buffer.count - holdBack
                if emitCount > 0 {
                    let idx = buffer.index(buffer.startIndex, offsetBy: emitCount)
                    emittedThinking.append(contentsOf: buffer[..<idx])
                    buffer.removeSubrange(..<idx)
                }
                break loop

            case .inContent:
                if reasoningEndedForced {
                    if let range = buffer.range(of: closeTag) {
                        emittedText.append(contentsOf: buffer[..<range.lowerBound])
                        buffer.removeSubrange(..<range.upperBound)
                        continue loop
                    }
                    if !flushing {
                        let holdBack = Self.longestSuffixMatchingPrefix(of: closeTag, in: buffer)
                        let emitCount = buffer.count - holdBack
                        if emitCount > 0 {
                            let idx = buffer.index(buffer.startIndex, offsetBy: emitCount)
                            emittedText.append(contentsOf: buffer[..<idx])
                            buffer.removeSubrange(..<idx)
                        }
                        break loop
                    }
                }
                emittedText.append(buffer)
                buffer = ""
                break loop

            case .inGemmaLabel, .inGemmaChannel:
                state = .inContent
            }
        }

        return (text: emittedText, thinking: emittedThinking)
    }

    // MARK: - 2. Gemma 4 Labeled Channel State Machine (`<|channel>thought\n...<channel|>`)

    private mutating func drainGemmaChannel(
        openPrefix: String,
        closeTag: String,
        flushing: Bool
    ) -> (text: String, thinking: String) {
        var emittedText = ""
        var emittedThinking = ""

        loop: while !buffer.isEmpty {
            switch state {
            case .initial, .inContent:
                if reasoningEndedForced {
                    if let range = buffer.range(of: closeTag) {
                        emittedText.append(contentsOf: buffer[..<range.lowerBound])
                        buffer.removeSubrange(..<range.upperBound)
                        continue loop
                    }
                    emittedText.append(buffer)
                    buffer = ""
                    break loop
                }

                // Check for `<|channel>` opener
                if let openRange = buffer.range(of: openPrefix) {
                    let before = String(buffer[..<openRange.lowerBound])
                    if !before.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        emittedText.append(before)
                    }
                    buffer.removeSubrange(buffer.startIndex..<openRange.upperBound)
                    state = .inGemmaLabel(openPrefix: openPrefix, closeTag: closeTag)
                    continue loop
                }

                // Also recover if prompt prefilled `<|channel>thought\n` and model emits `...<channel|>`
                if case .initial = state, let strayClose = buffer.range(of: closeTag) {
                    emittedThinking.append(contentsOf: buffer[..<strayClose.lowerBound])
                    var remainder = String(buffer[strayClose.upperBound...])
                    if remainder.hasPrefix("\n") {
                        remainder.removeFirst()
                    }
                    buffer = remainder
                    state = .inContent
                    continue loop
                }

                if flushing {
                    emittedText.append(buffer)
                    buffer = ""
                    break loop
                }

                let holdBack = Self.longestSuffixMatchingPrefix(of: openPrefix, in: buffer)
                let emitCount = buffer.count - holdBack
                if emitCount > 0 {
                    let idx = buffer.index(buffer.startIndex, offsetBy: emitCount)
                    emittedText.append(contentsOf: buffer[..<idx])
                    buffer.removeSubrange(..<idx)
                }
                break loop

            case .inGemmaLabel(_, let activeClose):
                if let newlineRange = buffer.range(of: "\n") {
                    let label = buffer[..<newlineRange.lowerBound]
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                        .lowercased()
                    buffer.removeSubrange(buffer.startIndex..<newlineRange.upperBound)
                    let isResponseChannel = (label == "content" || label == "final" || label == "response")
                    state = .inGemmaChannel(isThought: !isResponseChannel, closeTag: activeClose)
                    continue loop
                }
                if buffer.count > 32 || flushing {
                    // Malformed or unterminated label -> route safely to thinking (`MLXLMCommon.ReasoningChannel`)
                    state = .inGemmaChannel(isThought: true, closeTag: activeClose)
                    continue loop
                }
                break loop

            case .inGemmaChannel(let isThought, let activeClose):
                if let closeRange = buffer.range(of: activeClose) {
                    let body = String(buffer[..<closeRange.lowerBound])
                    if isThought {
                        emittedThinking.append(body)
                    } else {
                        emittedText.append(body)
                    }
                    var remainder = String(buffer[closeRange.upperBound...])
                    if remainder.hasPrefix("\n") {
                        remainder.removeFirst()
                    }
                    buffer = remainder
                    state = .inContent
                    continue loop
                }
                if flushing {
                    if isThought {
                        emittedThinking.append(buffer)
                    } else {
                        emittedText.append(buffer)
                    }
                    buffer = ""
                    break loop
                }
                let holdBack = Self.longestSuffixMatchingPrefix(of: activeClose, in: buffer)
                let emitCount = buffer.count - holdBack
                if emitCount > 0 {
                    let idx = buffer.index(buffer.startIndex, offsetBy: emitCount)
                    let chunk = String(buffer[..<idx])
                    if isThought {
                        emittedThinking.append(chunk)
                    } else {
                        emittedText.append(chunk)
                    }
                    buffer.removeSubrange(..<idx)
                }
                break loop

            case .inThinking:
                state = .inContent
            }
        }

        return (text: emittedText, thinking: emittedThinking)
    }

    // MARK: - 3. OpenAI Harmony Channel State Machine (`<|channel|>analysis<|message|>...<|end|>`)

    private mutating func drainHarmonyChannel(flushing: Bool) -> (text: String, thinking: String) {
        var emittedText = ""
        var emittedThinking = ""
        let msgTag = "<|message|>"
        let endTag = "<|end|>"
        let returnTag = "<|return|>"
        let channelTag = "<|channel|>"
        let startAssistant = "<|start|>assistant"

        loop: while !buffer.isEmpty {
            switch state {
            case .initial:
                // Strip leading `<|start|>assistant` if present
                if buffer.hasPrefix(startAssistant) {
                    buffer.removeFirst(startAssistant.count)
                    continue loop
                }
                if let msgRange = buffer.range(of: msgTag) {
                    let header = buffer[..<msgRange.lowerBound].lowercased()
                    buffer.removeSubrange(buffer.startIndex..<msgRange.upperBound)
                    if header.contains("analysis") || header.contains("commentary") || header.contains("thought") {
                        state = .inThinking(closeTag: endTag)
                    } else {
                        state = .inContent
                    }
                    continue loop
                }
                // Wait for `<|message|>` if header looks like a Harmony channel prefix
                let trimmed = buffer.drop(while: \.isWhitespace)
                if !flushing && (
                    trimmed.hasPrefix("<|")
                    || "<|channel|>".hasPrefix(trimmed)
                    || "<|start|>".hasPrefix(trimmed)
                    || trimmed.hasPrefix("analysis")
                    || trimmed.hasPrefix("final")
                ) && buffer.count < 64 {
                    break loop
                }
                state = .inContent

            case .inThinking(let closeTag):
                if let endRange = buffer.range(of: closeTag) {
                    emittedThinking.append(contentsOf: buffer[..<endRange.lowerBound])
                    buffer.removeSubrange(buffer.startIndex..<endRange.upperBound)
                    state = .initial
                    continue loop
                }
                // Also check if `<|channel|>final<|message|>` appears directly without `<|end|>`
                if let finalRange = buffer.range(of: "<|channel|>final<|message|>") {
                    emittedThinking.append(contentsOf: buffer[..<finalRange.lowerBound])
                    buffer.removeSubrange(buffer.startIndex..<finalRange.upperBound)
                    state = .inContent
                    continue loop
                }
                if flushing {
                    emittedThinking.append(buffer)
                    buffer = ""
                    break loop
                }
                let holdEnd = Self.longestSuffixMatchingPrefix(of: closeTag, in: buffer)
                let holdChan = Self.longestSuffixMatchingPrefix(of: "<|channel|>final<|message|>", in: buffer)
                let holdBack = max(holdEnd, holdChan)
                let emitCount = buffer.count - holdBack
                if emitCount > 0 {
                    let idx = buffer.index(buffer.startIndex, offsetBy: emitCount)
                    emittedThinking.append(contentsOf: buffer[..<idx])
                    buffer.removeSubrange(..<idx)
                }
                break loop

            case .inContent:
                // Strip trailing `<|end|>` or `<|return|>` markers inside content
                for marker in [endTag, returnTag, startAssistant, "\(channelTag)final\(msgTag)"] {
                    while let r = buffer.range(of: marker) {
                        buffer.removeSubrange(r)
                    }
                }
                if flushing {
                    emittedText.append(buffer)
                    buffer = ""
                    break loop
                }
                let holdEnd = Self.longestSuffixMatchingPrefix(of: endTag, in: buffer)
                let holdRet = Self.longestSuffixMatchingPrefix(of: returnTag, in: buffer)
                let holdChan = Self.longestSuffixMatchingPrefix(of: "\(channelTag)final\(msgTag)", in: buffer)
                let holdBack = max(holdEnd, max(holdRet, holdChan))
                let emitCount = buffer.count - holdBack
                if emitCount > 0 {
                    let idx = buffer.index(buffer.startIndex, offsetBy: emitCount)
                    emittedText.append(contentsOf: buffer[..<idx])
                    buffer.removeSubrange(..<idx)
                }
                break loop

            case .inGemmaLabel, .inGemmaChannel:
                state = .inContent
            }
        }

        return (text: emittedText, thinking: emittedThinking)
    }

    // MARK: - 4. Recipient-Addressed Channel State Machine (`to=self<|message|>...<|eom|>`)

    private mutating func drainRecipientChannel(flushing: Bool) -> (text: String, thinking: String) {
        var emittedText = ""
        var emittedThinking = ""
        let msgTag = "<|message|>"
        let eomTag = "<|eom|>"
        let eotTag = "<|eot|>"
        let startAssistant = "<|start|>assistant"

        loop: while !buffer.isEmpty {
            switch state {
            case .initial:
                if let msgRange = buffer.range(of: msgTag) {
                    let header = buffer[..<msgRange.lowerBound].lowercased()
                    buffer.removeSubrange(buffer.startIndex..<msgRange.upperBound)
                    if header.contains("to=self") {
                        state = .inThinking(closeTag: eomTag)
                    } else {
                        state = .inContent
                    }
                    continue loop
                }
                let trimmed = buffer.drop(while: \.isWhitespace)
                if !flushing && (
                    trimmed.hasPrefix("to=")
                    || "to=self".hasPrefix(trimmed)
                    || "to=user".hasPrefix(trimmed)
                    || trimmed.hasPrefix("<|")
                    || "<|start|>".hasPrefix(trimmed)
                ) && buffer.count < 64 {
                    break loop
                }
                state = .inContent

            case .inThinking(let closeTag):
                if let eomRange = buffer.range(of: closeTag) {
                    emittedThinking.append(contentsOf: buffer[..<eomRange.lowerBound])
                    buffer.removeSubrange(buffer.startIndex..<eomRange.upperBound)
                    state = .initial
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
                for marker in [eotTag, eomTag, startAssistant, "to=user<|message|>"] {
                    while let r = buffer.range(of: marker) {
                        buffer.removeSubrange(r)
                    }
                }
                if flushing {
                    emittedText.append(buffer)
                    buffer = ""
                    break loop
                }
                let holdEot = Self.longestSuffixMatchingPrefix(of: eotTag, in: buffer)
                let holdEom = Self.longestSuffixMatchingPrefix(of: eomTag, in: buffer)
                let holdUser = Self.longestSuffixMatchingPrefix(of: "to=user<|message|>", in: buffer)
                let holdBack = max(holdEot, max(holdEom, holdUser))
                let emitCount = buffer.count - holdBack
                if emitCount > 0 {
                    let idx = buffer.index(buffer.startIndex, offsetBy: emitCount)
                    emittedText.append(contentsOf: buffer[..<idx])
                    buffer.removeSubrange(..<idx)
                }
                break loop

            case .inGemmaLabel, .inGemmaChannel:
                state = .inContent
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
        "<turn|>",
        "<|eot|>",
        "<|eom|>",
    ]

    private let tokens: [String]
    private var buffer: String = ""

    public init(tokens: [String] = StripSpecials.defaultSpecialTokens) {
        let merged = Set(StripSpecials.defaultSpecialTokens).union(tokens.filter { !$0.isEmpty })
        // Sort longest-first so compound markers are stripped before shorter substrings
        self.tokens = merged.sorted { $0.count > $1.count }
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
