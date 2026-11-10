import Foundation
#if canImport(Jinja)
import Jinja
#endif

/// Errors thrown while parsing or evaluating a Jinja2 chat template.
public enum JinjaTemplateError: LocalizedError, Sendable, Equatable {
    case syntaxError(String)
    case templateRaisedException(String)
    case evaluationError(String)

    public var errorDescription: String? {
        switch self {
        case .syntaxError(let detail):
            return "Jinja template syntax error: \(detail)"
        case .templateRaisedException(let message):
            return "Template raised exception: \(message)"
        case .evaluationError(let detail):
            return "Jinja template evaluation error: \(detail)"
        }
    }
}

/// Character-offset span locating a rendered special token inside a prompt preview (`conversation_generate_api.py`).
public struct PromptMarkerSpan: Codable, Sendable, Equatable {
    public let start: Int
    public let end: Int
    public let token: String

    public init(start: Int, end: Int, token: String) {
        self.start = start
        self.end = end
        self.token = token
    }
}

/// Structural capabilities derived directly from a chat template's source (`chat_template_caps`).
public struct TemplateCapabilities: Codable, Sendable, Equatable {
    public var supportsSystemRole: Bool
    public var supportsTools: Bool
    public var supportsParallelToolCalls: Bool
    public var supportsToolCallID: Bool
    public var supportsVision: Bool
    public var supportsAudio: Bool
    public var supportsThinking: Bool
    public var requiresAlternatingRoles: Bool
    public var requiresTypedContent: Bool

    enum CodingKeys: String, CodingKey {
        case supportsSystemRole = "supports_system_role"
        case supportsTools = "supports_tools"
        case supportsParallelToolCalls = "supports_parallel_tool_calls"
        case supportsToolCallID = "supports_tool_call_id"
        case supportsVision = "supports_vision"
        case supportsAudio = "supports_audio"
        case supportsThinking = "supports_thinking"
        case requiresAlternatingRoles = "requires_alternating_roles"
        case requiresTypedContent = "requires_typed_content"
    }

    public init(
        supportsSystemRole: Bool = true,
        supportsTools: Bool = false,
        supportsParallelToolCalls: Bool = false,
        supportsToolCallID: Bool = false,
        supportsVision: Bool = false,
        supportsAudio: Bool = false,
        supportsThinking: Bool = false,
        requiresAlternatingRoles: Bool = false,
        requiresTypedContent: Bool = false
    ) {
        self.supportsSystemRole = supportsSystemRole
        self.supportsTools = supportsTools
        self.supportsParallelToolCalls = supportsParallelToolCalls
        self.supportsToolCallID = supportsToolCallID
        self.supportsVision = supportsVision
        self.supportsAudio = supportsAudio
        self.supportsThinking = supportsThinking
        self.requiresAlternatingRoles = requiresAlternatingRoles
        self.requiresTypedContent = requiresTypedContent
    }
}

/// Detailed prompt render result used by `/v1/conversations/{id}/prompt` and the UI Prompt Inspector.
public struct PromptRenderResult: Codable, Sendable, Equatable {
    public let prompt: String
    public let source: String
    public let markers: [PromptMarkerSpan]
    public let continuation: Bool
    public let droppedMedia: Int
    public let unrenderedMedia: Int

    enum CodingKeys: String, CodingKey {
        case prompt
        case source
        case markers
        case continuation
        case droppedMedia = "dropped_media"
        case unrenderedMedia = "unrendered_media"
    }

    public init(
        prompt: String,
        source: String,
        markers: [PromptMarkerSpan] = [],
        continuation: Bool = false,
        droppedMedia: Int = 0,
        unrenderedMedia: Int = 0
    ) {
        self.prompt = prompt
        self.source = source
        self.markers = markers
        self.continuation = continuation
        self.droppedMedia = droppedMedia
        self.unrenderedMedia = unrenderedMedia
    }
}

/// Dynamic runtime value for `HeylookJinjaRenderer`.
public enum JinjaValue: Sendable, Equatable {
    case undefined
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JinjaValue])
    case tuple([JinjaValue])
    case object([String: JinjaValue])
    case namespace(NamespaceBox)

    public final class NamespaceBox: @unchecked Sendable, Equatable {
        private let lock = NSLock()
        private var fields: [String: JinjaValue]

        public init(_ fields: [String: JinjaValue] = [:]) {
            self.fields = fields
        }

        public func get(_ key: String) -> JinjaValue {
            lock.lock()
            defer { lock.unlock() }
            return fields[key] ?? .undefined
        }

        public func set(_ key: String, value: JinjaValue) {
            lock.lock()
            defer { lock.unlock() }
            fields[key] = value
        }

        public func snapshot() -> [String: JinjaValue] {
            lock.lock()
            defer { lock.unlock() }
            return fields
        }

        public static func == (lhs: NamespaceBox, rhs: NamespaceBox) -> Bool {
            lhs.snapshot() == rhs.snapshot()
        }
    }

    public var isTruthy: Bool {
        switch self {
        case .undefined, .null:
            return false
        case .bool(let b):
            return b
        case .int(let i):
            return i != 0
        case .double(let d):
            return d != 0.0
        case .string(let s):
            return !s.isEmpty
        case .array(let a), .tuple(let a):
            return !a.isEmpty
        case .object(let o):
            return !o.isEmpty
        case .namespace(let ns):
            return !ns.snapshot().isEmpty
        }
    }

    public var renderString: String {
        switch self {
        case .undefined, .null:
            return ""
        case .bool(let b):
            return b ? "True" : "False"
        case .int(let i):
            return String(i)
        case .double(let d):
            if d.rounded() == d, let i = Int(exactly: d) {
                return "\(i).0"
            }
            return String(d)
        case .string(let s):
            return s
        case .array(let items):
            let rendered = items.map { $0.reprString }.joined(separator: ", ")
            return "[\(rendered)]"
        case .tuple(let items):
            let rendered = items.map { $0.reprString }.joined(separator: ", ")
            return items.count == 1 ? "(\(rendered),)" : "(\(rendered))"
        case .object(let dict):
            let pairs = dict.keys.sorted().map { k in
                "\(JinjaValue.string(k).reprString): \((dict[k] ?? .null).reprString)"
            }
            return "{\(pairs.joined(separator: ", "))}"
        case .namespace(let ns):
            let snap = ns.snapshot()
            let pairs = snap.keys.sorted().map { k in "\(k)=\((snap[k] ?? .null).reprString)" }
            return "namespace(\(pairs.joined(separator: ", ")))"
        }
    }

    public var reprString: String {
        switch self {
        case .string(let s):
            let escaped = s
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "'", with: "\\'")
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\r", with: "\\r")
                .replacingOccurrences(of: "\t", with: "\\t")
            return "'\(escaped)'"
        case .null, .undefined:
            return "None"
        default:
            return renderString
        }
    }

    public var asNumber: Double? {
        switch self {
        case .int(let i): return Double(i)
        case .double(let d): return d
        case .bool(let b): return b ? 1.0 : 0.0
        default: return nil
        }
    }

    public var asInt: Int? {
        switch self {
        case .int(let i): return i
        case .double(let d): return Int(d)
        case .bool(let b): return b ? 1 : 0
        case .string(let s): return Int(s.trimmingCharacters(in: .whitespacesAndNewlines))
        default: return nil
        }
    }

    public func toJSONObject() -> Any {
        switch self {
        case .undefined, .null:
            return NSNull()
        case .bool(let b):
            return b
        case .int(let i):
            return i
        case .double(let d):
            return d
        case .string(let s):
            return s
        case .array(let a), .tuple(let a):
            return a.map { $0.toJSONObject() }
        case .object(let o):
            var out: [String: Any] = [:]
            for (k, v) in o where v != .undefined {
                out[k] = v.toJSONObject()
            }
            return out
        case .namespace(let ns):
            var out: [String: Any] = [:]
            for (k, v) in ns.snapshot() where v != .undefined {
                out[k] = v.toJSONObject()
            }
            return out
        }
    }

    public static func fromAny(_ any: Any?) -> JinjaValue {
        guard let any else { return .null }
        if any is NSNull { return .null }
        if let v = any as? JinjaValue { return v }
        if let s = any as? String { return .string(s) }
        if let b = any as? Bool { return .bool(b) }
        if let n = any as? NSNumber {
            if CFGetTypeID(n) == CFBooleanGetTypeID() {
                return .bool(n.boolValue)
            }
            let objCType = String(cString: n.objCType)
            if objCType == "d" || objCType == "f" {
                return .double(n.doubleValue)
            }
            return .int(n.intValue)
        }
        if let i = any as? Int { return .int(i) }
        if let d = any as? Double { return .double(d) }
        if let arr = any as? [Any] { return .array(arr.map { fromAny($0) }) }
        if let dict = any as? [String: Any] {
            var out: [String: JinjaValue] = [:]
            for (k, v) in dict {
                out[k] = fromAny(v)
            }
            return .object(out)
        }
        return .string(String(describing: any))
    }

    #if canImport(Jinja)
    fileprivate func toHuggingFaceJinjaValue() -> Jinja.Value {
        switch self {
        case .undefined:
            return .undefined
        case .null:
            return .null
        case .bool(let b):
            return .boolean(b)
        case .int(let i):
            return .int(i)
        case .double(let d):
            return .double(d)
        case .string(let s):
            return .string(s)
        case .array(let a), .tuple(let a):
            return .array(a.map { $0.toHuggingFaceJinjaValue() })
        case .object(let o):
            var mapped: [String: Jinja.Value] = [:]
            for (k, v) in o {
                mapped[k] = v.toHuggingFaceJinjaValue()
            }
            return .object(mapped)
        case .namespace(let ns):
            var mapped: [String: Jinja.Value] = [:]
            for (k, v) in ns.snapshot() {
                mapped[k] = v.toHuggingFaceJinjaValue()
            }
            return .object(mapped)
        }
    }
    #endif
}

/// Real Jinja2 lexer, recursive-descent AST parser, validator, and evaluator for HuggingFace & GGUF chat templates.
/// Uses `huggingface/swift-jinja` when linked (`HEYLOOK_ENABLE_NATIVE_MLX=1`) and falls back to the in-tree
/// recursive-descent Jinja2 AST engine in pure-Swift builds so template evaluation is always real.
public enum HeylookJinjaRenderer {
    /// Built-in real ChatML Jinja2 template used only when a model directory ships zero chat template files.
    public static let defaultChatMLTemplate = """
    {%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\\n' }}
    {%- if message['thinking'] is defined and message['thinking'] %}
    {{- '<think>\\n' + message['thinking'] + '\\n</think>\\n' }}
    {%- endif %}
    {%- if message['content'] is string %}
    {{- message['content'] + '<|im_end|>\\n' }}
    {%- else %}
    {%- for part in message['content'] %}
    {%- if part['type'] == 'image' %}
    {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
    {%- elif part['type'] == 'audio' %}
    {{- '<|audio_start|><|audio_pad|><|audio_end|>' }}
    {%- elif part['type'] == 'text' %}
    {{- part['text'] }}
    {%- endif %}
    {%- endfor %}
    {{- '<|im_end|>\\n' }}
    {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking %}
    {{- '<think>\\n' }}
    {%- endif %}
    {%- endif %}
    """

    /// Strictly validates Jinja2 template syntax, rejecting empty strings, unclosed `{{` / `{%` / `{#` tags,
    /// invalid expressions, and mismatched or unclosed `if` / `for` / `macro` / `set` blocks.
    public static func validateSyntax(_ template: String) throws {
        if template.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw JinjaTemplateError.syntaxError("Chat template must not be empty.")
        }
        let tokens = try TemplateLexer.lex(template, lstripBlocks: true, trimBlocks: true)
        _ = try TemplateParser.parse(tokens)
        #if canImport(Jinja)
        do {
            _ = try Jinja.Template(template, with: .init(lstripBlocks: true, trimBlocks: true))
        } catch {
            throw JinjaTemplateError.syntaxError(error.localizedDescription)
        }
        #endif
    }

    /// Evaluates a Jinja2 template string against `context`.
    public static func render(
        template: String,
        context: [String: JinjaValue],
        lstripBlocks: Bool = true,
        trimBlocks: Bool = true
    ) throws -> String {
        let tokens = try TemplateLexer.lex(template, lstripBlocks: lstripBlocks, trimBlocks: trimBlocks)
        let program = try TemplateParser.parse(tokens)

        #if canImport(Jinja)
        do {
            let hfTemplate = try Jinja.Template(
                template,
                with: .init(lstripBlocks: lstripBlocks, trimBlocks: trimBlocks)
            )
            var hfContext: [String: Jinja.Value] = [:]
            for (k, v) in context {
                hfContext[k] = v.toHuggingFaceJinjaValue()
            }
            return try hfTemplate.render(hfContext)
        } catch let err as JinjaTemplateError {
            throw err
        } catch {
            // If swift-jinja lacks a Python-specific method/filter (e.g. strftime_now or custom slice),
            // execute our parsed AST interpreter which implements full HuggingFace template builtins.
            let evaluator = TemplateEvaluator(rootGlobals: context)
            return try evaluator.evaluate(program)
        }
        #else
        let evaluator = TemplateEvaluator(rootGlobals: context)
        return try evaluator.evaluate(program)
        #endif
    }

    /// Finds every non-overlapping special-token span in `prompt` sorted by character offset
    /// (longest-match-first at any tied start position, matching Python `conversation_generate_api.py`).
    public static func extractMarkers(
        in prompt: String,
        specialTokens: Set<String>
    ) -> [PromptMarkerSpan] {
        let candidates = specialTokens
            .filter { !$0.isEmpty }
            .sorted { lhs, rhs in
                if lhs.count != rhs.count { return lhs.count > rhs.count }
                return lhs < rhs
            }
        guard !candidates.isEmpty, !prompt.isEmpty else { return [] }

        let chars = Array(prompt)
        let candChars: [(token: String, chars: [Character])] = candidates.map { ($0, Array($0)) }
        var spans: [PromptMarkerSpan] = []
        var idx = 0
        let total = chars.count

        while idx < total {
            var matched: (token: String, len: Int)?
            for cand in candChars {
                let len = cand.chars.count
                guard idx + len <= total else { continue }
                var ok = true
                for j in 0..<len where chars[idx + j] != cand.chars[j] {
                    ok = false
                    break
                }
                if ok {
                    matched = (cand.token, len)
                    break
                }
            }
            if let matched {
                spans.append(PromptMarkerSpan(start: idx, end: idx + matched.len, token: matched.token))
                idx += matched.len
            } else {
                idx += 1
            }
        }
        return spans
    }

    /// Derives `TemplateCapabilities` directly from the Jinja template text.
    public static func detectCapabilities(template: String) -> TemplateCapabilities {
        guard !template.isEmpty else {
            return TemplateCapabilities()
        }
        let supportsSystem = template.contains("'system'")
            || template.contains("\"system\"")
            || template.contains("system_message")
        let supportsTools = template.contains("tools") || template.contains("tool_calls")
        let supportsParallelTools = supportsTools && template.contains("for tool_call in")
        let supportsToolCallID = template.contains("tool_call_id")
        let supportsVision = template.contains("'image'")
            || template.contains("\"image\"")
            || template.contains("image_url")
            || template.contains("<|image")
            || template.contains("<|vision_start|>")
            || template.contains("<start_of_image>")
            || template.contains("[IMG]")
            || template.contains("<__media__>")
        let supportsAudio = template.contains("'audio'")
            || template.contains("\"audio\"")
            || template.contains("audio_url")
            || template.contains("<|audio")
        let supportsThinking = template.contains("enable_thinking")
            || template.contains("reasoning_effort")
            || template.contains("reasoning_strength")
            || template.contains("thinking_mode")
            || template.contains("reasoning_content")
            || template.contains("<think>")
            || template.contains("<|thought|>")
            || template.contains("<|channel>thought")
            || template.contains("<|channel|>analysis")
        let requiresAlternating = template.contains("Conversation roles must alternate")
            || template.contains("loop.index0 % 2")
            || template.contains("prev_role")
        let requiresTypedContent = (supportsVision || supportsAudio)
            && (template.contains("content is not string")
                || template.contains("content is iterable")
                || template.contains("for content in message['content']")
                || template.contains("for item in message.content")
                || template.contains("for part in message['content']"))

        return TemplateCapabilities(
            supportsSystemRole: supportsSystem,
            supportsTools: supportsTools,
            supportsParallelToolCalls: supportsParallelTools,
            supportsToolCallID: supportsToolCallID,
            supportsVision: supportsVision,
            supportsAudio: supportsAudio,
            supportsThinking: supportsThinking,
            requiresAlternatingRoles: requiresAlternating,
            requiresTypedContent: requiresTypedContent
        )
    }

    /// Evaluates whether a Jinja2 chat template preserves exact byte prefixes across appended turns
    /// (`chat_template_files.py:check_prefix_stability`).
    public static func checkPrefixStability(template: String) -> (stable: Bool, note: String?) {
        guard !template.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (true, nil)
        }
        if template.contains("strftime_now") || template.contains("now(") {
            return (
                false,
                "Template calls strftime_now()/now(), changing system header timestamps across turns."
            )
        }

        func makeMsg(role: String, content: String, thinking: String? = nil) -> JinjaValue {
            var dict: [String: JinjaValue] = [
                "role": .string(role),
                "content": .string(content),
            ]
            if let thinking {
                dict["thinking"] = .string(thinking)
                dict["reasoning_content"] = .string(thinking)
            }
            return .object(dict)
        }

        let baseContext: [String: JinjaValue] = [
            "add_generation_prompt": .bool(false),
            "enable_thinking": .bool(true),
            "bos_token": .string(""),
            "eos_token": .string(""),
        ]

        // 1. Plain multi-turn prefix check ([U1, A1] vs [U1, A1, U2, A2])
        let msgs2Plain: [JinjaValue] = [
            makeMsg(role: "user", content: "Hello"),
            makeMsg(role: "assistant", content: "Hi there"),
        ]
        let msgs4Plain: [JinjaValue] = msgs2Plain + [
            makeMsg(role: "user", content: "Follow up"),
            makeMsg(role: "assistant", content: "Second reply"),
        ]
        var ctx2 = baseContext
        ctx2["messages"] = .array(msgs2Plain)
        var ctx4 = baseContext
        ctx4["messages"] = .array(msgs4Plain)

        if let p2 = try? render(template: template, context: ctx2),
           let p4 = try? render(template: template, context: ctx4),
           !p2.isEmpty,
           !p4.hasPrefix(p2) {
            return (
                false,
                "Template alters earlier rendered turns when subsequent messages are appended."
            )
        }

        // 2. Multi-turn reasoning prefix check (catches templates like Qwen3 that strip reasoning_content
        //    on turns prior to last_query_index!)
        let msgs2Think: [JinjaValue] = [
            makeMsg(role: "user", content: "Solve 1+1"),
            makeMsg(role: "assistant", content: "2", thinking: "1 plus 1 equals 2"),
        ]
        let msgs4Think: [JinjaValue] = msgs2Think + [
            makeMsg(role: "user", content: "And +1?"),
            makeMsg(role: "assistant", content: "3", thinking: "2 plus 1 equals 3"),
        ]
        var ctx2T = baseContext
        ctx2T["messages"] = .array(msgs2Think)
        var ctx4T = baseContext
        ctx4T["messages"] = .array(msgs4Think)

        if let p2T = try? render(template: template, context: ctx2T),
           let p4T = try? render(template: template, context: ctx4T),
           !p2T.isEmpty,
           !p4T.hasPrefix(p2T) {
            return (
                false,
                "Template strips or rewrites earlier reasoning blocks when new user turns are appended."
            )
        }

        return (true, nil)
    }
}

// MARK: - Jinja2 Lexer

private enum TemplateChunk: Equatable {
    case text(String)
    case expr(String)
    case stmt(String)
}

private enum TemplateLexer {
    static func lex(_ source: String, lstripBlocks: Bool, trimBlocks: Bool) throws -> [TemplateChunk] {
        enum RawPiece {
            case text(String)
            case tag(kind: Character, leftStrip: Bool, rightStrip: Bool, explicitLeftPlus: Bool, body: String)
        }

        let chars = Array(source)
        let n = chars.count
        var i = 0
        var rawPieces: [RawPiece] = []
        var textStart = 0

        func flushText(upTo end: Int) {
            if end > textStart {
                rawPieces.append(.text(String(chars[textStart..<end])))
            }
        }

        while i < n {
            if i + 1 < n, chars[i] == "{", (chars[i + 1] == "{" || chars[i + 1] == "%" || chars[i + 1] == "#") {
                let kind = chars[i + 1]
                let closingChar: Character = (kind == "{") ? "}" : kind
                flushText(upTo: i)

                var cursor = i + 2
                var leftStrip = false
                var explicitLeftPlus = false
                if cursor < n {
                    if chars[cursor] == "-" {
                        leftStrip = true
                        cursor += 1
                    } else if chars[cursor] == "+" {
                        explicitLeftPlus = true
                        cursor += 1
                    }
                }

                let bodyStart = cursor
                var inSingle = false
                var inDouble = false
                var escaped = false
                var foundClose: Int?
                var rightStrip = false

                while cursor < n {
                    let ch = chars[cursor]
                    if kind != "#" {
                        if escaped {
                            escaped = false
                            cursor += 1
                            continue
                        }
                        if ch == "\\" && (inSingle || inDouble) {
                            escaped = true
                            cursor += 1
                            continue
                        }
                        if ch == "'" && !inDouble {
                            inSingle.toggle()
                            cursor += 1
                            continue
                        }
                        if ch == "\"" && !inSingle {
                            inDouble.toggle()
                            cursor += 1
                            continue
                        }
                    }
                    if !inSingle && !inDouble {
                        if cursor + 1 < n && ch == closingChar && chars[cursor + 1] == "}" {
                            foundClose = cursor
                            break
                        }
                        if cursor + 2 < n && (ch == "-" || ch == "+") && chars[cursor + 1] == closingChar && chars[cursor + 2] == "}" {
                            rightStrip = (ch == "-")
                            foundClose = cursor
                            break
                        }
                    }
                    cursor += 1
                }

                guard let closeIdx = foundClose else {
                    throw JinjaTemplateError.syntaxError("Unclosed '{\(kind)' tag in template.")
                }

                let body = String(chars[bodyStart..<closeIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
                let afterClose: Int
                if chars[closeIdx] == "-" || chars[closeIdx] == "+" {
                    afterClose = closeIdx + 3
                } else {
                    afterClose = closeIdx + 2
                }

                rawPieces.append(.tag(
                    kind: kind,
                    leftStrip: leftStrip,
                    rightStrip: rightStrip,
                    explicitLeftPlus: explicitLeftPlus,
                    body: body
                ))
                i = afterClose
                textStart = i
            } else {
                i += 1
            }
        }
        flushText(upTo: n)

        // Apply Jinja2 whitespace control (`lstrip_blocks`, `trim_blocks`, `-`)
        var texts = [Int: String]()
        for (idx, piece) in rawPieces.enumerated() {
            if case .text(let s) = piece {
                texts[idx] = s
            }
        }

        for (idx, piece) in rawPieces.enumerated() {
            guard case .tag(let kind, let leftStrip, let rightStrip, let explicitLeftPlus, _) = piece else {
                continue
            }
            if leftStrip {
                if let prev = texts[idx - 1] {
                    texts[idx - 1] = stripTrailingAllWhitespace(prev)
                }
            } else if lstripBlocks && (kind == "%" || kind == "#") && !explicitLeftPlus {
                if let prev = texts[idx - 1] {
                    texts[idx - 1] = stripTrailingIndentAfterNewline(prev)
                }
            }

            if rightStrip {
                if let next = texts[idx + 1] {
                    texts[idx + 1] = stripLeadingAllWhitespace(next)
                }
            } else if trimBlocks && (kind == "%" || kind == "#") {
                if let next = texts[idx + 1] {
                    texts[idx + 1] = stripLeadingSingleNewline(next)
                }
            }
        }

        var chunks: [TemplateChunk] = []
        for (idx, piece) in rawPieces.enumerated() {
            switch piece {
            case .text:
                if let t = texts[idx], !t.isEmpty {
                    chunks.append(.text(t))
                }
            case .tag(let kind, _, _, _, let body):
                if kind == "#" {
                    continue
                } else if kind == "{" {
                    if body.isEmpty {
                        throw JinjaTemplateError.syntaxError("Empty '{{ }}' expression tag.")
                    }
                    chunks.append(.expr(body))
                } else if kind == "%" {
                    if body.isEmpty {
                        throw JinjaTemplateError.syntaxError("Empty '{% %}' statement tag.")
                    }
                    chunks.append(.stmt(body))
                }
            }
        }
        return chunks
    }

    private static func stripTrailingAllWhitespace(_ s: String) -> String {
        var scalars = Array(s)
        while let last = scalars.last, last.isWhitespace {
            scalars.removeLast()
        }
        return String(scalars)
    }

    private static func stripLeadingAllWhitespace(_ s: String) -> String {
        var idx = s.startIndex
        while idx < s.endIndex, s[idx].isWhitespace {
            idx = s.index(after: idx)
        }
        return String(s[idx...])
    }

    private static func stripLeadingSingleNewline(_ s: String) -> String {
        if s.hasPrefix("\r\n") {
            return String(s.dropFirst(2))
        }
        if s.hasPrefix("\n") {
            return String(s.dropFirst())
        }
        return s
    }

    private static func stripTrailingIndentAfterNewline(_ s: String) -> String {
        let chars = Array(s)
        var k = chars.count
        while k > 0 && (chars[k - 1] == " " || chars[k - 1] == "\t") {
            k -= 1
        }
        if k == 0 || chars[k - 1] == "\n" {
            return String(chars[0..<k])
        }
        return s
    }
}

// MARK: - Expression Tokenizer & AST

private enum ExprToken: Equatable {
    case ident(String)
    case int(Int)
    case double(Double)
    case string(String)
    case op(String)
    case lparen, rparen
    case lbracket, rbracket
    case lbrace, rbrace
    case comma, colon, dot, pipe, assign
}

private enum ExprTokenizer {
    static func tokenize(_ input: String) throws -> [ExprToken] {
        let chars = Array(input)
        let n = chars.count
        var i = 0
        var tokens: [ExprToken] = []

        while i < n {
            let ch = chars[i]
            if ch.isWhitespace {
                i += 1
                continue
            }
            if ch == "'" || ch == "\"" {
                let quote = ch
                i += 1
                var buf = ""
                var closed = false
                while i < n {
                    let c = chars[i]
                    if c == "\\" {
                        guard i + 1 < n else {
                            throw JinjaTemplateError.syntaxError("Unterminated escape in string literal.")
                        }
                        let esc = chars[i + 1]
                        switch esc {
                        case "n": buf.append("\n"); i += 2
                        case "r": buf.append("\r"); i += 2
                        case "t": buf.append("\t"); i += 2
                        case "\\": buf.append("\\"); i += 2
                        case "'": buf.append("'"); i += 2
                        case "\"": buf.append("\""); i += 2
                        case "u":
                            if i + 5 < n,
                               let hex = UInt32(String(chars[(i + 2)...(i + 5)]), radix: 16),
                               let scalar = UnicodeScalar(hex) {
                                buf.append(Character(scalar))
                                i += 6
                            } else {
                                buf.append("u")
                                i += 2
                            }
                        default:
                            buf.append(esc)
                            i += 2
                        }
                        continue
                    }
                    if c == quote {
                        closed = true
                        i += 1
                        break
                    }
                    buf.append(c)
                    i += 1
                }
                if !closed {
                    throw JinjaTemplateError.syntaxError("Unterminated string literal in '\(input)'.")
                }
                tokens.append(.string(buf))
                continue
            }

            if ch.isNumber {
                let start = i
                var hasDot = false
                while i < n && (chars[i].isNumber || (!hasDot && chars[i] == "." && i + 1 < n && chars[i + 1].isNumber)) {
                    if chars[i] == "." { hasDot = true }
                    i += 1
                }
                let raw = String(chars[start..<i])
                if hasDot {
                    guard let d = Double(raw) else {
                        throw JinjaTemplateError.syntaxError("Invalid float literal '\(raw)'.")
                    }
                    tokens.append(.double(d))
                } else {
                    guard let v = Int(raw) else {
                        throw JinjaTemplateError.syntaxError("Invalid integer literal '\(raw)'.")
                    }
                    tokens.append(.int(v))
                }
                continue
            }

            if ch.isLetter || ch == "_" {
                let start = i
                while i < n && (chars[i].isLetter || chars[i].isNumber || chars[i] == "_") {
                    i += 1
                }
                tokens.append(.ident(String(chars[start..<i])))
                continue
            }

            if i + 1 < n {
                let two = String([ch, chars[i + 1]])
                if ["==", "!=", "<=", ">=", "//", "**"].contains(two) {
                    tokens.append(.op(two))
                    i += 2
                    continue
                }
            }

            switch ch {
            case "(": tokens.append(.lparen); i += 1
            case ")": tokens.append(.rparen); i += 1
            case "[": tokens.append(.lbracket); i += 1
            case "]": tokens.append(.rbracket); i += 1
            case "{": tokens.append(.lbrace); i += 1
            case "}": tokens.append(.rbrace); i += 1
            case ",": tokens.append(.comma); i += 1
            case ":": tokens.append(.colon); i += 1
            case ".": tokens.append(.dot); i += 1
            case "|": tokens.append(.pipe); i += 1
            case "=": tokens.append(.assign); i += 1
            case "+", "-", "*", "/", "%", "~", "<", ">":
                tokens.append(.op(String(ch)))
                i += 1
            default:
                throw JinjaTemplateError.syntaxError("Unexpected character '\(ch)' in '\(input)'.")
            }
        }
        return tokens
    }
}

private indirect enum Expr {
    case literal(JinjaValue)
    case variable(String)
    case list([Expr])
    case tuple([Expr])
    case dict([(Expr, Expr)])
    case unary(op: String, Expr)
    case binary(op: String, Expr, Expr)
    case conditional(cond: Expr, ifTrue: Expr, ifFalse: Expr?)
    case getattr(Expr, String)
    case getitem(Expr, Expr)
    case slice(Expr, start: Expr?, stop: Expr?, step: Expr?)
    case call(Expr, args: [Expr], kwargs: [(String, Expr)])
    case filter(Expr, name: String, args: [Expr], kwargs: [(String, Expr)])
    case test(Expr, name: String, args: [Expr], negated: Bool)
}

private struct ExprParser {
    let tokens: [ExprToken]
    var pos: Int = 0

    var current: ExprToken? { pos < tokens.count ? tokens[pos] : nil }

    mutating func advance() -> ExprToken? {
        guard pos < tokens.count else { return nil }
        let tok = tokens[pos]
        pos += 1
        return tok
    }

    mutating func expect(_ expected: ExprToken) throws {
        guard advance() == expected else {
            throw JinjaTemplateError.syntaxError("Expected \(expected), got \(String(describing: current)).")
        }
    }

    static func parseFullExpression(_ source: String) throws -> Expr {
        let toks = try ExprTokenizer.tokenize(source)
        guard !toks.isEmpty else {
            throw JinjaTemplateError.syntaxError("Empty expression.")
        }
        var parser = ExprParser(tokens: toks)
        let expr = try parser.parseTupleOrTernary()
        if parser.pos < parser.tokens.count {
            throw JinjaTemplateError.syntaxError("Unexpected trailing token '\(parser.tokens[parser.pos])' in '\(source)'.")
        }
        return expr
    }

    mutating func parseTupleOrTernary() throws -> Expr {
        let first = try parseTernary()
        if current == .comma {
            var items = [first]
            while current == .comma {
                _ = advance()
                if current == nil || current == .rparen { break }
                items.append(try parseTernary())
            }
            return .tuple(items)
        }
        return first
    }

    mutating func parseTernary() throws -> Expr {
        let trueExpr = try parseOr()
        if current == .ident("if") {
            _ = advance()
            let cond = try parseOr()
            var falseExpr: Expr?
            if current == .ident("else") {
                _ = advance()
                falseExpr = try parseTernary()
            }
            return .conditional(cond: cond, ifTrue: trueExpr, ifFalse: falseExpr)
        }
        return trueExpr
    }

    mutating func parseOr() throws -> Expr {
        var left = try parseAnd()
        while current == .ident("or") {
            _ = advance()
            let right = try parseAnd()
            left = .binary(op: "or", left, right)
        }
        return left
    }

    mutating func parseAnd() throws -> Expr {
        var left = try parseNot()
        while current == .ident("and") {
            _ = advance()
            let right = try parseNot()
            left = .binary(op: "and", left, right)
        }
        return left
    }

    mutating func parseNot() throws -> Expr {
        if current == .ident("not") {
            _ = advance()
            return .unary(op: "not", try parseNot())
        }
        return try parseCompare()
    }

    mutating func parseCompare() throws -> Expr {
        let left = try parseConcat()
        var comparisons: [(kind: CompareStep, right: Expr)] = []

        enum CompareStep {
            case binary(String)
            case test(name: String, args: [Expr], negated: Bool)
        }

        while let tok = current {
            if case .op(let op) = tok, ["==", "!=", "<", "<=", ">", ">="].contains(op) {
                _ = advance()
                let right = try parseConcat()
                comparisons.append((.binary(op), right))
            } else if tok == .ident("in") {
                _ = advance()
                let right = try parseConcat()
                comparisons.append((.binary("in"), right))
            } else if tok == .ident("not"), pos + 1 < tokens.count, tokens[pos + 1] == .ident("in") {
                _ = advance()
                _ = advance()
                let right = try parseConcat()
                comparisons.append((.binary("not in"), right))
            } else if tok == .ident("is") {
                _ = advance()
                var negated = false
                if current == .ident("not") {
                    negated = true
                    _ = advance()
                }
                let testName: String
                if case .ident(let id) = current {
                    testName = id
                    _ = advance()
                } else if current == .op("in") {
                    testName = "in"
                    _ = advance()
                } else {
                    throw JinjaTemplateError.syntaxError("Expected test name after 'is'.")
                }
                var testArgs: [Expr] = []
                if current == .lparen {
                    _ = advance()
                    if current != .rparen {
                        while true {
                            testArgs.append(try parseTernary())
                            if current == .comma { _ = advance(); continue }
                            break
                        }
                    }
                    try expect(.rparen)
                } else if ["equalto", "eq", "ne", "lt", "le", "gt", "ge", "divisibleby", "in", "startingwith", "endingwith"].contains(testName),
                          let next = current,
                          next != .ident("and"), next != .ident("or"), next != .ident("if"), next != .ident("else"),
                          next != .rparen, next != .rbracket, next != .comma, next != .colon {
                    testArgs.append(try parseConcat())
                }
                comparisons.append((.test(name: testName, args: testArgs, negated: negated), .literal(.null)))
            } else {
                break
            }
        }

        if comparisons.isEmpty {
            return left
        }
        if comparisons.count == 1 {
            switch comparisons[0].kind {
            case .binary(let op):
                return .binary(op: op, left, comparisons[0].right)
            case .test(let name, let args, let negated):
                return .test(left, name: name, args: args, negated: negated)
            }
        }

        // Chained comparisons (`0 <= i < n`)
        var combined: Expr?
        var prev = left
        for step in comparisons {
            let clause: Expr
            switch step.kind {
            case .binary(let op):
                clause = .binary(op: op, prev, step.right)
                prev = step.right
            case .test(let name, let args, let negated):
                clause = .test(prev, name: name, args: args, negated: negated)
            }
            if let existing = combined {
                combined = .binary(op: "and", existing, clause)
            } else {
                combined = clause
            }
        }
        return combined ?? left
    }

    mutating func parseConcat() throws -> Expr {
        var left = try parseAddSub()
        while current == .op("~") {
            _ = advance()
            let right = try parseAddSub()
            left = .binary(op: "~", left, right)
        }
        return left
    }

    mutating func parseAddSub() throws -> Expr {
        var left = try parseMulDiv()
        while case .op(let op) = current, op == "+" || op == "-" {
            _ = advance()
            let right = try parseMulDiv()
            left = .binary(op: op, left, right)
        }
        return left
    }

    mutating func parseMulDiv() throws -> Expr {
        var left = try parseUnary()
        while case .op(let op) = current, ["*", "/", "//", "%"].contains(op) {
            _ = advance()
            let right = try parseUnary()
            left = .binary(op: op, left, right)
        }
        return left
    }

    mutating func parseUnary() throws -> Expr {
        if case .op(let op) = current, op == "-" || op == "+" {
            _ = advance()
            return .unary(op: op, try parseUnary())
        }
        return try parseFilter()
    }

    mutating func parseFilter() throws -> Expr {
        var expr = try parsePower()
        while current == .pipe {
            _ = advance()
            guard case .ident(let filterName) = advance() else {
                throw JinjaTemplateError.syntaxError("Expected filter name after '|'.")
            }
            var args: [Expr] = []
            var kwargs: [(String, Expr)] = []
            if current == .lparen {
                _ = advance()
                try parseCallArgs(args: &args, kwargs: &kwargs)
                try expect(.rparen)
            }
            expr = .filter(expr, name: filterName, args: args, kwargs: kwargs)
        }
        return expr
    }

    mutating func parsePower() throws -> Expr {
        let base = try parsePostfix()
        if current == .op("**") {
            _ = advance()
            let exp = try parseUnary()
            return .binary(op: "**", base, exp)
        }
        return base
    }

    mutating func parsePostfix() throws -> Expr {
        var expr = try parsePrimary()
        while let tok = current {
            if tok == .dot {
                _ = advance()
                if case .ident(let attr) = current {
                    _ = advance()
                    expr = .getattr(expr, attr)
                } else if case .int(let idx) = current {
                    _ = advance()
                    expr = .getitem(expr, .literal(.int(idx)))
                } else {
                    throw JinjaTemplateError.syntaxError("Expected attribute name after '.'.")
                }
            } else if tok == .lbracket {
                _ = advance()
                if current == .colon {
                    _ = advance()
                    let stop: Expr? = (current == .rbracket || current == .colon) ? nil : try parseTernary()
                    var step: Expr?
                    if current == .colon {
                        _ = advance()
                        if current != .rbracket {
                            step = try parseTernary()
                        }
                    }
                    try expect(.rbracket)
                    expr = .slice(expr, start: nil, stop: stop, step: step)
                } else {
                    let first = try parseTernary()
                    if current == .colon {
                        _ = advance()
                        let stop: Expr? = (current == .rbracket || current == .colon) ? nil : try parseTernary()
                        var step: Expr?
                        if current == .colon {
                            _ = advance()
                            if current != .rbracket {
                                step = try parseTernary()
                            }
                        }
                        try expect(.rbracket)
                        expr = .slice(expr, start: first, stop: stop, step: step)
                    } else {
                        try expect(.rbracket)
                        expr = .getitem(expr, first)
                    }
                }
            } else if tok == .lparen {
                _ = advance()
                var args: [Expr] = []
                var kwargs: [(String, Expr)] = []
                try parseCallArgs(args: &args, kwargs: &kwargs)
                try expect(.rparen)
                expr = .call(expr, args: args, kwargs: kwargs)
            } else {
                break
            }
        }
        return expr
    }

    mutating func parseCallArgs(args: inout [Expr], kwargs: inout [(String, Expr)]) throws {
        if current == .rparen { return }
        while true {
            if current == .rparen { break }
            if case .ident(let key) = current, pos + 1 < tokens.count, tokens[pos + 1] == .assign {
                _ = advance()
                _ = advance()
                let val = try parseTernary()
                kwargs.append((key, val))
            } else {
                args.append(try parseTernary())
            }
            if current == .comma {
                _ = advance()
                continue
            }
            break
        }
    }

    mutating func parsePrimary() throws -> Expr {
        guard let tok = advance() else {
            throw JinjaTemplateError.syntaxError("Unexpected end of expression.")
        }
        switch tok {
        case .int(let i):
            return .literal(.int(i))
        case .double(let d):
            return .literal(.double(d))
        case .string(let s):
            return .literal(.string(s))
        case .ident(let name):
            switch name {
            case "true", "True": return .literal(.bool(true))
            case "false", "False": return .literal(.bool(false))
            case "none", "None", "null": return .literal(.null)
            default: return .variable(name)
            }
        case .lparen:
            if current == .rparen {
                _ = advance()
                return .tuple([])
            }
            let inner = try parseTupleOrTernary()
            try expect(.rparen)
            return inner
        case .lbracket:
            var items: [Expr] = []
            if current != .rbracket {
                while true {
                    if current == .rbracket { break }
                    items.append(try parseTernary())
                    if current == .comma { _ = advance(); continue }
                    break
                }
            }
            try expect(.rbracket)
            return .list(items)
        case .lbrace:
            var pairs: [(Expr, Expr)] = []
            if current != .rbrace {
                while true {
                    if current == .rbrace { break }
                    let key = try parseTernary()
                    try expect(.colon)
                    let val = try parseTernary()
                    pairs.append((key, val))
                    if current == .comma { _ = advance(); continue }
                    break
                }
            }
            try expect(.rbrace)
            return .dict(pairs)
        default:
            throw JinjaTemplateError.syntaxError("Unexpected token \(tok) in expression.")
        }
    }
}

// MARK: - Statement AST & Recursive-Descent Parser

private indirect enum Stmt {
    case text(String)
    case output(Expr)
    case ifStmt(branches: [(cond: Expr, body: [Stmt])], elseBody: [Stmt])
    case forStmt(targets: [String], iter: Expr, filter: Expr?, body: [Stmt], elseBody: [Stmt])
    case setAssign(targets: [String], attr: String?, value: Expr)
    case setBlock(target: String, body: [Stmt])
    case macroDef(name: String, params: [(name: String, defaultVal: Expr?)], body: [Stmt])
    case doStmt(Expr)
    case breakStmt
    case continueStmt
}

private enum TemplateParser {
    static func parse(_ chunks: [TemplateChunk]) throws -> [Stmt] {
        var idx = 0
        let stmts = try parseBlock(chunks, idx: &idx, stopKeywords: [])
        if idx < chunks.count {
            throw JinjaTemplateError.syntaxError("Unexpected block closing tag.")
        }
        return stmts
    }

    private static func parseBlock(
        _ chunks: [TemplateChunk],
        idx: inout Int,
        stopKeywords: Set<String>
    ) throws -> [Stmt] {
        var result: [Stmt] = []

        while idx < chunks.count {
            switch chunks[idx] {
            case .text(let s):
                result.append(.text(s))
                idx += 1

            case .expr(let raw):
                let parsed = try ExprParser.parseFullExpression(raw)
                result.append(.output(parsed))
                idx += 1

            case .stmt(let raw):
                let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                let keyword = leadingIdentifier(trimmed)

                if stopKeywords.contains(keyword) {
                    return result
                }

                switch keyword {
                case "if":
                    let condSrc = dropKeyword("if", from: trimmed)
                    let cond = try ExprParser.parseFullExpression(condSrc)
                    idx += 1
                    var branches: [(cond: Expr, body: [Stmt])] = []
                    let firstBody = try parseBlock(chunks, idx: &idx, stopKeywords: ["elif", "else", "endif"])
                    branches.append((cond, firstBody))
                    var elseBody: [Stmt] = []

                    while idx < chunks.count, case .stmt(let nextRaw) = chunks[idx] {
                        let nextTrim = nextRaw.trimmingCharacters(in: .whitespacesAndNewlines)
                        let nextKw = leadingIdentifier(nextTrim)
                        if nextKw == "elif" {
                            let elifCondSrc = dropKeyword("elif", from: nextTrim)
                            let elifCond = try ExprParser.parseFullExpression(elifCondSrc)
                            idx += 1
                            let elifBody = try parseBlock(chunks, idx: &idx, stopKeywords: ["elif", "else", "endif"])
                            branches.append((elifCond, elifBody))
                        } else if nextKw == "else" {
                            idx += 1
                            elseBody = try parseBlock(chunks, idx: &idx, stopKeywords: ["endif"])
                            break
                        } else {
                            break
                        }
                    }

                    guard idx < chunks.count,
                          case .stmt(let closeRaw) = chunks[idx],
                          leadingIdentifier(closeRaw.trimmingCharacters(in: .whitespacesAndNewlines)) == "endif"
                    else {
                        throw JinjaTemplateError.syntaxError("Unclosed '{% if %}' block (missing '{% endif %}').")
                    }
                    idx += 1
                    result.append(.ifStmt(branches: branches, elseBody: elseBody))

                case "for":
                    let rest = dropKeyword("for", from: trimmed)
                    let (targets, iterExpr, filterExpr) = try parseForHeader(rest)
                    idx += 1
                    let body = try parseBlock(chunks, idx: &idx, stopKeywords: ["else", "endfor"])
                    var elseBody: [Stmt] = []
                    if idx < chunks.count,
                       case .stmt(let midRaw) = chunks[idx],
                       leadingIdentifier(midRaw.trimmingCharacters(in: .whitespacesAndNewlines)) == "else" {
                        idx += 1
                        elseBody = try parseBlock(chunks, idx: &idx, stopKeywords: ["endfor"])
                    }
                    guard idx < chunks.count,
                          case .stmt(let closeRaw) = chunks[idx],
                          leadingIdentifier(closeRaw.trimmingCharacters(in: .whitespacesAndNewlines)) == "endfor"
                    else {
                        throw JinjaTemplateError.syntaxError("Unclosed '{% for %}' block (missing '{% endfor %}').")
                    }
                    idx += 1
                    result.append(.forStmt(targets: targets, iter: iterExpr, filter: filterExpr, body: body, elseBody: elseBody))

                case "set":
                    let rest = dropKeyword("set", from: trimmed)
                    if let eqIdx = findTopLevelAssignIndex(in: rest) {
                        let lhs = String(rest[..<eqIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
                        let rhs = String(rest[rest.index(after: eqIdx)...]).trimmingCharacters(in: .whitespacesAndNewlines)
                        let valExpr = try ExprParser.parseFullExpression(rhs)
                        if lhs.contains(".") {
                            let parts = lhs.split(separator: ".", maxSplits: 1).map {
                                $0.trimmingCharacters(in: .whitespacesAndNewlines)
                            }
                            guard parts.count == 2, isValidIdentifier(parts[0]), isValidIdentifier(parts[1]) else {
                                throw JinjaTemplateError.syntaxError("Invalid namespace assignment target '\(lhs)'.")
                            }
                            result.append(.setAssign(targets: [parts[0]], attr: parts[1], value: valExpr))
                        } else {
                            let names = lhs.split(separator: ",").map {
                                $0.trimmingCharacters(in: .whitespacesAndNewlines)
                            }
                            guard !names.isEmpty, names.allSatisfy(isValidIdentifier) else {
                                throw JinjaTemplateError.syntaxError("Invalid '{% set %}' variable name '\(lhs)'.")
                            }
                            result.append(.setAssign(targets: names, attr: nil, value: valExpr))
                        }
                        idx += 1
                    } else {
                        guard isValidIdentifier(rest) else {
                            throw JinjaTemplateError.syntaxError("Invalid block '{% set \(rest) %}' target.")
                        }
                        idx += 1
                        let body = try parseBlock(chunks, idx: &idx, stopKeywords: ["endset"])
                        guard idx < chunks.count,
                              case .stmt(let closeRaw) = chunks[idx],
                              leadingIdentifier(closeRaw.trimmingCharacters(in: .whitespacesAndNewlines)) == "endset"
                        else {
                            throw JinjaTemplateError.syntaxError("Unclosed '{% set %}' block (missing '{% endset %}').")
                        }
                        idx += 1
                        result.append(.setBlock(target: rest, body: body))
                    }

                case "macro":
                    let rest = dropKeyword("macro", from: trimmed)
                    let (name, params) = try parseMacroHeader(rest)
                    idx += 1
                    let body = try parseBlock(chunks, idx: &idx, stopKeywords: ["endmacro"])
                    guard idx < chunks.count,
                          case .stmt(let closeRaw) = chunks[idx],
                          leadingIdentifier(closeRaw.trimmingCharacters(in: .whitespacesAndNewlines)) == "endmacro"
                    else {
                        throw JinjaTemplateError.syntaxError("Unclosed '{% macro \(name) %}' block (missing '{% endmacro %}').")
                    }
                    idx += 1
                    result.append(.macroDef(name: name, params: params, body: body))

                case "do":
                    let rest = dropKeyword("do", from: trimmed)
                    result.append(.doStmt(try ExprParser.parseFullExpression(rest)))
                    idx += 1

                case "break":
                    result.append(.breakStmt)
                    idx += 1

                case "continue":
                    result.append(.continueStmt)
                    idx += 1

                case "elif", "else", "endif", "endfor", "endset", "endmacro":
                    throw JinjaTemplateError.syntaxError("Unexpected '{% \(keyword) %}' without matching opening block.")

                default:
                    throw JinjaTemplateError.syntaxError("Unknown or unsupported Jinja statement tag '{% \(trimmed) %}'.")
                }
            }
        }

        return result
    }

    private static func leadingIdentifier(_ s: String) -> String {
        var out = ""
        for ch in s {
            if ch.isLetter || ch.isNumber || ch == "_" {
                out.append(ch)
            } else {
                break
            }
        }
        return out
    }

    private static func dropKeyword(_ kw: String, from s: String) -> String {
        String(s.dropFirst(kw.count)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func isValidIdentifier(_ s: String) -> Bool {
        guard let first = s.first, first.isLetter || first == "_" else { return false }
        return s.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
    }

    private static func findTopLevelAssignIndex(in s: String) -> String.Index? {
        var depth = 0
        var inSingle = false
        var inDouble = false
        var escaped = false
        var idx = s.startIndex

        while idx < s.endIndex {
            let ch = s[idx]
            if escaped {
                escaped = false
                idx = s.index(after: idx)
                continue
            }
            if ch == "\\" && (inSingle || inDouble) {
                escaped = true
                idx = s.index(after: idx)
                continue
            }
            if ch == "'" && !inDouble { inSingle.toggle() }
            else if ch == "\"" && !inSingle { inDouble.toggle() }
            else if !inSingle && !inDouble {
                if ch == "(" || ch == "[" || ch == "{" { depth += 1 }
                else if ch == ")" || ch == "]" || ch == "}" { depth = max(0, depth - 1) }
                else if ch == "=" && depth == 0 {
                    let prev: Character? = idx > s.startIndex ? s[s.index(before: idx)] : nil
                    let nextIdx = s.index(after: idx)
                    let next: Character? = nextIdx < s.endIndex ? s[nextIdx] : nil
                    if prev != "=" && prev != "!" && prev != "<" && prev != ">" && next != "=" {
                        return idx
                    }
                }
            }
            idx = s.index(after: idx)
        }
        return nil
    }

    private static func parseForHeader(_ raw: String) throws -> ([String], Expr, Expr?) {
        let toks = try ExprTokenizer.tokenize(raw)
        guard let inPos = toks.firstIndex(of: .ident("in")) else {
            throw JinjaTemplateError.syntaxError("Malformed '{% for %}' header (missing 'in'): '\(raw)'.")
        }
        var targets: [String] = []
        for tok in toks[0..<inPos] {
            if tok == .comma || tok == .lparen || tok == .rparen { continue }
            guard case .ident(let name) = tok else {
                throw JinjaTemplateError.syntaxError("Invalid loop target in '{% for \(raw) %}'.")
            }
            targets.append(name)
        }
        guard !targets.isEmpty else {
            throw JinjaTemplateError.syntaxError("Missing loop variable before 'in' in '{% for \(raw) %}'.")
        }

        let afterIn = Array(toks[(inPos + 1)...])
        guard !afterIn.isEmpty else {
            throw JinjaTemplateError.syntaxError("Missing iterable expression after 'in' in '{% for \(raw) %}'.")
        }

        // Split optional top-level `if <filter>` suffix on `for x in seq if cond`
        var depth = 0
        var filterIfIdx: Int?
        for (k, tok) in afterIn.enumerated() {
            switch tok {
            case .lparen, .lbracket, .lbrace: depth += 1
            case .rparen, .rbracket, .rbrace: depth = max(0, depth - 1)
            case .ident("if") where depth == 0 && k > 0:
                filterIfIdx = k
            default:
                break
            }
            if filterIfIdx != nil { break }
        }

        if let filterIfIdx {
            var iterParser = ExprParser(tokens: Array(afterIn[0..<filterIfIdx]))
            let iterExpr = try iterParser.parseTernary()
            var filtParser = ExprParser(tokens: Array(afterIn[(filterIfIdx + 1)...]))
            let filtExpr = try filtParser.parseTernary()
            return (targets, iterExpr, filtExpr)
        } else {
            var iterParser = ExprParser(tokens: afterIn)
            let iterExpr = try iterParser.parseTupleOrTernary()
            return (targets, iterExpr, nil)
        }
    }

    private static func parseMacroHeader(_ raw: String) throws -> (String, [(name: String, defaultVal: Expr?)]) {
        let toks = try ExprTokenizer.tokenize(raw)
        guard toks.count >= 3,
              case .ident(let name) = toks[0],
              toks[1] == .lparen,
              toks.last == .rparen
        else {
            throw JinjaTemplateError.syntaxError("Malformed '{% macro %}' signature: '\(raw)'.")
        }
        var parser = ExprParser(tokens: Array(toks[2..<(toks.count - 1)]))
        var params: [(name: String, defaultVal: Expr?)] = []
        while parser.current != nil {
            guard case .ident(let paramName) = parser.advance() else {
                throw JinjaTemplateError.syntaxError("Invalid macro parameter in '\(raw)'.")
            }
            var defExpr: Expr?
            if parser.current == .assign {
                _ = parser.advance()
                defExpr = try parser.parseTernary()
            }
            params.append((paramName, defExpr))
            if parser.current == .comma {
                _ = parser.advance()
            }
        }
        return (name, params)
    }
}

// MARK: - Jinja2 AST Evaluator

private final class TemplateEvaluator {
    private enum LoopSignal: Error {
        case breakLoop
        case continueLoop
    }

    private struct MacroRecord {
        let params: [(name: String, defaultVal: Expr?)]
        let body: [Stmt]
    }

    private var scopes: [[String: JinjaValue]]
    private var macros: [String: MacroRecord] = [:]

    init(rootGlobals: [String: JinjaValue]) {
        self.scopes = [rootGlobals]
    }

    func evaluate(_ stmts: [Stmt]) throws -> String {
        var out = ""
        try execBlock(stmts, into: &out)
        return out
    }

    private func lookup(_ name: String) -> JinjaValue {
        for scope in scopes.reversed() {
            if let val = scope[name] {
                return val
            }
        }
        return .undefined
    }

    private func assignInCurrentScope(_ name: String, value: JinjaValue) {
        scopes[scopes.count - 1][name] = value
    }

    private func execBlock(_ stmts: [Stmt], into output: inout String) throws {
        for stmt in stmts {
            switch stmt {
            case .text(let s):
                output.append(s)

            case .output(let expr):
                let val = try eval(expr)
                output.append(val.renderString)

            case .ifStmt(let branches, let elseBody):
                var matched = false
                for branch in branches {
                    if (try eval(branch.cond)).isTruthy {
                        try execBlock(branch.body, into: &output)
                        matched = true
                        break
                    }
                }
                if !matched {
                    try execBlock(elseBody, into: &output)
                }

            case .forStmt(let targets, let iterExpr, let filterExpr, let body, let elseBody):
                let rawSeq = try evalSequence(try eval(iterExpr))
                var items: [JinjaValue] = []
                if let filterExpr {
                    for elem in rawSeq {
                        scopes.append([:])
                        try bindTargets(targets, to: elem)
                        let keep = (try eval(filterExpr)).isTruthy
                        scopes.removeLast()
                        if keep { items.append(elem) }
                    }
                } else {
                    items = rawSeq
                }

                if items.isEmpty {
                    try execBlock(elseBody, into: &output)
                } else {
                    let count = items.count
                    scopes.append([:])
                    defer { scopes.removeLast() }

                    for (idx, item) in items.enumerated() {
                        try bindTargets(targets, to: item)
                        let loopDict: [String: JinjaValue] = [
                            "index": .int(idx + 1),
                            "index0": .int(idx),
                            "revindex": .int(count - idx),
                            "revindex0": .int(count - idx - 1),
                            "first": .bool(idx == 0),
                            "last": .bool(idx == count - 1),
                            "length": .int(count),
                            "previtem": idx > 0 ? items[idx - 1] : .undefined,
                            "nextitem": (idx + 1 < count) ? items[idx + 1] : .undefined,
                        ]
                        assignInCurrentScope("loop", value: .object(loopDict))
                        do {
                            try execBlock(body, into: &output)
                        } catch LoopSignal.continueLoop {
                            continue
                        } catch LoopSignal.breakLoop {
                            break
                        }
                    }
                }

            case .setAssign(let targets, let attr, let valueExpr):
                let val = try eval(valueExpr)
                if let attr {
                    let targetVar = targets[0]
                    if case .namespace(let ns) = lookup(targetVar) {
                        ns.set(attr, value: val)
                    } else {
                        throw JinjaTemplateError.evaluationError("Cannot set attribute '\(attr)' on non-namespace '\(targetVar)'.")
                    }
                } else {
                    try bindTargets(targets, to: val)
                }

            case .setBlock(let target, let body):
                var captured = ""
                try execBlock(body, into: &captured)
                assignInCurrentScope(target, value: .string(captured))

            case .macroDef(let name, let params, let body):
                macros[name] = MacroRecord(params: params, body: body)

            case .doStmt(let expr):
                _ = try eval(expr)

            case .breakStmt:
                throw LoopSignal.breakLoop

            case .continueStmt:
                throw LoopSignal.continueLoop
            }
        }
    }

    private func bindTargets(_ targets: [String], to value: JinjaValue) throws {
        if targets.count == 1 {
            assignInCurrentScope(targets[0], value: value)
            return
        }
        let seq = try evalSequence(value)
        guard seq.count == targets.count else {
            throw JinjaTemplateError.evaluationError("Cannot unpack \(seq.count) values into \(targets.count) targets.")
        }
        for (name, elem) in zip(targets, seq) {
            assignInCurrentScope(name, value: elem)
        }
    }

    private func evalSequence(_ val: JinjaValue) throws -> [JinjaValue] {
        switch val {
        case .array(let a), .tuple(let a):
            return a
        case .string(let s):
            return s.map { .string(String($0)) }
        case .object(let o):
            return o.keys.sorted().map { .string($0) }
        case .undefined, .null:
            return []
        default:
            throw JinjaTemplateError.evaluationError("Value '\(val.renderString)' is not iterable.")
        }
    }

    private func eval(_ expr: Expr) throws -> JinjaValue {
        switch expr {
        case .literal(let v):
            return v

        case .variable(let name):
            return lookup(name)

        case .list(let items):
            return .array(try items.map { try eval($0) })

        case .tuple(let items):
            return .tuple(try items.map { try eval($0) })

        case .dict(let pairs):
            var out: [String: JinjaValue] = [:]
            for (kExpr, vExpr) in pairs {
                let k = (try eval(kExpr)).renderString
                out[k] = try eval(vExpr)
            }
            return .object(out)

        case .unary(let op, let sub):
            let v = try eval(sub)
            switch op {
            case "not":
                return .bool(!v.isTruthy)
            case "-":
                if case .int(let i) = v { return .int(-i) }
                if let n = v.asNumber { return .double(-n) }
                throw JinjaTemplateError.evaluationError("Unary '-' requires numeric operand.")
            case "+":
                if case .int(let i) = v { return .int(i) }
                if let n = v.asNumber { return .double(n) }
                throw JinjaTemplateError.evaluationError("Unary '+' requires numeric operand.")
            default:
                return v
            }

        case .conditional(let cond, let ifTrue, let ifFalse):
            if (try eval(cond)).isTruthy {
                return try eval(ifTrue)
            } else if let ifFalse {
                return try eval(ifFalse)
            } else {
                return .undefined
            }

        case .binary(let op, let lhsExpr, let rhsExpr):
            if op == "and" {
                let l = try eval(lhsExpr)
                return l.isTruthy ? (try eval(rhsExpr)) : l
            }
            if op == "or" {
                let l = try eval(lhsExpr)
                return l.isTruthy ? l : (try eval(rhsExpr))
            }
            let l = try eval(lhsExpr)
            let r = try eval(rhsExpr)
            return try evalBinary(op: op, lhs: l, rhs: r)

        case .getattr(let baseExpr, let attr):
            let base = try eval(baseExpr)
            return resolveAttribute(base: base, attr: attr)

        case .getitem(let baseExpr, let keyExpr):
            let base = try eval(baseExpr)
            let key = try eval(keyExpr)
            return resolveSubscript(base: base, key: key)

        case .slice(let baseExpr, let startExpr, let stopExpr, let stepExpr):
            let base = try eval(baseExpr)
            let start = try startExpr.flatMap { try eval($0).asInt }
            let stop = try stopExpr.flatMap { try eval($0).asInt }
            let step = try stepExpr.flatMap { try eval($0).asInt } ?? 1
            return try applySlice(base: base, start: start, stop: stop, step: step)

        case .call(let fnExpr, let argExprs, let kwExprs):
            let args = try argExprs.map { try eval($0) }
            let kwargs = try kwExprs.map { ($0.0, try eval($0.1)) }

            if case .variable(let fnName) = fnExpr {
                return try evalFunctionCall(name: fnName, args: args, kwargs: kwargs)
            }
            if case .getattr(let recvExpr, let methodName) = fnExpr {
                if case .variable(let varName) = recvExpr, methodName == "append" {
                    let recv = lookup(varName)
                    if case .array(var arr) = recv, let first = args.first {
                        arr.append(first)
                        updateVariableInNearestScope(varName, value: .array(arr))
                        return .null
                    }
                }
                let recv = try eval(recvExpr)
                return try evalMethodCall(receiver: recv, method: methodName, args: args, kwargs: kwargs)
            }
            throw JinjaTemplateError.evaluationError("Expression is not callable.")

        case .filter(let targetExpr, let name, let argExprs, let kwExprs):
            let target = try eval(targetExpr)
            let args = try argExprs.map { try eval($0) }
            let kwargs = try kwExprs.map { ($0.0, try eval($0.1)) }
            return try applyFilter(target: target, name: name, args: args, kwargs: kwargs)

        case .test(let targetExpr, let name, let argExprs, let negated):
            let target = try eval(targetExpr)
            let args = try argExprs.map { try eval($0) }
            let res = try applyTest(target: target, name: name, args: args)
            return .bool(negated ? !res : res)
        }
    }

    private func updateVariableInNearestScope(_ name: String, value: JinjaValue) {
        for idx in scopes.indices.reversed() {
            if scopes[idx][name] != nil {
                scopes[idx][name] = value
                return
            }
        }
        assignInCurrentScope(name, value: value)
    }

    private func resolveAttribute(base: JinjaValue, attr: String) -> JinjaValue {
        switch base {
        case .object(let dict):
            return dict[attr] ?? .undefined
        case .namespace(let ns):
            return ns.get(attr)
        default:
            return .undefined
        }
    }

    private func resolveSubscript(base: JinjaValue, key: JinjaValue) -> JinjaValue {
        switch base {
        case .object(let dict):
            return dict[key.renderString] ?? .undefined
        case .namespace(let ns):
            return ns.get(key.renderString)
        case .array(let arr), .tuple(let arr):
            guard let rawIdx = key.asInt else { return .undefined }
            let idx = rawIdx < 0 ? arr.count + rawIdx : rawIdx
            guard idx >= 0 && idx < arr.count else { return .undefined }
            return arr[idx]
        case .string(let s):
            guard let rawIdx = key.asInt else { return .undefined }
            let chars = Array(s)
            let idx = rawIdx < 0 ? chars.count + rawIdx : rawIdx
            guard idx >= 0 && idx < chars.count else { return .undefined }
            return .string(String(chars[idx]))
        default:
            return .undefined
        }
    }

    private func applySlice(base: JinjaValue, start: Int?, stop: Int?, step: Int) throws -> JinjaValue {
        guard step != 0 else {
            throw JinjaTemplateError.evaluationError("Slice step cannot be zero.")
        }
        func sliceIndices(count: Int) -> [Int] {
            if count == 0 { return [] }
            var indices: [Int] = []
            if step > 0 {
                var s = start ?? 0
                var e = stop ?? count
                if s < 0 { s += count }
                if e < 0 { e += count }
                s = max(0, min(count, s))
                e = max(0, min(count, e))
                var cur = s
                while cur < e {
                    indices.append(cur)
                    cur += step
                }
            } else {
                var s = start ?? (count - 1)
                var e: Int
                if let stop {
                    e = stop < 0 ? stop + count : stop
                    e = max(-1, min(count - 1, e))
                } else {
                    e = -1
                }
                if s < 0 { s += count }
                s = max(-1, min(count - 1, s))
                var cur = s
                while cur > e {
                    if cur >= 0 && cur < count {
                        indices.append(cur)
                    }
                    cur += step
                }
            }
            return indices
        }

        switch base {
        case .array(let arr):
            return .array(sliceIndices(count: arr.count).map { arr[$0] })
        case .tuple(let arr):
            return .tuple(sliceIndices(count: arr.count).map { arr[$0] })
        case .string(let s):
            let chars = Array(s)
            return .string(String(sliceIndices(count: chars.count).map { chars[$0] }))
        default:
            return .array([])
        }
    }

    private func evalBinary(op: String, lhs: JinjaValue, rhs: JinjaValue) throws -> JinjaValue {
        switch op {
        case "~":
            return .string(lhs.renderString + rhs.renderString)
        case "+":
            if case .string(let a) = lhs, case .string(let b) = rhs {
                return .string(a + b)
            }
            if case .array(let a) = lhs, case .array(let b) = rhs {
                return .array(a + b)
            }
            if case .tuple(let a) = lhs, case .tuple(let b) = rhs {
                return .tuple(a + b)
            }
            if case .int(let a) = lhs, case .int(let b) = rhs {
                return .int(a + b)
            }
            if let a = lhs.asNumber, let b = rhs.asNumber {
                return .double(a + b)
            }
            return .string(lhs.renderString + rhs.renderString)
        case "-":
            if case .int(let a) = lhs, case .int(let b) = rhs { return .int(a - b) }
            if let a = lhs.asNumber, let b = rhs.asNumber { return .double(a - b) }
            throw JinjaTemplateError.evaluationError("Operator '-' requires numeric operands.")
        case "*":
            if case .string(let s) = lhs, let n = rhs.asInt {
                return .string(String(repeating: s, count: max(0, n)))
            }
            if case .int(let a) = lhs, case .int(let b) = rhs { return .int(a * b) }
            if let a = lhs.asNumber, let b = rhs.asNumber { return .double(a * b) }
            throw JinjaTemplateError.evaluationError("Operator '*' requires numeric operands.")
        case "/":
            guard let a = lhs.asNumber, let b = rhs.asNumber, b != 0 else {
                throw JinjaTemplateError.evaluationError("Invalid division.")
            }
            return .double(a / b)
        case "//":
            guard let a = lhs.asNumber, let b = rhs.asNumber, b != 0 else {
                throw JinjaTemplateError.evaluationError("Invalid floor division.")
            }
            return .int(Int(floor(a / b)))
        case "%":
            if case .int(let a) = lhs, case .int(let b) = rhs, b != 0 {
                let rem = a % b
                return .int((rem < 0 && (b > 0)) || (rem > 0 && (b < 0)) ? rem + b : rem)
            }
            guard let a = lhs.asNumber, let b = rhs.asNumber, b != 0 else {
                throw JinjaTemplateError.evaluationError("Invalid modulo.")
            }
            return .double(a.truncatingRemainder(dividingBy: b))
        case "**":
            guard let a = lhs.asNumber, let b = rhs.asNumber else {
                throw JinjaTemplateError.evaluationError("Invalid power operands.")
            }
            return .double(pow(a, b))
        case "==":
            return .bool(valuesEqual(lhs, rhs))
        case "!=":
            return .bool(!valuesEqual(lhs, rhs))
        case "<", "<=", ">", ">=":
            if let a = lhs.asNumber, let b = rhs.asNumber {
                switch op {
                case "<": return .bool(a < b)
                case "<=": return .bool(a <= b)
                case ">": return .bool(a > b)
                default: return .bool(a >= b)
                }
            }
            let a = lhs.renderString
            let b = rhs.renderString
            switch op {
            case "<": return .bool(a < b)
            case "<=": return .bool(a <= b)
            case ">": return .bool(a > b)
            default: return .bool(a >= b)
            }
        case "in":
            return .bool(valueContains(container: rhs, item: lhs))
        case "not in":
            return .bool(!valueContains(container: rhs, item: lhs))
        default:
            throw JinjaTemplateError.evaluationError("Unsupported binary operator '\(op)'.")
        }
    }

    private func valuesEqual(_ a: JinjaValue, _ b: JinjaValue) -> Bool {
        switch (a, b) {
        case (.undefined, .undefined), (.null, .null), (.undefined, .null), (.null, .undefined):
            return true
        case (.bool(let x), .bool(let y)):
            return x == y
        case (.int(let x), .int(let y)):
            return x == y
        case (.int(let x), .double(let y)):
            return Double(x) == y
        case (.double(let x), .int(let y)):
            return x == Double(y)
        case (.double(let x), .double(let y)):
            return x == y
        case (.string(let x), .string(let y)):
            return x == y
        case (.array(let x), .array(let y)), (.tuple(let x), .tuple(let y)):
            guard x.count == y.count else { return false }
            return zip(x, y).allSatisfy { valuesEqual($0, $1) }
        default:
            return a == b
        }
    }

    private func valueContains(container: JinjaValue, item: JinjaValue) -> Bool {
        switch container {
        case .string(let s):
            return s.contains(item.renderString)
        case .array(let arr), .tuple(let arr):
            return arr.contains { valuesEqual($0, item) }
        case .object(let dict):
            return dict[item.renderString] != nil
        case .namespace(let ns):
            return ns.snapshot()[item.renderString] != nil
        default:
            return false
        }
    }

    private func evalFunctionCall(
        name: String,
        args: [JinjaValue],
        kwargs: [(String, JinjaValue)]
    ) throws -> JinjaValue {
        if let macro = macros[name] {
            scopes.append([:])
            defer { scopes.removeLast() }
            let kwMap = Dictionary(uniqueKeysWithValues: kwargs)
            for (idx, param) in macro.params.enumerated() {
                if idx < args.count {
                    assignInCurrentScope(param.name, value: args[idx])
                } else if let kwVal = kwMap[param.name] {
                    assignInCurrentScope(param.name, value: kwVal)
                } else if let defExpr = param.defaultVal {
                    assignInCurrentScope(param.name, value: try eval(defExpr))
                } else {
                    assignInCurrentScope(param.name, value: .undefined)
                }
            }
            var out = ""
            try execBlock(macro.body, into: &out)
            return .string(out)
        }

        switch name {
        case "raise_exception":
            let msg = args.first?.renderString ?? "Template exception"
            throw JinjaTemplateError.templateRaisedException(msg)

        case "namespace":
            var dict: [String: JinjaValue] = [:]
            for (k, v) in kwargs {
                dict[k] = v
            }
            return .namespace(JinjaValue.NamespaceBox(dict))

        case "range":
            let start: Int
            let stop: Int
            let step: Int
            if args.count == 1 {
                start = 0
                stop = args[0].asInt ?? 0
                step = 1
            } else if args.count == 2 {
                start = args[0].asInt ?? 0
                stop = args[1].asInt ?? 0
                step = 1
            } else if args.count >= 3 {
                start = args[0].asInt ?? 0
                stop = args[1].asInt ?? 0
                step = args[2].asInt ?? 1
            } else {
                return .array([])
            }
            guard step != 0 else { return .array([]) }
            var out: [JinjaValue] = []
            var cur = start
            if step > 0 {
                while cur < stop && out.count < 100_000 {
                    out.append(.int(cur))
                    cur += step
                }
            } else {
                while cur > stop && out.count < 100_000 {
                    out.append(.int(cur))
                    cur += step
                }
            }
            return .array(out)

        case "strftime_now":
            let fmt = args.first?.renderString ?? "%Y-%m-%d"
            return .string(formatStrftime(fmt, date: Date()))

        case "dict":
            var out: [String: JinjaValue] = [:]
            if let first = args.first, case .object(let base) = first {
                out = base
            }
            for (k, v) in kwargs {
                out[k] = v
            }
            return .object(out)

        default:
            return .undefined
        }
    }

    private func evalMethodCall(
        receiver: JinjaValue,
        method: String,
        args: [JinjaValue],
        kwargs: [(String, JinjaValue)]
    ) throws -> JinjaValue {
        switch receiver {
        case .string(let s):
            switch method {
            case "strip":
                if let charsArg = args.first?.renderString, !charsArg.isEmpty {
                    return .string(s.trimmingCharacters(in: CharacterSet(charactersIn: charsArg)))
                }
                return .string(s.trimmingCharacters(in: .whitespacesAndNewlines))
            case "lstrip":
                let set = (args.first?.renderString).flatMap { $0.isEmpty ? nil : CharacterSet(charactersIn: $0) }
                    ?? CharacterSet.whitespacesAndNewlines
                var idx = s.startIndex
                while idx < s.endIndex, s[idx].unicodeScalars.allSatisfy({ set.contains($0) }) {
                    idx = s.index(after: idx)
                }
                return .string(String(s[idx...]))
            case "rstrip":
                let set = (args.first?.renderString).flatMap { $0.isEmpty ? nil : CharacterSet(charactersIn: $0) }
                    ?? CharacterSet.whitespacesAndNewlines
                var chars = Array(s)
                while let last = chars.last, last.unicodeScalars.allSatisfy({ set.contains($0) }) {
                    chars.removeLast()
                }
                return .string(String(chars))
            case "startswith":
                guard let first = args.first else { return .bool(false) }
                if case .tuple(let opts) = first {
                    return .bool(opts.contains { s.hasPrefix($0.renderString) })
                }
                if case .array(let opts) = first {
                    return .bool(opts.contains { s.hasPrefix($0.renderString) })
                }
                return .bool(s.hasPrefix(first.renderString))
            case "endswith":
                guard let first = args.first else { return .bool(false) }
                if case .tuple(let opts) = first {
                    return .bool(opts.contains { s.hasSuffix($0.renderString) })
                }
                if case .array(let opts) = first {
                    return .bool(opts.contains { s.hasSuffix($0.renderString) })
                }
                return .bool(s.hasSuffix(first.renderString))
            case "split":
                if let sep = args.first, sep != .null, sep != .undefined {
                    let delim = sep.renderString
                    if delim.isEmpty {
                        return .array([.string(s)])
                    }
                    let maxSplit = args.count > 1 ? (args[1].asInt ?? -1) : -1
                    if maxSplit >= 0 {
                        var parts: [String] = []
                        var remainder = s
                        for _ in 0..<maxSplit {
                            guard let r = remainder.range(of: delim) else { break }
                            parts.append(String(remainder[..<r.lowerBound]))
                            remainder = String(remainder[r.upperBound...])
                        }
                        parts.append(remainder)
                        return .array(parts.map { .string($0) })
                    }
                    return .array(s.components(separatedBy: delim).map { .string($0) })
                }
                return .array(s.split(whereSeparator: { $0.isWhitespace }).map { .string(String($0)) })
            case "rsplit":
                if let sep = args.first, sep != .null, sep != .undefined {
                    let delim = sep.renderString
                    let maxSplit = args.count > 1 ? (args[1].asInt ?? -1) : -1
                    if maxSplit >= 0 {
                        var parts: [String] = []
                        var remainder = s
                        for _ in 0..<maxSplit {
                            guard let r = remainder.range(of: delim, options: .backwards) else { break }
                            parts.insert(String(remainder[r.upperBound...]), at: 0)
                            remainder = String(remainder[..<r.lowerBound])
                        }
                        parts.insert(remainder, at: 0)
                        return .array(parts.map { .string($0) })
                    }
                    return .array(s.components(separatedBy: delim).map { .string($0) })
                }
                return .array(s.split(whereSeparator: { $0.isWhitespace }).map { .string(String($0)) })
            case "lower":
                return .string(s.lowercased())
            case "upper":
                return .string(s.uppercased())
            case "title":
                return .string(s.capitalized)
            case "capitalize":
                guard let first = s.first else { return .string("") }
                return .string(String(first).uppercased() + s.dropFirst().lowercased())
            case "replace":
                guard args.count >= 2 else { return .string(s) }
                let oldStr = args[0].renderString
                let newStr = args[1].renderString
                if args.count >= 3, let limit = args[2].asInt, limit >= 0 {
                    var result = ""
                    var remainder = s
                    var replaced = 0
                    while replaced < limit, let r = remainder.range(of: oldStr) {
                        result += remainder[..<r.lowerBound] + newStr
                        remainder = String(remainder[r.upperBound...])
                        replaced += 1
                    }
                    result += remainder
                    return .string(result)
                }
                return .string(s.replacingOccurrences(of: oldStr, with: newStr))
            case "join":
                guard let first = args.first else { return .string("") }
                let seq = try evalSequence(first)
                return .string(seq.map { $0.renderString }.joined(separator: s))
            case "find":
                guard let sub = args.first?.renderString else { return .int(-1) }
                if let r = s.range(of: sub) {
                    return .int(s.distance(from: s.startIndex, to: r.lowerBound))
                }
                return .int(-1)
            case "isdigit":
                return .bool(!s.isEmpty && s.allSatisfy { $0.isNumber })
            default:
                return .undefined
            }

        case .object(let dict):
            switch method {
            case "get":
                guard let key = args.first?.renderString else { return .null }
                if let val = dict[key], val != .undefined {
                    return val
                }
                return args.count > 1 ? args[1] : .null
            case "items":
                return .array(dict.keys.sorted().map { k in .tuple([.string(k), dict[k] ?? .null]) })
            case "keys":
                return .array(dict.keys.sorted().map { .string($0) })
            case "values":
                return .array(dict.keys.sorted().map { dict[$0] ?? .null })
            default:
                return .undefined
            }

        case .array(let arr), .tuple(let arr):
            switch method {
            case "count":
                guard let target = args.first else { return .int(0) }
                return .int(arr.filter { valuesEqual($0, target) }.count)
            case "index":
                guard let target = args.first,
                      let idx = arr.firstIndex(where: { valuesEqual($0, target) }) else {
                    return .int(-1)
                }
                return .int(idx)
            default:
                return .undefined
            }

        default:
            return .undefined
        }
    }

    private func applyFilter(
        target: JinjaValue,
        name: String,
        args: [JinjaValue],
        kwargs: [(String, JinjaValue)]
    ) throws -> JinjaValue {
        let kwMap = Dictionary(uniqueKeysWithValues: kwargs)
        switch name {
        case "trim", "strip":
            let chars = args.first?.renderString
            if let chars, !chars.isEmpty {
                return .string(target.renderString.trimmingCharacters(in: CharacterSet(charactersIn: chars)))
            }
            return .string(target.renderString.trimmingCharacters(in: .whitespacesAndNewlines))

        case "lower":
            return .string(target.renderString.lowercased())

        case "upper":
            return .string(target.renderString.uppercased())

        case "title":
            return .string(target.renderString.capitalized)

        case "capitalize":
            let s = target.renderString
            guard let first = s.first else { return .string("") }
            return .string(String(first).uppercased() + s.dropFirst().lowercased())

        case "length", "count":
            switch target {
            case .string(let s): return .int(s.count)
            case .array(let a), .tuple(let a): return .int(a.count)
            case .object(let o): return .int(o.count)
            case .namespace(let ns): return .int(ns.snapshot().count)
            default: return .int(0)
            }

        case "first":
            let seq = try evalSequence(target)
            return seq.first ?? .undefined

        case "last":
            let seq = try evalSequence(target)
            return seq.last ?? .undefined

        case "reverse":
            if case .string(let s) = target {
                return .string(String(s.reversed()))
            }
            let seq = try evalSequence(target)
            return .array(seq.reversed())

        case "list":
            return .array(try evalSequence(target))

        case "string", "safe":
            return .string(target.renderString)

        case "int":
            let fallback = args.first?.asInt ?? 0
            return .int(target.asInt ?? fallback)

        case "float":
            let fallback = args.first?.asNumber ?? 0.0
            if let n = target.asNumber { return .double(n) }
            if let parsed = Double(target.renderString.trimmingCharacters(in: .whitespacesAndNewlines)) {
                return .double(parsed)
            }
            return .double(fallback)

        case "bool":
            return .bool(target.isTruthy)

        case "default", "d":
            let defVal = args.first ?? .string("")
            let checkTruthy = (args.count > 1 ? args[1].isTruthy : false)
                || (kwMap["boolean"]?.isTruthy ?? false)
            if target == .undefined || (checkTruthy && !target.isTruthy) {
                return defVal
            }
            return target

        case "join":
            let sep = args.first?.renderString ?? kwMap["d"]?.renderString ?? ""
            let attrName = kwMap["attribute"]?.renderString
            let seq = try evalSequence(target)
            let mapped = seq.map { elem -> String in
                if let attrName {
                    return resolveAttribute(base: elem, attr: attrName).renderString
                }
                return elem.renderString
            }
            return .string(mapped.joined(separator: sep))

        case "replace":
            guard args.count >= 2 else { return .string(target.renderString) }
            return try evalMethodCall(receiver: .string(target.renderString), method: "replace", args: args, kwargs: kwargs)

        case "tojson":
            let indent = args.first?.asInt ?? kwMap["indent"]?.asInt
            return .string(Self.formatJinjaJSON(target, indent: indent, depth: 0))

        case "indent":
            let width = args.first?.asInt ?? kwMap["width"]?.asInt ?? 4
            let indentFirst = (args.count > 1 ? args[1].isTruthy : false) || (kwMap["first"]?.isTruthy ?? false)
            let pad = String(repeating: " ", count: max(0, width))
            let lines = target.renderString.components(separatedBy: "\n")
            let indented = lines.enumerated().map { idx, line -> String in
                if line.isEmpty { return line }
                if idx == 0 && !indentFirst { return line }
                return pad + line
            }
            return .string(indented.joined(separator: "\n"))

        case "map":
            let seq = try evalSequence(target)
            if let attr = kwMap["attribute"]?.renderString ?? (args.count == 1 ? nil : nil) {
                return .array(seq.map { resolveAttribute(base: $0, attr: attr) })
            }
            if let filterName = args.first?.renderString {
                let restArgs = Array(args.dropFirst())
                return .array(try seq.map { try applyFilter(target: $0, name: filterName, args: restArgs, kwargs: []) })
            }
            return .array(seq)

        case "selectattr", "rejectattr":
            let reject = (name == "rejectattr")
            let seq = try evalSequence(target)
            guard let attr = args.first?.renderString else { return .array(seq) }
            let testName = args.count > 1 ? args[1].renderString : nil
            let testArgs = args.count > 2 ? Array(args.dropFirst(2)) : []
            var out: [JinjaValue] = []
            for item in seq {
                let prop = resolveAttribute(base: item, attr: attr)
                let matched: Bool
                if let testName {
                    matched = try applyTest(target: prop, name: testName, args: testArgs)
                } else {
                    matched = prop.isTruthy
                }
                if reject ? !matched : matched {
                    out.append(item)
                }
            }
            return .array(out)

        case "select", "reject":
            let reject = (name == "reject")
            let seq = try evalSequence(target)
            let testName = args.first?.renderString
            let testArgs = args.count > 1 ? Array(args.dropFirst()) : []
            var out: [JinjaValue] = []
            for item in seq {
                let matched = testName != nil
                    ? (try applyTest(target: item, name: testName!, args: testArgs))
                    : item.isTruthy
                if reject ? !matched : matched {
                    out.append(item)
                }
            }
            return .array(out)

        default:
            return target
        }
    }

    private func applyTest(target: JinjaValue, name: String, args: [JinjaValue]) throws -> Bool {
        switch name {
        case "defined":
            return target != .undefined
        case "undefined":
            return target == .undefined
        case "none", "null":
            return target == .null
        case "boolean":
            if case .bool = target { return true }
            return false
        case "true":
            return target == .bool(true)
        case "false":
            return target == .bool(false)
        case "string":
            if case .string = target { return true }
            return false
        case "number":
            if case .int = target { return true }
            if case .double = target { return true }
            return false
        case "integer":
            if case .int = target { return true }
            return false
        case "float":
            if case .double = target { return true }
            return false
        case "mapping":
            if case .object = target { return true }
            if case .namespace = target { return true }
            return false
        case "sequence", "iterable":
            switch target {
            case .array, .tuple, .string, .object: return true
            default: return false
            }
        case "odd":
            guard let i = target.asInt else { return false }
            return i % 2 != 0
        case "even":
            guard let i = target.asInt else { return false }
            return i % 2 == 0
        case "divisibleby":
            guard let i = target.asInt, let d = args.first?.asInt, d != 0 else { return false }
            return i % d == 0
        case "equalto", "eq":
            guard let other = args.first else { return false }
            return valuesEqual(target, other)
        case "ne":
            guard let other = args.first else { return false }
            return !valuesEqual(target, other)
        case "in":
            guard let seq = args.first else { return false }
            return valueContains(container: seq, item: target)
        case "startingwith":
            guard let prefix = args.first?.renderString else { return false }
            return target.renderString.hasPrefix(prefix)
        case "endingwith":
            guard let suffix = args.first?.renderString else { return false }
            return target.renderString.hasSuffix(suffix)
        default:
            return target.isTruthy
        }
    }

    private func formatStrftime(_ format: String, date: Date) -> String {
        let cal = Calendar(identifier: .gregorian)
        let comps = cal.dateComponents([.year, .month, .day, .hour, .minute, .second], from: date)
        let dfMonthFull = DateFormatter()
        dfMonthFull.locale = Locale(identifier: "en_US_POSIX")
        dfMonthFull.dateFormat = "MMMM"
        let dfMonthShort = DateFormatter()
        dfMonthShort.locale = Locale(identifier: "en_US_POSIX")
        dfMonthShort.dateFormat = "MMM"
        let dfDayFull = DateFormatter()
        dfDayFull.locale = Locale(identifier: "en_US_POSIX")
        dfDayFull.dateFormat = "EEEE"
        let dfDayShort = DateFormatter()
        dfDayShort.locale = Locale(identifier: "en_US_POSIX")
        dfDayShort.dateFormat = "EEE"

        return format
            .replacingOccurrences(of: "%Y", with: String(format: "%04d", comps.year ?? 2026))
            .replacingOccurrences(of: "%m", with: String(format: "%02d", comps.month ?? 1))
            .replacingOccurrences(of: "%d", with: String(format: "%02d", comps.day ?? 1))
            .replacingOccurrences(of: "%H", with: String(format: "%02d", comps.hour ?? 0))
            .replacingOccurrences(of: "%M", with: String(format: "%02d", comps.minute ?? 0))
            .replacingOccurrences(of: "%S", with: String(format: "%02d", comps.second ?? 0))
            .replacingOccurrences(of: "%B", with: dfMonthFull.string(from: date))
            .replacingOccurrences(of: "%b", with: dfMonthShort.string(from: date))
            .replacingOccurrences(of: "%A", with: dfDayFull.string(from: date))
            .replacingOccurrences(of: "%a", with: dfDayShort.string(from: date))
    }

    private static func formatJinjaJSON(_ value: JinjaValue, indent: Int?, depth: Int) -> String {
        switch value {
        case .null, .undefined:
            return "null"
        case .bool(let b):
            return b ? "true" : "false"
        case .int(let i):
            return String(i)
        case .double(let d):
            if d.isFinite {
                return d.rounded() == d && abs(d) < 1e15 ? "\(Int64(d)).0" : String(d)
            }
            return "null"
        case .string(let s):
            let enc = JSONEncoder()
            enc.outputFormatting = [.withoutEscapingSlashes]
            if let data = try? enc.encode(s), let encoded = String(data: data, encoding: .utf8) {
                return encoded
            }
            return "\"\(s)\""
        case .array(let items), .tuple(let items):
            guard !items.isEmpty else { return "[]" }
            if let ind = indent, ind > 0 {
                let childPad = String(repeating: " ", count: (depth + 1) * ind)
                let closePad = String(repeating: " ", count: depth * ind)
                let parts = items.map { childPad + formatJinjaJSON($0, indent: ind, depth: depth + 1) }
                return "[\n" + parts.joined(separator: ",\n") + "\n" + closePad + "]"
            }
            return "[" + items.map { formatJinjaJSON($0, indent: nil, depth: depth + 1) }.joined(separator: ", ") + "]"
        case .object(let dict):
            guard !dict.isEmpty else { return "{}" }
            let keys = dict.keys.sorted()
            if let ind = indent, ind > 0 {
                let childPad = String(repeating: " ", count: (depth + 1) * ind)
                let closePad = String(repeating: " ", count: depth * ind)
                let parts = keys.map { k in
                    let kJSON = formatJinjaJSON(.string(k), indent: nil, depth: depth + 1)
                    let vJSON = formatJinjaJSON(dict[k] ?? .null, indent: ind, depth: depth + 1)
                    return "\(childPad)\(kJSON): \(vJSON)"
                }
                return "{\n" + parts.joined(separator: ",\n") + "\n" + closePad + "}"
            }
            let parts = keys.map { k in
                let kJSON = formatJinjaJSON(.string(k), indent: nil, depth: depth + 1)
                let vJSON = formatJinjaJSON(dict[k] ?? .null, indent: nil, depth: depth + 1)
                return "\(kJSON): \(vJSON)"
            }
            return "{" + parts.joined(separator: ", ") + "}"
        case .namespace(let ns):
            return formatJinjaJSON(.object(ns.snapshot()), indent: indent, depth: depth)
        }
    }
}
