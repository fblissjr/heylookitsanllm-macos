import Foundation

/// File-derived reasoning stream protocol detected from a model's chat template and tokenizer (`VISION.md` Principle 2: "Derive, don't copy").
public enum ReasoningFormat: Sendable, Equatable, Codable {
    /// Delimiter-pair reasoning blocks such as `<think>...</think>`, `<|thought|>...<|/thought|>`, or `<thought>...</thought>`.
    /// When `prefillsThinking` is true, the chat template's generation prompt ends with an unclosed opening tag (`<think>\n`),
    /// so the model stream begins inside the reasoning span when thinking is enabled.
    case tagPair(open: String, close: String, prefillsThinking: Bool)

    /// Gemma 4 labeled channel protocol (`<|channel>thought\n...<channel|>`).
    case gemmaChannel(openPrefix: String, closeTag: String)

    /// OpenAI Harmony channel protocol (`<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>`).
    case harmonyChannel

    /// Recipient-addressed channel protocol (`Muse-Glimmer`: `to=self<|message|>...<|eom|><|start|>assistant to=user<|message|>...<|eot|>`).
    case recipientChannel

    /// Model does not emit structured reasoning blocks.
    case none

    /// Canonical wire identifier reported on `EngineDescription.reasoningFormat` and `/v1/models`.
    public var wireIdentifier: String {
        switch self {
        case .tagPair:
            return "tag_pair"
        case .gemmaChannel:
            return "gemma_channel"
        case .harmonyChannel:
            return "harmony"
        case .recipientChannel:
            return "recipient"
        case .none:
            return "none"
        }
    }

    /// True when the template's generation prompt opens a thinking block without closing it.
    public var prefillsThinking: Bool {
        if case .tagPair(_, _, let prefills) = self {
            return prefills
        }
        return false
    }

    /// Structural delimiters that `ReasoningParser` consumes to split thinking from content.
    /// These must NOT be pre-stripped by `StripSpecials` before `ReasoningParser` runs.
    public var structuralDelimiters: Set<String> {
        switch self {
        case .tagPair(let open, let close, _):
            return [open, close]
        case .gemmaChannel(let openPrefix, let closeTag):
            return [openPrefix, closeTag]
        case .harmonyChannel:
            return ["<|channel|>", "<|message|>", "<|start|>", "<|end|>"]
        case .recipientChannel:
            return ["<|message|>", "<|eom|>", "<|start|>"]
        case .none:
            return []
        }
    }
}

/// Template-derived thinking switch and reasoning depth controls (`enable_thinking`, `reasoning_effort`, `reasoning_strength`, `thinking_mode`).
public struct ThinkingControls: Sendable, Equatable, Codable {
    public let supportsThinking: Bool
    public let canDisableThinking: Bool
    public let defaultEnabled: Bool
    public let switchVariable: String?
    public let depthVariable: String?
    public let depthValues: [String]
    public let depthAliases: [String: String]
    public let depthDefault: String?
    public let depthUnknown: String?
    public let depthChangesPrefix: Bool?
    public let depthOffValues: [String]
    public let budgetEnforced: Bool?
    public let budgetReason: String?
    public let templateSourceName: String?

    public init(
        supportsThinking: Bool,
        canDisableThinking: Bool = true,
        defaultEnabled: Bool = true,
        switchVariable: String? = nil,
        depthVariable: String? = nil,
        depthValues: [String] = [],
        depthAliases: [String: String] = [:],
        depthDefault: String? = nil,
        depthUnknown: String? = nil,
        depthChangesPrefix: Bool? = nil,
        depthOffValues: [String] = [],
        budgetEnforced: Bool? = nil,
        budgetReason: String? = nil,
        templateSourceName: String? = nil
    ) {
        self.supportsThinking = supportsThinking
        self.canDisableThinking = canDisableThinking
        self.defaultEnabled = defaultEnabled
        self.switchVariable = switchVariable
        self.depthVariable = depthVariable
        self.depthValues = depthValues
        self.depthAliases = depthAliases
        self.depthDefault = depthDefault
        self.depthUnknown = depthUnknown
        self.depthChangesPrefix = depthChangesPrefix
        self.depthOffValues = depthOffValues
        self.budgetEnforced = budgetEnforced
        self.budgetReason = budgetReason
        self.templateSourceName = templateSourceName
    }

    public static let unsupported = ThinkingControls(
        supportsThinking: false,
        canDisableThinking: false,
        defaultEnabled: false
    )

    /// Validates a requested `thinking_depth` / `reasoning_effort` value against detected controls
    /// (`thinking_controls.py:check_depth`). Returns `nil` if accepted, or a human-readable refusal reason.
    public func checkDepth(_ value: String?) -> String? {
        guard let value, !value.isEmpty else { return nil }
        guard supportsThinking || depthVariable != nil else {
            return "this model's chat template has no thinking-depth control"
        }
        guard let depthVar = depthVariable else {
            return "this model's chat template has no thinking-depth control"
        }
        if depthUnknown == "verbatim" {
            return nil
        }
        if depthValues.contains(value) || depthAliases[value] != nil {
            return nil
        }
        let offered = depthValues.isEmpty ? "none" : depthValues.joined(separator: ", ")
        return "thinking depth '\(value)' is not offered by this model's chat template (\(depthVar): \(offered))"
    }

    /// Converts these controls into the canonical wire `EngineThinkingFacts` slot (`contract.py`).
    public func toEngineThinkingFacts(templateName: String? = nil) -> EngineThinkingFacts? {
        guard supportsThinking || switchVariable != nil || depthVariable != nil else {
            return nil
        }
        let depthFacts: EngineThinkingDepthFacts? = depthVariable.map { v in
            EngineThinkingDepthFacts(
                variable: v,
                values: depthValues,
                aliases: depthAliases,
                defaultValue: depthDefault,
                unknown: depthUnknown,
                changesPrefix: depthChangesPrefix,
                off: depthOffValues
            )
        }
        let budgetFacts: EngineThinkingBudgetFacts? = budgetEnforced.map { enforced in
            EngineThinkingBudgetFacts(
                enforced: enforced,
                reason: budgetReason ?? (
                    enforced
                        ? "heylook's engine closes the thinking block with one forced token"
                        : "the template's thinking markers were not found, so no cap can be forced"
                )
            )
        }
        return EngineThinkingFacts(
            switchVariable: switchVariable,
            depth: depthFacts,
            budget: budgetFacts,
            template: templateName ?? templateSourceName
        )
    }
}

/// Pure-Swift checkpoint inspector that derives a model's `ReasoningFormat`, `ThinkingControls`,
/// declared special tokens, and EOS token IDs directly from its own files (`chat_template.jinja`,
/// `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, or GGUF header) with
/// zero hardcoded model-family tables.
public struct ModelTemplateInfo: Sendable, Equatable {
    public let reasoningFormat: ReasoningFormat
    public let thinkingControls: ThinkingControls
    public let specialTokens: Set<String>
    public let strippableSpecialTokens: Set<String>
    public let eosTokenIDs: [Int]
    public let templateSource: String

    /// Inspects checkpoint files (`config.json`, `processor_config.json`, `preprocessor_config.json`,
    /// or `.aimodel/metadata.json`) to determine if the model directory ships a vision tower or image processor.
    /// Does not rely on hardcoded model_type lists or directory name substrings.
    public static func hasVisionConfiguration(
        in modelDirectory: URL,
        configJSON: [String: Any]? = nil
    ) -> Bool {
        let dirURL = modelDirectory.pathExtension.lowercased() == "gguf"
            ? modelDirectory.deletingLastPathComponent()
            : modelDirectory
        let cfg = configJSON ?? loadJSON(at: dirURL.appendingPathComponent("config.json"))
        if let visionCfg = cfg?["vision_config"] as? [String: Any], !visionCfg.isEmpty {
            return true
        }
        if let procCfg = loadJSON(at: dirURL.appendingPathComponent("processor_config.json")) {
            if procCfg["image_processor"] != nil || procCfg["patch_size"] != nil || procCfg["max_soft_tokens"] != nil {
                return true
            }
        }
        if let preprocCfg = loadJSON(at: dirURL.appendingPathComponent("preprocessor_config.json")) {
            if preprocCfg["image_processor_type"] != nil || preprocCfg["size"] != nil || preprocCfg["patch_size"] != nil {
                return true
            }
        }
        if let metaCfg = loadJSON(at: dirURL.appendingPathComponent("metadata.json")) {
            if (metaCfg["model_type"] as? String)?.lowercased() == "vlm" || metaCfg["vision"] != nil {
                return true
            }
            if let assets = metaCfg["assets"] as? [String: Any], assets["vision"] != nil {
                return true
            }
        }
        return false
    }

    public init(
        reasoningFormat: ReasoningFormat,
        thinkingControls: ThinkingControls,
        specialTokens: Set<String>,
        strippableSpecialTokens: Set<String>,
        eosTokenIDs: [Int] = [],
        templateSource: String = "tokenizer_config.json"
    ) {
        self.reasoningFormat = reasoningFormat
        self.thinkingControls = thinkingControls
        self.specialTokens = specialTokens
        self.strippableSpecialTokens = strippableSpecialTokens
        self.eosTokenIDs = eosTokenIDs
        self.templateSource = templateSource
    }

    /// Inspects a model checkpoint directory (or `.aimodel` bundle) and derives its reasoning format,
    /// thinking controls, special tokens, and EOS token IDs from disk.
    public static func inspect(
        modelDirectory: URL,
        chatTemplateOverride: String? = nil,
        modelID: String? = nil
    ) -> ModelTemplateInfo {
        let fm = FileManager.default
        let dirURL = modelDirectory.pathExtension.lowercased() == "gguf"
            ? modelDirectory.deletingLastPathComponent()
            : modelDirectory

        // If this is a CoreAI `.aimodel` bundle with a `tokenizer/` subdirectory, also inspect `tokenizer/`
        let tokenizerSubdir = dirURL.appendingPathComponent("tokenizer", isDirectory: true)
        let searchDirs: [URL] = fm.fileExists(atPath: tokenizerSubdir.path)
            ? [dirURL, tokenizerSubdir]
            : [dirURL]

        var specialTokens = Set<String>()
        var eosIDs: [Int] = []

        for folder in searchDirs {
            // 1. Read `tokenizer.json` -> `added_tokens` where `"special": true`
            if let tokJSON = loadJSON(at: folder.appendingPathComponent("tokenizer.json")),
               let addedTokens = tokJSON["added_tokens"] as? [[String: Any]] {
                for entry in addedTokens {
                    let isSpecial = (entry["special"] as? Bool) ?? false
                    if isSpecial, let content = entry["content"] as? String, !content.isEmpty {
                        specialTokens.insert(content)
                    }
                }
            }

            // 2. Read `tokenizer_config.json` -> `added_tokens_decoder`, `eos_token`, `bos_token`, `pad_token`
            if let tokCfg = loadJSON(at: folder.appendingPathComponent("tokenizer_config.json")) {
                if let decoderMap = tokCfg["added_tokens_decoder"] as? [String: Any] {
                    for (_, rawVal) in decoderMap {
                        if let dict = rawVal as? [String: Any],
                           (dict["special"] as? Bool) == true,
                           let content = dict["content"] as? String,
                           !content.isEmpty {
                            specialTokens.insert(content)
                        }
                    }
                }
                for key in ["eos_token", "bos_token", "pad_token", "eot_token"] {
                    if let s = tokCfg[key] as? String, !s.isEmpty {
                        specialTokens.insert(s)
                    } else if let dict = tokCfg[key] as? [String: Any],
                              let content = dict["content"] as? String,
                              !content.isEmpty {
                        specialTokens.insert(content)
                    }
                }
            }

            // 3. Read `generation_config.json` & `config.json` -> `eos_token_id`
            for cfgName in ["generation_config.json", "config.json"] {
                if let cfg = loadJSON(at: folder.appendingPathComponent(cfgName)) {
                    if let single = (cfg["eos_token_id"] as? NSNumber)?.intValue {
                        if !eosIDs.contains(single) { eosIDs.append(single) }
                    } else if let arr = cfg["eos_token_id"] as? [Any] {
                        for item in arr {
                            if let id = (item as? NSNumber)?.intValue, !eosIDs.contains(id) {
                                eosIDs.append(id)
                            }
                        }
                    }
                }
            }
        }

        // Resolve template from override or the 4-rung ladder
        let templateState: HeylookConfig.ChatTemplateState
        if let chatTemplateOverride, !chatTemplateOverride.isEmpty {
            templateState = HeylookConfig.ChatTemplateState(
                source: "override",
                hasOverride: true,
                template: chatTemplateOverride
            )
        } else {
            var resolved = HeylookConfig.inspectChatTemplate(in: dirURL)
            if resolved.template.isEmpty && fm.fileExists(atPath: tokenizerSubdir.path) {
                resolved = HeylookConfig.inspectChatTemplate(in: tokenizerSubdir)
            }
            templateState = resolved
        }

        let effectiveModelID = modelID ?? dirURL.lastPathComponent
        return analyze(
            templateText: templateState.template,
            specialTokens: specialTokens,
            eosTokenIDs: eosIDs,
            templateSource: templateState.source,
            modelID: effectiveModelID
        )
    }

    /// Analyzes raw chat template text and special tokens to derive `ReasoningFormat`,
    /// `ThinkingControls`, and `strippableSpecialTokens`.
    public static func analyze(
        templateText: String,
        specialTokens: Set<String> = [],
        eosTokenIDs: [Int] = [],
        templateSource: String = "tokenizer_config.json",
        modelID: String? = nil
    ) -> ModelTemplateInfo {
        let format = detectReasoningFormat(
            templateText: templateText,
            specialTokens: specialTokens
        )
        let controls = detectThinkingControls(
            templateText: templateText,
            reasoningFormat: format,
            templateSource: templateSource
        )

        var mergedSpecials = Set(StripSpecials.defaultSpecialTokens).union(specialTokens)
        if templateText.contains("<turn|>") {
            mergedSpecials.insert("<turn|>")
        }
        if templateText.contains("<|eot|>") {
            mergedSpecials.insert("<|eot|>")
        }
        if templateText.contains("<|eom|>") {
            mergedSpecials.insert("<|eom|>")
        }
        let strippable = mergedSpecials.subtracting(format.structuralDelimiters)

        return ModelTemplateInfo(
            reasoningFormat: format,
            thinkingControls: controls,
            specialTokens: mergedSpecials,
            strippableSpecialTokens: strippable,
            eosTokenIDs: eosTokenIDs,
            templateSource: templateSource
        )
    }

    // MARK: - Private Structural Detectors

    private static let depthProbeLevels: [String] = [
        "low", "medium", "high", "xhigh", "max", "minimal",
        "none", "off", "on", "auto", "enabled", "disabled", "adaptive",
    ]
    private static let garbageDepthSentinel = "zq-not-a-level"
    private static let probeConversation: [[String: JinjaValue]] = [
        ["role": .string("system"), "content": .string("SYSTEM-PROMPT")],
        ["role": .string("user"), "content": .string("FIRST-USER-TURN")],
        ["role": .string("assistant"), "content": .string("FIRST-REPLY")],
        ["role": .string("user"), "content": .string("SECOND-USER-TURN")],
    ]

    private static func detectReasoningFormat(
        templateText: String,
        specialTokens: Set<String>
    ) -> ReasoningFormat {
        // 1. OpenAI Harmony channel protocol: `<|channel|>analysis<|message|>...<|end|>`
        let hasHarmonyMarkers = (templateText.contains("<|channel|>") && templateText.contains("<|message|>"))
            || (specialTokens.contains("<|channel|>") && specialTokens.contains("<|message|>"))
        if hasHarmonyMarkers {
            return .harmonyChannel
        }

        // 2. Recipient-addressed channel protocol (Muse-Glimmer / Muse-Spark): `to=self<|message|>...<|eom|>`
        let hasRecipientMarkers = templateText.contains("<|message|>")
            && (templateText.contains("<|eom|>") || specialTokens.contains("<|eom|>"))
            && (templateText.contains("to=self") || templateText.contains("to=user"))
        if hasRecipientMarkers {
            return .recipientChannel
        }

        // 3. Gemma 4 labeled channel protocol: `<|channel>thought\n...<channel|>`
        let hasGemmaChannelMarkers = (templateText.contains("<|channel>") && templateText.contains("<channel|>"))
            || (specialTokens.contains("<|channel>") && specialTokens.contains("<channel|>"))
        if hasGemmaChannelMarkers {
            return .gemmaChannel(openPrefix: "<|channel>", closeTag: "<channel|>")
        }

        // 4. Tag-pair protocols: `<think>...</think>`, `<|thought|>...<|/thought|>`, `<thought>...</thought>`
        let candidatePairs: [(open: String, close: String)] = [
            ("<think>", "</think>"),
            ("<|thought|>", "<|/thought|>"),
            ("<thought>", "</thought>"),
        ]
        for pair in candidatePairs {
            if templateText.contains(pair.open)
                || templateText.contains(pair.close)
                || specialTokens.contains(pair.open) {
                let prefills = detectTemplatePrefillsThinking(
                    templateText: templateText,
                    openTag: pair.open,
                    closeTag: pair.close
                )
                return .tagPair(open: pair.open, close: pair.close, prefillsThinking: prefills)
            }
        }

        // 5. If the template's AST declares/reads `enable_thinking`, default to `<think>`/`</think>`
        if let compiled = try? HeylookJinjaRenderer.compile(templateText),
           compiled.undeclaredVariables().contains("enable_thinking") {
            return .tagPair(open: "<think>", close: "</think>", prefillsThinking: false)
        }

        return .none
    }

    /// Inspects the generation-prompt tail of `templateText` (after the last `add_generation_prompt`
    /// or after the message loop `{% endfor %}`) to determine if the template opens `openTag`
    /// without closing it with `closeTag`.
    private static func detectTemplatePrefillsThinking(
        templateText: String,
        openTag: String,
        closeTag: String
    ) -> Bool {
        guard !templateText.isEmpty else { return false }

        let searchRegion: Substring
        if let genPromptRange = templateText.range(of: "add_generation_prompt", options: .backwards) {
            searchRegion = templateText[genPromptRange.lowerBound...]
        } else if let endForRange = templateText.range(of: "endfor", options: .backwards) {
            searchRegion = templateText[endForRange.lowerBound...]
        } else {
            searchRegion = templateText[...]
        }

        guard let lastOpen = searchRegion.range(of: openTag, options: .backwards) else {
            return false
        }
        if let lastClose = searchRegion.range(of: closeTag, options: .backwards) {
            let beforeClose = searchRegion[..<lastClose.lowerBound]
            let openCount = beforeClose.components(separatedBy: openTag).count - 1
            let afterOpen = searchRegion[lastOpen.upperBound...]
            if !afterOpen.contains(closeTag) {
                return true
            }
            return openCount >= 2
        }
        return true
    }

    private struct DetectedDepthSpec {
        let variable: String
        let values: [String]
        let aliases: [String: String]
        let defaultValue: String?
        let unknown: String
        let changesPrefix: Bool?
        let off: [String]
    }

    private static func detectThinkingControls(
        templateText: String,
        reasoningFormat: ReasoningFormat,
        templateSource: String
    ) -> ThinkingControls {
        let trimmed = templateText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let compiled = try? HeylookJinjaRenderer.compile(templateText) else {
            if reasoningFormat != .none {
                return ThinkingControls(
                    supportsThinking: true,
                    canDisableThinking: false,
                    defaultEnabled: true,
                    budgetEnforced: true,
                    budgetReason: "heylook's engine closes the thinking block with one forced token",
                    templateSourceName: templateSource
                )
            }
            return .unsupported
        }

        let reads = compiled.undeclaredVariables()
        let switchVar: String? = reads.contains("enable_thinking") ? "enable_thinking" : nil
        let switchOn: [String: JinjaValue] = switchVar != nil ? ["enable_thinking": .bool(true)] : [:]

        // Candidate depth variables: free variables matching /reason|think|effort/i (excluding `enable_thinking`),
        // sorted by first occurrence in template source (`thinking_controls.py:detect`).
        let depthNameRegex = try? NSRegularExpression(pattern: "reason|think|effort", options: [.caseInsensitive])
        let candidateVars = reads
            .filter { v in
                guard v != "enable_thinking" else { return false }
                let range = NSRange(v.startIndex..<v.endIndex, in: v)
                return depthNameRegex?.firstMatch(in: v, range: range) != nil
            }
            .sorted { lhs, rhs in
                let li = templateText.range(of: lhs)?.lowerBound ?? templateText.endIndex
                let ri = templateText.range(of: rhs)?.lowerBound ?? templateText.endIndex
                return li != ri ? li < ri : lhs < rhs
            }

        var detectedDepth: DetectedDepthSpec?
        for variable in candidateVars {
            if let found = probeDepthVariable(
                compiled: compiled,
                body: templateText,
                variable: variable,
                switchOn: switchOn
            ) {
                detectedDepth = found
                break
            }
        }

        let supportsThinking = (reasoningFormat != .none) || (switchVar != nil) || (detectedDepth != nil)
        guard supportsThinking else {
            return .unsupported
        }

        let defaultEnabled = detectDefaultThinkingEnabled(
            compiled: compiled,
            switchVar: switchVar,
            reasoningFormat: reasoningFormat,
            templateText: templateText
        )

        let offValues = detectedDepth?.off ?? []
        let depthValues = detectedDepth?.values ?? []
        let canDisable = (switchVar != nil)
            || !offValues.isEmpty
            || depthValues.contains("none")
            || depthValues.contains("off")

        let budgetEnforced: Bool?
        let budgetReason: String?
        if switchVar != nil {
            let enforced = !reasoningFormat.structuralDelimiters.isEmpty
            budgetEnforced = enforced
            budgetReason = enforced
                ? "heylook's engine closes the thinking block with one forced token"
                : "the template's thinking markers were not found, so no cap can be forced"
        } else {
            budgetEnforced = nil
            budgetReason = nil
        }

        return ThinkingControls(
            supportsThinking: true,
            canDisableThinking: canDisable,
            defaultEnabled: defaultEnabled,
            switchVariable: switchVar,
            depthVariable: detectedDepth?.variable,
            depthValues: depthValues,
            depthAliases: detectedDepth?.aliases ?? [:],
            depthDefault: detectedDepth?.defaultValue,
            depthUnknown: detectedDepth?.unknown,
            depthChangesPrefix: detectedDepth?.changesPrefix,
            depthOffValues: offValues,
            budgetEnforced: budgetEnforced,
            budgetReason: budgetReason,
            templateSourceName: templateSource
        )
    }

    private static func renderProbe(
        _ compiled: HeylookJinjaRenderer.CompiledTemplate,
        extra: [String: JinjaValue]
    ) -> String? {
        var ctx: [String: JinjaValue] = [
            "messages": .array(probeConversation.map { .object($0) }),
            "add_generation_prompt": .bool(true),
            "bos_token": .string(""),
            "eos_token": .string(""),
        ]
        for (k, v) in extra {
            ctx[k] = v
        }
        return try? compiled.render(context: ctx)
    }

    private static func detectDefaultThinkingEnabled(
        compiled: HeylookJinjaRenderer.CompiledTemplate,
        switchVar: String?,
        reasoningFormat: ReasoningFormat,
        templateText: String
    ) -> Bool {
        if let switchVar {
            if let absent = renderProbe(compiled, extra: [:]),
               let onOut = renderProbe(compiled, extra: [switchVar: .bool(true)]),
               let offOut = renderProbe(compiled, extra: [switchVar: .bool(false)]) {
                if absent == offOut && absent != onOut {
                    return false
                }
                if absent == onOut && absent != offOut {
                    return true
                }
            }
        }
        if case .gemmaChannel = reasoningFormat {
            return templateText.contains("enable_thinking is not defined or enable_thinking")
                || !templateText.contains("enable_thinking is defined and enable_thinking")
        }
        return true
    }

    private static func probeDepthVariable(
        compiled: HeylookJinjaRenderer.CompiledTemplate,
        body: String,
        variable: String,
        switchOn: [String: JinjaValue]
    ) -> DetectedDepthSpec? {
        let absent = renderProbe(compiled, extra: switchOn)
        var garbageCtx = switchOn
        garbageCtx[variable] = .string(garbageDepthSentinel)
        let garbage = renderProbe(compiled, extra: garbageCtx)

        let unknown: String
        if let garbage {
            if garbage == absent {
                unknown = "ignored"
            } else if garbage.contains(garbageDepthSentinel) {
                unknown = "verbatim"
            } else {
                unknown = "fallback"
            }
        } else {
            unknown = "raises"
        }

        let tied = compiled.associatedDepthLiterals(for: variable)
        let candidates: [String]
        if unknown == "verbatim" {
            candidates = tied
        } else {
            let quotedLiterals = extractQuotedIdentifierLiterals(from: body)
            candidates = dedupPreservingOrder(tied + quotedLiterals + depthProbeLevels)
        }

        var groups: [String: [String]] = [:]
        var outputOrder: [String] = []
        for candidate in candidates {
            var ctx = switchOn
            ctx[variable] = .string(candidate)
            guard let rendered = renderProbe(compiled, extra: ctx) else { continue }
            if groups[rendered] == nil {
                outputOrder.append(rendered)
            }
            groups[rendered, default: []].append(candidate)
        }

        var groupByWord: [String: String] = [:]
        for (out, members) in groups {
            for m in members { groupByWord[m] = out }
        }
        let canonical = compiled.normalizedDepthWords(groupByWord: groupByWord)

        if (unknown == "ignored" || unknown == "fallback"), let garbage, let droppedMembers = groups.removeValue(forKey: garbage) {
            let tiedSet = Set(tied)
            let named = droppedMembers.filter { canonical.contains($0) || tiedSet.contains($0) }
            if !named.isEmpty && !canonical.isDisjoint(with: named) {
                groups[garbage] = named
            }
        }

        if groups.isEmpty && unknown != "verbatim" {
            return nil
        }

        let varStart = body.range(of: variable)?.lowerBound ?? body.startIndex
        func firstSeenOffset(_ value: String) -> Int {
            let single = "'\(value)'"
            let double = "\"\(value)\""
            var hitsAfter: [Int] = []
            var hitsAny: [Int] = []
            for needle in [single, double] {
                if let rAfter = body.range(of: needle, range: varStart..<body.endIndex) {
                    hitsAfter.append(body.distance(from: body.startIndex, to: rAfter.lowerBound))
                }
                if let rAny = body.range(of: needle) {
                    hitsAny.append(body.distance(from: body.startIndex, to: rAny.lowerBound))
                }
            }
            if let earliestAfter = hitsAfter.min() {
                return earliestAfter
            }
            if let earliestAny = hitsAny.min() {
                return body.count + earliestAny
            }
            return 2 * body.count + 1
        }

        var values: [String] = []
        var aliases: [String: String] = [:]
        var defaultValue: String?

        for out in outputOrder {
            guard let members = groups[out], !members.isEmpty else { continue }
            let spelling = members.enumerated().min { lhs, rhs in
                let lCanon = canonical.contains(lhs.element) ? 0 : 1
                let rCanon = canonical.contains(rhs.element) ? 0 : 1
                if lCanon != rCanon { return lCanon < rCanon }
                let lSeen = firstSeenOffset(lhs.element)
                let rSeen = firstSeenOffset(rhs.element)
                if lSeen != rSeen { return lSeen < rSeen }
                return lhs.offset < rhs.offset
            }?.element ?? members[0]

            values.append(spelling)
            for m in members where m != spelling {
                aliases[m] = spelling
            }
            if out == absent {
                defaultValue = spelling
            }
        }

        values.sort { firstSeenOffset($0) < firstSeenOffset($1) }

        let base = absent ?? outputOrder.first ?? ""
        var changesPrefix: Bool? = nil
        if let firstUserRange = base.range(of: "FIRST-USER-TURN") {
            let firstUserOffset = base.distance(from: base.startIndex, to: firstUserRange.lowerBound)
            var others = outputOrder.filter { $0 != base }
            if unknown == "verbatim", let garbage, garbage != base {
                others = [garbage]
            }
            if !others.isEmpty {
                let earliestDivergence = others.map { firstDivergenceOffset(base, $0) }.min() ?? firstUserOffset
                changesPrefix = earliestDivergence < firstUserOffset
            }
        }

        var offValues: [String] = []
        if !switchOn.isEmpty, let switchedOff = renderProbe(compiled, extra: ["enable_thinking": .bool(false)]) {
            offValues = values.filter { spelling in
                var ctx = switchOn
                ctx[variable] = .string(spelling)
                return renderProbe(compiled, extra: ctx) == switchedOff
            }
        }

        return DetectedDepthSpec(
            variable: variable,
            values: values,
            aliases: aliases,
            defaultValue: defaultValue,
            unknown: unknown,
            changesPrefix: changesPrefix,
            off: offValues
        )
    }

    private static func extractQuotedIdentifierLiterals(from body: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: "['\"]([A-Za-z][A-Za-z0-9_-]{0,23})['\"]") else {
            return []
        }
        let nsRange = NSRange(body.startIndex..<body.endIndex, in: body)
        var out: [String] = []
        var seen: Set<String> = []
        for match in regex.matches(in: body, range: nsRange) {
            guard match.numberOfRanges >= 2,
                  let range = Range(match.range(at: 1), in: body) else { continue }
            let token = String(body[range])
            if seen.insert(token).inserted {
                out.append(token)
            }
        }
        return out
    }

    private static func dedupPreservingOrder(_ items: [String]) -> [String] {
        var seen: Set<String> = []
        var out: [String] = []
        out.reserveCapacity(items.count)
        for item in items where seen.insert(item).inserted {
            out.append(item)
        }
        return out
    }

    private static func firstDivergenceOffset(_ a: String, _ b: String) -> Int {
        var idx = 0
        for (ca, cb) in zip(a, b) {
            if ca != cb { return idx }
            idx += 1
        }
        return min(a.count, b.count)
    }

    private static func loadJSON(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return dict
    }
}
