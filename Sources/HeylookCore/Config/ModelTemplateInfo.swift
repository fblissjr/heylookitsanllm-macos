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

    public init(
        supportsThinking: Bool,
        canDisableThinking: Bool = true,
        defaultEnabled: Bool = true,
        switchVariable: String? = nil,
        depthVariable: String? = nil,
        depthValues: [String] = []
    ) {
        self.supportsThinking = supportsThinking
        self.canDisableThinking = canDisableThinking
        self.defaultEnabled = defaultEnabled
        self.switchVariable = switchVariable
        self.depthVariable = depthVariable
        self.depthValues = depthValues
    }

    public static let unsupported = ThinkingControls(
        supportsThinking: false,
        canDisableThinking: false,
        defaultEnabled: false
    )
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
            specialTokens: specialTokens,
            modelID: modelID
        )
        let controls = detectThinkingControls(
            templateText: templateText,
            reasoningFormat: format,
            modelID: modelID
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

    private static func detectReasoningFormat(
        templateText: String,
        specialTokens: Set<String>,
        modelID: String?
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

        // 5. Fallback for DeepSeek-R1 / R1-Distill checkpoints whose chat_template may omit literal `<think>`
        //    (matching `MLXLMCommon.DeepSeekR1ConventionsResolver`)
        if let idLower = modelID?.lowercased(),
           idLower.contains("deepseek-r1") || idLower.contains("r1-distill") || idLower.contains("qwq") {
            return .tagPair(open: "<think>", close: "</think>", prefillsThinking: true)
        }

        // 6. If the template explicitly toggles `enable_thinking`, default to `<think>`/`</think>`
        if templateText.contains("enable_thinking") {
            return .tagPair(open: "<think>", close: "</think>", prefillsThinking: false)
        }

        // 7. When no template file exists on disk (e.g. synthetic or unpopulated directories),
        //    default to standard `<think>...</think>` tag-pair parsing so `<think>` blocks are still parsed.
        if templateText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && specialTokens.isEmpty {
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

        // Look at the generation-prompt region at the end of the template
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
            // If `</think>` appears AFTER the last `<think>` in the generation prompt (e.g. when
            // `enable_thinking == false` branch renders `<think>\n\n</think>\n\n`), verify whether
            // there is also an unclosed `<think>` branch before or after it.
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

    private static func detectThinkingControls(
        templateText: String,
        reasoningFormat: ReasoningFormat,
        modelID: String?
    ) -> ThinkingControls {
        let hasEnableThinking = templateText.contains("enable_thinking")

        // Detect depth variable (`reasoning_effort`, `reasoning_strength`, or `thinking_mode`)
        let candidateDepthVars = ["reasoning_effort", "reasoning_strength", "thinking_mode"]
        var detectedDepthVar: String?
        var detectedDepthValues: [String] = []

        for varName in candidateDepthVars where templateText.contains(varName) {
            detectedDepthVar = varName
            detectedDepthValues = extractDepthValues(for: varName, in: templateText)
            break
        }

        let supportsThinking = (reasoningFormat != .none) || hasEnableThinking || (detectedDepthVar != nil)
        guard supportsThinking else {
            return .unsupported
        }

        let switchVar: String? = hasEnableThinking ? "enable_thinking" : nil
        let canDisable = hasEnableThinking
            || detectedDepthValues.contains("none")
            || detectedDepthValues.contains("off")

        // Gemma 4's template defaults `enable_thinking` to false unless requested;
        // standard `<think>` templates default to true.
        let defaultEnabled: Bool
        if case .gemmaChannel = reasoningFormat {
            defaultEnabled = templateText.contains("enable_thinking is not defined or enable_thinking")
                || !templateText.contains("enable_thinking is defined and enable_thinking")
        } else {
            defaultEnabled = true
        }

        return ThinkingControls(
            supportsThinking: true,
            canDisableThinking: canDisable,
            defaultEnabled: defaultEnabled,
            switchVariable: switchVar,
            depthVariable: detectedDepthVar,
            depthValues: detectedDepthValues
        )
    }

    /// Scans `templateText` for string literals compared or listed alongside `variableName`
    /// (e.g., `reasoning_effort in ["low", "medium", "high"]` or `reasoning_strength == 'max'`).
    private static func extractDepthValues(for variableName: String, in templateText: String) -> [String] {
        let canonicalOrder = ["none", "minimal", "low", "medium", "high", "max"]
        var found = Set<String>()

        let lowerTemplate = templateText.lowercased()
        for candidate in canonicalOrder {
            let singleQuoted = "'\(candidate)'"
            let doubleQuoted = "\"\(candidate)\""
            if lowerTemplate.contains(singleQuoted) || lowerTemplate.contains(doubleQuoted) {
                found.insert(candidate)
            }
        }

        let ordered = canonicalOrder.filter { found.contains($0) }
        if !ordered.isEmpty {
            return ordered
        }

        // Sensible defaults when the template accepts the variable without enumerating literals inline
        switch variableName {
        case "reasoning_strength":
            return ["none", "low", "medium", "high", "max"]
        case "reasoning_effort":
            return ["low", "medium", "high"]
        default:
            return ["low", "medium", "high"]
        }
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
