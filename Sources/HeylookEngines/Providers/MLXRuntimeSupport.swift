import CryptoKit
import Foundation
import HeylookCore

#if canImport(CoreImage)
import CoreImage
#endif

#if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers)
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

#if canImport(MLXVLM)
import MLXVLM
#endif

// MARK: - Pre-Rendered VLM Tokenizer Proxy

/// Bridges `HeylookConfig.renderPromptDetailed` (including `chat_template.heylook.jinja`, sidecar templates,
/// prior `thinking`, and open-turn continuation) into `MLXVLM` `UserInputProcessor` implementations so
/// both LLMs and VLMs honor identical template rendering while still executing native VLM image expansion.
final class PreRenderedVLMTokenizerProxy: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let base: any MLXLMCommon.Tokenizer
    private let preRenderedTokens: [Int]

    init(base: any MLXLMCommon.Tokenizer, preRenderedTokens: [Int]) {
        self.base = base
        self.preRenderedTokens = preRenderedTokens
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        base.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        base.convertIdToToken(id)
    }

    func makeStreamingDetokenizer() -> any StreamingDetokenizer {
        base.makeStreamingDetokenizer()
    }

    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var unknownToken: String? { base.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        preRenderedTokens
    }
}

// MARK: - LRU Vision Feature / Preprocessed Image Cache

struct CachedVisionImage: @unchecked Sendable {
    let key: String
    let resizedImage: CIImage
    let width: Int
    let height: Int
    var lastUsedAt: Date
    let approxBytes: Int
}

/// Bounded LRU cache of preprocessed VLM images keyed by `(SHA-256 of image bytes + processor geometry)`.
/// Ensures repeat multimodal turns skip CPU resize/color-space work and produce identical media salts
/// so `APCSnapshotCache` can reuse KV prefixes across matching image sets.
final class VisionFeatureCache: @unchecked Sendable {
    private let maxEntries: Int
    private let maxBytes: Int
    private var entries: [String: CachedVisionImage] = [:]
    private var order: [String] = []
    private(set) var hits: Int = 0
    private(set) var misses: Int = 0

    init(maxEntries: Int = 16, maxBytes: Int = 512 * 1024 * 1024) {
        self.maxEntries = max(1, maxEntries)
        self.maxBytes = max(64 * 1024 * 1024, maxBytes)
    }

    static func cacheKey(imageData: Data, geometryKey: String) -> String {
        let digest = SHA256.hash(data: imageData)
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return "\(hex):\(geometryKey)"
    }

    static func mediaSalt(forKeys keys: [String]) -> String {
        guard !keys.isEmpty else { return "" }
        return keys.joined(separator: "|")
    }

    func get(key: String) -> CachedVisionImage? {
        guard var hit = entries[key] else {
            misses += 1
            return nil
        }
        hits += 1
        hit.lastUsedAt = Date()
        entries[key] = hit
        order.removeAll { $0 == key }
        order.append(key)
        return hit
    }

    func insert(key: String, resizedImage: CIImage, width: Int, height: Int) {
        let bytes = max(1, width * height * 4)
        let entry = CachedVisionImage(
            key: key,
            resizedImage: resizedImage,
            width: width,
            height: height,
            lastUsedAt: Date(),
            approxBytes: bytes
        )
        entries[key] = entry
        order.removeAll { $0 == key }
        order.append(key)
        enforceBounds()
    }

    func clear() {
        entries.removeAll()
        order.removeAll()
        hits = 0
        misses = 0
    }

    private func enforceBounds() {
        while order.count > maxEntries || totalBytes > maxBytes {
            guard let oldest = order.first else { break }
            order.removeFirst()
            entries.removeValue(forKey: oldest)
        }
    }

    private var totalBytes: Int {
        entries.values.reduce(0) { $0 + $1.approxBytes }
    }
}

// MARK: - Multi-Branch APC Snapshot Cache

struct APCSnapshotMatch: @unchecked Sendable {
    let caches: [any KVCache]?
    let state: LMOutput.State?
    let reusedPrefixTokens: Int
    let didTrim: Bool
    let missCause: String?
}

/// Multi-branch in-memory Automatic Prefix Cache (APC) modeled on `vlm_engine.py` (`APC_CHECKPOINT_ENTRIES = 16`).
/// Stores exact post-template `[Int]` token ID sequences paired with `mediaSalt` and copied `[any KVCache]`
/// snapshots at turn boundaries, system-prompt boundaries, and aligned checkpoints so multi-turn branching
/// and hybrid/SWA models reuse cached KV state without detokenize/re-encode drift or invalid SSM trims.
final class APCSnapshotCache: @unchecked Sendable {
    struct Entry: @unchecked Sendable {
        let id: UUID
        let tokenIDs: [Int]
        let mediaSalt: String
        let caches: [any KVCache]
        let state: LMOutput.State?
        var lastUsedAt: Date
        let isHybridOrSWA: Bool
    }

    private let maxEntries: Int
    private var entries: [Entry] = []
    private var evictedDueToMemory = false

    init(maxEntries: Int = 16) {
        self.maxEntries = max(1, maxEntries)
    }

    var isEmpty: Bool { entries.isEmpty }

    func reset() {
        entries.removeAll()
        evictedDueToMemory = false
    }

    func pruneForMemoryPressureIfNeeded(minHeadroomGB: Double = 2.0) {
        let mem = RAMFitCalculator.liveMemorySnapshot()
        if mem.reclaimableGB < minHeadroomGB && !entries.isEmpty {
            let removeCount = max(1, entries.count / 2)
            let sorted = entries.sorted { $0.lastUsedAt < $1.lastUsedAt }
            let toRemove = Set(sorted.prefix(removeCount).map(\.id))
            entries.removeAll { toRemove.contains($0.id) }
            evictedDueToMemory = true
        }
    }

    func match(
        promptTokenIDs: [Int],
        mediaSalt: String,
        isHybridOrSWA: Bool
    ) -> APCSnapshotMatch {
        pruneForMemoryPressureIfNeeded()

        guard !entries.isEmpty else {
            let cause = evictedDueToMemory ? "memory" : "cold"
            evictedDueToMemory = false
            return APCSnapshotMatch(caches: nil, state: nil, reusedPrefixTokens: 0, didTrim: false, missCause: cause)
        }

        let promptLen = promptTokenIDs.count
        guard promptLen > 1 else {
            return APCSnapshotMatch(caches: nil, state: nil, reusedPrefixTokens: 0, didTrim: false, missCause: "cold")
        }

        // Filter candidates matching exact mediaSalt first so different image sets never alias each other's KV prefix.
        let matchingSaltEntries = entries.filter { $0.mediaSalt == mediaSalt }
        guard !matchingSaltEntries.isEmpty else {
            let cause: String = (!mediaSalt.isEmpty || entries.contains { !$0.mediaSalt.isEmpty })
                ? "new_image_set"
                : "history_diverged"
            return APCSnapshotMatch(caches: nil, state: nil, reusedPrefixTokens: 0, didTrim: false, missCause: cause)
        }

        var bestEntryIndex: Int?
        var bestCommon = 0
        var bestRequiresTrim = false

        for idx in matchingSaltEntries.indices {
            let entry = matchingSaltEntries[idx]
            let limit = min(entry.tokenIDs.count, promptLen - 1)
            guard limit > 0 else { continue }
            var common = 0
            while common < limit && entry.tokenIDs[common] == promptTokenIDs[common] {
                common += 1
            }
            guard common > 0 else { continue }

            let excess = entry.tokenIDs.count - common
            if excess == 0 {
                if common > bestCommon {
                    bestCommon = common
                    bestEntryIndex = idx
                    bestRequiresTrim = false
                }
            } else if !isHybridOrSWA && canTrimPromptCache(entry.caches, numTokens: excess) {
                if common > bestCommon {
                    bestCommon = common
                    bestEntryIndex = idx
                    bestRequiresTrim = true
                }
            } else if isHybridOrSWA && entry.caches.allSatisfy({ $0.maxTrimCount >= excess }) {
                if common > bestCommon {
                    bestCommon = common
                    bestEntryIndex = idx
                    bestRequiresTrim = true
                }
            }
        }

        guard let winnerIdx = bestEntryIndex, bestCommon > 0 else {
            let cause = evictedDueToMemory ? "memory" : "history_diverged"
            evictedDueToMemory = false
            return APCSnapshotMatch(caches: nil, state: nil, reusedPrefixTokens: 0, didTrim: false, missCause: cause)
        }

        let winner = matchingSaltEntries[winnerIdx]
        if let globalIdx = entries.firstIndex(where: { $0.id == winner.id }) {
            entries[globalIdx].lastUsedAt = Date()
        }

        let copied = winner.caches.map { $0.copy() }
        if bestRequiresTrim {
            let excess = winner.tokenIDs.count - bestCommon
            if excess > 0 {
                let trimmed = trimPromptCache(copied, numTokens: excess)
                if trimmed != excess {
                    return APCSnapshotMatch(caches: nil, state: nil, reusedPrefixTokens: 0, didTrim: false, missCause: "history_diverged")
                }
            }
        }
        return APCSnapshotMatch(
            caches: copied,
            state: winner.state,
            reusedPrefixTokens: bestCommon,
            didTrim: bestRequiresTrim,
            missCause: nil
        )
    }

    func recordSnapshot(
        tokenIDs: [Int],
        mediaSalt: String,
        caches: [any KVCache],
        state: LMOutput.State?,
        isHybridOrSWA: Bool
    ) {
        guard !tokenIDs.isEmpty, !caches.isEmpty else { return }
        // Deduplicate exact tokenIDs + mediaSalt branch.
        if let existingIdx = entries.firstIndex(where: { $0.tokenIDs == tokenIDs && $0.mediaSalt == mediaSalt }) {
            entries.remove(at: existingIdx)
        }
        let copied = caches.map { $0.copy() }
        entries.append(
            Entry(
                id: UUID(),
                tokenIDs: tokenIDs,
                mediaSalt: mediaSalt,
                caches: copied,
                state: state,
                lastUsedAt: Date(),
                isHybridOrSWA: isHybridOrSWA
            )
        )
        while entries.count > maxEntries {
            if let oldestIdx = entries.indices.min(by: { entries[$0].lastUsedAt < entries[$1].lastUsedAt }) {
                entries.remove(at: oldestIdx)
            } else {
                break
            }
        }
    }
}

// MARK: - GPU Closing Thinking Token Injection Processor

/// Combines optional base `LogitProcessor` (repetition/presence penalties) with GPU-level closing
/// thinking token injection when either `RequestRegistry.requestReasoningEnd` OR `thinkingBudget`
/// fires mid-generation, forcing the model's closer sequence (`</think>` / `<channel|>`) on GPU.
struct ThinkingControlLogitProcessor: LogitProcessor {
    private enum Phase {
        case observing
        case forcingCloser(index: Int)
        case completed
    }

    private var base: LogitProcessor?
    private let requestID: String
    private let openTokenIDs: [Int]
    private let closeTokenIDs: [Int]
    private let thinkingBudget: Int?
    private var insideThinking: Bool
    private(set) var thinkingTokensSeen: Int = 0
    private var openMatchedPrefix: Int = 0
    private var closeMatchedPrefix: Int = 0
    private var phase: Phase = .observing

    init(
        base: LogitProcessor?,
        requestID: String,
        reasoningFormat: ReasoningFormat,
        thinkingEnabled: Bool,
        startsInThinking: Bool,
        thinkingBudget: Int?,
        tokenizer: any MLXLMCommon.Tokenizer
    ) {
        self.base = base
        self.requestID = requestID
        self.thinkingBudget = thinkingEnabled ? thinkingBudget : nil
        self.insideThinking = thinkingEnabled && startsInThinking

        switch reasoningFormat {
        case .tagPair(let open, let close, _):
            self.openTokenIDs = tokenizer.encode(text: open, addSpecialTokens: false)
            self.closeTokenIDs = tokenizer.encode(text: close, addSpecialTokens: false)
        case .gemmaChannel(let openPrefix, let closeTag):
            self.openTokenIDs = tokenizer.encode(text: openPrefix, addSpecialTokens: false)
            self.closeTokenIDs = tokenizer.encode(text: closeTag, addSpecialTokens: false)
        case .harmonyChannel, .recipientChannel, .none:
            self.openTokenIDs = []
            self.closeTokenIDs = []
        }
    }

    func copy() -> Self {
        var dup = self
        dup.base = base?.copy()
        return dup
    }

    mutating func prompt(_ prompt: MLXArray) {
        base?.prompt(prompt)
        let promptIDs = prompt.asArray(Int.self)
        for tok in promptIDs {
            advanceBoundary(tokenID: tok, countForBudget: false)
        }
        checkTriggers()
    }

    func process(logits: MLXArray) -> MLXArray {
        var effective = base?.process(logits: logits) ?? logits
        if case .forcingCloser(let index) = phase,
           index < closeTokenIDs.count {
            let targetID = closeTokenIDs[index]
            let vocab = effective.dim(-1)
            let vocabRange = MLXArray(Int32(0) ..< Int32(vocab))
            let allowedMask = vocabRange .== Int32(targetID)
            let masked = MLX.where(allowedMask, MLXArray(Float(0)), MLXArray(-Float.infinity))
            effective = masked.reshaped([1, vocab]).asType(effective.dtype)
        }
        return effective
    }

    mutating func didSample(token: MLXArray) {
        base?.didSample(token: token)
        eval(token)
        let tokID = token.item(Int.self)
        advanceBoundary(tokenID: tokID, countForBudget: true)
        checkTriggers()
    }

    mutating func didEmit(token: Int) {
        base?.didEmit(token: token)
    }

    mutating func finalizeGeneration() {
        base?.finalizeGeneration()
    }

    private mutating func advanceBoundary(tokenID: Int, countForBudget: Bool) {
        if case .completed = phase { return }

        if case .forcingCloser(let index) = phase {
            let nextIndex = index + 1
            if nextIndex >= closeTokenIDs.count {
                insideThinking = false
                phase = .completed
            } else {
                phase = .forcingCloser(index: nextIndex)
            }
            return
        }

        if !insideThinking {
            if !openTokenIDs.isEmpty {
                if tokenID == openTokenIDs[openMatchedPrefix] {
                    openMatchedPrefix += 1
                    if openMatchedPrefix == openTokenIDs.count {
                        insideThinking = true
                        openMatchedPrefix = 0
                    }
                } else {
                    openMatchedPrefix = (tokenID == openTokenIDs[0]) ? 1 : 0
                }
            }
            return
        }

        if countForBudget {
            thinkingTokensSeen += 1
        }
        if !closeTokenIDs.isEmpty {
            if tokenID == closeTokenIDs[closeMatchedPrefix] {
                closeMatchedPrefix += 1
                if closeMatchedPrefix == closeTokenIDs.count {
                    insideThinking = false
                    closeMatchedPrefix = 0
                    phase = .completed
                }
            } else {
                closeMatchedPrefix = (tokenID == closeTokenIDs[0]) ? 1 : 0
            }
        }
    }

    private mutating func checkTriggers() {
        guard case .observing = phase, insideThinking, !closeTokenIDs.isEmpty else { return }
        let regState = RequestRegistry.shared.state(for: requestID)
        let budgetHit = thinkingBudget.map { $0 >= 0 && thinkingTokensSeen >= $0 } ?? false
        if regState.isReasoningEndRequested || budgetHit {
            // Skip any suffix already emitted as part of partial closer match
            let startIdx = min(closeMatchedPrefix, closeTokenIDs.count - 1)
            phase = .forcingCloser(index: startIdx)
        }
    }
}

// MARK: - Staged Rotating KV Cache Wrapper (Public RotatingKVCache API Only)

/// Wraps a `RotatingKVCache` for speculative rounds so `numDraft + 1` candidate positions can be
/// evaluated bidirectionally/causally without permanently evicting wrapped ring history until commit.
final class HeylookStagedRotatingKVCache: KVCache {
    let live: RotatingKVCache
    private var stagedKeys: MLXArray?
    private var stagedValues: MLXArray?

    init(live: RotatingKVCache) {
        self.live = live
    }

    var stagedCount: Int { stagedKeys?.dim(2) ?? 0 }

    var offset: Int { live.offset + stagedCount }

    var maxSize: Int? { live.maxSize }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let nextK = stagedKeys.map { concatenated([$0, keys], axis: 2) } ?? keys
        let nextV = stagedValues.map { concatenated([$0, values], axis: 2) } ?? values
        self.stagedKeys = nextK
        self.stagedValues = nextV
        return presentation(stagedKeys: nextK, stagedValues: nextV)
    }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        live.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    var state: [MLXArray] {
        get { live.state }
        set { live.state = newValue }
    }

    var metaState: [String] {
        get { live.metaState }
        set { live.metaState = newValue }
    }

    func commit(retaining: Int) {
        defer {
            stagedKeys = nil
            stagedValues = nil
        }
        guard let sk = stagedKeys, let sv = stagedValues else { return }
        let keep = Swift.min(Swift.max(retaining, 0), sk.dim(2))
        guard keep > 0 else { return }
        _ = live.update(
            keys: sk[.ellipsis, ..<keep, 0...],
            values: sv[.ellipsis, ..<keep, 0...]
        )
    }

    private func presentation(stagedKeys: MLXArray, stagedValues: MLXArray) -> (MLXArray, MLXArray) {
        let bound = Swift.min(live.offset, (live.maxSize ?? Int.max) - 1)
        guard bound > 0, let tailView = logicalTail(bound: bound) else {
            return (stagedKeys, stagedValues)
        }
        return (
            concatenated([tailView.0, stagedKeys], axis: 2),
            concatenated([tailView.1, stagedValues], axis: 2)
        )
    }

    private func logicalTail(bound: Int) -> (MLXArray, MLXArray)? {
        let state = live.state
        guard state.count == 2 else { return nil }
        let keys = state[0]
        let values = state[1]
        let meta = live.metaState
        let keep = meta.count > 0 ? (Int(meta[0]) ?? 0) : 0
        let idx = meta.count > 4 ? (Int(meta[4]) ?? keys.dim(2)) : keys.dim(2)
        let wrapped = meta.count > 6 ? (Bool(meta[6]) ?? false) : false

        let available = wrapped ? keys.dim(2) : idx
        guard available > 0 else { return nil }
        let prefix = Swift.min(keep, available)
        let requested = Swift.min(Swift.max(bound, 0), available)
        let effBound = Swift.max(requested, prefix)

        if !wrapped || idx == available {
            if prefix == 0 || effBound == available {
                return (
                    keys[.ellipsis, (available - effBound) ..< available, 0...],
                    values[.ellipsis, (available - effBound) ..< available, 0...]
                )
            }
        } else if prefix == 0 && effBound <= idx {
            return (
                keys[.ellipsis, (idx - effBound) ..< idx, 0...],
                values[.ellipsis, (idx - effBound) ..< idx, 0...]
            )
        }

        let recent = Swift.min(effBound - prefix, idx - prefix)
        let older = effBound - prefix - recent
        let ranges = [0 ..< prefix, (available - older) ..< available, (idx - recent) ..< idx]
            .filter { !$0.isEmpty }
        func slice(_ array: MLXArray) -> MLXArray {
            let parts = ranges.map { array[.ellipsis, $0, 0...] }
            return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
        }
        return (slice(keys), slice(values))
    }

    func innerState() -> [MLXArray] {
        live.innerState() + [stagedKeys, stagedValues].compactMap { $0 }
    }

    func prepare(lengths: [Int]?) { live.prepare(lengths: lengths) }
    func prepare(lengths: MLXArray?) { live.prepare(lengths: lengths) }
    func finalize() { live.finalize() }
    var isTrimmable: Bool { false }
    @discardableResult func trim(_ n: Int) -> Int { 0 }
    func copy() -> any KVCache { fatalError("HeylookStagedRotatingKVCache is round-scoped") }
}

// MARK: - Lossless Leviathan MTP Speculative Token Iterator

/// Probability-preserving Leviathan speculative token iterator for `MTPDrafterModel` (`gemma4_assistant`).
/// Adds:
/// 1. Exact `p_target(x_i) / q_draft(x_i)` acceptance + normalized residual `max(0, p_target - q_draft)`
///    resampling at `temperature > 0` (exact lossless equivalence to target sampling distribution);
/// 2. Repetition/presence/frequency penalties propagated onto draft logits via cloned `draftProcessor`;
/// 3. Explicit `draft_max_context` threshold gating speculation cleanly when sequence depth exceeds draft limit;
/// 4. Keeps MTP enabled after multimodal prefill (priming drafter state cleanly via 1-token forward if needed).
struct LeviathanMTPTokenIterator: TokenIteratorProtocol, MTPStatsCollecting {
    private let mainModel: any LanguageModel
    private let drafter: any MTPDrafterModel
    private let parameters: GenerateParameters
    private let blockSize: Int
    private let draftMaxContext: Int?
    private let targetSampler: any LogitSampler
    private var targetProcessor: (any LogitProcessor)?
    private let mainCache: [any KVCache]
    private var processedTokensCount: Int
    private var mainState: LMOutput.State?
    private var drafterState: MTPDrafterState?
    private var y: LMInput.Text

    private var pendingTokens: [Int] = []
    private var pendingIndex: Int = 0
    private var pendingConsumedFromRound: Int = 0
    private var lastReturnedFromPending: Bool = false
    private(set) var primedTokenID: Int?
    private var lastRoundAccepted: Int?
    private var passthrough: Bool = false

    private var roundCount: Int = 0
    private var totalDrafted: Int = 0
    private var totalAccepted: Int = 0
    private var totalTargetCalls: Int = 0
    private var totalDraftCalls: Int = 0
    private var totalTargetVerified: Int = 0
    private var totalEmitted: Int = 0

    let maxTokens: Int?
    var promptPrefillTime: TimeInterval = 0
    private(set) var passthroughReason: String?

    var tokenCount: Int { totalEmitted }
    var proposedDraftTokens: Int { totalDrafted }
    var acceptedDraftTokens: Int { totalAccepted }

    var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        guard roundCount > 0 else { return nil }
        return SpeculativeDecodingTelemetry(
            roundCount: roundCount,
            draftTokenCount: totalDrafted,
            acceptedDraftTokenCount: totalAccepted,
            targetModelCallCount: totalTargetCalls,
            draftModelCallCount: totalDraftCalls,
            targetVerifiedTokenCount: totalTargetVerified,
            emittedTokenCount: totalEmitted
        )
    }

    var state: LMOutput.State? {
        Self.persistentTargetState(from: mainState)
    }

    var lastLogProbabilities: DeferredTokenLogProbabilities? { nil }

    var evictedTokenCount: Int {
        mainCache.lazy.map(\.evictedTokenCount).max() ?? 0
    }

    var reasoningTokenCount: Int? {
        (targetProcessor as? ThinkingControlLogitProcessor)?.thinkingTokensSeen
    }

    init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any MTPDrafterModel,
        mainCache: [any KVCache]? = nil,
        mainState: LMOutput.State? = nil,
        parameters: GenerateParameters,
        processor: (any LogitProcessor)?,
        blockSize: Int = 4,
        draftMaxContext: Int? = nil
    ) throws {
        try drafter.validateCompatibility(with: mainModel)
        let caches = try mainCache ?? mainModel.newCache(parameters: parameters)
        let inferredProcessed = caches.compactMap { ($0 as? RotatingKVCache)?.offset ?? $0.offset }.min() ?? 0
        self.mainModel = mainModel
        self.drafter = drafter
        self.parameters = parameters
        let maxDrafterBlock = drafter.maximumBlockSize ?? blockSize
        let rotatingMin = caches.compactMap { $0.maxSize }.min()
        let windowBound = rotatingMin.map { Swift.max(2, $0 + 1) } ?? .max
        self.blockSize = Swift.max(2, Swift.min(blockSize, Swift.min(maxDrafterBlock, windowBound)))
        self.draftMaxContext = draftMaxContext
        self.targetSampler = parameters.sampler()
        self.targetProcessor = processor
        self.mainCache = caches
        self.processedTokensCount = inferredProcessed
        self.mainState = mainState
        self.drafterState = (drafter as? any StatefulMTPDrafterModel)?.makeState(parameters: parameters)
        self.y = input.text
        self.maxTokens = parameters.maxTokens

        let start = Date.timeIntervalSinceReferenceDate
        try prepare(input: input, prefill: parameters.prefill)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - start
    }

    private mutating func prepare(input: LMInput, prefill: PrefillParameters) throws {
        targetProcessor?.prompt(input.text.tokens)
        prefill.prefix?(processedTokensCount)
        let inputLength = input.text.tokens.dim(1)

        var prefillState = Self.persistentTargetState(from: mainState) ?? LMOutput.State()
        prefillState[mtpEmitFlagKey] = true
        prefillState[mtpTargetLayerIdsKey] = drafter.targetLayerIds

        switch try mainModel.prepare(input, cache: mainCache, state: prefillState, prefill: prefill) {
        case .tokens(let tokens):
            let remaining = tokens.tokens.dim(1)
            processedTokensCount += Swift.max(0, inputLength - remaining)
            let result = mainModel(tokens[text: .newAxis], cache: mainCache, state: prefillState)
            processedTokensCount += remaining
            var logits = result.logits[0..., -1, 0...]
            logits = targetProcessor?.process(logits: logits) ?? logits
            let bonus = targetSampler.sample(logits: logits)
            eval(bonus)
            targetProcessor?.didSample(token: bonus)
            y = .init(tokens: bonus)
            mainState = result.state
            if !Self.reconcileSharedKV(&mainState, discarding: 0, cache: mainCache) {
                engagePassthrough(reason: "Target did not emit shared KV source indices")
            }
            pendingTokens.append(bonus.item(Int.self))
            let total = input.text.tokens.size
            prefill.progress?(total, total)

        case .logits(let prefillResult):
            processedTokensCount += inputLength
            var logits = prefillResult.logits[0..., -1, 0...]
            logits = targetProcessor?.process(logits: logits) ?? logits
            let bonus = targetSampler.sample(logits: logits)
            eval(bonus)
            targetProcessor?.didSample(token: bonus)
            y = .init(tokens: bonus)
            mainState = prefillResult.state
            let placeable = Self.reconcileSharedKV(&mainState, discarding: 0, cache: mainCache)
            if !placeable {
                engagePassthrough(reason: "Target did not emit shared KV source indices")
            }
            let missingState = mainState?[mtpLastHiddenStatesKey] == nil
                || (drafter.requiresSharedTargetKV && mainState?[mtpSharedKVStatesKey] == nil)
            if placeable && missingState {
                // Multimodal prefill (e.g. Gemma4 VLM image prefill) omits drafter state; prime via 1-token step!
                var primeState = mainState ?? prefillState
                primeState[mtpEmitFlagKey] = true
                let primed = mainModel(y[text: .newAxis], cache: mainCache, state: primeState)
                processedTokensCount += y.tokens.dim(0)
                mainState = primed.state
                if !Self.reconcileSharedKV(&mainState, discarding: 0, cache: mainCache) {
                    engagePassthrough(reason: "Target did not emit shared KV after prime step")
                }
                var newLogits = primed.logits[0..., -1, 0...]
                newLogits = targetProcessor?.process(logits: newLogits) ?? newLogits
                let secondBonus = targetSampler.sample(logits: newLogits)
                eval(secondBonus)
                targetProcessor?.didSample(token: secondBonus)
                y = .init(tokens: secondBonus)
                let bonusID = bonus.item(Int.self)
                let secondBonusID = secondBonus.item(Int.self)
                primedTokenID = bonusID
                pendingTokens.append(bonusID)
                pendingTokens.append(secondBonusID)
            } else {
                pendingTokens.append(bonus.item(Int.self))
            }
        }
    }

    mutating func next() -> Int? {
        if let maxTokens, totalEmitted >= maxTokens { return nil }
        if pendingIndex < pendingTokens.count {
            let tok = pendingTokens[pendingIndex]
            pendingIndex += 1
            pendingConsumedFromRound += 1
            lastReturnedFromPending = true
            totalEmitted += 1
            return tok
        }
        if passthrough {
            lastReturnedFromPending = false
            guard let tok = singleTargetStep() else { return nil }
            totalEmitted += 1
            return tok
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        pendingConsumedFromRound = 0
        lastReturnedFromPending = false
        autoreleasepool { runLeviathanRound() }

        if pendingTokens.isEmpty && passthrough {
            lastReturnedFromPending = false
            guard let tok = singleTargetStep() else { return nil }
            totalEmitted += 1
            return tok
        }
        guard pendingIndex < pendingTokens.count else { return nil }
        let tok = pendingTokens[pendingIndex]
        pendingIndex += 1
        pendingConsumedFromRound += 1
        lastReturnedFromPending = true
        totalEmitted += 1
        return tok
    }

    mutating func discardGeneratedToken() {
        totalEmitted = Swift.max(0, totalEmitted - 1)
        if lastReturnedFromPending {
            pendingConsumedFromRound = Swift.max(0, pendingConsumedFromRound - 1)
            lastReturnedFromPending = false
        }
    }

    mutating func recordEmittedToken(_ token: Int) {
        targetProcessor?.didEmit(token: token)
    }

    /// Trims verified-but-unconsumed lookahead tokens if generation halts early (EOS/maxTokens/abort),
    /// only when all layers in `mainCache` are safely trimmable so hybrid SWA/SSM offsets never skew.
    mutating func finalizeUnconsumedLookahead() {
        let inMainCacheFromBuffer = Swift.max(0, pendingTokens.count - 1)
        let excessInMainCache = Swift.max(0, inMainCacheFromBuffer - pendingConsumedFromRound)
        if excessInMainCache > 0 && mainCache.allSatisfy(\.isTrimmable) {
            let trimmed = trimPromptCache(mainCache, numTokens: excessInMainCache)
            processedTokensCount = Swift.max(0, processedTokensCount - trimmed)
        }
        targetProcessor?.finalizeGeneration()
    }

    private mutating func engagePassthrough(reason: String) {
        if passthroughReason == nil {
            passthroughReason = reason
        }
        passthrough = true
    }

    private mutating func singleTargetStep() -> Int? {
        if let maxTokens, totalEmitted >= maxTokens { return nil }
        var stepState = mainState
        stepState?[mtpEmitFlagKey] = false
        let res = mainModel(y[text: .newAxis], cache: mainCache, state: stepState)
        mainState = res.state
        processedTokensCount += y.tokens.dim(0)
        var logits = res.logits[0..., -1, 0...]
        logits = targetProcessor?.process(logits: logits) ?? logits
        let tok = targetSampler.sample(logits: logits)
        eval(tok)
        targetProcessor?.didSample(token: tok)
        let id = tok.item(Int.self)
        y = .init(tokens: tok)
        return id
    }

    private mutating func runLeviathanRound() {
        guard !passthrough else { return }

        let numDraft: Int
        if let maxTokens {
            let remaining = maxTokens - totalEmitted
            guard remaining > 0 else { return }
            let budget = Swift.min(remaining - 1, blockSize - 1)
            guard budget > 0 else {
                if let tok = singleTargetStep() {
                    pendingTokens.append(tok)
                }
                return
            }
            numDraft = budget
        } else {
            numDraft = blockSize - 1
        }

        let queryOffset = mainState?[mtpSharedKVOffsetsKey]?["full_attention"] ?? processedTokensCount
        if let draftMaxContext, queryOffset >= draftMaxContext {
            engagePassthrough(reason: "Sequence reached draft_max_context (\(draftMaxContext))")
            return
        }

        guard let state = mainState,
              let lastHidden = state[mtpLastHiddenStatesKey]
        else {
            engagePassthrough(reason: "Target did not emit mtpLastHiddenStatesKey")
            return
        }
        let sharedKV = state[mtpSharedKVStatesKey] ?? [:]
        if drafter.requiresSharedTargetKV && sharedKV["full_attention"] == nil {
            engagePassthrough(reason: "Target did not emit shared full_attention KV")
            return
        }

        let bonusSlotHidden: MLXArray
        if let idx = lastRoundAccepted {
            bonusSlotHidden = lastHidden[0..., idx ..< (idx + 1), 0...]
        } else {
            bonusSlotHidden = lastHidden[0..., (-1)..., 0...]
        }

        // Capture exact draft probabilities q_i while propagating cloned repetition/presence penalties
        let draftProc = targetProcessor?.copy()
        let capturingSampler = CapturingDraftSampler(
            parameters: parameters,
            processor: draftProc
        )

        let draftTokens: MLXArray
        if let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
           var currentState = drafterState {
            draftTokens = statefulDrafter.draftBlock(
                target: mainModel,
                lastToken: y.tokens,
                lastHidden: bonusSlotHidden,
                sharedKV: sharedKV,
                positionDeltas: state[mtpPositionDeltasKey],
                queryOffset: queryOffset,
                blockSize: numDraft + 1,
                state: &currentState,
                sampler: capturingSampler
            )
            drafterState = currentState
        } else {
            draftTokens = drafter.draftBlock(
                target: mainModel,
                lastToken: y.tokens,
                lastHidden: bonusSlotHidden,
                sharedKV: sharedKV,
                positionDeltas: state[mtpPositionDeltasKey],
                queryOffset: queryOffset,
                blockSize: numDraft + 1,
                sampler: capturingSampler
            )
        }
        let flatDraft = draftTokens.flattened()
        eval(flatDraft)
        let draftIDs = flatDraft.asArray(Int.self)
        let draftProbs = capturingSampler.capturedProbs

        // Stage rotating caches if needed so ring history is never overwritten before verify outcome
        var stagedWrappers: [Int: HeylookStagedRotatingKVCache] = [:]
        var verifyCaches: [any KVCache] = []
        var appendOnlyStartOffsets: [Int: Int] = [:]
        for (slot, cache) in mainCache.enumerated() {
            if let rot = cache as? RotatingKVCache {
                let staged = HeylookStagedRotatingKVCache(live: rot)
                stagedWrappers[slot] = staged
                verifyCaches.append(staged)
            } else {
                appendOnlyStartOffsets[slot] = cache.offset
                verifyCaches.append(cache)
            }
        }

        var verifyState = state
        verifyState[mtpEmitFlagKey] = true
        verifyState[mtpTargetLayerIdsKey] = drafter.targetLayerIds
        let verifyTokens = concatenated([y.tokens, flatDraft])
        let verifyInput = LMInput.Text(tokens: verifyTokens)
        let verifyResult = mainModel(verifyInput[text: .newAxis], cache: verifyCaches, state: verifyState)
        let verifyLogits = verifyResult.logits
        mainState = verifyResult.state
        let verifyStart = Swift.max(0, verifyInput.tokens.dim(0) - (numDraft + 1))

        var accepted = 0
        var finalToken: MLXArray?

        for i in 0 ..< numDraft {
            var stepLogits = verifyLogits[0..., verifyStart + i, 0...]
            stepLogits = targetProcessor?.process(logits: stepLogits) ?? stepLogits

            if parameters.temperature <= 0 {
                let targetTok = targetSampler.sample(logits: stepLogits)
                eval(targetTok)
                let targetVal = targetTok.item(Int.self)
                targetProcessor?.didSample(token: targetTok)
                pendingTokens.append(targetVal)
                guard targetVal == draftIDs[i] else {
                    finalToken = targetTok
                    break
                }
                accepted += 1
            } else {
                let pVec = Self.filteredProbabilities(logits: stepLogits, parameters: parameters)
                let qVec = (i < draftProbs.count) ? draftProbs[i] : pVec
                let draftId = draftIDs[i]
                let pScalar = pVec[0..., draftId].item(Float.self)
                let qScalar = qVec[0..., draftId].item(Float.self)
                let ratio: Float = qScalar > 0 ? Swift.min(1.0, pScalar / qScalar) : (pScalar > 0 ? 1.0 : 0.0)
                let u = Float.random(in: 0 ..< 1)

                if u < ratio {
                    let accArr = MLXArray([Int32(draftId)])
                    targetProcessor?.didSample(token: accArr)
                    pendingTokens.append(draftId)
                    accepted += 1
                } else {
                    // Lossless Leviathan residual resampling: norm(max(0, p_target - q_draft))
                    let residual = maximum(pVec - qVec, MLXArray(Float(0)))
                    let resSum = residual.sum().item(Float.self)
                    let corrDist = resSum > 1e-8 ? (residual / resSum) : pVec
                    let corrTok = categorical(log(maximum(corrDist, MLXArray(Float(1e-20)))))
                    eval(corrTok)
                    targetProcessor?.didSample(token: corrTok)
                    pendingTokens.append(corrTok.item(Int.self))
                    finalToken = corrTok
                    break
                }
            }
        }

        if finalToken == nil {
            var bonusLogits = verifyLogits[0..., verifyStart + accepted, 0...]
            bonusLogits = targetProcessor?.process(logits: bonusLogits) ?? bonusLogits
            let bonus = targetSampler.sample(logits: bonusLogits)
            eval(bonus)
            targetProcessor?.didSample(token: bonus)
            pendingTokens.append(bonus.item(Int.self))
            finalToken = bonus
        }

        let emittedFinalToken = finalToken!
        let retained = accepted + 1
        let rejected = numDraft - accepted

        for (slot, cache) in mainCache.enumerated() {
            if let staged = stagedWrappers[slot] {
                staged.commit(retaining: retained)
            } else if let startOff = appendOnlyStartOffsets[slot] {
                let written = cache.offset - startOff
                if written > retained {
                    _ = cache.trim(written - retained)
                }
            }
        }
        processedTokensCount += retained

        roundCount += 1
        totalDrafted += numDraft
        totalAccepted += accepted
        totalTargetCalls += 1
        totalDraftCalls += 1
        totalTargetVerified += numDraft + 1
        lastRoundAccepted = accepted

        if let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
           var currentState = drafterState {
            statefulDrafter.commitDrafterState(
                target: mainModel,
                targetHidden: mainState?[mtpLastHiddenStatesKey] ?? lastHidden,
                draftTokens: draftTokens,
                acceptedCount: accepted,
                finalToken: emittedFinalToken,
                positionDeltas: mainState?[mtpPositionDeltasKey],
                state: &currentState,
                sampler: targetSampler
            )
            drafterState = currentState
        }

        if !Self.reconcileSharedKV(&mainState, discarding: rejected, cache: mainCache) {
            engagePassthrough(reason: "Target stopped emitting shared KV source indices")
            mainState = nil
        }

        y = .init(tokens: emittedFinalToken)
    }

    static func filteredProbabilities(logits: MLXArray, parameters: GenerateParameters) -> MLXArray {
        var logprobs = logSoftmax((logits.dtype == .bfloat16 ? logits.asType(.float32) : logits), axis: -1)
        let negInf = MLXArray(-Float.infinity)
        if parameters.topP > 0 && parameters.topP < 1 {
            let sortedIndices = argSort(logprobs, axis: -1)
            let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
            let cumProbs = cumsum(exp(sortedLogprobs), axis: -1)
            let filtered = MLX.where(cumProbs .> (1 - parameters.topP), sortedLogprobs, negInf)
            logprobs = putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
        }
        if parameters.minP > 0 {
            let maxLogprob = logprobs.max(axis: -1, keepDims: true)
            let threshold = maxLogprob + log(parameters.minP)
            logprobs = MLX.where(logprobs .>= threshold, logprobs, negInf)
        }
        if parameters.topK > 0 {
            let vocab = logprobs.dim(-1)
            if parameters.topK < vocab {
                let maskIndices = argPartition(-logprobs, kth: parameters.topK - 1, axis: -1)[0..., parameters.topK...]
                logprobs = putAlong(logprobs, maskIndices, values: negInf, axis: -1)
            }
        }
        let temp = Swift.max(Float(1e-4), parameters.temperature)
        return softmax(logprobs * (1.0 / temp), axis: -1)
    }

    private static func reconcileSharedKV(
        _ state: inout LMOutput.State?,
        discarding: Int,
        cache: [any KVCache]
    ) -> Bool {
        guard let sharedKV = state?[mtpSharedKVStatesKey] else { return true }
        guard let sources = state?[mtpSharedKVSourceIndicesKey],
              sharedKV.keys.allSatisfy({ sources[$0] != nil })
        else { return false }

        state?[mtpSharedKVStatesKey] = sharedKV.reduce(into: [:]) { acc, entry in
            let (key, kv) = entry
            let sourceIdx = sources[key]!
            guard sourceIdx >= 0 && sourceIdx < cache.count else { return }
            let leaf = cache[sourceIdx]
            let bound = Swift.min(leaf.offset, leaf.maxSize ?? .max)
            var len = kv.0.dim(-2)
            if discarding > 0 {
                len = Swift.max(0, len - discarding)
            }
            let start = Swift.max(0, len - bound)
            acc[key] = (
                kv.0[.ellipsis, start ..< len, 0...],
                kv.1[.ellipsis, start ..< len, 0...]
            )
        }
        if discarding > 0, let offsets = state?[mtpSharedKVOffsetsKey] {
            state?[mtpSharedKVOffsetsKey] = offsets.mapValues { Swift.max(0, $0 - discarding) }
        }
        return true
    }

    private static func persistentTargetState(from state: LMOutput.State?) -> LMOutput.State? {
        guard var state else { return nil }
        state[mtpEmitFlagKey] = nil
        state[mtpCacheCheckpointIndexKey] = nil
        state[mtpLastHiddenStatesKey] = nil
        state[mtpSharedKVStatesKey] = nil
        state[mtpSharedKVSourceIndicesKey] = nil
        state[mtpSharedKVOffsetsKey] = nil
        state[mtpPositionDeltasKey] = nil
        return state
    }
}

/// Capturing `LogitSampler` passed into `MTPDrafterModel.draftBlock` so `LeviathanMTPTokenIterator`
/// receives exact draft probability distributions `q_i` while enforcing repetition/presence penalties.
private final class CapturingDraftSampler: LogitSampler, @unchecked Sendable {
    private let parameters: GenerateParameters
    private var processor: (any LogitProcessor)?
    private(set) var capturedProbs: [MLXArray] = []

    init(parameters: GenerateParameters, processor: (any LogitProcessor)?) {
        self.parameters = parameters
        self.processor = processor
    }

    func sample(logits: MLXArray) -> MLXArray {
        let processed = processor?.process(logits: logits) ?? logits
        let tok: MLXArray
        if parameters.temperature <= 0 {
            let f32 = processed.dtype == .bfloat16 ? processed.asType(.float32) : processed
            let probs = softmax(f32, axis: -1)
            capturedProbs.append(probs)
            tok = argMax(f32, axis: -1)
        } else {
            let probs = LeviathanMTPTokenIterator.filteredProbabilities(
                logits: processed,
                parameters: parameters
            )
            capturedProbs.append(probs)
            tok = categorical(log(maximum(probs, MLXArray(Float(1e-20)))))
        }
        processor?.didSample(token: tok)
        return tok
    }
}

#endif
