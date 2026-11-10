import Foundation
import HeylookCore

/// Canonical factory mapping an `EngineDescription` to its concrete `InferenceProvider` actor.
public enum ProviderFactory {
    public static func makeProvider(
        for description: EngineDescription,
        allowSyntheticFallback: Bool = false,
        remoteBaseURL: URL = URL(string: "http://127.0.0.1:8000")!
    ) -> any InferenceProvider {
        switch description.provider {
        case .mlx:
            return MLXSwiftProvider(description: description, allowSyntheticFallback: allowSyntheticFallback)
        case .coreai:
            return CoreAIProvider(description: description, allowSyntheticFallback: allowSyntheticFallback)
        case .gguf:
            return GGUFSubprocessProvider(description: description, allowSyntheticFallback: allowSyntheticFallback)
        case .remote:
            let resolvedURL = URL(string: description.modelPath) ?? remoteBaseURL
            return RemoteHeylookProvider(baseURL: resolvedURL, description: description)
        }
    }
}

/// Multi-chunk holdback filter matching `stop_sequences` across arbitrary token boundaries.
/// Never emits a matched stop sequence or partial prefix until disambiguated.
public struct StopSequenceFilter: Sendable {
    public let stopSequences: [String]
    private var pending: String = ""
    public private(set) var matchedStopSequence: String?

    public init(stopSequences: [String]) {
        self.stopSequences = stopSequences.filter { !$0.isEmpty }
    }

    public var isActive: Bool {
        !stopSequences.isEmpty
    }

    public mutating func feed(_ chunk: String) -> (emitted: String, matchedStop: String?) {
        guard !stopSequences.isEmpty else {
            return (chunk, nil)
        }
        if let matchedStopSequence {
            return ("", matchedStopSequence)
        }
        guard !chunk.isEmpty else {
            return ("", nil)
        }

        pending.append(chunk)

        // 1. Check for full stop sequence match (earliest match wins; longest wins on tie).
        var earliestRange: Range<String.Index>?
        var earliestStop: String?
        for stop in stopSequences {
            if let range = pending.range(of: stop) {
                if let currentEarliest = earliestRange {
                    if range.lowerBound < currentEarliest.lowerBound
                        || (range.lowerBound == currentEarliest.lowerBound && stop.count > (earliestStop?.count ?? 0)) {
                        earliestRange = range
                        earliestStop = stop
                    }
                } else {
                    earliestRange = range
                    earliestStop = stop
                }
            }
        }

        if let matchRange = earliestRange, let stop = earliestStop {
            let safePrefix = String(pending[..<matchRange.lowerBound])
            pending.removeAll(keepingCapacity: false)
            matchedStopSequence = stop
            return (safePrefix, stop)
        }

        // 2. Hold back the longest suffix of `pending` that is a proper prefix of any stop sequence.
        var holdbackCount = 0
        for stop in stopSequences {
            let maxCheck = min(pending.count, stop.count - 1)
            guard maxCheck > holdbackCount else { continue }
            for k in stride(from: maxCheck, through: holdbackCount + 1, by: -1) {
                if pending.suffix(k) == stop.prefix(k) {
                    holdbackCount = k
                    break
                }
            }
        }

        if holdbackCount == 0 {
            let out = pending
            pending.removeAll(keepingCapacity: true)
            return (out, nil)
        }

        let emitCount = pending.count - holdbackCount
        guard emitCount > 0 else {
            return ("", nil)
        }
        let splitIdx = pending.index(pending.startIndex, offsetBy: emitCount)
        let emitted = String(pending[..<splitIdx])
        pending = String(pending[splitIdx...])
        return (emitted, nil)
    }

    public mutating func flush() -> String {
        guard matchedStopSequence == nil, !pending.isEmpty else {
            let empty = ""
            pending.removeAll(keepingCapacity: false)
            return empty
        }
        let out = pending
        pending.removeAll(keepingCapacity: false)
        return out
    }
}

/// Deterministic conversation prefix-cache state machine tracking system prompt, message turn
/// prefixes, and exact post-template token IDs to produce truthful `CacheReport` telemetry
/// (`cold`, `reused`, `system_changed`, `history_diverged`) and branch-edit KV truncation counts.
public struct PrefixCacheTracker: Sendable {
    public struct TurnEntry: Sendable, Equatable {
        public let role: String
        public let content: String
        public let tokenCount: Int

        public init(role: String, content: String, tokenCount: Int) {
            self.role = role
            self.content = content
            self.tokenCount = tokenCount
        }
    }

    private var hasEvaluated: Bool = false
    private var cachedSystemPrompt: String?
    private var cachedSystemTokens: Int = 0
    private var cachedTurns: [TurnEntry] = []
    public private(set) var cachedTokenIDs: [Int] = []
    public private(set) var branchTruncationsCount: Int = 0

    public init() {}

    public var isWarm: Bool {
        hasEvaluated
    }

    public var cachedTurnCount: Int {
        cachedTurns.count
    }

    /// Resets all cached prefix state when a model is unloaded, evicted, or cleared.
    public mutating func reset() {
        hasEvaluated = false
        cachedSystemPrompt = nil
        cachedSystemTokens = 0
        cachedTurns.removeAll()
        cachedTokenIDs.removeAll()
        branchTruncationsCount = 0
    }

    public static func estimateTokens(for text: String) -> Int {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 0 }
        let words = trimmed.split(whereSeparator: \.isWhitespace).count
        return max(2, words * 2)
    }

    /// Evaluates exact post-template token ID prefix overlap against `cachedTokenIDs`.
    /// Always leaves at least 1 token to feed into `generate()` so logits are produced.
    public mutating func evaluateTokenIDs(_ promptTokenIDs: [Int]) -> (commonPrefixCount: Int, truncatedTrailingCount: Int) {
        guard !cachedTokenIDs.isEmpty, !promptTokenIDs.isEmpty else {
            cachedTokenIDs = promptTokenIDs
            return (0, 0)
        }
        let limit = min(cachedTokenIDs.count, max(0, promptTokenIDs.count - 1))
        var common = 0
        while common < limit && cachedTokenIDs[common] == promptTokenIDs[common] {
            common += 1
        }
        let truncated = max(0, cachedTokenIDs.count - common)
        if common > 0 && truncated > 0 {
            branchTruncationsCount += 1
        }
        cachedTokenIDs = promptTokenIDs
        return (common, truncated)
    }

    public mutating func appendGeneratedTokenIDs(_ tokenIDs: [Int]) {
        guard !tokenIDs.isEmpty else { return }
        cachedTokenIDs.append(contentsOf: tokenIDs)
    }

    /// Evaluates how much of `system` + `messages` reuses the cached KV prefix and updates
    /// the tracker state for subsequent turns. Accepts an optional `tokenCounter` closure for
    /// exact tokenizer-backed counts.
    public mutating func evaluate(
        system: String?,
        messages: [MessagesRequest.MessageTurn],
        tokenCounter: ((String) -> Int)? = nil
    ) -> CacheReport {
        let countFn = tokenCounter ?? Self.estimateTokens(for:)
        let normalizedSystem = (system?.isEmpty == true) ? nil : system
        let sysTokens = normalizedSystem.map(countFn) ?? 0
        let turnEntries = messages.map { turn in
            TurnEntry(
                role: turn.role,
                content: turn.content,
                tokenCount: max(2, countFn(turn.content))
            )
        }
        let totalPromptTokens = max(4, sysTokens + turnEntries.reduce(0) { $0 + $1.tokenCount })

        if !hasEvaluated {
            hasEvaluated = true
            cachedSystemPrompt = normalizedSystem
            cachedSystemTokens = sysTokens
            cachedTurns = turnEntries
            return CacheReport(
                promptTokens: totalPromptTokens,
                cachedTokens: 0,
                processedTokens: totalPromptTokens,
                outcome: .miss,
                cause: "cold"
            )
        }

        if normalizedSystem != cachedSystemPrompt {
            cachedSystemPrompt = normalizedSystem
            cachedSystemTokens = sysTokens
            cachedTurns = turnEntries
            cachedTokenIDs.removeAll()
            return CacheReport(
                promptTokens: totalPromptTokens,
                cachedTokens: 0,
                processedTokens: totalPromptTokens,
                outcome: .miss,
                cause: "system_changed"
            )
        }

        let priorTurns = Array(turnEntries.dropLast())
        guard !priorTurns.isEmpty else {
            if let firstNew = turnEntries.first,
               let firstCached = cachedTurns.first,
               firstNew.role == firstCached.role,
               firstNew.content == firstCached.content {
                let reused = min(totalPromptTokens - 1, sysTokens + firstCached.tokenCount)
                cachedTurns = turnEntries
                return CacheReport(
                    promptTokens: totalPromptTokens,
                    cachedTokens: max(1, reused),
                    processedTokens: max(1, totalPromptTokens - max(1, reused)),
                    outcome: .reused,
                    cause: nil
                )
            }
            cachedTurns = turnEntries
            cachedTokenIDs.removeAll()
            return CacheReport(
                promptTokens: totalPromptTokens,
                cachedTokens: 0,
                processedTokens: totalPromptTokens,
                outcome: .miss,
                cause: "history_diverged"
            )
        }

        var matchedTurnCount = 0
        var matchedTurnTokens = 0
        let checkLimit = min(priorTurns.count, cachedTurns.count)
        for idx in 0..<checkLimit {
            if priorTurns[idx].role == cachedTurns[idx].role,
               priorTurns[idx].content == cachedTurns[idx].content {
                matchedTurnCount += 1
                matchedTurnTokens += priorTurns[idx].tokenCount
            } else {
                break
            }
        }

        let previousCachedCount = cachedTurns.count
        var effectiveMatchedTurns = matchedTurnCount
        if matchedTurnCount == previousCachedCount,
           priorTurns.count == previousCachedCount + 1,
           priorTurns.last?.role == "assistant" {
            matchedTurnTokens += priorTurns.last?.tokenCount ?? 0
            effectiveMatchedTurns += 1
        }

        if matchedTurnCount > 0 && effectiveMatchedTurns < previousCachedCount {
            branchTruncationsCount += 1
        }

        cachedTurns = turnEntries

        guard matchedTurnCount > 0 else {
            cachedTokenIDs.removeAll()
            return CacheReport(
                promptTokens: totalPromptTokens,
                cachedTokens: 0,
                processedTokens: totalPromptTokens,
                outcome: .miss,
                cause: "history_diverged"
            )
        }

        let cachedCount = min(max(1, totalPromptTokens - 1), sysTokens + matchedTurnTokens)
        let processedCount = max(1, totalPromptTokens - cachedCount)
        return CacheReport(
            promptTokens: totalPromptTokens,
            cachedTokens: cachedCount,
            processedTokens: processedCount,
            outcome: .reused,
            cause: nil
        )
    }

    /// Appends the completed assistant reply to the cached turn prefix so the next turn
    /// matches both the user and assistant turns.
    public mutating func recordAssistantReply(
        _ content: String,
        tokenCounter: ((String) -> Int)? = nil
    ) {
        guard hasEvaluated, !content.isEmpty else { return }
        let countFn = tokenCounter ?? Self.estimateTokens(for:)
        cachedTurns.append(
            TurnEntry(
                role: "assistant",
                content: content,
                tokenCount: max(2, countFn(content))
            )
        )
    }
}

/// Raw stream event emitted by a provider backend (`MLXLMCommon`, `CoreAIRunner`, `llama-server` SSE,
/// or synthetic test fixture) into `StreamingPipeline.transformEventStream`.
public enum ProviderStreamEvent: Sendable {
    /// Unsplit token string to be parsed through `ReasoningParser` and scrubbed by `StripSpecials`.
    case rawToken(String)
    /// Pre-split reasoning token (e.g. from `llama-server` `delta.reasoning_content` or `MLXLMCommon.Generation.reasoning`).
    case splitReasoning(String)
    /// Pre-split visible text token (when upstream already separated `reasoning_content`), scrubbed only by `StripSpecials`.
    case splitContent(String)
    /// Runtime-observed token counts and throughput telemetry.
    case telemetry(
        promptTokens: Int? = nil,
        cachedTokens: Int? = nil,
        promptTPS: Double? = nil,
        generationTPS: Double? = nil,
        speculativeReport: SpeculativeReport? = nil
    )
}

/// Single canonical token-streaming state machine shared by `MLXSwiftProvider`, `CoreAIProvider`,
/// and `GGUFSubprocessProvider`. Centralizes immediate `RequestRegistry` cancellation,
/// `ReasoningParser` state transitions (`forceEndReasoning()`, `thinkingBudget`), `StripSpecials`,
/// multi-chunk `StopSequenceFilter`, and `CacheReport` / `SpeculativeReport` / phase timing assembly.
public enum StreamingPipeline {
    public static func transformEventStream(
        requestID: String,
        cacheReport: CacheReport,
        defaultSpeculativeReport: SpeculativeReport? = nil,
        startsInThinking: Bool = false,
        reasoningFormat: ReasoningFormat? = nil,
        thinkingEnabled: Bool = true,
        strippableSpecialTokens: [String]? = nil,
        stopSequences: [String] = [],
        maxTokens: Int? = nil,
        thinkingBudget: Int? = nil,
        computePlacement: ComputePlacement,
        registry: RequestRegistry,
        onReasoningEndHook: (@Sendable () async -> Void)? = nil,
        onAbortHook: (@Sendable () async -> Void)? = nil,
        onComplete: (@Sendable (String) async -> Void)? = nil,
        producer: @escaping @Sendable (AsyncThrowingStream<ProviderStreamEvent, Error>.Continuation) async throws -> Void
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        final class ProducerTaskBox: @unchecked Sendable {
            private let lock = NSLock()
            private var task: Task<Void, Never>?
            func set(_ t: Task<Void, Never>) {
                lock.lock()
                task = t
                lock.unlock()
            }
            func cancel() {
                lock.lock()
                let t = task
                lock.unlock()
                t?.cancel()
            }
        }
        let producerBox = ProducerTaskBox()

        let events = AsyncThrowingStream<ProviderStreamEvent, Error> { eventCont in
            let pTask = Task {
                do {
                    try await producer(eventCont)
                    eventCont.finish()
                } catch is CancellationError {
                    eventCont.finish()
                } catch let urlErr as URLError where urlErr.code == .cancelled {
                    eventCont.finish()
                } catch {
                    eventCont.finish(throwing: error)
                }
            }
            producerBox.set(pTask)
            eventCont.onTermination = { _ in
                pTask.cancel()
            }
        }

        // Register synchronous abort handler so `DELETE /v1/requests/{id}` cancels prefill / HTTP reads immediately
        registry.addCancelHandler(for: requestID) {
            producerBox.cancel()
            if let onAbortHook {
                Task { await onAbortHook() }
            }
        }

        return AsyncThrowingStream { continuation in
            let consumerTask = Task {
                var reasoningParser: ReasoningParser
                if let reasoningFormat {
                    reasoningParser = ReasoningParser(
                        format: reasoningFormat,
                        thinkingEnabled: thinkingEnabled,
                        startsInThinkingOverride: startsInThinking ? true : nil
                    )
                } else {
                    reasoningParser = ReasoningParser(startsInThinking: startsInThinking)
                }
                var stripSpecials = StripSpecials(
                    tokens: strippableSpecialTokens ?? StripSpecials.defaultSpecialTokens
                )
                var stopFilter = StopSequenceFilter(stopSequences: stopSequences)
                var generatedCount = 0
                var thinkingTokenCount = 0
                var contentTokenCount = 0
                var accumulatedVisibleText = ""
                var promptTokens = cacheReport.promptTokens
                var cachedTokens = cacheReport.cachedTokens
                var sawObservedCache = false
                var promptTPS = 0.0
                var generationTPS = 0.0
                var observedSpecReport = defaultSpeculativeReport
                var firedReasoningEndHook = false
                var finalStopReason: StopReason = .endTurn
                let memGB = Double(HardwareMemoryMonitor.snapshot().processResidentMB) / 1024.0

                let startWall = Date.timeIntervalSinceReferenceDate
                var firstTokenWall: TimeInterval?
                var thinkingStartWall: TimeInterval?
                var thinkingEndWall: TimeInterval?
                var contentStartWall: TimeInterval?

                func recordTokenTiming(hasThinking: Bool, hasContent: Bool) {
                    let now = Date.timeIntervalSinceReferenceDate
                    if firstTokenWall == nil && (hasThinking || hasContent) {
                        firstTokenWall = now
                    }
                    if hasThinking {
                        if thinkingStartWall == nil { thinkingStartWall = now }
                        thinkingEndWall = now
                    }
                    if hasContent {
                        if contentStartWall == nil { contentStartWall = now }
                        if thinkingStartWall != nil && thinkingEndWall == nil {
                            thinkingEndWall = now
                        }
                    }
                }

                func buildTimingMetrics() -> (ttft: Double?, thinkDur: Double?, contentDur: Double?, genDur: Double, computedGenTPS: Double) {
                    let endWall = Date.timeIntervalSinceReferenceDate
                    let genDurMs = max(0.0, (endWall - startWall) * 1000.0)
                    let ttftMs = firstTokenWall.map { max(0.0, ($0 - startWall) * 1000.0) }
                    let thinkMs: Double? = if let ts = thinkingStartWall {
                        max(0.0, ((thinkingEndWall ?? contentStartWall ?? endWall) - ts) * 1000.0)
                    } else {
                        nil
                    }
                    let contentMs: Double? = if let cs = contentStartWall {
                        max(0.0, (endWall - cs) * 1000.0)
                    } else {
                        nil
                    }
                    let decodeSecs = max(1e-4, endWall - (firstTokenWall ?? startWall))
                    let calcTPS = generatedCount > 0 ? (Double(generatedCount) / decodeSecs) : 0.0
                    return (ttftMs, thinkMs, contentMs, genDurMs, calcTPS)
                }

                do {
                    streamLoop: for try await event in events {
                        let state = registry.state(for: requestID)
                        if state.isAborted || Task.isCancelled {
                            producerBox.cancel()
                            if let onAbortHook {
                                await onAbortHook()
                            }
                            let flushed = reasoningParser.finalize()
                            let rawTail = stripSpecials.feed(flushed.text) + stripSpecials.finalize()
                            let (filteredTail, _) = stopFilter.feed(rawTail)
                            let flushedStop = stopFilter.flush()
                            let cleanText = filteredTail + flushedStop
                            accumulatedVisibleText.append(cleanText)
                            let timings = buildTimingMetrics()
                            continuation.yield(
                                GenerationChunk(
                                    text: cleanText,
                                    thinking: flushed.thinking,
                                    tokenCount: generatedCount,
                                    thinkingTokens: thinkingTokenCount,
                                    contentTokens: contentTokenCount,
                                    promptTokens: promptTokens,
                                    generationTPS: generationTPS > 0 ? generationTPS : timings.computedGenTPS,
                                    promptTPS: promptTPS,
                                    ttftMs: timings.ttft,
                                    thinkingDurationMs: timings.thinkDur,
                                    contentDurationMs: timings.contentDur,
                                    generationDurationMs: timings.genDur,
                                    peakMemoryGB: memGB,
                                    stopReason: .cancelled,
                                    cacheReport: cacheReport,
                                    speculativeReport: observedSpecReport,
                                    activeComputePlacement: computePlacement
                                )
                            )
                            continuation.finish()
                            return
                        }

                        let budgetExceeded = if let budget = thinkingBudget, budget > 0 {
                            thinkingTokenCount >= budget && !reasoningParser.reasoningEndedForced
                        } else {
                            false
                        }

                        if state.isReasoningEndRequested || budgetExceeded {
                            if !firedReasoningEndHook {
                                firedReasoningEndHook = true
                                if let onReasoningEndHook {
                                    await onReasoningEndHook()
                                }
                            }
                            if !reasoningParser.reasoningEndedForced {
                                let forced = reasoningParser.forceEndReasoning()
                                let scrubbedForced = stripSpecials.feed(forced.text)
                                let (cleanForcedText, matched) = stopFilter.feed(scrubbedForced)
                                accumulatedVisibleText.append(cleanForcedText)
                                recordTokenTiming(hasThinking: !forced.thinking.isEmpty, hasContent: !cleanForcedText.isEmpty)
                                if !cleanForcedText.isEmpty || !forced.thinking.isEmpty {
                                    continuation.yield(
                                        GenerationChunk(
                                            text: cleanForcedText,
                                            thinking: forced.thinking,
                                            tokenCount: generatedCount,
                                            thinkingTokens: thinkingTokenCount,
                                            contentTokens: contentTokenCount,
                                            promptTokens: promptTokens,
                                            generationTPS: generationTPS,
                                            promptTPS: promptTPS,
                                            peakMemoryGB: memGB,
                                            activeComputePlacement: computePlacement
                                        )
                                    )
                                }
                                if matched != nil {
                                    finalStopReason = .stopSequence
                                    producerBox.cancel()
                                    break streamLoop
                                }
                            }
                        }

                        switch event {
                        case .rawToken(let rawTok):
                            generatedCount += 1
                            let delta = reasoningParser.feed(rawTok)
                            let scrubbedText = stripSpecials.feed(delta.text)
                            let (cleanText, matched) = stopFilter.feed(scrubbedText)
                            if !delta.thinking.isEmpty {
                                thinkingTokenCount += 1
                            }
                            if !cleanText.isEmpty || (!scrubbedText.isEmpty && matched == nil) {
                                contentTokenCount += 1
                            }
                            accumulatedVisibleText.append(cleanText)
                            recordTokenTiming(hasThinking: !delta.thinking.isEmpty, hasContent: !cleanText.isEmpty)
                            if !cleanText.isEmpty || !delta.thinking.isEmpty {
                                continuation.yield(
                                    GenerationChunk(
                                        text: cleanText,
                                        thinking: delta.thinking,
                                        tokenCount: generatedCount,
                                        thinkingTokens: thinkingTokenCount,
                                        contentTokens: contentTokenCount,
                                        promptTokens: promptTokens,
                                        generationTPS: generationTPS,
                                        promptTPS: promptTPS,
                                        peakMemoryGB: memGB,
                                        activeComputePlacement: computePlacement
                                    )
                                )
                            }
                            if matched != nil {
                                finalStopReason = .stopSequence
                                producerBox.cancel()
                                break streamLoop
                            }

                        case .splitReasoning(let thinkTok):
                            if reasoningParser.reasoningEndedForced {
                                // Suppress buffered upstream reasoning chunks after forced reasoning end
                                // so internal chain-of-thought never leaks into visible assistant output.
                                continue
                            }
                            generatedCount += 1
                            if thinkingEnabled && !thinkTok.isEmpty {
                                thinkingTokenCount += 1
                                recordTokenTiming(hasThinking: true, hasContent: false)
                                continuation.yield(
                                    GenerationChunk(
                                        text: "",
                                        thinking: thinkTok,
                                        tokenCount: generatedCount,
                                        thinkingTokens: thinkingTokenCount,
                                        contentTokens: contentTokenCount,
                                        promptTokens: promptTokens,
                                        generationTPS: generationTPS,
                                        promptTPS: promptTPS,
                                        peakMemoryGB: memGB,
                                        activeComputePlacement: computePlacement
                                    )
                                )
                            }

                        case .splitContent(let contentTok):
                            generatedCount += 1
                            let scrubbedText = stripSpecials.feed(contentTok)
                            let (cleanText, matched) = stopFilter.feed(scrubbedText)
                            if !cleanText.isEmpty || (!scrubbedText.isEmpty && matched == nil) {
                                contentTokenCount += 1
                            }
                            accumulatedVisibleText.append(cleanText)
                            recordTokenTiming(hasThinking: false, hasContent: !cleanText.isEmpty)
                            if !cleanText.isEmpty {
                                continuation.yield(
                                    GenerationChunk(
                                        text: cleanText,
                                        thinking: "",
                                        tokenCount: generatedCount,
                                        thinkingTokens: thinkingTokenCount,
                                        contentTokens: contentTokenCount,
                                        promptTokens: promptTokens,
                                        generationTPS: generationTPS,
                                        promptTPS: promptTPS,
                                        peakMemoryGB: memGB,
                                        activeComputePlacement: computePlacement
                                    )
                                )
                            }
                            if matched != nil {
                                finalStopReason = .stopSequence
                                producerBox.cancel()
                                break streamLoop
                            }

                        case let .telemetry(pTok, cTok, pTPS, gTPS, spec):
                            if let pTok { promptTokens = pTok }
                            if let cTok {
                                cachedTokens = cTok
                                sawObservedCache = true
                            }
                            if let pTPS { promptTPS = pTPS }
                            if let gTPS { generationTPS = gTPS }
                            if let spec { observedSpecReport = spec }
                        }

                        if let limit = maxTokens, limit > 0, generatedCount >= limit {
                            finalStopReason = .maxTokens
                            producerBox.cancel()
                            break streamLoop
                        }
                    }

                    // Check once more if abort raced with producer stream completion
                    if registry.state(for: requestID).isAborted || Task.isCancelled {
                        finalStopReason = .cancelled
                    }

                    let tail = reasoningParser.finalize()
                    let rawTailText = stripSpecials.feed(tail.text) + stripSpecials.finalize()
                    let tailText: String
                    if finalStopReason == .stopSequence {
                        tailText = ""
                    } else {
                        let (filteredTail, matched) = stopFilter.feed(rawTailText)
                        if matched != nil {
                            finalStopReason = .stopSequence
                            tailText = filteredTail
                        } else {
                            tailText = filteredTail + stopFilter.flush()
                        }
                    }
                    accumulatedVisibleText.append(tailText)
                    if finalStopReason != .cancelled, let onComplete {
                        await onComplete(accumulatedVisibleText)
                    }
                    let finalCache: CacheReport = sawObservedCache
                        ? CacheReport(
                            promptTokens: promptTokens,
                            cachedTokens: cachedTokens,
                            processedTokens: max(0, promptTokens - cachedTokens),
                            outcome: cachedTokens > 0 ? .reused : .miss,
                            cause: cachedTokens > 0 ? nil : cacheReport.cause
                        )
                        : cacheReport
                    let timings = buildTimingMetrics()

                    continuation.yield(
                        GenerationChunk(
                            text: tailText,
                            thinking: finalStopReason == .stopSequence ? "" : tail.thinking,
                            tokenCount: generatedCount,
                            thinkingTokens: thinkingTokenCount,
                            contentTokens: contentTokenCount,
                            promptTokens: promptTokens,
                            generationTPS: generationTPS > 0 ? generationTPS : timings.computedGenTPS,
                            promptTPS: promptTPS,
                            ttftMs: timings.ttft,
                            thinkingDurationMs: timings.thinkDur,
                            contentDurationMs: timings.contentDur,
                            generationDurationMs: timings.genDur,
                            peakMemoryGB: memGB,
                            stopReason: finalStopReason,
                            cacheReport: finalCache,
                            speculativeReport: observedSpecReport,
                            activeComputePlacement: computePlacement
                        )
                    )
                    continuation.finish()
                } catch {
                    producerBox.cancel()
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                producerBox.cancel()
                consumerTask.cancel()
            }
        }
    }

    public static func runTokenStream(
        requestID: String,
        rawTokens: [String],
        cacheReport: CacheReport,
        speculativeReport: SpeculativeReport? = nil,
        startsInThinking: Bool,
        reasoningFormat: ReasoningFormat? = nil,
        thinkingEnabled: Bool = true,
        strippableSpecialTokens: [String]? = nil,
        stopSequences: [String] = [],
        maxTokens: Int? = nil,
        thinkingBudget: Int? = nil,
        computePlacement: ComputePlacement,
        registry: RequestRegistry,
        onComplete: (@Sendable (String) async -> Void)? = nil
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        transformEventStream(
            requestID: requestID,
            cacheReport: cacheReport,
            defaultSpeculativeReport: speculativeReport,
            startsInThinking: startsInThinking,
            reasoningFormat: reasoningFormat,
            thinkingEnabled: thinkingEnabled,
            strippableSpecialTokens: strippableSpecialTokens,
            stopSequences: stopSequences,
            maxTokens: maxTokens,
            thinkingBudget: thinkingBudget,
            computePlacement: computePlacement,
            registry: registry,
            onComplete: onComplete
        ) { eventCont in
            for tok in rawTokens {
                try Task.checkCancellation()
                eventCont.yield(.rawToken(tok))
            }
        }
    }
}
