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

/// Deterministic conversation prefix-cache state machine tracking system prompt and
/// message turn prefixes to produce exact `CacheReport` telemetry (`cold`, `reused`,
/// `system_changed`, `history_diverged`) and branch-edit KV truncation counts.
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
    public private(set) var branchTruncationsCount: Int = 0

    public init() {}

    public var isWarm: Bool {
        hasEvaluated
    }

    public var cachedTurnCount: Int {
        cachedTurns.count
    }

    /// Resets all cached prefix state when a model is unloaded or evicted.
    public mutating func reset() {
        hasEvaluated = false
        cachedSystemPrompt = nil
        cachedSystemTokens = 0
        cachedTurns.removeAll()
        branchTruncationsCount = 0
    }

    public static func estimateTokens(for text: String) -> Int {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 0 }
        let words = trimmed.split(whereSeparator: \.isWhitespace).count
        return max(2, words * 2)
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
/// and `GGUFSubprocessProvider`. Centralizes `RequestRegistry` polling (`isAborted`, `isReasoningEndRequested`),
/// `ReasoningParser` state transitions (`forceEndReasoning()`, `finalize()`), `StripSpecials` scrubbing,
/// and final `CacheReport` / `SpeculativeReport` assembly.
public enum StreamingPipeline {
    public static func transformEventStream(
        requestID: String,
        cacheReport: CacheReport,
        defaultSpeculativeReport: SpeculativeReport? = nil,
        startsInThinking: Bool = false,
        reasoningFormat: ReasoningFormat? = nil,
        thinkingEnabled: Bool = true,
        strippableSpecialTokens: [String]? = nil,
        computePlacement: ComputePlacement,
        registry: RequestRegistry,
        onReasoningEndHook: (@Sendable () async -> Void)? = nil,
        onAbortHook: (@Sendable () async -> Void)? = nil,
        onComplete: (@Sendable (String) async -> Void)? = nil,
        producer: @escaping @Sendable (AsyncThrowingStream<ProviderStreamEvent, Error>.Continuation) async throws -> Void
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let events = AsyncThrowingStream<ProviderStreamEvent, Error> { eventCont in
            Task {
                do {
                    try await producer(eventCont)
                    eventCont.finish()
                } catch {
                    eventCont.finish(throwing: error)
                }
            }
        }

        return AsyncThrowingStream { continuation in
            Task {
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
                var generatedCount = 0
                var accumulatedVisibleText = ""
                var promptTokens = cacheReport.promptTokens
                var cachedTokens = cacheReport.cachedTokens
                var sawObservedCache = false
                var promptTPS = 0.0
                var generationTPS = 0.0
                var observedSpecReport = defaultSpeculativeReport
                var firedReasoningEndHook = false
                let memGB = Double(HardwareMemoryMonitor.snapshot().processResidentMB) / 1024.0

                do {
                    for try await event in events {
                        let state = await registry.state(for: requestID)
                        if state.isAborted || Task.isCancelled {
                            if let onAbortHook {
                                await onAbortHook()
                            }
                            let flushed = reasoningParser.finalize()
                            let cleanText = stripSpecials.feed(flushed.text) + stripSpecials.finalize()
                            accumulatedVisibleText.append(cleanText)
                            continuation.yield(
                                GenerationChunk(
                                    text: cleanText,
                                    thinking: flushed.thinking,
                                    tokenCount: generatedCount,
                                    promptTokens: promptTokens,
                                    generationTPS: generationTPS,
                                    promptTPS: promptTPS,
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

                        if state.isReasoningEndRequested {
                            if !firedReasoningEndHook {
                                firedReasoningEndHook = true
                                if let onReasoningEndHook {
                                    await onReasoningEndHook()
                                }
                            }
                            if !reasoningParser.reasoningEndedForced {
                                let forced = reasoningParser.forceEndReasoning()
                                let cleanForcedText = stripSpecials.feed(forced.text)
                                accumulatedVisibleText.append(cleanForcedText)
                                if !cleanForcedText.isEmpty || !forced.thinking.isEmpty {
                                    continuation.yield(
                                        GenerationChunk(
                                            text: cleanForcedText,
                                            thinking: forced.thinking,
                                            tokenCount: generatedCount,
                                            promptTokens: promptTokens,
                                            generationTPS: generationTPS,
                                            promptTPS: promptTPS,
                                            peakMemoryGB: memGB,
                                            activeComputePlacement: computePlacement
                                        )
                                    )
                                }
                            }
                        }

                        switch event {
                        case .rawToken(let rawTok):
                            generatedCount += 1
                            let delta = reasoningParser.feed(rawTok)
                            let cleanText = stripSpecials.feed(delta.text)
                            accumulatedVisibleText.append(cleanText)
                            if !cleanText.isEmpty || !delta.thinking.isEmpty {
                                continuation.yield(
                                    GenerationChunk(
                                        text: cleanText,
                                        thinking: delta.thinking,
                                        tokenCount: generatedCount,
                                        promptTokens: promptTokens,
                                        generationTPS: generationTPS,
                                        promptTPS: promptTPS,
                                        peakMemoryGB: memGB,
                                        activeComputePlacement: computePlacement
                                    )
                                )
                            }
                        case .splitReasoning(let thinkTok):
                            generatedCount += 1
                            if reasoningParser.reasoningEndedForced {
                                let cleanText = stripSpecials.feed(thinkTok)
                                accumulatedVisibleText.append(cleanText)
                                if !cleanText.isEmpty {
                                    continuation.yield(
                                        GenerationChunk(
                                            text: cleanText,
                                            thinking: "",
                                            tokenCount: generatedCount,
                                            promptTokens: promptTokens,
                                            generationTPS: generationTPS,
                                            promptTPS: promptTPS,
                                            peakMemoryGB: memGB,
                                            activeComputePlacement: computePlacement
                                        )
                                    )
                                }
                            } else if thinkingEnabled && !thinkTok.isEmpty {
                                continuation.yield(
                                    GenerationChunk(
                                        text: "",
                                        thinking: thinkTok,
                                        tokenCount: generatedCount,
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
                            let cleanText = stripSpecials.feed(contentTok)
                            accumulatedVisibleText.append(cleanText)
                            if !cleanText.isEmpty {
                                continuation.yield(
                                    GenerationChunk(
                                        text: cleanText,
                                        thinking: "",
                                        tokenCount: generatedCount,
                                        promptTokens: promptTokens,
                                        generationTPS: generationTPS,
                                        promptTPS: promptTPS,
                                        peakMemoryGB: memGB,
                                        activeComputePlacement: computePlacement
                                    )
                                )
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
                    }

                    let tail = reasoningParser.finalize()
                    let tailText = stripSpecials.feed(tail.text) + stripSpecials.finalize()
                    accumulatedVisibleText.append(tailText)
                    if let onComplete {
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

                    continuation.yield(
                        GenerationChunk(
                            text: tailText,
                            thinking: tail.thinking,
                            tokenCount: generatedCount,
                            promptTokens: promptTokens,
                            generationTPS: generationTPS,
                            promptTPS: promptTPS,
                            peakMemoryGB: memGB,
                            stopReason: .endTurn,
                            cacheReport: finalCache,
                            speculativeReport: observedSpecReport,
                            activeComputePlacement: computePlacement
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
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
            computePlacement: computePlacement,
            registry: registry,
            onComplete: onComplete
        ) { eventCont in
            for tok in rawTokens {
                eventCont.yield(.rawToken(tok))
            }
        }
    }
}
