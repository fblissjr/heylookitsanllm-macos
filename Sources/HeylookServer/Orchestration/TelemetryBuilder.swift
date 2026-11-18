import Foundation
import HeylookCore
import HeylookEngines

/// Accumulates chunk telemetry faithfully across sparse/partial chunks and builds unified
/// `PerformanceExtension`, `MessagesResponse.Usage`, and `MessageStatsRecord` payloads.
public struct ChunkTelemetryAccumulator: Sendable {
    public private(set) var promptTokens: Int = 0
    public private(set) var completionTokens: Int = 0
    public private(set) var thinkingTokens: Int = 0
    public private(set) var contentTokens: Int = 0
    public private(set) var peakMemoryGB: Double = 0.0
    public private(set) var queueWaitMs: Double = 0.0
    public private(set) var promptTPS: Double = 0.0
    public private(set) var generationTPS: Double = 0.0
    public private(set) var ttftMs: Double?
    public private(set) var thinkingDurationMs: Double?
    public private(set) var contentDurationMs: Double?
    public private(set) var generationDurationMs: Double?
    public private(set) var stopReason: StopReason?
    public private(set) var cacheReport: CacheReport?
    public private(set) var speculativeReport: SpeculativeReport?
    public private(set) var computePlacement: ComputePlacement = .mlxGPU

    public init() {}

    /// Latches monotonic counts and non-zero rates across chunks (never overwriting a prior reading
    /// with a sparse default).
    public mutating func absorb(_ chunk: GenerationChunk, recordedQueueWaitMs: Double? = nil) {
        if chunk.promptTokens > 0 { promptTokens = chunk.promptTokens }
        if chunk.tokenCount > 0 { completionTokens = chunk.tokenCount }
        if chunk.thinkingTokens > 0 { thinkingTokens = chunk.thinkingTokens }
        if chunk.contentTokens > 0 { contentTokens = chunk.contentTokens }
        peakMemoryGB = max(peakMemoryGB, chunk.peakMemoryGB)
        if let q = recordedQueueWaitMs, q > 0 { queueWaitMs = q }
        if chunk.promptTPS > 0 { promptTPS = chunk.promptTPS }
        if chunk.generationTPS > 0 { generationTPS = chunk.generationTPS }
        if let ttft = chunk.ttftMs { ttftMs = ttftMs ?? ttft }
        if let td = chunk.thinkingDurationMs { thinkingDurationMs = td }
        if let cd = chunk.contentDurationMs { contentDurationMs = cd }
        if let gd = chunk.generationDurationMs { generationDurationMs = gd }
        if let sr = chunk.stopReason { stopReason = sr }
        if let cr = chunk.cacheReport { cacheReport = cr }
        if let sp = chunk.speculativeReport { speculativeReport = sp }
        computePlacement = chunk.activeComputePlacement
    }

    public mutating func setQueueWaitMs(_ ms: Double) {
        if ms > 0 { queueWaitMs = ms }
    }

    /// Anthropic `(input_tokens, cache_read_input_tokens)` derivation:
    /// when cache report exists, `input_tokens = processedTokens` and `cache_read_input_tokens = cachedTokens`;
    /// when absent, `input_tokens = promptTokens` and `cache_read_input_tokens = nil`.
    public func usageCounts(fallbackOutputTokens: Int) -> MessagesResponse.Usage {
        let outCount = completionTokens > 0 ? completionTokens : fallbackOutputTokens
        let inputCount: Int
        let cachedCount: Int?
        if let cacheReport {
            inputCount = cacheReport.processedTokens
            cachedCount = cacheReport.cachedTokens
        } else {
            inputCount = promptTokens
            cachedCount = nil
        }
        return MessagesResponse.Usage(
            inputTokens: inputCount,
            outputTokens: outCount,
            cacheReadInputTokens: cachedCount,
            thinkingTokens: thinkingTokens > 0 ? thinkingTokens : nil,
            contentTokens: contentTokens > 0 ? contentTokens : nil
        )
    }

    /// Canonical wire `PerformanceExtension` builder. Note `PerformanceExtension.init` already
    /// subtracts positive `queueWaitMs` from `generationDurationMs`, so pass the raw span.
    public func buildPerformance(
        requestDurationMs: Double? = nil,
        generationDurationMs overrideGenDurationMs: Double? = nil,
        thinkingDurationMs overrideThinkingMs: Double? = nil,
        contentDurationMs overrideContentMs: Double? = nil
    ) -> MessagesResponse.PerformanceExtension {
        let rawGenMs = overrideGenDurationMs ?? generationDurationMs
        let effThinkingMs = overrideThinkingMs ?? thinkingDurationMs
        let effContentMs = overrideContentMs ?? contentDurationMs
        let measuredTPS: Double? = generationTPS > 0 ? generationTPS : nil

        return MessagesResponse.PerformanceExtension(
            generationTPS: measuredTPS,
            promptTPS: promptTPS > 0 ? promptTPS : nil,
            tokensPerSecond: measuredTPS,
            peakMemoryGB: peakMemoryGB > 0 ? peakMemoryGB : nil,
            queueWaitMs: queueWaitMs > 0 ? queueWaitMs : nil,
            ttftMs: ttftMs,
            thinkingDurationMs: effThinkingMs,
            contentDurationMs: effContentMs,
            generationDurationMs: rawGenMs,
            requestDurationMs: requestDurationMs,
            computePlacement: computePlacement,
            cache: cacheReport,
            speculative: speculativeReport
        )
    }

    /// Strips prose `CacheReport.reason` before persisting into `message_stats` (`cause` carries the token).
    public func sanitizedCacheReportForStats() -> CacheReport? {
        guard let cacheReport else { return nil }
        return CacheReport(
            promptTokens: cacheReport.promptTokens,
            cachedTokens: cacheReport.cachedTokens,
            processedTokens: cacheReport.processedTokens,
            outcome: cacheReport.outcome,
            cause: cacheReport.cause,
            reason: nil
        )
    }

    public func buildMessageStats(
        messageID: String,
        conversationID: String,
        modelID: String,
        requestDurationMs: Double? = nil,
        generationDurationMs: Double? = nil,
        thinkingDurationMs: Double? = nil,
        contentDurationMs: Double? = nil,
        modelLoadMs: Double? = nil,
        fallbackOutputTokens: Int = 0,
        success: Bool = true
    ) -> MessageStatsRecord {
        let perf = buildPerformance(
            requestDurationMs: requestDurationMs,
            generationDurationMs: generationDurationMs,
            thinkingDurationMs: thinkingDurationMs,
            contentDurationMs: contentDurationMs
        )
        let outTokens = completionTokens > 0 ? completionTokens : fallbackOutputTokens
        return MessageStatsRecord(
            messageID: messageID,
            conversationID: conversationID,
            modelID: modelID,
            promptTokens: promptTokens,
            outputTokens: outTokens,
            thinkingTokens: thinkingTokens,
            contentTokens: contentTokens,
            cachedTokens: cacheReport?.cachedTokens ?? 0,
            promptTPS: perf.promptTPS ?? 0,
            generationTPS: perf.generationTPS ?? 0,
            ttftMs: perf.ttftMs,
            thinkingDurationMs: perf.thinkingDurationMs,
            contentDurationMs: perf.contentDurationMs,
            generationDurationMs: perf.generationDurationMs,
            requestDurationMs: perf.requestDurationMs,
            peakMemoryGB: perf.peakMemoryGB,
            queueWaitMs: perf.queueWaitMs,
            cache: sanitizedCacheReportForStats(),
            speculative: speculativeReport,
            specEmittedTokens: speculativeReport?.emittedTokens,
            modelLoadMs: modelLoadMs.flatMap { $0 >= 100.0 ? $0 : nil },
            success: success
        )
    }
}
