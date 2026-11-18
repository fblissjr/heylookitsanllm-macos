import Foundation
import HeylookCore
import HeylookEngines

/// Thread-safe cooperative cancellation signal triggered when an HTTP peer drops or
/// `RequestRegistry` aborts an in-flight request.
public final class StreamCancellationSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled: Bool = false
    private var callbacks: [@Sendable () -> Void] = []

    public init() {}

    public var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    public func cancel() {
        let toRun: [@Sendable () -> Void]
        lock.lock()
        if cancelled {
            lock.unlock()
            return
        }
        cancelled = true
        toRun = callbacks
        callbacks.removeAll()
        lock.unlock()
        for cb in toRun {
            cb()
        }
    }

    public func onCancel(_ block: @escaping @Sendable () -> Void) {
        var runNow = false
        lock.lock()
        if cancelled {
            runNow = true
        } else {
            callbacks.append(block)
        }
        lock.unlock()
        if runNow {
            block()
        }
    }
}

/// Direct-write chunked HTTP/1.1 SSE streamer with 5.0s Anthropic `event: ping` keepalive
/// spanning both prefill and decode pauses (zero in-memory line accumulation).
public actor SSEConnectionStreamer {
    public static let defaultKeepaliveIntervalSeconds: Double = 5.0
    public static let pingFrame = "event: ping\ndata: {\"type\":\"ping\"}\n\n"

    private let requestID: String?
    private let keepaliveIntervalSeconds: Double
    private let sendBytes: @Sendable (Data) async -> Void
    private let closeConnection: @Sendable () -> Void
    public private(set) var hasStreamed: Bool = false
    private var finished: Bool = false
    private var lastFrameAt: Date = Date()
    private var pingTask: Task<Void, Never>?

    public init(
        requestID: String?,
        keepaliveIntervalSeconds: Double = SSEConnectionStreamer.defaultKeepaliveIntervalSeconds,
        sendBytes: @escaping @Sendable (Data) async -> Void,
        closeConnection: @escaping @Sendable () -> Void
    ) {
        self.requestID = requestID
        self.keepaliveIntervalSeconds = max(0.05, keepaliveIntervalSeconds)
        self.sendBytes = sendBytes
        self.closeConnection = closeConnection
    }

    public func sendFrame(_ frame: String) async {
        guard !finished else { return }
        let frameBytes = Data(frame.utf8)
        guard !frameBytes.isEmpty else { return }
        lastFrameAt = Date()

        var wirePayload = Data()
        if !hasStreamed {
            hasStreamed = true
            var headerLines = [
                "HTTP/1.1 200 OK",
                "Content-Type: text/event-stream",
                "Cache-Control: no-cache",
                "Connection: close",
                "Transfer-Encoding: chunked",
            ]
            if let requestID, !requestID.isEmpty {
                headerLines.append("X-Request-ID: \(requestID)")
            }
            headerLines.append("")
            headerLines.append("")
            wirePayload.append(Data(headerLines.joined(separator: "\r\n").utf8))
            startKeepaliveLoop()
        }

        let hexLen = String(frameBytes.count, radix: 16)
        wirePayload.append(Data("\(hexLen)\r\n".utf8))
        wirePayload.append(frameBytes)
        wirePayload.append(Data("\r\n".utf8))

        await sendBytes(wirePayload)
    }

    private func startKeepaliveLoop() {
        pingTask?.cancel()
        let interval = keepaliveIntervalSeconds
        pingTask = Task { [weak self] in
            while !Task.isCancelled {
                let sleepInterval = await self?.nextKeepaliveDelay(interval: interval) ?? interval
                try? await Task.sleep(for: .seconds(sleepInterval))
                if Task.isCancelled { break }
                await self?.sendPingIfSilent(interval: interval)
            }
        }
    }

    private func nextKeepaliveDelay(interval: Double) -> Double {
        let elapsed = Date().timeIntervalSince(lastFrameAt)
        return max(0.02, interval - elapsed)
    }

    private func sendPingIfSilent(interval: Double) async {
        guard !finished else { return }
        if Date().timeIntervalSince(lastFrameAt) >= interval - 0.01 {
            await sendFrame(Self.pingFrame)
        }
    }

    public func finish() async {
        guard !finished else { return }
        finished = true
        pingTask?.cancel()
        pingTask = nil
        if hasStreamed {
            let trailer = Data("0\r\n\r\n".utf8)
            await sendBytes(trailer)
        }
        closeConnection()
    }
}

/// Terminal state produced by `SSEWirePipeline.consumeStreamAndFormatSSE`.
public struct StreamedGenerationOutcome: Sendable {
    public let messageID: String
    public let content: String
    public let thinking: String
    public let stopReason: StopReason
    public let stopSequence: String?
    public let telemetry: ChunkTelemetryAccumulator
    public let requestDurationMs: Double
    public let generationDurationMs: Double
    public let thinkingDurationMs: Double?
    public let contentDurationMs: Double?
    public let wasAborted: Bool
    public let error: (type: String, message: String)?
}

/// Single canonical SSE translation loop shared by stateless `/v1/messages` and persistent
/// conversation generation (`POST /v1/conversations/{id}/generate` & `voice-generate`).
public enum SSEWirePipeline {
    public static func encodeSSEEvent<T: Encodable>(event: String, payload: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let json = (try? encoder.encode(payload)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return "event: \(event)\ndata: \(json)\n\n"
    }

    public static func classifyStreamError(_ error: Error) -> (type: String, message: String) {
        if let wireErr = error as? WireValidationError {
            return ("invalid_request_error", wireErr.localizedDescription)
        }
        if let execErr = error as? ProviderExecutionError, case .invalidRequest = execErr {
            return ("invalid_request_error", execErr.localizedDescription)
        }
        if let schedErr = error as? ComputeSchedulerError {
            switch schedErr {
            case .modelNotFound:
                return ("invalid_request_error", schedErr.localizedDescription)
            case .queueOverloaded, .modelBusy:
                return ("server_error", schedErr.localizedDescription)
            }
        }
        return ("server_error", error.localizedDescription)
    }

    public static func consumeStreamAndFormatSSE(
        stream: AsyncThrowingStream<GenerationChunk, Error>,
        messageID: String = "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))",
        modelID: String,
        requestID: String,
        reasoningFormat: ReasoningFormat,
        startsInThinking: Bool = false,
        thinkingEnabled: Bool = false,
        stopSequences: [String] = [],
        requestStartedAt: Date = Date(),
        queueWaitMs: Double = 0,
        registry: RequestRegistry = .shared,
        onEvent: (@Sendable (String) async -> Void)?
    ) async -> StreamedGenerationOutcome {
        let emit: @Sendable (String) async -> Void = onEvent ?? { _ in }
        await emit(encodeSSEEvent(
            event: "message_start",
            payload: MessageStartSSEEvent(id: messageID, model: modelID, inputTokens: 0)
        ))

        var telemetry = ChunkTelemetryAccumulator()
        if queueWaitMs > 0 {
            telemetry.setQueueWaitMs(queueWaitMs)
        }

        let genStartedAt = Date()
        var firstThinkingAt: Date? = nil
        var firstContentAt: Date? = nil
        var lastPrefillProcessed: Int = -1

        var thinkingAccum = ""
        var contentAccum = ""
        var nextBlockIndex = 0
        var activeBlockType: String? = nil
        var activeBlockIndex: Int? = nil

        var stopFilter = StopSequenceFilter(stopSequences: stopSequences)
        var fallbackParser: ReasoningParser? = startsInThinking
            ? ReasoningParser(
                format: reasoningFormat,
                thinkingEnabled: true,
                startsInThinkingOverride: true
            )
            : nil
        var providerEmittedThinking = false
        var terminalError: (type: String, message: String)? = nil
        var matchedStopSeq: String? = nil
        var closedReasoningViaSignal = false

        func closeActiveBlockIfNeeded() async {
            if let idx = activeBlockIndex {
                await emit(encodeSSEEvent(event: "content_block_stop", payload: ContentBlockStopSSEEvent(index: idx)))
                activeBlockIndex = nil
                activeBlockType = nil
            }
        }

        func emitThinkingDelta(_ delta: String) async {
            guard !delta.isEmpty else { return }
            if firstThinkingAt == nil { firstThinkingAt = Date() }
            if activeBlockType != "thinking" {
                await closeActiveBlockIfNeeded()
                let idx = nextBlockIndex
                nextBlockIndex += 1
                activeBlockType = "thinking"
                activeBlockIndex = idx
                await emit(encodeSSEEvent(
                    event: "content_block_start",
                    payload: ContentBlockStartSSEEvent(index: idx, contentBlock: .thinking())
                ))
            }
            if let idx = activeBlockIndex {
                thinkingAccum += delta
                await emit(encodeSSEEvent(
                    event: "content_block_delta",
                    payload: ContentBlockDeltaSSEEvent(index: idx, delta: .thinkingDelta(delta))
                ))
            }
        }

        func emitTextDelta(_ delta: String) async {
            guard !delta.isEmpty else { return }
            if firstContentAt == nil { firstContentAt = Date() }
            if activeBlockType != "text" {
                await closeActiveBlockIfNeeded()
                let idx = nextBlockIndex
                nextBlockIndex += 1
                activeBlockType = "text"
                activeBlockIndex = idx
                await emit(encodeSSEEvent(
                    event: "content_block_start",
                    payload: ContentBlockStartSSEEvent(index: idx, contentBlock: .text())
                ))
            }
            if let idx = activeBlockIndex {
                contentAccum += delta
                await emit(encodeSSEEvent(
                    event: "content_block_delta",
                    payload: ContentBlockDeltaSSEEvent(index: idx, delta: .textDelta(delta))
                ))
            }
        }

        do {
            for try await chunk in stream {
                telemetry.absorb(chunk)

                if chunk.text.isEmpty && chunk.thinking.isEmpty && chunk.promptTokens > 0 && chunk.tokenCount == 0 {
                    if chunk.promptTokens != lastPrefillProcessed {
                        lastPrefillProcessed = chunk.promptTokens
                        await emit(HeylookProgressSSEEvent(processed: chunk.promptTokens, total: chunk.promptTokens).sseFrame())
                    }
                    continue
                }

                // Check cooperative reasoning-end signal mid-stream
                let reqState = registry.state(for: requestID)
                if reqState.isReasoningEndRequested && !closedReasoningViaSignal {
                    closedReasoningViaSignal = true
                    if activeBlockType == "thinking" {
                        await closeActiveBlockIfNeeded()
                    }
                    if var parser = fallbackParser {
                        let flushed = parser.forceEndReasoning()
                        fallbackParser = parser
                        if !flushed.thinking.isEmpty {
                            await emitThinkingDelta(flushed.thinking)
                        }
                        if !flushed.text.isEmpty {
                            let filtered = stopFilter.feed(flushed.text)
                            await emitTextDelta(filtered.emitted)
                            if let matched = filtered.matchedStop {
                                matchedStopSeq = matched
                                break
                            }
                        }
                    }
                }

                if !chunk.thinking.isEmpty {
                    providerEmittedThinking = true
                    fallbackParser = nil
                    if !closedReasoningViaSignal {
                        await emitThinkingDelta(chunk.thinking)
                    }
                }

                if !chunk.text.isEmpty {
                    if !providerEmittedThinking, var parser = fallbackParser {
                        let parsed = parser.feed(chunk.text)
                        fallbackParser = parser
                        if !parsed.thinking.isEmpty && !closedReasoningViaSignal {
                            await emitThinkingDelta(parsed.thinking)
                        }
                        if !parsed.text.isEmpty {
                            let filtered = stopFilter.feed(parsed.text)
                            await emitTextDelta(filtered.emitted)
                            if let matched = filtered.matchedStop {
                                matchedStopSeq = matched
                                break
                            }
                        }
                    } else {
                        let filtered = stopFilter.feed(chunk.text)
                        await emitTextDelta(filtered.emitted)
                        if let matched = filtered.matchedStop {
                            matchedStopSeq = matched
                            break
                        }
                    }
                }

                if registry.state(for: requestID).isAborted {
                    break
                }
            }
        } catch is CancellationError {
            // Treat Swift task cancellation cleanly as abort below
        } catch {
            let classified = classifyStreamError(error)
            terminalError = classified
            await closeActiveBlockIfNeeded()
            await emit(encodeSSEEvent(
                event: "error",
                payload: StreamErrorSSEEvent(errorType: classified.type, message: classified.message)
            ))
        }

        let endedAt = Date()
        let reqDurationMs = endedAt.timeIntervalSince(requestStartedAt) * 1000.0
        let genDurationMs = endedAt.timeIntervalSince(genStartedAt) * 1000.0
        let thinkingDurMs: Double? = firstThinkingAt.map { start in
            let endRef = firstContentAt ?? endedAt
            return max(0, endRef.timeIntervalSince(start) * 1000.0)
        }
        let contentDurMs: Double? = firstContentAt.map { start in
            max(0, endedAt.timeIntervalSince(start) * 1000.0)
        }

        if terminalError == nil {
            if !providerEmittedThinking, var parser = fallbackParser {
                let flushed = parser.finalize()
                if !flushed.thinking.isEmpty && !closedReasoningViaSignal {
                    await emitThinkingDelta(flushed.thinking)
                }
                if !flushed.text.isEmpty {
                    let filtered = stopFilter.feed(flushed.text)
                    await emitTextDelta(filtered.emitted)
                    if let matched = filtered.matchedStop {
                        matchedStopSeq = matched
                    }
                }
            }
            let trailingText = stopFilter.flush()
            if !trailingText.isEmpty && matchedStopSeq == nil {
                await emitTextDelta(trailingText)
            }
            await closeActiveBlockIfNeeded()
        }

        let wasAborted = registry.state(for: requestID).isAborted || Task.isCancelled
        let effectiveStopReason: StopReason
        if matchedStopSeq != nil {
            effectiveStopReason = .stopSequence
        } else if let reported = telemetry.stopReason {
            effectiveStopReason = reported
        } else if wasAborted {
            effectiveStopReason = .endTurn
        } else {
            effectiveStopReason = .endTurn
        }

        if terminalError == nil {
            let approxOutputTokens = max(
                telemetry.completionTokens,
                max(0, (thinkingAccum.count + contentAccum.count) / 4)
            )
            let usage = telemetry.usageCounts(fallbackOutputTokens: approxOutputTokens)
            let perf = telemetry.buildPerformance(
                requestDurationMs: reqDurationMs,
                generationDurationMs: genDurationMs,
                thinkingDurationMs: thinkingDurMs,
                contentDurationMs: contentDurMs
            )
            await emit(encodeSSEEvent(
                event: "message_delta",
                payload: MessageDeltaSSEEvent(
                    delta: MessageDelta(stopReason: effectiveStopReason, stopSequence: matchedStopSeq),
                    usage: usage
                )
            ))
            await emit(encodeSSEEvent(
                event: "message_stop",
                payload: MessageStopSSEEvent(performance: perf)
            ))
        }

        return StreamedGenerationOutcome(
            messageID: messageID,
            content: contentAccum,
            thinking: thinkingAccum,
            stopReason: effectiveStopReason,
            stopSequence: matchedStopSeq,
            telemetry: telemetry,
            requestDurationMs: reqDurationMs,
            generationDurationMs: genDurationMs,
            thinkingDurationMs: thinkingDurMs,
            contentDurationMs: contentDurMs,
            wasAborted: wasAborted,
            error: terminalError
        )
    }
}
