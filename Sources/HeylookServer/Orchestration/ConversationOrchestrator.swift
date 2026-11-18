import Foundation
import HeylookCore
import HeylookEngines

/// Actor-isolated server-owned saga orchestrating stateful conversation generation
/// (`POST /v1/conversations/{id}/generate` & `POST /v1/conversations/{id}/voice-generate`)
/// and live prompt previews (`POST /v1/conversations/{id}/prompt`).
///
/// Guarantees:
/// - `404` when the target conversation does not exist (all modes including `append`)
/// - `409 generation_in_progress` lock via `RequestRegistry.tryAcquireConversationLock`
/// - Detachable pub/sub subscriber stream so HTTP client disconnects NEVER abort generation
/// - Atomic SQLite tail commit (`replaceTailWithMessage` / `replaceTailWithUpdate`) ONLY after
///   non-empty assistant output or explicit abort with output (never deleting before streaming!)
/// - Full `user` AND `assistant` `continue`, open-thinking resume (`startsInThinking = true`),
///   special-token stripping on replayed history, and accurate `dropped_media` + `heylook_saved`
public actor ConversationOrchestrator {
    private enum CommitTarget: Sendable {
        case appendNew
        case replaceFromPosition(Int)
        case replaceAfterPosition(Int)
        case updateAnchorAndTrimTail(anchor: MessageRecord)
    }

    private final class SubscriberHub: @unchecked Sendable {
        private let lock = NSLock()
        private var subscribers: [UUID: AsyncStream<String>.Continuation] = [:]
        private var isFinished: Bool = false

        func subscribe() -> (id: UUID, stream: AsyncStream<String>) {
            let subID = UUID()
            var contOut: AsyncStream<String>.Continuation!
            let stream = AsyncStream<String> { cont in
                contOut = cont
            }
            lock.lock()
            if isFinished {
                lock.unlock()
                contOut.finish()
            } else {
                subscribers[subID] = contOut
                lock.unlock()
            }
            return (subID, stream)
        }

        func unsubscribe(_ id: UUID) {
            lock.lock()
            let cont = subscribers.removeValue(forKey: id)
            lock.unlock()
            cont?.finish()
        }

        func publish(_ frame: String) {
            lock.lock()
            let targets = Array(subscribers.values)
            lock.unlock()
            for cont in targets {
                cont.yield(frame)
            }
        }

        func finishAll() {
            lock.lock()
            isFinished = true
            let targets = Array(subscribers.values)
            subscribers.removeAll()
            lock.unlock()
            for cont in targets {
                cont.finish()
            }
        }
    }

    private final class AdmissionBox: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<Result<Void, Error>, Never>?
        private var settled: Result<Void, Error>?

        func waitForAdmission() async throws {
            let result: Result<Void, Error> = await withCheckedContinuation { cont in
                lock.lock()
                if let settled {
                    lock.unlock()
                    cont.resume(returning: settled)
                } else {
                    self.continuation = cont
                    lock.unlock()
                }
            }
            if case .failure(let err) = result {
                throw err
            }
        }

        func resolve(_ result: Result<Void, Error>) {
            lock.lock()
            guard settled == nil else {
                lock.unlock()
                return
            }
            settled = result
            let cont = continuation
            continuation = nil
            lock.unlock()
            cont?.resume(returning: result)
        }
    }

    private let store: SQLiteStore
    private let scheduler: ComputeScheduler
    private let registry: RequestRegistry
    private var activeTasks: [String: Task<Void, Never>] = [:]

    public init(
        store: SQLiteStore,
        scheduler: ComputeScheduler,
        registry: RequestRegistry = .shared
    ) {
        self.store = store
        self.scheduler = scheduler
        self.registry = registry
    }

    public func previewPrompt(
        conversationID: String,
        request: PromptPreviewRequestPayload
    ) async throws -> PromptPreviewWireResponse {
        guard let detail = try store.getConversationDetail(id: conversationID) else {
            throw WorkbenchError.notFound("Conversation '\(conversationID)' not found")
        }
        let conv = detail.conversation
        let requestedModel = request.overrides?.model?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let modelID = (requestedModel?.isEmpty == false ? requestedModel : conv.defaultModelID),
              !modelID.isEmpty else {
            throw WorkbenchError.badRequest("model is required (provide overrides.model or set conversation.model_id)")
        }
        guard let desc = await scheduler.modelDescription(for: modelID) else {
            throw WorkbenchError.badRequest("Model '\(modelID)' not found or not enabled")
        }
        guard desc.isLoaded else {
            throw WorkbenchError.conflict(
                code: "model_not_loaded",
                message: "Model '\(modelID)' is not loaded -- load it to preview prompt; preview never loads a model"
            )
        }

        let editedMessages = detail.messages.map { msg -> MessageRecord in
            if let editedText = request.edits[msg.id] {
                var copy = msg
                copy.content = editedText
                copy.contentBlocks = nil
                return copy
            }
            return msg
        }

        var promptMessages: [MessageRecord] = []
        var continuationFlag: Bool? = nil
        var continuationKind: String? = nil

        switch request.mode {
        case .append:
            promptMessages = editedMessages
            let rawText = request.userContent ?? ""
            if !rawText.isEmpty || !request.userParts.isEmpty {
                let parts: [MessagesRequest.MessageTurn.ContentPart] = request.userParts.isEmpty
                    ? (rawText.isEmpty ? [] : [.text(rawText)])
                    : request.userParts
                let syntheticUser = MessageRecord(
                    id: "preview-user",
                    conversationID: conversationID,
                    position: (editedMessages.last?.position ?? 0) + 1,
                    role: "user",
                    content: rawText.isEmpty ? partsTextSummary(parts) : rawText,
                    contentBlocks: parts.isEmpty ? nil : parts
                )
                promptMessages.append(syntheticUser)
            }
            guard !promptMessages.isEmpty else {
                throw WorkbenchError.badRequest("Cannot preview prompt for an empty conversation without user_content")
            }
            continuationFlag = false

        case .regenerate:
            guard let msgID = request.messageID?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !msgID.isEmpty else {
                throw WorkbenchError.badRequest("message_id is required for mode='regenerate'")
            }
            guard let anchorIdx = editedMessages.firstIndex(where: { $0.id == msgID }) else {
                throw WorkbenchError.notFound("Message '\(msgID)' not found in conversation '\(conversationID)'")
            }
            promptMessages = Array(editedMessages[..<anchorIdx])
            guard !promptMessages.isEmpty else {
                throw WorkbenchError.badRequest("Cannot regenerate the first message without prior context")
            }
            continuationFlag = false

        case .continue:
            guard let msgID = request.messageID?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !msgID.isEmpty else {
                throw WorkbenchError.badRequest("message_id is required for mode='continue'")
            }
            guard let anchorIdx = editedMessages.firstIndex(where: { $0.id == msgID }) else {
                throw WorkbenchError.notFound("Message '\(msgID)' not found in conversation '\(conversationID)'")
            }
            let anchor = editedMessages[anchorIdx]
            guard anchor.role == "user" || anchor.role == "assistant" else {
                throw WorkbenchError.badRequest("Cannot continue from role '\(anchor.role)'; expected 'user' or 'assistant'")
            }
            promptMessages = Array(editedMessages[...anchorIdx])
            if anchor.role == "assistant" {
                let openThinking = !(anchor.thinking ?? "").isEmpty
                    && anchor.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                continuationKind = openThinking ? "thinking" : "content"
                continuationFlag = true
            } else {
                continuationKind = "content"
                continuationFlag = false
            }
        }

        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        let tplInfo = ModelTemplateInfo.inspect(modelDirectory: dirURL, modelID: desc.modelID)

        if let depth = request.overrides?.thinkingDepth, !depth.isEmpty {
            try validateThinkingDepth(depth, description: desc, templateInfo: tplInfo)
        }

        let hydrated = try store.hydrateMessagesForProvider(
            conversationID: conversationID,
            messages: promptMessages
        )
        let stripped = Self.convertToProviderTurns(
            messages: hydrated,
            supportsVision: desc.supportsVision.value,
            supportsAudio: desc.supportsAudio.value,
            specialTokens: Array(tplInfo.specialTokens)
        )

        let effectiveSystem = request.overrides?.system ?? (conv.systemPrompt.isEmpty ? nil : conv.systemPrompt)
        let resolvedSampling = HeylookConfig.resolveEffectiveSampling(
            modelPath: desc.modelPath,
            preset: conv.params,
            request: SamplerSettings(
                thinking: request.overrides?.thinking?.enabledBool,
                thinkingDepth: request.overrides?.thinkingDepth,
                thinkingBudget: request.overrides?.thinkingBudget
            ),
            thinkingCapable: desc.supportsThinking.value
        ).effective

        let rendered = try HeylookConfig.renderPromptDetailed(
            in: FileManager.default.fileExists(atPath: dirURL.path) ? dirURL : nil,
            system: effectiveSystem,
            turns: stripped.turns,
            thinkingEnabled: resolvedSampling.thinking ?? desc.supportsThinking.value,
            thinkingDepth: resolvedSampling.thinkingDepth,
            continuation: continuationFlag,
            modelSupportsVision: desc.supportsVision.value,
            modelSupportsAudio: desc.supportsAudio.value
        )

        // Attempt optional live token-count inspection on resident provider without generating
        var liveTokenCount: Int? = nil
        if let provider = try? await scheduler.provider(for: modelID) {
            let previewReq = MessagesRequest(
                model: modelID,
                messages: stripped.turns,
                system: effectiveSystem,
                maxTokens: 1,
                thinking: resolvedSampling.thinking,
                thinkingDepth: resolvedSampling.thinkingDepth
            )
            if let live = try? await provider.renderPromptPreview(request: previewReq) {
                liveTokenCount = live.tokenCount
            }
        }

        let tokenStrings = rendered.markers.map { $0.token }
        let uniqueLongestFirstMarkers = Array(Set(tokenStrings))
            .sorted { a, b in
                if a.count != b.count { return a.count > b.count }
                return a < b
            }

        return PromptPreviewWireResponse(
            modelID: modelID,
            prompt: rendered.prompt,
            source: rendered.source,
            markers: uniqueLongestFirstMarkers,
            continuation: continuationKind,
            droppedMedia: DroppedMediaCounts(
                images: stripped.dropped.images,
                audio: stripped.dropped.audio
            ),
            unrenderedMedia: rendered.unrenderedMedia,
            tokenCount: liveTokenCount
        )
    }

    public func generate(
        conversationID: String,
        request: ConversationGenerateRequest,
        requestID: String,
        cancellationSignal: StreamCancellationSignal = StreamCancellationSignal(),
        detachOnSubscriberExit: Bool = true,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> HeylookSavedEvent {
        let requestStartedAt = Date()
        guard let detail = try store.getConversationDetail(id: conversationID) else {
            throw WorkbenchError.notFound("Conversation '\(conversationID)' not found")
        }
        let conv = detail.conversation
        let existingMessages = detail.messages

        let reqModelTrimmed = request.model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let modelID = (!reqModelTrimmed.isEmpty ? reqModelTrimmed : conv.defaultModelID),
              !modelID.isEmpty else {
            throw WorkbenchError.badRequest("model is required (set request.model or conversation.model_id)")
        }
        guard let desc = await scheduler.modelDescription(for: modelID) else {
            throw WorkbenchError.badRequest("Model '\(modelID)' not found or not enabled")
        }

        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        let tplInfo = ModelTemplateInfo.inspect(modelDirectory: dirURL, modelID: desc.modelID)

        if let depth = request.thinkingDepth, !depth.isEmpty {
            try validateThinkingDepth(depth, description: desc, templateInfo: tplInfo)
        }

        // Resolve 5-layer sampler cascade (builtin -> vendor -> sidecar -> preset/conv -> request)
        var basePresetParams = conv.params ?? SamplerSettings()
        if let presetID = conv.appliedPresetID,
           let preset = try? store.getPreset(id: presetID) {
            basePresetParams = SamplerCascade.resolve(builtin: basePresetParams, request: preset.settings).effective
        } else if let presetName = request.presetName?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !presetName.isEmpty,
                  let preset = try? store.listPresets().first(where: { $0.name.caseInsensitiveCompare(presetName) == .orderedSame }) {
            basePresetParams = SamplerCascade.resolve(builtin: basePresetParams, request: preset.settings).effective
        }

        var requestSampler = request.sampling ?? SamplerSettings()
        if let th = request.thinking { requestSampler.thinking = th }
        if let td = request.thinkingDepth { requestSampler.thinkingDepth = td }
        if let tb = request.thinkingBudget { requestSampler.thinkingBudget = tb }
        if let stops = request.stopSequences { requestSampler.stopSequences = stops }
        let resolvedSampling = HeylookConfig.resolveEffectiveSampling(
            modelPath: desc.modelPath,
            preset: basePresetParams,
            request: requestSampler,
            thinkingCapable: desc.supportsThinking.value
        ).effective

        let effectiveSystem = request.system ?? (conv.systemPrompt.isEmpty ? nil : conv.systemPrompt)

        // Prepare mode turns & commit target BEFORE mutating SQLite or acquiring stream
        var pendingNewUserMessage: MessageRecord? = nil
        let promptRows: [MessageRecord]
        let commitTarget: CommitTarget
        var startsInThinking = false
        var replyPosition = (existingMessages.last?.position ?? 0) + 1

        switch request.mode {
        case .append:
            let hasNewText = !request.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            let hasNewParts = !request.userParts.isEmpty
            if hasNewParts {
                try Self.validateUserParts(request.userParts)
            }
            if hasNewText || hasNewParts {
                let parts: [MessagesRequest.MessageTurn.ContentPart] = hasNewParts
                    ? request.userParts
                    : [.text(request.content)]
                let summaryText = hasNewText ? request.content : partsTextSummary(parts)
                let newPos = (existingMessages.last?.position ?? 0) + 1
                let userMsg = MessageRecord(
                    id: UUID().uuidString,
                    conversationID: conversationID,
                    position: newPos,
                    role: "user",
                    content: summaryText,
                    contentBlocks: parts
                )
                pendingNewUserMessage = userMsg
                promptRows = existingMessages + [userMsg]
                replyPosition = newPos + 1
            } else {
                guard !existingMessages.isEmpty else {
                    throw WorkbenchError.badRequest("Cannot generate in an empty conversation without user content")
                }
                promptRows = existingMessages
                replyPosition = (existingMessages.last?.position ?? 0) + 1
            }
            commitTarget = .appendNew

        case .regenerate:
            guard let msgID = request.messageID?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !msgID.isEmpty else {
                throw WorkbenchError.badRequest("message_id is required for mode='regenerate'")
            }
            guard let anchorIdx = existingMessages.firstIndex(where: { $0.id == msgID }) else {
                throw WorkbenchError.notFound("Message '\(msgID)' not found in conversation '\(conversationID)'")
            }
            let anchor = existingMessages[anchorIdx]
            promptRows = Array(existingMessages[..<anchorIdx])
            guard !promptRows.isEmpty else {
                throw WorkbenchError.badRequest("Cannot regenerate the first message without prior context")
            }
            replyPosition = anchor.position
            commitTarget = .replaceFromPosition(anchor.position)

        case .continue:
            guard let msgID = request.messageID?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !msgID.isEmpty else {
                throw WorkbenchError.badRequest("message_id is required for mode='continue'")
            }
            guard let anchorIdx = existingMessages.firstIndex(where: { $0.id == msgID }) else {
                throw WorkbenchError.notFound("Message '\(msgID)' not found in conversation '\(conversationID)'")
            }
            let anchor = existingMessages[anchorIdx]
            guard anchor.role == "user" || anchor.role == "assistant" else {
                throw WorkbenchError.badRequest("Cannot continue from role '\(anchor.role)'; expected 'user' or 'assistant'")
            }
            promptRows = Array(existingMessages[...anchorIdx])
            if anchor.role == "assistant" {
                startsInThinking = !(anchor.thinking ?? "").isEmpty
                    && anchor.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                replyPosition = anchor.position
                commitTarget = .updateAnchorAndTrimTail(anchor: anchor)
            } else {
                replyPosition = anchor.position + 1
                commitTarget = .replaceAfterPosition(anchor.position)
            }
        }

        // Hydrate prior SQLite rows (new in-memory append turn already carries inline base64 if present)
        let persistedRows = pendingNewUserMessage != nil ? Array(promptRows.dropLast()) : promptRows
        let hydratedPersisted = try store.hydrateMessagesForProvider(
            conversationID: conversationID,
            messages: persistedRows
        )
        let allHydratedPromptRows = pendingNewUserMessage.map { hydratedPersisted + [$0] } ?? hydratedPersisted
        let converted = Self.convertToProviderTurns(
            messages: allHydratedPromptRows,
            supportsVision: desc.supportsVision.value,
            supportsAudio: desc.supportsAudio.value,
            specialTokens: Array(tplInfo.specialTokens)
        )

        let providerRequest = MessagesRequest(
            model: modelID,
            messages: converted.turns,
            system: effectiveSystem,
            maxTokens: resolvedSampling.maxTokens ?? SamplerCascade.defaultMaxTokens,
            temperature: resolvedSampling.temperature,
            topP: resolvedSampling.topP,
            topK: resolvedSampling.topK,
            minP: resolvedSampling.minP,
            repetitionPenalty: resolvedSampling.repetitionPenalty,
            presencePenalty: resolvedSampling.presencePenalty,
            seed: resolvedSampling.seed,
            stream: true,
            thinking: resolvedSampling.thinking,
            thinkingConfig: request.thinkingConfig,
            thinkingDepth: resolvedSampling.thinkingDepth,
            thinkingBudget: resolvedSampling.thinkingBudget,
            stopSequences: resolvedSampling.stopSequences,
            responseFormat: request.responseFormat
        )

        // Acquire per-conversation exclusive generation lock (409 generation_in_progress on contention)
        guard let execID = registry.tryAcquireConversationLock(
            conversationID: conversationID,
            requestID: requestID,
            modelID: modelID
        ) else {
            throw WorkbenchError.conflict(
                code: "generation_in_progress",
                message: "Generation already in progress for conversation '\(conversationID)'"
            )
        }

        let hub = SubscriberHub()
        let subscription = hub.subscribe()
        let admission = AdmissionBox()
        let savedResultBox = SavedEventResultBox()
        let storeRef = self.store
        let schedulerRef = self.scheduler
        let registryRef = self.registry
        let mode = request.mode
        let droppedMedia = converted.dropped
        let pendingUserToSave = pendingNewUserMessage
        let startsInThinkingImmutable = startsInThinking
        let replyPositionImmutable = replyPosition

        let runnerTask = Task<Void, Never> { [weak self] in
            defer {
                registryRef.releaseConversationLock(
                    conversationID: conversationID,
                    requestID: requestID,
                    executionID: execID
                )
                hub.finishAll()
            }

            do {
                let savedEvent: HeylookSavedEvent = try await schedulerRef.withComputeLease(for: modelID) {
                    let provider = try await schedulerRef.provider(for: modelID)
                    admission.resolve(.success(()))

                    if let pendingUser = pendingUserToSave {
                        _ = try storeRef.appendMessage(pendingUser)
                    }

                    let stream = await PrefillProgressBridge.$current.withValue({ processed, total in
                        hub.publish(HeylookProgressSSEEvent(processed: processed, total: total).sseFrame())
                    }) {
                        await provider.generate(
                            requestID: requestID,
                            request: providerRequest,
                            effectiveSampling: resolvedSampling,
                            registry: registryRef
                        )
                    }

                    let outcome = await SSEWirePipeline.consumeStreamAndFormatSSE(
                        stream: stream,
                        modelID: modelID,
                        requestID: requestID,
                        reasoningFormat: tplInfo.reasoningFormat,
                        startsInThinking: startsInThinkingImmutable,
                        thinkingEnabled: resolvedSampling.thinking ?? desc.supportsThinking.value,
                        stopSequences: resolvedSampling.stopSequences ?? [],
                        requestStartedAt: requestStartedAt,
                        registry: registryRef,
                        onEvent: { frame in
                            hub.publish(frame)
                        }
                    )

                    let saved = try Self.commitGenerationOutcome(
                        store: storeRef,
                        conversationID: conversationID,
                        mode: mode,
                        commitTarget: commitTarget,
                        replyPosition: replyPositionImmutable,
                        modelID: modelID,
                        outcome: outcome,
                        droppedMedia: droppedMedia
                    )
                    let savedFrame = SSEWirePipeline.encodeSSEEvent(event: "heylook_saved", payload: saved)
                    hub.publish(savedFrame)
                    return saved
                }
                savedResultBox.set(.success(savedEvent))
            } catch {
                admission.resolve(.failure(error))
                savedResultBox.set(.failure(error))
            }
            await self?.removeActiveTask(conversationID: conversationID)
        }

        activeTasks[conversationID] = runnerTask

        do {
            try await admission.waitForAdmission()
        } catch {
            // Pre-stream admission failed (e.g. 503 model_overloaded); surface error cleanly
            activeTasks.removeValue(forKey: conversationID)
            throw error
        }

        if let onSSEFrame {
            if detachOnSubscriberExit {
                let subID = subscription.id
                cancellationSignal.onCancel {
                    hub.unsubscribe(subID)
                }
            }
            for await frame in subscription.stream {
                if detachOnSubscriberExit && cancellationSignal.isCancelled {
                    hub.unsubscribe(subscription.id)
                    break
                }
                await onSSEFrame(frame)
            }
            if let existing = try? store.getConversationDetail(id: conversationID),
               savedResultBox.result == nil {
                let wireMsgs = existing.messages.map(WireMessageDTO.init(from:))
                return HeylookSavedEvent(
                    conversationID: conversationID,
                    mode: request.mode,
                    endReason: "detached",
                    messages: wireMsgs,
                    droppedMedia: droppedMedia,
                    timing: MessagesResponse.PerformanceExtension()
                )
            }
        } else {
            await runnerTask.value
        }

        guard let finalResult = savedResultBox.result else {
            let wireMsgs = ((try? store.getConversationDetail(id: conversationID))?.messages ?? []).map(WireMessageDTO.init(from:))
            return HeylookSavedEvent(
                conversationID: conversationID,
                mode: request.mode,
                endReason: "detached",
                messages: wireMsgs,
                droppedMedia: droppedMedia,
                timing: MessagesResponse.PerformanceExtension()
            )
        }
        return try finalResult.get()
    }

    private func removeActiveTask(conversationID: String) {
        activeTasks.removeValue(forKey: conversationID)
    }

    private static func commitGenerationOutcome(
        store: SQLiteStore,
        conversationID: String,
        mode: ConversationGenerateMode,
        commitTarget: CommitTarget,
        replyPosition: Int,
        modelID: String,
        outcome: StreamedGenerationOutcome,
        droppedMedia: DroppedMediaCounts
    ) throws -> HeylookSavedEvent {
        let perf = outcome.telemetry.buildPerformance(
            requestDurationMs: outcome.requestDurationMs,
            generationDurationMs: outcome.generationDurationMs,
            thinkingDurationMs: outcome.thinkingDurationMs,
            contentDurationMs: outcome.contentDurationMs
        )
        let hasOutput = !outcome.content.isEmpty || !outcome.thinking.isEmpty
        let endReason: String
        if outcome.error != nil {
            endReason = "error"
        } else if outcome.wasAborted {
            endReason = "aborted"
        } else if outcome.stopReason == .maxTokens {
            endReason = "max_tokens"
        } else if outcome.stopReason == .stopSequence {
            endReason = "stop_sequence"
        } else {
            endReason = "complete"
        }

        // Truthful atomic persistence: NEVER delete existing tail or insert empty rows on failure/empty abort
        let shouldPersist = (outcome.error == nil && hasOutput) || (outcome.wasAborted && hasOutput)
        if shouldPersist {
            let messageID = UUID().uuidString
            let approxOutTokens = max(
                outcome.telemetry.completionTokens,
                max(1, (outcome.content.count + outcome.thinking.count) / 4)
            )
            let statsMessageID: String
            switch commitTarget {
            case .updateAnchorAndTrimTail(let anchor):
                statsMessageID = anchor.id
            default:
                statsMessageID = messageID
            }

            let stats = outcome.telemetry.buildMessageStats(
                messageID: statsMessageID,
                conversationID: conversationID,
                modelID: modelID,
                requestDurationMs: outcome.requestDurationMs,
                generationDurationMs: outcome.generationDurationMs,
                thinkingDurationMs: outcome.thinkingDurationMs,
                contentDurationMs: outcome.contentDurationMs,
                fallbackOutputTokens: approxOutTokens,
                success: outcome.error == nil
            )

            switch commitTarget {
            case .appendNew:
                let assistantMsg = MessageRecord(
                    id: messageID,
                    conversationID: conversationID,
                    position: replyPosition,
                    role: "assistant",
                    content: outcome.content,
                    thinking: outcome.thinking.isEmpty ? nil : outcome.thinking,
                    modelID: modelID,
                    stopReason: outcome.stopReason.rawValue,
                    tokenCount: approxOutTokens,
                    generationTPS: perf.generationTPS,
                    ttftMs: perf.ttftMs,
                    cacheReport: outcome.telemetry.sanitizedCacheReportForStats(),
                    speculativeReport: outcome.telemetry.speculativeReport,
                    computePlacement: outcome.telemetry.computePlacement
                )
                _ = try store.appendMessage(assistantMsg)
                try store.upsertMessageStats(stats)

            case .replaceFromPosition(let fromPos):
                let assistantMsg = MessageRecord(
                    id: messageID,
                    conversationID: conversationID,
                    position: fromPos,
                    role: "assistant",
                    content: outcome.content,
                    thinking: outcome.thinking.isEmpty ? nil : outcome.thinking,
                    modelID: modelID,
                    stopReason: outcome.stopReason.rawValue,
                    tokenCount: approxOutTokens,
                    generationTPS: perf.generationTPS,
                    ttftMs: perf.ttftMs,
                    cacheReport: outcome.telemetry.sanitizedCacheReportForStats(),
                    speculativeReport: outcome.telemetry.speculativeReport,
                    computePlacement: outcome.telemetry.computePlacement
                )
                _ = try store.replaceTailWithMessage(
                    conversationID: conversationID,
                    fromPosition: fromPos,
                    newMessage: assistantMsg,
                    stats: stats
                )

            case .replaceAfterPosition(let afterPos):
                let assistantMsg = MessageRecord(
                    id: messageID,
                    conversationID: conversationID,
                    position: afterPos + 1,
                    role: "assistant",
                    content: outcome.content,
                    thinking: outcome.thinking.isEmpty ? nil : outcome.thinking,
                    modelID: modelID,
                    stopReason: outcome.stopReason.rawValue,
                    tokenCount: approxOutTokens,
                    generationTPS: perf.generationTPS,
                    ttftMs: perf.ttftMs,
                    cacheReport: outcome.telemetry.sanitizedCacheReportForStats(),
                    speculativeReport: outcome.telemetry.speculativeReport,
                    computePlacement: outcome.telemetry.computePlacement
                )
                _ = try store.replaceTailWithMessage(
                    conversationID: conversationID,
                    afterPosition: afterPos,
                    newMessage: assistantMsg,
                    stats: stats
                )

            case .updateAnchorAndTrimTail(let anchor):
                let combinedContent = anchor.content + outcome.content
                let updatedThinking: String? = outcome.thinking.isEmpty
                    ? nil
                    : (anchor.thinking ?? "") + outcome.thinking
                _ = try store.replaceTailWithUpdate(
                    conversationID: conversationID,
                    anchorMessageID: anchor.id,
                    updatedContent: combinedContent,
                    updatedThinking: updatedThinking,
                    updatedContentBlocks: nil,
                    stopReason: outcome.stopReason.rawValue,
                    tokenCount: (anchor.tokenCount ?? 0) + approxOutTokens,
                    generationTPS: perf.generationTPS,
                    ttftMs: perf.ttftMs,
                    cacheReport: outcome.telemetry.sanitizedCacheReportForStats(),
                    speculativeReport: outcome.telemetry.speculativeReport,
                    computePlacement: outcome.telemetry.computePlacement,
                    stats: stats
                )
            }
        }

        let updatedMessages = (try? store.getConversationDetail(id: conversationID)?.messages) ?? []
        let wireMessages = updatedMessages.map(WireMessageDTO.init(from:))
        return HeylookSavedEvent(
            conversationID: conversationID,
            mode: mode,
            endReason: endReason,
            messages: wireMessages,
            droppedMedia: droppedMedia,
            timing: perf,
            error: outcome.error?.message
        )
    }

    private func validateThinkingDepth(
        _ depth: String,
        description: EngineDescription,
        templateInfo: ModelTemplateInfo
    ) throws {
        if let controls = description.thinkingControls {
            if let problem = controls.checkDepth(depth) {
                throw WorkbenchError.badRequest(problem)
            }
        } else if let problem = templateInfo.thinkingControls.checkDepth(depth) {
            throw WorkbenchError.badRequest(problem)
        }
    }

    private static func validateUserParts(_ parts: [MessagesRequest.MessageTurn.ContentPart]) throws {
        for part in parts {
            if case .media(let media) = part {
                guard media.kind == "image" || media.kind == "audio" else {
                    throw WorkbenchError.unprocessable(
                        message: "Unsupported media block kind '\(media.kind)' in conversation message"
                    )
                }
                guard !media.dataBase64.isEmpty else {
                    throw WorkbenchError.badRequest("Media block of kind '\(media.kind)' requires non-empty base64 data")
                }
            }
        }
    }

    private func partsTextSummary(_ parts: [MessagesRequest.MessageTurn.ContentPart]) -> String {
        parts.compactMap { part in
            if case .text(let s) = part { return s }
            return nil
        }.joined(separator: "\n")
    }

    /// Strips special delimiter tokens from replayed assistant turns and drops media unsupported
    /// by the active model, counting dropped images and audio accurately.
    private static func convertToProviderTurns(
        messages: [MessageRecord],
        supportsVision: Bool,
        supportsAudio: Bool,
        specialTokens: [String]
    ) -> (turns: [MessagesRequest.MessageTurn], dropped: DroppedMediaCounts) {
        let extraTokens = specialTokens + [
            "<think>", "</think>",
            "<|channel|>", "<channel|>",
            "<|thought|>", "<|/thought|>",
        ]
        var droppedImages = 0
        var droppedAudio = 0
        var turns: [MessagesRequest.MessageTurn] = []

        for msg in messages {
            let cleanContent: String
            let cleanThinking: String?
            if msg.role == "assistant" {
                cleanContent = stripSpecialTokens(msg.content, extraTokens: extraTokens)
                cleanThinking = msg.thinking.map { stripSpecialTokens($0, extraTokens: extraTokens) }
            } else {
                cleanContent = msg.content
                cleanThinking = msg.thinking
            }

            if let blocks = msg.contentBlocks, !blocks.isEmpty {
                var keptParts: [MessagesRequest.MessageTurn.ContentPart] = []
                for block in blocks {
                    switch block {
                    case .text(let text):
                        let cleaned = msg.role == "assistant"
                            ? stripSpecialTokens(text, extraTokens: extraTokens)
                            : text
                        if !cleaned.isEmpty {
                            keptParts.append(.text(cleaned))
                        }
                    case .thinking(let th):
                        let cleaned = stripSpecialTokens(th, extraTokens: extraTokens)
                        if !cleaned.isEmpty {
                            keptParts.append(.thinking(cleaned))
                        }
                    case .media(let media):
                        if media.kind == "image" {
                            if supportsVision {
                                keptParts.append(.media(media))
                            } else {
                                droppedImages += 1
                            }
                        } else if media.kind == "audio" {
                            if supportsAudio {
                                keptParts.append(.media(media))
                            } else {
                                droppedAudio += 1
                            }
                        }
                    }
                }
                if keptParts.isEmpty && !cleanContent.isEmpty {
                    keptParts = [.text(cleanContent)]
                }
                turns.append(
                    MessagesRequest.MessageTurn(
                        role: msg.role,
                        content: cleanContent,
                        thinking: cleanThinking,
                        parts: keptParts
                    )
                )
            } else {
                turns.append(
                    MessagesRequest.MessageTurn(
                        role: msg.role,
                        content: cleanContent,
                        thinking: cleanThinking
                    )
                )
            }
        }
        return (turns, DroppedMediaCounts(images: droppedImages, audio: droppedAudio))
    }

    private static func stripSpecialTokens(_ text: String, extraTokens: [String]) -> String {
        guard !text.isEmpty else { return text }
        var stripper = StripSpecials(tokens: extraTokens)
        return stripper.feed(text) + stripper.finalize()
    }

    private final class SavedEventResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: Result<HeylookSavedEvent, Error>?

        var result: Result<HeylookSavedEvent, Error>? {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }

        func set(_ value: Result<HeylookSavedEvent, Error>) {
            lock.lock()
            storage = value
            lock.unlock()
        }
    }
}
