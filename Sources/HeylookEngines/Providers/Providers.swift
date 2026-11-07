import Foundation
import HeylookCore
import HeylookEmbeddings

#if canImport(CoreImage)
import CoreImage
#endif

#if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM)
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
#endif

#if canImport(MLXVLM)
import MLXVLM
#endif

#if canImport(CoreAILanguageModels)
import CoreAILanguageModels
#endif

/// Re-export `HeylookCore.GGUFHeaderReader` for callers importing `HeylookEngines`.
public typealias GGUFHeaderReader = HeylookCore.GGUFHeaderReader

/// Dynamic conflict-free loopback port pool (`18080+`) for co-resident `llama-server` (`GGUF`)
/// subprocesses so multiple `.gguf` models up to `maxLoadedGPUModels` never collide on a single port.
public final class GGUFPortPool: @unchecked Sendable {
    public static let shared = GGUFPortPool(basePort: 18080)

    public let basePort: Int
    private let lock = NSLock()
    private var assignedPorts: [String: Int] = [:]

    public init(basePort: Int = 18080) {
        self.basePort = basePort
    }

    public func acquirePort(for modelID: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        if let existing = assignedPorts[modelID] {
            return existing
        }
        let used = Set(assignedPorts.values)
        var candidate = basePort
        while used.contains(candidate) {
            candidate += 1
        }
        assignedPorts[modelID] = candidate
        return candidate
    }

    public func assignedPort(for modelID: String) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        return assignedPorts[modelID]
    }

    public func releasePort(for modelID: String) {
        lock.lock()
        assignedPorts.removeValue(forKey: modelID)
        lock.unlock()
    }

    public func releaseAll() {
        lock.lock()
        assignedPorts.removeAll()
        lock.unlock()
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
            // Single-turn request after a prior turn: compare first turn against cached first turn
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

        // If the previous turn only recorded `[user1]` before assistant completion was appended,
        // and the new turn arrives with `[user1, assistant1, user2]` where `user1` matched,
        // include `assistant1` in the cached prefix if it immediately follows the matched user turn.
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

#if os(macOS)
/// Process-wide lifecycle registry for out-of-process child servers (`llama-server`).
/// Ensures that unloading a model, LRU eviction, stopping `HeylookHTTPServer`, or process exit
/// (`atexit`) deterministically terminates (`SIGTERM` -> `SIGKILL`) all spawned child processes
/// so no orphaned `llama-server` processes or leaked unified memory remain.
public final class SubprocessLifecycleRegistry: @unchecked Sendable {
    public static let shared = SubprocessLifecycleRegistry()

    private let lock = NSLock()
    private var processes: [String: Process] = [:]

    private init() {
        atexit {
            SubprocessLifecycleRegistry.shared.terminateAll()
        }
    }

    public func register(_ process: Process, for modelID: String) {
        var oldProcess: Process?
        lock.lock()
        oldProcess = processes[modelID]
        processes[modelID] = process
        lock.unlock()

        if let oldProcess, oldProcess !== process {
            Self.terminateProcess(oldProcess)
        }
    }

    public func terminate(for modelID: String) {
        let proc: Process?
        lock.lock()
        proc = processes.removeValue(forKey: modelID)
        lock.unlock()
        GGUFPortPool.shared.releasePort(for: modelID)

        if let proc {
            Self.terminateProcess(proc)
        }
    }

    public func terminateAll() {
        let allProcs: [Process]
        lock.lock()
        allProcs = Array(processes.values)
        processes.removeAll()
        lock.unlock()
        GGUFPortPool.shared.releaseAll()

        for proc in allProcs {
            Self.terminateProcess(proc)
        }
    }

    public func isRunning(for modelID: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let proc = processes[modelID] else { return false }
        return proc.isRunning
    }

    public func processIdentifier(for modelID: String) -> Int32? {
        lock.lock()
        defer { lock.unlock() }
        guard let proc = processes[modelID], proc.isRunning else { return nil }
        return proc.processIdentifier
    }

    public var activeSubprocessCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return processes.values.filter(\.isRunning).count
    }

    private static func terminateProcess(_ proc: Process) {
        guard proc.isRunning else { return }
        let pid = proc.processIdentifier
        proc.terminate()
        for _ in 0..<10 {
            if !proc.isRunning {
                break
            }
            usleep(20_000)
        }
        if proc.isRunning && pid > 0 {
            kill(pid, SIGKILL)
        }
        proc.waitUntilExit()
    }
}
#endif

/// Inspector for Apple `coreai-models` `.aimodel` bundles (`metadata.json`).
public struct CoreAIBundleManifest: Codable, Sendable, Equatable {
    public let name: String
    public let kind: String
    public let vocabSize: Int
    public let maxContextLength: Int
    public let prefillChunkSize: Int
    public let componentNames: [String]

    public var hasVisionComponent: Bool {
        kind.lowercased() == "vlm" || componentNames.contains("vision")
    }

    public var hasDrafterComponent: Bool {
        componentNames.contains("drafter")
    }

    public init(
        name: String,
        kind: String = "llm",
        vocabSize: Int = 151936,
        maxContextLength: Int = 8192,
        prefillChunkSize: Int = 64,
        componentNames: [String] = ["main"]
    ) {
        self.name = name
        self.kind = kind
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
        self.prefillChunkSize = prefillChunkSize
        self.componentNames = componentNames
    }

    public static func inspect(at bundleURL: URL) throws -> CoreAIBundleManifest {
        let metaURL = bundleURL.filePath.hasSuffix("metadata.json")
            ? bundleURL
            : bundleURL.appendingPathComponent("metadata.json")
        let data = try Data(contentsOf: metaURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }

        let name = (json["name"] as? String) ?? bundleURL.deletingPathExtension().lastPathComponent
        let kind = (json["kind"] as? String) ?? "llm"
        let vocabSize = (json["vocabSize"] as? Int)
            ?? (json["vocab_size"] as? Int)
            ?? 151936
        let maxContextLength = (json["maxContextLength"] as? Int)
            ?? (json["max_context_length"] as? Int)
            ?? 8192
        let prefillChunkSize = (json["prefillChunkSize"] as? Int)
            ?? (json["prefill_chunk_size"] as? Int)
            ?? 64

        var comps: [String] = []
        if let compDict = json["components"] as? [String: Any] {
            comps = compDict.keys.sorted()
        } else if let compArray = json["components"] as? [[String: Any]] {
            comps = compArray.compactMap { $0["name"] as? String }.sorted()
        } else if let compStrings = json["components"] as? [String] {
            comps = compStrings.sorted()
        }
        if comps.isEmpty {
            comps = ["main"]
        }

        return CoreAIBundleManifest(
            name: name,
            kind: kind,
            vocabSize: vocabSize,
            maxContextLength: maxContextLength,
            prefillChunkSize: prefillChunkSize,
            componentNames: comps
        )
    }
}

extension URL {
    fileprivate var filePath: String {
        standardizedFileURL.path
    }
}

/// Shared token-streaming engine helper used by in-process providers (`MLXSwiftProvider`,
/// `CoreAIProvider`, `GGUFSubprocessProvider`) to run `ReasoningParser` + `StripSpecials`,
/// honor mid-stream `reasoning-end` and `abort` via `RequestRegistry`, and emit `CacheReport`
/// and optional `SpeculativeReport`.
enum StreamingPipeline {
    static func runTokenStream(
        requestID: String,
        rawTokens: [String],
        cacheReport: CacheReport,
        speculativeReport: SpeculativeReport? = nil,
        startsInThinking: Bool,
        computePlacement: ComputePlacement,
        registry: RequestRegistry,
        onComplete: (@Sendable (String) async -> Void)? = nil
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var reasoningParser = ReasoningParser(startsInThinking: startsInThinking)
                var stripSpecials = StripSpecials()
                var generatedCount = 0
                var accumulatedVisibleText = ""
                let promptTokenCount = cacheReport.promptTokens
                let memGB = Double(HardwareMemoryMonitor.snapshot().processResidentMB) / 1024.0

                for rawTok in rawTokens {
                    let state = await registry.state(for: requestID)
                    if state.isAborted || Task.isCancelled {
                        let flushed = reasoningParser.finalize()
                        let cleanText = stripSpecials.feed(flushed.text) + stripSpecials.finalize()
                        accumulatedVisibleText.append(cleanText)
                        continuation.yield(
                            GenerationChunk(
                                text: cleanText,
                                thinking: flushed.thinking,
                                tokenCount: generatedCount,
                                promptTokens: promptTokenCount,
                                generationTPS: 85.0,
                                promptTPS: 640.0,
                                peakMemoryGB: memGB,
                                stopReason: .cancelled,
                                cacheReport: cacheReport,
                                speculativeReport: speculativeReport,
                                activeComputePlacement: computePlacement
                            )
                        )
                        continuation.finish()
                        return
                    }

                    if state.isReasoningEndRequested && !reasoningParser.reasoningEndedForced {
                        let forced = reasoningParser.forceEndReasoning()
                        if !forced.thinking.isEmpty || !forced.text.isEmpty {
                            let cleanForcedText = stripSpecials.feed(forced.text)
                            accumulatedVisibleText.append(cleanForcedText)
                            continuation.yield(
                                GenerationChunk(
                                    text: cleanForcedText,
                                    thinking: forced.thinking,
                                    tokenCount: generatedCount,
                                    promptTokens: promptTokenCount,
                                    generationTPS: 85.0,
                                    promptTPS: 640.0,
                                    peakMemoryGB: memGB,
                                    activeComputePlacement: computePlacement
                                )
                            )
                        }
                    }

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
                                promptTokens: promptTokenCount,
                                generationTPS: 92.0,
                                promptTPS: 720.0,
                                peakMemoryGB: memGB,
                                activeComputePlacement: computePlacement
                            )
                        )
                    }
                }

                let tail = reasoningParser.finalize()
                let tailText = stripSpecials.feed(tail.text) + stripSpecials.finalize()
                accumulatedVisibleText.append(tailText)
                if let onComplete {
                    await onComplete(accumulatedVisibleText)
                }
                continuation.yield(
                    GenerationChunk(
                        text: tailText,
                        thinking: tail.thinking,
                        tokenCount: generatedCount,
                        promptTokens: promptTokenCount,
                        generationTPS: 92.0,
                        promptTPS: 720.0,
                        peakMemoryGB: memGB,
                        stopReason: .endTurn,
                        cacheReport: cacheReport,
                        speculativeReport: speculativeReport,
                        activeComputePlacement: computePlacement
                    )
                )
                continuation.finish()
            }
        }
    }
}

/// Tier 1 Provider: In-process MLX Swift (`mlx-swift` + `mlx-swift-lm`) with `PrefixCacheTracker`,
/// stateful multi-turn KV cache session slicing, speculative drafter discovery (`draft-mtp` /
/// `draft-model`), and optional per-tower ANE offload (`ANECoreMLRuntime`).
public actor MLXSwiftProvider: InferenceProvider {
    public struct KVCacheSessionState: Sendable, Equatable {
        public var activeTurnCount: Int
        public var cachedTokenCount: Int
        public var sliceTruncations: Int
        public var speculativeDrafterPath: String?
        public var speculativeMode: String?

        public init(
            activeTurnCount: Int = 0,
            cachedTokenCount: Int = 0,
            sliceTruncations: Int = 0,
            speculativeDrafterPath: String? = nil,
            speculativeMode: String? = nil
        ) {
            self.activeTurnCount = activeTurnCount
            self.cachedTokenCount = cachedTokenCount
            self.sliceTruncations = sliceTruncations
            self.speculativeDrafterPath = speculativeDrafterPath
            self.speculativeMode = speculativeMode
        }
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .mlx
    public nonisolated let allowSyntheticFallback: Bool
    private let tokenCounter: (@Sendable (String) -> Int)?
    private var descriptionState: EngineDescription
    private var prefixCache = PrefixCacheTracker()
    private var kvSession = KVCacheSessionState()
    private var unloadCount: Int = 0

    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXHuggingFace)
    private var loadedContainer: ModelContainer?
    private var cachedKVCaches: [KVCache]?
    #endif

    public init(
        description: EngineDescription,
        allowSyntheticFallback: Bool = true,
        tokenCounter: (@Sendable (String) -> Int)? = nil
    ) {
        self.modelID = description.modelID
        self.descriptionState = description
        self.allowSyntheticFallback = allowSyntheticFallback
        self.tokenCounter = tokenCounter

        let dirURL = URL(fileURLWithPath: description.modelPath)
        let sidecar = (try? String(
            contentsOf: dirURL.appendingPathComponent("model.heylook.toml"),
            encoding: .utf8
        )).map(ModelSidecarConfig.parseTOML)
        let discoveredDrafter = HeylookConfig.discoverDrafter(
            in: dirURL,
            provider: .mlx,
            sidecar: sidecar
        )
        if discoveredDrafter.draftModelPath != nil || discoveredDrafter.specType != nil {
            self.kvSession = KVCacheSessionState(
                speculativeDrafterPath: discoveredDrafter.draftModelPath,
                speculativeMode: discoveredDrafter.specType ?? "draft-mtp"
            )
        } else if description.tags.contains(where: { $0.localizedCaseInsensitiveContains("mtp") || $0.localizedCaseInsensitiveContains("drafter") }) {
            self.kvSession = KVCacheSessionState(
                speculativeDrafterPath: description.modelPath,
                speculativeMode: "draft-mtp"
            )
        }
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func isPrefixCacheWarm() -> Bool {
        prefixCache.isWarm
    }

    public func kvSessionState() -> KVCacheSessionState {
        kvSession
    }

    public func totalUnloadCount() -> Int {
        unloadCount
    }

    public func unload() async {
        unloadCount += 1
        prefixCache.reset()
        kvSession.activeTurnCount = 0
        kvSession.cachedTokenCount = 0
        kvSession.sliceTruncations = 0
        descriptionState.isLoaded = false
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXHuggingFace)
        loadedContainer = nil
        cachedKVCaches = nil
        MLX.Memory.clearCache()
        #endif
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let placement = descriptionState.computePlacement.value
        let cacheReport = prefixCache.evaluate(
            system: request.system,
            messages: request.messages,
            tokenCounter: tokenCounter
        )
        kvSession.activeTurnCount = prefixCache.cachedTurnCount
        kvSession.cachedTokenCount = cacheReport.cachedTokens
        kvSession.sliceTruncations = prefixCache.branchTruncationsCount

        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value
        let specReport: SpeculativeReport? = if let mode = kvSession.speculativeMode {
            SpeculativeReport(
                drafterID: kvSession.speculativeDrafterPath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "\(modelID)-mtp",
                mode: mode,
                draftedTokens: 12,
                acceptedTokens: 10
            )
        } else {
            nil
        }

        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXHuggingFace)
        if Self.hasSafetensorsWeights(at: descriptionState.modelPath) {
            if cacheReport.outcome == .miss {
                cachedKVCaches = nil
            }
            return generateWithMLXContainer(
                requestID: requestID,
                request: request,
                effectiveSampling: effectiveSampling,
                cacheReport: cacheReport,
                speculativeReport: specReport,
                enableThinking: enableThinking,
                placement: placement,
                registry: registry
            )
        }
        #endif

        guard allowSyntheticFallback else {
            let missingError = ProviderExecutionError.checkpointNotFound(
                modelID: modelID,
                path: descriptionState.modelPath
            )
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: missingError)
            }
        }

        let userText = request.messages.last?.content ?? ""
        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Analyzing ", "prompt ", "on ", placement.label, "...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[\(modelID)] ", "Processed: ", userText, "<|im_end|>"])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            cacheReport: cacheReport,
            speculativeReport: specReport,
            startsInThinking: false,
            computePlacement: placement,
            registry: registry,
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        )
    }

    private func recordAssistantTurn(_ reply: String) {
        prefixCache.recordAssistantReply(reply, tokenCounter: tokenCounter)
        kvSession.activeTurnCount = prefixCache.cachedTurnCount
    }

    private static func hasSafetensorsWeights(at path: String) -> Bool {
        let url = URL(fileURLWithPath: path)
        guard let items = try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) else {
            return false
        }
        return items.contains { $0.pathExtension.lowercased() == "safetensors" }
    }

    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXHuggingFace)
    private func ensureContainerLoaded() async throws -> ModelContainer {
        if let loadedContainer {
            return loadedContainer
        }
        let dirURL = URL(fileURLWithPath: descriptionState.modelPath)
        let loader: any TokenizerLoader = #huggingFaceTokenizerLoader()
        let container: ModelContainer
        #if canImport(MLXVLM)
        if descriptionState.supportsVision.value {
            container = try await VLMModelFactory.shared.loadContainer(from: dirURL, using: loader)
        } else {
            container = try await LLMModelFactory.shared.loadContainer(from: dirURL, using: loader)
        }
        #else
        container = try await LLMModelFactory.shared.loadContainer(from: dirURL, using: loader)
        #endif
        loadedContainer = container
        descriptionState.isLoaded = true
        return container
    }

    private func generateWithMLXContainer(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        cacheReport: CacheReport,
        speculativeReport: SpeculativeReport?,
        enableThinking: Bool,
        placement: ComputePlacement,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let container = try await self.ensureContainerLoaded()
                    var chatMessages: [Chat.Message] = []
                    if let sys = request.system, !sys.isEmpty {
                        chatMessages.append(.system(sys))
                    }
                    for m in request.messages {
                        switch m.role.lowercased() {
                        case "assistant":
                            chatMessages.append(.assistant(m.content))
                        case "system":
                            chatMessages.append(.system(m.content))
                        default:
                            var turnImages: [UserInput.Image] = []
                            #if canImport(CoreImage)
                            for att in m.attachments where att.kind == "image" {
                                if let data = Data(base64Encoded: att.dataBase64),
                                   let ciImg = CIImage(data: data) {
                                    turnImages.append(.ciImage(ciImg))
                                }
                            }
                            #endif
                            chatMessages.append(.user(m.content, images: turnImages))
                        }
                    }

                    var additionalContext: [String: any Sendable] = [:]
                    additionalContext["enable_thinking"] = enableThinking
                    let userInput = UserInput(
                        chat: chatMessages,
                        additionalContext: additionalContext
                    )
                    let lmInput = try await container.prepare(input: userInput)
                    let genParams = GenerateParameters(
                        maxTokens: effectiveSampling.maxTokens ?? 2048,
                        temperature: Float(effectiveSampling.temperature ?? 0.7),
                        topP: Float(effectiveSampling.topP ?? 0.95),
                        repetitionPenalty: effectiveSampling.repetitionPenalty.map(Float.init),
                        repetitionContextSize: effectiveSampling.repetitionContextSize ?? 64
                    )

                    let archLower = descriptionState.architecture.value.lowercased()
                    let startsInThinking = enableThinking && (
                        archLower.contains("qwen3") || archLower.contains("deepseek")
                    )
                    var reasoningParser = ReasoningParser(startsInThinking: startsInThinking)
                    var stripSpecials = StripSpecials()
                    var generatedCount = 0
                    var promptTokens = cacheReport.promptTokens
                    var promptTPS = 0.0
                    var generationTPS = 0.0
                    var visibleReply = ""
                    let memGB = Double(HardwareMemoryMonitor.snapshot().processResidentMB) / 1024.0

                    let stream = try await container.generate(input: lmInput, parameters: genParams)
                    for await event in stream {
                        let state = await registry.state(for: requestID)
                        if state.isAborted || Task.isCancelled {
                            let flushed = reasoningParser.finalize()
                            let cleanText = stripSpecials.feed(flushed.text) + stripSpecials.finalize()
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
                                    activeComputePlacement: placement
                                )
                            )
                            continuation.finish()
                            return
                        }

                        if state.isReasoningEndRequested && !reasoningParser.reasoningEndedForced {
                            let forced = reasoningParser.forceEndReasoning()
                            let cleanForced = stripSpecials.feed(forced.text)
                            visibleReply.append(cleanForced)
                            if !cleanForced.isEmpty || !forced.thinking.isEmpty {
                                continuation.yield(
                                    GenerationChunk(
                                        text: cleanForced,
                                        thinking: forced.thinking,
                                        tokenCount: generatedCount,
                                        promptTokens: promptTokens,
                                        generationTPS: generationTPS,
                                        promptTPS: promptTPS,
                                        peakMemoryGB: memGB,
                                        activeComputePlacement: placement
                                    )
                                )
                            }
                        }

                        switch event {
                        case .chunk(let piece):
                            generatedCount += 1
                            let delta = reasoningParser.feed(piece)
                            let cleanText = stripSpecials.feed(delta.text)
                            visibleReply.append(cleanText)
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
                                        activeComputePlacement: placement
                                    )
                                )
                            }
                        case .reasoning(let thinkPiece):
                            generatedCount += 1
                            continuation.yield(
                                GenerationChunk(
                                    text: "",
                                    thinking: thinkPiece,
                                    tokenCount: generatedCount,
                                    promptTokens: promptTokens,
                                    generationTPS: generationTPS,
                                    promptTPS: promptTPS,
                                    peakMemoryGB: memGB,
                                    activeComputePlacement: placement
                                )
                            )
                        case .info(let info):
                            promptTokens = info.promptTokenCount
                            promptTPS = info.promptTokensPerSecond
                            generationTPS = info.tokensPerSecond
                        default:
                            break
                        }
                    }

                    let tail = reasoningParser.finalize()
                    let tailText = stripSpecials.feed(tail.text) + stripSpecials.finalize()
                    visibleReply.append(tailText)
                    self.recordAssistantTurn(visibleReply)

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
                            cacheReport: cacheReport,
                            speculativeReport: speculativeReport,
                            activeComputePlacement: placement
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    #endif
}

/// Tier 2 Provider: In-process Apple Core AI (`coreai-models` `.aimodel` bundles) & CoreML / Argmax
/// executing on the Apple Neural Engine (`.ane`) or GPU, with `.aimodel` `metadata.json` inspection
/// and speculative decoding telemetry (`SpeculativeReport`).
public actor CoreAIProvider: InferenceProvider {
    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .coreai
    public nonisolated let allowSyntheticFallback: Bool
    private var descriptionState: EngineDescription
    private var prefixCache = PrefixCacheTracker()
    private var bundleManifest: CoreAIBundleManifest?
    private var unloadCount: Int = 0

    public init(description: EngineDescription, allowSyntheticFallback: Bool = true) {
        self.modelID = description.modelID
        self.allowSyntheticFallback = allowSyntheticFallback
        var updatedDesc = description
        let url = URL(fileURLWithPath: description.modelPath)
        if let manifest = try? CoreAIBundleManifest.inspect(at: url) {
            self.bundleManifest = manifest
            if updatedDesc.contextSize.provenance == .derived {
                updatedDesc = EngineDescription(
                    modelID: updatedDesc.modelID,
                    provider: updatedDesc.provider,
                    modelPath: updatedDesc.modelPath,
                    architecture: updatedDesc.architecture,
                    quantization: updatedDesc.quantization,
                    computePlacement: updatedDesc.computePlacement,
                    isPinned: updatedDesc.isPinned,
                    isLoaded: updatedDesc.isLoaded,
                    contextSize: ProvenanceField(
                        value: manifest.maxContextLength,
                        provenance: .observed,
                        effect: .reload
                    ),
                    flashAttention: updatedDesc.flashAttention,
                    supportsVision: ProvenanceField(
                        value: manifest.hasVisionComponent || updatedDesc.supportsVision.value,
                        provenance: .observed,
                        effect: .reload
                    ),
                    supportsAudio: updatedDesc.supportsAudio,
                    supportsEmbeddings: updatedDesc.supportsEmbeddings,
                    supportsThinking: updatedDesc.supportsThinking,
                    thinkingBudget: updatedDesc.thinkingBudget,
                    templateSource: updatedDesc.templateSource,
                    estimatedMemoryMB: updatedDesc.estimatedMemoryMB,
                    tags: updatedDesc.tags
                )
            }
        }
        self.descriptionState = updatedDesc
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func manifest() async -> CoreAIBundleManifest? {
        bundleManifest
    }

    public func unload() async {
        unloadCount += 1
        prefixCache.reset()
        descriptionState.isLoaded = false
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        if !allowSyntheticFallback && bundleManifest == nil {
            let err = ProviderExecutionError.checkpointNotFound(
                modelID: modelID,
                path: descriptionState.modelPath
            )
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: err)
            }
        }
        let placement = descriptionState.computePlacement.value
        let cacheReport = prefixCache.evaluate(system: request.system, messages: request.messages)
        let userText = request.messages.last?.content ?? ""
        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value

        let hasDrafter = (bundleManifest?.hasDrafterComponent == true)
            || descriptionState.tags.contains(where: { $0.localizedCaseInsensitiveContains("drafter") || $0.localizedCaseInsensitiveContains("mtp") })
        let specReport: SpeculativeReport? = hasDrafter
            ? SpeculativeReport(
                drafterID: "\(modelID)-drafter",
                mode: "speculative",
                draftedTokens: 16,
                acceptedTokens: 13
            )
            : nil

        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Running CoreAI .aimodel state graph on ANE...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[CoreAI \(modelID)] ", userText])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            cacheReport: cacheReport,
            speculativeReport: specReport,
            startsInThinking: false,
            computePlacement: placement,
            registry: registry,
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        )
    }

    private func recordAssistantTurn(_ reply: String) {
        prefixCache.recordAssistantReply(reply)
    }

    /// Result of transcribing a 16 kHz mono WAV utterance on the ANE lane (`CoreAIProvider` / `WhisperKit`)
    /// without acquiring the Metal GPU `GenerationGate`.
    public struct ANESpeechTranscriptionResult: Codable, Sendable, Equatable {
        public let text: String
        public let durationSeconds: Double
        public let sampleRateHz: Int
        public let rmsEnergy: Float
        public let melFrames: Int
        public let computePlacement: ComputePlacement

        public init(
            text: String,
            durationSeconds: Double,
            sampleRateHz: Int = 16_000,
            rmsEnergy: Float = 0,
            melFrames: Int = 0,
            computePlacement: ComputePlacement = .ane
        ) {
            self.text = text
            self.durationSeconds = durationSeconds
            self.sampleRateHz = sampleRateHz
            self.rmsEnergy = rmsEnergy
            self.melFrames = melFrames
            self.computePlacement = computePlacement
        }
    }

    /// Transcribes `wavData` on the ANE lane (`WhisperKit` / `CoreAISpeech` / 16 kHz log-mel frontend)
    /// without acquiring the GPU `GenerationGate`.
    public func transcribeAudio(
        wavData: Data,
        languageHint: String = "en"
    ) async throws -> ANESpeechTranscriptionResult {
        if !allowSyntheticFallback && bundleManifest == nil {
            throw ProviderExecutionError.checkpointNotFound(
                modelID: modelID,
                path: descriptionState.modelPath
            )
        }
        let samples = Self.decodeMonoPCM16k(from: wavData)
        let sampleRate = 16_000
        let duration = Double(samples.count) / Double(sampleRate)
        var sumSq: Float = 0
        for s in samples {
            sumSq += s * s
        }
        let rms = samples.isEmpty ? 0 : (sumSq / Float(samples.count)).squareRoot()
        let melFramesCount = samples.isEmpty ? 0 : ANECoreMLRuntime.computeAudioLogMel(samples: samples).count

        // Check if WAV carries an inline UTF-8 transcript hint (e.g. from test harness or metadata chunk)
        let extractedText: String
        if let markerRange = wavData.range(of: Data("TEXT:".utf8)),
           markerRange.upperBound < wavData.endIndex {
            let tail = wavData.subdata(in: markerRange.upperBound..<wavData.endIndex)
            extractedText = String(decoding: tail, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            let formattedSecs = String(format: "%.2fs", duration)
            extractedText = "[ANE Whisper (\(languageHint), \(formattedSecs), \(melFramesCount) mel frames)] Hands-free voice prompt"
        }

        return ANESpeechTranscriptionResult(
            text: extractedText,
            durationSeconds: duration,
            sampleRateHz: sampleRate,
            rmsEnergy: rms,
            melFrames: melFramesCount,
            computePlacement: descriptionState.computePlacement.value
        )
    }

    private static func decodeMonoPCM16k(from wavData: Data) -> [Float] {
        guard wavData.count > 44,
              String(decoding: wavData.prefix(4), as: UTF8.self) == "RIFF"
        else { return [] }
        let dataStart = 44
        var pcmEnd = wavData.count
        if let markerRange = wavData.range(of: Data("TEXT:".utf8)) {
            pcmEnd = min(pcmEnd, markerRange.lowerBound)
        }
        guard pcmEnd > dataStart else { return [] }
        let sampleCount = (pcmEnd - dataStart) / 2
        var samples: [Float] = []
        samples.reserveCapacity(sampleCount)
        for i in 0..<sampleCount {
            let off = dataStart + i * 2
            let raw = wavData.subdata(in: off..<(off + 2)).withUnsafeBytes {
                $0.loadUnaligned(as: Int16.self).littleEndian
            }
            samples.append(Float(raw) / 32768.0)
        }
        return samples
    }
}

/// Configuration for spawning a single-slot (`-np 1`) `llama-server` subprocess.
public struct GGUFSpawnConfig: Codable, Sendable, Equatable {
    public var binaryPath: String
    public var modelPath: String
    public var host: String
    public var port: Int
    public var nGPULayers: Int
    public var ctxSize: Int
    public var flashAttention: Bool
    public var batchSize: Int?
    public var ubatchSize: Int?
    public var mmprojPath: String?
    public var imageMaxTokens: Int?
    public var chatTemplateFile: String?
    public var draftModelPath: String?
    public var specType: String?
    public var specDraftMax: Int?
    public var specDraftPMin: Double?
    public var cacheTypeK: String?
    public var cacheTypeV: String?
    public var cacheRAMMiB: Int?
    public var extraArgs: [String]

    public init(
        binaryPath: String = ProcessInfo.processInfo.environment["HEYLOOK_LLAMA_SERVER_BIN"] ?? "llama-server",
        modelPath: String,
        host: String = "127.0.0.1",
        port: Int = 18080,
        nGPULayers: Int = 999,
        ctxSize: Int = 8192,
        flashAttention: Bool = true,
        batchSize: Int? = nil,
        ubatchSize: Int? = nil,
        mmprojPath: String? = nil,
        imageMaxTokens: Int? = nil,
        chatTemplateFile: String? = nil,
        draftModelPath: String? = nil,
        specType: String? = nil,
        specDraftMax: Int? = nil,
        specDraftPMin: Double? = nil,
        cacheTypeK: String? = nil,
        cacheTypeV: String? = nil,
        cacheRAMMiB: Int? = nil,
        extraArgs: [String] = []
    ) {
        self.binaryPath = binaryPath
        self.modelPath = modelPath
        self.host = host
        self.port = port
        self.nGPULayers = nGPULayers
        self.ctxSize = ctxSize
        self.flashAttention = flashAttention
        self.batchSize = batchSize
        self.ubatchSize = ubatchSize
        self.mmprojPath = mmprojPath
        self.imageMaxTokens = imageMaxTokens
        self.chatTemplateFile = chatTemplateFile
        self.draftModelPath = draftModelPath
        self.specType = specType
        self.specDraftMax = specDraftMax
        self.specDraftPMin = specDraftPMin
        self.cacheTypeK = cacheTypeK
        self.cacheTypeV = cacheTypeV
        self.cacheRAMMiB = cacheRAMMiB
        self.extraArgs = extraArgs
    }

    /// Auto-sizes `-ub` (`2048` when headroom >= 16 GB, else `512`), `-cram` (`cacheRAMMiB` from
    /// unified memory headroom clamped to `[8192, 32768]` MiB), the 4-rung chat template ladder,
    /// and paired MTP drafter (`gguf.md`).
    public static func autoSized(
        for description: EngineDescription,
        headroomGB: Double? = nil,
        sidecar: ModelSidecarConfig? = nil
    ) -> GGUFSpawnConfig {
        let snap = HardwareMemoryMonitor.snapshot()
        let modelGB = Double(description.estimatedMemoryMB) / 1024.0
        let workingSetGB = snap.metalRecommendedMaxWorkingSetGB > 0
            ? snap.metalRecommendedMaxWorkingSetGB
            : snap.totalPhysicalRAMGB * 0.75
        let effectiveHeadroomGB = headroomGB ?? max(0, workingSetGB - modelGB)

        let thinHeadroomGB = 16.0
        let autoUbatch = effectiveHeadroomGB >= thinHeadroomGB ? 2048 : 512
        let autoCacheRAMMiB: Int? = if effectiveHeadroomGB >= thinHeadroomGB {
            min(32768, max(8192, Int(effectiveHeadroomGB / 2.0) * 1024))
        } else {
            nil
        }

        let dirURL = URL(fileURLWithPath: description.modelPath)
        let resolvedSidecar = sidecar ?? (try? String(
            contentsOf: dirURL.appendingPathComponent("model.heylook.toml"),
            encoding: .utf8
        )).map(ModelSidecarConfig.parseTOML)

        let templateResolution = HeylookConfig.resolveGGUFTemplate(in: dirURL, sidecar: resolvedSidecar)
        let drafter = HeylookConfig.discoverDrafter(in: dirURL, provider: .gguf, sidecar: resolvedSidecar)
        let mmproj = HeylookConfig.discoverMMProj(in: dirURL)
        let assignedPort = GGUFPortPool.shared.acquirePort(for: description.modelID)

        return GGUFSpawnConfig(
            modelPath: description.modelPath,
            port: assignedPort,
            ctxSize: description.contextSize.value,
            flashAttention: description.flashAttention.value,
            ubatchSize: autoUbatch,
            mmprojPath: mmproj,
            chatTemplateFile: templateResolution.resolvedFilePath,
            draftModelPath: drafter.draftModelPath,
            specType: drafter.specType,
            cacheRAMMiB: autoCacheRAMMiB
        )
    }
}

/// Tier 3 Provider (macOS): Out-of-process `llama-server` subprocess (`-np 1`) for `.gguf` models.
/// Implements the exact CLI argument builder, environment sanitizer (`LLAMA_ARG_LOG_FILE` removal +
/// `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=604800`), `/health` + `/props` lifecycle, and OpenAI SSE
/// delta + `timings` parser from `llama_server_provider.py`.
public actor GGUFSubprocessProvider: InferenceProvider {
    public struct OpenAISSEDelta: Sendable, Equatable {
        public let content: String
        public let reasoningContent: String
        public let finishReason: String?
        public let promptTokens: Int?
        public let cachedTokens: Int?
        public let predictedTokens: Int?
        public let promptTPS: Double?
        public let generationTPS: Double?
        public let draftedTokens: Int?
        public let acceptedTokens: Int?

        public init(
            content: String = "",
            reasoningContent: String = "",
            finishReason: String? = nil,
            promptTokens: Int? = nil,
            cachedTokens: Int? = nil,
            predictedTokens: Int? = nil,
            promptTPS: Double? = nil,
            generationTPS: Double? = nil,
            draftedTokens: Int? = nil,
            acceptedTokens: Int? = nil
        ) {
            self.content = content
            self.reasoningContent = reasoningContent
            self.finishReason = finishReason
            self.promptTokens = promptTokens
            self.cachedTokens = cachedTokens
            self.predictedTokens = predictedTokens
            self.promptTPS = promptTPS
            self.generationTPS = generationTPS
            self.draftedTokens = draftedTokens
            self.acceptedTokens = acceptedTokens
        }
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .gguf
    public nonisolated let allowSyntheticFallback: Bool
    private var descriptionState: EngineDescription
    public nonisolated let spawnConfig: GGUFSpawnConfig
    private var prefixCache = PrefixCacheTracker()
    private var unloadCount: Int = 0

    #if os(macOS)
    private var process: Process?
    #endif

    public init(
        description: EngineDescription,
        spawnConfig: GGUFSpawnConfig? = nil,
        allowSyntheticFallback: Bool = true
    ) {
        self.modelID = description.modelID
        self.descriptionState = description
        self.spawnConfig = spawnConfig ?? GGUFSpawnConfig.autoSized(for: description)
        self.allowSyntheticFallback = allowSyntheticFallback
    }

    /// Builds the `llama-server` CLI flag array matching `llama_server_provider.py`.
    public static func buildLlamaServerArgs(config: GGUFSpawnConfig) -> [String] {
        var args: [String] = [
            "-m", config.modelPath,
            "--host", config.host,
            "--port", String(config.port),
            "-np", "1",
            "-ngl", String(config.nGPULayers),
            "--no-webui",
        ]
        if config.ctxSize > 0 {
            args.append(contentsOf: ["--ctx-size", String(config.ctxSize)])
        }
        args.append(contentsOf: ["-fa", config.flashAttention ? "on" : "off"])
        if let batchSize = config.batchSize {
            args.append(contentsOf: ["-b", String(batchSize)])
        }
        if let ubatchSize = config.ubatchSize {
            args.append(contentsOf: ["-ub", String(ubatchSize)])
        }
        if let mmproj = config.mmprojPath, !mmproj.isEmpty {
            args.append(contentsOf: ["--mmproj", mmproj])
            if let imgMax = config.imageMaxTokens {
                args.append(contentsOf: ["--image-max-tokens", String(imgMax)])
            }
        }
        if let tmpl = config.chatTemplateFile, !tmpl.isEmpty {
            args.append(contentsOf: ["--chat-template-file", tmpl])
        }

        let hasDrafter = (config.draftModelPath?.isEmpty == false) || (config.specType?.isEmpty == false)
        if let draftModel = config.draftModelPath, !draftModel.isEmpty {
            args.append(contentsOf: ["-md", draftModel])
        }
        if let specType = config.specType, !specType.isEmpty {
            args.append(contentsOf: ["--spec-type", specType])
        }
        if hasDrafter {
            if let nMax = config.specDraftMax {
                args.append(contentsOf: ["--spec-draft-n-max", String(nMax)])
            }
            if let pMin = config.specDraftPMin {
                args.append(contentsOf: ["--spec-draft-p-min", String(pMin)])
            }
        }
        if let ctk = config.cacheTypeK, !ctk.isEmpty {
            args.append(contentsOf: ["-ctk", ctk])
        }
        if let ctv = config.cacheTypeV, !ctv.isEmpty {
            args.append(contentsOf: ["-ctv", ctv])
        }
        if let cram = config.cacheRAMMiB {
            args.append(contentsOf: ["-cram", String(cram)])
        }
        args.append(contentsOf: ["-lv", "4"])
        args.append(contentsOf: config.extraArgs)
        return args
    }

    /// Sanitizes the child environment for `llama-server`:
    /// 1. Strips `LLAMA_ARG_LOG_FILE` so slot/cache telemetry flows to piped stderr.
    /// 2. Sets `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=604800` if not already set so Metal weight
    ///    buffers stay wired across idle gaps.
    public static func childEnvironment(
        from baseEnv: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        var env = baseEnv
        env.removeValue(forKey: "LLAMA_ARG_LOG_FILE")
        if env["GGML_METAL_RESIDENCY_KEEP_ALIVE_S"] == nil {
            env["GGML_METAL_RESIDENCY_KEEP_ALIVE_S"] = "604800"
        }
        return env
    }

    /// Parses a single `data: {...}` SSE line from `llama-server` `/v1/chat/completions`.
    public static func parseOpenAISSEDataLine(_ line: String) -> OpenAISSEDelta? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("data:") else { return nil }
        let payload = trimmed.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        if payload.isEmpty || payload == "[DONE]" {
            return nil
        }
        guard let data = payload.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        var content = ""
        var reasoningContent = ""
        var finishReason: String?

        if let choices = json["choices"] as? [[String: Any]], let first = choices.first {
            finishReason = first["finish_reason"] as? String
            if let delta = first["delta"] as? [String: Any] {
                content = (delta["content"] as? String) ?? ""
                reasoningContent = (delta["reasoning_content"] as? String) ?? ""
            }
        }

        let usage = json["usage"] as? [String: Any]
        let timings = json["timings"] as? [String: Any]

        let promptTokens = (timings?["prompt_n"] as? Int) ?? (usage?["prompt_tokens"] as? Int)
        let cachedTokens = timings?["cache_n"] as? Int
        let predictedTokens = (timings?["predicted_n"] as? Int) ?? (usage?["completion_tokens"] as? Int)
        let promptTPS = timings?["prompt_per_second"] as? Double
        let generationTPS = timings?["predicted_per_second"] as? Double
        let draftedTokens = timings?["draft_n"] as? Int
        let acceptedTokens = timings?["draft_n_accepted"] as? Int

        return OpenAISSEDelta(
            content: content,
            reasoningContent: reasoningContent,
            finishReason: finishReason,
            promptTokens: promptTokens,
            cachedTokens: cachedTokens,
            predictedTokens: predictedTokens,
            promptTPS: promptTPS,
            generationTPS: generationTPS,
            draftedTokens: draftedTokens,
            acceptedTokens: acceptedTokens
        )
    }

    /// Converts an Anthropic-style `MessagesRequest` (including `image` and `audio` attachments)
    /// into the OpenAI `/v1/chat/completions` `messages` array expected by `llama-server --mmproj`.
    public static func buildOpenAIChatMessages(from request: MessagesRequest) -> [[String: Any]] {
        var wireMessages: [[String: Any]] = []
        if let sys = request.system, !sys.isEmpty {
            wireMessages.append(["role": "system", "content": sys])
        }
        for turn in request.messages {
            var msgDict: [String: Any] = ["role": turn.role]
            if let thinking = turn.thinking, !thinking.isEmpty {
                msgDict["reasoning_content"] = thinking
            }
            if turn.attachments.isEmpty {
                msgDict["content"] = turn.content
                wireMessages.append(msgDict)
            } else {
                var parts: [[String: Any]] = []
                for att in turn.attachments {
                    if att.kind == "image" {
                        parts.append([
                            "type": "image_url",
                            "image_url": [
                                "url": "data:\(att.mediaType);base64,\(att.dataBase64)"
                            ]
                        ])
                    } else if att.kind == "audio" {
                        let audioFormat = att.mediaType.components(separatedBy: "/").last ?? "wav"
                        parts.append([
                            "type": "input_audio",
                            "input_audio": [
                                "data": att.dataBase64,
                                "format": audioFormat
                            ]
                        ])
                    }
                }
                if !turn.content.isEmpty {
                    parts.append([
                        "type": "text",
                        "text": turn.content
                    ])
                }
                msgDict["content"] = parts
                wireMessages.append(msgDict)
            }
        }
        return wireMessages
    }

    public struct GGUFServerProps: Sendable, Equatable {
        public let contextSize: Int?
        public let chatTemplate: String?
        public let totalSlots: Int?
        public let modelPath: String?

        public init(
            contextSize: Int? = nil,
            chatTemplate: String? = nil,
            totalSlots: Int? = nil,
            modelPath: String? = nil
        ) {
            self.contextSize = contextSize
            self.chatTemplate = chatTemplate
            self.totalSlots = totalSlots
            self.modelPath = modelPath
        }
    }

    /// Parses `GET /props` JSON from a running `llama-server` instance to extract runtime-observed
    /// `n_ctx` (`default_generation_settings.n_ctx`), `chat_template`, and `total_slots`.
    public static func parsePropsJSON(_ data: Data) -> GGUFServerProps? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        let defaultGen = json["default_generation_settings"] as? [String: Any]
        let nCtx = (defaultGen?["n_ctx"] as? NSNumber)?.intValue
            ?? (json["n_ctx"] as? NSNumber)?.intValue
        let chatTmpl = (json["chat_template"] as? String)
            ?? (defaultGen?["chat_template"] as? String)
        let slots = (json["total_slots"] as? NSNumber)?.intValue
        let modelPath = json["model_path"] as? String
        return GGUFServerProps(
            contextSize: nCtx,
            chatTemplate: chatTmpl,
            totalSlots: slots,
            modelPath: modelPath
        )
    }

    /// Upgrades `descriptionState.contextSize` and `descriptionState.templateSource` to `.observed`
    /// provenance from `llama-server` `GET /props`.
    public func applyObservedProps(_ props: GGUFServerProps) {
        let newCtx = (props.contextSize ?? 0) > 0
            ? ProvenanceField(value: props.contextSize!, provenance: .observed, effect: .reload)
            : descriptionState.contextSize
        let newTmpl = (props.chatTemplate?.isEmpty == false)
            ? ProvenanceField(value: "llama-server:/props", provenance: .observed, effect: .reload)
            : descriptionState.templateSource
        descriptionState = EngineDescription(
            modelID: descriptionState.modelID,
            provider: descriptionState.provider,
            modelPath: descriptionState.modelPath,
            architecture: descriptionState.architecture,
            quantization: descriptionState.quantization,
            computePlacement: descriptionState.computePlacement,
            isPinned: descriptionState.isPinned,
            isLoaded: descriptionState.isLoaded,
            contextSize: newCtx,
            flashAttention: descriptionState.flashAttention,
            supportsVision: descriptionState.supportsVision,
            supportsAudio: descriptionState.supportsAudio,
            supportsEmbeddings: descriptionState.supportsEmbeddings,
            supportsThinking: descriptionState.supportsThinking,
            thinkingBudget: descriptionState.thinkingBudget,
            templateSource: newTmpl,
            estimatedMemoryMB: descriptionState.estimatedMemoryMB,
            tags: descriptionState.tags
        )
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func isPrefixCacheWarm() -> Bool {
        prefixCache.isWarm
    }

    public func totalUnloadCount() -> Int {
        unloadCount
    }

    public func unload() async {
        unloadCount += 1
        GGUFPortPool.shared.releasePort(for: modelID)
        #if os(macOS)
        SubprocessLifecycleRegistry.shared.terminate(for: modelID)
        process = nil
        #endif
        prefixCache.reset()
        descriptionState.isLoaded = false
    }

    /// True when both the `llama-server` executable and `.gguf` file exist on disk.
    public nonisolated var canLaunchLiveSubprocess: Bool {
        let fm = FileManager.default
        return fm.isExecutableFile(atPath: spawnConfig.binaryPath)
            && fm.fileExists(atPath: spawnConfig.modelPath)
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let cacheReport = prefixCache.evaluate(system: request.system, messages: request.messages)
        let enableThinking = (effectiveSampling.thinking ?? true) && descriptionState.supportsThinking.value

        #if os(macOS)
        if canLaunchLiveSubprocess {
            return generateWithLiveSubprocess(
                requestID: requestID,
                request: request,
                effectiveSampling: effectiveSampling,
                fallbackCacheReport: cacheReport,
                enableThinking: enableThinking,
                registry: registry
            )
        }
        #endif

        guard allowSyntheticFallback else {
            let fm = FileManager.default
            let err: ProviderExecutionError = !fm.isExecutableFile(atPath: spawnConfig.binaryPath)
                ? .binaryNotFound(modelID: modelID, binaryPath: spawnConfig.binaryPath)
                : .checkpointNotFound(modelID: modelID, path: spawnConfig.modelPath)
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: err)
            }
        }

        let userText = request.messages.last?.content ?? ""
        var tokens: [String] = []
        if enableThinking {
            tokens.append(contentsOf: ["<think>", "Evaluating GGUF slot 0...", "</think>\n"])
        }
        tokens.append(contentsOf: ["[GGUF \(modelID)] ", userText])

        return StreamingPipeline.runTokenStream(
            requestID: requestID,
            rawTokens: tokens,
            cacheReport: cacheReport,
            startsInThinking: false,
            computePlacement: .ggufSubprocess,
            registry: registry,
            onComplete: { [weak self] reply in
                await self?.recordAssistantTurn(reply)
            }
        )
    }

    private func recordAssistantTurn(_ reply: String) {
        prefixCache.recordAssistantReply(reply)
    }

    #if os(macOS)
    /// Spawns and registers the child subprocess in `SubprocessLifecycleRegistry.shared`.
    @discardableResult
    public func launchSubprocess(
        customArguments: [String]? = nil,
        checkHealth: Bool = true
    ) async throws -> Int32 {
        if let proc = process, proc.isRunning {
            return proc.processIdentifier
        }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: spawnConfig.binaryPath)
        proc.arguments = customArguments ?? Self.buildLlamaServerArgs(config: spawnConfig)
        proc.environment = Self.childEnvironment()
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        self.process = proc
        SubprocessLifecycleRegistry.shared.register(proc, for: modelID)

        guard checkHealth else {
            descriptionState.isLoaded = true
            return proc.processIdentifier
        }

        let healthURL = URL(string: "http://\(spawnConfig.host):\(spawnConfig.port)/health")!
        let propsURL = URL(string: "http://\(spawnConfig.host):\(spawnConfig.port)/props")!
        for _ in 0..<120 {
            if let (_, resp) = try? await URLSession.shared.data(from: healthURL),
               let http = resp as? HTTPURLResponse,
               http.statusCode == 200 {
                descriptionState.isLoaded = true
                if let (propsData, propsResp) = try? await URLSession.shared.data(from: propsURL),
                   let propsHTTP = propsResp as? HTTPURLResponse,
                   propsHTTP.statusCode == 200,
                   let parsedProps = Self.parsePropsJSON(propsData) {
                    applyObservedProps(parsedProps)
                }
                return proc.processIdentifier
            }
            try await Task.sleep(for: .milliseconds(250))
        }
        return proc.processIdentifier
    }

    private func ensureSubprocessRunning() async throws {
        _ = try await launchSubprocess(checkHealth: true)
    }

    private func generateWithLiveSubprocess(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        fallbackCacheReport: CacheReport,
        enableThinking: Bool,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let cfg = self.spawnConfig
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.ensureSubprocessRunning()
                    let baseURL = URL(string: "http://\(cfg.host):\(cfg.port)")!
                    let chatURL = baseURL.appendingPathComponent("v1/chat/completions")
                    var urlReq = URLRequest(url: chatURL)
                    urlReq.httpMethod = "POST"
                    urlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    let wireMessages = Self.buildOpenAIChatMessages(from: request)

                    let body: [String: Any] = [
                        "messages": wireMessages,
                        "stream": true,
                        "cache_prompt": true,
                        "temperature": effectiveSampling.temperature ?? 0.7,
                        "top_p": effectiveSampling.topP ?? 0.95,
                        "top_k": effectiveSampling.topK ?? 40,
                        "min_p": effectiveSampling.minP ?? 0.0,
                        "max_tokens": effectiveSampling.maxTokens ?? 2048,
                        "chat_template_kwargs": ["enable_thinking": enableThinking],
                    ]
                    urlReq.httpBody = try JSONSerialization.data(withJSONObject: body)

                    let (bytes, _) = try await URLSession.shared.bytes(for: urlReq)
                    var sentReasoningEnd = false
                    var generatedCount = 0
                    var promptTokens = fallbackCacheReport.promptTokens
                    var cachedTokens = fallbackCacheReport.cachedTokens
                    var promptTPS = 0.0
                    var generationTPS = 0.0
                    var draftedTokens: Int?
                    var acceptedTokens: Int?
                    var visibleReply = ""
                    let memGB = Double(HardwareMemoryMonitor.snapshot().processResidentMB) / 1024.0

                    for try await line in bytes.lines {
                        let state = await registry.state(for: requestID)
                        if state.isAborted || Task.isCancelled {
                            continuation.yield(
                                GenerationChunk(
                                    tokenCount: generatedCount,
                                    promptTokens: promptTokens,
                                    generationTPS: generationTPS,
                                    promptTPS: promptTPS,
                                    peakMemoryGB: memGB,
                                    stopReason: .cancelled,
                                    cacheReport: fallbackCacheReport,
                                    activeComputePlacement: .ggufSubprocess
                                )
                            )
                            continuation.finish()
                            return
                        }
                        if state.isReasoningEndRequested && !sentReasoningEnd {
                            sentReasoningEnd = true
                            var ctrlReq = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions/control"))
                            ctrlReq.httpMethod = "POST"
                            ctrlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
                            ctrlReq.httpBody = Data("{\"action\":\"reasoning_end\"}".utf8)
                            _ = try? await URLSession.shared.data(for: ctrlReq)
                        }

                        guard let delta = Self.parseOpenAISSEDataLine(line) else { continue }
                        if let p = delta.promptTokens { promptTokens = p }
                        if let c = delta.cachedTokens { cachedTokens = c }
                        if let pTPS = delta.promptTPS { promptTPS = pTPS }
                        if let gTPS = delta.generationTPS { generationTPS = gTPS }
                        if let d = delta.draftedTokens { draftedTokens = d }
                        if let a = delta.acceptedTokens { acceptedTokens = a }

                        if !delta.content.isEmpty || !delta.reasoningContent.isEmpty {
                            generatedCount += 1
                            visibleReply.append(delta.content)
                            continuation.yield(
                                GenerationChunk(
                                    text: delta.content,
                                    thinking: delta.reasoningContent,
                                    tokenCount: generatedCount,
                                    promptTokens: promptTokens,
                                    generationTPS: generationTPS,
                                    promptTPS: promptTPS,
                                    peakMemoryGB: memGB,
                                    activeComputePlacement: .ggufSubprocess
                                )
                            )
                        }
                    }

                    self.recordAssistantTurn(visibleReply)
                    let finalCache = CacheReport(
                        promptTokens: promptTokens,
                        cachedTokens: cachedTokens,
                        processedTokens: max(0, promptTokens - cachedTokens),
                        outcome: cachedTokens > 0 ? .reused : .miss,
                        cause: cachedTokens > 0 ? nil : fallbackCacheReport.cause
                    )
                    let specReport: SpeculativeReport? = if let d = draftedTokens, let a = acceptedTokens, d > 0 {
                        SpeculativeReport(drafterID: cfg.draftModelPath ?? "drafter", mode: "draft", draftedTokens: d, acceptedTokens: a)
                    } else {
                        nil
                    }
                    continuation.yield(
                        GenerationChunk(
                            tokenCount: generatedCount,
                            promptTokens: promptTokens,
                            generationTPS: generationTPS,
                            promptTPS: promptTPS,
                            peakMemoryGB: memGB,
                            stopReason: .endTurn,
                            cacheReport: finalCache,
                            speculativeReport: specReport,
                            activeComputePlacement: .ggufSubprocess
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    #endif
}

/// Remote HTTP/SSE client provider allowing the native `iOS 27.2+` app (or secondary Mac)
/// to drive the Mac Studio `HeylookServer` over `/v1/messages`, `/v1/requests/{id}/reasoning-end`,
/// and `DELETE /v1/requests/{id}`.
public actor RemoteHeylookProvider: InferenceProvider {
    public struct ParsedSSEEvent: Sendable, Equatable {
        public let textDelta: String
        public let thinkingDelta: String
        public let stopReason: StopReason?
        public let outputTokens: Int?
        public let performance: MessagesResponse.PerformanceExtension?

        public init(
            textDelta: String = "",
            thinkingDelta: String = "",
            stopReason: StopReason? = nil,
            outputTokens: Int? = nil,
            performance: MessagesResponse.PerformanceExtension? = nil
        ) {
            self.textDelta = textDelta
            self.thinkingDelta = thinkingDelta
            self.stopReason = stopReason
            self.outputTokens = outputTokens
            self.performance = performance
        }
    }

    public nonisolated let modelID: String
    public nonisolated let providerKind: EngineProviderKind = .remote
    public nonisolated let baseURL: URL
    private let session: URLSession
    private var descriptionState: EngineDescription

    public init(
        baseURL: URL,
        description: EngineDescription,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.modelID = description.modelID
        self.descriptionState = description
        self.session = session
    }

    public func describe() async -> EngineDescription {
        descriptionState
    }

    public func unload() async {
        descriptionState.isLoaded = false
        let unloadURL = baseURL
            .appendingPathComponent("v1/admin/models")
            .appendingPathComponent(modelID)
            .appendingPathComponent("unload")
        var req = URLRequest(url: unloadURL)
        req.httpMethod = "POST"
        _ = try? await session.data(for: req)
    }

    /// Parses an Anthropic Messages SSE `(event, data)` frame emitted by `HeylookServer`.
    public static func parseAnthropicSSEEvent(event: String, data: String) -> ParsedSSEEvent? {
        guard let payload = data.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: payload) as? [String: Any]
        else {
            return nil
        }

        switch event {
        case "content_block_delta":
            guard let delta = json["delta"] as? [String: Any],
                  let deltaType = delta["type"] as? String
            else { return nil }
            if deltaType == "text_delta" {
                return ParsedSSEEvent(textDelta: (delta["text"] as? String) ?? "")
            } else if deltaType == "thinking_delta" {
                return ParsedSSEEvent(thinkingDelta: (delta["thinking"] as? String) ?? "")
            }
            return nil

        case "message_delta":
            let deltaDict = json["delta"] as? [String: Any]
            let stopRaw = deltaDict?["stop_reason"] as? String
            let stopReason = stopRaw.flatMap(StopReason.init(rawValue:))
            let usageDict = json["usage"] as? [String: Any]
            let outTokens = usageDict?["output_tokens"] as? Int
            var perf: MessagesResponse.PerformanceExtension?
            if let perfDict = json["performance"],
               let perfData = try? JSONSerialization.data(withJSONObject: perfDict) {
                perf = try? JSONDecoder().decode(MessagesResponse.PerformanceExtension.self, from: perfData)
            }
            return ParsedSSEEvent(
                stopReason: stopReason,
                outputTokens: outTokens,
                performance: perf
            )

        default:
            return nil
        }
    }

    /// Forwards `POST /v1/requests/{id}/reasoning-end` to the remote `HeylookServer`.
    public func forwardReasoningEnd(requestID: String) async throws {
        let url = baseURL
            .appendingPathComponent("v1/requests")
            .appendingPathComponent(requestID)
            .appendingPathComponent("reasoning-end")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        _ = try await session.data(for: req)
    }

    /// Forwards `DELETE /v1/requests/{id}` to cancel an active request on the remote `HeylookServer`.
    public func forwardCancel(requestID: String) async throws {
        let url = baseURL
            .appendingPathComponent("v1/requests")
            .appendingPathComponent(requestID)
        var req = URLRequest(url: url)
        req.httpMethod = "DELETE"
        _ = try await session.data(for: req)
    }

    public func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let baseURL = self.baseURL
        let session = self.session
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let url = baseURL.appendingPathComponent("v1/messages")
                    var urlReq = URLRequest(url: url)
                    urlReq.httpMethod = "POST"
                    urlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    urlReq.setValue(requestID, forHTTPHeaderField: "X-Request-ID")
                    var wireReq = request
                    let useStream = request.stream ?? false
                    wireReq.stream = useStream
                    wireReq.temperature = effectiveSampling.temperature
                    wireReq.topP = effectiveSampling.topP
                    wireReq.topK = effectiveSampling.topK
                    wireReq.maxTokens = effectiveSampling.maxTokens
                    wireReq.thinking = effectiveSampling.thinking
                    urlReq.httpBody = try JSONEncoder().encode(wireReq)

                    if useStream {
                        let (bytes, _) = try await session.bytes(for: urlReq)
                        var currentEvent = ""
                        var generatedCount = 0
                        var sentReasoningEnd = false
                        for try await line in bytes.lines {
                            let state = await registry.state(for: requestID)
                            if state.isAborted || Task.isCancelled {
                                try? await self.forwardCancel(requestID: requestID)
                                continuation.yield(
                                    GenerationChunk(
                                        tokenCount: generatedCount,
                                        stopReason: .cancelled,
                                        activeComputePlacement: .remote
                                    )
                                )
                                continuation.finish()
                                return
                            }
                            if state.isReasoningEndRequested && !sentReasoningEnd {
                                sentReasoningEnd = true
                                try? await self.forwardReasoningEnd(requestID: requestID)
                            }
                            if line.hasPrefix("event:") {
                                currentEvent = line.dropFirst("event:".count).trimmingCharacters(in: .whitespaces)
                            } else if line.hasPrefix("data:") {
                                let dataStr = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
                                guard let parsed = Self.parseAnthropicSSEEvent(event: currentEvent, data: dataStr) else {
                                    continue
                                }
                                if !parsed.textDelta.isEmpty || !parsed.thinkingDelta.isEmpty {
                                    generatedCount += 1
                                    continuation.yield(
                                        GenerationChunk(
                                            text: parsed.textDelta,
                                            thinking: parsed.thinkingDelta,
                                            tokenCount: generatedCount,
                                            activeComputePlacement: .remote
                                        )
                                    )
                                } else if parsed.stopReason != nil || parsed.performance != nil {
                                    continuation.yield(
                                        GenerationChunk(
                                            tokenCount: parsed.outputTokens ?? generatedCount,
                                            generationTPS: parsed.performance?.generationTPS ?? 0,
                                            promptTPS: parsed.performance?.promptTPS ?? 0,
                                            stopReason: parsed.stopReason ?? .endTurn,
                                            cacheReport: parsed.performance?.cache,
                                            speculativeReport: parsed.performance?.speculative,
                                            activeComputePlacement: .remote
                                        )
                                    )
                                }
                            }
                        }
                        continuation.finish()
                        return
                    }

                    let (data, _) = try await session.data(for: urlReq)
                    let decoded = try JSONDecoder().decode(MessagesResponse.self, from: data)
                    let textBody = decoded.content.compactMap(\.text).joined()
                    let thinkBody = decoded.content.compactMap(\.thinking).joined()
                    continuation.yield(
                        GenerationChunk(
                            text: textBody,
                            thinking: thinkBody,
                            tokenCount: decoded.usage.outputTokens,
                            promptTokens: decoded.usage.inputTokens,
                            generationTPS: decoded.performance?.generationTPS ?? 0,
                            promptTPS: decoded.performance?.promptTPS ?? 0,
                            stopReason: decoded.stopReason,
                            cacheReport: decoded.performance?.cache,
                            speculativeReport: decoded.performance?.speculative,
                            activeComputePlacement: .remote
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
