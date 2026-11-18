import Accelerate
import CoreGraphics
import Darwin
import Foundation
import ImageIO
import os
import Testing
import UniformTypeIdentifiers
@testable import HeylookCore
@testable import HeylookEmbeddings
@testable import HeylookEngines
@testable import HeylookServer
@testable import HeylookUI

// MARK: - Process-Wide Environment Serialization Lock

/// Coordinates test blocks that explicitly mutate process-wide environment variables
/// (`HEYLOOK_CONFIG`, `HEYLOOK_ADMIN_TOKEN`) so concurrent Swift Testing suites never collide.
actor ProcessConfigGate {
    static let shared = ProcessConfigGate()
    private var isLocked = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func withLock<T: Sendable>(_ operation: @Sendable () async throws -> T) async rethrows -> T {
        await acquire()
        defer { release() }
        return try await operation()
    }

    private func acquire() async {
        if !isLocked {
            isLocked = true
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    private func release() {
        if !waiters.isEmpty {
            let next = waiters.removeFirst()
            next.resume()
        } else {
            isLocked = false
        }
    }
}

// MARK: - Deterministic Test Embedding Engine

/// Deterministic feature-hash embedding engine reserved strictly for automated tests.
/// Computes unit-normalized 768d vectors via character n-gram hashing + payload byte hashing so
/// Matryoshka projections, prompt validation, and retrieval metrics can run without model weights.
final class TestDeterministicEmbeddingEngine: EmbeddingEngine, @unchecked Sendable {
    let modelID: String
    let nativeDimensions: Int
    let supportedDimensions: [Int]
    let supportedModalities: Set<String>
    let knownPrompts: Set<String>
    let maxWindowSeconds: Double
    let simulatedVideoDurationSeconds: Double?
    let perItemDelayNanoseconds: UInt64

    private struct LockedCounters {
        var recordedInputs: [EmbeddingInput] = []
        var unloadCallCount: Int = 0
    }

    private let stateLock = OSAllocatedUnfairLock(initialState: LockedCounters())

    var recordedInputs: [EmbeddingInput] {
        stateLock.withLock { $0.recordedInputs }
    }

    var unloadCallCount: Int {
        stateLock.withLock { $0.unloadCallCount }
    }

    init(
        modelID: String = "embeddinggemma-2-300m-test",
        nativeDimensions: Int = Matryoshka.gemma2NativeDimensions,
        supportedDimensions: [Int] = Matryoshka.gemma2Dimensions,
        supportedModalities: Set<String> = ["text", "image", "audio", "video", "document"],
        knownPrompts: Set<String> = Set(EmbeddingTask.allCases.compactMap(\.promptName)),
        maxWindowSeconds: Double = 32.0,
        simulatedVideoDurationSeconds: Double? = nil,
        perItemDelayNanoseconds: UInt64 = 0
    ) {
        self.modelID = modelID
        self.nativeDimensions = nativeDimensions
        self.supportedDimensions = supportedDimensions
        self.supportedModalities = supportedModalities
        self.knownPrompts = knownPrompts
        self.maxWindowSeconds = maxWindowSeconds
        self.simulatedVideoDurationSeconds = simulatedVideoDurationSeconds
        self.perItemDelayNanoseconds = perItemDelayNanoseconds
    }

    func resetCounters() {
        stateLock.withLock { state in
            state.recordedInputs.removeAll()
            state.unloadCallCount = 0
        }
    }

    func unload() async {
        stateLock.withLock { state in
            state.unloadCallCount += 1
        }
    }

    func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int?
    ) async throws -> [EmbeddedOutput] {
        let targetDim = dimensions ?? nativeDimensions
        guard supportedDimensions.contains(targetDim) else {
            throw EmbeddingValidationError.unsupportedDimensions(
                requested: targetDim,
                supported: supportedDimensions
            )
        }

        stateLock.withLock { state in
            state.recordedInputs.append(contentsOf: inputs)
        }

        var outputs: [EmbeddedOutput] = []
        outputs.reserveCapacity(inputs.count)

        for (index, input) in inputs.enumerated() {
            try Task.checkCancellation()
            if perItemDelayNanoseconds > 0 {
                try await Task.sleep(nanoseconds: perItemDelayNanoseconds)
            }

            let resolvedPrompt = input.promptName
            if input.usesExplicitPromptName,
               let promptName = resolvedPrompt,
               !promptName.isEmpty,
               !knownPrompts.contains(promptName) {
                throw EmbeddingValidationError.unknownPromptName(
                    itemIndex: index,
                    promptName: promptName,
                    available: knownPrompts.sorted()
                )
            }

            if let videoOpts = try input.resolvedVideoWindows(itemIndex: index),
               let secs = videoOpts.seconds,
               secs > maxWindowSeconds {
                throw EmbeddingValidationError.videoWindowsExceedMaxSeconds(
                    itemIndex: index,
                    requested: secs,
                    maxSeconds: maxWindowSeconds
                )
            }

            var accumulated = [Float](repeating: 0, count: nativeDimensions)
            var tokenCount = 0
            var windowResults: [VideoWindowEmbeddingResult] = []

            for part in input.parts {
                switch part {
                case .text(let rawText):
                    let prefix = resolvedPrompt.map { "prompt:\($0)|" } ?? input.task.prefixText
                    let promptText = prefix + (input.title.map { "title:\($0)|" } ?? "") + rawText
                    let words = rawText.split { $0.isWhitespace || $0.isPunctuation }
                    tokenCount += max(1, words.count)
                    Self.accumulateTextFeatures(promptText, into: &accumulated)
                case let .media(kind, mimeType, data, url, sampling):
                    guard supportedModalities.contains(kind.rawValue) else {
                        throw EmbeddingValidationError.unsupportedModality(
                            itemIndex: index,
                            modality: kind.rawValue,
                            supported: supportedModalities.sorted()
                        )
                    }
                    let payload = data ?? (url.flatMap { try? Data(contentsOf: $0) }) ?? Data()
                    Self.accumulateBytesFeatures(
                        prefix: "\(kind.rawValue):\(mimeType):",
                        bytes: payload,
                        into: &accumulated
                    )
                    tokenCount += max(8, min(512, payload.count / 128 + 16))

                    if kind == .video,
                       let videoOpts = try input.resolvedVideoWindows(itemIndex: index),
                       let duration = simulatedVideoDurationSeconds,
                       duration > 0 {
                        let winSec = videoOpts.seconds ?? maxWindowSeconds
                        let ranges = WindowPlanner.consecutiveRanges(duration: duration, windowSeconds: winSec)
                        for (wIdx, range) in ranges.enumerated() {
                            var winVec = accumulated
                            Self.accumulateTextFeatures("win:\(wIdx):\(range.start):\(range.end)", into: &winVec)
                            let midTime = ((range.start + range.end) * 0.5 * 1000).rounded() / 1000
                            let sampleTime = sampling == .content ? midTime : range.start
                            windowResults.append(
                                VideoWindowEmbeddingResult(
                                    start: range.start,
                                    end: range.end,
                                    frameTimes: [sampleTime],
                                    vector: Matryoshka.project(winVec, dimensions: targetDim)
                                )
                            )
                        }
                    }
                }
            }

            let projected = Matryoshka.project(accumulated, dimensions: targetDim)
            outputs.append(
                EmbeddedOutput(
                    vector: projected,
                    windows: windowResults,
                    inputTokens: tokenCount,
                    promptName: resolvedPrompt
                )
            )
        }
        return outputs
    }

    func embed(
        text: String,
        task: EmbeddingTask = .document,
        title: String? = nil,
        dimensions: Int? = nil
    ) -> [Float] {
        var acc = [Float](repeating: 0, count: nativeDimensions)
        let prompt = task == .document
            ? EmbeddingTask.documentPrompt(title: title, text: text)
            : task.prefixText + text
        Self.accumulateTextFeatures(prompt, into: &acc)
        return Matryoshka.project(acc, dimensions: dimensions ?? nativeDimensions)
    }

    private static func accumulateTextFeatures(_ text: String, into buffer: inout [Float]) {
        let normalized = text.lowercased()
        let bytes = Array(normalized.utf8)
        guard !bytes.isEmpty else { return }
        for n in 2...4 where bytes.count >= n {
            for start in 0...(bytes.count - n) {
                var hash: UInt64 = 14_695_981_039_346_656_037
                for offset in 0..<n {
                    hash ^= UInt64(bytes[start + offset])
                    hash &*= 1_099_511_628_211
                }
                let bucket = Int(hash % UInt64(buffer.count))
                let sign: Float = ((hash >> 63) == 0) ? 1.0 : -1.0
                buffer[bucket] += sign
            }
        }
        for word in normalized.split(whereSeparator: { $0.isWhitespace || $0.isPunctuation }) {
            let wordBytes = Array(word.utf8)
            var hash: UInt64 = 14_695_981_039_346_656_037
            for b in wordBytes {
                hash ^= UInt64(b)
                hash &*= 1_099_511_628_211
            }
            let bucket = Int(hash % UInt64(buffer.count))
            let sign: Float = ((hash >> 62) & 1) == 0 ? 2.5 : -2.5
            buffer[bucket] += sign
        }
    }

    private static func accumulateBytesFeatures(prefix: String, bytes: Data, into buffer: inout [Float]) {
        accumulateTextFeatures(prefix, into: &buffer)
        let strideStep = max(1, bytes.count / 256)
        var i = 0
        while i < bytes.count {
            var hash: UInt64 = 14_695_981_039_346_656_037 ^ UInt64(i)
            hash &*= 1_099_511_628_211
            hash ^= UInt64(bytes[i])
            hash &*= 1_099_511_628_211
            let bucket = Int(hash % UInt64(buffer.count))
            let sign: Float = ((hash >> 61) & 1) == 0 ? 1.0 : -1.0
            buffer[bucket] += sign
            i += strideStep
        }
    }
}

// MARK: - Configurable Test Stub Inference Provider

/// Configurable `InferenceProvider` stub used strictly in `Tests/HeylookTests` to exercise real
/// `ComputeScheduler`, `WorkbenchService`, `ConversationOrchestrator`, and HTTP/SSE wiring without
/// production fallback shims.
actor TestStubInferenceProvider: InferenceProvider {
    struct RecordedInvocation: Sendable {
        let requestID: String
        let request: MessagesRequest
        let sampling: SamplerSettings
    }

    nonisolated let modelID: String
    nonisolated let providerKind: EngineProviderKind

    private var currentDescription: EngineDescription
    private var scriptedChunks: [GenerationChunk]
    private var prefillSteps: [(processed: Int, total: Int)]
    private var chunkDelayNanoseconds: UInt64
    private var failBeforeStreamError: Error?
    private var failMidStreamAfterCount: Int?
    private var failMidStreamError: Error?
    private var observedState: ObservedContractState
    private var customImagePlan: ImagePlanResult?
    private var customPromptPreview: PromptPreviewResult?
    private var warmUpError: Error?
    private var warmUpDelayNanoseconds: UInt64 = 0
    private var streamFactory: (@Sendable (String, MessagesRequest, SamplerSettings, RequestRegistry) async -> AsyncThrowingStream<GenerationChunk, Error>)?

    private(set) var invocations: [RecordedInvocation] = []
    private(set) var preloadCount: Int = 0
    private(set) var unloadCount: Int = 0
    private(set) var clearCacheCount: Int = 0
    private(set) var warmUpCount: Int = 0

    init(
        description: EngineDescription,
        chunks: [GenerationChunk] = [
            GenerationChunk(text: "hello", tokenCount: 1, contentTokens: 1, promptTokens: 4, generationTPS: 42.0, promptTPS: 120.0, ttftMs: 12.0, stopReason: .endTurn)
        ],
        prefillSteps: [(processed: Int, total: Int)] = [],
        chunkDelayNanoseconds: UInt64 = 0,
        failBeforeStreamError: Error? = nil,
        failMidStreamAfterCount: Int? = nil,
        failMidStreamError: Error? = nil,
        observedState: ObservedContractState = ObservedContractState(),
        imagePlan: ImagePlanResult? = nil,
        promptPreview: PromptPreviewResult? = nil,
        streamFactory: (@Sendable (String, MessagesRequest, SamplerSettings, RequestRegistry) async -> AsyncThrowingStream<GenerationChunk, Error>)? = nil
    ) {
        self.modelID = description.modelID
        self.providerKind = description.provider
        self.currentDescription = description
        self.scriptedChunks = chunks
        self.prefillSteps = prefillSteps
        self.chunkDelayNanoseconds = chunkDelayNanoseconds
        self.failBeforeStreamError = failBeforeStreamError
        self.failMidStreamAfterCount = failMidStreamAfterCount
        self.failMidStreamError = failMidStreamError
        self.observedState = observedState
        self.customImagePlan = imagePlan
        self.customPromptPreview = promptPreview
        self.streamFactory = streamFactory
    }

    func updateChunks(_ chunks: [GenerationChunk]) {
        self.scriptedChunks = chunks
    }

    func setMidStreamFailure(after count: Int?, error: Error?) {
        self.failMidStreamAfterCount = count
        self.failMidStreamError = error
    }

    func setBeforeStreamFailure(_ error: Error?) {
        self.failBeforeStreamError = error
    }

    func setObservedState(_ state: ObservedContractState) {
        self.observedState = state
    }

    func setImagePlan(_ plan: ImagePlanResult?) {
        self.customImagePlan = plan
    }

    func setLoaded(_ loaded: Bool) {
        let d = currentDescription
        self.currentDescription = EngineDescription(
            modelID: d.modelID,
            provider: d.provider,
            modelPath: d.modelPath,
            architecture: d.architecture,
            quantization: d.quantization,
            computePlacement: d.computePlacement,
            isPinned: d.isPinned,
            isLoaded: loaded,
            contextSize: d.contextSize,
            flashAttention: d.flashAttention,
            supportsVision: d.supportsVision,
            supportsAudio: d.supportsAudio,
            supportsEmbeddings: d.supportsEmbeddings,
            supportsThinking: d.supportsThinking,
            thinkingBudget: d.thinkingBudget,
            reasoningFormat: d.reasoningFormat,
            thinkingControls: d.thinkingControls,
            templateSource: d.templateSource,
            speculativeDrafter: d.speculativeDrafter,
            estimatedMemoryMB: d.estimatedMemoryMB,
            tags: d.tags
        )
    }

    func describe() async -> EngineDescription {
        currentDescription
    }

    func preload() async throws {
        preloadCount += 1
        setLoaded(true)
    }

    func clearCache() async -> Bool {
        clearCacheCount += 1
        return true
    }

    func unload() async {
        unloadCount += 1
        setLoaded(false)
    }

    func observedContractState() async -> ObservedContractState {
        observedState
    }

    func planImages(sizes: [[Int]]) async throws -> ImagePlanResult {
        try ModelSpecializationRegistry.validateImagePlanSizes(sizes)
        if let customImagePlan {
            return customImagePlan
        }
        guard currentDescription.supportsVision.value else {
            throw ProviderExecutionError.invalidRequest(
                detail: "Model '\(modelID)' does not support image input."
            )
        }
        let rows = sizes.map { pair in
            ImagePlanRow(size: pair, tokens: 256, target: [48 * 10, 48 * 10])
        }
        return ImagePlanResult(engine: providerKind.rawValue, images: rows, source: "test_stub")
    }

    func renderPromptPreview(request: MessagesRequest) async throws -> PromptPreviewResult {
        if let customPromptPreview {
            return customPromptPreview
        }
        return try defaultRenderPromptPreview(request: request)
    }

    private func defaultRenderPromptPreview(request: MessagesRequest) throws -> PromptPreviewResult {
        let desc = currentDescription
        let modelURL = URL(fileURLWithPath: desc.modelPath)
        let modelDir = HeylookConfig.resolveModelDirectoryURL(from: modelURL.path)
        let enableThinking = ModelSpecializationRegistry.resolveThinkingEnabled(
            requestThinking: request.thinking,
            sidecarThinking: nil,
            modelCapable: desc.supportsThinking.value
        )
        let rendered = try HeylookConfig.renderPromptDetailed(
            in: FileManager.default.fileExists(atPath: modelDir.path) ? modelDir : nil,
            system: request.system,
            turns: request.messages,
            thinkingEnabled: enableThinking,
            thinkingDepth: request.thinkingDepth
        )
        return PromptPreviewResult(
            prompt: rendered.prompt,
            tokenCount: nil,
            tokens: nil,
            templateSource: rendered.source,
            chatTemplateCaps: nil
        )
    }

    func warmUp1Token() async throws {
        warmUpCount += 1
        if warmUpDelayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: warmUpDelayNanoseconds)
        }
        if let warmUpError {
            throw warmUpError
        }
    }

    func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) async -> AsyncThrowingStream<GenerationChunk, Error> {
        invocations.append(RecordedInvocation(requestID: requestID, request: request, sampling: effectiveSampling))

        if let streamFactory {
            return await streamFactory(requestID, request, effectiveSampling, registry)
        }

        let chunks = scriptedChunks
        let prefill = prefillSteps
        let delay = chunkDelayNanoseconds
        let beforeErr = failBeforeStreamError
        let midAfter = failMidStreamAfterCount
        let midErr = failMidStreamError
        let progressCallback = PrefillProgressBridge.current

        return AsyncThrowingStream { continuation in
            let task = Task {
                if let beforeErr {
                    continuation.finish(throwing: beforeErr)
                    return
                }
                for step in prefill {
                    if registry.state(for: requestID).isAborted || Task.isCancelled {
                        continuation.finish()
                        return
                    }
                    await progressCallback?(step.processed, step.total)
                }
                for (index, chunk) in chunks.enumerated() {
                    if registry.state(for: requestID).isAborted || Task.isCancelled {
                        continuation.finish()
                        return
                    }
                    if delay > 0 {
                        try? await Task.sleep(nanoseconds: delay)
                    }
                    if registry.state(for: requestID).isAborted || Task.isCancelled {
                        continuation.finish()
                        return
                    }
                    if let midAfter, index >= midAfter, let midErr {
                        continuation.finish(throwing: midErr)
                        return
                    }
                    continuation.yield(chunk)
                }
                if let midAfter, chunks.count <= midAfter, let midErr {
                    continuation.finish(throwing: midErr)
                    return
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}

// MARK: - Isolated Temporary Sandbox & Router Test Harness

final class TestTempSandbox: @unchecked Sendable {
    let rootURL: URL
    let dbURL: URL
    let vectorDBURL: URL
    let modelsDir: URL
    let configURL: URL

    init(prefix: String = "heylook_test") throws {
        let base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("\(prefix)_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.rootURL = base
        self.dbURL = base.appendingPathComponent("workbench.sqlite")
        self.vectorDBURL = base.appendingPathComponent("vectors.sqlite")
        self.modelsDir = base.appendingPathComponent("models", isDirectory: true)
        self.configURL = base.appendingPathComponent("heylook.toml")
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
    }

    func cleanup() {
        try? FileManager.default.removeItem(at: rootURL)
    }
}

final class TestRouterHarness: @unchecked Sendable {
    let sandbox: TestTempSandbox
    let store: SQLiteStore
    let vectorStore: VectorStore
    let scheduler: ComputeScheduler
    let registry: RequestRegistry
    let router: HeylookRouter

    init(
        maxQueueDepth: Int = 8,
        maxLoadedModels: Int = 3,
        adminToken: String? = nil,
        isReadOnlyConfig: Bool = false,
        allowedHosts: [String] = [],
        bindHost: String = "127.0.0.1",
        embeddingEngine: (any EmbeddingEngine)? = nil
    ) throws {
        self.sandbox = try TestTempSandbox(prefix: "heylook_router")
        self.store = try SQLiteStore(databaseURL: sandbox.dbURL)
        self.vectorStore = try VectorStore(databaseURL: sandbox.vectorDBURL)
        self.scheduler = ComputeScheduler(
            maxLoadedGPUModels: maxLoadedModels,
            maxQueueDepth: maxQueueDepth
        )
        self.registry = RequestRegistry()
        self.router = HeylookRouter(
            store: store,
            scheduler: scheduler,
            registry: registry,
            embeddingEngine: embeddingEngine ?? TestDeterministicEmbeddingEngine(),
            bindHost: bindHost,
            allowedHosts: allowedHosts,
            adminToken: adminToken,
            isReadOnlyConfig: isReadOnlyConfig
        )
    }

    deinit {
        sandbox.cleanup()
    }

    func registerStub(
        id: String,
        provider: EngineProviderKind = .mlx,
        modelPath: String? = nil,
        supportsVision: Bool = false,
        supportsAudio: Bool = false,
        supportsEmbeddings: Bool = false,
        supportsThinking: Bool = false,
        thinkingControls: ThinkingControls? = nil,
        speculativeDrafter: String? = nil,
        isLoaded: Bool = true,
        chunks: [GenerationChunk] = [
            GenerationChunk(text: "ok", tokenCount: 1, contentTokens: 1, promptTokens: 4, generationTPS: 25.0, promptTPS: 80.0, ttftMs: 10.0, stopReason: .endTurn)
        ],
        prefillSteps: [(processed: Int, total: Int)] = [],
        chunkDelayNanoseconds: UInt64 = 0,
        failMidStreamAfterCount: Int? = nil,
        failMidStreamError: Error? = nil
    ) async -> TestStubInferenceProvider {
        let drafterField: ProvenanceField<String?> = if let speculativeDrafter {
            ProvenanceField(value: speculativeDrafter, provenance: .derived, effect: .reload)
        } else {
            ProvenanceField(value: nil, provenance: .notApplicable, effect: .reload)
        }
        let placement: ComputePlacement = provider == .gguf ? .ggufSubprocess : .mlxGPU
        let desc = EngineDescription(
            modelID: id,
            provider: provider,
            modelPath: modelPath ?? sandbox.modelsDir.appendingPathComponent(id).path,
            computePlacement: ProvenanceField(
                value: placement,
                provenance: .derived,
                effect: .reload,
                computePlacement: placement
            ),
            isLoaded: isLoaded,
            contextSize: ProvenanceField(value: 8192, provenance: .derived, effect: .reload),
            supportsVision: ProvenanceField(value: supportsVision, provenance: .derived, effect: .reload),
            supportsAudio: ProvenanceField(value: supportsAudio, provenance: .derived, effect: .reload),
            supportsEmbeddings: ProvenanceField(value: supportsEmbeddings, provenance: .derived, effect: .reload),
            supportsThinking: ProvenanceField(value: supportsThinking, provenance: .derived, effect: .perRequest),
            thinkingControls: thinkingControls,
            speculativeDrafter: drafterField
        )
        let stub = TestStubInferenceProvider(
            description: desc,
            chunks: chunks,
            prefillSteps: prefillSteps,
            chunkDelayNanoseconds: chunkDelayNanoseconds,
            failMidStreamAfterCount: failMidStreamAfterCount,
            failMidStreamError: failMidStreamError
        )
        await scheduler.register(provider: stub, description: desc)
        return stub
    }

    func send(
        _ method: String,
        _ path: String,
        queryItems: [URLQueryItem] = [],
        headers: [String: String] = [:],
        body: Data = Data(),
        onSSEFrame: (@Sendable (String) async -> Void)? = nil
    ) async -> HeylookHTTPResponse {
        await router.handle(
            method: method,
            path: path,
            queryItems: queryItems,
            headers: headers,
            body: body,
            onSSEFrame: onSSEFrame
        )
    }

    func sendJSON(
        _ method: String,
        _ path: String,
        queryItems: [URLQueryItem] = [],
        headers: [String: String] = [:],
        object: Any,
        onSSEFrame: (@Sendable (String) async -> Void)? = nil
    ) async throws -> HeylookHTTPResponse {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        var merged = headers
        if merged["Content-Type"] == nil {
            merged["Content-Type"] = "application/json"
        }
        return await router.handle(
            method: method,
            path: path,
            queryItems: queryItems,
            headers: merged,
            body: data,
            onSSEFrame: onSSEFrame
        )
    }

    static func withEnvironmentOverride(
        _ key: String,
        value: String?,
        operation: @Sendable () async throws -> Void
    ) async rethrows {
        try await ProcessConfigGate.shared.withLock {
            let prior = ProcessInfo.processInfo.environment[key]
            if let value {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
            defer {
                if let prior {
                    setenv(key, prior, 1)
                } else {
                    unsetenv(key)
                }
            }
            try await operation()
        }
    }
}

// MARK: - Media, Wire & Port Test Fixtures

enum TestFixtures {
    struct ParsedSSEFrame: Equatable, Sendable {
        let event: String?
        let id: String?
        let data: String

        func jsonObject() throws -> [String: Any]? {
            guard !data.isEmpty, let d = data.data(using: .utf8) else { return nil }
            return try JSONSerialization.jsonObject(with: d) as? [String: Any]
        }
    }

    static func decodeJSONObject(_ data: Data) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    static func decodeJSONArray(_ data: Data) throws -> [[String: Any]] {
        try #require(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    }

    static func parseSSEFrames(_ rawText: String) -> [ParsedSSEFrame] {
        let normalized = rawText.replacingOccurrences(of: "\r\n", with: "\n")
        let blocks = normalized.components(separatedBy: "\n\n")
        var frames: [ParsedSSEFrame] = []
        for block in blocks {
            let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            var eventName: String?
            var eventID: String?
            var dataLines: [String] = []
            for line in trimmed.components(separatedBy: "\n") {
                if line.hasPrefix("event:") {
                    eventName = String(line.dropFirst(6)).trimmingCharacters(in: .whitespaces)
                } else if line.hasPrefix("id:") {
                    eventID = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                } else if line.hasPrefix("data:") {
                    dataLines.append(String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces))
                }
            }
            if eventName != nil || eventID != nil || !dataLines.isEmpty {
                frames.append(
                    ParsedSSEFrame(
                        event: eventName,
                        id: eventID,
                        data: dataLines.joined(separator: "\n")
                    )
                )
            }
        }
        return frames
    }

    static func makeMinimalPNGData(
        width: Int = 16,
        height: Int = 16,
        red: UInt8 = 220,
        green: UInt8 = 64,
        blue: UInt8 = 48
    ) throws -> Data {
        let bytesPerRow = width * 4
        var rgba = [UInt8](repeating: 255, count: height * bytesPerRow)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                rgba[offset] = red
                rgba[offset + 1] = green
                rgba[offset + 2] = blue
                rgba[offset + 3] = 255
            }
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ),
        let cgImage = ctx.makeImage() else {
            throw NSError(domain: "TestFixtures", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to build CGImage"])
        }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else {
            throw NSError(domain: "TestFixtures", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to build PNG destination"])
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw NSError(domain: "TestFixtures", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to finalize PNG"])
        }
        return out as Data
    }

    static func makeWAVData(
        sampleRate: UInt32 = 48_000,
        channels: UInt16 = 2,
        durationSeconds: Double = 0.25,
        fllrPaddingBytes: Int = 0,
        frequencyHz: Double = 440.0
    ) -> Data {
        let frameCount = max(1, Int((Double(sampleRate) * durationSeconds).rounded()))
        var pcm = Data(capacity: frameCount * Int(channels) * 2)
        for frame in 0..<frameCount {
            let t = Double(frame) / Double(sampleRate)
            let sample = Int16((sin(2.0 * .pi * frequencyHz * t) * 14_000.0).rounded()).littleEndian
            for _ in 0..<Int(channels) {
                withUnsafeBytes(of: sample) { pcm.append(contentsOf: $0) }
            }
        }

        let fmtSize: UInt32 = 16
        let fllrSize = UInt32(max(0, fllrPaddingBytes))
        let dataSize = UInt32(pcm.count)
        let fllrOverhead: UInt32 = fllrPaddingBytes > 0 ? (8 + fllrSize) : 0
        let riffBodySize: UInt32 = 4 + (8 + fmtSize) + fllrOverhead + (8 + dataSize)

        var wav = Data("RIFF".utf8)
        var riffLE = riffBodySize.littleEndian
        withUnsafeBytes(of: &riffLE) { wav.append(contentsOf: $0) }
        wav.append(Data("WAVE".utf8))

        wav.append(Data("fmt ".utf8))
        var fmtLenLE = fmtSize.littleEndian
        var audioFmtLE = UInt16(1).littleEndian
        var chanLE = channels.littleEndian
        var srLE = sampleRate.littleEndian
        var byteRateLE = (sampleRate * UInt32(channels) * 2).littleEndian
        var blockAlignLE = (channels * 2).littleEndian
        var bpsLE = UInt16(16).littleEndian
        withUnsafeBytes(of: &fmtLenLE) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &audioFmtLE) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &chanLE) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &srLE) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &byteRateLE) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &blockAlignLE) { wav.append(contentsOf: $0) }
        withUnsafeBytes(of: &bpsLE) { wav.append(contentsOf: $0) }

        if fllrPaddingBytes > 0 {
            wav.append(Data("FLLR".utf8))
            var fllrLenLE = fllrSize.littleEndian
            withUnsafeBytes(of: &fllrLenLE) { wav.append(contentsOf: $0) }
            wav.append(Data(repeating: 0, count: fllrPaddingBytes))
        }

        wav.append(Data("data".utf8))
        var dataLenLE = dataSize.littleEndian
        withUnsafeBytes(of: &dataLenLE) { wav.append(contentsOf: $0) }
        wav.append(pcm)
        return wav
    }

    static func acquireFreeLoopbackPort() -> UInt16? {
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        defer { Darwin.close(fd) }
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr.s_addr = CFSwapInt32HostToBig(INADDR_LOOPBACK)
        let rc = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.bind(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard rc == 0 else { return nil }
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let gn = withUnsafeMutablePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.getsockname(fd, sockPtr, &len)
            }
        }
        guard gn == 0 else { return nil }
        return UInt16(bigEndian: addr.sin_port)
    }
}

actor SSEChunkCollector {
    private(set) var frames: [String] = []
    func append(_ frame: String) {
        frames.append(frame)
    }
    func joined() -> String {
        frames.joined()
    }
}
