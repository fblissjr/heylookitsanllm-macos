import CoreGraphics
import Foundation
import ImageIO
import Observation
import UniformTypeIdentifiers
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

// MARK: - Rich Multimodal Compare & Embedding Lab Types

public enum CompareBlockKind: String, Sendable, Equatable, Codable {
    case text
    case media
}

public enum CompareVideoWindowMode: String, CaseIterable, Identifiable, Sendable, Equatable, Codable {
    case wholeVideo = "whole"
    case fps = "fps"
    case content = "content"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .wholeVideo: return "Whole video"
        case .fps: return "Windows (uniform FPS)"
        case .content: return "Windows (scene change)"
        }
    }

    public var sampling: VideoWindowSampling {
        switch self {
        case .wholeVideo: return .off
        case .fps: return .fps
        case .content: return .content
        }
    }
}

public struct CompareBlock: Identifiable, Sendable, Equatable {
    public let id: String
    public var kind: CompareBlockKind
    public var text: String
    public var fileURL: URL?
    public var mediaKind: ArtifactKind?
    public var videoWindowMode: CompareVideoWindowMode
    public var videoWindowSecondsText: String

    public init(
        id: String = UUID().uuidString,
        kind: CompareBlockKind,
        text: String = "",
        fileURL: URL? = nil,
        mediaKind: ArtifactKind? = nil,
        videoWindowMode: CompareVideoWindowMode = .wholeVideo,
        videoWindowSecondsText: String = ""
    ) {
        self.id = id
        self.kind = kind
        self.text = text
        self.fileURL = fileURL
        self.mediaKind = mediaKind
        self.videoWindowMode = videoWindowMode
        self.videoWindowSecondsText = videoWindowSecondsText
    }

    public var parsedWindowSeconds: Double? {
        let trimmed = videoWindowSecondsText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let v = Double(trimmed), v.isFinite, v > 0 else { return nil }
        return v
    }

    public var videoWindowOptions: VideoWindowOptions? {
        guard mediaKind == .video, videoWindowMode != .wholeVideo else { return nil }
        return VideoWindowOptions(seconds: parsedWindowSeconds, sampling: videoWindowMode.sampling)
    }

    public var displayTitle: String {
        switch kind {
        case .text:
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? "(empty text block)" : trimmed
        case .media:
            return fileURL?.lastPathComponent ?? "(select media)"
        }
    }
}

public struct CompareRichItem: Identifiable, Sendable, Equatable {
    public let id: String
    public var label: String
    public var blocks: [CompareBlock]
    public var promptName: String?

    public init(
        id: String = UUID().uuidString,
        label: String,
        blocks: [CompareBlock],
        promptName: String? = nil
    ) {
        self.id = id
        self.label = label
        self.blocks = blocks
        self.promptName = promptName
    }

    public var textSummary: String {
        blocks.compactMap { block -> String? in
            guard block.kind == .text else { return nil }
            let t = block.text.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? nil : t
        }.joined(separator: " ")
    }

    public var primaryFileURL: URL? {
        blocks.first(where: { $0.kind == .media })?.fileURL
    }

    public var primaryArtifactKind: ArtifactKind {
        blocks.first(where: { $0.kind == .media })?.mediaKind ?? .text
    }

    public var asMatrixItem: CompareMatrixItem {
        CompareMatrixItem(
            id: id,
            label: label,
            kind: primaryArtifactKind,
            textContent: textSummary,
            fileURL: primaryFileURL
        )
    }
}

public struct CompareVideoWindowHit: Identifiable, Sendable, Equatable {
    public let id: String
    public let itemIndex: Int
    public let itemLabel: String
    public let fileURL: URL?
    public let windowIndex: Int
    public let start: Double
    public let end: Double
    public let frameTimes: [Double]
    public let score: Float
}

public struct CompareVectorInspection: Sendable, Equatable {
    public let label: String
    public let dim: Int
    public let l2Norm: Float
    public let first16: [Float]
}

public struct EmbeddingModelCatalogEntry: Identifiable, Sendable, Equatable {
    public let id: String
    public let label: String
    public let facts: EngineEmbeddingFacts
    public let provider: EngineProviderKind
    public let modelPath: String?
}

public struct LabItemEntry: Identifiable, Sendable, Equatable, Codable {
    public let id: String
    public let text: String?
    public let filePath: String?
    public let displayName: String?
    public let windows: CompareVideoWindowMode?
    public let windowSeconds: Double?

    public init(
        id: String = UUID().uuidString,
        text: String? = nil,
        filePath: String? = nil,
        displayName: String? = nil,
        windows: CompareVideoWindowMode? = nil,
        windowSeconds: Double? = nil
    ) {
        self.id = id
        self.text = text
        self.filePath = filePath
        self.displayName = displayName
        self.windows = windows
        self.windowSeconds = windowSeconds
    }

    public var summaryLabel: String {
        if let displayName, !displayName.isEmpty {
            return displayName
        }
        if let filePath, !filePath.isEmpty {
            return URL(fileURLWithPath: filePath).lastPathComponent
        }
        if let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(64))
        }
        return id
    }

    public var cacheSignature: String {
        "\(text ?? "")|\(filePath ?? "")|\(windows?.rawValue ?? "")|\(windowSeconds.map { String($0) } ?? "")"
    }
}

public struct EmbeddingsLabCase: Identifiable, Sendable, Equatable, Codable {
    public let id: String
    public let query: LabItemEntry
    public let positives: [LabItemEntry]
    public let negatives: [LabItemEntry]

    public init(id: String, query: LabItemEntry, positives: [LabItemEntry], negatives: [LabItemEntry]) {
        self.id = id
        self.query = query
        self.positives = positives
        self.negatives = negatives
    }
}

public struct EmbeddingsLabSet: Sendable, Equatable {
    public var name: String
    public var baseDirectoryURL: URL?
    public var cases: [EmbeddingsLabCase]

    public init(name: String, baseDirectoryURL: URL? = nil, cases: [EmbeddingsLabCase]) {
        self.name = name
        self.baseDirectoryURL = baseDirectoryURL
        self.cases = cases
    }
}

public struct EmbeddingsLabConfigSpec: Identifiable, Sendable, Equatable {
    public let id: String
    public var label: String
    public var modelID: String
    public var dimensions: Int
    public var queryPromptName: String?
    public var docPromptName: String?
    public var computePreset: ComputePreset?

    public init(
        id: String = UUID().uuidString,
        label: String,
        modelID: String,
        dimensions: Int,
        queryPromptName: String?,
        docPromptName: String?,
        computePreset: ComputePreset? = nil
    ) {
        self.id = id
        self.label = label
        self.modelID = modelID
        self.dimensions = dimensions
        self.queryPromptName = queryPromptName
        self.docPromptName = docPromptName
        self.computePreset = computePreset
    }
}

public struct EmbeddingsLabCaseResult: Identifiable, Sendable, Equatable {
    public let id: String
    public let querySummary: String
    public let bestPositiveScore: Float
    public let bestNegativeScore: Float
    public let margin: Float
    public let rank: Int
    public let hitAt1: Bool
    public let hitAt3: Bool
    public let reciprocalRank: Double
    public let beatenByLabel: String?
}

public struct EmbeddingsLabRunResult: Identifiable, Sendable, Equatable {
    public let id: String
    public let config: EmbeddingsLabConfigSpec
    public let recallAt1: Double
    public let recallAt3: Double
    public let mrr: Double
    public let meanMargin: Double
    public let worstMargin: Double
    public let elapsedMs: Double
    public let caseResults: [EmbeddingsLabCaseResult]
    public let placementLabel: String?
}

/// Domain controller for Multimodal Embeddings & ANE Media Finder Studio (`VectorStore`,
/// `IndexCoordinator`, `FolderWatcher`, multimodal Compare Matrix, ANE Tower Inspector,
/// and multi-config multimodal Embedding Lab).
///
/// Enforces Truthful Runtime: never silently substitutes synthetic hash vectors for real model output.
@MainActor
@Observable
public final class EmbeddingsStudioController {
    public static let maxCompareMatrixItems: Int = EmbeddingRequestLimits.maxItems

    public let vectorStore: VectorStore
    public let router: HeylookRouter

    public var embeddingsSubMode: EmbeddingsSubMode = .compareMatrix
    public var selectedBackend: EngineBackend = .nativeMLX
    public var modelComputeOptions: ModelComputeOptions = .hybridANETowers
    public var embeddingServerURLString: String = "http://127.0.0.1:8000"
    public var checkpointPath: String = NativeMLXEmbeddingGemma2.defaultCheckpointURL.path
    public var serverCapabilities: HeylookServerCapabilities?

    public private(set) var availableEmbeddingModels: [EmbeddingModelCatalogEntry] = []
    public var selectedEmbeddingModelID: String = ""

    public var embeddingQuery: String = ""
    public var queryAttachmentURL: URL?
    public var selectedDimension: Int = Matryoshka.gemma2NativeDimensions
    public var selectedSampling: VideoWindowSampling = .fps
    public var loadAudioFromVideo: Bool = false
    public var selectedKindFilter: ArtifactKind?
    public var selectedTaskOverride: EmbeddingTask?
    public var searchHits: [SearchHit] = []
    public var selectedHit: SearchHit?
    public var selectedSpanIndex: Int = 0
    public var embeddingStatusMessage: String = "Ready"
    public var isEmbeddingBusy: Bool = false
    public var isIndexing: Bool = false
    public var artifactCount: Int = 0
    public var chunkCount: Int = 0
    public var countsByKind: [ArtifactKind: Int] = [:]
    public var watchedFolders: [VectorStore.WatchedFolder] = []
    public var showEvalLabDrawer: Bool = false

    public var compareLeftText: String = "Swift 6.4 actor isolation for unified memory inference on Apple Silicon"
    public var compareRightText: String = "Running MLX and CoreML ANE vision towers concurrently on Mac Studio"
    public var compareCosineScore: Float?
    public var compareRichItems: [CompareRichItem] = [
        CompareRichItem(
            id: "compare-item-1",
            label: "Item 1",
            blocks: [CompareBlock(kind: .text, text: "Swift 6.4 actor isolation for unified memory inference on Apple Silicon")]
        ),
        CompareRichItem(
            id: "compare-item-2",
            label: "Item 2",
            blocks: [CompareBlock(kind: .text, text: "Running MLX and CoreML ANE vision towers concurrently on Mac Studio")]
        )
    ] {
        didSet {
            syncCompareItemsMirrorFromRich()
        }
    }
    public var compareItems: [CompareMatrixItem] = [
        CompareMatrixItem(
            id: "compare-item-1",
            label: "Item 1",
            kind: .text,
            textContent: "Swift 6.4 actor isolation for unified memory inference on Apple Silicon"
        ),
        CompareMatrixItem(
            id: "compare-item-2",
            label: "Item 2",
            kind: .text,
            textContent: "Running MLX and CoreML ANE vision towers concurrently on Mac Studio"
        )
    ]
    public var compareAsymmetricMode: Bool = false
    public var compareMatrixDimension: Int = Matryoshka.gemma2NativeDimensions {
        didSet {
            recomputeCompareMatrixScores()
            recomputeCompareRankings()
        }
    }
    public private(set) var compareRowVectors768: [[Float]] = []
    public private(set) var compareColVectors768: [[Float]] = []
    public private(set) var compareItemOutputs: [EmbeddedOutput] = []
    public private(set) var compareMatrixScores: [[Float]] = []
    public private(set) var compareRankedMatchesToItem1: [(index: Int, label: String, score: Float)] = []
    public private(set) var compareVideoWindowHits: [CompareVideoWindowHit] = []
    public var comparePreflightError: String?
    public var activeComparePreviewWindow: CompareVideoWindowHit?
    public var selectedCompareCellRow: Int = 0
    public var selectedCompareCellCol: Int = 1

    public var aneInspectionReports: [ANETowerInspector.ComponentComputeReport] = []
    public var benchmarkCases: [EmbeddingLabRunner.BenchmarkCase] = [
        .init(query: "Apple Neural Engine co-residency", expectedArtifactID: "doc-vision"),
        .init(query: "evict 70B LLM background indexing", expectedArtifactID: "doc-scheduler"),
        .init(query: "Matryoshka 768d to 128d truncation", expectedArtifactID: "doc-towers"),
        .init(query: "heylook_saved SQLite WAL mobile lock screen", expectedArtifactID: "doc-sse")
    ]
    public var labMetrics: [EmbeddingLabRunner.DimensionMetrics] = []
    public private(set) var labConfigRuns: [EmbeddingLabConfigRun] = []
    public var labSet: EmbeddingsLabSet = EmbeddingsLabSet(name: "Built-in Multimodal Sample (8 cases)", cases: [])
    public var labConfigQueue: [EmbeddingsLabConfigSpec] = []
    public private(set) var labSweepResults: [EmbeddingsLabRunResult] = []
    public var labSweepANELanePresets: Bool = false
    public var labProgressText: String?
    public var isRunningEvalLab: Bool = false
    public private(set) var activeEmbeddingRequestID: String?

    @ObservationIgnored public private(set) var activeEmbeddingEngine: any EmbeddingEngine
    @ObservationIgnored private var indexCoordinator: IndexCoordinator
    @ObservationIgnored private var embeddingSearchTask: Task<Void, Never>?
    @ObservationIgnored private var activeEmbeddingRunTask: Task<Void, Never>?
    @ObservationIgnored private var indexTask: Task<Void, Never>?
    @ObservationIgnored private var lastInspectedANEKey: String = ""
    @ObservationIgnored private var isSyncingCompareMirror: Bool = false
    #if os(macOS)
    @ObservationIgnored private var folderWatcher: FolderWatcher?
    #endif

    @ObservationIgnored public var onStatusMessage: (String?) -> Void = { _ in }

    public init(
        vectorStore: VectorStore,
        router: HeylookRouter,
        embeddingEngine: (any EmbeddingEngine)? = nil
    ) {
        self.vectorStore = vectorStore
        self.router = router
        let initialEngine = embeddingEngine ?? NativeMLXEmbeddingGemma2.makeBestAvailableEngine(computeOptions: .hybridANETowers)
        self.activeEmbeddingEngine = initialEngine
        self.indexCoordinator = IndexCoordinator(engine: initialEngine, store: vectorStore)
        self.labSet = Self.makeBuiltInMultimodalSampleSet()
        self.selectedEmbeddingModelID = initialEngine.modelID
    }

    // MARK: - Computed Properties

    public var hasActiveQuery: Bool {
        !embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || queryAttachmentURL != nil
    }

    public var effectiveQueryTask: EmbeddingTask {
        selectedTaskOverride ?? (selectedKindFilter == .code ? .codeRetrieval : .query)
    }

    public var boundEffectiveQueryTask: EmbeddingTask {
        get { effectiveQueryTask }
        set {
            selectedTaskOverride = newValue
            Task { await runEmbeddingSearch() }
        }
    }

    public var boundComputePreset: ComputePreset {
        get { modelComputeOptions.preset }
        set { applyComputePreset(newValue) }
    }

    public subscript(computeUnitFor component: EmbeddingComponent) -> TowerComputeUnit {
        get { modelComputeOptions[component] }
        set {
            modelComputeOptions[component] = newValue
            lastInspectedANEKey = ""
            setupEmbeddingCoordinator()
        }
    }

    public var activeQueryTowersLabel: String {
        let hasText = !embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let attachmentKind = queryAttachmentURL.flatMap { ArtifactKind.detect(from: $0) }
        switch (hasText, attachmentKind) {
        case (true, nil):
            return "Text Tower"
        case (true, .some(.image)):
            return "Text + Vision (Image <=280 tokens)"
        case (true, .some(.video)):
            return "Text + Vision (Video <=140 tokens/f)"
        case (true, .some(.audio)):
            return "Text + Audio (16 kHz Conformer)"
        case (true, .some(let k)):
            return "Text + \(k.rawValue.capitalized)"
        case (false, .some(.image)):
            return "Vision Tower (Image <=280 tokens)"
        case (false, .some(.video)):
            return "Vision Tower (Video <=140 tokens/f)"
        case (false, .some(.audio)):
            return "Audio Tower (16 kHz Conformer)"
        case (false, .some(let k)):
            return "Text Tower (\(k.rawValue.capitalized))"
        case (false, nil):
            return "Text / Vision / Audio"
        }
    }

    public var isActiveEmbeddingEngineSynthetic: Bool {
        false
    }

    // MARK: - Engine Setup & Shutdown

    public func syncEmbeddingEngineToRouter() async {
        await router.setEmbeddingEngine(activeEmbeddingEngine)
    }

    public func unloadActiveEngine() async {
        await activeEmbeddingEngine.unload()
    }

    public func stopBackgroundTasks() {
        embeddingSearchTask?.cancel()
        embeddingSearchTask = nil
        cancelActiveEmbeddingRun()
        indexTask?.cancel()
        indexTask = nil
        #if os(macOS)
        folderWatcher?.stop()
        #endif
    }

    public func setupEmbeddingCoordinator() {
        let dir = URL(fileURLWithPath: (checkpointPath as NSString).expandingTildeInPath)
        let engine: any EmbeddingEngine
        switch selectedBackend {
        case .heylook:
            let baseURL = URL(string: embeddingServerURLString) ?? URL(string: "http://127.0.0.1:8000")!
            let heylook = HeylookMLXEngine(baseURL: baseURL)
            engine = heylook
            Task {
                self.serverCapabilities = try? await heylook.fetchCapabilities()
            }
        case .nativeMLX:
            engine = NativeMLXEmbeddingGemma2.makeBestAvailableEngine(
                checkpointURL: dir,
                computeOptions: modelComputeOptions
            )
        }
        self.activeEmbeddingEngine = engine
        self.indexCoordinator = IndexCoordinator(engine: engine, store: vectorStore)
        let targetRouter = self.router
        Task {
            await targetRouter.setEmbeddingEngine(engine)
            await self.refreshEmbeddingModels()
        }
        refreshEmbeddingCounts()
        refreshWatchedFolders()
        refreshComputeReports()
    }

    public func refreshComputeReports() {
        let dir = URL(fileURLWithPath: (checkpointPath as NSString).expandingTildeInPath)
        let componentUnits = EmbeddingComponent.allCases.map { modelComputeOptions[$0].rawValue }.joined(separator: ",")
        lastInspectedANEKey = "\(dir.path)|\(modelComputeOptions.preset.rawValue)|\(componentUnits)"
        aneInspectionReports = ANETowerInspector.statusSnapshot(
            options: modelComputeOptions,
            checkpointDirectory: dir
        )
        Task {
            let full = await ANETowerInspector.inspect(
                checkpointDirectory: dir,
                options: modelComputeOptions
            )
            if full.count == EmbeddingComponent.allCases.count {
                self.aneInspectionReports = full
            }
        }
    }

    public func refreshComputeReportsIfNeeded() async {
        let dir = URL(fileURLWithPath: (checkpointPath as NSString).expandingTildeInPath)
        let componentUnits = EmbeddingComponent.allCases.map { modelComputeOptions[$0].rawValue }.joined(separator: ",")
        let aneKey = "\(dir.path)|\(modelComputeOptions.preset.rawValue)|\(componentUnits)"
        if aneKey != lastInspectedANEKey || aneInspectionReports.isEmpty {
            lastInspectedANEKey = aneKey
            if aneInspectionReports.isEmpty {
                aneInspectionReports = ANETowerInspector.statusSnapshot(options: modelComputeOptions, checkpointDirectory: dir)
            }
            let full = await ANETowerInspector.inspect(checkpointDirectory: dir, options: modelComputeOptions)
            if full.count == EmbeddingComponent.allCases.count {
                self.aneInspectionReports = full
            }
        }
    }

    public func applyComputePreset(_ preset: ComputePreset) {
        switch preset {
        case .allMLXGPU:
            modelComputeOptions = .allMLXGPU
        case .hybridANETowers:
            modelComputeOptions = .hybridANETowers
        case .allNeuralEngine:
            modelComputeOptions = .allNeuralEngine
        case .custom:
            break
        }
        lastInspectedANEKey = ""
        setupEmbeddingCoordinator()
    }

    public func refreshEmbeddingCounts() {
        guard let stats = try? vectorStore.stats() else { return }
        artifactCount = stats.artifactCount
        chunkCount = stats.chunkCount
        countsByKind = stats.countsByKind
    }

    public func refreshWatchedFolders() {
        guard let folders = try? vectorStore.listWatchedFolders() else { return }
        watchedFolders = folders
        #if os(macOS)
        let folderURLs = folders.map(\.url)
        if folderWatcher == nil {
            folderWatcher = FolderWatcher { [weak self] changedDirs in
                Task { @MainActor [weak self] in
                    guard let self, !self.isIndexing else { return }
                    self.startIndexing(changedDirs, persistFolders: false)
                }
            }
        }
        folderWatcher?.watch(urls: folderURLs)
        #endif
    }

    public func removeWatchedFolder(_ folder: VectorStore.WatchedFolder) {
        try? vectorStore.removeWatchedFolder(folder.url)
        refreshWatchedFolders()
    }

    public func reindexWatchedFolders() {
        let urls = watchedFolders.map(\.url)
        guard !urls.isEmpty else { return }
        startIndexing(urls, persistFolders: false)
    }

    public func startIndexing(_ urls: [URL], persistFolders: Bool = true) {
        guard !isActiveEmbeddingEngineSynthetic else {
            let msg = "Cannot index media without a real embedding checkpoint at '\(checkpointPath)' (or switch Engine to Remote Server)."
            embeddingStatusMessage = msg
            onStatusMessage(msg)
            return
        }
        indexTask?.cancel()
        if persistFolders {
            for url in urls {
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
                    try? vectorStore.addWatchedFolder(url, windowSampling: selectedSampling)
                }
            }
            refreshWatchedFolders()
        }
        indexTask = Task {
            await indexURLs(urls)
        }
    }

    public func cancelIndexing() {
        indexTask?.cancel()
        indexTask = nil
        isIndexing = false
        isEmbeddingBusy = false
        embeddingStatusMessage = "Indexing cancelled."
    }

    public func indexURLs(_ urls: [URL]) async {
        isEmbeddingBusy = true
        isIndexing = true
        defer {
            isEmbeddingBusy = false
            isIndexing = false
            refreshEmbeddingCounts()
            refreshWatchedFolders()
        }

        do {
            let result = try await indexCoordinator.indexURLs(
                urls,
                videoWindowSampling: selectedSampling,
                loadAudioFromVideo: loadAudioFromVideo
            ) { progress in
                Task { @MainActor in
                    self.embeddingStatusMessage = "Indexing (\(progress.processed)/\(progress.total)): \(progress.currentTitle)"
                }
            }
            embeddingStatusMessage = "Indexed \(result.indexed) artifacts (\(result.skipped.count) skipped)."
        } catch is CancellationError {
            embeddingStatusMessage = "Indexing cancelled."
        } catch {
            embeddingStatusMessage = "Index error: \(error.localizedDescription)"
        }
    }

    public func clearEmbeddingIndex() {
        do {
            try vectorStore.clearAll()
            searchHits = []
            selectedHit = nil
            refreshEmbeddingCounts()
            embeddingStatusMessage = "Index cleared."
        } catch {
            embeddingStatusMessage = "Clear failed: \(error.localizedDescription)"
        }
    }

    /// Seeds sample artifacts ONLY when a real embedding engine is active (never fabricates hash vectors!).
    public func seedSampleCorpusIfNeeded() async {
        setupEmbeddingCoordinator()
        await syncEmbeddingEngineToRouter()
        guard !isActiveEmbeddingEngineSynthetic else {
            let msg = "Sample corpus indexing requires a real EmbeddingGemma 2 checkpoint at '\(checkpointPath)' (or an active Heylook embedding server)."
            embeddingStatusMessage = msg
            onStatusMessage(msg)
            return
        }
        guard ((try? vectorStore.stats().artifactCount) ?? 0) == 0 else {
            refreshEmbeddingCounts()
            return
        }

        let samples: [(String, ArtifactKind, String, String)] = [
            (
                "doc-vision",
                .text,
                "VISION.md — Native Apple Silicon Workbench",
                "Heylook is a single-user multimodal AI workbench for macOS 27.2 and iOS 27.2 unifying MLX Metal GPU generation with ANE CoreML co-residency."
            ),
            (
                "doc-scheduler",
                .code,
                "ComputeScheduler.swift — Dual-Lane GPU & ANE Co-Residency",
                "The ComputeScheduler separates heavy autoregressive GPU models from ANE-resident embedding and vision towers so background media indexing never evicts a 70B LLM."
            ),
            (
                "doc-towers",
                .code,
                "ModelComputeOptions.swift — Per-Tower Compute Unit Selection",
                "Assign Vision Tower and Audio Conformer to CPU + Apple Neural Engine while keeping Text Projection on GPU or ANE with Matryoshka 768d to 128d truncation."
            ),
            (
                "doc-sse",
                .code,
                "MessagesWire.swift — Disconnect-Resilient SSE Generation",
                "Mobile iOS clients can lock their screen mid-generation while the Mac Studio finishes writing the assistant turn to SQLite WAL and emits heylook_saved."
            )
        ]

        do {
            let batchInputs = samples.map { _, _, title, snippet in
                EmbeddingInput(parts: [.text(snippet)], task: .document, title: title)
            }
            let batchOutputs = try await activeEmbeddingEngine.embedBatch(
                batchInputs,
                dimensions: Matryoshka.gemma2NativeDimensions
            )
            guard batchOutputs.count == samples.count else {
                throw ProviderExecutionError.runtimeUnavailable(
                    modelID: activeEmbeddingEngine.modelID,
                    detail: "Embedding batch returned \(batchOutputs.count)/\(samples.count) vectors."
                )
            }

            for (idx, (id, kind, title, snippet)) in samples.enumerated() {
                let artifact = Artifact(
                    id: id,
                    collectionID: "default",
                    kind: kind,
                    url: URL(fileURLWithPath: "/workspace/\(id).swift"),
                    title: title,
                    fingerprint: ContentFingerprint(byteCount: Int64(snippet.utf8.count), modifiedUnixMs: 1_700_000_000_000),
                    modelID: activeEmbeddingEngine.modelID
                )
                let chunk = EmbeddedChunk(
                    id: "\(id)#whole",
                    artifactID: id,
                    span: .whole,
                    vector: batchOutputs[idx].vector,
                    previewText: snippet
                )
                try vectorStore.upsert(artifact: artifact, chunks: [chunk], modelID: activeEmbeddingEngine.modelID)
            }
            refreshEmbeddingCounts()
            embeddingQuery = "Apple Neural Engine co-residency"
            await runEmbeddingSearch()
            await runEmbeddingLabEvaluation()
        } catch {
            embeddingStatusMessage = "Sample corpus error: \(error.localizedDescription)"
            onStatusMessage("Sample corpus error: \(error.localizedDescription)")
        }
    }

    public func scheduleEmbeddingSearchDebounced() {
        embeddingSearchTask?.cancel()
        embeddingSearchTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled else { return }
            await self?.runEmbeddingSearch()
        }
    }

    public func runEmbeddingSearch() async {
        embeddingSearchTask?.cancel()
        let q = embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty || queryAttachmentURL != nil else {
            searchHits = []
            selectedHit = nil
            return
        }
        guard !isActiveEmbeddingEngineSynthetic else {
            searchHits = []
            selectedHit = nil
            embeddingStatusMessage = "Load an EmbeddingGemma 2 checkpoint at '\(checkpointPath)' (or switch Engine to Remote Server) to search."
            return
        }

        isEmbeddingBusy = true
        embeddingStatusMessage = "Searching (\(selectedDimension)d)…"
        defer { isEmbeddingBusy = false }

        do {
            let kinds: Set<ArtifactKind>? = selectedKindFilter.map { [$0] }
            let results = try await indexCoordinator.search(
                queryText: q,
                exampleFileURL: queryAttachmentURL,
                dimensions: selectedDimension,
                kinds: kinds,
                taskOverride: selectedTaskOverride,
                limit: 30
            )
            searchHits = results
            if let existingID = selectedHit?.id, let keep = results.first(where: { $0.id == existingID }) {
                selectedHit = keep
            } else {
                selectedHit = results.first
            }
            selectedSpanIndex = 0
            embeddingStatusMessage = "Found \(results.count) results (\(selectedDimension)d Matryoshka slice)."
        } catch {
            searchHits = []
            selectedHit = nil
            embeddingStatusMessage = "Search failed: \(error.localizedDescription)"
            onStatusMessage("Search failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Embedding Model Discovery & Dynamic Capabilities

    public var activeEmbeddingModelEntry: EmbeddingModelCatalogEntry? {
        availableEmbeddingModels.first(where: { $0.id == selectedEmbeddingModelID })
            ?? availableEmbeddingModels.first
    }

    public var activeEmbeddingFacts: EngineEmbeddingFacts {
        if let entry = activeEmbeddingModelEntry {
            return entry.facts
        }
        let dir = URL(fileURLWithPath: (checkpointPath as NSString).expandingTildeInPath)
        if let desc = try? NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: dir) {
            return EmbeddingModelFacts.fromCheckpoint(desc, computePlacement: modelComputeOptions.preset.label)
        }
        return EngineEmbeddingFacts(
            modalities: ["text", "image", "audio", "video"],
            dimensions: Matryoshka.gemma2Dimensions,
            nativeDimensions: Matryoshka.gemma2NativeDimensions,
            maxInputTokens: 8_192,
            prompts: [:],
            normalization: "l2",
            videoAudio: "separate_audio_block",
            videoWindows: EngineEmbeddingVideoWindows(
                maxSeconds: 32.0,
                sampling: ["fps", "content"],
                defaultSampling: "fps"
            ),
            limits: EngineEmbeddingLimits(),
            provenance: "fallback_checkpoint_facts"
        )
    }

    public var availableEmbeddingPromptNames: [String] {
        activeEmbeddingFacts.prompts.keys.sorted()
    }

    public var availableEmbeddingDimensions: [Int] {
        let dims = activeEmbeddingFacts.dimensions
        return dims.isEmpty ? Matryoshka.gemma2Dimensions : dims.sorted()
    }

    public var supportsVideoWindowingForSelectedModel: Bool {
        activeEmbeddingFacts.videoWindows != nil || activeEmbeddingFacts.modalities.contains("video")
    }

    public func refreshEmbeddingModels() async {
        var entries: [EmbeddingModelCatalogEntry] = []
        var seen: Set<String> = []

        if selectedBackend == .heylook, let remoteURL = URL(string: embeddingServerURLString) {
            let remote = RemoteWorkbenchBackend(baseURL: remoteURL)
            let remoteModels = await remote.listPublicModels()
            for row in remoteModels where row.capabilities.contains("embeddings") || row.engine.embedding != nil {
                guard seen.insert(row.id).inserted else { continue }
                let facts = row.engine.embedding ?? EngineEmbeddingFacts(
                    modalities: row.modalities,
                    dimensions: Matryoshka.gemma2Dimensions,
                    nativeDimensions: Matryoshka.gemma2NativeDimensions,
                    prompts: [:],
                    provenance: "remote_server"
                )
                entries.append(
                    EmbeddingModelCatalogEntry(
                        id: row.id,
                        label: row.id,
                        facts: facts,
                        provider: row.provider,
                        modelPath: nil
                    )
                )
            }
        } else {
            let localPublicRows = await router.service.listPublicModels()
            let scheduled = await router.scheduler.listModels()
            for row in localPublicRows where row.capabilities.contains("embeddings") || row.engine.embedding != nil {
                guard seen.insert(row.id).inserted else { continue }
                let desc = scheduled.first(where: { $0.modelID == row.id })
                let facts = row.engine.embedding ?? EngineEmbeddingFacts(
                    modalities: row.modalities,
                    dimensions: Matryoshka.gemma2Dimensions,
                    nativeDimensions: Matryoshka.gemma2NativeDimensions,
                    prompts: [:],
                    provenance: "local_scheduler"
                )
                entries.append(
                    EmbeddingModelCatalogEntry(
                        id: row.id,
                        label: row.id,
                        facts: facts,
                        provider: row.provider,
                        modelPath: desc?.modelPath
                    )
                )
            }

            let dir = URL(fileURLWithPath: (checkpointPath as NSString).expandingTildeInPath)
            if !isActiveEmbeddingEngineSynthetic,
               !seen.contains(activeEmbeddingEngine.modelID),
               let desc = try? NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: dir) {
                let facts = EmbeddingModelFacts.fromCheckpoint(desc, computePlacement: modelComputeOptions.preset.label)
                entries.append(
                    EmbeddingModelCatalogEntry(
                        id: activeEmbeddingEngine.modelID,
                        label: activeEmbeddingEngine.modelID,
                        facts: facts,
                        provider: .mlx,
                        modelPath: dir.path
                    )
                )
                seen.insert(activeEmbeddingEngine.modelID)
            }
        }

        if entries.isEmpty {
            let dir = URL(fileURLWithPath: (checkpointPath as NSString).expandingTildeInPath)
            let facts: EngineEmbeddingFacts
            if let desc = try? NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: dir) {
                facts = EmbeddingModelFacts.fromCheckpoint(desc, computePlacement: modelComputeOptions.preset.label)
            } else {
                facts = activeEmbeddingFacts
            }
            entries.append(
                EmbeddingModelCatalogEntry(
                    id: activeEmbeddingEngine.modelID,
                    label: activeEmbeddingEngine.modelID,
                    facts: facts,
                    provider: .mlx,
                    modelPath: dir.path
                )
            )
        }

        availableEmbeddingModels = entries
        if selectedEmbeddingModelID.isEmpty || !entries.contains(where: { $0.id == selectedEmbeddingModelID }) {
            selectedEmbeddingModelID = entries.first?.id ?? activeEmbeddingEngine.modelID
        }
        let validDims = availableEmbeddingDimensions
        if !validDims.contains(compareMatrixDimension), let lastDim = validDims.last {
            compareMatrixDimension = lastDim
        }
    }

    public func selectEmbeddingModel(_ modelID: String) {
        selectedEmbeddingModelID = modelID
        if let entry = availableEmbeddingModels.first(where: { $0.id == modelID }),
           let path = entry.modelPath,
           entry.provider != .gguf {
            checkpointPath = path
            setupEmbeddingCoordinator()
        }
        let validDims = availableEmbeddingDimensions
        if !validDims.contains(compareMatrixDimension), let maxDim = validDims.last {
            compareMatrixDimension = maxDim
        }
        comparePreflightError = validateCompareItemsForLimits()
    }

    // MARK: - Cancellation Support

    public func cancelActiveEmbeddingRun() {
        activeEmbeddingRunTask?.cancel()
        activeEmbeddingRunTask = nil
        if let reqID = activeEmbeddingRequestID {
            activeEmbeddingRequestID = nil
            _ = router.registry.abort(requestID: reqID)
            if selectedBackend == .heylook, let baseURL = URL(string: embeddingServerURLString) {
                let remote = RemoteWorkbenchBackend(baseURL: baseURL)
                Task { try? await remote.abortRequest(id: reqID) }
            }
        }
        isEmbeddingBusy = false
        isRunningEvalLab = false
        labProgressText = nil
        embeddingStatusMessage = "Cancelled embedding request."
        onStatusMessage("Stopped embedding run.")
    }

    // MARK: - N×N Multimodal Compare Matrix (Ordered Multi-Block Items)

    private func syncCompareItemsMirrorFromRich() {
        guard !isSyncingCompareMirror else { return }
        isSyncingCompareMirror = true
        compareItems = compareRichItems.map(\.asMatrixItem)
        isSyncingCompareMirror = false
    }

    public func addCompareTextItem(_ initialText: String = "") {
        guard compareRichItems.count < Self.maxCompareMatrixItems else {
            onStatusMessage("Compare supports up to \(Self.maxCompareMatrixItems) items.")
            return
        }
        compareRichItems.append(
            CompareRichItem(
                label: "Item \(compareRichItems.count + 1)",
                blocks: [CompareBlock(kind: .text, text: initialText)]
            )
        )
        comparePreflightError = validateCompareItemsForLimits()
    }

    public func addCompareMediaFiles(_ urls: [URL]) {
        for url in urls {
            guard compareRichItems.count < Self.maxCompareMatrixItems else {
                onStatusMessage("Compare supports up to \(Self.maxCompareMatrixItems) items.")
                break
            }
            let kind = ArtifactKind.detect(from: url) ?? .document
            compareRichItems.append(
                CompareRichItem(
                    label: url.lastPathComponent,
                    blocks: [
                        CompareBlock(
                            kind: .media,
                            fileURL: url,
                            mediaKind: kind,
                            videoWindowMode: kind == .video ? .fps : .wholeVideo
                        )
                    ]
                )
            )
        }
        comparePreflightError = validateCompareItemsForLimits()
    }

    public func removeCompareItem(id: String) {
        compareRichItems.removeAll { $0.id == id }
        compareRowVectors768 = []
        compareColVectors768 = []
        compareItemOutputs = []
        compareMatrixScores = []
        compareRankedMatchesToItem1 = []
        compareVideoWindowHits = []
        selectedCompareCellRow = 0
        selectedCompareCellCol = min(1, max(0, compareRichItems.count - 1))
        comparePreflightError = validateCompareItemsForLimits()
    }

    public func moveCompareItem(id: String, delta: Int) {
        guard let idx = compareRichItems.firstIndex(where: { $0.id == id }) else { return }
        let target = idx + delta
        guard target >= 0 && target < compareRichItems.count else { return }
        compareRichItems.swapAt(idx, target)
    }

    public func addBlockToCompareItem(itemID: String, kind: CompareBlockKind) {
        guard let idx = compareRichItems.firstIndex(where: { $0.id == itemID }) else { return }
        guard compareRichItems[idx].blocks.count < EmbeddingRequestLimits.maxBlocksPerItem else {
            onStatusMessage("Each item allows up to \(EmbeddingRequestLimits.maxBlocksPerItem) blocks.")
            return
        }
        compareRichItems[idx].blocks.append(CompareBlock(kind: kind))
        comparePreflightError = validateCompareItemsForLimits()
    }

    public func addMediaBlockToCompareItem(itemID: String, url: URL) {
        guard let idx = compareRichItems.firstIndex(where: { $0.id == itemID }) else { return }
        guard compareRichItems[idx].blocks.count < EmbeddingRequestLimits.maxBlocksPerItem else {
            onStatusMessage("Each item allows up to \(EmbeddingRequestLimits.maxBlocksPerItem) blocks.")
            return
        }
        let kind = ArtifactKind.detect(from: url) ?? .document
        let block = CompareBlock(
            kind: .media,
            fileURL: url,
            mediaKind: kind,
            videoWindowMode: .wholeVideo
        )
        compareRichItems[idx].blocks.append(block)
        if compareRichItems[idx].label.hasPrefix("Item ") && compareRichItems[idx].blocks.count == 1 {
            compareRichItems[idx].label = url.lastPathComponent
        }
        comparePreflightError = validateCompareItemsForLimits()
    }

    public func removeBlockFromCompareItem(itemID: String, blockID: String) {
        guard let idx = compareRichItems.firstIndex(where: { $0.id == itemID }) else { return }
        guard compareRichItems[idx].blocks.count > 1 else {
            compareRichItems[idx].blocks = [CompareBlock(kind: .text)]
            comparePreflightError = validateCompareItemsForLimits()
            return
        }
        compareRichItems[idx].blocks.removeAll { $0.id == blockID }
        comparePreflightError = validateCompareItemsForLimits()
    }

    public func moveBlockInCompareItem(itemID: String, blockID: String, delta: Int) {
        guard let idx = compareRichItems.firstIndex(where: { $0.id == itemID }),
              let blockIdx = compareRichItems[idx].blocks.firstIndex(where: { $0.id == blockID }) else { return }
        let target = blockIdx + delta
        guard target >= 0 && target < compareRichItems[idx].blocks.count else { return }
        compareRichItems[idx].blocks.swapAt(blockIdx, target)
    }

    public func setCompareItemPrompt(itemID: String, promptName: String?) {
        guard let idx = compareRichItems.firstIndex(where: { $0.id == itemID }) else { return }
        let trimmed = promptName?.trimmingCharacters(in: .whitespacesAndNewlines)
        compareRichItems[idx].promptName = (trimmed?.isEmpty == false) ? trimmed : nil
        comparePreflightError = validateCompareItemsForLimits()
    }

    public func validateCompareItemsForLimits() -> String? {
        let items = compareRichItems
        let facts = activeEmbeddingFacts
        let maxItems = min(EmbeddingRequestLimits.maxItems, facts.limits?.maxBatchSize ?? EmbeddingRequestLimits.maxItems)
        if items.count < 2 {
            return "Add at least 2 items to compare."
        }
        if items.count > maxItems {
            return "Too many items (\(items.count)); maximum is \(maxItems)."
        }

        var totalTextChars = 0
        var totalRequestMediaBytes = 0
        let supportedPromptSet = Set(facts.prompts.keys)
        let maxWinSeconds = facts.videoWindows?.maxSeconds ?? 32.0

        for (itemIdx, item) in items.enumerated() {
            if item.blocks.isEmpty {
                return "\(item.label): add at least one text or media block."
            }
            if item.blocks.count > EmbeddingRequestLimits.maxBlocksPerItem {
                return "\(item.label): maximum \(EmbeddingRequestLimits.maxBlocksPerItem) blocks per item."
            }
            if let explicitPrompt = item.promptName,
               !explicitPrompt.isEmpty,
               !supportedPromptSet.isEmpty,
               !supportedPromptSet.contains(explicitPrompt) {
                return "\(item.label): unknown prompt_name '\(explicitPrompt)' for \(selectedEmbeddingModelID)."
            }

            var hasEffectiveContent = false
            var windowedVideoCount = 0
            for block in item.blocks {
                switch block.kind {
                case .text:
                    let trimmed = block.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        hasEffectiveContent = true
                    }
                    totalTextChars += block.text.count
                    if totalTextChars > EmbeddingRequestLimits.maxTextChars {
                        return "Total text exceeds \(EmbeddingRequestLimits.maxTextChars) character limit."
                    }
                case .media:
                    guard let fileURL = block.fileURL else { continue }
                    hasEffectiveContent = true
                    let detectedKind = block.mediaKind ?? ArtifactKind.detect(from: fileURL) ?? .document
                    if let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
                       let byteCount = (attrs[.size] as? NSNumber)?.intValue {
                        let limit = detectedKind == .video
                            ? EmbeddingRequestLimits.maxVideoBytes
                            : EmbeddingRequestLimits.maxImageOrAudioBytes
                        if byteCount > limit {
                            let mb = limit / (1024 * 1024)
                            return "\(fileURL.lastPathComponent) exceeds per-file \(mb) MB limit."
                        }
                        totalRequestMediaBytes += byteCount
                        if totalRequestMediaBytes > EmbeddingRequestLimits.maxRequestMediaBytes {
                            let reqMB = EmbeddingRequestLimits.maxRequestMediaBytes / (1024 * 1024)
                            return "Total media payload across items exceeds \(reqMB) MB request limit."
                        }
                    }

                    if detectedKind == .video && block.videoWindowMode != .wholeVideo {
                        windowedVideoCount += 1
                        if item.blocks.count != 1 {
                            return "Item \(itemIdx + 1) (\(item.label)): windowed video must be the only block in its item."
                        }
                        if !facts.modalities.contains("video") {
                            return "\(selectedEmbeddingModelID) does not advertise video modality."
                        }
                        let rawSecText = block.videoWindowSecondsText.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !rawSecText.isEmpty {
                            guard let secs = Double(rawSecText), secs.isFinite, secs > 0 else {
                                return "\(item.label): window seconds must be a positive number."
                            }
                            if secs > maxWinSeconds {
                                return "\(item.label): window seconds (\(secs)s) exceeds model max \(Int(maxWinSeconds))s."
                            }
                        }
                    }
                }
            }
            if !hasEffectiveContent && windowedVideoCount == 0 {
                return "\(item.label): enter text or attach a media file."
            }
        }
        return nil
    }

    private func buildRichCompareEmbeddingInput(
        for item: CompareRichItem,
        fallbackPromptName: String?
    ) throws -> EmbeddingInput {
        var parts: [EmbeddingPart] = []
        var explicitVideoWindows: VideoWindowOptions?

        for block in item.blocks {
            switch block.kind {
            case .text:
                let trimmed = block.text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    parts.append(.text(trimmed))
                }
            case .media:
                guard let fileURL = block.fileURL else { continue }
                let scoped = fileURL.startAccessingSecurityScopedResource()
                defer {
                    if scoped { fileURL.stopAccessingSecurityScopedResource() }
                }
                let detectedKind = block.mediaKind ?? ArtifactKind.detect(from: fileURL) ?? .document
                switch detectedKind {
                case .image, .audio:
                    parts.append(.mediaURL(kind: detectedKind, url: fileURL))
                case .video:
                    let sampling = block.videoWindowMode.sampling
                    parts.append(.mediaURL(kind: .video, url: fileURL, windowSampling: sampling))
                    if let vw = block.videoWindowOptions {
                        explicitVideoWindows = vw
                    }
                case .document:
                    let plans = try ArtifactProcessor.chunkDocument(at: fileURL)
                    if let firstText = plans.first(where: { !$0.text.isEmpty })?.text {
                        parts.append(.text(String(firstText.prefix(8000))))
                    } else if let firstThumb = plans.first(where: { $0.thumbnailJPEG != nil })?.thumbnailJPEG {
                        parts.append(.media(kind: .image, mimeType: "image/jpeg", data: firstThumb))
                    }
                case .text, .code:
                    let text = try String(contentsOf: fileURL, encoding: .utf8)
                    parts.append(.text(String(text.prefix(8000))))
                }
            }
        }

        if parts.isEmpty {
            parts.append(.text(item.label))
        }

        let explicitPrompt = item.promptName?.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedPrompt = (explicitPrompt?.isEmpty == false) ? explicitPrompt : fallbackPromptName
        return EmbeddingInput(
            parts: parts,
            promptName: resolvedPrompt,
            title: item.label,
            videoWindows: explicitVideoWindows
        )
    }

    private func preferredPromptName(matching keyword: String, exclude: String? = nil) -> String? {
        let keys = availableEmbeddingPromptNames
        if let exact = keys.first(where: { $0.localizedCaseInsensitiveContains(keyword) }) {
            return exact
        }
        return keys.first(where: { $0 != exclude })
    }

    private func scoreCandidateOutput(_ queryVector: [Float], candidate: EmbeddedOutput, dimensions: Int) -> Float {
        if !candidate.windows.isEmpty {
            return candidate.windows.map {
                Matryoshka.cosine(queryVector, $0.vector, dimensions: dimensions)
            }.max() ?? Matryoshka.cosine(queryVector, candidate.vector, dimensions: dimensions)
        }
        return Matryoshka.cosine(queryVector, candidate.vector, dimensions: dimensions)
    }

    public func recomputeCompareMatrixScores() {
        let n = compareRowVectors768.count
        guard n >= 2, compareColVectors768.count == n else {
            compareMatrixScores = []
            compareRankedMatchesToItem1 = []
            compareVideoWindowHits = []
            return
        }
        let dim = compareMatrixDimension
        var grid: [[Float]] = Array(repeating: Array(repeating: 0, count: n), count: n)
        for r in 0..<n {
            for c in 0..<n {
                if !compareItemOutputs.isEmpty, r == 0, c < compareItemOutputs.count, !compareItemOutputs[c].windows.isEmpty {
                    grid[r][c] = scoreCandidateOutput(compareRowVectors768[r], candidate: compareItemOutputs[c], dimensions: dim)
                } else {
                    grid[r][c] = Matryoshka.cosine(compareRowVectors768[r], compareColVectors768[c], dimensions: dim)
                }
            }
        }
        compareMatrixScores = grid
        selectedCompareCellRow = min(max(0, selectedCompareCellRow), n - 1)
        selectedCompareCellCol = min(max(0, selectedCompareCellCol), n - 1)
    }

    public func recomputeCompareRankings() {
        let n = compareRowVectors768.count
        guard n >= 2, compareColVectors768.count == n else {
            compareRankedMatchesToItem1 = []
            compareVideoWindowHits = []
            return
        }
        let dim = compareMatrixDimension
        let item0Vec = compareRowVectors768[0]

        var itemMatches: [(index: Int, label: String, score: Float)] = []
        var windowHits: [CompareVideoWindowHit] = []

        for idx in 1..<n where idx < compareRichItems.count {
            let label = compareRichItems[idx].label
            let score: Float
            if idx < compareItemOutputs.count {
                let output = compareItemOutputs[idx]
                score = scoreCandidateOutput(item0Vec, candidate: output, dimensions: dim)
                let videoURL = compareRichItems[idx].blocks.first(where: { $0.mediaKind == .video })?.fileURL
                    ?? compareRichItems[idx].primaryFileURL
                for (winIdx, win) in output.windows.enumerated() {
                    let wScore = Matryoshka.cosine(item0Vec, win.vector, dimensions: dim)
                    windowHits.append(
                        CompareVideoWindowHit(
                            id: "\(idx)-\(winIdx)",
                            itemIndex: idx,
                            itemLabel: label,
                            fileURL: videoURL,
                            windowIndex: winIdx,
                            start: win.start,
                            end: win.end,
                            frameTimes: win.frameTimes,
                            score: wScore
                        )
                    )
                }
            } else {
                score = Matryoshka.cosine(item0Vec, compareColVectors768[idx], dimensions: dim)
            }
            itemMatches.append((index: idx, label: label, score: score))
        }

        compareRankedMatchesToItem1 = itemMatches.sorted { $0.score > $1.score }
        compareVideoWindowHits = windowHits.sorted { $0.score > $1.score }
    }

    public func runCompareMatrix() async {
        if selectedBackend == .nativeMLX && isActiveEmbeddingEngineSynthetic {
            compareMatrixScores = []
            let msg = "Compare requires a real EmbeddingGemma 2 checkpoint at '\(checkpointPath)' (or Remote Server)."
            embeddingStatusMessage = msg
            onStatusMessage(msg)
            return
        }
        if let validationError = validateCompareItemsForLimits() {
            comparePreflightError = validationError
            embeddingStatusMessage = validationError
            onStatusMessage(validationError)
            return
        }
        comparePreflightError = nil

        let items = Array(compareRichItems.prefix(Self.maxCompareMatrixItems))
        let reqID = "emb-compare-\(UUID().uuidString.prefix(8))"
        activeEmbeddingRequestID = reqID
        isEmbeddingBusy = true
        embeddingStatusMessage = "Computing \(items.count)×\(items.count) multimodal matrix…"

        let task = Task { @MainActor in
            defer {
                if self.activeEmbeddingRequestID == reqID {
                    self.activeEmbeddingRequestID = nil
                }
                self.isEmbeddingBusy = false
                self.activeEmbeddingRunTask = nil
            }
            do {
                let queryPromptFallback = self.compareAsymmetricMode ? self.preferredPromptName(matching: "query") : nil
                let docPromptFallback = self.compareAsymmetricMode ? self.preferredPromptName(matching: "document", exclude: queryPromptFallback) : nil
                let requestDim = self.activeEmbeddingFacts.nativeDimensions

                if self.compareAsymmetricMode {
                    let rowInputs = try items.map { try self.buildRichCompareEmbeddingInput(for: $0, fallbackPromptName: queryPromptFallback) }
                    let colInputs = try items.map { try self.buildRichCompareEmbeddingInput(for: $0, fallbackPromptName: docPromptFallback) }
                    let rowOutputs = try await self.executeEmbeddingBatch(
                        inputs: rowInputs,
                        modelID: self.selectedEmbeddingModelID,
                        dimensions: requestDim,
                        requestID: "\(reqID)-row"
                    )
                    try Task.checkCancellation()
                    let colOutputs = try await self.executeEmbeddingBatch(
                        inputs: colInputs,
                        modelID: self.selectedEmbeddingModelID,
                        dimensions: requestDim,
                        requestID: "\(reqID)-col"
                    )
                    guard rowOutputs.count == items.count, colOutputs.count == items.count else {
                        throw ProviderExecutionError.runtimeUnavailable(
                            modelID: self.selectedEmbeddingModelID,
                            detail: "Compare matrix vector count mismatch."
                        )
                    }
                    self.compareRowVectors768 = rowOutputs.map(\.vector)
                    self.compareColVectors768 = colOutputs.map(\.vector)
                    self.compareItemOutputs = colOutputs
                } else {
                    let symInputs = try items.map { try self.buildRichCompareEmbeddingInput(for: $0, fallbackPromptName: nil) }
                    let outputs = try await self.executeEmbeddingBatch(
                        inputs: symInputs,
                        modelID: self.selectedEmbeddingModelID,
                        dimensions: requestDim,
                        requestID: reqID
                    )
                    guard outputs.count == items.count else {
                        throw ProviderExecutionError.runtimeUnavailable(
                            modelID: self.selectedEmbeddingModelID,
                            detail: "Compare expected \(items.count) vectors, got \(outputs.count)."
                        )
                    }
                    let vecs = outputs.map(\.vector)
                    self.compareRowVectors768 = vecs
                    self.compareColVectors768 = vecs
                    self.compareItemOutputs = outputs
                }

                self.recomputeCompareMatrixScores()
                self.recomputeCompareRankings()
                if self.compareMatrixScores.count >= 2 && self.compareMatrixScores[0].count >= 2 {
                    self.compareCosineScore = self.compareMatrixScores[0][1]
                }
                self.embeddingStatusMessage = "Computed \(items.count)×\(items.count) matrix (\(self.compareMatrixDimension)d active slice)."
            } catch is CancellationError {
                self.embeddingStatusMessage = "Stopped matrix evaluation."
            } catch {
                self.compareMatrixScores = []
                self.compareRankedMatchesToItem1 = []
                self.compareVideoWindowHits = []
                self.embeddingStatusMessage = "Matrix compare failed: \(error.localizedDescription)"
                self.onStatusMessage("Matrix compare failed: \(error.localizedDescription)")
            }
        }
        activeEmbeddingRunTask = task
        await task.value
    }

    public func inspectCompareVector(itemIndex: Int, useColumnVector: Bool = false) -> CompareVectorInspection? {
        let source = useColumnVector ? compareColVectors768 : compareRowVectors768
        guard itemIndex >= 0, itemIndex < source.count, itemIndex < compareRichItems.count else {
            return nil
        }
        let raw = source[itemIndex]
        let projected = Matryoshka.project(raw, dimensions: compareMatrixDimension)
        let sq = projected.reduce(Float(0)) { $0 + $1 * $1 }
        return CompareVectorInspection(
            label: compareRichItems[itemIndex].label,
            dim: projected.count,
            l2Norm: sqrt(sq),
            first16: Array(projected.prefix(16))
        )
    }

    public func scoresAcrossDimensionsForCompareCell(row: Int, col: Int) -> [(dim: Int, score: Float)] {
        guard row >= 0, col >= 0,
              row < compareRowVectors768.count,
              col < compareColVectors768.count else { return [] }
        let rv = compareRowVectors768[row]
        let cv = compareColVectors768[col]
        return availableEmbeddingDimensions.map { d in
            (dim: d, score: Matryoshka.cosine(rv, cv, dimensions: d))
        }
    }

    public func compareMatrixCSV() -> String {
        let n = min(compareRichItems.count, compareMatrixScores.count)
        guard n >= 2 else { return "" }
        func esc(_ s: String) -> String {
            "\"\(s.replacingOccurrences(of: "\"", with: "\"\""))\""
        }
        var lines: [String] = []
        let header = ([""] + compareRichItems.prefix(n).map { esc($0.label) }).joined(separator: ",")
        lines.append(header)
        for r in 0..<n {
            let rowScores = (0..<n).map { c in String(format: "%.4f", compareMatrixScores[r][c]) }
            lines.append(([esc(compareRichItems[r].label)] + rowScores).joined(separator: ","))
        }
        return lines.joined(separator: "\n")
    }

    public func compareMatrixJSON() -> String {
        let data = compareMatrixExportData()
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    public func compareMatrixExportData() -> Data {
        let n = min(compareRichItems.count, compareMatrixScores.count)
        guard n >= 2 else { return Data("{}".utf8) }
        let serializedItems: [[String: Any]] = (0..<n).map { idx in
            let item = compareRichItems[idx]
            var entry: [String: Any] = [
                "label": item.label,
                "prompt_name": item.promptName as Any,
                "blocks": item.blocks.map { block -> [String: Any] in
                    [
                        "kind": block.kind.rawValue,
                        "text": block.text,
                        "file": block.fileURL?.lastPathComponent as Any,
                        "windows": block.videoWindowMode.rawValue,
                    ]
                }
            ]
            if idx < compareItemOutputs.count {
                let out = compareItemOutputs[idx]
                let projected = Matryoshka.project(out.vector, dimensions: compareMatrixDimension)
                entry["vector_preview"] = Array(projected.prefix(16))
                if !out.windows.isEmpty {
                    entry["windows"] = out.windows.map { win -> [String: Any] in
                        [
                            "start": win.start,
                            "end": win.end,
                            "frame_times": win.frameTimes,
                        ]
                    }
                }
            }
            return entry
        }
        let payload: [String: Any] = [
            "model": selectedEmbeddingModelID,
            "dimensions": compareMatrixDimension,
            "mode": compareAsymmetricMode ? "asymmetric_query_document" : "per_item_or_symmetric",
            "items": serializedItems,
            "scores": Array(compareMatrixScores.prefix(n).map { Array($0.prefix(n)) })
        ]
        return (try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])) ?? Data("{}".utf8)
    }

    private func makeLabEntry(from item: CompareRichItem) -> LabItemEntry {
        let textJoin = item.textSummary
        let mediaBlock = item.blocks.first(where: { $0.kind == .media && $0.fileURL != nil })
        return LabItemEntry(
            id: item.id,
            text: textJoin.isEmpty ? nil : textJoin,
            filePath: mediaBlock?.fileURL?.path,
            displayName: item.label,
            windows: mediaBlock?.mediaKind == .video ? (mediaBlock?.videoWindowMode == .wholeVideo ? nil : mediaBlock?.videoWindowMode) : nil,
            windowSeconds: mediaBlock?.parsedWindowSeconds
        )
    }

    public func sendComparePairToLab(row: Int, col: Int) async {
        guard compareRichItems.count >= 2,
              row >= 0, row < compareRichItems.count,
              col >= 0, col < compareRichItems.count,
              row != col else { return }

        let queryEntry = makeLabEntry(from: compareRichItems[row])
        let posEntry = makeLabEntry(from: compareRichItems[col])
        let negEntries = compareRichItems.enumerated()
            .filter { $0.offset != row && $0.offset != col }
            .map { makeLabEntry(from: $0.element) }

        let createdCase = EmbeddingsLabCase(
            id: "compare-case-\(row + 1)-to-\(col + 1)",
            query: queryEntry,
            positives: [posEntry],
            negatives: negEntries
        )
        labSet = EmbeddingsLabSet(
            name: "Promoted from Compare (\(compareRichItems.count) items)",
            baseDirectoryURL: nil,
            cases: [createdCase]
        )
        if labConfigQueue.isEmpty {
            addCurrentConfigToLabQueue()
        }
        embeddingsSubMode = .evalLab
        onStatusMessage("Sent Compare items to Embedding Lab standalone set.")
    }

    public func promoteComparePairToRetrievalLab() async {
        await sendComparePairToLab(row: 0, col: 1)
    }

    public func runPairwiseCompare() {
        Task {
            if compareRichItems.count >= 2 {
                compareRichItems[0].blocks = [CompareBlock(kind: .text, text: compareLeftText)]
                compareRichItems[1].blocks = [CompareBlock(kind: .text, text: compareRightText)]
            }
            await runCompareMatrix()
        }
    }

    // MARK: - Multimodal Embedding Lab (`set.json` + 8-Case CoreGraphics Sample + Sweep)

    public func loadBuiltInMultimodalSampleSet() {
        labSet = Self.makeBuiltInMultimodalSampleSet()
        embeddingStatusMessage = "Loaded built-in 8-case multimodal benchmark set."
        onStatusMessage("Loaded 8-case multimodal sample benchmark set.")
    }

    public func addCurrentConfigToLabQueue(
        modelID: String? = nil,
        dimensions: Int? = nil,
        queryPromptName: String? = nil,
        docPromptName: String? = nil,
        customLabel: String? = nil
    ) {
        let model = modelID ?? selectedEmbeddingModelID
        let dims = dimensions ?? compareMatrixDimension
        let qPrompt = queryPromptName ?? preferredPromptName(matching: "query")
        let dPrompt = docPromptName ?? preferredPromptName(matching: "document", exclude: qPrompt)
        let promptDesc = (qPrompt == nil && dPrompt == nil) ? "no prompts" : "\(qPrompt ?? "—") / \(dPrompt ?? "—")"
        let label = customLabel ?? "\(model) • \(dims)d • \(promptDesc)"
        labConfigQueue.append(
            EmbeddingsLabConfigSpec(
                label: label,
                modelID: model,
                dimensions: dims,
                queryPromptName: qPrompt,
                docPromptName: dPrompt
            )
        )
    }

    public func addPromptsOnVsOffAblation() {
        let model = selectedEmbeddingModelID
        let dims = compareMatrixDimension
        let qPrompt = preferredPromptName(matching: "query") ?? availableEmbeddingPromptNames.first
        let dPrompt = preferredPromptName(matching: "document", exclude: qPrompt) ?? qPrompt
        if let qPrompt {
            labConfigQueue.append(
                EmbeddingsLabConfigSpec(
                    label: "\(model) • \(dims)d • prompts ON (\(qPrompt) / \(dPrompt ?? qPrompt))",
                    modelID: model,
                    dimensions: dims,
                    queryPromptName: qPrompt,
                    docPromptName: dPrompt ?? qPrompt
                )
            )
        }
        labConfigQueue.append(
            EmbeddingsLabConfigSpec(
                label: "\(model) • \(dims)d • prompts OFF",
                modelID: model,
                dimensions: dims,
                queryPromptName: nil,
                docPromptName: nil
            )
        )
    }

    public func removeLabConfig(id: String) {
        labConfigQueue.removeAll { $0.id == id }
    }

    public func clearLabConfigQueue() {
        labConfigQueue.removeAll()
    }

    public func runMultimodalEmbeddingLab() async {
        if selectedBackend == .nativeMLX && isActiveEmbeddingEngineSynthetic {
            let msg = "Embedding Lab requires a real EmbeddingGemma 2 checkpoint at '\(checkpointPath)' (or Remote Server)."
            embeddingStatusMessage = msg
            onStatusMessage(msg)
            return
        }
        guard !labSet.cases.isEmpty else {
            onStatusMessage("Load or import a multimodal benchmark set first.")
            return
        }

        let configsToRun: [EmbeddingsLabConfigSpec]
        if labSweepANELanePresets && selectedBackend == .nativeMLX {
            let base = labConfigQueue.first ?? EmbeddingsLabConfigSpec(
                label: selectedEmbeddingModelID,
                modelID: selectedEmbeddingModelID,
                dimensions: compareMatrixDimension,
                queryPromptName: preferredPromptName(matching: "query"),
                docPromptName: preferredPromptName(matching: "document")
            )
            configsToRun = [ComputePreset.allMLXGPU, .hybridANETowers, .allNeuralEngine].map { preset in
                EmbeddingsLabConfigSpec(
                    label: "\(base.modelID) • \(base.dimensions)d • \(preset.label)",
                    modelID: base.modelID,
                    dimensions: base.dimensions,
                    queryPromptName: base.queryPromptName,
                    docPromptName: base.docPromptName,
                    computePreset: preset
                )
            }
        } else if !labConfigQueue.isEmpty {
            configsToRun = labConfigQueue
        } else {
            let qPrompt = preferredPromptName(matching: "query")
            let dPrompt = preferredPromptName(matching: "document", exclude: qPrompt)
            configsToRun = [
                EmbeddingsLabConfigSpec(
                    label: "\(selectedEmbeddingModelID) • \(compareMatrixDimension)d",
                    modelID: selectedEmbeddingModelID,
                    dimensions: compareMatrixDimension,
                    queryPromptName: qPrompt,
                    docPromptName: dPrompt
                )
            ]
        }

        let reqID = "emb-lab-\(UUID().uuidString.prefix(8))"
        activeEmbeddingRequestID = reqID
        isRunningEvalLab = true
        labSweepResults.removeAll()

        let originalCompute = modelComputeOptions
        let task = Task { @MainActor in
            defer {
                if self.activeEmbeddingRequestID == reqID {
                    self.activeEmbeddingRequestID = nil
                }
                self.isRunningEvalLab = false
                self.labProgressText = nil
                self.activeEmbeddingRunTask = nil
                if self.labSweepANELanePresets && self.modelComputeOptions != originalCompute {
                    self.modelComputeOptions = originalCompute
                    self.setupEmbeddingCoordinator()
                }
            }

            do {
                for (cfgIdx, spec) in configsToRun.enumerated() {
                    try Task.checkCancellation()
                    if let preset = spec.computePreset, self.selectedBackend == .nativeMLX {
                        self.applyComputePreset(preset)
                        await self.syncEmbeddingEngineToRouter()
                    }
                    self.labProgressText = "Running config \(cfgIdx + 1)/\(configsToRun.count): \(spec.label)…"
                    let result = try await self.evaluateMultimodalSet(
                        set: self.labSet,
                        config: spec,
                        requestIDPrefix: "\(reqID)-c\(cfgIdx)"
                    )
                    self.labSweepResults.append(result)
                }
                self.embeddingStatusMessage = "Completed \(self.labSweepResults.count) Embedding Lab run(s)."
            } catch is CancellationError {
                self.embeddingStatusMessage = "Stopped Embedding Lab run."
            } catch {
                self.embeddingStatusMessage = "Embedding Lab failed: \(error.localizedDescription)"
                self.onStatusMessage("Embedding Lab failed: \(error.localizedDescription)")
            }
        }
        activeEmbeddingRunTask = task
        await task.value
    }

    private func buildLabEmbeddingInput(for entry: LabItemEntry, promptName: String?, baseDir: URL?) throws -> EmbeddingInput {
        var parts: [EmbeddingPart] = []
        if let text = entry.text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
            parts.append(.text(text))
        }
        var videoWindows: VideoWindowOptions?
        if let filePath = entry.filePath, !filePath.isEmpty {
            let fileURL: URL
            if filePath.hasPrefix("/") {
                fileURL = URL(fileURLWithPath: filePath)
            } else if let baseDir {
                fileURL = baseDir.appendingPathComponent(filePath)
            } else {
                fileURL = URL(fileURLWithPath: filePath)
            }
            let scoped = fileURL.startAccessingSecurityScopedResource()
            defer {
                if scoped { fileURL.stopAccessingSecurityScopedResource() }
            }
            let kind = ArtifactKind.detect(from: fileURL) ?? .image
            switch kind {
            case .image, .audio:
                parts.append(.mediaURL(kind: kind, url: fileURL))
            case .video:
                let sampling = entry.windows?.sampling ?? .off
                parts.append(.mediaURL(kind: .video, url: fileURL, windowSampling: sampling))
                if let mode = entry.windows, mode != .wholeVideo {
                    videoWindows = VideoWindowOptions(seconds: entry.windowSeconds, sampling: mode.sampling)
                }
            case .document:
                let plans = try ArtifactProcessor.chunkDocument(at: fileURL)
                if let firstText = plans.first(where: { !$0.text.isEmpty })?.text {
                    parts.append(.text(String(firstText.prefix(8000))))
                } else if let firstThumb = plans.first(where: { $0.thumbnailJPEG != nil })?.thumbnailJPEG {
                    parts.append(.media(kind: .image, mimeType: "image/jpeg", data: firstThumb))
                }
            case .text, .code:
                let text = try String(contentsOf: fileURL, encoding: .utf8)
                parts.append(.text(String(text.prefix(8000))))
            }
        }
        if parts.isEmpty {
            parts.append(.text(entry.summaryLabel))
        }
        return EmbeddingInput(parts: parts, promptName: promptName, title: entry.summaryLabel, videoWindows: videoWindows)
    }

    private func evaluateMultimodalSet(
        set: EmbeddingsLabSet,
        config: EmbeddingsLabConfigSpec,
        requestIDPrefix: String
    ) async throws -> EmbeddingsLabRunResult {
        let started = Date()

        // Deduplicate query/doc inputs by cacheSignature + promptName within this config.
        struct PlanItem {
            let key: String
            let entry: LabItemEntry
            let promptName: String?
        }

        var planByKey: [String: PlanItem] = [:]
        var orderedKeys: [String] = []

        func register(_ entry: LabItemEntry, prompt: String?) -> String {
            let key = "\(prompt ?? "")::\(entry.cacheSignature)"
            if planByKey[key] == nil {
                planByKey[key] = PlanItem(key: key, entry: entry, promptName: prompt)
                orderedKeys.append(key)
            }
            return key
        }

        var caseRefs: [(bench: EmbeddingsLabCase, qKey: String, posKeys: [String], negKeys: [String])] = []
        for bench in set.cases {
            let qKey = register(bench.query, prompt: config.queryPromptName)
            let pKeys = bench.positives.map { register($0, prompt: config.docPromptName) }
            let nKeys = bench.negatives.map { register($0, prompt: config.docPromptName) }
            caseRefs.append((bench, qKey, pKeys, nKeys))
        }

        var outputsByKey: [String: EmbeddedOutput] = [:]
        let batchSize = EmbeddingRequestLimits.maxItems
        var offset = 0
        while offset < orderedKeys.count {
            try Task.checkCancellation()
            let end = min(offset + batchSize, orderedKeys.count)
            let sliceKeys = Array(orderedKeys[offset..<end])
            labProgressText = "Embedding \(end)/\(orderedKeys.count) items for \(config.label)…"
            let inputs = try sliceKeys.map { key -> EmbeddingInput in
                let plan = planByKey[key]!
                return try buildLabEmbeddingInput(for: plan.entry, promptName: plan.promptName, baseDir: set.baseDirectoryURL)
            }
            let outputs = try await executeEmbeddingBatch(
                inputs: inputs,
                modelID: config.modelID,
                dimensions: config.dimensions,
                requestID: "\(requestIDPrefix)-b\(offset)"
            )
            guard outputs.count == sliceKeys.count else {
                throw ProviderExecutionError.runtimeUnavailable(
                    modelID: config.modelID,
                    detail: "Lab batch returned \(outputs.count)/\(sliceKeys.count) outputs."
                )
            }
            for (idx, k) in sliceKeys.enumerated() {
                outputsByKey[k] = outputs[idx]
            }
            offset = end
        }

        var caseResults: [EmbeddingsLabCaseResult] = []
        var hitsAt1 = 0
        var hitsAt3 = 0
        var rrSum = 0.0
        var margins: [Double] = []

        for ref in caseRefs {
            guard let qOut = outputsByKey[ref.qKey] else { continue }
            let posScores = ref.posKeys.compactMap { outputsByKey[$0] }.map {
                scoreCandidateOutput(qOut.vector, candidate: $0, dimensions: config.dimensions)
            }
            let negPairs: [(label: String, score: Float)] = zip(ref.bench.negatives, ref.negKeys).compactMap { negEntry, k in
                guard let negOut = outputsByKey[k] else { return nil }
                let s = scoreCandidateOutput(qOut.vector, candidate: negOut, dimensions: config.dimensions)
                return (negEntry.summaryLabel, s)
            }

            let bestPos = posScores.max() ?? -Float.infinity
            let bestNegPair = negPairs.max(by: { $0.score < $1.score })
            let bestNeg = bestNegPair?.score ?? -Float.infinity
            let margin = bestNeg.isFinite ? (bestPos - bestNeg) : bestPos
            let beatenCount = negPairs.filter { $0.score >= bestPos }.count
            let rank = 1 + beatenCount
            let hit1 = rank == 1
            let hit3 = rank <= 3
            let rr = 1.0 / Double(rank)

            if hit1 { hitsAt1 += 1 }
            if hit3 { hitsAt3 += 1 }
            rrSum += rr
            margins.append(Double(margin))

            caseResults.append(
                EmbeddingsLabCaseResult(
                    id: ref.bench.id,
                    querySummary: ref.bench.query.summaryLabel,
                    bestPositiveScore: bestPos,
                    bestNegativeScore: bestNeg,
                    margin: margin,
                    rank: rank,
                    hitAt1: hit1,
                    hitAt3: hit3,
                    reciprocalRank: rr,
                    beatenByLabel: hit1 ? nil : bestNegPair?.label
                )
            )
        }

        let count = max(1, caseResults.count)
        let elapsedMs = Date().timeIntervalSince(started) * 1000.0
        return EmbeddingsLabRunResult(
            id: UUID().uuidString,
            config: config,
            recallAt1: Double(hitsAt1) / Double(count),
            recallAt3: Double(hitsAt3) / Double(count),
            mrr: rrSum / Double(count),
            meanMargin: margins.isEmpty ? 0.0 : margins.reduce(0, +) / Double(margins.count),
            worstMargin: margins.min() ?? 0.0,
            elapsedMs: elapsedMs,
            caseResults: caseResults,
            placementLabel: config.computePreset?.label
        )
    }

    public func embeddingLabExportData() -> Data {
        let runsPayload: [[String: Any]] = labSweepResults.map { run in
            [
                "label": run.config.label,
                "model": run.config.modelID,
                "dimensions": run.config.dimensions,
                "query_prompt": run.config.queryPromptName as Any,
                "doc_prompt": run.config.docPromptName as Any,
                "placement": run.placementLabel as Any,
                "recall_at_1": run.recallAt1,
                "recall_at_3": run.recallAt3,
                "mrr": run.mrr,
                "mean_margin": run.meanMargin,
                "worst_margin": run.worstMargin,
                "elapsed_ms": run.elapsedMs,
                "cases": run.caseResults.map { c -> [String: Any] in
                    [
                        "id": c.id,
                        "query": c.querySummary,
                        "best_pos": c.bestPositiveScore,
                        "best_neg": c.bestNegativeScore,
                        "margin": c.margin,
                        "rank": c.rank,
                        "hit_1": c.hitAt1,
                        "hit_3": c.hitAt3,
                        "beaten_by": c.beatenByLabel as Any,
                    ]
                }
            ]
        }
        let payload: [String: Any] = [
            "set_name": labSet.name,
            "case_count": labSet.cases.count,
            "runs": runsPayload
        ]
        return (try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])) ?? Data("{}".utf8)
    }

    public func runEmbeddingLabEvaluation() async {
        await runMultimodalEmbeddingLab()
    }

    public func evaluateCurrentConfigInLab(customLabel: String? = nil) async {
        if let customLabel {
            addCurrentConfigToLabQueue(customLabel: customLabel)
        }
        await runMultimodalEmbeddingLab()
    }

    public func sweepComputePresetsInLab() async {
        guard selectedBackend == .nativeMLX else {
            await runMultimodalEmbeddingLab()
            return
        }
        let prev = labSweepANELanePresets
        labSweepANELanePresets = true
        await runMultimodalEmbeddingLab()
        labSweepANELanePresets = prev
    }

    public func clearLabConfigRuns() {
        labConfigRuns.removeAll()
        labSweepResults.removeAll()
    }

    public func importBenchmarkSetJSON(from url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer {
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
        do {
            let targetFileURL: URL
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
                targetFileURL = url.appendingPathComponent("set.json")
            } else {
                targetFileURL = url
            }

            let data = try Data(contentsOf: targetFileURL)
            let json = try JSONSerialization.jsonObject(with: data)
            let baseDir = targetFileURL.deletingLastPathComponent()

            let rawArray: [[String: Any]]
            var setName = baseDir.lastPathComponent
            if let arr = json as? [[String: Any]] {
                rawArray = arr
            } else if let dict = json as? [String: Any],
                      let arr = (dict["cases"] as? [[String: Any]]) ?? (dict["queries"] as? [[String: Any]]) {
                rawArray = arr
                if let explicitName = dict["name"] as? String, !explicitName.isEmpty {
                    setName = explicitName
                }
            } else {
                onStatusMessage("Unsupported set.json format.")
                return
            }

            func parseItem(_ raw: Any?, defaultID: String) -> LabItemEntry? {
                if let s = raw as? String {
                    let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
                    return trimmed.isEmpty ? nil : LabItemEntry(id: defaultID, text: trimmed)
                }
                if let dict = raw as? [String: Any] {
                    let text = (dict["text"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
                    let file = (dict["file"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
                    let winStr = dict["windows"] as? String
                    let winMode: CompareVideoWindowMode? = winStr.flatMap(CompareVideoWindowMode.init(rawValue:))
                    let winSec = dict["window_seconds"] as? Double ?? dict["seconds"] as? Double
                    if (text?.isEmpty ?? true) && (file?.isEmpty ?? true) {
                        return nil
                    }
                    return LabItemEntry(id: defaultID, text: text, filePath: file, windows: winMode, windowSeconds: winSec)
                }
                return nil
            }

            var parsedMultimodal: [EmbeddingsLabCase] = []
            var parsedLegacy: [EmbeddingLabRunner.BenchmarkCase] = []

            for (idx, row) in rawArray.enumerated() {
                if let rawPos = row["positives"] as? [Any],
                   let rawNeg = row["negatives"] as? [Any],
                   let qEntry = parseItem(row["query"], defaultID: "q-\(idx)") {
                    let caseID = (row["id"] as? String) ?? "case-\(idx + 1)"
                    let posEntries = rawPos.enumerated().compactMap { parseItem($0.element, defaultID: "\(caseID)-pos-\($0.offset)") }
                    let negEntries = rawNeg.enumerated().compactMap { parseItem($0.element, defaultID: "\(caseID)-neg-\($0.offset)") }
                    if !posEntries.isEmpty && !negEntries.isEmpty {
                        parsedMultimodal.append(
                            EmbeddingsLabCase(id: caseID, query: qEntry, positives: posEntries, negatives: negEntries)
                        )
                    }
                } else if let q = (row["query"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
                          !q.isEmpty {
                    let expected = (row["expected_artifact_id"] as? String)
                        ?? (row["expectedArtifactID"] as? String)
                        ?? (row["expected"] as? String)
                        ?? (row["id"] as? String)
                        ?? ""
                    if !expected.isEmpty {
                        parsedLegacy.append(.init(query: q, expectedArtifactID: expected))
                    }
                }
            }

            if !parsedMultimodal.isEmpty {
                labSet = EmbeddingsLabSet(name: setName, baseDirectoryURL: baseDir, cases: parsedMultimodal)
                embeddingStatusMessage = "Imported \(parsedMultimodal.count) multimodal cases from \(targetFileURL.lastPathComponent)."
                onStatusMessage(embeddingStatusMessage)
            } else if !parsedLegacy.isEmpty {
                benchmarkCases = parsedLegacy
                embeddingStatusMessage = "Imported \(parsedLegacy.count) corpus benchmark cases."
                onStatusMessage(embeddingStatusMessage)
            } else {
                onStatusMessage("No valid multimodal {id, query, positives, negatives} cases found in \(targetFileURL.lastPathComponent).")
            }
        } catch {
            onStatusMessage("Failed to import set.json: \(error.localizedDescription)")
        }
    }

    public func sendHitToPairwiseCompare(_ hit: SearchHit) {
        let q = embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        let leftVal = q.isEmpty ? hit.artifact.title : q
        let rightVal = hit.previewText ?? hit.artifact.title
        compareLeftText = leftVal
        compareRightText = rightVal

        let hitURL = FileManager.default.fileExists(atPath: hit.artifact.url.path) ? hit.artifact.url : nil
        var rightBlocks: [CompareBlock] = []
        if let hitURL {
            rightBlocks.append(
                CompareBlock(
                    kind: .media,
                    fileURL: hitURL,
                    mediaKind: hit.artifact.kind,
                    videoWindowMode: hit.artifact.kind == .video ? .fps : .wholeVideo
                )
            )
        }
        if let preview = hit.previewText, !preview.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            rightBlocks.append(CompareBlock(kind: .text, text: preview))
        }
        if rightBlocks.isEmpty {
            rightBlocks.append(CompareBlock(kind: .text, text: hit.artifact.title))
        }

        compareRichItems = [
            CompareRichItem(label: "Query", blocks: [CompareBlock(kind: .text, text: leftVal)]),
            CompareRichItem(label: hit.artifact.title, blocks: rightBlocks)
        ]
        embeddingsSubMode = .compareMatrix
        Task { await runCompareMatrix() }
    }

    public func pinHitToRetrievalLab(_ hit: SearchHit) async {
        let q = embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        let queryText = q.isEmpty ? hit.artifact.title : q
        if !benchmarkCases.contains(where: { $0.query == queryText && $0.expectedArtifactID == hit.artifact.id }) {
            benchmarkCases.append(.init(query: queryText, expectedArtifactID: hit.artifact.id))
        }
        let posEntry = LabItemEntry(
            id: hit.artifact.id,
            text: hit.previewText,
            filePath: FileManager.default.fileExists(atPath: hit.artifact.url.path) ? hit.artifact.url.path : nil,
            displayName: hit.artifact.title
        )
        let fallbackNegs = labSet.cases.first?.negatives ?? [LabItemEntry(id: "neg-1", text: "Unrelated distractor entry")]
        labSet.cases.append(
            EmbeddingsLabCase(
                id: "pinned-\(hit.artifact.id.prefix(8))",
                query: LabItemEntry(id: "q-\(UUID().uuidString.prefix(6))", text: queryText),
                positives: [posEntry],
                negatives: fallbackNegs
            )
        )
        embeddingsSubMode = .evalLab
    }

    // MARK: - Unified Embedding Execution (Local Engine + Remote HTTP `/v1/embeddings`)

    private func executeEmbeddingBatch(
        inputs: [EmbeddingInput],
        modelID: String,
        dimensions: Int,
        requestID: String
    ) async throws -> [EmbeddedOutput] {
        switch selectedBackend {
        case .heylook:
            guard let baseURL = URL(string: embeddingServerURLString) else {
                throw URLError(.badURL)
            }
            let encodedInputs: [[String: Any]] = try inputs.map { input in
                var blocks: [[String: Any]] = []
                for part in input.parts {
                    switch part {
                    case .text(let s):
                        blocks.append(["type": "text", "text": s])
                    case .media(let kind, let mimeType, let rawData, let url, _):
                        let payloadData: Data
                        if let rawData {
                            payloadData = rawData
                        } else if let url {
                            let scoped = url.startAccessingSecurityScopedResource()
                            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                            payloadData = try Data(contentsOf: url)
                        } else {
                            continue
                        }
                        let blockType: String = switch kind {
                        case .image: "image"
                        case .audio: "audio"
                        case .video: "video"
                        default: "image"
                        }
                        var blockObj: [String: Any] = [
                            "type": blockType,
                            "source": [
                                "type": "base64",
                                "media_type": mimeType,
                                "data": payloadData.base64EncodedString()
                            ]
                        ]
                        if kind == .video, let vw = input.videoWindows {
                            var winObj: [String: Any] = ["sampling": vw.sampling.rawValue]
                            if let sec = vw.seconds {
                                winObj["seconds"] = sec
                            }
                            blockObj["windows"] = winObj
                        }
                        blocks.append(blockObj)
                    }
                }
                var itemObj: [String: Any] = ["content": blocks]
                if input.usesExplicitPromptName {
                    itemObj["prompt_name"] = input.promptName as Any
                }
                return itemObj
            }

            let bodyDict: [String: Any] = [
                "model": modelID,
                "input": encodedInputs,
                "dimensions": dimensions
            ]
            let bodyData = try JSONSerialization.data(withJSONObject: bodyDict)
            var urlReq = URLRequest(url: baseURL.appendingPathComponent("v1").appendingPathComponent("embeddings"))
            urlReq.httpMethod = "POST"
            urlReq.httpBody = bodyData
            urlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
            urlReq.setValue(requestID, forHTTPHeaderField: "X-Request-ID")

            let (data, resp) = try await URLSession.shared.data(for: urlReq)
            if let http = resp as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                let msg = String(data: data, encoding: .utf8) ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
                throw ProviderExecutionError.runtimeUnavailable(modelID: modelID, detail: "HTTP \(http.statusCode): \(msg)")
            }
            let decoded = try JSONDecoder().decode(EmbeddingWireResponse.self, from: data)
            let rows = decoded.data.sorted { $0.index < $1.index }
            return rows.map { row in
                let wins = (row.windows ?? []).map {
                    VideoWindowEmbeddingResult(start: $0.start, end: $0.end, frameTimes: $0.frameTimes, vector: $0.embedding)
                }
                return EmbeddedOutput(
                    vector: row.embedding,
                    windows: wins,
                    inputTokens: decoded.inputTokens,
                    promptName: row.promptName
                )
            }

        case .nativeMLX:
            let engine = try await resolveLocalEngine(for: modelID)
            let execID = router.registry.register(requestID: requestID, modelID: engine.modelID)
            let runnerTask = Task {
                try await engine.embedResult(inputs, dimensions: dimensions).outputs
            }
            router.registry.addCancelHandler(for: requestID, executionID: execID) {
                runnerTask.cancel()
            }
            defer {
                router.registry.finish(requestID: requestID, executionID: execID)
            }
            return try await runnerTask.value
        }
    }

    private func resolveLocalEngine(for modelID: String) async throws -> any EmbeddingEngine {
        if let entry = availableEmbeddingModels.first(where: { $0.id == modelID }),
           let path = entry.modelPath {
            let url = URL(fileURLWithPath: path)
            if entry.provider == .gguf {
                #if os(macOS)
                let sidecar = HeylookConfig.loadSidecar(in: url)
                let mmproj: String? = if case let .string(val)? = sidecar?.extraConfig["mmproj_path"] {
                    val
                } else {
                    nil
                }
                return try GGUFEmbeddingEngine(
                    modelPath: url,
                    mmprojPath: mmproj.map { URL(fileURLWithPath: $0) },
                    modelID: modelID,
                    ctxSize: sidecar?.ctxSize,
                    flashAttention: sidecar?.flashAttn ?? false
                )
                #endif
            } else if path != checkpointPath {
                let candidate = NativeMLXEmbeddingGemma2.makeBestAvailableEngine(
                    checkpointURL: url,
                    computeOptions: modelComputeOptions
                )
                return candidate
            }
        }
        return activeEmbeddingEngine
    }

    // MARK: - CoreGraphics 8-Case Multimodal Sample Generator (`embed-kit.js` v2.0.244)

    private static func makeBuiltInMultimodalSampleSet() -> EmbeddingsLabSet {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("heylook-sample-shapes", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let redCircleURL = dir.appendingPathComponent("red-circle.png")
        let blueSquareURL = dir.appendingPathComponent("blue-square.png")
        let greenTriURL = dir.appendingPathComponent("green-triangle.png")
        let blackCrossURL = dir.appendingPathComponent("black-cross.png")

        writeSampleShapePNG(to: redCircleURL) { ctx in
            ctx.setFillColor(red: 0.863, green: 0.149, blue: 0.149, alpha: 1.0) // #dc2626
            ctx.fillEllipse(in: CGRect(x: 22, y: 22, width: 84, height: 84))
        }
        writeSampleShapePNG(to: blueSquareURL) { ctx in
            ctx.setFillColor(red: 0.145, green: 0.388, blue: 0.922, alpha: 1.0) // #2563eb
            ctx.fill(CGRect(x: 24, y: 24, width: 80, height: 80))
        }
        writeSampleShapePNG(to: greenTriURL) { ctx in
            ctx.setFillColor(red: 0.086, green: 0.639, blue: 0.290, alpha: 1.0) // #16a34a
            ctx.beginPath()
            ctx.move(to: CGPoint(x: 64, y: 20))
            ctx.addLine(to: CGPoint(x: 108, y: 104))
            ctx.addLine(to: CGPoint(x: 20, y: 104))
            ctx.closePath()
            ctx.fillPath()
        }
        writeSampleShapePNG(to: blackCrossURL) { ctx in
            ctx.setFillColor(red: 0.067, green: 0.094, blue: 0.153, alpha: 1.0) // #111827
            ctx.fill(CGRect(x: 54, y: 20, width: 20, height: 88))
            ctx.fill(CGRect(x: 20, y: 54, width: 88, height: 20))
        }

        let cases: [EmbeddingsLabCase] = [
            EmbeddingsLabCase(
                id: "sql-join",
                query: LabItemEntry(id: "q-sql", text: "how do I join two tables on a matching column in SQL"),
                positives: [
                    LabItemEntry(id: "p-sql", text: "SELECT u.name, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id;")
                ],
                negatives: [
                    LabItemEntry(id: "n-sql-1", text: "Python list comprehensions provide a concise way to transform elements with a filter expression."),
                    LabItemEntry(id: "n-sql-2", text: "HTTP status codes in the 4xx range indicate the client sent a request the server could not fulfill.")
                ]
            ),
            EmbeddingsLabCase(
                id: "sourdough",
                query: LabItemEntry(id: "q-bread", text: "why did my sourdough loaf turn out dense and gummy"),
                positives: [
                    LabItemEntry(id: "p-bread", text: "Underproofing happens when the dough is shaped before fermentation has created enough gas, producing large tunnels near the crust and a dense crumb below.")
                ],
                negatives: [
                    LabItemEntry(id: "n-bread-1", text: "The Rust borrow checker enforces exclusive mutable access at compile time without garbage collection."),
                    LabItemEntry(id: "n-bread-2", text: "PostgreSQL indexes store keys in B-tree pages so range scans avoid full table reads.")
                ]
            ),
            EmbeddingsLabCase(
                id: "python-copy",
                query: LabItemEntry(id: "q-py", text: "python iterate list while removing items safely"),
                positives: [
                    LabItemEntry(id: "p-py", text: "Mutating a list during a for-loop skips elements because indices shift. Iterate over a copy or build a new list with a comprehension: items = [x for x in items if keep(x)].")
                ],
                negatives: [
                    LabItemEntry(id: "n-py-1", text: "A JOIN clause combines rows from two relational tables based on a related column."),
                    LabItemEntry(id: "n-py-2", text: "CSS flexbox aligns items along a main axis and a cross axis inside a flex container.")
                ]
            ),
            EmbeddingsLabCase(
                id: "river",
                query: LabItemEntry(id: "q-river", text: "water flowing through a canyon"),
                positives: [
                    LabItemEntry(id: "p-river", text: "A narrow river cuts between steep sandstone walls, carrying snowmelt across smoothed boulders toward the valley below.")
                ],
                negatives: [
                    LabItemEntry(id: "n-river-1", text: "Circuit boards route copper traces between surface-mount resistors and decoupling capacitors."),
                    LabItemEntry(id: "n-river-2", text: "Standard guitar tuning sets six strings to E2 A2 D3 G3 B3 E4 across the fretboard.")
                ]
            ),
            EmbeddingsLabCase(
                id: "text-to-image-red",
                query: LabItemEntry(id: "q-t2i-red", text: "a red circle on a light gray background"),
                positives: [
                    LabItemEntry(id: "p-red", filePath: redCircleURL.path, displayName: "red-circle.png")
                ],
                negatives: [
                    LabItemEntry(id: "n-red-1", filePath: blueSquareURL.path, displayName: "blue-square.png"),
                    LabItemEntry(id: "n-red-2", filePath: greenTriURL.path, displayName: "green-triangle.png")
                ]
            ),
            EmbeddingsLabCase(
                id: "text-to-image-blue",
                query: LabItemEntry(id: "q-t2i-blue", text: "a blue square on a light background"),
                positives: [
                    LabItemEntry(id: "p-blue", filePath: blueSquareURL.path, displayName: "blue-square.png")
                ],
                negatives: [
                    LabItemEntry(id: "n-blue-1", filePath: redCircleURL.path, displayName: "red-circle.png"),
                    LabItemEntry(id: "n-blue-2", filePath: blackCrossURL.path, displayName: "black-cross.png")
                ]
            ),
            EmbeddingsLabCase(
                id: "text-to-image-triangle",
                query: LabItemEntry(id: "q-t2i-tri", text: "a green triangle"),
                positives: [
                    LabItemEntry(id: "p-tri", filePath: greenTriURL.path, displayName: "green-triangle.png")
                ],
                negatives: [
                    LabItemEntry(id: "n-tri-1", filePath: redCircleURL.path, displayName: "red-circle.png"),
                    LabItemEntry(id: "n-tri-2", filePath: blueSquareURL.path, displayName: "blue-square.png")
                ]
            ),
            EmbeddingsLabCase(
                id: "image-to-text",
                query: LabItemEntry(id: "q-i2t", filePath: redCircleURL.path, displayName: "red-circle.png"),
                positives: [
                    LabItemEntry(id: "p-i2t", text: "A simple geometric drawing of a solid red circle centered on a pale gray background.")
                ],
                negatives: [
                    LabItemEntry(id: "n-i2t-1", text: "A blue square printed in the middle of a white sheet."),
                    LabItemEntry(id: "n-i2t-2", text: "A green equilateral triangle pointing straight upward.")
                ]
            )
        ]

        return EmbeddingsLabSet(
            name: "Built-in Multimodal Sample (8 cases)",
            baseDirectoryURL: dir,
            cases: cases
        )
    }

    private static func writeSampleShapePNG(to url: URL, draw: (CGContext) -> Void) {
        if FileManager.default.fileExists(atPath: url.path) { return }
        let width = 128
        let height = 128
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return }

        // Flip to top-left origin matching web canvas coordinates (`embed-kit.js`).
        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1, y: -1)

        // #f5f5f5 light gray backdrop.
        ctx.setFillColor(red: 0.961, green: 0.961, blue: 0.961, alpha: 1.0)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))

        draw(ctx)

        guard let cgImage = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(
                  url as CFURL,
                  UTType.png.identifier as CFString,
                  1,
                  nil
              ) else { return }
        CGImageDestinationAddImage(dest, cgImage, nil)
        CGImageDestinationFinalize(dest)
    }
}
