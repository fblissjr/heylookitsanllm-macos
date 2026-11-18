import Foundation
import Observation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

/// Domain controller for Multimodal Embeddings & ANE Media Finder Studio (`VectorStore`,
/// `IndexCoordinator`, `FolderWatcher`, `N×N` Compare Matrix, ANE Tower Inspector, Retrieval Lab,
/// and `EvalLabSuite`).
///
/// Enforces Truthful Runtime: never silently substitutes synthetic hash vectors for real model output.
@MainActor
@Observable
public final class EmbeddingsStudioController {
    public static let maxCompareMatrixItems: Int = 16

    public let vectorStore: VectorStore
    public let router: HeylookRouter

    public var embeddingsSubMode: EmbeddingsSubMode = .search
    public var selectedBackend: EngineBackend = .nativeMLX
    public var modelComputeOptions: ModelComputeOptions = .hybridANETowers
    public var embeddingServerURLString: String = "http://127.0.0.1:8000"
    public var checkpointPath: String = NativeMLXEmbeddingGemma2.defaultCheckpointURL.path
    public var serverCapabilities: HeylookServerCapabilities?

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

    public var compareLeftText: String = "Swift 6.2 actor isolation for unified memory inference on Apple Silicon"
    public var compareRightText: String = "Running MLX and CoreML ANE vision towers concurrently on Mac Studio"
    public var compareCosineScore: Float?
    public var compareItems: [CompareMatrixItem] = [
        CompareMatrixItem(
            label: "Item 1",
            kind: .text,
            textContent: "Swift 6.2 actor isolation for unified memory inference on Apple Silicon"
        ),
        CompareMatrixItem(
            label: "Item 2",
            kind: .text,
            textContent: "Running MLX and CoreML ANE vision towers concurrently on Mac Studio"
        )
    ]
    public var compareAsymmetricMode: Bool = false
    public var compareMatrixDimension: Int = Matryoshka.gemma2NativeDimensions {
        didSet { recomputeCompareMatrixScores() }
    }
    public private(set) var compareRowVectors768: [[Float]] = []
    public private(set) var compareColVectors768: [[Float]] = []
    public private(set) var compareMatrixScores: [[Float]] = []
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
    public var evalLabSuiteReport: EvalLabSuite.SuiteReport?
    public var isRunningEvalLab: Bool = false

    @ObservationIgnored public private(set) var activeEmbeddingEngine: any EmbeddingEngine
    @ObservationIgnored private var indexCoordinator: IndexCoordinator
    @ObservationIgnored private var embeddingSearchTask: Task<Void, Never>?
    @ObservationIgnored private var indexTask: Task<Void, Never>?
    @ObservationIgnored private var lastInspectedANEKey: String = ""
    #if os(macOS)
    @ObservationIgnored private var folderWatcher: FolderWatcher?
    #endif

    @ObservationIgnored public var onStatusMessage: (String?) -> Void = { _ in }

    public init(vectorStore: VectorStore, router: HeylookRouter) {
        self.vectorStore = vectorStore
        self.router = router
        let initialEngine = NativeMLXEmbeddingGemma2.makeBestAvailableEngine(computeOptions: .hybridANETowers)
        self.activeEmbeddingEngine = initialEngine
        self.indexCoordinator = IndexCoordinator(engine: initialEngine, store: vectorStore)
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
        activeEmbeddingEngine is DeterministicHashEmbeddingEngine
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

    // MARK: - N×N Multimodal Compare Matrix & Multi-Config Retrieval Lab

    public func addCompareTextItem(_ initialText: String = "") {
        guard compareItems.count < Self.maxCompareMatrixItems else {
            onStatusMessage("Compare Matrix supports up to \(Self.maxCompareMatrixItems) items.")
            return
        }
        compareItems.append(
            CompareMatrixItem(
                label: "Item \(compareItems.count + 1)",
                kind: .text,
                textContent: initialText
            )
        )
    }

    public func addCompareMediaFiles(_ urls: [URL]) {
        for url in urls {
            guard compareItems.count < Self.maxCompareMatrixItems else {
                onStatusMessage("Compare Matrix supports up to \(Self.maxCompareMatrixItems) items.")
                break
            }
            let kind = ArtifactKind.detect(from: url) ?? .document
            compareItems.append(
                CompareMatrixItem(
                    label: url.lastPathComponent,
                    kind: kind,
                    textContent: "",
                    fileURL: url
                )
            )
        }
    }

    public func removeCompareItem(id: String) {
        compareItems.removeAll { $0.id == id }
        compareRowVectors768 = []
        compareColVectors768 = []
        compareMatrixScores = []
        selectedCompareCellRow = 0
        selectedCompareCellCol = min(1, max(0, compareItems.count - 1))
    }

    private func buildCompareEmbeddingInput(for item: CompareMatrixItem, task: EmbeddingTask) throws -> EmbeddingInput {
        var parts: [EmbeddingPart] = []
        let trimmedText = item.textContent.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedText.isEmpty {
            parts.append(.text(trimmedText))
        }
        if let fileURL = item.fileURL {
            let scoped = fileURL.startAccessingSecurityScopedResource()
            defer {
                if scoped { fileURL.stopAccessingSecurityScopedResource() }
            }
            let detectedKind = ArtifactKind.detect(from: fileURL) ?? item.kind
            switch detectedKind {
            case .image, .video, .audio:
                parts.append(.mediaURL(kind: detectedKind, url: fileURL, windowSampling: .off))
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
            parts.append(.text(item.label))
        }
        return EmbeddingInput(parts: parts, task: task, title: item.label)
    }

    public func recomputeCompareMatrixScores() {
        let n = compareRowVectors768.count
        guard n >= 2, compareColVectors768.count == n else {
            compareMatrixScores = []
            return
        }
        let dim = compareMatrixDimension
        var grid: [[Float]] = Array(repeating: Array(repeating: 0, count: n), count: n)
        for r in 0..<n {
            for c in 0..<n {
                grid[r][c] = Matryoshka.cosine(compareRowVectors768[r], compareColVectors768[c], dimensions: dim)
            }
        }
        compareMatrixScores = grid
        selectedCompareCellRow = min(max(0, selectedCompareCellRow), n - 1)
        selectedCompareCellCol = min(max(0, selectedCompareCellCol), n - 1)
    }

    public func runCompareMatrix() async {
        guard !isActiveEmbeddingEngineSynthetic else {
            compareMatrixScores = []
            let msg = "N×N Compare Matrix requires a real EmbeddingGemma 2 checkpoint at '\(checkpointPath)' (or Remote Server)."
            embeddingStatusMessage = msg
            onStatusMessage(msg)
            return
        }
        let items = Array(compareItems.prefix(Self.maxCompareMatrixItems))
        guard items.count >= 2 else {
            onStatusMessage("Add at least 2 items to compute an N×N similarity matrix.")
            return
        }

        isEmbeddingBusy = true
        embeddingStatusMessage = "Computing \(items.count)×\(items.count) multimodal matrix…"
        defer { isEmbeddingBusy = false }

        do {
            if compareAsymmetricMode {
                let rowInputs = try items.map { try buildCompareEmbeddingInput(for: $0, task: .query) }
                let colInputs = try items.map { try buildCompareEmbeddingInput(for: $0, task: .document) }
                let rowOutputs = try await activeEmbeddingEngine.embedBatch(
                    rowInputs,
                    dimensions: Matryoshka.gemma2NativeDimensions
                )
                let colOutputs = try await activeEmbeddingEngine.embedBatch(
                    colInputs,
                    dimensions: Matryoshka.gemma2NativeDimensions
                )
                guard rowOutputs.count == items.count, colOutputs.count == items.count else {
                    throw ProviderExecutionError.runtimeUnavailable(
                        modelID: activeEmbeddingEngine.modelID,
                        detail: "Compare matrix vector count mismatch."
                    )
                }
                compareRowVectors768 = rowOutputs.map(\.vector)
                compareColVectors768 = colOutputs.map(\.vector)
            } else {
                let symInputs = try items.map { try buildCompareEmbeddingInput(for: $0, task: .sts) }
                let outputs = try await activeEmbeddingEngine.embedBatch(
                    symInputs,
                    dimensions: Matryoshka.gemma2NativeDimensions
                )
                guard outputs.count == items.count else {
                    throw ProviderExecutionError.runtimeUnavailable(
                        modelID: activeEmbeddingEngine.modelID,
                        detail: "Compare matrix expected \(items.count) vectors, got \(outputs.count)."
                    )
                }
                let vecs = outputs.map(\.vector)
                compareRowVectors768 = vecs
                compareColVectors768 = vecs
            }
            recomputeCompareMatrixScores()
            if compareMatrixScores.count >= 2 && compareMatrixScores[0].count >= 2 {
                compareCosineScore = compareMatrixScores[0][1]
            }
            embeddingStatusMessage = "Computed \(items.count)×\(items.count) cosine matrix (\(compareMatrixDimension)d active slice)."
        } catch {
            compareMatrixScores = []
            embeddingStatusMessage = "Matrix compare failed: \(error.localizedDescription)"
            onStatusMessage("Matrix compare failed: \(error.localizedDescription)")
        }
    }

    public func scoresAcrossDimensionsForCompareCell(row: Int, col: Int) -> [(dim: Int, score: Float)] {
        guard row >= 0, col >= 0,
              row < compareRowVectors768.count,
              col < compareColVectors768.count else { return [] }
        let rv = compareRowVectors768[row]
        let cv = compareColVectors768[col]
        return Matryoshka.gemma2Dimensions.map { d in
            (dim: d, score: Matryoshka.cosine(rv, cv, dimensions: d))
        }
    }

    public func compareMatrixCSV() -> String {
        let n = min(compareItems.count, compareMatrixScores.count)
        guard n >= 2 else { return "" }
        func esc(_ s: String) -> String {
            "\"\(s.replacingOccurrences(of: "\"", with: "\"\""))\""
        }
        var lines: [String] = []
        let header = ([""] + compareItems.prefix(n).map { esc($0.label) }).joined(separator: ",")
        lines.append(header)
        for r in 0..<n {
            let rowScores = (0..<n).map { c in String(format: "%.4f", compareMatrixScores[r][c]) }
            lines.append(([esc(compareItems[r].label)] + rowScores).joined(separator: ","))
        }
        return lines.joined(separator: "\n")
    }

    public func compareMatrixJSON() -> String {
        let n = min(compareItems.count, compareMatrixScores.count)
        guard n >= 2 else { return "{}" }
        let payload: [String: Any] = [
            "dimensions": compareMatrixDimension,
            "mode": compareAsymmetricMode ? "asymmetric_query_document" : "symmetric_sts",
            "labels": Array(compareItems.prefix(n).map(\.label)),
            "scores": Array(compareMatrixScores.prefix(n).map { Array($0.prefix(n)) })
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]),
              let str = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return str
    }

    public func sendComparePairToLab(row: Int, col: Int) async {
        guard row >= 0, col >= 0, row < compareItems.count, col < compareItems.count else { return }
        let rowItem = compareItems[row]
        let colItem = compareItems[col]
        let queryText = rowItem.textContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? rowItem.label
            : rowItem.textContent.trimmingCharacters(in: .whitespacesAndNewlines)
        let targetID = searchHits.first(where: { $0.artifact.title == colItem.label })?.artifact.id
            ?? selectedHit?.artifact.id
            ?? colItem.label
        if !benchmarkCases.contains(where: { $0.query == queryText && $0.expectedArtifactID == targetID }) {
            benchmarkCases.append(.init(query: queryText, expectedArtifactID: targetID))
        }
        embeddingsSubMode = .evalLab
        await evaluateCurrentConfigInLab()
    }

    public func runPairwiseCompare() {
        guard !isActiveEmbeddingEngineSynthetic else {
            compareCosineScore = nil
            let msg = "Pairwise Compare requires a real EmbeddingGemma 2 checkpoint at '\(checkpointPath)' (or Remote Server)."
            embeddingStatusMessage = msg
            onStatusMessage(msg)
            return
        }

        let dim = selectedDimension
        let leftText = compareLeftText
        let rightText = compareRightText
        let engine = activeEmbeddingEngine
        Task { @MainActor in
            do {
                let outputs = try await engine.embedBatch(
                    [.text(leftText, task: .sts), .text(rightText, task: .sts)],
                    dimensions: dim
                )
                guard outputs.count == 2 else {
                    self.compareCosineScore = nil
                    self.embeddingStatusMessage = "Compare error: expected 2 vectors, received \(outputs.count)."
                    return
                }
                self.compareCosineScore = Matryoshka.cosine(outputs[0].vector, outputs[1].vector, dimensions: dim)
            } catch {
                self.compareCosineScore = nil
                self.embeddingStatusMessage = "Compare failed: \(error.localizedDescription)"
                self.onStatusMessage("Compare failed: \(error.localizedDescription)")
            }
        }
    }

    public func runEmbeddingLabEvaluation() async {
        guard !isActiveEmbeddingEngineSynthetic else {
            labMetrics = []
            return
        }
        do {
            labMetrics = try await EmbeddingLabRunner.evaluate(coordinator: indexCoordinator, cases: benchmarkCases)
        } catch {
            labMetrics = []
            embeddingStatusMessage = "Retrieval Lab failed: \(error.localizedDescription)"
        }
    }

    public func evaluateCurrentConfigInLab(customLabel: String? = nil) async {
        guard !isActiveEmbeddingEngineSynthetic else {
            let msg = "Embedding Lab requires a real EmbeddingGemma 2 checkpoint at '\(checkpointPath)' (or Remote Server)."
            embeddingStatusMessage = msg
            onStatusMessage(msg)
            return
        }
        guard !benchmarkCases.isEmpty else {
            onStatusMessage("Add or import at least one benchmark case first.")
            return
        }

        isRunningEvalLab = true
        defer { isRunningEvalLab = false }

        let started = Date()
        do {
            let dims = try await EmbeddingLabRunner.evaluate(coordinator: indexCoordinator, cases: benchmarkCases)
            labMetrics = dims

            var margins: [Double] = []
            for item in benchmarkCases {
                let hits = try await indexCoordinator.search(
                    queryText: item.query,
                    dimensions: Matryoshka.gemma2NativeDimensions,
                    limit: 5
                )
                if hits.count >= 2 {
                    margins.append(Double(hits[0].bestScore - hits[1].bestScore))
                } else if let first = hits.first {
                    margins.append(Double(first.bestScore))
                }
            }
            let elapsedMs = Date().timeIntervalSince(started) * 1000.0
            let meanMargin = margins.isEmpty ? 0.0 : margins.reduce(0, +) / Double(margins.count)
            let worstMargin = margins.min() ?? 0.0
            let split = modelComputeOptions.memorySplitMB
            let ckptName = selectedBackend == .heylook
                ? "Remote (\(embeddingServerURLString))"
                : URL(fileURLWithPath: (checkpointPath as NSString).expandingTildeInPath).lastPathComponent
            let unitBadges = EmbeddingComponent.allCases
                .map { "\(modelComputeOptions[$0].shortBadge)" }
                .joined(separator: "/")
            let runLabel = customLabel ?? "\(ckptName) • \(modelComputeOptions.preset.label)"

            let configRun = EmbeddingLabConfigRun(
                label: runLabel,
                checkpointName: ckptName,
                computePresetLabel: modelComputeOptions.preset.label,
                shortPlacementBadge: unitBadges,
                metricsByDim: dims,
                meanTop1Margin: meanMargin,
                worstTop1Margin: worstMargin,
                evalLatencyMs: elapsedMs,
                aneMemoryMB: split.aneMB,
                gpuMemoryMB: split.gpuMB
            )
            labConfigRuns.append(configRun)
            embeddingStatusMessage = "Evaluated \"\(runLabel)\" in \(Int(elapsedMs.rounded())) ms."
        } catch {
            embeddingStatusMessage = "Lab config run failed: \(error.localizedDescription)"
            onStatusMessage("Lab config run failed: \(error.localizedDescription)")
        }
    }

    public func sweepComputePresetsInLab() async {
        guard selectedBackend == .nativeMLX else {
            await evaluateCurrentConfigInLab(customLabel: "Remote Server")
            return
        }
        guard !isActiveEmbeddingEngineSynthetic else {
            let msg = "Compute sweep requires a real EmbeddingGemma 2 checkpoint at '\(checkpointPath)'."
            embeddingStatusMessage = msg
            onStatusMessage(msg)
            return
        }
        let originalOptions = modelComputeOptions
        let presetsToSweep: [ComputePreset] = [.allMLXGPU, .hybridANETowers, .allNeuralEngine]
        for preset in presetsToSweep {
            applyComputePreset(preset)
            await syncEmbeddingEngineToRouter()
            await evaluateCurrentConfigInLab()
        }
        modelComputeOptions = originalOptions
        setupEmbeddingCoordinator()
        await syncEmbeddingEngineToRouter()
    }

    public func clearLabConfigRuns() {
        labConfigRuns.removeAll()
    }

    public func importBenchmarkSetJSON(from url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer {
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
        do {
            let data = try Data(contentsOf: url)
            let json = try JSONSerialization.jsonObject(with: data)
            let rawItems: [[String: Any]]
            if let arr = json as? [[String: Any]] {
                rawItems = arr
            } else if let dict = json as? [String: Any],
                      let arr = (dict["cases"] as? [[String: Any]]) ?? (dict["queries"] as? [[String: Any]]) {
                rawItems = arr
            } else {
                onStatusMessage("Unsupported benchmark JSON format (expected array of {query, expected_artifact_id}).")
                return
            }

            var imported: [EmbeddingLabRunner.BenchmarkCase] = []
            for entry in rawItems {
                guard let q = (entry["query"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
                      !q.isEmpty else { continue }
                let expected = (entry["expected_artifact_id"] as? String)
                    ?? (entry["expectedArtifactID"] as? String)
                    ?? (entry["expected"] as? String)
                    ?? (entry["id"] as? String)
                    ?? ""
                guard !expected.isEmpty else { continue }
                imported.append(.init(query: q, expectedArtifactID: expected))
            }
            guard !imported.isEmpty else {
                onStatusMessage("No valid {query, expected_artifact_id} cases found in \(url.lastPathComponent).")
                return
            }
            benchmarkCases = imported
            onStatusMessage("Imported \(imported.count) benchmark cases from \(url.lastPathComponent).")
            Task { await evaluateCurrentConfigInLab() }
        } catch {
            onStatusMessage("Failed to import benchmark set: \(error.localizedDescription)")
        }
    }

    public func sendHitToPairwiseCompare(_ hit: SearchHit) {
        let q = embeddingQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        let leftVal = q.isEmpty ? hit.artifact.title : q
        let rightVal = hit.previewText ?? hit.artifact.title
        compareLeftText = leftVal
        compareRightText = rightVal

        compareItems = [
            CompareMatrixItem(label: "Query", kind: .text, textContent: leftVal),
            CompareMatrixItem(
                label: hit.artifact.title,
                kind: hit.artifact.kind,
                textContent: hit.previewText ?? "",
                fileURL: FileManager.default.fileExists(atPath: hit.artifact.url.path) ? hit.artifact.url : nil
            )
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
        embeddingsSubMode = .evalLab
        await evaluateCurrentConfigInLab()
    }

    public func promoteComparePairToRetrievalLab() async {
        guard let hit = selectedHit ?? searchHits.first else { return }
        let queryText = compareLeftText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !queryText.isEmpty else { return }
        if !benchmarkCases.contains(where: { $0.query == queryText && $0.expectedArtifactID == hit.artifact.id }) {
            benchmarkCases.append(.init(query: queryText, expectedArtifactID: hit.artifact.id))
        }
        embeddingsSubMode = .evalLab
        await evaluateCurrentConfigInLab()
    }

    @discardableResult
    public func runFullEvalLabSuite() async -> EvalLabSuite.SuiteReport? {
        isRunningEvalLab = true
        defer { isRunningEvalLab = false }
        do {
            let report = try await EvalLabSuite.runAll()
            evalLabSuiteReport = report
            if !report.matryoshkaBenchmarks.isEmpty {
                labMetrics = report.matryoshkaBenchmarks
            }
            return report
        } catch {
            onStatusMessage("Eval suite error: \(error.localizedDescription)")
            return nil
        }
    }
}
