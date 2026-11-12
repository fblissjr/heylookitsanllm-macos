import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import Observation
import UniformTypeIdentifiers
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

#if os(macOS)
import AppKit
#endif

/// Staged multimodal attachment supporting optional ≤2048px max-edge downscaling (`Shrink` vs `Full res`)
/// and in-session multi-turn replay across Chat turns.
public struct StagedChatAttachment: Identifiable, Sendable, Equatable {
    public let id: String
    public let kind: MessagesRequest.MediaKind
    public let mimeType: String
    public let filename: String
    public let originalData: Data
    public let shrunk2048Data: Data?
    public let shrunkMimeType: String?
    public let originalPixelWidth: Int?
    public let originalPixelHeight: Int?
    public let shrunkPixelWidth: Int?
    public let shrunkPixelHeight: Int?
    public var useShrunk2048: Bool

    public init(
        id: String = UUID().uuidString,
        kind: MessagesRequest.MediaKind,
        mimeType: String,
        filename: String,
        originalData: Data,
        shrunk2048Data: Data? = nil,
        shrunkMimeType: String? = nil,
        originalPixelWidth: Int? = nil,
        originalPixelHeight: Int? = nil,
        shrunkPixelWidth: Int? = nil,
        shrunkPixelHeight: Int? = nil,
        useShrunk2048: Bool? = nil
    ) {
        self.id = id
        self.kind = kind
        self.mimeType = mimeType
        self.filename = filename
        self.originalData = originalData
        self.shrunk2048Data = shrunk2048Data
        self.shrunkMimeType = shrunkMimeType
        self.originalPixelWidth = originalPixelWidth
        self.originalPixelHeight = originalPixelHeight
        self.shrunkPixelWidth = shrunkPixelWidth
        self.shrunkPixelHeight = shrunkPixelHeight
        self.useShrunk2048 = useShrunk2048 ?? (shrunk2048Data != nil)
    }

    public var canToggleShrink: Bool {
        kind == .image && shrunk2048Data != nil
    }

    public var effectiveData: Data {
        if useShrunk2048, let shrunk2048Data {
            return shrunk2048Data
        }
        return originalData
    }

    public var effectiveMimeType: String {
        if useShrunk2048, shrunk2048Data != nil, let shrunkMimeType {
            return shrunkMimeType
        }
        return mimeType
    }

    public var dimensionsLabel: String? {
        if useShrunk2048, let sw = shrunkPixelWidth, let sh = shrunkPixelHeight {
            return "\(sw)×\(sh)"
        }
        if let ow = originalPixelWidth, let oh = originalPixelHeight {
            return "\(ow)×\(oh)"
        }
        return nil
    }

    public var wireAttachment: MessagesRequest.MediaAttachment {
        MessagesRequest.MediaAttachment(
            id: id,
            kind: kind,
            mimeType: effectiveMimeType,
            base64Data: effectiveData.base64EncodedString(),
            filename: filename
        )
    }

    public static func make(
        kind: MessagesRequest.MediaKind,
        mimeType: String,
        filename: String,
        data: Data
    ) -> StagedChatAttachment {
        guard kind == .image,
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = props[kCGImagePropertyPixelWidth] as? Int,
              let height = props[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0 else {
            return StagedChatAttachment(
                kind: kind,
                mimeType: mimeType,
                filename: filename,
                originalData: data
            )
        }

        guard max(width, height) > 2048 else {
            return StagedChatAttachment(
                kind: .image,
                mimeType: mimeType,
                filename: filename,
                originalData: data,
                originalPixelWidth: width,
                originalPixelHeight: height
            )
        }

        let thumbOpts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: 2048
        ]
        guard let cgThumb = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOpts as CFDictionary) else {
            return StagedChatAttachment(
                kind: .image,
                mimeType: mimeType,
                filename: filename,
                originalData: data,
                originalPixelWidth: width,
                originalPixelHeight: height
            )
        }

        let mutableData = NSMutableData()
        let outUTType = mimeType == "image/png" ? UTType.png.identifier : UTType.jpeg.identifier
        let outMime = mimeType == "image/png" ? "image/png" : "image/jpeg"
        if let dest = CGImageDestinationCreateWithData(mutableData, outUTType as CFString, 1, nil) {
            let destProps: [CFString: Any] = outMime == "image/jpeg"
                ? [kCGImageDestinationLossyCompressionQuality: 0.9]
                : [:]
            CGImageDestinationAddImage(dest, cgThumb, destProps as CFDictionary)
            if CGImageDestinationFinalize(dest) {
                return StagedChatAttachment(
                    kind: .image,
                    mimeType: mimeType,
                    filename: filename,
                    originalData: data,
                    shrunk2048Data: mutableData as Data,
                    shrunkMimeType: outMime,
                    originalPixelWidth: width,
                    originalPixelHeight: height,
                    shrunkPixelWidth: cgThumb.width,
                    shrunkPixelHeight: cgThumb.height,
                    useShrunk2048: true
                )
            }
        }

        return StagedChatAttachment(
            kind: .image,
            mimeType: mimeType,
            filename: filename,
            originalData: data,
            originalPixelWidth: width,
            originalPixelHeight: height
        )
    }
}

public enum DestructiveActionPrompt: Identifiable, Equatable, Sendable {
    case clearEmbeddingIndex
    case unloadAllModels
    case deleteConversation(id: String, title: String)
    case deleteNotebook(id: String, title: String)
    case deleteMessage(id: String, conversationID: String)
    case deletePreset(id: String, name: String)
    case clearAllConversationsAndNotebooks

    public var id: String {
        switch self {
        case .clearEmbeddingIndex:
            return "clearEmbeddingIndex"
        case .unloadAllModels:
            return "unloadAllModels"
        case .deleteConversation(let id, _):
            return "deleteConversation:\(id)"
        case .deleteNotebook(let id, _):
            return "deleteNotebook:\(id)"
        case .deleteMessage(let id, _):
            return "deleteMessage:\(id)"
        case .deletePreset(let id, _):
            return "deletePreset:\(id)"
        case .clearAllConversationsAndNotebooks:
            return "clearAllConversationsAndNotebooks"
        }
    }

    public var title: String {
        switch self {
        case .clearEmbeddingIndex:
            return "Clear Embedding Index?"
        case .unloadAllModels:
            return "Unload All Resident Models?"
        case .deleteConversation:
            return "Delete Conversation?"
        case .deleteNotebook:
            return "Delete Notebook?"
        case .deleteMessage:
            return "Delete Message?"
        case .deletePreset:
            return "Delete Preset?"
        case .clearAllConversationsAndNotebooks:
            return "Clear All Conversations & Notebooks?"
        }
    }

    public var confirmLabel: String {
        switch self {
        case .clearEmbeddingIndex:
            return "Clear Index"
        case .unloadAllModels:
            return "Unload All Models"
        case .deleteConversation:
            return "Delete Conversation"
        case .deleteNotebook:
            return "Delete Notebook"
        case .deleteMessage:
            return "Delete Message"
        case .deletePreset:
            return "Delete Preset"
        case .clearAllConversationsAndNotebooks:
            return "Clear All Data"
        }
    }

    public var message: String {
        switch self {
        case .clearEmbeddingIndex:
            return "This removes all indexed artifacts and Matryoshka vectors from VectorStore SQLite."
        case .unloadAllModels:
            return "This unloads all GPU and ANE models, clears Metal/ANE caches, and terminates llama-server subprocesses."
        case .deleteConversation(_, let title):
            return "Permanently delete \"\(title)\" and all of its messages?"
        case .deleteNotebook(_, let title):
            return "Permanently delete notebook \"\(title)\"?"
        case .deleteMessage:
            return "Remove this message from the conversation? Surrounding turns will be preserved."
        case .deletePreset(_, let name):
            return "Permanently delete preset \"\(name)\"?"
        case .clearAllConversationsAndNotebooks:
            return "Permanently delete all conversations, messages, media blobs, turn stats, and notebooks? Saved presets and indexed embeddings are preserved."
        }
    }
}

public enum StructuredOutputMode: String, CaseIterable, Identifiable, Sendable {
    case text = "text"
    case jsonObject = "json_object"
    case jsonSchema = "json_schema"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .text: return "Text"
        case .jsonObject: return "JSON Object"
        case .jsonSchema: return "JSON Schema"
        }
    }
}

public struct StagedImagePlanEstimate: Sendable, Equatable {
    public let resizedWidth: Int
    public let resizedHeight: Int
    public let estimatedTokens: Int
    public let tokenCap: Int

    public init(resizedWidth: Int, resizedHeight: Int, estimatedTokens: Int, tokenCap: Int) {
        self.resizedWidth = resizedWidth
        self.resizedHeight = resizedHeight
        self.estimatedTokens = estimatedTokens
        self.tokenCap = tokenCap
    }
}

public enum EmbeddingsSubMode: String, CaseIterable, Identifiable, Sendable {
    case search = "Media Search"
    case compareMatrix = "N×N Compare"
    case evalLab = "Embedding Lab"

    public var id: String { rawValue }

    public var icon: String {
        switch self {
        case .search: return "magnifyingglass"
        case .compareMatrix: return "square.grid.3x3.fill"
        case .evalLab: return "chart.bar.xaxis"
        }
    }
}

public struct CompareMatrixItem: Identifiable, Sendable, Equatable {
    public let id: String
    public var label: String
    public var kind: ArtifactKind
    public var textContent: String
    public var fileURL: URL?

    public init(
        id: String = UUID().uuidString,
        label: String,
        kind: ArtifactKind = .text,
        textContent: String = "",
        fileURL: URL? = nil
    ) {
        self.id = id
        self.label = label
        self.kind = kind
        self.textContent = textContent
        self.fileURL = fileURL
    }
}

public struct EmbeddingLabConfigRun: Identifiable, Sendable, Equatable {
    public let id: String
    public let label: String
    public let checkpointName: String
    public let computePresetLabel: String
    public let shortPlacementBadge: String
    public let metricsByDim: [EmbeddingLabRunner.DimensionMetrics]
    public let meanTop1Margin: Double
    public let worstTop1Margin: Double
    public let evalLatencyMs: Double
    public let aneMemoryMB: Int
    public let gpuMemoryMB: Int
    public let timestampUnixMs: Int64

    public init(
        id: String = UUID().uuidString,
        label: String,
        checkpointName: String,
        computePresetLabel: String,
        shortPlacementBadge: String,
        metricsByDim: [EmbeddingLabRunner.DimensionMetrics],
        meanTop1Margin: Double,
        worstTop1Margin: Double,
        evalLatencyMs: Double,
        aneMemoryMB: Int,
        gpuMemoryMB: Int,
        timestampUnixMs: Int64 = SQLiteStore.nowMillis()
    ) {
        self.id = id
        self.label = label
        self.checkpointName = checkpointName
        self.computePresetLabel = computePresetLabel
        self.shortPlacementBadge = shortPlacementBadge
        self.metricsByDim = metricsByDim
        self.meanTop1Margin = meanTop1Margin
        self.worstTop1Margin = worstTop1Margin
        self.evalLatencyMs = evalLatencyMs
        self.aneMemoryMB = aneMemoryMB
        self.gpuMemoryMB = gpuMemoryMB
        self.timestampUnixMs = timestampUnixMs
    }
}

public enum PerfTimeWindow: String, CaseIterable, Identifiable, Sendable {
    case oneHour = "1h"
    case sixHours = "6h"
    case twentyFourHours = "24h"
    case sevenDays = "7d"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .oneHour: return "1h"
        case .sixHours: return "6h"
        case .twentyFourHours: return "24h"
        case .sevenDays: return "7d"
        }
    }
}

@MainActor
@Observable
public final class HeylookWorkbenchViewModel {
    public var activeSurface: HeylookSurface = .chat {
        didSet {
            guard oldValue != activeSurface, !isApplyingProgrammaticState else { return }
            handleActiveSurfaceTransition(from: oldValue, to: activeSurface)
        }
    }

    // Core dependencies
    public let store: SQLiteStore
    public let vectorStore: VectorStore
    public let scheduler: ComputeScheduler
    public let registry: RequestRegistry
    public let router: HeylookRouter
    public let server: HeylookHTTPServer
    public var config: HeylookConfig

    // Models & Engine State
    public var registeredModels: [EngineDescription] = [] {
        didSet {
            recomputeFilteredModels()
            recomputeActiveModelDescription()
            refreshASRAvailability()
        }
    }
    public private(set) var generationModels: [EngineDescription] = []
    public private(set) var filteredModels: [EngineDescription] = []
    public private(set) var modelCountsByProvider: [EngineProviderKind: Int] = [:]
    public var selectedModelID: String = "" {
        didSet {
            recomputeActiveModelDescription()
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
    }
    public private(set) var activeModelDescription: EngineDescription?
    public private(set) var loadingModelIDs: Set<String> = []
    public var pendingModelSwitchTargetID: String?
    public var modelSwitchWarningMessage: String?

    public var editingSidecarModel: EngineDescription?
    public var showMLXConverterSheet: Bool = false
    public var convertingSourceModel: EngineDescription?
    public var pendingConfirmation: DestructiveActionPrompt?
    public var memorySnapshot: HardwareMemoryMonitor.MemorySnapshot = HardwareMemoryMonitor.snapshot()
    public var schedulerMetrics: ComputeScheduler.SchedulerMetrics = ComputeScheduler.SchedulerMetrics(
        gpuGateBusy: false,
        gpuQueueDepth: 0,
        maxQueueDepth: 2,
        activeANETasks: 0,
        loadedGPUModels: [],
        loadedANEModels: [],
        pinnedModels: []
    )

    // Chat Workbench State
    public var conversations: [ConversationRecord] = []
    public var conversationSearchQuery: String = ""
    public var selectedConversationID: String?
    public var currentMessages: [MessageRecord] = []
    public var messages: [MessageRecord] {
        get { currentMessages }
        set { currentMessages = newValue }
    }
    public var draftPrompt: String = "" {
        didSet {
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
    }
    public var draftSystemPrompt: String = "" {
        didSet {
            guard !isApplyingProgrammaticState else { return }
            if selectedPresetName != nil {
                isPresetModified = true
            }
            scheduleSystemPromptSave()
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
    }
    public var stagedAttachments: [StagedChatAttachment] = [] {
        didSet {
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
    }
    public var draftAttachments: [MessagesRequest.MediaAttachment] {
        stagedAttachments.map(\.wireAttachment)
    }
    public private(set) var messageAttachmentsByID: [String: [StagedChatAttachment]] = [:]
    public private(set) var lastDroppedMediaCount: Int = 0

    // Structured Output (`response_format`) Controls
    public var responseFormatMode: StructuredOutputMode = .text
    public var responseFormatSchemaName: String = "structured_response"
    public var responseFormatSchemaJSON: String = """
    {
      "type": "object",
      "properties": {
        "answer": { "type": "string" }
      },
      "required": ["answer"],
      "additionalProperties": false
    }
    """
    public var responseFormatStrict: Bool = true

    public var isGenerating: Bool = false
    public private(set) var generatingConversationID: String?
    public var activeRequestID: String?
    public private(set) var isEndingThinking: Bool = false
    public private(set) var streamStatusLine: String?
    public var streamingThinking: String = ""
    public var streamingContent: String = ""
    public var streamingTPS: Double = 0
    public var streamingTokens: Int = 0
    public var showPromptPreview: Bool = false {
        didSet {
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
    }
    public private(set) var promptPreviewText: String = ""
    public private(set) var promptPreviewSource: String = ""
    public private(set) var promptPreviewMarkers: [PromptMarkerSpan] = []
    public private(set) var isRenderingPromptPreview: Bool = false

    public var showLoadPanel: Bool = false
    public var showInspector: Bool = true
    public var selectedProviderFilter: EngineProviderKind? = nil {
        didSet { recomputeFilteredModels() }
    }

    public var isRecordingVoice: Bool = false
    public var autoSendVoiceTurn: Bool = false
    public var selectedASRModelID: String? = nil {
        didSet { refreshASRAvailability() }
    }
    public private(set) var availableASRModels: [EngineDescription] = []
    public private(set) var activeASRBackendLabel: String = "Apple SpeechTranscriber (ANE)"
    public var lastVoiceTranscription: CoreAIProvider.ANESpeechTranscriptionResult?
    public private(set) var resolvedASRModelPath: String?
    public var isVoiceASRConfigured: Bool {
        resolvedASRModelPath != nil
            || CoreAIProvider.isSystemSpeechAvailable
            || connectedRemoteServerURL != nil
    }
    private var pendingVoiceAutoSend: Bool?
    private var voiceRecorder: AVAudioRecorder?
    private var voiceRecordingURL: URL?

    // Honest Optional Sampler & Thinking Cascade Controls (`nil` = Cascade Default)
    public var presets: [PresetRecord] = [] {
        didSet {
            recomputeFilteredPresets()
            if let selID = selectedPresetRecordID, !presets.contains(where: { $0.id == selID }) {
                selectedPresetRecordID = presets.first?.id
            } else if selectedPresetRecordID == nil {
                selectedPresetRecordID = presets.first?.id
            }
        }
    }
    public private(set) var filteredPresets: [PresetRecord] = []
    public var presetSearchQuery: String = "" {
        didSet { recomputeFilteredPresets() }
    }
    public var presetScopeFilter: String? = nil {
        didSet { recomputeFilteredPresets() }
    }
    public var selectedPresetRecordID: String? = nil
    public var selectedPresetName: String? = nil
    public var isPresetModified: Bool = false

    public var temperatureOverride: Double? = nil { didSet { onSamplerOverrideEdited() } }
    public var topPOverride: Double? = nil { didSet { onSamplerOverrideEdited() } }
    public var topKOverride: Int? = nil { didSet { onSamplerOverrideEdited() } }
    public var minPOverride: Double? = nil { didSet { onSamplerOverrideEdited() } }
    public var repetitionPenaltyOverride: Double? = nil { didSet { onSamplerOverrideEdited() } }
    public var repetitionContextSizeOverride: Int? = nil { didSet { onSamplerOverrideEdited() } }
    public var presencePenaltyOverride: Double? = nil { didSet { onSamplerOverrideEdited() } }
    public var maxTokensOverride: Int? = nil { didSet { onSamplerOverrideEdited() } }
    public var seedOverride: Int? = nil { didSet { onSamplerOverrideEdited() } }
    public var stopSequencesOverride: [String]? = nil { didSet { onSamplerOverrideEdited() } }
    public var thinkingEnabledOverride: Bool? = nil { didSet { onSamplerOverrideEdited() } }
    public var thinkingDepthOverride: String? = nil { didSet { onSamplerOverrideEdited() } }
    public var thinkingBudgetOverride: Int? = nil { didSet { onSamplerOverrideEdited() } }

    // Message Edit State
    public var editingMessageID: String?
    public var editingContent: String = ""
    public var editingThinking: String = ""

    // Notebook State
    public var notebooks: [NotebookRecord] = []
    public var selectedNotebookID: String?
    public var notebookTitle: String = "Untitled Notebook" {
        didSet {
            guard !isApplyingProgrammaticState else { return }
            noteNotebookUserEdited()
        }
    }
    public var notebookSystemPrompt: String = "" {
        didSet {
            guard !isApplyingProgrammaticState else { return }
            if selectedPresetName != nil {
                isPresetModified = true
            }
            noteNotebookUserEdited()
        }
    }
    public var notebookBody: String = ""
    public var notebookSelectedRange: NSRange = NSRange(location: 0, length: 0)
    public var isNotebookGenerating: Bool = false
    public private(set) var activeNotebookRequestID: String?
    public private(set) var activeNotebookStreamingID: String?
    public private(set) var isNotebookDirty: Bool = false

    // Embeddings & ANE Media Finder Studio State
    public static let maxCompareMatrixItems: Int = 16
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
        didSet {
            recomputeCompareMatrixScores()
        }
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

    // Perf, Observability Log & GGUF Subprocess Telemetry
    public var selectedPerfWindow: PerfTimeWindow = .twentyFourHours {
        didSet {
            refreshPerfAndObservability()
        }
    }
    public private(set) var perfProfileSummary: SQLiteStore.PerformanceProfileSummary?
    public var observabilityMinLevelFilter: String? = nil {
        didSet { refreshObservabilityEntries() }
    }
    public var observabilityCategoryFilter: String = "" {
        didSet { refreshObservabilityEntries() }
    }
    public var observabilitySearchQuery: String = "" {
        didSet { refreshObservabilityEntries() }
    }
    public private(set) var observabilityEntries: [HeylookObservabilityLog.Entry] = []
    public var boundObservabilityLoggerLevel: String {
        get { HeylookObservabilityLog.shared.level }
        set {
            HeylookObservabilityLog.shared.setLevel(newValue)
            refreshObservabilityEntries()
        }
    }

    public var activeGGUFPorts: [String: Int] = [:] {
        didSet {
            sortedActiveGGUFPorts = activeGGUFPorts
                .sorted { $0.value < $1.value }
                .map { "\($0.key)::\($0.value)" }
        }
    }
    public private(set) var sortedActiveGGUFPorts: [String] = []
    public var activeSubprocessCount: Int = 0

    // Server & Bonjour Remote Connection State
    public var isServerRunning: Bool = false
    public var serverPort: UInt16 = 8000
    public var statusBanner: String?
    public var lastChatError: String?
    public var discoveredServers: [DiscoveredHeylookServer] = []
    public var isBrowsingBonjour: Bool = false
    public var remoteServerURLString: String = "http://mac-studio.local:8000"
    public var connectedRemoteServerURL: String?

    // Private State & Debounce Tasks
    private static let orphanConversationKey = "__orphan__"
    private var isApplyingProgrammaticState: Bool = false
    private var composerDraftByConversationID: [String: String] = [:]
    private var composerMediaByConversationID: [String: [StagedChatAttachment]] = [:]
    private var systemPromptSaveTask: Task<Void, Never>?
    private var dirtySystemPromptConversationID: String?
    private var promptPreviewTask: Task<Void, Never>?
    private var notebookAutoSaveTask: Task<Void, Never>?
    private var embeddingSearchTask: Task<Void, Never>?
    private var metricsLoopTask: Task<Void, Never>?
    private var storeMutationTask: Task<Void, Never>?
    private var lastInspectedANEKey: String = ""

    private let bonjourBrowser = BonjourServiceBrowser()
    private var activeEmbeddingEngine: any EmbeddingEngine
    private var indexCoordinator: IndexCoordinator
    private var indexTask: Task<Void, Never>?
    #if os(macOS)
    private var folderWatcher: FolderWatcher?
    #endif

    public init(autoBootstrap: Bool = true) {
        let dbURL = HeylookConfig.defaultStoreURL
        let vecDbURL = HeylookConfig.defaultVectorStoreURL
        let cfg = HeylookConfig.loadDefault()
        self.config = cfg
        self.store = (try? SQLiteStore(databaseURL: dbURL))
            ?? (try! SQLiteStore(databaseURL: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("heylook_fallback.sqlite")))
        self.vectorStore = (try? VectorStore(databaseURL: vecDbURL))
            ?? (try! VectorStore(databaseURL: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("heylook_vec_fallback.sqlite")))
        self.scheduler = ComputeScheduler(maxLoadedGPUModels: cfg.maxLoadedModels, maxQueueDepth: cfg.maxQueueDepth)
        self.registry = RequestRegistry.shared
        let initialEngine = NativeMLXEmbeddingGemma2.makeBestAvailableEngine(computeOptions: .hybridANETowers)
        self.activeEmbeddingEngine = initialEngine
        self.indexCoordinator = IndexCoordinator(engine: initialEngine, store: self.vectorStore)
        self.router = HeylookRouter(
            store: self.store,
            scheduler: self.scheduler,
            registry: self.registry,
            embeddingEngine: initialEngine,
            bindHost: cfg.serverHost,
            allowedHosts: cfg.allowedHosts,
            adminToken: cfg.adminToken ?? ProcessInfo.processInfo.environment["HEYLOOK_ADMIN_TOKEN"],
            isReadOnlyConfig: HeylookConfig.isModelConfigReadOnly
        )
        self.server = HeylookHTTPServer(
            router: self.router,
            host: cfg.serverHost,
            port: UInt16(cfg.serverPort)
        )
        self.serverPort = UInt16(cfg.serverPort)
        self.refreshASRAvailability()

        if autoBootstrap {
            Task {
                await bootstrap()
            }
        }
    }

    public func bootstrap() async {
        seedDefaultPresetsIfNeeded()

        let discovered = config.discoverConfiguredModels()
        await scheduler.registerDiscoveredModels(discovered)

        registeredModels = await scheduler.listModels()
        if selectedModelID.isEmpty {
            selectedModelID = generationModels.first(where: { $0.isLoaded })?.modelID
                ?? generationModels.first?.modelID
                ?? registeredModels.first?.modelID
                ?? ""
        }

        setupEmbeddingCoordinator()
        await syncEmbeddingEngineToRouter()
        refreshStoreData()
        await refreshMetrics()

        metricsLoopTask?.cancel()
        metricsLoopTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(3))
                guard let self else { break }
                await self.refreshMetrics()
            }
        }

        storeMutationTask?.cancel()
        let stream = store.mutationStream()
        storeMutationTask = Task { [weak self] in
            for await event in stream {
                guard !Task.isCancelled, let self else { break }
                self.handleStoreMutationEvent(event)
            }
        }
    }

    private func handleStoreMutationEvent(_ event: StoreMutationEvent) {
        switch event {
        case .conversationUpserted(let upsertedID):
            conversations = (try? store.listConversations()) ?? conversations
            if selectedConversationID == upsertedID && activeSurface == .chat && !isGenerating {
                if let conv = conversations.first(where: { $0.id == upsertedID }) {
                    isApplyingProgrammaticState = true
                    draftSystemPrompt = conv.systemPrompt
                    applyDocumentSamplingState(params: conv.params, appliedPresetID: conv.appliedPresetID)
                    isApplyingProgrammaticState = false
                }
            }
        case .conversationDeleted:
            conversations = (try? store.listConversations()) ?? conversations
            if let sel = selectedConversationID, !conversations.contains(where: { $0.id == sel }) {
                selectedConversationID = nil
                if let next = conversations.first {
                    selectConversation(next.id)
                } else {
                    currentMessages = []
                }
            }
        case .messagesChanged(let convID):
            if selectedConversationID == convID && !(isGenerating && generatingConversationID == convID) {
                if let loaded = try? loadAndHydrateMessages(conversationID: convID) {
                    currentMessages = loaded
                }
            }
            refreshPerfAndObservability()
        case .notebookUpserted, .notebookDeleted:
            notebooks = (try? store.listNotebooks()) ?? notebooks
        case .presetsChanged:
            presets = (try? store.listPresets()) ?? presets
        case .dataCleared:
            selectedConversationID = nil
            currentMessages = []
            selectedNotebookID = nil
            messageAttachmentsByID.removeAll()
            composerDraftByConversationID.removeAll()
            composerMediaByConversationID.removeAll()
            refreshStoreData()
            refreshPerfAndObservability()
        }
    }

    private func recomputeActiveModelDescription() {
        activeModelDescription = registeredModels.first(where: { $0.modelID == selectedModelID })
    }

    private func recomputeFilteredModels() {
        generationModels = registeredModels.filter { !$0.supportsEmbeddings.value && !$0.isSpeechRecognizer }
        var counts: [EngineProviderKind: Int] = [:]
        for model in registeredModels {
            counts[model.provider, default: 0] += 1
        }
        modelCountsByProvider = counts
        if let provider = selectedProviderFilter {
            filteredModels = registeredModels.filter { $0.provider == provider }
        } else {
            filteredModels = registeredModels
        }
    }

    private func recomputeFilteredPresets() {
        let q = presetSearchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        filteredPresets = presets.filter { preset in
            if let scope = presetScopeFilter, !scope.isEmpty, preset.scope != scope {
                return false
            }
            guard !q.isEmpty else { return true }
            return preset.name.localizedCaseInsensitiveContains(q)
                || preset.description.localizedCaseInsensitiveContains(q)
                || (preset.systemPrompt?.localizedCaseInsensitiveContains(q) ?? false)
        }
    }

    private func handleActiveSurfaceTransition(from oldSurface: HeylookSurface, to newSurface: HeylookSurface) {
        if oldSurface == .notebook {
            flushNotebookSaveIfDirty()
        } else if oldSurface == .chat {
            flushSystemPromptSave()
            persistActiveConversationSamplingState()
        }

        if newSurface == .notebook,
           let nbID = selectedNotebookID,
           let nb = (try? store.getNotebook(id: nbID)) ?? notebooks.first(where: { $0.id == nbID }) {
            isApplyingProgrammaticState = true
            notebookTitle = nb.title
            notebookSystemPrompt = nb.systemPrompt
            if let modelID = nb.modelID, !modelID.isEmpty,
               generationModels.contains(where: { $0.modelID == modelID }) {
                selectedModelID = modelID
            }
            applyDocumentSamplingState(params: nb.params, appliedPresetID: nb.appliedPresetID)
            isApplyingProgrammaticState = false
        } else if newSurface == .chat,
                  let convID = selectedConversationID,
                  let conv = (try? store.getConversation(id: convID)) ?? conversations.first(where: { $0.id == convID }) {
            isApplyingProgrammaticState = true
            draftSystemPrompt = conv.systemPrompt
            if let defModel = conv.defaultModelID, !defModel.isEmpty,
               generationModels.contains(where: { $0.modelID == defModel }) {
                selectedModelID = defModel
            }
            applyDocumentSamplingState(params: conv.params, appliedPresetID: conv.appliedPresetID)
            isApplyingProgrammaticState = false
        } else if newSurface == .perf {
            refreshPerfAndObservability()
        }
    }

    private func onSamplerOverrideEdited() {
        guard !isApplyingProgrammaticState else { return }
        if selectedPresetName != nil {
            isPresetModified = true
        }
        if activeSurface == .notebook {
            noteNotebookUserEdited()
        } else {
            persistActiveConversationSamplingState()
        }
        if showPromptPreview {
            schedulePromptPreviewRefresh()
        }
    }

    private func refreshASRAvailability() {
        let fm = FileManager.default
        var models: [EngineDescription] = registeredModels.filter {
            $0.provider == .coreai && $0.isSpeechRecognizer
        }
        if CoreAIProvider.isSystemSpeechAvailable,
           !models.contains(where: { $0.modelID == CoreAIProvider.systemSpeechModelID }) {
            models.append(CoreAIProvider.systemSpeechDescription())
        }
        availableASRModels = models

        var resolvedPath: String?
        if let envPath = ProcessInfo.processInfo.environment["HEYLOOK_ASR_MODEL_PATH"],
           !envPath.isEmpty,
           fm.fileExists(atPath: (envPath as NSString).expandingTildeInPath) {
            resolvedPath = (envPath as NSString).expandingTildeInPath
        } else if fm.fileExists(atPath: "models/ane-speech-whisper.aimodel") {
            resolvedPath = "models/ane-speech-whisper.aimodel"
        } else if let candidate = registeredModels.first(where: { desc in
            guard desc.isSpeechRecognizer else { return false }
            let expanded = (desc.modelPath as NSString).expandingTildeInPath
            return fm.fileExists(atPath: expanded)
        }) {
            resolvedPath = (candidate.modelPath as NSString).expandingTildeInPath
        }
        resolvedASRModelPath = resolvedPath

        let chosenModel: EngineDescription? = {
            if let explicitID = selectedASRModelID {
                return models.first(where: { $0.modelID == explicitID })
            }
            if CoreAIProvider.isCoreAISpeechLinked,
               let coreAISpeech = models.first(where: { $0.modelID != CoreAIProvider.systemSpeechModelID }) {
                return coreAISpeech
            }
            return models.first(where: { $0.modelID == CoreAIProvider.systemSpeechModelID }) ?? models.first
        }()

        if let chosenModel {
            switch chosenModel.architecture.value.lowercased() {
            case "parakeet_tdt":
                activeASRBackendLabel = "Parakeet TDT (ANE)"
            case "whisper":
                activeASRBackendLabel = "Whisper (ANE)"
            case "apple_speech_transcriber":
                activeASRBackendLabel = "Apple SpeechTranscriber (ANE)"
            default:
                activeASRBackendLabel = "\(chosenModel.modelID) (ANE)"
            }
        } else if resolvedPath != nil {
            activeASRBackendLabel = "CoreAI Speech (ANE)"
        } else if connectedRemoteServerURL != nil {
            activeASRBackendLabel = "Remote Server ASR"
        } else {
            activeASRBackendLabel = "Voice ASR Unavailable"
        }
    }

    public func confirmDestructiveAction(_ action: DestructiveActionPrompt) {
        switch action {
        case .clearEmbeddingIndex:
            clearEmbeddingIndex()
        case .unloadAllModels:
            Task { await unloadAllModels() }
        case .deleteConversation(let id, _):
            deleteConversation(id)
        case .deleteNotebook(let id, _):
            deleteNotebook(id)
        case .deleteMessage(let id, let conversationID):
            deleteSingleMessage(id: id, conversationID: conversationID)
        case .deletePreset(let id, _):
            deletePresetByID(id)
        case .clearAllConversationsAndNotebooks:
            clearAllStoreData()
        }
    }

    // MARK: - Demo Workspace & Metrics

    public func loadSampleDemoWorkspace() async {
        let demos = Self.defaultDemoModels()
        await scheduler.registerDiscoveredModels(demos, allowSyntheticFallback: true)
        registeredModels = await scheduler.listModels()
        if selectedModelID.isEmpty || !generationModels.contains(where: { $0.modelID == selectedModelID }) {
            selectedModelID = generationModels.first?.modelID ?? registeredModels.first?.modelID ?? ""
        }
        await refreshMetrics()
        statusBanner = "Loaded sample demo workspace ([DEMO] synthetic models — configure heylook.toml for local weights)."
    }

    public static func defaultDemoModels() -> [EngineDescription] {
        [
            EngineDescription(
                modelID: "gemma-4-26b-a4b-it-mlx",
                provider: .mlx,
                modelPath: "models/mlx/Gemma-4-26B-A4B-IT",
                architecture: ProvenanceField(value: "gemma4", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "4bit", provenance: .derived, effect: .reload),
                computePlacement: ProvenanceField(value: .hybridANEGPU, provenance: .configured, effect: .reload, computePlacement: .hybridANEGPU),
                isPinned: false,
                isLoaded: true,
                contextSize: ProvenanceField(value: 32768, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
                thinkingControls: ThinkingControls(
                    supportsThinking: true,
                    canDisableThinking: true,
                    defaultEnabled: true,
                    switchVariable: "enable_thinking"
                ),
                estimatedMemoryMB: 14800,
                tags: ["demo", "chat", "vision", "thinking"]
            ),
            EngineDescription(
                modelID: "qwen3.5-0.8b-mlx-8bit",
                provider: .mlx,
                modelPath: "models/mlx/Qwen3.5-0.8B-MLX-8bit",
                architecture: ProvenanceField(value: "qwen3_5", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "8bit", provenance: .derived, effect: .reload),
                computePlacement: ProvenanceField(value: .mlxGPU, provenance: .derived, effect: .reload, computePlacement: .mlxGPU),
                isPinned: false,
                isLoaded: false,
                contextSize: ProvenanceField(value: 32768, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: false, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
                thinkingControls: ThinkingControls(
                    supportsThinking: true,
                    canDisableThinking: true,
                    defaultEnabled: true,
                    switchVariable: "enable_thinking"
                ),
                estimatedMemoryMB: 980,
                tags: ["demo", "fast", "thinking_depth"]
            ),
            EngineDescription(
                modelID: "coreai-llama-3.2-3b-ane",
                provider: .coreai,
                modelPath: "models/coreai/Llama-3.2-3B.aimodel",
                architecture: ProvenanceField(value: "llama", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "int4-palettized", provenance: .derived, effect: .reload),
                computePlacement: ProvenanceField(value: .ane, provenance: .configured, effect: .reload, computePlacement: .ane),
                isPinned: true,
                isLoaded: true,
                contextSize: ProvenanceField(value: 8192, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: false, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
                estimatedMemoryMB: 2100,
                tags: ["demo", "ane", "coreai"]
            ),
            EngineDescription(
                modelID: "embeddinggemma-2-MLX-BF16",
                provider: .mlx,
                modelPath: "models/embeddings/embeddinggemma-2-MLX-BF16",
                architecture: ProvenanceField(value: "embedding_gemma2", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "bf16+ane", provenance: .configured, effect: .reload),
                computePlacement: ProvenanceField(value: .ane, provenance: .configured, effect: .reload, computePlacement: .ane),
                isPinned: true,
                isLoaded: true,
                contextSize: ProvenanceField(value: 8192, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsAudio: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsEmbeddings: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: false, provenance: .derived, effect: .perRequest),
                estimatedMemoryMB: 1488,
                tags: ["demo", "embeddings", "ane", "multimodal"]
            )
        ]
    }

    public func refreshStoreData() {
        do {
            conversations = try store.listConversations()
            presets = try store.listPresets()
            notebooks = try store.listNotebooks()

            if selectedConversationID == nil, let first = conversations.first {
                selectConversation(first.id)
            }
            if selectedNotebookID == nil, let firstNb = notebooks.first {
                isApplyingProgrammaticState = true
                selectedNotebookID = firstNb.id
                notebookTitle = firstNb.title
                notebookSystemPrompt = firstNb.systemPrompt
                notebookBody = firstNb.body
                notebookSelectedRange = NSRange(location: (firstNb.body as NSString).length, length: 0)
                if activeSurface == .notebook {
                    applyDocumentSamplingState(params: firstNb.params, appliedPresetID: firstNb.appliedPresetID)
                }
                isNotebookDirty = false
                isApplyingProgrammaticState = false
            } else if notebooks.isEmpty {
                isApplyingProgrammaticState = true
                notebookTitle = "Untitled Notebook"
                notebookSystemPrompt = ""
                notebookBody = ""
                notebookSelectedRange = NSRange(location: 0, length: 0)
                isNotebookDirty = false
                isApplyingProgrammaticState = false
            }
        } catch {
            statusBanner = "Store error: \(error.localizedDescription)"
        }
    }

    public func refreshMetrics() async {
        memorySnapshot = HardwareMemoryMonitor.snapshot()
        schedulerMetrics = await scheduler.metrics()
        registeredModels = await scheduler.listModels()
        activeGGUFPorts = GGUFPortPool.shared.activeAllocations()
        #if os(macOS)
        activeSubprocessCount = SubprocessLifecycleRegistry.shared.activeSubprocessCount
        #endif
        refreshPerfAndObservability()

        let dir = URL(fileURLWithPath: (checkpointPath as NSString).expandingTildeInPath)
        let opts = modelComputeOptions
        let componentUnits = EmbeddingComponent.allCases.map { opts[$0].rawValue }.joined(separator: ",")
        let aneKey = "\(dir.path)|\(opts.preset.rawValue)|\(componentUnits)"
        if aneKey != lastInspectedANEKey || aneInspectionReports.isEmpty {
            lastInspectedANEKey = aneKey
            if aneInspectionReports.isEmpty {
                aneInspectionReports = ANETowerInspector.statusSnapshot(options: opts, checkpointDirectory: dir)
            }
            let full = await ANETowerInspector.inspect(checkpointDirectory: dir, options: opts)
            if full.count == EmbeddingComponent.allCases.count {
                self.aneInspectionReports = full
            }
        }
    }

    public func refreshPerfAndObservability() {
        perfProfileSummary = try? store.queryPerformanceProfile(window: selectedPerfWindow.rawValue, recentLimit: 50)
        refreshObservabilityEntries()
    }

    public func refreshObservabilityEntries() {
        let cat = observabilityCategoryFilter.trimmingCharacters(in: .whitespacesAndNewlines)
        let search = observabilitySearchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        observabilityEntries = HeylookObservabilityLog.shared.recentEntries(
            limit: 300,
            level: observabilityMinLevelFilter,
            category: cat.isEmpty ? nil : cat,
            search: search.isEmpty ? nil : search
        ).reversed()
    }

    public func clearObservabilityEntries() {
        HeylookObservabilityLog.shared.clear()
        refreshObservabilityEntries()
        statusBanner = "Cleared observability event log."
    }

    public func exportObservabilityJSONL() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let chronological = observabilityEntries.reversed()
        return chronological.compactMap { entry in
            guard let data = try? encoder.encode(entry) else { return nil }
            return String(data: data, encoding: .utf8)
        }.joined(separator: "\n")
    }

    // MARK: - Presets, Per-Document Sampling & Honest 5-Layer Sampler Cascade

    private func seedDefaultPresetsIfNeeded() {
        // SQLiteStore.init seeds default presets once via store_meta ('presets_seeded' = '1'),
        // so deleted built-in presets stay deleted across restarts.
        presets = (try? store.listPresets()) ?? []
        // Never auto-apply a preset on startup so vendor/sidecar defaults remain honest!
    }

    public var unmaskedManualSamplerOverlay: SamplerSettings {
        SamplerSettings(
            temperature: temperatureOverride,
            topP: topPOverride,
            topK: topKOverride,
            minP: minPOverride,
            repetitionPenalty: repetitionPenaltyOverride,
            repetitionContextSize: repetitionContextSizeOverride,
            presencePenalty: presencePenaltyOverride,
            maxTokens: maxTokensOverride,
            seed: seedOverride,
            thinking: thinkingEnabledOverride,
            thinkingDepth: thinkingDepthOverride,
            thinkingBudget: thinkingBudgetOverride,
            stopSequences: stopSequencesOverride
        )
    }

    public var currentManualParamsForPersistence: SamplerSettings? {
        hasAnyManualSamplerOverride ? unmaskedManualSamplerOverlay : nil
    }

    public var activePresetIDForPersistence: String? {
        guard let name = selectedPresetName else { return nil }
        return presets.first(where: { $0.name == name })?.id
    }

    public func applyDocumentSamplingState(params: SamplerSettings?, appliedPresetID: String?) {
        clearManualSamplerOverridesInternal()
        if let presetID = appliedPresetID,
           let preset = presets.first(where: { $0.id == presetID }) {
            selectedPresetName = preset.name
        } else {
            selectedPresetName = nil
        }
        if let p = params {
            temperatureOverride = p.temperature
            topPOverride = p.topP
            topKOverride = p.topK
            minPOverride = p.minP
            repetitionPenaltyOverride = p.repetitionPenalty
            repetitionContextSizeOverride = p.repetitionContextSize
            presencePenaltyOverride = p.presencePenalty
            maxTokensOverride = p.maxTokens
            seedOverride = p.seed
            stopSequencesOverride = p.stopSequences
            thinkingEnabledOverride = p.thinking
            thinkingDepthOverride = p.thinkingDepth
            thinkingBudgetOverride = p.thinkingBudget
            isPresetModified = (selectedPresetName != nil && hasAnyManualSamplerOverride)
        } else {
            isPresetModified = false
        }
    }

    public func persistActiveConversationSamplingState() {
        guard let convID = selectedConversationID,
              var conv = try? store.getConversation(id: convID) else { return }
        let newParams = currentManualParamsForPersistence
        let newPresetID = activePresetIDForPersistence
        guard conv.params != newParams || conv.appliedPresetID != newPresetID else { return }
        conv.params = newParams
        conv.appliedPresetID = newPresetID
        conv.updatedUnixMs = SQLiteStore.nowMillis()
        try? store.upsertConversation(conv)
        conversations = (try? store.listConversations()) ?? conversations
    }

    public func applyPreset(_ preset: PresetRecord) {
        isApplyingProgrammaticState = true
        selectedPresetName = preset.name
        selectedPresetRecordID = preset.id
        isPresetModified = false
        clearManualSamplerOverridesInternal()
        let shouldApplySystemPrompt = (preset.scope == "full") || (preset.systemPrompt != nil)
        if shouldApplySystemPrompt, let sp = preset.systemPrompt {
            if activeSurface == .notebook {
                notebookSystemPrompt = sp
            } else {
                draftSystemPrompt = sp
            }
        }
        isApplyingProgrammaticState = false
        if activeSurface == .notebook {
            saveNotebook()
        } else {
            if shouldApplySystemPrompt && preset.systemPrompt != nil {
                scheduleSystemPromptSave()
            }
            persistActiveConversationSamplingState()
        }
        if showPromptPreview {
            schedulePromptPreviewRefresh()
        }
    }

    public func clearPresetSelection() {
        isApplyingProgrammaticState = true
        selectedPresetName = nil
        isPresetModified = false
        clearManualSamplerOverridesInternal()
        isApplyingProgrammaticState = false
        if activeSurface == .notebook {
            saveNotebook()
        } else {
            persistActiveConversationSamplingState()
        }
        if showPromptPreview {
            schedulePromptPreviewRefresh()
        }
    }

    public var hasAnyManualSamplerOverride: Bool {
        temperatureOverride != nil
            || topPOverride != nil
            || topKOverride != nil
            || minPOverride != nil
            || repetitionPenaltyOverride != nil
            || repetitionContextSizeOverride != nil
            || presencePenaltyOverride != nil
            || maxTokensOverride != nil
            || seedOverride != nil
            || stopSequencesOverride != nil
            || thinkingEnabledOverride != nil
            || thinkingDepthOverride != nil
            || thinkingBudgetOverride != nil
    }

    public func resetManualSamplerOverrides() {
        isApplyingProgrammaticState = true
        clearManualSamplerOverridesInternal()
        isPresetModified = false
        isApplyingProgrammaticState = false
        if activeSurface == .notebook {
            saveNotebook()
        } else {
            persistActiveConversationSamplingState()
        }
        if showPromptPreview {
            schedulePromptPreviewRefresh()
        }
    }

    public func resetSamplerOverrides() {
        resetManualSamplerOverrides()
    }

    private func clearManualSamplerOverridesInternal() {
        temperatureOverride = nil
        topPOverride = nil
        topKOverride = nil
        minPOverride = nil
        repetitionPenaltyOverride = nil
        repetitionContextSizeOverride = nil
        presencePenaltyOverride = nil
        maxTokensOverride = nil
        seedOverride = nil
        stopSequencesOverride = nil
        thinkingEnabledOverride = nil
        thinkingDepthOverride = nil
        thinkingBudgetOverride = nil
    }

    public func updateActivePreset() {
        guard let name = selectedPresetName,
              let existing = presets.first(where: { $0.name == name }) else { return }
        let activeSys = activeSurface == .notebook ? notebookSystemPrompt : draftSystemPrompt
        let trimmedSys = activeSys.trimmingCharacters(in: .whitespacesAndNewlines)
        let updated = PresetRecord(
            id: existing.id,
            name: existing.name,
            scope: existing.scope,
            settings: currentResolvedSamplers.effective,
            systemPrompt: existing.scope == "full" ? (trimmedSys.isEmpty ? nil : activeSys) : existing.systemPrompt,
            description: existing.description
        )
        do {
            let saved = try store.updatePreset(id: existing.id, updated)
            presets = try store.listPresets()
            lastChatError = nil
            applyPreset(saved)
            statusBanner = "Updated preset \"\(saved.name)\"."
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Preset update failed: \(error.localizedDescription)"
        }
    }

    public func saveCurrentAsPreset(name: String, scope: String? = nil, description: String = "") {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let activeSys = activeSurface == .notebook ? notebookSystemPrompt : draftSystemPrompt
        let trimmedSys = activeSys.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedScope = scope ?? (trimmedSys.isEmpty ? "sampler" : "full")
        let record = PresetRecord(
            name: trimmed,
            scope: resolvedScope,
            settings: currentResolvedSamplers.effective,
            systemPrompt: resolvedScope == "full" ? (trimmedSys.isEmpty ? nil : activeSys) : nil,
            description: description
        )
        do {
            let created = try store.createPreset(record)
            presets = try store.listPresets()
            lastChatError = nil
            applyPreset(created)
            statusBanner = "Saved preset \"\(created.name)\"."
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Preset save failed: \(error.localizedDescription)"
        }
    }

    @discardableResult
    public func createNewPresetRecord(
        name: String,
        scope: String = "sampler",
        description: String = "",
        systemPrompt: String? = nil,
        settings: SamplerSettings = SamplerSettings()
    ) -> PresetRecord? {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let cleanSys = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        let record = PresetRecord(
            name: trimmed,
            scope: scope,
            settings: settings,
            systemPrompt: (scope == "full" && !(cleanSys ?? "").isEmpty) ? systemPrompt : nil,
            description: description.trimmingCharacters(in: .whitespacesAndNewlines)
        )
        do {
            let created = try store.createPreset(record)
            presets = try store.listPresets()
            selectedPresetRecordID = created.id
            lastChatError = nil
            statusBanner = "Created preset \"\(created.name)\"."
            return created
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Preset creation failed: \(error.localizedDescription)"
            return nil
        }
    }

    @discardableResult
    public func updatePresetRecord(_ updated: PresetRecord) -> PresetRecord? {
        let previousName = presets.first(where: { $0.id == updated.id })?.name
        do {
            let saved = try store.updatePreset(id: updated.id, updated)
            presets = try store.listPresets()
            selectedPresetRecordID = saved.id
            if let previousName, selectedPresetName == previousName {
                selectedPresetName = saved.name
            }
            lastChatError = nil
            statusBanner = "Saved preset \"\(saved.name)\"."
            return saved
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Preset update failed: \(error.localizedDescription)"
            return nil
        }
    }

    @discardableResult
    public func duplicatePresetRecord(_ source: PresetRecord) -> PresetRecord? {
        var candidateName = "\(source.name) Copy"
        var suffix = 2
        while presets.contains(where: { $0.name.caseInsensitiveCompare(candidateName) == .orderedSame }) {
            candidateName = "\(source.name) Copy \(suffix)"
            suffix += 1
        }
        return createNewPresetRecord(
            name: candidateName,
            scope: source.scope,
            description: source.description,
            systemPrompt: source.systemPrompt,
            settings: source.settings
        )
    }

    public func deletePresetByID(_ id: String) {
        let target = presets.first(where: { $0.id == id })
        do {
            _ = try store.deletePreset(id: id)
            presets = try store.listPresets()
            if let target, selectedPresetName == target.name {
                clearPresetSelection()
            }
            if selectedPresetRecordID == id {
                selectedPresetRecordID = presets.first?.id
            }
            lastChatError = nil
            statusBanner = "Deleted preset \"\(target?.name ?? id)\"."
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Preset delete failed: \(error.localizedDescription)"
        }
    }

    public func deleteActivePreset() {
        guard let name = selectedPresetName,
              let existing = presets.first(where: { $0.name == name }) else { return }
        deletePresetByID(existing.id)
    }

    public var activeModelSupportsThinking: Bool {
        if let controls = activeModelDescription?.thinkingControls {
            return controls.supportsThinking
        }
        return activeModelDescription?.supportsThinking.value ?? false
    }

    public var activeModelCanDisableThinking: Bool {
        if let controls = activeModelDescription?.thinkingControls {
            return controls.canDisableThinking
        }
        return activeModelSupportsThinking
    }

    public var activeModelDepthOptions: [String] {
        activeModelDescription?.thinkingControls?.depthValues ?? []
    }

    public var effectiveThinkingEnabled: Bool {
        guard activeModelSupportsThinking else { return false }
        if !activeModelCanDisableThinking { return true }
        if let explicit = currentResolvedSamplers.effective.thinking {
            return explicit
        }
        return activeModelDescription?.thinkingControls?.defaultEnabled ?? true
    }

    public var effectiveThinkingDepth: String {
        if let explicit = currentResolvedSamplers.effective.thinkingDepth,
           activeModelDepthOptions.contains(explicit) {
            return explicit
        }
        return activeModelDepthOptions.first ?? "medium"
    }

    public var requestSamplerOverlay: SamplerSettings {
        SamplerSettings(
            temperature: temperatureOverride,
            topP: topPOverride,
            topK: topKOverride,
            minP: minPOverride,
            repetitionPenalty: repetitionPenaltyOverride,
            repetitionContextSize: repetitionContextSizeOverride,
            presencePenalty: presencePenaltyOverride,
            maxTokens: maxTokensOverride,
            seed: seedOverride,
            thinking: activeModelSupportsThinking ? thinkingEnabledOverride : nil,
            thinkingDepth: activeModelDepthOptions.isEmpty ? nil : thinkingDepthOverride,
            thinkingBudget: activeModelSupportsThinking ? thinkingBudgetOverride : nil,
            stopSequences: stopSequencesOverride
        )
    }

    public var currentResolvedSamplers: SamplerCascade.ResolvedSampling {
        let activePreset = selectedPresetName.flatMap { name in
            presets.first(where: { $0.name == name })?.settings
        }
        return HeylookConfig.resolveEffectiveSampling(
            modelPath: activeModelDescription?.modelPath,
            preset: activePreset,
            request: requestSamplerOverlay
        )
    }

    // MARK: - Off-MainActor Debounced Prompt Preview

    public func schedulePromptPreviewRefresh() {
        promptPreviewTask?.cancel()
        guard showPromptPreview else { return }

        let activeModel = activeModelDescription
        let sysPrompt = draftSystemPrompt
        let msgs = currentMessages
        let attachmentsMap = messageAttachmentsByID
        let staged = stagedAttachments
        let draft = draftPrompt
        let thinkOn = effectiveThinkingEnabled
        let depthVal = activeModelDepthOptions.isEmpty ? nil : effectiveThinkingDepth

        isRenderingPromptPreview = true
        promptPreviewTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(150))
            guard !Task.isCancelled else { return }

            let dirURL = activeModel.map { HeylookConfig.resolveModelDirectoryURL(from: $0.modelPath) }
            let supportsVis = activeModel?.supportsVision.value
            let supportsAud = activeModel?.supportsAudio.value

            let rendered = await Task.detached(priority: .utility) { () -> (prompt: String, source: String, markers: [PromptMarkerSpan]) in
                var turns: [MessagesRequest.MessageTurn] = msgs.map { msg in
                    let turnStaged = attachmentsMap[msg.id] ?? []
                    return MessagesRequest.MessageTurn(
                        role: msg.role,
                        content: msg.content,
                        thinking: msg.thinking,
                        attachments: turnStaged.map(\.wireAttachment),
                        parts: turnStaged.isEmpty ? msg.contentBlocks : nil
                    )
                }
                let trimmedDraft = draft.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedDraft.isEmpty || !staged.isEmpty {
                    turns.append(
                        MessagesRequest.MessageTurn(
                            role: "user",
                            content: draft,
                            thinking: nil,
                            attachments: staged.map(\.wireAttachment)
                        )
                    )
                }

                let trimmedSys = sysPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                do {
                    let detailed = try HeylookConfig.renderPromptDetailed(
                        in: dirURL,
                        system: trimmedSys.isEmpty ? nil : sysPrompt,
                        turns: turns,
                        thinkingEnabled: thinkOn,
                        thinkingDepth: depthVal,
                        modelSupportsVision: supportsVis,
                        modelSupportsAudio: supportsAud
                    )
                    var badges: [String] = [detailed.source]
                    if detailed.continuation {
                        badges.append("continuation")
                    }
                    if !detailed.markers.isEmpty {
                        badges.append("\(detailed.markers.count) markers")
                    }
                    if detailed.droppedMedia > 0 {
                        badges.append("\(detailed.droppedMedia) media omitted")
                    }
                    if detailed.unrenderedMedia > 0 {
                        badges.append("\(detailed.unrenderedMedia) media unrendered by template")
                    }
                    return (detailed.prompt, badges.joined(separator: " • "), detailed.markers)
                } catch {
                    return ("/* Jinja template error: \(error.localizedDescription) */", "template_error", [])
                }
            }.value

            guard !Task.isCancelled else { return }
            self?.promptPreviewText = rendered.prompt
            self?.promptPreviewSource = rendered.source
            self?.promptPreviewMarkers = rendered.markers
            self?.isRenderingPromptPreview = false
        }
    }

    // MARK: - Model Selection & Multimodal Loss Pre-Switch Guard

    public func selectGenerationModel(_ targetModelID: String) {
        guard targetModelID != selectedModelID else {
            pendingModelSwitchTargetID = nil
            modelSwitchWarningMessage = nil
            return
        }
        guard let target = registeredModels.first(where: { $0.modelID == targetModelID }) else {
            selectedModelID = targetModelID
            return
        }

        let convImages = countConversationAttachments(ofKind: .image)
        let convAudio = countConversationAttachments(ofKind: .audio)
        var dropped: [String] = []
        if convImages > 0 && !target.supportsVision.value {
            dropped.append("\(convImages) image\(convImages == 1 ? "" : "s") (no vision)")
        }
        if convAudio > 0 && !target.supportsAudio.value {
            dropped.append("\(convAudio) audio clip\(convAudio == 1 ? "" : "s") (no audio)")
        }

        if !dropped.isEmpty {
            pendingModelSwitchTargetID = targetModelID
            modelSwitchWarningMessage = "Switching to '\(targetModelID)' will omit \(dropped.joined(separator: " and ")) in this conversation."
            return
        }

        commitModelSelection(targetModelID)
    }

    public func confirmPendingModelSwitch() {
        guard let target = pendingModelSwitchTargetID else { return }
        pendingModelSwitchTargetID = nil
        modelSwitchWarningMessage = nil
        commitModelSelection(target)
    }

    public func cancelPendingModelSwitch() {
        pendingModelSwitchTargetID = nil
        modelSwitchWarningMessage = nil
    }

    private func commitModelSelection(_ targetModelID: String) {
        selectedModelID = targetModelID
        if let convID = selectedConversationID,
           var conv = try? store.getConversation(id: convID) {
            conv.defaultModelID = targetModelID
            conv.updatedUnixMs = Int64(Date().timeIntervalSince1970 * 1000)
            try? store.upsertConversation(conv)
            conversations = (try? store.listConversations()) ?? conversations
        }
    }

    private func countConversationAttachments(ofKind kind: MessagesRequest.MediaKind) -> Int {
        var count = stagedAttachments.filter { $0.kind == kind }.count
        for msg in currentMessages {
            count += (messageAttachmentsByID[msg.id] ?? []).filter { $0.kind == kind }.count
        }
        return count
    }

    public func omittedMediaNote(for messageID: String) -> String? {
        guard let attachments = messageAttachmentsByID[messageID], !attachments.isEmpty else {
            return nil
        }
        let supportsVis = activeModelDescription?.supportsVision.value ?? false
        let supportsAud = activeModelDescription?.supportsAudio.value ?? false
        let droppedImages = supportsVis ? 0 : attachments.filter { $0.kind == .image }.count
        let droppedAudio = supportsAud ? 0 : attachments.filter { $0.kind == .audio }.count
        var parts: [String] = []
        if droppedImages > 0 {
            parts.append("\(droppedImages) image\(droppedImages == 1 ? "" : "s")")
        }
        if droppedAudio > 0 {
            parts.append("\(droppedAudio) audio clip\(droppedAudio == 1 ? "" : "s")")
        }
        guard !parts.isEmpty else { return nil }
        let modelLabel = selectedModelID.isEmpty ? "current model" : selectedModelID
        return "\(parts.joined(separator: " & ")) not sent to \(modelLabel)"
    }

    // MARK: - Per-Conversation State, System Prompt Persistence & Turn Actions

    public var filteredConversations: [ConversationRecord] {
        let q = conversationSearchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return conversations }
        return conversations.filter {
            $0.title.localizedCaseInsensitiveContains(q) || $0.systemPrompt.localizedCaseInsensitiveContains(q)
        }
    }

    private func scheduleSystemPromptSave() {
        guard let convID = selectedConversationID else { return }
        dirtySystemPromptConversationID = convID
        systemPromptSaveTask?.cancel()
        systemPromptSaveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled else { return }
            self?.flushSystemPromptSave()
        }
    }

    public func flushSystemPromptSave() {
        systemPromptSaveTask?.cancel()
        systemPromptSaveTask = nil
        guard let convID = dirtySystemPromptConversationID,
              var conv = try? store.getConversation(id: convID) else {
            dirtySystemPromptConversationID = nil
            return
        }
        dirtySystemPromptConversationID = nil
        guard conv.systemPrompt != draftSystemPrompt else { return }
        conv.systemPrompt = draftSystemPrompt
        conv.updatedUnixMs = Int64(Date().timeIntervalSince1970 * 1000)
        try? store.upsertConversation(conv)
        conversations = (try? store.listConversations()) ?? conversations
    }

    private func stashComposerState(for key: String) {
        composerDraftByConversationID[key] = draftPrompt
        composerMediaByConversationID[key] = stagedAttachments
    }

    private func restoreComposerState(for key: String) {
        draftPrompt = composerDraftByConversationID[key] ?? ""
        stagedAttachments = composerMediaByConversationID[key] ?? []
    }

    public func createNewConversation() {
        flushSystemPromptSave()
        persistActiveConversationSamplingState()
        if let prevID = selectedConversationID {
            stashComposerState(for: prevID)
        }
        do {
            let model = selectedModelID.isEmpty ? (generationModels.first?.modelID ?? "") : selectedModelID
            let conv = ConversationRecord(
                title: "New Chat",
                systemPrompt: "",
                defaultModelID: model.isEmpty ? nil : model
            )
            try store.upsertConversation(conv)
            conversations = try store.listConversations()
            selectedConversationID = conv.id
            currentMessages = []
            editingMessageID = nil
            isApplyingProgrammaticState = true
            draftSystemPrompt = ""
            draftPrompt = ""
            stagedAttachments = []
            applyDocumentSamplingState(params: nil, appliedPresetID: nil)
            isApplyingProgrammaticState = false
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        } catch {
            statusBanner = "Could not create conversation: \(error.localizedDescription)"
        }
    }

    public func selectConversation(_ id: String) {
        if selectedConversationID == id && !currentMessages.isEmpty {
            return
        }
        flushSystemPromptSave()
        persistActiveConversationSamplingState()
        if let prevID = selectedConversationID, prevID != id {
            stashComposerState(for: prevID)
        } else if selectedConversationID == nil {
            stashComposerState(for: Self.orphanConversationKey)
        }

        selectedConversationID = id
        editingMessageID = nil
        pendingModelSwitchTargetID = nil
        modelSwitchWarningMessage = nil

        do {
            currentMessages = try loadAndHydrateMessages(conversationID: id)
        } catch {
            statusBanner = "Failed to load messages: \(error.localizedDescription)"
        }

        isApplyingProgrammaticState = true
        if let conv = conversations.first(where: { $0.id == id }) ?? (try? store.getConversation(id: id)) {
            draftSystemPrompt = conv.systemPrompt
            if let defModel = conv.defaultModelID,
               !defModel.isEmpty,
               generationModels.contains(where: { $0.modelID == defModel }) {
                selectedModelID = defModel
            }
            applyDocumentSamplingState(params: conv.params, appliedPresetID: conv.appliedPresetID)
        } else {
            draftSystemPrompt = ""
            applyDocumentSamplingState(params: nil, appliedPresetID: nil)
        }
        restoreComposerState(for: id)
        isApplyingProgrammaticState = false

        if showPromptPreview {
            schedulePromptPreviewRefresh()
        }
    }

    public func cloneConversation(id: String, upToPosition: Int? = nil, newTitle: String? = nil) {
        flushSystemPromptSave()
        persistActiveConversationSamplingState()
        do {
            let cloned = try store.cloneConversation(id: id, upToPosition: upToPosition, newTitle: newTitle)
            conversations = try store.listConversations()
            selectConversation(cloned.id)
            activeSurface = .chat
            if let upToPosition {
                statusBanner = "Forked conversation at turn #\(upToPosition + 1) → \"\(cloned.title)\"."
            } else {
                statusBanner = "Cloned conversation → \"\(cloned.title)\"."
            }
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Clone failed: \(error.localizedDescription)"
        }
    }

    public func clearAllStoreData() {
        if isGenerating {
            cancelGeneration()
        }
        if isNotebookGenerating {
            cancelNotebookGeneration()
        }
        do {
            try store.clearAllData()
            selectedConversationID = nil
            currentMessages = []
            selectedNotebookID = nil
            messageAttachmentsByID.removeAll()
            composerDraftByConversationID.removeAll()
            composerMediaByConversationID.removeAll()
            refreshStoreData()
            refreshPerfAndObservability()
            statusBanner = "Cleared all conversations, messages, media blobs, stats, and notebooks."
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Clear all data failed: \(error.localizedDescription)"
        }
    }

    public func renameConversation(id: String, newTitle: String) {
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, var conv = try? store.getConversation(id: id) else { return }
        conv.title = trimmed
        conv.updatedUnixMs = Int64(Date().timeIntervalSince1970 * 1000)
        try? store.upsertConversation(conv)
        conversations = (try? store.listConversations()) ?? conversations
    }

    public func deleteConversation(_ id: String) {
        let msgIDs = ((try? store.listMessages(conversationID: id)) ?? []).map(\.id)
        for mID in msgIDs {
            messageAttachmentsByID[mID] = nil
        }
        composerDraftByConversationID[id] = nil
        composerMediaByConversationID[id] = nil
        if dirtySystemPromptConversationID == id {
            dirtySystemPromptConversationID = nil
        }

        _ = try? store.deleteConversation(id: id)
        conversations = (try? store.listConversations()) ?? []
        if selectedConversationID == id {
            selectedConversationID = nil
            if let next = conversations.first {
                selectConversation(next.id)
            } else {
                currentMessages = []
                isApplyingProgrammaticState = true
                draftSystemPrompt = ""
                draftPrompt = ""
                stagedAttachments = []
                applyDocumentSamplingState(params: nil, appliedPresetID: nil)
                isApplyingProgrammaticState = false
            }
        }
    }

    // MARK: - Multimodal Chat Attachments, Image Token Plan, Structured Output & Paste

    public func estimateStagedImagePlan(for attachment: StagedChatAttachment) -> StagedImagePlanEstimate? {
        guard attachment.kind == .image else { return nil }
        let width: Int
        let height: Int
        if attachment.useShrunk2048, let sw = attachment.shrunkPixelWidth, let sh = attachment.shrunkPixelHeight {
            width = sw
            height = sh
        } else if let ow = attachment.originalPixelWidth, let oh = attachment.originalPixelHeight {
            width = ow
            height = oh
        } else {
            return nil
        }
        guard width > 0, height > 0 else { return nil }

        var tokenCap = 280
        if let active = activeModelDescription {
            let dirURL = HeylookConfig.resolveModelDirectoryURL(from: active.modelPath)
            let sidecarCap = loadSidecarConfig(for: active).imageMaxTokens
            tokenCap = max(16, sidecarCap ?? HeylookConfig.estimateImageTokenBudget(in: dirURL))
        }
        let maxPixels = tokenCap * 28 * 28
        let rawPixels = width * height
        let scale = rawPixels > maxPixels ? sqrt(Double(maxPixels) / Double(rawPixels)) : 1.0
        let targetW = max(28, Int((Double(width) * scale / 28.0).rounded() * 28.0))
        let targetH = max(28, Int((Double(height) * scale / 28.0).rounded() * 28.0))
        let estTokens = min(tokenCap, max(16, (targetW / 28) * (targetH / 28)))
        return StagedImagePlanEstimate(
            resizedWidth: targetW,
            resizedHeight: targetH,
            estimatedTokens: estTokens,
            tokenCap: tokenCap
        )
    }

    public var totalStagedImageTokens: Int {
        stagedAttachments.reduce(0) { sum, att in
            guard att.kind == .image else { return sum }
            return sum + (estimateStagedImagePlan(for: att)?.estimatedTokens ?? 280)
        }
    }

    public var responseFormatValidationError: String? {
        guard responseFormatMode == .jsonSchema else { return nil }
        let trimmedName = responseFormatSchemaName.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedName.isEmpty {
            return "Schema name cannot be empty."
        }
        let raw = responseFormatSchemaJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = raw.data(using: .utf8) else {
            return "Invalid UTF-8 in JSON schema."
        }
        guard let decoded = try? JSONDecoder().decode(JSONValue.self, from: data),
              case .object = decoded else {
            return "JSON Schema must be a valid JSON object ({ ... })."
        }
        return nil
    }

    public func buildValidatedResponseFormat() throws -> ResponseFormat? {
        switch responseFormatMode {
        case .text:
            return nil
        case .jsonObject:
            return ResponseFormat(type: "json_object")
        case .jsonSchema:
            if let err = responseFormatValidationError {
                throw ProviderExecutionError.invalidRequest(detail: err)
            }
            let data = Data(responseFormatSchemaJSON.utf8)
            let schemaObj = try JSONDecoder().decode(JSONValue.self, from: data)
            let trimmedName = responseFormatSchemaName.trimmingCharacters(in: .whitespacesAndNewlines)
            return ResponseFormat(
                type: "json_schema",
                jsonSchema: ResponseFormat.JSONSchemaSpec(
                    name: trimmedName,
                    schema: schemaObj,
                    strict: responseFormatStrict
                )
            )
        }
    }

    public func attachChatMedia(from urls: [URL]) {
        for url in urls {
            attachChatMediaFile(url: url)
        }
    }

    public func attachChatMediaFile(url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer {
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
        guard let data = try? Data(contentsOf: url), !data.isEmpty else {
            statusBanner = "Unable to read attachment '\(url.lastPathComponent)'."
            return
        }
        let ext = url.pathExtension.lowercased()
        let audioExts: Set<String> = ["wav", "mp3", "m4a", "aac", "flac"]
        let imageExts: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "gif", "tiff"]

        let kind: MessagesRequest.MediaKind
        let mime: String
        if audioExts.contains(ext) {
            kind = .audio
            switch ext {
            case "mp3": mime = "audio/mp3"
            case "m4a", "aac": mime = "audio/mp4"
            case "flac": mime = "audio/flac"
            default: mime = "audio/wav"
            }
        } else if imageExts.contains(ext) {
            kind = .image
            switch ext {
            case "png": mime = "image/png"
            case "webp": mime = "image/webp"
            case "gif": mime = "image/gif"
            default: mime = "image/jpeg"
            }
        } else {
            statusBanner = "Unsupported chat attachment '\(url.lastPathComponent)' (attach an image or audio file)."
            return
        }

        stagedAttachments.append(
            StagedChatAttachment.make(
                kind: kind,
                mimeType: mime,
                filename: url.lastPathComponent,
                data: data
            )
        )
    }

    #if os(macOS)
    @discardableResult
    public func pasteAttachmentFromClipboard() -> Bool {
        let pb = NSPasteboard.general
        if let urls = pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL],
           !urls.isEmpty {
            let validExts: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "gif", "tiff", "wav", "mp3", "m4a", "aac", "flac"]
            let mediaURLs = urls.filter { validExts.contains($0.pathExtension.lowercased()) }
            if !mediaURLs.isEmpty {
                for url in mediaURLs {
                    attachChatMediaFile(url: url)
                }
                return true
            }
        }
        if let pngData = pb.data(forType: .png), !pngData.isEmpty {
            let stamp = Int(Date().timeIntervalSince1970)
            stagedAttachments.append(
                StagedChatAttachment.make(
                    kind: .image,
                    mimeType: "image/png",
                    filename: "pasted-\(stamp).png",
                    data: pngData
                )
            )
            return true
        }
        if let tiffData = pb.data(forType: .tiff),
           let bitmap = NSBitmapImageRep(data: tiffData),
           let pngData = bitmap.representation(using: .png, properties: [:]) {
            let stamp = Int(Date().timeIntervalSince1970)
            stagedAttachments.append(
                StagedChatAttachment.make(
                    kind: .image,
                    mimeType: "image/png",
                    filename: "pasted-\(stamp).png",
                    data: pngData
                )
            )
            return true
        }
        return false
    }
    #endif

    public func toggleAttachmentShrink(id: String) {
        guard let idx = stagedAttachments.firstIndex(where: { $0.id == id }),
              stagedAttachments[idx].canToggleShrink else { return }
        stagedAttachments[idx].useShrunk2048.toggle()
    }

    public func removeDraftAttachment(id: String) {
        stagedAttachments.removeAll { $0.id == id }
    }

    public func removeChatAttachment(id: String) {
        removeDraftAttachment(id: id)
    }

    public var chatAttachmentCapabilityWarning: String? {
        guard !stagedAttachments.isEmpty else { return nil }
        guard let desc = activeModelDescription else {
            return "Select a generation model before sending media attachments."
        }
        let hasImage = stagedAttachments.contains { $0.kind == .image }
        let hasAudio = stagedAttachments.contains { $0.kind == .audio }
        if hasImage && !desc.supportsVision.value {
            return "Model '\(desc.modelID)' does not support image inputs (supports_vision = false)."
        }
        if hasAudio && !desc.supportsAudio.value {
            return "Model '\(desc.modelID)' does not support audio inputs (supports_audio = false)."
        }
        return nil
    }

    @discardableResult
    public func transcribeVoiceToComposer(
        wavData: Data,
        languageHint: String = "en",
        autoSend: Bool? = nil
    ) async -> CoreAIProvider.ANESpeechTranscriptionResult? {
        guard isVoiceASRConfigured else {
            statusBanner = "Voice input unavailable: neither Apple SpeechTranscriber nor a CoreAISpeech .aimodel bundle is available."
            return nil
        }

        do {
            let transcript: CoreAIProvider.ANESpeechTranscriptionResult
            if let remoteBase = connectedRemoteServerURL,
               let baseURL = URL(string: remoteBase) {
                var urlReq = URLRequest(url: baseURL.appendingPathComponent("v1/audio/transcriptions"))
                urlReq.httpMethod = "POST"
                urlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
                let payload = AudioTranscriptionRequest(
                    audioBase64: wavData.base64EncodedString(),
                    languageHint: languageHint,
                    model: selectedASRModelID
                )
                urlReq.httpBody = try JSONEncoder().encode(payload)
                let (respData, resp) = try await URLSession.shared.data(for: urlReq)
                if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                    let msg = String(decoding: respData, as: UTF8.self)
                    throw ProviderExecutionError.runtimeUnavailable(
                        modelID: selectedASRModelID ?? "remote-asr",
                        detail: "Remote /v1/audio/transcriptions HTTP \(http.statusCode): \(msg)"
                    )
                }
                transcript = try JSONDecoder().decode(CoreAIProvider.ANESpeechTranscriptionResult.self, from: respData)
            } else {
                transcript = try await scheduler.transcribeVoice(
                    wavData: wavData,
                    languageHint: languageHint,
                    preferredASRModelID: selectedASRModelID
                )
            }

            lastVoiceTranscription = transcript
            await refreshMetrics()

            let cleanedText = transcript.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleanedText.isEmpty else {
                statusBanner = "No speech detected in recording."
                return transcript
            }

            let trimmedDraft = draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedDraft.isEmpty {
                draftPrompt = cleanedText
            } else {
                draftPrompt = "\(trimmedDraft) \(cleanedText)"
            }

            let shouldAutoSend = autoSend ?? autoSendVoiceTurn
            if shouldAutoSend {
                await sendChatPrompt()
            } else {
                statusBanner = String(
                    format: "Dictated %.1fs via %@ (%@).",
                    transcript.durationSeconds,
                    transcript.modelID,
                    transcript.computePlacement.rawValue.uppercased()
                )
            }
            return transcript
        } catch {
            statusBanner = "ANE voice transcription failed: \(error.localizedDescription)"
            return nil
        }
    }

    @discardableResult
    public func sendVoiceTurn(
        wavData: Data,
        languageHint: String = "en"
    ) async -> CoreAIProvider.ANESpeechTranscriptionResult? {
        await transcribeVoiceToComposer(
            wavData: wavData,
            languageHint: languageHint,
            autoSend: true
        )
    }

    public func toggleVoiceRecording(autoSendOnStop: Bool? = nil) {
        Task { await performToggleVoiceRecording(autoSendOnStop: autoSendOnStop) }
    }

    private func performToggleVoiceRecording(autoSendOnStop: Bool? = nil) async {
        if isRecordingVoice {
            voiceRecorder?.stop()
            voiceRecorder = nil
            isRecordingVoice = false
            let resolvedAutoSend = autoSendOnStop ?? pendingVoiceAutoSend ?? autoSendVoiceTurn
            pendingVoiceAutoSend = nil
            guard let url = voiceRecordingURL else { return }
            voiceRecordingURL = nil
            defer { try? FileManager.default.removeItem(at: url) }
            if let wavData = try? Data(contentsOf: url), wavData.count > 44 {
                _ = await transcribeVoiceToComposer(wavData: wavData, autoSend: resolvedAutoSend)
            }
            return
        }

        guard isVoiceASRConfigured else {
            statusBanner = "Voice input unavailable: no ASR runtime configured."
            return
        }

        let granted = await AVAudioApplication.requestRecordPermission()
        guard granted else {
            statusBanner = "Microphone permission denied. Grant Microphone access in System Settings > Privacy & Security."
            return
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("heylook-voice-\(UUID().uuidString).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16_000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false
        ]
        do {
            #if os(iOS)
            let session = AVAudioSession.sharedInstance()
            try? session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try? session.setActive(true)
            #endif
            let recorder = try AVAudioRecorder(url: tempURL, settings: settings)
            recorder.prepareToRecord()
            if recorder.record() {
                pendingVoiceAutoSend = autoSendOnStop
                voiceRecordingURL = tempURL
                voiceRecorder = recorder
                isRecordingVoice = true
                statusBanner = "Recording 16 kHz mono voice input (\(activeASRBackendLabel))…"
            }
        } catch {
            statusBanner = "Microphone capture failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Chat Send, Streaming & Turn Actions

    public func sendChatPrompt() async {
        let promptText = draftPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard (!promptText.isEmpty || !stagedAttachments.isEmpty), !isGenerating else { return }

        // Hard capability gate — never send when staged media violates active model capabilities!
        if let warning = chatAttachmentCapabilityWarning {
            statusBanner = warning
            return
        }
        if let schemaErr = responseFormatValidationError {
            statusBanner = "Structured output schema invalid: \(schemaErr)"
            return
        }

        let modelID = selectedModelID.isEmpty ? (generationModels.first?.modelID ?? "") : selectedModelID
        guard !modelID.isEmpty else {
            statusBanner = "No generation model selected or configured. Add scan_folders in heylook.toml or load the sample workspace."
            return
        }

        if let existingConvID = selectedConversationID, registry.isConversationGenerating(existingConvID) {
            statusBanner = "Conversation is already generating (HTTP client or another session holds the conversation lock)."
            return
        }

        flushSystemPromptSave()
        let pendingStaged = stagedAttachments

        if selectedConversationID == nil {
            let conv = ConversationRecord(
                title: "New Chat",
                systemPrompt: draftSystemPrompt,
                defaultModelID: modelID,
                params: currentManualParamsForPersistence,
                appliedPresetID: activePresetIDForPersistence
            )
            try? store.upsertConversation(conv)
            conversations = (try? store.listConversations()) ?? []
            selectedConversationID = conv.id
            currentMessages = []
        } else {
            persistActiveConversationSamplingState()
        }
        guard let convID = selectedConversationID else { return }

        draftPrompt = ""
        stagedAttachments = []
        composerDraftByConversationID[convID] = ""
        composerMediaByConversationID[convID] = []
        composerDraftByConversationID[Self.orphanConversationKey] = nil
        composerMediaByConversationID[Self.orphanConversationKey] = nil

        do {
            let nextPos = (currentMessages.map(\.position).max() ?? -1) + 1
            let fallbackAttachmentLabel = pendingStaged.map(\.filename).joined(separator: ", ")
            let storedContent = promptText.isEmpty ? "[Attached: \(fallbackAttachmentLabel)]" : promptText
            let contentBlocks: [MessagesRequest.MessageTurn.ContentPart]? = pendingStaged.isEmpty
                ? nil
                : (pendingStaged.map { .media($0.wireAttachment) } + (promptText.isEmpty ? [] : [.text(promptText)]))
            let userMsg = MessageRecord(
                conversationID: convID,
                position: nextPos,
                role: "user",
                content: storedContent,
                modelID: modelID,
                contentBlocks: contentBlocks
            )
            if !pendingStaged.isEmpty {
                messageAttachmentsByID[userMsg.id] = pendingStaged
            }
            try store.appendMessage(userMsg)
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            if var conv = try store.getConversation(id: convID), conv.title == "New Chat" {
                conv.title = String(storedContent.prefix(44))
                conv.defaultModelID = modelID
                conv.updatedUnixMs = Int64(Date().timeIntervalSince1970 * 1000)
                try store.upsertConversation(conv)
                conversations = try store.listConversations()
            }
            await streamAssistantReply(for: convID, modelOverride: modelID)
        } catch {
            statusBanner = "Chat error: \(error.localizedDescription)"
        }
    }

    private func loadAndHydrateMessages(conversationID: String) throws -> [MessageRecord] {
        let loaded = try store.listMessages(conversationID: conversationID, hydrateMedia: true)
        for msg in loaded {
            guard messageAttachmentsByID[msg.id] == nil, let blocks = msg.contentBlocks else { continue }
            var hydrated: [StagedChatAttachment] = []
            for block in blocks {
                if case .media(let att) = block,
                   !att.dataBase64.isEmpty,
                   let data = Data(base64Encoded: att.dataBase64),
                   !data.isEmpty {
                    let kind: MessagesRequest.MediaKind = (att.kind == "audio") ? .audio : .image
                    hydrated.append(
                        StagedChatAttachment.make(
                            kind: kind,
                            mimeType: att.mediaType,
                            filename: att.filename ?? (kind == .audio ? "audio.wav" : "image.jpg"),
                            data: data
                        )
                    )
                }
            }
            if !hydrated.isEmpty {
                messageAttachmentsByID[msg.id] = hydrated
            }
        }
        return loaded
    }

    private static func updatingTextInContentBlocks(
        _ existingBlocks: [MessagesRequest.MessageTurn.ContentPart]?,
        newContent: String,
        newThinking: String?
    ) -> [MessagesRequest.MessageTurn.ContentPart]? {
        guard let existingBlocks else { return nil }
        var rebuilt: [MessagesRequest.MessageTurn.ContentPart] = []
        if let th = newThinking, !th.isEmpty {
            rebuilt.append(.thinking(th))
        }
        for block in existingBlocks {
            if case .media = block {
                rebuilt.append(block)
            }
        }
        if !newContent.isEmpty || rebuilt.isEmpty {
            rebuilt.append(.text(newContent))
        }
        return rebuilt
    }

    public func beginEditingMessage(_ message: MessageRecord) {
        editingMessageID = message.id
        editingContent = message.content
        editingThinking = message.thinking ?? ""
    }

    public func cancelEditingMessage() {
        editingMessageID = nil
        editingContent = ""
        editingThinking = ""
    }

    /// Saves edits to `content` and `thinking` in place WITHOUT deleting downstream turns.
    public func saveEditedMessageInPlace(_ message: MessageRecord) {
        guard let convID = selectedConversationID else { return }
        if registry.isConversationGenerating(convID) {
            statusBanner = "Cannot edit while conversation is actively generating."
            return
        }
        do {
            let trimmedThinking = editingThinking.trimmingCharacters(in: .whitespacesAndNewlines)
            var updated = message
            updated.content = editingContent
            if message.role == "assistant" {
                updated.thinking = trimmedThinking.isEmpty ? nil : editingThinking
            }
            updated.contentBlocks = Self.updatingTextInContentBlocks(
                updated.contentBlocks,
                newContent: updated.content,
                newThinking: updated.thinking
            )
            try store.appendMessage(updated)
            cancelEditingMessage()
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Failed to save message edit: \(error.localizedDescription)"
        }
    }

    /// Saves edits, truncates turns after `message`, and generates a fresh assistant reply.
    public func saveAndRegenerateEditedMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating else { return }
        if registry.isConversationGenerating(convID) {
            statusBanner = "Conversation is already generating."
            return
        }
        do {
            if message.role == "assistant" {
                cancelEditingMessage()
                try store.deleteMessages(conversationID: convID, fromPosition: message.position)
                currentMessages = try loadAndHydrateMessages(conversationID: convID)
                await streamAssistantReply(for: convID)
                return
            }

            var updated = message
            updated.content = editingContent
            updated.contentBlocks = Self.updatingTextInContentBlocks(
                updated.contentBlocks,
                newContent: updated.content,
                newThinking: updated.thinking
            )
            try store.appendMessage(updated)
            try store.deleteMessages(conversationID: convID, fromPosition: message.position + 1)
            cancelEditingMessage()
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            await streamAssistantReply(for: convID)
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Failed to regenerate message: \(error.localizedDescription)"
        }
    }

    /// Saves edits, truncates turns after `message`, and continues generation (supporting assistant prefill!).
    public func saveAndContinueEditedMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating else { return }
        if registry.isConversationGenerating(convID) {
            statusBanner = "Conversation is already generating."
            return
        }
        do {
            let trimmedThinking = editingThinking.trimmingCharacters(in: .whitespacesAndNewlines)
            var updated = message
            updated.content = editingContent
            if message.role == "assistant" {
                updated.thinking = trimmedThinking.isEmpty ? nil : editingThinking
            }
            updated.contentBlocks = Self.updatingTextInContentBlocks(
                updated.contentBlocks,
                newContent: updated.content,
                newThinking: updated.thinking
            )
            try store.appendMessage(updated)
            try store.deleteMessages(conversationID: convID, fromPosition: message.position + 1)
            cancelEditingMessage()
            currentMessages = try loadAndHydrateMessages(conversationID: convID)

            if updated.role == "assistant" {
                await streamAssistantReply(for: convID, continuingAssistantMessage: updated)
            } else {
                await streamAssistantReply(for: convID)
            }
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Failed to continue edited message: \(error.localizedDescription)"
        }
    }

    public func regenerateAssistantMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating else { return }
        if registry.isConversationGenerating(convID) {
            statusBanner = "Conversation is already generating."
            return
        }
        do {
            try store.deleteMessages(conversationID: convID, fromPosition: message.position)
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            await streamAssistantReply(for: convID)
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Failed to regenerate turn: \(error.localizedDescription)"
        }
    }

    public func continueAssistantMessage(_ message: MessageRecord) async {
        guard let convID = selectedConversationID, !isGenerating, message.role == "assistant" else { return }
        if registry.isConversationGenerating(convID) {
            statusBanner = "Conversation is already generating."
            return
        }
        do {
            try store.deleteMessages(conversationID: convID, fromPosition: message.position + 1)
            currentMessages = try loadAndHydrateMessages(conversationID: convID)
            await streamAssistantReply(for: convID, continuingAssistantMessage: message)
        } catch {
            lastChatError = error.localizedDescription
            statusBanner = "Failed to continue assistant turn: \(error.localizedDescription)"
        }
    }

    /// Deletes a single message row while preserving all subsequent turns at contiguous positions.
    public func deleteSingleMessage(id: String, conversationID: String) {
        if registry.isConversationGenerating(conversationID) {
            statusBanner = "Cannot delete messages while conversation is actively generating."
            return
        }
        _ = try? store.deleteMessage(conversationID: conversationID, messageID: id)
        messageAttachmentsByID[id] = nil
        if editingMessageID == id {
            cancelEditingMessage()
        }
        if selectedConversationID == conversationID {
            currentMessages = (try? loadAndHydrateMessages(conversationID: conversationID)) ?? []
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
        }
    }

    public func streamAssistantReply(
        for convID: String,
        modelOverride: String? = nil,
        continuingAssistantMessage: MessageRecord? = nil
    ) async {
        let modelID = modelOverride
            ?? (selectedModelID.isEmpty ? (generationModels.first?.modelID ?? "") : selectedModelID)
        guard !modelID.isEmpty else {
            statusBanner = "No generation model selected or configured. Add scan_folders in heylook.toml or load the sample workspace."
            return
        }

        let reqID = "req_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12))"
        guard let execID = registry.tryAcquireConversationLock(
            conversationID: convID,
            requestID: reqID,
            modelID: modelID
        ) else {
            statusBanner = "Conversation '\(convID)' is already generating (shared lock active)."
            HeylookObservabilityLog.shared.record(
                level: "WARN",
                category: "ui.chat",
                message: "Blocked concurrent generation on locked conversation",
                requestID: reqID,
                metadata: ["conversation_id": convID, "model": modelID]
            )
            return
        }
        defer {
            registry.releaseConversationLock(
                conversationID: convID,
                requestID: reqID,
                executionID: execID
            )
        }

        let requestStartedAt = Date()
        activeRequestID = reqID
        isGenerating = true
        generatingConversationID = convID
        isEndingThinking = false
        streamingThinking = continuingAssistantMessage?.thinking ?? ""
        streamingContent = continuingAssistantMessage?.content ?? ""
        streamingTPS = 0
        streamingTokens = 0

        let targetDesc = await scheduler.modelDescription(for: modelID)
        if targetDesc?.isLoaded == false {
            streamStatusLine = "Loading \(modelID)… (first token follows model load)"
        } else {
            streamStatusLine = "Waiting for first token…"
        }

        do {
            let convMessages = try loadAndHydrateMessages(conversationID: convID)
            if let cont = continuingAssistantMessage, selectedConversationID == convID {
                currentMessages = convMessages.filter { $0.id != cont.id }
            }

            let supportsVis = targetDesc?.supportsVision.value ?? false
            let supportsAud = targetDesc?.supportsAudio.value ?? false
            var droppedMedia = 0

            let turns: [MessagesRequest.MessageTurn] = convMessages.map { msg in
                let staged = messageAttachmentsByID[msg.id] ?? []
                if !staged.isEmpty {
                    let allowedWire = staged.compactMap { att -> MessagesRequest.MediaAttachment? in
                        if att.kind == .image && !supportsVis {
                            droppedMedia += 1
                            return nil
                        }
                        if att.kind == .audio && !supportsAud {
                            droppedMedia += 1
                            return nil
                        }
                        return att.wireAttachment
                    }
                    return MessagesRequest.MessageTurn(
                        role: msg.role,
                        content: msg.content,
                        thinking: msg.thinking,
                        attachments: allowedWire
                    )
                } else if let blocks = msg.contentBlocks, !blocks.isEmpty {
                    var keptBlocks: [MessagesRequest.MessageTurn.ContentPart] = []
                    for block in blocks {
                        if case .media(let att) = block {
                            if att.kind == "image" && !supportsVis {
                                droppedMedia += 1
                                continue
                            }
                            if att.kind == "audio" && !supportsAud {
                                droppedMedia += 1
                                continue
                            }
                        }
                        keptBlocks.append(block)
                    }
                    return MessagesRequest.MessageTurn(
                        role: msg.role,
                        content: msg.content,
                        thinking: msg.thinking,
                        attachments: [],
                        parts: keptBlocks
                    )
                } else {
                    return MessagesRequest.MessageTurn(
                        role: msg.role,
                        content: msg.content,
                        thinking: msg.thinking,
                        attachments: []
                    )
                }
            }
            lastDroppedMediaCount = droppedMedia
            if droppedMedia > 0 {
                HeylookObservabilityLog.shared.record(
                    level: "INFO",
                    category: "ui.chat",
                    message: "Omitted \(droppedMedia) historical media attachment(s) unsupported by \(modelID)",
                    requestID: reqID,
                    metadata: ["model": modelID, "dropped_media": "\(droppedMedia)"]
                )
            }

            let provider = try await scheduler.provider(for: modelID)
            let resolved = currentResolvedSamplers
            let effective = resolved.effective
            let convSystemPrompt = (try? store.getConversation(id: convID))?.systemPrompt ?? draftSystemPrompt
            let trimmedSys = convSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            let validatedResponseFormat = try buildValidatedResponseFormat()

            let req = MessagesRequest(
                model: modelID,
                messages: turns,
                system: trimmedSys.isEmpty ? nil : convSystemPrompt,
                maxTokens: effective.maxTokens,
                temperature: effective.temperature,
                topP: effective.topP,
                topK: effective.topK,
                minP: effective.minP,
                repetitionPenalty: effective.repetitionPenalty,
                repetitionContextSize: effective.repetitionContextSize,
                presencePenalty: effective.presencePenalty,
                seed: effective.seed,
                stream: true,
                thinking: activeModelSupportsThinking ? effectiveThinkingEnabled : nil,
                thinkingDepth: activeModelDepthOptions.isEmpty ? nil : effectiveThinkingDepth,
                thinkingBudget: activeModelSupportsThinking ? effective.thinkingBudget : nil,
                stopSequences: effective.stopSequences,
                responseFormat: validatedResponseFormat
            )

            let finalChunk = try await scheduler.withComputeLease(for: modelID) { [registry] () throws -> GenerationChunk in
                var lastChunk = GenerationChunk()
                let stream = await provider.generate(
                    requestID: reqID,
                    request: req,
                    effectiveSampling: effective,
                    registry: registry
                )
                for try await chunk in stream {
                    await MainActor.run {
                        if !chunk.thinking.isEmpty || !chunk.text.isEmpty {
                            self.streamStatusLine = nil
                        } else if chunk.promptTokens > 0 && self.streamStatusLine != nil {
                            self.streamStatusLine = "Prefilling \(chunk.promptTokens) prompt tokens…"
                        }
                        if !chunk.thinking.isEmpty {
                            self.streamingThinking.append(chunk.thinking)
                        }
                        if !chunk.text.isEmpty {
                            self.isEndingThinking = false
                            self.streamingContent.append(chunk.text)
                        }
                        if chunk.tokenCount > 0 { self.streamingTokens = chunk.tokenCount }
                        if chunk.generationTPS > 0 { self.streamingTPS = chunk.generationTPS }
                    }
                    lastChunk = chunk
                }
                return lastChunk
            }

            let requestDurationMs = Date().timeIntervalSince(requestStartedAt) * 1000.0

            if !streamingContent.isEmpty || !streamingThinking.isEmpty {
                let savedMessageID: String
                if var cont = continuingAssistantMessage {
                    cont.content = streamingContent
                    cont.thinking = streamingThinking.isEmpty ? nil : streamingThinking
                    cont.contentBlocks = Self.updatingTextInContentBlocks(
                        cont.contentBlocks,
                        newContent: cont.content,
                        newThinking: cont.thinking
                    )
                    cont.modelID = modelID
                    cont.stopReason = (finalChunk.stopReason ?? .endTurn).rawValue
                    cont.tokenCount = finalChunk.tokenCount > 0 ? finalChunk.tokenCount : cont.tokenCount
                    cont.ttftMs = finalChunk.ttftMs ?? cont.ttftMs
                    cont.cacheReport = finalChunk.cacheReport ?? cont.cacheReport
                    cont.speculativeReport = finalChunk.speculativeReport ?? cont.speculativeReport
                    cont.generationTPS = finalChunk.generationTPS > 0 ? finalChunk.generationTPS : cont.generationTPS
                    cont.computePlacement = finalChunk.activeComputePlacement
                    try store.appendMessage(cont)
                    savedMessageID = cont.id
                } else {
                    let nextPos = (convMessages.map(\.position).max() ?? -1) + 1
                    let assistantMsg = MessageRecord(
                        conversationID: convID,
                        position: nextPos,
                        role: "assistant",
                        content: streamingContent,
                        thinking: streamingThinking.isEmpty ? nil : streamingThinking,
                        modelID: modelID,
                        stopReason: finalChunk.stopReason ?? .endTurn,
                        cacheReport: finalChunk.cacheReport,
                        speculativeReport: finalChunk.speculativeReport,
                        generationTPS: finalChunk.generationTPS,
                        computePlacement: finalChunk.activeComputePlacement,
                        tokenCount: finalChunk.tokenCount > 0 ? finalChunk.tokenCount : nil,
                        ttftMs: finalChunk.ttftMs
                    )
                    try store.appendMessage(assistantMsg)
                    savedMessageID = assistantMsg.id
                }

                let stats = MessageStatsRecord(
                    messageID: savedMessageID,
                    conversationID: convID,
                    modelID: modelID,
                    promptTokens: finalChunk.promptTokens,
                    outputTokens: finalChunk.tokenCount,
                    thinkingTokens: finalChunk.thinkingTokens,
                    contentTokens: finalChunk.contentTokens,
                    cachedTokens: finalChunk.cacheReport?.cachedTokens ?? 0,
                    promptTPS: finalChunk.promptTPS,
                    generationTPS: finalChunk.generationTPS,
                    ttftMs: finalChunk.ttftMs,
                    thinkingDurationMs: finalChunk.thinkingDurationMs,
                    contentDurationMs: finalChunk.contentDurationMs,
                    generationDurationMs: finalChunk.generationDurationMs,
                    requestDurationMs: requestDurationMs
                )
                try? store.upsertMessageStats(stats)
                HeylookObservabilityLog.shared.record(
                    level: "INFO",
                    category: "ui.chat",
                    message: "Completed assistant turn (\(finalChunk.tokenCount) tok @ \(String(format: "%.1f", finalChunk.generationTPS)) tok/s)",
                    requestID: reqID,
                    metadata: [
                        "conversation_id": convID,
                        "model": modelID,
                        "output_tokens": "\(finalChunk.tokenCount)",
                        "prompt_tokens": "\(finalChunk.promptTokens)",
                        "stop_reason": (finalChunk.stopReason ?? .endTurn).rawValue
                    ]
                )
            }
            if selectedConversationID == convID {
                currentMessages = try loadAndHydrateMessages(conversationID: convID)
            }
        } catch {
            statusBanner = "Generation failed: \(error.localizedDescription)"
            HeylookObservabilityLog.shared.record(
                level: "ERROR",
                category: "ui.chat",
                message: "Chat generation failed: \(error.localizedDescription)",
                requestID: reqID,
                metadata: ["conversation_id": convID, "model": modelID]
            )
            if selectedConversationID == convID {
                currentMessages = (try? loadAndHydrateMessages(conversationID: convID)) ?? currentMessages
            }
        }

        isGenerating = false
        generatingConversationID = nil
        activeRequestID = nil
        isEndingThinking = false
        streamStatusLine = nil
        streamingThinking = ""
        streamingContent = ""
        if selectedConversationID == convID && showPromptPreview {
            schedulePromptPreviewRefresh()
        }
        await refreshMetrics()
    }

    public func endThinkingNow() {
        guard let reqID = activeRequestID, !isEndingThinking else { return }
        isEndingThinking = true
        _ = registry.requestReasoningEnd(requestID: reqID)
    }

    public func cancelGeneration() {
        guard let reqID = activeRequestID else { return }
        _ = registry.abort(requestID: reqID)
    }

    // MARK: - Notebook Operations (Inline Cursor Continuation, Selection Analysis, Stop & Auto-Save)

    public func noteNotebookUserEdited() {
        isNotebookDirty = true
        notebookAutoSaveTask?.cancel()
        notebookAutoSaveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled else { return }
            self?.saveNotebook()
        }
    }

    public func flushNotebookSaveIfDirty() {
        notebookAutoSaveTask?.cancel()
        notebookAutoSaveTask = nil
        if isNotebookDirty {
            saveNotebook()
        }
    }

    public func createNewNotebook() {
        if isNotebookGenerating {
            cancelNotebookGeneration()
            if activeNotebookStreamingID == selectedNotebookID {
                saveNotebook()
            }
        }
        flushNotebookSaveIfDirty()
        do {
            let model = selectedModelID.isEmpty ? (generationModels.first?.modelID ?? "") : selectedModelID
            let nb = NotebookRecord(
                title: "Untitled Notebook",
                body: "",
                systemPrompt: "",
                modelID: model.isEmpty ? nil : model
            )
            try store.upsertNotebook(nb)
            notebooks = try store.listNotebooks()
            isApplyingProgrammaticState = true
            selectedNotebookID = nb.id
            notebookTitle = nb.title
            notebookSystemPrompt = ""
            notebookBody = nb.body
            notebookSelectedRange = NSRange(location: 0, length: 0)
            if activeSurface == .notebook {
                applyDocumentSamplingState(params: nil, appliedPresetID: nil)
            }
            isNotebookDirty = false
            isApplyingProgrammaticState = false
        } catch {
            statusBanner = "Could not create notebook: \(error.localizedDescription)"
        }
    }

    public func selectNotebook(_ id: String) {
        guard id != selectedNotebookID else { return }
        if isNotebookGenerating {
            cancelNotebookGeneration()
            if activeNotebookStreamingID == selectedNotebookID {
                saveNotebook()
            }
        }
        flushNotebookSaveIfDirty()
        isApplyingProgrammaticState = true
        selectedNotebookID = id
        let fresh = (try? store.getNotebook(id: id)) ?? notebooks.first(where: { $0.id == id })
        if let nb = fresh {
            notebookTitle = nb.title
            notebookSystemPrompt = nb.systemPrompt
            notebookBody = nb.body
            notebookSelectedRange = NSRange(location: (nb.body as NSString).length, length: 0)
            if activeSurface == .notebook {
                if let modelID = nb.modelID, !modelID.isEmpty,
                   generationModels.contains(where: { $0.modelID == modelID }) {
                    selectedModelID = modelID
                }
                applyDocumentSamplingState(params: nb.params, appliedPresetID: nb.appliedPresetID)
            }
        }
        isNotebookDirty = false
        isApplyingProgrammaticState = false
    }

    public func deleteNotebook(_ id: String) {
        if activeNotebookStreamingID == id || (selectedNotebookID == id && isNotebookGenerating) {
            cancelNotebookGeneration()
        }
        if selectedNotebookID == id {
            notebookAutoSaveTask?.cancel()
            isNotebookDirty = false
        }
        _ = try? store.deleteNotebook(id: id)
        notebooks = (try? store.listNotebooks()) ?? []
        if selectedNotebookID == id {
            selectedNotebookID = nil
            if let first = notebooks.first {
                selectNotebook(first.id)
            } else {
                isApplyingProgrammaticState = true
                notebookTitle = "Scratchpad"
                notebookSystemPrompt = ""
                notebookBody = ""
                notebookSelectedRange = NSRange(location: 0, length: 0)
                if activeSurface == .notebook {
                    applyDocumentSamplingState(params: nil, appliedPresetID: nil)
                }
                isNotebookDirty = false
                isApplyingProgrammaticState = false
            }
        }
    }

    public func saveNotebook() {
        notebookAutoSaveTask?.cancel()
        notebookAutoSaveTask = nil
        do {
            let targetID = selectedNotebookID ?? UUID().uuidString
            let existing = selectedNotebookID.flatMap { try? store.getNotebook(id: $0) }
            let model = selectedModelID.isEmpty ? (generationModels.first?.modelID ?? "") : selectedModelID
            let resolvedParams = activeSurface == .notebook ? currentManualParamsForPersistence : existing?.params
            let resolvedPresetID = activeSurface == .notebook ? activePresetIDForPersistence : existing?.appliedPresetID
            let nb = NotebookRecord(
                id: targetID,
                title: notebookTitle,
                body: notebookBody,
                systemPrompt: notebookSystemPrompt,
                modelID: model.isEmpty ? existing?.modelID : model,
                params: resolvedParams,
                appliedPresetID: resolvedPresetID
            )
            try store.upsertNotebook(nb)
            selectedNotebookID = nb.id
            notebooks = try store.listNotebooks()
            isNotebookDirty = false
        } catch {
            statusBanner = "Notebook save failed: \(error.localizedDescription)"
        }
    }

    public func cancelNotebookGeneration() {
        guard let reqID = activeNotebookRequestID else { return }
        _ = registry.abort(requestID: reqID)
    }

    public var hasNotebookSelection: Bool {
        notebookSelectedRange.length > 0
    }

    /// Streams raw inline continuation directly at `notebookSelectedRange` without synthetic prompt wrappers.
    public func continueNotebookGeneration() async {
        guard !isNotebookGenerating else { return }
        let modelID = selectedModelID.isEmpty ? (generationModels.first?.modelID ?? "") : selectedModelID
        guard !modelID.isEmpty else {
            statusBanner = "No generation model selected for notebook continuation."
            return
        }

        let nsBody = notebookBody as NSString
        let clamped = Self.clampUTF16Range(notebookSelectedRange, length: nsBody.length)
        let insertionPoint = clamped.location + clamped.length
        let head = nsBody.substring(to: insertionPoint)
        let tail = nsBody.substring(from: insertionPoint)

        guard !head.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            statusBanner = "Place the cursor after text in the notebook to continue from."
            return
        }

        let trimmedNbSys = notebookSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        await streamNotebookInsertion(
            modelID: modelID,
            systemOverride: trimmedNbSys.isEmpty ? nil : notebookSystemPrompt,
            promptContent: head,
            head: head,
            prefixInjection: "",
            tail: tail
        )
    }

    public func summarizeNotebook() async {
        guard !isNotebookGenerating else { return }
        let modelID = selectedModelID.isEmpty ? (generationModels.first?.modelID ?? "") : selectedModelID
        guard !modelID.isEmpty else {
            statusBanner = "No generation model selected for notebook summary."
            return
        }
        let body = notebookBody
        guard !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        await streamNotebookInsertion(
            modelID: modelID,
            systemOverride: "You are a concise technical writer. Summarize the key points in clean bullet form.",
            promptContent: body,
            head: body,
            prefixInjection: "\n\n## Summary\n",
            tail: ""
        )
    }

    /// Analyzes ONLY the user's highlighted `notebookSelectedRange` span and inserts the critique after the selection.
    public func analyzeNotebookSelection() async {
        guard !isNotebookGenerating else { return }
        let modelID = selectedModelID.isEmpty ? (generationModels.first?.modelID ?? "") : selectedModelID
        guard !modelID.isEmpty else {
            statusBanner = "No generation model selected for notebook analysis."
            return
        }

        let nsBody = notebookBody as NSString
        let clamped = Self.clampUTF16Range(notebookSelectedRange, length: nsBody.length)
        guard clamped.length > 0 else {
            statusBanner = "Select a text span in the notebook to analyze."
            return
        }

        let selectedSpan = nsBody.substring(with: clamped)
        let insertionPoint = clamped.location + clamped.length
        let head = nsBody.substring(to: insertionPoint)
        let tail = nsBody.substring(from: insertionPoint)

        await streamNotebookInsertion(
            modelID: modelID,
            systemOverride: "Critique the key assumptions, edge cases, and concrete improvements for the selected passage.",
            promptContent: selectedSpan,
            head: head,
            prefixInjection: "\n\n## Selection Analysis\n",
            tail: tail
        )
    }

    private func streamNotebookInsertion(
        modelID: String,
        systemOverride: String?,
        promptContent: String,
        head: String,
        prefixInjection: String,
        tail: String
    ) async {
        if selectedNotebookID == nil {
            saveNotebook()
        }
        guard let targetNotebookID = selectedNotebookID else {
            statusBanner = "Could not persist active notebook before streaming."
            return
        }

        let initialExisting = try? store.getNotebook(id: targetNotebookID)
        let targetSystemPrompt = selectedNotebookID == targetNotebookID ? notebookSystemPrompt : (initialExisting?.systemPrompt ?? "")
        let targetParams = activeSurface == .notebook ? currentManualParamsForPersistence : initialExisting?.params
        let targetAppliedPresetID = activeSurface == .notebook ? activePresetIDForPersistence : initialExisting?.appliedPresetID

        let reqID = "nb_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(10))"
        activeNotebookRequestID = reqID
        activeNotebookStreamingID = targetNotebookID
        isNotebookGenerating = true
        statusBanner = nil

        let fixedPrefix = head + prefixInjection
        let fixedPrefixUTF16 = (fixedPrefix as NSString).length
        var workingBody = fixedPrefix + tail
        if !prefixInjection.isEmpty && selectedNotebookID == targetNotebookID {
            isApplyingProgrammaticState = true
            notebookBody = workingBody
            notebookSelectedRange = NSRange(location: fixedPrefixUTF16, length: 0)
            isApplyingProgrammaticState = false
        }

        do {
            let provider = try await scheduler.provider(for: modelID)
            let effective = currentResolvedSamplers.effective
            let req = MessagesRequest(
                model: modelID,
                messages: [MessagesRequest.MessageTurn(role: "user", content: promptContent)],
                system: systemOverride,
                maxTokens: effective.maxTokens ?? 1024,
                temperature: effective.temperature,
                topP: effective.topP,
                topK: effective.topK,
                minP: effective.minP,
                repetitionPenalty: effective.repetitionPenalty,
                repetitionContextSize: effective.repetitionContextSize,
                presencePenalty: effective.presencePenalty,
                seed: effective.seed,
                stream: true,
                thinking: activeModelSupportsThinking ? effectiveThinkingEnabled : nil,
                thinkingDepth: activeModelDepthOptions.isEmpty ? nil : effectiveThinkingDepth,
                thinkingBudget: activeModelSupportsThinking ? effective.thinkingBudget : nil,
                stopSequences: effective.stopSequences
            )
            registry.register(requestID: reqID, conversationID: nil, modelID: modelID)

            try await scheduler.withComputeLease(for: modelID) { [registry] in
                defer {
                    registry.finish(requestID: reqID)
                }
                var streamedDelta = ""
                let stream = await provider.generate(
                    requestID: reqID,
                    request: req,
                    effectiveSampling: effective,
                    registry: registry
                )
                for try await chunk in stream where !chunk.text.isEmpty {
                    streamedDelta.append(chunk.text)
                    let combined = fixedPrefix + streamedDelta + tail
                    let newCursor = fixedPrefixUTF16 + (streamedDelta as NSString).length
                    await MainActor.run {
                        workingBody = combined
                        if self.selectedNotebookID == targetNotebookID {
                            self.isApplyingProgrammaticState = true
                            self.notebookBody = combined
                            self.notebookSelectedRange = NSRange(location: newCursor, length: 0)
                            self.isApplyingProgrammaticState = false
                        }
                    }
                }
            }
        } catch {
            statusBanner = "Notebook generation error: \(error.localizedDescription)"
        }

        if activeNotebookRequestID == reqID {
            activeNotebookRequestID = nil
            activeNotebookStreamingID = nil
            isNotebookGenerating = false
        }

        // Persist directly to `targetNotebookID` (never via `saveNotebook()`, which reads `selectedNotebookID`/`notebookBody`)
        // so switching to another notebook mid-stream never overwrites the newly selected notebook!
        if let latestExisting = try? store.getNotebook(id: targetNotebookID) {
            let persisted = NotebookRecord(
                id: targetNotebookID,
                title: selectedNotebookID == targetNotebookID ? notebookTitle : latestExisting.title,
                body: workingBody,
                systemPrompt: selectedNotebookID == targetNotebookID ? notebookSystemPrompt : (latestExisting.systemPrompt.isEmpty ? targetSystemPrompt : latestExisting.systemPrompt),
                modelID: modelID,
                params: latestExisting.params ?? targetParams,
                appliedPresetID: latestExisting.appliedPresetID ?? targetAppliedPresetID
            )
            try? store.upsertNotebook(persisted)
            notebooks = (try? store.listNotebooks()) ?? notebooks
            if selectedNotebookID == targetNotebookID {
                isNotebookDirty = false
            }
        }
    }

    private static func clampUTF16Range(_ range: NSRange, length: Int) -> NSRange {
        let loc = min(max(0, range.location), length)
        let rem = max(0, length - loc)
        let len = min(max(0, range.length), rem)
        return NSRange(location: loc, length: len)
    }

    public func saveNotebookToConversations() {
        flushNotebookSaveIfDirty()
        let content = notebookBody.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else {
            statusBanner = "Write some content in the Notebook before sending to Chat."
            return
        }
        do {
            let model = selectedModelID.isEmpty ? (generationModels.first?.modelID ?? "") : selectedModelID
            let conv = ConversationRecord(
                title: "Notebook: \(notebookTitle)",
                systemPrompt: notebookSystemPrompt,
                defaultModelID: model.isEmpty ? nil : model,
                params: currentManualParamsForPersistence,
                appliedPresetID: activePresetIDForPersistence
            )
            try store.upsertConversation(conv)
            try store.appendMessage(
                MessageRecord(
                    conversationID: conv.id,
                    position: 0,
                    role: "user",
                    content: notebookBody,
                    modelID: model.isEmpty ? nil : model
                )
            )
            conversations = try store.listConversations()
            selectConversation(conv.id)
            activeSurface = .chat
        } catch {
            statusBanner = "Could not export notebook: \(error.localizedDescription)"
        }
    }

    // MARK: - Embeddings & ANE Media Finder Studio (Truthful Runtime — Zero Hash Fallback)

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

    public func syncEmbeddingEngineToRouter() async {
        await router.setEmbeddingEngine(activeEmbeddingEngine)
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
            statusBanner = msg
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
            statusBanner = msg
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
            statusBanner = "Sample corpus error: \(error.localizedDescription)"
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
            // Do NOT auto-force `showInspector = true` during query typing; open inspector only on explicit hit selection!
            embeddingStatusMessage = "Found \(results.count) results (\(selectedDimension)d Matryoshka slice)."
        } catch {
            searchHits = []
            selectedHit = nil
            embeddingStatusMessage = "Search failed: \(error.localizedDescription)"
            statusBanner = "Search failed: \(error.localizedDescription)"
        }
    }

    // MARK: - N×N Multimodal Compare Matrix (`≤16` Items, Live Matryoshka Slicing) & Multi-Config Lab

    public func addCompareTextItem(_ initialText: String = "") {
        guard compareItems.count < Self.maxCompareMatrixItems else {
            statusBanner = "Compare Matrix supports up to \(Self.maxCompareMatrixItems) items."
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
                statusBanner = "Compare Matrix supports up to \(Self.maxCompareMatrixItems) items."
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
            statusBanner = msg
            return
        }
        let items = Array(compareItems.prefix(Self.maxCompareMatrixItems))
        guard items.count >= 2 else {
            statusBanner = "Add at least 2 items to compute an N×N similarity matrix."
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
            statusBanner = "Matrix compare failed: \(error.localizedDescription)"
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
            let escaped = s.replacingOccurrences(of: "\"", with: "\"\"")
            return "\"\(escaped)\""
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

    /// Truthful pairwise cosine similarity — never falls back to synthetic hash vectors!
    public func runPairwiseCompare() {
        guard !isActiveEmbeddingEngineSynthetic else {
            compareCosineScore = nil
            let msg = "Pairwise Compare requires a real EmbeddingGemma 2 checkpoint at '\(checkpointPath)' (or Remote Server)."
            embeddingStatusMessage = msg
            statusBanner = msg
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
                self.statusBanner = "Compare failed: \(error.localizedDescription)"
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
            statusBanner = msg
            return
        }
        guard !benchmarkCases.isEmpty else {
            statusBanner = "Add or import at least one benchmark case first."
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
            statusBanner = "Lab config run failed: \(error.localizedDescription)"
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
            statusBanner = msg
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
                statusBanner = "Unsupported benchmark JSON format (expected array of {query, expected_artifact_id})."
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
                statusBanner = "No valid {query, expected_artifact_id} cases found in \(url.lastPathComponent)."
                return
            }
            benchmarkCases = imported
            statusBanner = "Imported \(imported.count) benchmark cases from \(url.lastPathComponent)."
            Task { await evaluateCurrentConfigInLab() }
        } catch {
            statusBanner = "Failed to import benchmark set: \(error.localizedDescription)"
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
            statusBanner = "Eval suite error: \(error.localizedDescription)"
            return nil
        }
    }

    // MARK: - Server Lifecycle & Bonjour Remote Discovery

    public func toggleServer() async {
        do {
            if isServerRunning {
                await server.stop()
                isServerRunning = false
                HeylookObservabilityLog.shared.record(
                    level: "INFO",
                    category: "server",
                    message: "Embedded HTTP/SSE server stopped"
                )
                statusBanner = "Embedded server stopped."
            } else {
                try await server.start()
                isServerRunning = true
                HeylookObservabilityLog.shared.record(
                    level: "INFO",
                    category: "server",
                    message: "Embedded HTTP/SSE server listening on http://127.0.0.1:\(serverPort)"
                )
                statusBanner = "Embedded server listening on http://127.0.0.1:\(serverPort)"
            }
            refreshObservabilityEntries()
        } catch {
            statusBanner = "Server toggle error: \(error.localizedDescription)"
        }
    }

    public func startBonjourDiscovery() {
        isBrowsingBonjour = true
        bonjourBrowser.start { [weak self] servers in
            Task { @MainActor [weak self] in
                self?.discoveredServers = servers
            }
        }
    }

    public func stopBonjourDiscovery() {
        bonjourBrowser.stop()
        isBrowsingBonjour = false
    }

    public func connectToRemoteServer(_ urlString: String) async {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let baseURL = URL(string: trimmed) else {
            statusBanner = "Invalid remote server URL: \(urlString)"
            return
        }
        remoteServerURLString = trimmed
        embeddingServerURLString = trimmed
        connectedRemoteServerURL = trimmed

        var remoteDescriptions: [EngineDescription] = []
        let modelsURL = baseURL.appendingPathComponent("v1/models")
        if let (data, resp) = try? await URLSession.shared.data(from: modelsURL),
           let http = resp as? HTTPURLResponse,
           http.statusCode == 200,
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let list = json["data"] as? [[String: Any]] {
            for item in list {
                if let id = (item["id"] as? String) ?? (item["modelID"] as? String) {
                    let remoteDesc = EngineDescription(
                        modelID: id,
                        provider: .remote,
                        modelPath: trimmed,
                        architecture: ProvenanceField(value: "remote-heylook", provenance: .observed),
                        quantization: ProvenanceField(value: "remote", provenance: .observed),
                        computePlacement: ProvenanceField(value: .remote, provenance: .observed),
                        contextSize: ProvenanceField(value: 32768, provenance: .observed),
                        flashAttention: ProvenanceField(value: true, provenance: .observed),
                        supportsVision: ProvenanceField(value: true, provenance: .observed),
                        supportsThinking: ProvenanceField(value: true, provenance: .observed),
                        tags: ["remote", "bonjour"]
                    )
                    remoteDescriptions.append(remoteDesc)
                }
            }
        }

        if remoteDescriptions.isEmpty {
            let fallbackDesc = EngineDescription(
                modelID: "remote-mac-studio-default",
                provider: .remote,
                modelPath: trimmed,
                architecture: ProvenanceField(value: "remote-heylook", provenance: .configured),
                quantization: ProvenanceField(value: "remote", provenance: .configured),
                computePlacement: ProvenanceField(value: .remote, provenance: .configured),
                contextSize: ProvenanceField(value: 32768, provenance: .configured),
                flashAttention: ProvenanceField(value: true, provenance: .configured),
                supportsVision: ProvenanceField(value: true, provenance: .configured),
                supportsThinking: ProvenanceField(value: true, provenance: .configured),
                tags: ["remote", "bonjour"]
            )
            remoteDescriptions = [fallbackDesc]
        }

        for desc in remoteDescriptions {
            let provider = RemoteHeylookProvider(baseURL: baseURL, description: desc)
            await scheduler.register(provider: provider, description: desc)
        }
        registeredModels = await scheduler.listModels()
        if let firstRemote = remoteDescriptions.first {
            selectedModelID = firstRemote.modelID
        }
        statusBanner = "Connected to remote Mac Studio at \(trimmed) (\(remoteDescriptions.count) models)."
    }

    // MARK: - Model Lifecycle, `[scan]` Folders, RAM Fit & Optional Sidecar (`model.heylook.toml`) Administration

    public func assessModelRAMFit(
        _ model: EngineDescription,
        ctxSizeOverride: Int? = nil,
        sidecarOverride: ModelSidecarConfig? = nil
    ) -> GGUFSpawnConfig.RAMFitAssessment {
        GGUFSpawnConfig.assessRAMFit(
            for: model,
            ctxSize: ctxSizeOverride,
            sidecar: sidecarOverride
        )
    }

    public func applyQuickContextSize(_ ctxSize: Int, for model: EngineDescription) async {
        guard !HeylookConfig.isModelConfigReadOnly else {
            statusBanner = "Model configuration is read-only (HEYLOOK_READONLY_MODELS=1)."
            return
        }
        var sidecar = loadSidecarConfig(for: model)
        sidecar.ctxSize = ctxSize
        await saveModelSidecar(sidecar, for: model)
    }

    public func addModelScanFolder(_ url: URL) async {
        let scoped = url.startAccessingSecurityScopedResource()
        defer {
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
        guard !HeylookConfig.isModelConfigReadOnly else {
            statusBanner = "Model configuration is read-only (HEYLOOK_READONLY_MODELS=1)."
            return
        }
        do {
            config = try HeylookConfig.addScanFolder(url.standardizedFileURL.path)
            await rescanConfiguredModels()
            statusBanner = "Added scan folder '\(url.lastPathComponent)' to heylook.toml."
        } catch {
            statusBanner = "Failed to add scan folder: \(error.localizedDescription)"
        }
    }

    public func removeModelScanFolder(_ folder: String) async {
        guard !HeylookConfig.isModelConfigReadOnly else {
            statusBanner = "Model configuration is read-only (HEYLOOK_READONLY_MODELS=1)."
            return
        }
        do {
            config = try HeylookConfig.removeScanFolder(folder)
            await rescanConfiguredModels()
            statusBanner = "Removed scan folder '\(folder)' from heylook.toml."
        } catch {
            statusBanner = "Failed to remove scan folder: \(error.localizedDescription)"
        }
    }

    public func toggleModelPin(_ model: EngineDescription) async {
        await scheduler.setPinned(!model.isPinned, for: model.modelID)
        await refreshMetrics()
    }

    public func toggleModelLoaded(_ model: EngineDescription) async {
        if model.isLoaded {
            await scheduler.unloadModel(model.modelID)
            await refreshMetrics()
            statusBanner = "Unloaded \(model.modelID)."
        } else {
            guard !loadingModelIDs.contains(model.modelID) else { return }
            loadingModelIDs.insert(model.modelID)
            defer { loadingModelIDs.remove(model.modelID) }

            do {
                _ = try await scheduler.activateModel(model.modelID)
                await refreshMetrics()
                statusBanner = "Loaded \(model.modelID)."
            } catch {
                await refreshMetrics()
                statusBanner = "Failed to load \(model.modelID): \(error.localizedDescription)"
            }
        }
    }

    public func unloadAllModels() async {
        await scheduler.unloadAllModels()
        await activeEmbeddingEngine.unload()
        await refreshMetrics()
        statusBanner = "Unloaded all models, cleared Metal/ANE caches, and terminated background subprocesses."
    }

    public func rescanConfiguredModels() async {
        config = HeylookConfig.loadDefault()
        let discovered = config.discoverConfiguredModels()
        await scheduler.registerDiscoveredModels(discovered)
        registeredModels = await scheduler.listModels()
        if selectedModelID.isEmpty, let first = generationModels.first ?? registeredModels.first {
            selectedModelID = first.modelID
        }
        await refreshMetrics()
        statusBanner = "Rescanned heylook.toml (\(registeredModels.count) models across \(config.scanFolders.count) scan folders)."
    }

    public func shutdown() async {
        flushSystemPromptSave()
        persistActiveConversationSamplingState()
        flushNotebookSaveIfDirty()
        storeMutationTask?.cancel()
        storeMutationTask = nil
        metricsLoopTask?.cancel()
        promptPreviewTask?.cancel()
        embeddingSearchTask?.cancel()
        indexTask?.cancel()
        indexTask = nil
        #if os(macOS)
        folderWatcher?.stop()
        #endif
        stopBonjourDiscovery()
        if isServerRunning {
            await server.stop()
            isServerRunning = false
        } else {
            await scheduler.unloadAllModels()
            await activeEmbeddingEngine.unload()
        }
        await refreshMetrics()
    }

    public func unloadGGUFAllocation(_ allocationEntry: String) async {
        let modelID = allocationEntry.components(separatedBy: "::").first ?? allocationEntry
        await scheduler.unloadModel(modelID)
        GGUFPortPool.shared.releasePort(for: modelID)
        await refreshMetrics()
        statusBanner = "Unloaded GGUF subprocess for \(modelID)."
    }

    public func hasSidecarFileOnDisk(for model: EngineDescription) -> Bool {
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        let sidecarURL = dirURL.appendingPathComponent("model.heylook.toml")
        return FileManager.default.fileExists(atPath: sidecarURL.path)
    }

    /// Loads `model.heylook.toml` when present on disk; otherwise returns an EMPTY `ModelSidecarConfig()`
    /// (all `nil`s!) so saving `pinned` or `tags` never pollutes untouched fields with synthetic defaults!
    public func loadSidecarConfig(for model: EngineDescription) -> ModelSidecarConfig {
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        let sidecarURL = dirURL.appendingPathComponent("model.heylook.toml")
        if let text = try? String(contentsOf: sidecarURL, encoding: .utf8) {
            return ModelSidecarConfig.parseTOML(text)
        }
        return ModelSidecarConfig()
    }

    public func saveModelSidecar(_ sidecar: ModelSidecarConfig, for model: EngineDescription) async {
        guard !HeylookConfig.isModelConfigReadOnly else {
            statusBanner = "Model configuration is read-only (HEYLOOK_READONLY_MODELS=1)."
            return
        }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        let sidecarURL = dirURL.appendingPathComponent("model.heylook.toml")
        var isDir: ObjCBool = false
        do {
            if sidecar == ModelSidecarConfig() {
                if FileManager.default.fileExists(atPath: sidecarURL.path) {
                    try FileManager.default.removeItem(at: sidecarURL)
                }
                await rescanConfiguredModels()
                if showPromptPreview {
                    schedulePromptPreviewRefresh()
                }
                statusBanner = "Removed model.heylook.toml overrides for \(model.modelID) (reverted to derived defaults)."
                return
            }

            if FileManager.default.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue {
                try sidecar.toTOML().write(to: sidecarURL, atomically: true, encoding: .utf8)
            }
            _ = try await scheduler.applySidecarUpdate(sidecar, for: model.modelID)
            await refreshMetrics()
            if showPromptPreview {
                schedulePromptPreviewRefresh()
            }
            statusBanner = "Saved model.heylook.toml sidecar for \(model.modelID)."
        } catch {
            statusBanner = "Failed to write model.heylook.toml: \(error.localizedDescription)"
        }
    }

    public func loadChatTemplateState(for model: EngineDescription) -> HeylookConfig.ChatTemplateState {
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        return HeylookConfig.inspectChatTemplate(
            in: dirURL,
            fallbackSource: model.templateSource.value
        )
    }

    public func validateJinjaTemplateSyntax(_ template: String) -> String? {
        do {
            try HeylookJinjaRenderer.validateSyntax(template)
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    @discardableResult
    public func saveChatTemplateOverride(_ template: String, for model: EngineDescription) async throws -> HeylookConfig.ChatTemplateState {
        guard !HeylookConfig.isModelConfigReadOnly else {
            throw ProviderExecutionError.invalidRequest(detail: "Model configuration is read-only (HEYLOOK_READONLY_MODELS=1).")
        }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        let state = try HeylookConfig.writeChatTemplateOverride(template, in: dirURL)
        let sidecar = loadSidecarConfig(for: model)
        _ = try? await scheduler.applySidecarUpdate(sidecar, for: model.modelID)
        await refreshMetrics()
        if showPromptPreview {
            schedulePromptPreviewRefresh()
        }
        statusBanner = "Saved chat_template.heylook.jinja override for \(model.modelID)."
        return state
    }

    @discardableResult
    public func revertChatTemplateOverride(for model: EngineDescription) async -> HeylookConfig.ChatTemplateState {
        guard !HeylookConfig.isModelConfigReadOnly else {
            statusBanner = "Model configuration is read-only (HEYLOOK_READONLY_MODELS=1)."
            return loadChatTemplateState(for: model)
        }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        let reverted = HeylookConfig.deleteChatTemplateOverride(
            in: dirURL,
            fallbackSource: model.templateSource.value == "chat_template.heylook.jinja"
                ? (model.provider == .gguf ? "gguf_embedded" : "tokenizer_config.json")
                : model.templateSource.value
        )
        let sidecar = loadSidecarConfig(for: model)
        _ = try? await scheduler.applySidecarUpdate(sidecar, for: model.modelID)
        await refreshMetrics()
        if showPromptPreview {
            schedulePromptPreviewRefresh()
        }
        statusBanner = "Reverted chat_template.heylook.jinja override for \(model.modelID) (active: \(reverted.source))."
        return reverted
    }
}
