import Accelerate
import Foundation
import HeylookCore

/// Task prompts published by EmbeddingGemma 2 (`config_sentence_transformers.json`).
public enum EmbeddingTask: String, Codable, CaseIterable, Sendable {
    case query
    case document
    case retrievalQuery = "Retrieval-query"
    case retrievalDocument = "Retrieval-document"
    case questionAnswering = "QuestionAnswering"
    case factVerification = "FactVerification"
    case sts = "STS"
    case clustering = "Clustering"
    case classification = "Classification"
    case codeRetrieval = "InstructionRetrieval"
    case none = ""

    public var promptName: String? {
        rawValue.isEmpty ? nil : rawValue
    }

    public var prefixText: String {
        switch self {
        case .query, .retrievalQuery:
            return "task: search result | query: "
        case .questionAnswering:
            return "task: question answering | query: "
        case .factVerification:
            return "task: fact checking | query: "
        case .codeRetrieval:
            return "task: code retrieval | query: "
        case .document, .retrievalDocument:
            return "title: none | text: "
        case .clustering:
            return "task: clustering | query: "
        case .classification:
            return "task: classification | query: "
        case .sts:
            return "task: sentence similarity | query: "
        case .none:
            return ""
        }
    }

    public var promptPrefix: String { prefixText }

    public static func documentPrompt(title: String?, text: String) -> String {
        let cleanTitle = title?
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
        if let cleanTitle, !cleanTitle.isEmpty {
            return "title: \(cleanTitle) | text: \(text)"
        }
        return "title: none | text: \(text)"
    }
}

/// Engine backend selection for embeddings.
public enum EngineBackend: String, Codable, CaseIterable, Sendable {
    case heylook
    case nativeMLX = "native"

    public var label: String {
        switch self {
        case .heylook: return "heylook Server (HTTP)"
        case .nativeMLX: return "In-Process MLX Swift / ANE"
        }
    }
}

/// Window sampling strategy for temporal artifacts (video/audio).
public enum VideoWindowSampling: String, Codable, CaseIterable, Sendable {
    case off = ""
    case fps = "fps"
    case content = "content"

    public var label: String {
        switch self {
        case .off: return "Whole video only"
        case .fps: return "Fixed 1 fps (32s windows)"
        case .content: return "Content-aware (32s windows)"
        }
    }
}

/// Requested temporal windowing options for a single video block (`v2.0.233`–`v2.0.244` parity).
public struct VideoWindowOptions: Codable, Sendable, Equatable {
    public let seconds: Double?
    public let sampling: VideoWindowSampling

    public init(seconds: Double? = nil, sampling: VideoWindowSampling = .fps) {
        self.seconds = seconds
        self.sampling = sampling
    }

    public static func validated(seconds: Double?, samplingRaw: String?) throws -> VideoWindowOptions {
        let sampling: VideoWindowSampling
        if let samplingRaw {
            guard let parsed = VideoWindowSampling(rawValue: samplingRaw), parsed != .off else {
                throw EmbeddingValidationError.invalidWindowSampling(samplingRaw)
            }
            sampling = parsed
        } else {
            sampling = .fps
        }
        if let seconds, (!seconds.isFinite || seconds <= 0) {
            throw EmbeddingValidationError.invalidWindowSeconds(seconds)
        }
        return VideoWindowOptions(seconds: seconds, sampling: sampling)
    }
}

/// Typed validation errors for `/v1/embeddings` inputs and checkpoint/model constraints.
public enum EmbeddingValidationError: LocalizedError, Sendable, Equatable {
    case unknownPromptName(itemIndex: Int, promptName: String, available: [String])
    case unsupportedDimensions(requested: Int, supported: [Int])
    case unsupportedModality(itemIndex: Int, modality: String, supported: [String])
    case videoWindowsUnsupported(itemIndex: Int, detail: String)
    case videoWindowsExceedMaxSeconds(itemIndex: Int, requested: Double, maxSeconds: Double)
    case invalidWindowSampling(String)
    case invalidWindowSeconds(Double)
    case windowedVideoRequiresSingleVideoBlock(itemIndex: Int)
    case windowedVideoNonVideoBlock(itemIndex: Int, kind: ArtifactKind)
    case invalidInput(statusCode: Int, detail: String)

    public var statusCode: Int {
        switch self {
        case .invalidWindowSampling,
             .invalidWindowSeconds,
             .windowedVideoRequiresSingleVideoBlock,
             .windowedVideoNonVideoBlock:
            return 422
        case let .invalidInput(statusCode, _):
            return statusCode
        case .unknownPromptName,
             .unsupportedDimensions,
             .unsupportedModality,
             .videoWindowsUnsupported,
             .videoWindowsExceedMaxSeconds:
            return 400
        }
    }

    public var wireValidationError: WireValidationError {
        let message = errorDescription ?? "Invalid embedding request"
        return statusCode == 422 ? .unprocessable(message) : .badRequest(message)
    }

    public var errorDescription: String? {
        switch self {
        case let .unknownPromptName(itemIndex, promptName, _):
            return "input[\(itemIndex)]: unknown prompt_name '\(promptName)'"
        case let .unsupportedDimensions(_, supported):
            return "Unsupported dimensions; choose \(supported)"
        case let .unsupportedModality(itemIndex, modality, _):
            return "input[\(itemIndex)]: unsupported modality '\(modality)'"
        case let .videoWindowsUnsupported(itemIndex, detail):
            return "input[\(itemIndex)]: \(detail)"
        case let .videoWindowsExceedMaxSeconds(itemIndex, _, maxSeconds):
            return "input[\(itemIndex)]: windows.seconds exceeds \(maxSeconds)"
        case let .invalidWindowSampling(raw):
            return "Invalid video window sampling '\(raw)'; expected 'fps' or 'content'"
        case let .invalidWindowSeconds(seconds):
            return "Invalid video window seconds (\(seconds)); must be > 0"
        case let .windowedVideoRequiresSingleVideoBlock(itemIndex):
            return "input[\(itemIndex)]: an item with video windows must contain only that video block"
        case let .windowedVideoNonVideoBlock(itemIndex, kind):
            return "input[\(itemIndex)]: windows applies to video blocks only (got '\(kind.rawValue)')"
        case let .invalidInput(_, detail):
            return detail
        }
    }
}

/// One ordered block inside a multimodal `EmbeddingInput`.
public enum EmbeddingPart: Sendable, Equatable {
    case text(String)
    case media(
        kind: ArtifactKind,
        mimeType: String,
        data: Data? = nil,
        url: URL? = nil,
        windowSampling: VideoWindowSampling = .off
    )

    public static func mediaURL(
        kind: ArtifactKind,
        mimeType: String? = nil,
        url: URL,
        windowSampling: VideoWindowSampling = .off
    ) -> EmbeddingPart {
        .media(
            kind: kind,
            mimeType: mimeType ?? ArtifactKind.mimeType(for: url),
            data: nil,
            url: url,
            windowSampling: windowSampling
        )
    }

    public func resolvedData() throws -> Data {
        switch self {
        case .text(let s):
            return Data(s.utf8)
        case let .media(_, _, data, url, _):
            if let data { return data }
            if let url { return try Data(contentsOf: url) }
            return Data()
        }
    }

    public var estimatedBytes: Int {
        switch self {
        case .text(let s):
            return s.utf8.count
        case let .media(_, _, data, url, _):
            if let data { return data.count }
            if let url,
               let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue {
                return size
            }
            return 0
        }
    }
}

/// One item submitted for embedding.
public struct EmbeddingInput: Sendable, Equatable {
    public let parts: [EmbeddingPart]
    public let task: EmbeddingTask
    public let title: String?
    public let promptName: String?
    public let usesExplicitPromptName: Bool
    public let videoWindows: VideoWindowOptions?

    public init(parts: [EmbeddingPart], task: EmbeddingTask = .document, title: String? = nil) {
        self.parts = parts
        self.task = task
        self.title = title
        self.promptName = task.promptName
        self.usesExplicitPromptName = false
        self.videoWindows = nil
    }

    /// Explicit wire/API initializer where `promptName == nil` applies NO task prefix at all
    /// and `videoWindows` carries validated `{seconds?, sampling}` parameters.
    public init(
        parts: [EmbeddingPart],
        promptName: String?,
        title: String? = nil,
        videoWindows: VideoWindowOptions? = nil,
        task: EmbeddingTask = .none
    ) {
        self.parts = parts
        self.task = task
        self.title = title
        self.promptName = promptName
        self.usesExplicitPromptName = true
        self.videoWindows = videoWindows
    }

    public static func text(_ value: String, task: EmbeddingTask, title: String? = nil) -> EmbeddingInput {
        EmbeddingInput(parts: [.text(value)], task: task, title: title)
    }

    public static func text(_ value: String, promptName: String? = nil, title: String? = nil) -> EmbeddingInput {
        EmbeddingInput(parts: [.text(value)], promptName: promptName, title: title)
    }

    public var hasNonWhitespaceText: Bool {
        parts.contains { part in
            if case .text(let s) = part {
                return !s.allSatisfy(\.isWhitespace)
            }
            return false
        }
    }

    /// Resolves and structurally validates requested video windows for this item.
    public func resolvedVideoWindows(itemIndex: Int = 0) throws -> VideoWindowOptions? {
        var partSampling: VideoWindowSampling?
        var nonVideoWindowKind: ArtifactKind?

        for part in parts {
            if case let .media(kind, _, _, _, sampling) = part, sampling != .off {
                if kind == .video {
                    if partSampling == nil {
                        partSampling = sampling
                    }
                } else {
                    nonVideoWindowKind = kind
                }
            }
        }

        if let kind = nonVideoWindowKind {
            throw EmbeddingValidationError.windowedVideoNonVideoBlock(itemIndex: itemIndex, kind: kind)
        }

        if let explicit = videoWindows {
            guard parts.count == 1,
                  case let .media(kind, _, _, _, _) = parts[0] else {
                throw EmbeddingValidationError.windowedVideoRequiresSingleVideoBlock(itemIndex: itemIndex)
            }
            guard kind == .video else {
                throw EmbeddingValidationError.windowedVideoNonVideoBlock(itemIndex: itemIndex, kind: kind)
            }
            if let seconds = explicit.seconds, !seconds.isFinite || seconds <= 0 {
                throw EmbeddingValidationError.invalidWindowSeconds(seconds)
            }
            return VideoWindowOptions(
                seconds: explicit.seconds,
                sampling: explicit.sampling == .off ? .fps : explicit.sampling
            )
        }

        guard let sampling = partSampling else {
            return nil
        }

        if usesExplicitPromptName && parts.count != 1 {
            throw EmbeddingValidationError.windowedVideoRequiresSingleVideoBlock(itemIndex: itemIndex)
        }

        return VideoWindowOptions(seconds: nil, sampling: sampling)
    }
}

/// Embedded temporal window vector with its time span and sampled frame timestamps.
public struct VideoWindowEmbeddingResult: Sendable, Equatable, Codable {
    public let start: Double
    public let end: Double
    public let frameTimes: [Double]
    public let vector: [Float]

    public init(start: Double, end: Double, frameTimes: [Double], vector: [Float]) {
        self.start = start
        self.end = end
        self.frameTimes = frameTimes
        self.vector = vector
    }
}

/// Output for one `EmbeddingInput`.
public struct EmbeddedOutput: Sendable, Equatable {
    public typealias WindowVector = VideoWindowEmbeddingResult

    public let vector: [Float]
    public let windows: [WindowVector]
    public let inputTokens: Int?
    public let promptName: String?

    public init(
        vector: [Float],
        windows: [WindowVector] = [],
        inputTokens: Int? = nil,
        promptName: String? = nil
    ) {
        self.vector = vector
        self.windows = windows
        self.inputTokens = inputTokens
        self.promptName = promptName
    }
}

/// Batch embedding output carrying per-item vectors plus cumulative `inputTokens` accounting.
public struct EmbeddingResult: Sendable, Equatable {
    public let outputs: [EmbeddedOutput]
    public let inputTokens: Int
    public let dimensions: Int
    public let modelID: String
    public let elapsedMS: Double

    public init(
        outputs: [EmbeddedOutput],
        inputTokens: Int = 0,
        dimensions: Int = Matryoshka.gemma2NativeDimensions,
        modelID: String = "",
        elapsedMS: Double = 0
    ) {
        self.outputs = outputs
        self.inputTokens = inputTokens
        self.dimensions = dimensions
        self.modelID = modelID
        self.elapsedMS = elapsedMS
    }
}

/// Protocol for a multimodal embedding backend.
public protocol EmbeddingEngine: Sendable {
    var modelID: String { get }
    var nativeDimensions: Int { get }
    var supportedDimensions: [Int] { get }

    func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int?
    ) async throws -> [EmbeddedOutput]

    func embedResult(
        _ inputs: [EmbeddingInput],
        dimensions: Int?
    ) async throws -> EmbeddingResult

    /// Releases any loaded CoreML / MLX model weights and memory-mapped token tables.
    func unload() async
}

public extension EmbeddingEngine {
    func embedResult(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> EmbeddingResult {
        let start = CFAbsoluteTimeGetCurrent()
        let outputs = try await embedBatch(inputs, dimensions: dimensions)
        let elapsedMS = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        let totalTokens = outputs.compactMap(\.inputTokens).reduce(0, +)
        return EmbeddingResult(
            outputs: outputs,
            inputTokens: totalTokens,
            dimensions: dimensions ?? nativeDimensions,
            modelID: modelID,
            elapsedMS: elapsedMS
        )
    }

    func unload() async {}
}

/// Matryoshka dimension truncation and hardware-accelerated L2 math (`Accelerate` / `vDSP`).
public enum Matryoshka {
    public static let gemma2Dimensions: [Int] = [128, 256, 512, 768]
    public static let gemma2NativeDimensions: Int = 768

    public static func project(_ vector: [Float], dimensions: Int) -> [Float] {
        let d = min(max(1, dimensions), vector.count)
        var slice = Array(vector.prefix(d))
        var sumSq: Float = 0
        vDSP_svesq(slice, 1, &sumSq, vDSP_Length(d))
        let norm = sqrt(sumSq)
        guard norm > 0, norm.isFinite else { return slice }
        var scale = 1.0 / norm
        vDSP_vsmul(slice, 1, &scale, &slice, 1, vDSP_Length(d))
        return slice
    }

    public static func cosine(_ a: [Float], _ b: [Float], dimensions: Int? = nil) -> Float {
        let d = min(a.count, b.count, dimensions ?? Int.max)
        guard d > 0 else { return 0 }
        let pa = d == a.count ? a : project(a, dimensions: d)
        let pb = d == b.count ? b : project(b, dimensions: d)
        var dot: Float = 0
        vDSP_dotpr(pa, 1, pb, 1, &dot, vDSP_Length(d))
        var na: Float = 0
        var nb: Float = 0
        vDSP_svesq(pa, 1, &na, vDSP_Length(d))
        vDSP_svesq(pb, 1, &nb, vDSP_Length(d))
        let denom = sqrt(na * nb)
        return denom > 0 ? dot / denom : 0
    }
}

/// Deterministic 768d n-gram / token-hash embedding engine (`EmbeddingEngine`) for fast local
/// indexing, offline unit testing, and instant Embeddings Studio preview when a full checkpoint
/// is not yet loaded.
@available(*, deprecated, message: "Test-only legacy shim; removed in Wave 4")
public struct DeterministicHashEmbeddingEngine: EmbeddingEngine, Sendable {
    public let modelID: String
    public let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions

    public init(modelID: String = "deterministic-hash-768d") {
        self.modelID = modelID
    }

    public func embed(text: String, dimensions: Int = Matryoshka.gemma2NativeDimensions) -> [Float] {
        var raw = [Float](repeating: 0, count: nativeDimensions)
        let tokens = text.lowercased().split { !$0.isLetter && !$0.isNumber }
        if tokens.isEmpty {
            raw[0] = 1.0
            return Matryoshka.project(raw, dimensions: dimensions)
        }
        for (pos, tok) in tokens.enumerated() {
            var h: UInt64 = 1469598103934665603
            for byte in tok.utf8 {
                h ^= UInt64(byte)
                h &*= 1099511628211
            }
            let idx1 = Int(h % UInt64(nativeDimensions))
            let idx2 = Int((h >> 17) % UInt64(nativeDimensions))
            let sign: Float = (h & 1) == 0 ? 1.0 : -1.0
            raw[idx1] += 1.0
            raw[idx2] += 0.5 * sign / Float(1 + (pos % 4))
        }
        return Matryoshka.project(raw, dimensions: dimensions)
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        let targetDim = dimensions ?? nativeDimensions
        return inputs.map { input in
            var combined = input.title ?? ""
            var windows: [EmbeddedOutput.WindowVector] = []
            for part in input.parts {
                switch part {
                case let .text(s):
                    combined += " " + s
                case let .media(kind, mimeType, data, mediaURL, windowSampling):
                    let byteLen = data?.count ?? part.estimatedBytes
                    let tag = mediaURL?.lastPathComponent ?? "\(kind.rawValue)_\(mimeType)_\(byteLen)"
                    combined += " media_\(tag)_\(windowSampling.rawValue)"
                    if kind == .video && windowSampling != .off {
                        let winVec = embed(text: "\(combined)_win_0_16_\(windowSampling.rawValue)", dimensions: targetDim)
                        let frames: [Double] = windowSampling == .fps ? [0.5, 4.5, 8.5, 12.5] : [0.5, 3.25, 9.75, 14.5]
                        windows.append(
                            EmbeddedOutput.WindowVector(
                                start: 0.0,
                                end: 16.0,
                                frameTimes: frames,
                                vector: winVec
                            )
                        )
                    }
                }
            }
            let vec = embed(text: combined, dimensions: targetDim)
            return EmbeddedOutput(vector: vec, windows: windows)
        }
    }
}


public enum EmbeddingEngineError: LocalizedError, Sendable, Equatable {
    case invalidResponse(status: Int, detail: String)
    case vectorCountMismatch(expected: Int, actual: Int)
    case checkpointNotFound(URL)
    case runtimeUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidResponse(status, detail):
            return "Embedding server error (\(status)): \(detail)"
        case let .vectorCountMismatch(expected, actual):
            return "Expected \(expected) vectors from engine, received \(actual)"
        case let .checkpointNotFound(url):
            return "EmbeddingGemma 2 MLX checkpoint not found at \(url.path)"
        case let .runtimeUnavailable(detail):
            return "Embedding runtime unavailable: \(detail)"
        }
    }
}

/// Canonical `/v1/embeddings` request limits shared between server validators, `/v1/capabilities`,
/// `/v1/models`, and `HeylookMLXEngine` client-side batch planning (`heylookitsanllm` `v2.0.239` parity).
public struct EmbeddingRequestLimits: Codable, Sendable, Equatable {
    public static let maxItems: Int = 16
    public static let maxBlocksPerItem: Int = 32
    public static let maxTextChars: Int = 131_072
    public static let maxImageOrAudioBytes: Int = 32 * 1024 * 1024
    public static let maxVideoBytes: Int = 256 * 1024 * 1024
    public static let maxRequestMediaBytes: Int = 256 * 1024 * 1024
    public static let mediaBytesByKind: [String: Int] = [
        "image": maxImageOrAudioBytes,
        "audio": maxImageOrAudioBytes,
        "video": maxVideoBytes,
    ]

    public let items: Int
    public let blocksPerItem: Int
    public let textChars: Int
    public let mediaBytes: [String: Int]
    public let requestMediaBytes: Int

    public init(
        items: Int = Self.maxItems,
        blocksPerItem: Int = Self.maxBlocksPerItem,
        textChars: Int = Self.maxTextChars,
        mediaBytes: [String: Int] = Self.mediaBytesByKind,
        requestMediaBytes: Int = Self.maxRequestMediaBytes
    ) {
        self.items = items
        self.blocksPerItem = blocksPerItem
        self.textChars = textChars
        self.mediaBytes = mediaBytes
        self.requestMediaBytes = requestMediaBytes
    }

    public static let `default` = EmbeddingRequestLimits()

    enum CodingKeys: String, CodingKey {
        case items
        case blocksPerItem = "blocks_per_item"
        case textChars = "text_chars"
        case mediaBytes = "media_bytes"
        case requestMediaBytes = "request_media_bytes"
    }
}

/// Capabilities reported by `heylook`'s `GET /v1/models` and `GET /v1/capabilities` endpoints for an embedding model.
public struct HeylookServerCapabilities: Codable, Sendable, Equatable {
    public let modelID: String
    public let limits: EmbeddingRequestLimits
    public let defaultWindowSeconds: Double
    public let videoSamplingModes: [String]
    public let prompts: [String: String]

    public var maxItems: Int { limits.items }
    public var maxRequestMediaBytes: Int { limits.requestMediaBytes }

    public init(
        modelID: String,
        limits: EmbeddingRequestLimits = .default,
        defaultWindowSeconds: Double = 32.0,
        videoSamplingModes: [String] = ["fps", "content"],
        prompts: [String: String] = [:]
    ) {
        self.modelID = modelID
        self.limits = limits
        self.defaultWindowSeconds = defaultWindowSeconds
        self.videoSamplingModes = videoSamplingModes
        self.prompts = prompts
    }

    public init(
        modelID: String,
        maxItems: Int,
        maxRequestMediaBytes: Int = EmbeddingRequestLimits.maxRequestMediaBytes,
        defaultWindowSeconds: Double = 32.0,
        videoSamplingModes: [String] = ["fps", "content"],
        prompts: [String: String] = [:]
    ) {
        self.init(
            modelID: modelID,
            limits: EmbeddingRequestLimits(
                items: maxItems,
                requestMediaBytes: maxRequestMediaBytes
            ),
            defaultWindowSeconds: defaultWindowSeconds,
            videoSamplingModes: videoSamplingModes,
            prompts: prompts
        )
    }
}

/// HTTP `EmbeddingEngine` talking to `heylook`'s `/v1/embeddings` endpoint with bounded
/// client-side sub-batching (`<= 16` items, `<= 192 MiB` raw media bytes) and `DELETE /v1/requests/{id}` cancellation.
public actor HeylookMLXEngine: EmbeddingEngine {
    public static let defaultMaxItemsPerRequest: Int = EmbeddingRequestLimits.maxItems
    public static let defaultMaxRequestMediaBytes: Int = EmbeddingRequestLimits.maxRequestMediaBytes

    public nonisolated let baseURL: URL
    public nonisolated let modelID: String
    public nonisolated let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public nonisolated let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions

    private var maxItemsPerRequest: Int
    private var maxRequestMediaBytes: Int
    private var didFetchCapabilities: Bool = false
    private let session: URLSession

    public init(
        baseURL: URL = URL(string: "http://127.0.0.1:8000")!,
        modelID: String = "embeddinggemma-2-MLX-BF16",
        maxItemsPerRequest: Int = HeylookMLXEngine.defaultMaxItemsPerRequest,
        maxRequestMediaBytes: Int = HeylookMLXEngine.defaultMaxRequestMediaBytes,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.modelID = modelID
        self.maxItemsPerRequest = max(1, maxItemsPerRequest)
        self.maxRequestMediaBytes = max(1, maxRequestMediaBytes)
        self.session = session
    }

    public static func planBatches(
        _ inputs: [EmbeddingInput],
        maxItems: Int = defaultMaxItemsPerRequest,
        maxRequestMediaBytes: Int = defaultMaxRequestMediaBytes
    ) -> [[EmbeddingInput]] {
        guard !inputs.isEmpty else { return [] }
        let capItems = max(1, maxItems)
        let maxRawBytes = max(1, Int(Double(maxRequestMediaBytes) * 0.75))

        var batches = [[EmbeddingInput]]()
        var current = [EmbeddingInput]()
        var currentBytes = 0

        for item in inputs {
            let itemBytes = item.parts.reduce(0) { $0 + $1.estimatedBytes }
            if !current.isEmpty && (current.count >= capItems || currentBytes + itemBytes > maxRawBytes) {
                batches.append(current)
                current = []
                currentBytes = 0
            }
            current.append(item)
            currentBytes += itemBytes
        }
        if !current.isEmpty {
            batches.append(current)
        }
        return batches
    }

    public func fetchCapabilities() async throws -> HeylookServerCapabilities {
        let url = baseURL.appendingPathComponent("v1/models")
        let (data, response) = try await session.data(from: url)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 200
        guard (200..<300).contains(status) else {
            throw EmbeddingEngineError.invalidResponse(
                status: status,
                detail: String(data: data, encoding: .utf8) ?? "HTTP \(status)"
            )
        }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = root["data"] as? [[String: Any]]
        else {
            return HeylookServerCapabilities(modelID: modelID)
        }
        func extractEmbeddingDict(_ row: [String: Any]) -> [String: Any]? {
            if let direct = row["embedding"] as? [String: Any] { return direct }
            if let engine = row["engine"] as? [String: Any],
               let nested = engine["embedding"] as? [String: Any] {
                return nested
            }
            return nil
        }
        let matching = rows.first {
            let id = ($0["id"] as? String) ?? ($0["modelID"] as? String)
            return id == modelID && extractEmbeddingDict($0) != nil
        } ?? rows.first { extractEmbeddingDict($0) != nil }

        guard let matching,
              let emb = extractEmbeddingDict(matching)
        else {
            return HeylookServerCapabilities(modelID: modelID)
        }
        let resolvedID = (matching["id"] as? String) ?? (matching["modelID"] as? String) ?? modelID
        let limitsDict = emb["limits"] as? [String: Any] ?? [:]
        let maxItems = (limitsDict["items"] as? Int) ?? Self.defaultMaxItemsPerRequest
        let blocksPerItem = (limitsDict["blocks_per_item"] as? Int) ?? EmbeddingRequestLimits.maxBlocksPerItem
        let textChars = (limitsDict["text_chars"] as? Int) ?? EmbeddingRequestLimits.maxTextChars
        let mediaBytes = (limitsDict["media_bytes"] as? [String: Int]) ?? EmbeddingRequestLimits.mediaBytesByKind
        let maxMediaBytes = (limitsDict["request_media_bytes"] as? Int) ?? Self.defaultMaxRequestMediaBytes
        let videoWin = emb["video_windows"] as? [String: Any] ?? [:]
        let winSec = (videoWin["default_seconds"] as? Double)
            ?? Double((videoWin["default_seconds"] as? Int) ?? 32)
        let samplingModes = (videoWin["sampling"] as? [String]) ?? ["fps", "content"]
        let prompts = (emb["prompts"] as? [String: String]) ?? [:]

        let caps = HeylookServerCapabilities(
            modelID: resolvedID,
            limits: EmbeddingRequestLimits(
                items: maxItems,
                blocksPerItem: blocksPerItem,
                textChars: textChars,
                mediaBytes: mediaBytes,
                requestMediaBytes: maxMediaBytes
            ),
            defaultWindowSeconds: winSec,
            videoSamplingModes: samplingModes,
            prompts: prompts
        )
        self.maxItemsPerRequest = max(1, caps.maxItems)
        self.maxRequestMediaBytes = max(1, caps.maxRequestMediaBytes)
        self.didFetchCapabilities = true
        return caps
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        guard !inputs.isEmpty else { return [] }
        if !didFetchCapabilities {
            self.didFetchCapabilities = true
            _ = try? await fetchCapabilities()
        }
        let batches = Self.planBatches(
            inputs,
            maxItems: maxItemsPerRequest,
            maxRequestMediaBytes: maxRequestMediaBytes
        )
        guard !batches.isEmpty else { return [] }

        var allOutputs = [EmbeddedOutput]()
        allOutputs.reserveCapacity(inputs.count)
        for batch in batches {
            try Task.checkCancellation()
            let batchOutputs = try await embedSingleBatch(batch, dimensions: dimensions)
            allOutputs.append(contentsOf: batchOutputs)
        }
        return allOutputs
    }

    private func embedSingleBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        guard !inputs.isEmpty else { return [] }
        let endpoint = baseURL.appendingPathComponent("v1/embeddings")
        let reqID = "heylook-\(UUID().uuidString.lowercased())"

        let wireItems: [[String: Any]] = try inputs.enumerated().map { itemIndex, item in
            let hasText = item.hasNonWhitespaceText
            let resolvedWindows = try item.resolvedVideoWindows(itemIndex: itemIndex)
            let isDocumentWithTitle = !item.usesExplicitPromptName
                && hasText
                && (item.task == .document || item.task == .retrievalDocument)
                && (item.title?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)
            var appliedTitlePrefix = false

            let blocks: [[String: Any]] = try item.parts.map { part in
                switch part {
                case let .text(text):
                    if isDocumentWithTitle && !appliedTitlePrefix
                        && !text.hasPrefix("title:") && !text.hasPrefix("task:") {
                        appliedTitlePrefix = true
                        let formatted = EmbeddingTask.documentPrompt(title: item.title, text: text)
                        return ["type": "text", "text": formatted]
                    }
                    return ["type": "text", "text": text]
                case let .media(kind, mimeType, _, _, windowSampling):
                    let rawData = try part.resolvedData()
                    var block: [String: Any] = [
                        "type": kind.rawValue,
                        "source": [
                            "type": "base64",
                            "media_type": mimeType,
                            "data": rawData.base64EncodedString(),
                        ],
                    ]
                    if kind == .video, let opts = resolvedWindows {
                        var winPayload: [String: Any] = ["sampling": opts.sampling.rawValue]
                        if let secs = opts.seconds {
                            winPayload["seconds"] = secs
                        }
                        block["windows"] = winPayload
                    } else if kind == .video && windowSampling != .off {
                        block["windows"] = ["sampling": windowSampling.rawValue]
                    }
                    return block
                }
            }
            var entry: [String: Any] = ["content": blocks]
            if item.usesExplicitPromptName {
                if let promptName = item.promptName {
                    entry["prompt_name"] = promptName
                }
            } else if hasText, !appliedTitlePrefix, let promptName = item.task.promptName {
                entry["prompt_name"] = promptName
            }
            return entry
        }

        var body: [String: Any] = [
            "model": modelID,
            "dimensions": dimensions ?? nativeDimensions,
            "input": wireItems,
        ]
        if let dimensions {
            body["dimensions"] = dimensions
        }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(reqID, forHTTPHeaderField: "X-Request-ID")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let maxAttempts = 4
        for attempt in 0..<maxAttempts {
            do {
                let (data, response) = try await session.data(for: request)
                let http = response as? HTTPURLResponse
                let status = http?.statusCode ?? 200
                if status == 429 || status == 503 {
                    if attempt + 1 < maxAttempts {
                        try await Task.sleep(nanoseconds: UInt64(1_500_000_000 * (attempt + 1)))
                        continue
                    }
                }
                guard (200..<300).contains(status) else {
                    let detail = (try? JSONDecoder().decode(ErrorEnvelope.self, from: data).detail)
                        ?? String(data: data, encoding: .utf8)
                        ?? "HTTP \(status)"
                    throw EmbeddingEngineError.invalidResponse(status: status, detail: detail)
                }
                let decoded = try JSONDecoder().decode(WireEmbeddingResult.self, from: data)
                guard decoded.data.count == inputs.count else {
                    throw EmbeddingEngineError.vectorCountMismatch(
                        expected: inputs.count,
                        actual: decoded.data.count
                    )
                }
                let sortedRows = decoded.data.sorted(by: { $0.index < $1.index })
                let totalTokens = decoded.input_tokens
                return sortedRows.enumerated().map { idx, row in
                    let wins = (row.windows ?? []).map {
                        VideoWindowEmbeddingResult(
                            start: $0.start,
                            end: $0.end,
                            frameTimes: $0.frame_times,
                            vector: $0.embedding
                        )
                    }
                    return EmbeddedOutput(
                        vector: row.embedding,
                        windows: wins,
                        inputTokens: idx == 0 ? totalTokens : 0,
                        promptName: row.prompt_name
                    )
                }
            } catch is CancellationError {
                await cancelRemoteRequest(id: reqID)
                throw CancellationError()
            }
        }
        throw EmbeddingEngineError.invalidResponse(status: 503, detail: "Retries exhausted")
    }

    private func cancelRemoteRequest(id: String) async {
        let cancelURL = baseURL
            .appendingPathComponent("v1/requests")
            .appendingPathComponent(id)
        var req = URLRequest(url: cancelURL)
        req.httpMethod = "DELETE"
        _ = try? await session.data(for: req)
    }
}

private struct ErrorEnvelope: Decodable {
    let detail: String?
}

private struct WireEmbeddingResult: Decodable {
    struct WindowRow: Decodable {
        let start: Double
        let end: Double
        let frame_times: [Double]
        let embedding: [Float]
    }

    struct VectorRow: Decodable {
        let index: Int
        let embedding: [Float]
        let prompt_name: String?
        let windows: [WindowRow]?
    }

    let model: String
    let dimensions: Int
    let data: [VectorRow]
    let input_tokens: Int?
    let elapsed_ms: Double?
}
