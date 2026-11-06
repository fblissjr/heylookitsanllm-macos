import Accelerate
import AVFoundation
import CoreImage
import Foundation

#if canImport(MLXVLM)
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers
#endif

public enum EmbeddingEngineError: LocalizedError, Sendable {
    case invalidResponse(status: Int, detail: String)
    case vectorCountMismatch(expected: Int, actual: Int)
    case checkpointNotFound(URL)

    public var errorDescription: String? {
        switch self {
        case let .invalidResponse(status, detail):
            return "Embedding server error (\(status)): \(detail)"
        case let .vectorCountMismatch(expected, actual):
            return "Expected \(expected) vectors from engine, received \(actual)"
        case let .checkpointNotFound(url):
            return "EmbeddingGemma 2 checkpoint not found at \(url.path)"
        }
    }
}

/// Capabilities reported by `GET /v1/models` for an embedding model.
public struct HeylookServerCapabilities: Codable, Sendable, Equatable {
    public let modelID: String
    public let maxItems: Int
    public let maxRequestMediaBytes: Int
    public let defaultWindowSeconds: Double
    public let videoSamplingModes: [String]
    public let prompts: [String: String]

    public init(
        modelID: String,
        maxItems: Int = 16,
        maxRequestMediaBytes: Int = 256 * 1024 * 1024,
        defaultWindowSeconds: Double = 32.0,
        videoSamplingModes: [String] = ["fps", "content"],
        prompts: [String: String] = [:]
    ) {
        self.modelID = modelID
        self.maxItems = maxItems
        self.maxRequestMediaBytes = maxRequestMediaBytes
        self.defaultWindowSeconds = defaultWindowSeconds
        self.videoSamplingModes = videoSamplingModes
        self.prompts = prompts
    }
}

/// Standalone MLX Swift + ANE integration for `EmbeddingGemma 2` (`embedding_gemma2`).
public enum NativeMLXEmbeddingGemma2 {

    public static var defaultCheckpointURL: URL {
        if let envPath = ProcessInfo.processInfo.environment["HEYLOOK_EMBEDDING_CHECKPOINT"], !envPath.isEmpty {
            return URL(fileURLWithPath: (envPath as NSString).expandingTildeInPath)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Storage/llms/google/embeddinggemma-2-MLX-BF16")
    }

    public struct CheckpointDescriptor: Codable, Sendable, Equatable {
        public let modelType: String
        public let hiddenSize: Int
        public let embeddingDim: Int
        public let numHiddenLayers: Int
        public let maxInputTokens: Int
        public let videoFPS: Double
        public let videoMaxFrames: Int
        public let videoMaxSoftTokens: Int
        public let imageMaxSoftTokens: Int
        public let prompts: [String: String]
        public let poolingModeMeanTokens: Bool

        public var maxWindowSeconds: Double {
            videoFPS > 0 ? Double(videoMaxFrames) / videoFPS : 32.0
        }

        public static func load(from directory: URL = defaultCheckpointURL) throws -> CheckpointDescriptor {
            let configURL = directory.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else {
                throw EmbeddingEngineError.checkpointNotFound(directory)
            }
            let rawConfig = try JSONSerialization.jsonObject(
                with: Data(contentsOf: configURL)
            ) as? [String: Any] ?? [:]

            let textConfig = rawConfig["text_config"] as? [String: Any] ?? rawConfig
            let modelType = (rawConfig["model_type"] as? String) ?? "embedding_gemma2"
            let hiddenSize = (textConfig["hidden_size"] as? Int) ?? 512
            let embeddingDim = (textConfig["embedding_dim"] as? Int) ?? 768
            let numHiddenLayers = (textConfig["num_hidden_layers"] as? Int) ?? 24

            let procURL = directory.appendingPathComponent("processor_config.json")
            let rawProc = (try? JSONSerialization.jsonObject(with: Data(contentsOf: procURL)) as? [String: Any]) ?? [:]
            let imageProc = rawProc["image_processor"] as? [String: Any] ?? [:]
            let videoProc = rawProc["video_processor"] as? [String: Any] ?? [:]
            let fps = (videoProc["fps"] as? Double) ?? Double((videoProc["fps"] as? Int) ?? 1)
            let maxFrames = (videoProc["max_frames"] as? Int) ?? 32
            let videoSoftTokens = (videoProc["max_soft_tokens"] as? Int) ?? 140
            let imageSoftTokens = (imageProc["max_soft_tokens"] as? Int) ?? 280

            let stURL = directory.appendingPathComponent("config_sentence_transformers.json")
            let rawST = (try? JSONSerialization.jsonObject(with: Data(contentsOf: stURL)) as? [String: Any]) ?? [:]
            let prompts = (rawST["prompts"] as? [String: String]) ?? [:]

            let poolURL = directory.appendingPathComponent("1_Pooling/config.json")
            let rawPool = (try? JSONSerialization.jsonObject(with: Data(contentsOf: poolURL)) as? [String: Any]) ?? [:]
            let meanPool = (rawPool["pooling_mode_mean_tokens"] as? Bool) ?? true

            return CheckpointDescriptor(
                modelType: modelType,
                hiddenSize: hiddenSize,
                embeddingDim: embeddingDim,
                numHiddenLayers: numHiddenLayers,
                maxInputTokens: 8192,
                videoFPS: fps,
                videoMaxFrames: maxFrames,
                videoMaxSoftTokens: videoSoftTokens,
                imageMaxSoftTokens: imageSoftTokens,
                prompts: prompts,
                poolingModeMeanTokens: meanPool
            )
        }
    }

    public static func activePatchCount(positionIDs: [[[Int]]]) -> Int {
        var maxValid = 0
        for grid in positionIDs {
            var validInGrid = 0
            for coord in grid where coord.count == 2 && (coord[0] != -1 || coord[1] != -1) {
                validInGrid += 1
            }
            if validInGrid > maxValid {
                maxValid = validInGrid
            }
        }
        return maxValid
    }

    public static func gatherPositionEmbeddings(
        tableX: [[Float]],
        tableY: [[Float]],
        patchPositions: [(x: Int, y: Int)]
    ) -> [[Float]] {
        let dim = tableX.first?.count ?? 0
        return patchPositions.map { pos in
            if pos.x < 0 || pos.y < 0 {
                return [Float](repeating: 0, count: dim)
            }
            let rx = tableX[min(max(0, pos.x), tableX.count - 1)]
            let ry = tableY[min(max(0, pos.y), tableY.count - 1)]
            var out = [Float](repeating: 0, count: dim)
            vDSP_vadd(rx, 1, ry, 1, &out, 1, vDSP_Length(dim))
            return out
        }
    }

    public static func meanPoolAndNormalize(
        tokenHiddenStates: [[Float]],
        attentionMask: [Int],
        dimensions: Int = Matryoshka.gemma2NativeDimensions
    ) -> [Float] {
        guard let hiddenWidth = tokenHiddenStates.first?.count, hiddenWidth > 0 else {
            return []
        }
        var sum = [Float](repeating: 0, count: hiddenWidth)
        var activeCount: Float = 0
        for (row, mask) in zip(tokenHiddenStates, attentionMask) where mask != 0 {
            vDSP_vadd(sum, 1, row, 1, &sum, 1, vDSP_Length(hiddenWidth))
            activeCount += Float(mask)
        }
        if activeCount > 0 {
            var inv = 1.0 / activeCount
            vDSP_vsmul(sum, 1, &inv, &sum, 1, vDSP_Length(hiddenWidth))
        }
        return Matryoshka.project(sum, dimensions: dimensions)
    }

    public static var isAvailableInProcess: Bool {
        #if canImport(MLXVLM) && canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        return true
        #else
        return false
        #endif
    }
}

/// HTTP/SSE client engine speaking `/v1/embeddings` and `/v1/models` with bounded media batching.
public actor HeylookMLXEngine: EmbeddingEngine {
    public static let defaultMaxItemsPerRequest = 16
    public static let defaultMaxRequestMediaBytes = 256 * 1024 * 1024

    public nonisolated let baseURL: URL
    public nonisolated let modelID: String
    public nonisolated let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public nonisolated let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions

    private var maxItemsPerRequest: Int
    private var maxRequestMediaBytes: Int
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
              let rows = root["data"] as? [[String: Any]],
              let matching = rows.first(where: { ($0["id"] as? String) == modelID && $0["embedding"] != nil })
                  ?? rows.first(where: { $0["embedding"] != nil }),
              let emb = matching["embedding"] as? [String: Any]
        else {
            return HeylookServerCapabilities(modelID: modelID)
        }
        let limits = emb["limits"] as? [String: Any] ?? [:]
        let maxItems = (limits["items"] as? Int) ?? Self.defaultMaxItemsPerRequest
        let maxMedia = (limits["request_media_bytes"] as? Int) ?? Self.defaultMaxRequestMediaBytes
        let prompts = (emb["prompts"] as? [String: String]) ?? [:]
        self.maxItemsPerRequest = max(1, maxItems)
        self.maxRequestMediaBytes = max(1, maxMedia)
        return HeylookServerCapabilities(
            modelID: (matching["id"] as? String) ?? modelID,
            maxItems: maxItems,
            maxRequestMediaBytes: maxMedia,
            prompts: prompts
        )
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        guard !inputs.isEmpty else { return [] }
        let batches = Self.planBatches(
            inputs,
            maxItems: maxItemsPerRequest,
            maxRequestMediaBytes: maxRequestMediaBytes
        )
        var allOutputs: [EmbeddedOutput] = []
        allOutputs.reserveCapacity(inputs.count)

        for batch in batches {
            let batchOut = try await sendSingleBatch(batch, dimensions: dimensions)
            allOutputs.append(contentsOf: batchOut)
        }
        return allOutputs
    }

    private func sendSingleBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int?
    ) async throws -> [EmbeddedOutput] {
        var wireInputs: [[String: Any]] = []
        for item in inputs {
            var contentBlocks: [[String: Any]] = []
            for part in item.parts {
                switch part {
                case .text(let s):
                    contentBlocks.append(["type": "text", "text": s])
                case let .media(kind, mimeType, _, _, sampling):
                    let raw = try part.resolvedData()
                    var block: [String: Any] = [
                        "type": kind.rawValue,
                        "source": [
                            "type": "base64",
                            "media_type": mimeType,
                            "data": raw.base64EncodedString(),
                        ],
                    ]
                    if kind == .video && sampling != .off {
                        block["window_sampling"] = sampling.rawValue
                    }
                    contentBlocks.append(block)
                }
            }
            var row: [String: Any] = ["content": contentBlocks]
            if item.hasNonWhitespaceText, let promptName = item.task.promptName {
                row["task"] = promptName
            }
            if let title = item.title {
                row["title"] = title
            }
            wireInputs.append(row)
        }

        var body: [String: Any] = [
            "model": modelID,
            "input": wireInputs,
        ]
        if let dimensions {
            body["dimensions"] = dimensions
        }

        let url = baseURL.appendingPathComponent("v1/embeddings")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: req)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 200
        guard (200..<300).contains(status) else {
            throw EmbeddingEngineError.invalidResponse(
                status: status,
                detail: String(data: data, encoding: .utf8) ?? "HTTP \(status)"
            )
        }

        struct WireResponse: Decodable {
            struct WindowRow: Decodable {
                let start: Double
                let end: Double
                let frame_times: [Double]?
                let embedding: [Float]
            }
            struct DataRow: Decodable {
                let index: Int
                let embedding: [Float]
                let windows: [WindowRow]?
            }
            let data: [DataRow]
        }

        let decoded = try JSONDecoder().decode(WireResponse.self, from: data)
        guard decoded.data.count == inputs.count else {
            throw EmbeddingEngineError.vectorCountMismatch(expected: inputs.count, actual: decoded.data.count)
        }
        return decoded.data.sorted(by: { $0.index < $1.index }).map { row in
            let wins = (row.windows ?? []).map {
                EmbeddedOutput.WindowVector(
                    start: $0.start,
                    end: $0.end,
                    frameTimes: $0.frame_times ?? [],
                    vector: $0.embedding
                )
            }
            return EmbeddedOutput(vector: row.embedding, windows: wins)
        }
    }
}
