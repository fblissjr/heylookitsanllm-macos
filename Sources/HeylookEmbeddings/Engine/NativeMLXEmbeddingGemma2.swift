import Accelerate
import AVFoundation
import CoreImage
import Foundation

#if canImport(MLXVLM)
import MLX
import MLXLMCommon
import MLXVLM
import Tokenizers

/// Direct `MLXLMCommon.TokenizerLoader` bridge to `swift-transformers` (`Tokenizers.AutoTokenizer`),
/// avoiding the external `MLXHuggingFaceMacros` (`swift-syntax`) compiler plugin dependency.
public struct HeylookTransformersTokenizerLoader: MLXLMCommon.TokenizerLoader, Sendable {
    private struct TokenizerBridge: MLXLMCommon.Tokenizer {
        let upstream: any Tokenizers.Tokenizer

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }

        func convertTokenToId(_ token: String) -> Int? {
            upstream.convertTokenToId(token)
        }

        func convertIdToToken(_ id: Int) -> String? {
            upstream.convertIdToToken(id)
        }

        var bosToken: String? { upstream.bosToken }
        var eosToken: String? { upstream.eosToken }
        var unknownToken: String? { upstream.unknownToken }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            do {
                return try upstream.applyChatTemplate(
                    messages: messages,
                    tools: tools,
                    additionalContext: additionalContext
                )
            } catch Tokenizers.TokenizerError.missingChatTemplate {
                throw MLXLMCommon.TokenizerError.missingChatTemplate
            }
        }
    }

    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(upstream: upstream)
    }
}
#endif

/// Standalone MLX Swift + CoreML ANE integration for `EmbeddingGemma 2` (`embedding_gemma2`).
public enum NativeMLXEmbeddingGemma2 {
    public static var defaultCheckpointURL: URL {
        if let envPath = ProcessInfo.processInfo.environment["HEYLOOK_EMBEDDING_CHECKPOINT"], !envPath.isEmpty {
            return URL(fileURLWithPath: envPath)
        }
        #if os(macOS)
        let baseDir = FileManager.default.homeDirectoryForCurrentUser
        #else
        let baseDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        #endif
        return baseDir.appendingPathComponent(".heylook/models/embeddinggemma-2-MLX-BF16")
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
            guard videoFPS > 0 else { return 32.0 }
            return Double(videoMaxFrames) / videoFPS
        }

        public static func load(from directory: URL = defaultCheckpointURL) throws -> CheckpointDescriptor {
            let configURL = directory.appendingPathComponent("config.json")
            let stURL = directory.appendingPathComponent("config_sentence_transformers.json")
            let poolURL = directory.appendingPathComponent("1_Pooling/config.json")
            let procURL = directory.appendingPathComponent("processor_config.json")

            guard FileManager.default.fileExists(atPath: configURL.path) else {
                throw CocoaError(.fileNoSuchFile)
            }

            let configData = try Data(contentsOf: configURL)
            let configJSON = (try JSONSerialization.jsonObject(with: configData) as? [String: Any]) ?? [:]
            let textConfig = (configJSON["text_config"] as? [String: Any]) ?? [:]
            let visionConfig = (configJSON["vision_config"] as? [String: Any]) ?? [:]

            var prompts = [String: String]()
            var maxSeqLength = 8192
            if let stData = try? Data(contentsOf: stURL),
               let stJSON = try? JSONSerialization.jsonObject(with: stData) as? [String: Any] {
                if let p = stJSON["prompts"] as? [String: String] {
                    prompts = p
                }
                if let maxLen = stJSON["max_seq_length"] as? Int {
                    maxSeqLength = maxLen
                }
            }

            var meanTokens = true
            var wordEmbeddingDimension = 768
            if let poolData = try? Data(contentsOf: poolURL),
               let poolJSON = try? JSONSerialization.jsonObject(with: poolData) as? [String: Any] {
                meanTokens = (poolJSON["pooling_mode_mean_tokens"] as? Bool) ?? true
                wordEmbeddingDimension = (poolJSON["word_embedding_dimension"] as? Int) ?? 768
            }

            var fps = 1.0
            var numFrames = 32
            var videoSoftTokens = 140
            var imageSoftTokens = (visionConfig["default_output_length"] as? Int) ?? 280
            if let procData = try? Data(contentsOf: procURL),
               let procJSON = try? JSONSerialization.jsonObject(with: procData) as? [String: Any] {
                if let vp = procJSON["video_processor"] as? [String: Any] {
                    fps = (vp["fps"] as? Double) ?? Double((vp["fps"] as? Int) ?? 1)
                    numFrames = (vp["num_frames"] as? Int) ?? 32
                    videoSoftTokens = (vp["max_soft_tokens"] as? Int) ?? 140
                }
                if let ip = procJSON["image_processor"] as? [String: Any],
                   let maxSoft = ip["max_soft_tokens"] as? Int {
                    imageSoftTokens = maxSoft
                }
            }

            return CheckpointDescriptor(
                modelType: (configJSON["model_type"] as? String) ?? "embedding_gemma2",
                hiddenSize: (textConfig["hidden_size"] as? Int) ?? 512,
                embeddingDim: (configJSON["embedding_dim"] as? Int) ?? wordEmbeddingDimension,
                numHiddenLayers: (textConfig["num_hidden_layers"] as? Int) ?? 24,
                maxInputTokens: maxSeqLength,
                videoFPS: fps,
                videoMaxFrames: numFrames,
                videoMaxSoftTokens: videoSoftTokens,
                imageMaxSoftTokens: imageSoftTokens,
                prompts: prompts,
                poolingModeMeanTokens: meanTokens
            )
        }
    }

    /// Masked mean pooling followed by Matryoshka truncation and L2 normalization (`vDSP`).
    public static func maskedMeanPoolAndNormalize(
        projectedTokens: [[Float]],
        attentionMask: [Int],
        dimensions: Int = Matryoshka.gemma2NativeDimensions
    ) -> [Float] {
        guard let first = projectedTokens.first, !first.isEmpty else { return [] }
        let hiddenWidth = first.count
        var sum = [Float](repeating: 0, count: hiddenWidth)
        var activeCount: Float = 0

        for (rowIdx, tokenVec) in projectedTokens.enumerated() where tokenVec.count == hiddenWidth {
            let maskVal = rowIdx < attentionMask.count ? attentionMask[rowIdx] : 1
            guard maskVal != 0 else { continue }
            vDSP_vadd(sum, 1, tokenVec, 1, &sum, 1, vDSP_Length(hiddenWidth))
            activeCount += 1
        }

        if activeCount > 0 {
            var inv = 1.0 / activeCount
            vDSP_vsmul(sum, 1, &inv, &sum, 1, vDSP_Length(hiddenWidth))
        }
        return Matryoshka.project(sum, dimensions: dimensions)
    }

    public static var isAvailableInProcess: Bool {
        #if canImport(MLXVLM) && canImport(MLXLMCommon) && canImport(Tokenizers)
        return true
        #else
        return false
        #endif
    }

    /// Creates the best available in-process `EmbeddingEngine` for the current build and checkpoint.
    public static func makeBestAvailableEngine(
        checkpointURL: URL = defaultCheckpointURL,
        computeOptions: ModelComputeOptions = .allMLXGPU
    ) -> any EmbeddingEngine {
        #if canImport(MLXVLM)
        if FileManager.default.fileExists(atPath: checkpointURL.appendingPathComponent("config.json").path) {
            return MLXSwiftLMEmbeddingGemma2Engine(
                checkpointURL: checkpointURL,
                computeOptions: computeOptions
            )
        }
        #endif
        return HybridLocalEmbeddingEngine(
            checkpointURL: checkpointURL,
            computeOptions: computeOptions
        )
    }
}

/// Hybrid local `EmbeddingEngine` used when `MLXVLM` is not linked (e.g. fast `swift build`):
/// - Uses `ArtifactProcessor` + `WindowPlanner` for real video/audio window planning and frame extraction.
/// - Uses `ANECoreMLRuntime` when compiled CoreML `.mlmodelc` towers are present on disk.
/// - Falls back to `DeterministicHashEmbeddingEngine` so indexing, search, and UI work offline without external servers.
public actor HybridLocalEmbeddingEngine: EmbeddingEngine {
    public nonisolated let modelID: String
    public nonisolated let checkpointURL: URL
    public nonisolated let computeOptions: ModelComputeOptions
    public nonisolated let hasCompiledANETowers: Bool
    public nonisolated let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public nonisolated let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions

    private let aneRuntime: ANECoreMLRuntime
    private let fallbackHash: DeterministicHashEmbeddingEngine

    public init(
        checkpointURL: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL,
        computeOptions: ModelComputeOptions = .allMLXGPU
    ) {
        self.checkpointURL = checkpointURL
        self.computeOptions = computeOptions
        let anePresent = ANETowerInspector.locateBundle(for: .textBackbone, in: checkpointURL) != nil
        self.hasCompiledANETowers = anePresent
        self.modelID = anePresent ? checkpointURL.lastPathComponent : "deterministic-hash-768d"
        self.aneRuntime = ANECoreMLRuntime(checkpointDirectory: checkpointURL)
        self.fallbackHash = DeterministicHashEmbeddingEngine(modelID: "deterministic-hash-768d")
    }

    public func unload() async {
        await aneRuntime.unload()
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        let targetDim = dimensions ?? nativeDimensions
        var outputs = [EmbeddedOutput]()
        outputs.reserveCapacity(inputs.count)

        for item in inputs {
            try Task.checkCancellation()
            var seedText = item.title ?? ""
            var windowedVideoRequest: (url: URL, sampling: VideoWindowSampling)?

            for part in item.parts {
                switch part {
                case .text(let s):
                    seedText += " " + s
                case let .media(kind, mimeType, data, url, windowSampling):
                    seedText += " \(kind.rawValue) \(mimeType) \(url?.lastPathComponent ?? "") \(data?.count ?? 0)"
                    if kind == .video, let url, windowSampling != .off, windowedVideoRequest == nil {
                        windowedVideoRequest = (url, windowSampling)
                    }
                }
            }

            let primary: [Float]
            if let aneVec = try await tryEmbedItemOnANE(item: item, targetDim: targetDim) {
                primary = aneVec
            } else {
                primary = fallbackHash.embed(text: seedText, dimensions: targetDim)
            }
            var windows = [EmbeddedOutput.WindowVector]()

            if let (videoURL, sampling) = windowedVideoRequest,
               let candidates = try? await ArtifactProcessor.extractVideoCandidates(for: videoURL, sampling: sampling) {
                let planned = WindowPlanner.planWindows(
                    times: candidates.times,
                    duration: candidates.duration,
                    windowSeconds: 32.0,
                    strideSeconds: 16.0,
                    sampling: sampling,
                    fps: 1.0,
                    maxFrames: 32,
                    thumbnails: candidates.thumbnails
                )
                if planned.count > 1 {
                    for win in planned {
                        let winVec = fallbackHash.embed(
                            text: "\(seedText) window:\(win.start)-\(win.end)",
                            dimensions: targetDim
                        )
                        windows.append(
                            EmbeddedOutput.WindowVector(
                                start: win.start,
                                end: win.end,
                                frameTimes: win.frameTimes,
                                vector: winVec
                            )
                        )
                    }
                }
            }

            outputs.append(EmbeddedOutput(vector: primary, windows: windows))
        }
        return outputs
    }

    private func tryEmbedItemOnANE(item: EmbeddingInput, targetDim: Int) async throws -> [Float]? {
        guard hasCompiledANETowers else { return nil }
        let visionUnit: TowerComputeUnit = computeOptions.visionEncoder == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.visionEncoder
        let audioUnit: TowerComputeUnit = computeOptions.audioEncoder == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.audioEncoder
        let textUnit: TowerComputeUnit = computeOptions.textBackbone == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.textBackbone

        var tokenIDs: [Int] = [2] // <bos>
        var softTokens: [Float] = []

        for part in item.parts {
            switch part {
            case .text(let s):
                let formatted = (item.task == .document || item.task == .retrievalDocument)
                    ? EmbeddingTask.documentPrompt(title: item.title, text: s)
                    : (item.task.promptPrefix + s)
                for scalar in formatted.unicodeScalars.prefix(512) {
                    tokenIDs.append(100 + Int(scalar.value % 250_000))
                }
            case let .media(kind, _, _, url, _):
                guard let url else { continue }
                if kind == .image {
                    guard let prep = ANECoreMLRuntime.prepareImagePixels(from: url, forVideoBudget: false),
                          let enc = try await aneRuntime.encodeVision(
                              pixels: prep.pixels,
                              height: prep.height,
                              width: prep.width,
                              unit: visionUnit
                          )
                    else { return nil }
                    tokenIDs.append(255999)
                    tokenIDs.append(contentsOf: repeatElement(258880, count: enc.tokenCount))
                    tokenIDs.append(258882)
                    softTokens.append(contentsOf: enc.softTokens)
                } else if kind == .audio {
                    guard let pcm = try? await ArtifactProcessor.decodeMonoPCM(for: url),
                          let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: pcm),
                          let enc = try await aneRuntime.encodeAudio(
                              melFeatures: mel.features,
                              frameCount: mel.frameCount,
                              unit: audioUnit
                          )
                    else { return nil }
                    tokenIDs.append(256000)
                    tokenIDs.append(contentsOf: repeatElement(258881, count: enc.tokenCount))
                    tokenIDs.append(258883)
                    softTokens.append(contentsOf: enc.softTokens)
                }
            }
        }
        tokenIDs.append(1) // <eos>
        return try await aneRuntime.encodeTextAndFuse(
            tokenIDs: tokenIDs,
            softTokens: softTokens.isEmpty ? nil : softTokens,
            unit: textUnit,
            dimensions: targetDim
        )
    }
}

#if canImport(MLXVLM)
/// In-process `EmbeddingEngine` backed directly by `mlx-swift-lm`'s `MLXVLM.EmbeddingGemma2Embedding`
/// and `ANECoreMLRuntime`.
public actor MLXSwiftLMEmbeddingGemma2Engine: EmbeddingEngine {
    public nonisolated let modelID: String
    public nonisolated let checkpointURL: URL
    public nonisolated let computeOptions: ModelComputeOptions
    public nonisolated let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public nonisolated let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions

    private let descriptor: NativeMLXEmbeddingGemma2.CheckpointDescriptor
    private let customTokenizerLoader: (any TokenizerLoader)?
    private let aneRuntime: ANECoreMLRuntime
    private var cachedTokenizer: (any MLXLMCommon.Tokenizer)?
    private var cachedEmbedder: MLXVLM.EmbeddingGemma2Embedding?

    public init(
        checkpointURL: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL,
        computeOptions: ModelComputeOptions = .allMLXGPU,
        tokenizerLoader: (any TokenizerLoader)? = nil
    ) {
        self.checkpointURL = checkpointURL
        self.computeOptions = computeOptions
        self.modelID = checkpointURL.lastPathComponent
        self.customTokenizerLoader = tokenizerLoader
        self.aneRuntime = ANECoreMLRuntime(checkpointDirectory: checkpointURL)
        self.descriptor = (try? NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: checkpointURL))
            ?? Self.fallbackDescriptor
    }

    public init(
        modelDirectory: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL,
        computeOptions: ModelComputeOptions = .allMLXGPU,
        tokenizerLoader: any TokenizerLoader
    ) async throws {
        self.checkpointURL = modelDirectory
        self.computeOptions = computeOptions
        self.modelID = modelDirectory.lastPathComponent
        self.customTokenizerLoader = tokenizerLoader
        self.aneRuntime = ANECoreMLRuntime(checkpointDirectory: modelDirectory)
        self.descriptor = (try? NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: modelDirectory))
            ?? Self.fallbackDescriptor
        self.cachedEmbedder = try await MLXVLM.EmbeddingGemma2Embedding(
            modelDirectory: modelDirectory,
            tokenizerLoader: tokenizerLoader,
            componentLoading: .onFirstUse
        )
    }

    private func resolvedTokenizer() async throws -> any MLXLMCommon.Tokenizer {
        if let cachedTokenizer {
            return cachedTokenizer
        }
        let loader: any TokenizerLoader = customTokenizerLoader ?? HeylookTransformersTokenizerLoader()
        let tok = try await loader.load(from: checkpointURL)
        self.cachedTokenizer = tok
        return tok
    }

    private func resolvedEmbedder() async throws -> MLXVLM.EmbeddingGemma2Embedding {
        if let cachedEmbedder {
            return cachedEmbedder
        }
        let loader: any TokenizerLoader = customTokenizerLoader ?? HeylookTransformersTokenizerLoader()
        let loaded = try await MLXVLM.EmbeddingGemma2Embedding(
            modelDirectory: checkpointURL,
            tokenizerLoader: loader,
            componentLoading: .onFirstUse
        )
        self.cachedEmbedder = loaded
        return loaded
    }

    public func unload() async {
        cachedEmbedder = nil
        cachedTokenizer = nil
        await aneRuntime.unload()
        MLX.Memory.clearCache()
    }

    private static let fallbackDescriptor = NativeMLXEmbeddingGemma2.CheckpointDescriptor(
        modelType: "embedding_gemma2",
        hiddenSize: 512,
        embeddingDim: 768,
        numHiddenLayers: 24,
        maxInputTokens: 8192,
        videoFPS: 1.0,
        videoMaxFrames: 32,
        videoMaxSoftTokens: 140,
        imageMaxSoftTokens: 280,
        prompts: [:],
        poolingModeMeanTokens: true
    )

    private enum ResolvedPart: Sendable {
        case text(String)
        case image(URL)
        case video(URL)
        case audio(URL)
    }

    nonisolated private static func buildVLMInput(
        parts: [ResolvedPart],
        title: String?
    ) -> sending MLXVLM.EmbeddingInput {
        let vlmParts: [MLXVLM.EmbeddingInput.Part] = parts.map { part in
            switch part {
            case .text(let s): return .text(s)
            case .image(let url): return .image(.url(url))
            case .video(let url): return .video(.url(url))
            case .audio(let url): return .audio(.url(url))
            }
        }
        return MLXVLM.EmbeddingInput(vlmParts, title: title)
    }

    nonisolated private static func buildWindowVLMInput(
        frames: [(time: Double, image: CGImage)],
        audioWAVURL: URL?
    ) -> sending MLXVLM.EmbeddingInput {
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)
        let ciOptions: [CIImageOption: Any] = sRGB.map { [.colorSpace: $0] } ?? [:]
        let videoFrames = frames.map { frame in
            UserInput.VideoFrame(
                image: .ciImage(CIImage(cgImage: frame.image, options: ciOptions)),
                timeStamp: CMTime(seconds: frame.time, preferredTimescale: 90_000)
            )
        }
        var winParts: [MLXVLM.EmbeddingInput.Part] = [.video(.frames(videoFrames))]
        if let audioWAVURL {
            winParts.append(.text(" "))
            winParts.append(.audio(.url(audioWAVURL)))
        }
        return MLXVLM.EmbeddingInput(winParts)
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        var outputs: [EmbeddedOutput] = []
        outputs.reserveCapacity(inputs.count)

        for item in inputs {
            try Task.checkCancellation()
            let vlmTask = Self.mapTask(item.task)
            var resolvedParts: [ResolvedPart] = []
            var tempURLs: [URL] = []
            defer {
                for url in tempURLs {
                    try? FileManager.default.removeItem(at: url)
                }
            }

            var windowedVideoRequest: (url: URL, sampling: VideoWindowSampling)?
            var hasCompanionAudio = false

            for part in item.parts {
                switch part {
                case .text(let s):
                    resolvedParts.append(.text(s))
                case let .media(kind, _, data, url, windowSampling):
                    let resolvedURL: URL
                    if let url {
                        resolvedURL = url
                    } else if let data {
                        let tempURL = FileManager.default.temporaryDirectory
                            .appendingPathComponent("heylook-\(UUID().uuidString).\(Self.ext(for: kind))")
                        try data.write(to: tempURL)
                        tempURLs.append(tempURL)
                        resolvedURL = tempURL
                    } else {
                        continue
                    }

                    switch kind {
                    case .image:
                        resolvedParts.append(.image(resolvedURL))
                    case .video:
                        resolvedParts.append(.video(resolvedURL))
                        if windowSampling != .off && windowedVideoRequest == nil {
                            windowedVideoRequest = (resolvedURL, windowSampling)
                        }
                    case .audio:
                        resolvedParts.append(.audio(resolvedURL))
                        hasCompanionAudio = true
                    default:
                        break
                    }
                }
            }

            let primary: [Float]
            if let aneVec = try await tryEmbedOnANE(
                parts: resolvedParts,
                item: item,
                dimensions: dimensions ?? nativeDimensions
            ) {
                primary = aneVec
            } else {
                let embedder = try await resolvedEmbedder()
                primary = try await embedder.embed(
                    Self.buildVLMInput(parts: resolvedParts, title: item.title),
                    task: vlmTask,
                    dimensions: dimensions
                )
            }

            var windowVectors: [EmbeddedOutput.WindowVector] = []
            if let (videoURL, sampling) = windowedVideoRequest {
                windowVectors = try await embedVideoWindows(
                    videoURL: videoURL,
                    sampling: sampling,
                    includeAudio: hasCompanionAudio,
                    task: vlmTask,
                    dimensions: dimensions
                )
            }
            outputs.append(EmbeddedOutput(vector: primary, windows: windowVectors))
        }
        return outputs
    }

    private func tryEmbedOnANE(
        parts: [ResolvedPart],
        item: EmbeddingInput,
        dimensions: Int
    ) async throws -> [Float]? {
        let hasImageOrVideo = parts.contains {
            if case .image = $0 { return true }
            if case .video = $0 { return true }
            return false
        }
        let hasAudio = parts.contains {
            if case .audio = $0 { return true }
            return false
        }
        let usesANEMedia = (hasImageOrVideo && computeOptions.visionEncoder != .mlxGPU)
            || (hasAudio && computeOptions.audioEncoder != .mlxGPU)

        guard computeOptions.textBackbone != .mlxGPU || usesANEMedia else { return nil }
        for part in parts {
            switch part {
            case .text:
                break
            case .image, .video:
                guard computeOptions.visionEncoder != .mlxGPU else { return nil }
            case .audio:
                guard computeOptions.audioEncoder != .mlxGPU else { return nil }
            }
        }

        let tokenizer = try await resolvedTokenizer()
        let bosID = tokenizer.bosToken.flatMap({ tokenizer.convertTokenToId($0) }) ?? 2
        let eosID = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("<eos>") ?? 1

        var tokenIDs: [Int] = [bosID]
        var softTokens: [Float] = []
        var didPrependPrompt = false

        for part in parts {
            switch part {
            case .text(let rawText):
                let formatted: String
                if !didPrependPrompt && item.hasNonWhitespaceText {
                    didPrependPrompt = true
                    if item.task == .document || item.task == .retrievalDocument {
                        formatted = EmbeddingTask.documentPrompt(title: item.title, text: rawText)
                    } else {
                        formatted = item.task.promptPrefix + rawText
                    }
                } else {
                    formatted = rawText
                }
                if !formatted.isEmpty {
                    tokenIDs.append(contentsOf: tokenizer.encode(text: formatted, addSpecialTokens: false))
                }
            case .image(let imageURL):
                guard let prep = ANECoreMLRuntime.prepareImagePixels(from: imageURL, forVideoBudget: false),
                      let encoded = try await aneRuntime.encodeVision(
                          pixels: prep.pixels,
                          height: prep.height,
                          width: prep.width,
                          unit: computeOptions.visionEncoder
                      )
                else {
                    return nil
                }
                tokenIDs.append(255999) // <boi>
                tokenIDs.append(contentsOf: repeatElement(258880, count: encoded.tokenCount))
                tokenIDs.append(258882) // <eoi>
                softTokens.append(contentsOf: encoded.softTokens)
            case .audio(let audioURL):
                let pcm = try await ArtifactProcessor.decodeMonoPCM(for: audioURL)
                guard let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: pcm),
                      let encoded = try await aneRuntime.encodeAudio(
                          melFeatures: mel.features,
                          frameCount: mel.frameCount,
                          unit: computeOptions.audioEncoder
                      )
                else {
                    return nil
                }
                tokenIDs.append(256000) // <boa>
                tokenIDs.append(contentsOf: repeatElement(258881, count: encoded.tokenCount))
                tokenIDs.append(258883) // <eoa>
                softTokens.append(contentsOf: encoded.softTokens)
            case .video(let videoURL):
                let candidates = try await ArtifactProcessor.extractVideoCandidates(
                    for: videoURL,
                    sampling: .fps
                )
                let frameIndices = WindowPlanner.pickFPS(
                    times: candidates.times,
                    lo: 0,
                    hi: min(candidates.duration, descriptor.maxWindowSeconds),
                    fps: descriptor.videoFPS,
                    maxFrames: descriptor.videoMaxFrames
                )
                let frameTimes = frameIndices.map { candidates.times[$0] }
                let decoded = try await ArtifactProcessor.decodeVideoFrames(
                    for: videoURL,
                    at: frameTimes
                )
                guard !decoded.isEmpty else { return nil }
                for frame in decoded {
                    guard let prep = ANECoreMLRuntime.prepareImagePixels(from: frame.image, forVideoBudget: true),
                          let encoded = try await aneRuntime.encodeVision(
                              pixels: prep.pixels,
                              height: prep.height,
                              width: prep.width,
                              unit: computeOptions.visionEncoder
                          )
                    else {
                        return nil
                    }
                    tokenIDs.append(255999) // <boi>
                    tokenIDs.append(contentsOf: repeatElement(258884, count: encoded.tokenCount))
                    tokenIDs.append(258882) // <eoi>
                    softTokens.append(contentsOf: encoded.softTokens)
                }
            }
        }
        tokenIDs.append(eosID)

        let textFusionUnit: TowerComputeUnit = (computeOptions.textBackbone == .mlxGPU)
            ? .cpuAndGPU
            : computeOptions.textBackbone

        return try await aneRuntime.encodeTextAndFuse(
            tokenIDs: tokenIDs,
            softTokens: softTokens.isEmpty ? nil : softTokens,
            unit: textFusionUnit,
            dimensions: dimensions
        )
    }

    private func embedVideoWindows(
        videoURL: URL,
        sampling: VideoWindowSampling,
        includeAudio: Bool,
        task: MLXVLM.EmbeddingTask,
        dimensions: Int?
    ) async throws -> [EmbeddedOutput.WindowVector] {
        let candidates = try await ArtifactProcessor.extractVideoCandidates(
            for: videoURL,
            sampling: sampling
        )
        let planned = WindowPlanner.planWindows(
            times: candidates.times,
            duration: candidates.duration,
            windowSeconds: descriptor.maxWindowSeconds,
            strideSeconds: descriptor.maxWindowSeconds / 2.0,
            sampling: sampling,
            fps: descriptor.videoFPS,
            maxFrames: descriptor.videoMaxFrames,
            thumbnails: candidates.thumbnails
        )
        guard planned.count > 1 else { return [] }

        let pcmSamples: [Float] = includeAudio
            ? ((try? await ArtifactProcessor.decodeMonoPCM(for: videoURL)) ?? [])
            : []

        var out: [EmbeddedOutput.WindowVector] = []
        out.reserveCapacity(planned.count)

        for win in planned {
            try Task.checkCancellation()
            let decoded = try await ArtifactProcessor.decodeVideoFrames(
                for: videoURL,
                at: win.frameTimes
            )
            guard !decoded.isEmpty else { continue }

            let vec: [Float]
            if let aneVec = try await tryEmbedWindowOnANE(
                frames: decoded,
                pcmSamples: pcmSamples,
                windowStart: win.start,
                windowEnd: win.end,
                dimensions: dimensions ?? nativeDimensions
            ) {
                vec = aneVec
            } else {
                var tempAudioURL: URL?
                defer {
                    if let tempAudioURL {
                        try? FileManager.default.removeItem(at: tempAudioURL)
                    }
                }
                if !pcmSamples.isEmpty,
                   let sliceWAV = ArtifactProcessor.encodeAudioSliceWAV(
                       samples: pcmSamples,
                       sampleRate: ArtifactProcessor.audioSampleRate,
                       startSeconds: win.start,
                       endSeconds: min(win.end, win.start + ArtifactProcessor.audioWindowSeconds)
                   ) {
                    let wavURL = FileManager.default.temporaryDirectory
                        .appendingPathComponent("heylook-win-audio-\(UUID().uuidString).wav")
                    if (try? sliceWAV.write(to: wavURL)) != nil {
                        tempAudioURL = wavURL
                    }
                }
                let embedder = try await resolvedEmbedder()
                vec = try await embedder.embed(
                    Self.buildWindowVLMInput(frames: decoded, audioWAVURL: tempAudioURL),
                    task: task,
                    dimensions: dimensions
                )
            }
            out.append(
                EmbeddedOutput.WindowVector(
                    start: win.start,
                    end: win.end,
                    frameTimes: decoded.map { ($0.time * 1000).rounded() / 1000 },
                    vector: vec
                )
            )
        }
        return out
    }

    private func tryEmbedWindowOnANE(
        frames: [(time: Double, image: CGImage)],
        pcmSamples: [Float],
        windowStart: Double,
        windowEnd: Double,
        dimensions: Int
    ) async throws -> [Float]? {
        guard computeOptions.visionEncoder != .mlxGPU else { return nil }
        if !pcmSamples.isEmpty && computeOptions.audioEncoder == .mlxGPU {
            return nil
        }
        let tokenizer = try await resolvedTokenizer()
        let bosID = tokenizer.bosToken.flatMap({ tokenizer.convertTokenToId($0) }) ?? 2
        let eosID = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("<eos>") ?? 1

        var tokenIDs: [Int] = [bosID]
        var softTokens: [Float] = []

        for frame in frames {
            guard let prep = ANECoreMLRuntime.prepareImagePixels(from: frame.image, forVideoBudget: true),
                  let encoded = try await aneRuntime.encodeVision(
                      pixels: prep.pixels,
                      height: prep.height,
                      width: prep.width,
                      unit: computeOptions.visionEncoder
                  )
            else {
                return nil
            }
            tokenIDs.append(255999) // <boi>
            tokenIDs.append(contentsOf: repeatElement(258884, count: encoded.tokenCount))
            tokenIDs.append(258882) // <eoi>
            softTokens.append(contentsOf: encoded.softTokens)
        }

        if !pcmSamples.isEmpty {
            let sr = Double(ArtifactProcessor.audioSampleRate)
            let startIdx = max(0, Int((windowStart * sr).rounded()))
            let endSec = min(windowEnd, windowStart + ArtifactProcessor.audioWindowSeconds)
            let endIdx = min(pcmSamples.count, Int((endSec * sr).rounded()))
            if endIdx > startIdx {
                let slice = Array(pcmSamples[startIdx..<endIdx])
                if let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: slice),
                   let encodedAudio = try await aneRuntime.encodeAudio(
                       melFeatures: mel.features,
                       frameCount: mel.frameCount,
                       unit: computeOptions.audioEncoder
                   ) {
                    tokenIDs.append(contentsOf: tokenizer.encode(text: " ", addSpecialTokens: false))
                    tokenIDs.append(256000) // <boa>
                    tokenIDs.append(contentsOf: repeatElement(258881, count: encodedAudio.tokenCount))
                    tokenIDs.append(258883) // <eoa>
                    softTokens.append(contentsOf: encodedAudio.softTokens)
                }
            }
        }

        tokenIDs.append(eosID)
        let textFusionUnit: TowerComputeUnit = (computeOptions.textBackbone == .mlxGPU)
            ? .cpuAndGPU
            : computeOptions.textBackbone
        return try await aneRuntime.encodeTextAndFuse(
            tokenIDs: tokenIDs,
            softTokens: softTokens.isEmpty ? nil : softTokens,
            unit: textFusionUnit,
            dimensions: dimensions
        )
    }

    private static func mapTask(_ task: EmbeddingTask) -> MLXVLM.EmbeddingTask {
        switch task {
        case .query, .retrievalQuery: return .searchQuery
        case .document, .retrievalDocument: return .document
        case .questionAnswering: return .questionAnswering
        case .factVerification: return .factChecking
        case .sts: return .sentenceSimilarity
        case .clustering: return .clustering
        case .classification: return .classification
        case .codeRetrieval: return .codeRetrieval
        case .none: return .none
        }
    }

    private static func ext(for kind: ArtifactKind) -> String {
        switch kind {
        case .image: return "jpg"
        case .video: return "mp4"
        case .audio: return "wav"
        default: return "dat"
        }
    }
}
#endif
