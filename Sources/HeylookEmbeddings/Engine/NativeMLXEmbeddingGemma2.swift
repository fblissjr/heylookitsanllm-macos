import Accelerate
import AVFoundation
import CoreImage
import Foundation
import HeylookCore

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
        public let defaultPromptName: String?
        public let poolingModeMeanTokens: Bool
        public let hasVision: Bool
        public let hasAudio: Bool
        public let hasVideo: Bool
        public let imageTokenID: Int
        public let videoTokenID: Int
        public let boiTokenID: Int
        public let eoiTokenID: Int
        public let audioTokenID: Int
        public let boaTokenID: Int
        public let eoaTokenID: Int

        public init(
            modelType: String,
            hiddenSize: Int,
            embeddingDim: Int,
            numHiddenLayers: Int,
            maxInputTokens: Int,
            videoFPS: Double,
            videoMaxFrames: Int,
            videoMaxSoftTokens: Int,
            imageMaxSoftTokens: Int,
            prompts: [String: String],
            defaultPromptName: String? = nil,
            poolingModeMeanTokens: Bool,
            hasVision: Bool = true,
            hasAudio: Bool = true,
            hasVideo: Bool = true,
            imageTokenID: Int = 258880,
            videoTokenID: Int = 258884,
            boiTokenID: Int = 255999,
            eoiTokenID: Int = 258882,
            audioTokenID: Int = 258881,
            boaTokenID: Int = 256000,
            eoaTokenID: Int = 258883
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.embeddingDim = embeddingDim
            self.numHiddenLayers = numHiddenLayers
            self.maxInputTokens = maxInputTokens
            self.videoFPS = videoFPS
            self.videoMaxFrames = videoMaxFrames
            self.videoMaxSoftTokens = videoMaxSoftTokens
            self.imageMaxSoftTokens = imageMaxSoftTokens
            self.prompts = prompts
            self.defaultPromptName = defaultPromptName
            self.poolingModeMeanTokens = poolingModeMeanTokens
            self.hasVision = hasVision
            self.hasAudio = hasAudio
            self.hasVideo = hasVideo
            self.imageTokenID = imageTokenID
            self.videoTokenID = videoTokenID
            self.boiTokenID = boiTokenID
            self.eoiTokenID = eoiTokenID
            self.audioTokenID = audioTokenID
            self.boaTokenID = boaTokenID
            self.eoaTokenID = eoaTokenID
        }

        public var maxWindowSeconds: Double {
            guard videoFPS > 0 else { return 32.0 }
            return Double(videoMaxFrames) / videoFPS
        }

        public var availablePromptNames: [String] {
            prompts.keys.sorted()
        }

        public var supportedModalities: [String] {
            var mods = ["text"]
            if hasVision { mods.append("image") }
            if hasVideo { mods.append("video") }
            if hasAudio { mods.append("audio") }
            return mods
        }

        /// Validates an explicit `prompt_name` against the checkpoint's `config_sentence_transformers.json` map.
        public func validatePromptName(_ name: String, itemIndex: Int = 0) throws -> String {
            guard let resolved = prompts[name] else {
                throw EmbeddingValidationError.unknownPromptName(
                    itemIndex: itemIndex,
                    promptName: name,
                    available: availablePromptNames
                )
            }
            return resolved
        }

        public func embeddingFacts(computePlacement: String? = nil) -> EngineEmbeddingFacts {
            EmbeddingModelFacts.fromCheckpoint(self, computePlacement: computePlacement)
        }

        /// Resolves the prompt prefix and effective `prompt_name` for an `EmbeddingInput`.
        /// - Explicit mode (`usesExplicitPromptName == true`): `promptName == nil` returns empty prefix `""`;
        ///   unknown `promptName` throws typed `400` error (`EmbeddingValidationError.unknownPromptName`);
        ///   known `promptName` returns exact `prompts[promptName]`.
        /// - Legacy `EmbeddingTask` mode: prefers matching entry in `prompts` (`config_sentence_transformers.json`)
        ///   before falling back to `EmbeddingTask` hardcoded defaults.
        public func resolvePrompt(
            for input: EmbeddingInput,
            itemIndex: Int = 0
        ) throws -> (prefix: String, promptName: String?) {
            try EmbeddingModelFacts.resolvePrompt(
                for: input,
                catalog: EmbeddingModelFacts.PromptCatalog(
                    prompts: prompts,
                    defaultPromptName: defaultPromptName,
                    maxSeqLength: maxInputTokens
                ),
                itemIndex: itemIndex
            )
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
            let visionConfig = configJSON["vision_config"] as? [String: Any]
            let audioConfig = configJSON["audio_config"] as? [String: Any]

            var prompts = [String: String]()
            var defaultPromptName: String?
            var maxSeqLength = 8192
            if let stData = try? Data(contentsOf: stURL),
               let stJSON = try? JSONSerialization.jsonObject(with: stData) as? [String: Any] {
                if let rawPrompts = stJSON["prompts"] as? [String: Any] {
                    for (k, v) in rawPrompts {
                        if let str = v as? String {
                            prompts[k] = str
                        }
                    }
                }
                if let defName = stJSON["default_prompt_name"] as? String {
                    defaultPromptName = defName
                }
                if let maxLen = (stJSON["max_seq_length"] as? NSNumber)?.intValue {
                    maxSeqLength = maxLen
                }
            }

            var meanTokens = true
            var wordEmbeddingDimension = 768
            if let poolData = try? Data(contentsOf: poolURL),
               let poolJSON = try? JSONSerialization.jsonObject(with: poolData) as? [String: Any] {
                meanTokens = (poolJSON["pooling_mode_mean_tokens"] as? Bool) ?? true
                wordEmbeddingDimension = (poolJSON["word_embedding_dimension"] as? NSNumber)?.intValue ?? 768
            }

            var fps = 1.0
            var numFrames = 32
            var videoSoftTokens = 140
            var imageSoftTokens = (visionConfig?["default_output_length"] as? NSNumber)?.intValue ?? 280
            var hasVideoProcessor = false
            if let procData = try? Data(contentsOf: procURL),
               let procJSON = try? JSONSerialization.jsonObject(with: procData) as? [String: Any] {
                if let vp = procJSON["video_processor"] as? [String: Any] {
                    hasVideoProcessor = true
                    if let parsedFPS = (vp["fps"] as? NSNumber)?.doubleValue, parsedFPS > 0 {
                        fps = parsedFPS
                    }
                    numFrames = (vp["max_frames"] as? NSNumber)?.intValue
                        ?? (vp["num_frames"] as? NSNumber)?.intValue
                        ?? 32
                    videoSoftTokens = (vp["max_soft_tokens"] as? NSNumber)?.intValue ?? 140
                }
                if let ip = procJSON["image_processor"] as? [String: Any],
                   let maxSoft = (ip["max_soft_tokens"] as? NSNumber)?.intValue {
                    imageSoftTokens = maxSoft
                }
            }

            let imageTokenID = (configJSON["image_token_id"] as? NSNumber)?.intValue ?? 258880
            let videoTokenID = (configJSON["video_token_id"] as? NSNumber)?.intValue ?? 258884
            let boiTokenID = (configJSON["boi_token_id"] as? NSNumber)?.intValue ?? 255999
            let eoiTokenID = (configJSON["eoi_token_id"] as? NSNumber)?.intValue ?? 258882
            let audioTokenID = (configJSON["audio_token_id"] as? NSNumber)?.intValue ?? 258881
            let boaTokenID = (configJSON["boa_token_id"] as? NSNumber)?.intValue ?? 256000
            let eoaTokenID = (configJSON["eoa_token_index"] as? NSNumber)?.intValue
                ?? (configJSON["eoa_token_id"] as? NSNumber)?.intValue
                ?? 258883

            let hasVision = visionConfig != nil
            let hasAudio = audioConfig != nil
            let hasVideo = hasVision && ((configJSON["video_token_id"] != nil) || hasVideoProcessor)

            return CheckpointDescriptor(
                modelType: (configJSON["model_type"] as? String) ?? "embedding_gemma2",
                hiddenSize: (textConfig["hidden_size"] as? NSNumber)?.intValue ?? 512,
                embeddingDim: (configJSON["embedding_dim"] as? NSNumber)?.intValue ?? wordEmbeddingDimension,
                numHiddenLayers: (textConfig["num_hidden_layers"] as? NSNumber)?.intValue ?? 24,
                maxInputTokens: maxSeqLength,
                videoFPS: fps,
                videoMaxFrames: numFrames,
                videoMaxSoftTokens: videoSoftTokens,
                imageMaxSoftTokens: imageSoftTokens,
                prompts: prompts,
                defaultPromptName: defaultPromptName,
                poolingModeMeanTokens: meanTokens,
                hasVision: hasVision,
                hasAudio: hasAudio,
                hasVideo: hasVideo,
                imageTokenID: imageTokenID,
                videoTokenID: videoTokenID,
                boiTokenID: boiTokenID,
                eoiTokenID: eoiTokenID,
                audioTokenID: audioTokenID,
                boaTokenID: boaTokenID,
                eoaTokenID: eoaTokenID
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

    @available(*, deprecated, message: "allowSyntheticFallback has been removed; real checkpoints/towers are always required.")
    public static func makeBestAvailableEngine(
        checkpointURL: URL = defaultCheckpointURL,
        computeOptions: ModelComputeOptions = .allMLXGPU,
        allowSyntheticFallback: Bool
    ) -> any EmbeddingEngine {
        _ = allowSyntheticFallback
        return makeBestAvailableEngine(checkpointURL: checkpointURL, computeOptions: computeOptions)
    }
}

/// Lightweight SentencePiece/BPE `tokenizer.json` piece encoder used by `HybridLocalEmbeddingEngine`
/// when `Tokenizers` is not compiled in, eliminating fake Unicode-scalar modulo token IDs.
public struct CheckpointPieceTokenizer: Sendable {
    public let bosTokenID: Int
    public let eosTokenID: Int
    public let unkTokenID: Int
    private let tokenToID: [String: Int]
    private let tokenScores: [String: Float]
    private let maxPieceChars: Int
    private let usesSentencePieceSpace: Bool

    public static func load(from checkpointDirectory: URL) -> CheckpointPieceTokenizer? {
        let fm = FileManager.default
        let candidateURLs = [
            checkpointDirectory.appendingPathComponent("tokenizer.json"),
            checkpointDirectory.appendingPathComponent("tokenizer/tokenizer.json"),
        ]
        guard let tokURL = candidateURLs.first(where: { fm.fileExists(atPath: $0.path) }),
              let data = try? Data(contentsOf: tokURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any]
        else {
            return nil
        }

        var map: [String: Int] = [:]
        var scores: [String: Float] = [:]
        var maxLen = 1

        if let vocabDict = model["vocab"] as? [String: Any] {
            for (piece, rawID) in vocabDict {
                if let id = (rawID as? NSNumber)?.intValue {
                    map[piece] = id
                    if piece.count > maxLen { maxLen = piece.count }
                }
            }
        } else if let vocabList = model["vocab"] as? [[Any]] {
            // Unigram SentencePiece format: `[["<pad>", 0.0], ["<eos>", 0.0], ["▁the", -3.12], ...]`
            for (idx, entry) in vocabList.enumerated() where !entry.isEmpty {
                if let piece = entry[0] as? String {
                    map[piece] = idx
                    if entry.count > 1, let sc = (entry[1] as? NSNumber)?.floatValue {
                        scores[piece] = sc
                    }
                    if piece.count > maxLen { maxLen = piece.count }
                }
            }
        }

        if let added = root["added_tokens"] as? [[String: Any]] {
            for item in added {
                if let content = item["content"] as? String,
                   let id = (item["id"] as? NSNumber)?.intValue {
                    map[content] = id
                    if content.count > maxLen { maxLen = content.count }
                }
            }
        }

        guard !map.isEmpty else { return nil }
        let bos = map["<bos>"] ?? map["<s>"] ?? 2
        let eos = map["<eos>"] ?? map["</s>"] ?? 1
        let unk = map["<unk>"] ?? 3
        let usesSP = map.keys.contains { $0.contains("▁") }

        return CheckpointPieceTokenizer(
            bosTokenID: bos,
            eosTokenID: eos,
            unkTokenID: unk,
            tokenToID: map,
            tokenScores: scores,
            maxPieceChars: min(maxLen, 64),
            usesSentencePieceSpace: usesSP
        )
    }

    public func encode(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        let normalized: String
        if usesSentencePieceSpace {
            normalized = text.replacingOccurrences(of: " ", with: "▁")
        } else {
            normalized = text
        }
        let chars = Array(normalized)
        let n = chars.count

        if !tokenScores.isEmpty {
            // Viterbi dynamic programming segmentation over Unigram log-probabilities
            var bestScore = [Float](repeating: -.greatestFiniteMagnitude, count: n + 1)
            var bestPrev = [Int](repeating: 0, count: n + 1)
            var bestID = [Int](repeating: unkTokenID, count: n + 1)
            bestScore[0] = 0

            for i in 0..<n where bestScore[i] > -.greatestFiniteMagnitude / 2 {
                let upper = min(n, i + maxPieceChars)
                var matchedAny = false
                for j in (i + 1)...upper {
                    let sub = String(chars[i..<j])
                    if let tokID = tokenToID[sub] {
                        matchedAny = true
                        let pieceScore = tokenScores[sub] ?? -10.0
                        let cand = bestScore[i] + pieceScore
                        if cand > bestScore[j] {
                            bestScore[j] = cand
                            bestPrev[j] = i
                            bestID[j] = tokID
                        }
                    }
                }
                if !matchedAny {
                    // Byte fallback `<0xXX>` if present in Gemma vocab, else `<unk>`
                    let charStr = String(chars[i])
                    let utf8Bytes = Array(charStr.utf8)
                    if utf8Bytes.count == 1,
                       let byteID = tokenToID[String(format: "<0x%02X>", utf8Bytes[0])] {
                        let cand = bestScore[i] - 20.0
                        if cand > bestScore[i + 1] {
                            bestScore[i + 1] = cand
                            bestPrev[i + 1] = i
                            bestID[i + 1] = byteID
                        }
                    } else {
                        let cand = bestScore[i] - 25.0
                        if cand > bestScore[i + 1] {
                            bestScore[i + 1] = cand
                            bestPrev[i + 1] = i
                            bestID[i + 1] = unkTokenID
                        }
                    }
                }
            }

            var reversedIDs: [Int] = []
            var cur = n
            while cur > 0 {
                reversedIDs.append(bestID[cur])
                let prev = bestPrev[cur]
                cur = prev < cur ? prev : cur - 1
            }
            return reversedIDs.reversed()
        }

        // Greedy longest-match fallback for BPE vocab dicts
        var ids: [Int] = []
        var idx = 0
        while idx < n {
            var matchedLen = 0
            var matchedID: Int?
            let upper = min(n, idx + maxPieceChars)
            for end in stride(from: upper, through: idx + 1, by: -1) {
                let sub = String(chars[idx..<end])
                if let id = tokenToID[sub] {
                    matchedID = id
                    matchedLen = end - idx
                    break
                }
            }
            if let matchedID {
                ids.append(matchedID)
                idx += matchedLen
            } else {
                let utf8Bytes = Array(String(chars[idx]).utf8)
                var appendedByte = false
                for b in utf8Bytes {
                    if let byteID = tokenToID[String(format: "<0x%02X>", b)] {
                        ids.append(byteID)
                        appendedByte = true
                    }
                }
                if !appendedByte {
                    ids.append(unkTokenID)
                }
                idx += 1
            }
        }
        return ids
    }
}

/// Hybrid local `EmbeddingEngine` executing compiled CoreML `.mlmodelc` ANE towers (`ANECoreMLRuntime`)
/// and `ArtifactProcessor` + `WindowPlanner` temporal windowing.
/// Enforces Repo Rule #3 (*Truthful Runtime*): always requires compiled `.mlmodelc` towers and
/// `tokenizer.json`, throwing `EmbeddingEngineError.runtimeUnavailable` when absent.
public actor HybridLocalEmbeddingEngine: EmbeddingEngine {
    public nonisolated let modelID: String
    public nonisolated let checkpointURL: URL
    public nonisolated let computeOptions: ModelComputeOptions
    public nonisolated let hasCompiledANETowers: Bool
    public nonisolated let hasCompiledANEVision: Bool
    public nonisolated let hasCompiledANEAudio: Bool
    @available(*, deprecated, message: "Synthetic fallback has been removed; HybridLocalEmbeddingEngine always throws runtimeUnavailable when towers are missing.")
    public nonisolated var allowSyntheticFallback: Bool { false }
    public nonisolated let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public nonisolated let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions

    public nonisolated let descriptor: NativeMLXEmbeddingGemma2.CheckpointDescriptor
    private let aneRuntime: ANECoreMLRuntime
    private let pieceTokenizer: CheckpointPieceTokenizer?

    public nonisolated var availablePromptNames: [String] { descriptor.availablePromptNames }
    public nonisolated var defaultPromptName: String? { descriptor.defaultPromptName }

    public init(
        checkpointURL: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL,
        computeOptions: ModelComputeOptions = .allMLXGPU
    ) {
        self.checkpointURL = checkpointURL
        self.computeOptions = computeOptions
        let anePresent = ANETowerInspector.locateBundle(for: .textBackbone, in: checkpointURL) != nil
        self.hasCompiledANETowers = anePresent
        self.hasCompiledANEVision = ANETowerInspector.locateBundle(for: .visionEncoder, in: checkpointURL) != nil
        self.hasCompiledANEAudio = ANETowerInspector.locateBundle(for: .audioEncoder, in: checkpointURL) != nil
        self.modelID = checkpointURL.lastPathComponent
        self.aneRuntime = ANECoreMLRuntime(checkpointDirectory: checkpointURL)
        self.pieceTokenizer = CheckpointPieceTokenizer.load(from: checkpointURL)
        self.descriptor = (try? NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: checkpointURL))
            ?? Self.defaultDescriptor
    }

    @available(*, deprecated, message: "allowSyntheticFallback has been removed; HybridLocalEmbeddingEngine always requires real .mlmodelc towers.")
    public init(
        checkpointURL: URL = NativeMLXEmbeddingGemma2.defaultCheckpointURL,
        computeOptions: ModelComputeOptions = .allMLXGPU,
        allowSyntheticFallback: Bool
    ) {
        _ = allowSyntheticFallback
        self.init(checkpointURL: checkpointURL, computeOptions: computeOptions)
    }

    private static let defaultDescriptor = NativeMLXEmbeddingGemma2.CheckpointDescriptor(
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
        defaultPromptName: nil,
        poolingModeMeanTokens: true
    )

    public init(
        checkpointDirectory: URL,
        computeOptions: ModelComputeOptions = .allMLXGPU
    ) {
        self.init(
            checkpointURL: checkpointDirectory,
            computeOptions: computeOptions
        )
    }

    @available(*, deprecated, message: "allowSyntheticFallback has been removed; HybridLocalEmbeddingEngine always requires real .mlmodelc towers.")
    public init(
        checkpointDirectory: URL,
        computeOptions: ModelComputeOptions = .allMLXGPU,
        allowSyntheticFallback: Bool
    ) {
        _ = allowSyntheticFallback
        self.init(
            checkpointURL: checkpointDirectory,
            computeOptions: computeOptions
        )
    }

    public func unload() async {
        await aneRuntime.unload()
    }

    public func embedBatch(
        _ inputs: [EmbeddingInput],
        dimensions: Int? = nil
    ) async throws -> [EmbeddedOutput] {
        guard !inputs.isEmpty else { return [] }
        let targetDim = dimensions ?? nativeDimensions
        guard supportedDimensions.contains(targetDim) else {
            throw EmbeddingValidationError.unsupportedDimensions(
                requested: targetDim,
                supported: supportedDimensions
            )
        }

        var validatedPlans: [(
            promptPrefix: String,
            promptName: String?,
            windowOptions: VideoWindowOptions?
        )] = []
        validatedPlans.reserveCapacity(inputs.count)
        for (idx, item) in inputs.enumerated() {
            let winOpts = try item.resolvedVideoWindows(itemIndex: idx)
            if let winOpts {
                guard descriptor.hasVideo else {
                    throw EmbeddingValidationError.unsupportedModality(
                        itemIndex: idx,
                        modality: "video",
                        supported: descriptor.supportedModalities
                    )
                }
                if let secs = winOpts.seconds, secs > descriptor.maxWindowSeconds {
                    throw EmbeddingValidationError.videoWindowsExceedMaxSeconds(
                        itemIndex: idx,
                        requested: secs,
                        maxSeconds: descriptor.maxWindowSeconds
                    )
                }
            }
            for part in item.parts {
                if case let .media(kind, _, _, _, _) = part,
                   !descriptor.supportedModalities.contains(kind.rawValue) {
                    throw EmbeddingValidationError.unsupportedModality(
                        itemIndex: idx,
                        modality: kind.rawValue,
                        supported: descriptor.supportedModalities
                    )
                }
            }
            let promptRes = try descriptor.resolvePrompt(for: item, itemIndex: idx)
            validatedPlans.append((promptRes.prefix, promptRes.promptName, winOpts))
        }

        guard hasCompiledANETowers else {
            throw EmbeddingEngineError.runtimeUnavailable(
                "Compiled CoreML ANE towers (.mlmodelc) not found at '\(checkpointURL.path)' and native MLXVLM is not linked."
            )
        }

        var outputs = [EmbeddedOutput]()
        outputs.reserveCapacity(inputs.count)

        for (idx, item) in inputs.enumerated() {
            try Task.checkCancellation()
            let plan = validatedPlans[idx]
            var windowedVideoURL: URL?
            var hasCompanionAudio = false
            var tempVideoURL: URL?
            defer {
                if let tempVideoURL {
                    try? FileManager.default.removeItem(at: tempVideoURL)
                }
            }

            for part in item.parts {
                if case let .media(kind, _, data, url, windowSampling) = part {
                    if kind == .audio {
                        hasCompanionAudio = true
                    }
                    if kind == .video, (plan.windowOptions != nil || windowSampling != .off), windowedVideoURL == nil {
                        if let url {
                            windowedVideoURL = url
                        } else if let data {
                            let tmp = FileManager.default.temporaryDirectory
                                .appendingPathComponent("heylook-emb-vid-\(UUID().uuidString).mp4")
                            try data.write(to: tmp)
                            tempVideoURL = tmp
                            windowedVideoURL = tmp
                        }
                    }
                }
            }

            guard let aneRes = try await tryEmbedItemOnANE(
                item: item,
                itemIndex: idx,
                promptPrefix: plan.promptPrefix,
                targetDim: targetDim
            ) else {
                throw EmbeddingEngineError.runtimeUnavailable(
                    "ANE tower execution unavailable for checkpoint '\(checkpointURL.path)' (verify tokenizer.json and .mlmodelc bundles exist)."
                )
            }
            let primary = aneRes.vector
            var totalTokens = aneRes.tokenCount

            var windows = [VideoWindowEmbeddingResult]()
            if let videoURL = windowedVideoURL, let winOpts = plan.windowOptions {
                let winSeconds = winOpts.seconds ?? descriptor.maxWindowSeconds
                if let metadata = try? await ArtifactProcessor.inspectVideoMetadata(for: videoURL) {
                    let ranges = WindowPlanner.consecutiveRanges(
                        duration: metadata.duration,
                        windowSeconds: winSeconds
                    )
                    let pcmSamples: [Float] = hasCompanionAudio
                        ? ((try? await ArtifactProcessor.decodeMonoPCM(for: videoURL)) ?? [])
                        : []
                    for (lo, hi) in ranges {
                        try Task.checkCancellation()
                        let decodedFrames = (try? await ArtifactProcessor.decodeVideoWindowFrames(
                            for: videoURL,
                            metadata: metadata,
                            lo: lo,
                            hi: hi,
                            sampling: winOpts.sampling,
                            fps: descriptor.videoFPS,
                            maxFrames: descriptor.videoMaxFrames
                        )) ?? []
                        guard !decodedFrames.isEmpty else { continue }
                        let roundedStart = (lo * 1000).rounded() / 1000
                        let roundedEnd = (hi * 1000).rounded() / 1000
                        let frameTimes = decodedFrames.map { ($0.time * 1000).rounded() / 1000 }
                        guard let aneWin = try await tryEmbedVideoFramesOnANE(
                            frames: decodedFrames,
                            pcmSamples: pcmSamples,
                            windowStart: lo,
                            windowEnd: hi,
                            promptPrefix: plan.promptPrefix,
                            targetDim: targetDim
                        ) else {
                            throw EmbeddingEngineError.runtimeUnavailable(
                                "Failed to encode video window (\(roundedStart)s-\(roundedEnd)s) on ANE."
                            )
                        }
                        totalTokens += aneWin.tokenCount
                        windows.append(
                            VideoWindowEmbeddingResult(
                                start: roundedStart,
                                end: roundedEnd,
                                frameTimes: frameTimes,
                                vector: aneWin.vector
                            )
                        )
                    }
                }
            }

            outputs.append(
                EmbeddedOutput(
                    vector: primary,
                    windows: windows,
                    inputTokens: totalTokens > 0 ? totalTokens : nil,
                    promptName: plan.promptName
                )
            )
        }
        return outputs
    }

    private func tryEmbedVideoFramesOnANE(
        frames: [(time: Double, image: CGImage)],
        pcmSamples: [Float] = [],
        windowStart: Double = 0,
        windowEnd: Double = 0,
        promptPrefix: String = "",
        targetDim: Int
    ) async throws -> (vector: [Float], tokenCount: Int)? {
        guard let pieceTokenizer else { return nil }
        let visionUnit: TowerComputeUnit = computeOptions.visionEncoder == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.visionEncoder
        let audioUnit: TowerComputeUnit = computeOptions.audioEncoder == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.audioEncoder
        let textUnit: TowerComputeUnit = computeOptions.textBackbone == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.textBackbone

        var tokenIDs: [Int] = [pieceTokenizer.bosTokenID]
        if !promptPrefix.isEmpty {
            tokenIDs.append(contentsOf: pieceTokenizer.encode(promptPrefix))
        }
        var softTokens: [Float] = []

        for frame in frames {
            guard let prep = ANECoreMLRuntime.prepareImagePixels(from: frame.image, forVideoBudget: true),
                  let enc = try await aneRuntime.encodeVision(
                      pixels: prep.pixels,
                      height: prep.height,
                      width: prep.width,
                      unit: visionUnit
                  )
            else { return nil }
            tokenIDs.append(descriptor.boiTokenID)
            tokenIDs.append(contentsOf: repeatElement(descriptor.videoTokenID, count: enc.tokenCount))
            tokenIDs.append(descriptor.eoiTokenID)
            softTokens.append(contentsOf: enc.softTokens)
        }

        if !pcmSamples.isEmpty && windowEnd > windowStart {
            let sr = Double(ArtifactProcessor.audioSampleRate)
            let startIdx = max(0, Int((windowStart * sr).rounded()))
            let endSec = min(windowEnd, windowStart + ArtifactProcessor.audioWindowSeconds)
            let endIdx = min(pcmSamples.count, Int((endSec * sr).rounded()))
            if endIdx > startIdx {
                let slice = Array(pcmSamples[startIdx..<endIdx])
                if let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: slice),
                   let encAudio = try await aneRuntime.encodeAudio(
                       melFeatures: mel.features,
                       frameCount: mel.frameCount,
                       unit: audioUnit
                   ) {
                    tokenIDs.append(contentsOf: pieceTokenizer.encode(" "))
                    tokenIDs.append(descriptor.boaTokenID)
                    tokenIDs.append(contentsOf: repeatElement(descriptor.audioTokenID, count: encAudio.tokenCount))
                    tokenIDs.append(descriptor.eoaTokenID)
                    softTokens.append(contentsOf: encAudio.softTokens)
                }
            }
        }

        tokenIDs.append(pieceTokenizer.eosTokenID)
        guard let vec = try await aneRuntime.encodeTextAndFuse(
            tokenIDs: tokenIDs,
            softTokens: softTokens.isEmpty ? nil : softTokens,
            softTokenIDs: [descriptor.imageTokenID, descriptor.videoTokenID, descriptor.audioTokenID],
            unit: textUnit,
            dimensions: targetDim
        ) else {
            return nil
        }
        return (vec, tokenIDs.count)
    }

    private func tryEmbedItemOnANE(
        item: EmbeddingInput,
        itemIndex: Int,
        promptPrefix: String,
        targetDim: Int
    ) async throws -> (vector: [Float], tokenCount: Int)? {
        guard hasCompiledANETowers else { return nil }
        guard let pieceTokenizer else {
            throw EmbeddingEngineError.runtimeUnavailable(
                "Missing tokenizer.json in checkpoint directory '\(checkpointURL.path)' required for ANE text backbone."
            )
        }

        let visionUnit: TowerComputeUnit = computeOptions.visionEncoder == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.visionEncoder
        let audioUnit: TowerComputeUnit = computeOptions.audioEncoder == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.audioEncoder
        let textUnit: TowerComputeUnit = computeOptions.textBackbone == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.textBackbone

        var tokenIDs: [Int] = [pieceTokenizer.bosTokenID]
        var softTokens: [Float] = []
        var didPrependPrompt = false
        var tempURLs: [URL] = []
        defer {
            for tmp in tempURLs {
                try? FileManager.default.removeItem(at: tmp)
            }
        }

        if !promptPrefix.isEmpty && !item.hasNonWhitespaceText {
            didPrependPrompt = true
            tokenIDs.append(contentsOf: pieceTokenizer.encode(promptPrefix))
        }

        for part in item.parts {
            switch part {
            case .text(let s):
                let formatted: String
                if !didPrependPrompt && !promptPrefix.isEmpty {
                    didPrependPrompt = true
                    formatted = (s.hasPrefix("task:") || s.hasPrefix("title:"))
                        ? s
                        : (promptPrefix + s)
                } else {
                    formatted = s
                }
                if !formatted.isEmpty {
                    let encodedPieces = pieceTokenizer.encode(formatted)
                    tokenIDs.append(contentsOf: encodedPieces)
                }

            case let .media(kind, _, data, url, _):
                let mediaURL: URL
                if let url {
                    mediaURL = url
                } else if let data {
                    let ext: String
                    switch kind {
                    case .image: ext = "jpg"
                    case .audio: ext = "wav"
                    case .video: ext = "mp4"
                    default: ext = "bin"
                    }
                    let tmp = FileManager.default.temporaryDirectory
                        .appendingPathComponent("heylook-ane-\(UUID().uuidString).\(ext)")
                    try data.write(to: tmp)
                    tempURLs.append(tmp)
                    mediaURL = tmp
                } else {
                    continue
                }

                if kind == .image {
                    guard let prep = ANECoreMLRuntime.prepareImagePixels(from: mediaURL, forVideoBudget: false),
                          let enc = try await aneRuntime.encodeVision(
                              pixels: prep.pixels,
                              height: prep.height,
                              width: prep.width,
                              unit: visionUnit
                          )
                    else { return nil }
                    tokenIDs.append(descriptor.boiTokenID)
                    tokenIDs.append(contentsOf: repeatElement(descriptor.imageTokenID, count: enc.tokenCount))
                    tokenIDs.append(descriptor.eoiTokenID)
                    softTokens.append(contentsOf: enc.softTokens)
                } else if kind == .audio {
                    guard let pcm = try? await ArtifactProcessor.decodeMonoPCM(for: mediaURL),
                          !pcm.isEmpty,
                          let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: pcm),
                          let enc = try await aneRuntime.encodeAudio(
                              melFeatures: mel.features,
                              frameCount: mel.frameCount,
                              unit: audioUnit
                          )
                    else { return nil }
                    tokenIDs.append(descriptor.boaTokenID)
                    tokenIDs.append(contentsOf: repeatElement(descriptor.audioTokenID, count: enc.tokenCount))
                    tokenIDs.append(descriptor.eoaTokenID)
                    softTokens.append(contentsOf: enc.softTokens)
                } else if kind == .video {
                    guard let metadata = try await ArtifactProcessor.inspectVideoMetadata(for: mediaURL) else {
                        return nil
                    }
                    let sampleTimes = WindowPlanner.uniformSampleTimes(
                        duration: metadata.duration,
                        nativeFPS: metadata.nativeFPS,
                        fps: descriptor.videoFPS,
                        maxFrames: descriptor.videoMaxFrames
                    )
                    let decoded = try await ArtifactProcessor.decodeVideoFrames(
                        for: mediaURL,
                        at: sampleTimes,
                        startOffsetSeconds: metadata.startSeconds
                    )
                    guard !decoded.isEmpty else { return nil }
                    for frame in decoded {
                        guard let prep = ANECoreMLRuntime.prepareImagePixels(from: frame.image, forVideoBudget: true),
                              let enc = try await aneRuntime.encodeVision(
                                  pixels: prep.pixels,
                                  height: prep.height,
                                  width: prep.width,
                                  unit: visionUnit
                              )
                        else { return nil }
                        tokenIDs.append(descriptor.boiTokenID)
                        tokenIDs.append(contentsOf: repeatElement(descriptor.videoTokenID, count: enc.tokenCount))
                        tokenIDs.append(descriptor.eoiTokenID)
                        softTokens.append(contentsOf: enc.softTokens)
                    }
                }
            }
        }
        tokenIDs.append(pieceTokenizer.eosTokenID)
        if item.usesExplicitPromptName && tokenIDs.count > descriptor.maxInputTokens {
            throw EmbeddingValidationError.invalidInput(
                statusCode: 400,
                detail: "input[\(itemIndex)]: expanded input has \(tokenIDs.count) tokens; limit is \(descriptor.maxInputTokens)"
            )
        }
        guard let vec = try await aneRuntime.encodeTextAndFuse(
            tokenIDs: tokenIDs,
            softTokens: softTokens.isEmpty ? nil : softTokens,
            softTokenIDs: [descriptor.imageTokenID, descriptor.videoTokenID, descriptor.audioTokenID],
            unit: textUnit,
            dimensions: targetDim
        ) else {
            return nil
        }
        return (vec, tokenIDs.count)
    }
}

#if canImport(MLXVLM)
/// In-process `EmbeddingEngine` backed directly by `mlx-swift-lm`'s `MLXVLM.EmbeddingGemma2Embedding`
/// (pure GPU & 1,024-token padded batching), `MLXVLM.EmbeddingGemma2` (`hybrid_ane` GPU text/fusion +
/// ANE vision/audio towers via soft tokens), and `ANECoreMLRuntime`.
public actor MLXSwiftLMEmbeddingGemma2Engine: EmbeddingEngine {
    public nonisolated let modelID: String
    public nonisolated let checkpointURL: URL
    public nonisolated let computeOptions: ModelComputeOptions
    public nonisolated let nativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public nonisolated let supportedDimensions: [Int] = Matryoshka.gemma2Dimensions

    public nonisolated let descriptor: NativeMLXEmbeddingGemma2.CheckpointDescriptor
    public nonisolated var availablePromptNames: [String] { descriptor.availablePromptNames }
    public nonisolated var defaultPromptName: String? { descriptor.defaultPromptName }
    public nonisolated var supportedModalities: [String] { descriptor.supportedModalities }

    private nonisolated let hasANEVisionBundle: Bool
    private nonisolated let hasANEAudioBundle: Bool
    private nonisolated let hasANETextBundle: Bool

    private let customTokenizerLoader: (any TokenizerLoader)?
    private let aneRuntime: ANECoreMLRuntime
    private var cachedTokenizer: (any MLXLMCommon.Tokenizer)?
    private var cachedEmbedder: MLXVLM.EmbeddingGemma2Embedding?
    private var cachedHybridModel: MLXVLM.EmbeddingGemma2?

    private static let textBatchTokenLimit: Int = 1_024

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
        self.hasANEVisionBundle = ANETowerInspector.locateBundle(for: .visionEncoder, in: checkpointURL) != nil
        self.hasANEAudioBundle = ANETowerInspector.locateBundle(for: .audioEncoder, in: checkpointURL) != nil
        self.hasANETextBundle = ANETowerInspector.locateBundle(for: .textBackbone, in: checkpointURL) != nil
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
        self.hasANEVisionBundle = ANETowerInspector.locateBundle(for: .visionEncoder, in: modelDirectory) != nil
        self.hasANEAudioBundle = ANETowerInspector.locateBundle(for: .audioEncoder, in: modelDirectory) != nil
        self.hasANETextBundle = ANETowerInspector.locateBundle(for: .textBackbone, in: modelDirectory) != nil
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

    /// Loads `EmbeddingGemma2` directly for `hybrid_ane` mode (`textBackbone == .mlxGPU`),
    /// excluding media components executed on CoreML/ANE while leaving non-ANE components `.onFirstUse`.
    private func resolvedHybridModel(
        excludingVision: Bool,
        excludingAudio: Bool
    ) async throws -> MLXVLM.EmbeddingGemma2 {
        if let cachedHybridModel {
            return cachedHybridModel
        }
        let configData = try Data(contentsOf: checkpointURL.appendingPathComponent("config.json"))
        let config = try JSONDecoder.json5().decode(
            MLXVLM.EmbeddingGemma2Configuration.self,
            from: configData
        )
        let model = MLXVLM.EmbeddingGemma2(config)
        let base = try JSONDecoder.json5().decode(MLXLMCommon.BaseConfiguration.self, from: configData)
        var excluded: Set<ModelComponent> = []
        if excludingVision { excluded.insert(.vision) }
        if excludingAudio { excluded.insert(.audio) }
        try await MLXLMCommon.loadWeights(
            modelDirectory: checkpointURL,
            model: model,
            perLayerQuantization: base.perLayerQuantization,
            excludedComponents: excluded,
            componentLoading: .onFirstUse
        )
        try Task.checkCancellation()
        self.cachedHybridModel = model
        return model
    }

    public func unload() async {
        let hadMLX = (cachedEmbedder != nil) || (cachedHybridModel != nil)
        cachedEmbedder = nil
        cachedHybridModel = nil
        cachedTokenizer = nil
        await aneRuntime.unload()
        if hadMLX {
            MLX.Memory.clearCache()
        }
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
        defaultPromptName: nil,
        poolingModeMeanTokens: true
    )

    private enum ResolvedPart: Sendable {
        case text(String)
        case image(URL)
        case video(URL)
        case audio(URL)
    }

    private struct PreparedItem {
        let index: Int
        let item: EmbeddingInput
        let promptPrefix: String
        let promptName: String?
        let resolvedParts: [ResolvedPart]
        let isTextOnly: Bool
        let combinedText: String
        let windowedVideoURL: URL?
        let windowOptions: VideoWindowOptions?
        let hasCompanionAudio: Bool
    }

    nonisolated private static func buildVLMInput(
        parts: [ResolvedPart],
        promptPrefix: String
    ) -> sending MLXVLM.EmbeddingInput {
        var vlmParts: [MLXVLM.EmbeddingInput.Part] = []
        vlmParts.reserveCapacity(parts.count + 1)
        var didPrepend = false
        for part in parts {
            switch part {
            case .text(let s):
                if !didPrepend && !promptPrefix.isEmpty {
                    didPrepend = true
                    let text = (s.hasPrefix("task:") || s.hasPrefix("title:")) ? s : (promptPrefix + s)
                    vlmParts.append(.text(text))
                } else {
                    vlmParts.append(.text(s))
                }
            case .image(let url):
                if !didPrepend && !promptPrefix.isEmpty {
                    didPrepend = true
                    vlmParts.append(.text(promptPrefix))
                }
                vlmParts.append(.image(.url(url)))
            case .video(let url):
                if !didPrepend && !promptPrefix.isEmpty {
                    didPrepend = true
                    vlmParts.append(.text(promptPrefix))
                }
                vlmParts.append(.video(.url(url)))
            case .audio(let url):
                if !didPrepend && !promptPrefix.isEmpty {
                    didPrepend = true
                    vlmParts.append(.text(promptPrefix))
                }
                vlmParts.append(.audio(.url(url)))
            }
        }
        return MLXVLM.EmbeddingInput(vlmParts, title: nil)
    }

    nonisolated private static func buildWindowVLMInput(
        frames: [(time: Double, image: CGImage)],
        audioWAVURL: URL?,
        promptPrefix: String
    ) -> sending MLXVLM.EmbeddingInput {
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)
        let ciOptions: [CIImageOption: Any] = sRGB.map { [.colorSpace: $0] } ?? [:]
        let videoFrames = frames.map { frame in
            UserInput.VideoFrame(
                image: .ciImage(CIImage(cgImage: frame.image, options: ciOptions)),
                timeStamp: CMTime(seconds: frame.time, preferredTimescale: 90_000)
            )
        }
        var winParts: [MLXVLM.EmbeddingInput.Part] = []
        if !promptPrefix.isEmpty {
            winParts.append(.text(promptPrefix))
        }
        winParts.append(.video(.frames(videoFrames)))
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
        guard !inputs.isEmpty else { return [] }
        let targetDim = dimensions ?? nativeDimensions
        guard supportedDimensions.contains(targetDim) else {
            throw EmbeddingValidationError.unsupportedDimensions(
                requested: targetDim,
                supported: supportedDimensions
            )
        }

        var tempURLs: [URL] = []
        defer {
            for url in tempURLs {
                try? FileManager.default.removeItem(at: url)
            }
        }

        // Validate and prepare every item upfront before executing compute.
        var preparedItems: [PreparedItem] = []
        preparedItems.reserveCapacity(inputs.count)
        for (idx, item) in inputs.enumerated() {
            guard !item.parts.isEmpty else {
                throw EmbeddingValidationError.invalidInput(
                    statusCode: 422,
                    detail: "input[\(idx)]: embedding item content cannot be empty"
                )
            }
            let winOpts = try item.resolvedVideoWindows(itemIndex: idx)
            if let winOpts {
                guard descriptor.hasVideo else {
                    throw EmbeddingValidationError.unsupportedModality(
                        itemIndex: idx,
                        modality: "video",
                        supported: descriptor.supportedModalities
                    )
                }
                if let secs = winOpts.seconds, secs > descriptor.maxWindowSeconds {
                    throw EmbeddingValidationError.videoWindowsExceedMaxSeconds(
                        itemIndex: idx,
                        requested: secs,
                        maxSeconds: descriptor.maxWindowSeconds
                    )
                }
            }
            let promptRes = try descriptor.resolvePrompt(for: item, itemIndex: idx)

            var resolvedParts: [ResolvedPart] = []
            var windowedVideoURL: URL?
            var hasCompanionAudio = false
            var hasMedia = false
            var combinedText = ""

            for part in item.parts {
                switch part {
                case .text(let s):
                    resolvedParts.append(.text(s))
                    combinedText += s
                case let .media(kind, _, data, url, windowSampling):
                    guard descriptor.supportedModalities.contains(kind.rawValue) else {
                        throw EmbeddingValidationError.unsupportedModality(
                            itemIndex: idx,
                            modality: kind.rawValue,
                            supported: descriptor.supportedModalities
                        )
                    }
                    hasMedia = true
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
                        if (winOpts != nil || windowSampling != .off) && windowedVideoURL == nil {
                            windowedVideoURL = resolvedURL
                        }
                    case .audio:
                        resolvedParts.append(.audio(resolvedURL))
                        hasCompanionAudio = true
                    default:
                        break
                    }
                }
            }

            let isTextOnly = !hasMedia
            let formattedText: String
            if isTextOnly && !promptRes.prefix.isEmpty {
                formattedText = (combinedText.hasPrefix("task:") || combinedText.hasPrefix("title:"))
                    ? combinedText
                    : (promptRes.prefix + combinedText)
            } else {
                formattedText = combinedText
            }
            if isTextOnly && formattedText.allSatisfy(\.isWhitespace) {
                throw EmbeddingValidationError.invalidInput(
                    statusCode: 400,
                    detail: "input[\(idx)]: embedding input cannot be empty"
                )
            }

            preparedItems.append(
                PreparedItem(
                    index: idx,
                    item: item,
                    promptPrefix: promptRes.prefix,
                    promptName: promptRes.promptName,
                    resolvedParts: resolvedParts,
                    isTextOnly: isTextOnly,
                    combinedText: formattedText,
                    windowedVideoURL: windowedVideoURL,
                    windowOptions: winOpts,
                    hasCompanionAudio: hasCompanionAudio
                )
            )
        }

        let tokenizer = try await resolvedTokenizer()
        let bosID = tokenizer.bosToken.flatMap({ tokenizer.convertTokenToId($0) }) ?? 2
        let eosID = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("<eos>") ?? 1

        let wantsFullANE = computeOptions.textBackbone != .mlxGPU
        let wantsANEVision = computeOptions.visionEncoder != .mlxGPU && hasANEVisionBundle
        let wantsANEAudio = computeOptions.audioEncoder != .mlxGPU && hasANEAudioBundle
        let useHybridANE = !wantsFullANE && (wantsANEVision || wantsANEAudio)

        var outputs: [EmbeddedOutput?] = Array(repeating: nil, count: inputs.count)

        if wantsFullANE {
            guard hasANETextBundle else {
                throw EmbeddingEngineError.runtimeUnavailable(
                    "Compiled CoreML TextANE (.mlmodelc) required for textBackbone=\(computeOptions.textBackbone.rawValue) was not found at '\(checkpointURL.path)'."
                )
            }
            for prep in preparedItems {
                try Task.checkCancellation()
                let primaryRes = try await embedItemFullANE(
                    prep: prep,
                    tokenizer: tokenizer,
                    bosID: bosID,
                    eosID: eosID,
                    dimensions: targetDim
                )
                var windowVectors: [VideoWindowEmbeddingResult] = []
                var windowTokens = 0
                if let videoURL = prep.windowedVideoURL, let winOpts = prep.windowOptions {
                    let res = try await embedVideoWindows(
                        videoURL: videoURL,
                        windowOptions: winOpts,
                        includeAudio: prep.hasCompanionAudio,
                        promptPrefix: prep.promptPrefix,
                        dimensions: dimensions,
                        mode: .fullANE(tokenizer: tokenizer, bosID: bosID, eosID: eosID)
                    )
                    windowVectors = res.windows
                    windowTokens = res.tokens
                }
                outputs[prep.index] = EmbeddedOutput(
                    vector: primaryRes.vector,
                    windows: windowVectors,
                    inputTokens: primaryRes.tokens + windowTokens,
                    promptName: prep.promptName
                )
            }
            return outputs.compactMap { $0 }
        }

        if useHybridANE {
            let hybridModel = try await resolvedHybridModel(
                excludingVision: wantsANEVision,
                excludingAudio: wantsANEAudio
            )

            let textItems = preparedItems.filter(\.isTextOnly)
            if !textItems.isEmpty {
                let encodedTextRows = try textItems.map { prep -> (item: PreparedItem, tokens: [Int]) in
                    let bodyTokens = tokenizer.encode(text: prep.combinedText, addSpecialTokens: false)
                    let fullCount = bodyTokens.count + 2
                    if prep.item.usesExplicitPromptName && fullCount > descriptor.maxInputTokens {
                        throw EmbeddingValidationError.invalidInput(
                            statusCode: 400,
                            detail: "input[\(prep.index)]: expanded input has \(fullCount) tokens; limit is \(descriptor.maxInputTokens)"
                        )
                    }
                    let maxBody = max(0, descriptor.maxInputTokens - 2)
                    let tokens = [bosID] + Array(bodyTokens.prefix(maxBody)) + [eosID]
                    return (prep, tokens)
                }

                let batches = Self.makeLengthBatches(
                    lengths: encodedTextRows.map(\.tokens.count),
                    limit: Self.textBatchTokenLimit
                )
                for batchIndices in batches {
                    try Task.checkCancellation()
                    let rows = batchIndices.map { encodedTextRows[$0].tokens }
                    let width = rows.map(\.count).max() ?? 0
                    let ids = rows.flatMap { $0 + repeatElement(0, count: width - $0.count) }
                    let needsMask = !rows.allSatisfy { $0.count == width }
                    let maskArray: MLXArray? = needsMask
                        ? MLXArray(
                            rows.flatMap {
                                [Int32](repeating: 1, count: $0.count) + repeatElement(Int32(0), count: width - $0.count)
                            },
                            [rows.count, width]
                        )
                        : nil
                    let rawVectors = hybridModel.embed(
                        inputIds: MLXArray(ids.map(Int32.init), [rows.count, width]),
                        attentionMask: maskArray
                    )
                    let vectors = try Self.projectUnitRows(rawVectors, dimension: targetDim)
                    for (rowPos, vec) in zip(batchIndices, vectors) {
                        let row = encodedTextRows[rowPos]
                        outputs[row.item.index] = EmbeddedOutput(
                            vector: vec,
                            windows: [],
                            inputTokens: row.tokens.count,
                            promptName: row.item.promptName
                        )
                    }
                }
            }

            for prep in preparedItems where !prep.isTextOnly {
                try Task.checkCancellation()
                let primaryRes = try await embedMediaItemHybrid(
                    prep: prep,
                    model: hybridModel,
                    tokenizer: tokenizer,
                    bosID: bosID,
                    eosID: eosID,
                    wantsANEVision: wantsANEVision,
                    wantsANEAudio: wantsANEAudio,
                    dimensions: targetDim
                )
                var windowVectors: [VideoWindowEmbeddingResult] = []
                var windowTokens = 0
                if let videoURL = prep.windowedVideoURL, let winOpts = prep.windowOptions {
                    let res = try await embedVideoWindows(
                        videoURL: videoURL,
                        windowOptions: winOpts,
                        includeAudio: prep.hasCompanionAudio,
                        promptPrefix: prep.promptPrefix,
                        dimensions: dimensions,
                        mode: .hybridANE(
                            model: hybridModel,
                            tokenizer: tokenizer,
                            bosID: bosID,
                            eosID: eosID,
                            wantsANEVision: wantsANEVision,
                            wantsANEAudio: wantsANEAudio
                        )
                    )
                    windowVectors = res.windows
                    windowTokens = res.tokens
                }
                outputs[prep.index] = EmbeddedOutput(
                    vector: primaryRes.vector,
                    windows: windowVectors,
                    inputTokens: primaryRes.tokens + windowTokens,
                    promptName: prep.promptName
                )
            }
            return outputs.compactMap { $0 }
        }

        // Pure MLX GPU mode (`all_gpu`, or default `.hybridANETowers` when `.mlmodelc` bundles are absent).
        let embedder = try await resolvedEmbedder()
        let textItems = preparedItems.filter(\.isTextOnly)
        if !textItems.isEmpty {
            var batchInputs: [MLXVLM.EmbeddingInput] = []
            var batchTokens: [Int] = []
            batchInputs.reserveCapacity(textItems.count)
            batchTokens.reserveCapacity(textItems.count)

            for prep in textItems {
                let encodedCount = tokenizer.encode(text: prep.combinedText, addSpecialTokens: false).count + 2
                if prep.item.usesExplicitPromptName && encodedCount > descriptor.maxInputTokens {
                    throw EmbeddingValidationError.invalidInput(
                        statusCode: 400,
                        detail: "input[\(prep.index)]: expanded input has \(encodedCount) tokens; limit is \(descriptor.maxInputTokens)"
                    )
                }
                batchInputs.append(MLXVLM.EmbeddingInput([.text(prep.combinedText)], title: nil))
                batchTokens.append(min(encodedCount, descriptor.maxInputTokens))
            }

            let batchVectors: [[Float]]
            do {
                batchVectors = try await embedder.embed(batchInputs, task: .none, dimensions: targetDim)
            } catch let vlmError as MLXVLM.EmbeddingGemma2Embedding.Error {
                throw Self.mapVLMError(vlmError, itemIndex: textItems[0].index)
            }

            for (pos, vec) in batchVectors.enumerated() {
                let prep = textItems[pos]
                outputs[prep.index] = EmbeddedOutput(
                    vector: vec,
                    windows: [],
                    inputTokens: batchTokens[pos],
                    promptName: prep.promptName
                )
            }
        }

        for prep in preparedItems where !prep.isTextOnly {
            try Task.checkCancellation()
            let vlmInput = Self.buildVLMInput(parts: prep.resolvedParts, promptPrefix: prep.promptPrefix)
            let primary: [Float]
            do {
                primary = try await embedder.embed(vlmInput, task: .none, dimensions: targetDim)
            } catch let vlmError as MLXVLM.EmbeddingGemma2Embedding.Error {
                throw Self.mapVLMError(vlmError, itemIndex: prep.index)
            }
            let primaryTokens = await estimateItemTokenCount(prep: prep, tokenizer: tokenizer)
            if prep.item.usesExplicitPromptName && primaryTokens > descriptor.maxInputTokens {
                throw EmbeddingValidationError.invalidInput(
                    statusCode: 400,
                    detail: "input[\(prep.index)]: expanded input has \(primaryTokens) tokens; limit is \(descriptor.maxInputTokens)"
                )
            }

            var windowVectors: [VideoWindowEmbeddingResult] = []
            var windowTokens = 0
            if let videoURL = prep.windowedVideoURL, let winOpts = prep.windowOptions {
                let res = try await embedVideoWindows(
                    videoURL: videoURL,
                    windowOptions: winOpts,
                    includeAudio: prep.hasCompanionAudio,
                    promptPrefix: prep.promptPrefix,
                    dimensions: targetDim,
                    mode: .mlxGPU(embedder: embedder, tokenizer: tokenizer)
                )
                windowVectors = res.windows
                windowTokens = res.tokens
            }
            outputs[prep.index] = EmbeddedOutput(
                vector: primary,
                windows: windowVectors,
                inputTokens: primaryTokens + windowTokens,
                promptName: prep.promptName
            )
        }
        return outputs.compactMap { $0 }
    }

    private enum WindowExecutionMode {
        case mlxGPU(embedder: MLXVLM.EmbeddingGemma2Embedding, tokenizer: any MLXLMCommon.Tokenizer)
        case hybridANE(
            model: MLXVLM.EmbeddingGemma2,
            tokenizer: any MLXLMCommon.Tokenizer,
            bosID: Int,
            eosID: Int,
            wantsANEVision: Bool,
            wantsANEAudio: Bool
        )
        case fullANE(tokenizer: any MLXLMCommon.Tokenizer, bosID: Int, eosID: Int)
    }

    private func embedMediaItemHybrid(
        prep: PreparedItem,
        model: MLXVLM.EmbeddingGemma2,
        tokenizer: any MLXLMCommon.Tokenizer,
        bosID: Int,
        eosID: Int,
        wantsANEVision: Bool,
        wantsANEAudio: Bool,
        dimensions: Int
    ) async throws -> (vector: [Float], tokens: Int) {
        let visionUnit: TowerComputeUnit = computeOptions.visionEncoder == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.visionEncoder
        let audioUnit: TowerComputeUnit = computeOptions.audioEncoder == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.audioEncoder

        let videoToken = model.config.softTokenIDs.count > 1 ? descriptor.videoTokenID : descriptor.imageTokenID
        let effectiveVideoToken = model.config.softTokenIDs.contains(videoToken) ? videoToken : descriptor.imageTokenID

        enum SegmentKind {
            case text([Int])
            case mediaBlock([Int])
        }

        var segments: [SegmentKind] = []
        var softTokenArrays: [MLXArray] = []
        var pendingText = ""
        var didPrepend = false

        func flushText() {
            if !pendingText.isEmpty {
                segments.append(.text(tokenizer.encode(text: pendingText, addSpecialTokens: false)))
                pendingText = ""
            }
        }

        if !prep.promptPrefix.isEmpty && !prep.item.hasNonWhitespaceText {
            didPrepend = true
            pendingText += prep.promptPrefix
        }

        for part in prep.resolvedParts {
            switch part {
            case .text(let s):
                if !didPrepend && !prep.promptPrefix.isEmpty {
                    didPrepend = true
                    pendingText += (s.hasPrefix("task:") || s.hasPrefix("title:")) ? s : (prep.promptPrefix + s)
                } else {
                    pendingText += s
                }

            case .image(let imageURL):
                flushText()
                guard let imgPrep = ANECoreMLRuntime.prepareImagePixels(from: imageURL, forVideoBudget: false) else {
                    throw EmbeddingValidationError.invalidInput(
                        statusCode: 400,
                        detail: "input[\(prep.index)]: failed to decode image"
                    )
                }
                let softArray: MLXArray
                let tokenCount: Int
                if wantsANEVision,
                   let encoded = try await aneRuntime.encodeVision(
                       pixels: imgPrep.pixels,
                       height: imgPrep.height,
                       width: imgPrep.width,
                       unit: visionUnit
                   ) {
                    softArray = MLXArray(encoded.softTokens, [encoded.tokenCount, descriptor.hiddenSize])
                    tokenCount = encoded.tokenCount
                } else {
                    let pixels = MLXArray(imgPrep.pixels, [1, 3, imgPrep.height, imgPrep.width])
                    guard let features = try model.imageFeatures(pixels) else {
                        throw EmbeddingValidationError.unsupportedModality(
                            itemIndex: prep.index,
                            modality: "image",
                            supported: descriptor.supportedModalities
                        )
                    }
                    try MLX.checkedEval(features)
                    tokenCount = features.dim(1)
                    softArray = features.reshaped(-1, features.dim(-1))
                }
                softTokenArrays.append(softArray)
                segments.append(.mediaBlock(
                    [descriptor.boiTokenID] + repeatElement(descriptor.imageTokenID, count: tokenCount) + [descriptor.eoiTokenID]
                ))

            case .video(let videoURL):
                flushText()
                guard let metadata = try await ArtifactProcessor.inspectVideoMetadata(for: videoURL) else {
                    throw EmbeddingValidationError.invalidInput(
                        statusCode: 400,
                        detail: "input[\(prep.index)]: failed to decode video track"
                    )
                }
                let sampleTimes = WindowPlanner.uniformSampleTimes(
                    duration: metadata.duration,
                    nativeFPS: metadata.nativeFPS,
                    fps: descriptor.videoFPS,
                    maxFrames: descriptor.videoMaxFrames
                )
                let decoded = try await ArtifactProcessor.decodeVideoFrames(
                    for: videoURL,
                    at: sampleTimes,
                    startOffsetSeconds: metadata.startSeconds
                )
                guard !decoded.isEmpty else {
                    throw EmbeddingValidationError.invalidInput(
                        statusCode: 400,
                        detail: "input[\(prep.index)]: no frames decoded from video"
                    )
                }
                var frameTokensBlock: [Int] = []
                for frame in decoded {
                    guard let framePrep = ANECoreMLRuntime.prepareImagePixels(from: frame.image, forVideoBudget: true) else {
                        continue
                    }
                    let softArray: MLXArray
                    let tokenCount: Int
                    if wantsANEVision,
                       let encoded = try await aneRuntime.encodeVision(
                           pixels: framePrep.pixels,
                           height: framePrep.height,
                           width: framePrep.width,
                           unit: visionUnit
                       ) {
                        softArray = MLXArray(encoded.softTokens, [encoded.tokenCount, descriptor.hiddenSize])
                        tokenCount = encoded.tokenCount
                    } else {
                        let pixels = MLXArray(framePrep.pixels, [1, 3, framePrep.height, framePrep.width])
                        guard let features = try model.imageFeatures(pixels) else {
                            throw EmbeddingValidationError.unsupportedModality(
                                itemIndex: prep.index,
                                modality: "video",
                                supported: descriptor.supportedModalities
                            )
                        }
                        try MLX.checkedEval(features)
                        tokenCount = features.dim(1)
                        softArray = features.reshaped(-1, features.dim(-1))
                    }
                    softTokenArrays.append(softArray)
                    frameTokensBlock.append(descriptor.boiTokenID)
                    frameTokensBlock.append(contentsOf: repeatElement(effectiveVideoToken, count: tokenCount))
                    frameTokensBlock.append(descriptor.eoiTokenID)
                }
                segments.append(.mediaBlock(frameTokensBlock))

            case .audio(let audioURL):
                flushText()
                let pcm = try await ArtifactProcessor.decodeMonoPCM(for: audioURL)
                if let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: pcm) {
                    let softArray: MLXArray
                    let tokenCount: Int
                    if wantsANEAudio,
                       let encoded = try await aneRuntime.encodeAudio(
                           melFeatures: mel.features,
                           frameCount: mel.frameCount,
                           unit: audioUnit
                       ) {
                        softArray = MLXArray(encoded.softTokens, [encoded.tokenCount, descriptor.hiddenSize])
                        tokenCount = encoded.tokenCount
                    } else {
                        let melArray = MLXArray(mel.features, [mel.frameCount, ANECoreMLRuntime.audioMelBins])
                        guard let features = try model.audioFeatures(melArray) else {
                            throw EmbeddingValidationError.unsupportedModality(
                                itemIndex: prep.index,
                                modality: "audio",
                                supported: descriptor.supportedModalities
                            )
                        }
                        try MLX.checkedEval(features)
                        tokenCount = features.dim(1)
                        softArray = features.reshaped(-1, features.dim(-1))
                    }
                    softTokenArrays.append(softArray)
                    segments.append(.mediaBlock(
                        [descriptor.boaTokenID] + repeatElement(descriptor.audioTokenID, count: tokenCount) + [descriptor.eoaTokenID]
                    ))
                } else {
                    segments.append(.mediaBlock([descriptor.boaTokenID, descriptor.eoaTokenID]))
                }
            }
        }
        flushText()

        let mediaTokenCount = segments.reduce(0) { sum, seg in
            if case let .mediaBlock(ids) = seg { return sum + ids.count }
            return sum
        }
        let placeholders: Set<Int> = Set(model.config.softTokenIDs)
        var textBudget = descriptor.maxInputTokens - 2 - mediaTokenCount
        guard textBudget >= 0 else {
            throw EmbeddingValidationError.invalidInput(
                statusCode: 400,
                detail: "input[\(prep.index)]: expanded input exceeds \(descriptor.maxInputTokens) tokens"
            )
        }
        var tokenIDs = [bosID]
        for seg in segments {
            switch seg {
            case .mediaBlock(let ids):
                tokenIDs.append(contentsOf: ids)
            case .text(let ids):
                if !softTokenArrays.isEmpty && ids.contains(where: placeholders.contains) {
                    throw EmbeddingValidationError.invalidInput(
                        statusCode: 400,
                        detail: "input[\(prep.index)]: text next to media must not contain placeholder tokens"
                    )
                }
                if prep.item.usesExplicitPromptName && ids.count > textBudget {
                    throw EmbeddingValidationError.invalidInput(
                        statusCode: 400,
                        detail: "input[\(prep.index)]: expanded input exceeds \(descriptor.maxInputTokens) tokens"
                    )
                }
                let takeCount = min(ids.count, textBudget)
                tokenIDs.append(contentsOf: ids.prefix(takeCount))
                textBudget -= takeCount
            }
        }
        tokenIDs.append(eosID)

        let fusedSoft: MLXArray? = softTokenArrays.isEmpty ? nil : MLX.concatenated(softTokenArrays, axis: 0)
        let rawVector = model.embed(
            inputIds: MLXArray(tokenIDs.map(Int32.init), [1, tokenIDs.count]),
            attentionMask: nil,
            softTokens: fusedSoft
        )
        let row = try Self.projectUnitRows(rawVector, dimension: dimensions)[0]
        return (row, tokenIDs.count)
    }

    private func embedItemFullANE(
        prep: PreparedItem,
        tokenizer: any MLXLMCommon.Tokenizer,
        bosID: Int,
        eosID: Int,
        dimensions: Int
    ) async throws -> (vector: [Float], tokens: Int) {
        var tokenIDs: [Int] = [bosID]
        var softTokens: [Float] = []
        var didPrependPrompt = false

        if !prep.promptPrefix.isEmpty && !prep.item.hasNonWhitespaceText {
            didPrependPrompt = true
            tokenIDs.append(contentsOf: tokenizer.encode(text: prep.promptPrefix, addSpecialTokens: false))
        }

        for part in prep.resolvedParts {
            switch part {
            case .text(let rawText):
                let formatted: String
                if !didPrependPrompt && !prep.promptPrefix.isEmpty {
                    didPrependPrompt = true
                    formatted = (rawText.hasPrefix("task:") || rawText.hasPrefix("title:"))
                        ? rawText
                        : (prep.promptPrefix + rawText)
                } else {
                    formatted = rawText
                }
                if !formatted.isEmpty {
                    tokenIDs.append(contentsOf: tokenizer.encode(text: formatted, addSpecialTokens: false))
                }

            case .image(let imageURL):
                guard let imgPrep = ANECoreMLRuntime.prepareImagePixels(from: imageURL, forVideoBudget: false),
                      let encoded = try await aneRuntime.encodeVision(
                          pixels: imgPrep.pixels,
                          height: imgPrep.height,
                          width: imgPrep.width,
                          unit: computeOptions.visionEncoder
                      )
                else {
                    throw EmbeddingEngineError.runtimeUnavailable(
                        "VisionANE.mlmodelc execution unavailable at '\(checkpointURL.path)'."
                    )
                }
                tokenIDs.append(descriptor.boiTokenID)
                tokenIDs.append(contentsOf: repeatElement(descriptor.imageTokenID, count: encoded.tokenCount))
                tokenIDs.append(descriptor.eoiTokenID)
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
                    throw EmbeddingEngineError.runtimeUnavailable(
                        "AudioANE.mlmodelc execution unavailable at '\(checkpointURL.path)'."
                    )
                }
                tokenIDs.append(descriptor.boaTokenID)
                tokenIDs.append(contentsOf: repeatElement(descriptor.audioTokenID, count: encoded.tokenCount))
                tokenIDs.append(descriptor.eoaTokenID)
                softTokens.append(contentsOf: encoded.softTokens)

            case .video(let videoURL):
                guard let metadata = try await ArtifactProcessor.inspectVideoMetadata(for: videoURL) else {
                    throw EmbeddingValidationError.invalidInput(
                        statusCode: 400,
                        detail: "input[\(prep.index)]: failed to decode video track"
                    )
                }
                let sampleTimes = WindowPlanner.uniformSampleTimes(
                    duration: metadata.duration,
                    nativeFPS: metadata.nativeFPS,
                    fps: descriptor.videoFPS,
                    maxFrames: descriptor.videoMaxFrames
                )
                let decoded = try await ArtifactProcessor.decodeVideoFrames(
                    for: videoURL,
                    at: sampleTimes,
                    startOffsetSeconds: metadata.startSeconds
                )
                guard !decoded.isEmpty else {
                    throw EmbeddingValidationError.invalidInput(
                        statusCode: 400,
                        detail: "input[\(prep.index)]: no frames decoded from video"
                    )
                }
                for frame in decoded {
                    guard let framePrep = ANECoreMLRuntime.prepareImagePixels(from: frame.image, forVideoBudget: true),
                          let encoded = try await aneRuntime.encodeVision(
                              pixels: framePrep.pixels,
                              height: framePrep.height,
                              width: framePrep.width,
                              unit: computeOptions.visionEncoder
                          )
                    else {
                        throw EmbeddingEngineError.runtimeUnavailable(
                            "VisionANE.mlmodelc execution unavailable for video frame."
                        )
                    }
                    tokenIDs.append(descriptor.boiTokenID)
                    tokenIDs.append(contentsOf: repeatElement(descriptor.videoTokenID, count: encoded.tokenCount))
                    tokenIDs.append(descriptor.eoiTokenID)
                    softTokens.append(contentsOf: encoded.softTokens)
                }
            }
        }
        tokenIDs.append(eosID)
        if prep.item.usesExplicitPromptName && tokenIDs.count > descriptor.maxInputTokens {
            throw EmbeddingValidationError.invalidInput(
                statusCode: 400,
                detail: "input[\(prep.index)]: expanded input has \(tokenIDs.count) tokens; limit is \(descriptor.maxInputTokens)"
            )
        }
        guard let vec = try await aneRuntime.encodeTextAndFuse(
            tokenIDs: tokenIDs,
            softTokens: softTokens.isEmpty ? nil : softTokens,
            softTokenIDs: [descriptor.imageTokenID, descriptor.videoTokenID, descriptor.audioTokenID],
            unit: computeOptions.textBackbone,
            dimensions: dimensions
        ) else {
            throw EmbeddingEngineError.runtimeUnavailable(
                "TextANE.mlmodelc execution unavailable at '\(checkpointURL.path)'."
            )
        }
        return (vec, tokenIDs.count)
    }

    private func embedVideoWindows(
        videoURL: URL,
        windowOptions: VideoWindowOptions,
        includeAudio: Bool,
        promptPrefix: String,
        dimensions: Int?,
        mode: WindowExecutionMode
    ) async throws -> (windows: [VideoWindowEmbeddingResult], tokens: Int) {
        guard let metadata = try await ArtifactProcessor.inspectVideoMetadata(for: videoURL) else {
            return ([], 0)
        }
        let windowSeconds = windowOptions.seconds ?? descriptor.maxWindowSeconds
        let ranges = WindowPlanner.consecutiveRanges(
            duration: metadata.duration,
            windowSeconds: windowSeconds
        )
        guard !ranges.isEmpty else { return ([], 0) }

        let pcmSamples: [Float] = includeAudio
            ? ((try? await ArtifactProcessor.decodeMonoPCM(for: videoURL)) ?? [])
            : []

        let targetDim = dimensions ?? nativeDimensions
        var out: [VideoWindowEmbeddingResult] = []
        out.reserveCapacity(ranges.count)
        var totalTokens = 0

        for (lo, hi) in ranges {
            try Task.checkCancellation()
            let decoded = try await ArtifactProcessor.decodeVideoWindowFrames(
                for: videoURL,
                metadata: metadata,
                lo: lo,
                hi: hi,
                sampling: windowOptions.sampling,
                fps: descriptor.videoFPS,
                maxFrames: descriptor.videoMaxFrames
            )
            guard !decoded.isEmpty else { continue }

            let frameTimes = decoded.map { ($0.time * 1000).rounded() / 1000 }
            let winStart = (lo * 1000).rounded() / 1000
            let winEnd = (hi * 1000).rounded() / 1000

            let vec: [Float]
            let winTokens: Int

            switch mode {
            case let .mlxGPU(embedder, tokenizer):
                var tempAudioURL: URL?
                defer {
                    if let tempAudioURL {
                        try? FileManager.default.removeItem(at: tempAudioURL)
                    }
                }
                var audioMelFrames = 0
                if !pcmSamples.isEmpty,
                   let sliceWAV = ArtifactProcessor.encodeAudioSliceWAV(
                       samples: pcmSamples,
                       sampleRate: ArtifactProcessor.audioSampleRate,
                       startSeconds: lo,
                       endSeconds: min(hi, lo + ArtifactProcessor.audioWindowSeconds)
                   ) {
                    let wavURL = FileManager.default.temporaryDirectory
                        .appendingPathComponent("heylook-win-audio-\(UUID().uuidString).wav")
                    if (try? sliceWAV.write(to: wavURL)) != nil {
                        tempAudioURL = wavURL
                        let sr = Double(ArtifactProcessor.audioSampleRate)
                        let sIdx = max(0, Int((lo * sr).rounded()))
                        let eIdx = min(pcmSamples.count, Int((min(hi, lo + ArtifactProcessor.audioWindowSeconds) * sr).rounded()))
                        if eIdx - sIdx >= 400 {
                            let rawMelFrames = min(ANECoreMLRuntime.audioMelFrames, 1 + ((eIdx - sIdx) - 400) / 160)
                            audioMelFrames = min(750, max(1, (((rawMelFrames + 1) / 2) + 1) / 2))
                        }
                    }
                }
                vec = try await embedder.embed(
                    Self.buildWindowVLMInput(frames: decoded, audioWAVURL: tempAudioURL, promptPrefix: promptPrefix),
                    task: .none,
                    dimensions: targetDim
                )
                let promptTokens = promptPrefix.isEmpty ? 0 : tokenizer.encode(text: promptPrefix, addSpecialTokens: false).count
                let frameTokenBlock = decoded.reduce(0) { acc, frame in
                    let estPerFrame = min(
                        descriptor.videoMaxSoftTokens,
                        max(1, Int((Double(frame.image.width * frame.image.height) / (28.0 * 28.0)).rounded()))
                    )
                    return acc + 2 + estPerFrame
                }
                winTokens = 2 + promptTokens + frameTokenBlock + (audioMelFrames > 0 ? (3 + audioMelFrames) : 0)

            case let .hybridANE(model, tokenizer, bosID, eosID, wantsANEVision, wantsANEAudio):
                let res = try await embedWindowHybridANE(
                    frames: decoded,
                    pcmSamples: pcmSamples,
                    windowStart: lo,
                    windowEnd: hi,
                    promptPrefix: promptPrefix,
                    dimensions: targetDim,
                    model: model,
                    tokenizer: tokenizer,
                    bosID: bosID,
                    eosID: eosID,
                    wantsANEVision: wantsANEVision,
                    wantsANEAudio: wantsANEAudio
                )
                vec = res.vector
                winTokens = res.tokens

            case let .fullANE(tokenizer, bosID, eosID):
                let res = try await embedWindowFullANE(
                    frames: decoded,
                    pcmSamples: pcmSamples,
                    windowStart: lo,
                    windowEnd: hi,
                    promptPrefix: promptPrefix,
                    dimensions: targetDim,
                    tokenizer: tokenizer,
                    bosID: bosID,
                    eosID: eosID
                )
                vec = res.vector
                winTokens = res.tokens
            }

            totalTokens += winTokens
            out.append(
                VideoWindowEmbeddingResult(
                    start: winStart,
                    end: winEnd,
                    frameTimes: frameTimes,
                    vector: vec
                )
            )
        }
        return (out, totalTokens)
    }

    private func embedWindowHybridANE(
        frames: [(time: Double, image: CGImage)],
        pcmSamples: [Float],
        windowStart: Double,
        windowEnd: Double,
        promptPrefix: String,
        dimensions: Int,
        model: MLXVLM.EmbeddingGemma2,
        tokenizer: any MLXLMCommon.Tokenizer,
        bosID: Int,
        eosID: Int,
        wantsANEVision: Bool,
        wantsANEAudio: Bool
    ) async throws -> (vector: [Float], tokens: Int) {
        let visionUnit: TowerComputeUnit = computeOptions.visionEncoder == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.visionEncoder
        let audioUnit: TowerComputeUnit = computeOptions.audioEncoder == .mlxGPU
            ? .cpuAndNeuralEngine
            : computeOptions.audioEncoder

        let videoToken = model.config.softTokenIDs.contains(descriptor.videoTokenID)
            ? descriptor.videoTokenID
            : descriptor.imageTokenID

        var tokenIDs: [Int] = [bosID]
        if !promptPrefix.isEmpty {
            tokenIDs.append(contentsOf: tokenizer.encode(text: promptPrefix, addSpecialTokens: false))
        }
        var softTokenArrays: [MLXArray] = []

        for frame in frames {
            guard let prep = ANECoreMLRuntime.prepareImagePixels(from: frame.image, forVideoBudget: true) else {
                continue
            }
            let softArray: MLXArray
            let tokenCount: Int
            if wantsANEVision,
               let encoded = try await aneRuntime.encodeVision(
                   pixels: prep.pixels,
                   height: prep.height,
                   width: prep.width,
                   unit: visionUnit
               ) {
                softArray = MLXArray(encoded.softTokens, [encoded.tokenCount, descriptor.hiddenSize])
                tokenCount = encoded.tokenCount
            } else {
                let pixels = MLXArray(prep.pixels, [1, 3, prep.height, prep.width])
                guard let features = try model.imageFeatures(pixels) else {
                    throw EmbeddingValidationError.unsupportedModality(
                        itemIndex: 0,
                        modality: "video",
                        supported: descriptor.supportedModalities
                    )
                }
                try MLX.checkedEval(features)
                tokenCount = features.dim(1)
                softArray = features.reshaped(-1, features.dim(-1))
            }
            tokenIDs.append(descriptor.boiTokenID)
            tokenIDs.append(contentsOf: repeatElement(videoToken, count: tokenCount))
            tokenIDs.append(descriptor.eoiTokenID)
            softTokenArrays.append(softArray)
        }

        if !pcmSamples.isEmpty {
            let sr = Double(ArtifactProcessor.audioSampleRate)
            let startIdx = max(0, Int((windowStart * sr).rounded()))
            let endSec = min(windowEnd, windowStart + ArtifactProcessor.audioWindowSeconds)
            let endIdx = min(pcmSamples.count, Int((endSec * sr).rounded()))
            if endIdx > startIdx {
                let slice = Array(pcmSamples[startIdx..<endIdx])
                if let mel = ANECoreMLRuntime.computeAudioLogMel(pcmSamples: slice) {
                    let softArray: MLXArray
                    let tokenCount: Int
                    if wantsANEAudio,
                       let encAudio = try await aneRuntime.encodeAudio(
                           melFeatures: mel.features,
                           frameCount: mel.frameCount,
                           unit: audioUnit
                       ) {
                        softArray = MLXArray(encAudio.softTokens, [encAudio.tokenCount, descriptor.hiddenSize])
                        tokenCount = encAudio.tokenCount
                    } else {
                        let melArray = MLXArray(mel.features, [mel.frameCount, ANECoreMLRuntime.audioMelBins])
                        guard let features = try model.audioFeatures(melArray) else {
                            throw EmbeddingValidationError.unsupportedModality(
                                itemIndex: 0,
                                modality: "audio",
                                supported: descriptor.supportedModalities
                            )
                        }
                        try MLX.checkedEval(features)
                        tokenCount = features.dim(1)
                        softArray = features.reshaped(-1, features.dim(-1))
                    }
                    tokenIDs.append(contentsOf: tokenizer.encode(text: " ", addSpecialTokens: false))
                    tokenIDs.append(descriptor.boaTokenID)
                    tokenIDs.append(contentsOf: repeatElement(descriptor.audioTokenID, count: tokenCount))
                    tokenIDs.append(descriptor.eoaTokenID)
                    softTokenArrays.append(softArray)
                }
            }
        }

        tokenIDs.append(eosID)
        let fusedSoft: MLXArray? = softTokenArrays.isEmpty ? nil : MLX.concatenated(softTokenArrays, axis: 0)
        let rawVector = model.embed(
            inputIds: MLXArray(tokenIDs.map(Int32.init), [1, tokenIDs.count]),
            attentionMask: nil,
            softTokens: fusedSoft
        )
        let row = try Self.projectUnitRows(rawVector, dimension: dimensions)[0]
        return (row, tokenIDs.count)
    }

    private func embedWindowFullANE(
        frames: [(time: Double, image: CGImage)],
        pcmSamples: [Float],
        windowStart: Double,
        windowEnd: Double,
        promptPrefix: String,
        dimensions: Int,
        tokenizer: any MLXLMCommon.Tokenizer,
        bosID: Int,
        eosID: Int
    ) async throws -> (vector: [Float], tokens: Int) {
        var tokenIDs: [Int] = [bosID]
        if !promptPrefix.isEmpty {
            tokenIDs.append(contentsOf: tokenizer.encode(text: promptPrefix, addSpecialTokens: false))
        }
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
                throw EmbeddingEngineError.runtimeUnavailable(
                    "VisionANE.mlmodelc execution unavailable for video window."
                )
            }
            tokenIDs.append(descriptor.boiTokenID)
            tokenIDs.append(contentsOf: repeatElement(descriptor.videoTokenID, count: encoded.tokenCount))
            tokenIDs.append(descriptor.eoiTokenID)
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
                    tokenIDs.append(descriptor.boaTokenID)
                    tokenIDs.append(contentsOf: repeatElement(descriptor.audioTokenID, count: encodedAudio.tokenCount))
                    tokenIDs.append(descriptor.eoaTokenID)
                    softTokens.append(contentsOf: encodedAudio.softTokens)
                }
            }
        }

        tokenIDs.append(eosID)
        guard let vec = try await aneRuntime.encodeTextAndFuse(
            tokenIDs: tokenIDs,
            softTokens: softTokens.isEmpty ? nil : softTokens,
            softTokenIDs: [descriptor.imageTokenID, descriptor.videoTokenID, descriptor.audioTokenID],
            unit: computeOptions.textBackbone,
            dimensions: dimensions
        ) else {
            throw EmbeddingEngineError.runtimeUnavailable(
                "TextANE.mlmodelc execution unavailable for video window."
            )
        }
        return (vec, tokenIDs.count)
    }

    private func estimateItemTokenCount(
        prep: PreparedItem,
        tokenizer: any MLXLMCommon.Tokenizer
    ) async -> Int {
        var total = 2
        if !prep.promptPrefix.isEmpty && !prep.item.hasNonWhitespaceText {
            total += tokenizer.encode(text: prep.promptPrefix, addSpecialTokens: false).count
        }
        var didPrepend = false
        for part in prep.resolvedParts {
            switch part {
            case .text(let s):
                let t = (!didPrepend && !prep.promptPrefix.isEmpty && !s.hasPrefix("task:") && !s.hasPrefix("title:"))
                    ? (prep.promptPrefix + s)
                    : s
                didPrepend = true
                total += tokenizer.encode(text: t, addSpecialTokens: false).count
            case .image:
                total += 2 + descriptor.imageMaxSoftTokens
            case .video(let url):
                if let md = try? await ArtifactProcessor.inspectVideoMetadata(for: url) {
                    let count = WindowPlanner.uniformSampleTimes(
                        duration: md.duration,
                        nativeFPS: md.nativeFPS,
                        fps: descriptor.videoFPS,
                        maxFrames: descriptor.videoMaxFrames
                    ).count
                    total += max(1, count) * (2 + descriptor.videoMaxSoftTokens)
                } else {
                    total += 2 + descriptor.videoMaxSoftTokens
                }
            case .audio(let url):
                if let pcm = try? await ArtifactProcessor.decodeMonoPCM(for: url), pcm.count >= 400 {
                    let capped = min(pcm.count, 480_000)
                    let rawFrames = 1 + (capped - 400) / 160
                    let audioTokens = min(750, max(1, (((rawFrames + 1) / 2) + 1) / 2))
                    total += 2 + audioTokens
                } else {
                    total += 2
                }
            }
        }
        return min(total, descriptor.maxInputTokens)
    }

    /// Groups sequences into padded batches holding at most `limit` tokens (longest first-matched ascending).
    private static func makeLengthBatches(lengths: [Int], limit: Int) -> [[Int]] {
        let order = lengths.indices.sorted { (lengths[$0], $0) < (lengths[$1], $1) }
        var batches: [[Int]] = []
        for idx in order {
            if let last = batches.last, (last.count + 1) * lengths[idx] <= limit {
                batches[batches.count - 1].append(idx)
            } else {
                batches.append([idx])
            }
        }
        return batches
    }

    private static func projectUnitRows(_ vectors: MLXArray, dimension: Int) throws -> [[Float]] {
        try MLX.checkedEval(vectors)
        let width = vectors.dim(1)
        let values = vectors.asArray(Float.self)
        return try stride(from: 0, to: values.count, by: width).map { start in
            let row = values[start ..< start + width]
            guard row.allSatisfy(\.isFinite), row.contains(where: { $0 != 0 }) else {
                throw EmbeddingEngineError.invalidResponse(
                    status: 500,
                    detail: "Embedding model produced a non-finite or zero vector."
                )
            }
            return Matryoshka.project(Array(row), dimensions: dimension)
        }
    }

    private static func mapVLMError(
        _ error: MLXVLM.EmbeddingGemma2Embedding.Error,
        itemIndex: Int
    ) -> Error {
        switch error {
        case .emptyInput:
            return EmbeddingValidationError.invalidInput(
                statusCode: 400,
                detail: "input[\(itemIndex)]: embedding input must contain text or media"
            )
        case .unsupportedMedia:
            return EmbeddingValidationError.unsupportedModality(
                itemIndex: itemIndex,
                modality: "media",
                supported: []
            )
        case .invalidMedia:
            return EmbeddingValidationError.invalidInput(
                statusCode: 400,
                detail: "input[\(itemIndex)]: media item could not be decoded"
            )
        case .placeholderInText:
            return EmbeddingValidationError.invalidInput(
                statusCode: 400,
                detail: "input[\(itemIndex)]: text next to media must not contain media placeholder tokens"
            )
        case .contextExceeded:
            return EmbeddingValidationError.invalidInput(
                statusCode: 400,
                detail: "input[\(itemIndex)]: expanded input exceeds model context length"
            )
        case .invalidDimension:
            return EmbeddingValidationError.unsupportedDimensions(
                requested: 0,
                supported: Matryoshka.gemma2Dimensions
            )
        case .invalidEmbedding:
            return EmbeddingEngineError.invalidResponse(
                status: 500,
                detail: error.localizedDescription
            )
        }
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
