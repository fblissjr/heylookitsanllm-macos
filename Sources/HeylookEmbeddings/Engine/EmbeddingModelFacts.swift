import Foundation
import HeylookCore

/// Canonical `EngineEmbeddingFacts` derivation (`embedding_models.py` parity) from on-disk MLX,
/// CoreAI, and GGUF checkpoints without running inference or hardcoding fake defaults.
public enum EmbeddingModelFacts {
    public static let gemma2Dimensions: [Int] = Matryoshka.gemma2Dimensions
    public static let gemma2NativeDimensions: Int = Matryoshka.gemma2NativeDimensions
    public static let gemma2MaxInputTokens: Int = 8_192

    /// Prompt catalog loaded from `config_sentence_transformers.json`.
    public struct PromptCatalog: Sendable, Equatable {
        public let prompts: [String: String]
        public let defaultPromptName: String?
        public let maxSeqLength: Int?

        public init(
            prompts: [String: String] = [:],
            defaultPromptName: String? = nil,
            maxSeqLength: Int? = nil
        ) {
            self.prompts = prompts
            self.defaultPromptName = defaultPromptName
            self.maxSeqLength = maxSeqLength
        }

        public static func load(from directory: URL) -> PromptCatalog {
            let url = directory.appendingPathComponent("config_sentence_transformers.json")
            guard let data = try? Data(contentsOf: url),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                return PromptCatalog()
            }
            var parsed = [String: String]()
            if let raw = obj["prompts"] as? [String: Any] {
                for (k, v) in raw {
                    if let s = v as? String {
                        parsed[k] = s
                    }
                }
            }
            let defName = obj["default_prompt_name"] as? String
            let maxSeq = (obj["max_seq_length"] as? NSNumber)?.intValue
            return PromptCatalog(prompts: parsed, defaultPromptName: defName, maxSeqLength: maxSeq)
        }
    }

    /// Derives `EngineEmbeddingFacts` directly from a loaded `CheckpointDescriptor`.
    public static func fromCheckpoint(
        _ descriptor: NativeMLXEmbeddingGemma2.CheckpointDescriptor,
        computePlacement: String? = nil
    ) -> EngineEmbeddingFacts {
        let nativeDim = max(1, descriptor.embeddingDim)
        let dimensions = supportedDimensions(forNativeDimension: nativeDim)
        let modalities = descriptor.supportedModalities
        let videoWindows: EngineEmbeddingVideoWindows? = modalities.contains("video")
            ? EngineEmbeddingVideoWindows(
                maxSeconds: descriptor.maxWindowSeconds,
                sampling: ["fps", "content"],
                defaultSampling: "fps"
            )
            : nil

        return EngineEmbeddingFacts(
            modalities: modalities,
            dimensions: dimensions,
            nativeDimensions: nativeDim,
            maxInputTokens: descriptor.maxInputTokens,
            prompts: descriptor.prompts,
            normalization: "l2",
            videoAudio: "separate_audio_block",
            videoWindows: videoWindows,
            limits: EngineEmbeddingLimits(
                maxBatchSize: EmbeddingRequestLimits.maxItems,
                maxTotalTokens: descriptor.maxInputTokens,
                maxImagesPerItem: 8,
                maxAudioSeconds: ArtifactProcessor.audioWindowSeconds
            ),
            provenance: "model_files_and_architecture_contract",
            loadProfile: nil,
            computePlacement: computePlacement
        )
    }

    /// Derives `EngineEmbeddingFacts` from an on-disk checkpoint path (`MLX`, `CoreAI`/`ANE`, or `GGUF`).
    /// Returns `nil` when the checkpoint is not an identified embedding model.
    public static func deriveFacts(
        provider: EngineProviderKind,
        modelPath: URL,
        mmprojPath: URL? = nil,
        computePlacement: String? = nil,
        ctxSize: Int? = nil,
        nBatch: Int? = nil,
        nUBatch: Int? = nil
    ) -> EngineEmbeddingFacts? {
        switch provider {
        case .gguf:
            return deriveGGUFFacts(
                modelPath: modelPath,
                mmprojPath: mmprojPath,
                computePlacement: computePlacement,
                ctxSize: ctxSize,
                nBatch: nBatch,
                nUBatch: nUBatch
            )
        case .mlx, .coreai:
            guard let descriptor = try? NativeMLXEmbeddingGemma2.CheckpointDescriptor.load(from: modelPath) else {
                return nil
            }
            let isGemma2Emb = descriptor.modelType == "embedding_gemma2"
            let hasPooling = FileManager.default.fileExists(
                atPath: modelPath.appendingPathComponent("1_Pooling/config.json").path
            )
            guard isGemma2Emb || hasPooling || !descriptor.prompts.isEmpty else {
                return nil
            }
            return fromCheckpoint(descriptor, computePlacement: computePlacement)
        case .remote:
            return nil
        }
    }

    /// Validates GGUF embedding non-causal batch constraints (`embedding_models.py:gguf_embedding_profile`):
    /// `ctx_size <= n_ubatch <= n_batch` and `ctx_size <= maxContext`.
    public static func ggufEmbeddingProfile(
        ctxSize: Int? = nil,
        nBatch: Int? = nil,
        nUBatch: Int? = nil,
        maxContext: Int = gemma2MaxInputTokens,
        hasDraftOrSpec: Bool = false,
        hasExtraArgs: Bool = false
    ) throws -> [String: Int] {
        let context = ctxSize ?? maxContext
        guard context > 0 && context <= maxContext else {
            throw EmbeddingValidationError.invalidInput(
                statusCode: 400,
                detail: "Embedding context cannot exceed \(maxContext) (requested \(context))"
            )
        }
        let batch = nBatch ?? context
        let ubatch = nUBatch ?? context
        guard context <= ubatch && ubatch <= batch else {
            throw EmbeddingValidationError.invalidInput(
                statusCode: 400,
                detail: "Embeddings require ctx_size <= n_ubatch <= n_batch (got ctx_size=\(context), n_ubatch=\(ubatch), n_batch=\(batch))"
            )
        }
        if hasDraftOrSpec || hasExtraArgs {
            throw EmbeddingValidationError.invalidInput(
                statusCode: 400,
                detail: "Embedding models do not accept drafting or extra_args"
            )
        }
        return [
            "ctx_size": context,
            "n_batch": batch,
            "n_ubatch": ubatch,
        ]
    }

    /// Matryoshka dimension ladder for a model's native dimension.
    public static func supportedDimensions(forNativeDimension nativeDim: Int) -> [Int] {
        if nativeDim == gemma2NativeDimensions {
            return gemma2Dimensions
        }
        var ladder = [128, 256, 512].filter { $0 < nativeDim }
        if !ladder.contains(nativeDim) {
            ladder.append(nativeDim)
        }
        return ladder
    }

    /// Resolves `PromptCatalog` and prompt validation uniformly for both MLX and GGUF models.
    public static func resolvePrompt(
        for input: EmbeddingInput,
        catalog: PromptCatalog,
        itemIndex: Int = 0
    ) throws -> (prefix: String, promptName: String?) {
        if input.usesExplicitPromptName {
            guard let requested = input.promptName else {
                return ("", nil)
            }
            guard let value = catalog.prompts[requested] else {
                throw EmbeddingValidationError.unknownPromptName(
                    itemIndex: itemIndex,
                    promptName: requested,
                    available: catalog.prompts.keys.sorted()
                )
            }
            return (value, requested)
        }
        guard input.task != .none, input.hasNonWhitespaceText else {
            return ("", nil)
        }
        let keys: [String] = switch input.task {
        case .query: ["query", "Retrieval-query"]
        case .retrievalQuery: ["Retrieval-query", "query"]
        case .document: ["document", "Retrieval-document"]
        case .retrievalDocument: ["Retrieval-document", "document"]
        case .questionAnswering: ["QuestionAnswering", "question_answering"]
        case .factVerification: ["FactVerification", "fact_verification", "fact_checking"]
        case .sts: ["STS", "sts", "sentence_similarity"]
        case .clustering: ["Clustering", "clustering"]
        case .classification: ["Classification", "classification"]
        case .codeRetrieval: ["InstructionRetrieval", "code_retrieval", "code"]
        case .none: []
        }
        if let matchedKey = keys.first(where: { catalog.prompts[$0] != nil }),
           let prefix = catalog.prompts[matchedKey] {
            if (input.task == .document || input.task == .retrievalDocument),
               let rawTitle = input.title?.split(whereSeparator: \.isWhitespace).joined(separator: " "),
               !rawTitle.isEmpty,
               prefix.hasPrefix("title: none | text:") {
                return ("title: \(rawTitle) | text: ", matchedKey)
            }
            return (prefix, matchedKey)
        }
        if input.task == .document || input.task == .retrievalDocument {
            let rawTitle = input.title?.split(whereSeparator: \.isWhitespace).joined(separator: " ")
            if let rawTitle, !rawTitle.isEmpty {
                return ("title: \(rawTitle) | text: ", input.task.promptName)
            }
        }
        return (input.task.prefixText, input.task.promptName)
    }

    // MARK: - Private GGUF Helpers

    private static func deriveGGUFFacts(
        modelPath: URL,
        mmprojPath: URL?,
        computePlacement: String?,
        ctxSize: Int?,
        nBatch: Int?,
        nUBatch: Int?
    ) -> EngineEmbeddingFacts? {
        guard let ggufFile = resolvePrimaryGGUFFile(at: modelPath),
              let header = GGUFHeaderReader.readFileHeader(at: ggufFile),
              header.isEmbedding
        else {
            return nil
        }

        let arch = header.architecture ?? ""
        let isGemma2 = (arch == "gemma-embedding2")
        let rootDir = ggufFile.deletingLastPathComponent()

        let resolvedMMProj = mmprojPath ?? resolveMMProjFile(in: rootDir)
        let mmprojHeader = resolvedMMProj.flatMap { GGUFHeaderReader.readFileHeader(at: $0) }

        var modalities = ["text"]
        let hasVision = (mmprojHeader?.hasVisionEncoder == true) || header.hasVisionEncoder
        let hasAudio = (mmprojHeader?.hasAudioEncoder == true) || header.hasAudioEncoder
        if hasVision {
            modalities.append(contentsOf: ["image", "video"])
        }
        if hasAudio {
            modalities.append("audio")
        }

        let rawEmbLen: UInt64? = {
            if !arch.isEmpty, let v = header.uintMetadata["\(arch).embedding_length"] {
                return v
            }
            return header.uintMetadata.first(where: { $0.key.hasSuffix(".embedding_length") })?.value
        }()
        let nativeDim = rawEmbLen.map(Int.init) ?? (isGemma2 ? gemma2NativeDimensions : 768)
        let dimensions = isGemma2 ? gemma2Dimensions : supportedDimensions(forNativeDimension: nativeDim)
        let headerCtx = header.contextLength ?? gemma2MaxInputTokens
        let maxInputTokens = isGemma2 ? min(headerCtx, gemma2MaxInputTokens) : headerCtx

        let catalog = PromptCatalog.load(from: rootDir)
        let loadProfile = try? ggufEmbeddingProfile(
            ctxSize: ctxSize,
            nBatch: nBatch,
            nUBatch: nUBatch,
            maxContext: maxInputTokens
        )

        return EngineEmbeddingFacts(
            modalities: modalities,
            dimensions: dimensions,
            nativeDimensions: nativeDim,
            maxInputTokens: maxInputTokens,
            prompts: catalog.prompts,
            normalization: "l2",
            videoAudio: hasAudio ? "separate_audio_block" : nil,
            videoWindows: nil,
            limits: EngineEmbeddingLimits(
                maxBatchSize: EmbeddingRequestLimits.maxItems,
                maxTotalTokens: maxInputTokens,
                maxImagesPerItem: hasVision ? 8 : 0,
                maxAudioSeconds: hasAudio ? ArtifactProcessor.audioWindowSeconds : 0
            ),
            provenance: "model_files_and_architecture_contract",
            loadProfile: loadProfile,
            computePlacement: computePlacement
        )
    }

    static func resolvePrimaryGGUFFile(at url: URL) -> URL? {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            return nil
        }
        if !isDir.boolValue {
            return url.pathExtension.lowercased() == "gguf" ? url : nil
        }
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        return entries
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            .first(where: { !$0.lastPathComponent.lowercased().contains("mmproj") })
    }

    static func resolveMMProjFile(in directory: URL) -> URL? {
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        return entries
            .filter {
                $0.pathExtension.lowercased() == "gguf"
                    && $0.lastPathComponent.lowercased().contains("mmproj")
            }
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            .first
    }
}
