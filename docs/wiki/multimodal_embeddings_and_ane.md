# 4. Multimodal Embeddings, ANE Towers & Matryoshka Store

`HeylookEmbeddings` (`Sources/HeylookEmbeddings/`) delivers native multimodal embeddings across text, code, images, PDFs, audio, and video windows using **EmbeddingGemma 2**, optional compiled CoreML ANE towers (`.mlmodelc`), GGUF embedding subprocesses (`llama-server --embeddings`), and an Accelerate/`vDSP`-backed SQLite WAL `VectorStore`.

---

## 4.1 Truthful Multimodal & GGUF Embedding Engines

1. **`MLXSwiftLMEmbeddingGemma2Engine`** (`Sources/HeylookEmbeddings/Engine/NativeMLXEmbeddingGemma2.swift`):
   - **1,024-Token Padded Batch Text Embedding**: Batches text-only inputs into padded forward passes capped at `textBatchTokenLimit = 1_024` via `makeLengthBatches(lengths:limit:)`, reducing Swift/Metal kernel dispatch overhead across bulk corpus and lab sweeps.
   - **True `hybrid_ane` ANE->GPU Bridge (No `TextANE.mlmodelc` Needed)**: Under `.hybridANETowers` (`compute=hybrid_ane`), `resolvedHybridModel` loads MLX weights *excluding* ANE-offloaded vision/audio towers, runs `VisionEncoder.mlmodelc` / `AudioEncoder.mlmodelc` on Apple Neural Engine via **`ANECoreMLRuntime`** to extract `softTokens`, and injects those `[Float]` soft tokens directly into GPU **`MLXVLM.EmbeddingGemma2.embed(inputIds:attentionMask:softTokens:)`**. Only `.allANE` requires a compiled `TextANE.mlmodelc` bundle (using sequence length buckets `[64, 320, 512, 768, 1024]`).
   - **Consecutive Single-Window Video Decoding**: When a request supplies `windows.seconds` (including clips `<= 16s`), **`WindowPlanner.consecutiveRanges`** partitions `[0, duration]` into non-overlapping consecutive windows and **`ArtifactProcessor.decodeVideoWindowFrames`** decodes **one window at a time** (bounded peak memory) using uniform FPS (`pickFPS`) or scene-change (`pickContent`) sampling, optionally extracting matched window audio.
2. **`GGUFEmbeddingEngine`** (`Sources/HeylookEmbeddings/Engine/GGUFEmbeddingEngine.swift`):
   - Launches an isolated `llama-server --embeddings --pooling <mode>` subprocess scoped by `EmbeddingModelFacts.ggufEmbeddingProfile` (`ctx_size <= n_ubatch <= n_batch` enforced safely).
   - Truthfully rejects video `windows` payloads with HTTP **`400`** because GGUF `llama-server --embeddings` supports text-only embedding inputs.
3. **`HybridLocalEmbeddingEngine`**:
   - Coordinates ANE-tower vs MLX fallback routing strictly across real checkpoints; synthetic fallback flags and `DeterministicHashEmbeddingEngine` have been completely removed from production `Sources/`.

---

## 4.2 Dynamic Task Prompts & Contract Facts (`EmbeddingModelFacts`)

Defined in `Sources/HeylookEmbeddings/Engine/EmbeddingModelFacts.swift`:

- **`PromptCatalog`**: Reads `config_sentence_transformers.json` directly from the model directory to discover `prompts` mappings and `default_prompt_name` (for example `Retrieval-query`, `Retrieval-document`, `STS`, `Clustering`, `Classification`), validating API `prompt_name` values against real checkpoint definitions instead of hardcoded task tables.
- **`EngineEmbeddingFacts`**: `EmbeddingModelFacts.deriveFacts(...)` populates truthful modalities (`text`, `image`, `audio`, `video` when supported), `native_dimensions`, Matryoshka projections (`dimensions`), `max_input_tokens`, `prompts`, `normalization`, `video_audio`, `video_windows`, `limits`, `provenance`, `load_profile`, and `compute_placement` on `/v1/models` and `/v1/models/*id`.

---

## 4.3 `ANECoreMLRuntime`, Limits & Media Preprocessing

- **Vision Buckets (`ANECoreMLRuntime` in `ModelComputeOptions.swift`)**: Aspect-preserving upright EXIF-oriented `CGImage` rendering into compiled CoreML Vision buckets (`[480, 672]`, `[672, 480]`, `[528, 528]` for video budgets; `[672, 960]`, `[960, 672]`, `[768, 768]` for standalone images) with sRGB `[1, 3, H, W]` `[0, 1]` normalization.
- **Audio Tower**: 16 kHz mono PCM resampled into `30s` windows (`750`-token output ceiling after `conv0`/`conv1` subsampling).
- **Strict Request Limits (`EmbeddingRequestLimits` in `EmbeddingEngine.swift`)**:
  - Max **`16`** items per request, **`32`** content blocks per item, **`131,072`** text characters per text block, **`32 MiB`** per decoded image/audio blob, and **`256 MiB`** aggregate video payload budget per request, enforced via `EmbeddingValidationError` (`400` / `413`).
- **Indexing Pipelines (`ArtifactProcessor` + `WindowPlanner` + `FolderWatcher` + `IndexCoordinator`)**:
  - Indexes folders recursively across text/code (`ArtifactSpan`), PDFs (`PDFKit` per-page chunking), images, audio (`30s` chunks), and video (`32s` windows with `16s` stride for corpus indexing; consecutive `windows.seconds` windows for `/v1/embeddings` video window queries).

---

## 4.4 Matryoshka `VectorStore` (`Store/VectorStore.swift`)

- **Nested Representations**: Stores full native float vectors (`768d` for EmbeddingGemma 2) in SQLite `WAL` (`embeddings`, `artifacts`, `spans`) and projects dynamically to L2-normalized **`768d` / `512d` / `256d` / `128d`** Matryoshka prefixes via `Accelerate`/`vDSP` (`Matryoshka.project`).
- **Truthful Model Invalidation**: Records the indexing `model_id` per indexed artifact/span so switching embedding models never silently compares vectors across incompatible latent spaces.
