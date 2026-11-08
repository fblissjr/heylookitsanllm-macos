# Multimodal Embeddings, ANE Towers & Matryoshka Vector Store

Covers `Sources/HeylookEmbeddings/` (`Domain/`, `Engine/`, `Pipeline/`, `Store/`, and `IndexCoordinator.swift`).

---

## 1. Extensible `Artifact` & `ArtifactSpan` Model (`Artifact.swift`)

Every indexed file is represented as a parent `Artifact` (`ArtifactKind`: `.video`, `.image`, `.audio`, `.text`, `.code`, `.document`) paired with one or more `EmbeddedChunk` rows carrying a modality-specific `ArtifactSpan`:
- `.whole`: Full image, short text file, or single-span item.
- `.temporal(start: Double, end: Double, frameTimes: [Double])`: Overlapping `32s` video windows (`16s` stride) with exact extracted keyframe timestamps, or `30s` audio windows (`15s` stride).
- `.lines(startLine: Int, endLine: Int)`: Source code and markdown line ranges.
- `.pages(startPage: Int, endPage: Int)`: `PDFKit` page ranges.

> **Parent-Artifact Deduplication Rule**: `VectorStore.search` groups matching chunk rows by `artifact.id`, never returning duplicate cards for multiple windows of the same video or pages of the same PDF. Each `SearchHit` carries `bestScore`, `bestSpan`, and ordered `spanMatches` (with `scoresByDimension` across `768d`, `512d`, `256d`, and `128d`).

---

## 2. Matryoshka Representation Learning (MRL) Without Re-Indexing (`VectorStore.swift`)

- `VectorStore` stores full **768-dimensional** `Float32` embeddings once in SQLite (`WAL` mode).
- At query time, `Matryoshka.sliceAndNormalize` slices the first $d \in \{768, 512, 256, 128\}$ dimensions and L2-normalizes in hardware using Apple `Accelerate` (`vDSP_svesq` + `vDSP_vsdiv` + `vDSP_dotpr`).
- **Truthful Model-ID Invalidation (`VectorStore.isUpToDate`)**:
  `isUpToDate` tracks `(byte_count, modified_ms, window_sampling, model_id)`.
  - Real `EmbeddingGemma 2` engines (`MLXSwiftLMEmbeddingGemma2Engine` and `HybridLocalEmbeddingEngine` when compiled `.mlmodelc` towers are present) report `modelID = "embeddinggemma-2-MLX-BF16"`.
  - When neither `MLXVLM` nor compiled `.mlmodelc` ANE bundles are present on disk, `DeterministicHashEmbeddingEngine` and `HybridLocalEmbeddingEngine` report `modelID = "deterministic-hash-768d"`, ensuring that installing real weights automatically invalidates placeholder hash rows on the next index pass.

---

## 3. Per-Tower Compute Placement & `ANECoreMLRuntime` (`ModelComputeOptions.swift` & `NativeMLXEmbeddingGemma2.swift`)

`ModelComputeOptions` allows routing the 3 towers of `EmbeddingGemma 2` independently across **CPU + Neural Engine (`.cpuAndNeuralEngine`)**, **MLX GPU (`.mlxGPU`)**, **CoreML GPU (`.cpuAndGPU`)**, or **CPU Only (`.cpuOnly`)**:

| Component (`EmbeddingComponent`) | Compiled Bundle | ANE Op Residency | Est. Memory |
| :--- | :--- | :--- | :--- |
| **Vision Encoder (`visionEncoder`)** | `EmbeddingGemma2VisionANE.mlmodelc` | `100%` ANE | `~335 MB` |
| **Audio Conformer (`audioEncoder`)** | `EmbeddingGemma2AudioANE.mlmodelc` | `99%` ANE | `~611 MB` |
| **Text Backbone + Projection (`textBackbone`)** | `EmbeddingGemma2TextANE.mlmodelc` + `embed_tokens_fp16.bin` | `100%` ANE | `~542 MB` |

### ANE Execution Path (`HybridLocalEmbeddingEngine` & `ANECoreMLRuntime`)
- When compiled `.mlmodelc` bundles exist in the checkpoint directory (`hasCompiledANETowers == true`), `HybridLocalEmbeddingEngine` decodes images via `CGImageSource`, computes 16 kHz 128-bin HTK log-mel spectrograms via `ANECoreMLRuntime.computeAudioLogMel`, and runs `aneRuntime.encodeVision`, `aneRuntime.encodeAudio`, and `aneRuntime.encodeTextAndFuse` directly on the Neural Engine.
- **FP16 Overflow Mitigation (`ANESafeRMSNorm`)**: Because the Apple Neural Engine executes strictly in IEEE 754 `FP16` (`max = 65,504`), `scripts/export_ane_coreml.py` pre-scales hidden states by $1/16$ before squaring inside `RMSNorm` layers so large Gemma residual activations never overflow to `NaN` on ANE hardware.

---

## 4. Video & Audio Windowing Pipeline (`ArtifactProcessor.swift`)

- **Duration-Scaled Frame Budgets (`WindowPlanner.pickContent`)**:
  Scales frame selection to the window duration (`min(maxFrames, max(2, ceil((hi - lo) * fps)))`) and falls back to all candidate frames when `< 3` anchors exist, preventing short clips or trailing windows from feeding 4x dense frames into a 1 fps vision processor.
- **Upright Portrait Frame Decoding**:
  Every `AVAssetImageGenerator` instance sets `appliesPreferredTrackTransform = true` so portrait iPhone clips are rotated upright before vision encoding.
- **Soundtrack Indexing (`loadAudioFromVideo`)**:
  When enabled, `IndexCoordinator` extracts both `.temporal` visual keyframe windows and `16 kHz` mono PCM audio windows (`30s` window, `15s` stride) for the same video artifact.
