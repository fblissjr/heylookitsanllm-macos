# Sharp Edges & Incident Rationales (`docs/wiki/sharp_edges.md`)

Every non-obvious mechanism in `heylookitsanllm-macos` exists because a real hardware, subprocess, or protocol edge case bit in practice. Read the relevant section before changing provider, template, or embedding code.

---

## 1. Truthful Runtime vs. Silent Synthetic Fallbacks

- **What bit**: Early scaffolding defaulted `allowSyntheticFallback: Bool = true` on `MLXSwiftProvider`, `CoreAIProvider`, and `GGUFSubprocessProvider`, and defaulted `DeterministicHashEmbeddingEngine.modelID` to `"embeddinggemma-2-MLX-BF16"`. When a checkpoint path was misconfigured or `swift build` ran without Metal shaders, the server returned HTTP `200` with synthetic `[model] Processed: ...` text or cached deterministic hash vectors under the real `embeddinggemma-2-MLX-BF16` model ID—poisoning `VectorStore.isUpToDate` even after real weights were installed.
- **The mechanism**:
  1. All production providers (`MLXSwiftProvider`, `CoreAIProvider`, `GGUFSubprocessProvider`, and `ProviderFactory.makeProvider`) default `allowSyntheticFallback = false` and throw `ProviderExecutionError` (`checkpointNotFound`, `binaryNotFound`, `runtimeUnavailable`).
  2. `DeterministicHashEmbeddingEngine` and `HybridLocalEmbeddingEngine` (when compiled `.mlmodelc` towers are absent) report `modelID = "deterministic-hash-768d"`, never `"embeddinggemma-2-MLX-BF16"`.
  3. Neither `HeylookWorkbenchViewModel.bootstrap()` nor `heylook-cli` auto-seeds fake models or fake corpus rows on startup; synthetic demo data is loaded only when explicitly requested via `loadSampleDemoWorkspace()` or unit test fixtures (`allowSyntheticFallback: true`).

---

## 2. GGUF (`llama-server`) Subprocess & Vision Sharp Edges

- **Non-Causal Vision Projector Batch Abort (`clip.projector_type`)**:
  - **What bit**: Non-causal vision encoders in `llama.cpp` (`gemma4v`, `qwen2vl`, `qwen2.5vl`, `qwen3vl`, `qwen3_5`, `pixtral`, `internvl`, `minicpmv`) encode an entire image in a single non-causal attention pass (`GGML_ASSERT(n_tokens <= n_ubatch)`). If `--image-max-tokens` exceeds `-ub` (`ubatchSize`), `llama-server` aborts inside `llama_context::decode`.
  - **The mechanism**: `GGUFHeaderReader` inspects `clip.projector_type` in `mmproj-*.gguf`, sets `hasNonCausalVisionProjector = true` for non-causal projector families, and `GGUFSpawnConfig.autoSized` clamps `imageMaxTokens = min(requestedImageMaxTokens, ubatchSize)`.
- **Inherited `LLAMA_ARG_*` Environment Pollution**:
  - **What bit**: `llama-server` parses every `LLAMA_ARG_*` environment variable as a CLI flag. If the parent shell sets `LLAMA_ARG_LOG_FILE` or `LLAMA_ARG_N_PARALLEL`, it silently overrides `GGUFSpawnConfig`.
  - **The mechanism**: `GGUFSubprocessProvider.childEnvironment(from:)` strips all `LLAMA_ARG_*` keys and sets `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=604800` so Metal weight buffers stay wired across idle periods.
- **Mid-Stream `llama-server` Error Frames**:
  - **What bit**: When `llama-server` encounters a mid-stream KV slot exhaustion or decode failure after returning HTTP `200`, it emits `data: {"error": {"message": "..."}}` over SSE. Ignoring non-`choices` frames caused the stream to end silently with `stop_reason = "end_turn"`.
  - **The mechanism**: `GGUFSubprocessProvider.parseOpenAISSEDataLine` parses top-level `"error"` objects into `OpenAISSEDelta.errorMessage` and throws `ProviderExecutionError.runtimeUnavailable`.

---

## 3. MLX Swift & Reasoning Parser Sharp Edges

- **Dual Reasoning Parser Conflict in `mlx-swift-lm`**:
  - **What bit**: `MLXLMCommon` includes its own basic `ReasoningConfig` tag splitter on some `ModelConfiguration` types, which conflicts with `HeylookCore`'s 4-protocol `ReasoningParser` (`.tagPair`, `.gemmaChannel`, `.harmonyChannel`, `.recipientChannel`) and mid-stream `forceEndReasoning()`.
  - **The mechanism**: `MLXSwiftProvider` sets `config.reasoningConfig = nil` after loading the container so `StreamingPipeline.transformEventStream` is the single authoritative reasoning parser across all providers.
- **Vision-Config Stubs on Text-Only Checkpoints**:
  - **What bit**: Some Hugging Face text-only conversions of multimodal families (e.g., `Qwen3.5` or `Gemma 4` text-only variants) leave a stub `vision_config` in `config.json`. Calling `VLMModelFactory.shared` fails when vision tower weights are absent.
  - **The mechanism**: `MLXSwiftProvider.ensureContainerLoaded()` attempts `VLMModelFactory.shared` first when `supportsVision.value` is `true`, and automatically falls back to `LLMModelFactory.shared` while updating `supportsVision` to `false` (`.observed` provenance).
- **Clean `MessageRecord.content` for KV Prefix Reuse**:
  - **What bit**: Prepending synthetic `[IMAGE: filename.png]` strings to `MessageRecord.content` in `HeylookWorkbenchViewModel.sendChatPrompt()` mutated the text history stored in `SQLiteStore`, causing `PrefixCacheTracker` to report `history_diverged` on subsequent turns.
  - **The mechanism**: `MessageRecord.content` stores the user's clean `promptText` (falling back to attachment filenames only when `promptText` is empty), while media payloads travel in `MessageTurn.attachments`.

---

## 4. Apple Neural Engine (ANE) & Multimodal Media Sharp Edges

- **FP16 Overflow (`65,504`) in Gemma RMSNorm Layers on ANE**:
  - **What bit**: The Apple Neural Engine executes `MLModel` graphs in IEEE 754 `FP16`. Squaring hidden states (`x * x`) inside standard `RMSNorm` layers overflows `65,504` on later Gemma 2/3/4 layers, producing `NaN` embeddings on ANE while passing on CPU/GPU (`FP32`/`BF16`).
  - **The mechanism**: `scripts/export_ane_coreml.py` wraps normalization layers in `ANESafeRMSNorm`, scaling inputs by $1/16$ before squaring (`(x * 0.0625)^2`), which is mathematically invariant after normalization and keeps all intermediate activations well within `FP16` bounds.
- **Portrait iPhone Video Orientation**:
  - **What bit**: `AVAssetImageGenerator` defaults `appliesPreferredTrackTransform = false`, decoding portrait iPhone videos sideways (`90°` rotated) and degrading vision embedding recall.
  - **The mechanism**: Every `AVAssetImageGenerator` in `ArtifactProcessor` sets `appliesPreferredTrackTransform = true`.
