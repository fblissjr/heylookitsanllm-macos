# 3. Providers, Specialization & Heterogeneous Compute

`HeylookEngines` (`Sources/HeylookEngines/`) coordinates four inference backends (`mlx`, `gguf`, `coreai`, `remote`), signature-based model specializations, decoupled ANE speech transcription, and Apple Silicon unified-memory fitting across dual hardware lanes (`GenerationGate` + `ANELane`).

---

## 3.1 `InferenceProvider` Contract & `ComputeScheduler`

- **`InferenceProvider`** (`Sources/HeylookEngines/ComputeScheduler.swift` + `Specialization/ModelSpecialization.swift`):
  - Core lifecycle & generation: `load()`, `unload()`, `generate(messages:settings:requestID:...)`.
  - Protocol requirements dispatched dynamically across every backend (`mlx`, `gguf`, `coreai`, `remote`): `planImages(sizes:)`, `renderPromptPreview(request:)`, `observedContractState()`, and `warmUp1Token()`.
  - Future runtimes (such as a planned `LiteRT` backend in Post-Parity Horizons) plug in cleanly behind `InferenceProvider` + `ModelSpecialization` without altering `ComputeScheduler`, `WorkbenchService`, or wire types.
- **Dual-Lane `ComputeScheduler`**:
  - **`GenerationGate`**: Serializes GPU weight generation behind a bounded FIFO queue (`maxQueueDepth`, default `2`, surfacing `503 model_overloaded` + `Retry-After: 1` when saturated).
  - **`ANELane`**: Tracks active Apple Neural Engine workloads independently so ANE `.mlmodelc`/CoreAI/ASR tasks run concurrently with GPU weight generation without stalling `GenerationGate`.
  - **Cross-Lane `withGPUEmbeddingLease`**: Coordinates GPU embedding bursts cleanly around loaded generation models and live memory pressure eviction (`evictionPlan`), respecting pinned models and returning `409 model_busy` when attempting to unload an actively generating model (`tryUnloadModel`).
  - **`PrefillProgressBridge`** (`@TaskLocal`): Publishes prefill progress events upward to SSE `heylook_progress` frames during long multimodal/prompt prefill.

---

## 3.2 Signature-Based `ModelSpecialization` (`ModelSpecialization.swift`)

Instead of branching on model-ID strings inside provider loops, `ModelSpecializationRegistry` inspects real checkpoint metadata via **`CheckpointSignature`** (`config.json`, `preprocessor_config.json`, GGUF header metadata, and paired drafter `config.json`) to bind a declarative **`ModelSpecialization`**:

1. **`Gemma4Specialization`**:
   - Vision alignment: enforces **`48px`** (`16 * 3` patch/pooling) edge rounding in `resizeCIImageIfNeeded` (replacing legacy fixed `28px` assumptions).
   - Speculative pairing: validates `gemma4_assistant` `backbone_hidden_size == target.hidden_size` so mismatched draft heads never load silently.
   - Multimodal speculative support: permits Leviathan MTP speculative decoding after multimodal prefill.
   - GGUF arguments: enables `--ctx-checkpoints 32 --checkpoint-min-step 256` and **suppresses `--cache-reuse`** (which corrupts SWA KV slots).
2. **`HybridRecurrentSWASpecialization`**:
   - Matches hybrid linear-attention / sliding-window-attention / state-space architectures (`qwen3_5`, `lfm2`, `granite`, GGUF sliding-window/recurrent headers).
   - Suppresses incompatible KV shifting (`--cache-reuse` excluded) and enables checkpointed slot restoration (`--ctx-checkpoints 32 --checkpoint-min-step 256`).
3. **`StandardAttentionSpecialization`**:
   - Full-attention dense and MoE checkpoints where standard prefix KV shifting and `--cache-reuse` are valid.

---

## 3.3 `MLXSwiftProvider` & `MLXRuntimeSupport`

Built under `HEYLOOK_ENABLE_NATIVE_MLX=1` (`mlx-swift`, `mlx-swift-lm`, `swift-transformers`), `MLXSwiftProvider` delegates low-level KV, vision, speculative, and template mechanics to `Sources/HeylookEngines/Providers/MLXRuntimeSupport.swift`:

- **`PreRenderedVLMTokenizerProxy`**: Wraps rendered Jinja2 chat prompts so upstream `mlx-swift-lm` VLM processors preserve exact prompt boundaries and multimodal placeholders instead of re-applying or stripping template structure.
- **`VisionFeatureCache`**: Thread-safe LRU cache (max **16 entries / 512 MB**) keyed by media digest and resolved geometry, eliminating redundant vision encoder passes across turns when paired with `resizeCIImageIfNeeded` (`alignmentMultiplePx`).
- **`APCSnapshotCache` & `HeylookStagedRotatingKVCache`**: Prefix KV snapshot cache (max **16 entries**) keyed by `(tokenIDs, mediaSalt)` with automatic memory-pressure pruning. When images differ (`mediaSalt`), visual KV entries never collide with stale text-only prefixes.
- **`ThinkingControlLogitProcessor`**: Enforces thinking budgets and open/close reasoning transitions at logit sampling time on GPU (`thinking_budget`, forced close tokens).
- **`LeviathanMTPTokenIterator`**: Lossless speculative token verification using exact `p_target / q_draft` rejection sampling + normalized residual resampling when `temperature > 0` (not greedy-only argmax mismatch), paired with cloned penalty `draftProcessor`, `draft_max_context` passthrough, and 1-token multimodal prefill priming where permitted by `ModelSpecialization`.
- **`MLXModelConverter`**: Pure Swift checkpoint quantization (`MLXLMCommon.convert`) supporting `affine`, `mxfp4`, `mxfp8`, `nvfp4`, and `q4_zero` calibration directly from CLI (`heylook-cli convert-mlx`) or the Models sheet.

---

## 3.4 `RAMFitCalculator` (`RAMFitCalculator.swift`)

`RAMFitCalculator` combines Mach host memory statistics (`host_statistics64`), Metal GPU device limits (`recommendedMaxWorkingSetSize`, `maxBufferLength`), and macOS `sysctl iogpu.wired_limit_mb` to evaluate pre-load safety:

- **Headroom Thresholds**: Enforces `thinHeadroomGB = 24.0` warning headroom (`RAMFitEvaluator` additionally applies `thinKVHeadroomGB = 4.0` for KV slack) and emits `sysctlSuggestMB` recommendations when wired GPU ceiling tuning helps.
- **Architecture-Aware KV Geometry**: Uses checkpoint layer-type maps (`full_attention` vs SWA vs recurrent/SSM state sizes) and KV quantization scheme (`cache_type_k` / `cache_type_v` / `kv_bits`), automatically evaluating `droppedDrafterForMemory` if a paired speculative drafter would breach headroom.
- **Provider-Truthful Verdicts**: Working-set overrun produces a **hard `fail` on MLX** (preventing Metal allocation panics/paging storms) vs a **`warn` on GGUF** (where mmap/host-RAM offloading remains viable).

---

## 3.5 `GGUFSubprocessProvider` & `StreamingPipeline`

- **Process Isolation (`GGUFSubprocessProvider` + `SubprocessSupport`)**:
  - Spawns one isolated `llama-server` subprocess per loaded GGUF model via `GGUFPortPool` and `SubprocessLifecycleRegistry`, automatically pairing discovered `mmproj*.gguf` projectors and validated drafters.
  - `GGUFSpawnConfig.autoSized` derives flags from `ModelSpecialization`: omits `--cache-reuse` on SWA/hybrid/Gemma 4 models, enables `--ctx-checkpoints 32 --checkpoint-min-step 256` on hybrid/SWA slots, and passes `--reasoning-preserve`.
  - Queries live `/apply-template`, `/props`, and `/v1/chat/completions/input_tokens` (used by `/v1/models/*id/image-plan`).
- **Reply-Only Stop Cutting & Mid-Stream Control**:
  - `buildLlamaServerArgs` intentionally omits user `stop` sequences from the upstream `llama-server` JSON body so the model is never prematurely aborted while inside `<think>...</think>`. Instead, **`StopSequenceFilter`** in `StreamingPipeline.swift` buffers partial suffix matches and cuts **only visible reply `delta.text`**.
  - **`CompletionIDBox`** captures the SSE `"id"` emitted by `llama-server` so `POST /v1/requests/{id}/reasoning-end` can forward `/v1/chat/completions/control` (`reasoning_end`) to interrupt reasoning cleanly and transition to final output.
  - **`PrefixCacheTracker`** reports truthful `0` prefix hits when no tokenizer count is available rather than inventing cache savings.

---

## 3.6 `CoreAIProvider` & `SpeechTranscriptionService`

- **`SpeechTranscriptionService`** (`Sources/HeylookEngines/Speech/SpeechTranscriptionService.swift`): Decouples on-device ANE speech recognition from the LLM generation queue, dispatching `CoreAISpeech.SpeechRecognitionModel` (Parakeet TDT / Whisper bundles) and Apple's `SpeechAnalyzer` / `SpeechTranscriber` behind availability checks.
- **RIFF Subchunk Walking (`decodeRIFFChunksMonoPCM`)**: Walks WAV headers chunk-by-chunk rather than assuming a 44-byte header, safely skipping `FLLR`, `JUNK`, `LIST`, or metadata subchunks to reach exact `fmt ` and `data` audio payloads before 16 kHz mono resampling.
- **`CoreAIProvider`**: Executes `.aimodel` bundles under `HEYLOOK_ENABLE_COREAI=1` on ANE/CoreML and delegates audio transcription requests directly to `SpeechTranscriptionService`.
