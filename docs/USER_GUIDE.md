# Heylook Native Workbench — Operator & API User Guide

Welcome to **`heylookitsanllm-macos`** (`heylook`) — a native **Swift 6.4 (`v6` language mode)** multiplatform (`macOS 27.2+` and `iOS 27.2+`, with `27.3+` SDK features guarded behind `#available(macOS 27.3, iOS 27.3, *)`) personal AI inference workbench, multimodal media/embedding studio, and embedded HTTP/SSE server built specifically for Apple Silicon.

---

## 1. Quick Start & Launching

### Build & Run Requirements
- **Hardware**: Apple Silicon Mac (`M1` / `M2` / `M3` / `M4` / `Ultra`) and/or `iOS 27.2+` / `iPadOS 27.2+` paired client.
- **OS & Toolchain**: `macOS 27.2+` (`27.2.1` supported; optional `27.3+` enhancements guarded cleanly) and Xcode 27.x (Swift 6.4 toolchain).

### Building & Launching from CLI / Terminal

```bash
# Fast SwiftPM unit + wire contract test suite (pure Swift logic, SQLite, HTTP/SSE)
swift test

# Full native Xcode build (compiles Metal kernels -> default.metallib, links MLX/CoreAI packages & TCC signing)
./scripts/build_native.sh build
./scripts/build_native.sh app

# Run the unified CLI against local checkpoints
./scripts/build_native.sh cli doctor
./scripts/build_native.sh cli models
./scripts/build_native.sh cli serve 8000
```

> [!TIP]
> Always use `./scripts/build_native.sh` when exercising GPU Metal inference (`MLXSwiftProvider`) or Apple Neural Engine bundles (`CoreAIProvider`, `SpeechTranscriptionService`). Raw `swift build` does not compile Metal shaders into `default.metallib` or attach TCC-hardened app bundle entitlements.

---

## 2. Configuration (`heylook.toml`, `model.heylook.toml`, Env Vars)

### Root `heylook.toml`
By default Heylook reads `$HEYLOOK_CONFIG`, `./heylook.toml`, or `~/Library/Application Support/Heylook/heylook.toml`. An example layout:

```toml
default_model = "gemma-4-26b-a4b-it-mlx"
max_loaded_models = 3
max_queue_depth = 2
server_host = "127.0.0.1"      # Set "0.0.0.0" to expose LAN + Bonjour _heylook._tcp
server_port = 8000
allowed_hosts = ["localhost", "127.0.0.1"]
# admin_token = "replace-with-strong-secret"  # Optional X-Heylook-Admin-Token guard

[scan]
folders = ["./models/mlx", "./models/gguf", "./models/coreai"]
scan_interval_seconds = 900

[settings]
observability_level = "minimal"      # off | minimal | standard | debug
observability_retention_days = 7
mlx_cache_limit_gb = 8
```

### Per-Model Override Sidecars (`model.heylook.toml`)
Place overrides directly inside a checkpoint folder (or save them from the **Models (`⌘5`)** config sheet). **Blank or unset keys emit `null` and remove overrides cleanly** via `TOMLCodec.updateModelSidecar` — empty sidecar files are removed automatically so derived checkpoint defaults continue to apply.

```toml
# Example model.heylook.toml
ctx_size = 32768
flash_attn = "on"
image_max_tokens = 280
compute_unit = "mlx_gpu"
# Sampling overrides (layers above vendor generation_config.json):
temperature = 1.0
top_p = 0.95
```

### Chat Template Override Ladder (`inspectChatTemplate`)
Heylook resolves each model's Jinja2 chat template in strict priority without mutating vendor checkpoint files:
1. Inline or referenced `chat_template` / `chat_template_path` in `model.heylook.toml`
2. `chat_template.heylook.jinja` inside the model directory
3. `chat_template.jinja` sidecar (unless `use_sidecar_chat_template = false`)
4. `tokenizer_config.json` -> `chat_template.json` -> embedded GGUF `tokenizer.chat_template`

### Environment Variables

| Variable | Purpose |
| :--- | :--- |
| `HEYLOOK_CONFIG` | Explicit path to `heylook.toml`. |
| `HEYLOOK_MODELS_DIR` | Additional scan directory appended during model discovery. |
| `HEYLOOK_STORE_DB_PATH` | Override SQLite WAL database path (`heylook.sqlite`). |
| `HEYLOOK_VECTOR_DB_PATH` | Override multimodal vector store SQLite path (`heylook_vectors.sqlite`). |
| `HEYLOOK_DB_PATH` | Shared database override (`heylook.sqlite` or vector DB). |
| `HEYLOOK_READONLY_MODEL_CONFIG=1` | Disables local `heylook.toml`, `model.heylook.toml`, and `chat_template.heylook.jinja` writes (`409` on local mutation endpoints; paired remote servers remain unaffected). |
| `HEYLOOK_FRONTEND_DIR` | Optional directory of static web assets served from `/` with `ETag` + gzip (`zero SPA fallback`). |
| `HEYLOOK_ENABLE_NATIVE_MLX=1` | Links `mlx-swift`, `mlx-swift-lm`, and `swift-transformers` remote SwiftPM dependencies. |
| `HEYLOOK_ENABLE_COREAI=1` | Links optional `apple/coreai-models` package for `.aimodel` / `CoreAISpeech` execution. |

---

## 3. Six-Surface Workbench Guide

Navigate across all six native surfaces via sidebar or keyboard shortcuts **`⌘1` – `⌘6`**, press **`?`** (outside text fields) for the shortcut cheat sheet, and press **`Escape`** to dismiss open drawers, sheets, inline editors, or confirmations top-down before cancelling active generation.

### Surface 1 (`⌘1`): Chat (`ChatWorkbenchSurface`)
- **Inline Model Load / Reload Panel**: Click the status chip (`Load` / `Reload` / `Loaded`) in the top readout bar to expand `ModelLoadPanelView`. Configure GGUF `ctx_size` (auto ceiling, power-of-two steps, or custom tokens) and `flash_attn` (`default` / `on` / `off` / `auto`), view active `stale_reload_fields`, and click **`Load`** (1-token JIT warm), **`Reload (Apply Changes)`**, or **`Unload`**.
- **4-State System Prompt Chip**: Shows live prompt provenance in one pill:
  1. `No system prompt`
  2. `System prompt: <preset.name>`
  3. `<preset.name> (modified)` *(highlighted orange when edited away from the stamped preset)*
  4. `System prompt: custom`
- **Prompt Preview Drawer (`Eye` toggle)**: Renders exact formatted template tokens and byte counts live via `POST /v1/conversations/:id/prompt` without running inference.
- **Multimodal & Voice Composer**:
  - Drag-and-drop or paste images/audio (`image-plan` estimates visual token budget per checkpoint alignment, e.g., `48px` on Gemma 4; toggle **2048px shrink** per attachment).
  - Tap the microphone button to record speech and send hands-free via decoupled ANE ASR (`SpeechTranscriptionService` / `/v1/conversations/:id/voice-generate`).
- **Turn Actions & Smart Tail Follow**:
  - Edit any message inline (`Save in place`, `Save & Regenerate`, `Save & Continue`), continue partial assistant **or** user turns (including mid-`<think>` resume), interrupt reasoning cleanly (`End Thinking`), regenerate, clone up to any turn, or delete individual messages.
  - `ScrollViewReader` tracks scroll position (`followingTail`) and reveals a floating **`Jump to latest`** pill when you scroll upward during active generation.

### Surface 2 (`⌘2`): Notebook (`NotebookSurface`)
- Long-form completion & drafting pad using `CursorAwareTextEditor` with per-document model picker, preset menu, collapsible system prompt bar, streaming insertion/continuation, and automatic `@Environment(\.scenePhase)` save flushing.

### Surface 3 (`⌘3`): Presets (`PresetsSurface`)
- Starts cleanly unseeded (no built-in sample clutter).
- Create, search, edit, and preview sparse sampler presets under **"Preset is a Copy"** + **"Null-Means-Cascade"**:
  - All 13 optional controls (`SamplerCascade.SamplerFieldKey`) leave untouched fields as `nil` (`null`), inheriting from `model.heylook.toml` -> `generation_config.json` -> builtin defaults (`temperature = 1.0`, `top_p = 0.95`, `max_tokens = 16384`).
  - Choose scope **`sampler`** (parameters only) or **`full`** (parameters + system prompt).
  - Built-in optimistic locking (`Reload Latest Copy` prompt on `expectedUpdatedUnixMs` conflict) and live compatibility preview against any generation model.

### Surface 4 (`⌘4`): Embeddings & ANE (`EmbeddingsStudioSurface`)
Organized into three mode tabs (`Compare | Embedding Lab | Media Corpus (mbed)`):
1. **Compare**:
   - Compose multi-block items mixing text, images, audio, and video.
   - For video blocks, select `Whole video`, `Windows (uniform FPS)`, or `Windows (scene change)` and optional `windows.seconds` chunk duration.
   - Run cross-similarity matrices across Matryoshka projections (`768d`, `512d`, `256d`, `128d`), inspect pairwise vector tails, and click any entry in **Closest video windows to Item 1** to open **`CompareVideoWindowPreviewSheet`** (native `AVPlayer` bounded to window `[start, end]` with clickable `frame_times` seek pills).
2. **Embedding Lab**:
   - Import external `set.json` multimodal benchmarks or load the built-in 8-case CoreGraphics multimodal test set.
   - Queue multi-configuration sweeps across models, Matryoshka dimensions, `prompt_name` on/off ablations (`config_sentence_transformers.json`), and compute lanes (`all_gpu` vs `hybrid_ane` vs `all_ane`), inspect Recall@1/MRR tables, and export JSON results.
3. **Media Corpus (`mbed`)**:
   - Optional local corpus indexing (`FolderWatcher` + `IndexCoordinator`) supporting text, source code, multi-page PDFs (`PDFKit`), images, `30s` audio windows, and `32s`/`16s`-stride video indexing with instant cosine retrieval.

### Surface 5 (`⌘5`): Models (`ModelsInspectorSurface`)
- **Watch Folders (`heylook.toml [scan]`)**: Add/remove directories and rescan MLX, GGUF, CoreAI `.aimodel`, and ANE speech recognizer bundles. When no models are registered, the empty state points cleanly to Watch Folders or Bonjour peer pairing without injecting synthetic fixtures.
- **Model Rows & Live RAM Fit**: Shows provider, architecture, quantization, compute placement, live `RAMFitCalculator` headroom badges (`24.0 GiB` thin-headroom warning; hard fail on MLX GPU working-set overrun vs warning on GGUF), and yellow `stale_reload_fields` warnings with a one-click **`Reload now`**.
- **Schema-Driven Model Config Editor (`ModelConfigEditorSheet`)**:
  - Dynamically renders fields from `/v1/admin/model-options` (`ProviderConfigSchema`), grouped into **`Applies immediately`**, **`Requires reload`**, **`Advanced`**, and **`Fixed for this process`**.
  - Clicking **`Unset`** or clearing a field sends `null` to delete the override cleanly rather than freezing defaults. Read-only mode (`HEYLOOK_READONLY_MODEL_CONFIG=1`) disables local edits safely.
  - Embeds a live **`RAMFitMeterView`** (recalculating on draft edits) and 4+ rung **`ChatTemplateInspectorView`** with Jinja syntax + refused-shape checks and **`chat_template.heylook.jinja`** editing.
- **Pure-Swift MLX Quantization (`MLXModelConverterSheet`)**: Quantize and shard checkpoints in `affine`, `mxfp4`, `mxfp8`, `nvfp4`, or `q4_zero` formats straight from the sheet.

### Surface 6 (`⌘6`): Perf & Server (`PerfServerSurface`)
- **Zero Background Polling**: Metrics update cleanly on surface visit, generation finish, SQLite mutations, or manual **`Refresh Profile`** click (never via wasteful 3s background timers).
- **Hardware Telemetry & `MLComputePlan`**: Host RAM, Metal working set, GPU `GenerationGate` / `ANELane` activity, loaded model cards, and `EmbeddingGemma 2` ANE operator compatibility breakdown.
- **3 SQLite Performance Profile Tables (`1h` / `6h` / `24h` / `7d`)**:
  - `TimingBreakdownTableView` (`Operation`, `Avg ms`, `Count`, `%`)
  - `PerModelCacheTableView` (`Model`, `Reqs`, `Hit %`, `Saved tok`, `Outcomes`, `Why`)
  - `RecentTrendsTableView` (hourly throughput, prompt/gen TPS, cache savings)
- **Embedded HTTP/SSE & Bonjour Truth**: Clearly distinguishes `LOCALHOST ONLY` (`127.0.0.1`) vs `LAN + BONJOUR` (`0.0.0.0`, advertising `_heylook._tcp` for iOS 27.2+ and remote Mac discovery).
- **Operational `[settings]` & Structured JSONL Logs**: Edit `observability_level`, retention days, and `mlx_cache_limit_gb` live, switch between **`events.jsonl`** and **`metrics.jsonl`** disk streams, or inspect/export the in-memory `HeylookObservabilityLog` ring buffer.

---

## 4. Unified CLI (`heylook-cli`)

| Command | Usage |
| :--- | :--- |
| `doctor` / `status` | Inspect unified memory, Metal ceiling, compute lanes, ANE ASR readiness, and vector store stats. |
| `models` | List discovered MLX, GGUF, CoreAI, and speech recognizer models from `heylook.toml`. |
| `chat` | `heylook-cli chat [--model <id_or_path>] [--max-tokens <n>] [--no-thinking] "<prompt>"` |
| `transcribe` | `heylook-cli transcribe <audio-path> [--model <id_or_path>] [--lang <locale>]` |
| `index` | `heylook-cli index <path> [--sampling fps\|content\|off] [--no-audio-from-video] [--compute all_gpu\|hybrid_ane\|all_ane] [--db <path>]` |
| `search` | `heylook-cli search "<query>" [--dim 768\|512\|256\|128] [--task query\|similarity\|code\|qa\|fact] [--media <path>] [--db <path>]` |
| `watched` | List `FSEvents`-watched folders stored in the multimodal vector database. |
| `inspect-ane` | Inspect compiled `.mlmodelc` CoreML ANE tower inputs/outputs and `MLComputePlan` status. |
| `inspect-checkpoint` | Inspect local `EmbeddingGemma 2` checkpoint files and `config_sentence_transformers.json` prompts. |
| `convert-mlx` | `heylook-cli convert-mlx <src-dir> <dst-dir> [--bits 4\|8] [--group-size 32\|64] [--mode affine\|mxfp4\|mxfp8\|nvfp4] [--calibration standard\|q4_zero] [--overwrite] [--dry-run]` |
| `eval-lab` | Run truthful multi-turn KV chain, VLM parity, and Matryoshka Recall@K probes (zero synthetic fallbacks). |
| `serve` | `heylook-cli serve [port]` — start the standalone `/v1/*` HTTP/SSE server. |

---

## 5. HTTP & SSE API Reference (`/v1/*`)

Key invariants for API callers and remote iOS/macOS clients:
- **Public vs Admin Model Payloads**: `GET /v1/models` returns `PublicModelWireRow` (**zero `model_path`**) with full `engine: EngineContract` provenance; `GET /v1/admin/models` requires `X-Heylook-Admin-Token` when `admin_token` is configured and includes `config` (explicit keys only), `source`, and `stale_reload_fields`.
- **State & Concurrency**: Mutating or deleting a conversation or running `/prompt` / `/generate` while the conversation holds an active lock returns HTTP **`409 generation_in_progress`**.
- **Truncation & Media**:
  - Truncate messages strictly after a turn via `DELETE /v1/conversations/:id/messages?after=<pos>` (missing `after` returns `422`).
  - Stored images/audio deduplicate into `(conversation_id, media_id)` 64-hex SHA-256 blobs served at `GET /v1/conversations/:id/media/:media_id` (`X-Content-Type-Options: nosniff`).
- **SSE Events**: Long prefill and generation emit **`event: ping`** every `5.0s`, **`heylook_progress`**, **`content_block_delta`** (`thinking_delta` emits both `thinking` and `text`), **`message_stop`** (with truthful `performance` metrics excluding queue wait from duration), and terminal **`heylook_saved`**. Client disconnects never cancel server-side generation (`DELETE /v1/requests/:id` cancels explicitly; `POST /v1/requests/:id/reasoning-end` stops thinking cleanly).
- **Embeddings**: `POST /v1/embeddings` supports dynamic checkpoint `prompt_name` validation (`config_sentence_transformers.json`), 1,024-token padded text batches, consecutive single-window `windows.seconds` video decoding on MLX/`hybrid_ane`, and GGUF text embeddings (GGUF rejects video `windows` with `400`).
