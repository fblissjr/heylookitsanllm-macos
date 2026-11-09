# User Guide (`heylookitsanllm-macos`)

Complete guide to configuring and using **`heylook`** across the native macOS/iOS SwiftUI Workbench (`HeylookApp`), the command-line tool (`heylook-cli`), and the embedded `/v1/*` HTTP & SSE server.

For deep internal subsystem architecture, see the [LLM Knowledge Wiki](./wiki/README.md).

---

## Table of Contents

1. [Getting Started & Configuration](#1-getting-started--configuration)
2. [Surface 1: Chat Workbench](#2-surface-1-chat-workbench)
3. [Surface 2: Notebook](#3-surface-2-notebook)
4. [Surface 3: Embeddings & Media Finder Studio](#4-surface-3-embeddings--media-finder-studio)
5. [Surface 4: Models & Engine Inspector](#5-surface-4-models--engine-inspector)
6. [Surface 5: Performance, Memory & Server Control (and iOS Sync)](#6-surface-5-performance-memory--server-control-and-ios-sync)
7. [macOS MenuBar Controller](#7-macos-menubar-controller)
8. [CLI Reference (`heylook-cli`)](#8-cli-reference-heylook-cli)
9. [Connecting External Tools to the `/v1/*` Server](#9-connecting-external-tools-to-the-v1-server)

---

## 1. Getting Started & Configuration

### Building & Launching the App

To build and launch `heylook` with full in-process **MLX Swift** (`default.metallib` Metal shaders), **Core AI** (`.aimodel` bundles), and **CoreML ANE** support:

```bash
# Launch the native macOS desktop app
./scripts/build_native.sh app

# (Optional) One-command environment, Xcode 27.2/27.3+, Metal, iOS Simulator, skills & swift-lsp updater
./scripts/update_dev_env.sh

# Or open Package.swift in Xcode 27.2 / 27.3+ and run the `heylook-app` scheme on My Mac or iPhone/iPad
```

> **Deployment Baseline (`27.2+`) & Xcode 27.3 ("Golden Gate") SDK Support**: `Package.swift` and `HeylookWorkbench.app` (`LSMinimumSystemVersion = 27.2`) target **`macOS 27.2+`** and **`iOS 27.2+`** so binaries compiled with the **`Xcode 27.3` SDK** (`MacOSX27.3.sdk` / `iPhoneSimulator27.3.sdk`) launch and run natively on `macOS 27.2` hosts while adopting `27.3` enhancements (such as `.toolbarContainerBehavior(.automatic)`) via runtime `#available(macOS 27.3, iOS 27.3, *)` checks.
>
> **Toolchain Note (`xcrun swift` vs. Homebrew `swift`)**: If Homebrew's open-source `swift` formula (`/opt/homebrew/opt/swift/bin/swift`) appears before `/usr/bin/swift` in your `$PATH`, running bare `swift build` prints a benign linker warning when compiling `Package.swift` itself (`ld: warning: building for macOS-14.0, but linking with dylib '/usr/lib/swift/libswiftCore.dylib' which was built for newer version 27.0`) because Homebrew SwiftPM hardcodes `-target arm64-apple-macosx14.0` for manifest evaluation against Xcode's `MacOSX27.2.sdk`. To avoid the warning, either prefix SwiftPM commands with `xcrun` (`xcrun swift build`, `xcrun swift test`) or run `brew unlink swift` so `swift` resolves to Xcode's native `arm64-apple-macosx27.2.0` toolchain (`/usr/bin/swift`).

### Configuring Model Discovery (`heylook.toml`)

`heylook` discovers your local models from `heylook.toml` and never registers fake or synthetic models on startup. Copy `heylook.toml.example` to get started:

```bash
cp heylook.toml.example heylook.toml
```

It checks the following locations in order:

1. `HEYLOOK_CONFIG` environment variable (explicit file path)
2. `./heylook.toml` in the current working directory or repository root (gitignored)
3. `~/Library/Application Support/Heylook/heylook.toml`
4. `~/.heylook/heylook.toml`
5. `HEYLOOK_MODELS_DIR` environment variable (directory scan)

Example `heylook.toml`:

```toml
host = "127.0.0.1"
port = 8000
max_loaded_models = 2
max_queue_depth = 4

[scan]
folders = [
  "~/.heylook/models/mlx",
  "~/.heylook/models/gguf",
  "~/.heylook/models/coreai"
]

# Optional explicit model entries (override scan defaults)
[[models]]
id = "Qwen3.5-27B-MLX-4bit"
provider = "mlx"
path = "~/.heylook/models/mlx/Qwen3.5-27B-MLX-4bit"
pinned = true
ctx_size = 32768
```

What `heylook` discovers automatically inside each scan folder:
- **MLX directories**: Any directory containing `config.json` and `.safetensors` weights (auto-detecting vision towers, reasoning format, thinking controls, and `mtp.` speculative weights).
- **GGUF checkpoints**: Single `.gguf` files, multi-shard `*-00001-of-*.gguf` sets, per-quant subdirectories (`UD-Q6_K_XL/`, `Q8_0/`), `mmproj*.gguf` vision/audio projectors, and sibling speculative drafters (`-MTP-`, `dspark`, `-assistant`, `MTP/`).
- **Core AI bundles**: `.aimodel` directories containing `metadata.json` and compiled `.mlmodelc` components.

> **Empty Workspace Banner**: If you launch the app before configuring `heylook.toml`, the top of the window displays an **Empty Workspace** banner. Click **Re-scan `heylook.toml`** after adding models, or click **Load Sample Demo Workspace** if you want to explore the UI layout with explicitly labeled synthetic fixtures.

---

## 2. Surface 1: Chat Workbench

The **Chat Workbench** (`ChatWorkbenchSurface`) uses a unified 3-region `NavigationSplitView` + trailing `.inspector` layout for multi-turn conversations, multimodal vision/audio prompting, reasoning control, and 5-layer sampler tuning.

### Layout & Top Controls (`ChatTopControlBar` & Toolbar)
- **Left Sidebar (`ChatConversationsSidebarSection`)**: Switch between saved conversations, start a **New Conversation (`⌘N`)**, or swipe to delete a thread.
- **Window Toolbar**:
  - **Model Picker**: Select any discovered model (`MLX`, `CoreAI`, `GGUF`, or `Remote`) with live `[LOADED]` and `[PIN]` indicators.
  - **Inspector Toggle (`sidebar.trailing`)**: Show or hide the trailing **Load & Samplers Inspector** (`LoadAndSamplerInspectorView`).
  - **Quick Actions**: **New Conversation (`⌘N`)** and **New Notebook (`⇧⌘N`)** (plus an overflow menu on iOS).
- **Top Control Bar (`ChatTopControlBar`)**:
  - **Preset Menu**: Switch between seeded or custom sampler presets (*Balanced*, *Deterministic Code*, *Deep Thinking*, *Creative*) with an active checkmark indicator.
  - **Thinking Toggle**: Enables or disables model reasoning (`enable_thinking`). Automatically disabled when the active model's template does not support thinking or locked on when the template thinks unconditionally (such as Harmony `gpt-oss`).
  - **Depth Picker (`low` | `medium` | `high` | `max`)**: Appears automatically when the active model's template declares `reasoning_effort` or `reasoning_strength` values.
  - **Prompt Preview Toggle (`⇧⌘P`)**: Opens or closes the collapsible **`ChatPromptPreviewDrawer`** directly below the control bar, showing the exact rendered Jinja prompt, active `template_source` (e.g., `chat_template.heylook.jinja` or `chat_template.jinja`), and a **Copy Prompt** button.

### Trailing Load & Samplers Inspector (`LoadAndSamplerInspectorView`)
The right-hand `.inspector` column has two tabs:
1. **`Samplers` Tab**:
   - Displays the active model's **Context Window**, **Compute Placement**, **Template Source**, and **Speculative Drafter** status.
   - Interactive sliders for **Temperature**, **Top P**, **Top K**, **Min P**, **Max Tokens**, and **Thinking Budget**, plus a **Reset** button to clear session overrides.
   - **Provenance Pills (`SamplerProvenancePill`)**: Every slider displays where its current value came from:
     - `derived` — read from the model's `config.json`, GGUF header, or `generation_config.json`.
     - `configured` — set in `model.heylook.toml` or the active Preset.
     - `observed` — reported by the live engine (`llama-server` `/props` or `.aimodel` `metadata.json`).
     - `request` — overridden by you in the current session (`userOverriddenSamplerKeys`).
2. **`Prompt` Tab**:
   - Edit the per-session **System Prompt Override** with a one-click **Reset to Default** button.

### Transcript, Thinking Trace & Conversation Branching (`MessageRowView`)
- **Collapsible Reasoning Trace & Live `End Thinking Early`**: While a model is streaming its chain of thought (`<think>`, `<|channel>thought`, Harmony `analysis`, or `to=self`), expand or collapse the **Reasoning Trace** disclosure group or click **End Thinking Early** mid-stream to transition immediately into the final answer without aborting the request.
- **Inline Message Editing & `Save & Continue` Branching**:
  - Click the **Edit (pencil)** button (`28×28` touch-safe target) on any user or assistant message.
  - Click **Save & Continue** to update that message at its position, truncate all subsequent turns in SQLite (`WAL`), slice the in-process MLX `[KVCache]` back to the shared token prefix, and regenerate from that point.
- **Per-Turn Telemetry Badges**: Every assistant turn reports `generationTPS` (`tok/s`), KV cache prefix reuse (`cache X/Y`, `reused`/`cold`/`history_diverged`), speculative acceptance rate (`spec X/Y`), and active compute placement (`mlx_gpu`, `ane`, `gguf_subprocess`).

### Liquid Glass Composer, Multimodal Attachments & ANE Voice Turns (`ChatComposerBar`)
- **Images, Audio & Video Attachments**: Click the **Paperclip (`Attach Media`)** button (`NSOpenPanel` on macOS, `.fileImporter` on iOS) to stage `.png`, `.jpg`, `.webp`, `.wav`, or `.mp4` files.
  - Each staged **`ChatAttachmentPillView`** shows its modality icon, filename, estimated vision soft-token budget (`~280 tok`, derived from the model's vision config), and a `28×28` remove button.
  - If the active model does not support vision or audio, `ChatComposerBar` displays an inline capability warning before you send.
- **Hands-Free ANE Voice Turn (`Mic`)**: Click the **Microphone** button to record `16 kHz` mono PCM audio and click **Stop (waveform)** to transcribe speech on the Apple Neural Engine (`CoreAIProvider.transcribeAudio`) and hand off into disconnect-resilient generation (`POST /v1/conversations/{id}/voice-generate`).

### Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Cmd + Return` | Send message (or continue writing in Notebook) |
| `Escape` | Stop active generation stream |
| `Cmd + N` | Start a new conversation |
| `Cmd + Shift + N` | Create a new notebook |
| `Cmd + Shift + P` | Toggle live Prompt Preview drawer |
| `Cmd + 1` … `Cmd + 5` | Switch between the 5 Workbench surfaces |

---

## 3. Surface 2: Notebook Scratchpad

The **Notebook** surface (`NotebookScratchpadSurface`) provides a continuous, distraction-free markdown and code synthesis buffer:
- **Contextual Left Sidebar (`NotebooksSidebarSection`)**: Switch between saved notebooks or click **`+` (`⇧⌘N`)** to create a new notebook.
- **Top Bar**: Edit the notebook title directly, inspect live **word** and **character** counts, or trigger AI synthesis actions:
  - **Insert Summary**: Appends a concise bulleted summary of the current document buffer.
  - **Analyze Selection**: Critiques assumptions, trade-offs, and action items in the current notebook text.
  - **Continue Writing (`⌘↩`)**: Streams model completions directly onto the end of the document buffer using the active model and sampler cascade.

---

## 4. Surface 3: Embeddings & Media Finder Studio

The **Embeddings & Media Finder Studio** (`EmbeddingsStudioSurface`) folds the standalone `mbed` desktop application into `heylook`, powered by **`EmbeddingGemma 2`** (`embeddinggemma-2-MLX-BF16`, `768d` -> `128d` Matryoshka representations) inside the unified 3-region `NavigationSplitView` + `.inspector` + collapsible bottom **Eval Lab Drawer** architecture:

### Region 1: Contextual Left Sidebar (`EmbeddingsSidebarSections`)
1. **Modality Filter (`ModalityFilterRow`)**: Filter indexed artifacts across **All Modalities**, **Video**, **Image**, **Audio**, **Code**, **Docs**, or **Text** with live per-modality item badges.
2. **Index & Watched Folders**:
   - **Add Files or Folders…**: Opens `NSOpenPanel` on macOS or `.fileImporter` on iOS to index files and register watched directories (`FolderWatcher` via `FSEvents`).
   - **Re-scan Watched Folders**: Re-indexes modified files and prunes deleted files.
   - **Clear Index…**: Protected by a confirmation dialog (`DestructiveActionPrompt.clearEmbeddingIndex`) before wiping the SQLite vector store.
   - **Watched Folders List**: Displays each monitored folder with a `28×28` remove button and swipe-to-delete support.
3. **`MLX EmbeddingGemma 2` Settings**:
   - **Engine**: Switch between `Hybrid ANE + MLX (EmbeddingGemma 2)` (`HybridLocalEmbeddingEngine` + `ANECoreMLRuntime`), `MLX Swift LM (EmbeddingGemma 2)`, or `Deterministic 768d (Offline Test)`.
   - **Matryoshka Dim**: Slice and L2-normalize vectors at query time to **`768d`**, **`512d`**, **`256d`**, or **`128d`** via Apple `Accelerate` (`vDSP`) without re-indexing.
   - **Query Task**: Choose `Retrieval Query`, `Semantic Similarity`, `Code Retrieval`, `Question Answering`, or `Fact Verification`.
   - **Video Sampling & Soundtrack**: Choose **`1 fps`**, **`Content Keyframes`** (novelty-based anchor selection over `32×32` RGB thumbnails within `16s`-stride overlapping `32s` windows), or **`Off`**, and toggle **Include Video Soundtrack** (`16 kHz` `30s` audio windows).
4. **Compute Placement (ANE / GPU)**:
   - Choose a placement preset (**`Hybrid (ANE Vision/Audio + GPU Text)`**, **`All ANE (Zero GPU Eviction)`**, or **`All GPU (Max Throughput)`**) or independently route the **Vision Encoder** (`100% ANE`), **Audio Conformer** (`99% ANE`), and **Text Backbone + Projection** (`100% ANE`) across `CPU + Neural Engine (ANE)`, `MLX Metal GPU`, `CPU + GPU (CoreML)`, or `CPU Only`.
   - Inspect the live **CoreML ANE Towers** status badge and **`ComputeMemorySplitBar`** showing ANE vs. GPU MB residency.

### Region 2: Center Studio (`EmbeddingsStudioSurface` & `EmbeddingsEvalLabDrawer`)
- **Search Header (`EmbeddingsSearchHeaderBar`)**:
  - Live search bar with `28×28` clear button, **`Attach Media`** button for multimodal query-by-example (blending an image, audio, or video attachment with your text query), and **`Search`**.
  - **`ScoringPipelineBanner`**: Interactive Liquid Glass feature pills summarizing the active Engine, Matryoshka slice (`768d MRL`), Query Task, Video Windowing mode, and Compute Placement.
- **Status Toolbar (`EmbeddingsStatusToolbar`)**:
  - Shows active indexing progress with a **`Stop Indexing`** button, **`Add Files/Folders…`**, **`Eval Lab`** drawer toggle, and result/corpus counters.
- **Deduplicated Results List (`HitRowView`)**:
  - Parent-artifact deduplication ensures a multi-window video or multi-page PDF appears **once**, ranked by its top-scoring `bestSpan`.
  - Each row renders a non-blocking off-main-thread **`AsyncImage` `ThumbnailBadge`**, modality badge, best span label (e.g., `0:16–0:48` or `lines 10–45`), primary cosine similarity pill, and inline `768d`/`512d`/`256d`/`128d` Matryoshka sub-score pills.
- **Collapsible Bottom `EmbeddingsEvalLabDrawer` (`EvalLabSuite` & Pairwise Compare)**:
  - Toggle the **Eval Lab** button in the status toolbar to open the bottom evaluation drawer:
    - **Run Full Eval Suite**: Executes `EvalLabSuite.runAll()` (`chain_probe` KV prefix reuse, `vlm_parity_probe` vision token budgeting & HTTP 422 gating, and `matryoshka_recall_probe` `Recall@1` / `MRR` across `768d`/`512d`/`256d`/`128d`).
    - **Left vs. Right Pairwise Cosine Compare**: Compare any two passages or artifact spans across all four Matryoshka dimensions (`768d`, `512d`, `256d`, `128d`) and click **`Add to Lab`** (**Compare-to-Lab Bridge**) to promote the pair into the benchmark suite.

### Region 3: Trailing Inspector (`EmbeddingsInspectorView`)
- **Header & Quick Actions**: Shows modality icon, title, file size, path, **`Reveal in Finder`**, **`Compare in Lab`** (loads into the Left slot of the Eval Lab comparator), and **`Pin to Lab`** (loads into the Right slot and opens the Eval Lab drawer).
- **`ScoredFeatureBreakdownCard`**: Interactive bar chart comparing cosine similarity across `128d`, `256d`, `512d`, and `768d` with the active Matryoshka dimension highlighted.
- **Modality-Aware Preview & Click-to-Seek Keyframes (`KeyframeStamp`)**:
  - **Video & Audio**: Inline `AVPlayerView` (macOS) / `VideoPlayer` (iOS) with clickable **`KeyframeStamp`** seek pills (`0:16`, `0:24`, etc.) and clickable **`Matching Spans`** rows that seek the player directly to the matching timestamp.
  - **Images**: Non-blocking `AsyncImage` preview.
  - **Code, Text & Docs**: Monospaced preview with line/page span badges and metadata key-value table.

---

## 5. Surface 4: Models & Engine Inspector

The **Models & Engine Inspector** (`ModelsInspectorSurface`) gives you complete visibility and control over every discovered checkpoint:

- **Top Action Bar (`ModelsTopActionBar`) & Residency Controls**:
  - **Quantize / Convert MLX…**: Opens **`MLXModelConverterSheet`** to quantize an unquantized `.safetensors` directory in 100% pure Swift (`MLXLMCommon.convert`) with configurable **Bits** (`4-bit` / `8-bit`), **Group Size** (`32` / `64` / `128`), **Mode** (`affine`, `mxfp4`, `mxfp8`, `nvfp4`), **Calibration** (`standard` or `q4_zero`), **Dry Run**, and **Overwrite**.
  - **Re-scan**: Reloads `heylook.toml` and refreshes all discovered MLX, CoreAI, and GGUF checkpoints.
  - **Unload All Models…**: Protected by a confirmation dialog (`DestructiveActionPrompt.unloadAllModels`); reclaims all GPU/ANE memory, clears Metal caches, and terminates all `llama-server` subprocesses.
- **Model Registry Cards (`ModelRegistryRowView`)**:
  - **Pin / Unpin**: Pinned GPU models remain resident in unified memory and are protected from automatic LRU eviction when another GPU model is loaded.
  - **Load / Unload**: Explicitly pre-warm a model into memory or unload it immediately.
  - **Quantize…**: Opens `MLXModelConverterSheet` pre-populated with that MLX model's source path.
  - **Sidecar (`ModelSidecarEditorSheet`)**:
    - Edit `model.heylook.toml` settings (`ctx_size`, `flash_attn`, `supports_thinking`, `thinking_budget`, `image_max_tokens`, `compute_unit`, default samplers, and speculative `drafter_path`).
    - **4-Rung Chat Template Override (`chat_template.heylook.jinja`)**: Inspect the active template rung and edit, **Save Template Override**, or **Revert Override** (`chat_template.heylook.jinja`) directly in the sheet without mutating vendor `chat_template.jinja`.
- **Provenance & ANE Tower Inspector**:
  - Expand any model card to inspect every `ProvenanceField` (`value`, `provenance`, `effect`) alongside its `reasoningFormat` and `thinkingControls`.

---

## 6. Surface 5: Performance, Memory & Server Control (and iOS Sync)

The **Performance, Queue & Server** surface (`PerfServerSurface`) provides live hardware, subprocess, and network control:

- **Embedded HTTP & SSE Server Card**:
  - Start or stop `HeylookHTTPServer` (configurable port `1024...65535`, default `127.0.0.1:8000`) and inspect all live `/v1/*` routes.
- **Heterogeneous Compute Scheduler Card**:
  - Displays live **GPU Queue Depth**, **Loaded GPU Models** (with **`Unload All Models…`** confirmation action), and **Co-Resident ANE Models** alongside Unified Memory & Metal 88% wired limit gauges.
- **Active `llama-server` Subprocesses (`GGUFPortPool`)**:
  - Lists every running `llama-server` subprocess bound to its loopback port (`viewModel.sortedActiveGGUFPorts`, `127.0.0.1:18080+`) with per-model **Kill & Unload** buttons.
- **Remote Mac Studio / Tailscale Discovery (`_heylook._tcp`)**:
  - Enter a Mac Studio URL (`http://mac-studio.local:8000`) or click **Scan Local Network (Bonjour `_heylook._tcp`)** on `iOS 27.2+` or a secondary Mac to connect via `RemoteHeylookProvider` with one click.
- **Request Latency & Speculative Acceptance Chart**:
  - Plots per-turn token throughput (`tok/s`) and speculative decoding acceptance rates across recent assistant turns.

---

## 7. macOS MenuBar Controller

When running `HeylookApp` on macOS, a persistent **`heylook` icon** lives in the system MenuBar (`MenuBarExtra`):
- View embedded server status (`Running :8000` vs. `Stopped`) and start/stop the server without opening the main window.
- See how many models are currently loaded in GPU/ANE memory and trigger **Unload All Models** in one click.

---

## 8. CLI Reference (`heylook-cli`)

Run via `./scripts/build_native.sh cli <subcommand>` (full Metal + MLX build) or `swift run heylook-cli <subcommand>`:

| Subcommand | Description & Example |
| :--- | :--- |
| `doctor` | Inspect Apple Silicon memory budget, configured `heylook.toml` paths, and `llama-server` binary readiness:<br>`./scripts/build_native.sh cli doctor` |
| `status` | Print dual-lane scheduler state, memory budget, and vector store counts:<br>`./scripts/build_native.sh cli status` |
| `models` | List all discovered models from `heylook.toml` with provider, placement, context length, thinking format, and template source:<br>`./scripts/build_native.sh cli models` |
| `chat` | Stream live inference in the terminal with thinking & sampler flags:<br>`./scripts/build_native.sh cli chat --model Qwen3.5-0.8B-MLX-8bit --max-tokens 256 "Hello!"` |
| `serve` | Start the standalone `/v1/*` HTTP/SSE server and Bonjour advertiser:<br>`./scripts/build_native.sh cli serve 8000` |
| `index` | Index a directory or file into the SQLite Matryoshka vector store:<br>`./scripts/build_native.sh cli index ./docs --sampling content --audio-from-video --compute hybrid_ane` |
| `search` | Search indexed artifacts with Matryoshka dimension slicing and task prompts:<br>`./scripts/build_native.sh cli search "dual-lane scheduler" --dim 512 --task query` |
| `watched` | List folders monitored by `FolderWatcher`:<br>`./scripts/build_native.sh cli watched` |
| `inspect-ane` | Run `MLComputePlan` inspection on compiled `.mlmodelc` ANE towers:<br>`./scripts/build_native.sh cli inspect-ane ~/.heylook/models/embeddinggemma-2-MLX-BF16` |
| `inspect-checkpoint` | Inspect an `EmbeddingGemma 2` checkpoint directory and compute placement plan:<br>`./scripts/build_native.sh cli inspect-checkpoint ~/.heylook/models/embeddinggemma-2-MLX-BF16` |
| `convert-mlx` | Quantize a `.safetensors` checkpoint directory in pure Swift:<br>`./scripts/build_native.sh cli convert-mlx ./raw-bf16 ./quant-4bit --bits 4 --group-size 64 --mode affine` |
| `eval-lab` | Run the automated verification lab (`chain_probe`, `vlm_parity_probe`, `matryoshka_recall_probe`):<br>`./scripts/build_native.sh cli eval-lab` |

---

## 9. Connecting External Tools to the `/v1/*` Server

With the embedded server running (`./scripts/build_native.sh cli serve 8000` or toggled on in **Perf & Server**), any Anthropic Messages-compatible client or script can call `http://127.0.0.1:8000`:

### Streaming Chat (`POST /v1/messages`)

```bash
curl -N http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-0.8B-MLX-8bit",
    "max_tokens": 512,
    "stream": true,
    "thinking": true,
    "thinking_depth": "medium",
    "messages": [
      {"role": "user", "content": "Explain Matryoshka representation learning in two paragraphs."}
    ]
  }'
```

### Ending Thinking Mid-Stream (`POST /v1/requests/{id}/reasoning-end`)

```bash
curl -X POST http://127.0.0.1:8000/v1/requests/<request_id>/reasoning-end
```

### Multimodal Embeddings (`POST /v1/embeddings`)

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embeddinggemma-2-MLX-BF16",
    "dimensions": 512,
    "task": "query",
    "input": ["heterogeneous Apple Silicon ANE and GPU scheduling"]
  }'
```
