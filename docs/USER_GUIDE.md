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

# Or open Package.swift in Xcode 26+ and run the `heylook-app` scheme on My Mac or iPhone/iPad
```

### Configuring Model Discovery (`heylook.toml`)

`heylook` discovers your local models from `heylook.toml` and never registers fake or synthetic models on startup. It checks the following locations in order:

1. `HEYLOOK_CONFIG` environment variable (explicit file path)
2. `./heylook.toml` in the current working directory
3. `~/.heylook/heylook.toml`
4. `HEYLOOK_MODELS_DIR` environment variable (directory scan)

Example `~/.heylook/heylook.toml`:

```toml
host = "127.0.0.1"
port = 8000
max_loaded_models = 2
max_queue_depth = 4

[scan]
folders = [
  "~/Models/mlx",
  "~/Models/gguf",
  "~/Models/coreai"
]

# Optional explicit model entries (override scan defaults)
[[models]]
id = "Qwen3.5-27B-MLX-4bit"
provider = "mlx"
path = "~/Models/mlx/Qwen3.5-27B-MLX-4bit"
pinned = true
context_size = 32768
```

What `heylook` discovers automatically inside each scan folder:
- **MLX directories**: Any directory containing `config.json` and `.safetensors` weights (auto-detecting vision towers, reasoning format, thinking controls, and `mtp.` speculative weights).
- **GGUF checkpoints**: Single `.gguf` files, multi-shard `*-00001-of-*.gguf` sets, per-quant subdirectories (`UD-Q6_K_XL/`, `Q8_0/`), `mmproj*.gguf` vision/audio projectors, and sibling speculative drafters (`-MTP-`, `dspark`, `-assistant`, `MTP/`).
- **Core AI bundles**: `.aimodel` directories containing `metadata.json` and compiled `.mlmodelc` components.

> **Empty Workspace Banner**: If you launch the app before configuring `heylook.toml`, the top of the window displays an **Empty Workspace** banner. Click **Re-scan `heylook.toml`** after adding models, or click **Load Sample Demo Workspace** if you want to explore the UI layout with explicitly labeled synthetic fixtures.

---

## 2. Surface 1: Chat Workbench

The **Chat Workbench** is your primary interface for multi-turn conversations, multimodal vision/audio prompting, reasoning control, and sampler tuning.

### Selecting & Loading Models
- Use the **Model Picker** in the top toolbar to select any discovered model (`MLX`, `CoreAI`, `GGUF`, or `Remote`).
- Click **Load & Samplers** (slider icon) to open the **Load & Sampler Panel**:
  - **Provenance Badges**: Every setting displays where its current value came from:
    - `derived` — read from the model's `config.json`, GGUF header, or `generation_config.json`.
    - `configured` — set in the model's `model.heylook.toml` sidecar or active Preset.
    - `observed` — reported by the live engine (e.g., `llama-server` `/props` or `.aimodel` `metadata.json`).
    - `request` — overridden by you in the current session.
  - **Effect Classes**: Badges indicate whether changing a field takes effect immediately (`per_request`), reloads the model (`reload`), or restarts the server (`restart`).

### Thinking & Reasoning Depth Controls
`heylook` inspects each model's `chat_template.jinja` and tokenizer files to configure its reasoning controls automatically:
- **Think Toggle**: Enables or disables model reasoning (`enable_thinking`). If a model's template thinks unconditionally (such as Harmony `gpt-oss`), the toggle is locked on and explains why in its tooltip.
- **Depth Picker (`low` | `medium` | `high` | `max`)**: Appears automatically when the active model's template supports `reasoning_effort` or `reasoning_strength`.
- **Collapsible Thinking Block & Live `End thinking`**: While a model is streaming its chain of thought (`<think>`, `<|channel>thought`, Harmony `analysis`, or `to=self`), you can expand/collapse the thinking block or click **End thinking** mid-stream to immediately transition the model into its final answer without aborting the request.

### Presets, Sampler Cascade & Prompt Preview
- **Preset Bar**: Quickly switch between seeded or custom sampler presets (*Balanced*, *Deterministic Code*, *Deep Thinking*, *Creative*) or click **Reset Overrides** to return to the model's file-derived defaults.
- **Prompt Preview (`Cmd+Shift+P`)**: Opens the exact rendered Jinja chat template prompt (including system instructions, thinking kwargs, and special tokens) before or after sending a message.

### Multimodal Attachments & Hands-Free Voice Turns
- **Images & Audio**: Click **Attach Media** or drag and drop `.png`, `.jpg`, `.webp`, or `.wav` files into the composer.
  - Each staged image pill displays its estimated vision soft-token budget (`~280 tok`, derived from the model's vision config).
  - If the active model does not support vision or audio, `heylook` warns you immediately in the composer before you send.
- **Hands-Free Voice Turn**: Click the **Mic / Voice Turn** button to transcribe speech on the Apple Neural Engine (`16 kHz` log-mel frontend) and hand off directly into disconnect-resilient server generation (`POST /v1/conversations/{id}/voice-generate`).

### Inline Message Editing & Conversation Branching
- Hover over any prior user or assistant message and click **Edit**.
- Click **Save & Continue** to truncate subsequent turns in SQLite (`WAL`), slice the in-process MLX `[KVCache]` back to the shared token prefix, and regenerate from that point—reporting exact cache reuse (`cache X/Y tokens`) in the message footer.

### Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Cmd + Return` | Send message |
| `Escape` | Stop active generation stream |
| `Cmd + N` | Start a new conversation |
| `Cmd + Shift + P` | Toggle live Prompt Preview |

---

## 3. Surface 2: Notebook

The **Notebook** surface provides a continuous, distraction-free document buffer for long-form drafting, code sketching, and prompt iteration:
- Select a notebook from the left sidebar or click **New Notebook**.
- Type or paste content into the main editor and click **Continue Writing** to stream model completions directly into the document buffer.
- Click **Save to Conversations** at any time to promote a notebook buffer into a structured multi-turn conversation in the Chat Workbench.

---

## 4. Surface 3: Embeddings & Media Finder Studio

The **Embeddings & Media Finder Studio** folds the standalone `mbed` desktop application into `heylook`, powered by **`EmbeddingGemma 2`** (`embeddinggemma-2-MLX-BF16`, `768d` -> `128d` Matryoshka representations).

Use the segmented switcher at the top of the surface to choose between three modes:

### Mode A: Media Finder (3-Column Workspace)
1. **Left Column — Compute Placement, Indexing & Watched Folders**:
   - **Per-Tower Compute Placement**: Independently route the **Vision Encoder** (`100% ANE`), **Audio Encoder** (`99% ANE`), and **Text Backbone** (`100% ANE`) across **Neural Engine (ANE)**, **MLX GPU**, **CoreML GPU**, or **CPU Only**, with a live **Compute Memory Split Bar** showing ANE vs. GPU MB residency.
   - **Indexing Options**:
     - **Video Sampling**: Choose **`1 fps`** (fast metadata path) or **`Content`** (novelty-based anchor selection over `32x32` RGB frame thumbnails within `16s`-stride overlapping `32s` windows).
     - **Index Audio from Video**: Toggle on to extract and embed the `16 kHz` audio soundtrack (`30s` windows with `15s` stride) alongside visual video windows.
   - **Watched Folders**: Add directories to monitor via macOS `FSEvents`; new or modified files are re-indexed automatically and deleted files are pruned.
2. **Center Column — Hybrid Search & Deduped Artifact Grid**:
   - **Matryoshka Dimension Picker**: Slice and L2-normalize vectors at query time to **`768d`**, **`512d`**, **`256d`**, or **`128d`** using Apple `Accelerate` (`vDSP`) without re-indexing.
   - **Task Prompt Picker**: Select any of the 10 `EmbeddingGemma 2` task prompts (`Search Query`, `Semantic Similarity`, `Code Retrieval`, `Question Answering`, `Fact Verification`, `Clustering`, `Classification`, etc.).
   - **Hybrid Example Attachment**: Attach a reference image, audio clip, or text snippet to blend example embeddings with your text query.
   - **Parent-Artifact Deduplication**: A 5-minute video with 19 temporal windows or a multi-page PDF appears **once** in results, ranked by its top-scoring `bestSpan`.
3. **Right Column — Span Inspector & Media Player**:
   - Displays inline **`AVPlayerView`** playback for video and audio artifacts.
   - Click any **`frameTimes` jump pill** (e.g., `00:16–00:48 @ 22.5s`) to seek the video player directly to the matching visual moment.
   - Inspect the **`ScoredFeatureBreakdownCard`** comparing cosine similarity across all four Matryoshka dimensions (`768d` / `512d` / `256d` / `128d`).

### Mode B: Pairwise Cosine Matrix Compare
- Enter or paste multiple text passages or artifact labels to compute an interactive **N×N pairwise cosine similarity heatmap** at the selected Matryoshka dimension.
- Click **Promote to Embedding Lab** (**Compare-to-Lab Bridge**) to convert your comparison items into a repeatable benchmark dataset.

### Mode C: Embedding Lab (`Recall@K` & `MRR`)
- Run automated retrieval evaluations across your corpus and compare **`Recall@1`**, **`Recall@3`**, **`MRR`**, and query latency across **`768d`**, **`512d`**, **`256d`**, and **`128d`** slices side by side.

---

## 5. Surface 4: Models & Engine Inspector

The **Models & Engine Inspector** gives you complete visibility and control over every discovered checkpoint:

- **Residency Controls**:
  - **Pin / Unpin**: Pinned GPU models remain resident in unified memory and are protected from automatic LRU eviction when another GPU model is loaded.
  - **Load / Unload**: Explicitly pre-warm a model into memory or unload it immediately (releasing `ModelContainer`, CoreML `MLModel` instances, or terminating its `llama-server` subprocess and calling `MLX.Memory.clearCache()`).
  - **Unload All Models**: One-click button in the top bar to reclaim all GPU/ANE memory and stop all `llama-server` subprocesses.
- **Edit Sidecar (`model.heylook.toml`)**:
  - Click **Edit Sidecar** on any model to configure `context_size`, `flash_attention`, `supports_thinking`, `thinking_budget`, `image_max_tokens` (clamped automatically against `ubatch_size` for non-causal GGUF vision projectors), default sampler overrides, and speculative drafter paths.
- **Convert MLX Model (Pure-Swift Quantization)**:
  - Click **Convert MLX** to quantize an unquantized `.safetensors` directory in 100% pure Swift (`MLXLMCommon.convert`):
    - **Bits**: `4-bit` or `8-bit`
    - **Group Size**: `32` or `64`
    - **Mode**: `affine`, `mxfp4`, `mxfp8`, or `nvfp4`
    - **Calibration**: `standard` or `q4_zero`
- **Engine Provenance & ANE Tower Inspector**:
  - Inspect every field of the selected model's `EngineDescription` alongside its provenance (`derived`, `configured`, `observed`) and `EffectClass`.
  - Inspect compiled `EmbeddingGemma2*ANE.mlmodelc` towers via Apple's `MLComputePlan` to verify exact Neural Engine operator residency (`100%` Vision, `99%` Audio, `100%` Text).

---

## 6. Surface 5: Performance, Memory & Server Control (and iOS Sync)

- **Unified Memory & Metal Wired Budget Gauge**:
  - Displays total Apple Silicon unified memory (e.g., `192 GB`), baseline resident footprint, estimated active model footprint, and remaining safe headroom under the 88% Metal wired limit.
- **Dual-Lane Scheduler Telemetry**:
  - Monitors the **GPU `GenerationGate`** (active model, queue depth, max loaded GPU models) alongside the **Co-Resident ANE Lane** (concurrent embedding and speech tasks that never evict GPU LLMs).
- **Embedded `/v1/*` Server Toggle**:
  - Start or stop the embedded HTTP/SSE server (default `127.0.0.1:8000`) and advertise it on the local network via Bonjour (`_heylook._tcp`).
- **Bonjour LAN Scanner (`iOS 27.2+` & Secondary Macs)**:
  - On an iPhone, iPad, or secondary Mac running `HeylookApp`, open **Perf & Server** and click **Scan LAN (`_heylook._tcp`)**.
  - Click **Connect** next to your Mac Studio to register its models via `RemoteHeylookProvider` and stream chats, thinking blocks, and embeddings from your phone with disconnect-resilient server handoff.

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
