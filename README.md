# heylookitsanllm-macos (`heylook`)

Native Swift 6.2 personal inference workbench, multimodal media/embedding finder, and embedded HTTP/SSE server for Apple Silicon (`macOS 27.2+` and `iOS 27.2+`), designed for heterogeneous compute across **Metal GPU (`MLX Swift`)**, **Apple Neural Engine (`CoreAI` / `CoreML` / `Argmax`)**, and **`GGUF` (`llama-server`)**.

## Key Capabilities

- **Unified 3-Tier Engine Contract + Remote Sync (`HeylookEngines`)**:
  - **`MLXSwiftProvider`**: In-process Metal GPU LLM, VLM (`CIImage` inputs), and `EmbeddingGemma 2` inference (`mlx-swift` + `mlx-swift-lm` with `default.metallib` & `#huggingFaceTokenizerLoader()`), automatic prompt prefix caching (`PrefixCacheTracker`), `startsInThinking` for `qwen3` / `deepseek` reasoning checkpoints, per-tower ANE offload (`ANECoreMLRuntime`), and deterministic `unload()` (`MLX.GPU.clearCache()`).
  - **`CoreAIProvider`**: In-process execution of `.aimodel` bundles (`CoreAILanguageModels`, `CoreAISpeech`, `CoreAIDiffusion`) and `argmax-oss-swift` (`WhisperKit`, `TTSKit`, `SpeakerKit`) on the Apple Neural Engine and GPU with `SpeculativeReport` telemetry.
  - **`GGUFSubprocessProvider` (macOS)**: Crash-isolated `llama-server` subprocess per loaded model (`-np 1`) managed by `SubprocessLifecycleRegistry` (`SIGTERM` -> `SIGKILL` + `atexit` teardown on unload, LRU eviction, or server stop), with 192 GB auto-sizing (`-ub 2048`, `-cram`), 4-rung `chat_template` ladder (`chat_template_path` > `chat_template.heylook.jinja` > `chat_template.jinja` > `gguf_embedded`), automatic `[MTP]` speculative drafter pairing (`dspark`, `-MTP-`, `-assistant`), `--mmproj` vision projector discovery, multi-turn `reasoning_content` forwarding, and mid-stream `/v1/chat/completions/control` `reasoning_end`.
  - **`RemoteHeylookProvider`**: Connects the native `iOS 27.2+` app (or secondary Mac) to a Mac Studio `HeylookServer` over HTTP/SSE with Bonjour `_heylook._tcp` auto-discovery (`BonjourServiceBrowser`) alongside local on-device ANE/MLX models.
- **Dual-Lane Heterogeneous Compute Scheduler (`ComputeScheduler`)**:
  - Serializes heavy Metal GPU generation through a FIFO `GenerationGate` (`max_queue_depth`, HTTP `503 model_overloaded` on overflow), automatically invokes `provider.unload()` on unpinned GPU LRU eviction or `unloadAllModels()`, and lets ANE-resident models (`EmbeddingGemma 2` ANE towers, `CoreAI` ANE `.aimodel` assets, speech models) run concurrently on the Neural Engine without evicting pinned 192 GB GPU models.
- **5-Layer Deterministic Sampler Cascade (`SamplerCascade`)**:
  - Resolves `builtin` -> `vendor generation_config.json` -> `model.heylook.toml` -> `preset` -> `request` with per-field provenance (`derived`, `configured`, `request`).
- **Folded Multimodal Embeddings & Media Finder (`HeylookEmbeddings`)**:
  - Integrates `mbed` directly: indexes `.video` (`16s`-stride overlapping `32s` windows with fixed `1 fps` or content-aware novelty selection + optional 16 kHz soundtrack), `.audio` (`30s` windows), `.image`, `.document` (`PDFKit` pages), `.code`, and `.text` into a SQLite (`WAL`) + `Accelerate` (`vDSP`) Matryoshka (`768d`/`512d`/`256d`/`128d`) vector store.
- **Embedded `/v1/*` Server (`HeylookServer`)**:
  - Serves incremental SSE (`SSEConnectionStreamer`) and non-streaming `POST /v1/messages` (with HTTP `422` multimodal capability gating), disconnect-resilient `POST /v1/conversations/{id}/generate` (with `event: heylook_saved`), model-derived `POST /v1/conversations/{id}/prompt` and `POST /v1/models/{id}/image-plan`, `POST /v1/embeddings`, `GET /v1/capabilities`, `GET /v1/models` (with full `EngineDescription` provenance), `POST /v1/admin/models/unload-all`, `GET/PATCH /v1/admin/models/{id}` + `load`/`unload`/`chat-template`, `/v1/conversations/*`, `/v1/notebooks/*`, `/v1/presets/*`, `/v1/requests/{id}/reasoning-end`, and `/v1/system/metrics`.
- **5-Surface Multiplatform SwiftUI Workbench (`HeylookUI`)**:
  1. **Chat Workbench** (Load Panel with per-slider provenance pills, Preset Bar, model-derived Prompt Preview, multimodal drag-and-drop image/audio pills with `~tok` token budget badges, collapsible Thinking Block with live **End thinking**, inline message edit + **Save & Continue**, and `Cmd+Return` / `Esc` / `Cmd+N` / `Cmd+Shift+P` shortcuts).
  2. **Notebook** (continuous buffer + **Save to Conversations**).
  3. **Embeddings & Media Finder Studio** (3-column Media Finder with `AVPlayerView` & `frameTimes` jump pills, `ScoredFeatureBreakdownCard`, Pairwise Cosine Matrix Compare, and `Recall@K` / `MRR` Embedding Lab).
  4. **Models & Engine Inspector** (`ModelSidecarEditorSheet` for `model.heylook.toml`, Pin / Load / Unload + **Unload All Models** controls, `EngineDescription` provenance table, and per-tower ANE residency inspector).
  5. **Performance, Memory & Server Control** (192 GB Unified Memory & Metal wired limit gauge, ANE vs GPU residency bar, embedded server toggle, and Bonjour `_heylook._tcp` LAN scanner).

## Quick Start

```bash
# Fast SwiftPM build & 24-suite verification run
swift build
swift test
swift run heylook-cli eval-lab

# Full native Xcode build (compiles Cmlx default.metallib + links MLX Swift & mlx-swift-lm from GitHub)
./scripts/build_native.sh build
./scripts/build_native.sh app

# Inspect hardware, 192 GB unified memory, discovered models, and ANE tower compute plans
./scripts/build_native.sh cli doctor
./scripts/build_native.sh cli models
./scripts/build_native.sh cli inspect-ane ./models/embeddinggemma-2-MLX-BF16
./scripts/build_native.sh cli inspect-checkpoint ./models/embeddinggemma-2-MLX-BF16

# Stream live Metal GPU inference from a configured MLX checkpoint
./scripts/build_native.sh cli chat --model Qwen3.5-0.8B-MLX-8bit --max-tokens 96 "Explain unified memory + ANE co-residency."

# Index media/docs with hybrid ANE + MLX EmbeddingGemma 2 and search with Matryoshka attribution
./scripts/build_native.sh cli index ./docs --compute hybrid_ane --db /tmp/heylook-vectors.sqlite
./scripts/build_native.sh cli search "dual-lane GPU and ANE co-residency scheduler" --dim 768 --db /tmp/heylook-vectors.sqlite

# Start the embedded /v1/* HTTP & SSE server (default port 8000)
./scripts/build_native.sh cli serve 8000
```

## Documentation

- [VISION.md](./VISION.md) — Core principles and project boundaries.
- [AGENTS.md](./AGENTS.md) — Operating guide and repository rules.
- [docs/wiki/architecture_overview.md](./docs/wiki/architecture_overview.md) — System topology and subsystem deep dive.
- [docs/project/CURRENT.md](./docs/project/CURRENT.md) & [docs/project/TODO.md](./docs/project/TODO.md) — Status and roadmap.
