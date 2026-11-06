# heylookitsanllm-macos (`heylook`)

Native Swift 6.2 personal inference workbench, multimodal media/embedding finder, and embedded HTTP/SSE server for Apple Silicon (`macOS 27.2+` and `iOS 27.2+`), designed for heterogeneous compute across **Metal GPU (`MLX Swift`)**, **Apple Neural Engine (`CoreAI` / `CoreML` / `Argmax`)**, and **`GGUF` (`llama-server`)**.

## Key Capabilities

- **Unified 3-Tier Engine Contract + Remote Sync (`HeylookEngines`)**:
  - **`MLXSwiftProvider`**: In-process Metal GPU LLM, VLM, and `EmbeddingGemma 2` inference (`mlx-swift` + `mlx-swift-lm`) with per-tower ANE offload (`ANECoreMLRuntime`).
  - **`CoreAIProvider`**: In-process execution of `.aimodel` bundles (`CoreAILanguageModels`, `CoreAISpeech`, `CoreAIDiffusion`) and `argmax-oss-swift` (`WhisperKit`, `TTSKit`, `SpeakerKit`) on the Apple Neural Engine and GPU.
  - **`GGUFSubprocessProvider` (macOS)**: Crash-isolated `llama-server` subprocess per loaded model (`-np 1`) with pure-Swift GGUF v2/v3 header inspection, `/props` capability discovery, and mid-stream `/v1/chat/completions/control` `reasoning_end`.
  - **`RemoteHeylookProvider`**: Connects the native `iOS 27.2+` app (or secondary Mac) to a Mac Studio `HeylookServer` over HTTP/SSE alongside local on-device ANE/MLX models.
- **Dual-Lane Heterogeneous Compute Scheduler (`ComputeScheduler`)**:
  - Serializes heavy Metal GPU generation through a FIFO `GenerationGate` (`max_queue_depth`, HTTP `503 model_overloaded` on overflow) while letting ANE-resident models (`EmbeddingGemma 2` ANE towers, `CoreAI` ANE `.aimodel` assets, speech models) run concurrently on the Neural Engine without evicting pinned 192 GB GPU models.
- **Folded Multimodal Embeddings & Media Finder (`HeylookEmbeddings`)**:
  - Integrates `mbed` directly: indexes `.video` (`16s`-stride overlapping `32s` windows with fixed `1 fps` or content-aware novelty selection + optional 16 kHz soundtrack), `.audio` (`30s` windows), `.image`, `.document` (`PDFKit` pages), `.code`, and `.text` into a SQLite (`WAL`) + `Accelerate` (`vDSP`) Matryoshka (`768d`/`512d`/`256d`/`128d`) vector store.
- **Embedded `/v1/*` Server (`HeylookServer`)**:
  - Built on **Hummingbird 2**, serving `POST /v1/messages`, disconnect-resilient `POST /v1/conversations/{id}/generate` (with `heylook_saved`), `POST /v1/embeddings`, `GET /v1/models` (with full `EngineDescription` provenance), `/v1/conversations/*`, `/v1/notebooks/*`, `/v1/presets/*`, `/v1/requests/{id}/reasoning-end`, and `/v1/system/metrics`.
- **5-Surface Multiplatform SwiftUI Workbench (`HeylookUI`)**:
  1. **Chat Workbench** (Load Panel, Preset Bar, Prompt Preview, collapsible Thinking Block with live **End thinking**, inline message edit + **Save & Continue**, and per-turn cache/draft/compute telemetry).
  2. **Notebook** (continuous buffer + **Save to Conversations**).
  3. **Embeddings & Media Finder Studio** (3-column Media Finder with `AVPlayerView` & `frameTimes` jump pills, Pairwise Cosine Matrix Compare, and `Recall@K` / `MRR` Embedding Lab).
  4. **Models & Engine Inspector** (`heylook.toml` & `model.heylook.toml` sidecar editor, `EngineDescription` provenance table, and per-tower ANE residency inspector).
  5. **Performance, Memory & Server Control** (192 GB Unified Memory & Metal wired limit gauge, ANE vs GPU residency bar, and embedded server toggle).

## Quick Start

```bash
# Fast SwiftPM build & test suite
swift build
swift test

# Inspect hardware, unified memory, and configuration
swift run heylook-cli doctor
swift run heylook-cli models
swift run heylook-cli inspect-ane ./models/embeddinggemma-2-MLX-BF16
swift run heylook-cli chat "Explain unified memory + ANE co-residency."
swift run heylook-cli search "Apple Neural Engine"

# Start the embedded /v1/* HTTP & SSE server (default port 8000)
swift run heylook-cli serve 8000

# Full native Xcode build (compiles Metal shaders + links MLX Swift & CoreAI from GitHub)
./scripts/build_native.sh build
./scripts/build_native.sh app
```

## Documentation

- [VISION.md](./VISION.md) — Core principles and project boundaries.
- [AGENTS.md](./AGENTS.md) — Operating guide and repository rules.
- [docs/wiki/architecture_overview.md](./docs/wiki/architecture_overview.md) — System topology and subsystem deep dive.
- [docs/project/CURRENT.md](./docs/project/CURRENT.md) & [docs/project/TODO.md](./docs/project/TODO.md) — Status and roadmap.
