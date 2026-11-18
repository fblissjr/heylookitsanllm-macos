# heylookitsanllm-macos (`heylook`)

A native **Swift 6.4 (`v6` language mode)** multiplatform (`macOS 27.2+` and `iOS 27.2+` deployment floor supporting `27.2.1` hosts and `#available(macOS 27.3, iOS 27.3, *)` APIs) personal AI inference workbench, multimodal media/embedding studio, and embedded HTTP/SSE server for Apple Silicon.

Built to achieve complete behavioral and frontend parity with `heylookitsanllm` (`v2.0.251`) on top of an Apple Silicon–native, truthful-runtime foundation.

---

## Key Capabilities

### 1. Six-Surface Native SwiftUI Workbench (`⌘1` – `⌘6`)
- **`⌘1` Chat (`ChatWorkbenchSurface`)**: Multi-turn streaming with reasoning traces, inline **Model Load / Reload Panel** (`ctx_size`, `flash_attn`, 1-token JIT warm, `Reload (Apply Changes)`), **4-state system prompt chip**, live Prompt Preview drawer, staged `image-plan` budget inspector + 2048px resize toggle, voice-to-turn ANE ASR recording, inline message edit/regenerate/continue/delete, and smart `ScrollViewReader` tail-follow (`Jump to latest`).
- **`⌘2` Notebook (`NotebookSurface`)**: Distraction-free drafting pad (`CursorAwareTextEditor`), per-document model picker, preset cascade, collapsible system prompt drawer, and `@Environment(\.scenePhase)` flush.
- **`⌘3` Presets (`PresetsSurface`)**: Clean unseeded preset library (`COLLATE NOCASE`), 13-key sparse `"Null-Means-Cascade"` overrides, optimistic concurrency locking (`Reload Latest Copy`), and live model compatibility preview.
- **`⌘4` Embeddings & ANE (`EmbeddingsStudioSurface`)**:
  - **Compare**: Multi-block text/image/audio/video items, `windows.seconds` (uniform FPS or scene-change), Matryoshka cosine heatmap (`768d`/`512d`/`256d`/`128d`), vector inspector, and clickable video window hits opening `CompareVideoWindowPreviewSheet` (`AVPlayer` bounded playback + sampled `frame_times`).
  - **Embedding Lab**: Import custom `set.json` benchmarks or run the built-in 8-case CoreGraphics multimodal sample suite across model/dimension/prompt/ANE-lane sweep configs.
  - **Media Corpus (`mbed`)**: Optional local corpus ingestion across text, code, PDFs (`PDFKit`), images, `30s` audio windows, and `32s`/`16s`-stride video windows (`VectorStore` SQLite + `Accelerate`/`vDSP`).
- **`⌘5` Models (`ModelsInspectorSurface`)**: Watch folders (`heylook.toml [scan]`), live `RAMFitCalculator` headroom badges (`24.0 GiB` thin-headroom warning; MLX hard fail vs GGUF warning), `stale_reload_fields` + `Reload now`, schema-driven `ModelConfigEditorSheet` (with integrated `RAMFitMeterView` & 4+ rung `ChatTemplateInspectorView`), and pure-Swift `MLXModelConverterSheet`.
- **`⌘6` Perf & Server (`PerfServerSurface`)**: Hardware RAM/Metal/scheduler strip, `EmbeddingGemma 2` `MLComputePlan` ANE operator plan, 3 SQLite profile tables (`TimingBreakdownTableView`, `PerModelCacheTableView`, `RecentTrendsTableView`) across `1h`/`6h`/`24h`/`7d`, truthful localhost vs LAN + Bonjour (`_heylook._tcp`) status, persistent `[settings]` & `events.jsonl`/`metrics.jsonl` disk log reader, and `HeylookObservabilityLog` ring viewer (zero background 3s polling).

### 2. Multi-Backend Heterogeneous Compute (`HeylookEngines` & `HeylookEmbeddings`)
- **`MLXSwiftProvider` (`HEYLOOK_ENABLE_NATIVE_MLX=1`)**: `PreRenderedVLMTokenizerProxy`, `VisionFeatureCache` (16 / 512 MB LRU), `APCSnapshotCache` keyed on `(tokenIDs, mediaSalt)`, `ThinkingControlLogitProcessor`, lossless `LeviathanMTPTokenIterator` rejection/residual sampling, `Gemma4Specialization` (`48px` vision alignment, `gemma4_assistant` `backbone_hidden_size` pairing), `HybridRecurrentSWASpecialization`, and pure-Swift `MLXModelConverter` (`affine`, `mxfp4`, `mxfp8`, `nvfp4`, `q4_zero`).
- **`GGUFSubprocessProvider`**: Isolated `llama-server` subprocesses (`GGUFPortPool`), automatic `mmproj` discovery, `GGUFSpawnConfig.autoSized` (SWA `--cache-reuse` suppression, `--ctx-checkpoints 32 --checkpoint-min-step 256`, `--reasoning-preserve`), reply-only `StopSequenceFilter`, and `CompletionIDBox` mid-stream `reasoning_end`.
- **`CoreAIProvider` & `SpeechTranscriptionService`**: `.aimodel` Neural Engine execution (`HEYLOOK_ENABLE_COREAI=1`) and decoupled on-device ASR (`CoreAISpeech` + `SpeechAnalyzer`/`SpeechTranscriber`) with subchunk-walking `decodeRIFFChunksMonoPCM`.
- **Multimodal & GGUF Embeddings**: `MLXSwiftLMEmbeddingGemma2Engine` (1,024-token padded text batching, true `hybrid_ane` ANE->GPU `softTokens` bridge without `TextANE.mlmodelc`, consecutive single-window `windows.seconds` video decoding, dynamic `config_sentence_transformers.json` prompts) plus `GGUFEmbeddingEngine` (`llama-server --embeddings`).

### 3. Truthful Engine Contract, SQLite `WAL` & Embedded HTTP/SSE Server (`HeylookCore` & `HeylookServer`)
- **`EngineContract`**: `Fact<Value>` (`"value": null` on `.unknown` / `.notApplicable`), `SettingFact`, zero-path `PublicModelWireRow`, explicit-only `AdminModelWireRow`, `ProviderConfigSchema` (13 derived `SamplerCascade` keys, `422` `file_only` guard), comment-preserving `TOMLCodec`, and `garbageDepthSentinel` template probing.
- **`WorkbenchService` & `ConversationOrchestrator`**: Unified actor saga shared by `HeylookServer` and `HeylookUI`, providing `409 generation_in_progress` locking, detachable `SubscriberHub` streaming, atomic SQLite tail commits (`replaceTailWithMessage` / `replaceTailWithUpdate`), scoped 64-hex `media_blobs` (`nosniff`), non-compacting `deleteMessage` + `DELETE ?after=`, `PatternRouter` greedy `*model_id`, HTTP/1.1 Keep-Alive + chunked transport, and `5.0s` `event: ping` SSE framing.

---

## Quick Start

```bash
# 1. Run fast Swift Testing unit + contract suite
swift test

# 2. Full native Xcode build (Metal shaders -> default.metallib + MLX/CoreAI + TCC signing)
./scripts/build_native.sh build

# 3. Launch the native macOS workbench app
./scripts/build_native.sh app

# 4. Run CLI diagnostics, indexing, quantization, evals, or headless server
./scripts/build_native.sh cli doctor
./scripts/build_native.sh cli models
./scripts/build_native.sh cli serve 8000
```

> [!NOTE]
> All SwiftPM dependencies in `Package.swift` use canonical remote GitHub URLs gated behind `HEYLOOK_ENABLE_NATIVE_MLX=1` (`mlx-swift`, `mlx-swift-lm`, `swift-transformers`) and `HEYLOOK_ENABLE_COREAI=1` (`apple/coreai-models`). Set `HEYLOOK_READONLY_MODEL_CONFIG=1` to run with local model/config writes disabled (`409`).

---

## Documentation Index

- **[VISION.md](./VISION.md)** — Core design beliefs, tradeoff compass, and non-goals.
- **[AGENTS.md](./AGENTS.md)** — Operating rules, build/test commands, and LLM Knowledge Wiki index router.
- **[docs/USER_GUIDE.md](./docs/USER_GUIDE.md)** — Operator handbook for the 6-surface workbench, `heylook.toml`, `heylook-cli`, and `/v1/*` APIs.
- **[docs/wiki/README.md](./docs/wiki/README.md)** — Deep-dive architecture wiki across all 8 subsystems.
- **[docs/project/CURRENT.md](./docs/project/CURRENT.md)** & **[docs/project/TODO.md](./docs/project/TODO.md)** — Subsystem readiness matrix and parity roadmap.
