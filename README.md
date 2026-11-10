# heylookitsanllm-macos (`heylook`)

Native Swift 6.2 multiplatform (`macOS 27.2+` and `iOS 27.2+`) personal inference workbench, multimodal embedding/retrieval studio, and embedded Anthropic Messages-conformant `/v1/*` HTTP/SSE server for Apple Silicon—orchestrating heterogeneous compute across **`MLX Swift` (Metal GPU)**, **`CoreAI` / `CoreML` (Apple Neural Engine)**, **`GGUF` (`llama-server` subprocesses)**, and **`RemoteHeylookProvider` (Bonjour `_heylook._tcp`)**.

- **[User Guide (`docs/USER_GUIDE.md`)](./docs/USER_GUIDE.md)** — How to configure `heylook.toml`, drive all 6 workbench surfaces (`⌘1`–`⌘6`), manage presets, index & compare multimodal artifacts, quantize models, and use `heylook-cli` and `/v1/*`.
- **[LLM Knowledge Wiki (`docs/wiki/README.md`)](./docs/wiki/README.md)** — Deep-dive subsystem architecture, engine contracts, and sharp edges.

---

## Architecture & Capabilities

- **Core Domain, Jinja2 AST, Provenance & SQLite WAL Store (`HeylookCore`)**:
  - **Provenance & 5-Layer Sampler Cascade** (`EngineDescription.swift`, `SamplerCascade.swift`): Deterministic resolution across `builtin` -> `vendor` (`generation_config.json` / GGUF `general.sampling.*`) -> `model.heylook.toml` -> `preset` -> `request` (`SamplerLayerSource`: `builtin`, `vendor`, `model_config`, `preset`, `request`) with dual `snake_case`/`camelCase` `SamplerSettings` `Codable` and per-field engine provenance (`ProvenanceSource`: `derived`, `configured`, `observed`, `observed_cached`, `unknown`, `not_applicable`).
  - **Jinja2 AST Engine, Template Inspection & Multi-Protocol Reasoning** (`HeylookJinjaRenderer.swift`, `ModelTemplateInfo.swift`, `ReasoningParser.swift`): Recursive-descent Jinja2 lexer/AST parser/evaluator (`HeylookJinjaRenderer`, bridged with `huggingface/swift-jinja` when linked) supporting `validateSyntax`, `detectCapabilities`, `checkPrefixStability`, and special-token `PromptMarkerSpan` extraction; 4-rung `ChatTemplateInspection` (`chat_template_path` > `chat_template.heylook.jinja` > `chat_template.jinja` > `gguf_embedded` / `tokenizer_config.json`) reporting `origin`, `writable`, `inertReason`, `stale`, `prefixStable`, `caps`, and Hugging Face `.metadata` SHA-1/SHA-256 `modifiedSinceDownload`; file-derived `ModelTemplateInfo` & `ThinkingControls`; and 4-protocol streaming `ReasoningParser` (`.tagPair`, `.gemmaChannel`, `.harmonyChannel`, `.recipientChannel`) with dynamic `StripSpecials` and mid-stream `forceEndReasoning()`.
  - **Streaming GGUF Inspection, Non-Destructive TOML & Observability** (`HeylookConfig.swift`): `64 MB` streaming `GGUFHeaderReader` parsing past `256k+` vocab arrays without truncation (including non-causal `clip.projector_type`, hybrid-recurrent SSM, and built-in MTP metadata), non-destructive `heylook.toml` (`addScanFolder` / `removeScanFolder`) & override-only `model.heylook.toml` sidecars, and `HeylookObservabilityLog` (`level` threshold filtering, `retentionLimit` ring pruning, JSONL event persistence).
  - **Unified SQLite3 WAL Store** (`SQLiteStore.swift`): Zero-dependency `SQLiteStore` (`WAL` mode) with SHA-256 deduplicated `media_blobs` + orphan GC, multimodal `content_blocks_json` hydration, `message_stats` + `queryPerformanceProfile` (`PerformanceProfileSummary`), turn-level `cloneConversation(id:upToPosition:)`, single-turn `deleteMessage` position compaction, `COLLATE NOCASE` unique `presets`, `clearAllData()`, and live `mutationStream()` sync.

- **Dual-Lane Compute Scheduler & Modular Providers (`HeylookEngines`)**:
  - **`ComputeScheduler` & `StreamingPipeline`** (`ComputeScheduler.swift`, `Providers/StreamingPipeline.swift`): FIFO Metal GPU `GenerationGate` (`max_queue_depth`) paired with a concurrent `ANELane` so ANE-resident towers run alongside pinned GPU models without LRU eviction; eager `provider.preload()` warmup and deterministic `provider.unload()` reclamation; unified `StreamingPipeline` with multi-chunk `StopSequenceFilter` holdback, `thinkingBudget` auto-`forceEndReasoning()`, and immediate `RequestRegistry.addCancelHandler` abort; and truthful runtime defaults (`allowSyntheticFallback = false`).
  - **`MLXSwiftProvider` & `MLXModelConverter`** (`Providers/MLXSwiftProvider.swift`): In-process Metal GPU LLM, VLM (with Lanczos `image_max_tokens` downscaling and VLM-to-LLM factory fallback), and `HeylookTransformersTokenizerLoader`; multi-branch `[KVCache]` snapshot cloning/trimming with `isTrimmable` hybrid guards (`MambaCache` / `ArraysCache`); speculative decoding via `MTPDrafterModelFactory` (`mtpDrafter`) and `LLMModelFactory` (`draftModel`) with `speculativeDecodingTelemetry`; and pure-Swift `.safetensors` quantization (`MLXModelConverter`).
  - **`GGUFSubprocessProvider`** (`Providers/GGUFSubprocessProvider.swift`, macOS): Crash-isolated `llama-server` subprocess per loaded model managed by `SubprocessLifecycleRegistry` (`SIGTERM` -> `SIGKILL` + `atexit` teardown) on OS-socket-probed loopback ports (`GGUFPortPool`, `18080+`), dynamic unified-memory `assessRAMFit` (`RAMFitAssessment`) with automatic external drafter drop under tight Metal headroom and drafter-less retry on launch failure, media-blind sidecar fallback when `--mmproj` is present, `--reasoning-preserve` / `--ctx-checkpoints` / `--cache-reuse` / `json_schema` / `stop`, and live `GET /props` provenance upgrades.
  - **`CoreAIProvider`** (`Providers/CoreAIProvider.swift`): In-process execution of Schema `0.1` and `0.2` `.aimodel` bundles (`CoreAIBundleManifest` + `CoreAILanguageModels` `LanguageModelBundle` / `CoreAIRunner`) with `dflash` / `mtp` `SpeculativeReport` telemetry. Its `transcribeAudio` path validates `16 kHz` mono PCM + `128`-bin HTK log-mel frames on the ANE lane and truthfully returns `ProviderExecutionError.runtimeUnavailable` until an on-device ASR tower (`CoreAISpeech` / `WhisperKit`) is linked (tracked in Next Horizon `H.1`).
  - **`RemoteHeylookProvider`** (`Providers/RemoteHeylookProvider.swift`): Connects `iOS 27.2+` or secondary macOS peers to a remote `HeylookServer` over HTTP/SSE with Bonjour `_heylook._tcp` auto-discovery (`BonjourServiceBrowser`).

- **Multimodal Embeddings, Audio Resampling & ANE Towers (`HeylookEmbeddings`)**:
  - Indexes `.video` (`16s`-stride overlapping `32s` windows with `1 fps` or content-aware novelty selection + optional soundtrack via SDK 27 async `AVAssetReader.outputProvider(for:)`), `.audio` (`30s` windows with `AVAudioFile` + `AVAudioConverter` resampling to `16 kHz` mono `Float32`), `.image`, `.document` (`PDFKit` pages), `.code`, and `.text`.
  - Pairs `CheckpointPieceTokenizer` (BPE merge-rank + Viterbi Unigram `tokenizer.json` segmentation) and `NativeMLXEmbeddingGemma2` (`MLXSwiftLMEmbeddingGemma2Engine` + `HybridLocalEmbeddingEngine`) with `ANECoreMLRuntime` (`.mlmodelc` Vision/Audio/Text towers), `ANETowerInspector` (`MLComputePlan`), and `VectorStore` (`SQLite3` + `Accelerate` `vDSP` Matryoshka `768d`/`512d`/`256d`/`128d` slicing with truthful `model_id` invalidation).

- **Embedded `/v1/*` HTTP/SSE Server & Security Guards (`HeylookServer`)**:
  - Unified incremental SSE framing (`consumeStreamAndFormatSSE` + `SSEConnectionStreamer`) with `X-Request-ID` validation & echo across:
    - `POST /v1/messages` (streaming & non-streaming Anthropic wire, HTTP `400` `thinking_depth` validation, HTTP `422` multimodal capability gating, `stop_sequences`, `response_format`)
    - `/v1/conversations` CRUD, `POST /v1/conversations/{id}/clone`, `/v1/conversations/{id}/messages` CRUD, `GET /v1/conversations/{id}/media/{media_id}`, `POST` / `DELETE /v1/conversations/{id}/generate` (`append` / `regenerate` / `continue` with shared `RequestRegistry.shared` `409 generation_in_progress` lock, `heylook_saved`, and `dropped_media`), `POST /v1/conversations/{id}/generate/reasoning-end`, `POST /v1/conversations/{id}/voice-generate`, and `GET` / `POST /v1/conversations/{id}/prompt`
    - `/v1/notebooks` CRUD + `POST /v1/notebooks/{id}/generate`, and `/v1/presets` CRUD
    - `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/models/{id}/load`, `POST /v1/models/{id}/unload`, and `POST /v1/models/{id}/image-plan`
    - `/v1/admin/models` (`GET`, `POST`, `POST /v1/admin/models/reload`, `POST /v1/admin/models/unload-all`, `POST /v1/admin/models/validate`, `POST /v1/admin/models/fit`, `GET` / `PATCH /v1/admin/models/scan-config`, `GET /v1/admin/models/model-options`) and `/v1/admin/models/{id}` (`GET`, `PATCH` / `PUT`, `DELETE`, `POST /load`, `POST /unload`, `GET` / `PUT` / `DELETE /chat-template`)
    - `GET` / `PATCH /v1/admin/config`, `DELETE /v1/admin/config/{key}`, and `GET` / `PATCH` / `DELETE /v1/admin/config/logs`
    - `GET /v1/performance/profile/{time_range}`, `GET /v1/system/metrics`, `GET /v1/capabilities`, `GET /v1/status` (`/health`, `/v1/health`), `POST /v1/embeddings`, `POST /v1/cache/clear`, `POST /v1/data/clear`, `POST /v1/telemetry/events`, `DELETE /v1/requests/{id}`, and `POST /v1/requests/{id}/reasoning-end`
  - Enforces strict `Host` DNS-rebinding rejection on both loopback (`127.0.0.1`) and wildcard (`0.0.0.0`), `X-Heylook-Admin-Token` (`401`/`403`) admin authentication, and `HEYLOOK_READONLY_MODEL_CONFIG` (`409`) write protection.

---

## 6-Surface Multiplatform SwiftUI Workbench (`HeylookUI` & `HeylookApp`)

Built on a unified 3-region `NavigationSplitView` + trailing `.inspector(isPresented:)` (`HeylookWorkbenchView.swift`) with cached `@Observable` projections (`activeModelDescription`, `sortedActiveGGUFPorts`), item-driven `DestructiveActionPrompt` confirmations, `#available(macOS 27.3, iOS 27.3, *)` `ToolbarContainerBehavior27_3Modifier`, and live `SQLiteStore.mutationStream()` + `RequestRegistry.shared` synchronization:

1. **`⌘1` Chat** (`Surfaces/ChatWorkbenchSurface.swift`, `Surfaces/MarkdownMessageView.swift`): Streaming GFM Markdown with fenced code-block copy, `followingTail` auto-scroll, `ThinkingTraceCardView` (`End thinking`), persistent turn media (`media_blobs`), `⇧⌘V` clipboard image paste + live `image-plan` token/resize pill, per-turn `Save` / `Save & Regenerate` / `Save & Continue` / `Regenerate` / `Fork`, highlighted `PromptMarkerSpan` prompt preview, and structured `ResponseFormat` (`Text` / `JSON Object` / `JSON Schema`).
2. **`⌘2` Notebook** (`Surfaces/NotebookAndEmbeddingsSurfaces.swift`, `Surfaces/CursorAwareTextEditor.swift`): Cursor-aware inline streaming (`prefix + delta + suffix`), selection-gated `"Analyze Selection"`, `"Insert Summary"`, `"Continue at Cursor"`, per-notebook `systemPrompt` / `params` / `appliedPresetID` persistence, `Stop` abort, and `500 ms` debounced auto-save.
3. **`⌘3` Presets** (`Surfaces/PresetsSurface.swift`): Master-detail CRUD across `sampler` and `full` scopes (`nil`-means-cascade optional field overrides, `Create Preset` / `Save Changes`, duplicate, apply, confirmed `Delete…`) paired with two-verb `Update` / `Save as…` controls in the inspector.
4. **`⌘4` Embeddings & Media Studio** (`Surfaces/NotebookAndEmbeddingsSurfaces.swift`): Local ANE/SQLite Media Finder (`HitRowView`, `EmbeddingsInspectorView` with inline `AVPlayerView` / `VideoPlayer` `KeyframeStamp` seek pills & `ScoredFeatureBreakdownCard`) + `≤16`-item N×N Multimodal Cosine Matrix (`EmbeddingsCompareMatrixView` with live `768`/`512`/`256`/`128d` subspace re-slicing, CSV/JSON export, and `Send Pair to Retrieval Lab`) + Multi-Config `GPU / Hybrid / ANE` Retrieval Lab sweeps (`EmbeddingsMultiConfigLabView` + `set.json` import) + `EmbeddingsEvalLabDrawer`.
5. **`⌘5` Models & Inspector** (`Surfaces/ModelsAndSystemSurfaces.swift`): `heylook.toml` `[scan] folders` editor, live `RAMFitAssessment` headroom badges + quick `ctx_size` reload picker, override-only `ModelSidecarEditorSheet` (`model.heylook.toml`), 4-rung `ChatTemplateInspection` + live Jinja2 syntax validation (`chat_template.heylook.jinja`), and pure-Swift `MLXModelConverterSheet`.
6. **`⌘6` Perf, Observability & Server** (`Surfaces/ModelsAndSystemSurfaces.swift`, `Sources/HeylookApp/HeylookApp.swift`): Live Mach RAM used/available & CPU %, real `MLComputePlan` ANE op %, `1h` / `6h` / `24h` / `7d` `PerformanceProfileSummary` tables, `HeylookObservabilityLog` JSONL viewer, active `llama-server` ports (`sortedActiveGGUFPorts`), Bonjour `_heylook._tcp` LAN scanner, and macOS `MenuBarExtra` (`Open Workbench`).

---

## Quick Start

```bash
# Fast offline SwiftPM build & 40-test verification suite
# (Prefix with `xcrun` if Homebrew `swift` precedes `/usr/bin/swift` in $PATH)
xcrun swift build
xcrun swift test
xcrun swift run heylook-cli eval-lab

# (Optional) One-command environment, Xcode, Metal, iOS Simulator, skills & swift-lsp updater
./scripts/update_dev_env.sh

# Full native Xcode build & test (compiles Cmlx default.metallib + links mlx-swift, mlx-swift-lm & swift-transformers)
./scripts/build_native.sh build
./scripts/build_native.sh test
./scripts/build_native.sh app

# Inspect host unified memory, discovered models, and ANE tower compute plans
./scripts/build_native.sh cli doctor
./scripts/build_native.sh cli status
./scripts/build_native.sh cli models
./scripts/build_native.sh cli inspect-ane ./models/embeddinggemma-2-MLX-BF16
./scripts/build_native.sh cli inspect-checkpoint ./models/embeddinggemma-2-MLX-BF16

# Quantize a safetensors checkpoint in pure Swift (affine, mxfp4, mxfp8, nvfp4; standard or q4_zero calibration)
./scripts/build_native.sh cli convert-mlx ./models/raw-bf16 ./models/quant-4bit --bits 4 --group-size 64 --mode affine

# Export Core AI (.aimodel) bundles (macOS GPU 4bit/FP4/FP8 or iOS ANE 4-bit palettized)
./scripts/export_coreai.sh list --type llm
./scripts/export_coreai.sh llm Qwen/Qwen3.5-9B
./scripts/export_coreai.sh gemma4 google/gemma-4-E4B-it --include-vision

# Stream live Metal GPU inference from a configured or repo-relative MLX checkpoint
./scripts/build_native.sh cli chat --model ./models/Qwen3.5-0.8B-MLX-8bit --max-tokens 96 "Explain unified memory + ANE co-residency."

# Index media/docs with hybrid ANE + MLX EmbeddingGemma 2, list watched folders, and search with Matryoshka attribution
./scripts/build_native.sh cli index ./docs --sampling content --audio-from-video --compute hybrid_ane --db /tmp/heylook-vectors.sqlite
./scripts/build_native.sh cli search "dual-lane GPU and ANE co-residency scheduler" --dim 768 --task query --db /tmp/heylook-vectors.sqlite
./scripts/build_native.sh cli watched

# Start the embedded /v1/* HTTP & SSE server (default port 8000)
./scripts/build_native.sh cli serve 8000
```

---

## Documentation & LLM Knowledge Wiki

| Document | Purpose |
| :--- | :--- |
| **[docs/USER_GUIDE.md](./docs/USER_GUIDE.md)** | Operator guide to all 6 Workbench surfaces (`⌘1`–`⌘6`), `heylook.toml` setup, `heylook-cli`, and `/v1/*` API usage. |
| **[VISION.md](./VISION.md)** | Core principles (*Visibility*, *Derive, Don't Copy*, *Truthful Runtime*, *Heterogeneous Compute*) and project boundaries. |
| **[AGENTS.md](./AGENTS.md)** | Operating guide, module table, and **LLM Knowledge Wiki Index Router**. |
| **[docs/wiki/README.md](./docs/wiki/README.md)** | Master hub for the 8-page **LLM Knowledge Wiki**:<br>• [architecture_overview.md](./docs/wiki/architecture_overview.md)<br>• [engine_contract_and_provenance.md](./docs/wiki/engine_contract_and_provenance.md)<br>• [providers_and_heterogeneous_compute.md](./docs/wiki/providers_and_heterogeneous_compute.md)<br>• [multimodal_embeddings_and_ane.md](./docs/wiki/multimodal_embeddings_and_ane.md)<br>• [wire_api_and_sqlite_store.md](./docs/wiki/wire_api_and_sqlite_store.md)<br>• [model_family_onboarding.md](./docs/wiki/model_family_onboarding.md)<br>• [ui_architecture_and_macos27.md](./docs/wiki/ui_architecture_and_macos27.md)<br>• [sharp_edges.md](./docs/wiki/sharp_edges.md) |
| **[docs/project/CURRENT.md](./docs/project/CURRENT.md)** & **[TODO.md](./docs/project/TODO.md)** | Subsystem readiness matrix (`SOLID`), completed **Phases 0–14** roadmap, and **Next Horizons (`H.1`–`H.4`)**. |
