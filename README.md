# heylookitsanllm-macos (`heylook`)

Native Swift 6.2 personal inference workbench, multimodal media/embedding studio, and embedded HTTP/SSE server for Apple Silicon (`macOS 27.2+` and `iOS 27.2+`), designed for heterogeneous compute across **Metal GPU (`MLX Swift`)**, **Apple Neural Engine (`CoreAI` / `CoreML` / `Argmax`)**, and **`GGUF` (`llama-server`)**.

- **[User Guide (`docs/USER_GUIDE.md`)](./docs/USER_GUIDE.md)** — How to configure `heylook.toml`, drive all 5 workbench surfaces, index & search media, quantize models, and use `heylook-cli` and the `/v1/*` API.
- **[LLM Knowledge Wiki (`docs/wiki/README.md`)](./docs/wiki/README.md)** — Deep-dive subsystem architecture, engine contracts, and sharp edges.

---

## Key Capabilities

- **Unified 3-Tier Engine Contract + Remote Sync (`HeylookEngines`)**:
  - **`MLXSwiftProvider`** (`Providers/MLXSwiftProvider.swift`): In-process Metal GPU LLM, VLM (`CIImage` inputs with automatic VLM-to-LLM factory fallback), and `EmbeddingGemma 2` inference (`mlx-swift` + `mlx-swift-lm` with `default.metallib` & `#huggingFaceTokenizerLoader()`), file-derived `ModelTemplateInfo` (`extraEOSTokens`, `eosTokenIds`, and `reasoning_effort`/`reasoning_strength` depth controls), 4-protocol `ReasoningParser` (`.tagPair`, `.gemmaChannel`, `.harmonyChannel`, `.recipientChannel`), stateful multi-turn `[KVCache]` slicing (`MLXLMCommon.generate(input:cache:parameters:context:)` + `PrefixCacheTracker`), pure-Swift weight quantization (`MLXModelConverter`), and deterministic `unload()` (`MLX.Memory.clearCache()`).
  - **`CoreAIProvider`** (`Providers/CoreAIProvider.swift`): In-process execution of Schema `0.1` and `0.2` `.aimodel` bundles (`CoreAILanguageModels` `LanguageModelBundle` / `CoreAIRunner`) and `16 kHz` ANE speech frontend (`transcribeAudio`) on the Apple Neural Engine and GPU with `dflash` / `mtp` `SpeculativeReport` telemetry.
  - **`GGUFSubprocessProvider`** (`Providers/GGUFSubprocessProvider.swift`, macOS): Crash-isolated `llama-server` subprocess per loaded model (`-np 1`) on dynamic loopback ports (`GGUFPortPool`, `18080+`) managed by `SubprocessLifecycleRegistry` (`SIGTERM` -> `SIGKILL` + `atexit` teardown on unload, LRU eviction, or server stop), with 192 GB auto-sizing (`-ub 2048`, `-cram`, and hybrid-recurrent SSM `--cache-ram 8192`), non-causal vision projector (`clip.projector_type`) `--image-max-tokens <= ubatchSize` clamping, 4-rung `chat_template` ladder (`chat_template_path` > `chat_template.heylook.jinja` > `chat_template.jinja` > `gguf_embedded`), dynamic `chat_template_kwargs` (`enable_thinking` + `reasoning_effort`/`reasoning_strength`), automatic `[MTP]` speculative drafter pairing (`dspark`, `-MTP-`, `-assistant`, and built-in `*.nextn_predict_layers`), `--mmproj` vision/audio projector discovery, multi-turn `reasoning_content` forwarding, and mid-stream `/v1/chat/completions/control` `reasoning_end`.
  - **`RemoteHeylookProvider`** (`Providers/RemoteHeylookProvider.swift`): Connects the native `iOS 27.2+` app (or secondary Mac) to a Mac Studio `HeylookServer` over HTTP/SSE with Bonjour `_heylook._tcp` auto-discovery (`BonjourServiceBrowser`) and full `SamplerSettings` + `thinkingDepth` forwarding alongside local on-device ANE/MLX models.
- **Truthful Runtime & Dual-Lane Compute Scheduler (`ComputeScheduler`)**:
  - Production providers and embedding engines never fabricate silent synthetic outputs (`allowSyntheticFallback` defaults to `false`).
  - Serializes heavy Metal GPU generation through a FIFO `GenerationGate` (`max_queue_depth`, HTTP `503 model_overloaded` on overflow), automatically invokes `provider.unload()` on unpinned GPU LRU eviction or `unloadAllModels()`, and lets ANE-resident models (`EmbeddingGemma 2` ANE towers, `CoreAI` ANE `.aimodel` assets, speech models) run concurrently on the Neural Engine without evicting pinned 192 GB GPU models.
- **5-Layer Deterministic Sampler Cascade (`SamplerCascade`)**:
  - Resolves `builtin` -> `vendor generation_config.json` / GGUF `general.sampling.*` -> `model.heylook.toml` -> `preset` -> `request` with per-field provenance (`derived`, `configured`, `observed`, `request`).
- **Folded Multimodal Embeddings & Media Finder (`HeylookEmbeddings`)**:
  - Integrates `mbed` at full fidelity: indexes `.video` (`16s`-stride overlapping `32s` windows with fixed `1 fps` or content-aware novelty selection + optional 16 kHz soundtrack), `.audio` (`30s` windows), `.image`, `.document` (`PDFKit` pages), `.code`, and `.text` into a SQLite (`WAL`) + `Accelerate` (`vDSP`) Matryoshka (`768d`/`512d`/`256d`/`128d`) vector store with per-tower ANE/GPU compute placement (`ANECoreMLRuntime`).
- **Embedded `/v1/*` Server (`HeylookServer`)**:
  - Serves unified incremental SSE (`consumeStreamAndFormatSSE` + `SSEConnectionStreamer`) and non-streaming `POST /v1/messages` (with HTTP `400` `thinking_depth` validation and HTTP `422` multimodal capability gating), disconnect-resilient `POST /v1/conversations/{id}/generate` & `POST /v1/conversations/{id}/voice-generate` (`event: heylook_saved` & `event: heylook_voice_transcribed`), model-derived `POST /v1/conversations/{id}/prompt` and `POST /v1/models/{id}/image-plan`, `POST /v1/embeddings` (structured `content`/`prompt_name`/`windows` wire + `EmbeddingRequestLimits` HTTP `422` validation), `GET /v1/capabilities`, `GET /v1/models`, `/v1/admin/models/*`, `/v1/conversations/*`, `/v1/notebooks/*`, `/v1/presets/*`, `/v1/requests/{id}/reasoning-end`, and `/v1/system/metrics`.
- **5-Surface Multiplatform SwiftUI Workbench (`HeylookUI` & `HeylookApp`)**:
  - **Unified 3-Region `NavigationSplitView` + Native `.inspector`**: Single `NavigationSplitView` with contextual sidebar sections (`HeylookSidebarView`), a full-bleed center stage (`HeylookDetailColumnView`), and a native trailing `.inspector(isPresented:)` drawer—with cached `@Observable` projections (`activeModelDescription`, `sortedActiveGGUFPorts`), `28x28` minimum hit targets, item-driven destructive confirmation dialogs (`DestructiveActionPrompt`), and `#available(macOS 27.3, iOS 27.3, *)` Golden Gate toolbar container behavior.
  1. **Chat Workbench** (`Surfaces/ChatWorkbenchSurface.swift`): `ChatTopControlBar` (`Preset` menu, `Thinking` toggle, model-derived `Depth` picker `low`/`medium`/`high`/`max`, `Prompt Preview` toggle), `ChatPromptPreviewDrawer`, `LoadAndSamplerInspectorView` with per-slider provenance pills, `ChatComposerBar` (Liquid Glass container with `NSOpenPanel`/`.fileImporter` attachments, `ChatAttachmentPillView` `~tok` badges, and hands-free ANE voice turn handoff), collapsible Thinking Block with live **End Thinking Early**, inline message edit + **Save & Continue**, and `Cmd+Return` / `Esc` / `Cmd+N` / `Cmd+Shift+P` shortcuts.
  2. **Notebook** (`Surfaces/NotebookAndEmbeddingsSurfaces.swift`): Continuous scratchpad buffer with contextual sidebar notebook switcher, **Insert Summary**, **Analyze Selection**, and **Continue Writing** (`Cmd+Return`).
  3. **Embeddings & Media Finder Studio** (`Surfaces/NotebookAndEmbeddingsSurfaces.swift`): Unified 3-region Media Finder with `EmbeddingsSidebarSections` (modality filter, watched folders, Matryoshka dimension & task pickers, per-tower ANE/GPU compute placement & `ComputeMemorySplitBar`), `EmbeddingsSearchHeaderBar` (`ScoringPipelineBanner` glass pills + multimodal query attachment), `HitRowView` results with non-blocking `AsyncImage` thumbnails, `EmbeddingsInspectorView` (`ScoredFeatureBreakdownCard`, `AVPlayerView`/`VideoPlayer` with `KeyframeStamp` seek pills, code/text preview, **Compare-to-Lab Bridge**), and a collapsible bottom `EmbeddingsEvalLabDrawer` (`EvalLabSuite` `Recall@K` / `MRR` & pairwise cosine compare).
  4. **Models & Engine Inspector** (`Surfaces/ModelsAndSystemSurfaces.swift`): `ModelsTopActionBar` (`Quantize / Convert MLX…`, `Re-scan`, `Unload All Models…` confirmation), `ModelSidecarEditorSheet` for `model.heylook.toml` + interactive 4-rung `chat_template.heylook.jinja` override editor, `MLXModelConverterSheet` for pure-Swift `.safetensors` quantization, Pin / Load / Unload controls, `EngineDescription` provenance table, and per-tower ANE residency inspector.
  5. **Performance, Memory & Server Control** (`Surfaces/ModelsAndSystemSurfaces.swift`): 192 GB Unified Memory & Metal wired limit gauge, ANE vs GPU residency bar, active `llama-server` subprocesses (`sortedActiveGGUFPorts`), embedded server toggle, and Bonjour `_heylook._tcp` LAN scanner.

---

## Quick Start

```bash
# Fast SwiftPM build & 29-test verification run
# (Use `xcrun swift` if Homebrew `swift` is installed first in $PATH to avoid
# Homebrew's macOS-14.0 Package.swift manifest linker warning against MacOSX27.2.sdk)
xcrun swift build
xcrun swift test
xcrun swift run heylook-cli eval-lab

# (Optional) One-command environment, Xcode, Metal, iOS Simulator, skills & swift-lsp updater
./scripts/update_dev_env.sh

# Full native Xcode build & test (compiles Cmlx default.metallib + links MLX Swift & mlx-swift-lm from GitHub)
./scripts/build_native.sh build
./scripts/build_native.sh test
./scripts/build_native.sh app

# Inspect hardware, 192 GB unified memory, discovered models, and ANE tower compute plans
./scripts/build_native.sh cli doctor
./scripts/build_native.sh cli models
./scripts/build_native.sh cli inspect-ane ~/.heylook/models/embeddinggemma-2-MLX-BF16
./scripts/build_native.sh cli inspect-checkpoint ~/.heylook/models/embeddinggemma-2-MLX-BF16

# Quantize a safetensors checkpoint in 100% pure Swift (affine, mxfp4, mxfp8, nvfp4, q4_zero)
./scripts/build_native.sh cli convert-mlx ./models/raw-bf16 ./models/quant-4bit --bits 4 --group-size 64 --mode affine

# Export Core AI (.aimodel) bundles (macOS GPU 4bit/FP4/FP8 or iOS ANE 4-bit palettized)
./scripts/export_coreai.sh list --type llm
./scripts/export_coreai.sh llm Qwen/Qwen3.5-9B
./scripts/export_coreai.sh gemma4 google/gemma-4-E4B-it --include-vision

# Stream live Metal GPU inference from a configured MLX checkpoint
./scripts/build_native.sh cli chat --model Qwen3.5-0.8B-MLX-8bit --max-tokens 96 "Explain unified memory + ANE co-residency."

# Index media/docs with hybrid ANE + MLX EmbeddingGemma 2 and search with Matryoshka attribution
./scripts/build_native.sh cli index ./docs --compute hybrid_ane --db /tmp/heylook-vectors.sqlite
./scripts/build_native.sh cli search "dual-lane GPU and ANE co-residency scheduler" --dim 768 --db /tmp/heylook-vectors.sqlite

# Start the embedded /v1/* HTTP & SSE server (default port 8000)
./scripts/build_native.sh cli serve 8000
```

---

## Documentation & LLM Knowledge Wiki

| Document | Purpose |
| :--- | :--- |
| **[docs/USER_GUIDE.md](./docs/USER_GUIDE.md)** | Complete guide to the 5-surface Workbench app, `heylook.toml` setup, `heylook-cli`, and `/v1/*` API usage. |
| **[VISION.md](./VISION.md)** | Core principles (*Visibility*, *Derive, Don't Copy*, *Truthful Runtime*, *Heterogeneous Compute*) and project boundaries. |
| **[AGENTS.md](./AGENTS.md)** | Operating guide, module table, and **LLM Knowledge Wiki Index Router**. |
| **[docs/wiki/README.md](./docs/wiki/README.md)** | Master hub for the 8-page **LLM Knowledge Wiki**:<br>• [architecture_overview.md](./docs/wiki/architecture_overview.md)<br>• [engine_contract_and_provenance.md](./docs/wiki/engine_contract_and_provenance.md)<br>• [providers_and_heterogeneous_compute.md](./docs/wiki/providers_and_heterogeneous_compute.md)<br>• [multimodal_embeddings_and_ane.md](./docs/wiki/multimodal_embeddings_and_ane.md)<br>• [wire_api_and_sqlite_store.md](./docs/wiki/wire_api_and_sqlite_store.md)<br>• [model_family_onboarding.md](./docs/wiki/model_family_onboarding.md)<br>• [ui_architecture_and_macos27.md](./docs/wiki/ui_architecture_and_macos27.md)<br>• [sharp_edges.md](./docs/wiki/sharp_edges.md) |
| **[docs/project/CURRENT.md](./docs/project/CURRENT.md)** & **[TODO.md](./docs/project/TODO.md)** | Subsystem readiness matrix (`SOLID`) and completed Phases 0–13 roadmap. |
