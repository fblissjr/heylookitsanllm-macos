# AGENTS.md

Operating guide and **LLM Knowledge Wiki Index Router** for `heylookitsanllm-macos` — a native Swift 6.4 (`v6` language mode) multiplatform (`macOS 27.2+` and `iOS 27.2+` deployment floor supporting `27.2.1` hosts and `27.3+` SDK features behind `#available(macOS 27.3, iOS 27.3, *)` guards) personal inference workbench, multimodal media/embedding studio, and embedded HTTP/SSE server for Apple Silicon.

## Where to Look First

- **[VISION.md](./VISION.md)**: Core design principles (*Commodity Upstream, Surgical Specialization*, *Runtime-Agnostic Core*, *Truthful Runtime*, *Derive, Don't Copy*, *Heterogeneous Silicon Harmony*, *Intimate in the Hand, Effortless at the Desk*, *Sovereign by Construction*). Breaks ties when a design choice is unclear.
- **[docs/USER_GUIDE.md](./docs/USER_GUIDE.md)**: Complete operator guide to the 6-surface Workbench app, `heylook.toml` setup, `heylook-cli`, and `/v1/*` API usage.
- **[docs/wiki/README.md](./docs/wiki/README.md)**: Master entrypoint to the **LLM Knowledge Wiki**.
- **[docs/project/CURRENT.md](./docs/project/CURRENT.md)** and **[docs/project/TODO.md](./docs/project/TODO.md)**: Honest subsystem audit matrix against `heylookitsanllm` `v2.0.251` and the Parity Completion Roadmap (**Phases 16A–16E**) + Post-Parity Horizons.

## LLM Knowledge Wiki Router (Read Before Editing an Area)

Before modifying a subsystem, read the corresponding wiki page from `docs/wiki/`:

| Area / Subsystem | File Glob / Targets | Knowledge Wiki Page |
| :--- | :--- | :--- |
| **System Topology & Invariants** | `Package.swift`, `Sources/HeylookApp/**`, `Sources/HeylookCLI/**` | [docs/wiki/architecture_overview.md](./docs/wiki/architecture_overview.md) |
| **Engine Contract, Provenance, Samplers & Templates** | `Sources/HeylookCore/Engine/**`, `Sources/HeylookCore/Sampling/**`, `Sources/HeylookCore/Parsing/**`, `Sources/HeylookCore/Config/**` | [docs/wiki/engine_contract_and_provenance.md](./docs/wiki/engine_contract_and_provenance.md) |
| **Providers, Streaming Pipeline & Dual-Lane Compute** | `Sources/HeylookEngines/**` | [docs/wiki/providers_and_heterogeneous_compute.md](./docs/wiki/providers_and_heterogeneous_compute.md) |
| **Multimodal Embeddings, ANE Towers & Matryoshka Store** | `Sources/HeylookEmbeddings/**` | [docs/wiki/multimodal_embeddings_and_ane.md](./docs/wiki/multimodal_embeddings_and_ane.md) |
| **HTTP/SSE Wire Contract & SQLite WAL Store** | `Sources/HeylookServer/**`, `Sources/HeylookCore/Wire/**`, `Sources/HeylookCore/Store/**` | [docs/wiki/wire_api_and_sqlite_store.md](./docs/wiki/wire_api_and_sqlite_store.md) |
| **Model Onboarding, Quantization & Export Playbooks** | `Sources/HeylookCore/Config/ModelTemplateInfo.swift`, `scripts/**` | [docs/wiki/model_family_onboarding.md](./docs/wiki/model_family_onboarding.md) |
| **SwiftUI Architecture & macOS 27.2+ Design System** | `Sources/HeylookUI/**`, `Sources/HeylookApp/**` | [docs/wiki/ui_architecture_and_macos27.md](./docs/wiki/ui_architecture_and_macos27.md) |
| **Sharp Edges & Incident Rationales** | Cross-cutting (`HeylookCore`, `HeylookEmbeddings`, `HeylookEngines`, `HeylookServer`, `HeylookUI`) | [docs/wiki/sharp_edges.md](./docs/wiki/sharp_edges.md) |

## Module Architecture

| Target | Platforms | Key Files & Role |
| :--- | :--- | :--- |
| `HeylookCore` | `macOS 27.2+`, `iOS 27.2+` | `EngineDescription` / `ProvenanceField` & wire `EngineContract` (`Fact<T>`, `SettingFact`, zero-path `PublicModelWireRow`, explicit `AdminModelWireRow`), `ProviderConfigSchema` (13 derived sampler fields, `422` `file_only` guard), comment-preserving `TOMLCodec`, 5-layer `SamplerCascade` (`1.0 / 0.95 / 16384`, `VENDOR_SAMPLING_KEYS`), `HeylookJinjaRenderer` (`checkPrefixStability`) & `ModelTemplateInfo` (`garbageDepthSentinel` probe, 4-protocol `ReasoningParser`, dynamic `StripSpecials`), `HeylookConfig` (`heylook.toml`, `model.heylook.toml` with `image_max_tokens`, 4+ rung template ladder, 64 MB `GGUFHeaderReader`), `ObservabilityDiskLog` + `HeylookObservabilityLog`, non-compacting `SQLiteStore` (`WAL`, scoped 64-hex `media_blobs`, unseeded `presets`, `message_stats`, `mutationStream()`), and `MessagesWire` / `WireDTOs`. |
| `HeylookEmbeddings` | `macOS 27.2+`, `iOS 27.2+` | `Artifact` & `ArtifactSpan`, `VectorStore` (`SQLite3` + `vDSP` Matryoshka `768d`/`512d`/`256d`/`128d` + truthful `model_id` invalidation), `EmbeddingModelFacts` (`PromptCatalog` from `config_sentence_transformers.json`), `MLXSwiftLMEmbeddingGemma2Engine` (1,024-token padded batching, true `hybrid_ane` ANE->GPU `softTokens` bridge without `TextANE.mlmodelc`, consecutive single-window `windows.seconds` video decoding), `GGUFEmbeddingEngine` (`llama-server --embeddings`), `ArtifactProcessor` & `WindowPlanner`, `CheckpointPieceTokenizer`, `FolderWatcher`, `IndexCoordinator`, `EmbeddingRequestLimits`, `ModelComputeOptions`, `ANECoreMLRuntime`, `HybridLocalEmbeddingEngine`, and `ANETowerInspector`. |
| `HeylookEngines` | `macOS 27.2+`, `iOS 27.2+` | `ComputeScheduler` (`InferenceProvider` protocol, `GenerationGate` + `ANELane` + `withGPUEmbeddingLease` + `409 model_busy`), `RAMFitCalculator` (`thinHeadroomGB = 24.0`, MLX hard fail vs GGUF warn, `sysctlSuggestMB`), `ModelSpecialization` (`CheckpointSignature`, `Gemma4Specialization` `48px` & `backbone_hidden_size`, `HybridRecurrentSWASpecialization`, `StandardAttentionSpecialization`), `SpeechTranscriptionService` (`CoreAISpeech` / `SpeechTranscriber` + `decodeRIFFChunksMonoPCM`), `StreamingPipeline` (`StopSequenceFilter` reply-only cut), `MLXRuntimeSupport` (`PreRenderedVLMTokenizerProxy`, `VisionFeatureCache`, `APCSnapshotCache` + `mediaSalt`, `ThinkingControlLogitProcessor`, `LeviathanMTPTokenIterator`), `MLXSwiftProvider` & `MLXModelConverter`, `CoreAIProvider`, `GGUFSubprocessProvider` (`GGUFSpawnConfig` SWA `--cache-reuse` suppression, `CompletionIDBox`), `RemoteHeylookProvider`, and `SubprocessSupport`. |
| `HeylookServer` | `macOS 27.2+`, `iOS 27.2+` | `HeylookHTTPServer`, `Orchestration/` (`WorkbenchBackend`, `WorkbenchService`, `ConversationOrchestrator` atomic tail saga + `SubscriberHub` + `409 generation_in_progress`, `RAMFitEvaluator`, `ChunkTelemetryAccumulator`), `Transport/` (`HTTPConnectionHandler` Keep-Alive + chunked, `SSEConnectionStreamer` `5s` `event: ping`, `SSEWirePipeline`), `Routing/` (`PatternRouter` greedy `*id`, `ErrorMapping`), `Routes/` (7 route modules incl. `DELETE ?after=`, scoped `nosniff` media, optional `HEYLOOK_FRONTEND_DIR` `StaticFrontendRoutes`), and `Discovery/` (`BonjourServiceBrowser` `_heylook._tcp`). |
| `HeylookUI` | `macOS 27.2+`, `iOS 27.2+` | Thin `HeylookWorkbenchViewModel` sharing one `WorkbenchService` with `HeylookRouter`, `Controllers/` (`DocumentSettingsController` separate `chatSettings`/`notebookSettings`, `RemoteWorkbenchBackend`, `ChatController`, `NotebookAndPresetsController`, `EmbeddingsStudioController`, `ModelsAndPerfController` zero 3s polling), `HeylookRootView` (`AdaptiveContrastStyle`, layered `Escape`, `@Environment(\.scenePhase)` flush, `ToolbarContainerBehavior27_3Modifier`), and modular 6-surface views under `Surfaces/` (`⌘1` Chat, `⌘2` Notebook, `⌘3` Presets, `⌘4` Embeddings & ANE, `⌘5` Models, `⌘6` Perf & Server). |
| `HeylookApp` | `macOS 27.2+`, `iOS 27.2+` | Native desktop & mobile app entry point (`@Environment(\.scenePhase)` flush) + macOS `MenuBarExtra` server & model lifecycle controller. |
| `heylook-cli` | `macOS 27.2+` | Unified CLI (`HeylookCLI.swift` + `EvalLabSuite.swift` `CLIEvalLabSuite` with zero synthetic fallback: `serve`, `status`, `doctor`, `models`, `chat`, `transcribe`, `index`, `search`, `watched`, `inspect-ane`, `inspect-checkpoint`, `convert-mlx`, `eval-lab`). |
| `HeylookTests` | `macOS 27.2+`, `iOS 27.2+` | Swift Testing (`@Suite`, `@Test`) 28-test modular unit and contract suite across 5 files (`EngineContractTests.swift`, `SamplerAndTemplateTests.swift`, `MessagesAndSagaContractTests.swift`, `EmbeddingsAndSpecializationTests.swift`, `WorkbenchControllersTests.swift`) + `Support/TestStubProviders.swift`. |

## Commands

- Fast SwiftPM build & unit/contract test suite:
  - `swift build`
  - `swift test`
  - Sandboxed test execution (when `/var/folders` module cache is restricted):
    `GIT_CONFIG_GLOBAL=/dev/null TMPDIR=/tmp CLANG_MODULE_CACHE_PATH=/tmp/clang-cache swift test --disable-sandbox --scratch-path /tmp/heylook-build --cache-path /tmp/heylook-spm-cache -Xswiftc -module-cache-path -Xswiftc /tmp/swift-module-cache`
- CLI commands:
  - `swift run heylook-cli doctor`
  - `swift run heylook-cli status`
  - `swift run heylook-cli models`
  - `swift run heylook-cli index <path> [--sampling fps|content|off] [--audio-from-video] [--compute all_gpu|hybrid_ane|all_ane] [--db <path>]`
  - `swift run heylook-cli search "<query>" [--dim 768|512|256|128] [--task query|similarity|code|qa|fact] [--media <path>] [--db <path>]`
  - `swift run heylook-cli watched`
  - `swift run heylook-cli inspect-ane [dir]`
  - `swift run heylook-cli inspect-checkpoint [dir]`
  - `swift run heylook-cli convert-mlx <src-dir> <dst-dir> [--bits 4|8] [--group-size 32|64] [--mode affine|mxfp4|mxfp8|nvfp4] [--calibration standard|q4_zero] [--overwrite] [--dry-run]`
  - `swift run heylook-cli chat [--model <id_or_path>] [--max-tokens <n>] [--no-thinking] "<prompt>"`
  - `swift run heylook-cli transcribe <audio-path> [--model <id_or_path>] [--lang <locale>]`
  - `swift run heylook-cli eval-lab`
  - `swift run heylook-cli serve 8000`
- Model conversion, export & agent skills sync scripts:
  - One-command environment, Xcode, Metal, iOS Simulator, skills & `swift-lsp` plugin updater (set `MLX_SWIFT_LM_DIR`, `COREAI_REPO_DIR`, and `HEYLOOK_SKILLS_PLUGIN_URL` when overriding remote defaults): `./scripts/update_dev_env.sh [--upgrade-xcode] [--update-packages] [--build]`
  - Sync Apple Xcode 27.x (`xcrun agent skills export`), Xcode `AdditionalDocumentation`, and `coreai-models` skills (`COREAI_REPO_DIR`) into `.agents/skills/`: `./scripts/sync_agent_skills.sh`
  - Pure-Swift MLX quantization (`MLXLMCommon.convert`): `./scripts/build_native.sh cli convert-mlx <src-dir> <dst-dir> --bits 4 --group-size 64 --mode affine`
  - Core AI `.aimodel` export (`coreai-models` + `xcrun coreai-build`): `./scripts/export_coreai.sh [list|llm|gemma4|whisper|parakeet|embedding|compile] ...`
  - 3-Tower CoreML ANE `EmbeddingGemma 2` export (`coremltools` + `xcrun coremlcompiler`): `uv run --python 3.12 --with coremltools --with safetensors --with "torch<2.8" --with "numpy<2" python ./scripts/export_ane_coreml.py --compile`
- Full native Xcode build (compiling Metal shaders `default.metallib` + linking `mlx-swift`, `mlx-swift-lm`, `swift-transformers`, and optional `coreai-models` via remote GitHub packages, plus TCC-hardened `Info.plist` and local `Apple Development` identity detection with ad-hoc fallback):
  - `./scripts/build_native.sh build`
  - `./scripts/build_native.sh test`
  - `./scripts/build_native.sh app`
  - `./scripts/build_native.sh cli <subcommand>`

## Repo Rules

1. **Remote SwiftPM Dependencies Only in `Package.swift`**: Always declare external dependencies using canonical remote `.package(url: "https://github.com/...", ...)` references (`mlx-swift`, `mlx-swift-lm`, `swift-transformers`, optional `coreai-models`), never local filesystem paths (`.package(path: ...)`), and never reference phantom packages not in `Package.swift`.
2. **Path Privacy**: Never commit personal home-directory paths, relative sibling workspace traversals, or personal full names in tracked files. Use repo-relative paths, `FileManager.default.homeDirectoryForCurrentUser`, or environment variables (`HEYLOOK_CONFIG`, `HEYLOOK_MODELS_DIR`, `HEYLOOK_DB_PATH`, `HEYLOOK_READONLY_MODEL_CONFIG`, `COREAI_REPO_DIR`, `MLX_SWIFT_LM_DIR`, `HEYLOOK_SKILLS_PLUGIN_URL`).
3. **Truthful Runtime (No Silent Synthetic Fallbacks)**: Production providers, embedding engines, and evaluation suites must never silently fabricate output or masquerade hash vectors as real model embeddings (`allowSyntheticFallback` and `DeterministicHashEmbeddingEngine` have been completely removed from production `Sources/`).
4. **Derive, Don't Copy**: Derive capabilities, sampler schemas, and provenance fields from single canonical Swift definitions and model files.
5. **No Migration Code**: Solo deployment. Schema updates in `SQLiteStore` and `VectorStore` use idempotent `CREATE TABLE IF NOT EXISTS` or clean table recreation.
6. **Git Hygiene**: Stage files explicitly by name (`git add <file>`), never `git add -A` or `git add .`.
7. **Read-Only Model Config Mode**: Use `HEYLOOK_READONLY_MODEL_CONFIG=1` when running read-only (`409` on model/config/template writes).
