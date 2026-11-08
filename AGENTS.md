# AGENTS.md

Operating guide and **LLM Knowledge Wiki Index Router** for `heylookitsanllm-macos` — a native Swift 6.2 multiplatform (`macOS 27.2+` and `iOS 27.2+`) personal inference workbench, multimodal media/embedding studio, and embedded HTTP/SSE server for Apple Silicon.

## Where to Look First

- **[VISION.md](./VISION.md)**: Core design principles (*Visibility*, *Derive, Don't Copy*, *Truthful Runtime*, *Heterogeneous Apple Silicon Co-Residency*). Breaks ties when a design choice is unclear.
- **[docs/wiki/README.md](./docs/wiki/README.md)**: Master entrypoint to the **LLM Knowledge Wiki**.
- **[docs/project/CURRENT.md](./docs/project/CURRENT.md)** and **[docs/project/TODO.md](./docs/project/TODO.md)**: Subsystem readiness matrix and phased roadmap.

## LLM Knowledge Wiki Router (Read Before Editing an Area)

Before modifying a subsystem, read the corresponding wiki page from `docs/wiki/`:

| Area / Subsystem | File Glob / Targets | Knowledge Wiki Page |
| :--- | :--- | :--- |
| **System Topology & Invariants** | `Package.swift`, `Sources/HeylookApp/**`, `Sources/HeylookCLI/**` | [docs/wiki/architecture_overview.md](./docs/wiki/architecture_overview.md) |
| **Engine Contract, Provenance, Samplers & Templates** | `Sources/HeylookCore/Domain/**`, `Sources/HeylookCore/Config/**` | [docs/wiki/engine_contract_and_provenance.md](./docs/wiki/engine_contract_and_provenance.md) |
| **Providers, Streaming Pipeline & Dual-Lane Compute** | `Sources/HeylookEngines/**` | [docs/wiki/providers_and_heterogeneous_compute.md](./docs/wiki/providers_and_heterogeneous_compute.md) |
| **Multimodal Embeddings, ANE Towers & Matryoshka Store** | `Sources/HeylookEmbeddings/**` | [docs/wiki/multimodal_embeddings_and_ane.md](./docs/wiki/multimodal_embeddings_and_ane.md) |
| **HTTP/SSE Wire Contract & SQLite WAL Store** | `Sources/HeylookServer/**`, `Sources/HeylookCore/Wire/**`, `Sources/HeylookCore/Store/**` | [docs/wiki/wire_api_and_sqlite_store.md](./docs/wiki/wire_api_and_sqlite_store.md) |
| **Model Onboarding, Quantization & Export Playbooks** | `Sources/HeylookCore/Domain/ModelTemplateInfo.swift`, `scripts/**` | [docs/wiki/model_family_onboarding.md](./docs/wiki/model_family_onboarding.md) |
| **Sharp Edges & Incident Rationales** | Cross-cutting (`HeylookCore`, `HeylookEmbeddings`, `HeylookEngines`, `HeylookServer`, `HeylookUI`) | [docs/wiki/sharp_edges.md](./docs/wiki/sharp_edges.md) |

## Module Architecture

| Target | Platforms | Key Files & Role |
| :--- | :--- | :--- |
| `HeylookCore` | `macOS 27.2+`, `iOS 27.2+` | `EngineDescription` & `ProvenanceField`, 5-layer `SamplerCascade`, file-derived `ModelTemplateInfo` & `ThinkingControls`, 4-protocol `ReasoningParser` & dynamic `StripSpecials`, `HeylookConfig` (`heylook.toml`, `model.heylook.toml` with `image_max_tokens`, `discoverConfiguredModels()`, 4-rung `chat_template` ladder, `GGUFHeaderReader` with non-causal `clip.projector_type` detection), `SQLiteStore` (`WAL` mode), and `/v1/messages` & `/v1/embeddings` wire types. |
| `HeylookEmbeddings` | `macOS 27.2+`, `iOS 27.2+` | `Artifact` & `ArtifactSpan`, `VectorStore` (`SQLite3` + `vDSP` Matryoshka `768d`/`512d`/`256d`/`128d` + truthful `model_id` invalidation), `ArtifactProcessor` & `WindowPlanner` (`16s`-stride `32s` video windows, `30s` audio windows, `PDFKit` page chunking), `FolderWatcher`, `IndexCoordinator`, `EmbeddingRequestLimits`, `EmbeddingLabRunner`, `ModelComputeOptions`, `ANECoreMLRuntime` + `HybridLocalEmbeddingEngine` (executing compiled Vision/Audio/Text `.mlmodelc` ANE towers), and `ANETowerInspector`. |
| `HeylookEngines` | `macOS 27.2+`, `iOS 27.2+` | `ComputeScheduler` (`GenerationGate` + `ANELane` + `registerDiscoveredModels`), `ProviderFactory` & `StreamingPipeline` (`Providers/StreamingPipeline.swift`), `MLXSwiftProvider` & `MLXModelConverter` (`Providers/MLXSwiftProvider.swift`), `CoreAIProvider` & `CoreAIBundleManifest` (`Providers/CoreAIProvider.swift`), `GGUFSubprocessProvider` & `GGUFSpawnConfig` (`Providers/GGUFSubprocessProvider.swift`), `RemoteHeylookProvider` (`Providers/RemoteHeylookProvider.swift`), and `SubprocessLifecycleRegistry` & `GGUFPortPool` (`Providers/SubprocessSupport.swift`). |
| `HeylookServer` | `macOS 27.2+`, `iOS 27.2+` | `HeylookHTTPServer`, `HeylookRouter` (with unified `consumeStreamAndFormatSSE`), `SSEConnectionStreamer`, `EvalLabSuite`, and `BonjourServiceBrowser` (`_heylook._tcp`) serving `/v1/messages` (with HTTP `400` `thinking_depth` validation & HTTP `422` capability gating), `/v1/conversations/{id}/generate` & `/voice-generate` (`heylook_saved`), `/v1/conversations/{id}/prompt`, `/v1/models/{id}/image-plan`, `/v1/embeddings`, `/v1/capabilities`, `/v1/models`, and `/v1/admin/models/*`. |
| `HeylookUI` | `macOS 27.2+`, `iOS 27.2+` | `HeylookWorkbenchViewModel` (`HeylookWorkbenchViewModel.swift`), `HeylookRootView` (`HeylookWorkbenchView.swift`), and modular surface views under `Surfaces/` (`ChatWorkbenchSurface.swift`, `NotebookAndEmbeddingsSurfaces.swift`, `ModelsAndSystemSurfaces.swift`). |
| `HeylookApp` | `macOS 27.2+`, `iOS 27.2+` | Native desktop & mobile app entry point + macOS `MenuBarExtra` server & model lifecycle controller. |
| `heylook-cli` | `macOS 27.2+` | Unified CLI (`serve`, `status`, `doctor`, `models`, `chat`, `index`, `search`, `watched`, `inspect-ane`, `inspect-checkpoint`, `convert-mlx`, `eval-lab`). |
| `HeylookTests` | `macOS 27.2+`, `iOS 27.2+` | Swift Testing (`@Suite`, `@Test`) 28-test unit, lifecycle, truthful-runtime, and wire-contract verification suite. |

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
  - `swift run heylook-cli eval-lab`
  - `swift run heylook-cli serve 8000`
- Model conversion & export scripts:
  - Pure-Swift MLX quantization (`MLXLMCommon.convert`): `./scripts/build_native.sh cli convert-mlx <src-dir> <dst-dir> --bits 4 --group-size 64 --mode affine`
  - Core AI `.aimodel` export (`coreai-models` + `xcrun coreai-build`): `./scripts/export_coreai.sh [list|llm|gemma4|embedding|compile] ...`
  - 3-Tower CoreML ANE `EmbeddingGemma 2` export (`coremltools` + `xcrun coremlcompiler`): `uv run --python 3.12 --with coremltools --with safetensors --with "torch<2.8" --with "numpy<2" python ./scripts/export_ane_coreml.py --compile`
- Full native Xcode build (compiling Metal shaders `default.metallib` + linking `mlx-swift`, `mlx-swift-lm`, `coreai-models`, and `argmax-oss-swift` via remote GitHub packages):
  - `./scripts/build_native.sh build`
  - `./scripts/build_native.sh test`
  - `./scripts/build_native.sh app`
  - `./scripts/build_native.sh cli <subcommand>`

## Repo Rules

1. **Remote SwiftPM Dependencies Only in `Package.swift`**: Always declare external dependencies using canonical remote `.package(url: "https://github.com/...", ...)` references, never local filesystem paths (`.package(path: ...)`).
2. **Path Privacy**: Never commit personal home-directory paths or personal full names in tracked files. Use repo-relative paths, `FileManager.default.homeDirectoryForCurrentUser`, or environment variables (`HEYLOOK_CONFIG`, `HEYLOOK_MODELS_DIR`, `HEYLOOK_DB_PATH`).
3. **Truthful Runtime (No Silent Synthetic Fallbacks)**: Production providers and embedding engines must never silently fabricate output or masquerade hash vectors as real model embeddings (`allowSyntheticFallback` defaults to `false`).
4. **Derive, Don't Copy**: Derive capabilities, sampler schemas, and provenance fields from single canonical Swift definitions and model files.
5. **No Migration Code**: Solo deployment. Schema updates in `SQLiteStore` and `VectorStore` use idempotent `CREATE TABLE IF NOT EXISTS` or clean table recreation.
6. **Git Hygiene**: Stage files explicitly by name (`git add <file>`), never `git add -A` or `git add .`.
