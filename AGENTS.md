# AGENTS.md

Operating guide for `heylookitsanllm-macos` — a native Swift 6.2 multiplatform (`macOS 27.2+` and `iOS 27.2+`) personal inference workbench, multimodal media/embedding finder, and embedded Hummingbird 2 HTTP/SSE server for Apple Silicon.

## Where to look first

- [VISION.md](./VISION.md): what the project is for. Breaks ties when a design choice is unclear.
- [docs/wiki/architecture_overview.md](./docs/wiki/architecture_overview.md): end-to-end system topology, module boundaries, dual-lane GPU/ANE scheduler, and wire contract.
- [docs/wiki/model_family_onboarding.md](./docs/wiki/model_family_onboarding.md): 100% Swift model-family support matrix, file-derived `ModelTemplateInfo` onboarding, and playbooks for porting new architectures to Tier 1 (`mlx-swift-lm`) and Tier 2 (`coreai-models` / ANE).
- [docs/project/CURRENT.md](./docs/project/CURRENT.md) and [docs/project/TODO.md](./docs/project/TODO.md): subsystem status and phased backlog.

## Module Architecture

| Target | Platforms | Role |
| :--- | :--- | :--- |
| `HeylookCore` | `macOS 27.2+`, `iOS 27.2+` | Domain models, `EngineDescription` provenance contract (with `reasoningFormat` & `thinkingControls`), 5-layer `SamplerCascade` (`builtin` -> `vendor generation_config.json` / GGUF `general.sampling.*` -> `model.heylook.toml` -> `preset` -> `request`), `ModelTemplateInfo`, 4-protocol `ReasoningParser` & dynamic `StripSpecials`, `HeylookConfig` (`heylook.toml`, `model.heylook.toml` sidecars, `discoverConfiguredModels()` merging `scan_folders` + `[[models]]`, `resolveEffectiveSampling`, `defaultStoreURL` / `defaultVectorStoreURL`, 4-rung `chat_template` ladder, `GGUFHeaderReader` KV metadata parser, `estimateImageTokenBudget`, `renderPromptPreview`, and `mmproj`/drafter auto-discovery), `SQLiteStore` (`WAL` mode for conversations, messages, notebooks, presets), and Anthropic `/v1/messages` (`thinking_depth` / `reasoning_effort`) & `/v1/embeddings` wire types. |
| `HeylookEmbeddings` | `macOS 27.2+`, `iOS 27.2+` | Folded from `mbed` (`MbedCore`): `Artifact`, `ArtifactSpan`, `VectorStore` (`SQLite3` + `vDSP` Matryoshka `768d`/`512d`/`256d`/`128d`), `ArtifactProcessor` & `WindowPlanner` (`16s`-stride `32s` video windows, `30s` audio windows, `PDFKit` page chunking), `FolderWatcher`, `IndexCoordinator`, `EmbeddingLabRunner`, `ModelComputeOptions`, `ANECoreMLRuntime` (with `unload()`), and `ANETowerInspector`. |
| `HeylookEngines` | `macOS 27.2+`, `iOS 27.2+` | `InferenceProvider` (with `unload()`) & `ProviderExecutionError`, `ComputeScheduler` (serializing Metal GPU inference via `GenerationGate`, enforcing GPU LRU eviction teardown, and allowing concurrent ANE co-residency), `SubprocessLifecycleRegistry` (`SIGTERM` -> `SIGKILL` + `atexit`), `GGUFPortPool` (`18080+`), `HardwareMemoryMonitor`, `PrefixCacheTracker` (`branchTruncationsCount`), `MLXSwiftProvider` (`KVCacheSessionState` + `MLXLMCommon.generate(input:cache:parameters:context:)`, `ModelConfiguration` `extraEOSTokens`, VLM-to-LLM factory fallback, + `SpeculativeReport`), `MLXModelConverter` (pure-Swift `.safetensors` quantization & sharding via `LLMModelFactory.shared.convert` / `MLXLMCommon.convert`), `CoreAIProvider` (Schema 0.1/0.2 `.aimodel` manifest, `CoreAILanguageModels` `LanguageModelBundle`/`CoreAIRunner`, + `transcribeAudio`), `GGUFSubprocessProvider` (macOS, `ModelTemplateInfo` `chat_template_kwargs`, `ReasoningParser`, + `/props` provenance), and `RemoteHeylookProvider`. |
| `HeylookServer` | `macOS 27.2+`, `iOS 27.2+` | Embedded HTTP + incremental SSE server (`HeylookHTTPServer` + `SSEConnectionStreamer` + `EvalLabSuite` + `BonjourServiceBrowser` `_heylook._tcp`) implementing `/v1/messages` (with HTTP 422 multimodal capability gating), `/v1/conversations/{id}/generate` & `/v1/conversations/{id}/voice-generate` (disconnect-resilient server-owned generation + `heylook_saved`), model-derived `/v1/conversations/{id}/prompt` & `/v1/models/{id}/image-plan`, `/v1/embeddings`, `/v1/capabilities`, `/v1/models`, `/v1/admin/models/unload-all`, `/v1/admin/models/{id}` (`GET`/`PATCH`/`load`/`unload`/`chat-template`), `/v1/conversations`, `/v1/notebooks`, `/v1/presets`, `/v1/requests/{id}`, and `/v1/system/metrics`. |
| `HeylookUI` | `macOS 27.2+`, `iOS 27.2+` | Shared SwiftUI 5-surface workbench: **Chat Workbench** (with model-derived `ThinkingControls` & `Depth` picker), **Notebook**, **Embeddings & Media Finder Studio**, **Models & Engine Inspector**, and **Performance & Server Control**. |
| `HeylookApp` | `macOS 27.2+`, `iOS 27.2+` | Native desktop & mobile app entry point + macOS `MenuBarExtra` server & model lifecycle controller. |
| `heylook-cli` | `macOS 27.2+` | Unified CLI (`serve`, `status`, `doctor`, `models`, `chat`, `index`, `search`, `watched`, `inspect-ane`, `inspect-checkpoint`, `convert-mlx`, `eval-lab`). |
| `HeylookTests` | `macOS 27.2+`, `iOS 27.2+` | Swift Testing (`@Suite`, `@Test`) 26-test unit, lifecycle, and wire-contract verification suite. |

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

1. **Remote SwiftPM Dependencies Only in `Package.swift`**: Always declare external dependencies using canonical remote `.package(url: "https://github.com/...", ...)` references, never local filesystem paths (`.package(path: ...)`). Use `swift package edit` or Xcode local overrides if testing uncommitted local framework changes.
2. **Path Privacy**: Never commit personal home-directory paths or personal full names in tracked files. Use repo-relative paths, `FileManager.default.homeDirectoryForCurrentUser`, or environment variables (`HEYLOOK_CONFIG`, `HEYLOOK_MODELS_DIR`, `HEYLOOK_DB_PATH`).
3. **Derive, Don't Copy**: Derive capabilities, sampler schemas, and provenance fields from single canonical Swift definitions and model files.
4. **No Migration Code**: Solo deployment. Schema updates in `SQLiteStore` use idempotent `CREATE TABLE IF NOT EXISTS` or clean table recreation.
5. **Git Hygiene**: Stage files explicitly by name (`git add <file>`), never `git add -A` or `git add .`.
