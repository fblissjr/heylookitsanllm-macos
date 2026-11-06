# AGENTS.md

Operating guide for `heylookitsanllm-macos` — a native Swift 6.2 multiplatform (`macOS 27.2+` and `iOS 27.2+`) personal inference workbench, multimodal media/embedding finder, and embedded Hummingbird 2 HTTP/SSE server for Apple Silicon.

## Where to look first

- [VISION.md](./VISION.md): what the project is for. Breaks ties when a design choice is unclear.
- [docs/wiki/architecture_overview.md](./docs/wiki/architecture_overview.md): end-to-end system topology, module boundaries, dual-lane GPU/ANE scheduler, and wire contract.
- [docs/project/CURRENT.md](./docs/project/CURRENT.md) and [docs/project/TODO.md](./docs/project/TODO.md): subsystem status and phased backlog.

## Module Architecture

| Target | Platforms | Role |
| :--- | :--- | :--- |
| `HeylookCore` | `macOS 27.2+`, `iOS 27.2+` | Domain models, `EngineDescription` provenance contract, 5-layer `SamplerCascade` (`builtin` -> `vendor generation_config.json` -> `model.heylook.toml` -> `preset` -> `request`), `ReasoningParser` & `StripSpecials`, `HeylookConfig` (`heylook.toml`, `model.heylook.toml` sidecars, 4-rung `chat_template` ladder, and `mmproj`/drafter auto-discovery), `SQLiteStore` (`WAL` mode for conversations, messages, notebooks, presets), and Anthropic `/v1/messages` & `/v1/embeddings` wire types. |
| `HeylookEmbeddings` | `macOS 27.2+`, `iOS 27.2+` | Folded from `mbed` (`MbedCore`): `Artifact`, `ArtifactSpan`, `VectorStore` (`SQLite3` + `vDSP` Matryoshka `768d`/`512d`/`256d`/`128d`), `ArtifactProcessor` & `WindowPlanner` (`16s`-stride `32s` video windows, `30s` audio windows, `PDFKit` page chunking), `FolderWatcher`, `IndexCoordinator`, `EmbeddingLabRunner`, `ModelComputeOptions`, `ANECoreMLRuntime`, and `ANETowerInspector`. |
| `HeylookEngines` | `macOS 27.2+`, `iOS 27.2+` | `InferenceProvider` & `EmbeddingProvider` protocols, `ComputeScheduler` (serializing Metal GPU inference via `GenerationGate` while allowing concurrent ANE co-residency), `HardwareMemoryMonitor`, `PrefixCacheTracker`, `MLXSwiftProvider`, `CoreAIProvider`, `GGUFSubprocessProvider` (macOS), and `RemoteHeylookProvider`. |
| `HeylookServer` | `macOS 27.2+`, `iOS 27.2+` | Embedded HTTP + SSE server (`HeylookHTTPServer` + `BonjourServiceBrowser` `_heylook._tcp`) implementing `/v1/messages` (with HTTP 422 multimodal capability gating), `/v1/conversations/{id}/generate` (disconnect-resilient server-owned generation + `heylook_saved`), `/v1/conversations/{id}/prompt`, `/v1/models/{id}/image-plan`, `/v1/embeddings`, `/v1/capabilities`, `/v1/models`, `/v1/admin/models/{id}` (`GET`/`PATCH`/`load`/`unload`/`chat-template`), `/v1/conversations`, `/v1/notebooks`, `/v1/presets`, `/v1/requests/{id}`, and `/v1/system/metrics`. |
| `HeylookUI` | `macOS 27.2+`, `iOS 27.2+` | Shared SwiftUI 5-surface workbench: **Chat Workbench**, **Notebook**, **Embeddings & Media Finder Studio**, **Models & Engine Inspector**, and **Performance & Server Control**. |
| `HeylookApp` | `macOS 27.2+`, `iOS 27.2+` | Native desktop & mobile app entry point + macOS `MenuBarExtra` server controller. |
| `heylook-cli` | `macOS 27.2+` | Unified CLI (`serve`, `status`, `doctor`, `models`, `chat`, `index`, `search`, `watched`, `inspect-ane`, `inspect-checkpoint`, `eval-lab`). |
| `HeylookTests` | `macOS 27.2+`, `iOS 27.2+` | Swift Testing (`@Suite`, `@Test`) 19-test unit and wire-contract verification suite. |

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
  - `swift run heylook-cli chat [--model <id_or_path>] [--max-tokens <n>] [--no-thinking] "<prompt>"`
  - `swift run heylook-cli serve 8000`
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
