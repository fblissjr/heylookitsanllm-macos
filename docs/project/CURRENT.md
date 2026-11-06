# Current State (`heylookitsanllm-macos`)

Last updated: 2026-11-06 (`main`)

Tracks the live readiness of each subsystem in `heylookitsanllm-macos` (`SOLID`, `SCAFFOLDED`, or `PLANNED`).

---

## Subsystem Matrix

| Subsystem | Target / Files | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Engine Contract & Provenance** | `HeylookCore` (`EngineDescription.swift`) | **SOLID** | Full `EngineDescription`, `ProvenanceSource`, `EffectClass`, `ComputePlacement`, `CacheReport`, `GenerationChunk`, and `StopReason`. |
| **Sampler Cascade & Reasoning Parser** | `HeylookCore` (`SamplerCascade.swift`, `ReasoningParser.swift`) | **SOLID** | Deterministic cascade (`builtin` -> `model.heylook.toml` -> `preset` -> `request`), streaming `<think>...</think>` parser with mid-stream `forceEndReasoning()`, and `StripSpecials` automaton. |
| **TOML Config & Sidecar Engine** | `HeylookCore` (`HeylookConfig.swift`) | **SOLID** | Reads/writes `heylook.toml` and per-folder `model.heylook.toml` sidecars; auto-discovers `.safetensors`, `.aimodel`, `.mlmodelc`, and `.gguf` model folders. |
| **Unified SQLite Store** | `HeylookCore` (`SQLiteStore.swift`) | **SOLID** | Zero-dependency SQLite3 (`WAL` mode) store for conversations, branched messages (`deleteMessagesFrom`), thinking blocks, notebooks, and seeded presets. |
| **Multimodal Embeddings & ANE Runtime** | `HeylookEmbeddings` (folded from `mbed`) | **SOLID** | Full `mbed` fidelity: `Artifact`, `ArtifactSpan`, `VectorStore` (`vDSP` Matryoshka `768d`..`128d`), `WindowPlanner`, `ArtifactProcessor`, `FolderWatcher` (`FSEvents`), `IndexCoordinator` (`indexURLs`, `discoverFiles`, `isUpToDate`, `pruneMissingArtifacts`, `buildChunks` with video soundtrack slicing), `HeylookMLXEngine` (`planBatches` + `DELETE /v1/requests/{id}` cancellation), `NativeMLXEmbeddingGemma2` (`MLXSwiftLMEmbeddingGemma2Engine` + `HybridLocalEmbeddingEngine`), `ModelComputeOptions`, `ANECoreMLRuntime` (`MLMultiArray` Vision/Audio/Text ANE prediction + 512-point `vDSP` FFT & 128-bin HTK mel filterbank), and `ANETowerInspector`. |
| **Heterogeneous Compute Scheduler & Providers** | `HeylookEngines` | **SOLID (Core) / SCAFFOLDED (Live Bridges)** | Dual-lane `ComputeScheduler` (`GPU GenerationGate` + `ANE Co-Residency Lane`), `HardwareMemoryMonitor` (192 GB unified RAM + wired budget), `GGUFHeaderReader`, `RemoteHeylookProvider`, and conditional `MLXSwiftProvider` / `CoreAIProvider` / `GGUFSubprocessProvider`. |
| **Embedded HTTP/SSE Server** | `HeylookServer` (`HeylookHTTPServer.swift`) | **SOLID** | `HeylookRouter` & `HeylookHTTPServer` (`NWListener` + Bonjour `_heylook._tcp` + optional Hummingbird 2) serving `/v1/messages`, disconnect-resilient `/v1/conversations/{id}/generate` (`event: heylook_saved`), `/v1/models`, `/v1/conversations`, `/v1/notebooks`, `/v1/presets`, `/v1/requests/{id}/reasoning-end`, and `/v1/system/metrics`. |
| **Multiplatform SwiftUI Workbench** | `HeylookUI`, `HeylookApp`, `heylook-cli` | **SOLID** | Shared `macOS 27.2+` and `iOS 27.2+` 5-surface SwiftUI app (`Chat`, `Notebook`, full 3-column `mbed` `Embeddings & Media Studio` with inline `AVPlayerView`, clickable `frameTimes` jump pills, `ScoredFeatureBreakdownCard`, and drag-and-drop/`NSOpenPanel` indexing, `Models`, `Perf & Server`) + macOS `MenuBarExtra` and `heylook-cli` (`doctor`, `status`, `models`, `index`, `search`, `watched`, `inspect-ane`, `inspect-checkpoint`, `chat`, `serve`). |
| **Verification Suite** | `HeylookTests` (`HeylookTests.swift`) | **SOLID** | 10/10 Swift Testing (`@Suite`, `@Test`) unit and wire-contract suites passing (`SamplerCascade`, `ReasoningParser` + `forceEndReasoning`, `HeylookConfig` TOML round-trip, `SQLiteStore` WAL & truncation, `Matryoshka` `vDSP` + `WindowPlanner` + `VectorStore` parent deduplication, `ModelComputeOptions` + `ANECoreMLRuntime` 16 kHz log-mel, `IndexCoordinator` folder indexing/deduplication/pruning + `HeylookMLXEngine.planBatches` + `maskedMeanPoolAndNormalize`, `GGUFHeaderReader`, Dual-Lane `ComputeScheduler` ANE co-residency, and `HeylookRouter` `/v1/*`). |
