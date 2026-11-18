# `heylookitsanllm-macos` — Current Status & Readiness Audit

**Benchmark Baseline**: `heylookitsanllm` `v2.0.251`  
**Toolchain / Targets**: Swift 6.4 (`v6` language mode) on `macOS 27.2+` (`27.2.1` hosts + `#available(macOS 27.3, iOS 27.3, *)` guards) & `iOS 27.2+` Simulator  
**Overall Parity Status**: **Wave 1–4 Behavioral & 6-Surface Parity Complete** *(offline unit/contract suite, native MLX build, and iOS Simulator target verified green; live Apple Silicon model/ASR hardware benchmarks remain under Post-Parity Horizons)*

---

## 1. Subsystem Readiness Matrix

| Subsystem | Target | Parity Status | Architecture & Key Components Implemented |
| :--- | :--- | :--- | :--- |
| **Truthful Engine Contract & Schema** | `HeylookCore` | **PARITY COMPLETE** | `EngineContract`, `Fact<Value>` (`"value": null` for `.unknown` / `.notApplicable`), `SettingFact`, zero-path `PublicModelWireRow`, explicit `AdminModelWireRow` (`source`, `stale_reload_fields`), `ProviderConfigSchema` (13 derived sampler fields, `422` `file_only` guard, `/reload` validation), comment-preserving `TOMLCodec` (deletes empty sidecars). |
| **Samplers, Templates & Reasoning** | `HeylookCore` | **PARITY COMPLETE** | 5-layer `SamplerCascade` (`1.0 / 0.95 / 16384`, `VENDOR_SAMPLING_KEYS`), `ModelTemplateInfo` AST + `garbageDepthSentinel` `renderProbe` (`ignored \| verbatim \| fallback \| raises`, `changes_prefix`, `off`, `aliases`), `HeylookJinjaRenderer.checkPrefixStability`, 4+ rung template ladder (`chat_template.heylook.jinja`), 4-protocol `ReasoningParser` + dynamic `StripSpecials`. |
| **SQLite WAL Store & Observability** | `HeylookCore` | **PARITY COMPLETE** | `SQLiteStore` (`WAL`, conversation-scoped 64-hex `media_blobs` with `nosniff`, non-compacting `deleteMessage`, `deleteMessagesAfter`, atomic `replaceTailWithMessage` / `replaceTailWithUpdate`, unseeded `COLLATE NOCASE` `presets`, `message_stats`, `mutationStream()`), `ObservabilityDiskLog` (`events.jsonl` / `metrics.jsonl`, 50 MB rotation, hourly sweep, `[settings]`). |
| **MLX, Gemma 4, SWA, RAM Fit & ASR** | `HeylookEngines` | **PARITY COMPLETE** | `ComputeScheduler` (`InferenceProvider` protocol dispatch, `GenerationGate` + `ANELane`, `withGPUEmbeddingLease`), `ModelSpecialization` (`Gemma4Specialization` `48px` alignment & `backbone_hidden_size`, `HybridRecurrentSWASpecialization`, `StandardAttentionSpecialization`), `MLXRuntimeSupport` (`PreRenderedVLMTokenizerProxy`, `VisionFeatureCache`, `APCSnapshotCache` + `mediaSalt`, `ThinkingControlLogitProcessor`, lossless `LeviathanMTPTokenIterator`), `RAMFitCalculator` (`24.0 GiB` headroom, MLX hard fail vs GGUF warning, `sysctlSuggestMB`), `GGUFSubprocessProvider` (SWA `--cache-reuse` suppression, `--ctx-checkpoints 32 --checkpoint-min-step 256`, `--reasoning-preserve`, `StopSequenceFilter` reply-only cut, `CompletionIDBox`), `SpeechTranscriptionService` (`decodeRIFFChunksMonoPCM`). |
| **Multimodal & GGUF Embeddings** | `HeylookEmbeddings` | **PARITY COMPLETE** | `EmbeddingModelFacts` (`PromptCatalog` from `config_sentence_transformers.json`), `MLXSwiftLMEmbeddingGemma2Engine` (1,024-token padded text batching, true `hybrid_ane` ANE->GPU `softTokens` bridge without requiring `TextANE.mlmodelc`, consecutive single-window `windows.seconds` video decoding for all clips incl. `<= 16s`), `GGUFEmbeddingEngine` (`llama-server --embeddings`, `400` video reject), `VectorStore` (`vDSP` `768d -> 128d` Matryoshka), zero `DeterministicHashEmbeddingEngine` / zero synthetic fallback in `Sources/`. |
| **HTTP/SSE Wire Server & Saga** | `HeylookServer` | **PARITY COMPLETE** | Modular `Orchestration/` (`WorkbenchBackend`, `WorkbenchService`, `ConversationOrchestrator` atomic tail saga + detachable `SubscriberHub` + `409 generation_in_progress`, `ChunkTelemetryAccumulator`), `Transport/` (`HTTPConnectionHandler` HTTP/1.1 Keep-Alive + chunked, `SSEConnectionStreamer` `5s` `event: ping`, `SSEWirePipeline` dual-key `thinking_delta` + `heylook_progress` + `performance` + `heylook_saved`), `Routing/` (`PatternRouter` greedy `*id`, `ErrorMapping`), 7 `Routes/` modules (`DELETE ?after=`, optional `StaticFrontendRoutes`), `BonjourServiceBrowser`. |
| **Unified CLI & Eval Lab** | `HeylookCLI` | **PARITY COMPLETE** | `HeylookCLI.swift` + standalone `EvalLabSuite.swift` (`CLIEvalLabSuite` with zero synthetic fallbacks; `LegacyEvalLabBridge.swift` deleted). |
| **Native 6-Surface SwiftUI Workbench** | `HeylookUI`, `HeylookApp` | **PARITY COMPLETE** | Single `WorkbenchService` shared across `ChatController` and `HeylookRouter`, 6 `@Observable` controllers (`DocumentSettingsController` separate `chatSettings`/`notebookSettings` `"Preset is a Copy"` + `"Null-Means-Cascade"`, `RemoteWorkbenchBackend`, `ChatController`, `NotebookAndPresetsController`, `EmbeddingsStudioController`, `ModelsAndPerfController` zero 3s polling), `AdaptiveContrastStyle`, `@Environment(\.scenePhase)` flush, layered `Escape`, `ScrollViewReader` tail follow (`Jump to latest`), and full `⌘1`–`⌘6` surface parity (Load/Reload panel, 4-state sys-prompt chip, Compare `CompareVideoWindowPreviewSheet` + 8-case Embedding Lab + `Media Corpus (mbed)`, schema-driven `ModelConfigEditorSheet`, 3 Perf tables + disk JSONL reader). |
| **Contract & Unit Test Suite** | `HeylookTests` | **VERIFIED GREEN** | 5 focused Swift Testing files (`EngineContractTests.swift`, `SamplerAndTemplateTests.swift`, `MessagesAndSagaContractTests.swift`, `EmbeddingsAndSpecializationTests.swift`, `WorkbenchControllersTests.swift`) + `Support/TestStubProviders.swift` (`TestStubInferenceProvider`, `TestDeterministicEmbeddingEngine`). `allowSyntheticFallback`, `LegacyEvalLabBridge.swift`, and `DeterministicHashEmbeddingEngine` completely removed from `Sources/`. |

---

## 2. Verified Build & Test Results (Wave 4)

- **Swift Testing Unit & Contract Suite (`swift test`)**: **`28` tests across `5` suites passed** (`0` failures, `0` skipped assertions).
  - Note: Under macOS terminal sandbox Seatbelt rules where `NWListener` port binding is restricted, HTTP/1.1 Keep-Alive framing (`Data.dropFirst` index reset) and `Transfer-Encoding: chunked` parsing + `HeylookRouter` dispatch are verified deterministically in-process via `HTTPConnectionHandler.ConnectionBufferActor` and `HeylookHTTPServer.dispatchRawHTTPBuffer`.
- **Native MLX Build (`HEYLOOK_ENABLE_NATIVE_MLX=1 swift build --build-system native`)**: Built `heylook-cli`, `heylook-app`, and `--build-tests` cleanly with `0` errors.
- **Cross-Platform Target (`iOS 27.2` Simulator)**: `swift build --target HeylookUI --triple arm64-apple-ios27.2-simulator` built cleanly with `0` errors.
- **Static & Hygiene Checks**: Zero `allowSyntheticFallback`, `LegacyEvalLabBridge`, `DeterministicHashEmbeddingEngine`, `evalLabSuiteReport`, `runFullEvalLabSuite`, host home-directory paths, sibling workspace traversals, or older Swift toolchain strings in `Sources/`.

---

## 3. Open Live Hardware & Post-Parity Work

The following items require physical Apple Silicon checkpoint weights / hardware devices or next-generation OS features and remain intentionally open under **Post-Parity Horizons** in [`TODO.md`](./TODO.md):

1. **Live ASR Benchmarks**: Hardware WER & latency comparisons between Core AI `Parakeet TDT` / `Whisper` bundles and Apple `SpeechAnalyzer` / `SpeechTranscriber` across real multi-speaker recordings.
2. **Live BF16 `convert-mlx` & Model Eval Bank**: End-to-end quantization throughput benchmarks and multi-turn `eval-lab` / KV-chain / VLM parity runs against real local weights.
3. **Physical `iOS 27.2+` Device Testing**: Field pairing of iPhone/iPad over Bonjour (`_heylook._tcp`) and Tailscale with background `heylook_saved` disconnect recovery.
4. **Future Runtimes & `27.3+` OS APIs**: `LiteRT` integration behind `InferenceProvider` + `ModelSpecialization`, plus `macOS/iOS 27.3+` Siri `AppIntents` (`ToolIntent` / `IndexedEntity`) and durable TCC / Full Disk Access workflows.
