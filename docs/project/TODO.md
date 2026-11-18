# `heylookitsanllm-macos` — Parity Roadmap & Next Horizons

Master parity completion log against `heylookitsanllm` `v2.0.251`, plus open hardware benchmarks and post-parity horizons.

---

## Parity Completion Roadmap (Phases `16A` – `16E`)

### Phase 16A — Truthful Core, Schema Config & SQLite/Wire Parity *(Wave 1)*
- [x] `EngineContract`, `Fact<Value>` (forced `null` for `.unknown`/`.notApplicable`), `SettingFact`, zero-path `PublicModelWireRow`, and explicit `AdminModelWireRow` (`source`, `stale_reload_fields`).
- [x] `ProviderConfigSchema` (single source of truth deriving 13 `SamplerCascade` fields, `422` `isFileOnly` guard on `PATCH`, `400` load-setting guard on `/reload`, `staleReloadFields`) & comment-preserving `TOMLCodec` (cleanly removing empty sidecars).
- [x] 5-layer `SamplerCascade` aligned to `temperature = 1.0`, `top_p = 0.95`, `max_tokens = 16384`, `VENDOR_SAMPLING_KEYS`, and `0/1.0` off sentinels.
- [x] Render-probe `ThinkingControls` (`garbageDepthSentinel` classifying `ignored | verbatim | fallback | raises`), `checkPrefixStability`, and 4+ rung template ladder (`chat_template.heylook.jinja`, `modified_since_download`, `refused_shapes`).
- [x] `SQLiteStore` WAL parity: unseeded `COLLATE NOCASE` `presets`, conversation-scoped 64-hex `media_blobs` (`nosniff`), non-compacting `deleteMessage`, `deleteMessagesAfter`, atomic `replaceTailWithMessage` / `replaceTailWithUpdate`, `ObservabilityDiskLog` (`events.jsonl` / `metrics.jsonl`), and `WireDTOs` / retired-field `422` validation.

### Phase 16B — Engine Specialization, MLX Support, Speech & RAM Fit *(Wave 2)*
- [x] Signature-driven `ModelSpecialization` (`CheckpointSignature`, `Gemma4Specialization` `48px` vision alignment & `gemma4_assistant` `backbone_hidden_size` pairing, `HybridRecurrentSWASpecialization`, `StandardAttentionSpecialization`).
- [x] `MLXRuntimeSupport`: `PreRenderedVLMTokenizerProxy`, `VisionFeatureCache` (16 / 512 MB LRU), `APCSnapshotCache` keyed on `(tokenIDs, mediaSalt)`, `ThinkingControlLogitProcessor`, and lossless `LeviathanMTPTokenIterator` (`p_target / q_draft` rejection + residual resampling when `temperature > 0`).
- [x] `RAMFitCalculator` (`thinHeadroomGB = 24.0`, Mach + Metal + `iogpu.wired_limit_mb`, `sysctlSuggestMB`, hard `fail` on MLX vs `warn` on GGUF, layer-aware KV estimation & `droppedDrafterForMemory`).
- [x] `GGUFSubprocessProvider` parity: SWA/hybrid `--cache-reuse` suppression, `--ctx-checkpoints 32 --checkpoint-min-step 256`, `--reasoning-preserve`, `StopSequenceFilter` reply-only stop cut, and `CompletionIDBox` mid-stream `reasoning_end`.
- [x] Decoupled `SpeechTranscriptionService` (`CoreAISpeech` + `SpeechAnalyzer`/`SpeechTranscriber`) with subchunk-walking `decodeRIFFChunksMonoPCM`.

### Phase 16C — Multimodal Embeddings, Modular Server Saga & CLI Eval Lab *(Wave 2)*
- [x] `EmbeddingModelFacts` (`PromptCatalog` parsed dynamically from `config_sentence_transformers.json`) and `GGUFEmbeddingEngine` (`llama-server --embeddings`, `ctx_size <= n_ubatch <= n_batch`, `400` rejection on video `windows`).
- [x] `MLXSwiftLMEmbeddingGemma2Engine`: 1,024-token padded batch text embedding, true `hybrid_ane` ANE->GPU `softTokens` fusion without needing `TextANE.mlmodelc`, and consecutive single-window `windows.seconds` video decoding (all durations including `<= 16s`).
- [x] Modular `HeylookServer`: `WorkbenchBackend` / `WorkbenchService`, `ConversationOrchestrator` (`409 generation_in_progress`, detachable `SubscriberHub`, non-destructive atomic tail commits, `user`/`assistant` `continue`, open-thinking resume), `PatternRouter` greedy `*id`, `HTTPConnectionHandler` (Keep-Alive + chunked), `SSEConnectionStreamer` (`5s` `event: ping`, `heylook_progress`, dual-key `thinking_delta`, `performance`, `heylook_saved`), and optional `StaticFrontendRoutes`.
- [x] Standalone `CLIEvalLabSuite` (`Sources/HeylookCLI/EvalLabSuite.swift`) with zero synthetic fallbacks.

### Phase 16D — Modular `@Observable` Controllers & 6-Surface Parity *(Waves 3A & 3B)*
- [x] `HeylookWorkbenchViewModel` split across 6 controllers (`DocumentSettingsController` with separate `chatSettings`/`notebookSettings` `"Preset is a Copy"` + `"Null-Means-Cascade"`, `RemoteWorkbenchBackend`, `ChatController`, `NotebookAndPresetsController`, `EmbeddingsStudioController`, `ModelsAndPerfController` with zero 3s polling).
- [x] Shell hardening: `AdaptiveContrastStyle`, `@Environment(\.scenePhase)` flush, layered `Escape` dismissal precedence, `?` shortcut outside text inputs, and `ScrollViewReader` tail follow (`Jump to latest`).
- [x] All 6 surfaces at parity (`⌘1` Chat Load/Reload panel & 4-state sys-prompt chip; `⌘2` Notebook empty-start & debounced autosave; `⌘3` Presets optimistic locking & compatibility preview; `⌘4` Compare `CompareVideoWindowPreviewSheet` + 8-case Embedding Lab + `Media Corpus (mbed)`; `⌘5` schema-driven `ModelConfigEditorSheet` + `RAMFitMeterView` + 4+ rung `ChatTemplateInspectorView`; `⌘6` 3 Perf tables + Bonjour/HTTP truth + `[settings]` / JSONL disk log reader).

### Phase 16E — Shim Removal, Contract Suite & Documentation Sync *(Wave 4)*
- [x] Unified `ChatController` and `HeylookRouter` onto one shared `WorkbenchService` actor instance; deleted `LegacyEvalLabBridge.swift`, moved `DeterministicHashEmbeddingEngine` out of `Sources/` into `Tests/HeylookTests/Support/TestStubProviders.swift`, and removed all `allowSyntheticFallback` parameters/properties from `Sources/`.
- [x] Fixed Wave 4 contract audit discoveries: HTTP/1.1 Keep-Alive `Data.dropFirst` index crash (`HTTPConnectionHandler`), `InferenceProvider` dynamic dispatch requirements (`ComputeScheduler`), and `404` `modelNotFound` mapping in voice/ASR endpoints (`WorkbenchService`).
- [x] Migrated monolithic `HeylookTests.swift` into 5 focused Swift Testing suites (`EngineContractTests.swift`, `SamplerAndTemplateTests.swift`, `MessagesAndSagaContractTests.swift`, `EmbeddingsAndSpecializationTests.swift`, `WorkbenchControllersTests.swift`) + `Support/TestStubProviders.swift` (`28` tests passed across `5` suites).
- [x] Synchronized `VISION.md`, `AGENTS.md`, `README.md`, `docs/USER_GUIDE.md`, `docs/wiki/*.md`, `docs/project/CURRENT.md`, and `docs/project/TODO.md` with Wave 1–4 truth.

---

## Post-Parity Horizons *(Open Hardware & Ecosystem Frontiers)*

- [ ] **H.1 — Live Hardware ASR & Speech Benchmarks**: Measure real-world WER, latency, and ANE utilization comparing Core AI `Parakeet TDT` / `Whisper` bundles against Apple `SpeechAnalyzer` / `SpeechTranscriber`.
- [ ] **H.2 — Live BF16 `convert-mlx` Throughput & Local Eval Bank**: Benchmark pure-Swift MLX quantization (`affine`, `mxfp4`, `mxfp8`, `nvfp4`, `q4_zero`) and run `eval-lab` / multi-turn KV probes against live checkpoints on Apple Silicon.
- [ ] **H.3 — Physical `iOS 27.2+` / `iPadOS 27.2+` Field Deployment**: Verify Bonjour `_heylook._tcp` LAN discovery, Tailscale remote pairing, compact touch ergonomics, and background `heylook_saved` recovery on physical devices.
- [ ] **H.4 — Multi-Runtime Extension (`LiteRT`)**: Add `LiteRT` alongside `MLXSwiftProvider`, `CoreAIProvider`, `GGUFSubprocessProvider`, and `RemoteHeylookProvider` behind `InferenceProvider` and `ModelSpecialization`.
- [ ] **H.5 — `macOS / iOS 27.3+` Siri `AppIntents` & TCC Custody**: Implement Siri `ToolIntent`, Spotlight `IndexedEntity`, and durable TCC / Full Disk Access scan-folder workflows behind `#available(macOS 27.3, iOS 27.3, *)`.

---

## Historical Port Log (Phases 0–15)

- **Phases 0–5**: Initial native Swift package scaffold, MLX / GGUF / Core AI providers, dual-lane `ComputeScheduler`, `/v1/messages`, and SQLite WAL persistence.
- **Phases 6–10**: Multimodal `EmbeddingGemma 2` MLX + `.mlmodelc` ANE towers, Matryoshka `VectorStore`, FSEvents corpus indexing, `Compare` & `Embedding Lab`, and pure-Swift `convert-mlx`.
- **Phases 11–15**: Jinja2 template engine (`HeylookJinjaRenderer`), 4-protocol `ReasoningParser`, `CoreAISpeech` / `SpeechTranscriber` wiring, Swift 6.4 strict concurrency migration, and `macOS 27.2+` / `iOS 27.2+` (`27.3` guarded) split views.
