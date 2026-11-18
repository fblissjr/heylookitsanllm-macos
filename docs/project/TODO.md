# Backlog & Parity Completion Roadmap (`heylookitsanllm-macos`)

Last updated: 2026-11-18 (`main`, benchmarked against `heylookitsanllm` `v2.0.251`)

Tracks the remediation roadmap closing all core, engine, wire/server, frontend, and verification gaps against `heylookitsanllm` (`v2.0.251`) on `Swift 6.4` (`macOS 27.2+` and `iOS 27.2+`, supporting `27.2.1` hosts and `#available(macOS 27.3, iOS 27.3, *)` OS hooks), followed by post-parity native horizons.

---

## Phase 16A: Truthful Core & Schema Contract (`HeylookCore`) — *Wave 1 Active*

- [ ] **16A.1 Canonical `EngineContract`, `Fact<Value>` & Wire Rows (`Sources/HeylookCore/Engine/`)**:
  - Implement generic `Fact<Value>` (`{value, provenance, source}` encoding `value: null` for `.unknown` and `.notApplicable`) and `SettingFact` (`{value, configured, auto, reason, provenance, effect}`).
  - Implement full `EngineContract` (`runtime`, `context: {length, running}`, `template`, `settings`, `cache`, `thinking`, `speculative`, `decoding`, `embedding`, `image`, `steering`) matching `heylookitsanllm` `providers/contract.py`.
  - Add path-free `PublicModelWireRow` (never exposing `modelPath` on `GET /v1/models`) and `AdminModelWireRow` (only explicitly configured keys, `stale_reload_fields`, `source: "config"|"discovered"`).
  - Upgrade `SpeculativeReport` (`emittedTokens`, `acceptanceRate`, `draftShare`) and `CacheReport` (`summaryBadge`, `humanReadableCause`, `causeWords`: `cold`, `memory`, `new_image_set`, `no_common_prefix`, `probable_template_diverged`, `probable_budget_skipped`, `probable_evicted`).
- [ ] **16A.2 Declarative `ProviderConfigSchema` (`Sources/HeylookCore/Config/ProviderConfigSchema.swift`)**:
  - Define `ConfigFieldDescriptor` (`name`, `type`, `bounds`, `enumValues`, `effect`, `arg`, `ui: .normal | .advanced | .hidden`, `shape`, `reason`, `isFileOnly`, `isLoadSetting`, `description`) across `mlx`, `gguf`, and `coreai`.
  - Derive all 13 sampler configuration keys directly from `SamplerCascade.SamplerFieldKey` (zero hand-copied lists) and expose single-source helpers for `/v1/admin/model-options`, `PATCH` `422` validation (`isFileOnly` / unknown / out-of-range), `/reload` parsing, and `stale_reload_fields` diffing.
- [ ] **16A.3 Spec-Accurate `TOMLCodec` & Truthful Discovery (`Sources/HeylookCore/Config/`)**:
  - Replace comma-splitting `SimpleTOML` with a recursive-descent TOML parser (`[scan]`, `[settings]`, `[[models]]`, quoted strings with commas/escapes, multiline strings/arrays, inline tables) plus a comment-preserving surgical TOML editor (removing empty `model.heylook.toml` sidecars on `null` unset).
  - Unify `HEYLOOK_READONLY_MODEL_CONFIG` in `HeylookConfig.isReadOnlyModelConfig` and drop legacy `default_model` fallback.
  - Remove the `id.contains("embed")` modality hack (derive strictly from `SentenceTransformers` / `config.json` / GGUF `pooling_type`) and remove `defaultChatMLTemplate` masquerading as `tokenizer_config.json`.
  - Upgrade `ChatTemplateInspection` (`sources` ladder, `override_template`, `inert_reason`, `stale: Bool?`, `refused_shapes` message-shape probes via `HeylookJinjaRenderer`).
  - Upgrade `GGUFHeaderReader` to detect SWA (`*.attention.sliding_window > 0`) and `linear_attention`, detect `gemma4`/`siglip` vision projectors, scan `nextn` / `dflash` / `draft-eagle3` drafters, and verify `gemma4_assistant` `backbone_hidden_size` in `discoverDrafter`.
- [ ] **16A.4 `SamplerCascade` Floor & Render-Probe `ThinkingControls` (`Sampling/`, `Config/ModelTemplateInfo.swift`)**:
  - Align `SamplerCascade` floor to `temperature: 1.0`, `top_p: 0.95`, `max_tokens: 16384` (`top_k` unset by default, `thinking` defaulting to `modelSupportsThinking`), restrict vendor keys to `temperature`, `top_p`, `top_k`, and expose `samplerDefaults` + `samplerSources` (`"model" | "vendor" | "default"`) from one cascade call.
  - Remove `"deepseek-r1"` / `"qwq"` ID substring heuristics in `ModelTemplateInfo` and derive `ThinkingControls` (`switch`, `depth`, `aliases`, `default`, `unknown`, `changes_prefix`, `off`, `budget`) from Jinja AST inspection + candidate & `"zq-not-a-level"` render probes.
- [ ] **16A.5 Truthful `SQLiteStore`, Observability Disk Log & Strict Wire DTOs (`Store/`, `Wire/`)**:
  - Remove `seedDefaultPresetsIfNeeded` from `SQLiteStore`; scope `media_blobs` per `(conversation_id, media_id)` 64-hex SHA-256 IDs with MIME sanitization and `url` block relocation (`hydrateMedia: false` returns URLs; hydration resolves bytes for providers); copy blobs cleanly in `cloneConversation`.
  - Stop compacting positions in `deleteMessage`; add `deleteMessagesAfter(conversationID:afterPosition:)`, atomic `replaceTailWithMessage`, and `replaceTailWithUpdate` (preserving anchor `modelID`); join `message_stats` onto returned `MessageRecord`s; implement full `queryPerformanceProfile(range:)` (`timing_breakdown`, per-model `cache`, 8-hour `trends`).
  - Add JSONL `events` / `metrics` disk log persistence (`ObservabilityDiskLog.swift`) read newest-first.
  - Upgrade `MessagesWire` (`ThinkingParameter` bool or `{type, budget_tokens}`, explicit `422` on retired keys, sampler bounds & `stop_sequences` validation, nested+flat media blocks, `StopReason.cancelled` -> `"max_tokens"`, `stop_sequence`, `message_stop.performance`, dual `thinking`+`text` on `thinking_delta`, `heylook_progress`, `heylook_saved`) and dedicated `WireDTOs` (`conversations` `generating`, `notebooks` `content`, `presets` `params` with `422` unknown-key validation).

---

## Phase 16B: Engine & Gemma 4 / EmbeddingGemma 2 Parity (`HeylookEngines`, `HeylookEmbeddings`)

- [ ] **16B.1 Clean Synthetic Fallbacks & `RuntimeAdapter` Seams**:
  - Delete production synthetic echo/hash fallback branches from `MLXSwiftProvider`, `GGUFSubprocessProvider`, `CoreAIProvider`, and `HybridLocalEmbeddingEngine`; move test doubles strictly to `Tests/`.
  - Introduce clean, composable `RuntimeAdapter` and model-specialization seams so `MLXSwift`, `CoreAI`, `GGUF`, and future runtimes (`LiteRT`) report a unified `EngineContract` (`describeObserved()`).
- [ ] **16B.2 `MLXSwiftProvider` VLM Cache, Prefill Progress & Lossless MTP Parity**:
  - Replace legacy `0.7 / 2048` fallback literals with `SamplerCascade` resolution.
  - Replace single-slot `MLXKVCacheBox` multimodal wipe (`lmInput.image != nil`) with unified multimodal prefix + vision feature caching and Gemma 4 `BatchRotatingKVCache` single-row transaction semantics.
  - Implement lossless Leviathan MTP rejection sampling for non-greedy (`temperature > 0.0`) requests matching `v2.0.249–v2.0.250`, stateless penalty MTP support (`presence_penalty`, `repetition_penalty`), `draft_max_context` prompt-length gate, Gemma 4 visual padding trim + single-row attention mask fast paths (each carrying a retirement test/probe), and `heylook_progress` chunked prefill events.
- [ ] **16B.3 `GGUFSubprocessProvider` `stop_sequences`, SWA/Cache & Reasoning-End Parity**:
  - Align sampling defaults with `SamplerCascade` and fix double `stop_sequences` filtering between `llama-server` `body["stop"]` and `StreamingPipeline` `StopSequenceFilter`.
  - Capture active completion `"id"` from the first SSE frame in `generateWithLiveSubprocess` and pass `{"id": cmplID, "action": "reasoning_end"}` to `/v1/chat/completions/control`.
  - Wire SWA (`sliding_window`) and `linear_attention` checks into `GGUFSpawnConfig`, mark `--cache-reuse` `not_applicable` when `mmproj != nil` or on hybrid/SWA models, and pass `--reasoning-preserve` and `--checkpoint-min-step` matching `llama_server_provider.py`.
- [ ] **16B.4 `EmbeddingGemma 2`, `CoreAI` & `EvalLabSuite` Parity**:
  - Replace hardcoded `28 px` geometry assumptions in `image-plan` with checkpoint-derived vision patch size and spatial merge factors.
  - Verify `EmbeddingGemma 2` hybrid ANE+GPU towers, `embedding.limits`, and `video_windows` (`fps` default + scaled `content` budget) against live checkpoints.
  - Fix the `EvalLabSuite.runAll(allowSyntheticSelfTest: false)` fallthrough bug (line 2961) so live mode never falls into synthetic hash/echo probes when real models exist.

---

## Phase 16C: Modular `/v1/*` Server & Canonical Saga (`HeylookServer`)

- [ ] **16C.1 Decompose `HeylookHTTPServer.swift` (3,193 lines)**:
  - Split `HeylookHTTPServer.swift` into cohesive modules (`HTTPServerTransport`, `HeylookRouter`, `MessagesRoutes`, `ConversationsRoutes`, `ModelsAndAdminRoutes`, `EmbeddingsRoutes`, `NotebooksAndPresetsRoutes`, `PerfAndObservabilityRoutes`, `EvalLabSuite`) sharing one host-check, `X-Request-ID`, admin-token (`X-Heylook-Admin-Token`), `HEYLOOK_READONLY_MODEL_CONFIG`, and JSON error middleware path.
- [ ] **16C.2 Canonical Conversation Generation Saga (`/v1/conversations/{id}/generate`)**:
  - Refactor `handleConversationGenerate` to perform atomic post-generation persistence (`replaceTailWithMessage` on `regenerate` and `replaceTailWithUpdate` on `continue`, preserving anchor `modelID`), emit exact `heylook_progress`, `message_stop.performance`, and `heylook_saved` SSE frames, and act as the single canonical execution path for both HTTP clients and the native SwiftUI workbench.
- [ ] **16C.3 Schema-Driven Admin & Public Model Endpoints**:
  - Serve path-free `PublicModelWireRow` on `GET /v1/models`, `AdminModelWireRow` on `GET /v1/admin/models`, declarative `ProviderConfigSchema` options on `GET /v1/admin/model-options`, `422` validation + `stale_reload_fields` on `PATCH /v1/admin/models/{id}`, model-derived geometry on `POST /v1/models/{id}/image-plan`, and newest-first JSONL tailing on `GET /v1/admin/config/logs`.

---

## Phase 16D: Full `frontend/` 6-Surface Parity & ViewModel Decomposition (`HeylookUI`, `HeylookApp`)

- [ ] **16D.1 Decompose `HeylookWorkbenchViewModel.swift` (4,507 lines)**:
  - Break the monolithic ViewModel into focused `@Observable` coordinators (`ChatCoordinator`, `ModelRegistryCoordinator`, `NotebooksCoordinator`, `PresetsCoordinator`, `EmbeddingsLabCoordinator`, `TelemetryAndServerCoordinator`), routing all Chat generation/regeneration/continuation through the canonical conversation generation coordinator (eliminating UI bypass of `/v1/conversations/{id}/generate` locks and atomic tail persistence) and removing `"Load Sample Demo Workspace"` / `"Load Sample Demo Corpus"` (`allowSyntheticFallback: true`) from production UI.
- [ ] **16D.2 Chat & Engine Panel Parity (`chat.js`, `engine.js`, `prompt-preview.js`)**:
  - Render exact tokenizer-backed prompt markers and token counts in `ChatPromptPreviewDrawer` (removing hardcoded `specialTokens` and `chars/4` approximations).
  - Upgrade `ChatEngineReadoutBar` and inspector to display full `EngineContract` facts (`context.running` vs ceiling, `cache` `causeWords`, `speculative` acceptance/draft share, `template.caps`, thinking-budget sub-row) and honor `SQLiteStore.mutationStream()` / active generation lock state cleanly across surfaces.
- [ ] **16D.3 Notebook & Presets Parity (`notebook.js`, `presets.js`)**:
  - Add `"Save to Conversations"` (`kind: "notebook"`) persistence and `refreshAfterResume` parity to Notebook.
  - Update Presets defaults/placeholders (`1.0 / 0.95 / 16384`), rename/clear UX, and strict `SamplerParams` validation (`params` DTO).
- [ ] **16D.4 Embeddings Compare + Embedding Lab Parity (`embeddings.js`, `embed-kit.js`)**:
  - Align the Embeddings surface with `heylookitsanllm` `v2.0.244` (`Compare` + `Embedding Lab` workflows, `engine.embedding.limits` preflight, windowed video preview with `frame_times` jump pills + **Send to Embedding Lab**, built-in 8-case multimodal sample set + `set.json` folder runner) without synthetic hash corpus leaks.
- [ ] **16D.5 Models & Perf Surface Parity (`models.js`, `model-config.js`, `perf.js`)**:
  - Drive `ModelSidecarEditorSheet` from `ProviderConfigSchema` (`ui: normal | advanced | hidden`, hiding `isFileOnly` fields, displaying `stale_reload_fields`, `RAMFitAssessment` meter, and full `sources` template ladder + `refused_shapes`).
  - Upgrade `PerfServerSurface` to display `timing_breakdown`, per-model `cache` stats (`cache_share`, `outcomes`, `causes`), 8-hour `trends`, and live `events`/`metrics` JSONL log controls (`observability_level`, `observability_retention_days`).

---

## Phase 16E: Contract Suite & Live Certification (`HeylookTests`, `heylook-cli`)

- [ ] **16E.1 Pure-Invariant & Route-Through Contract Suite**:
  - Update and harden `HeylookTests` to eliminate reliance on `0.7 / 40 / 2048` defaults, seeded bundled presets, and `allowSyntheticFallback: true` in production contract tests; add dedicated assertions for `EngineContract`, `ProviderConfigSchema`, `TOMLCodec`, probe-derived `ThinkingControls`, `SQLiteStore` atomic tail & scoped `media_blobs`, `MessagesWire` `400`/`422` rules, `StopSequenceFilter`, and truthful `EvalLabSuite` failure when models are absent.
- [ ] **16E.2 Live Hardware Certification (`MLX`, `GGUF`, `EmbeddingGemma 2`, `CoreAI`)**:
  - Execute live `heylook-cli eval-lab`, multi-turn `chain_probe`, `vlm_parity_probe` (including Gemma 4 lossless MTP & VLM cache reuse), GGUF SWA/`--reasoning-preserve`/MTP probes, and three-arm HTTP/SSE smoke checks across `macOS 27.2+` and `iOS 27.2+` Simulator.
  - Refresh `docs/wiki/*.md` and `docs/USER_GUIDE.md` so all subsystem docs match the post-16A–16E implementation.

---

## Post-Parity Horizons (After Phase 16E Baseline Is Green)

- [ ] **H.1 `macOS 27.3+` / `iOS 27.3+` Apple Intelligence & Spotlight Integration**:
  - Expose sovereign local capabilities via `#available(macOS 27.3, iOS 27.3, *)` Siri `@AppIntent(.tool)` (`ToolIntent`) actions (local inference, ANE audio transcription, multimodal retrieval) and Spotlight `IndexedEntity` indexing backed by `SQLiteStore` and `VectorStore`.
- [ ] **H.2 `LiteRT` `RuntimeAdapter` & Composable Model Specializations**:
  - Implement a `LiteRT` provider conforming to the unified `RuntimeAdapter` and `EngineContract` (`describeObserved()`), alongside modular, retirement-tested specialization hooks for active model families.
- [ ] **H.3 Live `Parakeet TDT` vs. `Apple SpeechTranscriber` ANE Benchmark**:
  - Run head-to-head warm/cold latency, Real-Time Factor (`RTFx`), RSS delta, and multi-clip accuracy comparisons over `POST /v1/audio/transcriptions` and `POST /v1/conversations/{id}/voice-generate`.
- [ ] **H.4 Physical `iOS 27.2+` / `27.3+` Device Deployment & Durable TCC/FDA Workflows**:
  - Package and test signed `HeylookWorkbench` builds on physical iPhone/iPad (`CoreAI` `.aimodel` on-device + Bonjour `_heylook._tcp` Mac Studio failover) and verify Apple Development-signed TCC protected-folder & Full Disk Access custody across macOS rebuilds (`scripts/build_native.sh`).

---

## Historical Port Log (Phases 0–15 — Pre-Parity Initial Scaffold)

> **Note:** Phases 0–15 built the initial Swift 6.4 multi-target package (`HeylookCore`, `HeylookEmbeddings`, `HeylookEngines`, `HeylookServer`, `HeylookUI`, `HeylookApp`, `heylook-cli`, `HeylookTests`). Earlier status notes overstated parity readiness against `heylookitsanllm`; the verified gaps from that baseline are now tracked in Phases **16A–16E** above.

- **Phases 0–3 (Scaffold, Initial Providers, `/v1/*` Stubs & Bonjour)**: Created multi-target `Package.swift` (`mlx-swift`, `mlx-swift-lm`, `swift-transformers`, optional `coreai-models`), dual-lane `ComputeScheduler`, initial `SQLiteStore` / `VectorStore` (`vDSP` Matryoshka), Bonjour `_heylook._tcp` discovery (`BonjourServiceBrowser`), and initial SwiftUI workbench.
- **Phases 4–10 (MMProj, Vendor Sampling, Metal Build, `SubprocessLifecycleRegistry`, GGUF Header Reader, `EvalLabSuite`)**: Wired `GGUFSubprocessProvider`, `--mmproj` forwarding, `SubprocessLifecycleRegistry`, `GGUFPortPool`, `AVAssetReader.outputProvider(for:)`, `AVAudioConverter` 16 kHz resampling, and initial `EvalLabSuite` / `heylook-cli` commands.
- **Phases 11–15 (4-Protocol `ReasoningParser`, `HeylookJinjaRenderer`, File Splits, Observation Caching, Dual-Tier ANE ASR)**: Added `HeylookJinjaRenderer`, 4-protocol `ReasoningParser`, split initial `Providers.swift` and `HeylookWorkbenchView.swift` into focused files, added `CoreAISpeech` / `SpeechTranscriber` ANE voice transcription + `/v1/audio/transcriptions`, and added `PresetsSurface`, `MarkdownMessageView`, and `CursorAwareTextEditor`.
