# `heylookitsanllm-macos` — Parity Roadmap & Next Horizons

Master parity completion log against `heylookitsanllm` `v2.0.251`, active parity-hardening backlog (**Phase 17**), and parked post-parity horizons.

> **Detailed Engineering Blueprint**: [`PARITY_EXECUTION_PLAN.md`](./PARITY_EXECUTION_PLAN.md)

---

## Phase 17 — Holistic Simplification & True `heylookitsanllm` `v2.0.251` Parity *(Waves A–E)*

> Adversarial red-team review of `fd6a3c1` identified concrete build, MLX/MTP, GGUF, embedding DSP, HTTP/SSE wire, `heylook.toml`, and SwiftUI controller gaps. Wave A–E prioritize **deleting duplicate bespoke wrappers** in favor of upstream `mlx-swift-lm` primitives and one shared `WorkbenchService` spine.

### Wave A — Native Build, Metal Staging & Scope Gates *(Gate)*
- [ ] **A.1 Package Defaults & Upstream Libraries**: Default `enableNativeMLX = true` on macOS (opt-out `HEYLOOK_DISABLE_NATIVE_MLX=1`), pin `mlx-swift-lm` commit revision, keep `enableCoreAI` opt-in, and link upstream `MLXGuidedGeneration` & `MLXEmbedders` in `Package.swift`.
- [ ] **A.2 Portable CLI & Metal Shader Tests**: Stage portable `.build/native/bin/heylook-cli` alongside `mlx-swift_Cmlx.bundle` (`default.metallib`), preflight Metal shader bundles before crashing, and run `#if canImport(MLX)` tests via `scripts/build_native.sh test`.
- [ ] **A.3 Truthful `heylook-cli doctor`**: Derive MLX, `default.metallib`, `llama-server`, and optional CoreAI/ASR lines from compile + runtime probes.
- [ ] **A.4 Feature Gates & Subtraction**: Add `FeatureFlags` (`HEYLOOK_ENABLE_EXPERIMENTAL_ASR`, `HEYLOOK_ENABLE_EXPERIMENTAL_CORPUS`, `HEYLOOK_ENABLE_COREAI`) + `RuntimeConfigurationApplying` protocol; remove dead `pipeline_ratios` / `pipeline_peers` schema fields and Notebook summary/analyze macros.

### Wave B — Engines: MLX, MTP, GGUF & Embeddings *(Track 1)*
- [ ] **B.1 Turn-2 APC & `LMOutput.State`**: Fix `[0, seqLen]` axis-0 slice bug, adopt `PreparedInputSplitting` + `_apc_media_token_ids` span protection, add multi-boundary capture (`system` + prefill-end) for `MambaCache` / saturated `RotatingKVCache`, and propagate `LMOutput.State` (`ropeDeltas`) into all iterators.
- [ ] **B.2 `LeviathanMTPTokenIterator` Cleanup**: Delete `HeylookStagedRotatingKVCache` (`RotatingKVCache.copy()` verify instead), squeeze `pVec`/`qVec` to `[1, vocab]`, set `mtpCacheCheckpointIndexKey = 1` + `rewindSpeculativePromptCache` for `Qwen3.5` Mamba MTP, and wrap forwards with `withPreparedCache`.
- [ ] **B.3 Embedded MTP & Diffusion Dispatch**: Load in-checkpoint `mtp.*` weights via `MTPDrafterModelFactory` (`Qwen35CheckpointPolicy(layout: .embedded)`), and dispatch masked-diffusion models via `NemotronLabsDiffusionModel` (or explicit `400`).
- [ ] **B.4 `MLXGuidedGeneration` & Unified `GenerationComponents`**: Wire `json_object`/`json_schema` constraints via `MLXGuidedGeneration`, gate off spec decoding cleanly, apply `logit_bias`, enforce `generated_only` penalties (`prompt(_:)` no-op), inject `RequestRegistry`, preserve rank, add `.harmonyChannel`/`.recipientChannel`, thread through all iterators, and load `generation_config.json` `eos_token_id` arrays.
- [ ] **B.5 `/image-plan` & Clean VLM Inputs**: Enforce resident-only `409 model_not_loaded`, multiply `THW * vision_config.patch_size`, send `"add_generation_prompt": false` on GGUF token probes, delete fake `VisionFeatureCache` + premature `28px` resize, and harden `PreRenderedVLMTokenizerProxy` `encode` path.
- [ ] **B.6 Runtime Config, Metal Cache Limit & Idle Reaper**: Wire `MLX.GPU.set(cacheLimit:)`, `WiredMemoryTicket`, global/per-model `idle_unload_seconds` + memory-pressure eviction, and all MLX (`ctx_size`, `kv_bits`, `kv_group_size`, `prefill_step_size`, `draft_*`, `chat_template_source`) & GGUF (`spec_*`, `n_cpu_moe`, `batch_size`, `ubatch_size`, `cache_ram_mb`, `ctx_checkpoints`, `checkpoint_min_step`, `extra_args`) settings; remove `max_position_embeddings` draft fallback.
- [ ] **B.7 GGUF Hardening & Live RSS**: Clamp `--image-max-tokens <= effectiveUbatch`, use `RAMFitCalculator.thinHeadroomGB` (`24.0`), gate `--reasoning-preserve` & switchless `enable_thinking`, emit `KNOBS_OFF`, stop re-parsing `llama-server` output via `ReasoningParser`, fix `StreamingPipeline` token counting, parse `server_prompt_cache` witness logs, auto-discover `<stem>-mtp` drafters with crash fallback, unify `llama-server` lookup, and report `proc_pidinfo` RSS.
- [ ] **B.8 Embeddings Repair**: Delete broken hand-rolled `computeAudioLogMel`/`prepareImagePixels` in favor of upstream `MLXVLM.EmbeddingGemma2AudioProcessor`/`ImageProcessor`; inspect `TextANE` `MLModelDescription` shapes dynamically; support `MLXEmbedders`; drain `GGUFEmbeddingEngine` `stderr`, batch `/embeddings`, register/unload cleanly, and support arbitrary `384d`–`4096d` dims; pool `768d` video windows before Matryoshka slice and keep single-window timestamps.

### Wave C — Core Wire Contract, SQLite Store & `heylook.toml` *(Track 2)*
- [ ] **C.1 Missing Routes**: Register `DELETE /v1/conversations/{id}/generate`, `POST /v1/conversations/{id}/generate/reasoning-end`, and `GET /openapi.json`.
- [ ] **C.2 Canonical Keys**: Snake-case `CacheReport` & `SpeculativeReport`, canonical `reasoning_effort` & `thinking_budget_tokens` wire keys, and forward `repetition_context_size`.
- [ ] **C.3 `/v1/conversations/{id}/prompt`**: Support structured `edits` (`content`/`thinking`), `{images, audio}` `unrendered_media`, `char_count`, `mode`, `provider`, `continuation: true` on `user` anchors, and GGUF `reasoning_content` + `parts`.
- [ ] **C.4 `EngineContract` & Admin API**: Align `engine.embedding.limits`, `engine.runtime = "mlx-vlm"`, template-less `null` thinking, 7-key `engine.cache`, `null` decoding fields on non-diffusion, GGUF `flash_attn.auto = "on"`, `/chat-template` (`heylook_override` + `_PROBE_SHAPES`), `/fit` (`422` missing weights + `JSONValue` overrides), `/reload` (`400` MLX `ctx_size`), `PUT /v1/admin/config` `"memory_only"` + `null` reset, and `/v1/performance/profile` `resource_timeline` + stateless `/v1/messages`.
- [ ] **C.5 Messages, SSE, Saga & Explicit `null`s**: Echo `X-Request-ID`, register `requestID` before load, apply `StripSpecials`, detect open-thought prefill, `422` retired fields, stop fabricating `generation_tps`, SSE abort `stop_reason = "max_tokens"`, `heylook_saved.end_reason` enum, lock before snapshot, `0`-based initial positions, explicit `null` JSON keys, and SSRF guard in `embed`.
- [ ] **C.6 `SQLiteStore`**: Clear `content_blocks_json` when edits remove all images (`COALESCE` fix) and preserve `message_stats` on `cloneConversationWithMessages`.
- [ ] **C.7 Unified `TOMLCodec` & Multi-GGUF Isolation**: Unify `loadDefault()` / `activeConfigURL()`, route all writes/removals (incl. nested `[models.config]`) through comment-preserving `TOMLCodec`, honor `scan_interval_seconds = 0`, and key sidecars/weights per `.gguf` file.
- [ ] **C.8 Static Web v3 & HTTP Transport**: Auto-discover/serve web v3 `/`, `icon.svg`, `/favicon.ico`, `Cache-Control: no-cache` on `/js/*` & `/css/*`, `HEAD` `Content-Length`, and IPv6 `[::1]` `Host` header.

### Wave D — SwiftUI 6-Surface Workbench Parity *(Track 3)*
- [ ] **D.1 Chat Editor & Attachments**: Persist deleted images cleanly, emit `"thinking": null` on remote PATCH, hide `Save & Regenerate` on assistant turns, fix `Preview prompt` + block ordering, and show `"Fit W×H"` ahead of `"Full res"` on `>2048px` photos.
- [ ] **D.2 Stream Recovery & Live Controls**: Keep `heylook_saved` partials on subsequent `event: error`, add `503 Retry-After` (`MAX_BUSY_RETRIES = 3`), show live Stop / End Thinking when `generating == true`, and enforce `refuseWhileUnsaved`.
- [ ] **D.3 Contract Wiring & Inline Load Bar**: Wire `activeContractProvider`, keep `activeAdminRow` fresh across surfaces, expose inline Load/Reload bar in Chat for MLX & GGUF, and commit custom `ctx_size` on submit/blur.
- [ ] **D.4 Presets & Sliders**: Fix Inspector `Save…` scope (`"full"` / `"sampler"`), route Presets `Apply` per surface, clamp `Presence Penalty` to `0.0...2.0`, and store `Stop Sequences` without comma-splitting `\n\n`.
- [ ] **D.5 Unified Notebook Spine**: Route Notebook generation and remote `Save to Chat` through `WorkbenchBackend` (`WorkbenchService`).
- [ ] **D.6 Unified Embeddings Studio & `Recall@3` Pool**: Route `embed` through `WorkbenchBackend`, rank `Embedding Lab` across the deduplicated ~20-item shared benchmark pool, remove fake `seedSampleCorpusIfNeeded`, derive honest readiness, and keep security-scoped URL access.
- [ ] **D.7 Models, Perf, Markdown & `⌘,` Settings**: Persist `toggleModelPin`, use `patchAdminModel` & comment-safe TOML in Models, gate MLX `Load` on `RAMFit` `fail`, add draft guard + `Clear all conversations & notebooks`, show live GGUF RSS & ANE queue state in Perf, fix Markdown loose lists/images/URLs, and wire `⌘,` Settings inputs.

### Wave E — Native Metal Verification & Live Smoke Gate *(Merge & Verify)*
- [ ] **E.1 Clean Tree & Path Privacy**: `find-external-paths.sh -d .` exits `0`; verify subtraction targets removed cleanly.
- [ ] **E.2 Dual Build**: `./scripts/build_native.sh build` (portable CLI + `default.metallib`) and `iOS 27.2` Simulator build green.
- [ ] **E.3 Full Swift Testing (`#if canImport(MLX)` Active)**: `./scripts/build_native.sh test` passes all unit & contract suites.
- [ ] **E.4 Live Hardware Matrix**: Verify live MLX dense/thinking/JSON-schema/APC, VLM `/image-plan` + Fit, `Qwen3.5` MTP & `Gemma 4` past `sliding_window`, GGUF vision/witness/RSS/`KNOBS_OFF`, and batched multimodal/GGUF embeddings.
- [ ] **E.5 Live Web v3 Check**: Serve `heylookitsanllm` web v3 frontend via `HEYLOOK_FRONTEND_DIR` and verify all 6 pages cleanly.

---

## Completed Foundation Waves *(Phases 0–16E, `261288b`–`fd6a3c1`)*

- **Phase 16A–16E (`Waves 1–4`)**: Modular 6-target package layout (`HeylookCore`, `HeylookEmbeddings`, `HeylookEngines`, `HeylookServer`, `HeylookUI`, `HeylookApp`, `heylook-cli`), `EngineContract`, `SamplerCascade`, AST `ModelTemplateInfo`, comment-preserving `TOMLCodec`, `SQLiteStore` WAL spine, modular `HeylookServer` routes, 6-controller `@Observable` SwiftUI refactor, synthetic-fallback deletion (`allowSyntheticFallback`, `DeterministicHashEmbeddingEngine` removed from `Sources/`), and 28 offline unit/contract tests.
- **Phases 0–15**: Initial native SwiftPM scaffold, Jinja2 renderer, `ReasoningParser`, dual-lane scheduler skeleton, `EmbeddingGemma 2` foundation, Swift 6.4 strict concurrency migration, and `macOS 27.2+` / `iOS 27.2+` (`27.3` guarded) split views.

---

## Post-Parity Horizons *(Deferred Until Phase 17 Is Green)*

- [ ] **H.1 — Multi-Mac Distributed Cluster Inference**: `mlx.distributed` (`ring` / `jaccl` Thunderbolt RDMA, `pipeline_peers`, `pipeline_ratios`) and `ggml-rpc` (`llama-server --rpc`) behind `ComputeScheduler`.
- [ ] **H.2 — Forkless Upstream `mlx-swift-lm` Contributions & Kernels**: Upstream post-ViT `MLXArray` feature caching inside `LanguageModel.prepare` and Gemma 4 Metal QMV/MTP optimizations.
- [ ] **H.3 — Live ASR & Speech Workflows (`Parakeet` / Apple Speech)**: Ungate composer mic dictation, `/v1/audio/transcriptions`, `/v1/conversations/{id}/voice-generate`, and benchmark `Parakeet TDT` / `Whisper` `.aimodel` vs `SpeechAnalyzer` / `SpeechTranscriber`.
- [ ] **H.4 — Local `Media Corpus (mbed)` & `CoreAIProvider` (`.aimodel`)**: Ungate corpus indexing inspector and Neural Engine `.aimodel` LLM scheduling beyond `mlx` + `gguf`.
- [ ] **H.5 — Multi-Runtime Extension (`LiteRT`), Physical `iOS 27.2+` & `27.3+` OS APIs**: `LiteRT` behind `InferenceProvider`, physical iPhone/iPad Bonjour/Tailscale validation, and `#available(macOS 27.3, iOS 27.3, *)` Siri `AppIntents` (`ToolIntent`), Spotlight `IndexedEntity`, and durable TCC / Full Disk Access custody.
