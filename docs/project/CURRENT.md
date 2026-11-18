# `heylookitsanllm-macos` — Current Status & Readiness Audit

**Benchmark Baseline**: `heylookitsanllm` `v2.0.251`  
**Toolchain / Targets**: Swift 6.4 (`v6` language mode) on `macOS 27.2+` (`27.2.1` hosts + `#available(macOS 27.3, iOS 27.3, *)` guards) & `iOS 27.2+` Simulator  
**Overall Parity Status**: **Waves 1–4 Modular Foundation Landed (`fd6a3c1`); Phase 17 (`Waves A–E`) Holistic Simplification & `v2.0.251` Parity Hardening Queued**  
**Execution Blueprint**: See **[PARITY_EXECUTION_PLAN.md](./PARITY_EXECUTION_PLAN.md)**

---

## 1. Subsystem Readiness Matrix

> [!IMPORTANT]
> An adversarial code-level audit of `fd6a3c1` against `heylookitsanllm` (`v2.0.251`) and upstream `mlx-swift-lm` established that while Waves 1–4 delivered the modular 6-target package layout, SQLite `WAL` foundation, and 28 non-MLX offline tests, native MLX execution, live GGUF subprocess parity, `/v1/*` wire schemas, and 6-surface workbench flows still contain blocking gaps tracked in **Phase 17 (`Waves A–E`)**.

| Subsystem | Target | Audit Status | What Works Today (`fd6a3c1`) & What Phase 17 Closes Next |
| :--- | :--- | :--- | :--- |
| **Native Build & Metal Packaging** | `Package.swift`, `scripts/` | **WAVE A QUEUED** | `build_native.sh` compiles Metal shaders for `.app`, and offline `swift test` passes with MLX compiled out (`enableNativeMLX = false` by default). **Phase 17 Wave A** enables native MLX by default on macOS, pins `mlx-swift-lm`, links upstream `MLXGuidedGeneration` & `MLXEmbedders`, stages a portable `.build/native/bin/heylook-cli` + `mlx-swift_Cmlx.bundle`, runs `#if canImport(MLX)` tests, and gates post-parity ASR / `Media Corpus` / `coreai` UI/routes. |
| **Engine Contract, Samplers & Templates** | `HeylookCore` | **WAVE C QUEUED** | `EngineContract`, `SamplerCascade`, AST-probing `ModelTemplateInfo`, `TOMLCodec`, and `ReasoningParser` compile and pass offline checks. **Phase 17 Wave C** aligns `CacheReport`/`SpeculativeReport` snake_case fields, canonical `reasoning_effort` & `thinking_budget_tokens` wire keys, `engine.embedding.limits` / `engine.runtime = "mlx-vlm"`, comment-preserving `TOMLCodec` single-path saves, `scan_interval_seconds = 0`, multi-GGUF sidecar isolation, and drops unimplemented `pipeline_ratios`/`pipeline_peers`. |
| **SQLite WAL Store & Observability** | `HeylookCore` | **WAVE C QUEUED** | `WAL` store, scoped 64-hex `media_blobs`, unseeded `presets`, atomic tail sagas, and JSONL disk logs are wired. **Phase 17 Wave C** fixes `COALESCE` image-clearing on edited messages, copies `message_stats` on clones, includes `resource_timeline` + stateless `/v1/messages` in `/v1/performance/profile`, and removes fabricated `generation_tps`. |
| **MLX, MTP, VLM & GGUF Engines** | `HeylookEngines` | **WAVE B QUEUED** | Modular providers, `RAMFitCalculator`, `ModelSpecialization`, `ComputeScheduler` exist structurally. **Phase 17 Wave B** fixes Turn-2 APC axis-1 / `PreparedInputSplitting` + `LMOutput.State` + multi-boundary capture, deletes `HeylookStagedRotatingKVCache` & fake `VisionFeatureCache`, repairs 3D `qVec` + Mamba checkpointing in `LeviathanMTPTokenIterator`, enables embedded `Qwen3.5` MTP & diffusion dispatch, integrates `MLXGuidedGeneration` (`response_format`), enforces `generated_only` penalties, fixes `/image-plan` `409` + `THW * patchSize`, wires `mlx_cache_limit_gb` & idle reaper, and hardens GGUF vision `n_ubatch`, `KNOBS_OFF`, cache witness, and live `proc_pidinfo` RSS. |
| **Multimodal & GGUF Embeddings** | `HeylookEmbeddings` | **WAVE B QUEUED** | `VectorStore`, `ArtifactProcessor`, and `GGUFEmbeddingEngine` foundation landed without synthetic fallbacks. **Phase 17 Wave B** deletes broken custom `computeAudioLogMel`/`prepareImagePixels` in favor of upstream `MLXVLM.EmbeddingGemma2AudioProcessor` / `ImageProcessor`, inspects `TextANE.mlmodelc` shapes dynamically, links `MLXEmbedders`, fixes `GGUFEmbeddingEngine` `stderr` pipe deadlock + batched `/embeddings` + arbitrary `384d`–`4096d` dims, pools `768d` video windows before Matryoshka slice, and preserves single-window timestamps. |
| **HTTP/SSE Wire API & Saga** | `HeylookServer` | **WAVE C QUEUED** | Keep-Alive HTTP/1.1 parser, `PatternRouter`, and `5s` SSE ping work in-process. **Phase 17 Wave C** registers `DELETE /v1/conversations/{id}/generate`, `POST .../reasoning-end`, and `/openapi.json`; aligns `/prompt`, `/fit`, `/reload`, `/chat-template`, and `/config`; fixes `abort` `stop_reason` / `heylook_saved.end_reason` and pre-lock saga race; and restores static web v3 hosting + IPv6 `[::1]` `Host` checks. |
| **6-Surface SwiftUI Workbench** | `HeylookUI`, `HeylookApp` | **WAVE D QUEUED** | Modular 6-controller `@Observable` architecture and surfaces (`⌘1`–`⌘6`) compile on macOS and iOS Simulator. **Phase 17 Wave D** parks non-parity ASR mic/`Media Corpus`/Notebook macro clutter, fixes deleted-image resurrection, `Save & Regenerate`, `Preview prompt`, `"Fit W×H"` label, `heylook_saved` recovery + `503` retry, inline Load bar, preset `Save…` scope, Notebook & Embeddings Studio bypasses (`WorkbenchService` spine), shared-pool `Recall@3` math, removes startup fake DB seeding, fixes Models/Perf/Markdown/`⌘,` Settings. |
| **Contract, Metal & Live Test Matrix** | `HeylookTests` | **PARTIAL (`28` non-MLX)** | `28` offline non-MLX unit/contract tests pass (`swift test` with MLX disabled). **Zero `#if canImport(MLX)` tests, zero live checkpoints, zero live `llama-server` runs, and zero live web v3 browser sessions have run yet** — **Phase 17 Wave E** verifies the full matrix. |

---

## 2. Verified Baseline Checks (`fd6a3c1`)

- **Offline Unit & Contract Suite (`swift test`, MLX compiled out)**: **`28` tests across `5` suites passed** (`0` failures). Note: `#if canImport(MLX)` codepaths remain untested until **Wave A + Wave E** execute `xcodebuild test` with `default.metallib`.
- **Cross-Platform Target (`iOS 27.2` Simulator)**: `swift build --target HeylookUI --triple arm64-apple-ios27.2-simulator` built with `0` errors.
- **Static Hygiene Checks**: `allowSyntheticFallback` and `DeterministicHashEmbeddingEngine` remain removed from `Sources/`.

---

## 3. Deferred Beyond Parity (Post-Parity Horizons)

Per explicit owner prioritization (`heylookitsanllm` `v2.0.251` parity before net-new expansion), the following items are intentionally parked or gated until **Phase 17 (`Waves A–E`)** passes live verification:

1. **Multi-Mac Cluster Sharding**: Distributed `mlx.distributed` (`pipeline_peers`, `pipeline_ratios`, Thunderbolt RDMA / ring) and `ggml-rpc` (`llama-server --rpc`).
2. **Forkless Upstream `mlx-swift-lm` Strategy**: Consuming upstream `mlx-swift` and `mlx-swift-lm` untouched for parity; upstream-invasive hooks (post-ViT `MLXArray` feature caching inside `LanguageModel.prepare`, custom Gemma 4 Metal `uint2` QMV kernels) move to **Horizon 2** / upstream PRs.
3. **Voice ASR / Parakeet & Apple Speech**: Live mic dictation composer button, CLI `transcribe`, `/v1/audio/transcriptions`, and `/v1/conversations/{id}/voice-generate` remain gated behind `HEYLOOK_ENABLE_EXPERIMENTAL_ASR=1` (`H.1`).
4. **Local `Media Corpus (mbed)` UI & `CoreAIProvider` (`.aimodel`)**: Retired in `heylookitsanllm` (`v2.0.243`) / native-only extensions remain gated behind `HEYLOOK_ENABLE_EXPERIMENTAL_CORPUS=1` and `HEYLOOK_ENABLE_COREAI=1`.
5. **Physical `iOS 27.2+` Pairing, `LiteRT` Adapter, & `27.3+` OS APIs**: `LiteRT` behind `InferenceProvider`, plus `#available(macOS 27.3, iOS 27.3, *)` Siri `AppIntents` (`ToolIntent`), Spotlight `IndexedEntity`, and durable TCC custody.
