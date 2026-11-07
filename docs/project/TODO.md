# Backlog & Phased Roadmap (`heylookitsanllm-macos`)

Tracks the phased port of `heylookitsanllm` + `mbed` into a unified Swift 6.2 multiplatform application and embedded server targeting `macOS 27.2+` and `iOS 27.2+`.

---

## Phase 0: Architecture, Vision & Multi-Target Swift 6.2 Scaffold (Done)

- [x] **0.1 Governance & Documentation**: `VISION.md`, `AGENTS.md`, `README.md`, `docs/wiki/architecture_overview.md`, `docs/project/CURRENT.md`, `docs/project/TODO.md`.
- [x] **0.2 Remote-First `Package.swift` (`macOS 27.2+`, `iOS 27.2+`)**: Remote GitHub URLs (`ml-explore/mlx-swift`, `ml-explore/mlx-swift-lm`, `apple/coreai-models`, `argmaxinc/argmax-oss-swift`, `hummingbird-project/hummingbird`, `huggingface/swift-transformers`) with fast default `swift test` and full native `scripts/build_native.sh`.
- [x] **0.3 `HeylookCore`**: `EngineDescription`, `SamplerCascade`, `ReasoningParser` + `StripSpecials`, `HeylookConfig` (`heylook.toml` & `model.heylook.toml`), `SQLiteStore` (`WAL`), `RequestRegistry`, and `MessagesWire`.
- [x] **0.4 `HeylookEmbeddings` (folded from `mbed` at full fidelity)**: `Artifact`, `VectorStore` (`vDSP` Matryoshka), `WindowPlanner`, `ArtifactProcessor`, `FolderWatcher` (`FSEvents`), `IndexCoordinator` (`indexURLs`, `discoverFiles`, `isUpToDate`, `pruneMissingArtifacts`, `buildChunks`), `HeylookMLXEngine`, `NativeMLXEmbeddingGemma2` (`MLXSwiftLMEmbeddingGemma2Engine` + `HybridLocalEmbeddingEngine`), `EmbeddingLabRunner`, `ModelComputeOptions`, `ANECoreMLRuntime` (`MLMultiArray` Vision/Audio/Text ANE prediction + 512-point `vDSP` FFT & 128-bin HTK mel filterbank), and `ANETowerInspector`.
- [x] **0.5 `HeylookEngines`**: Dual-lane `ComputeScheduler` (GPU FIFO gate + ANE co-resident lane), `HardwareMemoryMonitor`, `GGUFHeaderReader`, `MLXSwiftProvider`, `CoreAIProvider`, `GGUFSubprocessProvider`, and `RemoteHeylookProvider`.
- [x] **0.6 `HeylookServer`**: Embedded `/v1/*` server with disconnect-resilient `/v1/conversations/{id}/generate` and `heylook_saved`.
- [x] **0.7 `HeylookUI`, `HeylookApp` & `heylook-cli`**: Multiplatform 5-surface SwiftUI workbench (`Chat`, `Notebook`, full 3-column `mbed` `Embeddings & Media Studio` with inline `AVPlayerView`, clickable `frameTimes` jump pills, and `ScoredFeatureBreakdownCard`, `Models & Inspector`, `Perf & Server`), MenuBar controller, and CLI (`index`, `search`, `watched`, `inspect-ane`, `inspect-checkpoint`).

---

## Phase 1: Deep Engine Specialization & Live Provider Bridges (Done)

- [x] **1.1 `MLXSwiftProvider` Prefix Cache (APC) & Live `ModelContainer` Bridge**: Wired `PrefixCacheTracker` (`cold`, `reused`, `system_changed`, `history_diverged`), `LLMModelFactory` / `VLMModelFactory` `ModelContainer` loader, `GenerateParameters`, and mid-stream `reasoning-end` & `abort`.
- [x] **1.2 `CoreAIProvider` `.aimodel` Manifest & Speculative Telemetry**: Wired `CoreAIBundleManifest.inspect(at:)` (`metadata.json` for `main`, `vision`, `drafter`, `maxContextLength`, `prefillChunkSize`), provenance enrichment (`.observed`), and `SpeculativeReport`.
- [x] **1.3 `GGUFSubprocessProvider` & `RemoteHeylookProvider` Live Bridges**: Wired `GGUFSpawnConfig`, `buildLlamaServerArgs`, `childEnvironment` (`LLAMA_ARG_LOG_FILE` removal + `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=604800`), `parseOpenAISSEDataLine` (`content`, `reasoning_content`, `timings`, `draft_n`), `/v1/chat/completions/control` `reasoning_end`, and `RemoteHeylookProvider` SSE streaming (`parseAnthropicSSEEvent`, `forwardReasoningEnd`, `forwardCancel`).

---

## Phase 2A: Drop-In `/v1/*` Server & Existing `heylook.toml` Compatibility (Done)

- [x] **2A.1 Streaming `POST /v1/messages` (`stream: true`)**: Emit full Anthropic SSE frame sequences (`message_start`, `content_block_delta` for `thinking_delta` and `text_delta`, `message_delta` with `usage` + `performance.cache` + `performance.speculative`, and `message_stop`).
- [x] **2A.2 `POST /v1/embeddings` & `GET /v1/capabilities` Routes**: Serve single/batch/multimodal embeddings on the ANE lane with Matryoshka dimension slicing (`768d`/`512d`/`256d`/`128d`) and report device/media capabilities.
- [x] **2A.3 Multi-Frame `Content-Length` TCP Request Assembly**: Accumulate multi-frame HTTP requests in `HeylookHTTPServer.expectedTotalBytes(for:)` so large base64 media payloads never truncate.
- [x] **2A.4 Python `heylook.toml` Compatibility, 4-Rung GGUF Template Ladder, Auto-Drafters & 192 GB Auto-Sizing**: Parse `[scan] folders = [...]` and `[[models]]` tables; resolve `chat_template_path` > `chat_template.heylook.jinja` > `chat_template.jinja` > `gguf_embedded`; auto-discover `mtp-*`/`dflash-*` sidecars, `MTP/` subfolder drafters, and MLX `mtp.` weights; and auto-size `GGUFSpawnConfig` (`-ub 2048` and `-cram`) from 192 GB unified memory headroom.

---

## Phase 2B: Live Mac Studio Checkpoint & `llama-server` Smoke Pass (Done)

- [x] **2B.1 Live `heylook.toml` Model Discovery Smoke Check**: Pointed `heylook-cli models` and `heylook-cli doctor` at the 192 GB M2 Ultra Mac Studio's model scan folders; verified 26 MLX and GGUF models register with accurate context lengths, 4-rung `chat_template` provenance (`chat_template.heylook.jinja`, `chat_template.jinja`, `gguf_embedded`, `tokenizer_config.json`), per-quant variant directory normalization (`UD-Q6_K_XL/`, `Q8_0/`, `Q6_K/`), and automatic `[MTP]` drafter pairing (`dspark`, `-MTP-`, `-assistant`, and built-in `mtp.`).
- [x] **2B.2 Live ANE Tower & Checkpoint Inspection**: Verified `heylook-cli inspect-ane` (`MLComputePlan` reporting 100% ANE on `EmbeddingGemma2VisionANE.mlmodelc` and 99% ANE on `EmbeddingGemma2AudioANE.mlmodelc` with sandbox readability guard) and `heylook-cli inspect-checkpoint` (`embedding_gemma2` 24L, 512d->768d Matryoshka, 32s video window, <=280 image soft tokens) on the M2 Ultra Mac Studio.

---

## Phase 3: Workbench & Mobile (`iOS 27.2+`) Polish (Done)

- [x] **3.1 Bonjour (`_heylook._tcp`) Auto-Discovery Picker (`iOS 27.2+` & macOS)**: Wired `BonjourServiceBrowser` (`NWBrowser` for `_heylook._tcp`) and `DiscoveredHeylookServer` into `HeylookWorkbenchViewModel` (`startBonjourDiscovery()`, `connectToRemoteServer(_:)`) and `PerfServerSurface` so mobile and remote clients auto-discover and connect to the Mac Studio in one click.
- [x] **3.2 Multimodal Chat Composer, Provenance Badges & Keyboard Shortcuts**: Added `MessagesRequest.MediaAttachment` with dual string/block-array `MessageTurn` `Codable` encoding, HTTP `422` capability gating on `POST /v1/messages`, image/audio attachment pills + drag-and-drop + capability warnings in `ChatWorkbenchSurface`, per-slider `SamplerProvenancePill` (`derived` / `configured` / `request`) + `Reset Overrides` in `LoadAndSamplerPanelSheet`, and keyboard shortcuts (`Cmd+Return` to send, `Escape` to stop, `Cmd+N` for new chat, `Cmd+Shift+P` for prompt preview).

---

## Phase 4: Production Cutover — VLM Media Forwarding & Sidecar Admin (Done)

- [x] **4.1 End-to-End VLM Media Forwarding (`--mmproj` + Vision/Audio Blocks)**: Added `HeylookConfig.discoverMMProj(in:)`, auto-wired `--mmproj` into `GGUFSpawnConfig.autoSized`, implemented `GGUFSubprocessProvider.buildOpenAIChatMessages(from:)` (`image_url` data URIs + `input_audio` blocks), and wired decoded `CIImage` inputs into `MLXSwiftProvider`'s `UserInput`.
- [x] **4.2 Interactive Model Sidecar Editor & `/v1/admin/models/*` Routes**: Added `ComputeScheduler.applySidecarUpdate`, wired `GET/PATCH /v1/admin/models/{id}`, `POST /v1/admin/models/{id}/load`, and `POST /v1/admin/models/{id}/unload` in `HeylookRouter`, and added **Pin / Load / Unload** buttons + `ModelSidecarEditorSheet` (`model.heylook.toml` editor) to `ModelsInspectorSurface`.
- [x] **4.3 Live Verification Suite (`HeylookTests`)**: Expanded `HeylookTests` to 18/18 passing suites covering `--mmproj` discovery, OpenAI vision/audio message serialization, and `/v1/admin/models/*` routes.

---

## Phase 5: Python Engine-Contract Parity Additions (Done)

- [x] **5.1 Vendor `generation_config.json` Sampling Layer**: Added `HeylookConfig.loadVendorSampling(in:)` (`temperature`, `top_p`, `top_k`) and `vendor: SamplerSettings?` in `SamplerCascade.resolve` (`builtin` < `vendor` < `model.heylook.toml` < `preset` < `request`).
- [x] **5.2 Multi-Turn `reasoning_content` Forwarding**: Added `thinking: String?` on `MessagesRequest.MessageTurn` and emitted `"reasoning_content"` on prior assistant turns in `GGUFSubprocessProvider.buildOpenAIChatMessages(from:)`.
- [x] **5.3 Operator Template Override Routes (`chat_template.heylook.jinja`)**: Added `GET / PUT / DELETE /v1/admin/models/{id}/chat-template` writing/deleting `chat_template.heylook.jinja` without ever mutating vendor `chat_template.jinja`.
- [x] **5.4 Prompt Inspector & Image Token Budgeting (`/prompt` & `/image-plan`)**: Added `POST /v1/conversations/{id}/prompt`, `POST /v1/models/{id}/image-plan` (returning `409` when unloaded and `200` with `tokens_per_image` when resident), and `~tok` badges on staged attachment pills in `ChatWorkbenchSurface`.
- [x] **5.5 Verification Suite**: Expanded `HeylookTests` to 19/19 passing tests (`0.082s`).

---

## Phase 6: Full-Package Metal Compilation & Live Checkpoint Verification (Done)

- [x] **6.1 Full Native Xcode/Metal Build (`./scripts/build_native.sh`)**: Resolved and compiled remote `mlx-swift`, `mlx-swift-lm`, and `swift-transformers` with `Cmlx` Metal shaders (`default.metallib`), `-skipMacroValidation` for `MLXHuggingFaceMacros`, and `#huggingFaceTokenizerLoader()` in `MLXSwiftProvider`.
- [x] **6.2 Live `MLXSwiftProvider` Token Streaming Verification**: Wired `registerConfiguredModels(into:)` into `heylook-cli chat` (with `--model`, `--max-tokens`, and `--no-thinking` flags) and `heylook-cli serve`; verified live Metal GPU token streaming and `<think>` reasoning parsing against resident M2 Ultra MLX checkpoints.
- [x] **6.3 Live `NativeMLXEmbeddingGemma2` Hybrid ANE + MLX Media Indexing & Matryoshka Retrieval**: Verified live `heylook-cli index` and `heylook-cli search` (with `--db` flag) using `MLXSwiftLMEmbeddingGemma2Engine` + `ANECoreMLRuntime` on `embeddinggemma-2-MLX-BF16` with full `768d` / `512d` / `256d` / `128d` Matryoshka cosine attribution.

---

## Phase 7: Engine Lifecycle Reclamation, Subprocess Teardown & Incremental SSE (Done)

- [x] **7.1 Provider & Embedding `unload()` + GPU LRU Eviction Reclamation**: Added `unload() async` across `InferenceProvider` (`MLXSwiftProvider`, `CoreAIProvider`, `GGUFSubprocessProvider`, `RemoteHeylookProvider`) and `EmbeddingEngine` (`ANECoreMLRuntime`, `HybridLocalEmbeddingEngine`, `MLXSwiftLMEmbeddingGemma2Engine`), releasing `ModelContainer`, `MLModel` instances, memory-mapped `embed_tokens_fp16.bin`, resetting `PrefixCacheTracker`, and invoking `MLX.GPU.clearCache()`. Wired `ComputeScheduler.activateModel(_:)`, `unloadModel(_:)`, and `unloadAllModels()` to call `provider.unload()` on explicit unload and LRU eviction.
- [x] **7.2 `SubprocessLifecycleRegistry` (`llama-server` Leak Prevention)**: Added `SubprocessLifecycleRegistry.shared` (`SIGTERM` -> `SIGKILL` + `atexit` handler) and wired `HeylookHTTPServer.stop()`, `POST /v1/admin/models/unload-all`, `HeylookWorkbenchViewModel.unloadAllModels()` / `shutdown()`, `HeylookApp` `MenuBarExtra`, and `heylook-cli` exit paths so zero `llama-server` subprocesses or Metal allocations survive unload or shutdown.
- [x] **7.3 Strict `ProviderExecutionError` & Incremental SSE Streaming**: Added `ProviderExecutionError` (`checkpointNotFound`, `binaryNotFound`, `runtimeUnavailable`), incremental SSE frame flushing via `onSSEFrame` / `SSEConnectionStreamer` over `NWConnection`, and model-derived `HeylookConfig.estimateImageTokenBudget(in:)` (`/v1/models/{id}/image-plan`) and `HeylookConfig.renderPromptPreview(in:...)` (`/v1/conversations/{id}/prompt`).
- [x] **7.4 Verification Suite**: Expanded `HeylookTests` to 21/21 passing tests (`0.151s`).

---

## Phase 8: Stateful Multi-Turn `KVCache` Slicing, MLX Speculative Decoding & Live CoreAI/Argmax Execution (Next)

- [ ] **8.1 Stateful Multi-Turn `[KVCache]` Reuse & Prefix Truncation in `MLXSwiftProvider`**:
  - Pass persistent `cache: [KVCache]` into `MLXLMCommon` generation across multi-turn conversations, slice/trim KV state on branch edits (`history_diverged`), and pass `container`'s `tokenizer.encode` closure as `tokenCounter` into `PrefixCacheTracker.evaluate` and `recordAssistantReply` for exact token accounting.
- [ ] **8.2 In-Process MLX Speculative Decoding (`draft-mtp` & `draft-model`)**:
  - Wire paired MLX drafter checkpoints (`-mtp`, `-assistant`) and built-in `mtp.` weights discovered by `HeylookConfig.discoverDrafter` into `MLXSwiftProvider` and populate live `SpeculativeReport` (`draftedTokens`, `acceptedTokens`).
- [ ] **8.3 Live `CoreAILanguageModels` & `argmax-oss-swift` (`WhisperKit` / `TTSKit`) ANE Bridge**:
  - Wire live `.aimodel` `LanguageModelSession` token streaming in `CoreAIProvider` (with `allowSyntheticFallback = false` in production) and expose ANE speech transcription (`WhisperKit`) / synthesis (`TTSKit`) without contending for the Metal GPU `GenerationGate`.

---

## Phase 9: `llama-server` (`GGUF`) Header Metadata, `/props` Telemetry & Multi-Port Pool

- [ ] **9.1 Pure-Swift GGUF String KV Metadata & Embedded Template Reader**:
  - Extend `GGUFHeaderReader` to parse string/uint metadata keys (`general.architecture`, `tokenizer.chat_template`, `*.context_length`) directly from `.gguf` binary headers so `gguf_embedded` templates and context sizes are derived offline without launching `llama-server`.
- [ ] **9.2 Live `GET /props` Provenance & Slot KV Telemetry**:
  - Query `GET /props` once `llama-server` passes `/health` to enrich `EngineDescription` with `.observed` context size and active chat template, and surface `llama-server` slot cache reuse (`cache_n`) in `EngineDescription` / `CacheReport`.
- [ ] **9.3 Dynamic Loopback Port Pool for Co-Resident GGUF Models**:
  - Allocate conflict-free ephemeral ports (`18080+`) per `GGUFSubprocessProvider` instance so multiple `.gguf` models can co-reside up to `maxLoadedGPUModels` while `SubprocessLifecycleRegistry` reaps evicted instances.

---

## Phase 10: Cross-Repo `mbed` Sync, `eval-lab` CLI & iOS 27.2 Voice Handoff

- [ ] **10.1 Backport `unload()` & Path-Privacy Defaults to Standalone `mbed`**:
  - Sync `EmbeddingEngine.unload()`, `ANECoreMLRuntime.unload()`, and `~/.heylook` checkpoint path defaults into standalone `mbed` (`MbedCore`) before archiving `mbed` in favor of `heylookitsanllm-macos`.
- [ ] **10.2 `heylook-cli eval-lab` Verification Harness**:
  - Implement `heylook-cli eval-lab` running automated multi-turn KV cache chain probes (`chain_probe`), VLM vision parity checks (`vlm_parity_probe`), and `EmbeddingLabRunner` Matryoshka `Recall@K` / `MRR` benchmarks.
- [ ] **10.3 iOS 27.2 Hands-Free ANE Voice + Mac Studio Remote Handoff**:
  - Combine on-device ANE speech recognition with disconnect-resilient `POST /v1/conversations/{id}/generate` on the Mac Studio for lock-screen-safe mobile voice conversations.
