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

## Phase 8: Stateful Multi-Turn `KVCache` Slicing, MLX Speculative Decoding & Live CoreAI/Argmax Execution (Done)

- [x] **8.1 Stateful Multi-Turn `[KVCache]` Reuse & Prefix Truncation in `MLXSwiftProvider`**:
  - Added `MLXSwiftProvider.KVCacheSessionState` (`activeTurnCount`, `cachedTokenCount`, `sliceTruncations`, `speculativeDrafterPath`, `speculativeMode`), `PrefixCacheTracker.branchTruncationsCount`, and custom `tokenCounter` support on `MLXSwiftProvider` so multi-turn prefix reuse, branch-edit KV truncation slicing, and unload resets are tracked deterministically.
- [x] **8.2 In-Process MLX Speculative Decoding (`draft-mtp` & `draft-model`)**:
  - Wired `HeylookConfig.discoverDrafter` into `MLXSwiftProvider.init` and emitted live `SpeculativeReport` (`draft-mtp` / `draft-model`) on `GenerationChunk`s across both in-process `MLX` and streaming pipelines.
- [x] **8.3 `CoreAIProvider` ANE Speech Transcription Bridge (`transcribeAudio`)**:
  - Added `CoreAIProvider.ANESpeechTranscriptionResult` and `transcribeAudio(wavData:languageHint:)` decoding 16 kHz mono PCM, computing RMS energy and 128-bin HTK log-mel frames via `ANECoreMLRuntime.computeAudioLogMel`, and running on the ANE lane without acquiring the Metal GPU `GenerationGate`.

---

## Phase 9: `llama-server` (`GGUF`) Header Metadata, `/props` Telemetry & Multi-Port Pool (Done)

- [x] **9.1 Pure-Swift GGUF String/UInt KV Metadata & Embedded Template Reader**:
  - Extended `GGUFHeaderReader` (`HeylookCore`) with full GGUF v2/v3 KV cursor parsing (`stringMetadata`, `uintMetadata`, `architecture`, `embeddedChatTemplate`, `contextLength`, `readFileHeader(at:)`, and `buildMinimalGGUF`), and wired `HeylookConfig.discoverModels` and `HeylookConfig.renderPromptPreview` to read `general.architecture`, `*.context_length`, and `tokenizer.chat_template` directly from `.gguf` headers offline.
- [x] **9.2 Live `GET /props` Provenance & Slot KV Telemetry**:
  - Added `GGUFSubprocessProvider.GGUFServerProps`, `parsePropsJSON(_:)`, and `applyObservedProps(_:)`, and wired `launchSubprocess` to query `GET /props` after `/health` passes so `contextSize` and `templateSource` upgrade to `.observed` provenance (`"llama-server:/props"`).
- [x] **9.3 Dynamic Loopback Port Pool for Co-Resident GGUF Models (`GGUFPortPool`)**:
  - Added `GGUFPortPool.shared` (`18080+`) assigning conflict-free loopback ports per `modelID` in `GGUFSpawnConfig.autoSized` and releasing ports in `GGUFSubprocessProvider.unload()` and `SubprocessLifecycleRegistry.terminate`.

---

## Phase 10: Cross-Repo `mbed` Sync, `eval-lab` CLI & iOS 27.2 Voice Handoff (Done)

- [x] **10.1 Backport `unload()` & Path-Privacy Defaults to Standalone `mbed`**:
  - Backported `EmbeddingEngine.unload()`, `ANECoreMLRuntime.unload()`, `MLXSwiftLMEmbeddingGemma2Engine.unload()`, and `~/.heylook/models/embeddinggemma-2-MLX-BF16` path-privacy defaults into standalone `mbed` (`MbedCore` & `scripts/export_ane_coreml.py`), verified with 16/16 passing `MbedCoreTests` and zero path leaks.
- [x] **10.2 `heylook-cli eval-lab` Verification Harness (`EvalLabSuite`)**:
  - Implemented `EvalLabSuite.runAll()` and `heylook-cli eval-lab` running automated multi-turn KV prefix cache chain probes (`chain_probe`), VLM vision token budget & HTTP 422 capability gate parity checks (`vlm_parity_probe`), and `EmbeddingLabRunner` Matryoshka (`768d`/`512d`/`256d`/`128d`) `Recall@1` / `Recall@3` / `MRR` benchmarks (`matryoshka_recall_probe`).
- [x] **10.3 iOS 27.2 Hands-Free ANE Voice + Disconnect-Resilient Generation Handoff**:
  - Added `POST /v1/conversations/{id}/voice-generate` (`ConversationVoiceGenerateRequest` + `event: heylook_voice_transcribed` + `event: heylook_saved`) in `HeylookRouter` and `HeylookWorkbenchViewModel.sendVoiceTurn(wavData:)` combining ANE speech transcription with disconnect-resilient server-owned generation.
- [x] **10.4 Verification Suite**:
  - Expanded `HeylookTests` to **24/24** passing unit, lifecycle, and wire-contract suites (`0.144s`).

---

## Phase 11: Zero-Hardcoding Pure-Swift Model Family Onboarding (Done)

- [x] **11.1 File-Derived `ModelTemplateInfo` & `ThinkingControls` (`HeylookCore`)**:
  - Added `ModelTemplateInfo`, `ReasoningFormat` (`.tagPair`, `.gemmaChannel`, `.harmonyChannel`, `.recipientChannel`, `.none`), and `ThinkingControls` (`supportsThinking`, `canDisableThinking`, `defaultEnabled`, `switchVariable`, `depthVariable`, `depthValues`), inspecting `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and `hasVisionConfiguration(in:configJSON:)` directly from checkpoint files without static `model_type` string sets or hardcoded model-family branches.
- [x] **11.2 4-Protocol `ReasoningParser` & Dynamic `StripSpecials`**:
  - Upgraded `ReasoningParser` to stream `.tagPair` (`<think>...</think>`), `.gemmaChannel` (`<|channel>thought\n...<channel|>`), `.harmonyChannel` (`<|channel|>analysis<|message|>...`), and `.recipientChannel` (`to=self<|message|>...<|eom|>`) across arbitrary token chunk boundaries, and seeded `StripSpecials` from checkpoint-declared special tokens (`<turn|>`, `<|eot|>`, `<|eom|>`).
- [x] **11.3 `CoreAIBundleManifest` Schema 0.2, `CoreAILanguageModels` Execution & `GGUFHeaderReader` Extended KV Metadata**:
  - Added `coreai-models` Schema 0.2 nested `metadata.json` support (`language.*`, `vision.backbone`, `speculative.drafter_kind`, `assets.*`) in `CoreAIBundleManifest`, wired `CoreAIProvider` to execute `.aimodel` bundles via `CoreAILanguageModels` (`LanguageModelBundle` / `CoreAIRunner`) and report `drafterKind` (`dflash` / `mtp`) and `ModelTemplateInfo` provenance, and extended `GGUFHeaderReader` to parse `BOOL` (`clip.has_vision_encoder`, `clip.has_audio_encoder`), `FLOAT32`/`FLOAT64` (`general.sampling.*`), hybrid-recurrent SSM (`*.ssm.*` -> `--cache-ram 8192`), and built-in MTP (`*.nextn_predict_layers` -> `--spec-type draft-mtp`).
- [x] **11.4 End-to-End Wire, Provider (`MLXSwiftProvider` KV Cache & `GGUFSubprocessProvider`) & SwiftUI `ThinkingControls` Integration**:
  - Wired `thinking_depth` and `reasoning_effort` + full `SamplerSettings` (`repetition_penalty`, `repetition_context_size`, `presence_penalty`, `seed`) on `MessagesRequest` and `RemoteHeylookProvider` into `SamplerCascade`, connected `ModelTemplateInfo` (`extraEOSTokens`, `eosTokenIds`, `reasoningConfig: nil` to prevent dual-parser conflicts, automatic VLM-to-LLM factory fallback, and `MLXLMCommon.generate(input:cache:parameters:context:)` with `cachedKVCaches` reuse/trim) into `MLXSwiftProvider`, wired `chat_template_kwargs` + `ReasoningParser` + `/props` `.observed` upgrades into `GGUFSubprocessProvider`, and added dynamic **Think** toggle disabling and **Depth** segmented picker (`low`/`medium`/`high`/`max`) to `HeylookWorkbenchView` and `heylook-cli models`.
- [x] **11.5 Model Family Onboarding & Porting Playbook (`docs/wiki/model_family_onboarding.md`)**:
  - Documented the 3-tier pure-Swift model support matrix and step-by-step playbooks for porting new architectures to Tier 1 (`mlx-swift-lm`) and Tier 2 (`coreai-models` / ANE).
- [x] **11.6 Verification Suite**:
  - Expanded `HeylookTests` to **26/26** passing unit, lifecycle, and wire-contract suites (`0.155s`).
