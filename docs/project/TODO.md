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
- [x] **8.3 `CoreAIProvider` ANE Audio Log-Mel Front-End & Truthful Speech Gate (`transcribeAudio`)**:
  - Added `CoreAIProvider.ANESpeechTranscriptionResult` and `transcribeAudio(wavData:languageHint:)` validating 16 kHz mono PCM + 128-bin HTK log-mel frames via `ANECoreMLRuntime.computeAudioLogMel` on the ANE lane, and throwing `ProviderExecutionError.runtimeUnavailable` until a real ASR tower (`CoreAISpeech` / `WhisperKit`) is linked (never returning fabricated placeholder transcripts).

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
  - Wired `thinking_depth` and `reasoning_effort` + full `SamplerSettings` (`repetition_penalty`, `repetition_context_size`, `presence_penalty`, `seed`) on `MessagesRequest`, `POST /v1/conversations/{id}/prompt`, and `RemoteHeylookProvider` into `SamplerCascade`, connected `ModelTemplateInfo` (`extraEOSTokens`, `eosTokenIds`, `reasoningConfig: nil` to prevent dual-parser conflicts, automatic VLM-to-LLM factory fallback, and `MLXLMCommon.generate(input:cache:parameters:context:)` with `MLXKVCacheBox` reuse/trim) into `MLXSwiftProvider`, wired `chat_template_kwargs` + `ReasoningParser` + `/props` `.observed` upgrades into `GGUFSubprocessProvider`, and added dynamic **Think** toggle disabling, **Depth** segmented picker (`low`/`medium`/`high`/`max`), and the **Compare-to-Lab Bridge** to `HeylookWorkbenchView` and `heylook-cli`.
- [x] **11.5 Model Family Onboarding, Conversion Pipelines & Porting Playbook (`docs/wiki/model_family_onboarding.md`)**:
  - Added `MLXModelConverter` (`heylook-cli convert-mlx` + `MLXModelConverterSheet` in `ModelsInspectorSurface` wrapping `LLMModelFactory.shared.convert` / `MLXLMCommon.convert` for `affine`, `mxfp4`, `mxfp8`, `nvfp4`, and `q4_zero` calibration), wired `EmbeddingRequestLimits` (`v2.0.239` canonical bounds + structured `content`/`prompt_name`/`windows` `/v1/embeddings` wire), synced full 3-tower CoreML ANE exporter in `scripts/export_ane_coreml.py`, added `scripts/export_coreai.sh` wrapping `coreai-models`, and documented the 3-tier pure-Swift model support matrix, Tier 1 / Tier 2 porting playbooks, hybrid ANE+GPU+CPU placement rules, and MLX Swift vs. Core AI trade-offs.
- [x] **11.6 Verification Suite**:
  - Expanded `HeylookTests` to **27/27** passing unit, lifecycle, and wire-contract suites across both offline SwiftPM (`0.158s`) and full native `xcodebuild test` (`0.145s`).

---

## Phase 12: Architectural Simplification, Truthful Runtime & LLM Knowledge Wiki (Done)

- [x] **12.1 Truthful Runtime (Zero Silent Synthetic Fallbacks)**:
  - Defaulted `allowSyntheticFallback = false` across `MLXSwiftProvider`, `CoreAIProvider`, `GGUFSubprocessProvider`, and `ProviderFactory.makeProvider`; removed silent fake model registration and fake corpus seeding from `HeylookWorkbenchViewModel.bootstrap()`, `heylook-cli search`, and `heylook-cli chat` (exposing explicit `loadSampleDemoWorkspace()` in UI); and set `DeterministicHashEmbeddingEngine` / `HybridLocalEmbeddingEngine` fallback `modelID` to `"deterministic-hash-768d"` so hash vectors never poison `VectorStore.isUpToDate` for `"embeddinggemma-2-MLX-BF16"`, while wiring `HybridLocalEmbeddingEngine` to execute compiled Vision/Audio/Text `.mlmodelc` ANE towers via `ANECoreMLRuntime` when present.
- [x] **12.2 Codebase Simplification & Modular File Splits**:
  - Split monolithic `Sources/HeylookEngines/Providers/Providers.swift` (2,923 lines) into `StreamingPipeline.swift` (unifying the 4 token-streaming loops into a single `StreamingPipeline.transformEventStream` state machine + `ProviderFactory`), `MLXSwiftProvider.swift`, `CoreAIProvider.swift`, `GGUFSubprocessProvider.swift`, `RemoteHeylookProvider.swift`, and `SubprocessSupport.swift`.
  - Split monolithic `Sources/HeylookUI/HeylookWorkbenchView.swift` (4,155 lines) into `HeylookWorkbenchViewModel.swift`, `HeylookWorkbenchView.swift`, and `Surfaces/` (`ChatWorkbenchSurface.swift`, `NotebookAndEmbeddingsSurfaces.swift`, `ModelsAndSystemSurfaces.swift`), and unified SSE framing in `HeylookRouter.consumeStreamAndFormatSSE`.
- [x] **12.3 Engine & Wire Contract Hardening**:
  - Preserved `reasoningFormat` and `thinkingControls` in `ComputeScheduler.applySidecarUpdate`; wired `HeylookWorkbenchViewModel.streamAssistantReply` to honor `currentResolvedSamplers` (`SamplerCascade`) and store clean `MessageRecord.content` without `[IMAGE: ...]` prefix pollution; added HTTP `400` validation for unsupported `thinking_depth` / `reasoning_effort` on `POST /v1/messages`; added `clip.projector_type` non-causal vision detection in `GGUFHeaderReader` + `--image-max-tokens <= ubatchSize` clamping in `GGUFSpawnConfig.autoSized`; and surfaced mid-stream `llama-server` `data: {"error": ...}` SSE frames.
- [x] **12.4 LLM Knowledge Wiki & `AGENTS.md` Index Router**:
  - Created the 8-page `docs/wiki/` knowledge base (`README.md`, `architecture_overview.md`, `engine_contract_and_provenance.md`, `providers_and_heterogeneous_compute.md`, `multimodal_embeddings_and_ane.md`, `wire_api_and_sqlite_store.md`, `model_family_onboarding.md`, `sharp_edges.md`) and refactored `AGENTS.md` as an Index Router.
- [x] **12.5 Verification Suite**:
  - Expanded `HeylookTests` to **28/28** passing unit, lifecycle, truthful-runtime, and wire-contract suites (`0.165s`).

---

## Phase 13: SwiftUI 27.3 ("Golden Gate") & iOS 27.3 Specialist Audit, Subview Factoring & Observation Caching (Done)

- [x] **13.1 Granular `@Observable` View-Model Projections (`HeylookWorkbenchViewModel.swift`)**:
  - Cached `activeModelDescription: EngineDescription?` and `sortedActiveGGUFPorts: [(modelID: String, port: Int)]` in `didSet` observers on `registeredModels`, `selectedModelID`, and `activeGGUFPorts` so SwiftUI views do not subscribe transitively to the entire `registeredModels` array or sort dictionaries on every `body` evaluation.
  - Added `DestructiveActionPrompt` (`clearEmbeddingIndex`, `unloadAllModels`) and `pendingConfirmation: DestructiveActionPrompt?` for item-driven `.confirmationDialog("Confirm Action", item: $viewModel.pendingConfirmation)` prompts (`VISION.md` Principle 8).
- [x] **13.2 Focused `@Observable` Subview Factoring Across All 5 Surfaces**:
  - Extracted `StatusBannerBar`, `EmptyModelsDiscoveryBanner`, and `ToolbarContainerBehavior27_3Modifier` (`if #available(macOS 27.3, iOS 27.3, *)`) in `HeylookWorkbenchView.swift`, plus `#if os(iOS)` `ToolbarOverflowMenu`.
  - Extracted `ChatTopControlBar`, `ChatPromptPreviewDrawer`, `ChatComposerBar`, and `ChatAttachmentPillView` in `ChatWorkbenchSurface.swift`, added multiplatform `.fileImporter` on `#if !os(macOS)`, and enforced `28x28` minimum hit targets on compact icon controls.
  - Extracted `EmbeddingsSearchHeaderBar`, `EmbeddingsStatusToolbar`, `EmbeddingsEvalLabDrawer`, and `ModalityFilterRow` in `NotebookAndEmbeddingsSurfaces.swift`; replaced `ForEach(Array(frames.enumerated()), id: \.offset)` with `KeyframeStamp: Identifiable`; replaced synchronous main-thread `NSImage(contentsOf:)` with cross-platform `AsyncImage(url:)`; and converted `ScoredFeatureBreakdownCard` `.overlay(...)` to trailing-closure `.overlay { ... }`.
  - Extracted `ModelsTopActionBar` in `ModelsAndSystemSurfaces.swift` and bound `PerfServerSurface` to `viewModel.sortedActiveGGUFPorts`.
- [x] **13.3 Multi-Target `swift-lsp` Diagnostics & Unit Test Coverage**:
  - Enhanced `sourcekit_mcp.py` (`ensure_local_swiftpm_dependency_modules` + `@main` `-parse-as-library` handling) so `swift_diagnostics(platform: "both", min_version: "27.3")` verifies `HeylookUI`, `HeylookApp`, and `HeylookTests` with `0` errors across `arm64-apple-macosx27.3` and `arm64-apple-ios27.3-simulator`.
  - Added `@MainActor @Test` `testWorkbenchViewModelCachedObservableProjectionsAndDestructivePrompt` to `HeylookTests` (**29/29** tests passing).

---

## Phase 14: Python v2.0.244 Engine, Jinja2 AST, GGUF RAM-Fit, SQLite Media-Blob & `/v1/*` Wire Parity (Done)

- [x] **14.1 Real Jinja2 AST Engine, Template Inspection & Piece Tokenizer (`HeylookCore`)**:
  - Implemented `HeylookJinjaRenderer` (recursive-descent Jinja2 lexer, AST parser, `validateSyntax`, `detectCapabilities`, `checkPrefixStability`, special-token `PromptMarkerSpan` scanner, and `huggingface/swift-jinja` bridge when linked), `ChatTemplateInspection` with Hugging Face `.cache/huggingface/download/*.metadata` git-blob SHA-1 / LFS SHA-256 `checkModifiedSinceDownload`, `CheckpointPieceTokenizer` (BPE merge-rank + Viterbi Unigram `tokenizer.json` piece segmentation), non-destructive `HeylookConfig.addScanFolder` / `removeScanFolder`, and `HeylookObservabilityLog` (`level` filtering, `retentionLimit`, JSONL persistence).
- [x] **14.2 Deep `MLXSwiftProvider`, `GGUFSubprocessProvider` & `StreamingPipeline` Parity (`HeylookEngines`)**:
  - Added `StopSequenceFilter` (multi-chunk holdback), `thinkingBudget` auto-`forceEndReasoning()`, and immediate `RequestRegistry.addCancelHandler` cancellation in `StreamingPipeline`.
  - Upgraded `MLXSwiftProvider` with multi-branch `[KVCache]` snapshot cloning/trimming (checking `isTrimmable` on hybrid `MambaCache`/`ArraysCache`), conditional `<think>` KV prefix tracking (`sawReasoning`), real speculative decoding (`draft-mtp` via `MTPDrafterModelFactory` -> `MLXLMCommon.generate(..., mtpDrafter:)` and `draft-model` via `LLMModelFactory` -> `MLXLMCommon.generate(..., draftModel:)` + `speculativeDecodingTelemetry`), and VLM `image_max_tokens` downscaling.
  - Upgraded `GGUFSubprocessProvider` with OS-socket-probed `GGUFPortPool`, `GGUFSpawnConfig.assessRAMFit` (dropping external drafter under tight Metal working-set headroom + retry without drafter on launch failure), media-blind sidecar fallback to embedded GGUF template when `--mmproj` is present, and `--reasoning-preserve` / `--ctx-checkpoints` / `--cache-reuse` / `json_schema` / `stop` support.
- [x] **14.3 Full `/v1/*` Wire Contract, Security Middleware & `SQLiteStore` Media-Blob / Fork Parity (`HeylookCore` & `HeylookServer`)**:
  - Upgraded `SQLiteStore` with SHA-256 deduplicated `media_blobs` + orphan GC, multimodal `content_blocks_json` persistence & hydration, `message_stats` + `queryPerformanceProfile` (`1h`/`6h`/`24h`/`7d`/`30d` + `byModel`), `cloneConversation(id:upToPosition:)`, single-turn `deleteMessage` position compaction, `COLLATE NOCASE` unique presets seeded once via `store_meta`, `clearAllData()`, and `mutationStream()`.
  - Completed `HeylookRouter` route & security parity: strict `Host` DNS-rebinding rejection on loopback and `0.0.0.0`, `X-Heylook-Admin-Token` `401`/`403` auth, `HEYLOOK_READONLY_MODEL_CONFIG` `409` guard, `X-Request-ID` regex `422` validation + SSE echo, shared `RequestRegistry.shared` per-conversation `409 generation_in_progress` locks, `append`/`regenerate`/`continue` `/v1/conversations/{id}/generate`, `GET`/`POST /v1/conversations/{id}/prompt`, `/v1/conversations/{id}/clone`, `/v1/notebooks` CRUD + SSE `/generate`, `/v1/presets` CRUD, `/v1/models/reload`, `/v1/admin/models/validate`, `/v1/admin/models/{id}/fit`, `/v1/admin/models/{id}/cache/clear`, `/v1/admin/config` + `/logs`, `/v1/performance/profile/{range}`, `/v1/telemetry/events`, and `/v1/data/clear`.
- [x] **14.4 Audio Resampling, SDK 27 `AVAssetReader` Migration & Expanded Test Suite (`HeylookEmbeddings` & `HeylookTests`)**:
  - Added arbitrary WAV sample-rate / channel decoding via `AVAudioFile` + `AVAudioConverter` -> 16 kHz mono `Float32` and migrated `ArtifactProcessor` to SDK 27 async `AVAssetReader.outputProvider(for:)` with zero compiler warnings.
- [x] **14.5 6-Surface SwiftUI Workbench, N×N Compare Matrix, Multi-Config Retrieval Lab & 40-Test Verification Suite (`HeylookUI`, `HeylookApp`, `HeylookTests`)**:
  - Expanded the workbench to 6 surfaces (`Chat`, `Notebook`, `Presets`, `Embeddings & ANE`, `Models`, `Perf & Server`, `⌘1`–`⌘6`), added `PresetsSurface`, `MarkdownMessageView`, `CursorAwareTextEditor`, N×N multimodal Compare Matrix, Multi-Config Retrieval Lab, live `SQLiteStore.mutationStream()` + shared `RequestRegistry.shared` conversation locks, and expanded `HeylookTests` to `40/40` passing tests across both offline pure-Swift and `HEYLOOK_ENABLE_NATIVE_MLX=1` builds.

---

## Phase 15 (H.1): On-Device ASR & Voice Turn Live Wiring (`System SpeechTranscriber` + `CoreAISpeech`) (Done)

- [x] **15.1 Dual-Tier On-Device ANE Speech Recognition (`CoreAIProvider` & `Package.swift`)**:
  - Linked `Speech` and `AVFoundation` on `HeylookEngines` plus optional `CoreAISpeech` (`apple/coreai-models` under `HEYLOOK_ENABLE_COREAI=1`), wiring **Tier A** out-of-the-box Apple `SpeechAnalyzer` + `SpeechTranscriber` (`apple-speech-transcriber`) and **Tier B** `CoreAISpeech.SpeechRecognitionModel` (`Whisper` & `Parakeet TDT` `.aimodel` bundles) inside `CoreAIProvider.transcribeAudio(wavData:languageHint:)`.
  - Replaced naive byte-44 WAV slicing with `CoreAIProvider.decodeRIFFChunksMonoPCM` + `ArtifactProcessor.decodeMonoPCM` so `AVAudioRecorder` 4 KB `FLLR` alignment padding and arbitrary sample rates/channels decode accurately to 16 kHz mono `Float32`.
- [x] **15.2 Speech Bundle Discovery, Persistent ANE Residency & `/v1/audio/transcriptions` (`HeylookCore`, `HeylookEngines`, `HeylookServer`)**:
  - Added `EngineDescription.isSpeechRecognizer`, `HeylookConfig.resolveCoreAIBundleRoot(for:scanRoot:)`, and `inspectSpeechRecognizerArchitecture(in:)` so top-level `<model>.aimodel` directories never shift to their parent `scanRoot` and auto-derive `parakeet_tdt`/`whisper` capabilities.
  - Added persistent ANE-lane ASR residency in `ComputeScheduler.transcribeVoice` + `voiceASRSummary()`, added standalone `POST /v1/audio/transcriptions`, upgraded `POST /v1/conversations/{id}/voice-generate` with `asr_model` override and `422 empty_transcription` silence protection, and enriched `GET /v1/capabilities` with `voice_asr`.
- [x] **15.3 Workbench Composer Dictation, `heylook-cli transcribe` & Export Scripts (`HeylookUI`, `HeylookCLI`, `scripts/`)**:
  - Enabled out-of-the-box composer Mic dictation (`transcribeVoiceToComposer` Dictate-to-Composer default + `⌥`-click / context-menu Hands-Free Auto-Send + microphone permission prompt + ASR model selector), filtered pure ASR bundles out of `generationModels`, added `heylook-cli transcribe <audio-path>` + `doctor` Voice ASR diagnostics, and added `whisper` & `parakeet` subcommands to `scripts/export_coreai.sh` + `NSSpeechRecognitionUsageDescription` in `scripts/build_native.sh`.
- [x] **15.4 Verification Suite (`HeylookTests`)**:
  - Expanded `HeylookTests` to `44/44` passing unit, lifecycle, RIFF/`FLLR` audio, ANE ASR residency, and `/v1/*` wire tests.

---

## Next Horizons (Optional Post-Phase 15 Frontiers)

- [x] **H.1 On-Device ASR Tower Integration (`System SpeechTranscriber` + `CoreAISpeech`)**: Completed in Phase 15 above.
- [ ] **H.2 Live Hardware Conversion & Throughput Certification**:
  - Run live `heylook-cli convert-mlx` (`4-bit affine`, `mxfp4`, `q4_zero`) and `scripts/export_coreai.sh` against resident unquantized BF16 checkpoints and compare prefill/decode behavior and memory residency across Tier 1 (`MLXSwiftProvider`) and Tier 2 (`CoreAIProvider`).
- [ ] **H.3 Physical `iOS 27.2+` Device Deployment & Hybrid Failover**:
  - Package and deploy `HeylookApp` to physical iPhone/iPad (`iOS 27.2+`) with on-device 4-bit ANE-palettized `.aimodel` bundles (`./scripts/export_coreai.sh --platform ios`) alongside Bonjour `_heylook._tcp` remote failover.
- [ ] **H.4 Tier 2 Core AI Architecture Expansion**:
  - Extend `coreai-models` export rules (ANE prefill + GPU decode) as upstream adds new multimodal and reasoning architectures.
