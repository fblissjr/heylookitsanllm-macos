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
