# Backlog & Phased Roadmap (`heylookitsanllm-macos`)

Tracks the phased port of `heylookitsanllm` + `mbed` into a unified Swift 6.2 multiplatform application and server targeting `macOS 27.2+` and `iOS 27.2+`.

---

## Phase 0: Architecture, Vision & Multi-Target Swift 6.2 Scaffold

- [x] **0.1 Governance & Documentation**: `VISION.md`, `AGENTS.md`, `README.md`, `docs/wiki/architecture_overview.md`, `docs/project/CURRENT.md`, `docs/project/TODO.md`.
- [x] **0.2 Remote-First `Package.swift` (`macOS 27.2+`, `iOS 27.2+`)**: Remote GitHub URLs (`ml-explore/mlx-swift`, `ml-explore/mlx-swift-lm`, `apple/coreai-models`, `argmaxinc/argmax-oss-swift`, `hummingbird-project/hummingbird`, `huggingface/swift-transformers`) with fast default `swift test` and full native `scripts/build_native.sh`.
- [x] **0.3 `HeylookCore`**: `EngineDescription`, `SamplerCascade`, `ReasoningParser` + `StripSpecials`, `HeylookConfig` (`heylook.toml` & `model.heylook.toml`), `SQLiteStore` (`WAL`), `RequestRegistry`, and `MessagesWire`.
- [x] **0.4 `HeylookEmbeddings` (folded from `mbed` at full fidelity)**: `Artifact`, `VectorStore` (`vDSP` Matryoshka), `WindowPlanner`, `ArtifactProcessor`, `FolderWatcher` (`FSEvents`), `IndexCoordinator` (`indexURLs`, `discoverFiles`, `isUpToDate`, `pruneMissingArtifacts`, `buildChunks`), `HeylookMLXEngine`, `NativeMLXEmbeddingGemma2` (`MLXSwiftLMEmbeddingGemma2Engine` + `HybridLocalEmbeddingEngine`), `EmbeddingLabRunner`, `ModelComputeOptions`, `ANECoreMLRuntime` (`MLMultiArray` Vision/Audio/Text ANE prediction + 512-point `vDSP` FFT & 128-bin HTK mel filterbank), and `ANETowerInspector`.
- [x] **0.5 `HeylookEngines`**: Dual-lane `ComputeScheduler` (GPU FIFO gate + ANE co-resident lane), `HardwareMemoryMonitor`, `GGUFHeaderReader`, `MLXSwiftProvider`, `CoreAIProvider`, `GGUFSubprocessProvider`, and `RemoteHeylookProvider`.
- [x] **0.6 `HeylookServer`**: Embedded Hummingbird 2 `/v1/*` server with disconnect-resilient `/v1/conversations/{id}/generate` and `heylook_saved`.
- [x] **0.7 `HeylookUI`, `HeylookApp` & `heylook-cli`**: Multiplatform 5-surface SwiftUI workbench (`Chat`, `Notebook`, full 3-column `mbed` `Embeddings & Media Studio` with inline `AVPlayerView`, clickable `frameTimes` jump pills, and `ScoredFeatureBreakdownCard`, `Models & Inspector`, `Perf & Server`), MenuBar controller, CLI (`index`, `search`, `watched`, `inspect-ane`, `inspect-checkpoint`), and 10/10 `HeylookTests` suite.

---

## Phase 1: Deep Engine Specialization (Next)

- [ ] **1.1 `MLXSwiftProvider` Prefix Cache (APC) & Vision Feature Cache**: Wire `MLXLMCommon` `KVCache` / prompt cache reuse metrics (`prompt_tokens`, `cached_tokens`, `processed_tokens`, `cause`) and mid-stream `reasoning-end` token injection.
- [ ] **1.2 `CoreAIProvider` Stateful `.aimodel` Session**: Wire `CoreAILanguageModels` `InferenceEngine` (`keyCache`, `valueCache`, `convStates`, `recurrentStates` for Qwen 3.5/3.8 and Gemma 4 PLE + MTP drafter) and `CoreAISpeech` / `WhisperKit` / `TTSKit` endpoints.
- [ ] **1.3 `GGUFSubprocessProvider` Live Supervisor**: Wire `llama-server` process spawn, health readiness polling, `/props` `chat_template_caps` ingestion, and `/v1/chat/completions/control` `reasoning_end`.

---

## Phase 2: Mobile Sync & Live Parity

- [ ] **2.1 Bonjour (`_heylook._tcp`) Auto-Discovery on iOS 27.2**: Zero-config discovery of the Mac Studio server from the iPhone/iPad app over LAN or Tailscale.
- [ ] **2.2 Contract & Smoke Suite Parity**: Run the full `/v1/messages` and `/v1/conversations/{id}/generate` SSE contract suite against `HeylookServer`.
