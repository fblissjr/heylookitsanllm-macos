# Providers, Unified Streaming Pipeline & Heterogeneous Compute

Covers `Sources/HeylookEngines/` (`ComputeScheduler.swift` and `Providers/`).

---

## 1. Dual-Lane Heterogeneous `ComputeScheduler` (`ComputeScheduler.swift`)

Apple Silicon unifies CPU, Metal GPU, and Apple Neural Engine (ANE) over one physical memory pool (up to 192 GB on M2 Ultra), but Metal GPU command queues and ANE `MLModel` execution have fundamentally different concurrency profiles. `ComputeScheduler` separates them into two lanes:

1. **Metal GPU Lane (`GenerationGate`)**:
   - Serializes heavy autoregressive LLM/VLM decoding (`MLXSwiftProvider`, GPU-placed `CoreAIProvider`, and `GGUFSubprocessProvider`) via an async FIFO gate bounded by `maxQueueDepth` (default `2`).
   - When the queue is full, `withComputeLease(for:)` immediately throws `ComputeSchedulerError.queueOverloaded`, which `HeylookRouter` translates into HTTP `503 model_overloaded` with `Retry-After: 2`.
   - Enforces LRU eviction across unpinned GPU models (`maxLoadedGPUModels`, default `2`), calling `await provider.unload()` on evicted models to reclaim Metal buffers and terminate `llama-server` subprocesses.
2. **Apple Neural Engine Co-Resident Lane (`ANELane`)**:
   - Models and towers whose `computePlacement.runsOnNeuralEngineLane` is `true` (`.ane` or `.hybridANEGPU`, such as `EmbeddingGemma 2` ANE towers and `CoreAIProvider` speech transcription) bypass `GenerationGate` and execute concurrently on the ANE without evicting resident GPU LLMs.

---

## 2. `ProviderFactory` & Unified `StreamingPipeline` (`StreamingPipeline.swift`)

All provider instantiation goes through `ProviderFactory.makeProvider(for:allowSyntheticFallback:remoteBaseURL:)` (or `scheduler.registerDiscoveredModels(_:allowSyntheticFallback:)`), which guarantees:
- **Truthful Runtime by Default**: `allowSyntheticFallback` defaults to **`false`** in production across `MLXSwiftProvider`, `CoreAIProvider`, and `GGUFSubprocessProvider`. Missing checkpoints or binaries fail loudly with `ProviderExecutionError` (`checkpointNotFound`, `binaryNotFound`, `runtimeUnavailable`). Only explicit offline unit-test fixtures or the user-triggered **"Load Sample Demo Workspace"** action pass `allowSyntheticFallback: true`.
- **Single Token-Streaming State Machine (`StreamingPipeline.transformEventStream`)**:
  Instead of duplicating `ReasoningParser`, `StripSpecials`, `StopSequenceFilter` (holding back partial stop-sequence suffixes across token boundaries), `thinkingBudget` enforcement (triggering `forceEndReasoning()` once thinking token count reaches budget), immediate cancellation (`RequestRegistry.addCancelHandler`), TPS calculation, and final summary chunk emission across providers, every provider emits `ProviderStreamEvent`s (`.rawToken(String)`, `.splitReasoning(String)`, `.splitContent(String)`, `.telemetry(...)`) into `StreamingPipeline.transformEventStream`.

---

## 3. Provider Implementations (`Sources/HeylookEngines/Providers/`)

| File | Actor / Type | Key Responsibilities |
| :--- | :--- | :--- |
| `MLXSwiftProvider.swift` | `MLXSwiftProvider` & `MLXModelConverter` | Loads `.safetensors` checkpoints via `VLMModelFactory.shared` (with automatic fallback to `LLMModelFactory.shared` if a checkpoint ships a `vision_config` stub for an LLM-only `model_type`), disables `MLXLMCommon`'s built-in `reasoningConfig` so `StreamingPipeline` owns reasoning extraction cleanly, maintains multi-branch `[KVCache]` snapshots (`KVCacheSessionState` + `PrefixCacheTracker`) with `isTrimmable` guards for hybrid `MambaCache`/`ArraysCache` models and `sawReasoning` `<think>` token alignment, executes real speculative decoding (`draft-mtp` via `MTPDrafterModelFactory` -> `MLXLMCommon.generate(..., mtpDrafter:)` and `draft-model` via `LLMModelFactory` -> `MLXLMCommon.generate(..., draftModel:)` + `speculativeDecodingTelemetry`), downscales oversized VLM images to fit `image_max_tokens`, and wraps `MLXLMCommon.convert` in `MLXModelConverter`. |
| `CoreAIProvider.swift` | `CoreAIProvider` & `CoreAIBundleManifest` | Inspects Schema `0.1`/`0.2` LLM bundles (`language.max_context_length`, `vision.backbone`, `speculative.drafter_kind`) and `speech_recognizer` bundles (`parakeet_tdt` `metadata.json` + legacy `whisper` `encoder.aimodel`/`decoder.aimodel`), executes `.aimodel` LLMs via `CoreAILanguageModels`, and drives two on-device ANE speech recognition backends in `transcribeAudio(wavData:languageHint:)`: Tier A out-of-the-box Apple `SpeechAnalyzer` + `SpeechTranscriber` (`apple-speech-transcriber`) and Tier B `CoreAISpeech.SpeechRecognitionModel` (`Whisper` & `Parakeet TDT`), backed by RIFF/WAVE subchunk-walking `decodeAudioMonoPCM` (`FLLR`-safe + multi-rate/channel resampling) and persistent ANE-lane residency in `ComputeScheduler.transcribeVoice`. |
| `GGUFSubprocessProvider.swift` | `GGUFSubprocessProvider` & `GGUFSpawnConfig` | Spawns one `llama-server -np 1 --no-webui` subprocess per loaded GGUF model on an OS-socket-probed loopback port from `GGUFPortPool` (`18080+`), scrubs `LLAMA_ARG_*` env vars, evaluates unified memory headroom via `GGUFSpawnConfig.assessRAMFit` (dropping external drafter if memory is tight and retrying without drafter if `llama-server` rejects an incompatible drafter at launch), skips media-blind sidecar templates when `--mmproj` is active, passes `--reasoning-preserve`, `--ctx-checkpoints`, and `--cache-reuse`, clamps `--image-max-tokens <= ubatchSize` for non-causal vision projectors, queries `GET /props` on startup to upgrade `contextSize` and `templateSource` to `.observed`, forwards `chat_template_kwargs`, `stop`, and `response_format` (`json_schema`), and surfaces mid-stream `data: {"error": ...}` SSE frames as `ProviderExecutionError`. |
| `RemoteHeylookProvider.swift` | `RemoteHeylookProvider` | Streams `POST /v1/messages` SSE frames from a remote peer discovered via Bonjour (`_heylook._tcp`) or Tailscale, forwarding `DELETE /v1/requests/{id}` cancellation and `POST /v1/requests/{id}/reasoning-end`. |
| `SubprocessSupport.swift` | `SubprocessLifecycleRegistry` & `GGUFPortPool` | Probes free TCP ports via `Darwin.bind` before allocation and tracks all child `llama-server` PIDs in a thread-safe registry with a C `atexit` hook and `SIGTERM` -> `SIGKILL` escalation so zero orphaned subprocesses survive model eviction, server stop, or app quit. |
