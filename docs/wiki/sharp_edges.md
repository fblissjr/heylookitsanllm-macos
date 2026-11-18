# 8. Sharp Edges & Incident Rationales

Battle-tested invariants surfaced while aligning `heylookitsanllm-macos` across `HeylookCore`, `HeylookEmbeddings`, `HeylookEngines`, `HeylookServer`, and `HeylookUI`.

---

### 1. Truncate Query Param Is `DELETE ?after=`, Never `?from=`, and Single Deletes Do Not Compact Positions
- **Incident**: Treating message truncation as inclusive `?from=` or compacting subsequent message positions shifts indices out from under active conversation branches, orphan GC, and concurrent UI clients.
- **Invariant**: `DELETE /v1/conversations/:id/messages?after=<pos>` deletes strictly `position > after` (missing or non-integer `after` returns HTTP `422`), and `deleteMessage(conversationID:messageID:)` removes only the targeted row without renumbering subsequent rows (`gaps are valid` in `SQLiteStore`).

### 2. Never Forward User `stop` Sequences Upstream to `llama-server`
- **Incident**: Forwarding `body["stop"]` to `llama-server` terminates generation immediately if a stop string (like `---` or `User:`) happens to appear inside a `<think>...</think>` reasoning block before any user-facing reply begins.
- **Invariant**: `GGUFSubprocessProvider.buildLlamaServerArgs` intentionally omits `stop` from the upstream `llama-server` JSON payload. `StreamingPipeline`'s `StopSequenceFilter` holds back partial suffix matches and cuts **visible `delta.text` only**, never `<think>` reasoning deltas.

### 3. Never Save `.effective` Cascade Snapshots Back Into Presets or `model.heylook.toml` ("Preset is a Copy" + "Null-Means-Cascade")
- **Incident**: Persisting resolved `.effective` cascade values freezes stale vendor/builtin defaults (`temperature = 1.0`, `top_p = 0.95`, `max_tokens = 16384`) into user presets and sidecar files, silently masking future model updates.
- **Invariant**: Unset controls store `nil` (serialized as `null` to delete overrides on `PATCH`), `ModelSidecarConfig.toExplicitJSONDictionary()` exports only explicitly set keys, and empty `model.heylook.toml` sidecars are deleted by `TOMLCodec.updateModelSidecar` when all keys are cleared.

### 4. Preserving VLM Prompts & Image-Keyed APC Snapshots (`PreRenderedVLMTokenizerProxy` & `mediaSalt`)
- **Incident**: Passing plain strings into `mlx-swift-lm` VLM pipelines re-applies template logic over pre-rendered Jinja chat prompts (dropping vision placeholders or wiping KV continuity), and keying KV prefix snapshots on `tokenIDs` alone allows different images with identical prompt tokens to collide.
- **Invariant**: `MLXSwiftProvider` wraps rendered prompts in `PreRenderedVLMTokenizerProxy`, caches encoded visual features by image digest + geometry in `VisionFeatureCache` (16 entries / 512 MB), and keys `APCSnapshotCache` on `(tokenIDs, mediaSalt)`.

### 5. Gemma 4 Vision `48px` Patch Alignment vs Hardcoded `28px` Strides
- **Incident**: Assuming a fixed `28px` vision patch grid misaligns Gemma 4 `16 * 3 = 48px` pooling boundaries, inflating visual tokens or failing shape checks.
- **Invariant**: Vision alignment is derived per-checkpoint via `ModelSpecialization` (`Gemma4Specialization.alignmentMultiplePx = 48`), never hardcoded globally as `28px`.

### 6. Suppress `--cache-reuse` on SWA & Hybrid Recurrent GGUF Models
- **Incident**: Enabling `--cache-reuse` KV shifting on sliding-window attention (`gemma4`, SWA models) or hybrid recurrent/SSM checkpoints (`qwen3_5`, `lfm2`, `granite`) corrupts recurrent/sliding KV state slots across multi-turn prompts.
- **Invariant**: `GGUFSpawnConfig.autoSized` checks `ModelSpecialization`: SWA, Gemma 4, and hybrid recurrent architectures **exclude `--cache-reuse`** and instead configure `--ctx-checkpoints 32 --checkpoint-min-step 256 --reasoning-preserve`.

### 7. Non-Destructive Generation Sagas, Client Disconnects & `409 generation_in_progress`
- **Incident**: Truncating old turns *before* generation starts destroys user history if prefill fails (`409`/`503`/OOM) or aborts early, and tying generation lifetime to the HTTP socket aborts long mobile requests on brief Wi-Fi drops.
- **Invariant**: `ConversationOrchestrator` acquires `RequestRegistry.tryAcquireConversationLock` (`409` on overlapping mutations/previews/generates), streams via detachable `SubscriberHub`, and executes `SQLiteStore.replaceTailWithMessage` or `replaceTailWithUpdate` in one transaction **only after** output settles.

### 8. Subchunk-Safe RIFF Audio Decoding (`decodeRIFFChunksMonoPCM`)
- **Incident**: Fixed-offset `44`-byte WAV header parsers misread real-world recordings containing `FLLR`, `JUNK`, `LIST`, or metadata subchunks as corrupt noise.
- **Invariant**: `SpeechTranscriptionService.decodeRIFFChunksMonoPCM` walks RIFF subchunk headers explicitly until locating exact `fmt ` and `data` payloads before resampling to 16 kHz mono PCM.

### 9. Embedding Truthfulness: `hybrid_ane` vs `all_ane`, `windows.seconds`, and Zero Synthetic Fallbacks
- **Rules**:
  - Under `hybrid_ane`, `MLXSwiftLMEmbeddingGemma2Engine` runs Vision/Audio `.mlmodelc` on ANE and bridges `softTokens` into MLX GPU without requiring `TextANE.mlmodelc`; only `all_ane` requires compiled `TextANE.mlmodelc`.
  - When `windows.seconds` is requested (including clips `<= 16s`), decode consecutive non-overlapping windows one window at a time (`WindowPlanner.consecutiveRanges` + `ArtifactProcessor.decodeVideoWindowFrames`).
  - `GGUFEmbeddingEngine` enforces `ctx_size <= n_ubatch <= n_batch` and rejects video `windows` requests with HTTP `400`.
  - Production `Sources/` contains zero `allowSyntheticFallback` toggles and zero `DeterministicHashEmbeddingEngine`.

### 10. Swift 6.4 (`v6`), macOS `27.2.1` + `27.3` Availability & Sandbox Traps
- **Platform Guards**: Keep `27.3+` APIs (`ToolbarContainerBehavior27_3Modifier`, `SpeechAnalyzer` / future Siri `ToolIntent`) inside `#available(macOS 27.3, iOS 27.3, *)` so `27.2.1` hosts and `27.2+` deployment targets stay green.
- **Xcode Build Required for Metal/TCC**: Raw `swift build` cannot compile `.metal` kernels into `default.metallib` or embed TCC-hardened `Info.plist` entitlements; use `./scripts/build_native.sh`.
- **Sandboxed Git / Cache Paths**: Prefix sandboxed git invocations with `GIT_CONFIG_GLOBAL=/dev/null` when `~/.gitconfig.local` is unreadable outside the sandbox, and route SwiftPM scratch/module caches to `/tmp` when `/var/folders` is restricted.
