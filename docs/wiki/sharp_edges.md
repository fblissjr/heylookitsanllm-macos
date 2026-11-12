# Sharp Edges & Incident Rationales (`docs/wiki/sharp_edges.md`)

Every non-obvious mechanism in `heylookitsanllm-macos` exists because a real hardware, subprocess, or protocol edge case bit in practice. Read the relevant section before changing provider, template, or embedding code.

---

## 1. Truthful Runtime vs. Silent Synthetic Fallbacks

- **What bit**: Early scaffolding defaulted `allowSyntheticFallback: Bool = true` on `MLXSwiftProvider`, `CoreAIProvider`, and `GGUFSubprocessProvider`, and defaulted `DeterministicHashEmbeddingEngine.modelID` to `"embeddinggemma-2-MLX-BF16"`. When a checkpoint path was misconfigured or `swift build` ran without Metal shaders, the server returned HTTP `200` with synthetic `[model] Processed: ...` text or cached deterministic hash vectors under the real `embeddinggemma-2-MLX-BF16` model ID—poisoning `VectorStore.isUpToDate` even after real weights were installed.
- **The mechanism**:
  1. All production providers (`MLXSwiftProvider`, `CoreAIProvider`, `GGUFSubprocessProvider`, and `ProviderFactory.makeProvider`) default `allowSyntheticFallback = false` and throw `ProviderExecutionError` (`checkpointNotFound`, `binaryNotFound`, `runtimeUnavailable`).
  2. `DeterministicHashEmbeddingEngine` and `HybridLocalEmbeddingEngine` (when compiled `.mlmodelc` towers are absent) report `modelID = "deterministic-hash-768d"`, never `"embeddinggemma-2-MLX-BF16"`.
  3. Neither `HeylookWorkbenchViewModel.bootstrap()` nor `heylook-cli` auto-seeds fake models or fake corpus rows on startup; synthetic demo data is loaded only when explicitly requested via `loadSampleDemoWorkspace()` or unit test fixtures (`allowSyntheticFallback: true`).

---

## 2. GGUF (`llama-server`) Subprocess & Vision Sharp Edges

- **Non-Causal Vision Projector Batch Abort (`clip.projector_type`)**:
  - **What bit**: Non-causal vision encoders in `llama.cpp` (`gemma4v`, `qwen2vl`, `qwen2.5vl`, `qwen3vl`, `qwen3_5`, `pixtral`, `internvl`, `minicpmv`) encode an entire image in a single non-causal attention pass (`GGML_ASSERT(n_tokens <= n_ubatch)`). If `--image-max-tokens` exceeds `-ub` (`ubatchSize`), `llama-server` aborts inside `llama_context::decode`.
  - **The mechanism**: `GGUFHeaderReader` inspects `clip.projector_type` in `mmproj-*.gguf`, sets `hasNonCausalVisionProjector = true` for non-causal projector families, and `GGUFSpawnConfig.autoSized` clamps `imageMaxTokens = min(requestedImageMaxTokens, ubatchSize)`.
- **Inherited `LLAMA_ARG_*` Environment Pollution**:
  - **What bit**: `llama-server` parses every `LLAMA_ARG_*` environment variable as a CLI flag. If the parent shell sets `LLAMA_ARG_LOG_FILE` or `LLAMA_ARG_N_PARALLEL`, it silently overrides `GGUFSpawnConfig`.
  - **The mechanism**: `GGUFSubprocessProvider.childEnvironment(from:)` strips all `LLAMA_ARG_*` keys and sets `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=604800` so Metal weight buffers stay wired across idle periods.
- **Mid-Stream `llama-server` Error Frames**:
  - **What bit**: When `llama-server` encounters a mid-stream KV slot exhaustion or decode failure after returning HTTP `200`, it emits `data: {"error": {"message": "..."}}` over SSE. Ignoring non-`choices` frames caused the stream to end silently with `stop_reason = "end_turn"`.
  - **The mechanism**: `GGUFSubprocessProvider.parseOpenAISSEDataLine` parses top-level `"error"` objects into `OpenAISSEDelta.errorMessage` and throws `ProviderExecutionError.runtimeUnavailable`.
- **Large Vocab Array Truncation & Multi-GGUF Drafter Shadowing**:
  - **What bit**: Qwen3.5 and Gemma 4 `.gguf` files store `tokenizer.ggml.tokens` (256k+ strings, several MB) ahead of `tokenizer.chat_template` and `*.context_length`. Capping GGUF header reads at 2 MB or 4,096 array items aborted header parsing before reaching the template or context length. Separately, placing `draft-*.gguf` or `mtp-*.gguf` alongside a main `.gguf` in one directory caused alphabetical `firstGGUFFile` lookup to pick `draft-*.gguf` as the primary model.
  - **The mechanism**: `GGUFHeaderReader` streams header KV entries directly via `FileHandle` past arbitrary-length vocab arrays (skipping element payloads without materializing 256k strings), and `firstGGUFFile` / `GGUFSpawnConfig.autoSized` exclude `mmproj`, `draft*`, `*-draft*`, `mtp-*`, `*-mtp*`, `eagle*`, and `*-dspark-*` when selecting the primary `.gguf`.
- **Media-Blind Sidecar `chat_template.jinja` When `--mmproj` Is Active**:
  - **What bit**: Some community GGUF repos ship a text-only `chat_template.jinja` sidecar next to a multimodal `.gguf` + `mmproj-*.gguf` whose embedded GGUF header template *does* include `<__media__>` / `<|vision_start|>`. Passing `--chat-template-file chat_template.jinja` stripped image markers from multimodal turns.
  - **The mechanism**: When `mmprojPath != nil` and Rung 3 (`chat_template.jinja`, non-operator override) has `supportsVision == false && supportsAudio == false` while the `.gguf` header's embedded template supports media, `GGUFSpawnConfig.autoSized` omits `--chat-template-file` so `llama-server` uses the multimodal embedded template.

---

## 3. MLX Swift & Reasoning Parser Sharp Edges

- **Dual Reasoning Parser Conflict in `mlx-swift-lm`**:
  - **What bit**: `MLXLMCommon` includes its own basic `ReasoningConfig` tag splitter on some `ModelConfiguration` types, which conflicts with `HeylookCore`'s 4-protocol `ReasoningParser` (`.tagPair`, `.gemmaChannel`, `.harmonyChannel`, `.recipientChannel`) and mid-stream `forceEndReasoning()`.
  - **The mechanism**: `MLXSwiftProvider` sets `config.reasoningConfig = nil` after loading the container so `StreamingPipeline.transformEventStream` is the single authoritative reasoning parser across all providers.
- **Hybrid Recurrent (`MambaCache` / `ArraysCache`) Non-Trimmable KV Snapshots & `<think>` Token Alignment**:
  - **What bit**: Hybrid SSM+Attention models (`Qwen3.5`, `Jamba`, `Nemotron-H`, `Falcon-H1`) mix standard `KVCacheSimple` layers with recurrent state caches (`MambaCache` / `ArraysCache`) where `cache.isTrimmable == false`. Attempting `cache.trim(...)` on a diverged branch corrupts recurrent state; similarly, unconditionally stitching `<think>...</think>` into cached assistant token histories when `thinking == false` (`sawReasoning == false`) shifts KV positions by two tokens on the next turn.
  - **The mechanism**: `MLXSwiftProvider` clones `[KVCache]` snapshots at system-prompt and turn boundaries, only trims caches when `allSatisfy(\.isTrimmable)` holds (falling back to the system checkpoint or cold prefill for non-trimmable hybrid models), and wraps assistant reasoning tags in `sessionTokenHistory` iff `sawReasoning` was true on that turn.
- **Vision-Config Stubs on Text-Only Checkpoints**:
  - **What bit**: Some Hugging Face text-only conversions of multimodal families (e.g., `Qwen3.5` or `Gemma 4` text-only variants) leave a stub `vision_config` in `config.json`. Calling `VLMModelFactory.shared` fails when vision tower weights are absent.
  - **The mechanism**: `MLXSwiftProvider.ensureContainerLoaded()` attempts `VLMModelFactory.shared` first when `supportsVision.value` is `true`, and automatically falls back to `LLMModelFactory.shared` while updating `supportsVision` to `false` (`.observed` provenance).
- **Clean `MessageRecord.content` for KV Prefix Reuse**:
  - **What bit**: Prepending synthetic `[IMAGE: filename.png]` strings to `MessageRecord.content` in `HeylookWorkbenchViewModel.sendChatPrompt()` mutated the text history stored in `SQLiteStore`, causing `PrefixCacheTracker` to report `history_diverged` on subsequent turns.
  - **The mechanism**: `MessageRecord.content` stores the user's clean `promptText` (falling back to attachment filenames only when `promptText` is empty), while media payloads travel in `MessageTurn.attachments`.

---

## 4. Apple Neural Engine (ANE) & Multimodal Media Sharp Edges

- **FP16 Overflow (`65,504`) in Gemma RMSNorm Layers on ANE**:
  - **What bit**: The Apple Neural Engine executes `MLModel` graphs in IEEE 754 `FP16`. Squaring hidden states (`x * x`) inside standard `RMSNorm` layers overflows `65,504` on later Gemma 2/3/4 layers, producing `NaN` embeddings on ANE while passing on CPU/GPU (`FP32`/`BF16`).
  - **The mechanism**: `scripts/export_ane_coreml.py` wraps normalization layers in `ANESafeRMSNorm`, scaling inputs by $1/16$ before squaring (`(x * 0.0625)^2`), which is mathematically invariant after normalization and keeps all intermediate activations well within `FP16` bounds.
- **Portrait iPhone Video Orientation**:
  - **What bit**: `AVAssetImageGenerator` defaults `appliesPreferredTrackTransform = false`, decoding portrait iPhone videos sideways (`90°` rotated) and degrading vision embedding recall.
  - **The mechanism**: Every `AVAssetImageGenerator` in `ArtifactProcessor` sets `appliesPreferredTrackTransform = true`.
- **`ANELargeModelCompilerService` 30–35 GB Temporary Scratch Pre-Allocation on Core AI `.aimodel` Compilation**:
  - **What bit**: When compiling multi-gigabyte Core AI `.aimodel` bundles (`xcrun coreai-build` or first-load `AIModel(asset:)` compilation via `ANELargeModelCompilerService`), macOS pre-allocates 30–35 GB of temporary scratch space under `/private/var/tmp/` (`com.apple.ane.compiler.*`). If free disk space is below ~35 GB, `ANELargeModelCompilerService` fails mid-compile or leaves partial compiler artifacts in `/private/var/tmp/`.
  - **The mechanism**: Verify at least 40 GB of free disk space (`df -h /private/var/tmp`) before invoking `./scripts/export_coreai.sh compile`, and pre-compile `.aimodel` bundles ahead of time rather than triggering cold on-demand compilation during an interactive chat turn.

---

## 5. Toolchain & Build System Sharp Edges

- **Homebrew `swift` (`macOS-14.0`) vs. Xcode `MacOSX27.2.sdk` (`27.0`) Manifest Linker Warning**:
  - **What bit**: When `/opt/homebrew/opt/swift/bin/swift` precedes `/usr/bin/swift` in `$PATH`, running bare `swift build` emits `ld: warning: building for macOS-14.0, but linking with dylib '/usr/lib/swift/libswiftCore.dylib' which was built for newer version 27.0` while compiling `Package.swift` into a temporary manifest executable (`heylookitsanllm-macos-manifest`). Homebrew's open-source `swift-package` hardcodes `MinimumDeploymentTarget.default = macOS 14.0` for manifest compilation, while linking against Xcode's `MacOSX27.2.sdk` `.tbd` stubs (`27.0`).
  - **The mechanism**: Package targets themselves compile cleanly for `arm64-apple-macosx27.2`. To silence the manifest-loader warning on the CLI, invoke Xcode's native Swift toolchain via `xcrun swift build` / `/usr/bin/swift build` (whose host target is `arm64-apple-macosx27.2.0`) or `brew unlink swift`.
- **Multi-Target `swiftc -typecheck` & Single-File `@main` Diagnostics (`swift-lsp`)**:
  - **What bit**: Running `xcrun swiftc -typecheck` directly on a target's `.swift` files in a multi-target SwiftPM package (`HeylookUI` importing `HeylookCore`, `HeylookEmbeddings`, `HeylookEngines`, `HeylookServer`) fails with `error: no such module 'HeylookCore'` on `iOS` Simulator when only a macOS `.build` exists, and running `swiftc -typecheck` on a single-file `@main` target (`Sources/HeylookApp/HeylookApp.swift`) without `-parse-as-library` fails with `'main' attribute cannot be used in a module that contains top-level code`.
  - **The mechanism**: `sourcekit_mcp.py` (`ensure_local_swiftpm_dependency_modules`) topologically emits `.swiftmodule` interfaces (`-emit-module -experimental-allow-module-with-compiler-errors`) for sibling `Sources/<Target>` modules into `/tmp/swift-lsp-local-modules-<hash>-<platform>-<min_version>` and automatically appends `-parse-as-library` when any target file contains `@main`.
- **Guarding `27.3`-Only SDK Symbols When `LSMinimumSystemVersion` is `27.2`**:
  - **What bit**: Compiling with `Xcode 27.3 Beta` (`MacOSX27.3.sdk`) on a host Mac running `macOS 27.2` allows calling `@available(anyAppleOS 27.3, *)` APIs (such as `.toolbarContainerBehavior(.automatic)`), but raising `Package.swift` `platforms:` or `LSMinimumSystemVersion` to `27.3` prevents `HeylookWorkbench.app` and `swift test` from launching on `macOS 27.2`—and calling a `27.3` symbol unguarded with a `27.2` deployment target fails compilation or crashes at launch.
  - **The mechanism**: Keep `Package.swift` `platforms:` at `.macOS("27.2")` and `.iOS("27.2")` (and `LSMinimumSystemVersion = 27.2` in `scripts/build_native.sh`), and wrap `27.3`-only view modifiers inside an `if #available(macOS 27.3, iOS 27.3, *)` `ViewModifier` (`ToolbarContainerBehavior27_3Modifier`).

---

## 6. SwiftUI & AppKit Window / Split-View Sharp Edges (`HeylookApp` & `HeylookUI`)

- **Unresponsive Keyboard Input & Non-Resizable Windows on Bare SwiftPM Executables**:
  - **What bit**: Launching a raw SwiftPM executable (`heylook-app`) outside a `.app` bundle starts the process with `NSApplication.ActivationPolicy.prohibited` / `.accessory`. The window appears on screen, but AppKit never makes it key (`NSTextInputClient` keyboard events never reach `TextField` or `TextEditor`) and may omit `.resizable` from `window.styleMask`.
  - **The mechanism**: `HeylookApp` wires `@NSApplicationDelegateAdaptor(HeylookAppDelegate.self)` calling `NSApplication.shared.setActivationPolicy(.regular)`, `NSApplication.shared.activate()`, `window.styleMask.insert(.resizable)`, and `window.makeKeyAndOrderFront(nil)`, while `scripts/build_native.sh` packages `HeylookWorkbench.app` with `Contents/Info.plist`.
- **Nested `HSplitView` Inside `NavigationSplitView` Collapsing to Bottom-Left**:
  - **What bit**: Placing an AppKit-backed `HSplitView` inside the `detail:` column of a `NavigationSplitView` without infinite frame bounds causes `NSSplitView` to collapse to its intrinsic minimum size in the bottom-left corner and push the leading sidebar off-screen.
  - **The mechanism**: `HeylookRootView` uses a single `NavigationSplitView` + contextual `HeylookSidebarView` sections + native trailing `.inspector(isPresented:)` modifier, with `.frame(maxWidth: .infinity, maxHeight: .infinity)` on the detail container.
- **`ToolbarOverflowMenu` and `.topBarPinnedTrailing` Are Unavailable on macOS 27**:
  - **What bit**: Some community WWDC26 / Golden Gate blog posts place `ToolbarOverflowMenu` and `ToolbarItem(placement: .topBarPinnedTrailing)` inside macOS `NavigationSplitView` examples. In `MacOSX27.2.sdk` (`SwiftUI.swiftinterface` lines 8033 & 31555), both symbols are annotated `@available(iOS 27.0, visionOS 27.0, *) @available(macOS, unavailable)` and fail to compile on macOS.
  - **The mechanism**: Use `.visibilityPriority(.high)` / `.visibilityPriority(.low)` and `.contentMarginsRemoved()` on shared multiplatform toolbars, and gate `ToolbarOverflowMenu` / `.topBarPinnedTrailing` inside `#if os(iOS)`.
- **Xcode 27 `@State` Macro Double-Assignment & Inline `Binding(get:set:)` Closures**:
  - **What bit**: In SDK 27, `@State` is a macro that synthesizes backing storage. Providing both a default value at declaration (`@State private var x = 0`) and assigning `self.x = y` in `init` either fails to compile (`Variable used before being initialized`) or silently ignores the `init` assignment. Separately, inline `Binding(get:set:)` closures inside `body` allocate a new closure on every evaluation and trigger redundant child invalidations.
  - **The mechanism**: Declare `@State private var` with either a declaration default OR an `init` assignment (never both), and bind `@Observable` view-model projections via `@Bindable` KeyPaths (`$viewModel.boundTemperature`) instead of closure bindings.
- **Transitive `@Observable` Array Subscription (`registeredModels.first(where:)`) & View-Body Sorting**:
  - **What bit**: Computing `activeModelDescription` as `registeredModels.first(where: { $0.id == selectedModelID })` or sorting `activeGGUFPorts` inside `View.body` causes every view reading `activeModelDescription` (`ChatTopControlBar`, `ChatComposerBar`, `LoadAndSamplerInspectorView`) to register an `ObservationRegistrar` access on the entire `registeredModels` array—triggering full view invalidations whenever any background model's residency or sidecar metadata changes.
  - **The mechanism**: Cache `activeModelDescription: EngineDescription?` and `sortedActiveGGUFPorts: [(modelID: String, port: Int)]` as stored properties on `HeylookWorkbenchViewModel`, updated inside `didSet` observers on `registeredModels`, `selectedModelID`, and `activeGGUFPorts`.
- **Main-Thread `NSImage(contentsOf:)` Decoding Stalls in `List` Rows**:
  - **What bit**: Calling `NSImage(contentsOf: url)` synchronously inside `ThumbnailBadge.body` (`HitRowView`) blocks `@MainActor` on disk I/O and image decoding during fast scroll over large image/media search results, and requires `#if canImport(AppKit)` branching.
  - **The mechanism**: Use SwiftUI's cross-platform `AsyncImage(url:)` in both `ThumbnailBadge` and `EmbeddingsInspectorView` so image loading and decoding run asynchronously off the main thread on both `macOS 27.2+` and `iOS 27.2+`.
- **Antigravity (`agy`) Skill Discovery vs. `~/.agents/skills` Sandbox Block**:
  - **What bit**: Exporting Xcode's `xcrun agent skills export` bundles only to `~/.agents/skills/` leaves them outside `agy`'s workspace discovery root (`<repo>/.agents/skills/`) and outside the terminal sandbox's readable paths (`Operation not permitted` when reading `references/*.md`).
  - **The mechanism**: `./scripts/sync_agent_skills.sh` exports Xcode 27.x skills, Xcode's `AdditionalDocumentation` guides, and `apple/coreai-models` skills directly into `<repo>/.agents/skills/` (gitignored), where `agy` discovers them at highest workspace precedence with full in-sandbox read access.

---

## 7. On-Device Audio & `.aimodel` Bundle Discovery Sharp Edges (`HeylookCore` & `HeylookEngines`)

- **Apple `AVAudioRecorder` Writes a ~4 KB `FLLR` Alignment Chunk Before `data`**:
  - **What bit**: Assuming a WAV file's PCM samples start at fixed byte offset `44` fails silently on recordings captured with macOS/iOS `AVAudioRecorder`: CoreAudio inserts a ~4,052-byte zero-padded `FLLR` subchunk between `fmt ` and `data` to page-align sample data, causing naive offset-44 readers to ingest ~127 ms of leading zeros (or misread stereo/48 kHz frames as 16 kHz mono).
  - **The mechanism**: `CoreAIProvider.decodeRIFFChunksMonoPCM` walks actual RIFF subchunks (`fmt `, `FLLR`, `JUNK`, `LIST`, `data`), supports 8/16/24/32-bit PCM and IEEE Float32, downmixes multi-channel frames to mono, and resamples arbitrary sample rates to 16 kHz via `ArtifactProcessor.resampleMonoFloatPCM` (falling back to `ArtifactProcessor.decodeMonoPCM` for compressed `M4A`/`MP3`/`FLAC`/`CAF` containers).
- **Top-Level `<model>.aimodel` Bundle Roots vs. Nested Component `.aimodel` / `.mlmodelc` Subdirectories**:
  - **What bit**: `coreai-models` ships both single-directory `<name>.aimodel` bundles placed directly inside a scan folder AND multi-component speech/LLM folders whose inner components are named `encoder.aimodel`, `decoder.aimodel`, `decoder_step.aimodel`, `joint.aimodel`, or `main.mlmodelc`. Unconditionally calling `url.deletingLastPathComponent()` on every `.aimodel` directory shifts a top-level `<scanRoot>/parakeet.aimodel` bundle's `modelPath` up to `<scanRoot>` itself!
  - **The mechanism**: `HeylookConfig.resolveCoreAIBundleRoot(for:scanRoot:)` shifts to the parent directory ONLY when the `.aimodel`/`.mlmodelc` directory is an inner component name (`encoder`, `decoder`, `decoder_step`, `joint`, `main`, `vision`, `audio`, `text`, `embed`, `drafter`) or lives inside an outer `.aimodel` container, and calls `enumerator.skipDescendants()` once a `.aimodel` or `.mlmodelc` package is matched.
- **`CoreAISpeech.Decoder` Shadows `Swift.Decoder` in `Codable` Initializers**:
  - **What bit**: When `HEYLOOK_ENABLE_COREAI=1` links `CoreAISpeech` from `apple/coreai-models`, any file importing `CoreAISpeech` inherits `CoreAISpeech.Decoder` (its acoustic speech decoder protocol), causing `init(from decoder: Decoder)` to fail with `value of type 'any Decoder' has no member 'container'`.
  - **The mechanism**: Always qualify `Codable` initializers in `CoreAIProvider.swift` as `public init(from decoder: any Swift.Decoder) throws`.
- **`URL.deletingPathExtension()` Truncates Model Directory Names Containing Version Dots (`0.6b`, `3.5`)**:
  - **What bit**: Calling `bundleURL.deletingPathExtension().lastPathComponent` on a directory such as `parakeet-tdt-0.6b-v3_float16_static` treats `.6b-v3_float16_static` as a file extension and truncates the derived model ID to `parakeet-tdt-0`.
  - **The mechanism**: `CoreAIBundleManifest.canonicalBundleID(for:)` prefers `metadata.json` `"name"` when present and strips `.pathExtension` **only** when the extension is `"aimodel"` or `"mlmodelc"`.
- **`NWListener(using:on:)` Throws `EINVAL` (`NWError 22`) When `NWParameters.requiredLocalEndpoint` Already Sets a Port**:
  - **What bit**: Setting `params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: nwPort)` to restrict loopback binding and then calling `NWListener(using: params, on: nwPort)` causes `Network.framework` (`libnetwork`) to reject the double-specified port with `NWError error 22 - Invalid argument`.
  - **The mechanism**: In `HeylookHTTPServer.start()`, invoke `try NWListener(using: params)` (`on: .any`) whenever `params.requiredLocalEndpoint` carries `(host, port)`, and pass `on: nwPort` only for wildcard (`0.0.0.0` / `::`) listeners.
