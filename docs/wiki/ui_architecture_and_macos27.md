# 7. SwiftUI Architecture & macOS 27.2+ Design System

`HeylookUI` (`Sources/HeylookUI/`) and `HeylookApp` (`Sources/HeylookApp/`) implement a native **Swift 6.4 (`v6` language mode)** multiplatform (`macOS 27.2+` and `iOS 27.2+`, with `#available(macOS 27.3, iOS 27.3, *)` API enhancements) 6-surface workbench sharing one `WorkbenchBackend` orchestration boundary (`WorkbenchService` locally or `RemoteWorkbenchBackend` over LAN/Bonjour).

---

## 7.1 Modular `@Observable` Controller Architecture

Instead of one monolithic view-model, `HeylookWorkbenchViewModel` (`Sources/HeylookUI/HeylookWorkbenchViewModel.swift`) acts as a thin coordinator holding six specialized `@Observable` domain controllers (`Sources/HeylookUI/Controllers/`):

1. **`DocumentSettingsController`** (`chatSettings` and `notebookSettings` — strictly separate instances):
   - Enforces **"Preset is a Copy"**, **"Null-Means-Cascade"** (clearing overrides sends `nil`/`null` so underlying model/vendor/builtin cascade layers flow through), and **"Override-Box Rule"** (`UserDefaults` session restore + sparse override tracking).
   - Implements `presetForNewDoc()` and capability reconciliation against the active `EngineContract`, plus drift reporting (`DocumentPresetDriftReport`) and `PendingPromptReplacement` prompts when a preset's system prompt conflicts with user edits.
2. **`ChatController`**: Manages multi-turn conversation loading, attachment staging/shrinking (`StagedChatAttachment`), image-plan queries (`POST /v1/models/*id/image-plan`), voice/ASR transcription turns, debounced prompt previews (`POST /v1/conversations/:id/prompt`), inline message edit/regenerate/continue/delete, and SSE streaming adoption.
3. **`NotebookAndPresetsController`**: Drives freeform `Notebook` persistence/generation and unseeded `Preset` creation, search, optimistic-timestamp reload/save (`expectedUpdatedUnixMs`), and parameter patching.
4. **`EmbeddingsStudioController`**: Coordinates rich multimodal **Compare**, **Embedding Lab** sweeps (`set.json` + 8-case built-in CoreGraphics multimodal suite), and local **Media Corpus (`mbed`)** indexing/search.
5. **`ModelsAndPerfController`**: Manages model registry projections, `ProviderConfigSchema` cache, per-model config/template drafts, live `RAMFitEvaluator` gates, watch folders (`scan-config`), operational `[settings]` + disk JSONL reader, and Bonjour (`_heylook._tcp`) / remote connection state. **Zero unconditional 3-second polling loop** — metrics, performance profiles, and observability update strictly on demand (surface transitions, generation completion, explicit user refresh, and `SQLiteStore.mutationStream()`).
6. **`RemoteWorkbenchBackend`**: Conforms to `WorkbenchBackend` to proxy all REST/SSE operations cleanly to a remote Mac server when paired via Bonjour or manual URL.

---

## 7.2 Six-Surface Layout, Shortcuts & Cross-Cutting UX

`HeylookSurface` (`Sources/HeylookUI/HeylookWorkbenchView.swift`) defines the canonical 6-surface order:

| Shortcut | Surface | Primary View (`Sources/HeylookUI/Surfaces/`) |
| :--- | :--- | :--- |
| **`⌘1`** | **Chat** | `ChatWorkbenchSurface.swift` (`ChatWorkbenchSurface`, `LoadAndSamplerInspectorView`) |
| **`⌘2`** | **Notebook** | `NotebookAndEmbeddingsSurfaces.swift` (`NotebookSurface`) |
| **`⌘3`** | **Presets** | `PresetsSurface.swift` (`PresetsSurface`) |
| **`⌘4`** | **Embeddings & ANE** | `NotebookAndEmbeddingsSurfaces.swift` (`EmbeddingsStudioSurface`) |
| **`⌘5`** | **Models** | `ModelsAndSystemSurfaces.swift` (`ModelsInspectorSurface`, `ModelConfigEditorSheet`, `MLXModelConverterSheet`) |
| **`⌘6`** | **Perf & Server** | `ModelsAndSystemSurfaces.swift` (`PerfServerSurface`) |

### Key UX & Accessibility Mechanisms

- **`AdaptiveContrastStyle`** (`HeylookWorkbenchView.swift`): Reads `@Environment(\.colorSchemeContrast)` and scales border strokes, chip background opacities, and banner contrast automatically when **Increase Contrast** is enabled.
- **`@Environment(\.scenePhase)` Persistence (`HeylookApp.swift`, `NotebookSurface`)**: Invokes `viewModel.flushPendingEdits()` when transitioning out of `.active` so debounced system prompt, sampler, and notebook edits flush to SQLite WAL safely.
- **Layered `Escape` Key Precedence (`handleLayeredEscape`)**: Pressing `Escape` dismisses overlays strictly top-down before aborting compute: Shortcuts sheet -> Confirmation dialog -> Prompt replacement dialog -> `ModelConfigEditorSheet` -> `MLXModelConverterSheet` -> Eval Lab drawer -> Model Load panel -> Model switch warning -> Inline message edit -> Prompt preview drawer -> Right Inspector -> only then cancels active Chat or Notebook generation. (Pressing `?` outside text inputs opens the Keyboard Shortcuts sheet.)
- **Chat Tail Tracking (`ChatWorkbenchSurface.swift`)**: Wraps the message list in `ScrollViewReader` with `.onScrollGeometryChange(for: Bool.self)` tracking `distanceFromBottom < 100` (`followingTail`). Switching conversations, adding messages, or receiving streaming deltas while `followingTail == true` scrolls cleanly via `proxy.scrollTo("chat-bottom", anchor: .bottom)`; scrolling upward detaches tail follow and reveals a floating **`Jump to latest`** capsule overlay during active generation.

---

## 7.3 6-Surface Feature Highlights

1. **Chat (`⌘1`)**:
   - **Inline Model Load / Reload Panel**: Configure GGUF `ctx_size` (auto, power-of-two steps, custom range) and `flash_attn` (`default` / `on` / `off` / `auto`), then click **`Load`** (1-token JIT warm) or **`Reload (Apply Changes)`** / **`Unload`**, surfacing live `stale_reload_fields`.
   - **4-State System Prompt Chip**: Displays `No system prompt`, `System prompt: <preset.name>`, `<preset.name> (modified)` (highlighted orange), or `System prompt: custom`.
   - **Multimodal Composer & Inspector**: Drag/paste images or audio (`image-plan` budget preview + 2048px shrink toggle), live voice-to-turn ASR, inline prompt preview drawer, and 5-layer sparse `LoadAndSamplerInspectorView`.
2. **Notebook (`⌘2`)**: Distraction-free text workbench with `CursorAwareTextEditor`, model & preset selectors, collapsible system prompt drawer, and mid-document streaming continuation.
3. **Presets (`⌘3`)**: Clean unseeded preset library (`COLLATE NOCASE`), search filter, optimistic-lock `Reload Latest Copy` conflict safeguard, 13 sparse sampler overrides (`Null-Means-Cascade`), scope selector (`full` vs sampling-only), and live model compatibility preview.
4. **Embeddings & ANE (`⌘4`)**:
   - **Compare**: Multi-block items (text + image/audio/video), per-block `windows.seconds` (FPS or scene-change) configuration, similarity heatmap grid, vector norm/tail inspection, and interactive **closest video window hits** that launch **`CompareVideoWindowPreviewSheet`** with native `AVPlayer` seek and clickable `frame_times` pills.
   - **Embedding Lab**: Import custom `set.json` benchmarks or load the built-in 8-case CoreGraphics multimodal benchmark, queue multi-config sweeps (models, Matryoshka dims, `prompt_name` on/off ablations, ANE compute lanes), inspect Recall@1/MRR tables, and export JSON.
   - **Media Corpus (`mbed`)**: Optional local folder/file corpus indexing (`FolderWatcher` + `IndexCoordinator`) and multimodal cosine retrieval across text, code, PDF, image, audio, and 32s/16s-stride video spans.
5. **Models (`⌘5`)**:
   - Watch folders (`heylook.toml [scan]`), empty-state guidance to configure watch folders or pair over Bonjour, RAM fit verdicts, `stale_reload_fields` + immediate **`Reload now`**, pure-Swift **`MLXModelConverterSheet`**, and schema-driven **`ModelConfigEditorSheet`** (built dynamically from `ProviderConfigSchema`, sending `null` when clearing an override so derived/vendor values are never frozen, with integrated **`RAMFitMeterView`** and 4+ rung **`ChatTemplateInspectorView`**).
6. **Perf & Server (`⌘6`)**:
   - Live Mach/Metal memory strip, loaded-model telemetry cards, `EmbeddingGemma 2` `MLComputePlan` ANE operator readout, three SQLite performance tables (`TimingBreakdownTableView`, `PerModelCacheTableView`, `RecentTrendsTableView`) across `1h / 6h / 24h / 7d`, truthful HTTP & Bonjour (`_heylook._tcp` active only on wildcard `0.0.0.0`) status, persistent operational `[settings]` editor + disk `events.jsonl` / `metrics.jsonl` table reader, and in-memory `HeylookObservabilityLog` ring viewer.

---

## 7.4 Native Xcode & Metal Build Notes

- `swift build` / `swift test` validates Swift 6.4 actor isolation, core logic, wire contracts, SQLite WAL, and UI controllers cleanly without needing GPU shaders.
- **Full Apple Silicon app / Metal / CoreML / TCC build** runs via Xcode using `./scripts/build_native.sh [build|test|app|cli]` so Metal shaders compile into `default.metallib`, native packages link properly, and TCC-hardened `Info.plist` entitlements + signing attach cleanly (using `#available(macOS 27.3, iOS 27.3, *)` such as `ToolbarContainerBehavior27_3Modifier` on `27.3+` hosts while staying `27.2.1`-compatible).
