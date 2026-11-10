# SwiftUI Architecture & macOS 27.2+ (Golden Gate) Design System (`docs/wiki/ui_architecture_and_macos27.md`)

This document defines the authoritative UI architecture, layout hierarchy, focus & window lifecycle rules, and **macOS 27.2+ / iOS 27.2+ (Swift 6.2)** design patterns for `HeylookUI` (`Sources/HeylookUI/`) and `HeylookApp` (`Sources/HeylookApp/`).

---

## 1. Authoritative SDK & Reference Sources

All UI code in `heylookitsanllm-macos` is grounded in five canonical sources:

1. **Installed Xcode `MacOSX27.3.sdk` & `iPhoneSimulator27.3.sdk` `.swiftinterface` Headers (Targeting `27.2+` Deployment Floor)**:
   - Verified via `swift-lsp` (`swift_sdk_interface` and `swift_diagnostics`) against:
     - `SwiftUI.framework/Modules/SwiftUI.swiftmodule/arm64e-apple-macos.swiftinterface`
     - `SwiftUICore.framework/Modules/SwiftUICore.swiftmodule/arm64e-apple-macos.swiftinterface`
     - `AppKit.framework/Modules/AppKit.swiftmodule/arm64e-apple-macos.swiftinterface`
   - `Package.swift` `platforms:` and `scripts/build_native.sh` `LSMinimumSystemVersion` remain at `27.2` so binaries built with the `Xcode 27.3` SDK execute natively on `macOS 27.2` and `iOS 27.2` hosts without unguarded symbol crashes. APIs introduced in `27.3` (such as `.toolbarContainerBehavior(.automatic)`) are guarded with `if #available(macOS 27.3, iOS 27.3, *)`.
2. **Apple First-Party Xcode 27.x Agent Skills (`./scripts/sync_agent_skills.sh` -> `.agents/skills/`)**:
   - Exported via `xcrun agent skills export --replace-existing .agents/skills`:
     - `swiftui-specialist` (`references/structure.md`, `dataflow.md`, `environment.md`, `modifiers.md`, `localization.md`, `animations.md`, `foreach.md`, `soft-deprecation.md`, `soft-deprecated-apis.md`)
     - `swiftui-whats-new-27` (`references/state-macro.md`, `content-builder.md`, `toolbar.md`, `reorderable.md`, `swipe-actions.md`, `item-binding.md`, `async-image.md`)
     - `app-resizability` (dynamic window/scene sizing, Stage Manager, and iPhone Duo / iOS 27 resizability)
     - `building-document-based-swiftui-applications` (`ReadableDocument`, `WritableDocument`, `DocumentGroupLaunchScene`)
     - `modernize-tests`, `xctrace`, `app-intents-specialist`, `app-intents-whats-new-27`
   - Plus Apple's `coreai-models` skills (`working-with-coreai`, `model-authoring`, `model-compression-exploration`).
3. **Apple Xcode Intelligence Documentation Guides (`.agents/skills/xcode-27-guides/references/`)**:
   - `SwiftUI-Implementing-Liquid-Glass-Design.md` (`.glassEffect`, `GlassEffectContainer`, `.buttonStyle(.glass)`, `.buttonStyle(.glassProminent)`, `.backgroundExtensionEffect()`)
   - `AppKit-Implementing-Liquid-Glass-Design.md` (`NSGlassEffectView`, `NSGlassEffectContainerView`)
   - `SwiftUI-New-Toolbar-Features.md` (`.toolbar(id:)`, `ToolbarSpacer(.fixed | .flexible)`, `DefaultToolbarItem`, `.searchToolbarBehavior(.minimize)`, `.sharedBackgroundVisibility`)
   - `SwiftUI-Styled-Text-Editing.md` (`AttributedString`, `AttributedTextSelection`, `TextEditor(text:selection:)`)
   - `Swift-Concurrency-Updates.md` (Swift 6.2/6.4 `@MainActor` isolation, isolated conformances, `@concurrent` background offloading)
   - `FoundationModels-Using-on-device-LLM-in-your-app.md` (declarative snapshot streaming into `@Observable` state)
4. **Upstream Apple Reference Code (`coderef/apple-mlx/mlx-swift-examples`)**:
   - `Applications/MLXChatExample/` (`MLXChatExampleApp.swift`, `ChatView.swift`, `Views/ConversationView.swift`, `Views/PromptField.swift`, `ViewModels/ChatViewModel.swift`) for `@MainActor @Observable` view-model separation, `.defaultScrollAnchor(.bottom, for: .sizeChanges)`, and media attachment preview strips.
5. **Heylook Core Principles ([`VISION.md`](../../VISION.md))**:
   - *Nothing happens where it cannot be seen*: Every setting surfaces its `value`, `provenance` (`derived`, `configured`, `observed`, `observed_cached`, `request`), `effect` (`per_request`, `reload`, `restart`), and `computePlacement`.
   - *Derive, don't copy*: Thinking depth options (`ThinkingControls`) and vision/audio attachment gating are derived from the model's own files.
   - *The interface tells the truth*: Discloses live token throughput, KV prefix cache reuse (`cache X/Y`), thinking budgets (`End thinking`), and ANE vs. GPU memory residency splits.

---

## 2. Window Activation, Resizability & Bundle Packaging (`Sources/HeylookApp/`)

### 2.1 Why Bare SwiftPM Executables Lose Keyboard Focus on macOS
When a macOS executable is launched from a CLI build directory (`.build/debug/heylook-app`) without a `.app` bundle or explicit activation policy, macOS starts the process with `NSApplication.ActivationPolicy.prohibited` / `.accessory`. The window renders on screen, but **AppKit refuses to make it key or route keyboard events (`NSTextInputClient`) to any `TextField` or `TextEditor`**.

### 2.2 Mandatory App Lifecycle Rules (`HeylookApp.swift`)
1. **`@NSApplicationDelegateAdaptor(HeylookAppDelegate.self)`**:
   - In `applicationWillFinishLaunching`, call `NSApplication.shared.setActivationPolicy(.regular)`.
   - In `applicationDidFinishLaunching`, call `NSApplication.shared.activate()` and ensure every `NSWindow` includes `.resizable` in `window.styleMask` and calls `makeKeyAndOrderFront(nil)`.
2. **Scene Modifiers on `WindowGroup`**:
   - Always configure `.defaultSize(width: 1360, height: 860)` and `.windowResizability(.contentMinSize)`.
   - Include standard macOS menu commands: `SidebarCommands()`, `InspectorCommands()`, and surface shortcuts (`⌘1`–`⌘6`, `⌘N`, `⇧⌘N`).
3. **`.app` Bundle Assembly (`scripts/build_native.sh`)**:
   - `./scripts/build_native.sh build` and `./scripts/build_native.sh app` assemble `HeylookWorkbench.app` with `Contents/Info.plist` (`CFBundlePackageType = APPL`, `LSMinimumSystemVersion = 27.2`) alongside `mlx-swift_Cmlx.bundle` (`default.metallib`).

---

## 3. Single-Hierarchy Split & Inspector Topology (`Sources/HeylookUI/`)

### 3.1 Never Nest `HSplitView` Inside `NavigationSplitView`
Placing an AppKit-backed `HSplitView` inside the `detail:` column of a `NavigationSplitView` without explicit infinite frame bounds causes `NSSplitView` to collapse to its intrinsic minimum size in the bottom-left corner and push the leading sidebar column off-screen.

### 3.2 Canonical 3-Region Workbench Hierarchy
`HeylookRootView` (`HeylookWorkbenchView.swift`) owns the single top-level layout hierarchy:

```text
NavigationSplitView(columnVisibility: $columnVisibility)
├── Sidebar: HeylookSidebarView (.listStyle(.sidebar), min: 250, ideal: 280, max: 340)
│   ├── Section("Workbench") — 6 primary surfaces (Chat, Notebook, Embeddings, Presets, Models, Perf & Server)
│   ├── Contextual Surface Sections:
│   │   ├── .chat       -> Conversations list + New Conversation (⌘N)
│   │   ├── .notebook   -> Saved Notebooks list + New Notebook (⇧⌘N)
│   │   ├── .embeddings -> Modality Filter, Index & Watched Folders, EmbeddingGemma 2, ANE/GPU Placement
│   │   ├── .presets    -> PresetsSidebarSection (Scope filter, Search, New Preset)
│   │   ├── .models     -> Provider Filter (All, MLX, CoreAI, GGUF, Remote) + Model Actions
│   │   └── .perf       -> Telemetry & Server quick actions
│   └── Bottom Safe-Area Bar -> Server status (:8000) + GPU/ANE loaded model counts
│
└── Detail: Active Surface View (.frame(maxWidth: .infinity, maxHeight: .infinity))
    ├── Dismissible Status / Discovery Banners
    ├── Surface Content (ChatWorkbenchSurface | NotebookSurface | EmbeddingsStudioSurface | PresetsSurface | ModelsInspectorSurface | PerfServerSurface)
    └── .inspector(isPresented: $viewModel.showInspector) (.inspectorColumnWidth(min: 300, ideal: 340, max: 430))
        ├── .embeddings -> EmbeddingsInspectorView (AVPlayerView, Matryoshka 768d/512d/256d/128d bars, frame seek pills)
        └── .chat / .notebook / .presets / .models / .perf -> LoadAndSamplerInspectorView (5-layer Sampler Cascade + System Prompt)
```

---

## 4. First-Responder Focus & Streaming Transcript Rules

1. **Explicit `@FocusState` + `.defaultFocus` on Every Primary Input**:
   - `ChatWorkbenchSurface`: `@FocusState private var isPromptFocused: Bool` bound via `.focused($isPromptFocused)` and `.defaultFocus($isPromptFocused, true)` on the prompt `TextField`.
   - `EmbeddingsStudioSurface`: `@FocusState private var isSearchFocused: Bool` bound to the search `TextField`.
   - `NotebookSurface`: `@FocusState private var isBodyFocused: Bool` bound to the document `TextEditor`.
2. **Bottom-Anchored Streaming Transcripts**:
   - Use `ScrollView` + `LazyVStack` with `.defaultScrollAnchor(.bottom, for: .sizeChanges)` (following `MLXChatExample/Views/ConversationView.swift`) so streaming tokens keep the bottom of the transcript visible without jittery manual `scrollTo` loops on every token.
3. **Standard Empty States**:
   - Always use `ContentUnavailableView` with actionable starter buttons when a conversation, search result set, or model registry is empty.

---

## 5. Adopting Liquid Glass & macOS 27.2+ APIs

When styling controls, toolbars, and floating bars on `macOS 27.2+` and `iOS 27.2+`:

1. **Liquid Glass Button Styles (`GlassButtonStyle` / `GlassProminentButtonStyle`)**:
   - Use `.buttonStyle(.glass)` for secondary interactive controls and `.buttonStyle(.glassProminent)` for primary call-to-action buttons (`Send`, `Search`, `Continue Writing`, `Run Eval Suite`).
2. **Liquid Glass Containers & Surfaces (`GlassEffectContainer` & `.glassEffect`)**:
   - Wrap adjacent glass pills or floating control bars in `GlassEffectContainer(spacing:)` so effects batch in a single rendering pass and blend smoothly.
   - Apply `.glassEffect(.regular, in: .rect(cornerRadius: ...))` or `.glassEffect(.regular.interactive(), in: .capsule)` *after* layout/padding modifiers.
3. **Toolbar Grouping, Overflow & 27.3 Container Behavior (`ToolbarSpacer`, `.visibilityPriority`, `ToolbarOverflowMenu`, `.toolbarContainerBehavior`)**:
   - Separate logical groups in `.toolbar` using `ToolbarSpacer(.fixed)` or `ToolbarSpacer(.flexible)` rather than embedding manual `Divider()` views inside a single `ToolbarItem`.
   - Apply `.visibilityPriority(.high)` (`ToolbarItemVisibilityPriority`) to primary toolbar controls so they stay visible in narrow windows before overflowing.
   - Apply `.modifier(ToolbarContainerBehavior27_3Modifier())` (`if #available(macOS 27.3, iOS 27.3, *) { content.toolbarContainerBehavior(.automatic) }`) on the root `NavigationSplitView`.
   - **Dual-SDK Availability Rule**: `ToolbarOverflowMenu` and `ToolbarItemPlacement.topBarPinnedTrailing` are `@available(iOS 27.0, visionOS 27.0, *) @available(macOS, unavailable)` (and `ToolbarContainerBehavior.local` is `@available(iOS 27.3, *) @available(macOS, unavailable)`). Never use iOS-only toolbar symbols unguarded in multiplatform views; gate `ToolbarOverflowMenu` with `#if os(iOS)`.
4. **Item-Driven Confirmation Dialogs (`confirmationDialog(_:item:)`)**:
   - Destructive actions (`Clear Index…` in `EmbeddingsSidebarSections` and `Unload All Models…` across the sidebar, `ModelsTopActionBar`, and `PerfServerSurface`) set `viewModel.pendingConfirmation` (`HeylookWorkbenchViewModel.DestructiveActionPrompt`) and present `.confirmationDialog("Confirm Action", item: $viewModel.pendingConfirmation)` at the root `HeylookWorkbenchView` level per `swiftui-whats-new-27/references/item-binding.md`.
5. **Swift 6.2 / 6.4 `@MainActor` `@Observable` Isolation**:
   - `HeylookWorkbenchViewModel` is `@MainActor @Observable`. Background work (`ComputeScheduler`, `HybridLocalEmbeddingEngine`, `HeylookHTTPServer`, `MLXModelConverter`) runs inside dedicated actors or `@concurrent` async functions and yields `Sendable` value types back to `@MainActor`.

---

## 6. `swiftui-specialist` View Factoring & `@Observable` Dataflow Rules

Per Apple's exported `swiftui-specialist` and `swiftui-whats-new-27` skills:

1. **Separate `View` Structs Over `@ViewBuilder` Computed Properties (`structure.md`)**:
   - A `View` struct is SwiftUI's unit of invalidation. Multi-section layouts must be separate `View` structs rather than monolithic `body` closures or `private func` helpers on the parent view:
     - `HeylookWorkbenchView`: `StatusBannerBar`, `EmptyModelsDiscoveryBanner`, `ChatConversationsSidebarSection`, `NotebooksSidebarSection`, `EmbeddingsSidebarSections`, `PresetsSidebarSection`, `ModelsFilterSidebarSection`, `PerfQuickActionsSidebarSection`.
     - `ChatWorkbenchSurface` & `MarkdownMessageView`: `ChatTopControlBar`, `ChatPromptPreviewDrawer`, `ChatComposerBar`, `ChatAttachmentPillView`, `MessageBubbleView`, `StreamingAssistantView`, `MarkdownMessageView`.
     - `NotebookAndEmbeddingsSurfaces` & `CursorAwareTextEditor`: `EmbeddingsSearchHeaderBar`, `EmbeddingsStatusToolbar`, `EmbeddingsEvalLabDrawer`, `ModalityFilterRow`, `EmbeddingsInspectorView`, `ScoredFeatureBreakdownCard`, `CursorAwareTextEditor`.
     - `PresetsSurface`: Responsive master-detail `sampler` / `full` scope Preset editor and per-document preset applicator.
     - `ModelsInspectorSurface`: `ModelsTopActionBar`, `ModelRegistryRowView`.
2. **Cached `@Observable` Stored Projections & `@Bindable` KeyPaths (`dataflow.md` & `foreach.md`)**:
   - Pass `@Observable` models into child views via `@Bindable var viewModel: HeylookWorkbenchViewModel` (never `@Binding var viewModel: HeylookWorkbenchViewModel`).
   - Cache derived collections and lookups (`activeModelDescription`, `filteredModels`, `modelCountsByProvider`, `sortedActiveGGUFPorts`) in `didSet` hooks on `HeylookWorkbenchViewModel` instead of recomputing `.first(where:)`, `.filter`, or `.sorted` inside `View.body` or `ForEach` arguments.
   - Pass `Sequence.enumerated()` directly to `ForEach(HeylookSurface.allCases.enumerated(), id: \.element.id)` (Swift 6.1+ `EnumeratedSequence: RandomAccessCollection`) without wrapping in `Array(...)`, or wrap indexed domain elements in a dedicated `Identifiable` value type (such as `KeyframeStamp: Identifiable` and precomputed `sortedAttributes` in `EmbeddingsInspectorView`).
   - Never construct `Binding(get: { ... }, set: { ... })` closures inside `body`. Expose computed getter/setter properties or subscripts on `@Observable` (`$viewModel.boundTemperature`, `$viewModel.boundEffectiveQueryTask`, `$viewModel[computeUnitFor: component]`) and bind via `@Bindable` KeyPaths.
3. **`@State` Macro Rules in SDK 27 (`state-macro.md`)**:
   - Always mark `@State` properties `private` so `@Observable` reference types initialize lazily once per view lifetime.
   - Never provide both a declaration-site default value (`@State private var x = 0`) and an `init` assignment (`self.x = y`); drop the declaration default when initializing in `init`.
4. **`@ContentBuilder` Trailing-Closure Modifiers (`content-builder.md`)**:
   - Always use the trailing-closure form for `.overlay { ... }` and `.background { ... }` when passing modified `ShapeStyle` or view expressions so `@ContentBuilder` resolves overloads unambiguously.
5. **Non-Blocking Off-Main-Thread Image Loading (`AsyncImage(url:)` over `NSImage(contentsOf:)` — `async-image.md`)**:
   - Never call synchronous `NSImage(contentsOf: url)` inside `List` rows (`ThumbnailBadge`) or inspector views (`EmbeddingsInspectorView`). Use cross-platform `AsyncImage(url:)` so disk I/O and image decoding happen off `@MainActor` across both `macOS 27.2+` and `iOS 27.2+`.
6. **Dual-Platform `swift_diagnostics` Verification (`macos` & `ios` `27.3`)**:
   - Every UI change across `Sources/HeylookUI/`, `Sources/HeylookApp/`, and `Tests/HeylookTests/` is typechecked against both `arm64-apple-macosx27.3` and `arm64-apple-ios27.3-simulator` via `swift_diagnostics(platform: "both", min_version: "27.3")` (which automatically compiles/caches intra-package SwiftPM target `.swiftmodule` dependencies and passes `-parse-as-library` for `@main` entrypoints).

---

## 7. Golden Gate Material Layers, Touch on Mac & SDK 27 Runtime Behaviors

1. **Four Material Layers & Zero Structural Card Budget**:
   - Keep `.glassEffect` restricted to floating/transient chrome (composer bar, sidebar status pill, empty-state action cluster). Keep primary scrolling content (`List`, `MessageBubbleView`, `HitRowView`, `TextEditor`) on calm standard materials or subtle fills.
   - Read `@Environment(\.appearsActive)` on custom sidebar footers so status chrome dims when the window loses key focus.
2. **macOS 27 SDK Linked Runtime Behavior Rules**:
   - **Menu Item Icons (`#1`)**: Linking against `MacOSX27.x.sdk` hides menu item images by default. Attach `.labelStyle(.titleAndIcon)` inside `Menu` / `CommandGroup` items whenever an icon conveys state (such as active preset checkmarks).
   - **`NSOpenPanel` Content Types (`#2`) & Multiplatform `.fileImporter` Parity**: `NSOpenPanel` automatically derives `canChooseFiles` / `canChooseDirectories` from `allowedContentTypes`. On `#if !os(macOS)` (`iOS 27.2+`), pair file selection actions with `.fileImporter(isPresented:allowedContentTypes:allowsMultipleSelection:)` so media attachment and corpus indexing work on both macOS and iOS.
   - **Sheet & Popover Modifier Reset (`#6`)**: Control modifiers (`.controlSize`, `.buttonStyle`, `.tint`) applied outside `.sheet` or `.popover` reset inside the presented sheet on macOS 27; apply control modifiers directly inside the sheet's root content.
3. **Touch on Mac (28pt+ Hit Targets & Hover-Free Discoverability)**:
   - Ensure compact icon-only buttons (`StatusBannerBar` dismiss, `ChatAttachmentPillView` remove, `MessageBubbleView` edit pencil, watched-folder remove, search clear) apply `.frame(minWidth: 28, minHeight: 28).contentShape(Rectangle())` and never gate actions behind mouse-hover-only affordances so direct touch (`GestureInputKinds.directTouch` / Sidecar / PROLED displays) and iOS 27 scenes have full parity.
