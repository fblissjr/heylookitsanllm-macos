#!/usr/bin/env bash
# Synchronizes Apple's official Xcode 27.x agent skills (xcrun agent skills export),
# Xcode's bundled IDEIntelligenceChat AdditionalDocumentation guides, and optional
# Apple CoreAI agent skills (working-with-coreai, model-authoring,
# model-compression-exploration) into the workspace's .agents/skills/ directory
# for automatic Antigravity (agy) discovery and sandbox-safe reading.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILLS_DIR="${1:-$REPO_ROOT/.agents/skills}"

mkdir -p "$SKILLS_DIR"

echo "==> 1. Exporting official Apple Xcode 27.x Agent Skills via xcrun agent skills export..."
if xcrun --find agent >/dev/null 2>&1; then
    xcrun agent skills export --replace-existing "$SKILLS_DIR"
else
    echo "    [warn] 'xcrun agent' not found in active Xcode toolchain; skipping Xcode skill export."
fi

echo "==> 2. Materializing Xcode IDEIntelligenceChat AdditionalDocumentation guides..."
DEV_DIR="$(xcode-select -p 2>/dev/null || echo "/Applications/Xcode.app/Contents/Developer")"
XCODE_CONTENTS="$(cd "$DEV_DIR/.." && pwd)"
ADD_DOCS_DIR="$XCODE_CONTENTS/PlugIns/IDEIntelligenceChat.framework/Versions/A/Resources/AdditionalDocumentation"

if [[ -d "$ADD_DOCS_DIR" ]]; then
    GUIDES_SKILL_DIR="$SKILLS_DIR/xcode-27-guides"
    mkdir -p "$GUIDES_SKILL_DIR/references"
    cp -f "$ADD_DOCS_DIR"/*.md "$GUIDES_SKILL_DIR/references/"
    cat > "$GUIDES_SKILL_DIR/SKILL.md" <<'EOF'
---
name: xcode-27-guides
description: "Authoritative Apple Xcode 27 AdditionalDocumentation guides covering Liquid Glass in SwiftUI/AppKit/UIKit/WidgetKit, SwiftUI New Toolbar Features, Styled Text Editing (AttributedString/AttributedTextSelection), Swift 6.2/6.4 Concurrency Updates, InlineArray & Span, FoundationModels on-device LLM, AppIntents Updates, SwiftData Class Inheritance, StoreKit, WebKit, and 3D Swift Charts."
---

# Xcode 27 Additional Documentation Guides (`xcode-27-guides`)

These guides are bundled directly inside Xcode 27 (`IDEIntelligenceChat.framework/Versions/A/Resources/AdditionalDocumentation/`) and provide Apple's canonical implementation patterns and code examples. Read the matching reference file before implementing or reviewing code in these areas:

## Liquid Glass & Design System
- [`references/SwiftUI-Implementing-Liquid-Glass-Design.md`](references/SwiftUI-Implementing-Liquid-Glass-Design.md): `.glassEffect`, `GlassEffectContainer`, `.buttonStyle(.glass)`, `.buttonStyle(.glassProminent)`, `.glassEffectID`, `.backgroundExtensionEffect()`.
- [`references/AppKit-Implementing-Liquid-Glass-Design.md`](references/AppKit-Implementing-Liquid-Glass-Design.md): `NSGlassEffectView`, `NSGlassEffectContainerView`.
- [`references/UIKit-Implementing-Liquid-Glass-Design.md`](references/UIKit-Implementing-Liquid-Glass-Design.md): UIKit Liquid Glass materials and containers.
- [`references/WidgetKit-Implementing-Liquid-Glass-Design.md`](references/WidgetKit-Implementing-Liquid-Glass-Design.md): Accented and glass widget rendering.

## SwiftUI Controls, Toolbars & Text Editing
- [`references/SwiftUI-New-Toolbar-Features.md`](references/SwiftUI-New-Toolbar-Features.md): `ToolbarSpacer(.fixed | .flexible)`, `.toolbar(id:)`, `DefaultToolbarItem`, `.searchToolbarBehavior(.minimize)`, `.sharedBackgroundVisibility`.
- [`references/SwiftUI-Styled-Text-Editing.md`](references/SwiftUI-Styled-Text-Editing.md): `TextEditor(text:selection:)` with `AttributedString` and `AttributedTextSelection`.
- [`references/SwiftUI-WebKit-Integration.md`](references/SwiftUI-WebKit-Integration.md): Native `WebView` and `WebPage` in SwiftUI.
- [`references/SwiftUI-AlarmKit-Integration.md`](references/SwiftUI-AlarmKit-Integration.md): AlarmKit scheduling and presentation.
- [`references/Swift-Charts-3D-Visualization.md`](references/Swift-Charts-3D-Visualization.md): `Chart3D` and 3D mark types.

## Swift 6.2 / 6.4 Concurrency & Low-Level Primitives
- [`references/Swift-Concurrency-Updates.md`](references/Swift-Concurrency-Updates.md): `@MainActor` isolation, isolated conformances, `@concurrent` background offloading, task execution semantics.
- [`references/Swift-InlineArray-Span.md`](references/Swift-InlineArray-Span.md): Fixed-size stack `InlineArray<N, Element>` and non-escaping `Span<Element>` / `RawSpan`.
- [`references/Foundation-AttributedString-Updates.md`](references/Foundation-AttributedString-Updates.md): Paragraph and text selection attributes in `Foundation`.

## On-Device AI, App Intents & Persistence
- [`references/FoundationModels-Using-on-device-LLM-in-your-app.md`](references/FoundationModels-Using-on-device-LLM-in-your-app.md): `SystemLanguageModel`, `LanguageModelSession`, `@Generable`, `@Guide`, tool calling, and snapshot streaming.
- [`references/AppIntents-Updates.md`](references/AppIntents-Updates.md): App Intents, visual intelligence, and interactive snippets.
- [`references/Implementing-Visual-Intelligence-in-iOS.md`](references/Implementing-Visual-Intelligence-in-iOS.md): Visual Intelligence integration.
- [`references/SwiftData-Class-Inheritance.md`](references/SwiftData-Class-Inheritance.md): `@Model` inheritance and polymorphic queries.
EOF
    GUIDE_COUNT="$(find "$GUIDES_SKILL_DIR/references" -maxdepth 1 -name '*.md' | wc -l | tr -d ' ')"
    echo "    ✓ Materialized xcode-27-guides ($GUIDE_COUNT Apple markdown guides)"
else
    echo "    [warn] AdditionalDocumentation not found at $ADD_DOCS_DIR"
fi

echo "==> 3. Syncing Apple CoreAI skills (if local checkout is available)..."
COREAI_CANDIDATES=(
    "${HEYLOOK_COREAI_DIR:-}"
    "$REPO_ROOT/coderef/apple-coreai"
    "$REPO_ROOT/../../mlx/apple-coreai"
)

SYNCED_COREAI=0
for candidate in "${COREAI_CANDIDATES[@]}"; do
    [[ -z "$candidate" || ! -d "$candidate" ]] && continue
    for skill_src in \
        "$candidate/coreai-models/skills/skills"/* \
        "$candidate/coreai/coreai-model-zoo/skills/skills"/*; do
        if [[ -d "$skill_src" && -f "$skill_src/SKILL.md" ]]; then
            skill_name="$(basename "$skill_src")"
            rm -rf "$SKILLS_DIR/$skill_name"
            cp -R "$skill_src" "$SKILLS_DIR/$skill_name"
            echo "    ✓ Synced CoreAI skill: $skill_name"
            SYNCED_COREAI=$((SYNCED_COREAI + 1))
        fi
    done
    if [[ "$SYNCED_COREAI" -gt 0 ]]; then
        break
    fi
done

if [[ "$SYNCED_COREAI" -eq 0 ]]; then
    echo "    [info] No local apple-coreai checkout found (set HEYLOOK_COREAI_DIR to sync CoreAI skills)."
fi

TOTAL_SKILLS="$(find "$SKILLS_DIR" -mindepth 2 -maxdepth 2 -name 'SKILL.md' | wc -l | tr -d ' ')"
echo "==> Done! $TOTAL_SKILLS total skills ready in $SKILLS_DIR"
