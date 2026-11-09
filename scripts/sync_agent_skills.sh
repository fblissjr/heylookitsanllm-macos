#!/usr/bin/env bash
# Synchronizes project-specific external agent skills into the workspace's
# .agents/skills/ directory for Antigravity (agy) discovery and sandbox-safe reading.
#
# Architecture (2-Tier Skill Boundary):
#   - Tier 1 (Global Plugin — 'swift-lsp'): Indexes all 170+ Xcode 27.x Agent Skills,
#     AdditionalDocumentation guides, Swift diagnostic docs, and .swiftinterface headers
#     directly from $(xcode-select -p) via 'swift_sdk_interface(module: "XcodeSkills" | "XcodeDocs")'.
#   - Tier 2 (Repo-Local — '.agents/skills/'): Holds ONLY project-specific external skills
#     not bundled in Xcode.app — specifically the 5 Apple CoreAI ('apple/coreai-models') skills.
#
# Usage:
#   ./scripts/sync_agent_skills.sh [--include-xcode] [target-skills-dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INCLUDE_XCODE=0
SKILLS_DIR="$REPO_ROOT/.agents/skills"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-xcode|--all-xcode)
            INCLUDE_XCODE=1
            shift
            ;;
        -h|--help)
            sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            SKILLS_DIR="$1"
            shift
            ;;
    esac
done

mkdir -p "$SKILLS_DIR"
chmod -R u+w "$SKILLS_DIR" 2>/dev/null || true

XCODE_SKILL_NAMES=(
    "adopt-c-bounds-safety"
    "app-intents-specialist"
    "app-intents-whats-new-27"
    "app-resizability"
    "audit-xcode-security-settings"
    "building-document-based-swiftui-applications"
    "device-interaction"
    "macos-touch-gestures"
    "modernize-tests"
    "siri-app-integration"
    "siri-evaluations"
    "swiftui-specialist"
    "swiftui-whats-new-27"
    "translation"
    "translation-coordinator"
    "xcode-27-guides"
    "xctrace"
)

if [[ "$INCLUDE_XCODE" -eq 0 ]]; then
    echo "==> 1. Pruning duplicate Xcode skills from $SKILLS_DIR (served globally by 'swift-lsp')..."
    PRUNED=0
    for name in "${XCODE_SKILL_NAMES[@]}"; do
        if [[ -d "$SKILLS_DIR/$name" ]]; then
            chmod -R u+w "$SKILLS_DIR/$name" 2>/dev/null || true
            rm -rf "$SKILLS_DIR/$name"
            PRUNED=$((PRUNED + 1))
        fi
    done
    echo "    ✓ Removed $PRUNED duplicate Xcode skill directories (pass --include-xcode to export them locally)."
else
    DEV_DIR="$(xcode-select -p 2>/dev/null || echo "/Applications/Xcode.app/Contents/Developer")"
    XCODE_CONTENTS="$(cd "$DEV_DIR/.." && pwd)"
    IDE_CHAT_RES="$XCODE_CONTENTS/PlugIns/IDEIntelligenceChat.framework/Versions/A/Resources"
    ADD_DOCS_DIR="$IDE_CHAT_RES/AdditionalDocumentation"

    echo "==> 1. Exporting official Apple Xcode 27.x Agent Skills (--include-xcode)..."
    if xcrun --find agent >/dev/null 2>&1 && xcrun agent skills export --replace-existing "$SKILLS_DIR" 2>/dev/null; then
        chmod -R u+w "$SKILLS_DIR" 2>/dev/null || true
    elif [[ -d "$IDE_CHAT_RES" ]]; then
        for tmpl in "$IDE_CHAT_RES"/*.idechatprompttemplate; do
            [[ -f "$tmpl" ]] || continue
            if head -n 1 "$tmpl" | grep -q '^---$'; then
                skill_name="$(basename "$tmpl" .idechatprompttemplate)"
                mkdir -p "$SKILLS_DIR/$skill_name/references"
                cp -f "$tmpl" "$SKILLS_DIR/$skill_name/SKILL.md"
                chmod u+w "$SKILLS_DIR/$skill_name/SKILL.md"
                for ref in "$IDE_CHAT_RES/${skill_name}-ref-"*.md.packaged; do
                    [[ -f "$ref" ]] || continue
                    ref_base="$(basename "$ref" .md.packaged)"
                    ref_name="${ref_base#${skill_name}-ref-}.md"
                    cp -f "$ref" "$SKILLS_DIR/$skill_name/references/$ref_name"
                    chmod u+w "$SKILLS_DIR/$skill_name/references/$ref_name"
                done
                echo "    ✓ Unpacked Xcode skill: $skill_name"
            fi
        done
    fi

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

Read the matching reference file in `references/` before implementing or reviewing code in these areas.
EOF
        GUIDE_COUNT="$(find "$GUIDES_SKILL_DIR/references" -maxdepth 1 -name '*.md' | wc -l | tr -d ' ')"
        echo "    ✓ Materialized xcode-27-guides ($GUIDE_COUNT Apple markdown guides)"
    fi
fi

echo "==> 2. Syncing project-specific Apple CoreAI skills..."
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
            chmod -R u+w "$SKILLS_DIR/$skill_name" 2>/dev/null || true
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
echo "==> Done! $TOTAL_SKILLS project-local skills ready in $SKILLS_DIR"
