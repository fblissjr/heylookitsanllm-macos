#!/usr/bin/env bash
# Unified maintenance script to keep Xcode 27.x, the Metal toolchain, the iOS arm64
# Simulator runtime, local Apple MLX / Core AI reference checkouts, Xcode + Core AI
# Agent Skills (.agents/skills & ~/.agents/skills), Antigravity (agy) plugins
# (including swift-lsp), and SwiftPM dependencies up to date.
#
# Usage:
#   ./scripts/update_dev_env.sh [options]
#
# Options:
#   --upgrade-xcode     Install the latest Xcode prerelease via 'xcodes' and select it
#   --update-xcodes     Update the 'xcodes' CLI binary to the latest GitHub release
#   --update-packages   Run 'swift package update' across all optional remote packages
#   --build             Run './scripts/build_native.sh build' after updating
#   --audit-xcode       Run 'sourcekit_mcp.py audit-xcode' against the active Xcode
#   --force-downloads   Force re-downloading Metal toolchain and iOS Simulator even if installed
#   --prune-old-sims    Delete older iOS Simulator disk images that do not match the active SDK
#   --skip-sim          Skip 'xcodebuild -downloadPlatform iOS -architectureVariant arm64'
#   --skip-plugins      Skip updating Antigravity (agy) plugins
#   --dry-run           Print actions without executing downloads or mutations
#   -h, --help          Show this help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

UPGRADE_XCODE=0
UPDATE_XCODES=0
UPDATE_PACKAGES=0
RUN_BUILD=0
AUDIT_XCODE=0
FORCE_DOWNLOADS=0
PRUNE_OLD_SIMS=0
SKIP_SIM=0
SKIP_PLUGINS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --upgrade-xcode)
            UPGRADE_XCODE=1
            AUDIT_XCODE=1
            shift
            ;;
        --update-xcodes)
            UPDATE_XCODES=1
            shift
            ;;
        --update-packages)
            UPDATE_PACKAGES=1
            shift
            ;;
        --build)
            RUN_BUILD=1
            shift
            ;;
        --audit-xcode)
            AUDIT_XCODE=1
            shift
            ;;
        --force-downloads)
            FORCE_DOWNLOADS=1
            shift
            ;;
        --prune-old-sims)
            PRUNE_OLD_SIMS=1
            shift
            ;;
        --skip-sim)
            SKIP_SIM=1
            shift
            ;;
        --skip-plugins)
            SKIP_PLUGINS=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (run with --help for usage)" >&2
            exit 1
            ;;
    esac
done

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "    [dry-run] $*"
    else
        "$@"
    fi
}

echo "========================================================================"
echo " heylookitsanllm-macos — Toolchain, SDK, Skills & Plugin Updater"
echo "========================================================================"

# ------------------------------------------------------------------------------
# 1. Optional xcodes CLI & Xcode Prerelease Upgrade
# ------------------------------------------------------------------------------
echo ""
echo "==> [1/6] Checking Xcode & 'xcodes' CLI..."

if [[ "$UPDATE_XCODES" -eq 1 ]] || { [[ "$UPGRADE_XCODE" -eq 1 ]] && ! command -v xcodes >/dev/null 2>&1; }; then
    echo "    Installing/updating prebuilt 'xcodes' binary in /opt/homebrew/bin..."
    if command -v brew >/dev/null 2>&1 && ! command -v aria2c >/dev/null 2>&1; then
        run_cmd brew install aria2
    fi
    if [[ "$DRY_RUN" -eq 0 ]]; then
        curl -fL https://github.com/XcodesOrg/xcodes/releases/latest/download/xcodes.zip -o /tmp/xcodes.zip
        unzip -o /tmp/xcodes.zip -d /opt/homebrew/bin
        chmod +x /opt/homebrew/bin/xcodes
        rm -f /tmp/xcodes.zip
    else
        echo "    [dry-run] curl -fL https://github.com/XcodesOrg/xcodes/releases/latest/download/xcodes.zip ..."
    fi
fi

if [[ "$UPGRADE_XCODE" -eq 1 ]]; then
    echo "    Upgrading to latest Xcode prerelease via 'xcodes'..."
    run_cmd xcodes install --latest-prerelease --experimental-unxip --select
    run_cmd sudo xcodebuild -runFirstLaunch -checkForNewerComponents
else
    ACTIVE_DEV_DIR="$(xcode-select -p 2>/dev/null || echo "unknown")"
    XCODE_VER="$(xcodebuild -version 2>/dev/null | tr '\n' ' ' || echo "unknown")"
    echo "    Active developer dir : $ACTIVE_DEV_DIR"
    echo "    Active Xcode version : $XCODE_VER"
    echo "    (Tip: pass --upgrade-xcode when a new Xcode beta/release drops)"
fi

# ------------------------------------------------------------------------------
# 2. Metal Toolchain & iOS Simulator Runtime (arm64 Only — No watchOS/tvOS/visionOS)
# ------------------------------------------------------------------------------
echo ""
echo "==> [2/6] Ensuring Metal Toolchain & iOS (arm64) Simulator Runtime..."
ACTIVE_SDK_VER="$(xcodebuild -version -sdk macosx SDKVersion 2>/dev/null || echo "27.2")"

if [[ "$FORCE_DOWNLOADS" -eq 0 ]] && xcodebuild -showComponent metaltoolchain 2>/dev/null | grep -q '^Status: installed'; then
    echo "    ✓ Metal Toolchain already installed for Xcode $ACTIVE_SDK_VER (pass --force-downloads to re-download)."
else
    run_cmd xcodebuild -downloadComponent metaltoolchain
fi

if [[ "$SKIP_SIM" -eq 0 ]]; then
    if [[ "$FORCE_DOWNLOADS" -eq 0 ]] && xcrun simctl runtime list 2>/dev/null | grep -q "iOS ${ACTIVE_SDK_VER}.*(Ready)"; then
        echo "    ✓ iOS ${ACTIVE_SDK_VER} Simulator runtime already Ready (pass --force-downloads to re-download)."
    else
        run_cmd xcodebuild -downloadPlatform iOS -architectureVariant arm64
    fi
else
    echo "    [skip] Skipping iOS Simulator runtime check (--skip-sim)."
fi

if [[ "$PRUNE_OLD_SIMS" -eq 1 ]]; then
    echo "    Pruning superseded iOS Simulator disk images (keeping iOS ${ACTIVE_SDK_VER})..."
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        if [[ "$line" != *"iOS ${ACTIVE_SDK_VER}"* ]]; then
            sim_uuid="$(echo "$line" | grep -oE '[0-9A-F]{8}-([0-9A-F]{4}-){3}[0-9A-F]{12}' || true)"
            if [[ -n "$sim_uuid" ]]; then
                echo "    Deleting old simulator runtime: $line"
                run_cmd xcrun simctl runtime delete "$sim_uuid"
            fi
        fi
    done < <(xcrun simctl runtime list 2>/dev/null | grep '^iOS ' || true)
fi

# ------------------------------------------------------------------------------
# 3. Local Apple MLX & Core AI Reference Checkouts
# ------------------------------------------------------------------------------
echo ""
echo "==> [3/6] Updating local Apple MLX & Core AI reference checkouts (if present)..."
REF_REPOS=(
    "$REPO_ROOT/../../mlx/apple-mlx/mlx-swift"
    "$REPO_ROOT/../../mlx/apple-mlx/mlx-swift-lm"
    "$REPO_ROOT/../../mlx/apple-mlx/mlx-swift-examples"
    "$REPO_ROOT/../../mlx/apple-coreai/coreai-models"
    "$REPO_ROOT/coderef/mlx-swift"
    "$REPO_ROOT/coderef/mlx-swift-lm"
    "$REPO_ROOT/coderef/apple-coreai"
)

UPDATED_REFS=0
for repo in "${REF_REPOS[@]}"; do
    if [[ -d "$repo/.git" ]]; then
        repo_name="$(basename "$repo")"
        echo "    Pulling latest in $repo_name..."
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "    [dry-run] git -C $repo pull --ff-only"
        else
            git -C "$repo" pull --ff-only || echo "    [warn] Could not fast-forward $repo_name (local changes or offline)."
        fi
        UPDATED_REFS=$((UPDATED_REFS + 1))
    fi
done
if [[ "$UPDATED_REFS" -eq 0 ]]; then
    echo "    [info] No git-backed local MLX/CoreAI reference checkouts found."
fi

# ------------------------------------------------------------------------------
# 4. Sync Xcode 27.x Agent Skills, AdditionalDocumentation Guides & Core AI Skills
# ------------------------------------------------------------------------------
echo ""
echo "==> [4/6] Syncing Agent Skills into workspace (.agents/skills) and global (~/.agents/skills)..."
run_cmd "$SCRIPT_DIR/sync_agent_skills.sh" "$REPO_ROOT/.agents/skills"

if [[ -d "$HOME/.agents/skills" ]] || [[ "${SYNC_GLOBAL_SKILLS:-1}" == "1" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "    [dry-run] $SCRIPT_DIR/sync_agent_skills.sh $HOME/.agents/skills"
    else
        "$SCRIPT_DIR/sync_agent_skills.sh" "$HOME/.agents/skills" || echo "    [warn] Global ~/.agents/skills sync skipped (run outside sandbox if needed)."
    fi
fi

# ------------------------------------------------------------------------------
# 5. Update Antigravity (agy) Plugins (swift-lsp, heylook-provider, path-privacy, etc.)
# ------------------------------------------------------------------------------
echo ""
echo "==> [5/6] Updating Antigravity (agy) plugins..."
if [[ "$SKIP_PLUGINS" -eq 1 ]]; then
    echo "    [skip] Skipping Antigravity plugin update (--skip-plugins)."
elif command -v agy >/dev/null 2>&1; then
    PLUGIN_SOURCE="${HEYLOOK_AGY_PLUGINS_SOURCE:-https://github.com/fblissjr/fb-gemini-plugins/tree/main/skills}"
    echo "    Installing/updating plugins from $PLUGIN_SOURCE..."
    run_cmd agy plugin install "$PLUGIN_SOURCE"
else
    echo "    [info] 'agy' CLI not found on PATH; skipping plugin update."
fi

# ------------------------------------------------------------------------------
# 6. Probe swift-lsp Toolchain & Optional SwiftPM Package Update / Build
# ------------------------------------------------------------------------------
echo ""
echo "==> [6/6] Verifying Swift LSP Toolchain & SDKs..."
SWIFT_LSP_SCRIPT="$HOME/.gemini/config/plugins/swift-lsp/skills/swift-lsp/scripts/sourcekit_mcp.py"
if [[ -f "$SWIFT_LSP_SCRIPT" ]]; then
    run_cmd python3 "$SWIFT_LSP_SCRIPT" probe --workspace "$REPO_ROOT" --min-version "$ACTIVE_SDK_VER"
    if [[ "$AUDIT_XCODE" -eq 1 ]]; then
        echo ""
        echo "==> Auditing active Xcode $ACTIVE_SDK_VER SDK & bundled skills via swift-lsp..."
        run_cmd python3 "$SWIFT_LSP_SCRIPT" audit-xcode --available "$ACTIVE_SDK_VER"
    fi
else
    xcodebuild -showsdks | grep -E "macOS|iOS" || true
fi

if [[ "$UPDATE_PACKAGES" -eq 1 ]]; then
    echo ""
    echo "==> Updating remote SwiftPM dependencies (Package.resolved)..."
    run_cmd env HEYLOOK_ENABLE_ALL_PACKAGES=1 xcrun swift package --package-path "$REPO_ROOT" update
fi

if [[ "$RUN_BUILD" -eq 1 ]]; then
    echo ""
    echo "==> Building native Metal shaders & SwiftPM targets..."
    run_cmd "$SCRIPT_DIR/build_native.sh" build
fi

echo ""
echo "==> All up to date!"
