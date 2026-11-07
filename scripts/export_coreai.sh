#!/usr/bin/env bash
set -euo pipefail

# Export or compile models to Apple Core AI (.aimodel) bundles for heylookitsanllm-macos.
# Wraps `coreai.model.registry`, `coreai.llm.export`, `coreai.embedding.export`,
# `models/gemma4/export.py`, and `xcrun coreai-build compile`.
#
# Environment variables:
#   COREAI_MODELS_DIR : Path to a local `apple/coreai-models` checkout
#   HEYLOOK_MODELS_DIR: Default output directory for exported `.aimodel` bundles
#                       (defaults to "${HOME}/.heylook/models")

SUBCOMMAND="${1:-help}"
shift || true

DEFAULT_OUT_DIR="${HEYLOOK_MODELS_DIR:-${HOME}/.heylook/models}"

resolve_coreai_repo() {
  if [[ -n "${COREAI_MODELS_DIR:-}" && -d "${COREAI_MODELS_DIR}" ]]; then
    echo "${COREAI_MODELS_DIR}"
    return 0
  fi
  for candidate in \
    "${HOME}/mlx/apple-coreai/coreai-models" \
    "${HOME}/mlx/apple-coreai/coreai/coreai-models" \
    "../coreai-models"; do
    if [[ -d "${candidate}" ]]; then
      (cd "${candidate}" && pwd)
      return 0
    fi
  done
  echo "Error: Could not locate coreai-models checkout. Set COREAI_MODELS_DIR=/path/to/coreai-models." >&2
  return 1
}

case "${SUBCOMMAND}" in
  list)
    REPO="$(resolve_coreai_repo)"
    cd "${REPO}"
    exec uv run coreai.model.registry --list-models "$@"
    ;;
  llm)
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 llm <hf-id-or-preset> [--platform macOS|iOS] [--compression 4bit|4bit_weights_8bit_kv_cache|fp4_weights_fp8_activations_fp8_kv_cache|fp8_weights_fp8_activations_fp8_kv_cache|4bit_weight_palettized_group32|4bit_weight_palettized_group8|none] [--output-dir <dir>]" >&2
      exit 1
    fi
    REPO="$(resolve_coreai_repo)"
    mkdir -p "${DEFAULT_OUT_DIR}"
    cd "${REPO}"
    exec uv run coreai.llm.export --output-dir "${DEFAULT_OUT_DIR}" "$@"
    ;;
  gemma4)
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 gemma4 <google/gemma-4-E2B-it|google/gemma-4-E4B-it> [--include-vision] [--max-context-length <pow2>] [--compression none] [--output-dir <dir>]" >&2
      exit 1
    fi
    MODEL_ID="$1"
    shift || true
    REPO="$(resolve_coreai_repo)"
    mkdir -p "${DEFAULT_OUT_DIR}"
    cd "${REPO}/models/gemma4"
    exec uv run export.py --model "${MODEL_ID}" --output-dir "${DEFAULT_OUT_DIR}" "$@"
    ;;
  embedding)
    MODEL_NAME="${1:-embeddinggemma-2}"
    shift || true
    REPO="$(resolve_coreai_repo)"
    mkdir -p "${DEFAULT_OUT_DIR}"
    cd "${REPO}"
    exec uv run coreai.embedding.export "${MODEL_NAME}" --output-dir "${DEFAULT_OUT_DIR}" "$@"
    ;;
  compile)
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 compile <path/to/model.aimodel> [--platform macOS|iOS]" >&2
      exit 1
    fi
    exec xcrun coreai-build compile "$@"
    ;;
  *)
    cat >&2 <<'EOF'
Usage: ./scripts/export_coreai.sh <subcommand> [options]

Subcommands:
  list [--type llm|diffusion|utility] [--platform macOS|iOS]
      List supported models and export presets in `coreai-models`.

  llm <hf-id-or-preset> [args...]
      Export an LLM (e.g. `Qwen/Qwen3.5-9B`, `Qwen/Qwen3-1.7B --platform iOS`,
      or `--experimental org/Model --compression fp4_weights_fp8_activations_fp8_kv_cache`)
      to `${HEYLOOK_MODELS_DIR:-~/.heylook/models}`.

  gemma4 <google/gemma-4-E2B-it|google/gemma-4-E4B-it> [--include-vision] [args...]
      Export Gemma 4 with static context buckets, INT8 PLE sidecar, and optional
      multi-bucket vision encoder (`--include-vision`).

  embedding [embeddinggemma-2] [--platform macOS|iOS] [--text-only] [args...]
      Export EmbeddingGemma 2 to `.aimodel` (macOS GPU or iOS ANE).

  compile <path/to/model.aimodel> [--platform macOS|iOS]
      Ahead-of-time compile a `.aimodel` asset using `xcrun coreai-build compile`.
EOF
    exit 1
    ;;
esac
