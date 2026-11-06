#!/usr/bin/env bash
set -euo pipefail

# Build, test, or run heylookitsanllm-macos with full remote SwiftPM native packages enabled
# (mlx-swift, mlx-swift-lm, coreai-models, argmax-oss-swift, hummingbird) and Xcode Metal shader compilation.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-build}"
shift || true

export HEYLOOK_ENABLE_NATIVE_MLX="${HEYLOOK_ENABLE_NATIVE_MLX:-1}"
export HEYLOOK_ENABLE_COREAI="${HEYLOOK_ENABLE_COREAI:-0}"
export HEYLOOK_ENABLE_ARGMAX="${HEYLOOK_ENABLE_ARGMAX:-0}"
export HEYLOOK_ENABLE_HUMMINGBIRD="${HEYLOOK_ENABLE_HUMMINGBIRD:-1}"

cd "${REPO_ROOT}"

find_products_dir() {
  local dir
  dir="$(xcodebuild -scheme heylook-app -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation -showBuildSettings 2>/dev/null \
    | awk -F' = ' '/^[[:space:]]*BUILT_PRODUCTS_DIR = / {print $2; exit}' || true)"
  echo "${dir}"
}

needs_build() {
  local bin="$1"
  if [[ -z "${bin}" || ! -x "${bin}" ]]; then
    return 0
  fi
  if [[ -n "$(find "${REPO_ROOT}/Sources" "${REPO_ROOT}/Package.swift" -newer "${bin}" -print -quit 2>/dev/null)" ]]; then
    return 0
  fi
  return 1
}

case "${ACTION}" in
  build|test)
    xcodebuild "${ACTION}" \
      -scheme heylookitsanllm-macos-Package \
      -destination 'platform=macOS,arch=arm64' \
      -skipPackagePluginValidation \
      "$@"
    ;;
  cli)
    PRODUCTS_DIR="$(find_products_dir)"
    if needs_build "${PRODUCTS_DIR}/heylook-cli"; then
      xcodebuild build -scheme heylookitsanllm-macos-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation
      PRODUCTS_DIR="$(find_products_dir)"
    fi
    exec "${PRODUCTS_DIR}/heylook-cli" "$@"
    ;;
  app)
    PRODUCTS_DIR="$(find_products_dir)"
    if needs_build "${PRODUCTS_DIR}/heylook-app"; then
      xcodebuild build -scheme heylookitsanllm-macos-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation
      PRODUCTS_DIR="$(find_products_dir)"
    fi
    exec "${PRODUCTS_DIR}/heylook-app" "$@"
    ;;
  *)
    echo "Usage: $0 [build|test|cli <args...>|app]" >&2
    exit 1
    ;;
esac
