#!/usr/bin/env bash
set -euo pipefail

# Build, test, or run heylookitsanllm-macos with remote SwiftPM native packages enabled
# (mlx-swift, mlx-swift-lm, swift-transformers / swift-jinja, optional coreai-models) and Xcode Metal shader compilation.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-build}"
shift || true

export HEYLOOK_ENABLE_NATIVE_MLX="${HEYLOOK_ENABLE_NATIVE_MLX:-1}"
export HEYLOOK_ENABLE_COREAI="${HEYLOOK_ENABLE_COREAI:-0}"

cd "${REPO_ROOT}"

find_products_dir() {
  local dir
  dir="$(ls -td "${HOME}"/Library/Developer/Xcode/DerivedData/heylookitsanllm-macos-*/Build/Products/Debug 2>/dev/null | head -n 1 || true)"
  if [[ -z "${dir}" || ! -d "${dir}" ]]; then
    dir="$(xcodebuild -scheme heylook-app -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation -skipMacroValidation -showBuildSettings 2>/dev/null \
      | awk -F' = ' '/^[[:space:]]*BUILT_PRODUCTS_DIR = / {print $2; exit}' || true)"
  fi
  if [[ -z "${dir}" || ! -d "${dir}" ]]; then
    dir="${REPO_ROOT}/.build/out/Products/Debug"
  fi
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

assemble_app_bundle() {
  local products_dir="$1"
  local app_dir="${products_dir}/HeylookWorkbench.app"
  mkdir -p "${app_dir}/Contents/MacOS" "${app_dir}/Contents/Resources"
  cp -f "${products_dir}/heylook-app" "${app_dir}/Contents/MacOS/heylook-app"
  chmod +x "${app_dir}/Contents/MacOS/heylook-app"

  for bundle in "${products_dir}"/*.bundle; do
    if [[ -d "${bundle}" ]]; then
      local bname
      bname="$(basename "${bundle}")"
      rm -rf "${app_dir}/Contents/Resources/${bname}" "${app_dir}/Contents/MacOS/${bname}"
      cp -R "${bundle}" "${app_dir}/Contents/Resources/${bname}"
      ln -s "../Resources/${bname}" "${app_dir}/Contents/MacOS/${bname}"
    fi
  done

  cat > "${app_dir}/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>heylook-app</string>
    <key>CFBundleIdentifier</key>
    <string>com.heylook.workbench</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Heylook Workbench</string>
    <key>CFBundleDisplayName</key>
    <string>Heylook Workbench</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>27.2</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>Heylook Workbench uses the microphone for 16 kHz ANE voice prompt transcription.</string>
    <key>NSSpeechRecognitionUsageDescription</key>
    <string>Heylook Workbench transcribes voice prompts locally on the Apple Neural Engine.</string>
    <key>NSLocalNetworkUsageDescription</key>
    <string>Heylook Workbench discovers and serves _heylook._tcp inference peers on your local network.</string>
    <key>NSBonjourServices</key>
    <array>
        <string>_heylook._tcp</string>
    </array>
</dict>
</plist>
PLIST
  echo "${app_dir}"
}

case "${ACTION}" in
  build)
    xcodebuild build \
      -scheme heylook-app \
      -destination 'platform=macOS,arch=arm64' \
      -skipPackagePluginValidation \
      -skipMacroValidation \
      "$@"
    xcodebuild build \
      -scheme heylook-cli \
      -destination 'platform=macOS,arch=arm64' \
      -skipPackagePluginValidation \
      -skipMacroValidation \
      "$@"
    PRODUCTS_DIR="$(find_products_dir)"
    if [[ -x "${PRODUCTS_DIR}/heylook-app" ]]; then
      assemble_app_bundle "${PRODUCTS_DIR}" >/dev/null
    fi
    ;;
  test)
    xcodebuild test \
      -scheme heylookitsanllm-macos-Package \
      -destination 'platform=macOS,arch=arm64' \
      -only-testing:HeylookTests \
      -skipPackagePluginValidation \
      -skipMacroValidation \
      "$@"
    ;;
  cli)
    PRODUCTS_DIR="$(find_products_dir)"
    if needs_build "${PRODUCTS_DIR}/heylook-cli"; then
      xcodebuild build -scheme heylook-cli -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation -skipMacroValidation
      PRODUCTS_DIR="$(find_products_dir)"
    fi
    exec "${PRODUCTS_DIR}/heylook-cli" "$@"
    ;;
  app)
    PRODUCTS_DIR="$(find_products_dir)"
    if needs_build "${PRODUCTS_DIR}/heylook-app"; then
      xcodebuild build -scheme heylook-app -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation -skipMacroValidation
      PRODUCTS_DIR="$(find_products_dir)"
    fi
    APP_BUNDLE="$(assemble_app_bundle "${PRODUCTS_DIR}")"
    exec "${APP_BUNDLE}/Contents/MacOS/heylook-app" "$@"
    ;;
  *)
    echo "Usage: $0 [build|test|cli <args...>|app]" >&2
    exit 1
    ;;
esac
