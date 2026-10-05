#!/usr/bin/env bash
# scripts/package-release.sh - build the release archive for THIS machine's
# platform (what .github/workflows/release.yml runs on each native runner).
#
# Usage: ./scripts/package-release.sh [VERSION] [OUT_DIR]
#   VERSION  embedded via -ldflags and used in archive names
#            (default: `git describe --tags --always --dirty`, or "dev")
#   OUT_DIR  where to write the archive (default: ./dist)
#
# The ONNX Runtime Go bindings use CGO, so the binary is built natively with
# a C compiler (gcc/clang; MinGW gcc on Windows) rather than cross-compiled.
# The ONNX Runtime library and the models are downloaded on first run by
# `webfilter setup`, not bundled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

VERSION="${1:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"
OUT_DIR="${2:-$REPO_ROOT/dist}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

LDFLAGS="-s -w"
LDFLAGS="$LDFLAGS -X github.com/yjlion/onnx-web-filter/internal/version.Version=${VERSION}"
LDFLAGS="$LDFLAGS -X github.com/yjlion/onnx-web-filter/internal/version.Commit=${COMMIT}"
LDFLAGS="$LDFLAGS -X github.com/yjlion/onnx-web-filter/internal/version.BuildDate=${BUILD_DATE}"

goos="$(go env GOOS)"
goarch="$(go env GOARCH)"
ext=""
[[ "$goos" == "windows" ]] && ext=".exe"

mkdir -p "$OUT_DIR"
name="webfilter-${VERSION}-${goos}-${goarch}"
stage="$OUT_DIR/$name"
rm -rf "$stage"
mkdir -p "$stage/config" "$stage/policies" "$stage/docs"
echo "[package] building ${goos}/${goarch} ..."
CGO_ENABLED=1 go build -trimpath \
  -ldflags "$LDFLAGS" \
  -o "$stage/webfilter${ext}" ./cmd/webfilter
cp config/settings.example.json "$stage/config/"
cp policies/default.json.example "$stage/policies/"
cp README.md LICENSE "$stage/"
cp docs/install.md docs/ml.md docs/policies.md "$stage/docs/"
if [[ "$goos" == "windows" ]]; then
  if command -v zip >/dev/null; then
    (cd "$OUT_DIR" && zip -qr "$name.zip" "$name")
  else
    (cd "$OUT_DIR" && powershell -NoProfile -Command "Compress-Archive -Path '$name' -DestinationPath '$name.zip' -Force")
  fi
else
  tar -C "$OUT_DIR" -czf "$OUT_DIR/$name.tar.gz" "$name"
fi
rm -rf "$stage"
echo "[package] done: $OUT_DIR"
ls -la "$OUT_DIR"
