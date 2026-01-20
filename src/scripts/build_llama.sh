#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="$ROOT_DIR/vendor/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"

if [ ! -d "$LLAMA_DIR" ]; then
  mkdir -p "$ROOT_DIR/vendor"
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

GLSLC_BIN=""
if command -v glslc >/dev/null 2>&1; then
  GLSLC_BIN="$(command -v glslc)"
elif command -v glslangValidator >/dev/null 2>&1; then
  GLSLC_BIN="$ROOT_DIR/scripts/glslc"
  chmod +x "$GLSLC_BIN"
else
  echo "glslc not found. Install Vulkan SDK or glslang before building Vulkan backend." >&2
  exit 1
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_VULKAN=ON \
  -DVulkan_GLSLC_EXECUTABLE="$GLSLC_BIN" \
  -DBUILD_SHARED_LIBS=OFF

cmake --build "$BUILD_DIR" -j"$(nproc)"
