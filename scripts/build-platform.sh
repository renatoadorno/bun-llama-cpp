#!/usr/bin/env bash
set -euo pipefail

# Build native libraries for a target platform.
# Usage: ./scripts/build-platform.sh [output_dir]
#
# Environment variables:
#   CMAKE_EXTRA_FLAGS  — additional cmake flags (e.g., -DGGML_METAL=OFF)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/llama.cpp"
OUTPUT_DIR="${1:-$ROOT_DIR/packages/bun-llama-cpp-darwin-arm64}"

OS="$(uname -s)"
ARCH="$(uname -m)"
echo "Building for $OS-$ARCH"
echo "Output: $OUTPUT_DIR"

# Step 1: Build llama.cpp
echo ""
echo "=== Building llama.cpp ==="
cd "$LLAMA_DIR"

CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON"

if [[ "$OS" == "Darwin" ]]; then
  CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_METAL=ON"
fi

cmake -B build $CMAKE_FLAGS ${CMAKE_EXTRA_FLAGS:-}
cmake --build build --config Release -j "$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)"

# Step 2: Build shims
echo ""
echo "=== Building libllama_shims ==="
cd "$ROOT_DIR"

if [[ "$OS" == "Darwin" ]]; then
  EXT="dylib"
  cc -shared -fPIC -o native/libllama_shims.$EXT native/llama_shims.c \
    -I llama.cpp/include -I llama.cpp/ggml/include \
    -undefined dynamic_lookup
else
  EXT="so"
  cc -shared -fPIC -o native/libllama_shims.$EXT native/llama_shims.c \
    -I llama.cpp/include -I llama.cpp/ggml/include
fi

# Step 3: Copy binaries to output directory
echo ""
echo "=== Copying binaries to $OUTPUT_DIR ==="
mkdir -p "$OUTPUT_DIR"

cp "$LLAMA_DIR/build/bin/libllama.$EXT" "$OUTPUT_DIR/"
cp "$LLAMA_DIR/build/bin/libggml"*".$EXT" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$ROOT_DIR/native/libllama_shims.$EXT" "$OUTPUT_DIR/"

# Copy Metal shaders if on macOS
if [[ "$OS" == "Darwin" ]]; then
  cp "$LLAMA_DIR/build/bin/ggml-metal.metal" "$OUTPUT_DIR/" 2>/dev/null || true
  cp "$LLAMA_DIR/build/bin/default.metallib" "$OUTPUT_DIR/" 2>/dev/null || true
fi

echo ""
echo "=== Build complete ==="
ls -lh "$OUTPUT_DIR/"*.$EXT "$OUTPUT_DIR/"*.metal 2>/dev/null || true
