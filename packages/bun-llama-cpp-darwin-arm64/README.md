# bun-llama-cpp-darwin-arm64

Prebuilt native binaries for [bun-llama-cpp](https://github.com/renatoadorno/bun-llama-cpp) on macOS Apple Silicon (arm64).

This package is automatically installed as an optional dependency of `bun-llama-cpp`. You should not need to install it directly.

## Contents

- `libllama.dylib` — llama.cpp shared library (Metal-accelerated)
- `libllama_shims.dylib` — FFI shim layer for struct-by-value handling
- `libggml-*.dylib` — GGML backend libraries (base, cpu, metal, blas)
- `ggml-metal.metal` — Metal compute shaders
