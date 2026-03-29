# bun-llama-cpp

High-performance FFI bindings from [Bun](https://bun.sh) to [llama.cpp](https://github.com/ggml-org/llama.cpp). Direct `dlopen` to `libllama` with near-zero overhead.

> **Status:** Early development — API may change between minor versions.

## Features

- **Direct FFI** — `bun:ffi` `dlopen` with ~10ns call overhead (vs ~50ns N-API)
- **Metal acceleration** — Full GPU offload on Apple Silicon
- **Worker isolation** — All llama.cpp calls run in a dedicated Bun Worker
- **Streaming tokens** — `onToken` callback for real-time output
- **Abort support** — `AbortSignal` with `SharedArrayBuffer` cross-thread signaling
- **Serial queue** — Thread-safe concurrent `infer()` calls via Promise-based queue
- **Graceful shutdown** — Proper GPU/Metal buffer cleanup on `dispose()`

## Install

```bash
bun add bun-llama-cpp
```

The correct native binaries for your platform are installed automatically via `optionalDependencies`.

### Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| macOS | Apple Silicon (arm64) | ✅ Supported |
| macOS | Intel (x64) | 🔜 Planned |
| Linux | x64 | 🔜 Planned |
| Linux | arm64 | 🔜 Planned |

## Quick Start

```typescript
import { LlamaModel } from 'bun-llama-cpp'

const llm = await LlamaModel.load('./model.gguf', { preset: 'medium' })

const result = await llm.infer('Explain quantum computing in 3 sentences.', {
  maxTokens: 200,
  onToken: (text) => process.stdout.write(text),
  signal: AbortSignal.timeout(30_000),
})

console.log(`\n\nTokens: ${result.tokenCount}`)
await llm.dispose()
```

## API

### `LlamaModel.load(modelPath, config?)`

Load a GGUF model file and return an initialized `LlamaModel`.

```typescript
const llm = await LlamaModel.load('./model.gguf', {
  preset: 'medium',     // 'small' | 'medium' | 'large'
  nGpuLayers: 99,       // GPU layers (99 = all)
  nCtx: 4096,           // Context window size
  nThreads: 4,          // CPU threads
  temperature: 0.7,     // Sampling temperature
  topK: 40,             // Top-K sampling
  topP: 0.9,            // Top-P (nucleus) sampling
  minP: 0.05,           // Min-P sampling
})
```

### `llm.infer(prompt, options)`

Run inference with streaming output.

```typescript
const result = await llm.infer(prompt, {
  maxTokens: 512,
  onToken: (text) => { /* streaming callback */ },
  signal: AbortSignal.timeout(30_000),
})
// result: { text: string, tokenCount: number, aborted: boolean }
```

### `llm.dispose()`

Free GPU/Metal buffers and terminate the worker. Always call this when done.

## Building from Source

### Prerequisites

- [Bun](https://bun.sh) >= 1.1.0
- CMake >= 3.14
- C/C++ compiler (Xcode CLI tools on macOS)

### Clone & Build

```bash
git clone --recurse-submodules https://github.com/renatoadorno/bun-llama-cpp.git
cd bun-llama-cpp
bun install

# Build llama.cpp (Metal enabled on macOS)
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
cd ..

# Build the C shim
bun run build:shims

# Run the demo (requires a GGUF model in models/)
bun run start
```

### Run Tests

```bash
bun run test
```

> Tests load a ~5GB model and run real inference. Timeouts are 60–180 seconds.

## Architecture

```
LlamaModel (public API)
  └── Bun Worker (thread boundary)
       ├── libllama.dylib    ← llama.cpp via dlopen
       └── libllama_shims.dylib  ← C shim for struct-by-value
```

The C shim exists because `bun:ffi` cannot pass C structs by value. It wraps llama.cpp functions that take/return structs, using buffer pointers instead.

## Roadmap

See [docs/strategy.md](docs/strategy.md) for the full roadmap:

- Embeddings & ranking support
- Parallel sequences (multi-user inference)
- Multi-model orchestration
- Grammar/JSON mode
- Speculative decoding

## License

[MIT](LICENSE)

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov
- [Bun](https://bun.sh) by Jarred Sumner
