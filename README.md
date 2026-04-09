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
- **Embeddings** — `embed()` / `embedMany()` returning native `Float32Array`, encoder model support
- **Parallel sequences** — `inferParallel()` with continuous batching and KV prefix sharing
- **Multi-model registry** — `ModelRegistry` + `ModelPipeline` for embed → generate pipelines

## Install

This package is published on **GitHub Packages** and requires a GitHub PAT with `read:packages` scope.

### 1. Configure authentication

Add to your project's `.npmrc` (never commit this file):

```
@renatoadorno:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=YOUR_GITHUB_PAT
```

### 2. Install

```bash
bun add @renatoadorno/bun-llama-cpp
```

The correct platform binary package is installed automatically via `optionalDependencies` — no extra install needed.

For a complete step-by-step setup guide, see **[docs/getting-started.md](docs/getting-started.md)**.

### Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| macOS | Apple Silicon (arm64) | ✅ Supported |
| macOS | Intel (x64) | 🔜 Planned |
| Linux | x64 | 🔜 Planned |
| Linux | arm64 | 🔜 Planned |

## Quick Start

```typescript
import { LlamaModel } from '@renatoadorno/bun-llama-cpp'

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
  metrics: true,   // optional: collect timing data
})
// result: { text, tokenCount, aborted, metrics? }
```

### `llm.applyTemplate(messages, options?)`

Format a conversation using the model's built-in chat template.

```typescript
const prompt = await llm.applyTemplate([
  { role: 'system', content: 'You are helpful.' },
  { role: 'user', content: 'Hello!' },
])
```

### Embeddings

Load with `embeddings: true` to use a model as an encoder:

```ts
const embedder = await LlamaModel.load('./nomic-embed-text-v1.5.Q4_K_M.gguf', {
  preset: 'small',
  embeddings: true,
  poolingType: 1, // MEAN — required for nomic-embed-text
})

const vector = await embedder.embed('search_query: capital of France')
// → Float32Array[768]

const vectors = await embedder.embedMany(['doc 1', 'doc 2', 'doc 3'])
// → Float32Array[]

await embedder.dispose()
```

Embedding mode is mutually exclusive with generation — an embedding model cannot call `infer()`.

### Parallel Inference

Load with `nSeqMax > 1` to run multiple sequences concurrently on the same GPU batch:

```ts
const llm = await LlamaModel.load('./model.gguf', { preset: 'large', nSeqMax: 4 })

// Batch: all sequences run in lock-step on the GPU
const results = await llm.inferParallel([
  { prompt: 'Translate to French: Hello', onToken: (t) => process.stdout.write(t) },
  { prompt: 'Translate to Spanish: Hello', onToken: (t) => process.stdout.write(t) },
])

// With nSeqMax > 1, infer() bypasses the serial queue and runs concurrently
// via the batch engine — multiple requests execute in parallel on the GPU
await Promise.all([
  llm.infer('Prompt A', { onToken: (t) => process.stdout.write(t) }),
  llm.infer('Prompt B', { onToken: (t) => process.stdout.write(t) }),
])

await llm.dispose()
```

Use `llm.warmup(systemPrompt)` to pre-compute the KV cache for a shared system prompt before calling `inferParallel()` — this gives ~8× speedup when all sequences share the same prefix.

### Multi-Model Orchestration

`ModelRegistry` manages multiple models by name. `ModelPipeline` orchestrates embed → rerank → generate without owning a vector store:

```ts
import { ModelRegistry, ModelPipeline, assertCapability } from '@renatoadorno/bun-llama-cpp'

const registry = new ModelRegistry()
await registry.load('embed', './nomic-embed-text.gguf', { embeddings: true, poolingType: 1 })
await registry.load('gen',   './qwen3-8b.gguf',         { preset: 'medium' })

console.log(registry.status('embed')) // 'ready'

const pipeline = new ModelPipeline(registry.get('embed'), registry.get('gen'))

// Rerank candidates (retrieved from your vector store)
const ranked = await pipeline.rerank('search_query: tallest mountain', [
  'search_document: Mount Everest is 8848 meters tall',
  'search_document: The Amazon is the largest river',
])
// → [{ doc: 'Mount Everest...', score: 0.92 }, { doc: 'The Amazon...', score: 0.41 }]

// Generate answer using top context
const context = ranked.slice(0, 2).map(r => r.doc).join('\n')
const result = await pipeline.generate(context, 'What is the tallest mountain?', {
  maxTokens: 80,
  onToken: (t) => process.stdout.write(t),
})

await registry.disposeAll()
```

`assertCapability(model, 'embed' | 'generate')` throws `CapabilityMismatchError` with an actionable message if the model lacks the required capability. Errors carry a `code` for programmatic handling: `CAPABILITY_MISMATCH`, `MODEL_NOT_FOUND`.

### `llm.metadata`

Model info populated after `load()` — no extra call needed.

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

- ✅ Embeddings & ranking support
- ✅ Parallel sequences (multi-user inference)
- ✅ Multi-model orchestration
- Grammar/JSON mode
- Speculative decoding

## License

[MIT](LICENSE)

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov
- [Bun](https://bun.sh) by Jarred Sumner
