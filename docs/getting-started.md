# Getting Started with bun-llama-cpp

This guide walks you through installing and using `bun-llama-cpp` in a new project from scratch.

---

## Prerequisites

- [Bun](https://bun.sh) >= 1.1.0
- macOS Apple Silicon (arm64) — only supported platform in v0.1.0
- A GGUF model file (see [Download a Model](#download-a-model))
- A GitHub account (required for GitHub Packages authentication)

---

## 1. Authenticate with GitHub Packages

`bun-llama-cpp` is published on **GitHub Packages**, which requires authentication even for public packages. You need a GitHub Personal Access Token (PAT) with `read:packages` scope.

### Create a PAT

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **Generate new token (classic)**
3. Give it a name (e.g. `bun-llama-cpp`)
4. Check **`read:packages`** — that's the only scope needed
5. Click **Generate token** and copy the value

### Configure the registry

Create (or edit) a `.npmrc` file in the root of your project:

```
@renatoadorno:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=YOUR_GITHUB_PAT
```

Replace `YOUR_GITHUB_PAT` with the token you just created.

> **Tip:** To configure globally (available across all projects), create the file at `~/.npmrc` instead.

> **Security:** Never commit your `.npmrc` with the token to git. Add it to `.gitignore`:
> ```
> echo ".npmrc" >> .gitignore
> ```

---

## 2. Install

```bash
bun add @renatoadorno/bun-llama-cpp
```

**You don't need to install the platform package manually.** The main package declares `@renatoadorno/bun-llama-cpp-darwin-arm64` as an `optionalDependency`. Bun automatically installs it when it detects you're on macOS ARM64.

After installation, your `node_modules` will have both:

```
node_modules/
  @renatoadorno/
    bun-llama-cpp/          ← TypeScript source + public API
    bun-llama-cpp-darwin-arm64/  ← prebuilt .dylib binaries (auto-installed)
```

The library resolves the correct `.dylib` path at runtime — no configuration needed.

---

## 3. Download a Model

`bun-llama-cpp` loads models in **GGUF format**. Any model from [Hugging Face](https://huggingface.co) in GGUF format works.

Recommended starting point — Qwen3-8B (good quality, fits in ~8GB RAM):

```bash
# Create a models directory in your project
mkdir -p models

# Download using huggingface-cli (pip install huggingface-hub)
huggingface-cli download \
  Qwen/Qwen3-8B-GGUF \
  qwen3-8b-q4_k_m.gguf \
  --local-dir ./models
```

Or download directly from the browser and place it in `./models/`.

### Choosing quantization

| Suffix | Size | Quality | RAM needed |
|--------|------|---------|------------|
| `Q2_K` | ~3GB | Low | 4GB |
| `Q4_K_M` | ~5GB | Good | 8GB |
| `Q5_K_M` | ~6GB | Better | 10GB |
| `Q8_0` | ~9GB | Near-lossless | 12GB |
| `F16` | ~16GB | Full precision | 24GB |

For development, `Q4_K_M` is a good default.

---

## 4. Basic Usage

Create a file `index.ts`:

```typescript
import { LlamaModel } from '@renatoadorno/bun-llama-cpp'

const llm = await LlamaModel.load('./models/qwen3-8b-q4_k_m.gguf', {
  preset: 'medium',
})

const result = await llm.infer('Explain what Bun FFI is in 2 sentences.', {
  maxTokens: 200,
  onToken: (text) => process.stdout.write(text),
})

process.stdout.write('\n')
await llm.dispose()
```

Run it:

```bash
bun run index.ts
```

---

## 5. Using Chat Templates

Models like Qwen3, Llama 3, and Mistral have built-in chat templates that format conversations correctly. Use `applyTemplate()` instead of crafting prompt strings manually:

```typescript
import { LlamaModel } from '@renatoadorno/bun-llama-cpp'
import type { ChatMessage } from '@renatoadorno/bun-llama-cpp'

const llm = await LlamaModel.load('./models/qwen3-8b-q4_k_m.gguf', {
  preset: 'medium',
})

const messages: ChatMessage[] = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is the capital of France?' },
]

const prompt = await llm.applyTemplate(messages, { addAssistant: true })

const result = await llm.infer(prompt, {
  maxTokens: 100,
  onToken: (text) => process.stdout.write(text),
})

process.stdout.write('\n')
await llm.dispose()
```

---

## 6. Performance Metrics

Pass `metrics: true` to get timing and throughput data:

```typescript
const result = await llm.infer(prompt, {
  maxTokens: 200,
  onToken: (text) => process.stdout.write(text),
  metrics: true,
})

console.log(`\nPrompt: ${result.metrics!.promptTokens} tokens in ${result.metrics!.promptMs.toFixed(0)}ms`)
console.log(`Generate: ${result.metrics!.generatedTokens} tokens at ${result.metrics!.tokensPerSec.toFixed(1)} tok/s`)
```

---

## 7. Aborting Inference

Use the standard `AbortSignal` API to cancel in-flight generation:

```typescript
const controller = new AbortController()

// Cancel after 10 seconds
setTimeout(() => controller.abort(), 10_000)

const result = await llm.infer(prompt, {
  maxTokens: 500,
  onToken: (text) => process.stdout.write(text),
  signal: controller.signal,
})

if (result.aborted) {
  console.log('\nGeneration was cancelled.')
}
```

`AbortSignal.timeout()` also works:

```typescript
signal: AbortSignal.timeout(10_000)
```

---

## 8. Configuration Reference

### Presets

| Preset | nGpuLayers | nCtx | nThreads | Use case |
|--------|-----------|------|----------|----------|
| `small` | 1 | 512 | 4 | Fast tests, low memory |
| `medium` | 99 | 2048 | 4 | General use |
| `large` | 99 | 8192 | 8 | Long contexts |

### Full config options

```typescript
const llm = await LlamaModel.load('./model.gguf', {
  preset: 'medium',       // base preset ('small' | 'medium' | 'large')

  // Override any preset value:
  nGpuLayers: 99,         // GPU layers (99 = all layers on GPU)
  nCtx: 4096,             // Context window in tokens
  nThreads: 4,            // CPU threads for non-GPU layers

  sampler: {
    temp: 0.8,            // Sampling temperature (0 = greedy, 1 = random)
    topK: 50,             // Top-K sampling
    topP: 0.9,            // Top-P (nucleus) sampling
    minP: 0.05,           // Min-P sampling
    seed: 0xFFFFFFFF,     // Random seed (0xFFFFFFFF = random)

    // Repetition penalties
    repeatPenalty: 1.1,   // Penalty for repeated tokens (1.0 = disabled)
    frequencyPenalty: 0.0, // Penalty proportional to token frequency
    presencePenalty: 0.0,  // Flat penalty for any previously seen token
    penaltyLastN: 64,      // How many previous tokens to consider
  },
})
```

---

## 9. Model Metadata

After loading, inspect model info without extra API calls:

```typescript
const llm = await LlamaModel.load('./model.gguf', { preset: 'small' })

console.log(llm.metadata)
// {
//   nParams: 8030000000,  // parameter count
//   nEmbd: 4096,          // embedding dimension
//   nCtxTrain: 32768,     // training context length
//   nLayers: 36,          // transformer layers
//   desc: "Qwen3 8B Q4_K - Medium",
//   sizeBytes: 4685817856 // disk size
// }
```

---

## 10. Concurrent Requests

Multiple `infer()` calls are safe — they are serialized automatically via an internal queue:

```typescript
const llm = await LlamaModel.load('./model.gguf', { preset: 'medium' })

// By default (nSeqMax = 1), these are serialized via an internal queue.
// With nSeqMax > 1, they run truly concurrently — see Section 12.
const [a, b, c] = await Promise.all([
  llm.infer('Question 1', { onToken: () => {}, maxTokens: 50 }),
  llm.infer('Question 2', { onToken: () => {}, maxTokens: 50 }),
  llm.infer('Question 3', { onToken: () => {}, maxTokens: 50 }),
])
```

---

## 11. Embeddings

Load a model in embedding mode to convert text into dense vectors for semantic search and similarity tasks.

```typescript
import { LlamaModel } from '@renatoadorno/bun-llama-cpp'

const embedder = await LlamaModel.load('./models/nomic-embed-text-v1.5.Q4_K_M.gguf', {
  preset: 'small',
  embeddings: true,
  poolingType: 1, // MEAN — required for nomic-embed-text
})

// Single embedding
const vector = await embedder.embed('search_query: What is the capital of France?')
console.log(vector) // Float32Array[768]

// Batch embeddings
const vectors = await embedder.embedMany([
  'search_document: Paris is the capital of France',
  'search_document: London is the capital of the UK',
])
// → Float32Array[]

await embedder.dispose()
```

**Embedding mode is mutually exclusive with generation.** A model loaded with `embeddings: true` cannot call `infer()`, and a generative model cannot call `embed()`. Use `assertCapability()` to validate early:

```typescript
import { assertCapability, CapabilityMismatchError } from '@renatoadorno/bun-llama-cpp'

try {
  assertCapability(embedder, 'embed')   // passes silently ✅
  assertCapability(llm, 'generate')     // passes silently ✅
  assertCapability(llm, 'embed')        // throws CapabilityMismatchError ❌
} catch (e) {
  if (e instanceof CapabilityMismatchError) {
    console.error(e.message)
    // "Model must be loaded with embeddings: true to embed text. Re-load with { embeddings: true, poolingType: 1 }."
  }
}
```

### Recommended embedding models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `nomic-embed-text-v1.5.Q4_K_M.gguf` | 768 | Best for retrieval; use `poolingType: 1` (MEAN) |

### Cosine similarity (user-side utility)

The library returns raw `Float32Array` vectors — similarity computation is your responsibility:

```typescript
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0
  for (let i = 0; i < a.length; i++) {
    dot   += a[i]! * b[i]!
    normA += a[i]! * a[i]!
    normB += b[i]! * b[i]!
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

const [queryVec, ...docVecs] = await embedder.embedMany([query, ...documents])
const scores = documents.map((doc, i) => ({
  doc,
  score: cosineSimilarity(queryVec!, docVecs[i]!),
}))
scores.sort((a, b) => b.score - a.score)
```

---

## 12. Parallel Inference

Load with `nSeqMax > 1` to run multiple sequences concurrently on the same GPU batch. All sequences advance together in each decode step — throughput scales with batch size.

```typescript
const llm = await LlamaModel.load('./models/qwen3-8b-q4_k_m.gguf', {
  preset: 'large',
  nSeqMax: 4, // up to 4 parallel sequences
})
```

### Method 1: `inferParallel()` — synchronized batch

All sequences start together. Best for homogeneous batch workloads:

```typescript
const results = await llm.inferParallel([
  {
    prompt: 'Translate to French: Hello world /no_think',
    onToken: (t) => process.stdout.write(t),
    maxTokens: 50,
  },
  {
    prompt: 'Translate to Spanish: Hello world /no_think',
    onToken: (t) => process.stdout.write(t),
    maxTokens: 50,
  },
])

for (const result of results) {
  console.log(`\n${result.text} (${result.tokenCount} tokens)`)
}
```

### Method 2: `Promise.all` with `infer()` — concurrent queue

When `nSeqMax > 1`, concurrent `infer()` calls are dispatched to the batch engine and run in parallel (not serialized). Best for heterogeneous workloads where requests arrive at different times:

```typescript
const [a, b] = await Promise.all([
  llm.infer('Question A /no_think', { onToken: () => {}, maxTokens: 30 }),
  llm.infer('Question B /no_think', { onToken: () => {}, maxTokens: 30 }),
])
```

### Method 3: Per-sequence abort

Each sequence in `inferParallel()` accepts its own `AbortController`:

```typescript
const controller = new AbortController()

const results = await llm.inferParallel([
  {
    prompt: 'Write a very long essay /no_think',
    onToken: (t) => {
      process.stdout.write(t)
      if (someCondition) controller.abort() // abort this sequence only
    },
    maxTokens: 500,
    signal: controller.signal,
  },
  {
    prompt: 'Short answer /no_think',
    onToken: (t) => process.stdout.write(t),
    maxTokens: 20,
    // no signal — runs to completion regardless
  },
])

console.log(`Seq 0 aborted: ${results[0]!.aborted}`)
```

### Method 4: Warmup + prefix sharing (~8× speedup)

Pre-compute the KV cache for a shared system prompt so all parallel sequences skip the prefill step:

```typescript
// Build the system prompt prefix using the model's chat template
const systemPrompt = await llm.applyTemplate([
  { role: 'system', content: 'You are a math tutor. Answer concisely. /no_think' },
], { addAssistant: false })

// Pre-compute KV cache for the shared prefix (runs once)
const warmupTokens = await llm.warmup(systemPrompt)
console.log(`Cached ${warmupTokens} tokens`)

// All sequences reuse the cached prefix — only user turns need prefill
const results = await llm.inferParallel([
  {
    prompt: systemPrompt + '\nUser: What is 2 + 2?\nAssistant:',
    onToken: (t) => process.stdout.write(t),
    maxTokens: 30,
    metrics: true,
  },
  {
    prompt: systemPrompt + '\nUser: What is 10 × 10?\nAssistant:',
    onToken: (t) => process.stdout.write(t),
    maxTokens: 30,
    metrics: true,
  },
])

for (const result of results) {
  if (result.metrics) {
    console.log(`${result.metrics.generatedTokens} tokens at ${result.metrics.tokensPerSec.toFixed(1)} tok/s`)
  }
}

await llm.dispose()
```

> **Note:** Call `warmup()` once after load, before the first `inferParallel()`. `nSeqMax` must be `> 1` for `warmup()` to work.

---

## 13. Multi-Model Orchestration

Use `ModelRegistry` to manage multiple models by name, and `ModelPipeline` to orchestrate embed → rerank → generate workflows.

### ModelRegistry — load and manage models by name

```typescript
import { ModelRegistry, ModelNotFoundError } from '@renatoadorno/bun-llama-cpp'

const registry = new ModelRegistry()

await registry.load('embed', './models/nomic-embed-text-v1.5.Q4_K_M.gguf', {
  preset: 'small',
  embeddings: true,
  poolingType: 1,
})
await registry.load('gen', './models/qwen3-8b-q4_k_m.gguf', { preset: 'medium' })

// Check status
console.log(registry.status('embed'))        // 'ready'
console.log(registry.status('not-loaded'))   // 'unknown'

// Get a ready model
const embedder = registry.get('embed')

// Unload a specific model
await registry.unload('embed')

// Dispose everything (in reverse load order)
await registry.disposeAll()
```

**`load()` is idempotent and deduplicates concurrent calls** — calling it twice for the same name (even concurrently) creates only one Worker.

**Error recovery:**

```typescript
try {
  await registry.load('bad', '/nonexistent/model.gguf')
} catch (e) {
  console.log(registry.status('bad')) // 'error'
}

// get() on a failed model throws with context
try {
  registry.get('bad')
} catch (e) {
  if (e instanceof ModelNotFoundError) {
    console.log(e.code)    // 'MODEL_NOT_FOUND'
    console.log(e.message) // "Model 'bad' failed to load: ... Call registry.load('bad', path, config) again to retry."
  }
}

// Retry by calling load() again with the correct path
await registry.load('bad', './models/qwen3-8b-q4_k_m.gguf', { preset: 'small' })
```

### ModelPipeline — embed → rerank → generate

`ModelPipeline` orchestrates multi-step RAG. The caller retrieves candidates from their own vector store and passes them in:

```typescript
import { ModelPipeline } from '@renatoadorno/bun-llama-cpp'

const pipeline = new ModelPipeline(
  registry.get('embed'), // must have embeddings: true
  registry.get('gen'),   // must be a generative model
)

// 1. Embed a query (returns Float32Array)
const queryVec = await pipeline.embed('search_query: What is the tallest mountain?')

// 2. Rerank candidates from your vector store
const candidates = [
  'search_document: Mount Everest stands at 8,848 meters above sea level.',
  'search_document: The Amazon River is the largest by discharge volume.',
  'search_document: K2 is the second-tallest mountain at 8,611 meters.',
]
const ranked = await pipeline.rerank(
  'search_query: What is the tallest mountain?',
  candidates,
)
// → [{ doc: 'Mount Everest...', score: 0.94 }, { doc: 'K2...', score: 0.87 }, ...]

// 3. Generate answer from top context
const context = ranked.slice(0, 2).map(r => r.doc).join('\n')
const result = await pipeline.generate(context, 'What is the tallest mountain?', {
  maxTokens: 80,
  onToken: (t) => process.stdout.write(t),
})
// → { text: 'Mount Everest, at 8,848 meters...', tokenCount: 22, aborted: false }

process.stdout.write('\n')
```

**Constructor validates model types immediately** — passing a generative model as the embed argument throws `CapabilityMismatchError` at construction time, not at call time:

```typescript
new ModelPipeline(genModel, genModel)    // ❌ throws CapabilityMismatchError
new ModelPipeline(embedModel, genModel)  // ✅
```

### Typed errors

All `bun-llama-cpp` errors extend `LlamaCppError` and carry a `code` string for programmatic handling:

```typescript
import { LlamaCppError, CapabilityMismatchError, ModelNotFoundError } from '@renatoadorno/bun-llama-cpp'

try {
  registry.get('not-loaded')
} catch (e) {
  if (e instanceof LlamaCppError) {
    console.log(e.code)    // 'MODEL_NOT_FOUND' | 'CAPABILITY_MISMATCH'
    console.log(e.message) // actionable message
  }
}
```

| Code | Thrown by | Cause |
|------|-----------|-------|
| `MODEL_NOT_FOUND` | `registry.get()` | Model not loaded, still loading, or failed |
| `CAPABILITY_MISMATCH` | `assertCapability()`, `ModelPipeline` constructor | Wrong model type for the operation |

---

## Troubleshooting

### `Library not loaded: @rpath/libggml.0.dylib`

The `.dylib` was built for a different path. This happens when running scripts directly instead of through `bun run`. Set the library path manually:

```bash
DYLD_LIBRARY_PATH=node_modules/@renatoadorno/bun-llama-cpp-darwin-arm64 bun run index.ts
```

### `Failed to resolve package @renatoadorno/bun-llama-cpp`

Your `.npmrc` is not configured or the token is wrong. Check:

```bash
cat .npmrc
# should show:
# @renatoadorno:registry=https://npm.pkg.github.com
# //npm.pkg.github.com/:_authToken=<your-token>
```

### Model loads but output is garbled

You may be passing a raw prompt to a chat-tuned model. Use `applyTemplate()` to format the prompt correctly for the model's expected input format.

### `applyTemplate` throws "Chat template not supported"

The model doesn't have a built-in template (uncommon for modern models). Use a manual prompt format for that model's expected syntax, or switch to a model with a built-in chat template.
