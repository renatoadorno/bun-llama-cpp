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

// These run sequentially, not in parallel (llama.cpp is single-context)
const [a, b, c] = await Promise.all([
  llm.infer('Question 1', { onToken: () => {}, maxTokens: 50 }),
  llm.infer('Question 2', { onToken: () => {}, maxTokens: 50 }),
  llm.infer('Question 3', { onToken: () => {}, maxTokens: 50 }),
])
```

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
