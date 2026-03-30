# Phase 1: Foundations — Design Spec

## Context

bun-llama-cpp currently covers ~37% of the llama.cpp API surface. Phase 1 adds 5 independent "quick win" features that fill critical gaps: repetition penalties (the #1 user complaint for local LLMs), model metadata (required by Phase 4 multi-model), chat templates (table-stakes for chat APIs), FIM tokens (enables IDE/code-completion use cases), and performance metrics (debugging/benchmarking). All features are non-invasive — they extend without changing existing architecture.

## Delivery

- **5 separate PRs**, one per feature, in complexity-ascending order
- Each PR is a vertical slice: C shim (if needed) → FFI bindings → worker handler → model facade → types → tests

## Implementation Order

1. **FIM Tokens** — simplest (6 direct FFI bindings)
2. **Model Metadata** — direct FFI + buffer handling for `llama_model_desc`
3. **Performance Metrics** — needs C shims (struct-by-value returns)
4. **Repetition Penalties** — direct FFI, sampler chain modification
5. **Chat Templates** — most complex shim (array of structs construction)

---

## PR 1: FIM Tokens

### FFI Bindings (`src/worker/ffi.ts`)

6 direct bindings to `libllama`, all `(vocab: pointer) → i32`:

- `llama_vocab_fim_pre`, `llama_vocab_fim_suf`, `llama_vocab_fim_mid`
- `llama_vocab_fim_pad`, `llama_vocab_fim_rep`, `llama_vocab_fim_sep`

### Types (`src/types.ts`)

```ts
export interface FimTokens {
  pre: number;  // fill-in-middle prefix token
  suf: number;  // suffix token
  mid: number;  // middle token
  pad: number;  // padding token
  rep: number;  // repository token
  sep: number;  // separator token
}
```

Worker protocol additions:
```ts
| { type: 'getFimTokens' }                     // request
| { type: 'fimTokens'; data: FimTokens }       // response
```

### Model API (`src/model.ts`)

```ts
async getFimTokens(): Promise<FimTokens>
```

Returns token IDs. Value `-1` indicates token not supported by the model.

### Exports (`src/index.ts`)

Export `FimTokens` type.

### Test (`tests/fim.test.ts`)

- Returns valid token IDs (each ≥ -1)
- Known FIM-capable models return positive IDs for pre/suf/mid

---

## PR 2: Model Metadata

### FFI Bindings (`src/worker/ffi.ts`)

6 direct bindings to `libllama`:

```
llama_model_n_params(model)    → u64
llama_model_n_embd(model)      → i32
llama_model_n_ctx_train(model)  → i32
llama_model_n_layer(model)     → i32
llama_model_size(model)        → u64
llama_model_desc(model, buf, buf_size) → i32  (writes to buffer)
```

### Types (`src/types.ts`)

```ts
export interface ModelMetadata {
  nParams: number;    // total parameters
  nEmbd: number;      // embedding dimension
  nCtxTrain: number;  // training context window
  nLayers: number;    // number of layers
  desc: string;       // model description (e.g. "Qwen3 8B Q4_K_M")
  sizeBytes: number;  // total tensor size in bytes
}
```

Worker protocol: metadata piggybacked on `'ready'` response:
```ts
{ type: 'ready'; metadata: ModelMetadata }
```

### Worker (`src/worker/inference.ts` + `llm.worker.ts`)

After `initModel()`, collect metadata from model pointer and include in `'ready'` message.

### Model API (`src/model.ts`)

```ts
readonly metadata: ModelMetadata  // populated during load(), synchronous access
```

### Exports (`src/index.ts`)

Export `ModelMetadata` type.

### Test (`tests/metadata.test.ts`)

- `nParams > 0`, `nEmbd > 0`, `nCtxTrain > 0`, `nLayers > 0`
- `desc` is non-empty string
- `sizeBytes > 0`

---

## PR 3: Performance Metrics

### C Shims (`src/llama_shims.c`)

`llama_perf_context()` and `llama_perf_sampler()` return structs by value — bun:ffi cannot handle this. Shims write to caller-provided buffers:

```c
size_t shim_sizeof_perf_context_data(void);
size_t shim_sizeof_perf_sampler_data(void);
void   shim_perf_context(const llama_context *ctx, void *out);
void   shim_perf_sampler(const llama_sampler *chain, void *out);
```

### FFI Bindings (`src/worker/ffi.ts`)

Shim bindings + direct bindings for reset:
```
shim_sizeof_perf_context_data()        → u64
shim_sizeof_perf_sampler_data()        → u64
shim_perf_context(ctx, buf)            → void
shim_perf_sampler(chain, buf)          → void
llama_perf_context_reset(ctx)          → void
```

### Types (`src/types.ts`)

```ts
export interface InferMetrics {
  promptTokens: number;
  generatedTokens: number;
  promptMs: number;
  generateMs: number;
  tokensPerSec: number;  // calculated: generatedTokens / (generateMs / 1000)
}
```

`InferOptions` gains: `metrics?: boolean`
`InferResult.metrics` is `InferMetrics | undefined`

Worker protocol: `metrics` flag forwarded in `'infer'` request, metrics included in `'done'` response when requested.

### Worker (`src/worker/inference.ts`)

After generation loop completes (before cleanup):
1. Call `shim_perf_context()` to read context perf data
2. Call `shim_perf_sampler()` to read sampler perf data
3. Parse struct fields from buffer (offsets based on C struct layout)
4. Call `llama_perf_context_reset()` for next inference

### Model API (`src/model.ts`)

```ts
const result = await llm.infer(prompt, { metrics: true })
// result.metrics: InferMetrics | undefined
```

### Test (extends `tests/inference.test.ts`)

- With `metrics: true`: result.metrics exists, all fields > 0, tokensPerSec > 0
- Without `metrics`: result.metrics is undefined

---

## PR 4: Repetition Penalties

### FFI Bindings (`src/worker/ffi.ts`)

Direct binding — returns pointer:
```
llama_sampler_init_penalties(last_n: i32, repeat: f32, freq: f32, present: f32) → pointer
```

### Types (`src/types.ts`)

`SamplerConfig` gains optional fields:
```ts
repeatPenalty?: number;     // default: 1.1 (1.0 = disabled)
frequencyPenalty?: number;  // default: 0.0 (disabled)
presencePenalty?: number;   // default: 0.0 (disabled)
penaltyLastN?: number;      // default: 64 (0 = disable, -1 = ctx size)
```

### Worker (`src/worker/inference.ts`)

Modify sampler chain construction in `initModel()`. Insert penalties sampler **after top-k** (llama.h recommends applying top-k/top-p first for performance):

```
top-p → min-p → top-k → penalties → temp → dist
```

Only add penalties sampler if any penalty is non-default (repeatPenalty !== 1.0 or freq/present !== 0.0).

### Model API (`src/model.ts`)

No new methods — penalties flow through existing `SamplerConfig` in `ModelConfig`:

```ts
const llm = await LlamaModel.load('./model.gguf', {
  sampler: { repeatPenalty: 1.2, penaltyLastN: 128 }
})
```

### Presets (`src/presets.ts`)

Add default penalty values to base sampler config.

### Test (extends `tests/inference.test.ts`)

- Output with penalties enabled shows less repetition than without
- Config values are respected (different penalties = different outputs)

---

## PR 5: Chat Templates

### FFI Bindings (`src/worker/ffi.ts`)

Direct binding for template extraction:
```
llama_model_chat_template(model, name) → pointer  (returns const char*)
```

### C Shim (`src/llama_shims.c`)

Accepts parallel string arrays (easier for FFI than array of structs):

```c
int32_t shim_chat_apply_template(
    const char *tmpl,       // template string (NULL = model default)
    const char **roles,     // array of role C strings
    const char **contents,  // array of content C strings
    size_t n_msg,           // number of messages
    bool add_ass,           // add assistant turn prefix
    char *buf,              // output buffer
    int32_t length          // buffer size
);
```

Internally constructs `llama_chat_message[]` from parallel arrays, calls `llama_chat_apply_template()`.

### Types (`src/types.ts`)

```ts
export interface ChatMessage {
  role: string;
  content: string;
}
```

Worker protocol:
```ts
| { type: 'applyTemplate'; id: string; messages: ChatMessage[]; options?: { addAssistant?: boolean } }
| { type: 'templateResult'; id: string; text: string }
```

### Model API (`src/model.ts`)

```ts
async applyTemplate(
  messages: ChatMessage[],
  options?: { addAssistant?: boolean }  // default: true
): Promise<string>
```

Uses model's built-in template (extracted via `llama_model_chat_template`). Falls back to ChatML if model has no template.

### Exports (`src/index.ts`)

Export `ChatMessage` type.

### Test (`tests/chat-template.test.ts`)

- Single user message formats correctly
- Multi-turn (system + user + assistant + user) formats correctly
- `addAssistant: false` omits assistant prefix
- Known model (Qwen3) produces expected template markers

---

## Test Structure

```
tests/
  fim.test.ts              ← PR1
  metadata.test.ts         ← PR2
  inference.test.ts        ← PR3 + PR4 (metrics, penalties)
  chat-template.test.ts    ← PR5
```

Existing `src/llm.worker.test.ts` moves to `tests/` in PR1 (or stays and new tests go alongside in `tests/`).

All tests share a single model load in `beforeAll` (load is ~5s). Tests use 60-180s timeouts due to GPU inference.

## Verification

For each PR:
1. `bun run build:shims` (if shims changed)
2. `bun run test` — all tests pass
3. `bun run start` — demo still works
4. Manual verification of the specific feature via a test script or updated demo

---

## Files Modified (Summary)

| File | PRs |
|------|-----|
| `src/types.ts` | 1, 2, 3, 4, 5 |
| `src/worker/ffi.ts` | 1, 2, 3, 4, 5 |
| `src/worker/llm.worker.ts` | 1, 2, 3, 5 |
| `src/model.ts` | 1, 2, 3, 4, 5 |
| `src/index.ts` | 1, 2, 5 |
| `src/worker/inference.ts` | 2, 3, 4 |
| `src/presets.ts` | 4 |
| `src/llama_shims.c` | 3, 5 |
