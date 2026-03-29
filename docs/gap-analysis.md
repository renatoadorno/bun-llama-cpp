# bun-llama-cpp — Gap Analysis

> Comprehensive mapping of current capabilities vs. llama.cpp API surface.
> Generated from source code inspection of all `src/` files and `llama.cpp/include/llama.h`.

---

## 1. Current State Summary

### What the project does today

| Layer | File(s) | Responsibility |
|-------|---------|----------------|
| Public API | `index.ts`, `model.ts` | `LlamaModel.load()` → `infer()` → `dispose()` |
| Types | `types.ts` | `ModelConfig`, `SamplerConfig`, `InferOptions`, `InferResult`, worker protocol |
| Presets | `presets.ts` | small / medium / large with `resolveConfig()` merge |
| Queue | `queue.ts` | Promise-chain serial queue (one inference at a time) |
| Worker router | `worker/llm.worker.ts` | Message dispatch: init / infer / shutdown; stderr muting |
| FFI bindings | `worker/ffi.ts` | `dlopen` wrappers for libllama (41 fn) + libshims (25 fn) |
| Tokenizer | `worker/tokenizer.ts` | `tokenize()`, `tokenPiece()`, EOG detection |
| Inference | `worker/inference.ts` | `initModel()`, `runInference()`, `cleanup()` |
| C shim | `llama_shims.c` | Struct-by-value wrappers for bun:ffi compatibility |

### Bound FFI functions (66 total)

**From libllama (41):**

| Category | Functions |
|----------|-----------|
| Backend | `llama_backend_init`, `llama_backend_free` |
| Model lifecycle | `llama_model_free`, `llama_free` |
| Logging | `llama_log_set` |
| Vocab | `llama_model_get_vocab`, `llama_vocab_bos`, `llama_vocab_eos`, `llama_vocab_eot`, `llama_vocab_is_eog` |
| Tokenization | `llama_tokenize`, `llama_token_to_piece` |
| Context | `llama_n_ctx`, `llama_get_memory`, `llama_memory_clear` |
| Sampler chain | `llama_sampler_chain_add`, `llama_sampler_init_greedy`, `llama_sampler_init_top_k`, `llama_sampler_init_temp`, `llama_sampler_init_dist`, `llama_sampler_sample`, `llama_sampler_accept`, `llama_sampler_reset`, `llama_sampler_free` |

**From libshims (25):**

| Category | Functions |
|----------|-----------|
| Size queries | `shim_sizeof_model_params`, `shim_sizeof_context_params`, `shim_sizeof_batch`, `shim_sizeof_sampler_chain_params` |
| Default params | `shim_model_default_params`, `shim_context_default_params`, `shim_sampler_chain_default_params` |
| Param setters | `shim_model_params_set_n_gpu_layers`, `shim_ctx_params_set_n_ctx`, `shim_ctx_params_set_n_threads` |
| Model/ctx | `shim_model_load_from_file`, `shim_init_from_model` |
| Batch | `shim_batch_init`, `shim_batch_clear`, `shim_batch_add`, `shim_batch_free`, `shim_decode` |
| Sampler | `shim_sampler_chain_init`, `shim_sampler_init_top_p`, `shim_sampler_init_min_p` |

### Sampler options currently available

| Sampler | Configurable | Config field |
|---------|-------------|--------------|
| top-k | ✅ | `sampler.topK` |
| top-p (nucleus) | ✅ | `sampler.topP` |
| min-p | ✅ | `sampler.minP` |
| Temperature | ✅ | `sampler.temp` |
| Distribution (seed) | ✅ | `sampler.seed` |
| Greedy | ✅ (via temp=0 fallback) | — |

Chain order: top-p → min-p → top-k → temp → dist

### Threading model

```
Main Thread                    Worker Thread
┌─────────────┐               ┌──────────────────┐
│ LlamaModel  │  postMessage  │ llm.worker.ts    │
│ ┌─────────┐ │ ────────────► │ ┌──────────────┐ │
│ │SerialQ  │ │               │ │ FFI calls    │ │
│ │(Promise)│ │ SharedArray   │ │ (synchronous │ │
│ │         │ │ Buffer abort  │ │  blocking)   │ │
│ └─────────┘ │ ◄───────────► │ └──────────────┘ │
└─────────────┘               └──────────────────┘
```

- **One worker, one model, one context** — all serial
- Abort via `SharedArrayBuffer` + `Atomics` (bypasses blocked event loop)
- Worker keeps its own global state (`libs`, `state`)

---

## 2. llama.cpp API Coverage

### Total API surface: ~180+ functions in `llama.h`

### Functions bound: **41 from libllama + 25 shim wrappers = 66**

### Coverage by category:

| Category | Available in llama.h | Bound | Coverage |
|----------|---------------------|-------|----------|
| Backend init/free | 2 | 2 | 100% |
| Model load/free | 6 | 2 (via shim) | 33% |
| Context create/free | 2 | 2 (via shim) | 100% |
| Vocab / special tokens | 15+ | 5 | ~33% |
| Tokenization | 3 | 2 | 67% |
| Batch / decode | 5 | 5 (via shim) | 100% |
| KV cache / memory mgmt | 9 | 1 (`memory_clear`) | 11% |
| Embeddings | 4 | 0 | **0%** |
| Samplers | 20+ init functions | 7 | ~35% |
| Grammar | 3 | 0 | **0%** |
| Model metadata | 20+ | 0 | **0%** |
| Context settings | 5 | 0 | **0%** |
| LoRA adapters | 5 | 0 | **0%** |
| State save/load | 9 | 0 | **0%** |
| Chat templates | 2 | 0 | **0%** |
| Performance info | 4 | 0 | **0%** |
| Encode (embeddings) | 1 | 0 | **0%** |
| Thread pool | 2 | 0 | **0%** |

---

## 3. Gap Map

### 3.1 Feature Gap Table

| # | Feature | Current State | What's Missing | Difficulty | Impact |
|---|---------|--------------|----------------|------------|--------|
| 1 | **Embeddings** | Not supported | `llama_set_embeddings`, `llama_encode`, `llama_get_embeddings_ith/seq`. Need `embeddings: true` on context params, `llama_encode()` instead of `llama_decode()`, float pointer read-back. New shims: `shim_ctx_params_set_embeddings`, `shim_encode`, `shim_get_embeddings_ith`. | Medium | 🔴 High — required for RAG |
| 2 | **Grammar / constrained generation** | Not supported | `llama_sampler_init_grammar(vocab, grammar_str, grammar_root)`. Needs vocab pointer (already available) and GBNF grammar string from user. New shim: `shim_sampler_init_grammar`. Sampler chain must insert grammar sampler before distribution sampler. | Medium | 🔴 High — JSON mode, structured output |
| 3 | **Parallel sequences (same model)** | Single sequence only | `n_seq_max > 1` on context params, `llama_memory_seq_cp/rm/keep`, batch with multiple `seq_id`. Requires prompt-sharing via KV cache copy. SerialQueue must be replaced with concurrent dispatch per sequence slot. | Hard | 🔴 High — throughput |
| 4 | **Multiple models** | Single model per process | Each model needs its own Worker. `LlamaModel` already encapsulates worker, so multiple instances *should* work — but needs testing for Metal/GPU memory contention and `llama_backend_init` idempotency across workers. | Medium | 🟡 Medium |
| 5 | **KV cache management** | Only `memory_clear` | `llama_memory_seq_rm(mem, seq_id, p0, p1)` — trim prefix/suffix. `llama_memory_seq_add` — shift positions. `llama_memory_seq_cp` — copy for parallel sequences. `llama_memory_can_shift` — check if model supports context shifting. | Medium | 🟡 Medium — context window management |
| 6 | **Context window shift/extend** | Not supported | When prompt + generation exceed `n_ctx`: detect overflow, call `llama_memory_seq_rm` to trim oldest tokens, `llama_memory_seq_add` to re-index positions. Common pattern for "infinite" generation. | Medium | 🟡 Medium |
| 7 | **Repetition / presence penalties** | Not supported | `llama_sampler_init_penalties(last_n, repeat, freq, present)`. Pure addition to sampler chain. New shim: `shim_sampler_init_penalties`. | Easy | 🟡 Medium |
| 8 | **DRY sampler** | Not supported | `llama_sampler_init_dry(vocab, n_ctx_train, multiplier, base, allowed_length, penalty_last_n, seq_breakers, num_breakers)`. Needs string array passing via FFI. | Medium | 🟢 Low |
| 9 | **Mirostat sampling** | Not supported | `llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)` and `llama_sampler_init_mirostat_v2(seed, tau, eta)`. Simple sampler additions. | Easy | 🟢 Low |
| 10 | **Model metadata / introspection** | Not supported | `llama_model_n_params`, `llama_model_n_embd`, `llama_model_n_ctx_train`, `llama_model_n_layer`, `llama_model_desc`, `llama_model_size`, `llama_model_meta_*`. All return scalars — direct `dlopen` bindings, no shims needed. | Easy | 🟡 Medium — model selection, validation |
| 11 | **Chat template application** | Not supported | `llama_chat_apply_template(tmpl, messages, n_msg, add_ass, buf, len)`. Needs `llama_chat_message` struct handling (two pointers: role + content). New shim for struct. | Medium | 🔴 High — chat/conversation API |
| 12 | **LoRA adapters** | Not supported | `llama_adapter_lora_init(model, path)`, `llama_set_adapters_lora(ctx, adapters, n, scales)`. Pointer array management. | Medium | 🟡 Medium |
| 13 | **Reranking / scoring** | Not supported | Reranking uses embedding model + `llama_set_causal_attn(ctx, false)` + `llama_encode()` + read `llama_get_embeddings_seq`. Depends on embedding support (gap #1). | Hard | 🟡 Medium — RAG ranking |
| 14 | **Speculative decoding** | Not supported | No dedicated API in llama.h — implemented as a pattern: draft model generates candidates, main model verifies in batch. Requires multi-model + batch decode. | Very Hard | 🟢 Low (niche) |
| 15 | **State save/load** | Not supported | `llama_state_seq_save_file`, `llama_state_seq_load_file`. Persist conversation KV cache to disk. Buffer + file path management. | Medium | 🟢 Low |
| 16 | **Batch multi-sequence decode** | Single token decode only | Current `shim_batch_add` supports `seq_id` param but always uses seq 0. Batch can hold tokens for multiple sequences and decode together for throughput. Requires sequence management layer. | Hard | 🔴 High — server throughput |
| 17 | **Detokenize** | Token-by-token only | `llama_detokenize(vocab, tokens, n_tokens, text, text_len, remove_special, unparse_special)` — batch detokenization. Minor quality-of-life improvement. | Easy | 🟢 Low |
| 18 | **FIM (Fill-in-Middle)** | Not supported | Special FIM tokens: `llama_vocab_fim_pre/suf/mid/pad/rep/sep`. Already in vocab API, just not bound. Used for code completion. | Easy | 🟡 Medium — code use cases |
| 19 | **Quantization** | Not supported | `llama_model_quantize(fname_inp, fname_out, params)`. Large struct (`llama_model_quantize_params`). Offline tool, not real-time. | Medium | 🟢 Low |
| 20 | **Performance metrics** | Not supported | `llama_perf_context`, `llama_perf_sampler`, `llama_memory_breakdown_print`. Useful for benchmarking. | Easy | 🟢 Low |

### 3.2 Priority Tiers

**Tier 1 — Core missing features (high impact, enables new use cases):**
1. Embeddings (#1) — unlocks RAG pipelines
2. Grammar / constrained generation (#2) — unlocks JSON/structured output
3. Chat template application (#11) — unlocks proper conversation API
4. Repetition penalties (#7) — basic quality improvement

**Tier 2 — Throughput & multi-use (medium difficulty, multiplier effect):**
5. Parallel sequences (#3) — throughput multiplier
6. KV cache management (#5) — context window control
7. Context shift/extend (#6) — long generation support
8. Model metadata (#10) — introspection & validation
9. Batch multi-sequence (#16) — server-grade throughput

**Tier 3 — Extended capabilities:**
10. Multiple models (#4) — test & document
11. LoRA adapters (#12) — fine-tune switching
12. Reranking (#13) — RAG quality
13. FIM tokens (#18) — code completion

**Tier 4 — Nice-to-have:**
14. DRY sampler (#8)
15. Mirostat (#9)
16. State save/load (#15)
17. Speculative decoding (#14)
18. Quantization (#19)
19. Performance metrics (#20)
20. Batch detokenize (#17)

---

## 4. Architecture Constraints & Recommendations

### 4.1 Single-worker model limitations

| Constraint | Impact | Resolution |
|------------|--------|------------|
| One worker = one model = one context | No parallel inference | For parallel sequences: keep single worker, add sequence management inside it. For multiple models: spawn multiple workers (already possible). |
| SerialQueue blocks concurrent requests | Throughput ceiling | Replace with sequence-aware dispatcher that maps request → seq_id, then batch-decodes multiple sequences per step. |
| Worker event loop blocked during FFI | Can't receive messages during decode | Already mitigated via SharedArrayBuffer abort. For sequence management, the batch decode loop must interleave sequence management between decode steps. |
| No shared memory between workers | Can't share KV cache or model weights across workers | llama.cpp internally memory-maps model weights, so multiple workers loading the same model file will share physical RAM via OS mmap. KV caches are always per-context. |

### 4.2 C shim changes needed

| Feature | New shims required |
|---------|-------------------|
| Embeddings | `shim_ctx_params_set_embeddings(buf, bool)`, `shim_ctx_params_set_pooling_type(buf, int)`, `shim_encode(ctx, batch_buf)`, `shim_get_embeddings_ith(ctx, i)` → returns `ptr` |
| Grammar | `shim_sampler_init_grammar(vocab, str, root)` — wraps struct return |
| Penalties | `shim_sampler_init_penalties(last_n, repeat, freq, present)` — wraps struct return |
| Context params | `shim_ctx_params_set_n_batch(buf, n)`, `shim_ctx_params_set_n_seq_max(buf, n)`, `shim_ctx_params_set_flash_attn(buf, type)` |
| KV cache ops | `shim_memory_seq_rm(mem, seq, p0, p1)`, `shim_memory_seq_cp(...)`, `shim_memory_seq_add(...)`, `shim_memory_seq_keep(...)` — all take `llama_memory_t` which is an opaque pointer |
| Chat template | `shim_chat_apply_template(tmpl, roles[], contents[], n, add_ass, buf, len)` — needs to build `llama_chat_message` array |
| Causal attn | `shim_set_causal_attn(ctx, bool)` — simple wrapper |
| Set embeddings | `shim_set_embeddings(ctx, bool)` — simple wrapper |

### 4.3 FFI patterns that need to change

| Current pattern | Limitation | New pattern needed |
|-----------------|-----------|-------------------|
| All pointers as `number` | Works, but no typing | Keep — bun:ffi constraint |
| Struct fields via individual setter shims | Verbose, one shim per field | Acceptable for now; consider a bulk setter `shim_ctx_params_set_all(buf, json)` if field count grows |
| Single batch buffer in `LlamaState` | Can't batch multiple sequences | Batch buffer is already seq-aware (`shim_batch_add` takes `seq_id`). Just need to call it with different seq_ids. |
| Float pointer return for embeddings | Not yet handled | Use `ptr` return type + `new Float32Array(buffer, ptr, n_embd)` to read embedding vectors. Bun supports `toArrayBuffer(ptr, offset, size)` for this. |
| String arrays (for DRY breakers) | Not yet handled | Allocate array of C strings as contiguous buffer, pass pointer + count |

### 4.4 Can Bun Workers share memory/pointers?

| Mechanism | Supported | Notes |
|-----------|-----------|-------|
| `SharedArrayBuffer` | ✅ Yes | Already used for abort signaling. Can share typed arrays. |
| FFI pointers across workers | ❌ No | Pointers from one worker's `dlopen` are not valid in another worker. Each worker must `dlopen` independently. |
| Model weight sharing | ✅ Via OS | Multiple workers loading the same GGUF file share physical pages via mmap (handled by OS, not Bun). |
| `postMessage` with Transferable | ✅ Yes | `ArrayBuffer` can be transferred (zero-copy move) between threads, but pointer values inside are not portable. |

---

## 5. Implementation Roadmap Sketch

### Phase 1: Low-hanging fruit (Easy, 1–2 days each)

```
┌─ Repetition penalties ──── new shim + sampler chain slot
├─ Model metadata ────────── direct dlopen bindings, no shims
├─ FIM tokens ────────────── bind llama_vocab_fim_* (already in vocab API)
└─ Performance metrics ───── bind llama_perf_context/sampler
```

### Phase 2: Core features (Medium, 3–5 days each)

```
┌─ Embeddings ────────────── new context param setter + encode path + float read
├─ Grammar ───────────────── new shim + sampler integration + GBNF string passing
├─ Chat templates ────────── new shim wrapping llama_chat_message array
└─ KV cache management ──── bind memory_seq_* functions
```

### Phase 3: Architecture changes (Hard, 1–2 weeks)

```
┌─ Parallel sequences ───── sequence manager, batch dispatcher, concurrent API
├─ Context shifting ──────── automatic overflow detection + trim/shift
└─ Multi-model testing ──── document Metal memory behavior, test concurrent load
```

### Phase 4: Advanced (Hard, exploratory)

```
┌─ Reranking ─────────────── depends on embeddings + causal_attn toggle
├─ Speculative decoding ──── multi-model orchestration pattern
└─ State persistence ─────── save/load KV cache to disk
```

---

## 6. Key Observations

1. **The project has solid foundations** — the two-library FFI pattern, worker isolation, serial queue, and abort mechanism are well-engineered. The shim approach correctly solves bun:ffi's struct limitations.

2. **Embedding support is the biggest gap** — it's the #1 unlock for RAG, reranking, and semantic search. The plumbing is close: `llama_encode` is just `llama_decode` without KV cache, and `llama_get_embeddings_ith` returns a float pointer that can be read with `toArrayBuffer`.

3. **Grammar/constrained output is critical** — JSON mode is table-stakes for agent/tool-use patterns. The implementation is straightforward: one new sampler in the chain.

4. **Parallel sequences don't require multiple workers** — llama.cpp supports `n_seq_max > 1` on a single context. The batch already supports `seq_id`. The challenge is the control plane: mapping requests to sequence slots, sharing prompt prefixes via `memory_seq_cp`, and scheduling decode steps across active sequences.

5. **Multiple models already work in theory** — each `LlamaModel.load()` spawns its own worker with its own `dlopen`. The risk is GPU memory contention on Metal, which needs empirical testing.

6. **Context shifting is easy once KV cache ops are bound** — the pattern is: detect when `pos >= n_ctx`, call `memory_seq_rm` to drop the oldest N tokens, call `memory_seq_add` with delta `-N` to shift positions down.

7. **~114 llama.h functions are unbound** — but many are niche (quantization, state serialization, LoRA). The 20–30 functions in Tier 1–2 would cover 90% of practical use cases.
