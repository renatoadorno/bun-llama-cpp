# Copilot Instructions — bun-llama-cpp

## Runtime & Tooling

Use **Bun** exclusively — never Node.js, npm, yarn, or npx.

- `bun install` / `bun run <script>` / `bunx <pkg>`
- Bun auto-loads `.env` — don't use dotenv
- Prefer `Bun.file` over `node:fs`, `Bun.serve()` over express, `bun:sqlite` over better-sqlite3

## Commands

```bash
# Build the C shim (required after llama.cpp changes)
bun run build:shims

# Full platform build (llama.cpp + shims + copy binaries)
./scripts/build-platform.sh

# Run example demo
bun run start

# Run with hot-reload
bun run dev

# Run tests
bun run test

# Run a single test file
DYLD_LIBRARY_PATH=llama.cpp/build/bin bun test tests/inference.test.ts

# Run tests and see results (bun suppresses output without a TTY)
DYLD_LIBRARY_PATH=llama.cpp/build/bin bun test tests/foo.test.ts --reporter=junit --reporter-outfile=/tmp/out.xml
```

All runtime commands require `DYLD_LIBRARY_PATH=llama.cpp/build/bin` (already set in package.json scripts).

## Project Vision

High-performance FFI bindings from Bun to llama.cpp — the most performant JS/TS library for local LLM inference. Key differentiators vs node-llama-cpp:

- **FFI ~10ns** vs N-API ~50ns (5× faster call overhead)
- **Worker isolation** — main thread never blocks
- **SharedArrayBuffer abort** — works even during blocked FFI calls
- **Float32 native** — half the memory vs Float64
- **Zero build step** — dlopen universal, no cmake-js per platform

## Architecture

This is a **reusable library** providing low-level FFI bindings from Bun to `libllama.dylib` (llama.cpp). The public API is a `LlamaModel` class that hides all FFI and worker internals.

### Module Structure

- `src/index.ts` — Public entry point. Exports `LlamaModel`, types, and presets.
- `src/model.ts` — `LlamaModel` facade class. Manages worker lifecycle, serial queue (thread-safety), and AbortSignal cancellation.
- `src/types.ts` — All public types (`ModelConfig`, `InferOptions`, `InferResult`, `Preset`) and internal worker protocol types.
- `src/presets.ts` — Preset configs (small/medium/large) with `resolveConfig()` merge logic.
- `src/queue.ts` — Promise-chain serial queue ensuring one inference at a time.
- `src/lib-resolver.ts` — Runtime library path resolution. Priority: platform package (npm) → local dev (llama.cpp/build/bin/) → error.
- `src/worker/` — Everything that runs inside the Bun Worker (never imported by consumers):
  - `llm.worker.ts` — Thin message router delegating to modules below
  - `ffi.ts` — `dlopen` bindings for libllama + shims, typed interfaces
  - `tokenizer.ts` — `tokenize()`, `tokenPiece()`, EOT detection
  - `inference.ts` — Model init, generation loop with abort support, cleanup

### Two-library FFI pattern

1. **libllama.dylib** (`llama.cpp/build/bin/`) — upstream llama.cpp. Opened first via `dlopen`.
2. **libllama_shims.dylib** — C shim (`src/llama_shims.c`) compiled with `-undefined dynamic_lookup`. Must be opened **after** libllama.

The shim exists because `bun:ffi` cannot handle C structs passed by value.

### Distribution (optionalDependencies pattern)

```
bun-llama-cpp                          ← main package (TypeScript only)
├── bun-llama-cpp-darwin-arm64         ← prebuilt binaries (macOS Apple Silicon)
├── bun-llama-cpp-darwin-x64           ← planned
├── bun-llama-cpp-linux-x64            ← planned
└── bun-llama-cpp-linux-arm64          ← planned
```

Platform packages live in `packages/`. npm auto-installs only the correct one for the user's OS/arch. `src/lib-resolver.ts` resolves the right .dylib path at runtime.

### llama.cpp Submodule

`llama.cpp/` is a git submodule pinned to a specific commit. Clone with `--recurse-submodules`. Update via `git submodule update`.

### Thread Safety

All llama.cpp interaction runs inside a **Bun Worker** (single-threaded boundary). The `LlamaModel` class enforces serial execution via a Promise-based queue — multiple `infer()` calls are enqueued and run one at a time.

### Consumer API

```ts
const llm = await LlamaModel.load('./model.gguf', { preset: 'medium' })
const result = await llm.infer(prompt, {
  onToken: (text) => process.stdout.write(text),
  signal: AbortSignal.timeout(30_000),
})
await llm.dispose() // Always dispose — frees Metal/GPU buffers
```

## Design Principles

1. **Performance-first** — FFI direto, zero-copy SharedArrayBuffer, Float32 nativo. Cada camada de abstração é opcional.
2. **Controle fino** — Simples (load→infer→dispose), Intermediário (embeddings, grammar), Avançado (sequences, scheduling, pipelines). Cada nível é opt-in.
3. **Composable** — Primitivas que se combinam: `LlamaModel`, `ModelRegistry`, `LlamaEmbedPool`, `ModelPipeline`. Cada componente funciona independente.
4. **Bun-native** — Explorar Workers, SharedArrayBuffer, bun:ffi toArrayBuffer, bun:sqlite, JavaScriptCore.

## Roadmap (see docs/strategy.md for details)

- **Fase 1 — Fundamentos**: ✅ DONE (PR #1) — repetition penalties, model metadata, chat templates, FIM tokens, performance metrics
- **Fase 2 — Embeddings**: EmbeddingContext mode, true batch embedding (multi-seq_id), reranking, cosine similarity
- **Fase 3 — Sequences Paralelas**: multi-sequence context, KV prefix sharing, continuous batching, smart scheduling
- **Fase 4 — Multi-Modelo**: model registry, VRAM tracking, lazy loading, pipeline cascading (embed→rerank→generate)
- **Fase 5 — Advanced**: grammar/JSON mode, speculative decoding, KV cache quantization, context shift

## Key Conventions

- **Buffer management**: Structs are `Buffer.alloc(size)` where size comes from `shim_sizeof_*()`. Fields set via shim setter functions.
- **Pointer casting**: FFI pointers use `as unknown as number` — intentional, required by bun:ffi.
- **Graceful shutdown**: Always call `dispose()`. Direct `worker.terminate()` skips GPU cleanup and causes Metal assertion failures.
- **New shims**: When llama.cpp functions take/return C structs by value, add a wrapper in `src/llama_shims.c` that accepts/returns pointers instead.
- **New direct bindings**: When llama.cpp functions use only scalars/pointers (no structs by value), bind directly in `src/worker/ffi.ts` — no shim needed.
- **Tests are slow**: Tests load a ~5GB model and run inference, so timeouts are 60–180 seconds.
- **Float32 over Float64**: Always use Float32Array for embeddings and float data. Never convert to number[] or Float64Array.
- **Commit messages**: Use conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`).

## Known Quirks

- **`llama_perf_context()` returns zeros** — use JS `performance.now()` for timing instead. The llama.cpp header itself says "avoid using in third-party apps."
- **`llama_model_desc` snprintf semantics** — return value is chars *would* be written; clamp with `Math.min(rawLen, BUF_SIZE - 1)` to avoid buffer overread.
- **`bun test` suppresses output** when not in a TTY — use `--reporter=junit --reporter-outfile=<file>` to capture results in CI or non-interactive shells.
- **`alloca` in C shims** — always add a bounds check (e.g. `n > 1024`) before using `alloca` for user-controlled sizes.
- **`shim_chat_apply_template` returns `-1`** for unsupported templates — check before calling `Buffer.alloc(needed + 1)`.
- **Qwen3-8B has 36 layers**, not 32 — verify model-specific values by running the model before hardcoding in tests.
- **New worker methods** follow the pattern: `queue.enqueue(() => doXxx())` + `doXxx()` with a `Promise` + `onmessage` + 5s timeout.

## Reference Documentation

- `docs/strategy.md` — Full roadmap, competitive analysis, and 8 original innovations
- `docs/gap-analysis.md` — Current API coverage (37%) and detailed gap table
- `docs/embeddings.md` — Embedding strategy and FFI functions needed
- `docs/parallel-sequences.md` — Sequence architecture and continuous batching
- `docs/multi-model.md` — Multi-model orchestration and VRAM management
- `docs/ref-node-llama-cpp.md` — What node-llama-cpp does well and poorly
- `docs/ref-qmd.md` — qmd pipeline patterns and lessons