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

# Run example demo
bun run start

# Run with hot-reload
bun run dev

# Run tests
bun run test

# Run a single test file
DYLD_LIBRARY_PATH=llama.cpp/build/bin bun test src/llm.worker.test.ts
```

All runtime commands require `DYLD_LIBRARY_PATH=llama.cpp/build/bin` (already set in package.json scripts).

## Architecture

This is a **reusable library** providing low-level FFI bindings from Bun to `libllama.dylib` (llama.cpp). The public API is a `LlamaModel` class that hides all FFI and worker internals.

### Module Structure

- `src/index.ts` — Public entry point. Exports `LlamaModel`, types, and presets.
- `src/model.ts` — `LlamaModel` facade class. Manages worker lifecycle, serial queue (thread-safety), and AbortSignal cancellation.
- `src/types.ts` — All public types (`ModelConfig`, `InferOptions`, `InferResult`, `Preset`) and internal worker protocol types.
- `src/presets.ts` — Preset configs (small/medium/large) with `resolveConfig()` merge logic.
- `src/queue.ts` — Promise-chain serial queue ensuring one inference at a time.
- `src/worker/` — Everything that runs inside the Bun Worker (never imported by consumers):
  - `llm.worker.ts` — Thin message router delegating to modules below
  - `ffi.ts` — `dlopen` bindings for libllama + shims, typed interfaces
  - `tokenizer.ts` — `tokenize()`, `tokenPiece()`, EOT detection
  - `inference.ts` — Model init, generation loop with abort support, cleanup

### Two-library FFI pattern

1. **libllama.dylib** (`llama.cpp/build/bin/`) — upstream llama.cpp. Opened first via `dlopen`.
2. **libllama_shims.dylib** (`src/libllama_shims.dylib`) — C shim (`src/llama_shims.c`) compiled with `-undefined dynamic_lookup`. Must be opened **after** libllama.

The shim exists because `bun:ffi` cannot handle C structs passed by value.

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

## Key Conventions

- **Buffer management**: Structs are `Buffer.alloc(size)` where size comes from `shim_sizeof_*()`. Fields set via shim setter functions.
- **Pointer casting**: FFI pointers use `as unknown as number` — intentional, required by bun:ffi.
- **Graceful shutdown**: Always call `dispose()`. Direct `worker.terminate()` skips GPU cleanup and causes Metal assertion failures.
- **Tests are slow**: Tests load a ~5GB model and run inference, so timeouts are 60–180 seconds.
