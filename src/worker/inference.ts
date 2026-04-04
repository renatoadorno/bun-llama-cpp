import { toArrayBuffer, type Pointer } from 'bun:ffi'
import type { LibLlama, LibShims } from './ffi.ts'
import type { ResolvedConfig, ModelMetadata, InferMetrics } from '../types.ts'
import { tokenize, tokenPiece, isEndOfGeneration, isSpecialToken } from './tokenizer.ts'

export interface LlamaState {
  modelPtr: number
  ctxPtr: number
  vocabPtr: number
  samplerPtr: number | null  // null in embedding mode (primary sampler / single-seq)
  samplers: number[]         // per-sequence samplers (length = nSeqMax)
  batchBuf: Buffer
  chatTemplatePtr: number
  nSeqMax: number
  batchCapacity: number      // max tokens per batch (n_tokens passed to shim_batch_init)
}

/** Initialize llama backend, load model, create context and sampler. */
export function initModel(
  L: LibLlama,
  S: LibShims,
  modelPath: string,
  config: ResolvedConfig,
): LlamaState {
  // Suppress llama.cpp stderr noise
  L.llama_log_set(null as unknown as ArrayBufferView, null as unknown as ArrayBufferView)
  L.llama_backend_init()

  // Model params
  const mpBuf = Buffer.alloc(Number(S.shim_sizeof_model_params()))
  S.shim_model_default_params(mpBuf)
  S.shim_model_params_set_n_gpu_layers(mpBuf, config.nGpuLayers)

  const modelPathBuf = Buffer.from(modelPath + '\0', 'utf8')
  const modelPtr = S.shim_model_load_from_file(modelPathBuf, mpBuf)
  if (!modelPtr) throw new Error(`Failed to load model: ${modelPath}`)

  const vocabPtr = L.llama_model_get_vocab(modelPtr)
  const chatTemplatePtr = L.llama_model_chat_template(modelPtr, null as unknown as number)

  // Context params
  const cpBuf = Buffer.alloc(Number(S.shim_sizeof_context_params()))
  S.shim_context_default_params(cpBuf)
  S.shim_ctx_params_set_n_ctx(cpBuf, config.nCtx)
  S.shim_ctx_params_set_n_threads(cpBuf, config.nThreads)

  if (config.embeddings) {
    S.shim_ctx_params_set_embeddings(cpBuf, true)
    S.shim_ctx_params_set_pooling_type(cpBuf, config.poolingType)
  }

  if (config.nSeqMax > 1) {
    S.shim_ctx_params_set_n_seq_max(cpBuf, config.nSeqMax)
    // Total KV cache = per-sequence context × number of sequences
    S.shim_ctx_params_set_n_ctx(cpBuf, config.nCtx * config.nSeqMax)
    // Ensure batch can hold a full single-sequence prefill
    S.shim_ctx_params_set_n_batch(cpBuf, config.nCtx)
  }

  const ctxPtr = S.shim_init_from_model(modelPtr, cpBuf)
  if (!ctxPtr) throw new Error('Failed to create context')

  // Batch buffer (persistent — internal arrays allocated by libllama)
  const batchCapacity = config.nCtx
  const batchBuf = Buffer.alloc(Number(S.shim_sizeof_batch()))
  S.shim_batch_init(batchBuf, batchCapacity, 0, config.nSeqMax)

  // Sampler chain: only for generation mode
  let samplerPtr: number | null = null
  const samplers: number[] = []
  if (!config.embeddings) {
    const nSamplers = config.nSeqMax
    for (let s = 0; s < nSamplers; s++) {
      const scpBuf = Buffer.alloc(Number(S.shim_sizeof_sampler_chain_params()))
      S.shim_sampler_chain_default_params(scpBuf)
      const ptr = S.shim_sampler_chain_init(scpBuf) as unknown as number

      const { sampler: sc } = config
      L.llama_sampler_chain_add(ptr, S.shim_sampler_init_top_p(sc.topP, 1))
      L.llama_sampler_chain_add(ptr, S.shim_sampler_init_min_p(sc.minP, 1))
      L.llama_sampler_chain_add(ptr, L.llama_sampler_init_top_k(sc.topK))

      const rp = sc.repeatPenalty ?? 1.1
      const fp = sc.frequencyPenalty ?? 0.0
      const pp = sc.presencePenalty ?? 0.0
      const pn = sc.penaltyLastN ?? 64
      if (rp !== 1.0 || fp !== 0.0 || pp !== 0.0) {
        L.llama_sampler_chain_add(ptr, L.llama_sampler_init_penalties(pn, rp, fp, pp))
      }

      L.llama_sampler_chain_add(ptr, L.llama_sampler_init_temp(sc.temp))
      L.llama_sampler_chain_add(ptr, L.llama_sampler_init_dist(sc.seed))
      samplers.push(ptr)
    }
    samplerPtr = samplers[0]!
  }

  return { modelPtr, ctxPtr, vocabPtr, samplerPtr, samplers, batchBuf, chatTemplatePtr, nSeqMax: config.nSeqMax, batchCapacity }
}

/** Collect model metadata from loaded model. */
export function collectMetadata(L: LibLlama, modelPtr: number): ModelMetadata {
  const BUF_SIZE = 256
  const descBuf = Buffer.alloc(BUF_SIZE)
  const rawLen = L.llama_model_desc(modelPtr, descBuf, BUF_SIZE)
  const descLen = Math.min(rawLen, BUF_SIZE - 1)

  return {
    nParams: Number(L.llama_model_n_params(modelPtr)),
    nEmbd: L.llama_model_n_embd(modelPtr),
    nCtxTrain: L.llama_model_n_ctx_train(modelPtr),
    nLayers: L.llama_model_n_layer(modelPtr),
    desc: descBuf.subarray(0, descLen).toString('utf8'),
    sizeBytes: Number(L.llama_model_size(modelPtr)),
  }
}

export interface InferCallbacks {
  onToken: (text: string) => void
  isAborted: () => boolean
  collectMetrics: boolean
}

/** Run inference: prefill prompt tokens then generate up to maxTokens. */
export function runInference(
  L: LibLlama,
  S: LibShims,
  state: LlamaState,
  prompt: string,
  maxTokens: number,
  callbacks: InferCallbacks,
): { tokenCount: number; aborted: boolean; metrics?: InferMetrics } {
  if (state.samplerPtr === null) throw new Error('Cannot infer on an embedding model — use embed() or embedMany()')
  const { ctxPtr, vocabPtr, samplerPtr, batchBuf } = state
  const tokens = tokenize(L, vocabPtr, prompt)

  // Clear KV cache and sampler state
  const mem = L.llama_get_memory(ctxPtr)
  L.llama_memory_clear(mem, false)
  L.llama_sampler_reset(samplerPtr)

  // Prefill: process prompt tokens, request logits only for last
  S.shim_batch_clear(batchBuf)
  for (let i = 0; i < tokens.length; i++) {
    const ok = S.shim_batch_add(batchBuf, state.batchCapacity, tokens[i]!, i, 0, i === tokens.length - 1)
    if (!ok) throw new Error(`Batch overflow at token ${i}/${tokens.length} — prompt exceeds batch capacity (${state.batchCapacity}). Reduce prompt length or increase nCtx.`)
  }
  const prefillStart = callbacks.collectMetrics ? performance.now() : 0
  const rc = S.shim_decode(ctxPtr, batchBuf)
  if (rc !== 0) throw new Error(`llama_decode (prefill) failed: ${rc}`)
  const prefillMs = callbacks.collectMetrics ? performance.now() - prefillStart : 0

  // Generation loop
  const generateStart = callbacks.collectMetrics ? performance.now() : 0
  let pos = tokens.length
  let tokenCount = 0

  for (let i = 0; i < maxTokens; i++) {
    if (callbacks.isAborted()) return { tokenCount, aborted: true }

    const token = L.llama_sampler_sample(samplerPtr, ctxPtr, -1)
    const piece = tokenPiece(L, vocabPtr, token)

    if (isEndOfGeneration(L, vocabPtr, token, piece)) break

    L.llama_sampler_accept(samplerPtr, token)
    tokenCount++

    // Stream non-special tokens
    if (piece && !isSpecialToken(piece)) {
      callbacks.onToken(piece)
    }

    // Single-token batch for next decode step
    S.shim_batch_clear(batchBuf)
    const ok = S.shim_batch_add(batchBuf, state.batchCapacity, token, pos, 0, true)
    if (!ok) throw new Error(`Batch full at generation step — position ${pos} exceeds capacity ${state.batchCapacity}`)
    const rc2 = S.shim_decode(ctxPtr, batchBuf)
    if (rc2 !== 0) throw new Error(`llama_decode (step ${i}) failed: ${rc2}`)
    pos++
  }

  let metrics: InferMetrics | undefined
  if (callbacks.collectMetrics) {
    const generateMs = performance.now() - generateStart
    const promptTokens = tokens.length
    const generatedTokens = tokenCount
    const tokensPerSec = generateMs > 0 ? generatedTokens / (generateMs / 1000) : 0

    metrics = { promptTokens, generatedTokens, promptMs: prefillMs, generateMs, tokensPerSec }
  }

  return { tokenCount, aborted: false, metrics }
}

/** Run a single text through the embedding forward pass. Returns a copied Float32Array. */
export function runEmbed(
  L: LibLlama,
  S: LibShims,
  state: LlamaState,
  text: string,
): Float32Array {
  const tokens = tokenize(L, state.vocabPtr, text)
  const n_embd = L.llama_model_n_embd(state.modelPtr) as number

  const nCtx = L.llama_n_ctx(state.ctxPtr) as number
  if (tokens.length > nCtx) throw new Error(`Input too long: ${tokens.length} tokens exceeds context length ${nCtx}`)

  const tempBatch = Buffer.alloc(Number(S.shim_sizeof_batch()))
  S.shim_batch_init(tempBatch, tokens.length, 0, 1)
  try {
    for (let j = 0; j < tokens.length; j++) {
      const ok = S.shim_batch_add(tempBatch, tokens.length, tokens[j]!, j, 0, false)
      if (!ok) throw new Error(`Embed batch overflow at token ${j}`)
    }

    const hasEncoder = L.llama_model_has_encoder(state.modelPtr) as boolean
    // For encoder models, llama_encode manages its own state;
    // KV-cache clear is only needed for decoder-only models used as embedders.
    if (!hasEncoder) {
      const mem = L.llama_get_memory(state.ctxPtr)
      L.llama_memory_clear(mem, false)
    }

    const ret = hasEncoder
      ? S.shim_encode(state.ctxPtr, tempBatch) as number
      : S.shim_decode(state.ctxPtr, tempBatch) as number
    if (ret !== 0) throw new Error(`embedding forward pass failed: ${ret}`)

    const ptr = L.llama_get_embeddings_seq(state.ctxPtr, 0) as unknown as Pointer
    if (!ptr) throw new Error('null embedding pointer for sequence 0')

    const raw = new Float32Array(toArrayBuffer(ptr, 0, n_embd * 4))
    return raw.slice()  // copy from C-owned memory before next FFI call
  } finally {
    S.shim_batch_free(tempBatch)
  }
}

/** Embed multiple texts sequentially, returning one Float32Array per text. */
export function runEmbedBatch(
  L: LibLlama,
  S: LibShims,
  state: LlamaState,
  texts: string[],
): Float32Array[] {
  return texts.map(t => runEmbed(L, S, state, t))
}

/**
 * Pre-compute KV cache for a shared system prompt on sequence 0.
 * Returns the number of prefix tokens processed.
 * Subsequent inferParallel calls can share this prefix via seq_cp.
 */
export function warmupPrefix(
  L: LibLlama,
  S: LibShims,
  state: LlamaState,
  systemPrompt: string,
): number {
  if (state.samplerPtr === null) throw new Error('Cannot warmup an embedding model')
  const { ctxPtr, vocabPtr, batchBuf } = state
  const tokens = tokenize(L, vocabPtr, systemPrompt)

  const mem = L.llama_get_memory(ctxPtr)
  L.llama_memory_clear(mem, false)

  // Prefill system prompt on sequence 0
  S.shim_batch_clear(batchBuf)
  for (let i = 0; i < tokens.length; i++) {
    const ok = S.shim_batch_add(batchBuf, state.batchCapacity, tokens[i]!, i, 0, i === tokens.length - 1)
    if (!ok) throw new Error(`Warmup batch overflow at token ${i}/${tokens.length} — system prompt too long for batch capacity (${state.batchCapacity}). Reduce system prompt or increase nCtx.`)
  }
  const rc = S.shim_decode(ctxPtr, batchBuf)
  if (rc !== 0) throw new Error(`warmup prefill failed: ${rc}`)

  return tokens.length
}

/** Free all llama resources (GPU buffers, model, context). */
export function cleanup(L: LibLlama, S: LibShims, state: LlamaState): void {
  try {
    for (const ptr of state.samplers) {
      L.llama_sampler_free(ptr)
    }
  } catch {}
  try { if (state.batchBuf.length > 0) S.shim_batch_free(state.batchBuf) } catch {}
  try { if (state.ctxPtr) L.llama_free(state.ctxPtr) } catch {}
  try { if (state.modelPtr) L.llama_model_free(state.modelPtr) } catch {}
  try { L.llama_backend_free() } catch {}
}
