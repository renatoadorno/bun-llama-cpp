import { toArrayBuffer, type Pointer } from 'bun:ffi'
import type { LibLlama, LibShims } from './ffi.ts'
import type { ResolvedConfig, ModelMetadata, InferMetrics, ParallelInferRequest, ParallelInferResult } from '../types.ts'
import { tokenize, tokenPiece, isEndOfGeneration, isSpecialToken } from './tokenizer.ts'
import { SequenceAllocator, type SequenceSlot } from './sequence-allocator.ts'

export interface LlamaState {
  modelPtr: number
  ctxPtr: number
  vocabPtr: number
  samplerPtr: number | null  // null in embedding mode (primary sampler / single-seq)
  samplers: number[]         // per-sequence samplers (length = nSeqMax)
  batchBuf: Buffer
  chatTemplatePtr: number
  nSeqMax: number
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
  }

  const ctxPtr = S.shim_init_from_model(modelPtr, cpBuf)
  if (!ctxPtr) throw new Error('Failed to create context')

  // Batch buffer (persistent — internal arrays allocated by libllama)
  const batchBuf = Buffer.alloc(Number(S.shim_sizeof_batch()))
  S.shim_batch_init(batchBuf, config.nCtx, 0, config.nSeqMax)

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

  return { modelPtr, ctxPtr, vocabPtr, samplerPtr, samplers, batchBuf, chatTemplatePtr, nSeqMax: config.nSeqMax }
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
    S.shim_batch_add(batchBuf, tokens[i]!, i, 0, i === tokens.length - 1)
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
    S.shim_batch_add(batchBuf, token, pos, 0, true)
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
      S.shim_batch_add(tempBatch, tokens[j]!, j, 0, false)
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

export interface ParallelInferCallbacks {
  onToken: (seqIndex: number, text: string) => void
  isAborted: (seqIndex: number) => boolean
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
    S.shim_batch_add(batchBuf, tokens[i]!, i, 0, i === tokens.length - 1)
  }
  const rc = S.shim_decode(ctxPtr, batchBuf)
  if (rc !== 0) throw new Error(`warmup prefill failed: ${rc}`)

  return tokens.length
}

/**
 * Run parallel inference on multiple prompts using different sequence slots.
 * All sequences share the same KV context and are decoded in the same batch.
 *
 * If warmupTokens > 0, assumes KV cache for seq 0 already has a prefix
 * of that length (from warmupPrefix). Copies it to other sequences via seq_cp.
 */
export function runInferParallel(
  L: LibLlama,
  S: LibShims,
  state: LlamaState,
  requests: ParallelInferRequest[],
  callbacks: ParallelInferCallbacks,
  warmupTokens = 0,
): ParallelInferResult[] {
  if (state.samplerPtr === null) throw new Error('Cannot infer on an embedding model')
  if (requests.length > state.nSeqMax) {
    throw new Error(`Too many parallel requests (${requests.length}) for nSeqMax=${state.nSeqMax}`)
  }

  const { ctxPtr, vocabPtr, samplers, batchBuf } = state
  const mem = L.llama_get_memory(ctxPtr)

  // If no warmup prefix, clear everything
  if (warmupTokens === 0) {
    L.llama_memory_clear(mem, false)
  }

  // Sort by priority (higher first)
  const sorted = requests
    .map((r, i) => ({ ...r, originalIndex: i }))
    .sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0))

  // Allocator for sequence slots
  const allocator = new SequenceAllocator(state.nSeqMax)

  interface SeqState {
    slot: SequenceSlot
    tokens: Int32Array
    position: number
    maxTokens: number
    tokenCount: number
    aborted: boolean
    chunks: string[]
    prefillStart: number
    prefillMs: number
    generateStart: number
    collectMetrics: boolean
    abortFlag: Int32Array
    originalIndex: number
    done: boolean
    batchIndex: number // index of this seq's logits token in the batch
  }

  const seqStates: SeqState[] = []

  for (const req of sorted) {
    const slot = allocator.acquire(String(req.originalIndex), req.priority ?? 0)
    if (!slot) throw new Error('Failed to acquire sequence slot')

    const tokens = tokenize(L, vocabPtr, req.prompt)
    const samplerPtr = samplers[slot.seqId]!
    L.llama_sampler_reset(samplerPtr)

    // If warmup prefix exists and this is not seq 0, copy KV cache
    if (warmupTokens > 0 && slot.seqId !== 0) {
      L.llama_memory_seq_cp(mem, 0, slot.seqId, 0, warmupTokens)
    } else if (warmupTokens > 0 && slot.seqId === 0) {
      // seq 0 already has the prefix from warmup — just reset sampler
    } else {
      // No warmup — clear this sequence's KV
      L.llama_memory_seq_rm(mem, slot.seqId, -1, -1)
    }

    seqStates.push({
      slot,
      tokens,
      position: warmupTokens, // start after prefix
      maxTokens: req.maxTokens,
      tokenCount: 0,
      aborted: false,
      chunks: [],
      prefillStart: 0,
      prefillMs: 0,
      generateStart: 0,
      collectMetrics: req.collectMetrics ?? false,
      abortFlag: req.abortFlag,
      originalIndex: req.originalIndex,
      done: false,
      batchIndex: -1,
    })
  }

  // ── Phase 1: Prefill all sequences ──
  // Each sequence prefills its own prompt tokens (after any shared prefix)
  S.shim_batch_clear(batchBuf)
  for (const ss of seqStates) {
    ss.prefillStart = ss.collectMetrics ? performance.now() : 0
    for (let i = 0; i < ss.tokens.length; i++) {
      const isLast = (i === ss.tokens.length - 1)
      S.shim_batch_add(batchBuf, ss.tokens[i]!, ss.position + i, ss.slot.seqId, isLast)
    }
  }

  const rcPrefill = S.shim_decode(ctxPtr, batchBuf)
  if (rcPrefill !== 0) throw new Error(`parallel prefill decode failed: ${rcPrefill}`)

  for (const ss of seqStates) {
    ss.position += ss.tokens.length
    ss.prefillMs = ss.collectMetrics ? performance.now() - ss.prefillStart : 0
    ss.generateStart = ss.collectMetrics ? performance.now() : 0
    ss.slot.state = 'generating'
  }

  // ── Phase 2: Generation loop (round-robin) ──
  const active = () => seqStates.filter(s => !s.done)

  while (active().length > 0) {
    // Check for aborts
    for (const ss of active()) {
      if (Atomics.load(ss.abortFlag, 0) !== 0) {
        ss.aborted = true
        ss.done = true
        L.llama_memory_seq_rm(mem, ss.slot.seqId, -1, -1)
        allocator.release(ss.slot.seqId)
      }
    }

    const currentActive = active()
    if (currentActive.length === 0) break

    // Sample from previous decode for each active sequence
    S.shim_batch_clear(batchBuf)
    let batchIdx = 0

    for (const ss of currentActive) {
      if (ss.tokenCount >= ss.maxTokens) {
        ss.done = true
        L.llama_memory_seq_rm(mem, ss.slot.seqId, -1, -1)
        allocator.release(ss.slot.seqId)
        continue
      }

      const samplerPtr = samplers[ss.slot.seqId]!
      const token = L.llama_sampler_sample(samplerPtr, ctxPtr, -1)
      const piece = tokenPiece(L, vocabPtr, token)

      if (isEndOfGeneration(L, vocabPtr, token, piece)) {
        ss.done = true
        L.llama_memory_seq_rm(mem, ss.slot.seqId, -1, -1)
        allocator.release(ss.slot.seqId)
        continue
      }

      L.llama_sampler_accept(samplerPtr, token)
      ss.tokenCount++

      if (piece && !isSpecialToken(piece)) {
        ss.chunks.push(piece)
        callbacks.onToken(ss.originalIndex, piece)
      }

      // Add this sequence's next token to the batch
      S.shim_batch_add(batchBuf, token, ss.position, ss.slot.seqId, true)
      ss.position++
      ss.batchIndex = batchIdx++
    }

    // Decode all active sequences in one GPU pass
    const remaining = active()
    if (remaining.length === 0) break

    const rcGen = S.shim_decode(ctxPtr, batchBuf)
    if (rcGen !== 0) throw new Error(`parallel generation decode failed: ${rcGen}`)
  }

  // ── Build results in original order ──
  const results: ParallelInferResult[] = new Array(requests.length)
  for (const ss of seqStates) {
    let metrics: InferMetrics | undefined
    if (ss.collectMetrics) {
      const generateMs = performance.now() - ss.generateStart
      const tokensPerSec = generateMs > 0 ? ss.tokenCount / (generateMs / 1000) : 0
      metrics = {
        promptTokens: ss.tokens.length,
        generatedTokens: ss.tokenCount,
        promptMs: ss.prefillMs,
        generateMs,
        tokensPerSec,
      }
    }
    results[ss.originalIndex] = {
      text: ss.chunks.join(''),
      tokenCount: ss.tokenCount,
      aborted: ss.aborted,
      metrics,
    }
  }

  return results
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
