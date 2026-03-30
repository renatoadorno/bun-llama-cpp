import type { LibLlama, LibShims } from './ffi.ts'
import type { ResolvedConfig, ModelMetadata, InferMetrics } from '../types.ts'
import { tokenize, tokenPiece, isEndOfGeneration, isSpecialToken } from './tokenizer.ts'

export interface LlamaState {
  modelPtr: number
  ctxPtr: number
  vocabPtr: number
  samplerPtr: number
  batchBuf: Buffer
  chatTemplatePtr: number
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

  const ctxPtr = S.shim_init_from_model(modelPtr, cpBuf)
  if (!ctxPtr) throw new Error('Failed to create context')

  // Batch buffer (persistent — internal arrays allocated by libllama)
  const batchBuf = Buffer.alloc(Number(S.shim_sizeof_batch()))
  S.shim_batch_init(batchBuf, config.nCtx, 0, 1)

  // Sampler chain: top-p → min-p → top-k → temp → dist
  const scpBuf = Buffer.alloc(Number(S.shim_sizeof_sampler_chain_params()))
  S.shim_sampler_chain_default_params(scpBuf)
  const samplerPtr = S.shim_sampler_chain_init(scpBuf)

  const { sampler: sc } = config
  L.llama_sampler_chain_add(samplerPtr, S.shim_sampler_init_top_p(sc.topP, 1))
  L.llama_sampler_chain_add(samplerPtr, S.shim_sampler_init_min_p(sc.minP, 1))
  L.llama_sampler_chain_add(samplerPtr, L.llama_sampler_init_top_k(sc.topK))

  // Penalties after top-k/top-p (llama.h: "apply top-k or top-p sampling first")
  const rp = sc.repeatPenalty ?? 1.1
  const fp = sc.frequencyPenalty ?? 0.0
  const pp = sc.presencePenalty ?? 0.0
  const pn = sc.penaltyLastN ?? 64
  if (rp !== 1.0 || fp !== 0.0 || pp !== 0.0) {
    L.llama_sampler_chain_add(samplerPtr, L.llama_sampler_init_penalties(pn, rp, fp, pp))
  }

  L.llama_sampler_chain_add(samplerPtr, L.llama_sampler_init_temp(sc.temp))
  L.llama_sampler_chain_add(samplerPtr, L.llama_sampler_init_dist(sc.seed))

  return { modelPtr, ctxPtr, vocabPtr, samplerPtr, batchBuf, chatTemplatePtr }
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
  const prefillStart = performance.now()
  const rc = S.shim_decode(ctxPtr, batchBuf)
  if (rc !== 0) throw new Error(`llama_decode (prefill) failed: ${rc}`)
  const prefillMs = performance.now() - prefillStart

  // Generation loop
  const generateStart = performance.now()
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

/** Free all llama resources (GPU buffers, model, context). */
export function cleanup(L: LibLlama, S: LibShims, state: LlamaState): void {
  try { if (state.samplerPtr) L.llama_sampler_free(state.samplerPtr) } catch {}
  try { if (state.batchBuf.length > 0) S.shim_batch_free(state.batchBuf) } catch {}
  try { if (state.ctxPtr) L.llama_free(state.ctxPtr) } catch {}
  try { if (state.modelPtr) L.llama_model_free(state.modelPtr) } catch {}
  try { L.llama_backend_free() } catch {}
}
