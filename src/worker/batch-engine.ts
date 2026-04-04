import type { LibLlama, LibShims } from './ffi.ts'
import type { InferMetrics } from '../types.ts'
import type { LlamaState } from './inference.ts'
import { SequenceAllocator } from './sequence-allocator.ts'
import { tokenize, tokenPiece, isEndOfGeneration, isSpecialToken } from './tokenizer.ts'

export interface BatchRequest {
  id: string
  prompt: string
  maxTokens: number
  priority: number
  abortFlag: Int32Array
  collectMetrics: boolean
  warmupTokens: number
}

export interface BatchCallbacks {
  onToken: (id: string, text: string) => void
  onDone: (id: string, result: { text: string; tokenCount: number; aborted: boolean; metrics?: InferMetrics }) => void
  onError: (id: string, error: string) => void
}

interface ActiveSeq {
  id: string
  seqId: number
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
  warmupTokens: number
  batchIndex: number
  state: 'pending_prefill' | 'generating'
}

/**
 * Continuous batching engine for the worker.
 *
 * Runs a cooperative loop via setTimeout(0), yielding between decode steps
 * so the worker event loop can process new incoming requests.
 *
 * New requests are admitted into the batch between generation steps —
 * no need to wait for existing sequences to finish.
 */
export class BatchEngine {
  private L: LibLlama
  private S: LibShims
  private state: LlamaState
  private allocator: SequenceAllocator
  private callbacks: BatchCallbacks

  private pending: BatchRequest[] = []
  private active: ActiveSeq[] = []
  private loopRunning = false

  constructor(L: LibLlama, S: LibShims, state: LlamaState, callbacks: BatchCallbacks) {
    this.L = L
    this.S = S
    this.state = state
    this.allocator = new SequenceAllocator(state.nSeqMax)
    this.callbacks = callbacks
  }

  get isActive(): boolean {
    return this.active.length > 0 || this.pending.length > 0
  }

  enqueue(request: BatchRequest): void {
    this.pending.push(request)
    if (!this.loopRunning) this.startLoop()
  }

  private startLoop(): void {
    this.loopRunning = true
    this.step()
  }

  private step(): void {
    try {
      // 1. Admit pending requests into free slots
      this.admitPending()

      // 2. Check for aborts on active sequences
      this.checkAborts()

      // 3. If nothing active, stop the loop
      if (this.active.length === 0) {
        this.loopRunning = false
        return
      }

      // 4. Prefill any sequences that need it
      const needsPrefill = this.active.filter(s => s.state === 'pending_prefill')
      if (needsPrefill.length > 0) {
        this.prefillBatch(needsPrefill)
      }

      // 5. Generate one token for each active generating sequence
      const generating = this.active.filter(s => s.state === 'generating')
      if (generating.length > 0) {
        this.generateStep(generating)
      }

      // 6. Yield to event loop, then continue
      if (this.active.length > 0 || this.pending.length > 0) {
        setTimeout(() => this.step(), 0)
      } else {
        this.loopRunning = false
      }
    } catch (e) {
      // Fatal error — notify all active sequences and stop
      for (const seq of this.active) {
        this.callbacks.onError(seq.id, String(e))
      }
      this.active = []
      this.loopRunning = false
    }
  }

  private admitPending(): void {
    while (this.pending.length > 0 && this.allocator.hasFreeSlots()) {
      const req = this.pending.shift()!
      const slot = this.allocator.acquire(req.id, req.priority)
      if (!slot) break

      const tokens = tokenize(this.L, this.state.vocabPtr, req.prompt)
      const samplerPtr = this.state.samplers[slot.seqId]!
      this.L.llama_sampler_reset(samplerPtr)

      const mem = this.L.llama_get_memory(this.state.ctxPtr)

      // Setup KV cache for this sequence
      if (req.warmupTokens > 0 && slot.seqId !== 0) {
        // Clean any previous user tokens, then copy warmup prefix
        this.L.llama_memory_seq_rm(mem, slot.seqId, -1, -1)
        this.L.llama_memory_seq_cp(mem, 0, slot.seqId, 0, req.warmupTokens)
      } else if (req.warmupTokens > 0 && slot.seqId === 0) {
        // seq 0 already has prefix — clear only user tokens
        this.L.llama_memory_seq_rm(mem, slot.seqId, req.warmupTokens, -1)
      } else {
        this.L.llama_memory_seq_rm(mem, slot.seqId, -1, -1)
      }

      this.active.push({
        id: req.id,
        seqId: slot.seqId,
        tokens,
        position: req.warmupTokens,
        maxTokens: req.maxTokens,
        tokenCount: 0,
        aborted: false,
        chunks: [],
        prefillStart: 0,
        prefillMs: 0,
        generateStart: 0,
        collectMetrics: req.collectMetrics,
        abortFlag: req.abortFlag,
        warmupTokens: req.warmupTokens,
        batchIndex: -1,
        state: 'pending_prefill',
      })
    }
  }

  private checkAborts(): void {
    const mem = this.L.llama_get_memory(this.state.ctxPtr)
    for (let i = this.active.length - 1; i >= 0; i--) {
      const seq = this.active[i]!
      if (Atomics.load(seq.abortFlag, 0) !== 0) {
        seq.aborted = true
        this.L.llama_memory_seq_rm(mem, seq.seqId, seq.warmupTokens > 0 ? seq.warmupTokens : -1, -1)
        this.allocator.release(seq.seqId)
        this.active.splice(i, 1)
        this.callbacks.onDone(seq.id, {
          text: seq.chunks.join(''),
          tokenCount: seq.tokenCount,
          aborted: true,
        })
      }
    }
  }

  private prefillBatch(seqs: ActiveSeq[]): void {
    const { ctxPtr, batchBuf } = this.state

    this.S.shim_batch_clear(batchBuf)
    for (const seq of seqs) {
      seq.prefillStart = seq.collectMetrics ? performance.now() : 0
      for (let i = 0; i < seq.tokens.length; i++) {
        const isLast = (i === seq.tokens.length - 1)
        this.S.shim_batch_add(batchBuf, seq.tokens[i]!, seq.position + i, seq.seqId, isLast)
      }
    }

    const rc = this.S.shim_decode(ctxPtr, batchBuf)
    if (rc !== 0) throw new Error(`continuous prefill decode failed: ${rc}`)

    // Transition to generating
    for (let i = 0; i < seqs.length; i++) {
      const seq = seqs[i]!
      seq.position += seq.tokens.length
      seq.prefillMs = seq.collectMetrics ? performance.now() - seq.prefillStart : 0
      seq.generateStart = seq.collectMetrics ? performance.now() : 0
      seq.state = 'generating'
      seq.batchIndex = i  // logits output index from prefill
    }
  }

  private generateStep(seqs: ActiveSeq[]): void {
    const { ctxPtr, vocabPtr, samplers, batchBuf } = this.state
    const mem = this.L.llama_get_memory(ctxPtr)

    // Sample from previous decode, build next batch
    this.S.shim_batch_clear(batchBuf)
    let batchIdx = 0
    const toRemove: number[] = []

    for (let i = 0; i < seqs.length; i++) {
      const seq = seqs[i]!

      if (seq.tokenCount >= seq.maxTokens) {
        this.finishSeq(seq, false)
        toRemove.push(i)
        continue
      }

      const samplerPtr = samplers[seq.seqId]!
      const token = this.L.llama_sampler_sample(samplerPtr, ctxPtr, seq.batchIndex)
      const piece = tokenPiece(this.L, vocabPtr, token)

      if (isEndOfGeneration(this.L, vocabPtr, token, piece)) {
        this.finishSeq(seq, false)
        toRemove.push(i)
        continue
      }

      this.L.llama_sampler_accept(samplerPtr, token)
      seq.tokenCount++

      if (piece && !isSpecialToken(piece)) {
        seq.chunks.push(piece)
        this.callbacks.onToken(seq.id, piece)
      }

      this.S.shim_batch_add(batchBuf, token, seq.position, seq.seqId, true)
      seq.position++
      seq.batchIndex = batchIdx++
    }

    // Remove finished sequences (reverse order to keep indices valid)
    for (let i = toRemove.length - 1; i >= 0; i--) {
      const idx = this.active.indexOf(seqs[toRemove[i]!]!)
      if (idx !== -1) this.active.splice(idx, 1)
    }

    // Decode remaining active sequences
    if (batchIdx > 0) {
      const rc = this.S.shim_decode(ctxPtr, batchBuf)
      if (rc !== 0) throw new Error(`continuous generation decode failed: ${rc}`)
    }
  }

  private finishSeq(seq: ActiveSeq, aborted: boolean): void {
    const mem = this.L.llama_get_memory(this.state.ctxPtr)
    this.L.llama_memory_seq_rm(mem, seq.seqId, seq.warmupTokens > 0 ? seq.warmupTokens : -1, -1)
    this.allocator.release(seq.seqId)

    let metrics: InferMetrics | undefined
    if (seq.collectMetrics) {
      const generateMs = performance.now() - seq.generateStart
      const tokensPerSec = generateMs > 0 ? seq.tokenCount / (generateMs / 1000) : 0
      metrics = {
        promptTokens: seq.tokens.length,
        generatedTokens: seq.tokenCount,
        promptMs: seq.prefillMs,
        generateMs,
        tokensPerSec,
      }
    }

    this.callbacks.onDone(seq.id, {
      text: seq.chunks.join(''),
      tokenCount: seq.tokenCount,
      aborted,
      metrics,
    })
  }
}
