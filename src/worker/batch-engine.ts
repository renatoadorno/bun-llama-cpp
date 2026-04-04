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
  private fatalError = false

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

  /** Stop the engine and notify all active/pending sequences with an error. */
  shutdown(): void {
    this.fatalError = true
    this.loopRunning = false
    for (const seq of this.active) {
      this.callbacks.onError(seq.id, 'Batch engine shutting down')
    }
    for (const req of this.pending) {
      this.callbacks.onError(req.id, 'Batch engine shutting down')
    }
    this.active = []
    this.pending = []
  }

  enqueue(request: BatchRequest): void {
    if (this.fatalError) {
      this.callbacks.onError(request.id, 'Batch engine is in fatal error state')
      return
    }
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

      // 4. Build unified batch: sample generating seqs + prefill new seqs in ONE decode
      this.unifiedStep()

      // 5. Yield to event loop, then continue
      if (this.active.length > 0 || this.pending.length > 0) {
        setTimeout(() => this.step(), 0)
      } else {
        this.loopRunning = false
      }
    } catch (e) {
      // Fatal error — notify all active and pending sequences, then stop
      for (const seq of this.active) {
        this.callbacks.onError(seq.id, String(e))
      }
      for (const req of this.pending) {
        this.callbacks.onError(req.id, String(e))
      }
      this.active = []
      this.pending = []
      this.fatalError = true
      this.loopRunning = false
    }
  }

  private assertSeqId(seqId: number): void {
    if (seqId < 0 || seqId >= this.state.nSeqMax)
      throw new Error(`seqId ${seqId} out of bounds [0, ${this.state.nSeqMax})`)
  }

  private admitPending(): void {
    while (this.pending.length > 0 && this.allocator.hasFreeSlots()) {
      const req = this.pending.shift()!
      const slot = this.allocator.acquire(req.id, req.priority)
      if (!slot) break

      this.assertSeqId(slot.seqId)

      const tokens = tokenize(this.L, this.state.vocabPtr, req.prompt)
      if (tokens.length === 0) {
        this.allocator.release(slot.seqId)
        this.callbacks.onError(req.id, 'Prompt produced 0 tokens after tokenization — check prompt content')
        continue
      }

      const samplerPtr = this.state.samplers[slot.seqId]!
      this.L.llama_sampler_reset(samplerPtr)

      const mem = this.L.llama_get_memory(this.state.ctxPtr)

      // Setup KV cache for this sequence
      if (req.warmupTokens > 0 && slot.seqId !== 0) {
        this.L.llama_memory_seq_rm(mem, slot.seqId, -1, -1)
        this.L.llama_memory_seq_cp(mem, 0, slot.seqId, -1, -1)
      } else if (req.warmupTokens > 0 && slot.seqId === 0) {
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

  /**
   * Unified step: build ONE batch with both prefill tokens and generation tokens,
   * then ONE shim_decode() call. This ensures logit indices are always valid.
   */
  private unifiedStep(): void {
    const { ctxPtr, vocabPtr, samplers, batchBuf } = this.state
    const generating = this.active.filter(s => s.state === 'generating')
    const needsPrefill = this.active.filter(s => s.state === 'pending_prefill')
    const finished: ActiveSeq[] = []

    // ── Part A: Sample from generating sequences ──
    // Must sample BEFORE clearing the batch — seq.batchIndex is a batch token
    // position from the previous decode, and output_ids maps those positions
    // to logit rows. Clearing + rebuilding the batch first would make the
    // old batchIndex point at a different token.
    const sampled: { seq: ActiveSeq; token: number }[] = []

    for (const seq of generating) {
      if (seq.batchIndex < 0) {
        finished.push(seq)
        this.callbacks.onError(seq.id, 'Internal error: sequence reached generation with invalid batchIndex')
        continue
      }
      if (seq.tokenCount >= seq.maxTokens) {
        finished.push(seq)
        continue
      }

      const samplerPtr = samplers[seq.seqId]!
      const token = this.L.llama_sampler_sample(samplerPtr, ctxPtr, seq.batchIndex)
      const piece = tokenPiece(this.L, vocabPtr, token)

      if (isEndOfGeneration(this.L, vocabPtr, token, piece)) {
        finished.push(seq)
        continue
      }

      this.L.llama_sampler_accept(samplerPtr, token)
      seq.tokenCount++

      if (piece && !isSpecialToken(piece)) {
        seq.chunks.push(piece)
        this.callbacks.onToken(seq.id, piece)
      }

      sampled.push({ seq, token })
    }

    // ── Clear batch and build new one ──
    // batchPos tracks the actual batch token position (index into the batch
    // arrays). llama_sampler_sample needs this — NOT the output row index.
    this.S.shim_batch_clear(batchBuf)
    let batchPos = 0

    // Add sampled tokens for generating sequences (all logits=true)
    for (const { seq, token } of sampled) {
      const ok = this.S.shim_batch_add(batchBuf, this.state.batchCapacity, token, seq.position, seq.seqId, true)
      if (!ok) throw new Error(`Batch full during generation — capacity ${this.state.batchCapacity} exceeded`)
      seq.position++
      seq.batchIndex = batchPos
      batchPos++
    }

    // ── Part B: Add prefill tokens for new sequences ──
    // Pre-check capacity: only admit prefills that fit in remaining budget
    let budgetRemaining = this.state.batchCapacity - sampled.length
    const fittingPrefills: ActiveSeq[] = []
    for (const seq of needsPrefill) {
      if (seq.tokens.length <= budgetRemaining) {
        fittingPrefills.push(seq)
        budgetRemaining -= seq.tokens.length
      }
      // else: stays in active as pending_prefill, will be tried next step
    }

    for (const seq of fittingPrefills) {
      seq.prefillStart = seq.collectMetrics ? performance.now() : 0
      for (let i = 0; i < seq.tokens.length; i++) {
        const isLast = (i === seq.tokens.length - 1)
        const ok = this.S.shim_batch_add(batchBuf, this.state.batchCapacity, seq.tokens[i]!, seq.position + i, seq.seqId, isLast)
        if (!ok) throw new Error(`Batch overflow during prefill for seqId=${seq.seqId} — token ${i}/${seq.tokens.length} exceeds capacity ${this.state.batchCapacity}`)
        if (isLast) seq.batchIndex = batchPos
        batchPos++
      }
    }

    // ── Remove finished sequences ──
    for (const seq of finished) {
      this.finishSeq(seq, false)
      const idx = this.active.indexOf(seq)
      if (idx !== -1) this.active.splice(idx, 1)
    }

    // ── Single decode for all active sequences ──
    if (batchPos > 0) {
      const rc = this.S.shim_decode(ctxPtr, batchBuf)
      if (rc !== 0) throw new Error(`continuous batch decode failed: ${rc}`)
    }

    // ── Transition prefilled sequences to generating ──
    for (const seq of fittingPrefills) {
      seq.position += seq.tokens.length
      seq.prefillMs = seq.collectMetrics ? performance.now() - seq.prefillStart : 0
      seq.generateStart = seq.collectMetrics ? performance.now() : 0
      seq.state = 'generating'
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
