import { resolveConfig } from './presets.ts'
import { SerialQueue } from './queue.ts'
import type {
  ModelConfig,
  InferOptions,
  InferResult,
  FimTokens,
  ModelMetadata,
  ChatMessage,
  ParallelInferResult,
  WorkerRequest,
  WorkerResponse,
} from './types.ts'

export interface ParallelInferOptions {
  prompt: string
  onToken: (text: string) => void
  maxTokens?: number
  signal?: AbortSignal
  priority?: number
  metrics?: boolean
}

export class LlamaModel {
  private worker: Worker
  private queue = new SerialQueue()
  private _isReady = false
  private _isBusy = false
  private _disposed = false
  private _metadata!: ModelMetadata
  private _embeddingMode = false
  private _nSeqMax = 1
  private _warmupTokens = 0
  private _activeInfers = 0

  // Persistent message router: dispatches by request ID
  private _handlers = new Map<string, (msg: WorkerResponse) => void>()
  // Fallback handler for messages without an ID (e.g. 'fimTokens')
  private _defaultHandler: ((msg: WorkerResponse) => void) | null = null

  private constructor(worker: Worker) {
    this.worker = worker
    this._setupRouter()
  }

  private _setupRouter(): void {
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data
      // Route by message ID if present
      if ('id' in msg && msg.id) {
        const handler = this._handlers.get(msg.id)
        if (handler) handler(msg)
      } else if (this._defaultHandler) {
        this._defaultHandler(msg)
      }
    }
  }

  private _registerHandler(id: string, handler: (msg: WorkerResponse) => void): void {
    this._handlers.set(id, handler)
  }

  private _unregisterHandler(id: string): void {
    this._handlers.delete(id)
  }

  /** Load a GGUF model and return an initialized LlamaModel instance. */
  static async load(modelPath: string, config?: ModelConfig): Promise<LlamaModel> {
    const resolved = resolveConfig(config)
    const worker = new Worker(new URL('./worker/llm.worker.ts', import.meta.url).href)
    const instance = new LlamaModel(worker)

    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Model load timeout (120s)')), 120_000)

      // Use a temporary default handler for the init response (no id)
      instance._defaultHandler = (msg: WorkerResponse) => {
        if (msg.type === 'ready') {
          clearTimeout(timer)
          instance._defaultHandler = null
          instance._metadata = msg.metadata
          instance._isReady = true
          instance._embeddingMode = resolved.embeddings
          instance._nSeqMax = resolved.nSeqMax
          resolve()
        } else if (msg.type === 'error') {
          clearTimeout(timer)
          instance._defaultHandler = null
          reject(new Error(msg.message))
        }
      }

      const initMsg: WorkerRequest = { type: 'init', modelPath, config: resolved }
      worker.postMessage(initMsg)
    })

    return instance
  }

  /** Run inference with streaming token output. */
  async infer(prompt: string, options: InferOptions): Promise<InferResult> {
    if (this._disposed) throw new Error('Model has been disposed')
    if (!this._isReady) throw new Error('Model is not ready')
    if (this._embeddingMode) throw new Error('Cannot call infer() on an embedding model — use embed() or embedMany()')

    // Continuous batching: bypass serial queue, join batch engine
    if (this._nSeqMax > 1) {
      return this.doConcurrentInfer(prompt, options)
    }

    return this.queue.enqueue(() => this.doInfer(prompt, options))
  }

  /** Concurrent infer via batch engine — multiple calls run in parallel. */
  private doConcurrentInfer(prompt: string, options: InferOptions): Promise<InferResult> {
    return new Promise<InferResult>((resolve, reject) => {
      this._activeInfers++
      const id = crypto.randomUUID()

      const abortBuf = new SharedArrayBuffer(4)
      const abortFlag = new Int32Array(abortBuf)

      const onAbort = () => Atomics.store(abortFlag, 0, 1)

      if (options.signal?.aborted) {
        this._activeInfers--
        resolve({ text: '', tokenCount: 0, aborted: true })
        return
      }

      options.signal?.addEventListener('abort', onAbort, { once: true })

      const timer = setTimeout(() => {
        Atomics.store(abortFlag, 0, 1)
        cleanup()
        reject(new Error('infer timeout (300s)'))
      }, 300_000)

      const cleanup = () => {
        clearTimeout(timer)
        options.signal?.removeEventListener('abort', onAbort)
        this._unregisterHandler(id)
        this._activeInfers--
      }

      this._registerHandler(id, (msg: WorkerResponse) => {
        switch (msg.type) {
          case 'seqToken': {
            options.onToken(msg.text)
            break
          }
          case 'seqDone': {
            cleanup()
            resolve({
              text: msg.text,
              tokenCount: msg.tokenCount,
              aborted: msg.aborted,
              metrics: msg.metrics,
            })
            break
          }
          case 'error': {
            cleanup()
            reject(new Error(msg.message))
            break
          }
        }
      })

      const inferMsg: WorkerRequest = {
        type: 'startInfer',
        id,
        prompt,
        maxTokens: options.maxTokens ?? 512,
        priority: 0,
        abortFlag,
        collectMetrics: options.metrics ?? false,
        warmupTokens: this._warmupTokens,
      }
      this.worker.postMessage(inferMsg)
    })
  }

  private doInfer(prompt: string, options: InferOptions): Promise<InferResult> {
    return new Promise<InferResult>((resolve, reject) => {
      this._isBusy = true
      const id = crypto.randomUUID()
      const chunks: string[] = []

      const abortBuf = new SharedArrayBuffer(4)
      const abortFlag = new Int32Array(abortBuf)

      const onAbort = () => {
        Atomics.store(abortFlag, 0, 1)
      }

      if (options.signal?.aborted) {
        this._isBusy = false
        resolve({ text: '', tokenCount: 0, aborted: true })
        return
      }

      options.signal?.addEventListener('abort', onAbort, { once: true })

      this._registerHandler(id, (msg: WorkerResponse) => {
        switch (msg.type) {
          case 'token': {
            chunks.push(msg.text)
            options.onToken(msg.text)
            break
          }
          case 'done': {
            options.signal?.removeEventListener('abort', onAbort)
            this._unregisterHandler(id)
            this._isBusy = false
            resolve({
              text: chunks.join(''),
              tokenCount: msg.tokenCount,
              aborted: false,
              metrics: msg.metrics,
            })
            break
          }
          case 'aborted': {
            options.signal?.removeEventListener('abort', onAbort)
            this._unregisterHandler(id)
            this._isBusy = false
            resolve({
              text: chunks.join(''),
              tokenCount: chunks.length,
              aborted: true,
            })
            break
          }
          case 'error': {
            options.signal?.removeEventListener('abort', onAbort)
            this._unregisterHandler(id)
            this._isBusy = false
            reject(new Error(msg.message))
            break
          }
        }
      })

      const inferMsg: WorkerRequest = {
        type: 'infer',
        id,
        prompt,
        maxTokens: options.maxTokens ?? 512,
        abortFlag,
        collectMetrics: options.metrics ?? false,
      }
      this.worker.postMessage(inferMsg)
    })
  }

  get isReady(): boolean { return this._isReady && !this._disposed }
  get isBusy(): boolean { return this._isBusy || this._activeInfers > 0 }

  /** Model metadata — populated during load(). */
  get metadata(): ModelMetadata { return this._metadata }

  /** Get Fill-in-Middle token IDs. Returns -1 for unsupported tokens. */
  async getFimTokens(): Promise<FimTokens> {
    if (this._disposed) throw new Error('Model has been disposed')
    if (!this._isReady) throw new Error('Model is not ready')

    return this.queue.enqueue(() => this.doGetFimTokens())
  }

  private doGetFimTokens(): Promise<FimTokens> {
    return new Promise<FimTokens>((resolve, reject) => {
      const timer = setTimeout(() => {
        this._defaultHandler = null
        reject(new Error('getFimTokens timeout (5s)'))
      }, 5_000)

      this._defaultHandler = (msg: WorkerResponse) => {
        if (msg.type === 'fimTokens') {
          clearTimeout(timer)
          this._defaultHandler = null
          resolve(msg.data)
        } else if (msg.type === 'error') {
          clearTimeout(timer)
          this._defaultHandler = null
          reject(new Error(msg.message))
        }
      }
      const req: WorkerRequest = { type: 'getFimTokens' }
      this.worker.postMessage(req)
    })
  }

  /** Apply the model's chat template to format messages into a prompt string. */
  async applyTemplate(
    messages: ChatMessage[],
    options?: { addAssistant?: boolean },
  ): Promise<string> {
    if (this._disposed) throw new Error('Model has been disposed')
    if (!this._isReady) throw new Error('Model is not ready')
    if (messages.length === 0) throw new Error('messages must not be empty')
    if (messages.length > 1024) throw new Error('messages must not exceed 1024 entries')
    for (const m of messages) {
      if (m.role.includes('\0') || m.content.includes('\0'))
        throw new Error('message role and content must not contain NUL characters')
    }

    return this.queue.enqueue(() => this.doApplyTemplate(messages, options))
  }

  private doApplyTemplate(
    messages: ChatMessage[],
    options?: { addAssistant?: boolean },
  ): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      const id = crypto.randomUUID()
      const timer = setTimeout(() => {
        this._unregisterHandler(id)
        reject(new Error('applyTemplate timeout (5s)'))
      }, 5_000)

      this._registerHandler(id, (msg: WorkerResponse) => {
        if (msg.type === 'templateResult') {
          clearTimeout(timer)
          this._unregisterHandler(id)
          resolve(msg.text)
        } else if (msg.type === 'error') {
          clearTimeout(timer)
          this._unregisterHandler(id)
          reject(new Error(msg.message))
        }
      })

      const req: WorkerRequest = {
        type: 'applyTemplate',
        id,
        messages,
        addAssistant: options?.addAssistant ?? true,
      }
      this.worker.postMessage(req)
    })
  }

  /** Embed a single text. Model must be loaded with embeddings: true. */
  async embed(text: string): Promise<Float32Array> {
    if (this._disposed) throw new Error('Model has been disposed')
    if (!this._isReady) throw new Error('Model is not ready')
    if (!this._embeddingMode) throw new Error('Load model with embeddings: true to use embed()')
    return this.queue.enqueue(() => this.doEmbed(text))
  }

  private doEmbed(text: string): Promise<Float32Array> {
    return new Promise<Float32Array>((resolve, reject) => {
      const id = crypto.randomUUID()
      const timer = setTimeout(() => {
        this._unregisterHandler(id)
        reject(new Error('embed timeout (60s)'))
      }, 60_000)

      this._registerHandler(id, (msg: WorkerResponse) => {
        if (msg.type === 'embedResult') {
          clearTimeout(timer)
          this._unregisterHandler(id)
          resolve(msg.vector)
        } else if (msg.type === 'error') {
          clearTimeout(timer)
          this._unregisterHandler(id)
          reject(new Error(msg.message))
        }
      })

      const req: WorkerRequest = { type: 'embed', id, text }
      this.worker.postMessage(req)
    })
  }

  /** Embed multiple texts sequentially. Model must be loaded with embeddings: true. */
  async embedMany(texts: string[]): Promise<Float32Array[]> {
    if (this._disposed) throw new Error('Model has been disposed')
    if (!this._isReady) throw new Error('Model is not ready')
    if (!this._embeddingMode) throw new Error('Load model with embeddings: true to use embedMany()')
    if (texts.length === 0) return []
    return this.queue.enqueue(() => this.doEmbedBatch(texts))
  }

  private doEmbedBatch(texts: string[]): Promise<Float32Array[]> {
    return new Promise<Float32Array[]>((resolve, reject) => {
      const id = crypto.randomUUID()
      const timer = setTimeout(() => {
        this._unregisterHandler(id)
        reject(new Error('embedMany timeout (60s)'))
      }, 60_000)

      this._registerHandler(id, (msg: WorkerResponse) => {
        if (msg.type === 'embedBatchResult') {
          clearTimeout(timer)
          this._unregisterHandler(id)
          resolve(msg.vectors)
        } else if (msg.type === 'error') {
          clearTimeout(timer)
          this._unregisterHandler(id)
          reject(new Error(msg.message))
        }
      })

      const req: WorkerRequest = { type: 'embedBatch', id, texts }
      this.worker.postMessage(req)
    })
  }

  /**
   * Pre-compute KV cache for a shared system prompt.
   * Subsequent inferParallel calls will share this prefix via O(1) KV copy.
   * Only works when nSeqMax > 1.
   */
  async warmup(systemPrompt: string): Promise<number> {
    if (this._disposed) throw new Error('Model has been disposed')
    if (!this._isReady) throw new Error('Model is not ready')
    if (this._embeddingMode) throw new Error('Cannot warmup an embedding model')
    if (this._nSeqMax <= 1) throw new Error('warmup requires nSeqMax > 1')

    return this.queue.enqueue(() => this.doWarmup(systemPrompt))
  }

  private doWarmup(systemPrompt: string): Promise<number> {
    return new Promise<number>((resolve, reject) => {
      const id = crypto.randomUUID()
      const timer = setTimeout(() => {
        this._unregisterHandler(id)
        reject(new Error('warmup timeout (60s)'))
      }, 60_000)

      this._registerHandler(id, (msg: WorkerResponse) => {
        if (msg.type === 'warmupDone') {
          clearTimeout(timer)
          this._unregisterHandler(id)
          this._warmupTokens = msg.tokenCount
          resolve(msg.tokenCount)
        } else if (msg.type === 'error') {
          clearTimeout(timer)
          this._unregisterHandler(id)
          reject(new Error(msg.message))
        }
      })

      const req: WorkerRequest = { type: 'warmup', id, systemPrompt }
      this.worker.postMessage(req)
    })
  }

  /**
   * Run multiple inferences in parallel using separate sequence slots.
   * All sequences are decoded in the same GPU batch for maximum throughput.
   * Requires nSeqMax > 1.
   *
   * If warmup() was called, the pre-computed system prompt KV is shared
   * across all sequences via O(1) copy.
   */
  async inferParallel(requests: ParallelInferOptions[]): Promise<ParallelInferResult[]> {
    if (this._disposed) throw new Error('Model has been disposed')
    if (!this._isReady) throw new Error('Model is not ready')
    if (this._embeddingMode) throw new Error('Cannot call inferParallel on an embedding model')
    if (this._nSeqMax <= 1) throw new Error('inferParallel requires nSeqMax > 1')
    if (requests.length > this._nSeqMax) {
      throw new Error(`Too many parallel requests (${requests.length}) for nSeqMax=${this._nSeqMax}`)
    }
    if (requests.length === 0) return []

    // Route through batch engine via concurrent infers — avoids separate
    // SequenceAllocator/context conflicts with the BatchEngine.
    const promises = requests.map(req =>
      this.doConcurrentInfer(req.prompt, {
        onToken: req.onToken,
        maxTokens: req.maxTokens,
        signal: req.signal,
        metrics: req.metrics,
      })
    )

    const results = await Promise.all(promises)
    return results.map(r => ({
      text: r.text,
      tokenCount: r.tokenCount,
      aborted: r.aborted,
      metrics: r.metrics,
    }))
  }

  /** Graceful shutdown — frees GPU/Metal buffers before terminating. */
  async dispose(): Promise<void> {
    if (this._disposed) return
    this._disposed = true
    this._isReady = false

    return new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        this.worker.terminate()
        resolve()
      }, 5_000)

      this.worker.addEventListener('close', () => {
        clearTimeout(timer)
        resolve()
      })

      const shutdownMsg: WorkerRequest = { type: 'shutdown' }
      this.worker.postMessage(shutdownMsg)
    })
  }
}
