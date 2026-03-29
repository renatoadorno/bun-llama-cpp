import { resolveConfig } from './presets.ts'
import { SerialQueue } from './queue.ts'
import type {
  ModelConfig,
  InferOptions,
  InferResult,
  FimTokens,
  ModelMetadata,
  WorkerRequest,
  WorkerResponse,
} from './types.ts'

export class LlamaModel {
  private worker: Worker
  private queue = new SerialQueue()
  private _isReady = false
  private _isBusy = false
  private _disposed = false
  private _metadata!: ModelMetadata

  private constructor(worker: Worker) {
    this.worker = worker
  }

  /** Load a GGUF model and return an initialized LlamaModel instance. */
  static async load(modelPath: string, config?: ModelConfig): Promise<LlamaModel> {
    const resolved = resolveConfig(config)
    const worker = new Worker(new URL('./worker/llm.worker.ts', import.meta.url).href)
    const instance = new LlamaModel(worker)

    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Model load timeout (120s)')), 120_000)

      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const msg = event.data
        if (msg.type === 'ready') {
          clearTimeout(timer)
          instance._metadata = msg.metadata
          instance._isReady = true
          resolve()
        } else if (msg.type === 'error') {
          clearTimeout(timer)
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

    return this.queue.enqueue(() => this.doInfer(prompt, options))
  }

  private doInfer(prompt: string, options: InferOptions): Promise<InferResult> {
    return new Promise<InferResult>((resolve, reject) => {
      this._isBusy = true
      const id = crypto.randomUUID()
      const chunks: string[] = []

      // SharedArrayBuffer for cross-thread abort signaling
      // (worker event loop is blocked during synchronous FFI calls)
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

      this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const msg = event.data

        switch (msg.type) {
          case 'token': {
            if (msg.id !== id) break
            chunks.push(msg.text)
            options.onToken(msg.text)
            break
          }
          case 'done': {
            if (msg.id !== id) break
            options.signal?.removeEventListener('abort', onAbort)
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
            if (msg.id !== id) break
            options.signal?.removeEventListener('abort', onAbort)
            this._isBusy = false
            resolve({
              text: chunks.join(''),
              tokenCount: chunks.length,
              aborted: true,
            })
            break
          }
          case 'error': {
            if (msg.id && msg.id !== id) break
            options.signal?.removeEventListener('abort', onAbort)
            this._isBusy = false
            reject(new Error(msg.message))
            break
          }
        }
      }

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
  get isBusy(): boolean { return this._isBusy }

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
      const timer = setTimeout(() => reject(new Error('getFimTokens timeout (5s)')), 5_000)

      this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const msg = event.data
        if (msg.type === 'fimTokens') {
          clearTimeout(timer)
          resolve(msg.data)
        } else if (msg.type === 'error') {
          clearTimeout(timer)
          reject(new Error(msg.message))
        }
      }
      const req: WorkerRequest = { type: 'getFimTokens' }
      this.worker.postMessage(req)
    })
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
