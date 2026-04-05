import { LlamaModel } from './model.ts'
import { ModelNotFoundError } from './errors.ts'
import type { ModelConfig, ModelStatus } from './types.ts'

export class ModelRegistry {
  private _models = new Map<string, LlamaModel>()
  private _loading = new Map<string, Promise<LlamaModel>>()
  private _statuses = new Map<string, Exclude<ModelStatus, 'unknown'>>()
  private _errors = new Map<string, string>()
  private _loadOrder: string[] = []

  /**
   * Load a model and register it under `name`.
   * Idempotent: if the model is already ready, resolves immediately.
   * Deduplicates concurrent calls: multiple load() calls for the same
   * name share one Worker, not N.
   */
  async load(name: string, path: string, config?: ModelConfig): Promise<void> {
    if (this._models.has(name)) return

    const existing = this._loading.get(name)
    if (existing) { await existing; return }

    this._statuses.set(name, 'loading')
    if (!this._loadOrder.includes(name)) this._loadOrder.push(name)

    const promise = LlamaModel.load(path, config)
    this._loading.set(name, promise)

    try {
      const model = await promise
      this._models.set(name, model)
      this._statuses.set(name, 'ready')
      this._errors.delete(name)
    } catch (e) {
      this._statuses.set(name, 'error')
      this._errors.set(name, String(e))
      throw e
    } finally {
      this._loading.delete(name)
    }
  }

  /**
   * Get a loaded model by name.
   * Throws ModelNotFoundError if not ready (unknown, loading, or error).
   */
  get(name: string): LlamaModel {
    const model = this._models.get(name)
    if (model) return model

    const status = this._statuses.get(name)
    if (status === 'loading') {
      throw new ModelNotFoundError(
        `Model '${name}' is still loading. Await registry.load('${name}', ...) before calling get().`,
      )
    }
    if (status === 'error') {
      const err = this._errors.get(name) ?? 'unknown error'
      throw new ModelNotFoundError(
        `Model '${name}' failed to load: ${err}. Call registry.load('${name}', path, config) again to retry.`,
      )
    }
    throw new ModelNotFoundError(
      `Model '${name}' not found. Call registry.load('${name}', path, config) first.`,
    )
  }

  /** Dispose and remove a model by name. No-op if not loaded. */
  async unload(name: string): Promise<void> {
    const model = this._models.get(name)
    if (model) {
      this._models.delete(name)
      this._statuses.delete(name)
      this._errors.delete(name)
      await model.dispose()
    }
  }

  /**
   * Dispose all registered models in reverse load order.
   * Safe to call on an empty registry.
   */
  async disposeAll(): Promise<void> {
    for (const name of [...this._loadOrder].reverse()) {
      await this.unload(name)
    }
    this._loadOrder = []
  }

  /**
   * Returns the current status of a model.
   * Returns 'unknown' for names that have never been passed to load().
   */
  status(name: string): ModelStatus {
    return this._statuses.get(name) ?? 'unknown'
  }
}
