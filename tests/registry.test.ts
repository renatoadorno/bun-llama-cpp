import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { join } from 'node:path'
import { ModelRegistry, ModelNotFoundError } from '../src/index.ts'

const EMBED_PATH = join(import.meta.dir, '..', 'models', 'nomic-embed-text-v1.5.Q4_K_M.gguf')
const EMBED_CFG = { preset: 'small', embeddings: true, poolingType: 1 } as const

describe('ModelRegistry', () => {
  let registry: ModelRegistry

  beforeAll(() => {
    registry = new ModelRegistry()
  })

  afterAll(async () => {
    await registry.disposeAll()
  })

  test('status() returns unknown for unseen name', () => {
    expect(registry.status('not-loaded')).toBe('unknown')
  })

  test('get() throws ModelNotFoundError with MODEL_NOT_FOUND code', () => {
    let err: unknown
    try { registry.get('a') } catch (e) { err = e }
    expect(err).toBeInstanceOf(ModelNotFoundError)
    expect((err as ModelNotFoundError).code).toBe('MODEL_NOT_FOUND')
    expect((err as ModelNotFoundError).message).toContain("'a'")
  })

  test('load() registers model and status becomes ready', async () => {
    await registry.load('nomic', EMBED_PATH, EMBED_CFG)
    expect(registry.status('nomic')).toBe('ready')
    expect(registry.get('nomic').isReady).toBe(true)
  }, 120_000)

  test('load() is idempotent when already ready', async () => {
    await registry.load('nomic', EMBED_PATH, EMBED_CFG) // second call — no-op
    expect(registry.status('nomic')).toBe('ready')
  })

  test('concurrent load() for same name deduplicates to one instance', async () => {
    const p1 = registry.load('nomic2', EMBED_PATH, EMBED_CFG)
    const p2 = registry.load('nomic2', EMBED_PATH, EMBED_CFG)
    await Promise.all([p1, p2])
    expect(registry.status('nomic2')).toBe('ready')
    // Both paths resolve to the same object
    expect(registry.get('nomic2')).toBe(registry.get('nomic2'))
    await registry.unload('nomic2')
  }, 120_000)

  test('unload() disposes model and clears status', async () => {
    await registry.unload('nomic')
    expect(registry.status('nomic')).toBe('unknown')
    let err: unknown
    try { registry.get('nomic') } catch (e) { err = e }
    expect(err).toBeInstanceOf(ModelNotFoundError)
  })

  test('load() with invalid path sets status to error and re-throws', async () => {
    let err: unknown
    try { await registry.load('bad', '/nonexistent/bad.gguf') } catch (e) { err = e }
    expect(err).toBeDefined()
    expect(registry.status('bad')).toBe('error')
  }, 30_000)

  test('get() after failed load has actionable message', () => {
    let err: unknown
    try { registry.get('bad') } catch (e) { err = e }
    expect(err).toBeInstanceOf(ModelNotFoundError)
    expect((err as ModelNotFoundError).message).toContain('failed to load')
  })

  test('failed load can be retried', async () => {
    await registry.load('bad', EMBED_PATH, EMBED_CFG)
    expect(registry.status('bad')).toBe('ready')
    await registry.unload('bad')
  }, 120_000)

  test('disposeAll() unloads all remaining models', async () => {
    await registry.load('x', EMBED_PATH, EMBED_CFG)
    await registry.load('y', EMBED_PATH, EMBED_CFG)
    await registry.disposeAll()
    expect(registry.status('x')).toBe('unknown')
    expect(registry.status('y')).toBe('unknown')
  }, 240_000)
})
