import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { join } from 'node:path'
import { LlamaModel } from '../src/index.ts'
import type { FimTokens } from '../src/index.ts'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

let llm: LlamaModel

beforeAll(async () => {
  llm = await LlamaModel.load(MODEL_PATH, { preset: 'small' })
}, 120_000)

afterAll(async () => {
  if (llm) await llm.dispose()
})

describe('getFimTokens', () => {
  test('returns FimTokens with valid token IDs', async () => {
    const fim = await llm.getFimTokens()

    expect(fim).toBeDefined()
    expect(typeof fim.pre).toBe('number')
    expect(typeof fim.suf).toBe('number')
    expect(typeof fim.mid).toBe('number')
    expect(typeof fim.pad).toBe('number')
    expect(typeof fim.rep).toBe('number')
    expect(typeof fim.sep).toBe('number')

    // Each token is either a valid ID (>= 0) or -1 (unsupported)
    for (const value of Object.values(fim)) {
      expect(value).toBeGreaterThanOrEqual(-1)
    }
  }, 60_000)
})
