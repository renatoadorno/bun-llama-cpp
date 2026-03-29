import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { join } from 'node:path'
import { LlamaModel } from '../src/index.ts'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

let llm: LlamaModel

beforeAll(async () => {
  llm = await LlamaModel.load(MODEL_PATH, { preset: 'small' })
}, 120_000)

afterAll(async () => {
  if (llm) await llm.dispose()
})

describe('metadata', () => {
  test('metadata is populated after load()', () => {
    const meta = llm.metadata

    expect(meta).toBeDefined()
    expect(meta.nParams).toBeGreaterThan(0)
    expect(meta.nEmbd).toBeGreaterThan(0)
    expect(meta.nCtxTrain).toBeGreaterThan(0)
    expect(meta.nLayers).toBeGreaterThan(0)
    expect(meta.desc).toBeTruthy()
    expect(typeof meta.desc).toBe('string')
    expect(meta.sizeBytes).toBeGreaterThan(0)
  })

  test('metadata has expected values for Qwen3-8B', () => {
    const meta = llm.metadata

    expect(meta.nLayers).toBe(32)
    expect(meta.nEmbd).toBe(4096)
    expect(meta.desc).toContain('Q4_K')
  })
})
