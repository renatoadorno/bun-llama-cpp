import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { join } from 'node:path'
import { LlamaModel } from '../src/index.ts'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

let llm: LlamaModel

beforeAll(async () => {
  llm = await LlamaModel.load(MODEL_PATH, {
    preset: 'small',
    embeddings: true,
    poolingType: 1,  // MEAN — correct for decoder-only models used as embedders
  })
}, 120_000)

afterAll(async () => {
  if (llm) await llm.dispose()
})

describe('embed', () => {
  test('returns Float32Array with model embedding dimensions', async () => {
    const vector = await llm.embed('hello world')
    expect(vector).toBeInstanceOf(Float32Array)
    expect(vector.length).toBe(llm.metadata.nEmbd)
    expect(vector.length).toBeGreaterThan(0)
  }, 60_000)

  test('returns different vectors for different texts', async () => {
    const v1 = await llm.embed('the cat sat on the mat')
    const v2 = await llm.embed('quantum chromodynamics')
    expect(v1.length).toBe(v2.length)
    const allSame = v1.every((x, i) => x === v2[i])
    expect(allSame).toBe(false)
  }, 120_000)

  test('returns same vector on repeated calls for the same text', async () => {
    const v1 = await llm.embed('deterministic test')
    const v2 = await llm.embed('deterministic test')
    expect(v1.every((x, i) => x === v2[i])).toBe(true)
  }, 120_000)
})

describe('embedMany', () => {
  test('returns array of Float32Array with correct count and dimensions', async () => {
    const vectors = await llm.embedMany(['hello', 'world', 'foo'])
    expect(vectors).toHaveLength(3)
    for (const v of vectors) {
      expect(v).toBeInstanceOf(Float32Array)
      expect(v.length).toBe(llm.metadata.nEmbd)
    }
  }, 120_000)

  test('empty array returns empty result immediately', async () => {
    const vectors = await llm.embedMany([])
    expect(vectors).toHaveLength(0)
  }, 30_000)

  test('single-text batch matches embed() result', async () => {
    const single = await llm.embed('batch parity check')
    const batch = await llm.embedMany(['batch parity check'])
    expect(batch[0]!.every((x, i) => x === single[i])).toBe(true)
  }, 120_000)

  test('all vectors in a batch have the same length', async () => {
    const texts = ['short', 'a slightly longer sentence for testing', 'x']
    const vectors = await llm.embedMany(texts)
    const dim = vectors[0]!.length
    expect(vectors.every(v => v.length === dim)).toBe(true)
  }, 120_000)
})
