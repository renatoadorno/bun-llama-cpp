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

describe('performance metrics', () => {
  test('returns metrics when metrics: true', async () => {
    const result = await llm.infer('Hello', {
      onToken: () => {},
      maxTokens: 10,
      metrics: true,
    })

    expect(result.metrics).toBeDefined()
    expect(result.metrics!.promptTokens).toBeGreaterThan(0)
    expect(result.metrics!.generatedTokens).toBeGreaterThan(0)
    expect(result.metrics!.promptMs).toBeGreaterThan(0)
    expect(result.metrics!.generateMs).toBeGreaterThan(0)
    expect(result.metrics!.tokensPerSec).toBeGreaterThan(0)
  }, 60_000)

  test('does not return metrics when metrics is omitted', async () => {
    const result = await llm.infer('Hello', {
      onToken: () => {},
      maxTokens: 10,
    })

    expect(result.metrics).toBeUndefined()
  }, 60_000)
})
