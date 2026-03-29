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

describe('repetition penalties', () => {
  test('penalties config is accepted and affects output', async () => {
    const prompt = 'Repeat the word "hello" as many times as you can:'

    // Run without penalties (disabled)
    const noPenalty = await LlamaModel.load(MODEL_PATH, {
      preset: 'small',
      sampler: { repeatPenalty: 1.0, frequencyPenalty: 0.0, presencePenalty: 0.0 },
    })
    const resultNoPenalty = await noPenalty.infer(prompt, {
      onToken: () => {},
      maxTokens: 50,
    })
    await noPenalty.dispose()

    // Run with strong penalties
    const withPenalty = await LlamaModel.load(MODEL_PATH, {
      preset: 'small',
      sampler: { repeatPenalty: 2.0, frequencyPenalty: 1.0, presencePenalty: 1.0 },
    })
    const resultWithPenalty = await withPenalty.infer(prompt, {
      onToken: () => {},
      maxTokens: 50,
    })
    await withPenalty.dispose()

    // With strong penalties, output should be different (less repetitive)
    expect(resultNoPenalty.text).not.toBe(resultWithPenalty.text)
  }, 300_000)
})
