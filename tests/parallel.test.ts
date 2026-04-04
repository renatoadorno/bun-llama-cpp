import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { join } from 'node:path'
import { LlamaModel } from '../src/index.ts'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

let llm: LlamaModel

beforeAll(async () => {
  llm = await LlamaModel.load(MODEL_PATH, { preset: 'small', nSeqMax: 4 })
}, 120_000)

afterAll(async () => {
  if (llm) await llm.dispose()
})

describe('inferParallel', () => {
  test('runs 2 parallel inferences', async () => {
    const tokens1: string[] = []
    const tokens2: string[] = []

    const results = await llm.inferParallel([
      { prompt: 'Count 1 to 5: /no_think', onToken: (t) => tokens1.push(t), maxTokens: 30 },
      { prompt: 'Name 3 colors: /no_think', onToken: (t) => tokens2.push(t), maxTokens: 30 },
    ])

    expect(results).toHaveLength(2)
    expect(results[0]!.text.length).toBeGreaterThan(0)
    expect(results[1]!.text.length).toBeGreaterThan(0)
    expect(results[0]!.aborted).toBe(false)
    expect(results[1]!.aborted).toBe(false)
    expect(tokens1.join('')).toBe(results[0]!.text)
    expect(tokens2.join('')).toBe(results[1]!.text)
  }, 180_000)

  test('empty request array returns empty results', async () => {
    const results = await llm.inferParallel([])
    expect(results).toHaveLength(0)
  })

  test('per-sequence abort stops only that sequence', async () => {
    const controller = new AbortController()
    const tokens: string[] = []

    const results = await llm.inferParallel([
      {
        prompt: 'Write a very long essay about nature /no_think',
        onToken: (t) => { tokens.push(t); if (tokens.length >= 5) controller.abort() },
        maxTokens: 500,
        signal: controller.signal,
      },
      { prompt: 'Say hello /no_think', onToken: () => {}, maxTokens: 20 },
    ])

    expect(results[0]!.aborted).toBe(true)
    expect(results[1]!.aborted).toBe(false)
    expect(results[1]!.text.length).toBeGreaterThan(0)
  }, 180_000)

  test('rejects too many requests for nSeqMax', () => {
    const requests = Array.from({ length: 5 }, () => ({
      prompt: 'Test', onToken: () => {}, maxTokens: 10,
    }))
    expect(llm.inferParallel(requests)).rejects.toThrow('Too many parallel requests')
  })

  test('collects metrics in parallel mode', async () => {
    const results = await llm.inferParallel([
      { prompt: 'Hello /no_think', onToken: () => {}, maxTokens: 10, metrics: true },
    ])

    expect(results[0]!.metrics).toBeDefined()
    expect(results[0]!.metrics!.promptTokens).toBeGreaterThan(0)
    expect(results[0]!.metrics!.generatedTokens).toBeGreaterThan(0)
    expect(results[0]!.metrics!.tokensPerSec).toBeGreaterThan(0)
  }, 180_000)
})

describe('warmup + prefix sharing', () => {
  test('warmup returns positive token count', async () => {
    const count = await llm.warmup('You are a helpful assistant.')
    expect(count).toBeGreaterThan(0)
  }, 60_000)

  test('parallel infer after warmup produces output', async () => {
    const results = await llm.inferParallel([
      { prompt: 'What is 2+2? /no_think', onToken: () => {}, maxTokens: 20 },
      { prompt: 'What is 3+3? /no_think', onToken: () => {}, maxTokens: 20 },
    ])

    expect(results).toHaveLength(2)
    expect(results[0]!.text.length).toBeGreaterThan(0)
    expect(results[1]!.text.length).toBeGreaterThan(0)
  }, 180_000)
})

describe('concurrent infer (nSeqMax > 1)', () => {
  test('Promise.all with two infer() calls resolves both', async () => {
    const [r1, r2] = await Promise.all([
      llm.infer('Say alpha /no_think', { onToken: () => {}, maxTokens: 20 }),
      llm.infer('Say beta /no_think', { onToken: () => {}, maxTokens: 20 }),
    ])

    expect(r1.text.length).toBeGreaterThan(0)
    expect(r2.text.length).toBeGreaterThan(0)
    expect(r1.aborted).toBe(false)
    expect(r2.aborted).toBe(false)
  }, 180_000)
})

describe('error handling', () => {
  test('inferParallel on disposed model throws', async () => {
    const m = await LlamaModel.load(MODEL_PATH, { preset: 'small', nSeqMax: 2 })
    await m.dispose()
    expect(m.inferParallel([{ prompt: 'hi', onToken: () => {}, maxTokens: 10 }]))
      .rejects.toThrow('disposed')
  }, 120_000)

  test('warmup rejects while inferences are active', async () => {
    const inferPromise = llm.infer('Tell me a story /no_think', {
      onToken: () => {},
      maxTokens: 50,
    })
    // warmup should reject because _activeInfers > 0
    await expect(llm.warmup('system')).rejects.toThrow('active')
    await inferPromise
  }, 180_000)

  test('infer with empty prompt rejects', () => {
    expect(llm.infer('', { onToken: () => {}, maxTokens: 10 }))
      .rejects.toThrow('non-empty')
  })
})
