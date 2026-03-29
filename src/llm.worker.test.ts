import { test, expect, afterAll } from 'bun:test'
import { join } from 'node:path'
import { LlamaModel } from './index.ts'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

let llm: LlamaModel

test('LlamaModel.load() initializes model within 120s', async () => {
  llm = await LlamaModel.load(MODEL_PATH, { preset: 'small' })
  expect(llm.isReady).toBe(true)
  expect(llm.isBusy).toBe(false)
}, 120_000)

test('infer() streams tokens and returns result', async () => {
  const tokens: string[] = []

  const result = await llm.infer('Hello', {
    onToken: (text) => tokens.push(text),
    maxTokens: 20,
  })

  expect(result.tokenCount).toBeGreaterThan(0)
  expect(result.aborted).toBe(false)
  expect(result.text).toBeTruthy()
  expect(tokens.length).toBeGreaterThan(0)
  expect(tokens.join('')).toBe(result.text)
}, 180_000)

test('infer() supports AbortSignal cancellation', async () => {
  const controller = new AbortController()
  const tokens: string[] = []

  // Abort after receiving a few tokens
  const result = await llm.infer('Tell me a long story about a dragon', {
    onToken: (text) => {
      tokens.push(text)
      if (tokens.length >= 5) controller.abort()
    },
    maxTokens: 500,
    signal: controller.signal,
  })

  expect(result.aborted).toBe(true)
  expect(tokens.length).toBeGreaterThanOrEqual(5)
  expect(tokens.length).toBeLessThan(500)
}, 180_000)

test('infer() with pre-aborted signal returns immediately', async () => {
  const controller = new AbortController()
  controller.abort()

  const result = await llm.infer('Hello', {
    onToken: () => { throw new Error('Should not receive tokens') },
    signal: controller.signal,
  })

  expect(result.aborted).toBe(true)
  expect(result.tokenCount).toBe(0)
  expect(result.text).toBe('')
})

afterAll(async () => {
  if (llm) await llm.dispose()
})
