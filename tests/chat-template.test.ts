import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { join } from 'node:path'
import { LlamaModel } from '../src/index.ts'
import type { ChatMessage } from '../src/index.ts'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

let llm: LlamaModel

beforeAll(async () => {
  llm = await LlamaModel.load(MODEL_PATH, { preset: 'small' })
}, 120_000)

afterAll(async () => {
  if (llm) await llm.dispose()
})

describe('applyTemplate', () => {
  test('formats single user message', async () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Hello!' },
    ]
    const result = await llm.applyTemplate(messages)

    expect(result).toBeTruthy()
    expect(result).toContain('Hello!')
    expect(result).toContain('user')
  }, 60_000)

  test('formats multi-turn conversation', async () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello!' },
      { role: 'user', content: 'How are you?' },
    ]
    const result = await llm.applyTemplate(messages)

    expect(result).toContain('system')
    expect(result).toContain('You are helpful.')
    expect(result).toContain('Hi')
    expect(result).toContain('Hello!')
    expect(result).toContain('How are you?')
  }, 60_000)

  test('addAssistant: false omits assistant prefix', async () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Hello!' },
    ]

    const withAssistant = await llm.applyTemplate(messages, { addAssistant: true })
    const withoutAssistant = await llm.applyTemplate(messages, { addAssistant: false })

    // With assistant should be longer (includes assistant turn prefix)
    expect(withAssistant.length).toBeGreaterThan(withoutAssistant.length)
  }, 60_000)

  test('Qwen3 model produces expected template markers', async () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Test' },
    ]
    const result = await llm.applyTemplate(messages)

    // Qwen3 uses ChatML-style markers
    expect(result).toContain('<|im_start|>')
    expect(result).toContain('<|im_end|>')
  }, 60_000)
})
