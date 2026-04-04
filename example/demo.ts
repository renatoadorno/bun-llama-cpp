import { LlamaModel } from '../src/index.ts'
import { join } from 'node:path'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

const llm = await LlamaModel.load(MODEL_PATH, { preset: 'large' })

// Use model's built-in chat template instead of hardcoding
const prompt = await llm.applyTemplate([
  { role: 'user', content: 'explique Bun FFI de forma detalhada.' },
])

const result = await llm.infer(prompt, {
  onToken: (text) => process.stdout.write(text),
  maxTokens: 800,
})

process.stdout.write('\n')
console.log(`[${result.tokenCount} tokens, aborted: ${result.aborted}]`)

await llm.dispose()
