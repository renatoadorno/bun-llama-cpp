import { LlamaModel } from '../src/index.ts'
import { join } from 'node:path'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

// Qwen3 chat template — /no_think disables thinking mode
const PROMPT = `<|im_start|>user
Explain Bun FFI in 3 sentences. /no_think<|im_end|>
<|im_start|>assistant
`

const llm = await LlamaModel.load(MODEL_PATH, { preset: 'medium' })

const result = await llm.infer(PROMPT, {
  onToken: (text) => process.stdout.write(text),
  maxTokens: 200,
})

process.stdout.write('\n')
console.log(`[${result.tokenCount} tokens, aborted: ${result.aborted}]`)

await llm.dispose()
