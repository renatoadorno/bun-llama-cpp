import { LlamaModel } from '../src/index.ts'
import { join } from 'node:path'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

console.log('Loading model with nSeqMax=4...')
const llm = await LlamaModel.load(MODEL_PATH, { preset: 'large', nSeqMax: 4 })
console.log('Model ready.\n')

// ── Example 1: Basic inferParallel ──────────────────────────────────
console.log('=== Example 1: Basic Parallel Inference ===\n')

const results = await llm.inferParallel([
  {
    prompt: await llm.applyTemplate([
      { role: 'user', content: 'Count from 1 to 5. /no_think' },
    ]),
    onToken: () => {},
    maxTokens: 50,
  },
  {
    prompt: await llm.applyTemplate([
      { role: 'user', content: 'Name 3 primary colors. /no_think' },
    ]),
    onToken: () => {},
    maxTokens: 50,
  },
])

for (let i = 0; i < results.length; i++) {
  console.log(`Result ${i + 1}: ${results[i]!.text.trim()}`)
  console.log(`  [${results[i]!.tokenCount} tokens, aborted: ${results[i]!.aborted}]\n`)
}

// ── Example 2: Warmup + Prefix Sharing ──────────────────────────────
console.log('=== Example 2: Warmup + Prefix Sharing ===\n')

const systemPrompt = await llm.applyTemplate([
  { role: 'system', content: 'You are a math tutor. Answer concisely.' },
], { addAssistant: false })

const warmupTokens = await llm.warmup(systemPrompt)
console.log(`Warmup: ${warmupTokens} tokens cached on seq 0\n`)

const mathResults = await llm.inferParallel([
  {
    prompt: await llm.applyTemplate([
      { role: 'system', content: 'You are a math tutor. Answer concisely.' },
      { role: 'user', content: 'What is 12 × 8? /no_think' },
    ]),
    onToken: () => {},
    maxTokens: 30,
    metrics: true,
  },
  {
    prompt: await llm.applyTemplate([
      { role: 'system', content: 'You are a math tutor. Answer concisely.' },
      { role: 'user', content: 'What is the square root of 144? /no_think' },
    ]),
    onToken: () => {},
    maxTokens: 30,
    metrics: true,
  },
])

for (let i = 0; i < mathResults.length; i++) {
  const r = mathResults[i]!
  console.log(`Result ${i + 1}: ${r.text.trim()}`)
  if (r.metrics) {
    console.log(`  [prompt: ${r.metrics.promptTokens} tok, generated: ${r.metrics.generatedTokens} tok, ${r.metrics.tokensPerSec.toFixed(1)} tok/s]\n`)
  }
}

// ── Example 3: Concurrent infer() via Promise.all ────────────────────
console.log('=== Example 3: Concurrent infer() via Promise.all ===\n')

const [rA, rB] = await Promise.all([
  llm.infer(
    await llm.applyTemplate([{ role: 'user', content: 'Say "hello world". /no_think' }]),
    { onToken: () => {}, maxTokens: 20 },
  ),
  llm.infer(
    await llm.applyTemplate([{ role: 'user', content: 'Say "goodbye world". /no_think' }]),
    { onToken: () => {}, maxTokens: 20 },
  ),
])

console.log(`A: ${rA.text.trim()}`)
console.log(`B: ${rB.text.trim()}\n`)

// ── Example 4: Per-Sequence Abort ───────────────────────────────────
console.log('=== Example 4: Per-Sequence Abort ===\n')

const controller = new AbortController()
const abortResults = await llm.inferParallel([
  {
    prompt: await llm.applyTemplate([
      { role: 'user', content: 'Write a very long essay about nature. /no_think' },
    ]),
    onToken: () => {
      if (!controller.signal.aborted) controller.abort()
    },
    maxTokens: 500,
    signal: controller.signal,
  },
  {
    prompt: await llm.applyTemplate([
      { role: 'user', content: 'Say "I completed successfully". /no_think' },
    ]),
    onToken: () => {},
    maxTokens: 30,
  },
])

console.log(`Seq 1 (aborted): ${abortResults[0]!.aborted} — "${abortResults[0]!.text.trim().slice(0, 50)}..."`)
console.log(`Seq 2 (completed): ${abortResults[1]!.aborted} — "${abortResults[1]!.text.trim()}"`)

console.log('\nDone.')
await llm.dispose()
