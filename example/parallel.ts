import { LlamaModel } from '../src/index.ts'
import { join } from 'node:path'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

// ANSI
const C = ['\x1b[36m', '\x1b[33m', '\x1b[32m', '\x1b[35m']
const RST = '\x1b[0m'
const DIM = '\x1b[2m'
const BOLD = '\x1b[1m'

/**
 * Live multi-line renderer: each sequence gets its own line
 * that grows in-place as tokens arrive.
 */
class LiveStream {
  private lines: string[]
  private thinking: boolean[]
  private n: number
  private started = false
  private cols: number

  constructor(n: number) {
    this.n = n
    this.lines = Array(n).fill('')
    this.thinking = Array(n).fill(false)
    this.cols = process.stdout.columns || 80
  }

  handler(seq: number) {
    return (token: string) => {
      if (token.includes('<think>')) { this.thinking[seq] = true; return }
      if (token.includes('</think>')) { this.thinking[seq] = false; return }
      if (this.thinking[seq]) return
      this.lines[seq] += token.replace(/\n/g, ' ')
      this.render()
    }
  }

  private render() {
    if (this.started) process.stdout.write(`\x1b[${this.n}A`)
    this.started = true
    for (let i = 0; i < this.n; i++) {
      const color = C[i % C.length]
      const prefix = `[seq ${i + 1}] `
      const maxText = this.cols - prefix.length - 1
      const text = this.lines[i]!.length > maxText
        ? '...' + this.lines[i]!.slice(-(maxText - 3))
        : this.lines[i]
      process.stdout.write(`\x1b[2K${color}${prefix}${RST}${text}\n`)
    }
  }
}

console.log('Loading model with nSeqMax=4...')
const llm = await LlamaModel.load(MODEL_PATH, { preset: 'large', nSeqMax: 4 })
console.log('Model ready.\n')

// ── Example 1: Basic inferParallel ──────────────────────────────────
console.log(`${BOLD}=== Example 1: Basic Parallel Inference ===${RST}\n`)

let stream = new LiveStream(2)
const results = await llm.inferParallel([
  {
    prompt: await llm.applyTemplate([
      { role: 'user', content: 'Count from 1 to 5. /no_think' },
    ]),
    onToken: stream.handler(0),
    maxTokens: 50,
  },
  {
    prompt: await llm.applyTemplate([
      { role: 'user', content: 'Name 3 primary colors. /no_think' },
    ]),
    onToken: stream.handler(1),
    maxTokens: 50,
  },
])

console.log(`${DIM}  ${results[0]!.tokenCount} + ${results[1]!.tokenCount} tokens${RST}\n`)

// ── Example 2: Concurrent infer() via Promise.all ────────────────────
console.log(`${BOLD}=== Example 2: Concurrent infer() via Promise.all ===${RST}\n`)

stream = new LiveStream(2)
const [rA, rB] = await Promise.all([
  llm.infer(
    await llm.applyTemplate([{ role: 'user', content: 'Say "hello world". /no_think' }]),
    { onToken: stream.handler(0), maxTokens: 20 },
  ),
  llm.infer(
    await llm.applyTemplate([{ role: 'user', content: 'Say "goodbye world". /no_think' }]),
    { onToken: stream.handler(1), maxTokens: 20 },
  ),
])

console.log(`${DIM}  ${rA.tokenCount} + ${rB.tokenCount} tokens${RST}\n`)

// ── Example 3: Per-Sequence Abort ───────────────────────────────────
console.log(`${BOLD}=== Example 3: Per-Sequence Abort ===${RST}\n`)

const controller = new AbortController()
let abortTokenCount = 0

stream = new LiveStream(2)
const seq1Handler = stream.handler(0)
const abortResults = await llm.inferParallel([
  {
    prompt: await llm.applyTemplate([
      { role: 'user', content: 'Write a very long essay about nature. /no_think' },
    ]),
    onToken: (t) => {
      seq1Handler(t)
      if (!t.includes('<think>') && !t.includes('</think>') && t.trim() !== '') abortTokenCount++
      if (abortTokenCount >= 10 && !controller.signal.aborted) controller.abort()
    },
    maxTokens: 500,
    signal: controller.signal,
  },
  {
    prompt: await llm.applyTemplate([
      { role: 'user', content: 'Say "I completed successfully". /no_think' },
    ]),
    onToken: stream.handler(1),
    maxTokens: 30,
  },
])

console.log(`${DIM}  seq 1: aborted=${abortResults[0]!.aborted}, seq 2: aborted=${abortResults[1]!.aborted}${RST}\n`)

// ── Example 4: Warmup + Prefix Sharing ──────────────────────────────
// NOTE: warmup sets persistent state (_warmupTokens), so this must be
// the last example — all subsequent infer() calls expect the warmed
// prefix at the start of the prompt.
console.log(`${BOLD}=== Example 4: Warmup + Prefix Sharing ===${RST}\n`)

const systemPrompt = await llm.applyTemplate([
  { role: 'system', content: 'You are a math tutor. Answer concisely.' },
], { addAssistant: false })

const warmupTokens = await llm.warmup(systemPrompt)
console.log(`${DIM}Warmup: ${warmupTokens} tokens cached on seq 0${RST}\n`)

stream = new LiveStream(2)
const mathResults = await llm.inferParallel([
  {
    prompt: await llm.applyTemplate([
      { role: 'system', content: 'You are a math tutor. Answer concisely.' },
      { role: 'user', content: 'What is 12 * 8? /no_think' },
    ]),
    onToken: stream.handler(0),
    maxTokens: 30,
    metrics: true,
  },
  {
    prompt: await llm.applyTemplate([
      { role: 'system', content: 'You are a math tutor. Answer concisely.' },
      { role: 'user', content: 'What is the square root of 144? /no_think' },
    ]),
    onToken: stream.handler(1),
    maxTokens: 30,
    metrics: true,
  },
])

for (let i = 0; i < mathResults.length; i++) {
  const r = mathResults[i]!
  if (r.metrics) {
    console.log(`${DIM}  seq ${i + 1}: ${r.metrics.generatedTokens} tok, ${r.metrics.tokensPerSec.toFixed(1)} tok/s${RST}`)
  }
}

console.log('\nDone.')
await llm.dispose()
