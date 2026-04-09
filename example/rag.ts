/**
 * RAG example — demonstrates Phase 4 multi-model orchestration.
 *
 * Flow:
 *   1. Load embed + gen models via ModelRegistry
 *   2. Build ModelPipeline
 *   3. Rerank candidate documents
 *   4. Generate answer using top-k context
 *
 * The pipeline does NOT own a vector store — that stays with the consumer.
 */
import { join } from 'node:path'
import { ModelRegistry, ModelPipeline } from '../src/index.ts'

const EMBED_PATH = join(import.meta.dir, '..', 'models', 'nomic-embed-text-v1.5.Q4_K_M.gguf')
const GEN_PATH   = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')

// ── 1. Load models ───────────────────────────────────────────────────

const registry = new ModelRegistry()

console.log('Loading models...')
await registry.load('embed', EMBED_PATH, { preset: 'small', embeddings: true, poolingType: 1 })
await registry.load('gen',   GEN_PATH,   { preset: 'small' })

console.log(`embed status: ${registry.status('embed')}`)
console.log(`gen   status: ${registry.status('gen')}`)
console.log()

// ── 2. Build pipeline ────────────────────────────────────────────────

const pipeline = new ModelPipeline(registry.get('embed'), registry.get('gen'))

// ── 3. In-memory document store (consumer owns this) ─────────────────

const DOCS = [
  'search_document: The Eiffel Tower is located in Paris, France. It was built in 1889.',
  'search_document: The Great Wall of China stretches over 21,196 kilometers.',
  'search_document: Mount Everest is the highest mountain in the world at 8,848 meters.',
  'search_document: The Amazon River is the largest river by discharge in the world.',
  'search_document: Tokyo is the capital of Japan and one of the most populous cities.',
]

const QUERY = 'search_query: What is the tallest mountain on Earth?'

// ── 4. Rerank candidates ─────────────────────────────────────────────

console.log(`Query: "${QUERY}"`)
console.log()

console.log('Ranking documents...')
const ranked = await pipeline.rerank(QUERY, DOCS)

console.log('Ranked results:')
for (const { doc, score } of ranked) {
  console.log(`  [${score.toFixed(4)}] ${doc.replace('search_document: ', '')}`)
}
console.log()

// ── 5. Generate answer using top-3 context ───────────────────────────

const top3 = ranked.slice(0, 3).map(r => r.doc.replace('search_document: ', '')).join('\n')

console.log('Generating answer...')
process.stdout.write('Answer: ')

const result = await pipeline.generate(top3, 'What is the tallest mountain on Earth?', {
  maxTokens: 80,
  onToken: (t) => process.stdout.write(t),
})

console.log()
console.log()
console.log(`Tokens: ${result.tokenCount}  Aborted: ${result.aborted}`)

// ── 6. Cleanup ───────────────────────────────────────────────────────

await registry.disposeAll()
