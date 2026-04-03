import { LlamaModel } from '../src/index.ts'
import { join } from 'node:path'

const MODEL_PATH = join(import.meta.dir, '..', 'models', 'nomic-embed-text-v1.5.Q4_K_M.gguf')

// nomic-embed-text-v1.5 is a BERT-style encoder model.
// Optional instruction prefixes improve retrieval quality:
//   "search_query: <text>"    — for query-side embeddings
//   "search_document: <text>" — for document-side embeddings
//   No prefix                 — for symmetric tasks (clustering, STS)

const llm = await LlamaModel.load(MODEL_PATH, {
  preset: 'small',
  embeddings: true,
  poolingType: 1, // MEAN — required for nomic-embed-text
})

console.log(`Model: ${llm.metadata.desc}`)
console.log(`Embedding dimensions: ${llm.metadata.nEmbd}`)
console.log()

// ── Cosine similarity (user-side utility — not part of the lib) ───────
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0
  for (let i = 0; i < a.length; i++) {
    dot   += a[i]! * b[i]!
    normA += a[i]! * a[i]!
    normB += b[i]! * b[i]!
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

// ── Example 1: Single embed ──────────────────────────────────────────
console.log('=== Single embed ===')
const query = 'search_query: What is the capital of France?'
const v = await llm.embed(query)
console.log(`"${query}"`)
console.log(`→ Float32Array[${v.length}]  first 4: [${Array.from(v.slice(0, 4)).map(x => x.toFixed(4)).join(', ')}]`)
console.log()

// ── Example 2: Batch embed + semantic similarity ─────────────────────
console.log('=== Semantic similarity ===')

const queryText = 'search_query: best way to learn machine learning'
const documents = [
  'search_document: Introduction to machine learning and neural networks',
  'search_document: Delicious pasta recipes for beginners',
  'search_document: Understanding gradient descent optimization',
  'search_document: How to grow tomatoes in a garden',
]

const [queryVec, ...docVecs] = await llm.embedMany([queryText, ...documents])

console.log(`Query: "${queryText}"`)
console.log()

const scores = documents.map((doc, i) => ({
  doc: doc.replace('search_document: ', ''),
  score: cosineSimilarity(queryVec!, docVecs[i]!),
}))
scores.sort((a, b) => b.score - a.score)

for (const { doc, score } of scores) {
  const bar = '█'.repeat(Math.round(score * 40))
  console.log(`  ${score.toFixed(4)}  ${bar}`)
  console.log(`         "${doc}"`)
  console.log()
}

// ── Example 3: Symmetric similarity (no prefix) ──────────────────────
console.log('=== Sentence similarity (symmetric, no prefix) ===')

const sentences = [
  'The cat sat on the mat.',
  'A feline rested on the rug.',
  'The dog ran through the park.',
]

const vecs = await llm.embedMany(sentences)

for (let i = 0; i < sentences.length; i++) {
  for (let j = i + 1; j < sentences.length; j++) {
    const sim = cosineSimilarity(vecs[i]!, vecs[j]!)
    console.log(`  [${sim.toFixed(4)}] "${sentences[i]}"  ↔  "${sentences[j]}"`)
  }
}

console.log()
await llm.dispose()
