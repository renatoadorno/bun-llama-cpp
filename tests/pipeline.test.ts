import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { join } from 'node:path'
import {
  LlamaModel,
  ModelPipeline,
  CapabilityMismatchError,
  assertCapability,
} from '../src/index.ts'

const GEN_PATH   = join(import.meta.dir, '..', 'models', 'qwen3-8b-q4_k_m.gguf')
const EMBED_PATH = join(import.meta.dir, '..', 'models', 'nomic-embed-text-v1.5.Q4_K_M.gguf')

let embedModel: LlamaModel
let genModel: LlamaModel

beforeAll(async () => {
  embedModel = await LlamaModel.load(EMBED_PATH, { preset: 'small', embeddings: true, poolingType: 1 })
  genModel   = await LlamaModel.load(GEN_PATH,   { preset: 'small' })
}, 240_000)

afterAll(async () => {
  if (embedModel) await embedModel.dispose()
  if (genModel)   await genModel.dispose()
})

// ── assertCapability ─────────────────────────────────────────────────

describe('assertCapability', () => {
  test('passes for embed model on embed capability', () => {
    expect(() => assertCapability(embedModel, 'embed')).not.toThrow()
  })

  test('passes for gen model on generate capability', () => {
    expect(() => assertCapability(genModel, 'generate')).not.toThrow()
  })

  test('throws CapabilityMismatchError when gen model used for embed', () => {
    let err: unknown
    try { assertCapability(genModel, 'embed') } catch (e) { err = e }
    expect(err).toBeInstanceOf(CapabilityMismatchError)
    expect((err as CapabilityMismatchError).code).toBe('CAPABILITY_MISMATCH')
    expect((err as CapabilityMismatchError).message).toContain('embeddings: true')
  })

  test('throws CapabilityMismatchError when embed model used for generate', () => {
    let err: unknown
    try { assertCapability(embedModel, 'generate') } catch (e) { err = e }
    expect(err).toBeInstanceOf(CapabilityMismatchError)
    expect((err as CapabilityMismatchError).message).toContain('embedding mode')
  })

  test('error message includes model name when provided', () => {
    let err: unknown
    try { assertCapability(genModel, 'embed', 'my-gen') } catch (e) { err = e }
    expect((err as CapabilityMismatchError).message).toContain("'my-gen'")
  })
})

// ── ModelPipeline constructor ────────────────────────────────────────

describe('ModelPipeline constructor', () => {
  test('constructs successfully with correct models', () => {
    expect(() => new ModelPipeline(embedModel, genModel)).not.toThrow()
  })

  test('throws CapabilityMismatchError when embed arg is a gen model', () => {
    expect(() => new ModelPipeline(genModel, genModel)).toThrow(CapabilityMismatchError)
  })

  test('throws CapabilityMismatchError when generate arg is an embed model', () => {
    expect(() => new ModelPipeline(embedModel, embedModel)).toThrow(CapabilityMismatchError)
  })
})

// ── ModelPipeline.embed() ────────────────────────────────────────────

describe('ModelPipeline.embed()', () => {
  let pipeline: ModelPipeline

  beforeAll(() => { pipeline = new ModelPipeline(embedModel, genModel) })

  test('returns Float32Array with positive length', async () => {
    const vec = await pipeline.embed('hello world')
    expect(vec).toBeInstanceOf(Float32Array)
    expect(vec.length).toBeGreaterThan(0)
  }, 30_000)
})

// ── ModelPipeline.rerank() ───────────────────────────────────────────

describe('ModelPipeline.rerank()', () => {
  let pipeline: ModelPipeline

  beforeAll(() => { pipeline = new ModelPipeline(embedModel, genModel) })

  test('returns empty array for empty docs', async () => {
    const results = await pipeline.rerank('query', [])
    expect(results).toHaveLength(0)
  })

  test('returns docs sorted by score descending', async () => {
    const query = 'search_query: machine learning algorithms'
    const docs = [
      'search_document: Delicious pasta recipes for dinner',
      'search_document: Introduction to gradient descent and neural networks',
    ]
    const results = await pipeline.rerank(query, docs)
    expect(results).toHaveLength(2)
    expect(results[0]!.score).toBeGreaterThanOrEqual(results[1]!.score)
    // The ML doc should rank higher than the pasta doc
    expect(results[0]!.doc).toBe(docs[1])
  }, 60_000)

  test('all scores are within cosine similarity range [-1, 1]', async () => {
    const results = await pipeline.rerank('test query', ['doc a', 'doc b', 'doc c'])
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(-1)
      expect(r.score).toBeLessThanOrEqual(1)
    }
  }, 60_000)
})

// ── ModelPipeline.generate() ─────────────────────────────────────────

describe('ModelPipeline.generate()', () => {
  let pipeline: ModelPipeline

  beforeAll(() => { pipeline = new ModelPipeline(embedModel, genModel) })

  test('returns non-empty text within maxTokens', async () => {
    const result = await pipeline.generate(
      'France is a country in Western Europe. Paris is its capital city.',
      'What is the capital of France?',
      { maxTokens: 20, onToken: () => {} },
    )
    expect(result.text.length).toBeGreaterThan(0)
    expect(result.tokenCount).toBeLessThanOrEqual(20)
    expect(result.aborted).toBe(false)
  }, 60_000)

  test('onToken streams tokens that join to match final text', async () => {
    const tokens: string[] = []
    const result = await pipeline.generate(
      'The speed of light is approximately 299,792,458 meters per second.',
      'How fast is light?',
      { maxTokens: 20, onToken: (t) => tokens.push(t) },
    )
    expect(tokens.join('')).toBe(result.text)
  }, 60_000)
})
