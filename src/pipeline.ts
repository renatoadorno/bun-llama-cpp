import { LlamaModel } from './model.ts'
import { assertCapability } from './errors.ts'
import type { RerankResult, PipelineGenerateOptions, PipelineGenerateResult } from './types.ts'

export class ModelPipeline {
  constructor(
    private embedModel: LlamaModel,
    private generateModel: LlamaModel,
  ) {
    assertCapability(embedModel, 'embed')
    assertCapability(generateModel, 'generate')
  }

  /** Embed text using the embed model. */
  async embed(text: string): Promise<Float32Array> {
    return this.embedModel.embed(text)
  }

  /**
   * Rank docs by relevance to query using bi-encoder cosine similarity.
   * Returns docs sorted by score descending (most relevant first).
   * Provide candidate strings pre-filtered from your vector store.
   * For large doc sets (1000+), pre-filter to top-k before calling this.
   */
  async rerank(query: string, docs: string[]): Promise<RerankResult[]> {
    if (docs.length === 0) return []

    const [queryVec, ...docVecs] = await this.embedModel.embedMany([query, ...docs])

    return docs
      .map((doc, i) => ({
        doc,
        score: ModelPipeline.cosineSim(queryVec!, docVecs[i]!),
      }))
      .sort((a, b) => b.score - a.score)
  }

  /**
   * Generate a response with the generate model.
   * Callers assemble `context` from rerank() output — this method does not
   * access a vector store internally.
   *
   * Built-in prompt format:
   *   Context:\n{context}\n\nQuestion: {question}\n\nAnswer:
   *
   * For custom prompt templates, call generateModel.infer() directly.
   */
  async generate(
    context: string,
    question: string,
    options?: PipelineGenerateOptions,
  ): Promise<PipelineGenerateResult> {
    const prompt = `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer:`
    const result = await this.generateModel.infer(prompt, {
      onToken: options?.onToken ?? (() => {}),
      maxTokens: options?.maxTokens,
      signal: options?.signal,
    })
    return {
      text: result.text,
      tokenCount: result.tokenCount,
      aborted: result.aborted,
    }
  }

  private static cosineSim(a: Float32Array, b: Float32Array): number {
    let dot = 0, normA = 0, normB = 0
    for (let i = 0; i < a.length; i++) {
      dot   += a[i]! * b[i]!
      normA += a[i]! * a[i]!
      normB += b[i]! * b[i]!
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB))
  }
}
