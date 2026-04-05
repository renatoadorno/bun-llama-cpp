import type { LlamaModel } from './model.ts'

export type LlamaCppErrorCode =
  | 'CAPABILITY_MISMATCH'
  | 'MODEL_NOT_FOUND'
  | 'OUT_OF_MEMORY'  // reserved for future use

export class LlamaCppError extends Error {
  readonly code: LlamaCppErrorCode

  constructor(code: LlamaCppErrorCode, message: string) {
    super(message)
    this.name = 'LlamaCppError'
    this.code = code
  }
}

export class CapabilityMismatchError extends LlamaCppError {
  constructor(message: string) {
    super('CAPABILITY_MISMATCH', message)
    this.name = 'CapabilityMismatchError'
  }
}

export class ModelNotFoundError extends LlamaCppError {
  constructor(message: string) {
    super('MODEL_NOT_FOUND', message)
    this.name = 'ModelNotFoundError'
  }
}

/**
 * Assert that a model has the required capability.
 * Throws CapabilityMismatchError with an actionable message if not.
 *
 * @param model - The LlamaModel instance to check.
 * @param capability - 'embed' or 'generate'.
 * @param modelName - Optional registry name for clearer error messages.
 */
export function assertCapability(
  model: LlamaModel,
  capability: 'embed' | 'generate',
  modelName?: string,
): void {
  const nameClause = modelName ? ` '${modelName}'` : ''
  const caps = model.capabilities

  if (capability === 'embed' && !caps.isEmbedder) {
    throw new CapabilityMismatchError(
      `Model${nameClause} must be loaded with embeddings: true to embed text. ` +
        `Re-load with { embeddings: true, poolingType: 1 }.`,
    )
  }

  if (capability === 'generate' && caps.isEmbedder) {
    throw new CapabilityMismatchError(
      `Model${nameClause} is in embedding mode and cannot generate text. ` +
        `Load a generative model without embeddings: true.`,
    )
  }
}
