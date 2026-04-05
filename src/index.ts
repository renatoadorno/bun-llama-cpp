export { LlamaModel } from './model.ts'
export type { ParallelInferOptions } from './model.ts'
export { resolveConfig } from './presets.ts'
export type {
  Preset,
  ModelConfig,
  SamplerConfig,
  InferOptions,
  InferResult,
  InferMetrics,
  FimTokens,
  ModelMetadata,
  ChatMessage,
  ParallelInferResult,
} from './types.ts'
export { ModelRegistry } from './registry.ts'
export { LlamaCppError, CapabilityMismatchError, ModelNotFoundError, assertCapability } from './errors.ts'
export type { LlamaCppErrorCode } from './errors.ts'
export type {
  ModelCapabilities,
  ModelStatus,
  RerankResult,
  PipelineGenerateOptions,
  PipelineGenerateResult,
} from './types.ts'
