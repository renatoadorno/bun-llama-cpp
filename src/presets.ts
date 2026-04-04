import type { ModelConfig, Preset, ResolvedConfig, SamplerConfig } from './types.ts'

const DEFAULT_SAMPLER: SamplerConfig = {
  topK: 50,
  topP: 0.9,
  temp: 0.8,
  minP: 0.05,
  seed: 0xFFFFFFFF,
  repeatPenalty: 1.1,
  frequencyPenalty: 0.0,
  presencePenalty: 0.0,
  penaltyLastN: 64,
}

const PRESETS: Record<Preset, ResolvedConfig> = {
  small: {
    nCtx: 2048, nThreads: 4, nGpuLayers: 99, maxTokens: 256,
    sampler: { ...DEFAULT_SAMPLER },
    embeddings: false, poolingType: 0, nSeqMax: 1,
  },
  medium: {
    nCtx: 4096, nThreads: 8, nGpuLayers: 99, maxTokens: 512,
    sampler: { ...DEFAULT_SAMPLER },
    embeddings: false, poolingType: 0, nSeqMax: 1,
  },
  large: {
    nCtx: 8192, nThreads: 8, nGpuLayers: 99, maxTokens: 2048,
    sampler: { ...DEFAULT_SAMPLER },
    embeddings: false, poolingType: 0, nSeqMax: 1,
  },
}

function validateConfig(config: ResolvedConfig): void {
  if (!Number.isInteger(config.nSeqMax) || config.nSeqMax < 1 || config.nSeqMax > 128)
    throw new Error(`nSeqMax must be between 1 and 128 (got ${config.nSeqMax})`)
  if (!Number.isInteger(config.nCtx) || config.nCtx < 128 || config.nCtx > 131072)
    throw new Error(`nCtx must be between 128 and 131072 (got ${config.nCtx})`)
  if (!Number.isInteger(config.maxTokens) || config.maxTokens < 1 || config.maxTokens > 131072)
    throw new Error(`maxTokens must be between 1 and 131072 (got ${config.maxTokens})`)
  if (!Number.isInteger(config.nThreads) || config.nThreads < 1 || config.nThreads > 256)
    throw new Error(`nThreads must be between 1 and 256 (got ${config.nThreads})`)
}

export function resolveConfig(config?: ModelConfig): ResolvedConfig {
  const base = PRESETS[config?.preset ?? 'medium']
  const resolved: ResolvedConfig = {
    nCtx:       config?.nCtx       ?? base.nCtx,
    nThreads:   config?.nThreads   ?? base.nThreads,
    nGpuLayers: config?.nGpuLayers ?? base.nGpuLayers,
    maxTokens:  config?.maxTokens  ?? base.maxTokens,
    sampler: {
      ...base.sampler,
      ...config?.sampler,
    },
    embeddings:  config?.embeddings  ?? false,
    poolingType: config?.poolingType ?? 0,
    nSeqMax:     config?.nSeqMax     ?? 1,
  }
  validateConfig(resolved)
  return resolved
}
