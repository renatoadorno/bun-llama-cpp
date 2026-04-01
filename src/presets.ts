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
    embeddings: false, poolingType: 0, nSeqMax: 64,
  },
  medium: {
    nCtx: 4096, nThreads: 8, nGpuLayers: 99, maxTokens: 512,
    sampler: { ...DEFAULT_SAMPLER },
    embeddings: false, poolingType: 0, nSeqMax: 64,
  },
  large: {
    nCtx: 8192, nThreads: 8, nGpuLayers: 99, maxTokens: 2048,
    sampler: { ...DEFAULT_SAMPLER },
    embeddings: false, poolingType: 0, nSeqMax: 64,
  },
}

export function resolveConfig(config?: ModelConfig): ResolvedConfig {
  const base = PRESETS[config?.preset ?? 'medium']
  return {
    nCtx:       config?.nCtx       ?? base.nCtx,
    nThreads:   config?.nThreads   ?? base.nThreads,
    nGpuLayers: config?.nGpuLayers ?? base.nGpuLayers,
    maxTokens:  config?.maxTokens  ?? base.maxTokens,
    sampler: {
      ...base.sampler,
      ...config?.sampler,
    },
    embeddings:  config?.embeddings  ?? false,
    poolingType: config?.poolingType ?? 0,   // UNSPECIFIED — callers set explicit type
    nSeqMax:     config?.nSeqMax     ?? 64,
  }
}
