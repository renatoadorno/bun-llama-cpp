import type { ModelConfig, Preset, ResolvedConfig, SamplerConfig } from './types.ts'

const DEFAULT_SAMPLER: SamplerConfig = {
  topK: 50,
  topP: 0.9,
  temp: 0.8,
  minP: 0.05,
  seed: 0xFFFFFFFF,
}

const PRESETS: Record<Preset, ResolvedConfig> = {
  small: {
    nCtx: 2048,
    nThreads: 4,
    nGpuLayers: 99,
    maxTokens: 256,
    sampler: { ...DEFAULT_SAMPLER },
  },
  medium: {
    nCtx: 4096,
    nThreads: 8,
    nGpuLayers: 99,
    maxTokens: 512,
    sampler: { ...DEFAULT_SAMPLER },
  },
  large: {
    nCtx: 8192,
    nThreads: 8,
    nGpuLayers: 99,
    maxTokens: 2048,
    sampler: { ...DEFAULT_SAMPLER },
  },
}

export function resolveConfig(config?: ModelConfig): ResolvedConfig {
  const base = PRESETS[config?.preset ?? 'medium']
  return {
    nCtx: config?.nCtx ?? base.nCtx,
    nThreads: config?.nThreads ?? base.nThreads,
    nGpuLayers: config?.nGpuLayers ?? base.nGpuLayers,
    maxTokens: config?.maxTokens ?? base.maxTokens,
    sampler: {
      ...base.sampler,
      ...config?.sampler,
    },
  }
}
