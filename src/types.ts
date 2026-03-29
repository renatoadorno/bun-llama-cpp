// ── Public types ─────────────────────────────────────────────────────

export type Preset = 'small' | 'medium' | 'large'

export interface SamplerConfig {
  topK: number
  topP: number
  temp: number
  minP: number
  seed: number
}

export interface ModelConfig {
  preset?: Preset
  nCtx?: number
  nThreads?: number
  nGpuLayers?: number
  maxTokens?: number
  sampler?: Partial<SamplerConfig>
}

export interface InferOptions {
  onToken: (text: string) => void
  maxTokens?: number
  signal?: AbortSignal
}

export interface InferResult {
  text: string
  tokenCount: number
  aborted: boolean
}

export interface FimTokens {
  pre: number
  suf: number
  mid: number
  pad: number
  rep: number
  sep: number
}

// ── Resolved config (all fields required, used internally) ──────────

export interface ResolvedConfig {
  nCtx: number
  nThreads: number
  nGpuLayers: number
  maxTokens: number
  sampler: SamplerConfig
}

// ── Worker protocol ─────────────────────────────────────────────────

export type WorkerRequest =
  | { type: 'init'; modelPath: string; config: ResolvedConfig }
  | { type: 'infer'; id: string; prompt: string; maxTokens: number; abortFlag: Int32Array }
  | { type: 'getFimTokens' }
  | { type: 'shutdown' }

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'token'; id: string; text: string }
  | { type: 'done'; id: string; tokenCount: number }
  | { type: 'aborted'; id: string }
  | { type: 'fimTokens'; data: FimTokens }
  | { type: 'error'; id?: string; message: string }
