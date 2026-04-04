// ── Public types ─────────────────────────────────────────────────────

export type Preset = 'small' | 'medium' | 'large'

export interface SamplerConfig {
  topK: number
  topP: number
  temp: number
  minP: number
  seed: number
  repeatPenalty?: number
  frequencyPenalty?: number
  presencePenalty?: number
  penaltyLastN?: number
}

export interface ModelConfig {
  preset?: Preset
  nCtx?: number
  nThreads?: number
  nGpuLayers?: number
  maxTokens?: number
  sampler?: Partial<SamplerConfig>
  embeddings?: boolean
  poolingType?: 0 | 1 | 2 | 3 | 4  // UNSPECIFIED=0 MEAN=1 CLS=2 LAST=3 RANK=4
  nSeqMax?: number  // max parallel sequences (default: 1)
}

export interface InferOptions {
  onToken: (text: string) => void
  maxTokens?: number
  signal?: AbortSignal
  metrics?: boolean
}

export interface InferResult {
  text: string
  tokenCount: number
  aborted: boolean
  metrics?: InferMetrics
}

export interface FimTokens {
  pre: number
  suf: number
  mid: number
  pad: number
  rep: number
  sep: number
}

export interface ModelMetadata {
  nParams: number
  nEmbd: number
  nCtxTrain: number
  nLayers: number
  desc: string
  sizeBytes: number
}

export interface InferMetrics {
  promptTokens: number
  generatedTokens: number
  promptMs: number
  generateMs: number
  tokensPerSec: number
}

export interface ChatMessage {
  role: string
  content: string
}

// ── Resolved config (all fields required, used internally) ──────────

export interface ResolvedConfig {
  nCtx: number
  nThreads: number
  nGpuLayers: number
  maxTokens: number
  sampler: SamplerConfig
  embeddings: boolean
  poolingType: number
  nSeqMax: number
}

// ── Worker protocol ─────────────────────────────────────────────────

export type WorkerRequest =
  | { type: 'init'; modelPath: string; config: ResolvedConfig }
  | { type: 'infer'; id: string; prompt: string; maxTokens: number; abortFlag: Int32Array; collectMetrics: boolean }
  | { type: 'getFimTokens' }
  | { type: 'applyTemplate'; id: string; messages: ChatMessage[]; addAssistant: boolean }
  | { type: 'embed';      id: string; text: string }
  | { type: 'embedBatch'; id: string; texts: string[] }
  | { type: 'inferParallel'; id: string; requests: ParallelInferRequest[] }
  | { type: 'warmup'; id: string; systemPrompt: string }
  | { type: 'shutdown' }

export interface ParallelInferRequest {
  prompt: string
  maxTokens: number
  priority?: number
  abortFlag: Int32Array
  collectMetrics?: boolean
}

export interface ParallelInferResult {
  text: string
  tokenCount: number
  aborted: boolean
  metrics?: InferMetrics
}

export type WorkerResponse =
  | { type: 'ready'; metadata: ModelMetadata }
  | { type: 'token'; id: string; text: string }
  | { type: 'done'; id: string; tokenCount: number; metrics?: InferMetrics }
  | { type: 'aborted'; id: string }
  | { type: 'fimTokens'; data: FimTokens }
  | { type: 'templateResult'; id: string; text: string }
  | { type: 'embedResult';      id: string; vector: Float32Array }
  | { type: 'embedBatchResult'; id: string; vectors: Float32Array[] }
  | { type: 'inferParallelResult'; id: string; results: ParallelInferResult[] }
  | { type: 'warmupDone'; id: string; tokenCount: number }
  | { type: 'parallelToken'; id: string; seqIndex: number; text: string }
  | { type: 'error'; id?: string; message: string }
