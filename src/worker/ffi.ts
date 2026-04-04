import { dlopen, FFIType } from 'bun:ffi'

export interface LibLlama {
  llama_backend_init: () => void
  llama_backend_free: () => void
  llama_model_free: (model: number) => void
  llama_free: (ctx: number) => void
  llama_log_set: (fn: unknown, data: unknown) => void

  llama_model_get_vocab: (model: number) => number
  llama_vocab_bos: (vocab: number) => number
  llama_vocab_eos: (vocab: number) => number
  llama_vocab_eot: (vocab: number) => number
  llama_vocab_is_eog: (vocab: number, token: number) => boolean

  llama_vocab_fim_pre: (vocab: number) => number
  llama_vocab_fim_suf: (vocab: number) => number
  llama_vocab_fim_mid: (vocab: number) => number
  llama_vocab_fim_pad: (vocab: number) => number
  llama_vocab_fim_rep: (vocab: number) => number
  llama_vocab_fim_sep: (vocab: number) => number

  llama_model_n_params: (model: number) => number
  llama_model_n_embd: (model: number) => number
  llama_model_n_ctx_train: (model: number) => number
  llama_model_n_layer: (model: number) => number
  llama_model_size: (model: number) => number
  llama_model_desc: (model: number, buf: Buffer, bufSize: number) => number
  llama_model_chat_template: (model: number, name: number | null) => number

  llama_get_embeddings_seq: (ctx: number, seqId: number) => number
  llama_model_has_encoder: (model: number) => boolean

  llama_tokenize: (
    vocab: number, text: Buffer, textLen: number,
    tokens: Int32Array, nTokensMax: number,
    addSpecial: boolean, parseSpecial: boolean,
  ) => number
  llama_token_to_piece: (
    vocab: number, token: number, buf: Buffer,
    length: number, lstrip: number, special: boolean,
  ) => number

  llama_n_ctx: (ctx: number) => number
  llama_get_memory: (ctx: number) => number
  llama_memory_clear: (mem: number, partial: boolean) => void
  llama_memory_seq_rm: (mem: number, seqId: number, p0: number, p1: number) => boolean
  llama_memory_seq_cp: (mem: number, src: number, dst: number, p0: number, p1: number) => void
  llama_memory_seq_keep: (mem: number, seqId: number) => void

  llama_sampler_chain_add: (chain: number, sampler: number) => void
  llama_sampler_init_greedy: () => number
  llama_sampler_init_top_k: (k: number) => number
  llama_sampler_init_temp: (temp: number) => number
  llama_sampler_init_dist: (seed: number) => number
  llama_sampler_init_penalties: (penaltyLastN: number, repeatPenalty: number, frequencyPenalty: number, presencePenalty: number) => number
  llama_sampler_sample: (chain: number, ctx: number, idx: number) => number
  llama_sampler_accept: (chain: number, token: number) => void
  llama_sampler_reset: (chain: number) => void
  llama_sampler_free: (chain: number) => void

}

export interface LibShims {
  shim_sizeof_model_params: () => number
  shim_sizeof_context_params: () => number
  shim_sizeof_batch: () => number
  shim_sizeof_sampler_chain_params: () => number

  shim_model_default_params: (buf: Buffer) => void
  shim_context_default_params: (buf: Buffer) => void
  shim_sampler_chain_default_params: (buf: Buffer) => void

  shim_model_params_set_n_gpu_layers: (buf: Buffer, n: number) => void
  shim_ctx_params_set_n_ctx: (buf: Buffer, n: number) => void
  shim_ctx_params_set_n_threads: (buf: Buffer, n: number) => void
  shim_ctx_params_set_embeddings: (buf: Buffer, v: boolean) => void
  shim_ctx_params_set_pooling_type: (buf: Buffer, type: number) => void
  shim_ctx_params_set_n_seq_max: (buf: Buffer, n: number) => void
  shim_ctx_params_set_n_batch: (buf: Buffer, n: number) => void
  shim_encode: (ctx: number, buf: Buffer) => number

  shim_model_load_from_file: (path: Buffer, params: Buffer) => number
  shim_init_from_model: (model: number, params: Buffer) => number

  shim_batch_init: (buf: Buffer, nTokens: number, embd: number, nSeqMax: number) => void
  shim_batch_clear: (buf: Buffer) => void
  shim_batch_add: (buf: Buffer, id: number, pos: number, seqId: number, logits: boolean) => void
  shim_batch_free: (buf: Buffer) => void
  shim_decode: (ctx: number, buf: Buffer) => number

  shim_sampler_chain_init: (buf: Buffer) => number
  shim_sampler_init_top_p: (p: number, minKeep: number) => number
  shim_sampler_init_min_p: (p: number, minKeep: number) => number

  shim_chat_apply_template: (tmpl: number | null, messagesPacked: Buffer, nMsg: number, addAss: boolean, buf: Buffer, length: number) => number
}

/**
 * Opens libllama and the shim library, returning typed symbol accessors.
 * libllama MUST be opened first so the shim resolves symbols via dynamic_lookup.
 */
export function openLibraries(libLlamaPath: string, libShimsPath: string) {
  const { symbols: L } = dlopen(libLlamaPath, {
    llama_backend_init:      { args: [],                        returns: FFIType.void },
    llama_backend_free:      { args: [],                        returns: FFIType.void },
    llama_model_free:        { args: [FFIType.ptr],              returns: FFIType.void },
    llama_free:              { args: [FFIType.ptr],              returns: FFIType.void },
    llama_log_set:           { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.void },

    llama_model_get_vocab:   { args: [FFIType.ptr],              returns: FFIType.ptr  },
    llama_vocab_bos:         { args: [FFIType.ptr],              returns: FFIType.i32  },
    llama_vocab_eos:         { args: [FFIType.ptr],              returns: FFIType.i32  },
    llama_vocab_eot:         { args: [FFIType.ptr],              returns: FFIType.i32  },
    llama_vocab_is_eog:      { args: [FFIType.ptr, FFIType.i32], returns: FFIType.bool },

    llama_vocab_fim_pre: { args: [FFIType.ptr], returns: FFIType.i32 },
    llama_vocab_fim_suf: { args: [FFIType.ptr], returns: FFIType.i32 },
    llama_vocab_fim_mid: { args: [FFIType.ptr], returns: FFIType.i32 },
    llama_vocab_fim_pad: { args: [FFIType.ptr], returns: FFIType.i32 },
    llama_vocab_fim_rep: { args: [FFIType.ptr], returns: FFIType.i32 },
    llama_vocab_fim_sep: { args: [FFIType.ptr], returns: FFIType.i32 },

    llama_model_n_params:    { args: [FFIType.ptr],                              returns: FFIType.u64 },
    llama_model_n_embd:      { args: [FFIType.ptr],                              returns: FFIType.i32 },
    llama_model_n_ctx_train: { args: [FFIType.ptr],                              returns: FFIType.i32 },
    llama_model_n_layer:     { args: [FFIType.ptr],                              returns: FFIType.i32 },
    llama_model_size:        { args: [FFIType.ptr],                              returns: FFIType.u64 },
    llama_model_desc:        { args: [FFIType.ptr, FFIType.ptr, FFIType.u64],    returns: FFIType.i32 },
    llama_model_chat_template: { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.ptr },

    llama_get_embeddings_seq: { args: [FFIType.ptr, FFIType.i32], returns: FFIType.ptr  },
    llama_model_has_encoder:  { args: [FFIType.ptr],              returns: FFIType.bool },

    llama_tokenize: {
      args: [
        FFIType.ptr, FFIType.cstring, FFIType.i32,
        FFIType.ptr, FFIType.i32,
        FFIType.bool, FFIType.bool,
      ],
      returns: FFIType.i32,
    },
    llama_token_to_piece: {
      args: [
        FFIType.ptr, FFIType.i32, FFIType.ptr,
        FFIType.i32, FFIType.i32, FFIType.bool,
      ],
      returns: FFIType.i32,
    },

    llama_n_ctx:         { args: [FFIType.ptr],              returns: FFIType.u32  },
    llama_get_memory:    { args: [FFIType.ptr],              returns: FFIType.ptr  },
    llama_memory_clear:  { args: [FFIType.ptr, FFIType.bool], returns: FFIType.void },
    llama_memory_seq_rm:   { args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32], returns: FFIType.bool },
    llama_memory_seq_cp:   { args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.i32], returns: FFIType.void },
    llama_memory_seq_keep: { args: [FFIType.ptr, FFIType.i32], returns: FFIType.void },

    llama_sampler_chain_add:   { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.void },
    llama_sampler_init_greedy: { args: [],                         returns: FFIType.ptr  },
    llama_sampler_init_top_k:  { args: [FFIType.i32],              returns: FFIType.ptr  },
    llama_sampler_init_temp:   { args: [FFIType.f32],              returns: FFIType.ptr  },
    llama_sampler_init_dist:       { args: [FFIType.u32],              returns: FFIType.ptr  },
    llama_sampler_init_penalties:  { args: [FFIType.i32, FFIType.f32, FFIType.f32, FFIType.f32], returns: FFIType.ptr },
    llama_sampler_sample:          { args: [FFIType.ptr, FFIType.ptr, FFIType.i32], returns: FFIType.i32 },
    llama_sampler_accept:      { args: [FFIType.ptr, FFIType.i32], returns: FFIType.void },
    llama_sampler_reset:       { args: [FFIType.ptr],              returns: FFIType.void },
    llama_sampler_free:        { args: [FFIType.ptr],              returns: FFIType.void },

  })

  const { symbols: S } = dlopen(libShimsPath, {
    shim_sizeof_model_params:         { args: [], returns: FFIType.u64 },
    shim_sizeof_context_params:       { args: [], returns: FFIType.u64 },
    shim_sizeof_batch:                { args: [], returns: FFIType.u64 },
    shim_sizeof_sampler_chain_params: { args: [], returns: FFIType.u64 },

    shim_model_default_params:         { args: [FFIType.ptr], returns: FFIType.void },
    shim_context_default_params:       { args: [FFIType.ptr], returns: FFIType.void },
    shim_sampler_chain_default_params: { args: [FFIType.ptr], returns: FFIType.void },

    shim_model_params_set_n_gpu_layers: { args: [FFIType.ptr, FFIType.i32], returns: FFIType.void },
    shim_ctx_params_set_n_ctx:          { args: [FFIType.ptr, FFIType.u32], returns: FFIType.void },
    shim_ctx_params_set_n_threads:      { args: [FFIType.ptr, FFIType.i32], returns: FFIType.void },
    shim_ctx_params_set_embeddings:     { args: [FFIType.ptr, FFIType.bool], returns: FFIType.void },
    shim_ctx_params_set_pooling_type:   { args: [FFIType.ptr, FFIType.i32],  returns: FFIType.void },
    shim_ctx_params_set_n_seq_max:      { args: [FFIType.ptr, FFIType.i32],  returns: FFIType.void },
    shim_ctx_params_set_n_batch:        { args: [FFIType.ptr, FFIType.i32],  returns: FFIType.void },
    shim_encode:                        { args: [FFIType.ptr, FFIType.ptr],  returns: FFIType.i32  },

    shim_model_load_from_file: { args: [FFIType.cstring, FFIType.ptr], returns: FFIType.ptr },
    shim_init_from_model:      { args: [FFIType.ptr,     FFIType.ptr], returns: FFIType.ptr },

    shim_batch_init:  { args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32], returns: FFIType.void },
    shim_batch_clear: { args: [FFIType.ptr],                                          returns: FFIType.void },
    shim_batch_add:   {
      args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.i32, FFIType.bool],
      returns: FFIType.void,
    },
    shim_batch_free:  { args: [FFIType.ptr],              returns: FFIType.void },
    shim_decode:      { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.i32  },

    shim_sampler_chain_init: { args: [FFIType.ptr],              returns: FFIType.ptr },
    shim_sampler_init_top_p: { args: [FFIType.f32, FFIType.i32], returns: FFIType.ptr },
    shim_sampler_init_min_p: { args: [FFIType.f32, FFIType.i32], returns: FFIType.ptr },

    shim_chat_apply_template: {
      args: [FFIType.ptr, FFIType.ptr, FFIType.i32, FFIType.bool, FFIType.ptr, FFIType.i32],
      returns: FFIType.i32,
    },
  })

  // Cast to typed interfaces — the `as unknown as number` pattern is required by bun:ffi
  return {
    L: L as unknown as LibLlama,
    S: S as unknown as LibShims,
  }
}
