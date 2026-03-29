import { dlopen, FFIType } from 'bun:ffi'
import type { WorkerRequest, WorkerResponse } from '../types.ts'
import { openLibraries } from './ffi.ts'
import { initModel, runInference, cleanup, collectMetadata, type LlamaState } from './inference.ts'
import { resolveLibPaths } from '../lib-resolver.ts'

declare var self: Worker

const { libllama: LIBLLAMA, libshims: LIBSHIMS } = resolveLibPaths()

let libs: ReturnType<typeof openLibraries> | null = null
let state: LlamaState | null = null

// Redirect fd 2 (stderr) to /dev/null via libc dup/dup2.
// This silences ggml_metal and create_tensor logs that bypass llama_log_set.
const libc = dlopen('libSystem.B.dylib', {
  open:  { args: [FFIType.cstring, FFIType.i32], returns: FFIType.i32 },
  close: { args: [FFIType.i32],                  returns: FFIType.i32 },
  dup:   { args: [FFIType.i32],                  returns: FFIType.i32 },
  dup2:  { args: [FFIType.i32, FFIType.i32],     returns: FFIType.i32 },
})

let savedStderrFd = -1

function muteStderr() {
  savedStderrFd = libc.symbols.dup(2) as number
  const devnull = libc.symbols.open(Buffer.from('/dev/null\0'), 1) as number
  libc.symbols.dup2(devnull, 2)
  libc.symbols.close(devnull)
}

function restoreStderr() {
  if (savedStderrFd >= 0) {
    libc.symbols.dup2(savedStderrFd, 2)
    libc.symbols.close(savedStderrFd)
    savedStderrFd = -1
  }
}

function post(msg: WorkerResponse) {
  self.postMessage(msg)
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data

  switch (msg.type) {
    case 'init': {
      try {
        muteStderr()
        libs = openLibraries(LIBLLAMA, LIBSHIMS)
        state = initModel(libs.L, libs.S, msg.modelPath, msg.config)
        restoreStderr()
        const metadata = collectMetadata(libs.L, state.modelPtr)
        post({ type: 'ready', metadata })
      } catch (e) {
        restoreStderr()
        post({ type: 'error', message: String(e) })
      }
      break
    }

    case 'getFimTokens': {
      if (!libs || !state) {
        post({ type: 'error', message: 'Worker not initialized' })
        break
      }
      const vocab = state.vocabPtr
      post({
        type: 'fimTokens',
        data: {
          pre: libs.L.llama_vocab_fim_pre(vocab),
          suf: libs.L.llama_vocab_fim_suf(vocab),
          mid: libs.L.llama_vocab_fim_mid(vocab),
          pad: libs.L.llama_vocab_fim_pad(vocab),
          rep: libs.L.llama_vocab_fim_rep(vocab),
          sep: libs.L.llama_vocab_fim_sep(vocab),
        },
      })
      break
    }

    case 'applyTemplate': {
      if (!libs || !state) {
        post({ type: 'error', id: msg.id, message: 'Worker not initialized' })
        break
      }
      try {
        // Pack messages as null-separated pairs: "role\0content\0role\0content\0"
        const parts: string[] = []
        for (const m of msg.messages) {
          parts.push(m.role + '\0' + m.content + '\0')
        }
        const packed = Buffer.from(parts.join(''), 'utf8')

        // Template pointer: use model's built-in or null for chatml default
        const tmplPtr = state.chatTemplatePtr

        // First call: get required buffer size
        const needed = libs.S.shim_chat_apply_template(
          tmplPtr ? (tmplPtr as unknown as Buffer) : null,
          packed,
          msg.messages.length,
          msg.addAssistant,
          Buffer.alloc(0),
          0,
        )

        // Second call: fill buffer
        const buf = Buffer.alloc(needed + 1)
        libs.S.shim_chat_apply_template(
          tmplPtr ? (tmplPtr as unknown as Buffer) : null,
          packed,
          msg.messages.length,
          msg.addAssistant,
          buf,
          buf.length,
        )

        const text = buf.subarray(0, needed).toString('utf8')
        post({ type: 'templateResult', id: msg.id, text })
      } catch (e) {
        post({ type: 'error', id: msg.id, message: String(e) })
      }
      break
    }

    case 'infer': {
      if (!libs || !state) {
        post({ type: 'error', id: msg.id, message: 'Worker not initialized' })
        break
      }
      try {
        const abortFlag = msg.abortFlag
        muteStderr()
        const result = runInference(libs.L, libs.S, state, msg.prompt, msg.maxTokens, {
          onToken: (text) => post({ type: 'token', id: msg.id, text }),
          isAborted: () => Atomics.load(abortFlag, 0) === 1,
          collectMetrics: msg.collectMetrics,
        })
        restoreStderr()
        if (result.aborted) {
          post({ type: 'aborted', id: msg.id })
        } else {
          post({ type: 'done', id: msg.id, tokenCount: result.tokenCount, metrics: result.metrics })
        }
      } catch (e) {
        restoreStderr()
        post({ type: 'error', id: msg.id, message: String(e) })
      }
      break
    }

    case 'shutdown': {
      muteStderr()
      if (libs && state) {
        cleanup(libs.L, libs.S, state)
      }
      process.exit(0)
    }
  }
}
