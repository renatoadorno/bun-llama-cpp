import { dlopen, FFIType } from 'bun:ffi'
import type { WorkerRequest, WorkerResponse } from '../types.ts'
import { openLibraries } from './ffi.ts'
import { initModel, runInference, cleanup, type LlamaState } from './inference.ts'
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
        post({ type: 'ready' })
      } catch (e) {
        restoreStderr()
        post({ type: 'error', message: String(e) })
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
        })
        restoreStderr()
        if (result.aborted) {
          post({ type: 'aborted', id: msg.id })
        } else {
          post({ type: 'done', id: msg.id, tokenCount: result.tokenCount })
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
