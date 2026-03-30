import { join } from 'node:path'
import { arch, platform } from 'node:os'

export interface LibPaths {
  libllama: string
  libshims: string
}

const PLATFORM_PACKAGES: Record<string, string> = {
  'darwin-arm64': '@renatoadorno/bun-llama-cpp-darwin-arm64',
  'darwin-x64': '@renatoadorno/bun-llama-cpp-darwin-x64',
  'linux-x64': '@renatoadorno/bun-llama-cpp-linux-x64',
  'linux-arm64': '@renatoadorno/bun-llama-cpp-linux-arm64',
}

function getExtension(): string {
  return platform() === 'darwin' ? 'dylib' : 'so'
}

function tryPlatformPackage(): LibPaths | null {
  const key = `${platform()}-${arch()}`
  const pkg = PLATFORM_PACKAGES[key]
  if (!pkg) return null

  try {
    const pkgDir = require.resolve(`${pkg}/package.json`)
    const dir = join(pkgDir, '..')
    const ext = getExtension()
    return {
      libllama: join(dir, `libllama.${ext}`),
      libshims: join(dir, `libllama_shims.${ext}`),
    }
  } catch {
    return null
  }
}

function tryLocalDev(): LibPaths | null {
  const root = join(import.meta.dir, '..')
  const ext = getExtension()

  const libllama = join(root, `llama.cpp/build/bin/libllama.${ext}`)
  const libshims = join(root, `src/libllama_shims.${ext}`)

  const llamaExists = Bun.file(libllama).size > 0
  const shimsExists = Bun.file(libshims).size > 0

  if (llamaExists && shimsExists) return { libllama, libshims }
  return null
}

/**
 * Resolve native library paths.
 * Priority: platform package (npm install) → local dev (built from source)
 */
export function resolveLibPaths(): LibPaths {
  const fromPackage = tryPlatformPackage()
  if (fromPackage) return fromPackage

  const fromLocal = tryLocalDev()
  if (fromLocal) return fromLocal

  const key = `${platform()}-${arch()}`
  throw new Error(
    `bun-llama-cpp: No native binaries found for ${key}.\n` +
    `Install the platform package: bun add bun-llama-cpp-${key}\n` +
    `Or build from source: cd llama.cpp && cmake -B build && cmake --build build --config Release`
  )
}
