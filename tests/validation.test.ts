import { describe, test, expect } from 'bun:test'
import { resolveConfig } from '../src/presets.ts'

describe('resolveConfig validation', () => {
  test('rejects nSeqMax = 0', () => {
    expect(() => resolveConfig({ nSeqMax: 0 })).toThrow('nSeqMax must be between 1 and 128')
  })

  test('rejects negative nSeqMax', () => {
    expect(() => resolveConfig({ nSeqMax: -1 })).toThrow('nSeqMax must be between 1 and 128')
  })

  test('rejects nSeqMax > 128', () => {
    expect(() => resolveConfig({ nSeqMax: 200 })).toThrow('nSeqMax must be between 1 and 128')
  })

  test('accepts valid nSeqMax', () => {
    expect(resolveConfig({ nSeqMax: 4 }).nSeqMax).toBe(4)
  })

  test('rejects nCtx < 128', () => {
    expect(() => resolveConfig({ nCtx: 64 })).toThrow('nCtx must be between 128 and 131072')
  })

  test('rejects maxTokens = 0', () => {
    expect(() => resolveConfig({ maxTokens: 0 })).toThrow('maxTokens must be between 1 and 131072')
  })

  test('rejects negative nThreads', () => {
    expect(() => resolveConfig({ nThreads: -2 })).toThrow('nThreads must be between 1 and 256')
  })

  test('default config is valid', () => {
    const c = resolveConfig()
    expect(c.nSeqMax).toBe(1)
    expect(c.nCtx).toBeGreaterThan(0)
    expect(c.maxTokens).toBeGreaterThan(0)
  })
})
