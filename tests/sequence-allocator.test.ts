import { describe, test, expect } from 'bun:test'
import { SequenceAllocator } from '../src/worker/sequence-allocator.ts'

describe('SequenceAllocator', () => {
  test('acquires slots up to capacity', () => {
    const alloc = new SequenceAllocator(3)
    expect(alloc.hasFreeSlots()).toBe(true)

    const s0 = alloc.acquire('req-1')
    const s1 = alloc.acquire('req-2')
    const s2 = alloc.acquire('req-3')

    expect(s0).not.toBeNull()
    expect(s1).not.toBeNull()
    expect(s2).not.toBeNull()

    // All taken
    expect(alloc.hasFreeSlots()).toBe(false)
    expect(alloc.acquire('req-4')).toBeNull()
  })

  test('release recycles slots', () => {
    const alloc = new SequenceAllocator(2)
    const s0 = alloc.acquire('req-1')!
    alloc.acquire('req-2')

    alloc.release(s0.seqId)
    expect(alloc.hasFreeSlots()).toBe(true)
    expect(alloc.activeCount).toBe(1)

    const s2 = alloc.acquire('req-3')
    expect(s2).not.toBeNull()
    expect(s2!.seqId).toBe(s0.seqId)
  })

  test('double release is a no-op', () => {
    const alloc = new SequenceAllocator(2)
    const s = alloc.acquire('req-1')!
    alloc.release(s.seqId)
    alloc.release(s.seqId)
    expect(alloc.activeCount).toBe(0)
  })

  test('all seqIds are unique and in range', () => {
    const alloc = new SequenceAllocator(4)
    const ids = new Set<number>()
    for (let i = 0; i < 4; i++) {
      const s = alloc.acquire(`req-${i}`)!
      expect(s.seqId).toBeGreaterThanOrEqual(0)
      expect(s.seqId).toBeLessThan(4)
      ids.add(s.seqId)
    }
    expect(ids.size).toBe(4)
  })

  test('getActive returns active slots only', () => {
    const alloc = new SequenceAllocator(3)
    alloc.acquire('a')
    const b = alloc.acquire('b')!
    alloc.acquire('c')

    expect(alloc.getActive()).toHaveLength(3)
    alloc.release(b.seqId)
    expect(alloc.getActive()).toHaveLength(2)
  })
})
