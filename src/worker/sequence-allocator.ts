/** Sequence slot state for parallel inference. */
export interface SequenceSlot {
  seqId: number
  state: 'idle' | 'prefilling' | 'generating'
  position: number
  requestId: string
  priority: number
}

/**
 * Pool-based allocator for sequence slots.
 * Manages which seq_ids are available and tracks active sequences.
 */
export class SequenceAllocator {
  private free: number[]
  private active: Map<number, SequenceSlot> = new Map()

  constructor(nSeqMax: number) {
    this.free = Array.from({ length: nSeqMax }, (_, i) => i)
  }

  acquire(requestId: string, priority = 0): SequenceSlot | null {
    const seqId = this.free.pop()
    if (seqId === undefined) return null

    const slot: SequenceSlot = {
      seqId,
      state: 'idle',
      position: 0,
      requestId,
      priority,
    }
    this.active.set(seqId, slot)
    return slot
  }

  release(seqId: number): void {
    if (!this.active.has(seqId)) return
    this.active.delete(seqId)
    this.free.push(seqId)
  }

  getActive(): SequenceSlot[] {
    return [...this.active.values()]
  }

  getSlot(seqId: number): SequenceSlot | undefined {
    return this.active.get(seqId)
  }

  hasFreeSlots(): boolean {
    return this.free.length > 0
  }

  get activeCount(): number {
    return this.active.size
  }
}
