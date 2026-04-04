/** Sequence slot for parallel inference. */
export interface SequenceSlot {
  seqId: number
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

  acquire(): SequenceSlot | null {
    const seqId = this.free.pop()
    if (seqId === undefined) return null

    const slot: SequenceSlot = { seqId }
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

  hasFreeSlots(): boolean {
    return this.free.length > 0
  }

  get activeCount(): number {
    return this.active.size
  }
}
