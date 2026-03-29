/**
 * Serial execution queue — ensures one task runs at a time.
 * Tasks are enqueued and executed in FIFO order via Promise chaining.
 */
export class SerialQueue {
  private tail: Promise<void> = Promise.resolve()

  enqueue<T>(task: () => Promise<T>): Promise<T> {
    const result = this.tail.then(task, () => task())
    // Keep the chain going regardless of task success/failure
    this.tail = result.then(() => {}, () => {})
    return result
  }
}
