import type { AniraWasmInstance } from '../../factory'
import { BaseWrapper } from '../BaseWrapper'

/**
 * TypeScript wrapper for anira::RingBuffer
 * Thread-safe C API wrapper
 */
export class RingBuffer extends BaseWrapper {
  constructor(wasmInstance: AniraWasmInstance) {
    super(wasmInstance, wasmInstance._ringbuffer_create())
  }

  destroy(): void {
    this._destroy(this.wasmInstance._ringbuffer_destroy)
  }

  initializeWithPositions(numChannels: number, numSamples: number): void {
    this.wasmInstance._ringbuffer_initialize_with_positions(
      this.ptr,
      numChannels,
      numSamples
    )
  }

  clearWithPositions(): void {
    this.wasmInstance._ringbuffer_clear_with_positions(this.ptr)
  }

  pushSample(channel: number, sample: number): void {
    this.wasmInstance._ringbuffer_push_sample(this.ptr, channel, sample)
  }

  popSample(channel: number): number {
    return this.wasmInstance._ringbuffer_pop_sample(this.ptr, channel)
  }

  getFutureSample(channel: number, offset: number): number {
    return this.wasmInstance._ringbuffer_get_future_sample(this.ptr, channel, offset)
  }

  getPastSample(channel: number, offset: number): number {
    return this.wasmInstance._ringbuffer_get_past_sample(this.ptr, channel, offset)
  }

  getAvailableSamples(channel: number): number {
    return this.wasmInstance._ringbuffer_get_available_samples(this.ptr, channel)
  }

  getAvailablePastSamples(channel: number): number {
    return this.wasmInstance._ringbuffer_get_available_past_samples(this.ptr, channel)
  }
}
