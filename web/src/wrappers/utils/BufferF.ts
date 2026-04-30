import type { AniraWasmInstance } from '../../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from '../BaseWrapper'

/**
 * TypeScript wrapper for anira::BufferF
 * Thread-safe C API wrapper
 */
export class BufferF extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    numChannels?: number,
    numSamples?: number
  ) {
    if (numChannels !== undefined && numSamples !== undefined) {
      super(wasmInstance, wasmInstance._bufferf_create_with_size(numChannels, numSamples))
    } else {
      super(wasmInstance, wasmInstance._bufferf_create())
    }
  }

  destroy(): void {
    this._destroy(this.wasmInstance._bufferf_destroy)
  }

  getNumChannels(): number {
    return this.wasmInstance._bufferf_get_num_channels(this.ptr)
  }

  getNumSamples(): number {
    return this.wasmInstance._bufferf_get_num_samples(this.ptr)
  }

  resize(numChannels: number, numSamples: number): void {
    this.wasmInstance._bufferf_resize(this.ptr, numChannels, numSamples)
  }

  getReadPointer(channel: number): number
  getReadPointer(channel: number, sampleIndex: number): number
  getReadPointer(channel: number, sampleIndex?: number): number {
    if (sampleIndex === undefined) {
      return this.wasmInstance._bufferf_get_read_pointer(this.ptr, channel)
    }
    return this.wasmInstance._bufferf_get_read_pointer_at(this.ptr, channel, sampleIndex)
  }

  getWritePointer(channel: number): number
  getWritePointer(channel: number, sampleIndex: number): number
  getWritePointer(channel: number, sampleIndex?: number): number {
    if (sampleIndex === undefined) {
      return this.wasmInstance._bufferf_get_write_pointer(this.ptr, channel)
    }
    return this.wasmInstance._bufferf_get_write_pointer_at(this.ptr, channel, sampleIndex)
  }

  getArrayOfReadPointers(): number[] {
    const numChannels = this.getNumChannels()
    const outArray = this.wasmInstance._malloc(numChannels * 4) // 4 bytes per pointer
    this.wasmInstance._bufferf_get_array_of_read_pointers(this.ptr, outArray)

    const result: number[] = []
    const view = new Uint32Array(this.wasmInstance.HEAPU32.buffer, outArray, numChannels)
    for (let i = 0; i < numChannels; i++) {
      result.push(view[i])
    }

    this.wasmInstance._free(outArray)
    return result
  }

  getArrayOfWritePointers(): number[] {
    const numChannels = this.getNumChannels()
    const outArray = this.wasmInstance._malloc(numChannels * 4) // 4 bytes per pointer
    this.wasmInstance._bufferf_get_array_of_write_pointers(this.ptr, outArray)

    const result: number[] = []
    const view = new Uint32Array(this.wasmInstance.HEAPU32.buffer, outArray, numChannels)
    for (let i = 0; i < numChannels; i++) {
      result.push(view[i])
    }

    this.wasmInstance._free(outArray)
    return result
  }

  data(): number {
    return this.wasmInstance._bufferf_data(this.ptr)
  }

  swapData(other: PossiblePointer<BufferF>): void
  swapData(rawPointer: number, size: number): void
  swapData(
    otherOrRawPointer: PossiblePointer<BufferF> | number,
    size?: number
  ): void {
    // The single-arg overload accepts either a `BufferF` instance or a raw
    // pointer to a `BufferF`; the two-arg overload accepts a raw float buffer
    // pointer plus its element count. Distinguishing via arg count is
    // unambiguous.
    if (size === undefined) {
      this.wasmInstance._bufferf_swap_data_with_buffer(
        this.ptr,
        resolvePtr(otherOrRawPointer as PossiblePointer<BufferF>)
      )
      return
    }
    this.wasmInstance._bufferf_swap_data_with_raw_pointer(
      this.ptr,
      otherOrRawPointer as number,
      size
    )
  }

  resetChannelPtr(): void {
    this.wasmInstance._bufferf_reset_channel_ptr(this.ptr)
  }

  getSample(channel: number, sampleIndex: number): number {
    return this.wasmInstance._bufferf_get_sample(this.ptr, channel, sampleIndex)
  }

  setSample(channel: number, sampleIndex: number, value: number): void {
    this.wasmInstance._bufferf_set_sample(this.ptr, channel, sampleIndex, value)
  }

  clear(): void {
    this.wasmInstance._bufferf_clear(this.ptr)
  }
}
