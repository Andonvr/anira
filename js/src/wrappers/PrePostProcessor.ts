import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import type { AniraWasmInstance } from '../factory'
import type { InferenceConfig } from './InferenceConfig'
import type { RingBuffer } from './utils/RingBuffer'
import type { BufferF } from './utils/BufferF'
import type { VectorBufferF, VectorRingBuffer } from './Vectors'

/**
 * TypeScript wrapper for anira::PrePostProcessor
 * Thread-safe C API wrapper
 */
export class PrePostProcessor extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    config: PossiblePointer<InferenceConfig>,
    createFn?: (configPtr: number) => number
  ) {
    const configPtr = resolvePtr(config)
    const creator = createFn ?? wasmInstance._prepostprocessor_create
    super(wasmInstance, creator(configPtr))
  }

  /**   * Destroy this buffer and free memory
   */
  destroy(): void {
    this._destroy(this.wasmInstance._prepostprocessor_destroy)
  }

  preProcess(
    ringBuffers: PossiblePointer<VectorRingBuffer>,
    buffers: PossiblePointer<VectorBufferF>,
    backend: number
  ): void {
    this.wasmInstance._prepostprocessor_pre_process(
      this.ptr,
      resolvePtr(ringBuffers),
      resolvePtr(buffers),
      backend
    )
  }

  postProcess(
    buffers: PossiblePointer<VectorBufferF>,
    ringBuffers: PossiblePointer<VectorRingBuffer>,
    backend: number
  ): void {
    this.wasmInstance._prepostprocessor_post_process(
      this.ptr,
      resolvePtr(buffers),
      resolvePtr(ringBuffers),
      backend
    )
  }

  setInput(value: number, channel: number, tensorIndex: number): void {
    this.wasmInstance._prepostprocessor_set_input(this.ptr, value, channel, tensorIndex)
  }

  setOutput(value: number, channel: number, tensorIndex: number): void {
    this.wasmInstance._prepostprocessor_set_output(this.ptr, value, channel, tensorIndex)
  }

  getInput(channel: number, tensorIndex: number): number {
    return this.wasmInstance._prepostprocessor_get_input(this.ptr, channel, tensorIndex)
  }

  getOutput(channel: number, tensorIndex: number): number {
    return this.wasmInstance._prepostprocessor_get_output(this.ptr, channel, tensorIndex)
  }

  popSamplesFromBuffer(
    ringBuffer: PossiblePointer<RingBuffer>,
    buffer: PossiblePointer<BufferF>,
    numSamples: number
  ): void {
    this.wasmInstance._prepostprocessor_pop_samples_from_buffer(
      this.ptr,
      resolvePtr(ringBuffer),
      resolvePtr(buffer),
      numSamples
    )
  }

  popSamplesFromBufferWindow(
    ringBuffer: PossiblePointer<RingBuffer>,
    buffer: PossiblePointer<BufferF>,
    numSamples: number,
    windowSize: number
  ): void {
    this.wasmInstance._prepostprocessor_pop_samples_from_buffer_window(
      this.ptr,
      resolvePtr(ringBuffer),
      resolvePtr(buffer),
      numSamples,
      windowSize
    )
  }

  popSamplesFromBufferWindowOffset(
    ringBuffer: PossiblePointer<RingBuffer>,
    buffer: PossiblePointer<BufferF>,
    numSamples: number,
    windowSize: number,
    offset: number
  ): void {
    this.wasmInstance._prepostprocessor_pop_samples_from_buffer_window_offset(
      this.ptr,
      resolvePtr(ringBuffer),
      resolvePtr(buffer),
      numSamples,
      windowSize,
      offset
    )
  }

  pushSamplesToBuffer(
    buffer: PossiblePointer<BufferF>,
    ringBuffer: PossiblePointer<RingBuffer>,
    numSamples: number
  ): void {
    this.wasmInstance._prepostprocessor_push_samples_to_buffer(
      this.ptr,
      resolvePtr(buffer),
      resolvePtr(ringBuffer),
      numSamples
    )
  }
}
