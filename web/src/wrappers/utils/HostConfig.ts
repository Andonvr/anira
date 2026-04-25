import type { AniraWasmInstance } from '../../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from '../BaseWrapper'
import type { InferenceConfig } from '../InferenceConfig'

/**
 * TypeScript wrapper for anira::HostConfig
 * Thread-safe C API wrapper
 */
export class HostConfig extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    bufferSize?: number,
    sampleRate?: number,
    allowSmallerBuffers?: boolean,
    tensorIndex?: number
  ) {
    if (
      bufferSize !== undefined &&
      sampleRate !== undefined &&
      allowSmallerBuffers !== undefined &&
      tensorIndex !== undefined
    ) {
      super(
        wasmInstance,
        wasmInstance._hostconfig_create_with_params(
          bufferSize,
          sampleRate,
          allowSmallerBuffers ? 1 : 0,
          tensorIndex
        )
      )
    } else {
      super(wasmInstance, wasmInstance._hostconfig_create())
    }
  }

  destroy(): void {
    this._destroy(this.wasmInstance._hostconfig_destroy)
  }

  // Property getters
  get bufferSize(): number {
    return this.wasmInstance._hostconfig_get_buffer_size(this.ptr)
  }

  get sampleRate(): number {
    return this.wasmInstance._hostconfig_get_sample_rate(this.ptr)
  }

  get allowSmallerBuffers(): boolean {
    return this.wasmInstance._hostconfig_get_allow_smaller_buffers(this.ptr) === 1
  }

  get tensorIndex(): number {
    return this.wasmInstance._hostconfig_get_tensor_index(this.ptr)
  }

  // Property setters
  set bufferSize(value: number) {
    this.wasmInstance._hostconfig_set_buffer_size(this.ptr, value)
  }

  set sampleRate(value: number) {
    this.wasmInstance._hostconfig_set_sample_rate(this.ptr, value)
  }

  set allowSmallerBuffers(value: boolean) {
    this.wasmInstance._hostconfig_set_allow_smaller_buffers(this.ptr, value ? 1 : 0)
  }

  set tensorIndex(value: number) {
    this.wasmInstance._hostconfig_set_tensor_index(this.ptr, value)
  }

  equals(other: PossiblePointer<HostConfig>): boolean {
    return this.wasmInstance._hostconfig_equals(this.ptr, resolvePtr(other)) === 1
  }

  notEquals(other: PossiblePointer<HostConfig>): boolean {
    return this.wasmInstance._hostconfig_not_equals(this.ptr, resolvePtr(other)) === 1
  }

  getRelativeBufferSize(
    inferenceConfig: PossiblePointer<InferenceConfig>,
    tensorIndex: number,
    input: boolean = true
  ): number {
    return this.wasmInstance._hostconfig_get_relative_buffer_size(
      this.ptr,
      resolvePtr(inferenceConfig),
      tensorIndex,
      input ? 1 : 0
    )
  }

  getRelativeSampleRate(
    inferenceConfig: PossiblePointer<InferenceConfig>,
    tensorIndex: number,
    input: boolean = true
  ): number {
    return this.wasmInstance._hostconfig_get_relative_sample_rate(
      this.ptr,
      resolvePtr(inferenceConfig),
      tensorIndex,
      input ? 1 : 0
    )
  }
}
