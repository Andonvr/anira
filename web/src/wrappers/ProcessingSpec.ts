import { type AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import { VectorSizeT } from './Vectors'

/**
 * TypeScript wrapper for anira::ProcessingSpec
 * Thread-safe C API wrapper
 */
export class ProcessingSpec extends BaseWrapper {
  constructor(wasmInstance: AniraWasmInstance)
  constructor(
    wasmInstance: AniraWasmInstance,
    preprocessChannels: PossiblePointer<VectorSizeT>,
    postprocessChannels: PossiblePointer<VectorSizeT>
  )
  constructor(
    wasmInstance: AniraWasmInstance,
    preprocessChannels: PossiblePointer<VectorSizeT>,
    postprocessChannels: PossiblePointer<VectorSizeT>,
    preprocessSize: PossiblePointer<VectorSizeT>,
    postprocessSize: PossiblePointer<VectorSizeT>
  )
  constructor(
    wasmInstance: AniraWasmInstance,
    preprocessChannels: PossiblePointer<VectorSizeT>,
    postprocessChannels: PossiblePointer<VectorSizeT>,
    preprocessSize: PossiblePointer<VectorSizeT>,
    postprocessSize: PossiblePointer<VectorSizeT>,
    internalModelLatency: PossiblePointer<VectorSizeT>
  )
  constructor(
    wasmInstance: AniraWasmInstance,
    preprocessChannels?: PossiblePointer<VectorSizeT>,
    postprocessChannels?: PossiblePointer<VectorSizeT>,
    preprocessSize?: PossiblePointer<VectorSizeT>,
    postprocessSize?: PossiblePointer<VectorSizeT>,
    internalModelLatency?: PossiblePointer<VectorSizeT>
  ) {
    if (preprocessChannels === undefined || postprocessChannels === undefined) {
      super(wasmInstance, wasmInstance._processingspec_create())
      return
    }

    const preChPtr = resolvePtr(preprocessChannels)
    const postChPtr = resolvePtr(postprocessChannels)
    const preChCount = wasmInstance._vector_size_t_size(preChPtr)
    const postChCount = wasmInstance._vector_size_t_size(postChPtr)

    if (preprocessSize === undefined || postprocessSize === undefined) {
      super(
        wasmInstance,
        wasmInstance._processingspec_create_with_channels(
          preChPtr,
          preChCount,
          postChPtr,
          postChCount
        )
      )
      return
    }

    const preSizePtr = resolvePtr(preprocessSize)
    const postSizePtr = resolvePtr(postprocessSize)
    const preSizeCount = wasmInstance._vector_size_t_size(preSizePtr)
    const postSizeCount = wasmInstance._vector_size_t_size(postSizePtr)

    if (internalModelLatency === undefined) {
      super(
        wasmInstance,
        wasmInstance._processingspec_create_full(
          preChPtr,
          preChCount,
          postChPtr,
          postChCount,
          preSizePtr,
          preSizeCount,
          postSizePtr,
          postSizeCount
        )
      )
      return
    }

    const latencyPtr = resolvePtr(internalModelLatency)
    const latencyCount = wasmInstance._vector_size_t_size(latencyPtr)
    super(
      wasmInstance,
      wasmInstance._processingspec_create_full_with_latency(
        preChPtr,
        preChCount,
        postChPtr,
        postChCount,
        preSizePtr,
        preSizeCount,
        postSizePtr,
        postSizeCount,
        latencyPtr,
        latencyCount
      )
    )
  }

  destroy(): void {
    this._destroy(this.wasmInstance._processingspec_destroy)
  }

  getPreprocessInputChannels(tensorIndex: number = 0): number {
    return this.wasmInstance._processingspec_get_preprocess_input_channels(
      this.ptr,
      tensorIndex
    )
  }

  getPostprocessOutputChannels(tensorIndex: number = 0): number {
    return this.wasmInstance._processingspec_get_postprocess_output_channels(
      this.ptr,
      tensorIndex
    )
  }

  getPreprocessInputSize(tensorIndex: number = 0): number {
    return this.wasmInstance._processingspec_get_preprocess_input_size(
      this.ptr,
      tensorIndex
    )
  }

  getPostprocessOutputSize(tensorIndex: number = 0): number {
    return this.wasmInstance._processingspec_get_postprocess_output_size(
      this.ptr,
      tensorIndex
    )
  }

  getInternalModelLatency(tensorIndex: number = 0): number {
    return this.wasmInstance._processingspec_get_internal_model_latency(
      this.ptr,
      tensorIndex
    )
  }

  getTensorInputSize(tensorIndex: number = 0): number {
    return this.wasmInstance._processingspec_get_tensor_input_size(
      this.ptr,
      tensorIndex
    )
  }

  getTensorOutputSize(tensorIndex: number = 0): number {
    return this.wasmInstance._processingspec_get_tensor_output_size(
      this.ptr,
      tensorIndex
    )
  }

  equals(other: PossiblePointer<ProcessingSpec>): boolean {
    return this.wasmInstance._processingspec_equals(this.ptr, resolvePtr(other)) === 1
  }

  notEquals(other: PossiblePointer<ProcessingSpec>): boolean {
    return this.wasmInstance._processingspec_not_equals(this.ptr, resolvePtr(other)) === 1
  }
}
