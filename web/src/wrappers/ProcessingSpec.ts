import { type AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import type { VectorSizeT } from './Vectors'

/**
 * TypeScript wrapper for anira::ProcessingSpec
 * Thread-safe C API wrapper
 */
export class ProcessingSpec extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    preprocessChannels: PossiblePointer<VectorSizeT>,
    postprocessChannels: PossiblePointer<VectorSizeT>,
    preprocessSize: PossiblePointer<VectorSizeT>,
    postprocessSize: PossiblePointer<VectorSizeT>
  ) {
    const preprocessChannelsPtr = resolvePtr(preprocessChannels)
    const postprocessChannelsPtr = resolvePtr(postprocessChannels)
    const preprocessSizePtr = resolvePtr(preprocessSize)
    const postprocessSizePtr = resolvePtr(postprocessSize)

    // Get sizes from the vector pointers
    const preChCount = wasmInstance._vector_size_t_size(preprocessChannelsPtr)
    const postChCount = wasmInstance._vector_size_t_size(postprocessChannelsPtr)
    const preSizeCount = wasmInstance._vector_size_t_size(preprocessSizePtr)
    const postSizeCount = wasmInstance._vector_size_t_size(postprocessSizePtr)

    super(
      wasmInstance,
      wasmInstance._processingspec_create_full(
        preprocessChannelsPtr,
        preChCount,
        postprocessChannelsPtr,
        postChCount,
        preprocessSizePtr,
        preSizeCount,
        postprocessSizePtr,
        postSizeCount
      )
    )
  }

  destroy(): void {
    this._destroy(this.wasmInstance._processingspec_destroy)
  }

  equals(other: PossiblePointer<ProcessingSpec>): boolean {
    return this.wasmInstance._processingspec_equals(this.ptr, resolvePtr(other)) === 1
  }

  notEquals(other: PossiblePointer<ProcessingSpec>): boolean {
    return this.wasmInstance._processingspec_not_equals(this.ptr, resolvePtr(other)) === 1
  }
}
