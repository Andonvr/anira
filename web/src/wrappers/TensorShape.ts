import { type AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import { TensorShapeList } from './Vectors'

/**
 * TypeScript wrapper for anira::TensorShape
 * Thread-safe C API wrapper
 */
export class TensorShape extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    inputShapeListOrPtr: PossiblePointer<TensorShapeList>,
    outputShapeListOrPtr: PossiblePointer<TensorShapeList>
  ) {
    const inputShapeList = resolvePtr(inputShapeListOrPtr)
    const outputShapeList = resolvePtr(outputShapeListOrPtr)
    const inputCount = wasmInstance._vector_vector_int64_size(inputShapeList)
    const outputCount = wasmInstance._vector_vector_int64_size(outputShapeList)
    super(
      wasmInstance,
      wasmInstance._tensorshape_create(
        inputShapeList,
        inputCount,
        outputShapeList,
        outputCount
      )
    )
  }

  destroy(): void {
    this._destroy(this.wasmInstance._tensorshape_destroy)
  }

  isUniversal(): boolean {
    return this.wasmInstance._tensorshape_is_universal(this.ptr) === 1
  }

  /**
   * Returns a non-owning view into `m_tensor_input_shape`. Do **not** call
   * `.destroy()` on it — the storage belongs to this `TensorShape`.
   */
  getTensorInputShape(): TensorShapeList {
    const ptr = this.wasmInstance._tensorshape_get_input_shape(this.ptr)
    return this.wrapPointer(TensorShapeList, ptr)
  }

  /**
   * Returns a non-owning view into `m_tensor_output_shape`. See
   * `getTensorInputShape` for ownership notes.
   */
  getTensorOutputShape(): TensorShapeList {
    const ptr = this.wasmInstance._tensorshape_get_output_shape(this.ptr)
    return this.wrapPointer(TensorShapeList, ptr)
  }

  equals(other: PossiblePointer<TensorShape>): boolean {
    return this.wasmInstance._tensorshape_equals(this.ptr, resolvePtr(other)) === 1
  }

  notEquals(other: PossiblePointer<TensorShape>): boolean {
    return this.wasmInstance._tensorshape_not_equals(this.ptr, resolvePtr(other)) === 1
  }
}
