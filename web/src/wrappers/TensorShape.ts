import { type AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import type { TensorShapeList } from './Vectors'

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

  equals(other: PossiblePointer<TensorShape>): boolean {
    return this.wasmInstance._tensorshape_equals(this.ptr, resolvePtr(other)) === 1
  }

  notEquals(other: PossiblePointer<TensorShape>): boolean {
    return this.wasmInstance._tensorshape_not_equals(this.ptr, resolvePtr(other)) === 1
  }
}
