import type { AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import type { VectorModelData, VectorTensorShape } from './Vectors'
import type { ProcessingSpec } from './ProcessingSpec'

/**
 * TypeScript wrapper for anira::InferenceConfig
 * Thread-safe C API wrapper
 */
export class InferenceConfig extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    modelDataVector: PossiblePointer<VectorModelData>,
    tensorShapeVector: PossiblePointer<VectorTensorShape>,
    processingSpec: PossiblePointer<ProcessingSpec>,
    maxInferenceTime: number,
    warmUp: number,
    sessionExclusiveProcessor: boolean,
    blockingRatio: number,
    numParallelProcessors: number
  ) {
    const modelDataVectorPtr = resolvePtr(modelDataVector)
    const tensorShapeVectorPtr = resolvePtr(tensorShapeVector)
    const processingSpecPtr = resolvePtr(processingSpec)

    // Get counts from the vector pointers
    const modelCount = wasmInstance._vector_model_data_size(modelDataVectorPtr)
    const tensorCount = wasmInstance._vector_tensor_shape_size(tensorShapeVectorPtr)
    super(
      wasmInstance,
      wasmInstance._inferenceconfig_create_full(
        modelDataVectorPtr,
        modelCount,
        tensorShapeVectorPtr,
        tensorCount,
        processingSpecPtr,
        maxInferenceTime,
        warmUp,
        sessionExclusiveProcessor ? 1 : 0,
        blockingRatio,
        numParallelProcessors
      )
    )
  }

  destroy(): void {
    this._destroy(this.wasmInstance._inferenceconfig_destroy)
  }

  getTensorInputSize(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_tensor_input_size(this.ptr, tensorIndex)
  }

  getTensorOutputSize(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_tensor_output_size(
      this.ptr,
      tensorIndex
    )
  }

  getPreprocessInputChannels(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_preprocess_input_channels(
      this.ptr,
      tensorIndex
    )
  }

  getPostprocessOutputChannels(tensorIndex: number = 0): number {
    return this.wasmInstance._inferenceconfig_get_postprocess_output_channels(
      this.ptr,
      tensorIndex
    )
  }

  getMaxInferenceTime(): number {
    return this.wasmInstance._inferenceconfig_get_max_inference_time(this.ptr)
  }

  setMaxInferenceTime(value: number): void {
    this.wasmInstance._inferenceconfig_set_max_inference_time(this.ptr, value)
  }

  getWarmUp(): number {
    return this.wasmInstance._inferenceconfig_get_warm_up(this.ptr)
  }

  setWarmUp(value: number): void {
    this.wasmInstance._inferenceconfig_set_warm_up(this.ptr, value)
  }

  getModelData(backend: number): number {
    return this.wasmInstance._inferenceconfig_get_model_data(this.ptr, backend)
  }
}
