import { type AniraWasmInstance } from '../factory'
import { BaseWrapper, type PossiblePointer, resolvePtr } from './BaseWrapper'
import type { InferenceConfig } from './InferenceConfig'
import type { PrePostProcessor } from './PrePostProcessor'
import type { HostConfig } from './utils/HostConfig'

/**
 * TypeScript wrapper for anira::InferenceHandler
 * Thread-safe C API wrapper
 */
export class InferenceHandler extends BaseWrapper {
  constructor(
    wasmInstance: AniraWasmInstance,
    preprocessor: PossiblePointer<PrePostProcessor>,
    config: PossiblePointer<InferenceConfig>,
    customProcessor?: PossiblePointer
  ) {
    const preprocessorPtr = resolvePtr(preprocessor)
    const configPtr = resolvePtr(config)
    if (customProcessor) {
      super(
        wasmInstance,
        wasmInstance._inferencehandler_create_with_custom_processor(
          preprocessorPtr,
          configPtr,
          resolvePtr(customProcessor)
        )
      )
    } else {
      super(
        wasmInstance,
        wasmInstance._inferencehandler_create(preprocessorPtr, configPtr)
      )
    }
  }

  destroy(): void {
    this._destroy(this.wasmInstance._inferencehandler_destroy)
  }

  setInferenceBackend(backend: number): void {
    this.wasmInstance._inferencehandler_set_inference_backend(this.ptr, backend)
  }

  prepare(hostConfig: PossiblePointer<HostConfig>): void {
    this.wasmInstance._inferencehandler_prepare(this.ptr, resolvePtr(hostConfig))
  }

  prepareWithLatency(
    hostConfig: PossiblePointer<HostConfig>,
    customLatency: number,
    tensorIndex: number
  ): void {
    this.wasmInstance._inferencehandler_prepare_with_latency(
      this.ptr,
      resolvePtr(hostConfig),
      customLatency,
      tensorIndex
    )
  }

  prepareWithLatencyVector(
    hostConfig: PossiblePointer<HostConfig>,
    latencyVector: Uint32Array
  ): void {
    const hostConfigPtr = resolvePtr(hostConfig)
    const latencyPtr = this.wasmInstance._malloc(latencyVector.length * 4)
    this.wasmInstance.HEAPU32.set(latencyVector, latencyPtr / 4)
    this.wasmInstance._inferencehandler_prepare_with_latency_vector(
      this.ptr,
      hostConfigPtr,
      latencyPtr,
      latencyVector.length
    )
    this.wasmInstance._free(latencyPtr)
  }

  process(dataPtr: number, numSamples: number, tensorIndex: number = 0): number {
    return this.wasmInstance._inferencehandler_process(
      this.ptr,
      dataPtr,
      numSamples,
      tensorIndex
    )
  }

  processSeparate(
    inputPtr: number,
    numInputSamples: number,
    outputPtr: number,
    numOutputSamples: number,
    tensorIndex: number = 0
  ): number {
    return this.wasmInstance._inferencehandler_process_separate(
      this.ptr,
      inputPtr,
      numInputSamples,
      outputPtr,
      numOutputSamples,
      tensorIndex
    )
  }

  processMulti(
    inputPtr: number,
    numInputPtr: number,
    outputPtr: number,
    numOutputPtr: number
  ): number {
    return this.wasmInstance._inferencehandler_process_multi(
      this.ptr,
      inputPtr,
      numInputPtr,
      outputPtr,
      numOutputPtr
    )
  }

  pushData(inputPtr: number, numSamples: number, tensorIndex: number = 0): void {
    this.wasmInstance._inferencehandler_push_data(
      this.ptr,
      inputPtr,
      numSamples,
      tensorIndex
    )
  }

  pushDataMulti(inputPtr: number, numSamplesPtr: number): void {
    this.wasmInstance._inferencehandler_push_data_multi(this.ptr, inputPtr, numSamplesPtr)
  }

  popData(outputPtr: number, numSamples: number, tensorIndex: number = 0): number {
    return this.wasmInstance._inferencehandler_pop_data(
      this.ptr,
      outputPtr,
      numSamples,
      tensorIndex
    )
  }

  popDataBlocking(
    outputPtr: number,
    numSamples: number,
    waitMs: number,
    tensorIndex: number = 0
  ): number {
    return this.wasmInstance._inferencehandler_pop_data_blocking(
      this.ptr,
      outputPtr,
      numSamples,
      waitMs,
      tensorIndex
    )
  }

  popDataMulti(outputPtr: number, numSamplesPtr: number): number {
    return this.wasmInstance._inferencehandler_pop_data_multi(
      this.ptr,
      outputPtr,
      numSamplesPtr
    )
  }

  popDataMultiBlocking(outputPtr: number, numSamplesPtr: number, waitMs: number): number {
    return this.wasmInstance._inferencehandler_pop_data_multi_blocking(
      this.ptr,
      outputPtr,
      numSamplesPtr,
      waitMs
    )
  }

  getLatency(tensorIndex: number = 0): number {
    return this.wasmInstance._inferencehandler_get_latency(this.ptr, tensorIndex)
  }

  getLatencyVector(): number[] {
    const vectorPtr = this.wasmInstance._inferencehandler_get_latency_vector(this.ptr)

    const dataPtr = this.wasmInstance.HEAPU32[vectorPtr / 4]
    const endPtr = this.wasmInstance.HEAPU32[vectorPtr / 4 + 1]
    const size = (endPtr - dataPtr) / 4 // size in elements (uint32)

    const result: number[] = []
    for (let i = 0; i < size; i++) {
      result.push(this.wasmInstance.HEAPU32[dataPtr / 4 + i])
    }
    return result
  }

  getAvailableSamples(tensorIndex: number = 0): number {
    return this.wasmInstance._inferencehandler_get_available_samples(
      this.ptr,
      tensorIndex
    )
  }

  setNonRealtime(nonRealtime: boolean): void {
    this.wasmInstance._inferencehandler_set_non_realtime(this.ptr, nonRealtime ? 1 : 0)
  }

  reset(): void {
    this.wasmInstance._inferencehandler_reset(this.ptr)
  }
}
