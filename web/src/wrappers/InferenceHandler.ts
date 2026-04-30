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

  getInferenceBackend(): number {
    return this.wasmInstance._inferencehandler_get_inference_backend(this.ptr)
  }

  prepare(hostConfig: PossiblePointer<HostConfig>): void
  prepare(
    hostConfig: PossiblePointer<HostConfig>,
    customLatency: number,
    tensorIndex?: number
  ): void
  prepare(hostConfig: PossiblePointer<HostConfig>, customLatency: Uint32Array): void
  prepare(
    hostConfig: PossiblePointer<HostConfig>,
    customLatency?: number | Uint32Array,
    tensorIndex?: number
  ): void {
    const hostConfigPtr = resolvePtr(hostConfig)
    if (customLatency === undefined) {
      this.wasmInstance._inferencehandler_prepare(this.ptr, hostConfigPtr)
      return
    }
    if (typeof customLatency === 'number') {
      this.wasmInstance._inferencehandler_prepare_with_latency(
        this.ptr,
        hostConfigPtr,
        customLatency,
        tensorIndex ?? 0
      )
      return
    }
    const latencyPtr = this.wasmInstance._malloc(customLatency.length * 4)
    this.wasmInstance.HEAPU32.set(customLatency, latencyPtr / 4)
    this.wasmInstance._inferencehandler_prepare_with_latency_vector(
      this.ptr,
      hostConfigPtr,
      latencyPtr,
      customLatency.length
    )
    this.wasmInstance._free(latencyPtr)
  }

  process(dataPtr: number, numSamples: number, tensorIndex?: number): number
  process(
    inputPtr: number,
    numInputSamples: number,
    outputPtr: number,
    numOutputSamples: number,
    tensorIndex?: number
  ): number
  process(
    a: number,
    b: number,
    c?: number,
    d?: number,
    e?: number
  ): number {
    // Discriminator: the separate-buffers overload always passes outputPtr at
    // position 4 (`d`). When `d` is undefined we're in the in-place form.
    if (d === undefined) {
      return this.wasmInstance._inferencehandler_process(this.ptr, a, b, c ?? 0)
    }
    return this.wasmInstance._inferencehandler_process_separate(
      this.ptr,
      a,
      b,
      c!,
      d,
      e ?? 0
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

  getAvailableSamples(tensorIndex: number, channel: number = 0): number {
    return this.wasmInstance._inferencehandler_get_available_samples(
      this.ptr,
      tensorIndex,
      channel
    )
  }

  setNonRealtime(nonRealtime: boolean): void {
    this.wasmInstance._inferencehandler_set_non_realtime(this.ptr, nonRealtime ? 1 : 0)
  }

  reset(): void {
    this.wasmInstance._inferencehandler_reset(this.ptr)
  }
}
