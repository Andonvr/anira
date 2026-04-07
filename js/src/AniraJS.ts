import { JSBackendBase } from './backends'
import { ONNXRuntimeWebBackend } from './backends/ONNXRuntimeWebBackend'
import {
  createAniraWasm,
  wasmUrl,
  type AniraWasmConfig,
  type AniraWasmInstance,
} from './factory'
import { createFactory, type Factory } from './utils'
import type {
  AudioWorkletConfigureMessage,
  AudioWorkletIOConfig,
  DestroyMessage,
  InitInferenceWorkerMessage,
  RegisterProcessorMessage,
  StartMessage,
} from './workers/messages'
import { waitForWorkerMessage } from './workers/messages'
import {
  BufferF,
  HostConfig,
  InferenceConfig,
  InferenceHandler,
  JSPrePostProcessor,
  ModelData,
  PrePostProcessor,
  ProcessingSpec,
  RingBuffer,
  TensorShape,
  TensorShapeList,
  VectorBufferF,
  VectorFloat,
  VectorInt64T,
  VectorModelData,
  VectorRingBuffer,
  VectorSizeT,
  VectorTensorShape,
  VectorUnsignedInt,
  VectorVectorInt64,
  createInferenceBackend,
  type InferenceBackendValues,
} from './wrappers'
import { resolvePtr, type PossiblePointer } from './wrappers/BaseWrapper'
import { WasmInferenceThread } from './wrappers/system/WasmInferenceThread'

export type ConfigureAudioWorkletIOOptions = Partial<AudioWorkletIOConfig>

export type ProcessorDescriptor = {
  backend: JSBackendBase
  className: string
}

export type InferenceWorker = {
  worker: Worker
  registerProcessor: (descriptor: ProcessorDescriptor) => Promise<void>
  stop: () => Promise<void>
}

export class AniraJS {
  protected wasmInstance: AniraWasmInstance
  protected memory: WebAssembly.Memory
  protected wasmBinary: ArrayBuffer | null = null
  private registeredProcessors: ProcessorDescriptor[] = []
  private registeredPrePostProcessors: Map<number, JSPrePostProcessor>
  private activeWorkers: InferenceWorker[] = []

  InferenceBackend: InferenceBackendValues
  Buffer: Factory<typeof BufferF>
  HostConfig: Factory<typeof HostConfig>
  InferenceConfig: Factory<typeof InferenceConfig>
  InferenceHandler: Factory<typeof InferenceHandler>
  JSBackendBase: Factory<typeof JSBackendBase>
  ONNXRuntimeWebBackend: Factory<typeof ONNXRuntimeWebBackend>
  JSPrePostProcessor: Factory<typeof JSPrePostProcessor>
  ModelData: Factory<typeof ModelData>
  PrePostProcessor: Factory<typeof PrePostProcessor>
  ProcessingSpec: Factory<typeof ProcessingSpec>
  RingBuffer: Factory<typeof RingBuffer>
  TensorShape: Factory<typeof TensorShape>
  WasmInferenceThread: Factory<typeof WasmInferenceThread>
  VectorBufferF: Factory<typeof VectorBufferF>
  VectorFloat: Factory<typeof VectorFloat>
  VectorInt64T: Factory<typeof VectorInt64T>
  VectorModelData: Factory<typeof VectorModelData>
  VectorRingBuffer: Factory<typeof VectorRingBuffer>
  VectorSizeT: Factory<typeof VectorSizeT>
  VectorTensorShape: Factory<typeof VectorTensorShape>
  VectorUnsignedInt: Factory<typeof VectorUnsignedInt>
  VectorVectorInt64: Factory<typeof VectorVectorInt64>
  TensorShapeList: Factory<typeof TensorShapeList>

  private static async initWasm(
    config?: AniraWasmConfig & Record<string, unknown>,
    memory?: WebAssembly.Memory
  ) {
    const wasmMemory =
      memory ??
      new WebAssembly.Memory({
        initial: 8192,
        maximum: 8192,
        shared: true,
      })
    const prePostRegistry = new Map<number, JSPrePostProcessor>()
    const { processPrePost: externalProcessPrePost, ...restConfig } = config ?? {}
    const wasmInstance = await createAniraWasm(wasmMemory, {
      ...restConfig,
      processPrePost: (
        prePostProcessorPtr: number,
        inputPtr: number,
        outputPtr: number,
        backend: number,
        phase: number
      ) => {
        const prePostProcessor = prePostRegistry.get(prePostProcessorPtr)
        if (prePostProcessor) {
          if (phase === 0) {
            prePostProcessor.preProcess(inputPtr, outputPtr, backend)
            return
          }
          if (phase === 1) {
            prePostProcessor.postProcess(inputPtr, outputPtr, backend)
            return
          }
          throw new Error(`Unknown pre/post phase: ${phase}`)
        }

        if (externalProcessPrePost) {
          externalProcessPrePost(prePostProcessorPtr, inputPtr, outputPtr, backend, phase)
          return
        }

        throw new Error(
          `JSPrePostProcessor with pointer ${prePostProcessorPtr} is not registered. ` +
            `Call registerPrePostProcessor() before processing.`
        )
      },
    })
    return { wasmInstance, wasmMemory, prePostRegistry }
  }

  static async create(
    config?: AniraWasmConfig & Record<string, unknown>,
    memory?: WebAssembly.Memory
  ): Promise<AniraJS> {
    const init = await AniraJS.initWasm(config, memory)
    return new AniraJS(init.wasmInstance, init.wasmMemory, init.prePostRegistry)
  }

  constructor(
    module: AniraWasmInstance,
    memory: WebAssembly.Memory,
    prePostRegistry?: Map<number, JSPrePostProcessor>
  ) {
    this.wasmInstance = module
    this.memory = memory
    this.registeredPrePostProcessors =
      prePostRegistry ?? new Map<number, JSPrePostProcessor>()

    this.InferenceBackend = createInferenceBackend(module)
    this.Buffer = createFactory(module, BufferF)
    this.HostConfig = createFactory(module, HostConfig)
    this.InferenceConfig = createFactory(module, InferenceConfig)
    this.InferenceHandler = createFactory(module, InferenceHandler)
    this.JSBackendBase = createFactory(module, JSBackendBase)
    this.ONNXRuntimeWebBackend = createFactory(module, ONNXRuntimeWebBackend)
    this.JSPrePostProcessor = createFactory(module, JSPrePostProcessor)
    this.ModelData = createFactory(module, ModelData)
    this.PrePostProcessor = createFactory(module, PrePostProcessor)
    this.ProcessingSpec = createFactory(module, ProcessingSpec)
    this.RingBuffer = createFactory(module, RingBuffer)
    this.TensorShape = createFactory(module, TensorShape)
    this.WasmInferenceThread = createFactory(module, WasmInferenceThread)
    this.VectorBufferF = createFactory(module, VectorBufferF)
    this.VectorFloat = createFactory(module, VectorFloat)
    this.VectorInt64T = createFactory(module, VectorInt64T)
    this.VectorModelData = createFactory(module, VectorModelData)
    this.VectorRingBuffer = createFactory(module, VectorRingBuffer)
    this.VectorSizeT = createFactory(module, VectorSizeT)
    this.VectorTensorShape = createFactory(module, VectorTensorShape)
    this.VectorUnsignedInt = createFactory(module, VectorUnsignedInt)
    this.VectorVectorInt64 = createFactory(module, VectorVectorInt64)
    this.TensorShapeList = createFactory(module, TensorShapeList)
  }

  stackRestore(ptr: number): void {
    this.wasmInstance.stackRestore(ptr)
  }

  malloc(size: number): number {
    return this.wasmInstance._malloc(size)
  }

  free(ptr: number): void {
    this.wasmInstance._free(ptr)
  }

  getMemory(): WebAssembly.Memory {
    return this.memory
  }

  getWasmInstance(): AniraWasmInstance {
    return this.wasmInstance
  }

  getHeapF32(): Float32Array {
    return this.wasmInstance.HEAPF32
  }

  getHeapU32(): Float32Array {
    return this.wasmInstance.HEAPU32
  }

  // ---- General utilities ----

  allocWasmString(str: string): number {
    const bytes = new TextEncoder().encode(str + '\0')
    const ptr = this.wasmInstance._malloc(bytes.length)
    new Uint8Array(this.wasmInstance.HEAPU32.buffer, ptr, bytes.length).set(bytes)
    return ptr
  }

  // ---- Worker & Audio Worklet helpers ----

  protected async ensureWasmBinary(): Promise<ArrayBuffer> {
    if (!this.wasmBinary) {
      const res = await fetch(wasmUrl)
      this.wasmBinary = await res.arrayBuffer()
    }
    return this.wasmBinary
  }

  protected allocateWorkerStack(): number {
    const WORKER_STACK_SIZE = 4194304 // 4 MB per worker stack
    const base = this.malloc(WORKER_STACK_SIZE)
    if (!base) throw new Error('Failed to allocate worker stack')
    return base + WORKER_STACK_SIZE
  }

  async registerProcessor(backend: JSBackendBase, className: string): Promise<void> {
    const descriptor: ProcessorDescriptor = { backend, className }
    this.registeredProcessors.push(descriptor)
    await Promise.all(this.activeWorkers.map((w) => w.registerProcessor(descriptor)))
  }

  registerPrePostProcessor(prePostProcessor: JSPrePostProcessor): void {
    this.registeredPrePostProcessors.set(prePostProcessor.getPointer(), prePostProcessor)
  }

  unregisterPrePostProcessor(
    prePostProcessor: PossiblePointer<JSPrePostProcessor>
  ): void {
    this.registeredPrePostProcessors.delete(resolvePtr(prePostProcessor))
  }

  getActiveWorkers(): readonly InferenceWorker[] {
    return this.activeWorkers
  }

  async spinUpInferenceWorker(workerOrUrl?: Worker | URL): Promise<InferenceWorker> {
    const inferenceThread = this.WasmInferenceThread()

    const inferenceStackPtr = this.allocateWorkerStack()
    let worker: Worker
    if (workerOrUrl instanceof Worker) {
      worker = workerOrUrl
    } else if (workerOrUrl) {
      worker = new Worker(workerOrUrl, { type: 'module' })
    } else {
      worker = new Worker(new URL('./workers/inference-worker.ts', import.meta.url), {
        type: 'module',
      })
    }

    worker.postMessage({
      type: 'initInferenceWorker',
      wasmMemory: this.memory,
      stackPtr: inferenceStackPtr,
      threadPtr: inferenceThread.getPointer(),
    } satisfies InitInferenceWorkerMessage)
    await waitForWorkerMessage(worker, 'ready')

    for (const { backend, className } of this.registeredProcessors) {
      worker.postMessage({
        type: 'registerProcessor',
        processorPtr: backend.getPointer(),
        className,
        inferenceConfigPtr: backend.inferenceConfigPtr || undefined,
      } satisfies RegisterProcessorMessage)
      await waitForWorkerMessage(worker, 'processorRegistered')
    }

    inferenceThread.start()
    worker.postMessage({ type: 'start' } satisfies StartMessage)

    const inferenceWorker: InferenceWorker = {
      worker,
      registerProcessor: async (descriptor: ProcessorDescriptor) => {
        inferenceThread.stop()
        await waitForWorkerMessage(worker, 'stopped')

        worker.postMessage({
          type: 'registerProcessor',
          processorPtr: descriptor.backend.getPointer(),
          className: descriptor.className,
          inferenceConfigPtr: descriptor.backend.inferenceConfigPtr || undefined,
        } satisfies RegisterProcessorMessage)
        await waitForWorkerMessage(worker, 'processorRegistered')

        inferenceThread.start()
        worker.postMessage({ type: 'start' } satisfies StartMessage)
      },
      stop: async () => {
        inferenceThread.stop()
        await waitForWorkerMessage(worker, 'stopped')
        worker.postMessage({ type: 'destroy' } satisfies DestroyMessage)
        const idx = this.activeWorkers.indexOf(inferenceWorker)
        if (idx !== -1) this.activeWorkers.splice(idx, 1)
      },
    }

    this.activeWorkers.push(inferenceWorker)
    return inferenceWorker
  }

  async registerAudioWorkletForContext(
    audioContext: AudioContext,
    workletUrl?: string | URL
  ): Promise<void> {
    const url =
      workletUrl ?? new URL('./workers/audio-worklet.bundled.js', import.meta.url)
    await audioContext.audioWorklet.addModule(url)
  }

  async configureAudioWorklet(
    audioContext: AudioContext,
    inferenceHandlerPtr: PossiblePointer<InferenceHandler>,
    prePostProcessorPtr: PossiblePointer<PrePostProcessor>,
    audioWorkletNodeName = 'inference-processor',
    ioOptions: ConfigureAudioWorkletIOOptions = {}
  ): Promise<AudioWorkletNode> {
    const wasmMemory = this.memory
    const ioConfig: AudioWorkletIOConfig = {
      maxBufferSize: ioOptions.maxBufferSize ?? 1024,
      inputNodeIndex: ioOptions.inputNodeIndex ?? 0,
      outputNodeIndex: ioOptions.outputNodeIndex ?? 0,
      inputChannels: ioOptions.inputChannels ?? 2,
      outputChannels: ioOptions.outputChannels ?? 2,
    }

    if (
      ioConfig.maxBufferSize <= 0 ||
      ioConfig.inputChannels <= 0 ||
      ioConfig.outputChannels <= 0
    ) {
      throw new Error(
        'Invalid AudioWorklet IO config: sizes and channel counts must be > 0'
      )
    }

    const wasmBinary = await this.ensureWasmBinary()

    const inferenceNode = new AudioWorkletNode(audioContext, audioWorkletNodeName)
    const processStackPtr = this.allocateWorkerStack()
    const bytesPerChannel = ioConfig.maxBufferSize * Float32Array.BYTES_PER_ELEMENT

    const inputDataBuffer = this.malloc(bytesPerChannel * ioConfig.inputChannels)
    const outputDataBuffer = this.malloc(bytesPerChannel * ioConfig.outputChannels)

    const inputBufferPtr = this.malloc(ioConfig.inputChannels * 4)
    const outputBufferPtr = this.malloc(ioConfig.outputChannels * 4)

    const inputPtrArray = new Uint32Array(
      wasmMemory.buffer,
      inputBufferPtr,
      ioConfig.inputChannels
    )
    const outputPtrArray = new Uint32Array(
      wasmMemory.buffer,
      outputBufferPtr,
      ioConfig.outputChannels
    )
    for (let i = 0; i < ioConfig.inputChannels; i++) {
      inputPtrArray[i] = inputDataBuffer + i * bytesPerChannel
    }
    for (let i = 0; i < ioConfig.outputChannels; i++) {
      outputPtrArray[i] = outputDataBuffer + i * bytesPerChannel
    }

    inferenceNode.port.start()
    inferenceNode.port.postMessage({
      type: 'configure',
      wasmMemory,
      wasmBinary,
      stackPtr: processStackPtr,
      inferenceHandlerPtr: resolvePtr(inferenceHandlerPtr),
      prePostProcessorPtr: resolvePtr(prePostProcessorPtr),
      inputBufferPtr,
      outputBufferPtr,
      inputDataBuffer,
      outputDataBuffer,
      ioConfig,
    } satisfies AudioWorkletConfigureMessage)

    await waitForWorkerMessage(inferenceNode.port, 'ready')
    return inferenceNode
  }
}
