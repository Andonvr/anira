import { JSBackendBase } from './JSBackendBase'
import { BufferF } from '../wrappers/utils/BufferF'
import { VectorBufferF, VectorInt64T, type TensorShapeList } from '../wrappers/Vectors'
import { InferenceConfig } from '../wrappers/InferenceConfig'
import { ModelData } from '../wrappers/ModelData'

// onnxruntime-web 1.19.2 doesn't expose this WASM module factory via its
// package `exports` field. The Vite config aliases this specifier to
// node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs.
import ortWasmFactory from 'onnxruntime-web/ort-wasm-simd-threaded.mjs'
import { createInferenceBackend } from '../wrappers'

// Shared ORT WASM module — loaded once and reused across all backend instances
// to avoid accumulating WebAssembly.Memory allocations (causes OOM on Safari).
let ortModulePromise: Promise<OrtWasmModule> | null = null

/**
 * Minimal type for the ORT WASM Emscripten module instance.
 * We use the module directly for synchronous access to the C API.
 */
export interface OrtWasmModule {
  _OrtInit(numThreads: number, loggingLevel: number): number
  _OrtCreateSessionOptions(
    graphOptLevel: number,
    enableCpuMemArena: number,
    enableMemPattern: number,
    executionMode: number,
    enableProfiling: number,
    profileFilePrefix: number,
    logId: number,
    logSeverityLevel: number,
    logVerbosityLevel: number,
    optimizedModelFilePath: number
  ): number
  _OrtReleaseSessionOptions(handle: number): number
  _OrtCreateSession(
    modelData: number,
    modelDataLength: number,
    sessionOptions: number
  ): number
  _OrtReleaseSession(handle: number): number
  _OrtGetInputOutputCount(
    sessionHandle: number,
    inputCountOffset: number,
    outputCountOffset: number
  ): number
  _OrtGetInputName(sessionHandle: number, index: number): number
  _OrtGetOutputName(sessionHandle: number, index: number): number
  _OrtCreateTensor(
    dataType: number,
    data: number,
    dataByteLength: number,
    dimsOffset: number,
    dimsLength: number,
    dataLocation: number
  ): number
  _OrtReleaseTensor(handle: number): number
  _OrtRun(
    sessionHandle: number,
    inputNamesOffset: number,
    inputValuesOffset: number,
    inputCount: number,
    outputNamesOffset: number,
    outputCount: number,
    outputValuesOffset: number,
    runOptionsHandle: number
  ): number
  _OrtGetTensorData(
    tensorHandle: number,
    dataTypeOffset: number,
    dataOffset: number,
    dimsOffset: number,
    dimsLengthOffset: number
  ): number
  _OrtGetLastError(errorCodeOffset: number, errorMessageOffset: number): void
  _OrtCreateRunOptions(
    logSeverityLevel: number,
    logVerbosityLevel: number,
    terminate: number,
    tag: number
  ): number
  _OrtReleaseRunOptions(handle: number): number
  _OrtFree(ptr: number): number
  _malloc(size: number): number
  _free(ptr: number): void
  stackSave(): number
  stackRestore(ptr: number): void
  stackAlloc(size: number): number
  setValue(ptr: number, value: number, type: string): void
  getValue(ptr: number, type: string): number
  UTF8ToString(ptr: number): string
  HEAPU8: Uint8Array
  HEAP32: Int32Array
  HEAPU32: Uint32Array
  HEAPF32: Float32Array
  PTR_SIZE: number
}

/** Per-input/output metadata stored after session creation. */
interface TensorMeta {
  namePtr: number
  dims: number[]
  flatSize: number
}

/**
 * onnxruntime-web 1.19.2's Emscripten build doesn't export the `getValue` /
 * `setValue` runtime helpers or the `PTR_SIZE` constant — those arrived with
 * wasm64 support in 1.21+. This backend drives the ORT C API directly and
 * relies on them, so we polyfill them onto the module after load. 1.19.2 is a
 * wasm32 build, so pointers are 4 bytes and `i64` / `*` accesses reduce to
 * 32-bit reads and writes. HEAP views are read fresh on each access so the
 * polyfill stays correct across memory growth.
 */
function patchOrtRuntime(ort: OrtWasmModule): void {
  if (ort.PTR_SIZE === undefined) {
    ort.PTR_SIZE = 4 // wasm32
  }

  if (typeof ort.getValue !== 'function') {
    ort.getValue = (ptr, type) => {
      switch (type) {
        case 'i1':
        case 'i8':
          return new Int8Array(ort.HEAPU8.buffer)[ptr]
        case 'i16':
          return new Int16Array(ort.HEAPU8.buffer)[ptr >> 1]
        case 'i32':
        case 'i64': // wasm32: low 32 bits
          return ort.HEAP32[ptr >> 2]
        case 'float':
          return ort.HEAPF32[ptr >> 2]
        case 'double':
          return new Float64Array(ort.HEAPU8.buffer)[ptr >> 3]
        case '*':
          return ort.HEAPU32[ptr >> 2]
        default:
          throw new Error(`ORT getValue: unsupported type "${type}"`)
      }
    }
  }

  if (typeof ort.setValue !== 'function') {
    ort.setValue = (ptr, value, type) => {
      switch (type) {
        case 'i1':
        case 'i8':
          new Int8Array(ort.HEAPU8.buffer)[ptr] = value
          break
        case 'i16':
          new Int16Array(ort.HEAPU8.buffer)[ptr >> 1] = value
          break
        case 'i32':
          ort.HEAP32[ptr >> 2] = value
          break
        case 'i64': // wasm32: low 32 bits, zero the high word
          ort.HEAP32[ptr >> 2] = value
          ort.HEAP32[(ptr >> 2) + 1] = 0
          break
        case 'float':
          ort.HEAPF32[ptr >> 2] = value
          break
        case 'double':
          new Float64Array(ort.HEAPU8.buffer)[ptr >> 3] = value
          break
        case '*':
          ort.HEAPU32[ptr >> 2] = value
          break
        default:
          throw new Error(`ORT setValue: unsupported type "${type}"`)
      }
    }
  }
}

/**
 * ONNX Runtime Web backend implementation.
 * Loads the ORT WASM module directly for synchronous inference in the
 * process() callback, mirroring the native OnnxRuntimeProcessor.
 */
export class ONNXRuntimeWebBackend extends JSBackendBase {
  private ort: OrtWasmModule | null = null
  private sessionHandle: number = 0
  private runOptionsHandle: number = 0
  private inputMeta: TensorMeta[] = []
  private outputMeta: TensorMeta[] = []

  /**
   * Async initialization: loads the ORT WASM module, creates an inference
   * session from the model binary stored in anira's shared WASM memory.
   * Called automatically by the worker handler after processor registration.
   */
  async init(): Promise<void> {
    const m = this.wasmInstance
    const inferenceBackend = createInferenceBackend(m)
    const configPtr = this.inferenceConfigPtr
    if (!configPtr) {
      throw new Error(
        'ONNXRuntimeWebBackend: no inferenceConfigPtr – was the backend registered correctly?'
      )
    }

    // --- Extract model binary from anira WASM memory ---
    const config = this.wrapPointer(InferenceConfig, configPtr)
    const customBackend = inferenceBackend.CUSTOM
    const modelDataPtr = config.getModelData(customBackend)
    if (!modelDataPtr) {
      throw new Error('ONNXRuntimeWebBackend: no model data for CUSTOM backend')
    }

    const modelData = this.wrapPointer(ModelData, modelDataPtr)
    let modelBytes: Uint8Array

    if (modelData.isBinary()) {
      const modelBinaryPtr = modelData.getDataPtr()
      const modelSize = modelData.getSize()
      modelBytes = new Uint8Array(m.HEAPU32.buffer, modelBinaryPtr, modelSize).slice()
    } else {
      const pathPtr = modelData.getDataPtr()
      const pathLen = modelData.getSize()
      const pathBytes = new Uint8Array(m.HEAPU32.buffer, pathPtr, pathLen).slice()
      const modelUrl = new TextDecoder().decode(pathBytes)
      const response = await fetch(modelUrl)
      if (!response.ok) {
        throw new Error(
          `ONNXRuntimeWebBackend: failed to fetch model from ${modelUrl}: ${response.status}`
        )
      }
      modelBytes = new Uint8Array(await response.arrayBuffer())
    }

    // --- Load ORT WASM module (shared singleton to avoid repeated memory allocs) ---
    ortModulePromise ??= (
      ortWasmFactory({ numThreads: 1 }) as Promise<OrtWasmModule>
    ).then((ort) => {
      patchOrtRuntime(ort)
      return ort
    })
    this.ort = await ortModulePromise
    const ort = this.ort

    if (ort._OrtInit(1, 3) !== 0) {
      throw new Error('ONNXRuntimeWebBackend: _OrtInit failed')
    }

    // --- Create session ---
    const sessionOpts = ort._OrtCreateSessionOptions(99, 1, 1, 0, 0, 0, 0, 3, 0, 0)
    if (sessionOpts === 0) {
      throw new Error('ONNXRuntimeWebBackend: _OrtCreateSessionOptions failed')
    }

    const modelOffset = ort._malloc(modelBytes.length)
    if (modelOffset === 0) {
      throw new Error('ONNXRuntimeWebBackend: ORT _malloc failed for model data')
    }
    ort.HEAPU8.set(modelBytes, modelOffset)

    this.sessionHandle = ort._OrtCreateSession(
      modelOffset,
      modelBytes.length,
      sessionOpts
    )
    ort._free(modelOffset)
    ort._OrtReleaseSessionOptions(sessionOpts)

    if (this.sessionHandle === 0) {
      const ptrSize = ort.PTR_SIZE
      const errStack = ort.stackSave()
      const errBuf = ort.stackAlloc(2 * ptrSize)
      ort._OrtGetLastError(errBuf, errBuf + ptrSize)
      const errCode = Number(ort.getValue(errBuf, ptrSize === 4 ? 'i32' : 'i64'))
      const errMsgPtr = Number(ort.getValue(errBuf + ptrSize, '*'))
      const errMsg = errMsgPtr ? ort.UTF8ToString(errMsgPtr) : ''
      ort.stackRestore(errStack)
      throw new Error(
        `ONNXRuntimeWebBackend: _OrtCreateSession failed (code=${errCode}): ${errMsg}`
      )
    }

    this.runOptionsHandle = ort._OrtCreateRunOptions(2, 0, 0, 0)
    if (this.runOptionsHandle === 0) {
      throw new Error('ONNXRuntimeWebBackend: _OrtCreateRunOptions failed')
    }

    // --- Query input / output metadata ---
    const ptrSize = ort.PTR_SIZE
    const countStack = ort.stackSave()
    const countBuf = ort.stackAlloc(2 * ptrSize)
    if (
      ort._OrtGetInputOutputCount(this.sessionHandle, countBuf, countBuf + ptrSize) !== 0
    ) {
      ort.stackRestore(countStack)
      throw new Error('ONNXRuntimeWebBackend: _OrtGetInputOutputCount failed')
    }
    const inputCount = Number(ort.getValue(countBuf, ptrSize === 4 ? 'i32' : 'i64'))
    const outputCount = Number(
      ort.getValue(countBuf + ptrSize, ptrSize === 4 ? 'i32' : 'i64')
    )
    ort.stackRestore(countStack)

    // onnxruntime-web 1.19.2 has no shape-metadata C-API, so tensor dims come
    // from the anira InferenceConfig (per-backend shape, falling back to the
    // universal shape). Names still come from ORT.
    const tensorShape = config.getTensorShape(customBackend)
    const inputShapes = this.readShapeList(tensorShape.getTensorInputShape())
    const outputShapes = this.readShapeList(tensorShape.getTensorOutputShape())

    this.inputMeta = this.queryMetadata(ort, true, inputCount, inputShapes)
    this.outputMeta = this.queryMetadata(ort, false, outputCount, outputShapes)

    // --- Warm-up inference (non-fatal, matches C++ behaviour) ---
    // Skip warm-up when any input has dynamic dims — we can't construct a
    // valid concrete shape without actual buffer data.
    const hasDynamicDims = this.inputMeta.some((m) => m.dims.includes(-1))
    if (!hasDynamicDims) {
      const warmUp = config.getWarmUp()
      for (let i = 0; i < warmUp; i++) {
        try {
          const inputs = this.createZeroInputs()
          const outputs = this.runOrt(inputs)
          for (const t of outputs) if (t !== 0) ort._OrtReleaseTensor(t)
        } catch {
          break
        }
      }
    }
  }

  override process(inputVecPtr: number, outputVecPtr: number): void {
    if (!this.ort || !this.sessionHandle) {
      super.process(inputVecPtr, outputVecPtr)
      return
    }

    const heapF32 = this.wasmInstance.HEAPF32
    const ort = this.ort
    const inputVec = this.wrapPointer(VectorBufferF, inputVecPtr)
    const outputVec = this.wrapPointer(VectorBufferF, outputVecPtr)

    const numInputBufs = inputVec.size()
    const numOutputBufs = outputVec.size()

    const inputTensors: number[] = []
    const allocs: number[] = []

    try {
      // --- Build input tensors from anira buffers ---
      for (let i = 0; i < Math.min(numInputBufs, this.inputMeta.length); i++) {
        const meta = this.inputMeta[i]
        const buf = this.wrapPointer(BufferF, inputVec.get(i))
        const channels = buf.getNumChannels()
        const samples = buf.getNumSamples()
        const totalFloats = channels * samples
        const byteLen = totalFloats * 4

        const dataOff = ort._malloc(byteLen)
        allocs.push(dataOff)

        // Copy channel data linearly into ORT memory
        for (let ch = 0; ch < channels; ch++) {
          const readPtr = buf.getReadPointer(ch)
          const inputOff = readPtr >> 2
          const ortF32 = new Float32Array(ort.HEAPU8.buffer, dataOff, totalFloats)
          for (let s = 0; s < samples; s++) {
            ortF32[ch * samples + s] = heapF32[inputOff + s]
          }
        }

        // Resolve dynamic dims (-1) from actual buffer dimensions
        const concreteDims = this.resolveDynamicDims(meta.dims, totalFloats)

        const stack = ort.stackSave()
        const dimsOff = ort.stackAlloc(concreteDims.length * ort.PTR_SIZE)
        for (let d = 0; d < concreteDims.length; d++) {
          ort.setValue(
            dimsOff + d * ort.PTR_SIZE,
            concreteDims[d],
            ort.PTR_SIZE === 4 ? 'i32' : 'i64'
          )
        }
        const tensor = ort._OrtCreateTensor(
          1,
          dataOff,
          byteLen,
          dimsOff,
          concreteDims.length,
          1
        )
        ort.stackRestore(stack)

        if (tensor === 0) throw new Error(`Failed to create ORT input tensor ${i}`)
        inputTensors.push(tensor)
      }

      // --- Run inference ---
      const outputs = this.runOrt(inputTensors)

      // --- Copy outputs to anira buffers ---
      const ptrSize = ort.PTR_SIZE
      for (
        let i = 0;
        i < Math.min(numOutputBufs, this.outputMeta.length, outputs.length);
        i++
      ) {
        const outTensor = outputs[i]
        if (outTensor === 0) continue

        const stack = ort.stackSave()
        const info = ort.stackAlloc(4 * ptrSize)
        ort._OrtGetTensorData(
          outTensor,
          info,
          info + ptrSize,
          info + 2 * ptrSize,
          info + 3 * ptrSize
        )
        const dataPtr = Number(ort.getValue(info + ptrSize, '*'))
        ort.stackRestore(stack)

        const outBuf = this.wrapPointer(BufferF, outputVec.get(i))
        const outCh = outBuf.getNumChannels()
        const outSamp = outBuf.getNumSamples()
        const totalOut = outCh * outSamp

        const ortF32 = new Float32Array(ort.HEAPU8.buffer, dataPtr, totalOut)
        for (let ch = 0; ch < outCh; ch++) {
          const writePtr = outBuf.getWritePointer(ch)
          const writeOff = writePtr >> 2
          for (let s = 0; s < outSamp; s++) {
            heapF32[writeOff + s] = ortF32[ch * outSamp + s]
          }
        }

        ort._OrtReleaseTensor(outTensor)
      }
    } finally {
      for (const t of inputTensors) ort._OrtReleaseTensor(t)
      for (const a of allocs) ort._free(a)
    }
  }

  override destroy(): void {
    if (this.ort && this.sessionHandle) {
      for (const meta of [...this.inputMeta, ...this.outputMeta]) {
        this.ort._OrtFree(meta.namePtr)
      }
      if (this.runOptionsHandle) {
        this.ort._OrtReleaseRunOptions(this.runOptionsHandle)
        this.runOptionsHandle = 0
      }
      this.ort._OrtReleaseSession(this.sessionHandle)
      this.sessionHandle = 0
    }
    this.ort = null
    this.inputMeta = []
    this.outputMeta = []
    super.destroy()
  }

  // ---- private helpers ----

  /**
   * Replace any dynamic dims (-1) with concrete values inferred from the
   * total number of elements. Only a single dynamic dim is supported.
   */
  private resolveDynamicDims(dims: number[], totalElements: number): number[] {
    const dynamicCount = dims.filter((d) => d === -1).length
    if (dynamicCount === 0) return dims

    if (dynamicCount > 1) {
      throw new Error(
        `ONNXRuntimeWebBackend: cannot resolve ${dynamicCount} dynamic dims — at most 1 is supported`
      )
    }

    const staticProduct = dims.reduce((a, d) => (d === -1 ? a : a * d), 1)
    const inferred = Math.floor(totalElements / staticProduct)

    return dims.map((d) => (d === -1 ? inferred : d))
  }

  /**
   * Build per-tensor metadata. Names come from ORT's 1.19.2 C-API
   * (`_OrtGetInputName` / `_OrtGetOutputName`); dims come from the anira
   * config shapes (1.19.2 exposes no shape-metadata function). Non-positive
   * config dims are treated as dynamic (-1) and resolved from buffer sizes at
   * process time.
   */
  private queryMetadata(
    ort: OrtWasmModule,
    isInput: boolean,
    count: number,
    shapes: number[][]
  ): TensorMeta[] {
    const result: TensorMeta[] = []

    for (let i = 0; i < count; i++) {
      const namePtr = isInput
        ? ort._OrtGetInputName(this.sessionHandle, i)
        : ort._OrtGetOutputName(this.sessionHandle, i)
      if (namePtr === 0) {
        throw new Error(
          `ONNXRuntimeWebBackend: failed to get ${isInput ? 'input' : 'output'} name at index ${i}`
        )
      }

      const dims = (shapes[i] ?? []).map((d) => (d > 0 ? d : -1))
      const staticDims = dims.filter((d) => d > 0)
      const flatSize =
        staticDims.length === dims.length ? staticDims.reduce((a, b) => a * b, 1) : 0

      result.push({ namePtr, dims, flatSize })
    }

    return result
  }

  /** Read an anira TensorShapeList (`vector<vector<int64>>`) into `number[][]`. */
  private readShapeList(list: TensorShapeList): number[][] {
    const shapes: number[][] = []
    for (let i = 0; i < list.size(); i++) {
      const inner = this.wrapPointer(VectorInt64T, list.get(i))
      const dims: number[] = []
      for (let d = 0; d < inner.size(); d++) {
        dims.push(Number(inner.get(d)))
      }
      shapes.push(dims)
    }
    return shapes
  }

  private createZeroInputs(): number[] {
    const ort = this.ort!
    const tensors: number[] = []

    for (const meta of this.inputMeta) {
      // For warm-up, replace dynamic dims (-1) with 1
      const concreteDims = meta.dims.map((d) => (d === -1 ? 1 : d))
      const flatSize = concreteDims.reduce((a, b) => a * b, 1)
      const byteLen = flatSize * 4
      const dataOff = ort._malloc(byteLen)
      ort.HEAPU8.fill(0, dataOff, dataOff + byteLen)

      const stack = ort.stackSave()
      const dimsOff = ort.stackAlloc(concreteDims.length * ort.PTR_SIZE)
      for (let d = 0; d < concreteDims.length; d++) {
        ort.setValue(
          dimsOff + d * ort.PTR_SIZE,
          concreteDims[d],
          ort.PTR_SIZE === 4 ? 'i32' : 'i64'
        )
      }
      const tensor = ort._OrtCreateTensor(
        1,
        dataOff,
        byteLen,
        dimsOff,
        concreteDims.length,
        1
      )
      ort.stackRestore(stack)

      tensors.push(tensor)
    }

    return tensors
  }

  private runOrt(inputTensors: number[]): number[] {
    const ort = this.ort!
    const ptrSize = ort.PTR_SIZE
    const inputCount = this.inputMeta.length
    const outputCount = this.outputMeta.length

    const stack = ort.stackSave()
    const inputNamesOff = ort.stackAlloc(inputCount * ptrSize)
    const inputValsOff = ort.stackAlloc(inputCount * ptrSize)
    const outputNamesOff = ort.stackAlloc(outputCount * ptrSize)
    const outputValsOff = ort.stackAlloc(outputCount * ptrSize)

    for (let i = 0; i < inputCount; i++) {
      ort.setValue(inputNamesOff + i * ptrSize, this.inputMeta[i].namePtr, '*')
      ort.setValue(inputValsOff + i * ptrSize, inputTensors[i], '*')
    }
    for (let i = 0; i < outputCount; i++) {
      ort.setValue(outputNamesOff + i * ptrSize, this.outputMeta[i].namePtr, '*')
      ort.setValue(outputValsOff + i * ptrSize, 0, '*')
    }

    const errorCode = ort._OrtRun(
      this.sessionHandle,
      inputNamesOff,
      inputValsOff,
      inputCount,
      outputNamesOff,
      outputCount,
      outputValsOff,
      this.runOptionsHandle
    )

    const outputs: number[] = []
    for (let i = 0; i < outputCount; i++) {
      outputs.push(Number(ort.getValue(outputValsOff + i * ptrSize, '*')))
    }
    ort.stackRestore(stack)

    if (errorCode !== 0) {
      const errStack = ort.stackSave()
      const errBuf = ort.stackAlloc(2 * ptrSize)
      ort._OrtGetLastError(errBuf, errBuf + ptrSize)
      const errCode = Number(ort.getValue(errBuf, ptrSize === 4 ? 'i32' : 'i64'))
      const errMsgPtr = Number(ort.getValue(errBuf + ptrSize, '*'))
      const errMsg = errMsgPtr ? ort.UTF8ToString(errMsgPtr) : ''
      ort.stackRestore(errStack)

      for (const t of outputs) if (t !== 0) ort._OrtReleaseTensor(t)
      throw new Error(
        `ONNXRuntimeWebBackend: _OrtRun failed (code=${errCode}): ${errMsg}`
      )
    }

    return outputs
  }
}
