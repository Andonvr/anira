import AniraJSFactory from '../wasm/AniraJS'
import jsUrl from '../wasm/AniraJS.js?url&no-inline'
import wasmUrl from '../wasm/AniraJS.wasm?url&no-inline'

export { wasmUrl }

export type AniraWasmConfig = {
  processBuffers?: (processorPtr: number, inputPtr: number, outputPtr: number) => void
  processPrePost?: (
    prePostProcessorPtr: number,
    inputPtr: number,
    outputPtr: number,
    backend: number,
    phase: number
  ) => void
  wasmBinary?: ArrayBuffer
}

// Export factory with WASM locateFile override
export const createAniraWasm = async (
  wasmMemory: WebAssembly.Memory,
  config?: AniraWasmConfig & Record<string, unknown>
) => {
  const { processBuffers, processPrePost, wasmBinary, ...rest } = config ?? {}
  const out = await AniraJSFactory({
    processBuffers: processBuffers ?? (() => {}),
    processPrePost: processPrePost ?? (() => {}),
    wasmBinary,
    ...rest,
    wasmMemory,
    locateFile: (path: string) => {
      if (path.endsWith('.wasm')) {
        return wasmUrl
      }
      if (path.endsWith('.js')) {
        return jsUrl
      }
      return path
    },
  })

  return {
    ...out,
    HEAPF32: out.HEAPF32 as Float32Array,
    HEAPU32: out.HEAPU32 as Float32Array,
  }
}
export type AniraWasmInstance = Awaited<ReturnType<typeof createAniraWasm>>
