import AniraJSFactory from '../wasm/AniraJS'

// Lazy-evaluated so the module can be imported in AudioWorkletGlobalScope
// where URL may not be available. The URLs are only needed on the main thread.
let _jsUrl: string | undefined
let _wasmUrl: string | undefined

const getJsUrl = () => (_jsUrl ??= new URL('../wasm/AniraJS.js', import.meta.url).href)
const getWasmUrl = () => (_wasmUrl ??= new URL('../wasm/AniraJS.wasm', import.meta.url).href)

export { getWasmUrl }

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
        return getWasmUrl()
      }
      if (path.endsWith('.js')) {
        return getJsUrl()
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
