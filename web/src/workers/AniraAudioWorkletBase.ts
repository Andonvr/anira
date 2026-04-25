import { AniraWeb } from '../AniraWeb'
import type { InferenceHandler } from '../wrappers'
import type {
  AudioWorkletConfigureMessage,
  AudioWorkletIOConfig,
  ReadyRespose,
} from './messages'

export type AniraWorkletState = {
  wasmMemory: WebAssembly.Memory
  aniraWeb: AniraWeb
  inferenceHandler: InferenceHandler
  prePostProcessorPtr: number
  inputBufferPtr: number
  outputBufferPtr: number
  inputDataBuffer: number
  outputDataBuffer: number
  ioConfig: AudioWorkletIOConfig
  inputChannelViews: Float32Array[]
  outputChannelViews: Float32Array[]
}

export class AniraAudioWorkletBase extends AudioWorkletProcessor {
  protected aniraState: AniraWorkletState | null = null

  private clearOutputs(outputs: Float32Array[][]): void {
    for (const outputNode of outputs) {
      for (const channel of outputNode) {
        channel.fill(0)
      }
    }
  }

  constructor(options?: AudioWorkletNodeOptions) {
    super(options)

    this.port.onmessage = async (e: MessageEvent<AudioWorkletConfigureMessage>) => {
      const message = e.data
      if (message.type !== 'configure') return

      const {
        inferenceHandlerPtr,
        prePostProcessorPtr,
        inputBufferPtr,
        outputBufferPtr,
        inputDataBuffer,
        outputDataBuffer,
        wasmMemory,
        wasmBinary,
        stackPtr,
        ioConfig,
      } = message

      const aniraWeb = await AniraWeb.create({ wasmBinary }, wasmMemory)
      aniraWeb.stackRestore(stackPtr)
      const inferenceHandler = aniraWeb.InferenceHandler.fromPointer(inferenceHandlerPtr)
      if (!inferenceHandler) {
        console.error('Failed to create inference handler from pointer')
        return
      }

      const bytesPerChannel = ioConfig.maxBufferSize * Float32Array.BYTES_PER_ELEMENT
      const inputChannelViews: Float32Array[] = []
      const outputChannelViews: Float32Array[] = []
      for (let i = 0; i < ioConfig.inputChannels; i++) {
        inputChannelViews.push(
          new Float32Array(
            wasmMemory.buffer,
            inputDataBuffer + i * bytesPerChannel,
            ioConfig.maxBufferSize
          )
        )
      }
      for (let i = 0; i < ioConfig.outputChannels; i++) {
        outputChannelViews.push(
          new Float32Array(
            wasmMemory.buffer,
            outputDataBuffer + i * bytesPerChannel,
            ioConfig.maxBufferSize
          )
        )
      }

      this.aniraState = {
        wasmMemory,
        aniraWeb,
        inferenceHandler,
        prePostProcessorPtr,
        inputBufferPtr,
        outputBufferPtr,
        inputDataBuffer,
        outputDataBuffer,
        ioConfig,
        inputChannelViews,
        outputChannelViews,
      }

      await this.onConfigured(this.aniraState)
      this.port.postMessage({ type: 'ready' } satisfies ReadyRespose)
    }
  }

  protected async onConfigured(_state: AniraWorkletState) {
    // Hook for subclasses that need one-time setup after configure.
  }

  protected processAudioBlock(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    state: AniraWorkletState,
    bufferSize: number
  ): void {
    const {
      inferenceHandler,
      inputBufferPtr,
      outputBufferPtr,
      ioConfig,
      inputChannelViews,
      outputChannelViews,
    } = state
    const inputNode = inputs[ioConfig.inputNodeIndex]
    const outputNode = outputs[ioConfig.outputNodeIndex]

    if (outputNode && outputNode.length > 0) {
      for (let ch = 0; ch < outputNode.length; ch++) {
        outputNode[ch].fill(0)
      }
    }

    if (!inputNode || inputNode.length === 0) {
      for (let ch = 0; ch < inputChannelViews.length; ch++) {
        inputChannelViews[ch].fill(0, 0, bufferSize)
      }
      return
    }

    const inputCount = Math.min(inputNode.length, inputChannelViews.length)
    for (let ch = 0; ch < inputCount; ch++) {
      inputChannelViews[ch].set(inputNode[ch], 0)
    }
    for (let ch = inputCount; ch < inputChannelViews.length; ch++) {
      inputChannelViews[ch].fill(0, 0, bufferSize)
    }

    const samplesProcessed = inferenceHandler.processSeparate(
      inputBufferPtr,
      bufferSize,
      outputBufferPtr,
      bufferSize,
      0
    )

    if (outputNode && outputNode.length > 0 && samplesProcessed > 0) {
      const outputCount = Math.min(outputNode.length, outputChannelViews.length)
      for (let ch = 0; ch < outputCount; ch++) {
        const src = outputChannelViews[ch]
        const dst = outputNode[ch]
        const n = Math.min(samplesProcessed, dst.length, ioConfig.maxBufferSize)
        for (let i = 0; i < n; i++) {
          dst[i] = src[i]
        }
      }
    }
  }

  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    _parameters: Record<string, Float32Array>
  ): boolean {
    if (!this.aniraState) {
      // AudioWorklet process() can run before the async configure handshake finishes.
      this.clearOutputs(outputs)
      return true
    }

    const outputNode = outputs[this.aniraState.ioConfig.outputNodeIndex]
    const inputNode = inputs[this.aniraState.ioConfig.inputNodeIndex]
    const requestedBufferSize = outputNode?.[0]?.length || inputNode?.[0]?.length || 0
    const bufferSize = Math.min(
      requestedBufferSize,
      this.aniraState.ioConfig.maxBufferSize
    )
    if (bufferSize === 0) return true

    this.processAudioBlock(inputs, outputs, this.aniraState, bufferSize)
    return true
  }
}
