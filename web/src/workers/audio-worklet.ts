import { AniraAudioWorkletBase } from './AniraAudioWorkletBase'

class InferenceWorklet extends AniraAudioWorkletBase {}

registerProcessor('inference-processor', InferenceWorklet)
