// Separate entry point for AudioWorklet-scope code.
// AudioWorkletProcessor is only available inside an AudioWorklet,
// so this must not be imported from the main thread.
export { AniraAudioWorkletBase, type AniraWorkletState } from './AniraAudioWorkletBase'
