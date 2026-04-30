import type { AniraWasmInstance } from './factory'
import { BufferF } from './wrappers/utils/BufferF'
import { RingBuffer } from './wrappers/utils/RingBuffer'

/**
 * Mirrors `anira::random_sample()` — uniform sample in [-1.0, 1.0).
 */
export const randomSample = (): number => Math.random() * 2 - 1

/**
 * Mirrors `anira::fill_buffer(BufferF&)` — overwrites every sample on every
 * channel with a fresh `randomSample`. Pure TS; goes through the wrapper
 * so it works for any pointer the wrapper owns.
 */
export const fillBuffer = (buffer: BufferF): void => {
  const numChannels = buffer.getNumChannels()
  const numSamples = buffer.getNumSamples()
  for (let i = 0; i < numChannels; i++) {
    for (let j = 0; j < numSamples; j++) {
      buffer.setSample(i, j, randomSample())
    }
  }
}

/**
 * Mirrors `anira::push_buffer_to_ringbuffer(BufferF const&, RingBuffer&)` —
 * pushes every sample of every channel into the ring buffer in order. Throws
 * if either buffer is uninitialized, matching the C++ pre-condition checks.
 *
 * `wasmInstance` is needed because the TS `RingBuffer` wrapper doesn't expose
 * the inherited `Buffer<float>` accessors (`get_num_channels` / `get_num_samples`);
 * we read those off the ring buffer's pointer via the `bufferf_*` bindings.
 */
export const pushBufferToRingbuffer = (
  wasmInstance: AniraWasmInstance,
  buffer: BufferF,
  ringbuffer: RingBuffer
): void => {
  const numChannels = buffer.getNumChannels()
  const numSamples = buffer.getNumSamples()
  if (numChannels === 0 || numSamples === 0) {
    throw new Error('Buffer is empty, cannot push to ring buffer.')
  }
  const rbPtr = ringbuffer.getPointer()
  const rbChannels = wasmInstance._bufferf_get_num_channels(rbPtr)
  const rbSamples = wasmInstance._bufferf_get_num_samples(rbPtr)
  if (rbChannels === 0 || rbSamples === 0) {
    throw new Error('Ring buffer is not initialized, cannot push samples.')
  }
  for (let i = 0; i < numChannels; i++) {
    for (let j = 0; j < numSamples; j++) {
      ringbuffer.pushSample(i, buffer.getSample(i, j))
    }
  }
}

/**
 * Returns the `ANIRA_VERSION` string baked into the WASM module at compile time.
 */
export const getAniraVersion = (wasmInstance: AniraWasmInstance): string => {
  const ptr = wasmInstance._anira_get_version()
  // ANIRA_VERSION is a static C string literal; bytes live in the .rodata
  // segment of the WASM image, so we just decode until the first NUL.
  const view = new Uint8Array(wasmInstance.HEAPU32.buffer, ptr, 256)
  let end = 0
  while (end < view.length && view[end] !== 0) end++
  return new TextDecoder().decode(view.subarray(0, end))
}
