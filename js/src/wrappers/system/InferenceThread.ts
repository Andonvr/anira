import { type AniraWasmInstance } from '../../factory'
import { BaseWrapper } from '../BaseWrapper'

export class InferenceThread extends BaseWrapper {
  constructor(wasmInstance: AniraWasmInstance) {
    super(wasmInstance, wasmInstance._inference_thread_create_from_context())
  }

  destroy(): void {
    this._destroy(this.wasmInstance._inference_thread_destroy)
  }

  execute(): boolean {
    return this.wasmInstance._inference_thread_execute(this.ptr) === 1
  }

  runLoop(): void {
    this.wasmInstance._inference_thread_run_loop(this.ptr)
  }

  stop(): void {
    this.wasmInstance._inference_thread_stop(this.ptr)
  }

  start(): void {
    this.wasmInstance._inference_thread_start(this.ptr)
  }

  shouldExit(): boolean {
    return this.wasmInstance._inference_thread_should_exit(this.ptr) === 1
  }

  isRunning(): boolean {
    return this.wasmInstance._inference_thread_is_running(this.ptr) === 1
  }
}
