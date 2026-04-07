#include <emscripten/emscripten.h>
#include <atomic>
#include "anira/InferenceHandler.h"
#include "anira/scheduler/Context.h"
#include "anira/scheduler/SessionElement.h"
#include "anira/utils/InferenceBackend.h"
#include <concurrentqueue.h>

/**
 * WasmInferenceThread: An allocation-free inference thread for WebAssembly workers.
 *
 * Background / why this exists
 * ─────────────────────────────
 * When sharing a single WebAssembly.Memory across multiple WASM module instances
 * (e.g. main thread + workers), only ONE instance owns the heap allocator
 * (emmalloc).  Any call to malloc/free from a different instance corrupts the
 * allocator state.
 *
 * The standard anira::InferenceThread::execute() calls
 *   moodycamel::ConcurrentQueue::try_dequeue(item)
 * with an *implicit* consumer token.  On the very first dequeue call from a new
 * operating-system thread, moodycamel internally allocates a ThreadToken object
 * to cache per-thread consumer state.  When execute() is called from a worker,
 * that allocation happens inside the worker's WASM instance — triggering the
 * allocator corruption described above.
 *
 * This class fixes the problem by pre-allocating the moodycamel::ConsumerToken
 * explicitly during construction, which MUST happen on the main thread (the
 * one that owns the allocator).  Subsequent execute() calls use the explicit
 * overload  try_dequeue(ConsumerToken&, item)  which is fully allocation-free
 * and therefore safe from any WASM instance / OS thread.
 *
 * Allocation audit for execute()
 * ───────────────────────────────
 *  1. try_dequeue(token, item)          – no alloc (token pre-allocated)
 *  2. shared_ptr copy / move            – no alloc (atomic ref-count ops only)
 *  3. BackendBase::process()            – no alloc (reads/writes pre-allocated buffers)
 *  4. done_atomic.store() / semaphore.release() – no alloc
 *  5. active_inferences fetch_add/sub   – no alloc
 *
 * Usage
 * ─────
 *  // Main thread:
 *  uintptr_t threadPtr = wasm_inference_thread_create(handlerPtr);
 *
 *  // Pass threadPtr to worker via postMessage, then in the worker:
 *  //   wasm_inference_thread_execute(threadPtr)  ← allocation-free
 */
class WasmInferenceThread {
public:
    WasmInferenceThread(moodycamel::ConcurrentQueue<anira::InferenceData>& queue)
        : m_queue(queue)
        , m_consumer_token(queue)   // allocates once — must run on the allocating WASM instance
    {}

    /**
     * Dequeue one inference item and execute it.
     * Completely allocation-free — safe to call from any WASM instance / worker thread.
     *
     * @return true  if an inference job was found and executed.
     *         false if the queue was empty.
     */
    bool execute() {
        // try_dequeue with an explicit ConsumerToken is the only allocation-free
        // dequeue path in moodycamel.  The ordinary implicit-token overload
        // allocates a per-thread ThreadToken on its first call from a new thread.
        if (m_queue.try_dequeue(m_consumer_token, m_current_data)) {
            auto* session = m_current_data.m_session.get();
            auto* tss     = m_current_data.m_thread_safe_struct.get();

            if (session && tss && session->m_initialized.load(std::memory_order::acquire)) {
                session->m_active_inferences.fetch_add(1, std::memory_order::release);

                // Dispatch to the appropriate backend based on m_current_backend.
                // shared_ptr copy increments the ref-count atomically — no alloc.
                auto backend = session->m_current_backend.load(std::memory_order_relaxed);
#ifdef USE_ONNXRUNTIME
                if (backend == anira::ONNX && session->m_onnx_processor != nullptr) {
                    session->m_onnx_processor->process(
                        tss->m_tensor_input_data,
                        tss->m_tensor_output_data,
                        m_current_data.m_session
                    );
                } else
#endif
                if (backend == anira::CUSTOM) {
                    session->m_custom_processor->process(
                        tss->m_tensor_input_data,
                        tss->m_tensor_output_data,
                        m_current_data.m_session
                    );
                }

                if (session->m_inference_config.m_blocking_ratio > 0.f) {
                    tss->m_done_semaphore.release();
                } else {
                    tss->m_done_atomic.store(true, std::memory_order::release);
                }

                session->m_active_inferences.fetch_sub(1, std::memory_order::release);
            }
            return true;
        }
        return false;
    }

    /**
     * Run the inference loop until stop() is called.
     * Completely allocation-free — safe to call from any WASM instance / worker thread.
     */
    void run_loop() {
        while (!should_exit()) {
            execute();
        }
    }

    void stop()  { m_should_exit.store(true,  std::memory_order::release); }
    void start() { m_should_exit.store(false, std::memory_order::release); }

    bool should_exit() const { return m_should_exit.load(std::memory_order::acquire); }
    bool is_running()  const { return !should_exit(); }

private:
    moodycamel::ConcurrentQueue<anira::InferenceData>& m_queue;

    /// Pre-allocated consumer token.  Construction happens once on the main thread;
    /// after that dequeue calls are allocation-free from any thread.
    moodycamel::ConsumerToken m_consumer_token;

    /// Reusable storage for the dequeued item.  Stored as a member so that the
    /// shared_ptr move/copy machinery re-uses the same control blocks without
    /// allocating new ones.
    anira::InferenceData m_current_data;

    std::atomic<bool> m_should_exit{false};
};


// ─── C API ────────────────────────────────────────────────────────────────────

extern "C" {

/**
 * Create a WasmInferenceThread directly from the Context's static inference queue.
 *
 * MUST be called from the MAIN WASM instance (the one that owns the allocator).
 * Does NOT require an InferenceHandler — only requires that a Context has been
 * initialized (i.e., at least one InferenceHandler has been created).
 *
 * @return Opaque pointer to the new WasmInferenceThread.
 */
EMSCRIPTEN_KEEPALIVE
uintptr_t wasm_inference_thread_create_from_context() {
    auto& queue = anira::Context::get_static_inference_queue();
    return reinterpret_cast<uintptr_t>(new WasmInferenceThread(queue));
}

/**
 * Execute one inference step.
 *
 * ALLOCATION-FREE — safe to call from WASM worker threads that share memory
 * with the main instance.
 *
 * @return 1 if an inference job was executed, 0 if the queue was empty.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_inference_thread_execute(uintptr_t ptr) {
    return reinterpret_cast<WasmInferenceThread*>(ptr)->execute() ? 1 : 0;
}

/**
 * Run the inference loop until stop() is called.
 *
 * ALLOCATION-FREE — safe to call from WASM worker threads that share memory
 * with the main instance.
 *
 * This is more efficient than calling execute() repeatedly from JavaScript,
 * as it keeps the tight loop in C++ and minimizes JS↔WASM boundary crossings.
 */
EMSCRIPTEN_KEEPALIVE
void wasm_inference_thread_run_loop(uintptr_t ptr) {
    reinterpret_cast<WasmInferenceThread*>(ptr)->run_loop();
}

/** Signal the thread loop to stop. */
EMSCRIPTEN_KEEPALIVE
void wasm_inference_thread_stop(uintptr_t ptr) {
    reinterpret_cast<WasmInferenceThread*>(ptr)->stop();
}

/** Reset the stop flag so the thread loop can run again. */
EMSCRIPTEN_KEEPALIVE
void wasm_inference_thread_start(uintptr_t ptr) {
    reinterpret_cast<WasmInferenceThread*>(ptr)->start();
}

/** @return 1 if the thread should exit, 0 otherwise. */
EMSCRIPTEN_KEEPALIVE
int wasm_inference_thread_should_exit(uintptr_t ptr) {
    return reinterpret_cast<WasmInferenceThread*>(ptr)->should_exit() ? 1 : 0;
}

/** @return 1 if the thread is running (i.e. not stopped), 0 otherwise. */
EMSCRIPTEN_KEEPALIVE
int wasm_inference_thread_is_running(uintptr_t ptr) {
    return reinterpret_cast<WasmInferenceThread*>(ptr)->is_running() ? 1 : 0;
}

/**
 * Destroy a WasmInferenceThread.
 *
 * Should be called from the MAIN WASM instance after the worker has been
 * stopped (so that no worker thread is still accessing the object).
 */
EMSCRIPTEN_KEEPALIVE
void wasm_inference_thread_destroy(uintptr_t ptr) {
    delete reinterpret_cast<WasmInferenceThread*>(ptr);
}

} // extern "C"
