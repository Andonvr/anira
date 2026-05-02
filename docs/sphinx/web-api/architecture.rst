Architecture
============

``anira-web`` is the WebAssembly distribution of anira: the same C++
library, compiled to WASM and wrapped in a
TypeScript API. The TypeScript layer's job is to spread that WASM
module across the browser threads anira needs in order to run
real-time inference — a main thread, the audio worklet thread, and one
or more inference worker threads.

.. code-block:: text

   ┌──────────────────┐    ┌────────────────────┐    ┌──────────────────────┐
   │   Main thread    │    │ Inference worker(s)│    │ Audio worklet thread │
   │                  │    │   (Web Worker(s))  │    │                      │
   │  Setup           │    │                    │    │  AudioWorklet-       │
   │                  │    │  Model inference   │    │   Processor          │
   │  Configuration   │◀──▶│  (WASM ONNX,       │◀──▶│  Real-time           │
   │                  │    │   onnxruntime-web, │    │   process()          │
   │  UI control      │    │   or custom)       │    │  Pre/Post-processor  │
   └──────────────────┘    └────────────────────┘    └──────────────────────┘
            ▲                       ▲                          ▲
            └───────────────────────┴──────────────────────────┘
                       Shared WebAssembly memory

The number of inference workers is up to you. Each call to
``aniraWeb.spinUpInferenceWorker()`` spawns a new Web Worker hosting an
``InferenceThread`` — the same primitive anira uses for its desktop
thread pool. One worker is enough for simple models on most machines; spawn more if
you see audio dropouts, so anira can run inference on multiple batches
in parallel.

All threads share a single ``WebAssembly.Memory`` instance, so
configuration objects, ring buffers, and tensor data live at the same
heap addresses everywhere. Cross-thread coordination uses message
passing for setup and atomics on shared memory for the real-time path.

Main Thread
-----------

The main thread is where you set up anira. Calling
``await AniraWeb.create()`` instantiates the WASM module and returns
the ``aniraWeb`` factory; from there you wire up the model, inference
configuration, and pre/post-processing the same way you would in C++.

The main thread also owns your UI. Non-streamable tensor values
written from here — through ``setInput`` and similar APIs — reach the
model without blocking the audio path, so a slider or toggle can
update the model from frame to frame.

Inference Worker
----------------

``await aniraWeb.spinUpInferenceWorker()`` starts a Web Worker that
owns inference execution. Pulling inference off the audio thread is
what keeps the audio worklet's ``process`` callback real-time-safe
even when a forward pass takes longer than one audio block.

The worker hosts the inference engine itself, regardless of where that
engine actually runs. ``anira-web`` ships with two built-in engines:
ONNX Runtime compiled into the WASM module, and ``onnxruntime-web`` on
the JavaScript side (:js:class:`ONNXRuntimeWebBackend`). User-written JS backends
also run on this worker. See :doc:`custom_inference_backends`.

You can also replace the worker entry point itself:

.. code-block:: typescript

   await aniraWeb.spinUpInferenceWorker(
     new URL('./customInferenceWorker.ts', import.meta.url)
   )

Audio Worklet Thread
--------------------

The browser's ``AudioWorkletGlobalScope`` runs the audio callback. Anira
ships with a default worklet that handles the common case: a
single-tensor model with in-place stereo or mono I/O. To install it:

.. code-block:: typescript

   await aniraWeb.registerAudioWorkletForContext(audioContext)
   const node = await aniraWeb.configureAudioWorklet(
     audioContext,
     inferenceHandler,
     ppProcessor
   )

For models that need more — multi-tensor I/O, a custom processing
buffer size, ``AudioParam`` integration, or a JS pre/post processor —
you provide a custom worklet file. See :doc:`custom_audio_worklets`.

.. note::
   :js:class:`JSPrePostProcessor` subclasses are constructed on the
   audio worklet thread, not on the main thread. Pre- and
   post-processing run in the real-time callback, so the JS object that
   implements them must live where that callback runs.

Three Customization Axes
------------------------

Most extension work falls into one of three independent categories,
each with its own page:

1. :doc:`custom_audio_worklets` — extend
   :js:class:`AniraAudioWorkletBase` for multi-tensor models
   (``processMulti``), custom ``maxBufferSize``, ``AudioParam``
   integration, or to host a custom :js:class:`JSPrePostProcessor`.
2. :doc:`custom_pre_post_processing` — subclass
   :js:class:`JSPrePostProcessor` to run JavaScript before and after
   inference (windowing, normalization, parameter clamping, etc.).
3. :doc:`custom_inference_backends` — replace the WASM-side runtime with
   a JavaScript backend. Built-in options
   (:js:class:`JSBackendBase`, :js:class:`ONNXRuntimeWebBackend`) and
   user-written backends both run on the inference worker.

Custom pre/post processing **requires** a custom worklet (because the
subclass must be instantiated on the audio thread); custom worklets and
custom backends are otherwise independent and can be combined freely.
