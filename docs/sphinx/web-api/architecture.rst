Architecture
============

.. note::

   This page is hand-written. Edit it directly under
   ``docs/sphinx/web-api/architecture.rst``.

anira-web splits inference work across three browser execution
contexts:

- **Main thread** -- hosts ``AniraWeb`` and the public API surface.
- **AudioWorklet** -- runs the real-time audio callback. Wraps the
  native ``InferenceHandler`` ``process()`` loop.
- **Web Worker** -- runs the actual model inference (ONNX Runtime
  Web). Communicates with the worklet via shared memory / message
  ports.

The ``wrappers/`` directory mirrors the C++ public API one-to-one
(``InferenceConfig``, ``ProcessingSpec``, ``ModelData``, etc.) so that
documentation written against the C++ types transfers with minimal
adaptation. The ``backends/`` directory contains the JS-side inference
backends; today only ``ONNXRuntimeWebBackend`` is supported.
