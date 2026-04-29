anira-web (TypeScript / WASM)
=============================

The ``anira-web`` package is a TypeScript wrapper around the Anira C++
library compiled to WebAssembly via Emscripten. It exposes the same
real-time inference primitives as the native build, adapted to the
browser's audio worklet and worker model.

This section is a mix of hand-written guides and an auto-generated API
reference. The reference pages are produced from TypeDoc output via
``sphinx-js``, mirroring the way the native C++ reference is produced
from Doxygen via Breathe -- so signatures, parameter lists, and
admonitions render in the same shibuya-themed style as the C++ pages.

.. toctree::
   :maxdepth: 2
   :caption: anira-web

   getting_started
   architecture
   reference/index
