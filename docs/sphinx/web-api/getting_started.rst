Getting Started with anira-web
==============================

.. note::

   This page is hand-written. Edit it directly under
   ``docs/sphinx/web-api/getting_started.rst``.

Installation
------------

.. code-block:: bash

   npm install anira-web

The package ships its WebAssembly artifacts under ``anira-web/wasm/``.
When bundling, make sure the ``.wasm`` files are copied to a path your
runtime can fetch.

First inference
---------------

A minimal end-to-end example is intentionally left as a TODO until the
public API stabilises. See the :doc:`reference/index` for the currently
exported symbols and the :doc:`architecture` page for how they fit
together.
