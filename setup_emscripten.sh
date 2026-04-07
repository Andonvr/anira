#!/bin/bash

# Exit immediately if any command fails
set -e

# Check if the modules directory exists
if [ -d "modules/emsdk" ]; then
  echo "Directory 'modules/emsdk' already exists. Skipping cloning."
  exit 0
fi

# Clone the onnxruntime repository recursively
echo "Cloning emscripten repository..."
git clone --recurse-submodules https://github.com/emscripten-core/emsdk.git modules/emsdk

# Checkout the specified tag
cd modules/emsdk
echo "Checking out tag 4.0.23..."
git checkout tags/4.0.23

# Install emscripten
./emsdk install 4.0.23
./emsdk activate 4.0.23
cd upstream/emscripten
npm install