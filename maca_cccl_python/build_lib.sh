#!/bin/bash

# Build shared library for MACA CCCL Transform API
# This creates libmaca_cccl_transform.so for Python ctypes binding

set -e

# MACA SDK path - UPDATE THIS to match your system
export MACA_PATH="/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707"
export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$LD_LIBRARY_PATH"

# Source directory - UPDATE THIS to match your file locations
# Assumes your files are in: project_2026_01_12/mcCub/c/parallel/
SOURCE_DIR="../project_2026_01_12/mcCub/c"

# Compiler and flags
COMPILER="mxcc"
INCLUDE_DIRS="-I${SOURCE_DIR}/parallel/include -I${MACA_PATH}/include"
LIBRARY_DIRS="-L${MACA_PATH}/lib"
LIBRARIES="-lmcruntime"

# Output library
OUTPUT="libmaca_cccl_transform.so"

echo "Building shared library: ${OUTPUT}"
echo "Source: ${SOURCE_DIR}/parallel/src/transform_official.cu"

# Compile to shared library
${COMPILER} -xmaca \
    -shared \
    -fPIC \
    ${INCLUDE_DIRS} \
    ${LIBRARY_DIRS} \
    ${LIBRARIES} \
    -std=c++17 \
    ${SOURCE_DIR}/parallel/src/transform_official.cu \
    -o ${OUTPUT}

if [ $? -eq 0 ]; then
    echo "✅ Build successful: ${OUTPUT}"
    ls -lh ${OUTPUT}
else
    echo "❌ Build failed"
    exit 1
fi
