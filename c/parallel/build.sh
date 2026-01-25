#!/bin/bash
#==============================================================================
# Build script for libcccl_maca.so
# Supports both static dispatch (Phase A) and JIT compilation (Phase B)
#==============================================================================

set -e

echo "=============================================================="
echo " Building libcccl_maca.so with JIT support"
echo "=============================================================="

# Configuration - adjust these paths for your environment
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export MCCUB_PATH="${MCCUB_PATH:-/path/to/mcCub}"

# Add MACA to path
export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:$LD_LIBRARY_PATH"

# Source directory
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "MACA_PATH: $MACA_PATH"
echo "MCCUB_PATH: $MCCUB_PATH"
echo "SRC_DIR: $SRC_DIR"

# Check for required tools
if ! command -v mxcc &> /dev/null; then
    echo "ERROR: mxcc not found. Please set MACA_PATH correctly."
    exit 1
fi

# Compile the library
mxcc -xmaca \
    -fPIC -shared \
    -DCCCL_C_EXPERIMENTAL \
    -D__MACA__ \
    -std=c++17 \
    -O2 \
    -I"$SRC_DIR/include" \
    -I"$SRC_DIR/src" \
    -I"$MACA_PATH/include" \
    -I"$MCCUB_PATH" \
    -L"$MACA_PATH/lib" \
    -L"$MACA_PATH/lib64" \
    -lmcruntime \
    -lmcrtc \
    "$SRC_DIR/src/reduce.cu" \
    -o "$SRC_DIR/libcccl_maca.so"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ libcccl_maca.so built successfully"
    echo ""
    echo "Exported symbols:"
    nm -D "$SRC_DIR/libcccl_maca.so" | grep cccl_device_reduce || echo "  (no symbols found)"
    echo ""
    echo "Library location: $SRC_DIR/libcccl_maca.so"
    echo ""
    echo "To use, set:"
    echo "  export LD_LIBRARY_PATH=$SRC_DIR:\$LD_LIBRARY_PATH"
else
    echo "❌ Build failed"
    exit 1
fi
