#!/bin/bash

set -e

echo "=============================================================="
echo "  Building libcccl_maca.so for Python binding                 "
echo "=============================================================="

export MACA_PATH="/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707"
export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:$LD_LIBRARY_PATH"

mxcc -xmaca \
    -fPIC -shared \
    -DCCCL_C_EXPERIMENTAL \
    -D__MACA__ \
    -std=c++17 \
    -O2 \
    -I./parallel/include \
    -I$MACA_PATH/include \
    -I/mnt/data/minxi/1_15/mcCub \
    -L$MACA_PATH/lib \
    -L$MACA_PATH/lib64 \
    -lmcruntime \
    parallel/src/reduce_official.cu \
    -o libcccl_maca.so

if [ $? -eq 0 ]; then
    echo "✅ libcccl_maca.so built successfully"
    echo ""
    echo "Exported symbols:"
    nm -D libcccl_maca.so | grep cccl_device_reduce
else
    echo "❌ Build failed"
    exit 1
fi
