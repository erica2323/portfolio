#!/bin/bash

# MACA CCCL Reduce Build Script - Phase 5 (True CCCL Style)

set -e

echo "=============================================================="
echo "  MACA CCCL Reduce Build (Phase 5: JIT + mcCub)            "
echo "=============================================================="
echo ""
echo "Architecture: JIT compile kernels that #include mcCub"
echo "This matches NVIDIA CCCL's design!"
echo ""

# 设置 MACA 路径
export MACA_PATH="/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707"
export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:$LD_LIBRARY_PATH"

# 检查 mxcc
if ! command -v mxcc &> /dev/null; then
    echo "❌ Error: mxcc not found"
    echo "   MACA_PATH: $MACA_PATH"
    echo "   PATH: $PATH"
    exit 1
fi

echo "✅ Using MACA at: $MACA_PATH"
echo "✅ mxcc found at: $(which mxcc)"
echo ""

# 编译选项
INCLUDE_DIRS="-I./parallel/include -I$MACA_PATH/include -I/mnt/data/minxi/1_15/mcCub"
LIBRARY_DIRS="-L$MACA_PATH/lib -L$MACA_PATH/lib64"
LIBRARIES="-lmcruntime"

echo "=============================================================="
echo " Compiling CCCL reduce (JIT + mcCub architecture)..."
echo "=============================================================="

mxcc -xmaca \
    $INCLUDE_DIRS \
    $LIBRARY_DIRS \
    $LIBRARIES \
    -std=c++17 \
    parallel/src/reduce_official.cu \
    parallel/test/test_reduce_cccl_style.cpp \
    -o test_reduce_cccl_style

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Compilation successful!"
    echo ""
    echo "=============================================================="
    echo " Build Complete!"
    echo "=============================================================="
    echo " Executable: test_reduce_cccl_style"
    echo ""
    echo " Phase 5 Implementation (True CCCL Style):"
    echo "   • JIT compilation at runtime"
    echo "   • Generated kernels #include <mccub/...>"
    echo "   • Wrapper kernel calls mcCub::DeviceReduce"
    echo "   • Two-phase execution (query + execute)"
    echo "   • CCCL_PLUS, CCCL_MINIMUM, CCCL_MAXIMUM"
    echo "   • Matches NVIDIA CCCL architecture!"
    echo ""
    echo " How it works:"
    echo "   1. Build phase: JIT compile kernel with mcCub headers"
    echo "   2. Execute phase: Launch wrapper that calls mcCub"
    echo "   3. mcCub handles the actual GPU reduction"
    echo ""
    echo " Run with: ./test_reduce_cccl_style"
    echo ""
else
    echo "❌ Compilation failed!"
    exit 1
fi
