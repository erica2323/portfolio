#!/bin/bash

# MACA CCCL Reduce 编译脚本 (Phase 1: Pointer + PLUS)

set -e

echo "=============================================================="
echo "  MACA CCCL Reduce Build (Phase 1)                          "
echo "=============================================================="
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
INCLUDE_DIRS="-I./parallel/include -I$MACA_PATH/include"
LIBRARY_DIRS="-L$MACA_PATH/lib -L$MACA_PATH/lib64"
LIBRARIES="-lmcruntime"

echo "=============================================================="
echo " Compiling reduce_official.cu..."
echo "=============================================================="

mxcc -xmaca \
    $INCLUDE_DIRS \
    $LIBRARY_DIRS \
    $LIBRARIES \
    -std=c++17 \
    parallel/src/reduce_official.cu \
    parallel/test/test_reduce_official.cpp \
    -o test_reduce_official

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Compilation successful!"
    echo ""
    echo "=============================================================="
    echo " Build Complete!"
    echo "=============================================================="
    echo " Executable: test_reduce_official"
    echo ""
    echo " Phase 1 Implementation:"
    echo "   • Two-phase reduction (single_tile + multi-tile)"
    echo "   • Pointer support (no iterator yet)"
    echo "   • CCCL_PLUS operation (sum)"
    echo "   • Support for int32, float32, etc."
    echo "   • JIT compilation with MCRTC"
    echo ""
    echo " Run with: ./test_reduce_official"
    echo ""
else
    echo "❌ Compilation failed!"
    exit 1
fi
