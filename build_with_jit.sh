#!/bin/bash

set -e

echo "=============================================================="
echo " Building libcccl_maca.so with JIT support "
echo "=============================================================="

export MACA_PATH="/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707"
export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:$LD_LIBRARY_PATH"

# Source files
SOURCES="parallel/src/reduce_official.cu parallel/src/jit_compiler.cu"

# Compiler flags
CFLAGS="-xmaca \
  -fPIC -shared \
  -DCCCL_C_EXPERIMENTAL \
  -D__MACA__ \
  -std=c++17 \
  -O2 \
  -I./parallel/include \
  -I$MACA_PATH/include \
  -I/mnt/data/minxi/1_15/mcCub"

# Linker flags
LDFLAGS="-L$MACA_PATH/lib \
  -L$MACA_PATH/lib64 \
  -lmcruntime \
  -lcrypto"  # For SHA256 in kernel cache

echo ""
echo "Compiling sources:"
echo "  - parallel/src/reduce_official.cu"
echo "  - parallel/src/jit_compiler.cu"
echo ""

mxcc $CFLAGS $LDFLAGS $SOURCES -o libcccl_maca.so

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ libcccl_maca.so built successfully (with JIT support)"
    echo ""
    echo "Exported symbols:"
    nm -D libcccl_maca.so | grep -E "(cccl_device_reduce|cccl_jit)" | head -20
    echo ""
    echo "📝 Next steps:"
    echo "1. Merge JIT extension into _bindings_maca.pyx"
    echo "2. Update _reduce_simplified.py with JIT API"
    echo "3. Rebuild Python extension: python setup_maca.py build_ext --inplace"
    echo "4. Run tests: python test_jit_reduce.py"
else
    echo "❌ Build failed"
    exit 1
fi
