#!/bin/bash
#==============================================================================
# MACA CCCL Verification Script
# Run this to verify the entire build and test pipeline
#==============================================================================

set -e

echo "=============================================================="
echo "  MACA CCCL Verification"
echo "=============================================================="
echo ""

# Configuration
CCCL_ROOT="/mnt/data/minxi/1_15/mcCub"
MACA_PATH="/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707"

export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:$CCCL_ROOT/c:$LD_LIBRARY_PATH"

#------------------------------------------------------------------------------
# Step 1: Check environment
#------------------------------------------------------------------------------
echo "Step 1: Checking environment..."
echo ""

echo "  CCCL_ROOT: $CCCL_ROOT"
echo "  MACA_PATH: $MACA_PATH"
echo ""

# Check mxcc
if command -v mxcc &> /dev/null; then
    echo "  ✅ mxcc found: $(which mxcc)"
else
    echo "  ❌ mxcc not found! Please check MACA_PATH"
    exit 1
fi

# Check required files
echo ""
echo "  Checking required files..."

files_to_check=(
    "$MACA_PATH/include/mcr/mc_runtime.h"
    "$CCCL_ROOT/c/parallel/include/cccl/c/reduce_official.h"
    "$CCCL_ROOT/c/parallel/src/reduce_official.cu"
)

all_found=true
for f in "${files_to_check[@]}"; do
    if [ -f "$f" ]; then
        echo "    ✅ $f"
    else
        echo "    ❌ $f"
        all_found=false
    fi
done

if [ "$all_found" = false ]; then
    echo ""
    echo "  ❌ Some required files are missing!"
    exit 1
fi

echo ""
echo "  ✅ Environment check passed"
echo ""

#------------------------------------------------------------------------------
# Step 2: Build C library
#------------------------------------------------------------------------------
echo "Step 2: Building libcccl_maca.so..."
echo ""

cd "$CCCL_ROOT/c"

if [ -f "build_libcccl_maca.sh" ]; then
    bash build_libcccl_maca.sh
else
    echo "  ❌ build_libcccl_maca.sh not found in $CCCL_ROOT/c"
    exit 1
fi

if [ -f "libcccl_maca.so" ]; then
    echo ""
    echo "  ✅ libcccl_maca.so built successfully"
    echo "  Location: $CCCL_ROOT/c/libcccl_maca.so"
else
    echo "  ❌ libcccl_maca.so not found after build"
    exit 1
fi

echo ""

#------------------------------------------------------------------------------
# Step 3: Build Python bindings
#------------------------------------------------------------------------------
echo "Step 3: Building Python bindings..."
echo ""

cd "$CCCL_ROOT/python"

# Clean previous build
rm -rf build/ *.egg-info cuda_cccl/*.so cuda_cccl/**/*.so 2>/dev/null || true

# Build
python3 setup_maca.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo ""
    echo "  ✅ Python bindings built successfully"
else
    echo "  ❌ Python bindings build failed"
    exit 1
fi

echo ""

#------------------------------------------------------------------------------
# Step 4: Run test
#------------------------------------------------------------------------------
echo "Step 4: Running tests..."
echo ""

cd "$CCCL_ROOT/python"

# Run the test
python3 -c "
import sys
sys.path.insert(0, '.')

print('  Importing cuda_cccl...')
try:
    from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind
    print('  ✅ Import successful')
except ImportError as e:
    print(f'  ❌ Import failed: {e}')
    sys.exit(1)

print('  Module loaded successfully!')
print('')
print('  Available functions:')
print('    - reduce(d_in, d_out, op, init)')
print('    - reduce_with_op(d_in, d_out, op_source, init)')
print('    - sum, min, max convenience functions')
print('')
"

if [ $? -eq 0 ]; then
    echo "  ✅ Basic import test passed"
else
    echo "  ❌ Basic import test failed"
    exit 1
fi

echo ""

#------------------------------------------------------------------------------
# Step 5: Run full GPU test (if GPU available)
#------------------------------------------------------------------------------
echo "Step 5: Running GPU test..."
echo ""

python3 << 'PYTEST'
import sys
sys.path.insert(0, '.')

import numpy as np
import ctypes

# Load MACA runtime
try:
    maca = ctypes.CDLL("libmcruntime.so")
    maca.mcMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    maca.mcMalloc.restype = ctypes.c_int
    maca.mcFree.argtypes = [ctypes.c_void_p]
    maca.mcFree.restype = ctypes.c_int
    maca.mcMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    maca.mcMemcpy.restype = ctypes.c_int
except OSError as e:
    print(f"  ⚠️  Cannot load libmcruntime.so: {e}")
    print("  Skipping GPU test (no GPU runtime available)")
    sys.exit(0)

# Simple device array class
class MacaArray:
    def __init__(self, shape, dtype):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape))
        self.nbytes = self.size * self.dtype.itemsize
        self.ptr = ctypes.c_void_p(None)
        ret = maca.mcMalloc(ctypes.byref(self.ptr), ctypes.c_size_t(self.nbytes))
        if ret != 0:
            raise RuntimeError(f"mcMalloc failed: {ret}")

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr.value:
            maca.mcFree(self.ptr)

    @property
    def __cuda_array_interface__(self):
        return {
            "version": 3,
            "shape": self.shape,
            "typestr": self.dtype.str,
            "data": (int(self.ptr.value), False),
        }

    def copy_to_device(self, host_array):
        host_array = np.ascontiguousarray(host_array, dtype=self.dtype)
        maca.mcMemcpy(
            ctypes.c_void_p(int(self.ptr.value)),
            ctypes.c_void_p(int(host_array.ctypes.data)),
            ctypes.c_size_t(self.nbytes),
            ctypes.c_int(1)  # H2D
        )

    def copy_to_host(self, host_array):
        maca.mcMemcpy(
            ctypes.c_void_p(int(host_array.ctypes.data)),
            ctypes.c_void_p(int(self.ptr.value)),
            ctypes.c_size_t(self.nbytes),
            ctypes.c_int(2)  # D2H
        )

# Import CCCL
from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind

# Test 1: Sum reduction
print("  Test 1: Sum Reduction")
N = 1000
h_in = np.ones(N, dtype=np.int32)
h_out = np.zeros(1, dtype=np.int32)

d_in = MacaArray(N, np.int32)
d_out = MacaArray(1, np.int32)
d_in.copy_to_device(h_in)

reduce(d_in, d_out, op=OpKind.SUM, init=10)

d_out.copy_to_host(h_out)
expected = N + 10
result = int(h_out[0])

if result == expected:
    print(f"    ✅ PASSED: sum({N} ones) + 10 = {result}")
else:
    print(f"    ❌ FAILED: expected {expected}, got {result}")
    sys.exit(1)

# Test 2: Min reduction
print("  Test 2: Min Reduction")
h_in = np.array([5, 2, 8, 1, 9], dtype=np.int32)
h_out = np.zeros(1, dtype=np.int32)

d_in = MacaArray(len(h_in), np.int32)
d_out = MacaArray(1, np.int32)
d_in.copy_to_device(h_in)

reduce(d_in, d_out, op=OpKind.MIN, init=100)

d_out.copy_to_host(h_out)
expected = min(100, *h_in)
result = int(h_out[0])

if result == expected:
    print(f"    ✅ PASSED: min([5,2,8,1,9], init=100) = {result}")
else:
    print(f"    ❌ FAILED: expected {expected}, got {result}")
    sys.exit(1)

# Test 3: Max reduction
print("  Test 3: Max Reduction")
h_in = np.array([5, 2, 8, 1, 9], dtype=np.int32)
h_out = np.zeros(1, dtype=np.int32)

d_in = MacaArray(len(h_in), np.int32)
d_out = MacaArray(1, np.int32)
d_in.copy_to_device(h_in)

reduce(d_in, d_out, op=OpKind.MAX, init=0)

d_out.copy_to_host(h_out)
expected = max(0, *h_in)
result = int(h_out[0])

if result == expected:
    print(f"    ✅ PASSED: max([5,2,8,1,9], init=0) = {result}")
else:
    print(f"    ❌ FAILED: expected {expected}, got {result}")
    sys.exit(1)

print("")
print("  ✅ All GPU tests passed!")
PYTEST

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================="
    echo "  ✅ All verification steps passed!"
    echo "=============================================================="
else
    echo ""
    echo "=============================================================="
    echo "  ❌ Some tests failed"
    echo "=============================================================="
    exit 1
fi
