# Phase B: Python Wrapper Layer - Build Instructions

This document explains how to build and test the Phase B Python wrapper layer for MACA CCCL.

## Overview

Phase B adds a Python layer on top of the Phase A C library, matching NVIDIA CCCL's Python API:

```
Python Application (test_reduce_phase_b.py)
    ↓
High-level API (algorithms.py)
    ↓
Cython Binding (_bindings_maca.pyx)
    ↓
C API (reduce_official.h/cu from Phase A)
    ↓
mcCub DispatchReduce
    ↓
MACA GPU
```

## File Structure

```
portfolio/
├── setup.py                              # Build script for Cython
├── test_reduce_phase_b.py                # Phase B test suite
├── cuda/
│   ├── __init__.py
│   ├── cccl/
│   │   ├── __init__.py
│   │   └── parallel/
│   │       ├── __init__.py
│   │       └── experimental/
│   │           ├── __init__.py
│   │           ├── algorithms.py         # High-level API (reduce_into)
│   │           ├── _cccl_interop.py      # Type conversion utilities
│   │           └── _bindings_maca.pyx    # Cython binding to C API
│   └── compute/
│       └── __init__.py                   # Convenience re-export
└── parallel/                             # Phase A C library (from previous work)
    ├── include/
    │   └── cccl/c/
    │       ├── reduce_official.h
    │       └── types_official.h
    ├── src/
    │   └── reduce_official.cu
    └── lib/
        └── libcccl_maca.so
```

## Prerequisites

1. **Phase A completed**: You must have `libcccl_maca.so` compiled from Phase A
2. **MACA runtime**: MACA SDK installed (default: `/opt/maca`)
3. **Python dependencies**:
   ```bash
   pip install numpy cython setuptools
   ```

## Environment Setup

Set environment variables for MACA and CCCL paths:

```bash
# MACA installation path
export MACA_PATH=/opt/maca

# CCCL C library path (from Phase A)
export CCCL_PATH=$(pwd)

# Add MACA runtime to library path
export LD_LIBRARY_PATH=$MACA_PATH/lib:$CCCL_PATH/parallel/lib:$LD_LIBRARY_PATH
```

Add to your `~/.bashrc` for persistence:

```bash
echo 'export MACA_PATH=/opt/maca' >> ~/.bashrc
echo 'export CCCL_PATH=/path/to/portfolio' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$MACA_PATH/lib:$CCCL_PATH/parallel/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Build Instructions

### Step 1: Verify Phase A Library

First, check that the Phase A C library exists:

```bash
ls -lh parallel/lib/libcccl_maca.so
```

If it doesn't exist, you need to compile it first:

```bash
cd parallel
mkdir -p build lib
mxcc -xmaca -std=c++14 -fPIC -shared \
     -I./include \
     -I$MACA_PATH/include \
     -L$MACA_PATH/lib -lmcruntime \
     -DCCCL_C_EXPERIMENTAL -D__MACA__ \
     ./src/reduce_official.cu \
     -o ./lib/libcccl_maca.so
cd ..
```

### Step 2: Build Cython Extension

Build the Python extension module in-place:

```bash
python setup.py build_ext --inplace
```

This will:
1. Compile `_bindings_maca.pyx` → `_bindings_maca.cpp`
2. Link against `libmcruntime.so` and `libcccl_maca.so`
3. Create `_bindings_maca.so` in the `cuda/cccl/parallel/experimental/` directory

Expected output:
```
Include directories: ['/opt/maca/include', '/path/to/numpy/include', '/path/to/parallel/include']
Library directories: ['/opt/maca/lib', '/path/to/parallel/lib']
running build_ext
building 'cuda.cccl.parallel.experimental._bindings_maca' extension
...
copying build/.../cuda/cccl/parallel/experimental/_bindings_maca.cpython-*.so -> cuda/cccl/parallel/experimental
```

### Step 3: Verify Build

Check that the compiled module exists:

```bash
find cuda -name "_bindings_maca*.so"
# Should output: cuda/cccl/parallel/experimental/_bindings_maca.cpython-39-x86_64-linux-gnu.so (or similar)
```

Test import:

```bash
python -c "from cuda.cccl.parallel.experimental import reduce_into; print('✓ Import successful')"
```

## Running Tests

### Phase B Test Suite

Run the Level 1 tests (basic pointer iterators):

```bash
python test_reduce_phase_b.py
```

Expected output:
```
============================================================
Phase B Test Suite: Python Wrapper Layer
============================================================
Test 1: Basic sum (int32, 1000 elements, init=10)
  Creating input array: 1000 ones (int32)
  Expected result: 1000 + 10 = 1010
  Result: 1010
  ✓ PASS

Test 2: Different operators
  add: ✓ PASS (result=100)
  min: ✓ PASS (result=0)
  max: ✓ PASS (result=99)

Test 3: Different data types
  int32: ✓ PASS (result=1000)
  int64: ✓ PASS (result=1000)
  float32: ✓ PASS (result=1000.0)
  float64: ✓ PASS (result=1000.0)

Test 4: String operator names
  'add': ✓ PASS (result=100)
  '+': ✓ PASS (result=100)
  'min': ✓ PASS (result=0)
  'max': ✓ PASS (result=99)

============================================================
Test Results Summary:
============================================================
Basic sum                      ✓ PASS
Different operators            ✓ PASS
Different dtypes               ✓ PASS
String operators               ✓ PASS
============================================================
Total: 4/4 tests passed

🎉 All Phase B tests passed!
```

### Quick Verification

Quick test to verify the API works:

```bash
python -c "
import numpy as np
from cuda.cccl.parallel.experimental import reduce_into

d_in = np.ones(1000, dtype=np.int32)
d_out = np.zeros(1, dtype=np.int32)
reduce_into(d_in, d_out, 'add', h_init=10)
print(f'Result: {d_out[0]} (expected: 1010)')
assert d_out[0] == 1010, 'Test failed!'
print('✓ API works correctly')
"
```

## Troubleshooting

### Error: "ImportError: cannot import name '_bindings_maca'"

**Cause**: Cython extension not compiled.

**Solution**:
```bash
python setup.py build_ext --inplace
```

### Error: "libmcruntime.so: cannot open shared object file"

**Cause**: MACA runtime library not in `LD_LIBRARY_PATH`.

**Solution**:
```bash
export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
```

Or find the correct path:
```bash
find /opt/maca -name "libmcruntime.so"
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
```

### Error: "libcccl_maca.so: cannot open shared object file"

**Cause**: Phase A library not compiled or not in `LD_LIBRARY_PATH`.

**Solution**:
```bash
# Check if library exists
ls parallel/lib/libcccl_maca.so

# If missing, compile Phase A first
cd parallel && [build commands] && cd ..

# Add to library path
export LD_LIBRARY_PATH=$(pwd)/parallel/lib:$LD_LIBRARY_PATH
```

### Error: "cccl_device_reduce_build failed with error X"

**Cause**: C API error during kernel compilation.

**Debug**:
1. Check that Phase A C library works standalone:
   ```bash
   ./parallel/test/test_reduce_phase_a
   ```

2. Enable verbose output in the Cython code (edit `_bindings_maca.pyx`):
   ```python
   print(f"Calling cccl_device_reduce_build with:")
   print(f"  d_in: {d_in_desc}")
   print(f"  d_out: {d_out_desc}")
   print(f"  op: {op_enum}")
   ```

3. Rebuild and retest:
   ```bash
   python setup.py build_ext --inplace --force
   python test_reduce_phase_b.py
   ```

## What Phase B Provides

✅ **Complete**:
- High-level Python API (`reduce_into`)
- Cython binding to C API
- Type conversion (numpy ↔ C types)
- Operator conversion (Python operators ↔ C enums)
- Kernel build result caching
- Support for pointer iterators
- Support for built-in types (int32, int64, float32, float64)
- Support for built-in operators (add, min, max, mul)

❌ **Not Yet Implemented** (Future Phases):
- NVRTC JIT for Python lambda functions (Phase C)
- Advanced iterators (Counting, Constant, Transform) (Phase D)
- Complex types, struct types (Phase E)
- Stream support (Phase E)
- cuNumeric/CuPy integration (Phase F)

## Next Steps

### Level 2 Tests (Phase C - NVRTC)

To run the full NVIDIA `test_reduce.py` test suite, you need Phase C which adds:
- NVRTC/MCRTC JIT compilation for Python operators
- Runtime code generation for custom lambda functions

### Level 3 Tests (Phase D - Iterators)

For advanced iterator tests:
- Counting iterator (0, 1, 2, 3, ...)
- Constant iterator (same value repeated)
- Transform iterator (apply function to input)
- CacheModified iterator (with caching hints)

### Level 4 Tests (Phase E - Advanced Types)

For complex type tests:
- Complex numbers
- Struct types (custom C++ types)
- Custom operators

## API Usage Examples

### Basic Reduction

```python
import numpy as np
from cuda.cccl.parallel.experimental import reduce_into

# Create input array
d_in = np.ones(1000, dtype=np.int32)
d_out = np.zeros(1, dtype=np.int32)

# Reduce with sum
reduce_into(d_in, d_out, 'add')
print(d_out[0])  # Output: 1000
```

### With Initial Value

```python
reduce_into(d_in, d_out, 'add', h_init=10)
print(d_out[0])  # Output: 1010
```

### Using Python Operators

```python
import operator

reduce_into(d_in, d_out, operator.add, h_init=10)
reduce_into(d_in, d_out, min)
reduce_into(d_in, d_out, max)
```

### Different Data Types

```python
# Float32
d_in_f32 = np.ones(1000, dtype=np.float32)
d_out_f32 = np.zeros(1, dtype=np.float32)
reduce_into(d_in_f32, d_out_f32, operator.add)

# Int64
d_in_i64 = np.arange(1000, dtype=np.int64)
d_out_i64 = np.zeros(1, dtype=np.int64)
reduce_into(d_in_i64, d_out_i64, max)
```

## Success Criteria

Phase B is complete when:
1. ✅ `python setup.py build_ext --inplace` succeeds
2. ✅ `from cuda.cccl.parallel.experimental import reduce_into` works
3. ✅ All 4 tests in `test_reduce_phase_b.py` pass
4. ✅ Can run basic reductions with different types and operators

You're ready for Phase C when you want to:
- Use Python lambda functions as operators
- Run NVIDIA's full `test_reduce.py` test suite
- Support custom user-defined operations
