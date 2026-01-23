# CCCL for MACA - Usage Guide

## Overview

This is a Python binding for NVIDIA CCCL (CUDA C++ Core Libraries) adapted for MACA GPUs. It provides high-performance parallel algorithms like reduce, scan, sort, etc.

**Current Status**: Phase A - Reduce operations with builtin operators

## Installation

### 1. Build the C++ library

```bash
cd /path/to/mcCub
./build_libcccl.sh
```

This creates `libcccl_maca.so` which implements the CCCL C API.

### 2. Build the Python bindings

```bash
python setup_maca.py build_ext --inplace
```

This compiles the Cython extension `_bindings_maca.so`.

### 3. Set library path

```bash
export LD_LIBRARY_PATH=/opt/maca-20250707/lib64:/path/to/mcCub:$LD_LIBRARY_PATH
```

## Quick Start

```python
import numpy as np
from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind, MacaArray

# Create device arrays
d_in = MacaArray(1000, dtype=np.int32)
d_out = MacaArray(1, dtype=np.int32)

# Copy data to device
h_in = np.ones(1000, dtype=np.int32)
d_in.copy_to_device(h_in)

# Run reduction
reduce(d_in, d_out, op=OpKind.SUM, init=0)

# Copy result back
h_out = np.zeros(1, dtype=np.int32)
d_out.copy_to_host(h_out)

print(f"Sum: {h_out[0]}")  # Should be 1000
```

## API Reference

### Reduce Operations

```python
reduce(d_in, d_out, op=OpKind.SUM, init=0, stream=None, num_items=None)
```

**Parameters**:
- `d_in`: Input device array (MacaArray or any object with `__cuda_array_interface__`)
- `d_out`: Output device array (single element)
- `op`: Operation kind (OpKind.SUM, OpKind.MIN, OpKind.MAX)
- `init`: Initial value for reduction
- `stream`: MACA stream handle (optional)
- `num_items`: Number of elements (auto-detected from array if not specified)

**Convenience functions**:
```python
sum(d_in, d_out, init=0, stream=None)
min(d_in, d_out, init=0, stream=None)
max(d_in, d_out, init=0, stream=None)
```

### MACA Runtime Wrapper

The `_maca_runtime` module provides safe Python wrappers for MACA runtime functions.

**CRITICAL**: The ctypes function signatures are declared correctly to prevent pointer truncation bugs.

#### MacaArray Class

```python
class MacaArray:
    """Device array with __cuda_array_interface__ support"""

    def __init__(self, shape, dtype)
    def copy_to_device(self, host_array: np.ndarray)
    def copy_to_host(self, host_array: np.ndarray)
```

#### Low-Level Functions

```python
# Memory allocation
ptr = malloc(nbytes)              # Returns device pointer as int
free(ptr)                         # Free device memory

# Memory transfers
memcpy_h2d(dev_ptr, host_array)   # Host → Device
memcpy_d2h(host_array, dev_ptr)   # Device → Host

# Convenience constructors
d_zeros = zeros(shape, dtype)     # Create zero-initialized array
d_ones = ones(shape, dtype)       # Create ones array
d_arr = from_numpy(host_array)    # Copy from NumPy
h_arr = to_numpy(d_arr)           # Copy to NumPy
```

## Supported Types

- `np.int32`, `np.int64`
- `np.uint8`, `np.uint16`, `np.uint32`, `np.uint64`
- `np.float32`, `np.float64`

## Supported Operations (Phase A)

- **SUM** (`OpKind.SUM`): Addition reduction
- **MIN** (`OpKind.MIN`): Minimum reduction
- **MAX** (`OpKind.MAX`): Maximum reduction

## Architecture

```
User Code (test_reduce_simplified.py)
    ↓
Python API (_reduce_simplified.py)
    ↓ uses
Runtime Wrapper (_maca_runtime.py)  ← CRITICAL: Correct ctypes signatures
    ↓
Cython Bindings (_bindings_maca.pyx)
    ↓
C API (reduce_official.h/cu → libcccl_maca.so)
    ↓
mcCub (DispatchReduce)
    ↓
MACA Runtime (libmcruntime.so)
```

## Important Notes

### Why `_maca_runtime.py` is Critical

**Without explicit ctypes argtypes declarations**, Python will guess function signatures incorrectly:

```python
# ❌ WRONG - ctypes guesses signature, truncates pointers on 64-bit
maca = ctypes.CDLL("libmcruntime.so")
maca.mcMalloc(...)  # SEGFAULT!

# ✅ CORRECT - explicit signature prevents truncation
maca.mcMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
maca.mcMalloc.restype = ctypes.c_int
```

**Always use the provided `_maca_runtime` module** instead of calling ctypes directly.

### Memory Management

- `MacaArray` automatically frees device memory in `__del__`
- For manual control, use `malloc()` / `free()`
- Keep Python objects alive during async GPU operations

### Stream Synchronization

```python
from cuda_cccl.cuda.cccl.parallel.experimental._maca_runtime import synchronize_stream

# Synchronize default stream
synchronize_stream(0)

# Synchronize custom stream
synchronize_stream(stream_handle)
```

## Testing

```bash
# Run tests
python test_reduce_simplified.py

# Expected output:
# === Test 1: Sum Reduction ===
# Result: 1010
# Expected: 1010
# ✅ PASSED
#
# === Test 2: Min Reduction ===
# Result: 0
# Expected: 0
# ✅ PASSED
#
# === Test 3: Max Reduction ===
# Result: 100
# Expected: 100
# ✅ PASSED
```

## Troubleshooting

### "Cannot load libmcruntime.so"

```bash
export LD_LIBRARY_PATH=/opt/maca-20250707/lib64:$LD_LIBRARY_PATH
```

### "Cannot load libcccl_maca.so"

Make sure you built it and added to library path:
```bash
cd /path/to/mcCub
./build_libcccl.sh
export LD_LIBRARY_PATH=/path/to/mcCub:$LD_LIBRARY_PATH
```

### Segmentation fault

This usually means:
1. ctypes signatures are wrong (should never happen with `_maca_runtime`)
2. Passing wrong pointer/size to C functions
3. Using freed memory

### Import errors

```bash
# Rebuild Cython extension
python setup_maca.py build_ext --inplace

# Make sure package structure exists
touch cuda_cccl/__init__.py
touch cuda_cccl/cuda/__init__.py
touch cuda_cccl/cuda/cccl/__init__.py
touch cuda_cccl/cuda/cccl/parallel/__init__.py
```

## Future Phases

- **Phase B**: Custom operators with LTO-IR
- **Phase C**: Custom iterators
- **Phase D**: More algorithms (scan, sort, etc.)
- **Phase E**: Multi-GPU support

## Contributing

When adding new runtime functions:

1. **ALWAYS** declare ctypes signatures in `_maca_runtime.py`
2. Add unit tests
3. Update this documentation

Example:
```python
# In _maca_runtime.py
_maca_runtime.mcNewFunction.argtypes = [ctypes.c_void_p, ctypes.c_int]
_maca_runtime.mcNewFunction.restype = ctypes.c_int
```
