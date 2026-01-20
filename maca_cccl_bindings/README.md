# MACA CCCL Python Bindings

GPU-accelerated parallel algorithms for MACA (Moore Threads) adapted from NVIDIA CUDA CCCL.

## 🎯 Features

- **Device Reduce**: Sum, Min, Max reductions with build/execute pattern
- **Type System**: Support for int8-int64, uint8-uint64, float32, float64
- **High Performance**: Cython-based bindings to C/mcCub layer
- **JIT Support**: Runtime kernel generation (mcRTC integration ready)
- **Compatible**: Works with NumPy arrays and device pointers

## 📋 Architecture

```
┌─────────────────────────────────────────┐
│         Python User API                  │
│  (DeviceReduce.sum/min/max)             │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Python Algorithm Layer              │
│         (_reduce.py)                     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Cython Bindings Layer               │
│     (_bindings_impl.pyx)                 │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         C API Layer                      │
│   (reduce_official.h/.cu)                │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      mcCub (MACA CUB Port)               │
│   DispatchReduce::Dispatch()             │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         MACA GPU Runtime                 │
│      (Moore Threads Hardware)            │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- MACA SDK (Moore Threads)
- mcCub library
- Python 3.7+
- NumPy
- Cython

### Installation

```bash
# Set environment variables
export MACA_PATH="/path/to/maca-sdk"
export MCCUB_PATH="/path/to/mcCub"
export CCCL_C_INCLUDE="./parallel/include"

# Build and install
chmod +x build_python.sh
./build_python.sh
```

### Basic Usage

```python
import numpy as np
from maca_cccl import DeviceReduce

# Create data
data = np.arange(10000, dtype=np.int32)

# Allocate device memory (use your MACA runtime API)
# d_data = maca.mem_copy_host_to_device(data)
# d_out = maca.mem_alloc(4)

# Build reduce operation (once)
reduce_sum = DeviceReduce.sum(np.int32)

# Execute (can reuse multiple times)
reduce_sum.compute(d_data, d_out, len(data))

# Get result
# result = maca.mem_copy_device_to_host(d_out)
```

## 📚 API Reference

### DeviceReduce

GPU-accelerated reduction operations.

#### Methods

**`DeviceReduce.sum(dtype, stream=None)`**
- Build sum reduction operation
- **Parameters:**
  - `dtype`: numpy dtype (int32, int64, float32, float64, etc.)
  - `stream`: MACA stream (optional)
- **Returns:** `DeviceReduceBuildResult` object

**`DeviceReduce.min(dtype, stream=None)`**
- Build minimum reduction operation
- Uses appropriate initial value (type.max)

**`DeviceReduce.max(dtype, stream=None)`**
- Build maximum reduction operation
- Uses appropriate initial value (type.min)

**`DeviceReduce.reduce(d_in, d_out, num_items, op_kind, dtype, initial_value=None, stream=None)`**
- One-shot reduction (build + execute)
- **Parameters:**
  - `d_in`: device pointer (int) or array with `__cuda_array_interface__`
  - `d_out`: device pointer for single result value
  - `num_items`: number of elements
  - `op_kind`: `OpKind.PLUS`, `OpKind.MINIMUM`, or `OpKind.MAXIMUM`
  - `dtype`: numpy dtype
  - `initial_value`: scalar (optional, auto-detected)
  - `stream`: MACA stream (optional)

### DeviceReduceBuildResult

Compiled reduction operation.

**`compute(d_in, d_out, num_items, stream=None)`**
- Execute the compiled reduction
- **Parameters:**
  - `d_in`: device input pointer
  - `d_out`: device output pointer
  - `num_items`: number of elements
  - `stream`: MACA stream (optional)

**Properties:**
- `dtype`: data type
- `op_kind`: operation kind

## 🔧 Advanced: JIT/mcRTC Support

For custom operations with runtime compilation:

```python
from maca_cccl._jit_templates import JITKernelTemplate

# Generate custom kernel source
template = JITKernelTemplate()
kernel_src = template.generate_reduce_kernel(
    dtype='float',
    op_name='Custom',
    custom_op_code='return a * b;'  # Product reduction
)

# Compile with mcRTC (requires implementation)
# compiler = mcRTCCompiler()
# module = compiler.compile(kernel_src, include_paths=[...])
# kernel = compiler.get_function(module, "reduce_kernel")
```

### Implementing mcRTC Support

To enable JIT compilation, implement in `_jit_templates.py`:

1. **Create mcRTC program**
   ```cpp
   mcrtcProgram prog;
   mcrtcCreateProgram(&prog, source, "kernel.cu", ...);
   ```

2. **Compile with options**
   ```cpp
   const char* opts[] = {"-std=c++17", "-I/path/to/mccub"};
   mcrtcCompileProgram(prog, 2, opts);
   ```

3. **Get LTOIR**
   ```cpp
   size_t ltoir_size;
   mcrtcGetLTOIRSize(prog, &ltoir_size);
   char* ltoir = new char[ltoir_size];
   mcrtcGetLTOIR(prog, ltoir);
   ```

4. **Load module**
   ```cpp
   mcModule_t module;
   mcModuleLoadData(&module, ltoir);
   mcModuleGetFunction(&kernel, module, "reduce_kernel");
   ```

See NVIDIA NVRTC documentation for reference:
https://docs.nvidia.com/cuda/nvrtc/index.html

## 📁 File Structure

```
maca_cccl_bindings/
├── __init__.py              # Package initialization
├── _bindings_impl.pyx       # Cython → C bindings
├── _reduce.py               # High-level reduce API
├── _jit_templates.py        # JIT kernel generation
├── typing.py                # Type system
├── setup.py                 # Build configuration
├── build_python.sh          # Build script
├── example_reduce.py        # Usage examples
└── README.md                # This file
```

## 🎓 Examples

### Example 1: Sum Reduction

```python
import numpy as np
from maca_cccl import DeviceReduce

# Host data
h_data = np.arange(10000, dtype=np.int32)

# Build once (expensive)
reduce_sum = DeviceReduce.sum(np.int32)

# Execute many times (fast)
for batch in data_batches:
    reduce_sum.compute(d_batch, d_out, len(batch))
```

### Example 2: Min/Max

```python
# Find min and max in parallel
reduce_min = DeviceReduce.min(np.float32)
reduce_max = DeviceReduce.max(np.float32)

reduce_min.compute(d_data, d_min, size)
reduce_max.compute(d_data, d_max, size)
```

### Example 3: Multiple Types

```python
# Support for various types
ops = {
    np.int32: DeviceReduce.sum(np.int32),
    np.int64: DeviceReduce.sum(np.int64),
    np.float32: DeviceReduce.sum(np.float32),
    np.float64: DeviceReduce.sum(np.float64),
}

# Use appropriate operation
for dtype, op in ops.items():
    op.compute(d_data, d_out, size)
```

## 🔍 Comparison with NVIDIA CCCL

| Feature | NVIDIA CCCL | MACA CCCL |
|---------|-------------|-----------|
| Backend | CUB | mcCub |
| Runtime | CUDA | MACA |
| JIT | NVRTC | mcRTC |
| Language | C++17 | C++17 |
| Python | Cython | Cython |
| API Style | Identical | Identical |

## 🐛 Troubleshooting

### Build Errors

**Error: `mxcc not found`**
```bash
export MACA_PATH="/path/to/maca-sdk"
export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
```

**Error: `Cannot find cccl/c/reduce_official.h`**
```bash
export CCCL_C_INCLUDE="/path/to/parallel/include"
```

**Error: `ImportError: cannot import name '_bindings_impl'`**
```bash
# Rebuild Cython extension
python3 setup.py build_ext --inplace
```

### Runtime Errors

**Error: `mcErrorInvalidValue`**
- Check that device pointers are valid
- Verify data type matches built operation
- Ensure num_items > 0

**Error: `RuntimeError: MACA reduce failed`**
- Check MACA runtime initialization
- Verify device memory allocation
- Check stream validity

## 📝 TODO

- [ ] Implement mcRTC compilation wrapper
- [ ] Add DeviceScan (prefix sum)
- [ ] Add DeviceSort (radix, merge)
- [ ] Add DeviceSegmentedReduce
- [ ] Add DeviceTransform
- [ ] Add iterator support (counting, transform, etc.)
- [ ] Add caching layer for compiled operations
- [ ] Performance benchmarks vs hand-written kernels
- [ ] Integration tests with MACA runtime

## 📄 License

Adapted from NVIDIA CUDA CCCL under Apache License 2.0.

## 🤝 Contributing

This is an adaptation of NVIDIA CCCL for MACA. When implementing new features:

1. Follow NVIDIA CCCL's C API structure
2. Use mcCub internal dispatch pattern
3. Maintain build/execute separation
4. Add type-safe Python wrappers
5. Include examples and tests

## 📞 Support

For MACA-specific issues, consult Moore Threads documentation.
For CCCL API questions, refer to NVIDIA CCCL documentation.

---

**Version:** 0.1.0 (Based on CCCL from 7 months ago - intentional!)
**Last Updated:** 2026-01-20
