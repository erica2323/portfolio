# MACA CCCL Python Bindings

Python bindings for **MACA Core Compute Libraries (CCCL)**, providing high-performance GPU algorithms for the MACA platform.

## Overview

MACA CCCL Python provides a Pythonic interface to MACA's GPU computing libraries, offering:

- **Device-level algorithms**: Reduction, scan, sort, and more
- **High performance**: Direct dispatch to optimized mcCub kernels
- **Easy to use**: NumPy-like interface for GPU operations
- **Flexible**: Support for custom operations and types

## Architecture

This implementation follows the **NVIDIA CCCL architecture pattern**:

```
┌─────────────────────────────────────────┐
│         Python User Code                │
│  (NumPy-like high-level API)            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│    Python Wrapper Layer                 │
│  (maca.compute.algorithms)              │
│  • Type conversion                      │
│  • Caching                              │
│  • Memory management                    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Cython Bindings Layer              │
│  (_bindings_impl.pyx)                   │
│  • Python ↔ C type conversion           │
│  • GIL release for GPU work             │
│  • Memory safety                        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         C API Layer                     │
│  (cccl/c/*.h)                           │
│  • Build/Execute separation             │
│  • Iterator abstraction                 │
│  • Type system                          │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      mcCub Library (C++)                │
│  Device-level GPU algorithms            │
│  (like NVIDIA CUB for CUDA)             │
└─────────────────────────────────────────┘
```

## Key Concepts

### Build-Execute Separation

MACA CCCL follows a **two-phase execution model**:

1. **Build Phase**: Compile operation with type information
   - Type checking and validation
   - Kernel specialization
   - One-time compilation cost

2. **Execute Phase**: Run on actual data
   - Fast execution
   - Can be called multiple times
   - Low overhead

```python
# Build phase (compile once)
reduce_op = make_reduce_into(sample_array, op="sum")

# Execute phase (run many times)
reduce_op(data1, output1)
reduce_op(data2, output2)  # Reuses compiled kernel!
```

### Type System

MACA CCCL uses a rich type system:

```python
from maca.compute.types import TypeInfo, OpKind

# Create type info from NumPy dtype
typeinfo = TypeInfo(np.float32)

# Or explicitly
typeinfo = TypeInfo(TypeEnum.FLOAT32, size=4, alignment=4)
```

### Operations

Support for well-known operations:

```python
from maca.compute import reduce_into

# Sum (addition)
reduce_into(d_in, d_out, op="sum")

# Minimum
reduce_into(d_in, d_out, op="min")

# Maximum
reduce_into(d_in, d_out, op="max")
```

## Installation

### Prerequisites

- Python 3.10+
- MACA SDK installed
- NumPy
- Cython 3.0+
- CMake 3.21+

### Build from Source

```bash
# Clone repository
git clone <your-repo>
cd python/maca_cccl

# Install in development mode
pip install -e .

# Or build wheel
pip install build
python -m build
```

### Environment Variables

```bash
# Set MACA installation path
export MACA_PATH=/opt/maca

# Set mcCub path
export MCCUB_PATH=/path/to/mcCub
```

## Usage Examples

### Basic Reduction

```python
import numpy as np
from maca.compute import reduce_into

# Create data
n = 1000000
h_data = np.arange(n, dtype=np.float32)

# Allocate device memory (using your MACA memory API)
d_input = maca.device_array(n, dtype=np.float32)
d_output = maca.device_array(1, dtype=np.float32)

# Copy to device
d_input.copy_from_host(h_data)

# Perform reduction on GPU
reduce_into(d_input, d_output, op="sum")

# Get result
result = d_output.copy_to_host()[0]
print(f"Sum: {result}")
```

### Reusable Operations (More Efficient)

```python
from maca.compute import make_reduce_into

# Build phase: Compile once
reduce_sum = make_reduce_into(d_sample, op="sum")

# Execute phase: Run multiple times
for dataset in datasets:
    d_input.copy_from_host(dataset)
    reduce_sum(d_input, d_output)
    results.append(d_output.copy_to_host()[0])
```

### Different Operations

```python
# Sum
reduce_into(d_in, d_out, op="sum")

# Minimum
reduce_into(d_in, d_out, op="min")

# Maximum
reduce_into(d_in, d_out, op="max")
```

### Different Types

```python
# Works with multiple types
types = [np.int32, np.int64, np.float32, np.float64]

for dtype in types:
    d_in = maca.device_array(n, dtype=dtype)
    d_out = maca.device_array(1, dtype=dtype)
    reduce_into(d_in, d_out, op="sum")
```

## API Reference

### High-Level Functions

#### `reduce_into(d_in, d_out, op="sum", initial_value=None, stream=None)`

Perform device-wide reduction (single-call interface).

**Parameters:**
- `d_in`: Input array on device
- `d_out`: Output buffer on device (single element)
- `op`: Operation ("sum", "min", "max")
- `initial_value`: Initial value for reduction (optional)
- `stream`: MACA stream handle (optional)

#### `make_reduce_into(d_in, op="sum", initial_value=None)`

Create reusable reduction operation (build phase).

**Parameters:**
- `d_in`: Sample input array (for type detection)
- `op`: Operation to perform
- `initial_value`: Initial value (optional)

**Returns:**
- Compiled reduction operation (callable)

### Low-Level Classes

#### `TypeInfo`

Type information for CCCL operations.

```python
typeinfo = TypeInfo(np.float32)
# or
typeinfo = TypeInfo(TypeEnum.FLOAT32, size=4, alignment=4)
```

#### `Op`

Operation wrapper.

```python
from maca.compute.op import Op, OpKind

op = Op(OpKind.PLUS, name="addition")
```

## How It Works

### 1. Type Conversion Flow

```
Python Value → NumPy dtype → CCCL TypeInfo → C struct
    ↓              ↓              ↓             ↓
  42.0    →   float32    →   FLOAT32    →  cccl_type_info
```

### 2. Operation Flow

```
Python Call
    ↓
reduce_into(d_in, d_out, op="sum")
    ↓
make_reduce_into() [Cached]
    ↓
_Reduce.__init__() [Build Phase]
    ↓
_bindings.device_reduce_build()
    ↓
cccl_device_reduce_build_ex() [C]
    ↓
Configure mcCub parameters
    ↓
Return build_result
    ↓
_Reduce.__call__() [Execute Phase]
    ↓
_bindings.device_reduce()
    ↓
cccl_device_reduce_ex() [C]
    ↓
dispatch_reduce_internal<T>()
    ↓
DispatchReduce<>::Dispatch() [mcCub]
    ↓
GPU Kernel Execution
```

### 3. Memory Management

- **Input/Output**: Device pointers extracted via `__cuda_array_interface__` or similar
- **Temporary Storage**: Managed by mcCub (allocated as needed)
- **Build Results**: Cleaned up automatically (RAII pattern in Cython)

### 4. Caching

Operations are cached by `(op_kind, dtype)` to avoid recompilation:

```python
@lru_cache(maxsize=128)
def _make_reduce_into_cached(op_kind, dtype_str, initial_value):
    # Compile operation
    return _Reduce(op, typeinfo, initial_value)
```

## Performance Tips

1. **Reuse compiled operations**:
   ```python
   # Good: Compile once
   op = make_reduce_into(d_sample, "sum")
   for data in datasets:
       op(data, output)

   # Bad: Recompile every time
   for data in datasets:
       reduce_into(data, output, "sum")  # Cached, but still overhead
   ```

2. **Use appropriate types**: Smaller types = faster transfers
   ```python
   # float32 is often enough and 2x faster than float64
   data = np.array(..., dtype=np.float32)
   ```

3. **Stream execution**: Use MACA streams for concurrency
   ```python
   reduce_into(d_in, d_out, stream=stream1)
   ```

## Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_reduce.py::TestReduceBasic::test_sum_reduce

# With coverage
pytest --cov=maca --cov-report=html tests/
```

## Troubleshooting

### Import Error: `_bindings_impl` not found

**Solution**: Make sure the Cython extension is compiled:
```bash
pip install -e .
```

### Type Error: Cannot extract device pointer

**Solution**: Ensure your array object provides `__cuda_array_interface__` or `data_ptr()` method.

### Build Error: MACA headers not found

**Solution**: Set `MACA_PATH` environment variable:
```bash
export MACA_PATH=/opt/maca
```

## Contributing

Contributions welcome! Please:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Submit pull request

## License

Apache 2.0

## References

- [NVIDIA CCCL](https://github.com/NVIDIA/cccl)
- [CUB Documentation](https://nvidia.github.io/cccl/cub/)
- [Cython Documentation](https://cython.readthedocs.io/)
