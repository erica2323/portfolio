# MACA CCCL Python Bindings - Quick Start Guide

Get up and running with MACA CCCL Python bindings in minutes!

## 📁 Project Structure

```
python/maca_cccl/
├── maca/                           # Python package
│   ├── __init__.py                 # Package root
│   └── compute/                    # Compute algorithms module
│       ├── __init__.py             # Module exports
│       ├── _bindings_impl.pyx      # 🔧 Cython bindings (C bridge)
│       ├── types.py                # Type system (TypeInfo, OpKind)
│       ├── op.py                   # Operation wrappers
│       └── algorithms/             # High-level algorithms
│           ├── __init__.py
│           └── _reduce.py          # Reduce implementation
│
├── tests/                          # Test suite
│   └── test_reduce.py              # Reduce tests
│
├── examples/                       # Usage examples
│   └── reduce_example.py           # Complete reduce example
│
├── pyproject.toml                  # 📦 Build configuration
├── CMakeLists.txt                  # 🔨 CMake build
├── build.sh                        # 🚀 Build script
├── README.md                       # Documentation
├── ARCHITECTURE.md                 # Architecture details
└── QUICKSTART.md                   # This file
```

## 🚀 Installation

### Step 1: Prerequisites

```bash
# Python 3.10+
python3 --version

# Install build dependencies
pip install cython numpy build

# Ensure MACA SDK is installed
export MACA_PATH=/opt/maca
```

### Step 2: Build and Install

```bash
cd python/maca_cccl

# Option A: Development install (recommended for development)
./build.sh develop

# Option B: Regular install
./build.sh install

# Option C: Build wheel
./build.sh wheel
pip install dist/*.whl
```

### Step 3: Verify Installation

```python
python3 -c "from maca.compute import reduce_into; print('✅ Success!')"
```

## 💡 Basic Usage

### Example 1: Sum Reduction

```python
import numpy as np
from maca.compute import reduce_into

# Create test data
n = 1000
h_data = np.arange(n, dtype=np.float32)

# Allocate device memory (pseudo-code, use your MACA API)
d_input = maca.device_array(n, dtype=np.float32)
d_output = maca.device_array(1, dtype=np.float32)

# Copy to device
d_input.copy_from_host(h_data)

# Perform reduction on GPU
reduce_into(d_input, d_output, op="sum")

# Get result
result = d_output.copy_to_host()[0]
print(f"Sum: {result}")  # Sum: 499500.0
```

### Example 2: Reusable Operations (Faster!)

```python
from maca.compute import make_reduce_into

# Build phase: Compile once
reduce_op = make_reduce_into(d_sample, op="sum")

# Execute phase: Use multiple times
for dataset in datasets:
    d_input.copy_from_host(dataset)
    reduce_op(d_input, d_output)  # Fast! No recompilation
    print(d_output.copy_to_host()[0])
```

### Example 3: Different Operations

```python
# Sum
reduce_into(d_in, d_out, op="sum")

# Minimum
reduce_into(d_in, d_out, op="min")

# Maximum
reduce_into(d_in, d_out, op="max")
```

## 🔧 How It Works

### Two-Phase Execution

MACA CCCL uses a **build-execute** pattern:

```python
# Phase 1: Build (compile operation with types)
reduce_op = make_reduce_into(d_sample, op="sum")
#                             ↑
#                    Uses type from array

# Phase 2: Execute (run on data)
reduce_op(d_input, d_output)
#         ↑        ↑
#      input     output (1 element)
```

### Architecture Flow

```
Your Python Code
      ↓
  reduce_into(d_in, d_out, "sum")
      ↓
  _Reduce.__call__()
      ↓
  Cython Bindings (_bindings_impl.pyx)
      ↓
  C API (cccl_device_reduce_ex)
      ↓
  mcCub Library (DispatchReduce)
      ↓
  GPU Kernel Execution
      ↓
  Result in d_out
```

## 📖 API Reference

### High-Level Functions

#### `reduce_into(d_in, d_out, op="sum", initial_value=None, stream=None)`

Perform device-wide reduction (single call).

**Parameters:**
- `d_in`: Input array on device
- `d_out`: Output buffer (1 element) on device
- `op`: Operation - `"sum"`, `"min"`, or `"max"`
- `initial_value`: Initial value (optional, auto-detected)
- `stream`: MACA stream handle (optional)

**Example:**
```python
reduce_into(d_input, d_output, op="sum")
```

#### `make_reduce_into(d_in, op="sum", initial_value=None)`

Create reusable reduction operation.

**Parameters:**
- `d_in`: Sample input (for type detection)
- `op`: Operation to perform
- `initial_value`: Initial value (optional)

**Returns:**
- Callable reduction operation

**Example:**
```python
reduce_op = make_reduce_into(d_sample, op="sum")
reduce_op(d_input, d_output)  # Execute
```

### Supported Types

- `np.int32` - 32-bit signed integer
- `np.int64` - 64-bit signed integer
- `np.float32` - 32-bit floating point
- `np.float64` - 64-bit floating point

### Supported Operations

| Operation | Aliases | Initial Value |
|-----------|---------|---------------|
| Sum       | `"sum"`, `"plus"`, `"add"` | 0 |
| Minimum   | `"min"`, `"minimum"` | type max |
| Maximum   | `"max"`, `"maximum"` | type min |

## 🧪 Testing

```bash
# Run all tests
./build.sh test

# Run specific test
pytest tests/test_reduce.py -v

# Run with output
pytest tests/test_reduce.py -v -s
```

## 🐛 Troubleshooting

### ImportError: _bindings_impl not found

**Problem:** Cython extension not built.

**Solution:**
```bash
./build.sh clean
./build.sh develop
```

### Cannot extract device pointer

**Problem:** Array object doesn't provide `__cuda_array_interface__`.

**Solution:** Your array must support one of:
- `__cuda_array_interface__` (standard)
- `data_ptr()` method (PyTorch-style)
- `.ctypes.data` attribute (NumPy, but warns)

**Example:**
```python
class MyDeviceArray:
    @property
    def __cuda_array_interface__(self):
        return {
            'data': (self._ptr, False),  # (pointer, read_only)
            'shape': (self.size,),
            'typestr': self.dtype.str,
            'version': 3
        }
```

### Build fails: MACA headers not found

**Problem:** CMake can't find MACA installation.

**Solution:**
```bash
export MACA_PATH=/opt/maca
./build.sh develop
```

### Build fails: mcCub not found

**Problem:** CMake can't find mcCub headers.

**Solution:**
```bash
export MCCUB_PATH=/path/to/mcCub
./build.sh develop
```

## 📚 Next Steps

1. **Run Examples:**
   ```bash
   python examples/reduce_example.py
   ```

2. **Read Documentation:**
   - `README.md` - Overview and usage
   - `ARCHITECTURE.md` - Deep dive into architecture

3. **Explore Code:**
   - `maca/compute/_bindings_impl.pyx` - Cython bindings
   - `maca/compute/algorithms/_reduce.py` - High-level API
   - `tests/test_reduce.py` - Test examples

4. **Adapt to Your Project:**
   - Implement device memory interface
   - Add your own operations
   - Extend to other algorithms (scan, sort)

## 💬 Common Patterns

### Pattern 1: Single Operation, Many Datasets

```python
# Compile once
reduce_sum = make_reduce_into(d_sample, "sum")

# Use many times
for data in datasets:
    d_input.copy_from_host(data)
    reduce_sum(d_input, d_output)
    results.append(d_output.copy_to_host()[0])
```

### Pattern 2: Different Operations, Same Data

```python
# Allocate output
d_out = maca.device_array(1, dtype=np.float32)

# Perform different reductions
reduce_into(d_data, d_out, "sum")
sum_result = d_out.copy_to_host()[0]

reduce_into(d_data, d_out, "min")
min_result = d_out.copy_to_host()[0]

reduce_into(d_data, d_out, "max")
max_result = d_out.copy_to_host()[0]
```

### Pattern 3: Type-Specific Processing

```python
def process_data(data, dtype):
    # Create device arrays with specific type
    d_in = maca.device_array(len(data), dtype=dtype)
    d_out = maca.device_array(1, dtype=dtype)

    # Copy and reduce
    d_in.copy_from_host(data.astype(dtype))
    reduce_into(d_in, d_out, "sum")

    return d_out.copy_to_host()[0]

# Use with different types
result_f32 = process_data(data, np.float32)
result_f64 = process_data(data, np.float64)
```

## 🎯 Performance Tips

1. **Reuse compiled operations** - Avoid recompiling same type/op combinations
2. **Use appropriate precision** - `float32` is often sufficient and 2x faster
3. **Batch operations** - Group similar operations together
4. **Use streams** - Enable concurrent execution
5. **Minimize transfers** - Keep data on device when possible

## 📞 Getting Help

- 📖 Read `README.md` for detailed documentation
- 🏗️ Read `ARCHITECTURE.md` for implementation details
- 🐛 Check troubleshooting section above
- 💻 Examine example code in `examples/`
- 🧪 Look at tests in `tests/` for usage patterns

Happy coding! 🚀
