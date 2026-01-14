# MACA CCCL Python Binding

Python binding for MACA CCCL Transform API using ctypes.

## 📁 File Structure

```
maca_cccl_python/
├── README.md              # This file
├── build_lib.sh           # Build script for shared library
├── maca_cccl.py          # Python ctypes wrapper
├── test_transform.py      # Test suite
└── examples/             # Usage examples (optional)
```

## 🔧 Prerequisites

1. **MACA SDK** installed
2. **Your C++ implementation** from previous session:
   - `project_2026_01_12/mcCub/c/parallel/include/cccl/c/types_official.h`
   - `project_2026_01_12/mcCub/c/parallel/include/cccl/c/transform_official.h`
   - `project_2026_01_12/mcCub/c/parallel/src/transform_official.cu`

3. **Python 3.x** with NumPy:
   ```bash
   pip install numpy
   ```

## 🚀 Quick Start

### Step 1: Update paths in build_lib.sh

Edit `build_lib.sh` and update these paths to match your system:

```bash
# Update MACA_PATH
export MACA_PATH="/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707"

# Update SOURCE_DIR (path to your C++ files)
SOURCE_DIR="../project_2026_01_12/mcCub/c"
```

### Step 2: Build shared library

```bash
cd maca_cccl_python
chmod +x build_lib.sh
./build_lib.sh
```

This creates `libmaca_cccl_transform.so`.

### Step 3: Set environment variables

```bash
export MACA_PATH="/path/to/your/maca-sdk"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$LD_LIBRARY_PATH"
```

### Step 4: Run tests

```bash
python3 test_transform.py
```

Expected output:
```
============================================================
MACA CCCL Transform Python Binding - Test Suite
============================================================

============================================================
Test 1: Built-in PLUS operation
============================================================
Input:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
Addend: 5
Output:   [ 5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
Expected: [ 5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
✅ Test 1: Built-in PLUS operation PASSED

... (more tests)

============================================================
Results: 7/7 tests passed
============================================================
```

## 📖 Python API Usage

### 1. Basic Import

```python
from maca_cccl import Transform, CCCLOpKind
import numpy as np
```

### 2. Built-in Operations

#### PLUS (add constant)

```python
# Method 1: Using Transform class
input_data = np.array([1, 2, 3, 4], dtype=np.int32)
addend = 5

transform = Transform(CCCLOpKind.PLUS, state=addend)
output = transform(input_data)
# Result: [6, 7, 8, 9]

# Method 2: Using convenience function
from maca_cccl import plus
output = plus(input_data, 5)
```

#### MULTIPLIES (multiply by constant)

```python
# Method 1: Using Transform class
input_data = np.array([2, 4, 6, 8], dtype=np.int32)
multiplier = 3

transform = Transform(CCCLOpKind.MULTIPLIES, state=multiplier)
output = transform(input_data)
# Result: [6, 12, 18, 24]

# Method 2: Using convenience function
from maca_cccl import multiplies
output = multiplies(input_data, 3)
```

### 3. Custom Operations (User C++ Code)

```python
# Method 1: Using Transform class
input_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)

user_code = """
__device__ int32_t user_op(int32_t x) {
    return x * x;  // Square operation
}
"""

transform = Transform(
    CCCLOpKind.STATELESS,
    in_dtype=np.int32,
    out_dtype=np.int32,
    user_code=user_code
)
output = transform(input_data)
# Result: [1, 4, 9, 16, 25]

# Method 2: Using convenience function
from maca_cccl import custom
output = custom(input_data, user_code)
```

### 4. Float Operations

```python
input_data = np.array([1.5, 2.5, 3.5], dtype=np.float32)

transform = Transform(
    CCCLOpKind.MULTIPLIES,
    in_dtype=np.float32,
    out_dtype=np.float32,
    state=2.0
)
output = transform(input_data)
# Result: [3.0, 5.0, 7.0]
```

### 5. Reusing Transform Objects

```python
# Build once, execute multiple times
transform = Transform(CCCLOpKind.PLUS, state=10)

# Execute on different inputs
result1 = transform(np.array([1, 2, 3], dtype=np.int32))
result2 = transform(np.array([10, 20, 30], dtype=np.int32))
result3 = transform(np.array([100, 200], dtype=np.int32))
```

## 🔍 Available Operations

### Built-in Operations

| Operation | CCCLOpKind | Description | State Parameter |
|-----------|------------|-------------|-----------------|
| Addition | `PLUS` | `out[i] = in[i] + addend` | addend (int/float) |
| Multiplication | `MULTIPLIES` | `out[i] = in[i] * multiplier` | multiplier (int/float) |
| Subtraction | `MINUS` | `out[i] = in[i] - subtrahend` | subtrahend (int/float) |
| Division | `DIVIDES` | `out[i] = in[i] / divisor` | divisor (int/float) |

**Note**: Currently only `PLUS` and `MULTIPLIES` are fully implemented in the C++ backend.

### Custom Operations

Use `CCCLOpKind.STATELESS` with `user_code` parameter to define custom operations:

```python
# Absolute value
code = "__device__ int32_t user_op(int32_t x) { return x < 0 ? -x : x; }"

# Clamp to range [0, 100]
code = "__device__ int32_t user_op(int32_t x) { return x < 0 ? 0 : (x > 100 ? 100 : x); }"

# Complex computation
code = """
__device__ int32_t user_op(int32_t x) {
    int32_t result = x * x + 2 * x + 1;
    return result > 1000 ? 1000 : result;
}
"""
```

## 🎯 API Reference

### Transform Class

```python
Transform(op_kind, in_dtype=np.int32, out_dtype=np.int32, state=None, user_code=None)
```

**Parameters:**
- `op_kind`: `CCCLOpKind` enum value
- `in_dtype`: NumPy dtype for input (default: `np.int32`)
- `out_dtype`: NumPy dtype for output (default: `np.int32`)
- `state`: State value for built-in operations (int or float)
- `user_code`: C++ source code string for custom operations

**Methods:**
- `__call__(input_data)`: Execute transform on NumPy array

**Returns:**
- NumPy array with transformed data

### Convenience Functions

```python
# Add constant
output = plus(input_data, addend)

# Multiply by constant
output = multiplies(input_data, multiplier)

# Custom operation
output = custom(input_data, code, in_dtype=np.int32, out_dtype=np.int32)
```

## 🐛 Troubleshooting

### Error: "Failed to load CCCL transform library"

**Solution**: Make sure you ran `build_lib.sh` first to create `libmaca_cccl_transform.so`.

### Error: "Failed to load MACA runtime library"

**Solution**: Set `MACA_PATH` environment variable:
```bash
export MACA_PATH="/path/to/maca-sdk"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$LD_LIBRARY_PATH"
```

### Error: "cccl_transform_build failed"

**Solution**: Check that your C++ source files are in the correct location specified in `build_lib.sh`.

### Error: "Input dtype doesn't match"

**Solution**: Make sure input NumPy array has the correct dtype:
```python
# Wrong
data = np.array([1, 2, 3])  # Default dtype might be int64

# Correct
data = np.array([1, 2, 3], dtype=np.int32)
```

## 📊 Performance Notes

- **Memory Management**: GPU memory is automatically allocated and freed
- **Reusability**: Build transform once, execute many times for best performance
- **Data Transfer**: Each call transfers data Host→Device→Host
- **Synchronization**: Automatic GPU synchronization after kernel execution

## 🚧 Current Limitations

1. **Unary Transform only**: Only one input array (Binary Transform not yet implemented)
2. **Limited built-in ops**: Only `PLUS` and `MULTIPLIES` fully tested
3. **Automatic memory**: No manual GPU memory control (could add advanced API)
4. **Single stream**: Uses default stream (could add multi-stream support)

## 🔜 Next Steps

1. **Implement Binary Transform**: Two input arrays
2. **Add Reduce operations**: Reduce array to single value
3. **Add Scan operations**: Prefix sum/scan
4. **Add more built-in operations**: MINUS, DIVIDES, etc.
5. **Advanced memory API**: User-managed GPU memory
6. **Stream support**: Concurrent kernel execution

## 📝 Example: Complete Workflow

```python
#!/usr/bin/env python3
import numpy as np
from maca_cccl import Transform, CCCLOpKind

# 1. Create input data
data = np.arange(1000, dtype=np.int32)

# 2. Build transform (compile kernel)
transform = Transform(CCCLOpKind.PLUS, state=42)

# 3. Execute transform (JIT execution on GPU)
result = transform(data)

# 4. Use result
print(f"Input:  {data[:10]}...")
print(f"Output: {result[:10]}...")
print(f"All correct: {np.allclose(result, data + 42)}")
```

## 📄 License

Same as MACA CCCL C++ implementation.

## 🤝 Contributing

This is a minimal working implementation. Contributions welcome for:
- Binary Transform
- More operation types
- Better error handling
- Performance optimizations
- Documentation improvements
