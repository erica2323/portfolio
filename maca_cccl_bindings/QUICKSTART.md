# Quick Start Guide - MACA CCCL Python Bindings

## 🚀 5-Minute Setup

### Step 1: Set Environment (30 seconds)

```bash
# Your MACA installation
export MACA_PATH="/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707"

# Your mcCub library
export MCCUB_PATH="/mnt/data/minxi/1_15/mcCub"

# Your CCCL C headers (parallel/include)
export CCCL_C_INCLUDE="./parallel/include"

# Add to shell profile for persistence
echo "export MACA_PATH=$MACA_PATH" >> ~/.bashrc
echo "export MCCUB_PATH=$MCCUB_PATH" >> ~/.bashrc
echo "export CCCL_C_INCLUDE=$CCCL_C_INCLUDE" >> ~/.bashrc
```

### Step 2: Install Dependencies (1 minute)

```bash
# Python dependencies
pip install numpy cython

# Or with conda
conda install numpy cython
```

### Step 3: Build (1 minute)

```bash
cd maca_cccl_bindings
chmod +x build_python.sh
./build_python.sh
```

Expected output:
```
✅ Python: 3.x
✅ NumPy installed
✅ Cython installed
✅ MACA found
✅ mcCub found
✅ Cython extension built successfully
```

### Step 4: Test (30 seconds)

```bash
python3 -c "from maca_cccl import DeviceReduce; print('✅ Import successful!')"
```

### Step 5: Run Example (1 minute)

```bash
python3 example_reduce.py
```

## 💻 Your First Reduce Operation

Create `my_first_reduce.py`:

```python
import numpy as np
from maca_cccl import DeviceReduce

# 1. Build the operation (compile once)
print("Building sum reduction...")
reduce_sum = DeviceReduce.sum(np.int32)
print("✅ Built!")

# 2. In real usage with MACA runtime:
# import maca_runtime as maca
#
# # Allocate and copy data
# h_data = np.arange(10000, dtype=np.int32)
# d_data = maca.mem_copy_host_to_device(h_data)
# d_out = maca.mem_alloc(np.int32().nbytes)
#
# # Execute reduction
# reduce_sum.compute(d_data, d_out, len(h_data))
#
# # Get result
# result = maca.mem_copy_device_to_host(d_out, 1, np.int32)
# print(f"Sum: {result[0]}")
# print(f"Expected: {h_data.sum()}")
#
# # Cleanup
# maca.mem_free(d_data)
# maca.mem_free(d_out)

print("\n✅ Ready to use with MACA runtime!")
```

Run it:
```bash
python3 my_first_reduce.py
```

## 🎯 Common Use Cases

### Use Case 1: Sum of Array

```python
from maca_cccl import reduce_sum
import numpy as np

# One-line reduction
reduce_sum(d_in, d_out, num_items, np.int32)
```

### Use Case 2: Find Min/Max

```python
from maca_cccl import DeviceReduce

reduce_min = DeviceReduce.min(np.float32)
reduce_max = DeviceReduce.max(np.float32)

reduce_min.compute(d_data, d_min, size)
reduce_max.compute(d_data, d_max, size)
```

### Use Case 3: Reuse Operation

```python
# Build once
reduce_op = DeviceReduce.sum(np.int64)

# Use many times
for batch in data_batches:
    reduce_op.compute(d_batch, d_out, len(batch))
```

## 🔧 Integration with Your C Layer

Your C layer remains **unchanged**. Python bindings call it like this:

```
Python API:          DeviceReduce.sum(np.int32)
       ↓
Cython Layer:        device_reduce_build(OpKind.PLUS, ...)
       ↓
Your C API:          cccl_device_reduce_build(...)
       ↓
Your C Code:         dispatch_reduce_internal<int32_t>(...)
       ↓
mcCub:               DispatchReduce<...>::Dispatch(...)
       ↓
MACA GPU:            [Hardware Execution]
```

## 📋 Supported Types

| Python/NumPy | C Type | Example |
|--------------|--------|---------|
| `np.int8` | `int8_t` | `DeviceReduce.sum(np.int8)` |
| `np.int16` | `int16_t` | `DeviceReduce.sum(np.int16)` |
| `np.int32` | `int32_t` | `DeviceReduce.sum(np.int32)` |
| `np.int64` | `int64_t` | `DeviceReduce.sum(np.int64)` |
| `np.uint8` | `uint8_t` | `DeviceReduce.sum(np.uint8)` |
| `np.uint16` | `uint16_t` | `DeviceReduce.sum(np.uint16)` |
| `np.uint32` | `uint32_t` | `DeviceReduce.sum(np.uint32)` |
| `np.uint64` | `uint64_t` | `DeviceReduce.sum(np.uint64)` |
| `np.float32` | `float` | `DeviceReduce.sum(np.float32)` |
| `np.float64` | `double` | `DeviceReduce.sum(np.float64)` |

## 📊 Supported Operations

| Python API | C Op | Description |
|------------|------|-------------|
| `DeviceReduce.sum()` | `CCCL_PLUS` | Sum all elements |
| `DeviceReduce.min()` | `CCCL_MINIMUM` | Find minimum |
| `DeviceReduce.max()` | `CCCL_MAXIMUM` | Find maximum |

## 🐛 Troubleshooting

### Problem: Import fails
```
ImportError: No module named 'maca_cccl'
```
**Solution:**
```bash
cd maca_cccl_bindings
python3 setup.py install --user
```

### Problem: Build fails with "mxcc not found"
```bash
export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
```

### Problem: "Cannot find reduce_official.h"
```bash
# Make sure your C layer is compiled and headers are accessible
export CCCL_C_INCLUDE="/path/to/parallel/include"
```

### Problem: Runtime error "mcErrorInvalidValue"
- Check device pointers are valid MACA device memory
- Verify num_items > 0
- Ensure dtype matches the data type

## 📖 Learn More

- **Full Documentation**: See `README.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Examples**: See `example_reduce.py`
- **JIT/mcRTC**: See `_jit_templates.py`

## 🎓 Next Steps

1. **Integrate with MACA Runtime**
   - Replace placeholder pointers with actual `maca.mem_alloc()`
   - Add proper host↔device memory transfers

2. **Add More Algorithms**
   - Copy `_reduce.py` → `_scan.py` for prefix sum
   - Copy `_reduce.py` → `_sort.py` for sorting
   - Use same pattern: Cython bindings → Python API

3. **Implement JIT/mcRTC** (optional)
   - For custom operators
   - See `_jit_templates.py` for template
   - Reference NVIDIA NVRTC documentation

4. **Performance Testing**
   - Benchmark vs hand-written kernels
   - Profile build time vs execute time
   - Test with various data sizes

## 💡 Pro Tips

1. **Build Once, Execute Many**
   ```python
   # Good ✅
   reduce_op = DeviceReduce.sum(np.int32)
   for _ in range(1000):
       reduce_op.compute(d_in, d_out, size)

   # Bad ❌
   for _ in range(1000):
       reduce_op = DeviceReduce.sum(np.int32)  # Rebuilds every time!
       reduce_op.compute(d_in, d_out, size)
   ```

2. **Reuse for Same Type**
   ```python
   # Create once for int32
   reduce_int32 = DeviceReduce.sum(np.int32)

   # Use with any int32 data
   reduce_int32.compute(d_data1, d_out1, size1)
   reduce_int32.compute(d_data2, d_out2, size2)
   ```

3. **Type Consistency**
   ```python
   # Data type must match build type
   reduce_op = DeviceReduce.sum(np.float32)
   # d_in and d_out must point to float32 data!
   reduce_op.compute(d_in, d_out, size)
   ```

## ✅ Checklist

- [ ] Set environment variables (MACA_PATH, etc.)
- [ ] Install dependencies (numpy, cython)
- [ ] Build Python extension
- [ ] Test import
- [ ] Run examples
- [ ] Integrate with your MACA runtime
- [ ] Test with real data

## 🎉 You're Ready!

You now have a complete Python interface to your MACA CCCL C layer!

```python
from maca_cccl import DeviceReduce
reduce_op = DeviceReduce.sum(np.int32)
reduce_op.compute(d_in, d_out, num_items)
```

**That's it! 🚀**

---

Questions? Check `README.md` or `IMPLEMENTATION_SUMMARY.md`
