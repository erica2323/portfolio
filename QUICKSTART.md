# Phase B Quick Start Guide

**TL;DR**: Copy-paste these commands to build and test Phase B.

## Prerequisites Check

```bash
# Check MACA installation
ls /opt/maca/lib/libmcruntime.so

# Check Phase A library exists
ls parallel/lib/libcccl_maca.so

# Check Python dependencies
python -c "import numpy, Cython; print('✓ Dependencies OK')"
```

If any fail, see `PHASE_B_BUILD.md` for setup instructions.

## Build Phase B (3 commands)

```bash
# 1. Set environment
export MACA_PATH=/opt/maca
export CCCL_PATH=$(pwd)
export LD_LIBRARY_PATH=$MACA_PATH/lib:$CCCL_PATH/parallel/lib:$LD_LIBRARY_PATH

# 2. Build Cython extension
python setup.py build_ext --inplace

# 3. Test import
python -c "from cuda.cccl.parallel.experimental import reduce_into; print('✓ Import works')"
```

## Run Tests

```bash
# Run full Phase B test suite
python test_reduce_phase_b.py

# Quick verification
python -c "
import numpy as np
from cuda.cccl.parallel.experimental import reduce_into
d_in = np.ones(1000, dtype=np.int32)
d_out = np.zeros(1, dtype=np.int32)
reduce_into(d_in, d_out, 'add', h_init=10)
print(f'Result: {d_out[0]} (expected: 1010)')
assert d_out[0] == 1010
print('✓ Test passed!')
"
```

## Expected Output

```
============================================================
Phase B Test Suite: Python Wrapper Layer
============================================================
Test 1: Basic sum (int32, 1000 elements, init=10)
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
Total: 4/4 tests passed

🎉 All Phase B tests passed!
```

## If Something Breaks

### Error: "libmcruntime.so not found"
```bash
# Find MACA lib directory
find /opt/maca -name "libmcruntime.so"
# Then export that path
export LD_LIBRARY_PATH=/path/to/maca/lib:$LD_LIBRARY_PATH
```

### Error: "libcccl_maca.so not found"
```bash
# Check if Phase A library exists
ls parallel/lib/libcccl_maca.so
# If missing, rebuild Phase A first
```

### Error: "cannot import _bindings_maca"
```bash
# Rebuild Cython extension
python setup.py build_ext --inplace --force
```

### Error: "cccl_device_reduce_build failed"
```bash
# Test Phase A standalone
cd parallel/test
./test_reduce_phase_a
# If Phase A fails, fix that first
```

## What You Can Do After Phase B

```python
import numpy as np
from cuda.cccl.parallel.experimental import reduce_into
import operator

# Basic reduction
d_in = np.ones(1000, dtype=np.int32)
d_out = np.zeros(1, dtype=np.int32)
reduce_into(d_in, d_out, 'add')  # Sum

# With initial value
reduce_into(d_in, d_out, 'add', h_init=10)

# Different operators
reduce_into(d_in, d_out, operator.add)  # Sum
reduce_into(d_in, d_out, min)            # Min
reduce_into(d_in, d_out, max)            # Max
reduce_into(d_in, d_out, operator.mul)   # Product

# String operators
reduce_into(d_in, d_out, '+')
reduce_into(d_in, d_out, 'min')
reduce_into(d_in, d_out, 'max')
reduce_into(d_in, d_out, 'mul')

# Different types
for dtype in [np.int32, np.int64, np.float32, np.float64]:
    d_in = np.ones(1000, dtype=dtype)
    d_out = np.zeros(1, dtype=dtype)
    reduce_into(d_in, d_out, operator.add)
    print(f"{dtype.__name__}: {d_out[0]}")
```

## What You CAN'T Do Yet (Need Phase C+)

```python
# ❌ Lambda functions (need Phase C)
reduce_into(d_in, d_out, lambda a, b: a + b)

# ❌ Advanced iterators (need Phase D)
from cuda.cccl.iterators import counting_iterator
it = counting_iterator(0)
reduce_into(it, d_out, operator.add, num_items=1000)

# ❌ Complex types (need Phase E)
d_in = np.array([1+2j, 3+4j], dtype=np.complex64)
reduce_into(d_in, d_out, operator.add)

# ❌ Streams (need Phase E)
stream = mc.Stream()
reduce_into(d_in, d_out, operator.add, stream=stream)
```

## Next Steps After Success

1. **Celebrate!** 🎉 You have a working Python API for MACA CCCL
2. **Try it on real data**: Test with your actual use cases
3. **Decide on Phase C**: Do you need lambda function support?
4. **Read the roadmap**: See `MACA_CCCL_ROADMAP.md` for the full plan

## Getting Help

- **Build issues**: `PHASE_B_BUILD.md` (detailed troubleshooting)
- **Architecture questions**: `MACA_CCCL_ROADMAP.md` (big picture)
- **Code questions**: Read comments in `algorithms.py`, `_cccl_interop.py`

## Quick Commands Reference

```bash
# Build
python setup.py build_ext --inplace

# Test
python test_reduce_phase_b.py

# Quick test
python -c "from cuda.cccl.parallel.experimental import reduce_into; print('✓')"

# Clean build
rm -rf build cuda/cccl/parallel/experimental/_bindings_maca*.{c,cpp,so}
python setup.py build_ext --inplace

# Check library paths
ldd cuda/cccl/parallel/experimental/_bindings_maca*.so

# Verify GPU
python -c "
import ctypes
mc = ctypes.CDLL('libmcruntime.so')
count = ctypes.c_int()
mc.mcGetDeviceCount(ctypes.byref(count))
print(f'MACA devices: {count.value}')
"
```

That's it! You should be up and running in < 5 minutes. 🚀
