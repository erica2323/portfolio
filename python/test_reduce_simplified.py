#!/usr/bin/env python3
"""
Simplified test for CCCL reduce (MACA port, no Numba)

This test demonstrates usage of the simplified Python API that only
supports builtin operators (OpKind.SUM, MIN, MAX, etc.)

Prerequisites:
1. Build libcccl_maca.so (C layer)
2. Build Python bindings: python setup.py build_ext --inplace
3. Set environment variables:
   export MACA_PATH=/usr/local/maca
   export LD_LIBRARY_PATH=$MACA_PATH/lib64:$LD_LIBRARY_PATH

Usage:
    python test_reduce_simplified.py
"""

import sys
import numpy as np
import ctypes

# Load MACA runtime
try:
    maca = ctypes.CDLL("libmcruntime.so")
except OSError:
    print("ERROR: Cannot load libmcruntime.so")
    print("Please set LD_LIBRARY_PATH to include MACA lib64 directory")
    sys.exit(1)


class MacaArray:
    """
    Simple wrapper for MACA device arrays.
    Provides __cuda_array_interface__ for compatibility with CCCL API.
    """
    def __init__(self, shape, dtype):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape))
        self.nbytes = self.size * self.dtype.itemsize

        # Allocate device memory
        self.ptr = ctypes.c_void_p()
        ret = maca.mcMalloc(ctypes.byref(self.ptr), self.nbytes)
        if ret != 0:
            raise RuntimeError(f"mcMalloc failed with error {ret}")

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            maca.mcFree(self.ptr)

    @property
    def __cuda_array_interface__(self):
        """
        Provide CUDA Array Interface for compatibility.
        """
        return {
            'version': 3,
            'shape': self.shape,
            'typestr': self.dtype.str,
            'data': (self.ptr.value, False),  # (ptr, read_only)
            'strides': None,
        }

    def copy_to_device(self, host_array):
        """Copy host array to device."""
        if host_array.nbytes != self.nbytes:
            raise ValueError("Size mismatch")
        ret = maca.mcMemcpy(
            self.ptr,
            host_array.ctypes.data,
            self.nbytes,
            1  # mcMemcpyHostToDevice
        )
        if ret != 0:
            raise RuntimeError(f"mcMemcpy H2D failed with error {ret}")

    def copy_to_host(self, host_array):
        """Copy device array to host."""
        if host_array.nbytes != self.nbytes:
            raise ValueError("Size mismatch")
        ret = maca.mcMemcpy(
            host_array.ctypes.data,
            self.ptr,
            self.nbytes,
            2  # mcMemcpyDeviceToHost
        )
        if ret != 0:
            raise RuntimeError(f"mcMemcpy D2H failed with error {ret}")


def test_reduce_sum():
    """Test sum reduction with init value."""
    print("\n=== Test 1: Sum Reduction ===")

    # Import CCCL API
    from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind

    # Create input: 1000 ones
    N = 1000
    h_in = np.ones(N, dtype=np.int32)
    h_out = np.zeros(1, dtype=np.int32)

    # Allocate device arrays
    d_in = MacaArray(N, np.int32)
    d_out = MacaArray(1, np.int32)

    # Copy to device
    d_in.copy_to_device(h_in)

    # Run reduce
    reduce(d_in, d_out, op=OpKind.SUM, init=10)

    # Copy result back
    d_out.copy_to_host(h_out)

    # Check result
    expected = N + 10  # 1000 ones + init value 10
    print(f"Input: {N} ones")
    print(f"Init value: 10")
    print(f"Result: {h_out[0]}")
    print(f"Expected: {expected}")

    if h_out[0] == expected:
        print("✅ PASSED")
        return True
    else:
        print("❌ FAILED")
        return False


def test_reduce_min():
    """Test min reduction."""
    print("\n=== Test 2: Min Reduction ===")

    from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind

    # Create input: [5, 2, 8, 1, 9]
    h_in = np.array([5, 2, 8, 1, 9], dtype=np.int32)
    h_out = np.zeros(1, dtype=np.int32)

    # Allocate device arrays
    d_in = MacaArray(len(h_in), np.int32)
    d_out = MacaArray(1, np.int32)

    # Copy to device
    d_in.copy_to_device(h_in)

    # Run reduce with init=0
    reduce(d_in, d_out, op=OpKind.MIN, init=0)

    # Copy result back
    d_out.copy_to_host(h_out)

    # Check result (min of init and array)
    expected = min(0, *h_in)  # 0
    print(f"Input: {h_in}")
    print(f"Init value: 0")
    print(f"Result: {h_out[0]}")
    print(f"Expected: {expected}")

    if h_out[0] == expected:
        print("✅ PASSED")
        return True
    else:
        print("❌ FAILED")
        return False


def test_reduce_max():
    """Test max reduction."""
    print("\n=== Test 3: Max Reduction ===")

    from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind

    # Create input: [5, 2, 8, 1, 9]
    h_in = np.array([5, 2, 8, 1, 9], dtype=np.int32)
    h_out = np.zeros(1, dtype=np.int32)

    # Allocate device arrays
    d_in = MacaArray(len(h_in), np.int32)
    d_out = MacaArray(1, np.int32)

    # Copy to device
    d_in.copy_to_device(h_in)

    # Run reduce with init=100
    reduce(d_in, d_out, op=OpKind.MAX, init=100)

    # Copy result back
    d_out.copy_to_host(h_out)

    # Check result (max of init and array)
    expected = max(100, *h_in)  # 100
    print(f"Input: {h_in}")
    print(f"Init value: 100")
    print(f"Result: {h_out[0]}")
    print(f"Expected: {expected}")

    if h_out[0] == expected:
        print("✅ PASSED")
        return True
    else:
        print("❌ FAILED")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CCCL Reduce Tests (Simplified, No Numba)")
    print("=" * 60)

    results = []
    results.append(test_reduce_sum())
    results.append(test_reduce_min())
    results.append(test_reduce_max())

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    sys.exit(0 if all(results) else 1)
