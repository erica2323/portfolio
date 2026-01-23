#!/usr/bin/env python3
"""
JIT-enabled CCCL reduce test (MACA port)

Demonstrates custom operator compilation via MCRTC
"""
import sys
import numpy as np
import ctypes

# ============================================================================
# Load MACA runtime
# ============================================================================
try:
    maca = ctypes.CDLL("libmcruntime.so")
except OSError:
    print("ERROR: Cannot load libmcruntime.so")
    sys.exit(1)

maca.mcMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
maca.mcMalloc.restype = ctypes.c_int
maca.mcFree.argtypes = [ctypes.c_void_p]
maca.mcFree.restype = ctypes.c_int
maca.mcMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
maca.mcMemcpy.restype = ctypes.c_int

# ============================================================================
# Simple device array wrapper (reuse from previous test)
# ============================================================================
class MacaArray:
    def __init__(self, shape, dtype):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape))
        self.nbytes = self.size * self.dtype.itemsize

        self.ptr = ctypes.c_void_p(None)
        ret = maca.mcMalloc(ctypes.byref(self.ptr), ctypes.c_size_t(self.nbytes))
        if ret != 0 or not self.ptr.value:
            raise RuntimeError(f"mcMalloc failed (ret={ret})")

    def __del__(self):
        try:
            if getattr(self, "ptr", None) is not None and self.ptr.value:
                maca.mcFree(self.ptr)
                self.ptr = ctypes.c_void_p(None)
        except Exception:
            pass

    @property
    def __cuda_array_interface__(self):
        return {
            "version": 3,
            "shape": self.shape,
            "typestr": self.dtype.str,
            "data": (int(self.ptr.value), False),
            "strides": None,
        }

    def copy_to_device(self, host_array: np.ndarray):
        host_array = np.ascontiguousarray(host_array, dtype=self.dtype)
        if host_array.nbytes != self.nbytes:
            raise ValueError(f"Size mismatch")
        ret = maca.mcMemcpy(
            ctypes.c_void_p(int(self.ptr.value)),
            ctypes.c_void_p(int(host_array.ctypes.data)),
            ctypes.c_size_t(self.nbytes),
            ctypes.c_int(1),  # H2D
        )
        if ret != 0:
            raise RuntimeError(f"mcMemcpy H2D failed")

    def copy_to_host(self, host_array: np.ndarray):
        if host_array.dtype != self.dtype or host_array.nbytes != self.nbytes:
            raise ValueError("Size/dtype mismatch")
        ret = maca.mcMemcpy(
            ctypes.c_void_p(int(host_array.ctypes.data)),
            ctypes.c_void_p(int(self.ptr.value)),
            ctypes.c_size_t(self.nbytes),
            ctypes.c_int(2),  # D2H
        )
        if ret != 0:
            raise RuntimeError(f"mcMemcpy D2H failed")

# ============================================================================
# Test 1: Builtin operators (should work as before)
# ============================================================================
def test_builtin_sum():
    print("\n=== Test 1: Builtin SUM (no JIT) ===")
    from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind

    N = 1000
    h_in = np.ones(N, dtype=np.int32)
    h_out = np.zeros(1, dtype=np.int32)

    d_in = MacaArray(N, np.int32)
    d_out = MacaArray(1, np.int32)

    d_in.copy_to_device(h_in)

    reduce(d_in, d_out, op=OpKind.SUM, init=10)

    d_out.copy_to_host(h_out)

    expected = N + 10
    print(f"Result: {h_out[0]}, Expected: {expected}")

    ok = (int(h_out[0]) == int(expected))
    print("✅ PASSED" if ok else "❌ FAILED")
    return ok

# ============================================================================
# Test 2: Custom operator via JIT (NEW!)
# ============================================================================
def test_jit_custom_multiply():
    """
    Test custom operator: multiply reduction (product)

    This requires JIT compilation since multiply is not a builtin
    """
    print("\n=== Test 2: Custom MULTIPLY via JIT ===")

    # Import extended API that supports custom operators
    try:
        from cuda_cccl.cuda.cccl.parallel.experimental import reduce_with_custom_op
    except ImportError:
        print("⚠️  JIT API not available yet - need to extend Python bindings")
        print("This test demonstrates the intended usage pattern")
        return None

    # Define custom operator as C++ lambda source code
    custom_op_code = """
    [](const auto& a, const auto& b) {
        return a * b;
    }
    """

    h_in = np.array([2, 3, 4, 5], dtype=np.int32)
    h_out = np.zeros(1, dtype=np.int32)

    d_in = MacaArray(len(h_in), np.int32)
    d_out = MacaArray(1, np.int32)

    d_in.copy_to_device(h_in)

    # Use JIT-compiled custom operator
    reduce_with_custom_op(
        d_in,
        d_out,
        op_code=custom_op_code,
        op_name="multiply_op",
        init=1
    )

    d_out.copy_to_host(h_out)

    expected = 1 * 2 * 3 * 4 * 5  # 120
    print(f"Input: {h_in}")
    print(f"Result: {h_out[0]}, Expected: {expected}")

    ok = (int(h_out[0]) == int(expected))
    print("✅ PASSED" if ok else "❌ FAILED")
    return ok

# ============================================================================
# Test 3: Custom operator - squared sum
# ============================================================================
def test_jit_squared_sum():
    """
    Test custom operator: sum of squares

    Operator: (a, b) => a + b*b
    """
    print("\n=== Test 3: Custom SQUARED SUM via JIT ===")

    try:
        from cuda_cccl.cuda.cccl.parallel.experimental import reduce_with_custom_op
    except ImportError:
        print("⚠️  JIT API not available yet")
        return None

    custom_op_code = """
    [](const auto& a, const auto& b) {
        return a + (b * b);
    }
    """

    h_in = np.array([1, 2, 3, 4], dtype=np.float32)
    h_out = np.zeros(1, dtype=np.float32)

    d_in = MacaArray(len(h_in), np.float32)
    d_out = MacaArray(1, np.float32)

    d_in.copy_to_device(h_in)

    reduce_with_custom_op(
        d_in,
        d_out,
        op_code=custom_op_code,
        op_name="squared_sum",
        init=0.0
    )

    d_out.copy_to_host(h_out)

    # Manual calculation: init=0
    # 0 + 1^2 = 1
    # 1 + 2^2 = 5
    # 5 + 3^2 = 14
    # 14 + 4^2 = 30
    expected = 30.0

    print(f"Input: {h_in}")
    print(f"Result: {h_out[0]:.1f}, Expected: {expected:.1f}")

    ok = (abs(h_out[0] - expected) < 0.01)
    print("✅ PASSED" if ok else "❌ FAILED")
    return ok

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CCCL Reduce with JIT Tests (MACA)")
    print("=" * 70)

    results = []

    # Test builtin operators (should work)
    results.append(test_builtin_sum())

    # Test JIT operators (will show intended API)
    jit_test_1 = test_jit_custom_multiply()
    if jit_test_1 is not None:
        results.append(jit_test_1)

    jit_test_2 = test_jit_squared_sum()
    if jit_test_2 is not None:
        results.append(jit_test_2)

    print("\n" + "=" * 70)
    print(f"Results: {sum(1 for r in results if r)}/{len(results)} tests passed")

    if None in [jit_test_1, jit_test_2]:
        print("\n📝 Next steps to enable JIT tests:")
        print("1. Extend _bindings_maca.pyx to accept op.code parameter")
        print("2. Add reduce_with_custom_op() to _reduce_simplified.py")
        print("3. Rebuild with JIT-enabled build.sh")

    print("=" * 70)

    sys.exit(0 if all(results) else 1)
