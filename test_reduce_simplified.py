#!/usr/bin/env python3
"""
Simplified test for CCCL reduce (MACA port, no Numba)

Uses the proper MACA runtime wrapper module for safe ctypes handling.
"""

import sys
import numpy as np

# Import MACA runtime wrapper (handles ctypes signatures correctly)
from cuda_cccl.cuda.cccl.parallel.experimental._maca_runtime import MacaArray


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_reduce_sum():
    print("\n=== Test 1: Sum Reduction ===")
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
    print(f"Input: {N} ones")
    print(f"Init value: 10")
    print(f"Result: {h_out[0]}")
    print(f"Expected: {expected}")

    ok = (int(h_out[0]) == int(expected))
    print("✅ PASSED" if ok else "❌ FAILED")
    return ok


def test_reduce_min():
    print("\n=== Test 2: Min Reduction ===")
    from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind

    h_in = np.array([5, 2, 8, 1, 9], dtype=np.int32)
    h_out = np.zeros(1, dtype=np.int32)

    d_in = MacaArray(len(h_in), np.int32)
    d_out = MacaArray(1, np.int32)

    d_in.copy_to_device(h_in)

    reduce(d_in, d_out, op=OpKind.MIN, init=0)

    d_out.copy_to_host(h_out)

    expected = min(0, *[int(x) for x in h_in])
    print(f"Input: {h_in}")
    print(f"Init value: 0")
    print(f"Result: {h_out[0]}")
    print(f"Expected: {expected}")

    ok = (int(h_out[0]) == int(expected))
    print("✅ PASSED" if ok else "❌ FAILED")
    return ok


def test_reduce_max():
    print("\n=== Test 3: Max Reduction ===")
    from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind

    h_in = np.array([5, 2, 8, 1, 9], dtype=np.int32)
    h_out = np.zeros(1, dtype=np.int32)

    d_in = MacaArray(len(h_in), np.int32)
    d_out = MacaArray(1, np.int32)

    d_in.copy_to_device(h_in)

    reduce(d_in, d_out, op=OpKind.MAX, init=100)

    d_out.copy_to_host(h_out)

    expected = max(100, *[int(x) for x in h_in])
    print(f"Input: {h_in}")
    print(f"Init value: 100")
    print(f"Result: {h_out[0]}")
    print(f"Expected: {expected}")

    ok = (int(h_out[0]) == int(expected))
    print("✅ PASSED" if ok else "❌ FAILED")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("CCCL Reduce Tests (Simplified, No Numba)")
    print("=" * 60)

    results = [test_reduce_sum(), test_reduce_min(), test_reduce_max()]

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    sys.exit(0 if all(results) else 1)
