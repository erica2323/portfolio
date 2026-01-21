"""
Phase B Test: Python wrapper layer for MACA CCCL reduce.

This test verifies that the Python API works correctly, matching
NVIDIA CCCL's test_reduce.py Level 1 tests (basic pointer iterators).
"""

import numpy as np
import operator
import sys


def test_basic_sum_int32():
    """Test basic sum reduction with int32."""
    print("Test 1: Basic sum (int32, 1000 elements, init=10)")

    try:
        from cuda.cccl.parallel.experimental import reduce_into
    except ImportError as e:
        print(f"SKIP: Cannot import reduce_into: {e}")
        print("You need to compile the Cython bindings first:")
        print("  python setup.py build_ext --inplace")
        return False

    # Create test data on host
    h_in = np.ones(1000, dtype=np.int32)
    h_out = np.zeros(1, dtype=np.int32)

    # TODO: In production, these would be GPU arrays
    # For now, we need to manually transfer to device
    print("  Creating input array: 1000 ones (int32)")
    print("  Expected result: 1000 + 10 = 1010")

    try:
        # Call reduce_into
        reduce_into(h_in, h_out, operator.add, h_init=10)

        # Check result
        result = h_out[0]
        expected = 1010

        print(f"  Result: {result}")
        if result == expected:
            print("  ✓ PASS")
            return True
        else:
            print(f"  ✗ FAIL: Expected {expected}, got {result}")
            return False

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_operators():
    """Test different reduction operators."""
    print("\nTest 2: Different operators")

    try:
        from cuda.cccl.parallel.experimental import reduce_into
    except ImportError:
        print("SKIP: reduce_into not available")
        return False

    tests = [
        ('add', operator.add, np.ones(100, dtype=np.int32), 0, 100),
        ('min', min, np.arange(100, dtype=np.int32), 1000, 0),
        ('max', max, np.arange(100, dtype=np.int32), -1000, 99),
    ]

    all_passed = True
    for name, op, h_in, init, expected in tests:
        h_out = np.zeros(1, dtype=np.int32)

        try:
            reduce_into(h_in, h_out, op, h_init=init)
            result = h_out[0]

            if result == expected:
                print(f"  {name}: ✓ PASS (result={result})")
            else:
                print(f"  {name}: ✗ FAIL (expected={expected}, got={result})")
                all_passed = False

        except Exception as e:
            print(f"  {name}: ✗ ERROR: {e}")
            all_passed = False

    return all_passed


def test_different_dtypes():
    """Test different data types."""
    print("\nTest 3: Different data types")

    try:
        from cuda.cccl.parallel.experimental import reduce_into
    except ImportError:
        print("SKIP: reduce_into not available")
        return False

    dtypes = [
        (np.int32, 1000, 1000),
        (np.int64, 1000, 1000),
        (np.float32, 1000.0, 1000.0),
        (np.float64, 1000.0, 1000.0),
    ]

    all_passed = True
    for dtype, expected, init in dtypes:
        h_in = np.ones(1000, dtype=dtype)
        h_out = np.zeros(1, dtype=dtype)

        try:
            reduce_into(h_in, h_out, operator.add, h_init=dtype(0))
            result = h_out[0]

            if abs(result - expected) < 1e-5:
                print(f"  {dtype.__name__}: ✓ PASS (result={result})")
            else:
                print(f"  {dtype.__name__}: ✗ FAIL (expected={expected}, got={result})")
                all_passed = False

        except Exception as e:
            print(f"  {dtype.__name__}: ✗ ERROR: {e}")
            all_passed = False

    return all_passed


def test_string_operators():
    """Test using string operator names."""
    print("\nTest 4: String operator names")

    try:
        from cuda.cccl.parallel.experimental import reduce_into
    except ImportError:
        print("SKIP: reduce_into not available")
        return False

    tests = [
        ('add', np.ones(100, dtype=np.int32), 0, 100),
        ('+', np.ones(100, dtype=np.int32), 0, 100),
        ('min', np.arange(100, dtype=np.int32), 1000, 0),
        ('max', np.arange(100, dtype=np.int32), -1000, 99),
    ]

    all_passed = True
    for op_str, h_in, init, expected in tests:
        h_out = np.zeros(1, dtype=np.int32)

        try:
            reduce_into(h_in, h_out, op_str, h_init=init)
            result = h_out[0]

            if result == expected:
                print(f"  '{op_str}': ✓ PASS (result={result})")
            else:
                print(f"  '{op_str}': ✗ FAIL (expected={expected}, got={result})")
                all_passed = False

        except Exception as e:
            print(f"  '{op_str}': ✗ ERROR: {e}")
            all_passed = False

    return all_passed


def main():
    print("=" * 60)
    print("Phase B Test Suite: Python Wrapper Layer")
    print("=" * 60)

    results = []
    results.append(("Basic sum", test_basic_sum_int32()))
    results.append(("Different operators", test_different_operators()))
    results.append(("Different dtypes", test_different_dtypes()))
    results.append(("String operators", test_string_operators()))

    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")

    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All Phase B tests passed!")
        print("\nNext steps:")
        print("  - Phase C: Add NVRTC JIT support for Python operators")
        print("  - Phase D: Add advanced iterator support")
        print("  - Phase E: Add complex types and streams")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
