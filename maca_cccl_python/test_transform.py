#!/usr/bin/env python3
"""
Test script for MACA CCCL Transform Python binding
Tests the same 3 operations as the C++ tests
"""

import numpy as np
import sys
from maca_cccl import Transform, CCCLOpKind, plus, multiplies, custom


def print_test_header(test_num, test_name):
    """Print test header"""
    print("\n" + "=" * 60)
    print(f"Test {test_num}: {test_name}")
    print("=" * 60)


def check_results(output, expected, test_name):
    """Check if output matches expected results"""
    if np.allclose(output, expected):
        print(f"✅ {test_name} PASSED")
        return True
    else:
        print(f"❌ {test_name} FAILED")
        print(f"Expected: {expected}")
        print(f"Got:      {output}")
        return False


def test1_builtin_plus():
    """Test 1: Built-in PLUS operation (add constant)"""
    print_test_header(1, "Built-in PLUS operation")

    # Input data
    input_data = np.arange(16, dtype=np.int32)
    addend = 5

    print(f"Input:  {input_data}")
    print(f"Addend: {addend}")

    # Create transform with PLUS operation
    transform = Transform(CCCLOpKind.PLUS, state=addend)

    # Execute
    output = transform(input_data)

    # Expected result
    expected = input_data + addend

    print(f"Output:   {output}")
    print(f"Expected: {expected}")

    return check_results(output, expected, "PLUS")


def test2_user_code_square():
    """Test 2: User-provided C++ code (square operation)"""
    print_test_header(2, "User C++ code (square)")

    # Input data
    input_data = np.arange(16, dtype=np.int32)

    print(f"Input: {input_data}")

    # User-defined C++ code
    user_code = """
__device__ int32_t user_op(int32_t x) {
    return x * x;
}
"""

    print(f"Operation: x * x")

    # Create transform with user code
    transform = Transform(
        CCCLOpKind.STATELESS,
        in_dtype=np.int32,
        out_dtype=np.int32,
        user_code=user_code
    )

    # Execute
    output = transform(input_data)

    # Expected result
    expected = input_data * input_data

    print(f"Output:   {output}")
    print(f"Expected: {expected}")

    return check_results(output, expected, "User code (square)")


def test3_builtin_multiplies():
    """Test 3: Built-in MULTIPLIES operation (multiply by constant)"""
    print_test_header(3, "Built-in MULTIPLIES operation")

    # Input data
    input_data = np.arange(16, dtype=np.int32)
    multiplier = 3

    print(f"Input:      {input_data}")
    print(f"Multiplier: {multiplier}")

    # Create transform with MULTIPLIES operation
    transform = Transform(CCCLOpKind.MULTIPLIES, state=multiplier)

    # Execute
    output = transform(input_data)

    # Expected result
    expected = input_data * multiplier

    print(f"Output:   {output}")
    print(f"Expected: {expected}")

    return check_results(output, expected, "MULTIPLIES")


def test4_convenience_plus():
    """Test 4: Convenience function - plus()"""
    print_test_header(4, "Convenience function: plus()")

    input_data = np.array([10, 20, 30, 40], dtype=np.int32)
    addend = 7

    print(f"Input:  {input_data}")
    print(f"Addend: {addend}")

    # Use convenience function
    output = plus(input_data, addend)

    expected = input_data + addend

    print(f"Output:   {output}")
    print(f"Expected: {expected}")

    return check_results(output, expected, "plus() convenience function")


def test5_convenience_multiplies():
    """Test 5: Convenience function - multiplies()"""
    print_test_header(5, "Convenience function: multiplies()")

    input_data = np.array([5, 10, 15, 20], dtype=np.int32)
    multiplier = 4

    print(f"Input:      {input_data}")
    print(f"Multiplier: {multiplier}")

    # Use convenience function
    output = multiplies(input_data, multiplier)

    expected = input_data * multiplier

    print(f"Output:   {output}")
    print(f"Expected: {expected}")

    return check_results(output, expected, "multiplies() convenience function")


def test6_convenience_custom():
    """Test 6: Convenience function - custom()"""
    print_test_header(6, "Convenience function: custom()")

    input_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)

    print(f"Input: {input_data}")

    # Custom operation: x * 2 + 1
    code = """
__device__ int32_t user_op(int32_t x) {
    return x * 2 + 1;
}
"""

    print("Operation: x * 2 + 1")

    # Use convenience function
    output = custom(input_data, code)

    expected = input_data * 2 + 1

    print(f"Output:   {output}")
    print(f"Expected: {expected}")

    return check_results(output, expected, "custom() convenience function")


def test7_float_operations():
    """Test 7: Float operations"""
    print_test_header(7, "Float operations")

    input_data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    multiplier = 2.0

    print(f"Input:      {input_data}")
    print(f"Multiplier: {multiplier}")

    # Create transform with MULTIPLIES for float32
    transform = Transform(
        CCCLOpKind.MULTIPLIES,
        in_dtype=np.float32,
        out_dtype=np.float32,
        state=multiplier
    )

    output = transform(input_data)
    expected = input_data * multiplier

    print(f"Output:   {output}")
    print(f"Expected: {expected}")

    return check_results(output, expected, "Float MULTIPLIES")


def main():
    """Run all tests"""
    print("=" * 60)
    print("MACA CCCL Transform Python Binding - Test Suite")
    print("=" * 60)

    # Core tests (matching C++ tests)
    results = []
    results.append(("Test 1: PLUS", test1_builtin_plus()))
    results.append(("Test 2: User code", test2_user_code_square()))
    results.append(("Test 3: MULTIPLIES", test3_builtin_multiplies()))

    # Additional tests (Python convenience functions)
    results.append(("Test 4: plus()", test4_convenience_plus()))
    results.append(("Test 5: multiplies()", test5_convenience_multiplies()))
    results.append(("Test 6: custom()", test6_convenience_custom()))
    results.append(("Test 7: Float ops", test7_float_operations()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
