#!/usr/bin/env python3
"""
MACA CCCL Python Bindings - Reduce Example

This example demonstrates how to use the MACA CCCL Python bindings
to perform device-wide reduction operations on GPU.

Note: This requires MACA device memory management APIs.
      Adapt the memory allocation/copy parts to your MACA Python bindings.
"""

import numpy as np
import time

# Import MACA CCCL
try:
    from maca.compute import reduce_into, make_reduce_into
    from maca.compute.types import OpKind
    print("✅ MACA CCCL imported successfully")
except ImportError as e:
    print(f"❌ Error importing MACA CCCL: {e}")
    print("   Make sure to build and install: ./build.sh develop")
    exit(1)

# TODO: Replace with your MACA Python bindings
# This is a placeholder to show the expected interface
class MockMacaArray:
    """Mock MACA device array for demonstration"""
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        self._data = np.zeros(size, dtype=dtype)
        self._ptr = self._data.ctypes.data

    def copy_from_host(self, host_array):
        self._data[:] = host_array

    def copy_to_host(self):
        return self._data.copy()

    @property
    def __cuda_array_interface__(self):
        return {
            'data': (self._ptr, False),
            'shape': (self.size,),
            'typestr': self._data.dtype.str,
            'version': 3
        }


def example_basic_reduction():
    """Example 1: Basic reduction (single call)"""
    print("\n" + "="*60)
    print("Example 1: Basic Reduction")
    print("="*60)

    # Create test data
    n = 1000000
    print(f"Creating array with {n:,} float32 values...")
    h_input = np.arange(n, dtype=np.float32)
    expected_sum = h_input.sum()

    # Allocate device memory (replace with your MACA API)
    print("Allocating device memory...")
    d_input = MockMacaArray(n, dtype=np.float32)
    d_output = MockMacaArray(1, dtype=np.float32)

    # Copy to device
    print("Copying data to device...")
    d_input.copy_from_host(h_input)

    # Perform reduction
    print("Performing reduction on GPU...")
    start = time.time()
    reduce_into(d_input, d_output, op="sum")
    elapsed = time.time() - start

    # Get result
    h_output = d_output.copy_to_host()
    result = h_output[0]

    # Verify
    print(f"\n✅ Results:")
    print(f"   Expected: {expected_sum:.2f}")
    print(f"   Got:      {result:.2f}")
    print(f"   Match:    {np.isclose(result, expected_sum)}")
    print(f"   Time:     {elapsed*1000:.3f} ms")


def example_reusable_reduction():
    """Example 2: Reusable reduction (compile once, execute many times)"""
    print("\n" + "="*60)
    print("Example 2: Reusable Reduction (More Efficient)")
    print("="*60)

    n = 1000000
    num_iterations = 10

    # Allocate device memory
    d_input = MockMacaArray(n, dtype=np.float32)
    d_output = MockMacaArray(1, dtype=np.float32)

    # Build phase: Compile once
    print(f"\n📦 Build phase: Compiling reduction operation...")
    build_start = time.time()
    reduce_op = make_reduce_into(d_input, op="sum")
    build_time = time.time() - build_start
    print(f"   Build time: {build_time*1000:.3f} ms")

    # Execute phase: Run multiple times
    print(f"\n🚀 Execute phase: Running {num_iterations} reductions...")
    execution_times = []

    for i in range(num_iterations):
        # Create new data
        h_input = np.random.randn(n).astype(np.float32)
        expected = h_input.sum()

        # Copy to device
        d_input.copy_from_host(h_input)

        # Execute (using compiled operation)
        start = time.time()
        reduce_op(d_input, d_output)
        elapsed = time.time() - start
        execution_times.append(elapsed)

        # Verify
        result = d_output.copy_to_host()[0]
        match = np.isclose(result, expected, rtol=1e-5)

        if i == 0:
            print(f"   Iteration {i+1}: {elapsed*1000:.3f} ms (match: {match})")

    avg_time = np.mean(execution_times) * 1000
    print(f"   ...")
    print(f"   Average execution time: {avg_time:.3f} ms")
    print(f"\n💡 Benefit: Compilation done once, execution is fast!")


def example_different_operations():
    """Example 3: Different reduction operations"""
    print("\n" + "="*60)
    print("Example 3: Different Operations (sum, min, max)")
    print("="*60)

    n = 1000000
    h_input = np.random.randn(n).astype(np.float32)

    # Allocate device memory
    d_input = MockMacaArray(n, dtype=np.float32)
    d_output = MockMacaArray(1, dtype=np.float32)
    d_input.copy_from_host(h_input)

    operations = [
        ("sum", h_input.sum()),
        ("min", h_input.min()),
        ("max", h_input.max()),
    ]

    print(f"\nProcessing {n:,} values...")
    for op_name, expected in operations:
        # Perform reduction
        reduce_into(d_input, d_output, op=op_name)
        result = d_output.copy_to_host()[0]
        match = np.isclose(result, expected, rtol=1e-5)

        print(f"   {op_name.upper():5s}: {result:12.4f} (expected: {expected:12.4f}, match: {match})")


def example_different_types():
    """Example 4: Different data types"""
    print("\n" + "="*60)
    print("Example 4: Different Data Types")
    print("="*60)

    n = 100000
    dtypes = [
        (np.int32, "int32"),
        (np.int64, "int64"),
        (np.float32, "float32"),
        (np.float64, "float64"),
    ]

    print(f"\nReducing {n:,} values of different types...")
    for dtype, name in dtypes:
        # Create data
        h_input = np.arange(n, dtype=dtype)
        expected = h_input.sum()

        # Allocate device memory
        d_input = MockMacaArray(n, dtype=dtype)
        d_output = MockMacaArray(1, dtype=dtype)
        d_input.copy_from_host(h_input)

        # Perform reduction
        start = time.time()
        reduce_into(d_input, d_output, op="sum")
        elapsed = time.time() - start

        result = d_output.copy_to_host()[0]
        match = np.isclose(float(result), float(expected), rtol=1e-5)

        print(f"   {name:8s}: sum={result:15.0f} (match: {match}, time: {elapsed*1000:.3f} ms)")


def example_performance_comparison():
    """Example 5: Performance comparison vs NumPy"""
    print("\n" + "="*60)
    print("Example 5: Performance Comparison")
    print("="*60)

    sizes = [1000, 10000, 100000, 1000000, 10000000]

    print("\n{:>12s} {:>15s} {:>15s} {:>10s}".format(
        "Size", "MACA (ms)", "NumPy (ms)", "Speedup"
    ))
    print("-" * 60)

    for n in sizes:
        # Create data
        h_input = np.random.randn(n).astype(np.float32)

        # NumPy version
        np_times = []
        for _ in range(10):
            start = time.time()
            _ = h_input.sum()
            np_times.append(time.time() - start)
        np_time = np.mean(np_times) * 1000

        # MACA version (mock - in reality would be faster on GPU)
        d_input = MockMacaArray(n, dtype=np.float32)
        d_output = MockMacaArray(1, dtype=np.float32)
        d_input.copy_from_host(h_input)

        maca_times = []
        for _ in range(10):
            start = time.time()
            reduce_into(d_input, d_output, op="sum")
            maca_times.append(time.time() - start)
        maca_time = np.mean(maca_times) * 1000

        speedup = np_time / maca_time if maca_time > 0 else 0

        print("{:>12,d} {:>15.3f} {:>15.3f} {:>10.2f}x".format(
            n, maca_time, np_time, speedup
        ))

    print("\n💡 Note: This uses mock arrays. Real GPU would show better speedup!")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("  MACA CCCL Python Bindings - Examples")
    print("="*60)

    examples = [
        ("Basic Reduction", example_basic_reduction),
        ("Reusable Reduction", example_reusable_reduction),
        ("Different Operations", example_different_operations),
        ("Different Types", example_different_types),
        ("Performance Comparison", example_performance_comparison),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("  All Examples Complete!")
    print("="*60)
    print("\n💡 Tips:")
    print("   • Use make_reduce_into() to compile once, execute many times")
    print("   • Reusable operations are cached automatically")
    print("   • Support for int32, int64, float32, float64")
    print("   • Operations: sum, min, max")


if __name__ == "__main__":
    main()
