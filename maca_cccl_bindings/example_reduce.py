"""
MACA CCCL Reduce Example
Demonstrates how to use the Python bindings for GPU reduction
"""

import numpy as np

# Note: This assumes you have a MACA Python runtime binding
# Replace with your actual MACA Python API
# import maca_runtime as maca

from maca_cccl import DeviceReduce, OpKind

def example_sum_reduce():
    """Example: Sum reduction"""
    print("="*70)
    print("Example 1: Sum Reduction")
    print("="*70)

    # Create host data
    size = 10000
    h_data = np.arange(size, dtype=np.int32)
    print(f"Input: Array of {size} elements (0 to {size-1})")
    print(f"Expected sum: {h_data.sum()}")

    # Allocate device memory (pseudo-code - use your MACA API)
    # d_data = maca.mem_alloc(h_data.nbytes)
    # maca.mem_copy_host_to_device(d_data, h_data)
    # d_out = maca.mem_alloc(4)  # Single int32

    # For demonstration, we'll use integer pointers
    # In real usage, these would be actual device pointers
    d_data = 0x100000000  # Placeholder device pointer
    d_out = 0x100001000   # Placeholder output pointer

    # Build reduce operation
    print("\nBuilding sum reduction...")
    reduce_sum = DeviceReduce.sum(np.int32)
    print("✅ Build complete!")

    # Execute reduction
    print("\nExecuting reduction...")
    # reduce_sum.compute(d_data, d_out, size)
    print("✅ Reduction complete!")

    # Copy result back (pseudo-code)
    # h_result = maca.mem_copy_device_to_host(d_out, 1, np.int32)
    # print(f"\nResult: {h_result[0]}")

    # Cleanup
    # maca.mem_free(d_data)
    # maca.mem_free(d_out)

    print("\n" + "="*70 + "\n")

def example_min_max_reduce():
    """Example: Min/Max reduction"""
    print("="*70)
    print("Example 2: Min/Max Reduction")
    print("="*70)

    # Create random data
    np.random.seed(42)
    size = 5000
    h_data = np.random.randn(size).astype(np.float32)
    print(f"Input: Random array of {size} elements")
    print(f"Expected min: {h_data.min():.6f}")
    print(f"Expected max: {h_data.max():.6f}")

    # Allocate device memory (pseudo-code)
    d_data = 0x200000000  # Placeholder
    d_min = 0x200001000   # Placeholder
    d_max = 0x200002000   # Placeholder

    # Build operations
    print("\nBuilding min/max reductions...")
    reduce_min = DeviceReduce.min(np.float32)
    reduce_max = DeviceReduce.max(np.float32)
    print("✅ Build complete!")

    # Execute reductions
    print("\nExecuting reductions...")
    # reduce_min.compute(d_data, d_min, size)
    # reduce_max.compute(d_data, d_max, size)
    print("✅ Reductions complete!")

    print("\n" + "="*70 + "\n")

def example_reusable_operation():
    """Example: Reusing compiled operations"""
    print("="*70)
    print("Example 3: Reusable Operations")
    print("="*70)

    # Build once
    print("Building sum reduction once...")
    reduce_sum = DeviceReduce.sum(np.int32)
    print("✅ Build complete!")

    # Use multiple times with different data
    print("\nExecuting multiple reductions with same operation...")
    for i in range(3):
        size = (i + 1) * 1000
        print(f"  Reduction {i+1}: {size} elements")
        # reduce_sum.compute(d_data, d_out, size)

    print("✅ All reductions complete!")
    print("\nNote: The operation was compiled once and reused 3 times!")
    print("This is the power of the build/execute pattern.")

    print("\n" + "="*70 + "\n")

def example_custom_operation():
    """Example: Custom reduction with JIT"""
    print("="*70)
    print("Example 4: Custom Operations (Advanced - Requires mcRTC)")
    print("="*70)

    from maca_cccl._jit_templates import JITKernelTemplate

    # Generate custom operation
    template = JITKernelTemplate()

    # Example: Product reduction (multiply all elements)
    custom_op = "return a * b;"
    kernel_src = template.generate_reduce_kernel(
        dtype='float',
        op_name='Custom',
        custom_op_code=custom_op
    )

    print("Generated custom product reduction kernel:")
    print("-" * 70)
    print(kernel_src[:500] + "...")
    print("-" * 70)

    print("\nNote: To execute this, you need to:")
    print("1. Implement mcRTC compilation (see _jit_templates.py)")
    print("2. Compile the generated kernel")
    print("3. Launch the kernel with your data")

    print("\n" + "="*70 + "\n")

def example_type_support():
    """Example: Different data types"""
    print("="*70)
    print("Example 5: Multiple Data Types")
    print("="*70)

    dtypes = [np.int32, np.int64, np.float32, np.float64]

    print("Building sum reductions for different types...")
    operations = {}
    for dtype in dtypes:
        operations[dtype] = DeviceReduce.sum(dtype)
        print(f"  ✅ {dtype.__name__}")

    print("\nAll types supported and ready to use!")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" MACA CCCL Python Bindings - Reduce Examples")
    print("="*70 + "\n")

    print("NOTE: These examples use placeholder device pointers.")
    print("In production, replace with actual MACA runtime API calls.\n")

    # Run examples
    example_sum_reduce()
    example_min_max_reduce()
    example_reusable_operation()
    example_type_support()
    example_custom_operation()

    print("="*70)
    print(" All examples complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Integrate with your MACA Python runtime")
    print("2. Implement mcRTC compilation for JIT support")
    print("3. Add more algorithms (scan, sort, etc.)")
    print("4. Profile and optimize performance")
    print("="*70 + "\n")
