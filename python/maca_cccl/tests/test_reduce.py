"""
Test suite for MACA CCCL Python bindings - Reduce operations

This demonstrates how to use the Python bindings with MACA device memory.
"""

import numpy as np
import pytest

# These would be your MACA Python bindings for device memory
# For now, we'll show the interface pattern
try:
    import maca  # Hypothetical MACA Python module
    HAS_MACA = True
except ImportError:
    HAS_MACA = False

try:
    from maca.compute import reduce_into, make_reduce_into
    from maca.compute.types import OpKind
    HAS_MACA_CCCL = True
except ImportError:
    HAS_MACA_CCCL = False


@pytest.mark.skipif(not (HAS_MACA and HAS_MACA_CCCL), reason="MACA not available")
class TestReduceBasic:
    """Basic reduction tests"""

    def test_sum_reduce(self):
        """Test sum reduction"""
        # Create input data
        n = 1000
        h_input = np.arange(n, dtype=np.float32)
        expected_sum = h_input.sum()

        # Allocate device memory
        d_input = maca.device_array(n, dtype=np.float32)
        d_output = maca.device_array(1, dtype=np.float32)

        # Copy to device
        d_input.copy_from_host(h_input)

        # Perform reduction
        reduce_into(d_input, d_output, op="sum")

        # Copy result back
        h_output = d_output.copy_to_host()

        # Verify
        assert np.isclose(h_output[0], expected_sum, rtol=1e-5)

    def test_min_reduce(self):
        """Test minimum reduction"""
        n = 1000
        h_input = np.random.randn(n).astype(np.float32)
        expected_min = h_input.min()

        d_input = maca.device_array(n, dtype=np.float32)
        d_output = maca.device_array(1, dtype=np.float32)
        d_input.copy_from_host(h_input)

        reduce_into(d_input, d_output, op="min")

        h_output = d_output.copy_to_host()
        assert np.isclose(h_output[0], expected_min, rtol=1e-5)

    def test_max_reduce(self):
        """Test maximum reduction"""
        n = 1000
        h_input = np.random.randn(n).astype(np.float32)
        expected_max = h_input.max()

        d_input = maca.device_array(n, dtype=np.float32)
        d_output = maca.device_array(1, dtype=np.float32)
        d_input.copy_from_host(h_input)

        reduce_into(d_input, d_output, op="max")

        h_output = d_output.copy_to_host()
        assert np.isclose(h_output[0], expected_max, rtol=1e-5)


@pytest.mark.skipif(not (HAS_MACA and HAS_MACA_CCCL), reason="MACA not available")
class TestReduceReusable:
    """Test reusable (compiled) reduce operations"""

    def test_reusable_reduce(self):
        """Test that we can compile once and execute multiple times"""
        n = 1000
        dtype = np.float32

        # Build phase: compile once
        d_sample = maca.device_array(n, dtype=dtype)
        reduce_op = make_reduce_into(d_sample, op="sum")

        # Execute phase: run multiple times
        for _ in range(5):
            h_input = np.random.randn(n).astype(dtype)
            expected_sum = h_input.sum()

            d_input = maca.device_array(n, dtype=dtype)
            d_output = maca.device_array(1, dtype=dtype)
            d_input.copy_from_host(h_input)

            # Use the compiled operation
            reduce_op(d_input, d_output)

            h_output = d_output.copy_to_host()
            assert np.isclose(h_output[0], expected_sum, rtol=1e-5)


@pytest.mark.skipif(not (HAS_MACA and HAS_MACA_CCCL), reason="MACA not available")
class TestReduceTypes:
    """Test different data types"""

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    def test_different_types(self, dtype):
        """Test reduction with different data types"""
        n = 1000
        h_input = np.arange(n, dtype=dtype)
        expected_sum = h_input.sum()

        d_input = maca.device_array(n, dtype=dtype)
        d_output = maca.device_array(1, dtype=dtype)
        d_input.copy_from_host(h_input)

        reduce_into(d_input, d_output, op="sum")

        h_output = d_output.copy_to_host()
        assert np.isclose(h_output[0], expected_sum, rtol=1e-5)


# Standalone example (not a test)
def example_usage():
    """
    Example of using MACA CCCL Python bindings

    This shows the typical workflow for using the reduce operation.
    """
    print("MACA CCCL Python Bindings - Reduce Example")
    print("=" * 60)

    # Setup
    n = 1000000
    print(f"\nReducing {n} float32 values...")

    # Create test data
    h_input = np.arange(n, dtype=np.float32)
    expected_sum = h_input.sum()
    print(f"Expected sum: {expected_sum}")

    # Allocate device memory (pseudo-code)
    # d_input = maca.malloc(n * 4)  # 4 bytes per float32
    # d_output = maca.malloc(4)

    # Copy to device
    # maca.memcpy_host_to_device(d_input, h_input)

    # Method 1: Single-call interface
    print("\nMethod 1: Single-call interface")
    # reduce_into(d_input, d_output, op="sum")

    # Method 2: Compile once, execute many times (more efficient)
    print("\nMethod 2: Reusable compiled operation")
    # reduce_op = make_reduce_into(d_input, op="sum")
    # reduce_op(d_input, d_output)  # Execute
    # reduce_op(d_input2, d_output2)  # Execute again on different data

    # Copy result back
    # h_output = np.empty(1, dtype=np.float32)
    # maca.memcpy_device_to_host(h_output, d_output)

    print("\n✅ Reduction complete!")
    # print(f"Result: {h_output[0]}")
    # print(f"Match: {np.isclose(h_output[0], expected_sum)}")


if __name__ == "__main__":
    example_usage()
