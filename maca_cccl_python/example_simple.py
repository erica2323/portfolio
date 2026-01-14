#!/usr/bin/env python3
"""
Simple example of using MACA CCCL Python binding
"""

import numpy as np
from maca_cccl import plus, multiplies, custom

print("=" * 60)
print("MACA CCCL Python Binding - Simple Examples")
print("=" * 60)

# Example 1: Add constant
print("\n📌 Example 1: Add 10 to all elements")
data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
print(f"Input:  {data}")

result = plus(data, 10)
print(f"Output: {result}")

# Example 2: Multiply by constant
print("\n📌 Example 2: Multiply all elements by 3")
data = np.array([10, 20, 30, 40], dtype=np.int32)
print(f"Input:  {data}")

result = multiplies(data, 3)
print(f"Output: {result}")

# Example 3: Custom operation (square)
print("\n📌 Example 3: Square each element")
data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
print(f"Input:  {data}")

code = """
__device__ int32_t user_op(int32_t x) {
    return x * x;
}
"""

result = custom(data, code)
print(f"Output: {result}")

# Example 4: Larger array
print("\n📌 Example 4: Process 1000 elements")
data = np.arange(1000, dtype=np.int32)
print(f"Input:  [0, 1, 2, ..., 999] (1000 elements)")

result = plus(data, 5)
print(f"Output: [5, 6, 7, ..., 1004] (1000 elements)")
print(f"First 10: {result[:10]}")
print(f"Last 10:  {result[-10:]}")

# Example 5: Float operations
print("\n📌 Example 5: Float operations")
data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
print(f"Input:  {data}")

result = multiplies(data, 2.0)
print(f"Output: {result}")

print("\n" + "=" * 60)
print("✅ All examples completed successfully!")
print("=" * 60)
