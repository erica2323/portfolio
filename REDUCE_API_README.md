# MACA CCCL Reduce API

This directory contains the implementation of the MACA CCCL Reduce API, following the same pattern as the Transform API.

## Files Structure

```
cccl/
├── c/
│   ├── types_official.h       # Common type definitions
│   ├── reduce_official.h      # Reduce API header
│   └── reduce_official.cu     # Reduce API implementation
test_reduce_official.cpp       # Test program
build_reduce.sh               # Build script
```

## API Overview

The Reduce API provides a C interface to mcCub's `cub::DeviceReduce` functionality.

### Key Functions

1. **cccl_reduce_build()** - Build/prepare a reduce operation
   ```c
   mcError_t cccl_reduce_build(
       cccl_reduce_build_result_t* build,
       cccl_type_info in_type,
       cccl_type_info out_type,
       cccl_op_t op
   );
   ```

2. **cccl_reduce_dispatch()** - Execute the reduce operation
   ```c
   mcError_t cccl_reduce_dispatch(
       cccl_reduce_build_result_t& build,
       const void* d_in,
       void* d_out,
       size_t num_items,
       const void* init,
       cccl_type_info in_type,
       cccl_op_t op,
       mcStream_t stream
   );
   ```

3. **cccl_reduce_cleanup()** - Clean up resources
   ```c
   mcError_t cccl_reduce_cleanup(
       cccl_reduce_build_result_t build
   );
   ```

### Supported Operations

- **CCCL_PLUS** - Sum reduction
- **CCCL_MULTIPLIES** - Product reduction
- **CCCL_MAXIMUM** - Maximum reduction
- **CCCL_MINIMUM** - Minimum reduction
- **CCCL_STATELESS** - User-defined reduction (requires NVRTC compilation)

### Supported Types

- INT8, INT16, INT32, INT64
- UINT8, UINT16, UINT32, UINT64
- FLOAT32, FLOAT64

## Test Program

The `test_reduce_official.cpp` program includes three tests:

1. **Test 1: SUM** - Reduces an array [0, 1, 2, ..., 15] to get sum = 120
2. **Test 2: MAX** - Finds the maximum value in a shuffled array
3. **Test 3: MIN** - Finds the minimum value in a shuffled array

## Building

### Prerequisites
- MACA SDK installed (typically in `/opt/maca`)
- `mcc` compiler available in PATH

### Compilation
```bash
chmod +x build_reduce.sh
./build_reduce.sh
```

Or manually:
```bash
mcc -I. -I/opt/maca/include \
    test_reduce_official.cpp \
    cccl/c/reduce_official.cu \
    -o test_reduce_official \
    -lmcrt
```

### Running
```bash
./test_reduce_official
```

Expected output:
```
========================================
MACA CCCL Reduce Test Suite
========================================

Test 1: Built-in SUM operation
========================================
Input: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Operation: SUM(array)
✅ SUM PASSED
   Result: 120

Test 2: Built-in MAX operation
========================================
Input: [15, 3, 7, 1, 9, 12, 4, 8, 2, 14, 6, 11, 5, 10, 13, 0]
Operation: MAX(array)
✅ MAX PASSED
   Result: 15

Test 3: Built-in MIN operation
========================================
Input: [15, 3, 7, 1, 9, 12, 4, 8, 2, 14, 6, 11, 5, 10, 13, 0]
Operation: MIN(array)
✅ MIN PASSED
   Result: 0

========================================
Test Summary
========================================
Passed: 3/3
========================================
```

## Implementation Details

### Architecture

The Reduce API follows the same pattern as the Transform API:

1. **Type-safe interface**: Uses `cccl_type_info` to specify input/output types
2. **Operation dispatch**: Supports both built-in and user-defined operations
3. **Template dispatch**: Internally uses C++ templates for type-specific implementations
4. **Resource management**: Handles temporary storage allocation/deallocation

### Internal Flow

```
cccl_reduce_build()
    ↓
cccl_reduce_dispatch()
    ↓
cccl_reduce_typed<T>()
    ↓
reduce_with_builtin_op<T>()
    ↓
reduce_impl<T, OpType>()
    ↓
cub::DeviceReduce::Reduce()
```

### Memory Management

The API automatically manages temporary storage required by CUB:
1. First call determines required temp storage size
2. Allocates device memory for temp storage
3. Executes reduce with temp storage
4. Stores temp storage pointer for cleanup
5. `cccl_reduce_cleanup()` frees the temp storage

## Future Enhancements

- [ ] Support for user-defined reduction operations (NVRTC compilation)
- [ ] Support for initial values in all operations
- [ ] Additional reduction operations (AND, OR, XOR, etc.)
- [ ] Performance benchmarking
- [ ] Python bindings (similar to transform API)

## Relationship to Transform API

Both APIs share:
- Common type system (`types_official.h`)
- Similar build/execute/cleanup pattern
- Operation dispatch mechanism
- Support for user-defined operations

Key differences:
- Transform: One-to-one mapping (N inputs → N outputs)
- Reduce: Many-to-one mapping (N inputs → 1 output)

## Testing with mcRTC

Similar to the transform API, you can verify mcCub's reduce API is available:

```bash
./test_mccub_api  # Tests DeviceTransform
```

For reduce-specific testing, you would create a similar program testing `cub::DeviceReduce::Reduce()`.
