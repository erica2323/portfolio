# MACA CCCL Python Bindings - Architecture Deep Dive

This document provides an in-depth look at how the Python bindings work, following the NVIDIA CCCL architecture pattern.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
3. [Data Flow](#data-flow)
4. [Memory Management](#memory-management)
5. [Type System](#type-system)
6. [Compilation Pipeline](#compilation-pipeline)
7. [Caching Strategy](#caching-strategy)
8. [Comparison with NVIDIA CCCL](#comparison-with-nvidia-cccl)

## Architecture Overview

MACA CCCL Python bindings follow a **5-layer architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Python User API (maca.compute.algorithms._reduce)  │
│  • High-level interface (reduce_into, make_reduce_into)     │
│  • NumPy-style API                                           │
│  • Automatic type detection                                  │
│  • Caching                                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 4: Python Wrappers (types.py, op.py)                  │
│  • TypeInfo: Type metadata management                        │
│  • Op: Operation wrappers                                    │
│  • Type conversion utilities                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 3: Cython Bindings (_bindings_impl.pyx)               │
│  • Python ↔ C type bridging                                 │
│  • GIL management (release during GPU work)                  │
│  • Memory safety (RAII patterns)                             │
│  • Cython classes wrapping C structs                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 2: C API (cccl/c/reduce_official.h)                   │
│  • Build/Execute separation                                  │
│  • Type system (cccl_type_info, cccl_type_enum)             │
│  • Operation system (cccl_op_t, cccl_op_kind_t)             │
│  • Iterator abstraction (cccl_iterator_t)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 1: mcCub Library (C++)                                │
│  • DispatchReduce<>::Dispatch()                              │
│  • Device-level parallel algorithms                          │
│  • GPU kernel execution                                      │
└─────────────────────────────────────────────────────────────┘
```

## Layer-by-Layer Breakdown

### Layer 5: Python User API

**File**: `maca/compute/algorithms/_reduce.py`

**Responsibilities**:
- Provide intuitive, NumPy-like API
- Handle array type detection
- Manage device pointer extraction
- Cache compiled operations
- Error handling and validation

**Key Classes**:

```python
class _Reduce:
    """Compiled reduction operation"""
    def __init__(self, op, d_in_typeinfo, initial_value):
        # Build phase: compile operation
        self.build_result = _bindings.device_reduce_build(...)

    def __call__(self, d_in, d_out, stream=None):
        # Execute phase: run on GPU
        _bindings.device_reduce(...)
```

**Key Functions**:

```python
def reduce_into(d_in, d_out, op="sum", ...):
    """Single-call interface: build + execute"""
    reduce_op = make_reduce_into(d_in, op, initial_value)
    reduce_op(d_in, d_out, stream)

@lru_cache(maxsize=128)
def _make_reduce_into_cached(op_kind, dtype_str, initial_value):
    """Cached compilation to avoid recompiling same types"""
    return _Reduce(op, typeinfo, initial_val)
```

### Layer 4: Python Wrappers

**Files**: `maca/compute/types.py`, `maca/compute/op.py`

**Responsibilities**:
- Convert between NumPy types and CCCL types
- Provide Pythonic operation wrappers
- Type validation

**TypeInfo Class**:

```python
class TypeInfo:
    """Maps NumPy dtype to CCCL type system"""
    def __init__(self, dtype):
        # np.float32 → (FLOAT32, size=4, alignment=4)
        self._type_enum = TypeEnum.FLOAT32
        self._size = 4
        self._alignment = 4

    def to_cython(self):
        """Convert to Cython TypeInfo object"""
        return _bindings.TypeInfo(self._type_enum, self._size, self._alignment)
```

**Op Class**:

```python
class Op:
    """Wraps CCCL operations"""
    def __init__(self, op_kind, name=None, ...):
        self._op_kind = op_kind  # OpKind.PLUS, etc.

    def to_cython(self):
        """Convert to Cython Op object"""
        return _bindings.Op(op_kind=self._op_kind, ...)
```

### Layer 3: Cython Bindings

**File**: `maca/compute/_bindings_impl.pyx`

**Responsibilities**:
- Bridge Python and C worlds
- Type conversion (Python scalars ↔ C types)
- GIL management
- Memory safety
- Expose C API to Python

**Key Components**:

```cython
# External C declarations
cdef extern from "cccl/c/reduce_official.h" nogil:
    ctypedef struct cccl_device_reduce_build_result_t:
        # ... C struct members

    mcError_t cccl_device_reduce_build_ex(...)
    mcError_t cccl_device_reduce_ex(...)

# Python wrapper classes
cdef class TypeInfo:
    cdef cccl_type_info _c_type  # C struct

    def __init__(self, type_enum, size, alignment):
        self._c_type.type = type_enum
        # ...

    cdef cccl_type_info get_c_type(self):
        return self._c_type

# Build function
def device_reduce_build(Op op, TypeInfo type_info, initial_value, ...):
    cdef cccl_device_reduce_build_result_t c_result
    cdef void* init_val_ptr = malloc(type_info.size)

    # Convert Python value to C
    _copy_scalar_to_ptr(initial_value, init_val_ptr, type_info.type)

    # Release GIL for C call
    with nogil:
        err = cccl_device_reduce_build_ex(&c_result, ...)

    # Wrap result in Python object
    result = DeviceReduceBuildResult()
    result.set_result(c_result)
    return result
```

**GIL Management**:

```cython
# Release GIL during GPU operations (Python can run in parallel)
with nogil:
    err = cccl_device_reduce_ex(...)
```

**Type Conversion**:

```cython
cdef void _copy_scalar_to_ptr(value, void* ptr, cccl_type_enum type_enum):
    """Copy Python scalar to C pointer"""
    if type_enum == CCCL_INT32:
        (<int32_t*>ptr)[0] = <int32_t>value
    elif type_enum == CCCL_FLOAT32:
        (<float*>ptr)[0] = <float>value
    # ...
```

### Layer 2: C API

**Files**: `parallel/include/cccl/c/types_official.h`, `parallel/include/cccl/c/reduce_official.h`, `parallel/src/reduce_official.cu`

**Responsibilities**:
- Define C interface to C++ implementation
- Type system
- Build/Execute separation
- Iterator abstraction

**Type System**:

```c
// Type enumeration
typedef enum cccl_type_enum {
    CCCL_INT32 = 2,
    CCCL_FLOAT32 = 9,
    // ...
} cccl_type_enum;

// Type information
typedef struct cccl_type_info {
    size_t size;
    size_t alignment;
    cccl_type_enum type;
} cccl_type_info;
```

**Operation System**:

```c
typedef enum cccl_op_kind_t {
    CCCL_PLUS = 2,
    CCCL_MINIMUM = 22,
    CCCL_MAXIMUM = 23,
    // ...
} cccl_op_kind_t;

typedef struct cccl_op_t {
    cccl_op_kind_t type;
    const char* name;
    // ...
} cccl_op_t;
```

**Build/Execute API**:

```c
// Build phase: Compile operation
mcError_t cccl_device_reduce_build_ex(
    cccl_device_reduce_build_result_t* build,
    cccl_op_t op,
    cccl_iterator_t d_in,
    void* initial_value,
    cccl_build_config* build_config
);

// Execute phase: Run on GPU
mcError_t cccl_device_reduce_ex(
    cccl_device_reduce_build_result_t build,
    cccl_iterator_t d_in,
    void* d_out,
    uint64_t num_items,
    mcStream_t stream
);
```

### Layer 1: mcCub Library

**Files**: mcCub headers (e.g., `mccub/device/dispatch/dispatch_reduce.cuh`)

**Responsibilities**:
- Optimized GPU algorithms
- Template-based C++ implementation
- Direct dispatch to GPU kernels

**Dispatch Pattern** (matches NVIDIA CUB exactly):

```cpp
template<typename T>
static mcError_t dispatch_reduce_internal(
    cccl_op_kind_t op_type,
    const T* d_in,
    T* d_out,
    int num_items,
    T init_value,
    mcStream_t stream
) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Phase 1: Query temp storage size
    DispatchReduce<const T*, T*, int, cub::Sum>::Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        cub::Sum(),
        T(),
        stream,
        false
    );

    // Allocate temp storage
    mcMalloc(&d_temp_storage, temp_storage_bytes);

    // Phase 2: Execute reduction
    DispatchReduce<const T*, T*, int, cub::Sum>::Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        cub::Sum(),
        T(),
        stream,
        false
    );

    mcFree(d_temp_storage);
    return mcSuccess;
}
```

## Data Flow

### Build Phase

```
Python:
  reduce_into(d_in, d_out, op="sum")
    ↓
  make_reduce_into(d_in, op="sum")
    ↓
  _make_reduce_into_cached(OpKind.PLUS, "float32", 0)
    ↓
  _Reduce.__init__(op, typeinfo, 0)
    ↓
Cython:
  _bindings.device_reduce_build(op_cython, typeinfo_cython, 0)
    ↓
  Convert types: Op → cccl_op_t, TypeInfo → cccl_type_info
    ↓
  Allocate init value: malloc(4 bytes)
    ↓
  Copy: 0 (Python) → init_val_ptr (C)
    ↓
  with nogil:
    ↓
C API:
  cccl_device_reduce_build_ex(&c_result, op, iter, init_ptr, NULL)
    ↓
  Create iterator: cccl_make_pointer_iterator(NULL, type)
    ↓
  Store configuration in build_result
    ↓
  return mcSuccess
    ↓
Cython:
  Wrap c_result in DeviceReduceBuildResult
    ↓
Python:
  return _Reduce object (with compiled build_result)
```

### Execute Phase

```
Python:
  reduce_op(d_in, d_out)  # _Reduce.__call__
    ↓
  Extract device pointers:
    d_in_ptr = d_in.__cuda_array_interface__['data'][0]
    d_out_ptr = d_out.__cuda_array_interface__['data'][0]
    ↓
  num_items = d_in.size
    ↓
Cython:
  _bindings.device_reduce(build_result, d_in_ptr, d_out_ptr, num_items, stream)
    ↓
  Create iterator with actual pointer:
    iter = cccl_make_pointer_iterator(<void*>d_in_ptr, type)
    ↓
  with nogil:
    ↓
C API:
  cccl_device_reduce_ex(build, iter, d_out_ptr, num_items, stream)
    ↓
  Extract pointer: void* d_in_ptr = iter.state
    ↓
  Type dispatch:
    switch (build.type.type) {
      case CCCL_FLOAT32:
        dispatch_reduce_internal<float>(...);
    }
    ↓
mcCub:
  dispatch_reduce_internal<float>(...)
    ↓
  DispatchReduce<...>::Dispatch(...)
    ↓
  [GPU Kernel Launch]
    ↓
  GPU reduces data, writes to d_out
    ↓
  return mcSuccess
    ↓
Python:
  Result now in d_out device memory
```

## Memory Management

### Host Side (Python/Cython)

**Initial Value**:
```cython
# Allocate
cdef void* init_val_ptr = malloc(type_info.size)

# Use
_copy_scalar_to_ptr(initial_value, init_val_ptr, type_info.type)

# Free (in cleanup)
if build->initial_value != nullptr:
    free(build->initial_value)
```

**Build Result**:
```cython
cdef class DeviceReduceBuildResult:
    cdef cccl_device_reduce_build_result_t _c_result
    cdef bint _owns_result

    def __dealloc__(self):
        if self._owns_result:
            cccl_device_reduce_cleanup(&self._c_result)
```

### Device Side (GPU)

**Temporary Storage**:
- Managed by mcCub internally
- Allocated during Phase 1 (query size)
- Freed after Phase 2 (execute)

**Input/Output Buffers**:
- Managed by user code
- Python bindings only use pointers
- No ownership transfer

## Type System

### Type Conversion Chain

```
Python         NumPy           Python           Cython          C
Value          dtype           TypeInfo         TypeInfo        cccl_type_info
─────          ─────           ────────         ────────        ──────────────
42.0     →     float32    →    TypeInfo    →    TypeInfo   →   {size:4,
                                (FLOAT32,        ._c_type        align:4,
                                 size:4,                         type:FLOAT32}
                                 align:4)
```

### Type Mapping Table

| NumPy dtype  | TypeEnum    | C type    | Size | Alignment |
|--------------|-------------|-----------|------|-----------|
| np.int32     | INT32       | int32_t   | 4    | 4         |
| np.int64     | INT64       | int64_t   | 8    | 8         |
| np.float32   | FLOAT32     | float     | 4    | 4         |
| np.float64   | FLOAT64     | double    | 8    | 8         |

## Compilation Pipeline

### Traditional Approach (NVIDIA CCCL)

```
Python
  ↓
Numba: Compile Python → LTOIR
  ↓
NVRTC: Compile LTOIR + CUB → PTX
  ↓
GPU: Execute
```

### MACA CCCL Approach (Current)

```
Python
  ↓
Direct mcCub Dispatch (C++)
  ↓
GPU: Execute
```

**Advantages**:
- Simpler: No NVRTC compilation step
- Faster build times: No runtime compilation
- Easier debugging: Standard C++ compilation

**Trade-offs**:
- Less flexible: Can't compile custom operations (yet)
- Only supports well-known ops (PLUS, MIN, MAX)

## Caching Strategy

### Level 1: Python LRU Cache

```python
@lru_cache(maxsize=128)
def _make_reduce_into_cached(op_kind, dtype_str, initial_value):
    """Cache by (operation, dtype, initial_value)"""
    return _Reduce(op, typeinfo, initial_val)
```

**Cache Key**: `(OpKind.PLUS, "float32", 0)`

**What's Cached**: Entire `_Reduce` object (includes build result)

**Lifetime**: Process lifetime (LRU eviction when cache full)

### Level 2: Build Result Cache (Implicit)

Build results are cached implicitly through Python object lifetime:

```python
# First call: builds operation
reduce_op = make_reduce_into(d_sample, "sum")  # Cache miss

# Subsequent calls with same signature: reuses cached object
reduce_op2 = make_reduce_into(d_sample2, "sum")  # Cache hit!
```

## Comparison with NVIDIA CCCL

### Similarities

| Feature | NVIDIA CCCL | MACA CCCL |
|---------|-------------|-----------|
| Architecture | 5-layer | 5-layer ✅ |
| Build/Execute | Separated | Separated ✅ |
| Type System | Rich, extensible | Rich, extensible ✅ |
| Iterator Abstraction | Yes | Yes ✅ |
| Cython Bindings | Yes | Yes ✅ |
| Caching | Yes | Yes ✅ |

### Differences

| Feature | NVIDIA CCCL | MACA CCCL |
|---------|-------------|-----------|
| Backend | CUB | mcCub |
| Runtime Compilation | NVRTC | Direct dispatch |
| Custom Operations | Full support | Not yet |
| Iterators | Full support | Pointer only |
| GPU | NVIDIA CUDA | MACA |

### Feature Parity

**Current Status**:
- ✅ Basic reduce (sum, min, max)
- ✅ Type system (int32, int64, float32, float64)
- ✅ Build/Execute separation
- ✅ Caching
- ✅ Python API
- ⏳ Custom operations (future)
- ⏳ Advanced iterators (future)
- ⏳ Scan operations (future)
- ⏳ Sort operations (future)

## Future Enhancements

1. **NVRTC-style Compilation**:
   - Add mcRTC support for runtime compilation
   - Enable custom user operations

2. **Advanced Iterators**:
   - TransformIterator
   - CountingIterator
   - ZipIterator

3. **More Algorithms**:
   - Scan (prefix sum)
   - Sort (merge sort, radix sort)
   - Histogram
   - Select

4. **Performance**:
   - Benchmark against NumPy
   - Optimize memory transfers
   - Stream support

5. **Testing**:
   - Comprehensive test suite
   - Performance benchmarks
   - CI/CD integration
