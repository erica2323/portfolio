# cython: language_level=3
# distutils: language=c++
"""
Cython bindings for CCCL C API (MACA port) with JIT support

Provides low-level Python bindings to libcccl_maca.so
Supports both builtin operations and user-defined operations via JIT
"""

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t, int32_t, int64_t, uintptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy

#------------------------------------------------------------------------------
# MACA runtime types
#------------------------------------------------------------------------------

cdef extern from "mcr/mc_runtime.h":
    ctypedef int mcError_t
    ctypedef void* mcStream_t
    ctypedef void* mcModule_t
    ctypedef void* mcFunction_t

    mcError_t mcMalloc(void** devPtr, size_t size)
    mcError_t mcFree(void* devPtr)
    mcError_t mcMemcpy(void* dst, const void* src, size_t count, int kind)
    mcError_t mcStreamSynchronize(mcStream_t stream)

    # mcMemcpyKind values
    int mcMemcpyHostToDevice
    int mcMemcpyDeviceToHost
    int mcMemcpyDeviceToDevice

#------------------------------------------------------------------------------
# CCCL C API types
#------------------------------------------------------------------------------

cdef extern from "cccl/c/types.h":
    ctypedef enum cccl_type_enum:
        CCCL_INT8    = 0
        CCCL_INT16   = 1
        CCCL_INT32   = 2
        CCCL_INT64   = 3
        CCCL_UINT8   = 4
        CCCL_UINT16  = 5
        CCCL_UINT32  = 6
        CCCL_UINT64  = 7
        CCCL_FLOAT16 = 8
        CCCL_FLOAT32 = 9
        CCCL_FLOAT64 = 10
        CCCL_STORAGE = 11
        CCCL_BOOLEAN = 12

    ctypedef enum cccl_op_kind_t:
        CCCL_STATELESS   = 0
        CCCL_STATEFUL    = 1
        CCCL_PLUS        = 2
        CCCL_MINUS       = 3
        CCCL_MULTIPLIES  = 4
        CCCL_MINIMUM     = 22
        CCCL_MAXIMUM     = 23

    ctypedef enum cccl_iterator_kind_t:
        CCCL_POINTER  = 0
        CCCL_ITERATOR = 1

    ctypedef enum cccl_op_code_type:
        CCCL_OP_BITCODE    = 0
        CCCL_OP_CPP_SOURCE = 1

    ctypedef struct cccl_type_info:
        size_t         size
        size_t         alignment
        cccl_type_enum type

    ctypedef void (*cccl_host_op_fn_ptr_t)(void*, uint64_t)

    ctypedef struct cccl_op_t:
        cccl_op_kind_t     type
        const char*        name
        const char*        code
        size_t             code_size
        cccl_op_code_type  code_type
        size_t             size
        size_t             alignment
        void*              state

    ctypedef struct cccl_value_t:
        cccl_type_info type
        void*          state

    ctypedef struct cccl_iterator_t:
        size_t               size
        size_t               alignment
        cccl_iterator_kind_t type
        cccl_op_t            advance
        cccl_op_t            dereference
        cccl_type_info       value_type
        void*                state
        cccl_host_op_fn_ptr_t host_advance

#------------------------------------------------------------------------------
# CCCL Reduce API
#------------------------------------------------------------------------------

cdef extern from "cccl/c/reduce.h":
    ctypedef struct cccl_device_reduce_build_result_t:
        cccl_type_info   type
        cccl_op_t        op
        void*            initial_value
        size_t           initial_value_size
        cccl_iterator_t  d_in_iterator
        void*            jit_module
        void*            jit_kernel
        int              uses_jit
        void*            jit_bitcode
        size_t           jit_bitcode_size

    mcError_t cccl_device_reduce_build(
        cccl_device_reduce_build_result_t* build,
        cccl_iterator_t d_in,
        cccl_iterator_t d_out,
        cccl_op_t op,
        cccl_value_t h_init,
        int cc_major,
        int cc_minor,
        const char* cub_path,
        const char* thrust_path,
        const char* libcudacxx_path,
        const char* ctk_path
    )

    mcError_t cccl_device_reduce(
        cccl_device_reduce_build_result_t build,
        void* temp_storage,
        size_t* temp_storage_bytes,
        cccl_iterator_t d_in,
        cccl_iterator_t d_out,
        uint64_t num_items,
        cccl_op_t op,
        cccl_value_t h_init,
        mcStream_t stream
    )

    mcError_t cccl_device_reduce_cleanup(
        cccl_device_reduce_build_result_t* build
    )

#------------------------------------------------------------------------------
# Internal Helpers
#------------------------------------------------------------------------------

cdef inline void _init_op_zero(cccl_op_t* op) noexcept nogil:
    op.type       = <cccl_op_kind_t>0
    op.name       = NULL
    op.code       = NULL
    op.code_size  = 0
    op.code_type  = <cccl_op_code_type>0
    op.size       = 0
    op.alignment  = 0
    op.state      = NULL

cdef inline void _init_op_builtin(cccl_op_t* op, cccl_op_kind_t kind) noexcept nogil:
    op.type       = kind
    op.name       = NULL
    op.code       = NULL
    op.code_size  = 0
    op.code_type  = CCCL_OP_BITCODE
    op.size       = 0
    op.alignment  = 1
    op.state      = NULL

cdef inline void _init_op_custom(
    cccl_op_t* op,
    const char* code,
    size_t code_size,
    cccl_op_code_type code_type,
    size_t value_size,
    size_t value_alignment
) noexcept nogil:
    op.type       = CCCL_STATELESS
    op.name       = NULL
    op.code       = code
    op.code_size  = code_size
    op.code_type  = code_type
    op.size       = value_size
    op.alignment  = value_alignment
    op.state      = NULL

cdef inline void _init_iterator_pointer(
    cccl_iterator_t* it,
    void*           ptr,
    cccl_type_enum  t,
    size_t          elem_size,
    size_t          elem_align
) noexcept nogil:
    it.size               = 0
    it.alignment          = 0
    it.type               = CCCL_POINTER
    it.state              = ptr
    it.value_type.size    = elem_size
    it.value_type.alignment = elem_align
    it.value_type.type    = t
    _init_op_zero(&it.advance)
    _init_op_zero(&it.dereference)
    it.host_advance       = NULL

#------------------------------------------------------------------------------
# Python API: Build with builtin operation
#------------------------------------------------------------------------------

def reduce_build(
    d_in_ptr,
    d_out_ptr,
    type_enum,
    op_kind,
    init_value,
    dtype_size,
    dtype_alignment,
    cc_major=9,
    cc_minor=0,
    cub_path=b"",
    thrust_path=b"",
    libcudacxx_path=b"",
    ctk_path=b""
):
    """
    Build reduce operation with builtin operator.

    Args:
        d_in_ptr: Device input pointer (as int)
        d_out_ptr: Device output pointer (as int)
        type_enum: Type enum (CCCL_INT32, etc.)
        op_kind: Operation kind (CCCL_PLUS, CCCL_MIN, CCCL_MAX)
        init_value: Pointer to initial value on host (as int)
        dtype_size: Size of data type
        dtype_alignment: Alignment of data type
        cub_path: Path to mcCub headers (for JIT, optional)
        ctk_path: Path to MACA toolkit (for JIT, optional)

    Returns:
        Build handle (pointer as int)
    """
    cdef cccl_device_reduce_build_result_t* build = <cccl_device_reduce_build_result_t*>malloc(
        sizeof(cccl_device_reduce_build_result_t)
    )
    if build == NULL:
        raise MemoryError("Failed to allocate build result struct")

    memset(build, 0, sizeof(cccl_device_reduce_build_result_t))

    cdef cccl_iterator_t d_in
    cdef cccl_iterator_t d_out
    cdef cccl_op_t op
    cdef cccl_value_t h_init
    cdef mcError_t err

    # Convert Python ints to pointers
    cdef void* in_ptr = <void*><uintptr_t><size_t>d_in_ptr
    cdef void* out_ptr = <void*><uintptr_t><size_t>d_out_ptr
    cdef uintptr_t init_addr = <uintptr_t><size_t>init_value

    if init_addr == 0:
        free(build)
        raise ValueError("init_value pointer is NULL")

    # Setup iterators
    _init_iterator_pointer(
        &d_in, in_ptr,
        <cccl_type_enum>type_enum,
        <size_t>dtype_size,
        <size_t>dtype_alignment
    )
    _init_iterator_pointer(
        &d_out, out_ptr,
        <cccl_type_enum>type_enum,
        <size_t>dtype_size,
        <size_t>dtype_alignment
    )

    # Setup builtin operator
    _init_op_builtin(&op, <cccl_op_kind_t>op_kind)

    # Setup init value
    memset(&h_init, 0, sizeof(cccl_value_t))
    h_init.type.size      = <size_t>dtype_size
    h_init.type.alignment = <size_t>dtype_alignment
    h_init.type.type      = <cccl_type_enum>type_enum
    h_init.state          = <void*>init_addr

    # Call build
    err = cccl_device_reduce_build(
        build,
        d_in, d_out,
        op, h_init,
        <int>cc_major, <int>cc_minor,
        <const char*>cub_path,
        <const char*>thrust_path,
        <const char*>libcudacxx_path,
        <const char*>ctk_path
    )

    if err != 0:
        free(build)
        raise RuntimeError(f"cccl_device_reduce_build failed with error {err}")

    return <size_t>build

#------------------------------------------------------------------------------
# Python API: Build with custom operation (JIT)
#------------------------------------------------------------------------------

def reduce_build_custom(
    d_in_ptr,
    d_out_ptr,
    type_enum,
    op_source,           # bytes: C++ source code
    init_value,
    dtype_size,
    dtype_alignment,
    cub_path=b"",
    ctk_path=b""
):
    """
    Build reduce operation with custom user-defined operator.

    The operator source should define a device function like:
        extern "C" __device__ void reduce_op_device_fn(
            void* result, const void* arg0, const void* arg1
        ) {
            // Your reduction logic here
        }

    Args:
        d_in_ptr: Device input pointer (as int)
        d_out_ptr: Device output pointer (as int)
        type_enum: Type enum (CCCL_INT32, etc.)
        op_source: C++ source code for the operation (bytes)
        init_value: Pointer to initial value on host (as int)
        dtype_size: Size of data type
        dtype_alignment: Alignment of data type
        cub_path: Path to mcCub headers
        ctk_path: Path to MACA toolkit

    Returns:
        Build handle (pointer as int)
    """
    cdef cccl_device_reduce_build_result_t* build = <cccl_device_reduce_build_result_t*>malloc(
        sizeof(cccl_device_reduce_build_result_t)
    )
    if build == NULL:
        raise MemoryError("Failed to allocate build result struct")

    memset(build, 0, sizeof(cccl_device_reduce_build_result_t))

    cdef cccl_iterator_t d_in
    cdef cccl_iterator_t d_out
    cdef cccl_op_t op
    cdef cccl_value_t h_init
    cdef mcError_t err

    # Convert Python ints to pointers
    cdef void* in_ptr = <void*><uintptr_t><size_t>d_in_ptr
    cdef void* out_ptr = <void*><uintptr_t><size_t>d_out_ptr
    cdef uintptr_t init_addr = <uintptr_t><size_t>init_value

    if init_addr == 0:
        free(build)
        raise ValueError("init_value pointer is NULL")

    # Setup iterators
    _init_iterator_pointer(
        &d_in, in_ptr,
        <cccl_type_enum>type_enum,
        <size_t>dtype_size,
        <size_t>dtype_alignment
    )
    _init_iterator_pointer(
        &d_out, out_ptr,
        <cccl_type_enum>type_enum,
        <size_t>dtype_size,
        <size_t>dtype_alignment
    )

    # Setup custom operator with C++ source
    cdef bytes op_bytes = op_source if isinstance(op_source, bytes) else op_source.encode('utf-8')
    _init_op_custom(
        &op,
        <const char*>op_bytes,
        len(op_bytes),
        CCCL_OP_CPP_SOURCE,
        <size_t>dtype_size,
        <size_t>dtype_alignment
    )

    # Setup init value
    memset(&h_init, 0, sizeof(cccl_value_t))
    h_init.type.size      = <size_t>dtype_size
    h_init.type.alignment = <size_t>dtype_alignment
    h_init.type.type      = <cccl_type_enum>type_enum
    h_init.state          = <void*>init_addr

    # Call build
    err = cccl_device_reduce_build(
        build,
        d_in, d_out,
        op, h_init,
        9, 0,  # cc_major, cc_minor (unused for MACA)
        <const char*>cub_path,
        b"",  # thrust_path
        b"",  # libcudacxx_path
        <const char*>ctk_path
    )

    if err != 0:
        free(build)
        raise RuntimeError(f"cccl_device_reduce_build (custom) failed with error {err}")

    return <size_t>build

#------------------------------------------------------------------------------
# Python API: Execute
#------------------------------------------------------------------------------

def reduce_execute(
    build_handle,
    d_in_ptr,
    d_out_ptr,
    num_items,
    type_enum,
    op_kind,
    init_value,
    dtype_size,
    dtype_alignment,
    stream_ptr=0
):
    """
    Execute reduce operation (two-phase).

    Args:
        build_handle: Handle from reduce_build
        d_in_ptr: Device input pointer
        d_out_ptr: Device output pointer
        num_items: Number of items to reduce
        type_enum: Type enum
        op_kind: Operation kind
        init_value: Pointer to initial value
        dtype_size: Size of data type
        dtype_alignment: Alignment of data type
        stream_ptr: MACA stream pointer (0 for default)
    """
    cdef cccl_device_reduce_build_result_t* build_ptr = \
        <cccl_device_reduce_build_result_t*><size_t>build_handle
    if build_ptr == NULL:
        raise ValueError("build_handle is NULL")

    cdef cccl_device_reduce_build_result_t build = build_ptr[0]

    cdef cccl_iterator_t d_in
    cdef cccl_iterator_t d_out
    cdef cccl_op_t op
    cdef cccl_value_t h_init
    cdef mcStream_t stream = <mcStream_t><size_t>stream_ptr
    cdef size_t temp_storage_bytes = 0
    cdef void* d_temp_storage = NULL
    cdef mcError_t err

    cdef void* in_ptr = <void*><uintptr_t><size_t>d_in_ptr
    cdef void* out_ptr = <void*><uintptr_t><size_t>d_out_ptr
    cdef uintptr_t init_addr = <uintptr_t><size_t>init_value

    if init_addr == 0:
        raise ValueError("init_value pointer is NULL")

    # Setup iterators
    _init_iterator_pointer(
        &d_in, in_ptr,
        <cccl_type_enum>type_enum,
        <size_t>dtype_size,
        <size_t>dtype_alignment
    )
    _init_iterator_pointer(
        &d_out, out_ptr,
        <cccl_type_enum>type_enum,
        <size_t>dtype_size,
        <size_t>dtype_alignment
    )

    # Setup operator (use builtin for execution)
    _init_op_builtin(&op, <cccl_op_kind_t>op_kind)

    # Setup init value
    memset(&h_init, 0, sizeof(cccl_value_t))
    h_init.type.size      = <size_t>dtype_size
    h_init.type.alignment = <size_t>dtype_alignment
    h_init.type.type      = <cccl_type_enum>type_enum
    h_init.state          = <void*>init_addr

    # Phase 1: Query temp storage bytes
    err = cccl_device_reduce(
        build,
        NULL,
        &temp_storage_bytes,
        d_in, d_out,
        <uint64_t>num_items,
        op, h_init,
        stream
    )
    if err != 0:
        raise RuntimeError(f"cccl_device_reduce (query) failed with error {err}")

    # Allocate temp storage
    if temp_storage_bytes > 0:
        err = mcMalloc(&d_temp_storage, temp_storage_bytes)
        if err != 0:
            raise RuntimeError(f"mcMalloc failed with error {err}")

    # Phase 2: Execute
    try:
        err = cccl_device_reduce(
            build,
            d_temp_storage,
            &temp_storage_bytes,
            d_in, d_out,
            <uint64_t>num_items,
            op, h_init,
            stream
        )
    finally:
        if d_temp_storage != NULL:
            mcFree(d_temp_storage)

    if err != 0:
        raise RuntimeError(f"cccl_device_reduce (execute) failed with error {err}")

    return 0

#------------------------------------------------------------------------------
# Python API: Cleanup
#------------------------------------------------------------------------------

def reduce_cleanup(build_handle):
    """
    Cleanup reduce build result and free memory.

    Args:
        build_handle: Handle from reduce_build
    """
    cdef cccl_device_reduce_build_result_t* build = \
        <cccl_device_reduce_build_result_t*><size_t>build_handle
    if build == NULL:
        raise ValueError("build_handle is NULL")

    cdef mcError_t err = cccl_device_reduce_cleanup(build)
    if err != 0:
        free(build)
        raise RuntimeError(f"cccl_device_reduce_cleanup failed with error {err}")

    free(build)
    return 0

#------------------------------------------------------------------------------
# GPU Memory Helpers
#------------------------------------------------------------------------------

def gpu_malloc(size_t size):
    """Allocate device memory."""
    cdef void* dev_ptr = NULL
    cdef mcError_t err = mcMalloc(&dev_ptr, size)
    if err != 0:
        raise RuntimeError(f"mcMalloc failed with error {err}")
    return <size_t>dev_ptr

def gpu_free(size_t dev_ptr):
    """Free device memory."""
    cdef mcError_t err = mcFree(<void*>dev_ptr)
    if err != 0:
        raise RuntimeError(f"mcFree failed with error {err}")
    return 0

def gpu_memcpy_h2d(size_t dst, size_t src, size_t count):
    """Copy host to device."""
    cdef mcError_t err = mcMemcpy(
        <void*>dst, <const void*>src, count, 1  # mcMemcpyHostToDevice
    )
    if err != 0:
        raise RuntimeError(f"mcMemcpy H2D failed with error {err}")
    return 0

def gpu_memcpy_d2h(size_t dst, size_t src, size_t count):
    """Copy device to host."""
    cdef mcError_t err = mcMemcpy(
        <void*>dst, <const void*>src, count, 2  # mcMemcpyDeviceToHost
    )
    if err != 0:
        raise RuntimeError(f"mcMemcpy D2H failed with error {err}")
    return 0
