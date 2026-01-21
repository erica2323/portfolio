# cython: language_level=3
# distutils: language=c++

"""
Cython bindings for CCCL C API (MACA port)

This provides low-level Python bindings to libcccl_maca.so
"""

from libc.stdint cimport uint64_t, int32_t
from libc.stdlib cimport malloc, free

# MACA runtime types
cdef extern from "mc_runtime.h":
    ctypedef int mcError_t
    ctypedef void* mcStream_t

    mcError_t mcMalloc(void** devPtr, size_t size)
    mcError_t mcFree(void* devPtr)
    mcError_t mcMemcpy(void* dst, const void* src, size_t count, int kind)
    mcError_t mcStreamSynchronize(mcStream_t stream)

    # mcMemcpyKind
    cdef int mcMemcpyHostToDevice
    cdef int mcMemcpyDeviceToHost
    cdef int mcMemcpyDeviceToDevice


# CCCL C API types
cdef extern from "cccl/c/types_official.h":
    ctypedef enum cccl_type_enum:
        CCCL_TYPE_INT8 = 0
        CCCL_TYPE_INT16 = 1
        CCCL_TYPE_INT32 = 2
        CCCL_TYPE_INT64 = 3
        CCCL_TYPE_UINT8 = 4
        CCCL_TYPE_UINT16 = 5
        CCCL_TYPE_UINT32 = 6
        CCCL_TYPE_UINT64 = 7
        CCCL_TYPE_FLOAT32 = 8
        CCCL_TYPE_FLOAT64 = 9

    ctypedef enum cccl_op_kind_t:
        CCCL_SUM = 0
        CCCL_MIN = 1
        CCCL_MAX = 2
        CCCL_PROD = 3

    ctypedef struct cccl_iterator_t:
        void* state
        cccl_type_enum type
        size_t size
        size_t alignment

    ctypedef struct cccl_value_t:
        void* state
        cccl_type_enum type
        size_t size
        size_t alignment

    ctypedef struct cccl_op_t:
        cccl_op_kind_t type
        void* state
        size_t size
        size_t alignment

    ctypedef void* cccl_device_reduce_build_result_t


# CCCL reduce API
cdef extern from "cccl/c/reduce_official.h":
    mcError_t cccl_device_reduce_build(
        cccl_device_reduce_build_result_t* build,
        cccl_iterator_t d_in,
        cccl_iterator_t d_out,
        cccl_op_t op,
        cccl_value_t h_init,
        int32_t cc_major,
        int32_t cc_minor,
        const char* cub_path,
        const char* thrust_path,
        const char* libcudacxx_path,
        const char* ctk_path
    )

    mcError_t cccl_device_reduce(
        cccl_device_reduce_build_result_t build,
        void* d_temp_storage,
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


# Python wrapper functions
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
    Build reduce operation (Phase 1).

    Returns build handle (void*).
    """
    cdef cccl_device_reduce_build_result_t build
    cdef cccl_iterator_t d_in
    cdef cccl_iterator_t d_out
    cdef cccl_op_t op
    cdef cccl_value_t h_init
    cdef mcError_t err

    # Setup input iterator
    d_in.state = <void*><size_t>d_in_ptr
    d_in.type = <cccl_type_enum>type_enum
    d_in.size = dtype_size
    d_in.alignment = dtype_alignment

    # Setup output iterator
    d_out.state = <void*><size_t>d_out_ptr
    d_out.type = <cccl_type_enum>type_enum
    d_out.size = dtype_size
    d_out.alignment = dtype_alignment

    # Setup operator
    op.type = <cccl_op_kind_t>op_kind
    op.state = NULL
    op.size = 0
    op.alignment = 1

    # Setup init value
    h_init.state = <void*><size_t>init_value
    h_init.type = <cccl_type_enum>type_enum
    h_init.size = dtype_size
    h_init.alignment = dtype_alignment

    # Call C API
    err = cccl_device_reduce_build(
        &build,
        d_in,
        d_out,
        op,
        h_init,
        cc_major,
        cc_minor,
        <const char*>cub_path,
        <const char*>thrust_path,
        <const char*>libcudacxx_path,
        <const char*>ctk_path
    )

    if err != 0:
        raise RuntimeError(f"cccl_device_reduce_build failed with error {err}")

    return <size_t>build


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
    Execute reduce operation (Phase 2).

    Two-phase execution:
    1. Pass d_temp_storage=NULL to query temp storage size
    2. Allocate temp storage and execute with it
    """
    cdef cccl_device_reduce_build_result_t build = <cccl_device_reduce_build_result_t><size_t>build_handle
    cdef cccl_iterator_t d_in
    cdef cccl_iterator_t d_out
    cdef cccl_op_t op
    cdef cccl_value_t h_init
    cdef mcStream_t stream = <mcStream_t><size_t>stream_ptr
    cdef size_t temp_storage_bytes = 0
    cdef void* d_temp_storage = NULL
    cdef mcError_t err

    # Setup iterators
    d_in.state = <void*><size_t>d_in_ptr
    d_in.type = <cccl_type_enum>type_enum
    d_in.size = dtype_size
    d_in.alignment = dtype_alignment

    d_out.state = <void*><size_t>d_out_ptr
    d_out.type = <cccl_type_enum>type_enum
    d_out.size = dtype_size
    d_out.alignment = dtype_alignment

    # Setup operator
    op.type = <cccl_op_kind_t>op_kind
    op.state = NULL
    op.size = 0
    op.alignment = 1

    # Setup init value
    h_init.state = <void*><size_t>init_value
    h_init.type = <cccl_type_enum>type_enum
    h_init.size = dtype_size
    h_init.alignment = dtype_alignment

    # Phase 1: Query temp storage size
    err = cccl_device_reduce(
        build,
        NULL,
        &temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        op,
        h_init,
        stream
    )

    if err != 0:
        raise RuntimeError(f"cccl_device_reduce (query) failed with error {err}")

    # Allocate temp storage
    if temp_storage_bytes > 0:
        err = mcMalloc(&d_temp_storage, temp_storage_bytes)
        if err != 0:
            raise RuntimeError(f"mcMalloc failed with error {err}")

    # Phase 2: Execute reduce
    err = cccl_device_reduce(
        build,
        d_temp_storage,
        &temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        op,
        h_init,
        stream
    )

    # Cleanup temp storage
    if d_temp_storage != NULL:
        mcFree(d_temp_storage)

    if err != 0:
        raise RuntimeError(f"cccl_device_reduce (execute) failed with error {err}")

    return 0


def reduce_cleanup(build_handle):
    """
    Cleanup reduce build result.
    """
    cdef cccl_device_reduce_build_result_t build = <cccl_device_reduce_build_result_t><size_t>build_handle
    cdef mcError_t err

    err = cccl_device_reduce_cleanup(&build)

    if err != 0:
        raise RuntimeError(f"cccl_device_reduce_cleanup failed with error {err}")

    return 0
