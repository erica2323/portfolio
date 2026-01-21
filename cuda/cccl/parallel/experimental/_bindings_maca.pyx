# distutils: language = c++
# cython: language_level = 3

"""
Cython bindings for MACA CCCL C API.
This module wraps the C reduce API for use from Python.
"""

from libc.stdint cimport uint64_t, int32_t, int64_t
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

cnp.import_array()


# MACA runtime types
cdef extern from "mc_runtime.h":
    ctypedef int mcError_t
    ctypedef struct mcStream_t:
        pass

    mcError_t mcMalloc(void** devPtr, size_t size)
    mcError_t mcFree(void* devPtr)
    mcError_t mcMemcpy(void* dst, const void* src, size_t count, int kind)

    # Memory copy kinds
    cdef enum:
        mcMemcpyHostToDevice = 1
        mcMemcpyDeviceToHost = 2


# CCCL C API types
cdef extern from "cccl/c/types_official.h":
    ctypedef enum cccl_type_enum:
        INT8
        INT16
        INT32
        INT64
        UINT8
        UINT16
        UINT32
        UINT64
        FP32
        FP64

    ctypedef enum cccl_op_kind_t:
        CCCL_PLUS
        CCCL_MIN
        CCCL_MAX
        CCCL_MUL

    ctypedef enum cccl_iterator_kind_t:
        CCCL_ITERATOR_POINTER
        CCCL_ITERATOR_COUNTING
        CCCL_ITERATOR_CONSTANT

    ctypedef struct cccl_type_info:
        cccl_type_enum type
        size_t size
        size_t alignment

    ctypedef struct cccl_value_t:
        cccl_type_info type
        void* state

    ctypedef struct cccl_iterator_t:
        cccl_iterator_kind_t kind
        cccl_type_info value_type
        void* state

    ctypedef struct cccl_op_t:
        cccl_op_kind_t kind
        cccl_type_info type

    ctypedef struct cccl_device_reduce_build_result_t:
        int cc_major
        int cc_minor
        void* cubin
        size_t cubin_size


# CCCL C API functions
cdef extern from "cccl/c/reduce_official.h":
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

    void cccl_device_reduce_cleanup(cccl_device_reduce_build_result_t* build)


# Type enum mapping
TYPE_ENUM_MAP = {
    'INT8': INT8,
    'INT16': INT16,
    'INT32': INT32,
    'INT64': INT64,
    'UINT8': UINT8,
    'UINT16': UINT16,
    'UINT32': UINT32,
    'UINT64': UINT64,
    'FP32': FP32,
    'FP64': FP64,
}

OP_ENUM_MAP = {
    'PLUS': CCCL_PLUS,
    'MIN': CCCL_MIN,
    'MAX': CCCL_MAX,
    'MUL': CCCL_MUL,
}

ITERATOR_KIND_MAP = {
    'POINTER': CCCL_ITERATOR_POINTER,
    'COUNTING': CCCL_ITERATOR_COUNTING,
    'CONSTANT': CCCL_ITERATOR_CONSTANT,
}


cdef cccl_type_info make_type_info(str dtype_enum):
    """Create C type info from dtype enum string."""
    cdef cccl_type_info info
    info.type = TYPE_ENUM_MAP[dtype_enum]

    # Set size and alignment based on type
    if dtype_enum == 'INT8' or dtype_enum == 'UINT8':
        info.size = 1
        info.alignment = 1
    elif dtype_enum == 'INT16' or dtype_enum == 'UINT16':
        info.size = 2
        info.alignment = 2
    elif dtype_enum == 'INT32' or dtype_enum == 'UINT32' or dtype_enum == 'FP32':
        info.size = 4
        info.alignment = 4
    elif dtype_enum == 'INT64' or dtype_enum == 'UINT64' or dtype_enum == 'FP64':
        info.size = 8
        info.alignment = 8

    return info


cdef class DeviceReduceBuildResult:
    """
    Wrapper for cccl_device_reduce_build_result_t.
    Manages the lifecycle of a compiled reduction kernel.
    """
    cdef cccl_device_reduce_build_result_t c_result
    cdef bint is_valid

    def __cinit__(self):
        self.is_valid = False

    def __dealloc__(self):
        if self.is_valid:
            cccl_device_reduce_cleanup(&self.c_result)

    def build(self, d_in_desc, d_out_desc, op_enum, init_value, init_dtype,
              cc_major=8, cc_minor=0,
              cub_path=b"", thrust_path=b"", libcudacxx_path=b"", ctk_path=b""):
        """
        Build (compile) the reduction kernel.

        Args:
            d_in_desc: Input iterator descriptor dict
            d_out_desc: Output iterator descriptor dict
            op_enum: Operation enum string ('PLUS', 'MIN', 'MAX', 'MUL')
            init_value: Initial value (numpy scalar)
            init_dtype: Data type enum string
            cc_major: Compute capability major version
            cc_minor: Compute capability minor version
        """
        cdef cccl_iterator_t d_in
        cdef cccl_iterator_t d_out
        cdef cccl_op_t op
        cdef cccl_value_t h_init
        cdef mcError_t err

        # Setup input iterator
        d_in.kind = ITERATOR_KIND_MAP[d_in_desc['kind']]
        d_in.value_type = make_type_info(d_in_desc['type'])
        if d_in_desc['kind'] == 'POINTER':
            d_in.state = <void*><uintptr_t>d_in_desc['ptr']
        else:
            raise NotImplementedError(f"Iterator kind {d_in_desc['kind']} not yet supported")

        # Setup output iterator
        d_out.kind = ITERATOR_KIND_MAP[d_out_desc['kind']]
        d_out.value_type = make_type_info(d_out_desc['type'])
        if d_out_desc['kind'] == 'POINTER':
            d_out.state = <void*><uintptr_t>d_out_desc['ptr']
        else:
            raise NotImplementedError(f"Iterator kind {d_out_desc['kind']} not yet supported")

        # Setup operator
        op.kind = OP_ENUM_MAP[op_enum]
        op.type = make_type_info(init_dtype)

        # Setup init value
        h_init.type = make_type_info(init_dtype)
        cdef cnp.ndarray init_arr = np.asarray(init_value, dtype=init_value.dtype)
        h_init.state = <void*>cnp.PyArray_DATA(init_arr)

        # Call C API
        err = cccl_device_reduce_build(
            &self.c_result,
            d_in, d_out, op, h_init,
            cc_major, cc_minor,
            cub_path, thrust_path, libcudacxx_path, ctk_path
        )

        if err != 0:
            raise RuntimeError(f"cccl_device_reduce_build failed with error {err}")

        self.is_valid = True

    def reduce(self, d_in_desc, d_out_desc, num_items, op_enum, init_value, init_dtype, stream=None):
        """
        Execute the reduction.

        Args:
            d_in_desc: Input iterator descriptor dict
            d_out_desc: Output iterator descriptor dict
            num_items: Number of items to reduce
            op_enum: Operation enum string
            init_value: Initial value (numpy scalar)
            init_dtype: Data type enum string
            stream: MACA stream (not yet supported)

        Returns:
            None (result is written to d_out)
        """
        if not self.is_valid:
            raise RuntimeError("BuildResult not initialized. Call build() first.")

        cdef cccl_iterator_t d_in
        cdef cccl_iterator_t d_out
        cdef cccl_op_t op
        cdef cccl_value_t h_init
        cdef size_t temp_storage_bytes = 0
        cdef void* temp_storage = NULL
        cdef mcError_t err
        cdef mcStream_t c_stream = <mcStream_t>NULL

        # Setup iterators
        d_in.kind = ITERATOR_KIND_MAP[d_in_desc['kind']]
        d_in.value_type = make_type_info(d_in_desc['type'])
        if d_in_desc['kind'] == 'POINTER':
            d_in.state = <void*><uintptr_t>d_in_desc['ptr']

        d_out.kind = ITERATOR_KIND_MAP[d_out_desc['kind']]
        d_out.value_type = make_type_info(d_out_desc['type'])
        if d_out_desc['kind'] == 'POINTER':
            d_out.state = <void*><uintptr_t>d_out_desc['ptr']

        # Setup operator
        op.kind = OP_ENUM_MAP[op_enum]
        op.type = make_type_info(init_dtype)

        # Setup init value
        h_init.type = make_type_info(init_dtype)
        cdef cnp.ndarray init_arr = np.asarray(init_value, dtype=init_value.dtype)
        h_init.state = <void*>cnp.PyArray_DATA(init_arr)

        # Query temp storage size
        err = cccl_device_reduce(
            self.c_result, NULL, &temp_storage_bytes,
            d_in, d_out, num_items, op, h_init, c_stream
        )
        if err != 0:
            raise RuntimeError(f"Failed to query temp storage size: error {err}")

        # Allocate temp storage
        if temp_storage_bytes > 0:
            err = mcMalloc(&temp_storage, temp_storage_bytes)
            if err != 0:
                raise RuntimeError(f"Failed to allocate temp storage: error {err}")

        # Execute reduction
        try:
            err = cccl_device_reduce(
                self.c_result, temp_storage, &temp_storage_bytes,
                d_in, d_out, num_items, op, h_init, c_stream
            )
            if err != 0:
                raise RuntimeError(f"cccl_device_reduce failed with error {err}")
        finally:
            # Clean up temp storage
            if temp_storage != NULL:
                mcFree(temp_storage)
