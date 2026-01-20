# cython: language_level=3
# distutils: language = c++

"""
MACA CCCL Cython Bindings Implementation
Wraps the C API for Python access
"""

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# External C declarations
cdef extern from "mc_runtime.h":
    ctypedef int mcError_t
    ctypedef void* mcStream_t
    ctypedef void* mcModule_t
    ctypedef void* mcFunction_t

    mcError_t mcSuccess
    mcError_t mcErrorInvalidValue
    mcError_t mcErrorMemoryAllocation

cdef extern from "cccl/c/types_official.h":
    ctypedef enum cccl_type_enum:
        CCCL_INT8 = 0
        CCCL_INT16 = 1
        CCCL_INT32 = 2
        CCCL_INT64 = 3
        CCCL_UINT8 = 4
        CCCL_UINT16 = 5
        CCCL_UINT32 = 6
        CCCL_UINT64 = 7
        CCCL_FLOAT16 = 8
        CCCL_FLOAT32 = 9
        CCCL_FLOAT64 = 10
        CCCL_STORAGE = 11
        CCCL_BOOLEAN = 12

    ctypedef enum cccl_op_kind_t:
        CCCL_STATELESS = 0
        CCCL_STATEFUL = 1
        CCCL_PLUS = 2
        CCCL_MINIMUM = 22
        CCCL_MAXIMUM = 23

    ctypedef enum cccl_iterator_kind_t:
        CCCL_POINTER = 0
        CCCL_ITERATOR = 1

    ctypedef struct cccl_type_info:
        size_t size
        size_t alignment
        cccl_type_enum type

    ctypedef struct cccl_op_t:
        cccl_op_kind_t type
        const char* name
        const char* code
        size_t code_size
        size_t size
        size_t alignment
        void* state

    ctypedef struct cccl_iterator_t:
        size_t size
        size_t alignment
        cccl_iterator_kind_t type
        cccl_op_t advance
        cccl_op_t dereference
        cccl_type_info value_type
        void* state
        void* host_advance

    ctypedef struct cccl_build_config:
        const char** extra_compile_flags
        size_t num_extra_compile_flags
        const char** extra_include_dirs
        size_t num_extra_include_dirs

cdef extern from "cccl/c/reduce_official.h":
    ctypedef struct cccl_device_reduce_build_result_t:
        void* bitcode
        size_t bitcode_size
        mcModule_t module
        mcFunction_t reduce_kernel
        cccl_type_info type
        cccl_op_t op
        void* initial_value
        size_t initial_value_size
        cccl_iterator_t d_in_iterator

    mcError_t cccl_device_reduce_build(
        cccl_device_reduce_build_result_t* build,
        cccl_op_t op,
        cccl_type_info type,
        void* initial_value,
        cccl_build_config* build_config
    )

    mcError_t cccl_device_reduce(
        cccl_device_reduce_build_result_t build,
        void* d_in,
        void* d_out,
        uint64_t num_items,
        mcStream_t stream
    )

    mcError_t cccl_device_reduce_cleanup(
        cccl_device_reduce_build_result_t* build
    )

#==============================================================================
# Python Type System
#==============================================================================

class CCCLType:
    """CCCL type enumeration"""
    INT8 = CCCL_INT8
    INT16 = CCCL_INT16
    INT32 = CCCL_INT32
    INT64 = CCCL_INT64
    UINT8 = CCCL_UINT8
    UINT16 = CCCL_UINT16
    UINT32 = CCCL_UINT32
    UINT64 = CCCL_UINT64
    FLOAT16 = CCCL_FLOAT16
    FLOAT32 = CCCL_FLOAT32
    FLOAT64 = CCCL_FLOAT64

class OpKind:
    """Operation kind enumeration"""
    STATELESS = CCCL_STATELESS
    STATEFUL = CCCL_STATEFUL
    PLUS = CCCL_PLUS
    MINIMUM = CCCL_MINIMUM
    MAXIMUM = CCCL_MAXIMUM

#==============================================================================
# Type Helper Functions
#==============================================================================

def get_cccl_type(dtype):
    """Convert numpy/Python dtype to CCCL type enum"""
    import numpy as np

    type_map = {
        np.int8: CCCL_INT8,
        np.int16: CCCL_INT16,
        np.int32: CCCL_INT32,
        np.int64: CCCL_INT64,
        np.uint8: CCCL_UINT8,
        np.uint16: CCCL_UINT16,
        np.uint32: CCCL_UINT32,
        np.uint64: CCCL_UINT64,
        np.float32: CCCL_FLOAT32,
        np.float64: CCCL_FLOAT64,
    }

    dtype = np.dtype(dtype)
    if dtype.type not in type_map:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return type_map[dtype.type]

def get_type_info(dtype):
    """Create cccl_type_info from numpy dtype"""
    import numpy as np

    cdef cccl_type_info info
    dtype = np.dtype(dtype)
    info.size = dtype.itemsize
    info.alignment = dtype.alignment
    info.type = get_cccl_type(dtype)

    return info

#==============================================================================
# DeviceReduce Build Result Wrapper
#==============================================================================

cdef class DeviceReduceBuildResult:
    """Python wrapper for cccl_device_reduce_build_result_t"""
    cdef cccl_device_reduce_build_result_t _c_build
    cdef object _dtype
    cdef object _op_kind

    def __cinit__(self):
        self._c_build.bitcode = NULL
        self._c_build.module = NULL
        self._c_build.initial_value = NULL

    def __dealloc__(self):
        """Cleanup C resources"""
        cccl_device_reduce_cleanup(&self._c_build)

    @property
    def dtype(self):
        return self._dtype

    @property
    def op_kind(self):
        return self._op_kind

    def compute(self, d_in, d_out, size_t num_items, stream=None):
        """
        Execute the reduce operation

        Parameters
        ----------
        d_in : device pointer (int)
            Input data pointer
        d_out : device pointer (int)
            Output data pointer
        num_items : int
            Number of elements to reduce
        stream : int, optional
            MACA stream handle
        """
        cdef void* d_in_ptr = <void*><size_t>d_in
        cdef void* d_out_ptr = <void*><size_t>d_out
        cdef mcStream_t c_stream = <mcStream_t><size_t>(stream if stream else 0)

        cdef mcError_t err = cccl_device_reduce(
            self._c_build,
            d_in_ptr,
            d_out_ptr,
            num_items,
            c_stream
        )

        if err != mcSuccess:
            raise RuntimeError(f"MACA reduce failed with error code: {err}")

#==============================================================================
# Build Function
#==============================================================================

def device_reduce_build(op_kind, dtype, initial_value):
    """
    Build (compile) a reduce operation

    Parameters
    ----------
    op_kind : int
        Operation kind (OpKind.PLUS, OpKind.MINIMUM, OpKind.MAXIMUM)
    dtype : numpy.dtype
        Data type
    initial_value : scalar
        Initial value for reduction

    Returns
    -------
    DeviceReduceBuildResult
        Compiled reduce operation
    """
    import numpy as np

    # Convert dtype
    dtype = np.dtype(dtype)
    cdef cccl_type_info type_info = get_type_info(dtype)

    # Create operation
    cdef cccl_op_t op
    op.type = <cccl_op_kind_t>op_kind
    op.name = NULL
    op.code = NULL
    op.code_size = 0
    op.size = 0
    op.alignment = 0
    op.state = NULL

    # Convert initial value to bytes
    initial_array = np.array(initial_value, dtype=dtype)
    cdef void* initial_ptr = <void*>np.PyArray_DATA(initial_array)

    # Create build result wrapper
    result = DeviceReduceBuildResult()
    result._dtype = dtype
    result._op_kind = op_kind

    # Call C API
    cdef mcError_t err = cccl_device_reduce_build(
        &result._c_build,
        op,
        type_info,
        initial_ptr,
        NULL  # build_config
    )

    if err != mcSuccess:
        raise RuntimeError(f"Failed to build reduce: error {err}")

    return result

#==============================================================================
# Utility: Extract pointer from buffer
#==============================================================================

def get_device_pointer(obj):
    """Extract device pointer from various objects (CuPy, PyTorch, etc.)"""
    # CuPy array
    if hasattr(obj, '__cuda_array_interface__'):
        return obj.__cuda_array_interface__['data'][0]

    # Raw integer pointer
    if isinstance(obj, int):
        return obj

    # PyTorch tensor
    if hasattr(obj, 'data_ptr'):
        return obj.data_ptr()

    raise TypeError(f"Cannot extract device pointer from {type(obj)}")
