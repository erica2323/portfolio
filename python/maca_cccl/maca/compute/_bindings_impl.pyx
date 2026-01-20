# distutils: language = c++
# cython: language_level = 3

"""
Cython bindings for MACA CCCL C API

This module provides the low-level bridge between Python and the MACA CCCL C library.
"""

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString

#==============================================================================
# External C Declarations
#==============================================================================

cdef extern from "mc_runtime.h" nogil:
    ctypedef void* mcStream_t
    ctypedef int mcError_t
    ctypedef void* mcModule_t
    ctypedef void* mcFunction_t

    mcError_t mcSuccess
    mcError_t mcErrorInvalidValue
    mcError_t mcErrorMemoryAllocation
    mcError_t mcErrorUnknown

cdef extern from "cccl/c/types_official.h" nogil:
    # Type enumerations
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

    ctypedef struct cccl_type_info:
        size_t size
        size_t alignment
        cccl_type_enum type

    # Operation types
    ctypedef enum cccl_op_kind_t:
        CCCL_STATELESS = 0
        CCCL_STATEFUL = 1
        CCCL_PLUS = 2
        CCCL_MINUS = 3
        CCCL_MULTIPLIES = 4
        CCCL_DIVIDES = 5
        CCCL_MODULUS = 6
        CCCL_EQUAL_TO = 7
        CCCL_NOT_EQUAL_TO = 8
        CCCL_GREATER = 9
        CCCL_LESS = 10
        CCCL_GREATER_EQUAL = 11
        CCCL_LESS_EQUAL = 12
        CCCL_LOGICAL_AND = 13
        CCCL_LOGICAL_OR = 14
        CCCL_LOGICAL_NOT = 15
        CCCL_BIT_AND = 16
        CCCL_BIT_OR = 17
        CCCL_BIT_XOR = 18
        CCCL_BIT_NOT = 19
        CCCL_IDENTITY = 20
        CCCL_NEGATE = 21
        CCCL_MINIMUM = 22
        CCCL_MAXIMUM = 23

    ctypedef enum cccl_op_code_type:
        CCCL_OP_LTOIR = 0
        CCCL_OP_CPP_SOURCE = 1

    ctypedef struct cccl_op_t:
        cccl_op_kind_t type
        const char* name
        const char* code
        size_t code_size
        cccl_op_code_type code_type
        size_t size
        size_t alignment
        void* state

    # Iterator types
    ctypedef enum cccl_iterator_kind_t:
        CCCL_POINTER = 0
        CCCL_ITERATOR = 1

    ctypedef struct cccl_value_t:
        cccl_type_info type
        void* state

    ctypedef void (*cccl_host_op_fn_ptr_t)(void*, void*)

    ctypedef struct cccl_iterator_t:
        size_t size
        size_t alignment
        cccl_iterator_kind_t type
        cccl_op_t advance
        cccl_op_t dereference
        cccl_type_info value_type
        void* state
        cccl_host_op_fn_ptr_t host_advance

    # Build config
    ctypedef struct cccl_build_config:
        const char** extra_compile_flags
        size_t num_extra_compile_flags
        const char** extra_include_dirs
        size_t num_extra_include_dirs

cdef extern from "cccl/c/reduce_official.h" nogil:
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

    mcError_t cccl_device_reduce_build_ex(
        cccl_device_reduce_build_result_t* build,
        cccl_op_t op,
        cccl_iterator_t d_in,
        void* initial_value,
        cccl_build_config* build_config
    )

    mcError_t cccl_device_reduce_ex(
        cccl_device_reduce_build_result_t build,
        cccl_iterator_t d_in,
        void* d_out,
        uint64_t num_items,
        mcStream_t stream
    )

    mcError_t cccl_device_reduce_cleanup(
        cccl_device_reduce_build_result_t* build
    )

    cccl_iterator_t cccl_make_pointer_iterator(void* ptr, cccl_type_info type)

#==============================================================================
# Python Wrapper Classes
#==============================================================================

cdef class TypeInfo:
    """Represents CCCL type information"""
    cdef cccl_type_info _c_type

    def __init__(self, type_enum, size, alignment):
        self._c_type.type = type_enum
        self._c_type.size = size
        self._c_type.alignment = alignment

    @property
    def type(self):
        return self._c_type.type

    @property
    def size(self):
        return self._c_type.size

    @property
    def alignment(self):
        return self._c_type.alignment

    cdef cccl_type_info get_c_type(self):
        return self._c_type


cdef class Op:
    """Represents a CCCL operation"""
    cdef cccl_op_t _c_op
    cdef bytes _name_bytes
    cdef bytes _code_bytes

    def __init__(self, op_kind, name=None, code=None, code_type=CCCL_OP_LTOIR,
                 size=0, alignment=0, state=None):
        self._c_op.type = op_kind
        self._c_op.size = size
        self._c_op.alignment = alignment
        self._c_op.code_type = code_type

        # Handle name
        if name is not None:
            self._name_bytes = name.encode('utf-8') if isinstance(name, str) else name
            self._c_op.name = <const char*>PyBytes_AsString(self._name_bytes)
        else:
            self._name_bytes = b""
            self._c_op.name = NULL

        # Handle code
        if code is not None:
            self._code_bytes = code if isinstance(code, bytes) else code.encode('utf-8')
            self._c_op.code = <const char*>PyBytes_AsString(self._code_bytes)
            self._c_op.code_size = len(self._code_bytes)
        else:
            self._code_bytes = b""
            self._c_op.code = NULL
            self._c_op.code_size = 0

        self._c_op.state = <void*><size_t>state if state is not None else NULL

    @property
    def type(self):
        return self._c_op.type

    cdef cccl_op_t get_c_op(self):
        return self._c_op


cdef class Iterator:
    """Represents a CCCL iterator"""
    cdef cccl_iterator_t _c_iter
    cdef TypeInfo _value_type
    cdef Op _advance_op
    cdef Op _deref_op

    def __init__(self, iter_kind, value_type, state_ptr,
                 advance_op=None, deref_op=None, size=0, alignment=0):
        self._c_iter.type = iter_kind
        self._c_iter.size = size
        self._c_iter.alignment = alignment
        self._c_iter.state = <void*><size_t>state_ptr
        self._c_iter.host_advance = NULL

        # Store TypeInfo
        self._value_type = value_type
        self._c_iter.value_type = value_type.get_c_type()

        # Store operations
        if advance_op is not None:
            self._advance_op = advance_op
            self._c_iter.advance = advance_op.get_c_op()

        if deref_op is not None:
            self._deref_op = deref_op
            self._c_iter.dereference = deref_op.get_c_op()

    @property
    def state(self):
        return <size_t>self._c_iter.state

    cdef cccl_iterator_t get_c_iter(self):
        return self._c_iter


cdef class DeviceReduceBuildResult:
    """Result of reduce build phase"""
    cdef cccl_device_reduce_build_result_t _c_result
    cdef bint _owns_result

    def __init__(self):
        self._owns_result = False

    def __dealloc__(self):
        if self._owns_result:
            cccl_device_reduce_cleanup(&self._c_result)

    cdef void set_result(self, cccl_device_reduce_build_result_t result):
        self._c_result = result
        self._owns_result = True

    cdef cccl_device_reduce_build_result_t get_result(self):
        return self._c_result


#==============================================================================
# Build Functions
#==============================================================================

def device_reduce_build(Op op, TypeInfo type_info, initial_value,
                        extra_flags=None, extra_includes=None):
    """
    Build phase: compile reduce operation

    Parameters:
    -----------
    op : Op
        Reduction operation (e.g., CCCL_PLUS, CCCL_MINIMUM, CCCL_MAXIMUM)
    type_info : TypeInfo
        Data type information
    initial_value : scalar
        Initial value for reduction (Python scalar)
    extra_flags : list of str, optional
        Extra compilation flags
    extra_includes : list of str, optional
        Extra include directories

    Returns:
    --------
    DeviceReduceBuildResult
        Compiled reduction operation
    """
    cdef cccl_device_reduce_build_result_t c_result
    cdef cccl_build_config build_config
    cdef cccl_build_config* build_config_ptr = NULL
    cdef mcError_t err

    # Allocate space for initial value
    cdef void* init_val_ptr = malloc(type_info.size)
    if init_val_ptr == NULL:
        raise MemoryError("Failed to allocate memory for initial value")

    try:
        # Convert Python initial value to C
        _copy_scalar_to_ptr(initial_value, init_val_ptr, type_info.type)

        # Create pointer iterator
        cdef cccl_iterator_t iter = cccl_make_pointer_iterator(NULL, type_info.get_c_type())

        # Call C build function
        with nogil:
            err = cccl_device_reduce_build_ex(
                &c_result,
                op.get_c_op(),
                iter,
                init_val_ptr,
                build_config_ptr
            )

        if err != mcSuccess:
            raise RuntimeError(f"Build failed with error code: {err}")

        # Create Python wrapper
        result = DeviceReduceBuildResult()
        result.set_result(c_result)
        return result

    finally:
        free(init_val_ptr)


#==============================================================================
# Execute Functions
#==============================================================================

def device_reduce(DeviceReduceBuildResult build, size_t d_in_ptr, size_t d_out_ptr,
                  uint64_t num_items, size_t stream=0):
    """
    Execute phase: run reduction on GPU

    Parameters:
    -----------
    build : DeviceReduceBuildResult
        Compiled reduction from build phase
    d_in_ptr : int
        Device pointer to input data (as integer)
    d_out_ptr : int
        Device pointer to output data (as integer, single element)
    num_items : int
        Number of items to reduce
    stream : int, optional
        MACA stream handle (default: 0 = default stream)

    Returns:
    --------
    None
    """
    cdef mcError_t err
    cdef cccl_device_reduce_build_result_t c_build = build.get_result()

    # Create iterator with actual data pointer
    cdef cccl_iterator_t iter = cccl_make_pointer_iterator(
        <void*>d_in_ptr,
        c_build.type
    )

    # Execute reduction
    with nogil:
        err = cccl_device_reduce_ex(
            c_build,
            iter,
            <void*>d_out_ptr,
            num_items,
            <mcStream_t>stream
        )

    if err != mcSuccess:
        raise RuntimeError(f"Reduce execution failed with error code: {err}")


#==============================================================================
# Helper Functions
#==============================================================================

cdef void _copy_scalar_to_ptr(value, void* ptr, cccl_type_enum type_enum) except *:
    """Copy Python scalar to C pointer"""
    if type_enum == CCCL_INT32:
        (<int32_t*>ptr)[0] = <int32_t>value
    elif type_enum == CCCL_INT64:
        (<int64_t*>ptr)[0] = <int64_t>value
    elif type_enum == CCCL_UINT32:
        (<uint32_t*>ptr)[0] = <uint32_t>value
    elif type_enum == CCCL_UINT64:
        (<uint64_t*>ptr)[0] = <uint64_t>value
    elif type_enum == CCCL_FLOAT32:
        (<float*>ptr)[0] = <float>value
    elif type_enum == CCCL_FLOAT64:
        (<double*>ptr)[0] = <double>value
    else:
        raise ValueError(f"Unsupported type: {type_enum}")


def make_pointer_iterator(size_t ptr, TypeInfo type_info):
    """Create a pointer iterator from device pointer"""
    cdef cccl_iterator_t c_iter = cccl_make_pointer_iterator(<void*>ptr, type_info.get_c_type())

    # Wrap in Python object
    iter_obj = Iterator(
        CCCL_POINTER,
        type_info,
        ptr,
        size=0,
        alignment=0
    )
    return iter_obj


# Export type enum constants
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

# Export operation constants
PLUS = CCCL_PLUS
MINUS = CCCL_MINUS
MULTIPLIES = CCCL_MULTIPLIES
MINIMUM = CCCL_MINIMUM
MAXIMUM = CCCL_MAXIMUM
