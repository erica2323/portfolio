# cython: language_level=3
# distutils: language=c++
"""
Extension to _bindings_maca.pyx to support JIT custom operators

Add this code to your existing _bindings_maca.pyx file
"""

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t, int32_t, uintptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset, strlen

# ============================================================================
# 🔥 NEW: Add this to your existing cdef extern declarations
# ============================================================================

# Assuming you already have the basic CCCL types declared,
# here's what you need to ADD for JIT support:

cdef extern from "cccl/c/types_official.h":
    # ... (existing declarations)

    # Ensure op_code_type is declared:
    ctypedef enum cccl_op_code_type:
        CCCL_OP_LTOIR      = 0
        CCCL_OP_CPP_SOURCE = 1

# ============================================================================
# 🔥 NEW: Python wrapper for JIT-enabled reduce_build
# ============================================================================

def reduce_build_with_custom_op(
    d_in_ptr,
    d_out_ptr,
    type_enum,
    op_kind,
    op_code: bytes,           # 🔥 NEW: operator source code
    op_name: bytes,           # 🔥 NEW: operator name
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
    Build reduce operation with custom JIT-compiled operator.

    Args:
        op_code: C++ lambda/functor source code (bytes)
        op_name: Name for the operator (bytes)
        ... (other args same as reduce_build)

    Returns:
        build handle (pointer as size_t)
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

    # Setup pointers
    cdef void* in_ptr = <void*><uintptr_t><size_t>d_in_ptr
    cdef void* out_ptr = <void*><uintptr_t><size_t>d_out_ptr
    cdef uintptr_t init_addr = <uintptr_t><size_t>init_value

    if init_addr == 0:
        free(build)
        raise ValueError("init_value pointer is NULL")

    # Setup iterators
    _init_iterator_pointer(
        &d_in,
        in_ptr,
        <cccl_type_enum>type_enum,
        <size_t>dtype_size,
        <size_t>dtype_alignment
    )
    _init_iterator_pointer(
        &d_out,
        out_ptr,
        <cccl_type_enum>type_enum,
        <size_t>dtype_size,
        <size_t>dtype_alignment
    )

    # 🔥 NEW: Setup custom operator with source code
    memset(&op, 0, sizeof(cccl_op_t))
    op.type = <cccl_op_kind_t>op_kind  # Should be CCCL_STATELESS (0)

    # Allocate and copy operator name
    cdef size_t name_len = len(op_name)
    cdef char* name_copy = <char*>malloc(name_len + 1)
    if name_copy == NULL:
        free(build)
        raise MemoryError("Failed to allocate operator name")
    memcpy(name_copy, <const char*>op_name, name_len)
    name_copy[name_len] = 0  # null terminate
    op.name = name_copy

    # Allocate and copy operator code
    cdef size_t code_len = len(op_code)
    cdef char* code_copy = <char*>malloc(code_len + 1)
    if code_copy == NULL:
        free(name_copy)
        free(build)
        raise MemoryError("Failed to allocate operator code")
    memcpy(code_copy, <const char*>op_code, code_len)
    code_copy[code_len] = 0  # null terminate (optional for source)
    op.code = code_copy
    op.code_size = code_len
    op.code_type = CCCL_OP_CPP_SOURCE  # 🔥 Mark as C++ source

    op.size = 0
    op.alignment = 1
    op.state = NULL

    # Setup init value
    memset(&h_init, 0, sizeof(cccl_value_t))
    h_init.type.size = <size_t>dtype_size
    h_init.type.alignment = <size_t>dtype_alignment
    h_init.type.type = <cccl_type_enum>type_enum
    h_init.state = <void*>init_addr

    # Call build API (will trigger JIT compilation)
    err = cccl_device_reduce_build(
        build,
        d_in,
        d_out,
        op,
        h_init,
        <int32_t>cc_major,
        <int32_t>cc_minor,
        <const char*>cub_path,
        <const char*>thrust_path,
        <const char*>libcudacxx_path,
        <const char*>ctk_path
    )

    # Clean up temporary allocations (build struct stores its own copies)
    free(name_copy)
    free(code_copy)

    if err != 0:
        free(build)
        raise RuntimeError(f"cccl_device_reduce_build (JIT) failed with error {err}")

    return <size_t>build

# ============================================================================
# 🔥 NEW: Clear JIT cache
# ============================================================================

def clear_jit_cache():
    """
    Clear the JIT kernel cache.

    This frees all cached JIT-compiled kernels.
    Useful for testing or when you want to force recompilation.
    """
    cdef extern from "cccl/c/jit_compiler.h":
        void cccl_jit_clear_cache()

    cccl_jit_clear_cache()
    return 0

# ============================================================================
# Note: The existing reduce_execute() and reduce_cleanup() functions
#       work with both builtin and JIT operators without modification,
#       because the build_handle already contains the compiled kernel.
# ============================================================================
