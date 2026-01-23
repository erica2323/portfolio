"""
High-level reduce API for CCCL (MACA port) with JIT support

Extended version that supports:
- Builtin operators (OpKind enum)
- Custom operators via JIT compilation
"""

import numpy as np
import ctypes
from typing import Union, Optional
from .._cccl_interop_simplified import (
    OpKind,
    numpy_dtype_to_type_enum,
    get_dtype_size,
    get_dtype_alignment,
)

try:
    from .. import _bindings_maca
    _BINDINGS_AVAILABLE = True
except ImportError:
    _BINDINGS_AVAILABLE = False

def _get_device_pointer(arr):
    """Extract device pointer from array-like object."""
    if hasattr(arr, "cuda_array_interface"):
        cuda_iface = arr.cuda_array_interface
        return cuda_iface["data"][0], cuda_iface["shape"], cuda_iface["typestr"]
    elif isinstance(arr, int):
        return arr, None, None
    else:
        raise TypeError(
            f"Unsupported array type: {type(arr)}. "
            "Expected object with cuda_array_interface or integer pointer."
        )

def _make_init_scalar_ptr(dtype: np.dtype, init_value):
    """
    Create a stable C pointer to a scalar init value.
    Returns (ptr, keepalive_object)
    """
    dtype = np.dtype(dtype)

    if dtype == np.int32:
        c_obj = ctypes.c_int32(int(init_value))
    elif dtype == np.int64:
        c_obj = ctypes.c_int64(int(init_value))
    elif dtype == np.float32:
        c_obj = ctypes.c_float(float(init_value))
    elif dtype == np.float64:
        c_obj = ctypes.c_double(float(init_value))
    elif dtype == np.uint8:
        c_obj = ctypes.c_uint8(int(init_value))
    elif dtype == np.uint16:
        c_obj = ctypes.c_uint16(int(init_value))
    elif dtype == np.uint32:
        c_obj = ctypes.c_uint32(int(init_value))
    elif dtype == np.uint64:
        c_obj = ctypes.c_uint64(int(init_value))
    else:
        raise ValueError(f"Unsupported dtype for init: {dtype}")

    return ctypes.addressof(c_obj), c_obj

# ============================================================================
# Original reduce function (builtin operators only)
# ============================================================================
def reduce(
    d_in,
    d_out,
    op: Union[OpKind, int] = OpKind.SUM,
    init: Union[int, float] = 0,
    stream=None,
    num_items: Optional[int] = None,
):
    """
    Reduce with builtin operators (SUM, MIN, MAX)

    Args:
        d_in: Input device array
        d_out: Output device array (single element)
        op: Builtin operator (OpKind enum)
        init: Initial value
        stream: MACA stream (optional)
        num_items: Number of items (auto-detected if not provided)
    """
    if not _BINDINGS_AVAILABLE:
        raise RuntimeError(
            "Cython bindings not available. "
            "Build the extension with: python setup_maca.py build_ext --inplace"
        )

    d_in_ptr, in_shape, in_typestr = _get_device_pointer(d_in)
    d_out_ptr, _, _ = _get_device_pointer(d_out)

    if num_items is None:
        if in_shape is None:
            raise ValueError("num_items must be specified when using raw pointers")
        num_items = int(np.prod(in_shape))

    if in_typestr is not None:
        dtype = np.dtype(in_typestr)
    elif hasattr(d_in, "dtype"):
        dtype = np.dtype(d_in.dtype)
    else:
        raise ValueError("Cannot determine dtype from input.")

    type_enum = int(numpy_dtype_to_type_enum(dtype))
    dtype_size = int(get_dtype_size(dtype))
    dtype_alignment = int(get_dtype_alignment(dtype))

    if isinstance(op, int):
        op = OpKind(op)
    op_kind = int(op)

    init_ptr, _init_keepalive = _make_init_scalar_ptr(dtype, init)

    stream_ptr = 0
    if stream is not None:
        if hasattr(stream, "ptr"):
            stream_ptr = int(stream.ptr)
        elif isinstance(stream, int):
            stream_ptr = int(stream)

    build_handle = None
    try:
        # Phase 1: Build
        build_handle = _bindings_maca.reduce_build(
            d_in_ptr=d_in_ptr,
            d_out_ptr=d_out_ptr,
            type_enum=type_enum,
            op_kind=op_kind,
            init_value=init_ptr,
            dtype_size=dtype_size,
            dtype_alignment=dtype_alignment,
        )

        # Phase 2: Execute
        _bindings_maca.reduce_execute(
            build_handle=build_handle,
            d_in_ptr=d_in_ptr,
            d_out_ptr=d_out_ptr,
            num_items=num_items,
            type_enum=type_enum,
            op_kind=op_kind,
            init_value=init_ptr,
            dtype_size=dtype_size,
            dtype_alignment=dtype_alignment,
            stream_ptr=stream_ptr,
        )

    finally:
        # Phase 3: Cleanup
        if build_handle is not None:
            _bindings_maca.reduce_cleanup(build_handle)

# ============================================================================
# 🔥 NEW: Custom operator reduce with JIT
# ============================================================================
def reduce_with_custom_op(
    d_in,
    d_out,
    op_code: str,
    op_name: str = "custom_op",
    init: Union[int, float] = 0,
    stream=None,
    num_items: Optional[int] = None,
    cc_major: int = 9,
    cc_minor: int = 0,
    cub_path: str = "",
    thrust_path: str = "",
    libcudacxx_path: str = "",
):
    """
    Reduce with custom operator via JIT compilation

    Args:
        d_in: Input device array
        d_out: Output device array (single element)
        op_code: C++ lambda/functor source code (e.g., "[](auto a, auto b) { return a + b; }")
        op_name: Name for the operator (for caching)
        init: Initial value
        stream: MACA stream (optional)
        num_items: Number of items (auto-detected if not provided)
        cc_major: Compute capability major version
        cc_minor: Compute capability minor version
        cub_path: Path to CUB headers
        thrust_path: Path to Thrust headers
        libcudacxx_path: Path to libcu++ headers

    Example:
        >>> # Custom multiply operator
        >>> op_code = "[](const auto& a, const auto& b) { return a * b; }"
        >>> reduce_with_custom_op(d_in, d_out, op_code, op_name="multiply", init=1)
    """
    if not _BINDINGS_AVAILABLE:
        raise RuntimeError("Cython bindings not available")

    d_in_ptr, in_shape, in_typestr = _get_device_pointer(d_in)
    d_out_ptr, _, _ = _get_device_pointer(d_out)

    if num_items is None:
        if in_shape is None:
            raise ValueError("num_items must be specified when using raw pointers")
        num_items = int(np.prod(in_shape))

    if in_typestr is not None:
        dtype = np.dtype(in_typestr)
    elif hasattr(d_in, "dtype"):
        dtype = np.dtype(d_in.dtype)
    else:
        raise ValueError("Cannot determine dtype from input.")

    type_enum = int(numpy_dtype_to_type_enum(dtype))
    dtype_size = int(get_dtype_size(dtype))
    dtype_alignment = int(get_dtype_alignment(dtype))

    # OpKind.STATELESS = 0 for custom operators
    op_kind = 0  # CCCL_STATELESS

    init_ptr, _init_keepalive = _make_init_scalar_ptr(dtype, init)

    stream_ptr = 0
    if stream is not None:
        if hasattr(stream, "ptr"):
            stream_ptr = int(stream.ptr)
        elif isinstance(stream, int):
            stream_ptr = int(stream)

    # Convert Python strings to bytes for C API
    op_code_bytes = op_code.encode('utf-8')
    op_name_bytes = op_name.encode('utf-8')
    cub_path_bytes = cub_path.encode('utf-8')
    thrust_path_bytes = thrust_path.encode('utf-8')
    libcudacxx_path_bytes = libcudacxx_path.encode('utf-8')

    build_handle = None
    try:
        # Phase 1: Build (with JIT compilation)
        build_handle = _bindings_maca.reduce_build_with_custom_op(
            d_in_ptr=d_in_ptr,
            d_out_ptr=d_out_ptr,
            type_enum=type_enum,
            op_kind=op_kind,
            op_code=op_code_bytes,
            op_name=op_name_bytes,
            init_value=init_ptr,
            dtype_size=dtype_size,
            dtype_alignment=dtype_alignment,
            cc_major=cc_major,
            cc_minor=cc_minor,
            cub_path=cub_path_bytes,
            thrust_path=thrust_path_bytes,
            libcudacxx_path=libcudacxx_path_bytes,
        )

        # Phase 2: Execute
        _bindings_maca.reduce_execute(
            build_handle=build_handle,
            d_in_ptr=d_in_ptr,
            d_out_ptr=d_out_ptr,
            num_items=num_items,
            type_enum=type_enum,
            op_kind=op_kind,
            init_value=init_ptr,
            dtype_size=dtype_size,
            dtype_alignment=dtype_alignment,
            stream_ptr=stream_ptr,
        )

    finally:
        # Phase 3: Cleanup
        if build_handle is not None:
            _bindings_maca.reduce_cleanup(build_handle)

# ============================================================================
# Convenience functions
# ============================================================================
sum = lambda d_in, d_out, init=0, stream=None: reduce(d_in, d_out, OpKind.SUM, init, stream)
min = lambda d_in, d_out, init=0, stream=None: reduce(d_in, d_out, OpKind.MIN, init, stream)
max = lambda d_in, d_out, init=0, stream=None: reduce(d_in, d_out, OpKind.MAX, init, stream)
