"""
High-level reduce API for CCCL (MACA port)

Supports:
- Builtin operators (SUM, MIN, MAX)
- User-defined operators via JIT compilation
"""

import numpy as np
import ctypes
from typing import Union, Optional, Callable
from enum import IntEnum


class OpKind(IntEnum):
    """Builtin operation types."""
    SUM  = 2
    MIN  = 22
    MAX  = 23
    PROD = 4


class TypeEnum(IntEnum):
    """CCCL type enums."""
    INT8    = 0
    INT16   = 1
    INT32   = 2
    INT64   = 3
    UINT8   = 4
    UINT16  = 5
    UINT32  = 6
    UINT64  = 7
    FLOAT16 = 8
    FLOAT32 = 9
    FLOAT64 = 10


_NUMPY_TO_TYPE = {
    np.int8:    TypeEnum.INT8,
    np.int16:   TypeEnum.INT16,
    np.int32:   TypeEnum.INT32,
    np.int64:   TypeEnum.INT64,
    np.uint8:   TypeEnum.UINT8,
    np.uint16:  TypeEnum.UINT16,
    np.uint32:  TypeEnum.UINT32,
    np.uint64:  TypeEnum.UINT64,
    np.float16: TypeEnum.FLOAT16,
    np.float32: TypeEnum.FLOAT32,
    np.float64: TypeEnum.FLOAT64,
}


def numpy_dtype_to_type_enum(dtype) -> TypeEnum:
    """Convert numpy dtype to CCCL type enum."""
    return _NUMPY_TO_TYPE[np.dtype(dtype).type]


def get_dtype_size(dtype) -> int:
    """Get size of dtype in bytes."""
    return np.dtype(dtype).itemsize


def get_dtype_alignment(dtype) -> int:
    """Get alignment of dtype."""
    return np.dtype(dtype).alignment


# Try to import bindings
try:
    from .. import _bindings_maca
    _BINDINGS_AVAILABLE = True
except ImportError:
    _BINDINGS_AVAILABLE = False


def _get_device_pointer(arr):
    """Extract device pointer from array-like object."""
    if hasattr(arr, "__cuda_array_interface__"):
        cuda_iface = arr.__cuda_array_interface__
        return cuda_iface["data"][0], cuda_iface["shape"], cuda_iface["typestr"]
    elif hasattr(arr, "cuda_array_interface"):
        cuda_iface = arr.cuda_array_interface
        return cuda_iface["data"][0], cuda_iface["shape"], cuda_iface["typestr"]
    elif isinstance(arr, int):
        return arr, None, None
    else:
        raise TypeError(
            f"Unsupported array type: {type(arr)}. "
            "Expected object with __cuda_array_interface__ or integer pointer."
        )


def _make_init_scalar_ptr(dtype: np.dtype, init_value):
    """
    Create a stable C pointer to a scalar init value.
    Returns (pointer, keepalive_object).
    """
    dtype = np.dtype(dtype)

    if dtype == np.int8:
        c_obj = ctypes.c_int8(int(init_value))
    elif dtype == np.int16:
        c_obj = ctypes.c_int16(int(init_value))
    elif dtype == np.int32:
        c_obj = ctypes.c_int32(int(init_value))
    elif dtype == np.int64:
        c_obj = ctypes.c_int64(int(init_value))
    elif dtype == np.uint8:
        c_obj = ctypes.c_uint8(int(init_value))
    elif dtype == np.uint16:
        c_obj = ctypes.c_uint16(int(init_value))
    elif dtype == np.uint32:
        c_obj = ctypes.c_uint32(int(init_value))
    elif dtype == np.uint64:
        c_obj = ctypes.c_uint64(int(init_value))
    elif dtype == np.float32:
        c_obj = ctypes.c_float(float(init_value))
    elif dtype == np.float64:
        c_obj = ctypes.c_double(float(init_value))
    else:
        raise ValueError(f"Unsupported dtype for init: {dtype}")

    return ctypes.addressof(c_obj), c_obj


def reduce(
    d_in,
    d_out,
    op: Union[OpKind, int] = OpKind.SUM,
    init: Union[int, float] = 0,
    stream=None,
    num_items: Optional[int] = None,
):
    """
    Reduce array with builtin operation.

    Args:
        d_in: Input device array
        d_out: Output device array (single element)
        op: Reduction operation (OpKind.SUM, OpKind.MIN, OpKind.MAX)
        init: Initial value for reduction
        stream: MACA stream (optional)
        num_items: Number of items (inferred from d_in if not specified)

    Example:
        >>> reduce(d_in, d_out, op=OpKind.SUM, init=0)
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


def reduce_with_op(
    d_in,
    d_out,
    op_source: str,
    init: Union[int, float] = 0,
    stream=None,
    num_items: Optional[int] = None,
    cub_path: str = "",
    ctk_path: str = "",
):
    """
    Reduce array with user-defined operation via JIT compilation.

    The operation source should define a device function with the signature:
        extern "C" __device__ void reduce_op_device_fn(
            void* result,
            const void* arg0,
            const void* arg1
        );

    Args:
        d_in: Input device array
        d_out: Output device array (single element)
        op_source: C++ source code for the reduction operation
        init: Initial value for reduction
        stream: MACA stream (optional)
        num_items: Number of items (inferred from d_in if not specified)
        cub_path: Path to mcCub headers (for JIT compilation)
        ctk_path: Path to MACA toolkit

    Example:
        >>> op_source = '''
        ... extern "C" __device__ void reduce_op_device_fn(
        ...     void* result, const void* arg0, const void* arg1
        ... ) {
        ...     int a = *static_cast<const int*>(arg0);
        ...     int b = *static_cast<const int*>(arg1);
        ...     *static_cast<int*>(result) = a > b ? a : b;  // custom max
        ... }
        ... '''
        >>> reduce_with_op(d_in, d_out, op_source, init=0)
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

    init_ptr, _init_keepalive = _make_init_scalar_ptr(dtype, init)

    stream_ptr = 0
    if stream is not None:
        if hasattr(stream, "ptr"):
            stream_ptr = int(stream.ptr)
        elif isinstance(stream, int):
            stream_ptr = int(stream)

    # Convert paths to bytes
    cub_path_bytes = cub_path.encode('utf-8') if isinstance(cub_path, str) else cub_path
    ctk_path_bytes = ctk_path.encode('utf-8') if isinstance(ctk_path, str) else ctk_path
    op_source_bytes = op_source.encode('utf-8') if isinstance(op_source, str) else op_source

    build_handle = None
    try:
        # Phase 1: Build with JIT
        build_handle = _bindings_maca.reduce_build_custom(
            d_in_ptr=d_in_ptr,
            d_out_ptr=d_out_ptr,
            type_enum=type_enum,
            op_source=op_source_bytes,
            init_value=init_ptr,
            dtype_size=dtype_size,
            dtype_alignment=dtype_alignment,
            cub_path=cub_path_bytes,
            ctk_path=ctk_path_bytes,
        )

        # Phase 2: Execute
        _bindings_maca.reduce_execute(
            build_handle=build_handle,
            d_in_ptr=d_in_ptr,
            d_out_ptr=d_out_ptr,
            num_items=num_items,
            type_enum=type_enum,
            op_kind=int(OpKind.SUM),  # Placeholder, actual op from JIT
            init_value=init_ptr,
            dtype_size=dtype_size,
            dtype_alignment=dtype_alignment,
            stream_ptr=stream_ptr,
        )

    finally:
        # Phase 3: Cleanup
        if build_handle is not None:
            _bindings_maca.reduce_cleanup(build_handle)


# Convenience functions
def sum(d_in, d_out, init=0, stream=None, num_items=None):
    """Sum reduction."""
    return reduce(d_in, d_out, OpKind.SUM, init, stream, num_items)


def min(d_in, d_out, init=0, stream=None, num_items=None):
    """Min reduction."""
    return reduce(d_in, d_out, OpKind.MIN, init, stream, num_items)


def max(d_in, d_out, init=0, stream=None, num_items=None):
    """Max reduction."""
    return reduce(d_in, d_out, OpKind.MAX, init, stream, num_items)
