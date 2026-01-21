"""
High-level reduce API for CCCL (MACA port)

Simplified version without Numba support.
Only supports builtin operators via OpKind enum.
"""

import numpy as np
import ctypes
from typing import Union, Optional
from .._cccl_interop import OpKind, numpy_dtype_to_type_enum, get_dtype_size, get_dtype_alignment

# Import Cython bindings
try:
    from .. import _bindings_maca
    _BINDINGS_AVAILABLE = True
except ImportError:
    _BINDINGS_AVAILABLE = False
    import warnings
    warnings.warn(
        "Cython bindings not available. Run 'python setup.py build_ext --inplace' to build them.",
        ImportWarning
    )


def _get_device_pointer(arr):
    """
    Extract device pointer from array-like object.

    Supports:
    - Objects with __cuda_array_interface__ (CuPy, PyTorch, etc.)
    - Raw integer pointers
    """
    if hasattr(arr, '__cuda_array_interface__'):
        cuda_iface = arr.__cuda_array_interface__
        return cuda_iface['data'][0], cuda_iface['shape'], cuda_iface['typestr']
    elif isinstance(arr, int):
        # Raw pointer
        return arr, None, None
    else:
        raise TypeError(
            f"Unsupported array type: {type(arr)}. "
            "Expected object with __cuda_array_interface__ or integer pointer."
        )


def reduce(
    d_in,
    d_out,
    op: Union[OpKind, int] = OpKind.SUM,
    init: Union[int, float] = 0,
    stream=None,
    num_items: Optional[int] = None
):
    """
    Compute reduction of input array.

    Parameters
    ----------
    d_in : array-like or int
        Input device array or device pointer.
        Must support __cuda_array_interface__ or be an integer pointer.

    d_out : array-like or int
        Output device array or device pointer (single element).
        Must support __cuda_array_interface__ or be an integer pointer.

    op : OpKind or int, optional
        Reduction operator. Default is OpKind.SUM.
        Supported operators:
        - OpKind.SUM: Sum reduction
        - OpKind.MIN: Minimum reduction
        - OpKind.MAX: Maximum reduction
        - OpKind.PROD: Product reduction (if supported)

    init : int or float, optional
        Initial value for reduction. Default is 0.

    stream : optional
        MACA stream for asynchronous execution. Default is None (default stream).

    num_items : int, optional
        Number of items to reduce. If not specified, inferred from d_in shape.

    Returns
    -------
    None
        Result is written to d_out.

    Examples
    --------
    >>> import cupy as cp  # or use MACA Python bindings
    >>> from cuda.cccl.parallel.experimental import reduce, OpKind
    >>>
    >>> # Create input array
    >>> d_in = cp.ones(1000, dtype=cp.int32)
    >>> d_out = cp.zeros(1, dtype=cp.int32)
    >>>
    >>> # Sum reduction
    >>> reduce(d_in, d_out, op=OpKind.SUM, init=10)
    >>> print(d_out)  # Should be 1010 (1000 ones + init value 10)
    >>>
    >>> # Min reduction
    >>> d_in = cp.array([5, 2, 8, 1, 9], dtype=cp.int32)
    >>> reduce(d_in, d_out, op=OpKind.MIN, init=0)
    >>> print(d_out)  # Should be 0 (min of init and array)

    Notes
    -----
    - This is a simplified version without Numba support
    - Only builtin operators (OpKind) are supported
    - Custom Python lambdas are NOT supported
    - Requires libcccl_maca.so to be built and accessible
    """
    if not _BINDINGS_AVAILABLE:
        raise RuntimeError(
            "Cython bindings not available. "
            "Build the extension with: python setup.py build_ext --inplace"
        )

    # Extract device pointers and metadata
    d_in_ptr, in_shape, in_typestr = _get_device_pointer(d_in)
    d_out_ptr, _, _ = _get_device_pointer(d_out)

    # Determine num_items
    if num_items is None:
        if in_shape is None:
            raise ValueError("num_items must be specified when using raw pointers")
        num_items = int(np.prod(in_shape))

    # Determine dtype
    if in_typestr is not None:
        dtype = np.dtype(in_typestr)
    elif hasattr(d_in, 'dtype'):
        dtype = d_in.dtype
    else:
        raise ValueError("Cannot determine dtype. Please use array with dtype or specify manually.")

    # Convert to CCCL types
    type_enum = int(numpy_dtype_to_type_enum(dtype))
    dtype_size = get_dtype_size(dtype)
    dtype_alignment = get_dtype_alignment(dtype)

    # Convert op to OpKind
    if isinstance(op, int):
        op = OpKind(op)
    op_kind = int(op)

    # Convert init value to C-compatible format
    init_c = dtype.type(init)  # Convert to numpy scalar of correct type
    init_ptr = ctypes.cast(
        ctypes.pointer(init_c.ctypes),
        ctypes.c_void_p
    ).value

    # Convert stream
    stream_ptr = 0
    if stream is not None:
        if hasattr(stream, 'ptr'):
            stream_ptr = stream.ptr
        elif isinstance(stream, int):
            stream_ptr = stream

    # Phase 1: Build
    build_handle = _bindings_maca.reduce_build(
        d_in_ptr=d_in_ptr,
        d_out_ptr=d_out_ptr,
        type_enum=type_enum,
        op_kind=op_kind,
        init_value=init_ptr,
        dtype_size=dtype_size,
        dtype_alignment=dtype_alignment
    )

    try:
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
            stream_ptr=stream_ptr
        )
    finally:
        # Phase 3: Cleanup
        _bindings_maca.reduce_cleanup(build_handle)


# Convenience aliases matching NVIDIA CCCL naming
sum = lambda d_in, d_out, init=0, stream=None: reduce(d_in, d_out, OpKind.SUM, init, stream)
min = lambda d_in, d_out, init=0, stream=None: reduce(d_in, d_out, OpKind.MIN, init, stream)
max = lambda d_in, d_out, init=0, stream=None: reduce(d_in, d_out, OpKind.MAX, init, stream)
