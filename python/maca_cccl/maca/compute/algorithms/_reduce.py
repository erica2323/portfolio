"""
Device-wide reduction operations

This module provides high-level Python interfaces to MACA CCCL's
device-wide reduction algorithms.
"""

import numpy as np
from functools import lru_cache
from ..types import TypeInfo, get_array_typeinfo, OpKind
from ..op import make_op
try:
    from .. import _bindings_impl as _bindings
except ImportError:
    _bindings = None


def _get_device_pointer(array):
    """
    Extract device pointer from array-like object

    Supports:
    - NumPy arrays (returns data pointer, but warns if not on device)
    - Arrays with __cuda_array_interface__ or similar
    - Objects with data_ptr() method
    """
    # Check for CUDA array interface
    if hasattr(array, '__cuda_array_interface__'):
        return array.__cuda_array_interface__['data'][0]

    # Check for data_ptr method (PyTorch-style)
    if hasattr(array, 'data_ptr'):
        return array.data_ptr()

    # Check for ctypes data pointer
    if hasattr(array, 'ctypes') and hasattr(array.ctypes, 'data'):
        import warnings
        warnings.warn(
            "Using NumPy array pointer - ensure data is on GPU device!",
            UserWarning
        )
        return array.ctypes.data

    # Try to get raw pointer
    if hasattr(array, 'data') and hasattr(array.data, 'ptr'):
        return array.data.ptr

    raise TypeError(f"Cannot extract device pointer from {type(array)}")


def _get_scalar_value(value, dtype):
    """Convert Python value to appropriate numpy scalar"""
    return np.dtype(dtype).type(value)


class _Reduce:
    """
    Internal class that wraps a compiled reduction operation

    This follows the NVIDIA CCCL pattern:
    1. Build phase: Compile the operation with types
    2. Execute phase: Run on actual data
    """

    def __init__(self, op, d_in_typeinfo, initial_value):
        """
        Build the reduce operation

        Parameters:
        -----------
        op : Op
            Reduction operation
        d_in_typeinfo : TypeInfo
            Input data type
        initial_value : scalar
            Initial value for reduction
        """
        if _bindings is None:
            raise ImportError("MACA CCCL bindings not available")

        self.op = op
        self.typeinfo = d_in_typeinfo
        self.initial_value = initial_value

        # Build phase: compile the operation
        self.build_result = _bindings.device_reduce_build(
            op.to_cython(),
            d_in_typeinfo.to_cython(),
            initial_value
        )

    def __call__(self, d_in, d_out, stream=None):
        """
        Execute the reduction

        Parameters:
        -----------
        d_in : array-like
            Input data on device
        d_out : array-like
            Output buffer on device (single element)
        stream : int, optional
            MACA stream handle

        Returns:
        --------
        None
            Result is written to d_out
        """
        # Get device pointers
        d_in_ptr = _get_device_pointer(d_in)
        d_out_ptr = _get_device_pointer(d_out)

        # Get number of items
        if hasattr(d_in, 'size'):
            num_items = d_in.size
        elif hasattr(d_in, '__len__'):
            num_items = len(d_in)
        else:
            raise ValueError("Cannot determine size of input array")

        # Get stream handle
        stream_handle = stream if stream is not None else 0

        # Execute reduction
        _bindings.device_reduce(
            self.build_result,
            d_in_ptr,
            d_out_ptr,
            num_items,
            stream_handle
        )


# Cache compiled operations by (op_kind, dtype)
@lru_cache(maxsize=128)
def _make_reduce_into_cached(op_kind, dtype_str, initial_value):
    """
    Cached factory for reduce operations

    This caching avoids recompiling the same operation multiple times.
    """
    dtype = np.dtype(dtype_str)
    typeinfo = TypeInfo(dtype)
    op = make_op(op_kind)
    initial_val = _get_scalar_value(initial_value, dtype)

    return _Reduce(op, typeinfo, initial_val)


def make_reduce_into(d_in, op="sum", initial_value=None):
    """
    Create a reusable reduction operation (build phase)

    This follows the CCCL pattern: separate build and execute phases.
    The build phase compiles the operation, which can then be executed
    multiple times on different data.

    Parameters:
    -----------
    d_in : array-like
        Sample input array (used to determine type)
    op : str or Op, optional
        Operation to perform. Can be:
        - "sum" or "plus" (default)
        - "min" or "minimum"
        - "max" or "maximum"
        - An Op object
    initial_value : scalar, optional
        Initial value for reduction (default: 0 for sum, inf/-inf for min/max)

    Returns:
    --------
    _Reduce
        Compiled reduction operation that can be called with (d_in, d_out)

    Examples:
    ---------
    >>> import numpy as np
    >>> from maca.compute import make_reduce_into
    >>>
    >>> # Build phase (compile once)
    >>> reduce_op = make_reduce_into(np.zeros(100, dtype=np.float32), op="sum")
    >>>
    >>> # Execute phase (run many times)
    >>> reduce_op(d_input, d_output)  # First dataset
    >>> reduce_op(d_input2, d_output2)  # Second dataset
    """
    # Get type information
    typeinfo = get_array_typeinfo(d_in)

    # Convert op to OpKind
    if isinstance(op, str):
        op_obj = make_op(op)
        op_kind = op_obj.op_kind
    elif isinstance(op, int):
        op_kind = OpKind(op)
    else:
        op_kind = op.op_kind

    # Determine initial value if not provided
    if initial_value is None:
        if op_kind == OpKind.PLUS:
            initial_value = 0
        elif op_kind == OpKind.MINIMUM:
            # Use dtype max
            if typeinfo.dtype == np.float32:
                initial_value = np.finfo(np.float32).max
            elif typeinfo.dtype == np.float64:
                initial_value = np.finfo(np.float64).max
            else:
                initial_value = np.iinfo(typeinfo.dtype).max
        elif op_kind == OpKind.MAXIMUM:
            # Use dtype min
            if typeinfo.dtype == np.float32:
                initial_value = np.finfo(np.float32).min
            elif typeinfo.dtype == np.float64:
                initial_value = np.finfo(np.float64).min
            else:
                initial_value = np.iinfo(typeinfo.dtype).min
        else:
            initial_value = 0

    # Use cached factory
    return _make_reduce_into_cached(
        op_kind,
        str(typeinfo.dtype),
        initial_value
    )


def reduce_into(d_in, d_out, op="sum", initial_value=None, stream=None):
    """
    Perform device-wide reduction (single-call interface)

    This is a convenience function that combines build and execute phases.
    For repeated reductions on the same type, use make_reduce_into() to
    build once and execute multiple times.

    Parameters:
    -----------
    d_in : array-like
        Input data on device
    d_out : array-like
        Output buffer on device (single element, will receive result)
    op : str or Op, optional
        Operation to perform (default: "sum")
    initial_value : scalar, optional
        Initial value for reduction
    stream : int, optional
        MACA stream handle

    Returns:
    --------
    None
        Result is written to d_out

    Examples:
    ---------
    >>> import numpy as np
    >>> from maca.compute import reduce_into
    >>>
    >>> # Allocate device memory (pseudo-code, use your device memory API)
    >>> d_input = device_array(1000, dtype=np.float32)
    >>> d_output = device_array(1, dtype=np.float32)
    >>>
    >>> # Perform reduction
    >>> reduce_into(d_input, d_output, op="sum")
    >>>
    >>> # Result is now in d_output
    >>> result = d_output.copy_to_host()[0]
    """
    # Build operation
    reduce_op = make_reduce_into(d_in, op=op, initial_value=initial_value)

    # Execute operation
    reduce_op(d_in, d_out, stream=stream)


# Convenience functions for common operations
def sum_into(d_in, d_out, stream=None):
    """Compute sum reduction"""
    return reduce_into(d_in, d_out, op="sum", stream=stream)


def min_into(d_in, d_out, stream=None):
    """Compute minimum reduction"""
    return reduce_into(d_in, d_out, op="min", stream=stream)


def max_into(d_in, d_out, stream=None):
    """Compute maximum reduction"""
    return reduce_into(d_in, d_out, op="max", stream=stream)
