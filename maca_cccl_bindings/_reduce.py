"""
MACA CCCL Device Reduce API
High-level Python interface for GPU reductions
"""

import numpy as np
from . import _bindings_impl
from .typing import OpKind

class DeviceReduce:
    """
    Device-level parallel reduction operations

    This class provides GPU-accelerated reduction operations using MACA CCCL.
    Operations follow a build-then-execute pattern for optimal performance.

    Example
    -------
    >>> import numpy as np
    >>> from maca_cccl import DeviceReduce
    >>>
    >>> # Create input data
    >>> data = np.arange(1000, dtype=np.int32)
    >>> d_data = maca.mem_copy_host_to_device(data)
    >>> d_out = maca.mem_alloc(4)  # Single int32
    >>>
    >>> # Build and execute sum reduction
    >>> reduce_op = DeviceReduce.sum(data.dtype)
    >>> reduce_op.compute(d_data, d_out, len(data))
    >>>
    >>> # Get result
    >>> result = maca.mem_copy_device_to_host(d_out, 1, np.int32)
    >>> print(result[0])  # Sum of 0..999
    """

    @staticmethod
    def sum(dtype, stream=None):
        """
        Build a sum reduction operation

        Parameters
        ----------
        dtype : numpy.dtype
            Data type for reduction
        stream : int, optional
            MACA stream for execution

        Returns
        -------
        DeviceReduceBuildResult
            Compiled reduction operation

        Example
        -------
        >>> reduce_sum = DeviceReduce.sum(np.int32)
        >>> reduce_sum.compute(d_in, d_out, num_items)
        """
        dtype = np.dtype(dtype)
        initial_value = dtype.type(0)
        return _bindings_impl.device_reduce_build(
            OpKind.PLUS,
            dtype,
            initial_value
        )

    @staticmethod
    def min(dtype, stream=None):
        """
        Build a minimum reduction operation

        Parameters
        ----------
        dtype : numpy.dtype
            Data type for reduction
        stream : int, optional
            MACA stream for execution

        Returns
        -------
        DeviceReduceBuildResult
            Compiled reduction operation

        Example
        -------
        >>> reduce_min = DeviceReduce.min(np.float32)
        >>> reduce_min.compute(d_in, d_out, num_items)
        """
        dtype = np.dtype(dtype)
        # Use maximum value as initial for min reduction
        if np.issubdtype(dtype, np.integer):
            initial_value = np.iinfo(dtype).max
        else:
            initial_value = np.finfo(dtype).max
        return _bindings_impl.device_reduce_build(
            OpKind.MINIMUM,
            dtype,
            initial_value
        )

    @staticmethod
    def max(dtype, stream=None):
        """
        Build a maximum reduction operation

        Parameters
        ----------
        dtype : numpy.dtype
            Data type for reduction
        stream : int, optional
            MACA stream for execution

        Returns
        -------
        DeviceReduceBuildResult
            Compiled reduction operation

        Example
        -------
        >>> reduce_max = DeviceReduce.max(np.float64)
        >>> reduce_max.compute(d_in, d_out, num_items)
        """
        dtype = np.dtype(dtype)
        # Use minimum value as initial for max reduction
        if np.issubdtype(dtype, np.integer):
            initial_value = np.iinfo(dtype).min
        else:
            initial_value = np.finfo(dtype).min
        return _bindings_impl.device_reduce_build(
            OpKind.MAXIMUM,
            dtype,
            initial_value
        )

    @staticmethod
    def reduce(d_in, d_out, num_items, op_kind, dtype, initial_value=None, stream=None):
        """
        One-shot reduction (build + execute in one call)

        Parameters
        ----------
        d_in : device pointer or array-like
            Input data on device
        d_out : device pointer or array-like
            Output location on device (single element)
        num_items : int
            Number of elements to reduce
        op_kind : OpKind
            Reduction operation (PLUS, MINIMUM, MAXIMUM)
        dtype : numpy.dtype
            Data type
        initial_value : scalar, optional
            Initial value for reduction
        stream : int, optional
            MACA stream

        Example
        -------
        >>> # Quick one-shot reduction
        >>> DeviceReduce.reduce(d_in, d_out, 1000, OpKind.PLUS, np.int32)
        """
        dtype = np.dtype(dtype)

        # Set default initial value
        if initial_value is None:
            if op_kind == OpKind.PLUS:
                initial_value = dtype.type(0)
            elif op_kind == OpKind.MINIMUM:
                initial_value = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else np.finfo(dtype).max
            elif op_kind == OpKind.MAXIMUM:
                initial_value = np.iinfo(dtype).min if np.issubdtype(dtype, np.integer) else np.finfo(dtype).min
            else:
                raise ValueError(f"Unsupported op_kind: {op_kind}")

        # Build
        reduce_op = _bindings_impl.device_reduce_build(op_kind, dtype, initial_value)

        # Execute
        d_in_ptr = _bindings_impl.get_device_pointer(d_in)
        d_out_ptr = _bindings_impl.get_device_pointer(d_out)
        reduce_op.compute(d_in_ptr, d_out_ptr, num_items, stream)

# Convenience functions
def reduce_sum(d_in, d_out, num_items, dtype, stream=None):
    """Convenience function for sum reduction"""
    return DeviceReduce.reduce(d_in, d_out, num_items, OpKind.PLUS, dtype, stream=stream)

def reduce_min(d_in, d_out, num_items, dtype, stream=None):
    """Convenience function for min reduction"""
    return DeviceReduce.reduce(d_in, d_out, num_items, OpKind.MINIMUM, dtype, stream=stream)

def reduce_max(d_in, d_out, num_items, dtype, stream=None):
    """Convenience function for max reduction"""
    return DeviceReduce.reduce(d_in, d_out, num_items, OpKind.MAXIMUM, dtype, stream=stream)
