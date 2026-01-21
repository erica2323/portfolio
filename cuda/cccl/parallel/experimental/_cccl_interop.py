"""
Type conversion utilities for MACA CCCL Python bindings.
Converts between Python types and C API types.
"""
import numpy as np
from typing import Any, Union
import operator


# Type mapping: numpy dtype -> C type enum
DTYPE_TO_C_TYPE = {
    np.dtype('int8'): 'INT8',
    np.dtype('int16'): 'INT16',
    np.dtype('int32'): 'INT32',
    np.dtype('int64'): 'INT64',
    np.dtype('uint8'): 'UINT8',
    np.dtype('uint16'): 'UINT16',
    np.dtype('uint32'): 'UINT32',
    np.dtype('uint64'): 'UINT64',
    np.dtype('float32'): 'FP32',
    np.dtype('float64'): 'FP64',
}


# Operator mapping: Python operator -> C op enum
OP_TO_C_OP = {
    operator.add: 'PLUS',
    'add': 'PLUS',
    '+': 'PLUS',
    operator.mul: 'MUL',
    'mul': 'MUL',
    '*': 'MUL',
    min: 'MIN',
    'min': 'MIN',
    max: 'MAX',
    'max': 'MAX',
}


def get_dtype_enum(dtype: np.dtype) -> str:
    """Convert numpy dtype to C type enum string."""
    dtype = np.dtype(dtype)  # Normalize
    if dtype not in DTYPE_TO_C_TYPE:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return DTYPE_TO_C_TYPE[dtype]


def get_op_enum(op: Any) -> str:
    """Convert Python operator to C op enum string."""
    if op not in OP_TO_C_OP:
        raise ValueError(f"Unsupported operator: {op}. Supported: {list(OP_TO_C_OP.keys())}")
    return OP_TO_C_OP[op]


def make_pointer_iterator(array: np.ndarray, writeable: bool = False) -> dict:
    """
    Create a pointer iterator descriptor for a numpy array.

    Args:
        array: Input numpy array
        writeable: Whether this is an output iterator (True) or input (False)

    Returns:
        dict with 'kind', 'type', and 'ptr' keys
    """
    return {
        'kind': 'POINTER',
        'type': get_dtype_enum(array.dtype),
        'ptr': array.ctypes.data,
        'writeable': writeable
    }


def make_constant_iterator(value: Any, dtype: np.dtype) -> dict:
    """
    Create a constant iterator that yields the same value.

    Args:
        value: The constant value
        dtype: The data type

    Returns:
        dict with 'kind', 'type', and 'value' keys
    """
    # Convert Python value to numpy scalar
    np_value = np.array(value, dtype=dtype)
    return {
        'kind': 'CONSTANT',
        'type': get_dtype_enum(dtype),
        'value': np_value
    }


def make_counting_iterator(start: int, dtype: np.dtype = np.int64) -> dict:
    """
    Create a counting iterator (0, 1, 2, 3, ...).

    Args:
        start: Starting value
        dtype: Integer type for the counter

    Returns:
        dict with 'kind', 'type', and 'start' keys
    """
    return {
        'kind': 'COUNTING',
        'type': get_dtype_enum(dtype),
        'start': start
    }


def get_init_value(dtype: np.dtype, op: Any, init: Any = None) -> Any:
    """
    Get the initial value for reduction.

    Args:
        dtype: The data type
        op: The reduction operator
        init: User-provided initial value, or None for default

    Returns:
        Initial value as numpy scalar
    """
    if init is not None:
        return np.array(init, dtype=dtype)

    # Default init values based on operator
    op_enum = get_op_enum(op)
    if op_enum == 'PLUS':
        return np.array(0, dtype=dtype)
    elif op_enum == 'MUL':
        return np.array(1, dtype=dtype)
    elif op_enum == 'MIN':
        # Return max value for the type
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).max
        else:
            return np.finfo(dtype).max
    elif op_enum == 'MAX':
        # Return min value for the type
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).min
        else:
            return np.finfo(dtype).min
    else:
        raise ValueError(f"Unknown operator: {op_enum}")
