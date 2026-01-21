"""
CCCL Interop Layer - Simplified Version (No Numba)

This is a simplified version of NVIDIA CCCL's _cccl_interop.py that:
- Does NOT use Numba for JIT compilation
- Only supports builtin operators via OpKind enum
- Cannot compile custom Python lambdas to GPU code

For MACA compatibility without Numba dependency.
"""

from enum import IntEnum
from typing import Union
import numpy as np


class OpKind(IntEnum):
    """
    Builtin reduction operators.
    These map directly to cccl_op_kind_t in C API.
    Values match actual CCCL C API from types_official.h
    """
    SUM = 2     # CCCL_PLUS
    MIN = 22    # CCCL_MINIMUM
    MAX = 23    # CCCL_MAXIMUM
    PROD = 4    # CCCL_MULTIPLIES


class TypeEnum(IntEnum):
    """
    Type enumeration matching cccl_type_enum in C API.
    """
    INT8 = 0
    INT16 = 1
    INT32 = 2
    INT64 = 3
    UINT8 = 4
    UINT16 = 5
    UINT32 = 6
    UINT64 = 7
    FLOAT32 = 8
    FLOAT64 = 9


# Mapping from NumPy dtypes to TypeEnum
_NUMPY_TO_TYPE_ENUM = {
    np.int8: TypeEnum.INT8,
    np.int16: TypeEnum.INT16,
    np.int32: TypeEnum.INT32,
    np.int64: TypeEnum.INT64,
    np.uint8: TypeEnum.UINT8,
    np.uint16: TypeEnum.UINT16,
    np.uint32: TypeEnum.UINT32,
    np.uint64: TypeEnum.UINT64,
    np.float32: TypeEnum.FLOAT32,
    np.float64: TypeEnum.FLOAT64,
}


def numpy_dtype_to_type_enum(dtype) -> TypeEnum:
    """Convert NumPy dtype to CCCL TypeEnum."""
    dtype = np.dtype(dtype)
    if dtype.type in _NUMPY_TO_TYPE_ENUM:
        return _NUMPY_TO_TYPE_ENUM[dtype.type]
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_dtype_size(dtype) -> int:
    """Get size of dtype in bytes."""
    return np.dtype(dtype).itemsize


def get_dtype_alignment(dtype) -> int:
    """Get alignment of dtype in bytes."""
    return np.dtype(dtype).alignment
