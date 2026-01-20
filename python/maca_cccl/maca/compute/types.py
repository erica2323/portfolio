"""
Type system for MACA CCCL Python bindings
"""

import numpy as np
from enum import IntEnum
try:
    from . import _bindings_impl as _bindings
except ImportError:
    _bindings = None


class TypeEnum(IntEnum):
    """CCCL type enumeration"""
    INT8 = 0
    INT16 = 1
    INT32 = 2
    INT64 = 3
    UINT8 = 4
    UINT16 = 5
    UINT32 = 6
    UINT64 = 7
    FLOAT16 = 8
    FLOAT32 = 9
    FLOAT64 = 10
    STORAGE = 11
    BOOLEAN = 12


class OpKind(IntEnum):
    """Operation kind enumeration"""
    STATELESS = 0
    STATEFUL = 1
    PLUS = 2
    MINUS = 3
    MULTIPLIES = 4
    DIVIDES = 5
    MODULUS = 6
    EQUAL_TO = 7
    NOT_EQUAL_TO = 8
    GREATER = 9
    LESS = 10
    GREATER_EQUAL = 11
    LESS_EQUAL = 12
    LOGICAL_AND = 13
    LOGICAL_OR = 14
    LOGICAL_NOT = 15
    BIT_AND = 16
    BIT_OR = 17
    BIT_XOR = 18
    BIT_NOT = 19
    IDENTITY = 20
    NEGATE = 21
    MINIMUM = 22
    MAXIMUM = 23


# Mapping from NumPy dtypes to CCCL types
_DTYPE_TO_CCCL = {
    np.dtype(np.int8): (TypeEnum.INT8, 1, 1),
    np.dtype(np.int16): (TypeEnum.INT16, 2, 2),
    np.dtype(np.int32): (TypeEnum.INT32, 4, 4),
    np.dtype(np.int64): (TypeEnum.INT64, 8, 8),
    np.dtype(np.uint8): (TypeEnum.UINT8, 1, 1),
    np.dtype(np.uint16): (TypeEnum.UINT16, 2, 2),
    np.dtype(np.uint32): (TypeEnum.UINT32, 4, 4),
    np.dtype(np.uint64): (TypeEnum.UINT64, 8, 8),
    np.dtype(np.float16): (TypeEnum.FLOAT16, 2, 2),
    np.dtype(np.float32): (TypeEnum.FLOAT32, 4, 4),
    np.dtype(np.float64): (TypeEnum.FLOAT64, 8, 8),
}


class TypeInfo:
    """
    Type information for CCCL operations

    Parameters:
    -----------
    dtype : np.dtype or TypeEnum
        Data type
    size : int, optional
        Size in bytes (auto-detected if not provided)
    alignment : int, optional
        Alignment in bytes (auto-detected if not provided)
    """

    def __init__(self, dtype, size=None, alignment=None):
        if isinstance(dtype, np.dtype):
            if dtype not in _DTYPE_TO_CCCL:
                raise ValueError(f"Unsupported dtype: {dtype}")
            type_enum, default_size, default_align = _DTYPE_TO_CCCL[dtype]
            self._type_enum = type_enum
            self._size = size if size is not None else default_size
            self._alignment = alignment if alignment is not None else default_align
            self._dtype = dtype
        elif isinstance(dtype, (TypeEnum, int)):
            self._type_enum = TypeEnum(dtype)
            if size is None or alignment is None:
                raise ValueError("size and alignment must be provided for TypeEnum")
            self._size = size
            self._alignment = alignment
            # Try to reverse-lookup dtype
            self._dtype = None
            for np_dtype, (te, s, a) in _DTYPE_TO_CCCL.items():
                if te == self._type_enum and s == size and a == alignment:
                    self._dtype = np_dtype
                    break
        else:
            raise TypeError(f"dtype must be np.dtype or TypeEnum, got {type(dtype)}")

    @property
    def type_enum(self):
        return self._type_enum

    @property
    def size(self):
        return self._size

    @property
    def alignment(self):
        return self._alignment

    @property
    def dtype(self):
        return self._dtype

    def to_cython(self):
        """Convert to Cython TypeInfo object"""
        if _bindings is None:
            raise ImportError("_bindings_impl module not available")
        return _bindings.TypeInfo(self._type_enum, self._size, self._alignment)

    def __repr__(self):
        return f"TypeInfo(type={self._type_enum.name}, size={self._size}, alignment={self._alignment})"


def dtype_to_cccl_type(dtype):
    """
    Convert NumPy dtype to CCCL TypeInfo

    Parameters:
    -----------
    dtype : np.dtype
        NumPy data type

    Returns:
    --------
    TypeInfo
        CCCL type information
    """
    return TypeInfo(dtype)


def get_array_typeinfo(array):
    """
    Get type information from array

    Parameters:
    -----------
    array : array-like
        NumPy array or compatible array

    Returns:
    --------
    TypeInfo
        Type information
    """
    if hasattr(array, 'dtype'):
        return dtype_to_cccl_type(array.dtype)
    else:
        raise TypeError(f"Cannot determine dtype from {type(array)}")
