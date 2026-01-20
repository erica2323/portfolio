"""
MACA CCCL Type System
Maps Python/NumPy types to CCCL types
"""

import numpy as np
from enum import IntEnum

class CCCLType(IntEnum):
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
    # Built-in operations
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

# Type mapping
NUMPY_TO_CCCL = {
    np.int8: CCCLType.INT8,
    np.int16: CCCLType.INT16,
    np.int32: CCCLType.INT32,
    np.int64: CCCLType.INT64,
    np.uint8: CCCLType.UINT8,
    np.uint16: CCCLType.UINT16,
    np.uint32: CCCLType.UINT32,
    np.uint64: CCCLType.UINT64,
    np.float32: CCCLType.FLOAT32,
    np.float64: CCCLType.FLOAT64,
}

def get_cccl_type(dtype):
    """Convert numpy dtype to CCCL type"""
    dtype = np.dtype(dtype)
    if dtype.type not in NUMPY_TO_CCCL:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return NUMPY_TO_CCCL[dtype.type]
