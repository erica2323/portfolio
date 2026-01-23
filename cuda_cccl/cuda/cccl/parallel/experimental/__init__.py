"""
CCCL Parallel Experimental API for MACA

This module provides Python bindings for NVIDIA CCCL algorithms adapted for MACA GPUs.
"""

# High-level reduce API
from ._reduce_simplified import reduce, sum, min, max

# Operator and type enums
from ._cccl_interop_simplified import OpKind, TypeEnum

# MACA runtime utilities
from ._maca_runtime import (
    MacaArray,
    malloc,
    free,
    memcpy,
    memcpy_h2d,
    memcpy_d2h,
    MemcpyKind,
    ErrorCode,
    zeros,
    ones,
    from_numpy,
    to_numpy,
)

__all__ = [
    # Reduce operations
    "reduce",
    "sum",
    "min",
    "max",

    # Enums
    "OpKind",
    "TypeEnum",

    # MACA runtime
    "MacaArray",
    "malloc",
    "free",
    "memcpy",
    "memcpy_h2d",
    "memcpy_d2h",
    "MemcpyKind",
    "ErrorCode",
    "zeros",
    "ones",
    "from_numpy",
    "to_numpy",
]
