"""
CCCL Parallel Experimental API for MACA

This module provides GPU-accelerated parallel algorithms with support for
both builtin and user-defined operations via JIT compilation.
"""

from .algorithms.reduce import (
    reduce,
    reduce_with_op,
    sum,
    min,
    max,
    OpKind,
    TypeEnum,
)

__all__ = [
    "reduce",
    "reduce_with_op",
    "sum",
    "min",
    "max",
    "OpKind",
    "TypeEnum",
]
