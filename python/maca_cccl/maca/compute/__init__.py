"""
maca.compute - Device-level parallel algorithms

This module provides high-performance GPU algorithms for reduction, scan, sort, etc.
"""

from . import algorithms
from .types import TypeInfo, dtype_to_cccl_type, OpKind
from .op import make_op

# Re-export commonly used items
from .algorithms import reduce_into, make_reduce_into

__all__ = [
    "algorithms",
    "TypeInfo",
    "dtype_to_cccl_type",
    "OpKind",
    "make_op",
    "reduce_into",
    "make_reduce_into",
]
