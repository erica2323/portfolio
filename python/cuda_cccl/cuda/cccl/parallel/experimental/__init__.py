"""
CCCL Parallel Algorithms - Experimental API (MACA port)

This is a simplified version without Numba support.
Only builtin operators (OpKind.SUM, MIN, MAX, etc.) are supported.
"""

from .algorithms._reduce import reduce
from ._cccl_interop import OpKind

__all__ = ['reduce', 'OpKind']
