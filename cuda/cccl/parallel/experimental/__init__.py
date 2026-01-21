"""
MACA CCCL Parallel Experimental API.

This module provides experimental parallel algorithms for MACA GPUs,
matching the NVIDIA CCCL API interface.
"""

from .algorithms import reduce_into

__all__ = ['reduce_into']
