"""
MACA Compute API.

Convenience module that re-exports reduce_into at the top level.
"""

from cuda.cccl.parallel.experimental import reduce_into

__all__ = ['reduce_into']
