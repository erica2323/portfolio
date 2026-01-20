"""
maca.compute.algorithms - High-level GPU algorithm implementations
"""

from ._reduce import reduce_into, make_reduce_into

__all__ = ["reduce_into", "make_reduce_into"]
