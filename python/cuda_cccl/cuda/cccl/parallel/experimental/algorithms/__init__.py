"""Parallel algorithms module."""

from .reduce import reduce, reduce_with_op, sum, min, max, OpKind, TypeEnum

__all__ = ["reduce", "reduce_with_op", "sum", "min", "max", "OpKind", "TypeEnum"]
