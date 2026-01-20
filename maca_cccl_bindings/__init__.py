"""
MACA CCCL Python Bindings
GPU-accelerated parallel algorithms for MACA (Moore Threads)
"""

from .typing import CCCLType, OpKind
from ._reduce import DeviceReduce, reduce_sum, reduce_min, reduce_max

__version__ = "0.1.0"

__all__ = [
    # Type system
    "CCCLType",
    "OpKind",

    # Algorithms
    "DeviceReduce",
    "reduce_sum",
    "reduce_min",
    "reduce_max",
]

# Version info
def get_version():
    """Get MACA CCCL version"""
    return __version__
