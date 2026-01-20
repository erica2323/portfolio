"""
MACA CCCL - Python bindings for MACA Core Compute Libraries

This package provides Pythonic access to MACA's high-performance GPU algorithms.
"""

__version__ = "0.1.0"

# Sub-packages
from . import compute

__all__ = ["compute", "__version__"]
