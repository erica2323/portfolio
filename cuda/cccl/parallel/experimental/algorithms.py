"""
High-level Python API for MACA CCCL parallel algorithms.
Provides reduce_into() function matching NVIDIA CCCL API.
"""
import numpy as np
from typing import Optional, Any, Union
import hashlib

try:
    from . import _bindings_maca as bindings
except ImportError:
    # Fallback if Cython module not compiled yet
    bindings = None

from . import _cccl_interop as interop


class _Reduce:
    """
    Manages reduction operations with build result caching.
    """
    def __init__(self):
        self._build_cache = {}  # key -> DeviceReduceBuildResult

    def _make_cache_key(self, dtype_enum: str, op_enum: str,
                        in_kind: str, out_kind: str) -> str:
        """Create a cache key for build results."""
        return f"{dtype_enum}_{op_enum}_{in_kind}_{out_kind}"

    def reduce_into(self,
                    d_in: Union[np.ndarray, dict],
                    d_out: Union[np.ndarray, dict],
                    op: Any,
                    h_init: Optional[Any] = None,
                    stream=None) -> None:
        """
        Performs a device-wide reduction operation.

        Args:
            d_in: Input data. Can be:
                  - numpy array (uses pointer iterator)
                  - dict with iterator descriptor (for advanced iterators)
            d_out: Output data. Can be:
                   - numpy array (single element, uses pointer iterator)
                   - dict with iterator descriptor
            op: Reduction operator. Can be:
                - Python function (operator.add, min, max, etc.)
                - String ('add', '+', 'min', 'max', etc.)
            h_init: Initial value for reduction. If None, uses default for operator.
            stream: MACA stream (not yet supported)

        Example:
            >>> import numpy as np
            >>> from cuda.cccl.parallel.experimental import reduce_into
            >>>
            >>> d_in = np.ones(1000, dtype=np.int32)
            >>> d_out = np.zeros(1, dtype=np.int32)
            >>> reduce_into(d_in, d_out, 'add', h_init=10)
            >>> print(d_out[0])  # Should be 1010
        """
        if bindings is None:
            raise ImportError(
                "Cython bindings not available. "
                "Please compile _bindings_maca.pyx first using setup.py"
            )

        # Convert inputs to iterator descriptors
        if isinstance(d_in, np.ndarray):
            d_in_desc = interop.make_pointer_iterator(d_in, writeable=False)
            dtype = d_in.dtype
            num_items = len(d_in)
        else:
            d_in_desc = d_in
            dtype = np.dtype(d_in['type'].lower())  # Convert enum to numpy dtype
            if 'num_items' not in d_in:
                raise ValueError("Iterator dict must include 'num_items' key")
            num_items = d_in['num_items']

        if isinstance(d_out, np.ndarray):
            d_out_desc = interop.make_pointer_iterator(d_out, writeable=True)
        else:
            d_out_desc = d_out

        # Get type and operator enums
        dtype_enum = interop.get_dtype_enum(dtype)
        op_enum = interop.get_op_enum(op)

        # Get init value
        init_value = interop.get_init_value(dtype, op, h_init)

        # Check cache for compiled kernel
        cache_key = self._make_cache_key(
            dtype_enum, op_enum,
            d_in_desc['kind'], d_out_desc['kind']
        )

        if cache_key not in self._build_cache:
            # Build (compile) the kernel
            build_result = bindings.DeviceReduceBuildResult()
            build_result.build(
                d_in_desc, d_out_desc, op_enum,
                init_value, dtype_enum,
                cc_major=8, cc_minor=0  # MACA default compute capability
            )
            self._build_cache[cache_key] = build_result
        else:
            build_result = self._build_cache[cache_key]

        # Execute the reduction
        build_result.reduce(
            d_in_desc, d_out_desc, num_items,
            op_enum, init_value, dtype_enum,
            stream=stream
        )


# Global singleton instance
_reduce_singleton = _Reduce()


def reduce_into(d_in: Union[np.ndarray, dict],
                d_out: Union[np.ndarray, dict],
                op: Any,
                h_init: Optional[Any] = None,
                stream=None) -> None:
    """
    Performs a device-wide reduction operation.

    This is the main entry point for the reduce API, matching NVIDIA CCCL's interface.

    Args:
        d_in: Input data (numpy array or iterator descriptor dict)
        d_out: Output data (numpy array or iterator descriptor dict)
        op: Reduction operator (function, string, or callable)
        h_init: Initial value for reduction (optional)
        stream: MACA stream (optional, not yet supported)

    Example:
        >>> import numpy as np
        >>> from cuda.cccl.parallel.experimental import reduce_into
        >>>
        >>> # Basic reduction
        >>> d_in = np.ones(1000, dtype=np.int32)
        >>> d_out = np.zeros(1, dtype=np.int32)
        >>> reduce_into(d_in, d_out, 'add')
        >>> print(d_out[0])  # 1000
        >>>
        >>> # With initial value
        >>> reduce_into(d_in, d_out, 'add', h_init=10)
        >>> print(d_out[0])  # 1010
        >>>
        >>> # Using Python operator
        >>> import operator
        >>> reduce_into(d_in, d_out, operator.add, h_init=10)
        >>> print(d_out[0])  # 1010
    """
    _reduce_singleton.reduce_into(d_in, d_out, op, h_init, stream)


__all__ = ['reduce_into']
