"""
Operation wrappers for MACA CCCL
"""

from .types import OpKind
try:
    from . import _bindings_impl as _bindings
except ImportError:
    _bindings = None


class Op:
    """
    Wrapper for CCCL operations

    This class provides a Python interface to CCCL operation objects,
    supporting both well-known operations (like PLUS, MINIMUM) and
    custom user-defined operations.

    Parameters:
    -----------
    op_kind : OpKind
        Kind of operation (e.g., OpKind.PLUS, OpKind.MINIMUM)
    name : str, optional
        Operation name for debugging
    code : str or bytes, optional
        Operation code (for custom operations)
    """

    def __init__(self, op_kind, name=None, code=None, code_type=None,
                 size=0, alignment=0, state=None):
        self._op_kind = op_kind
        self._name = name
        self._code = code
        self._code_type = code_type if code_type is not None else 0  # LTOIR
        self._size = size
        self._alignment = alignment
        self._state = state

    @property
    def op_kind(self):
        return self._op_kind

    @property
    def name(self):
        return self._name

    def to_cython(self):
        """Convert to Cython Op object"""
        if _bindings is None:
            raise ImportError("_bindings_impl module not available")

        return _bindings.Op(
            op_kind=self._op_kind,
            name=self._name,
            code=self._code,
            code_type=self._code_type,
            size=self._size,
            alignment=self._alignment,
            state=self._state
        )

    def __repr__(self):
        return f"Op(kind={OpKind(self._op_kind).name}, name={self._name})"


# Pre-defined operations
class WellKnownOps:
    """Collection of well-known operations"""

    @staticmethod
    def plus():
        """Addition operation"""
        return Op(OpKind.PLUS, name="plus")

    @staticmethod
    def minus():
        """Subtraction operation"""
        return Op(OpKind.MINUS, name="minus")

    @staticmethod
    def multiplies():
        """Multiplication operation"""
        return Op(OpKind.MULTIPLIES, name="multiplies")

    @staticmethod
    def minimum():
        """Minimum operation"""
        return Op(OpKind.MINIMUM, name="minimum")

    @staticmethod
    def maximum():
        """Maximum operation"""
        return Op(OpKind.MAXIMUM, name="maximum")


def make_op(op_kind_or_name):
    """
    Create an operation from kind or name

    Parameters:
    -----------
    op_kind_or_name : OpKind, int, or str
        Operation kind (e.g., OpKind.PLUS) or name (e.g., "plus", "sum")

    Returns:
    --------
    Op
        Operation object

    Examples:
    ---------
    >>> op = make_op(OpKind.PLUS)
    >>> op = make_op("plus")
    >>> op = make_op("sum")  # alias for plus
    """
    if isinstance(op_kind_or_name, Op):
        return op_kind_or_name

    if isinstance(op_kind_or_name, str):
        name = op_kind_or_name.lower()
        if name in ("plus", "sum", "add"):
            return WellKnownOps.plus()
        elif name in ("minus", "subtract"):
            return WellKnownOps.minus()
        elif name in ("multiplies", "multiply", "mul"):
            return WellKnownOps.multiplies()
        elif name in ("minimum", "min"):
            return WellKnownOps.minimum()
        elif name in ("maximum", "max"):
            return WellKnownOps.maximum()
        else:
            raise ValueError(f"Unknown operation name: {name}")

    if isinstance(op_kind_or_name, (OpKind, int)):
        op_kind = OpKind(op_kind_or_name)
        return Op(op_kind, name=op_kind.name.lower())

    raise TypeError(f"Cannot create Op from {type(op_kind_or_name)}")
