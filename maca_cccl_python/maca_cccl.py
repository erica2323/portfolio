"""
MACA CCCL Python Binding
Python ctypes wrapper for MACA CCCL Transform API
"""

import ctypes
from ctypes import c_void_p, c_char_p, c_size_t, c_uint64, c_int32, c_float, POINTER, Structure
from enum import IntEnum
import os
import numpy as np


# ============================================================================
# Enums
# ============================================================================

class CCCLType(IntEnum):
    """CCCL data type enumeration"""
    INT8 = 0
    INT16 = 1
    INT32 = 2
    INT64 = 3
    UINT8 = 4
    UINT16 = 5
    UINT32 = 6
    UINT64 = 7
    FLOAT16 = 8
    FLOAT32 = 9
    FLOAT64 = 10
    BFLOAT16 = 11
    BOOL = 12
    UNKNOWN = 13


class CCCLOpKind(IntEnum):
    """CCCL operation kind enumeration"""
    STATELESS = 0
    STATEFUL = 1
    PLUS = 2
    MINUS = 3
    MULTIPLIES = 4
    DIVIDES = 5
    MODULUS = 6
    BIT_AND = 7
    BIT_OR = 8
    BIT_XOR = 9
    EQUAL_TO = 10
    NOT_EQUAL_TO = 11
    LESS = 12
    LESS_EQUAL = 13
    GREATER = 14
    GREATER_EQUAL = 15
    LOGICAL_AND = 16
    LOGICAL_OR = 17
    MINIMUM = 18
    MAXIMUM = 19
    NEGATE = 20
    SQUARE_ROOT = 21
    ABSOLUTE_VALUE = 22
    IDENTITY = 23


class CCCLOpCodeType(IntEnum):
    """CCCL operation code type"""
    LTOIR = 0
    CPP_SOURCE = 1


class MCError(IntEnum):
    """MACA runtime error codes"""
    SUCCESS = 0
    ERROR_INVALID_VALUE = 1
    ERROR_MEMORY_ALLOCATION = 2
    ERROR_NOT_INITIALIZED = 3
    ERROR_LAUNCH_FAILURE = 4


# ============================================================================
# Structures
# ============================================================================

class cccl_type_info(Structure):
    """Type information structure"""
    _fields_ = [
        ("type", c_int32),      # cccl_type_enum
        ("size", c_size_t),     # sizeof type
        ("alignment", c_size_t) # alignof type
    ]


class cccl_op_t(Structure):
    """Operation descriptor structure"""
    _fields_ = [
        ("type", c_int32),          # cccl_op_kind_t
        ("name", c_char_p),         # Operation name
        ("code", c_char_p),         # LTO-IR or C++ source
        ("code_size", c_size_t),    # Size of code
        ("code_type", c_int32),     # cccl_op_code_type
        ("size", c_size_t),         # sizeof operation state
        ("alignment", c_size_t),    # alignof operation state
        ("state", c_void_p)         # Pointer to state (e.g., addend for PLUS)
    ]


class cccl_transform_build_result_t(Structure):
    """Transform build result structure"""
    _fields_ = [
        ("bitcode", c_void_p),      # LLVM bitcode
        ("bitcode_size", c_size_t), # Bitcode size
        ("module", c_void_p),       # mcModule_t
        ("kernel", c_void_p),       # mcFunction_t
        ("in_type", cccl_type_info),  # Input type
        ("out_type", cccl_type_info), # Output type
        ("op", cccl_op_t)           # Operation (for execute phase)
    ]


# ============================================================================
# Load shared library
# ============================================================================

class MACACCCLError(Exception):
    """Exception for MACA CCCL errors"""
    pass


class MACARuntimeError(Exception):
    """Exception for MACA runtime errors"""
    pass


def check_error(result, func, args):
    """Error checking callback for ctypes functions"""
    if result != 0:
        raise MACARuntimeError(f"{func.__name__} returned error code: {result}")
    return result


# Load MACA runtime library
try:
    maca_path = os.environ.get('MACA_PATH', '/opt/maca')
    libmaca = ctypes.CDLL(os.path.join(maca_path, 'lib', 'libmcruntime.so'))
except OSError as e:
    raise RuntimeError(f"Failed to load MACA runtime library: {e}")

# Load CCCL transform library
try:
    lib_path = os.path.join(os.path.dirname(__file__), 'libmaca_cccl_transform.so')
    libcccl = ctypes.CDLL(lib_path)
except OSError as e:
    raise RuntimeError(f"Failed to load CCCL transform library: {e}\n"
                      f"Make sure to run build_lib.sh first!")


# ============================================================================
# MACA Runtime API
# ============================================================================

# mcMalloc
mcMalloc = libmaca.mcMalloc
mcMalloc.argtypes = [POINTER(c_void_p), c_size_t]
mcMalloc.restype = c_int32
mcMalloc.errcheck = check_error

# mcFree
mcFree = libmaca.mcFree
mcFree.argtypes = [c_void_p]
mcFree.restype = c_int32
mcFree.errcheck = check_error

# mcMemcpy
mcMemcpy = libmaca.mcMemcpy
mcMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int32]
mcMemcpy.restype = c_int32
mcMemcpy.errcheck = check_error

# Memory copy kinds
mcMemcpyHostToDevice = 1
mcMemcpyDeviceToHost = 2
mcMemcpyDeviceToDevice = 3

# mcDeviceSynchronize
mcDeviceSynchronize = libmaca.mcDeviceSynchronize
mcDeviceSynchronize.argtypes = []
mcDeviceSynchronize.restype = c_int32
mcDeviceSynchronize.errcheck = check_error


# ============================================================================
# CCCL Transform API
# ============================================================================

# cccl_transform_build
cccl_transform_build = libcccl.cccl_transform_build
cccl_transform_build.argtypes = [
    POINTER(cccl_transform_build_result_t),  # build
    cccl_type_info,                          # in_type
    cccl_type_info,                          # out_type
    cccl_op_t                                # op
]
cccl_transform_build.restype = c_int32

# cccl_transform
cccl_transform = libcccl.cccl_transform
cccl_transform.argtypes = [
    cccl_transform_build_result_t,  # build
    c_void_p,                       # d_in
    c_void_p,                       # d_out
    c_uint64,                       # num_items
    c_void_p                        # stream (NULL for default)
]
cccl_transform.restype = c_int32

# cccl_transform_cleanup
cccl_transform_cleanup = libcccl.cccl_transform_cleanup
cccl_transform_cleanup.argtypes = [cccl_transform_build_result_t]
cccl_transform_cleanup.restype = c_int32


# ============================================================================
# High-level Python API
# ============================================================================

# NumPy dtype to CCCL type mapping
NUMPY_TO_CCCL = {
    np.int8: CCCLType.INT8,
    np.int16: CCCLType.INT16,
    np.int32: CCCLType.INT32,
    np.int64: CCCLType.INT64,
    np.uint8: CCCLType.UINT8,
    np.uint16: CCCLType.UINT16,
    np.uint32: CCCLType.UINT32,
    np.uint64: CCCLType.UINT64,
    np.float32: CCCLType.FLOAT32,
    np.float64: CCCLType.FLOAT64,
    np.bool_: CCCLType.BOOL,
}


def numpy_to_cccl_type(dtype):
    """Convert NumPy dtype to CCCL type_info"""
    dtype = np.dtype(dtype)
    if dtype.type not in NUMPY_TO_CCCL:
        raise ValueError(f"Unsupported dtype: {dtype}")

    cccl_type = NUMPY_TO_CCCL[dtype.type]
    return cccl_type_info(
        type=cccl_type,
        size=dtype.itemsize,
        alignment=dtype.itemsize
    )


class Transform:
    """High-level Transform API wrapper"""

    def __init__(self, op_kind, in_dtype=np.int32, out_dtype=np.int32,
                 state=None, user_code=None):
        """
        Create a transform operation

        Args:
            op_kind: CCCLOpKind enum value (e.g., CCCLOpKind.PLUS)
            in_dtype: NumPy dtype for input (default: np.int32)
            out_dtype: NumPy dtype for output (default: np.int32)
            state: State pointer for built-in ops (e.g., addend value)
            user_code: C++ source code for custom operations
        """
        self.op_kind = op_kind
        self.in_dtype = np.dtype(in_dtype)
        self.out_dtype = np.dtype(out_dtype)
        self.build_result = None

        # Create type info
        in_type = numpy_to_cccl_type(self.in_dtype)
        out_type = numpy_to_cccl_type(self.out_dtype)

        # Create operation descriptor
        op = cccl_op_t()
        op.type = op_kind
        op.name = b"transform_op"

        if user_code is not None:
            # User-provided C++ source
            op.code = user_code.encode('utf-8')
            op.code_size = len(op.code)
            op.code_type = CCCLOpCodeType.CPP_SOURCE
        else:
            # Built-in operation
            op.code = None
            op.code_size = 0
            op.code_type = CCCLOpCodeType.LTOIR

        op.size = 0
        op.alignment = 0

        # Handle state for built-in operations
        if state is not None:
            if isinstance(state, int):
                self._state_value = c_int32(state)
            elif isinstance(state, float):
                self._state_value = c_float(state)
            else:
                raise ValueError("State must be int or float")
            op.state = ctypes.cast(ctypes.pointer(self._state_value), c_void_p)
        else:
            op.state = None

        # Build the transform
        self.build_result = cccl_transform_build_result_t()
        result = cccl_transform_build(
            ctypes.byref(self.build_result),
            in_type,
            out_type,
            op
        )

        if result != 0:
            raise MACACCCLError(f"cccl_transform_build failed with code: {result}")

    def __call__(self, input_data):
        """
        Execute transform on input data

        Args:
            input_data: NumPy array (host) or GPU device pointer

        Returns:
            NumPy array with transformed data
        """
        if not isinstance(input_data, np.ndarray):
            raise ValueError("Input must be NumPy array")

        if input_data.dtype != self.in_dtype:
            raise ValueError(f"Input dtype {input_data.dtype} doesn't match {self.in_dtype}")

        n = input_data.size

        # Allocate GPU memory
        d_in = c_void_p()
        d_out = c_void_p()

        in_size = input_data.nbytes
        out_size = n * self.out_dtype.itemsize

        mcMalloc(ctypes.byref(d_in), in_size)
        mcMalloc(ctypes.byref(d_out), out_size)

        try:
            # Copy input to device
            mcMemcpy(d_in, input_data.ctypes.data, in_size, mcMemcpyHostToDevice)

            # Execute transform
            result = cccl_transform(
                self.build_result,
                d_in,
                d_out,
                n,
                None  # Default stream
            )

            if result != 0:
                raise MACACCCLError(f"cccl_transform failed with code: {result}")

            # Synchronize
            mcDeviceSynchronize()

            # Copy result back to host
            output_data = np.empty(n, dtype=self.out_dtype)
            mcMemcpy(output_data.ctypes.data, d_out, out_size, mcMemcpyDeviceToHost)

            return output_data

        finally:
            # Free GPU memory
            mcFree(d_in)
            mcFree(d_out)

    def __del__(self):
        """Cleanup resources"""
        if self.build_result is not None:
            cccl_transform_cleanup(self.build_result)


# ============================================================================
# Convenience functions
# ============================================================================

def plus(input_data, addend):
    """Add constant to all elements: out[i] = in[i] + addend"""
    transform = Transform(CCCLOpKind.PLUS, state=addend)
    return transform(input_data)


def multiplies(input_data, multiplier):
    """Multiply all elements by constant: out[i] = in[i] * multiplier"""
    transform = Transform(CCCLOpKind.MULTIPLIES, state=multiplier)
    return transform(input_data)


def custom(input_data, code, in_dtype=np.int32, out_dtype=np.int32):
    """
    Apply custom operation defined by C++ code

    Args:
        input_data: NumPy array
        code: C++ source code defining __device__ function user_op(x)
        in_dtype: Input data type
        out_dtype: Output data type

    Returns:
        Transformed NumPy array
    """
    transform = Transform(
        CCCLOpKind.STATELESS,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        user_code=code
    )
    return transform(input_data)


if __name__ == "__main__":
    print("MACA CCCL Python Binding")
    print("=" * 50)
    print(f"Library path: {lib_path}")
    print(f"MACA path: {maca_path}")
    print("\nAvailable operations:")
    for op in CCCLOpKind:
        print(f"  - {op.name}")
