"""
MACA Runtime Wrapper

Provides safe Python bindings to libmcruntime.so with correct ctypes signatures.

CRITICAL: ctypes requires explicit argtypes/restype declarations to handle
          pointers correctly on 64-bit systems. Without these, pointer
          arguments get truncated to 32-bit ints, causing segfaults.
"""

import ctypes
import sys
import numpy as np
from typing import Optional


# =============================================================================
# Load MACA Runtime Library
# =============================================================================

try:
    _maca_runtime = ctypes.CDLL("libmcruntime.so")
except OSError as e:
    raise RuntimeError(
        "Failed to load libmcruntime.so. "
        "Please set LD_LIBRARY_PATH to include MACA lib64 directory.\n"
        f"Error: {e}"
    ) from e


# =============================================================================
# Function Signature Declarations (CRITICAL!)
# =============================================================================

# mcError_t mcMalloc(void** devPtr, size_t size)
_maca_runtime.mcMalloc.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # void** devPtr
    ctypes.c_size_t                   # size_t size
]
_maca_runtime.mcMalloc.restype = ctypes.c_int

# mcError_t mcFree(void* devPtr)
_maca_runtime.mcFree.argtypes = [ctypes.c_void_p]  # void* devPtr
_maca_runtime.mcFree.restype = ctypes.c_int

# mcError_t mcMemcpy(void* dst, const void* src, size_t count, mcMemcpyKind kind)
_maca_runtime.mcMemcpy.argtypes = [
    ctypes.c_void_p,   # void* dst
    ctypes.c_void_p,   # const void* src
    ctypes.c_size_t,   # size_t count
    ctypes.c_int       # mcMemcpyKind kind
]
_maca_runtime.mcMemcpy.restype = ctypes.c_int

# mcError_t mcStreamSynchronize(mcStream_t stream)
if hasattr(_maca_runtime, "mcStreamSynchronize"):
    _maca_runtime.mcStreamSynchronize.argtypes = [ctypes.c_void_p]
    _maca_runtime.mcStreamSynchronize.restype = ctypes.c_int


# =============================================================================
# Constants
# =============================================================================

class MemcpyKind:
    """Memory copy direction constants"""
    HostToHost = 0
    HostToDevice = 1
    DeviceToHost = 2
    DeviceToDevice = 3
    Default = 4


class ErrorCode:
    """MACA error codes"""
    Success = 0
    InvalidValue = 1
    MemoryAllocation = 2
    # Add more as needed


# =============================================================================
# Low-Level API Functions
# =============================================================================

def malloc(size: int) -> int:
    """
    Allocate device memory.

    Args:
        size: Number of bytes to allocate

    Returns:
        Device pointer as integer

    Raises:
        RuntimeError: If allocation fails
    """
    dev_ptr = ctypes.c_void_p(None)
    ret = _maca_runtime.mcMalloc(ctypes.byref(dev_ptr), ctypes.c_size_t(size))

    if ret != ErrorCode.Success or not dev_ptr.value:
        raise RuntimeError(
            f"mcMalloc failed: error code {ret}, "
            f"requested {size} bytes, got ptr={dev_ptr.value}"
        )

    return int(dev_ptr.value)


def free(dev_ptr: int) -> None:
    """
    Free device memory.

    Args:
        dev_ptr: Device pointer (as integer)

    Raises:
        RuntimeError: If free fails
    """
    if dev_ptr == 0:
        return  # Null pointer, nothing to free

    ret = _maca_runtime.mcFree(ctypes.c_void_p(dev_ptr))
    if ret != ErrorCode.Success:
        raise RuntimeError(f"mcFree failed: error code {ret}")


def memcpy(dst: int, src: int, count: int, kind: int) -> None:
    """
    Copy memory between host and device.

    Args:
        dst: Destination pointer (as integer)
        src: Source pointer (as integer)
        count: Number of bytes to copy
        kind: Copy direction (MemcpyKind.HostToDevice, etc.)

    Raises:
        RuntimeError: If copy fails
    """
    ret = _maca_runtime.mcMemcpy(
        ctypes.c_void_p(dst),
        ctypes.c_void_p(src),
        ctypes.c_size_t(count),
        ctypes.c_int(kind)
    )

    if ret != ErrorCode.Success:
        raise RuntimeError(f"mcMemcpy failed: error code {ret}")


def memcpy_h2d(dev_ptr: int, host_array: np.ndarray) -> None:
    """
    Copy numpy array from host to device.

    Args:
        dev_ptr: Device pointer (as integer)
        host_array: NumPy array (must be contiguous)
    """
    host_array = np.ascontiguousarray(host_array)
    memcpy(
        dst=dev_ptr,
        src=int(host_array.ctypes.data),
        count=host_array.nbytes,
        kind=MemcpyKind.HostToDevice
    )


def memcpy_d2h(host_array: np.ndarray, dev_ptr: int, nbytes: Optional[int] = None) -> None:
    """
    Copy data from device to host numpy array.

    Args:
        host_array: NumPy array (must be contiguous)
        dev_ptr: Device pointer (as integer)
        nbytes: Number of bytes to copy (defaults to host_array.nbytes)
    """
    if nbytes is None:
        nbytes = host_array.nbytes

    if nbytes > host_array.nbytes:
        raise ValueError(
            f"Cannot copy {nbytes} bytes into array with {host_array.nbytes} bytes"
        )

    memcpy(
        dst=int(host_array.ctypes.data),
        src=dev_ptr,
        count=nbytes,
        kind=MemcpyKind.DeviceToHost
    )


def synchronize_stream(stream: int = 0) -> None:
    """
    Synchronize a MACA stream.

    Args:
        stream: Stream handle (0 for default stream)

    Raises:
        RuntimeError: If synchronization fails
    """
    if hasattr(_maca_runtime, "mcStreamSynchronize"):
        ret = _maca_runtime.mcStreamSynchronize(ctypes.c_void_p(stream))
        if ret != ErrorCode.Success:
            raise RuntimeError(f"mcStreamSynchronize failed: error code {ret}")


# =============================================================================
# High-Level Array Wrapper
# =============================================================================

class MacaArray:
    """
    High-level wrapper for MACA device arrays.

    Provides __cuda_array_interface__ for interoperability with CCCL and other
    libraries that support the CUDA Array Interface protocol.

    Example:
        >>> arr = MacaArray(1000, dtype=np.float32)
        >>> arr.copy_to_device(np.ones(1000, dtype=np.float32))
        >>> result = np.zeros(1000, dtype=np.float32)
        >>> arr.copy_to_host(result)
    """

    def __init__(self, shape, dtype):
        """
        Allocate device array.

        Args:
            shape: Array shape (int or tuple)
            dtype: NumPy dtype
        """
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape))
        self.nbytes = self.size * self.dtype.itemsize

        # Allocate device memory
        self.ptr = malloc(self.nbytes)

    def __del__(self):
        """Free device memory (best-effort, never raises)"""
        try:
            if hasattr(self, "ptr") and self.ptr != 0:
                free(self.ptr)
                self.ptr = 0
        except Exception:
            # Never raise from __del__
            pass

    @property
    def __cuda_array_interface__(self):
        """
        CUDA Array Interface v3 descriptor.

        Allows this object to be used with libraries that support CAI,
        including CCCL, CuPy, Numba, etc.
        """
        return {
            "version": 3,
            "shape": self.shape,
            "typestr": self.dtype.str,
            "data": (self.ptr, False),  # (ptr, read_only)
            "strides": None,
        }

    def copy_to_device(self, host_array: np.ndarray) -> None:
        """
        Copy host array to device.

        Args:
            host_array: NumPy array to copy

        Raises:
            ValueError: If array shape/dtype doesn't match
        """
        host_array = np.ascontiguousarray(host_array, dtype=self.dtype)

        if host_array.nbytes != self.nbytes:
            raise ValueError(
                f"Size mismatch: host array has {host_array.nbytes} bytes, "
                f"device array has {self.nbytes} bytes"
            )

        memcpy_h2d(self.ptr, host_array)

    def copy_to_host(self, host_array: np.ndarray) -> None:
        """
        Copy device data to host array.

        Args:
            host_array: NumPy array to copy into

        Raises:
            ValueError: If array dtype/size doesn't match
        """
        if host_array.dtype != self.dtype:
            raise ValueError(
                f"dtype mismatch: host array is {host_array.dtype}, "
                f"device array is {self.dtype}"
            )

        if host_array.nbytes != self.nbytes:
            raise ValueError(
                f"Size mismatch: host array has {host_array.nbytes} bytes, "
                f"device array has {self.nbytes} bytes"
            )

        memcpy_d2h(host_array, self.ptr)

    def __repr__(self):
        return f"MacaArray(shape={self.shape}, dtype={self.dtype}, ptr=0x{self.ptr:x})"


# =============================================================================
# Convenience Functions
# =============================================================================

def zeros(shape, dtype=np.float32):
    """Create a device array initialized to zeros"""
    arr = MacaArray(shape, dtype)
    host_zeros = np.zeros(shape, dtype=dtype)
    arr.copy_to_device(host_zeros)
    return arr


def ones(shape, dtype=np.float32):
    """Create a device array initialized to ones"""
    arr = MacaArray(shape, dtype)
    host_ones = np.ones(shape, dtype=dtype)
    arr.copy_to_device(host_ones)
    return arr


def from_numpy(host_array: np.ndarray):
    """Create a device array from a NumPy array"""
    arr = MacaArray(host_array.shape, host_array.dtype)
    arr.copy_to_device(host_array)
    return arr


def to_numpy(device_array: MacaArray) -> np.ndarray:
    """Copy a device array to a new NumPy array"""
    host_array = np.empty(device_array.shape, dtype=device_array.dtype)
    device_array.copy_to_host(host_array)
    return host_array
