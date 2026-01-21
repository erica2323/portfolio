"""
Setup script for CCCL Python bindings (MACA port)

Build instructions:
    python setup.py build_ext --inplace

This will generate _bindings_maca.so in cuda_cccl/cuda/cccl/parallel/experimental/
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Paths - ADJUST THESE FOR YOUR SYSTEM
MACA_ROOT = os.environ.get('MACA_PATH', '/usr/local/maca')
CCCL_ROOT = os.environ.get('CCCL_PATH', '/path/to/cccl')

# Include directories
include_dirs = [
    np.get_include(),
    f"{CCCL_ROOT}/parallel/include",      # For cccl/c/*.h
    f"{MACA_ROOT}/include",                # For mc_runtime.h
]

# Library directories
library_dirs = [
    f"{MACA_ROOT}/lib64",                  # For libmcruntime.so
    f"{CCCL_ROOT}/build",                  # For libcccl_maca.so (or wherever you built it)
]

# Libraries to link
libraries = [
    'mcruntime',
    'cccl_maca',
]

# Compiler flags
extra_compile_args = [
    '-std=c++14',
    '-D__MACA__',
    '-DCCCL_C_EXPERIMENTAL',
]

# Define Cython extension
extensions = [
    Extension(
        name="cuda_cccl.cuda.cccl.parallel.experimental._bindings_maca",
        sources=["cuda_cccl/cuda/cccl/parallel/experimental/_bindings_maca.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        language="c++",
    )
]

setup(
    name="cuda-cccl-maca",
    version="0.1.0",
    description="CCCL Python bindings for MACA (simplified, no Numba)",
    author="CCCL MACA Port",
    packages=[
        "cuda_cccl",
        "cuda_cccl.cuda",
        "cuda_cccl.cuda.cccl",
        "cuda_cccl.cuda.cccl.parallel",
        "cuda_cccl.cuda.cccl.parallel.experimental",
        "cuda_cccl.cuda.cccl.parallel.experimental.algorithms",
    ],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'embedsignature': True,
        }
    ),
    install_requires=[
        'numpy>=1.20',
        'cython>=0.29',
    ],
    zip_safe=False,
)
