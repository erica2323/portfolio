"""
Setup script for MACA CCCL Python bindings.

This script compiles the Cython bindings and creates a Python package
that can be installed with pip.

Usage:
    python setup.py build_ext --inplace
    python setup.py install
"""

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os


# Detect MACA installation paths
def find_maca_paths():
    """Find MACA installation paths."""
    maca_path = os.environ.get('MACA_PATH', '/opt/maca')

    # Common MACA paths
    include_dirs = [
        os.path.join(maca_path, 'include'),
        np.get_include(),
    ]

    library_dirs = [
        os.path.join(maca_path, 'lib'),
        os.path.join(maca_path, 'lib64'),
    ]

    # Check for libmcruntime.so
    runtime_lib = None
    for lib_dir in library_dirs:
        lib_path = os.path.join(lib_dir, 'libmcruntime.so')
        if os.path.exists(lib_path):
            runtime_lib = lib_dir
            break

    if runtime_lib:
        library_dirs.insert(0, runtime_lib)

    return include_dirs, library_dirs


# Detect CCCL C library paths
def find_cccl_paths():
    """Find CCCL C library installation."""
    # Assume parallel/include and parallel/lib from Phase A
    cccl_base = os.environ.get('CCCL_PATH', os.path.dirname(os.path.abspath(__file__)))

    include_dirs = [
        os.path.join(cccl_base, 'parallel', 'include'),
    ]

    library_dirs = [
        os.path.join(cccl_base, 'parallel', 'lib'),
        os.path.join(cccl_base, 'parallel', 'build'),  # Alternative build location
    ]

    return include_dirs, library_dirs


# Get all paths
maca_includes, maca_libs = find_maca_paths()
cccl_includes, cccl_libs = find_cccl_paths()

include_dirs = maca_includes + cccl_includes
library_dirs = maca_libs + cccl_libs

print("Include directories:", include_dirs)
print("Library directories:", library_dirs)


# Define the Cython extension
extensions = [
    Extension(
        name="cuda.cccl.parallel.experimental._bindings_maca",
        sources=["cuda/cccl/parallel/experimental/_bindings_maca.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["mcruntime", "cccl_maca"],  # Link against MACA runtime and CCCL C lib
        language="c++",
        extra_compile_args=[
            "-std=c++14",
            "-D__MACA__",
            "-DCCCL_C_EXPERIMENTAL",
        ],
        extra_link_args=[
            "-Wl,-rpath,$ORIGIN/../../../../parallel/lib",  # Runtime library search path
        ],
    )
]


setup(
    name="maca-cccl",
    version="0.1.0",
    description="MACA CCCL Python bindings",
    author="MACA CCCL Team",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'embedsignature': True,
        }
    ),
    install_requires=[
        "numpy>=1.20.0",
        "cython>=0.29.0",
    ],
    python_requires=">=3.7",
    zip_safe=False,
)
