"""
Setup script for MACA CCCL Python bindings
Builds Cython extensions and links to MACA C layer
"""

import os
import sys
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

# MACA paths - adjust to your installation
MACA_PATH = os.environ.get(
    "MACA_PATH",
    "/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707"
)
MCCUB_PATH = os.environ.get("MCCUB_PATH", "/mnt/data/minxi/1_15/mcCub")
CCCL_C_INCLUDE = os.environ.get("CCCL_C_INCLUDE", "./parallel/include")

# Check paths
if not os.path.exists(MACA_PATH):
    print(f"WARNING: MACA_PATH not found: {MACA_PATH}")
    print("Set MACA_PATH environment variable to your MACA installation")

if not os.path.exists(MCCUB_PATH):
    print(f"WARNING: MCCUB_PATH not found: {MCCUB_PATH}")
    print("Set MCCUB_PATH environment variable to your mcCub installation")

# Compiler and linker flags
include_dirs = [
    CCCL_C_INCLUDE,
    os.path.join(MACA_PATH, "include"),
    MCCUB_PATH,
    np.get_include(),
]

library_dirs = [
    os.path.join(MACA_PATH, "lib"),
    os.path.join(MACA_PATH, "lib64"),
]

libraries = ["mcruntime"]

extra_compile_args = [
    "-std=c++17",
    "-O3",
    "-fPIC",
]

extra_link_args = [
    f"-Wl,-rpath,{os.path.join(MACA_PATH, 'lib')}",
    f"-Wl,-rpath,{os.path.join(MACA_PATH, 'lib64')}",
]

# Define Cython extensions
extensions = [
    Extension(
        "maca_cccl._bindings_impl",
        sources=["_bindings_impl.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

# Read long description
long_description = """
MACA CCCL Python Bindings
=========================

GPU-accelerated parallel algorithms for MACA (Moore Threads GPU).

Features:
- Device Reduce (Sum, Min, Max)
- Type-safe Python API
- Cython-based high-performance bindings
- Compatible with NumPy arrays

Adapted from NVIDIA CUDA CCCL (CUDA C++ Core Libraries).
"""

# Setup configuration
setup(
    name="maca-cccl",
    version="0.1.0",
    description="MACA CCCL Python bindings for GPU-accelerated parallel algorithms",
    long_description=long_description,
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/maca-cccl",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "embedsignature": True,
            "boundscheck": False,
            "wraparound": False,
        },
    ),
    install_requires=[
        "numpy>=1.19",
        "cython>=0.29",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    zip_safe=False,
)

# Post-install message
print("\n" + "="*70)
print("MACA CCCL Python Bindings")
print("="*70)
print(f"MACA Path: {MACA_PATH}")
print(f"mcCub Path: {MCCUB_PATH}")
print(f"CCCL C Include: {CCCL_C_INCLUDE}")
print("\nBuild configuration:")
print(f"  Include dirs: {include_dirs}")
print(f"  Library dirs: {library_dirs}")
print(f"  Libraries: {libraries}")
print("="*70 + "\n")
