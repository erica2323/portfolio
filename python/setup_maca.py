"""
Setup script for CCCL Python bindings (MACA port)
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

#============================================
# Configuration - adjust for your environment
#============================================

# Try to get paths from environment, with defaults
MACA_ROOT = os.environ.get('MACA_PATH', '/opt/maca')
CCCL_ROOT = os.environ.get('CCCL_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CCCL_C_ROOT = os.path.join(CCCL_ROOT, 'c', 'parallel')

print("=" * 70)
print(" CCCL Python Bindings Setup (MACA)")
print("=" * 70)
print(f"MACA_ROOT: {MACA_ROOT}")
print(f"CCCL_ROOT: {CCCL_ROOT}")
print(f"CCCL_C_ROOT: {CCCL_C_ROOT}")
print()

# Verify key files exist
mc_runtime_h = os.path.join(MACA_ROOT, "include", "mcr", "mc_runtime.h")
libcccl_maca = os.path.join(CCCL_C_ROOT, "libcccl_maca.so")
reduce_h = os.path.join(CCCL_C_ROOT, "include", "cccl", "c", "reduce.h")

print("Checking files:")
print(f"  mc_runtime.h: {mc_runtime_h}")
print(f"    exists: {os.path.exists(mc_runtime_h)}")
print(f"  reduce.h: {reduce_h}")
print(f"    exists: {os.path.exists(reduce_h)}")
print(f"  libcccl_maca.so: {libcccl_maca}")
print(f"    exists: {os.path.exists(libcccl_maca)}")
print()

# Warn if files don't exist (but don't fail - they might be created later)
missing = []
if not os.path.exists(mc_runtime_h):
    missing.append(f"mc_runtime.h not found: {mc_runtime_h}")
if not os.path.exists(reduce_h):
    missing.append(f"reduce.h not found: {reduce_h}")
if not os.path.exists(libcccl_maca):
    missing.append(f"libcccl_maca.so not found: {libcccl_maca}")

if missing:
    print("⚠️  Missing files (build may fail):")
    for m in missing:
        print(f"    {m}")
    print()
else:
    print("✅ All files found!")
    print()

# Include directories
include_dirs = [
    np.get_include(),
    os.path.join(CCCL_C_ROOT, "include"),
    os.path.join(MACA_ROOT, "include"),
    CCCL_ROOT,
]

print("Include directories:")
for d in include_dirs:
    print(f"  -I{d}")
print()

# Library directories
library_dirs = [
    os.path.join(MACA_ROOT, "lib"),
    os.path.join(MACA_ROOT, "lib64"),
    CCCL_C_ROOT,
]

print("Library directories:")
for d in library_dirs:
    print(f"  -L{d}")
print()

# Libraries to link
libraries = [
    'mcruntime',
    'cccl_maca',
]

# Compiler flags
extra_compile_args = [
    '-std=c++17',
    '-D__MACA__',
    '-DCCCL_C_EXPERIMENTAL',
]

# Linker flags
extra_link_args = [
    f'-Wl,-rpath,{os.path.join(MACA_ROOT, "lib")}',
    f'-Wl,-rpath,{os.path.join(MACA_ROOT, "lib64")}',
    f'-Wl,-rpath,{CCCL_C_ROOT}',
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
        extra_link_args=extra_link_args,
        language="c++",
    )
]

print("Building extension: cuda_cccl.cuda.cccl.parallel.experimental._bindings_maca")
print("=" * 70)
print()

setup(
    name="cuda-cccl-maca",
    version="0.1.0",
    description="CCCL Python bindings for MACA with JIT support",
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
    python_requires='>=3.8',
    zip_safe=False,
)
