"""
Setup script for CCCL Python bindings (MACA port)
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# ============================================
# 硬编码路径 - 直接使用，不读环境变量
# ============================================

MACA_ROOT = '/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707'
CCCL_ROOT = '/mnt/data/minxi/1_15/mcCub'

print("=" * 70)
print("  CCCL Python Bindings Setup (MACA)")
print("=" * 70)
print(f"MACA_ROOT: {MACA_ROOT}")
print(f"CCCL_ROOT: {CCCL_ROOT}")
print()

# 验证关键文件
mc_runtime_h = f"{MACA_ROOT}/include/mcr/mc_runtime.h"
libcccl_maca = f"{CCCL_ROOT}/c/libcccl_maca.so"
reduce_official_h = f"{CCCL_ROOT}/c/parallel/include/cccl/c/reduce_official.h"

print("Checking files:")
print(f"  mc_runtime.h:      {mc_runtime_h}")
print(f"    exists: {os.path.exists(mc_runtime_h)}")
print(f"  reduce_official.h: {reduce_official_h}")
print(f"    exists: {os.path.exists(reduce_official_h)}")
print(f"  libcccl_maca.so:   {libcccl_maca}")
print(f"    exists: {os.path.exists(libcccl_maca)}")
print()

if not os.path.exists(mc_runtime_h):
    raise RuntimeError(f"❌ mc_runtime.h not found: {mc_runtime_h}")
if not os.path.exists(libcccl_maca):
    raise RuntimeError(f"❌ libcccl_maca.so not found: {libcccl_maca}")
if not os.path.exists(reduce_official_h):
    raise RuntimeError(f"❌ reduce_official.h not found: {reduce_official_h}")

print("✅ All files found!")
print()

# Include directories（按优先级排序）
include_dirs = [
    np.get_include(),
    f"{CCCL_ROOT}/c/parallel/include",
    f"{MACA_ROOT}/include",
    CCCL_ROOT,
]

print("Include directories:")
for d in include_dirs:
    print(f"  -I{d}")
print()

# Library directories
library_dirs = [
    f"{MACA_ROOT}/lib",
    f"{MACA_ROOT}/lib64",
    f"{CCCL_ROOT}/c",
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
    '-v',   # 显示详细编译信息
]

# Linker flags
extra_link_args = [
    f'-Wl,-rpath,{MACA_ROOT}/lib',
    f'-Wl,-rpath,{MACA_ROOT}/lib64',
    f'-Wl,-rpath,{CCCL_ROOT}/c',
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
    zip_safe=False,
)
