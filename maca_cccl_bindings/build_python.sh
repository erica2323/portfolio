#!/bin/bash

#==============================================================================
# MACA CCCL Python Bindings Build Script
# Builds Cython extensions and installs Python package
#==============================================================================

set -e

echo "=============================================================="
echo "  MACA CCCL Python Bindings - Build Script"
echo "=============================================================="
echo ""

# Set MACA paths (adjust to your installation)
export MACA_PATH="${MACA_PATH:-/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707}"
export MCCUB_PATH="${MCCUB_PATH:-/mnt/data/minxi/1_15/mcCub}"
export CCCL_C_INCLUDE="${CCCL_C_INCLUDE:-./parallel/include}"

echo "Configuration:"
echo "  MACA_PATH: $MACA_PATH"
echo "  MCCUB_PATH: $MCCUB_PATH"
echo "  CCCL_C_INCLUDE: $CCCL_C_INCLUDE"
echo ""

# Check dependencies
echo "Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi
echo "  ✅ Python: $(python3 --version)"

if ! python3 -c "import numpy" 2>/dev/null; then
    echo "❌ Error: NumPy not installed"
    echo "   Install with: pip install numpy"
    exit 1
fi
echo "  ✅ NumPy installed"

if ! python3 -c "import Cython" 2>/dev/null; then
    echo "❌ Error: Cython not installed"
    echo "   Install with: pip install cython"
    exit 1
fi
echo "  ✅ Cython installed"

# Check MACA installation
if [ ! -d "$MACA_PATH" ]; then
    echo "❌ Error: MACA_PATH not found: $MACA_PATH"
    exit 1
fi
echo "  ✅ MACA found"

if [ ! -d "$MCCUB_PATH" ]; then
    echo "❌ Error: MCCUB_PATH not found: $MCCUB_PATH"
    exit 1
fi
echo "  ✅ mcCub found"

echo ""
echo "=============================================================="
echo " Step 1: Clean previous builds"
echo "=============================================================="
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -f _bindings_impl.cpp
rm -f _bindings_impl*.so
echo "✅ Cleaned"

echo ""
echo "=============================================================="
echo " Step 2: Build Cython extension"
echo "=============================================================="
python3 setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✅ Cython extension built successfully"
else
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "=============================================================="
echo " Step 3: Install package (optional)"
echo "=============================================================="
read -p "Install package? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 setup.py install --user
    echo "✅ Package installed"
else
    echo "Skipped installation"
fi

echo ""
echo "=============================================================="
echo " Build Complete!"
echo "=============================================================="
echo ""
echo "Test the bindings:"
echo "  python3 example_reduce.py"
echo ""
echo "Or use in your code:"
echo "  from maca_cccl import DeviceReduce"
echo "  reduce_op = DeviceReduce.sum(np.int32)"
echo "  reduce_op.compute(d_in, d_out, num_items)"
echo ""
echo "=============================================================="
