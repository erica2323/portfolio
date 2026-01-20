#!/bin/bash

# MACA CCCL Python Bindings Build Script

set -e

echo "=============================================================="
echo "  MACA CCCL Python Bindings Build"
echo "=============================================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MACA_PATH="${MACA_PATH:-/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707}"
MCCUB_PATH="${MCCUB_PATH:-/mnt/data/minxi/1_15/mcCub}"
PARALLEL_PATH="$(pwd)/../../parallel"

echo "Configuration:"
echo "  MACA_PATH:     $MACA_PATH"
echo "  MCCUB_PATH:    $MCCUB_PATH"
echo "  PARALLEL_PATH: $PARALLEL_PATH"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python:${NC} $(python3 --version)"

# Check MACA
if [ ! -d "$MACA_PATH" ]; then
    echo -e "${RED}❌ Error: MACA not found at $MACA_PATH${NC}"
    echo "   Set MACA_PATH environment variable"
    exit 1
fi
echo -e "${GREEN}✅ MACA:${NC} $MACA_PATH"

# Check mcCub
if [ ! -d "$MCCUB_PATH" ]; then
    echo -e "${YELLOW}⚠️  Warning: mcCub not found at $MCCUB_PATH${NC}"
    echo "   Set MCCUB_PATH environment variable"
fi

# Check parallel headers
if [ ! -d "$PARALLEL_PATH/include" ]; then
    echo -e "${RED}❌ Error: CCCL C headers not found at $PARALLEL_PATH/include${NC}"
    exit 1
fi
echo -e "${GREEN}✅ CCCL C headers:${NC} $PARALLEL_PATH/include"

# Check Cython
if ! python3 -c "import Cython" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Cython not found, installing...${NC}"
    pip install cython>=3.0
fi
echo -e "${GREEN}✅ Cython:${NC} $(python3 -c 'import Cython; print(Cython.__version__)')"

echo ""
echo "=============================================================="
echo " Building Python Extension"
echo "=============================================================="
echo ""

# Export paths for CMake
export MACA_PATH
export MCCUB_PATH

# Build options
BUILD_TYPE="${BUILD_TYPE:-Release}"
VERBOSE="${VERBOSE:-0}"

if [ "$1" == "clean" ]; then
    echo "Cleaning build artifacts..."
    rm -rf build dist *.egg-info
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.so" -delete
    find . -type f -name "*.cpp" -path "*/maca/compute/*" -delete
    echo -e "${GREEN}✅ Clean complete${NC}"
    exit 0
fi

if [ "$1" == "develop" ] || [ "$1" == "dev" ]; then
    echo "Installing in development mode..."
    pip install -e . -v
    echo ""
    echo -e "${GREEN}✅ Development installation complete${NC}"
    echo ""
    echo "You can now:"
    echo "  import maca.compute"
    echo "  from maca.compute import reduce_into"
    exit 0
fi

if [ "$1" == "wheel" ]; then
    echo "Building wheel..."
    pip install build
    python3 -m build --wheel
    echo ""
    echo -e "${GREEN}✅ Wheel built successfully${NC}"
    echo ""
    ls -lh dist/*.whl
    exit 0
fi

if [ "$1" == "install" ]; then
    echo "Building and installing..."
    pip install .
    echo ""
    echo -e "${GREEN}✅ Installation complete${NC}"
    exit 0
fi

if [ "$1" == "test" ]; then
    echo "Running tests..."
    pip install -e .
    pip install pytest pytest-xdist
    pytest tests/ -v
    exit 0
fi

# Default: show help
echo "Usage: $0 [command]"
echo ""
echo "Commands:"
echo "  clean     - Remove build artifacts"
echo "  develop   - Install in development mode (editable)"
echo "  wheel     - Build wheel package"
echo "  install   - Build and install"
echo "  test      - Run tests"
echo ""
echo "Environment variables:"
echo "  MACA_PATH      - Path to MACA installation"
echo "  MCCUB_PATH     - Path to mcCub"
echo "  BUILD_TYPE     - Build type (Debug/Release, default: Release)"
echo "  VERBOSE        - Verbose build output (0/1, default: 0)"
echo ""
echo "Examples:"
echo "  $0 clean            # Clean build"
echo "  $0 develop          # Development install"
echo "  $0 wheel            # Build wheel"
echo "  $0 install          # Install package"
echo "  $0 test             # Run tests"
