#!/bin/bash

# Build script for MACA CCCL Reduce API test
# Requires MACA SDK to be installed

set -e

echo "Building MACA CCCL Reduce API..."

# Check if mcc is available
if ! command -v mcc &> /dev/null; then
    echo "Error: mcc compiler not found. Please install MACA SDK."
    echo "Expected location: /opt/maca/bin/mcc"
    exit 1
fi

# Set MACA include paths
MACA_INCLUDE="/opt/maca/include"
MACA_LIB="/opt/maca/lib"

# Compile the reduce test program
echo "Compiling test_reduce_official..."
mcc -I. -I${MACA_INCLUDE} \
    test_reduce_official.cpp \
    cccl/c/reduce_official.cu \
    -o test_reduce_official \
    -L${MACA_LIB} -lmcrt \
    -std=c++14

echo "Build successful!"
echo "Run with: ./test_reduce_official"
