# Setup Instructions for MACA CCCL Python Binding

## 📋 Prerequisites Checklist

- [ ] MACA SDK installed at known location
- [ ] Your C++ Transform implementation completed (from previous session)
- [ ] Python 3.x installed
- [ ] NumPy installed (`pip install numpy`)

## 📂 Step 1: Organize Your Files

You should have these files from your previous C++ implementation:

```
project_2026_01_12/mcCub/c/
├── parallel/
│   ├── include/
│   │   └── cccl/
│   │       └── c/
│   │           ├── types_official.h
│   │           └── transform_official.h
│   ├── src/
│   │   └── transform_official.cu
│   └── test/
│       └── test_transform_official.cpp
```

And these new Python binding files (from this session):

```
maca_cccl_python/
├── README.md
├── SETUP_INSTRUCTIONS.md (this file)
├── build_lib.sh
├── maca_cccl.py
├── test_transform.py
└── example_simple.py
```

## 🔧 Step 2: Update build_lib.sh Paths

Open `build_lib.sh` and update two critical paths:

### 2.1: Update MACA_PATH

Find this line:
```bash
export MACA_PATH="/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707"
```

Change it to **your actual MACA SDK location**:
```bash
export MACA_PATH="/YOUR/ACTUAL/PATH/TO/maca-sdk"
```

To find your MACA path:
```bash
# Try to find mxcc compiler
find /opt /usr/local /mnt -name "mxcc" 2>/dev/null | head -5

# Or check environment
echo $MACA_PATH
```

### 2.2: Update SOURCE_DIR

Find this line:
```bash
SOURCE_DIR="../project_2026_01_12/mcCub/c"
```

Change it to point to **your C++ implementation directory**:
```bash
SOURCE_DIR="/ABSOLUTE/PATH/TO/project_2026_01_12/mcCub/c"
```

**Example**: If your files are at `/home/user/work/project_2026_01_12/mcCub/c/`, use:
```bash
SOURCE_DIR="/home/user/work/project_2026_01_12/mcCub/c"
```

### 2.3: Verify Directory Structure

Make sure this file exists:
```bash
ls -l ${SOURCE_DIR}/parallel/src/transform_official.cu
```

Should show your transform implementation file.

## 🏗️ Step 3: Build Shared Library

### 3.1: Make build script executable
```bash
cd maca_cccl_python
chmod +x build_lib.sh
```

### 3.2: Run build
```bash
./build_lib.sh
```

Expected output:
```
Building shared library: libmaca_cccl_transform.so
Source: ../project_2026_01_12/mcCub/c/parallel/src/transform_official.cu
✅ Build successful: libmaca_cccl_transform.so
-rwxr-xr-x 1 user user 123456 Jan 14 12:00 libmaca_cccl_transform.so
```

### 3.3: Verify shared library
```bash
ls -lh libmaca_cccl_transform.so
file libmaca_cccl_transform.so
```

Should show:
```
libmaca_cccl_transform.so: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, ...
```

## 🌍 Step 4: Set Environment Variables

### 4.1: Add to your shell rc file

Add these lines to `~/.bashrc` or `~/.zshrc`:

```bash
# MACA SDK
export MACA_PATH="/YOUR/ACTUAL/PATH/TO/maca-sdk"
export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$LD_LIBRARY_PATH"
```

### 4.2: Reload shell config
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### 4.3: Verify environment
```bash
echo $MACA_PATH
echo $LD_LIBRARY_PATH
which mxcc
```

All should show valid paths.

## 🧪 Step 5: Run Tests

### 5.1: Simple test
```bash
cd maca_cccl_python
python3 example_simple.py
```

Expected output:
```
============================================================
MACA CCCL Python Binding - Simple Examples
============================================================

📌 Example 1: Add 10 to all elements
Input:  [1 2 3 4 5]
Output: [11 12 13 14 15]

📌 Example 2: Multiply all elements by 3
Input:  [10 20 30 40]
Output: [30 60 90 120]
...
✅ All examples completed successfully!
```

### 5.2: Full test suite
```bash
python3 test_transform.py
```

Expected output:
```
============================================================
Test 1: Built-in PLUS operation
============================================================
Input:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
Addend: 5
Output:   [ 5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
Expected: [ 5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
✅ Test 1: Built-in PLUS operation PASSED
...
============================================================
Results: 7/7 tests passed
============================================================
```

## 🐛 Troubleshooting

### Problem 1: "mxcc: command not found" during build

**Solution**: Update PATH in build_lib.sh:
```bash
export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
```

Make sure `$MACA_PATH/mxgpu_llvm/bin/mxcc` exists.

### Problem 2: "transform_official.cu: No such file or directory"

**Solution**: Check SOURCE_DIR in build_lib.sh points to correct location:
```bash
# Verify file exists
ls -l ${SOURCE_DIR}/parallel/src/transform_official.cu
```

### Problem 3: "cannot find -lmcruntime"

**Solution**: Check MACA library directory:
```bash
ls -l $MACA_PATH/lib/libmcruntime.so
```

Update LIBRARY_DIRS in build_lib.sh if needed.

### Problem 4: Python ImportError: "Failed to load CCCL transform library"

**Solution**: Make sure you're running Python from the same directory where `libmaca_cccl_transform.so` is located:
```bash
cd maca_cccl_python
python3 test_transform.py
```

Or update `lib_path` in `maca_cccl.py` to use absolute path.

### Problem 5: Python ImportError: "Failed to load MACA runtime library"

**Solution**: Set LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH="$MACA_PATH/lib:$LD_LIBRARY_PATH"
```

Verify:
```bash
ldd libmaca_cccl_transform.so | grep mcruntime
```

Should show path to libmcruntime.so.

### Problem 6: "mcMalloc failed" or GPU runtime errors

**Solution**: Check MACA GPU is accessible:
```bash
# Check if GPU is detected (if MACA has similar command to nvidia-smi)
# This is example, actual command may differ
maca-smi  # or similar MACA GPU info command
```

### Problem 7: Tests fail with wrong results

**Solution**: Verify C++ implementation is working first:
```bash
cd project_2026_01_12/mcCub/c
./test_transform_official
```

Should show all C++ tests passing before trying Python.

## ✅ Verification Checklist

After completing all steps:

- [ ] `build_lib.sh` paths updated
- [ ] `libmaca_cccl_transform.so` built successfully
- [ ] Environment variables set (`MACA_PATH`, `LD_LIBRARY_PATH`)
- [ ] `example_simple.py` runs successfully
- [ ] `test_transform.py` shows 7/7 tests passed

## 🎯 Quick Reference Commands

```bash
# Build shared library
cd maca_cccl_python
./build_lib.sh

# Set environment (adjust path)
export MACA_PATH="/path/to/maca-sdk"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$LD_LIBRARY_PATH"

# Run simple example
python3 example_simple.py

# Run full tests
python3 test_transform.py

# Use in your own code
python3 -c "from maca_cccl import plus; import numpy as np; print(plus(np.array([1,2,3], dtype=np.int32), 5))"
```

## 📞 Need Help?

If you encounter issues:

1. Check C++ implementation works: `./test_transform_official`
2. Verify MACA SDK installation: `which mxcc` and `ls $MACA_PATH/lib/libmcruntime.so`
3. Check library dependencies: `ldd libmaca_cccl_transform.so`
4. Try running with verbose Python errors: `python3 -v test_transform.py`

## 🚀 Next: Use in Your Code

Once everything works, you can use it in your own scripts:

```python
#!/usr/bin/env python3
import numpy as np
from maca_cccl import plus, multiplies, custom

# Your data
data = np.array([...], dtype=np.int32)

# Transform
result = plus(data, 100)

# Use result
print(result)
```

See `README.md` for full API documentation and more examples.
