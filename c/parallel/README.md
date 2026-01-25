# MACA CCCL - Parallel Algorithms Library

A port of NVIDIA's CCCL (CUDA C Core Libraries) to MACA GPU platform with JIT compilation support.

## Features

- **Static Dispatch (Phase A)**: Pre-compiled support for builtin operations (SUM, MIN, MAX)
- **JIT Compilation (Phase B)**: Runtime compilation of user-defined operations via MCRTC

## Architecture

```
c/parallel/
├── include/cccl/c/
│   ├── types.h              # Type definitions (matches NVIDIA CCCL)
│   └── reduce.h             # Reduce API
├── src/
│   ├── reduce.cu            # Main implementation
│   ├── mcrtc/               # MACA runtime compiler wrappers
│   │   ├── mcrtc_helper.h   # MCRTC program/module management
│   │   ├── mcjitlink_helper.h # JIT linking support
│   │   └── command_list.h   # Compilation orchestration
│   └── jit_templates/       # JIT code generation
│       ├── mappings/        # Runtime-to-template conversions
│       │   ├── type_info.h
│       │   ├── operation.h
│       │   └── iterator.h
│       └── templates/       # Device code templates
│           ├── operation.h
│           └── iterator.h
├── build.sh                 # Build script
└── libcccl_maca.so          # Output library
```

## Building

```bash
# Set environment
export MACA_PATH=/opt/maca
export MCCUB_PATH=/path/to/mcCub

# Build the library
./build.sh
```

## Usage

### C API

```c
#include <cccl/c/reduce.h>

// Build
cccl_device_reduce_build_result_t build;
cccl_device_reduce_build(&build, d_in, d_out, op, h_init, ...);

// Execute (two-phase)
size_t temp_bytes;
cccl_device_reduce(build, NULL, &temp_bytes, ...);  // Query
cccl_device_reduce(build, temp, &temp_bytes, ...);  // Execute

// Cleanup
cccl_device_reduce_cleanup(&build);
```

### Python API

```python
from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind

# Builtin operation
reduce(d_in, d_out, op=OpKind.SUM, init=0)

# User-defined operation (JIT)
from cuda_cccl.cuda.cccl.parallel.experimental import reduce_with_op

op_source = '''
extern "C" __device__ void reduce_op_device_fn(
    void* result, const void* arg0, const void* arg1
) {
    int a = *static_cast<const int*>(arg0);
    int b = *static_cast<const int*>(arg1);
    *static_cast<int*>(result) = a > b ? a : b;
}
'''
reduce_with_op(d_in, d_out, op_source, init=0)
```

## JIT Compilation Flow

1. **Code Generation**: Generate specialized C++ code from templates
2. **MCRTC Compilation**: Compile to LLVM bitcode via `mcrtcCompileProgram`
3. **Module Loading**: Load bitcode via `mcModuleLoadData`
4. **Kernel Resolution**: Get function pointer via `mcModuleGetFunction`
5. **Execution**: Launch via `mcModuleLaunchKernel`

## Comparison with NVIDIA CCCL

| Feature | NVIDIA CCCL | MACA CCCL |
|---------|-------------|-----------|
| Runtime Compiler | NVRTC | MCRTC |
| Output Format | PTX/CUBIN | LLVM Bitcode |
| Linking | nvJitLink | mcJitLink (TBD) |
| JIT Templates | ✓ | ✓ |
| Builtin Ops | ✓ | ✓ |
| Custom Ops | ✓ | ✓ |
| Custom Iterators | ✓ | ✓ (planned) |

## License

Same license as NVIDIA CCCL (Apache 2.0 with LLVM exception)
