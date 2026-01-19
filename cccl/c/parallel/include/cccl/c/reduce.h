//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CCCL_C_REDUCE_H
#define CCCL_C_REDUCE_H

#include <cccl/c/types.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Reduce Build Result
//==============================================================================

/**
 * @brief Build result containing compiled kernels and configuration
 */
typedef struct {
    // Compiled code
    void* bitcode;                // LLVM bitcode or compiled binary
    size_t bitcode_size;          // Size of bitcode

    // MACA/CUDA module
    mcModule_t module;            // Loaded module
    mcFunction_t kernel;          // Main reduction kernel

    // Configuration
    cccl_iterator_t input_it;     // Input iterator descriptor
    cccl_iterator_t output_it;    // Output iterator descriptor
    cccl_op_t op;                 // Operation descriptor
    cccl_value_t init;            // Initial value
    cccl_determinism_t determinism; // Determinism mode

    // Device info
    int cc_major;                 // Compute capability major
    int cc_minor;                 // Compute capability minor
} cccl_device_reduce_build_result_t;

//==============================================================================
// Build Functions
//==============================================================================

/**
 * @brief Build (compile) a device reduce operation
 *
 * This function performs JIT compilation of a reduction kernel customized
 * for the given iterator types, operation, and initial value.
 *
 * @param build_result        Output: build result containing compiled kernel
 * @param input               Input iterator descriptor
 * @param output              Output iterator descriptor
 * @param op                  Reduction operation
 * @param init                Initial value for reduction
 * @param determinism         Determinism mode
 * @param cc_major            Compute capability major version
 * @param cc_minor            Compute capability minor version
 * @param cub_path            Path to CUB library headers (for JIT)
 * @param thrust_path         Path to Thrust library headers (for JIT)
 * @param libcudacxx_path     Path to libcu++ headers (for JIT)
 * @param ctk_path            Path to CUDA Toolkit (for JIT)
 * @return CUDA_SUCCESS on success, error code otherwise
 */
CUresult cccl_device_reduce_build(
    cccl_device_reduce_build_result_t* build_result,
    cccl_iterator_t input,
    cccl_iterator_t output,
    cccl_op_t op,
    cccl_value_t init,
    cccl_determinism_t determinism,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path
);

/**
 * @brief Extended build with custom configuration
 *
 * Same as cccl_device_reduce_build but allows passing extra compile flags
 * and include directories.
 *
 * @param build_result        Output: build result
 * @param input               Input iterator descriptor
 * @param output              Output iterator descriptor
 * @param op                  Reduction operation
 * @param init                Initial value
 * @param determinism         Determinism mode
 * @param cc_major            Compute capability major
 * @param cc_minor            Compute capability minor
 * @param cub_path            Path to CUB headers
 * @param thrust_path         Path to Thrust headers
 * @param libcudacxx_path     Path to libcu++ headers
 * @param ctk_path            Path to CUDA Toolkit
 * @param config              Build configuration (extra flags, includes)
 * @return CUDA_SUCCESS on success, error code otherwise
 */
CUresult cccl_device_reduce_build_ex(
    cccl_device_reduce_build_result_t* build_result,
    cccl_iterator_t input,
    cccl_iterator_t output,
    cccl_op_t op,
    cccl_value_t init,
    cccl_determinism_t determinism,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path,
    cccl_build_config* config
);

//==============================================================================
// Execute Functions
//==============================================================================

/**
 * @brief Execute a device reduce operation (deterministic)
 *
 * This is a two-phase API:
 * 1. Call with d_temp_storage = NULL to query temp_storage_bytes
 * 2. Allocate temp storage and call again to execute
 *
 * @param build              Build result from cccl_device_reduce_build
 * @param d_temp_storage     Temporary storage (or NULL to query size)
 * @param temp_storage_bytes Input/Output: size of temp storage
 * @param input              Input iterator
 * @param output             Output iterator
 * @param num_items          Number of items to reduce
 * @param op                 Reduction operation (must match build)
 * @param init               Initial value (must match build)
 * @param stream             CUDA stream
 * @return CUDA_SUCCESS on success, error code otherwise
 */
CUresult cccl_device_reduce(
    cccl_device_reduce_build_result_t build,
    void* d_temp_storage,
    size_t* temp_storage_bytes,
    cccl_iterator_t input,
    cccl_iterator_t output,
    uint64_t num_items,
    cccl_op_t op,
    cccl_value_t init,
    CUstream stream
);

/**
 * @brief Execute a device reduce operation (non-deterministic, faster)
 *
 * Same as cccl_device_reduce but uses non-deterministic algorithm
 * (may produce different results across runs due to floating-point
 * associativity issues, but can be faster).
 *
 * @param build              Build result from cccl_device_reduce_build
 * @param d_temp_storage     Temporary storage (or NULL to query size)
 * @param temp_storage_bytes Input/Output: size of temp storage
 * @param input              Input iterator
 * @param output             Output iterator
 * @param num_items          Number of items to reduce
 * @param op                 Reduction operation
 * @param init               Initial value
 * @param stream             CUDA stream
 * @return CUDA_SUCCESS on success, error code otherwise
 */
CUresult cccl_device_reduce_nondeterministic(
    cccl_device_reduce_build_result_t build,
    void* d_temp_storage,
    size_t* temp_storage_bytes,
    cccl_iterator_t input,
    cccl_iterator_t output,
    uint64_t num_items,
    cccl_op_t op,
    cccl_value_t init,
    CUstream stream
);

//==============================================================================
// Cleanup
//==============================================================================

/**
 * @brief Free resources associated with a build result
 *
 * @param build Build result to clean up
 * @return CUDA_SUCCESS on success, error code otherwise
 */
CUresult cccl_device_reduce_cleanup(
    cccl_device_reduce_build_result_t* build
);

#ifdef __cplusplus
}
#endif

#endif // CCCL_C_REDUCE_H
