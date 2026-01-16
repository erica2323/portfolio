//==============================================================================
//
// MACA CCCL C Binding - Reduce API
// Adapted for MACA - Phase 1 (Pointer + PLUS only)
//
//==============================================================================

#ifndef CCCL_C_REDUCE_H
#define CCCL_C_REDUCE_H

#include <cccl/c/types_official.h>
#include <mc_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Reduce API
//==============================================================================

// Build result: stores compiled kernels
typedef struct
{
    void* bitcode;                     // Compiled LLVM bitcode
    size_t bitcode_size;               // Bitcode size in bytes
    mcModule_t module;                 // MACA module
    mcFunction_t single_tile_kernel;   // Kernel for small data (single tile)
    mcFunction_t reduction_kernel;     // Kernel for large data (multi-tile)
    cccl_type_info type;               // Data type
    cccl_op_t op;                      // Operation
    void* initial_value;               // Initial value for reduction
    size_t initial_value_size;         // Size of initial value
} cccl_device_reduce_build_result_t;

/**
 * Build phase: Compile reduce kernels
 *
 * @param build         Output: compilation result
 * @param op            Reduction operation (Phase 1: only CCCL_PLUS supported)
 * @param type          Data type (int32, float32, etc.)
 * @param initial_value Pointer to initial value (e.g., 0 for sum)
 * @param build_config  Build configuration (optional, can be NULL)
 * @return mcSuccess or error code
 */
mcError_t cccl_device_reduce_build(
    cccl_device_reduce_build_result_t* build,
    cccl_op_t op,
    cccl_type_info type,
    void* initial_value,
    cccl_build_config* build_config
);

/**
 * Execute phase: Run reduction
 *
 * @param build      Compilation result (from cccl_device_reduce_build)
 * @param d_in       Input data (GPU pointer)
 * @param d_out      Output data (GPU pointer, single value)
 * @param num_items  Number of items to reduce
 * @param stream     MACA stream (can be NULL)
 * @return mcSuccess or error code
 */
mcError_t cccl_device_reduce(
    cccl_device_reduce_build_result_t build,
    void* d_in,
    void* d_out,
    uint64_t num_items,
    mcStream_t stream
);

/**
 * Cleanup phase: Free resources
 *
 * @param build Compilation result
 * @return mcSuccess or error code
 */
mcError_t cccl_device_reduce_cleanup(
    cccl_device_reduce_build_result_t* build
);

#ifdef __cplusplus
}
#endif

#endif // CCCL_C_REDUCE_H
