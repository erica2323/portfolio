//==============================================================================
//
// MACA CCCL C Binding - Reduce API
// Adapted for MACA - Phase 5 (JIT + mcCub Integration - True CCCL Style)
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

// Build result: stores compiled kernels and runtime state
typedef struct
{
    void* bitcode;                     // Compiled LLVM bitcode
    size_t bitcode_size;               // Bitcode size in bytes
    mcModule_t module;                 // MACA module
    mcFunction_t reduce_kernel;        // JIT-compiled kernel that uses mcCub
    cccl_type_info type;               // Data type
    cccl_op_t op;                      // Operation
    void* initial_value;               // Initial value for reduction
    size_t initial_value_size;         // Size of initial value
    cccl_iterator_t d_in_iterator;     // Input iterator info
} cccl_device_reduce_build_result_t;

//==============================================================================
// Helper Functions
//==============================================================================

/**
 * Create a pointer iterator from a device pointer
 */
static inline cccl_iterator_t cccl_make_pointer_iterator(
    void* ptr,
    cccl_type_info type
) {
    cccl_iterator_t it;
    it.type = CCCL_POINTER;
    it.state = ptr;
    it.value_type = type;
    it.size = 0;
    it.alignment = 0;
    it.advance.type = CCCL_STATELESS;
    it.advance.code = nullptr;
    it.advance.code_size = 0;
    it.dereference.type = CCCL_STATELESS;
    it.dereference.code = nullptr;
    it.dereference.code_size = 0;
    it.host_advance = nullptr;
    return it;
}

//==============================================================================
// Build Functions - JIT Compilation Phase
//==============================================================================

/**
 * Build phase: JIT compile reduce kernel with mcCub (iterator version)
 * This generates source code that #includes mcCub headers and compiles it
 *
 * @param build         Output: compilation result
 * @param op            Reduction operation (CCCL_PLUS, CCCL_MINIMUM, CCCL_MAXIMUM)
 * @param d_in          Input iterator
 * @param initial_value Pointer to initial value
 * @param build_config  Build configuration (optional, can be NULL)
 * @return mcSuccess or error code
 */
mcError_t cccl_device_reduce_build_ex(
    cccl_device_reduce_build_result_t* build,
    cccl_op_t op,
    cccl_iterator_t d_in,
    void* initial_value,
    cccl_build_config* build_config
);

/**
 * Build phase: JIT compile reduce kernel with mcCub (pointer version - backward compatible)
 */
mcError_t cccl_device_reduce_build(
    cccl_device_reduce_build_result_t* build,
    cccl_op_t op,
    cccl_type_info type,
    void* initial_value,
    cccl_build_config* build_config
);

//==============================================================================
// Execute Functions - mcCub Execution Phase
//==============================================================================

/**
 * Execute phase: Run reduction using mcCub (iterator version)
 * Uses the JIT-compiled kernel that internally calls mcCub
 *
 * @param build      Compilation result
 * @param d_in       Input iterator
 * @param d_out      Output data (GPU pointer, single value)
 * @param num_items  Number of items to reduce
 * @param stream     MACA stream (can be NULL)
 * @return mcSuccess or error code
 */
mcError_t cccl_device_reduce_ex(
    cccl_device_reduce_build_result_t build,
    cccl_iterator_t d_in,
    void* d_out,
    uint64_t num_items,
    mcStream_t stream
);

/**
 * Execute phase: Run reduction using mcCub (pointer version - backward compatible)
 */
mcError_t cccl_device_reduce(
    cccl_device_reduce_build_result_t build,
    void* d_in,
    void* d_out,
    uint64_t num_items,
    mcStream_t stream
);

//==============================================================================
// Cleanup
//==============================================================================

/**
 * Cleanup phase: Free resources
 */
mcError_t cccl_device_reduce_cleanup(
    cccl_device_reduce_build_result_t* build
);

#ifdef __cplusplus
}
#endif

#endif // CCCL_C_REDUCE_H
