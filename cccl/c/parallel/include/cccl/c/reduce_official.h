//==============================================================================
//
// MACA CCCL C Binding - Reduce API
// Adapted for MACA - Phase 4 (mcCub Integration)
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

// Build result: stores configuration for mcCub calls
typedef struct
{
    cccl_type_info type;               // Data type
    cccl_op_t op;                      // Operation
    void* initial_value;               // Initial value for reduction
    size_t initial_value_size;         // Size of initial value
    cccl_iterator_t d_in_iterator;     // Input iterator info (for iterator mode)
} cccl_device_reduce_build_result_t;

//==============================================================================
// Helper Functions
//==============================================================================

/**
 * Create a pointer iterator from a device pointer
 * This is a convenience function for the common case of reducing a contiguous array
 *
 * @param ptr       Device pointer
 * @param type      Element type information
 * @return Iterator wrapping the pointer
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
// Build Functions
//==============================================================================

/**
 * Build phase: Save reduce configuration (iterator version)
 * Note: With mcCub integration, no JIT compilation is needed
 *
 * @param build         Output: configuration result
 * @param op            Reduction operation (CCCL_PLUS, CCCL_MINIMUM, CCCL_MAXIMUM)
 * @param d_in          Input iterator
 * @param initial_value Pointer to initial value
 * @param build_config  Build configuration (optional, can be NULL - ignored in mcCub mode)
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
 * Build phase: Save reduce configuration (pointer version - backward compatible)
 * Note: With mcCub integration, no JIT compilation is needed
 *
 * @param build         Output: configuration result
 * @param op            Reduction operation
 * @param type          Data type
 * @param initial_value Pointer to initial value
 * @param build_config  Build configuration (optional, can be NULL - ignored in mcCub mode)
 * @return mcSuccess or error code
 */
mcError_t cccl_device_reduce_build(
    cccl_device_reduce_build_result_t* build,
    cccl_op_t op,
    cccl_type_info type,
    void* initial_value,
    cccl_build_config* build_config
);

//==============================================================================
// Execute Functions
//==============================================================================

/**
 * Execute phase: Run reduction (iterator version)
 *
 * @param build      Compilation result
 * @param d_in       Input iterator (must match the one used in build)
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
 * Execute phase: Run reduction (pointer version - backward compatible)
 *
 * @param build      Compilation result
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

//==============================================================================
// Cleanup
//==============================================================================

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
