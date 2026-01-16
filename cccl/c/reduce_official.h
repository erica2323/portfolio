#ifndef CCCL_C_REDUCE_OFFICIAL_H
#define CCCL_C_REDUCE_OFFICIAL_H

#include "types_official.h"
#include <mc_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Build result structure for reduce
typedef struct {
    void* kernel_ptr;
    void* temp_storage;
    size_t temp_storage_bytes;
} cccl_reduce_build_result_t;

// Build reduce kernel
// Parameters:
//   build: Pointer to store the build result
//   in_type: Input data type information
//   out_type: Output data type information (usually same as in_type for reduce)
//   op: Reduction operation
mcError_t cccl_reduce_build(
    cccl_reduce_build_result_t* build,
    cccl_type_info in_type,
    cccl_type_info out_type,
    cccl_op_t op
);

// Execute reduce operation
// Parameters:
//   build: Build result from cccl_reduce_build
//   d_in: Device input array
//   d_out: Device output (single element)
//   num_items: Number of items to reduce
//   init: Initial value (optional, can be nullptr)
//   stream: MACA stream (optional, can be nullptr for default stream)
mcError_t cccl_reduce(
    cccl_reduce_build_result_t build,
    const void* d_in,
    void* d_out,
    size_t num_items,
    const void* init,
    mcStream_t stream
);

// Dispatcher based on type (internal function)
mcError_t cccl_reduce_dispatch(
    cccl_reduce_build_result_t& build,
    const void* d_in,
    void* d_out,
    size_t num_items,
    const void* init,
    cccl_type_info in_type,
    cccl_op_t op,
    mcStream_t stream
);

// Cleanup reduce resources
mcError_t cccl_reduce_cleanup(
    cccl_reduce_build_result_t build
);

#ifdef __cplusplus
}
#endif

#endif // CCCL_C_REDUCE_OFFICIAL_H
