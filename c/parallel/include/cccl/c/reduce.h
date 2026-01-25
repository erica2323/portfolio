//==============================================================================
// MACA CCCL C Binding - Reduce API
// Adapted for MACA with JIT support
//==============================================================================

#ifndef CCCL_C_REDUCE_H
#define CCCL_C_REDUCE_H

#include <cccl/c/types.h>
#include <mcr/mc_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Build Result Structure
//==============================================================================

typedef struct cccl_device_reduce_build_result_t
{
    // Configuration from build
    cccl_type_info   type;               // Data type
    cccl_op_t        op;                 // Operation
    void*            initial_value;      // Initial value for reduction
    size_t           initial_value_size; // Size of initial value
    cccl_iterator_t  d_in_iterator;      // Input iterator info

    // JIT compilation results (Phase B)
    void*            jit_module;         // mcModule_t (if JIT used)
    void*            jit_kernel;         // mcFunction_t (if JIT used)
    int              uses_jit;           // 1 if JIT-compiled, 0 for static dispatch
    void*            jit_bitcode;        // Compiled bitcode (for caching)
    size_t           jit_bitcode_size;   // Size of bitcode
} cccl_device_reduce_build_result_t;

//==============================================================================
// Helper Functions
//==============================================================================

/**
 * Create a pointer iterator from a device pointer
 */
static inline cccl_iterator_t cccl_make_pointer_iterator(
    void*          ptr,
    cccl_type_info type
) {
    cccl_iterator_t it;
    it.type               = CCCL_POINTER;
    it.state              = ptr;
    it.value_type         = type;
    it.size               = 0;
    it.alignment          = 0;
    it.advance.type       = CCCL_STATELESS;
    it.advance.code       = NULL;
    it.advance.code_size  = 0;
    it.dereference.type   = CCCL_STATELESS;
    it.dereference.code   = NULL;
    it.dereference.code_size = 0;
    it.host_advance       = NULL;
    return it;
}

/**
 * Create a builtin operation (e.g., CCCL_PLUS, CCCL_MINIMUM)
 */
static inline cccl_op_t cccl_make_builtin_op(cccl_op_kind_t kind) {
    cccl_op_t op;
    op.type       = kind;
    op.name       = NULL;
    op.code       = NULL;
    op.code_size  = 0;
    op.code_type  = CCCL_OP_BITCODE;
    op.size       = 0;
    op.alignment  = 1;
    op.state      = NULL;
    return op;
}

/**
 * Create a user-defined operation from C++ source
 */
static inline cccl_op_t cccl_make_source_op(
    const char* cpp_source,
    size_t      source_size,
    size_t      value_size,
    size_t      value_alignment
) {
    cccl_op_t op;
    op.type       = CCCL_STATELESS;
    op.name       = NULL;
    op.code       = cpp_source;
    op.code_size  = source_size;
    op.code_type  = CCCL_OP_CPP_SOURCE;
    op.size       = value_size;
    op.alignment  = value_alignment;
    op.state      = NULL;
    return op;
}

/**
 * Create a user-defined operation from pre-compiled bitcode
 */
static inline cccl_op_t cccl_make_bitcode_op(
    const char* bitcode,
    size_t      bitcode_size,
    size_t      value_size,
    size_t      value_alignment
) {
    cccl_op_t op;
    op.type       = CCCL_STATELESS;
    op.name       = NULL;
    op.code       = bitcode;
    op.code_size  = bitcode_size;
    op.code_type  = CCCL_OP_BITCODE;
    op.size       = value_size;
    op.alignment  = value_alignment;
    op.state      = NULL;
    return op;
}

//==============================================================================
// API Functions
//==============================================================================

/**
 * Build phase: Configure and compile reduction operation
 *
 * For builtin operations (CCCL_PLUS, etc.): uses static dispatch (fast)
 * For user-defined operations: uses JIT compilation via MCRTC
 *
 * @param build      Output: build result structure
 * @param d_in       Input iterator
 * @param d_out      Output iterator
 * @param op         Reduction operation
 * @param h_init     Initial value (host pointer)
 * @param cc_major   Compute capability major (unused for MACA)
 * @param cc_minor   Compute capability minor (unused for MACA)
 * @param cub_path   Path to mcCub headers (for JIT)
 * @param thrust_path Path to thrust headers (for JIT, optional)
 * @param libcudacxx_path Path to libcu++ headers (for JIT, optional)
 * @param ctk_path   Path to MACA toolkit (for JIT)
 *
 * @return mcSuccess on success, error code otherwise
 */
mcError_t cccl_device_reduce_build(
    cccl_device_reduce_build_result_t* build,
    cccl_iterator_t                    d_in,
    cccl_iterator_t                    d_out,
    cccl_op_t                          op,
    cccl_value_t                       h_init,
    int                                cc_major,
    int                                cc_minor,
    const char*                        cub_path,
    const char*                        thrust_path,
    const char*                        libcudacxx_path,
    const char*                        ctk_path
);

/**
 * Compute phase: Execute the reduction
 *
 * Two-phase execution:
 *   1. If temp_storage == NULL: query required temp storage bytes
 *   2. If temp_storage != NULL: execute using provided temp storage
 *
 * @param build              Build result from cccl_device_reduce_build
 * @param temp_storage       Temporary storage (NULL for query)
 * @param temp_storage_bytes Input/output: temp storage size
 * @param d_in               Input iterator
 * @param d_out              Output iterator
 * @param num_items          Number of items to reduce
 * @param op                 Reduction operation
 * @param h_init             Initial value
 * @param stream             MACA stream
 *
 * @return mcSuccess on success, error code otherwise
 */
mcError_t cccl_device_reduce(
    cccl_device_reduce_build_result_t build,
    void*                             temp_storage,
    size_t*                           temp_storage_bytes,
    cccl_iterator_t                   d_in,
    cccl_iterator_t                   d_out,
    uint64_t                          num_items,
    cccl_op_t                         op,
    cccl_value_t                      h_init,
    mcStream_t                        stream
);

/**
 * Cleanup phase: Free resources
 *
 * @param build  Build result to cleanup
 *
 * @return mcSuccess on success, error code otherwise
 */
mcError_t cccl_device_reduce_cleanup(
    cccl_device_reduce_build_result_t* build
);

#ifdef __cplusplus
}
#endif

#endif // CCCL_C_REDUCE_H
