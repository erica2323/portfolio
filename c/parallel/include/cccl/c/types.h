//==============================================================================
// MACA CCCL C Binding - Types
// Adapted for MACA (matches NVIDIA CCCL C API)
//==============================================================================

#ifndef CCCL_C_TYPES_H
#define CCCL_C_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Type System
//==============================================================================

typedef enum cccl_type_enum
{
    CCCL_INT8    = 0,
    CCCL_INT16   = 1,
    CCCL_INT32   = 2,
    CCCL_INT64   = 3,
    CCCL_UINT8   = 4,
    CCCL_UINT16  = 5,
    CCCL_UINT32  = 6,
    CCCL_UINT64  = 7,
    CCCL_FLOAT16 = 8,
    CCCL_FLOAT32 = 9,
    CCCL_FLOAT64 = 10,
    CCCL_STORAGE = 11,
    CCCL_BOOLEAN = 12,
} cccl_type_enum;

typedef struct cccl_type_info
{
    size_t         size;
    size_t         alignment;
    cccl_type_enum type;
} cccl_type_info;

//==============================================================================
// Operation Types
//==============================================================================

typedef enum cccl_op_kind_t
{
    // Arbitrary semantics, without state.
    CCCL_STATELESS = 0,

    // Arbitrary semantics, with state.
    CCCL_STATEFUL  = 1,

    // Well-known semantics (like C++ <functional>)
    CCCL_PLUS          = 2,
    CCCL_MINUS         = 3,
    CCCL_MULTIPLIES    = 4,
    CCCL_DIVIDES       = 5,
    CCCL_MODULUS       = 6,
    CCCL_EQUAL_TO      = 7,
    CCCL_NOT_EQUAL_TO  = 8,
    CCCL_GREATER       = 9,
    CCCL_LESS          = 10,
    CCCL_GREATER_EQUAL = 11,
    CCCL_LESS_EQUAL    = 12,
    CCCL_LOGICAL_AND   = 13,
    CCCL_LOGICAL_OR    = 14,
    CCCL_LOGICAL_NOT   = 15,
    CCCL_BIT_AND       = 16,
    CCCL_BIT_OR        = 17,
    CCCL_BIT_XOR       = 18,
    CCCL_BIT_NOT       = 19,
    CCCL_IDENTITY      = 20,
    CCCL_NEGATE        = 21,
    CCCL_MINIMUM       = 22,
    CCCL_MAXIMUM       = 23,
} cccl_op_kind_t;

typedef enum cccl_op_code_type
{
    CCCL_OP_BITCODE    = 0,   // Pre-compiled LLVM bitcode (MACA)
    CCCL_OP_CPP_SOURCE = 1,   // C++ source code (for JIT)
} cccl_op_code_type;

typedef struct cccl_op_t
{
    cccl_op_kind_t     type;       // Operation kind
    const char*        name;       // Operation name (optional, for debugging)
    const char*        code;       // Bitcode or C++ source code
    size_t             code_size;  // Size of code in bytes
    cccl_op_code_type  code_type;  // Type of code (BITCODE or CPP_SOURCE)
    size_t             size;       // Size of operation state
    size_t             alignment;  // Alignment of operation state
    void*              state;      // User state (if needed)
} cccl_op_t;

//==============================================================================
// Value Type (for initial values, etc.)
//==============================================================================

typedef struct cccl_value_t
{
    cccl_type_info type;
    void*          state;
} cccl_value_t;

//==============================================================================
// Iterator System
//==============================================================================

typedef enum cccl_iterator_kind_t
{
    CCCL_POINTER  = 0,   // Simple pointer iterator
    CCCL_ITERATOR = 1,   // Custom iterator (needs JIT)
} cccl_iterator_kind_t;

typedef union
{
    int64_t  signed_offset;
    uint64_t unsigned_offset;
} cccl_increment_t;

typedef void (*cccl_host_op_fn_ptr_t)(void*, cccl_increment_t);

typedef struct cccl_iterator_t
{
    size_t               size;          // Size of iterator state
    size_t               alignment;     // Alignment of iterator state
    cccl_iterator_kind_t type;          // Pointer or custom iterator
    cccl_op_t            advance;       // Advance operation (for custom)
    cccl_op_t            dereference;   // Dereference operation (for custom)
    cccl_type_info       value_type;    // Type of values produced
    void*                state;         // Iterator state (pointer for CCCL_POINTER)
    cccl_host_op_fn_ptr_t host_advance; // Host-side advance function
} cccl_iterator_t;

//==============================================================================
// Build Configuration
//==============================================================================

typedef struct cccl_build_config
{
    const char** extra_compile_flags;
    size_t       num_extra_compile_flags;
    const char** extra_include_dirs;
    size_t       num_extra_include_dirs;
} cccl_build_config;

//==============================================================================
// Other Enumerations
//==============================================================================

typedef enum cccl_sort_order_t
{
    CCCL_ASCENDING  = 0,
    CCCL_DESCENDING = 1,
} cccl_sort_order_t;

typedef enum cccl_init_kind_t
{
    CCCL_VALUE_INIT        = 0,
    CCCL_FUTURE_VALUE_INIT = 1,
    CCCL_NO_INIT           = 2,
} cccl_init_kind_t;

typedef enum cccl_determinism_t
{
    CCCL_NOT_GUARANTEED = 0,
    CCCL_RUN_TO_RUN     = 1,
    CCCL_GPU_TO_GPU     = 2,
} cccl_determinism_t;

//==============================================================================
// Helper Macros
//==============================================================================

#define CCCL_TYPE_INFO(T, ENUM) \
    (cccl_type_info){ sizeof(T), alignof(T), ENUM }

#define CCCL_TYPE_INFO_INT32  CCCL_TYPE_INFO(int32_t, CCCL_INT32)
#define CCCL_TYPE_INFO_INT64  CCCL_TYPE_INFO(int64_t, CCCL_INT64)
#define CCCL_TYPE_INFO_FLOAT  CCCL_TYPE_INFO(float,   CCCL_FLOAT32)
#define CCCL_TYPE_INFO_DOUBLE CCCL_TYPE_INFO(double,  CCCL_FLOAT64)

#ifdef __cplusplus
}
#endif

#endif // CCCL_C_TYPES_H
