#ifndef CCCL_C_TYPES_OFFICIAL_H
#define CCCL_C_TYPES_OFFICIAL_H

#include <mc_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Type enumeration
typedef enum {
    CCCL_INT8,
    CCCL_INT16,
    CCCL_INT32,
    CCCL_INT64,
    CCCL_UINT8,
    CCCL_UINT16,
    CCCL_UINT32,
    CCCL_UINT64,
    CCCL_FLOAT32,
    CCCL_FLOAT64
} cccl_type_enum;

// Type information structure
typedef struct {
    cccl_type_enum type;
    size_t size;
    size_t alignment;
} cccl_type_info;

// Operation type enumeration
typedef enum {
    CCCL_STATELESS,   // User-defined stateless operation
    CCCL_PLUS,        // Built-in plus operation
    CCCL_MULTIPLIES,  // Built-in multiplies operation
    CCCL_MAXIMUM,     // Built-in maximum operation
    CCCL_MINIMUM      // Built-in minimum operation
} cccl_op_type;

// Code type enumeration
typedef enum {
    CCCL_OP_CPP_SOURCE,  // C++ source code
    CCCL_OP_LTOIR        // LTOIR binary (not implemented yet)
} cccl_op_code_type;

// Operation structure
typedef struct {
    cccl_op_type type;
    const char* name;
    const char* code;
    size_t code_size;
    cccl_op_code_type code_type;
    size_t size;
    size_t alignment;
    void* state;
} cccl_op_t;

#ifdef __cplusplus
}
#endif

#endif // CCCL_C_TYPES_OFFICIAL_H
