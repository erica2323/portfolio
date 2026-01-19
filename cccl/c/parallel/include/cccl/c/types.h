//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CCCL_C_TYPES_H
#define CCCL_C_TYPES_H

#include <stddef.h>
#include <stdint.h>

// MACA runtime types (equivalent to CUDA Driver API)
#include <mc_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Return Types (CUDA Driver API compatible)
//==============================================================================

// Use mcError_t as CUresult equivalent for MACA
typedef mcError_t CUresult;
typedef mcStream_t CUstream;

// CUDA success codes
#define CUDA_SUCCESS mcSuccess

//==============================================================================
// Type System
//==============================================================================

/**
 * @brief Enumeration of supported data types
 */
typedef enum {
    CCCL_INT8,
    CCCL_INT16,
    CCCL_INT32,
    CCCL_INT64,
    CCCL_UINT8,
    CCCL_UINT16,
    CCCL_UINT32,
    CCCL_UINT64,
    CCCL_FLOAT16,
    CCCL_FLOAT32,
    CCCL_FLOAT64,
    CCCL_STORAGE,  // Opaque storage type
} cccl_type_enum;

/**
 * @brief Type information structure
 */
typedef struct {
    cccl_type_enum type;  // Type identifier
    size_t size;          // Size in bytes
    size_t alignment;     // Alignment requirement
} cccl_type_info;

//==============================================================================
// Value Type (wraps arbitrary values)
//==============================================================================

/**
 * @brief Value container - can hold any type
 */
typedef struct {
    void* data;           // Pointer to the value
    cccl_type_info type;  // Type information
} cccl_value_t;

//==============================================================================
// Operation Types
//==============================================================================

/**
 * @brief Well-known operation kinds
 */
typedef enum {
    CCCL_PLUS,      // Addition
    CCCL_MINIMUM,   // Minimum
    CCCL_MAXIMUM,   // Maximum
    CCCL_CUSTOM,    // User-defined operation
} cccl_op_kind_t;

/**
 * @brief Operation callable types
 */
typedef enum {
    CCCL_STATELESS,  // No state (pure function)
    CCCL_STATEFUL,   // Has state
} cccl_callable_kind_t;

/**
 * @brief Callable descriptor for operations
 */
typedef struct {
    cccl_callable_kind_t type;  // Stateless or stateful
    const char* code;           // CUDA source code (C++ or LTO-IR)
    size_t code_size;           // Size of code
    void* state;                // State pointer (for stateful ops)
    size_t state_size;          // Size of state
    size_t state_alignment;     // Alignment of state
} cccl_callable_t;

/**
 * @brief Operation descriptor
 */
typedef struct {
    cccl_op_kind_t type;        // Operation kind
    cccl_callable_t callable;   // Function implementation
    const char* name;           // Function name (e.g., "op")
    size_t name_size;           // Size of name
} cccl_op_t;

//==============================================================================
// Iterator Types
//==============================================================================

/**
 * @brief Iterator kinds
 */
typedef enum {
    CCCL_POINTER,           // Simple pointer
    CCCL_COUNTING,          // Counting iterator
    CCCL_CONSTANT,          // Constant iterator
    CCCL_TRANSFORM,         // Transform iterator
    CCCL_RANDOM_ACCESS,     // Custom random access iterator
} cccl_iterator_kind_t;

/**
 * @brief Iterator descriptor
 */
typedef struct {
    cccl_iterator_kind_t type;  // Iterator kind
    void* state;                // Iterator state (e.g., pointer, counter)
    size_t size;                // State size
    size_t alignment;           // State alignment
    cccl_type_info value_type;  // Type of dereferenced value

    // Advance operation (for custom iterators)
    cccl_callable_t advance;

    // Dereference operation (for custom iterators)
    cccl_callable_t dereference;

    // Host-side advance function (optional)
    void (*host_advance)(void* state, int64_t n);
} cccl_iterator_t;

//==============================================================================
// Build Configuration
//==============================================================================

/**
 * @brief Determinism mode
 */
typedef enum {
    CCCL_RUN_TO_RUN,      // Deterministic across runs
    CCCL_NOT_GUARANTEED,  // Non-deterministic (faster)
} cccl_determinism_t;

/**
 * @brief Build configuration for JIT compilation
 */
typedef struct {
    const char** extra_compile_flags;  // Additional compiler flags
    size_t num_extra_compile_flags;    // Number of flags
    const char** extra_include_dirs;   // Additional include directories
    size_t num_extra_include_dirs;     // Number of directories
} cccl_build_config;

//==============================================================================
// Helper Functions
//==============================================================================

/**
 * @brief Get type info for a given type enum
 */
static inline cccl_type_info cccl_type_info_from_enum(cccl_type_enum type) {
    cccl_type_info info;
    info.type = type;

    switch (type) {
        case CCCL_INT8:    info.size = 1; info.alignment = 1; break;
        case CCCL_INT16:   info.size = 2; info.alignment = 2; break;
        case CCCL_INT32:   info.size = 4; info.alignment = 4; break;
        case CCCL_INT64:   info.size = 8; info.alignment = 8; break;
        case CCCL_UINT8:   info.size = 1; info.alignment = 1; break;
        case CCCL_UINT16:  info.size = 2; info.alignment = 2; break;
        case CCCL_UINT32:  info.size = 4; info.alignment = 4; break;
        case CCCL_UINT64:  info.size = 8; info.alignment = 8; break;
        case CCCL_FLOAT16: info.size = 2; info.alignment = 2; break;
        case CCCL_FLOAT32: info.size = 4; info.alignment = 4; break;
        case CCCL_FLOAT64: info.size = 8; info.alignment = 8; break;
        default:           info.size = 0; info.alignment = 0; break;
    }

    return info;
}

/**
 * @brief Create a simple pointer iterator
 */
static inline cccl_iterator_t cccl_make_pointer_iterator(void* ptr, cccl_type_info type) {
    cccl_iterator_t it;
    it.type = CCCL_POINTER;
    it.state = ptr;
    it.value_type = type;
    it.size = sizeof(void*);
    it.alignment = alignof(void*);
    it.advance.type = CCCL_STATELESS;
    it.advance.code = nullptr;
    it.advance.code_size = 0;
    it.advance.state = nullptr;
    it.advance.state_size = 0;
    it.advance.state_alignment = 0;
    it.dereference.type = CCCL_STATELESS;
    it.dereference.code = nullptr;
    it.dereference.code_size = 0;
    it.dereference.state = nullptr;
    it.dereference.state_size = 0;
    it.dereference.state_alignment = 0;
    it.host_advance = nullptr;
    return it;
}

/**
 * @brief Create a value container
 */
static inline cccl_value_t cccl_make_value(void* data, cccl_type_info type) {
    cccl_value_t val;
    val.data = data;
    val.type = type;
    return val;
}

/**
 * @brief Create a well-known operation (PLUS, MIN, MAX)
 */
static inline cccl_op_t cccl_make_builtin_op(cccl_op_kind_t kind) {
    cccl_op_t op;
    op.type = kind;
    op.callable.type = CCCL_STATELESS;
    op.callable.code = nullptr;
    op.callable.code_size = 0;
    op.callable.state = nullptr;
    op.callable.state_size = 0;
    op.callable.state_alignment = 0;
    op.name = "op";
    op.name_size = 2;
    return op;
}

#ifdef __cplusplus
}
#endif

#endif // CCCL_C_TYPES_H
