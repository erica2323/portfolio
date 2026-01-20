//==============================================================================
// Type Info Mapping for JIT Templates
// Maps CCCL C types to C++ type names for code generation
//==============================================================================

#pragma once

#include <cccl/c/types_official.h>
#include <string>
#include <sstream>

namespace cccl {
namespace jit {

// Get C++ type name from cccl_type_info
inline std::string type_to_name(cccl_type_enum type, bool with_namespace = false) {
    std::string prefix = with_namespace ? "::cub::" : "";

    switch (type) {
        case CCCL_INT8:     return "int8_t";
        case CCCL_INT16:    return "int16_t";
        case CCCL_INT32:    return "int32_t";
        case CCCL_INT64:    return "int64_t";
        case CCCL_UINT8:    return "uint8_t";
        case CCCL_UINT16:   return "uint16_t";
        case CCCL_UINT32:   return "uint32_t";
        case CCCL_UINT64:   return "uint64_t";
        case CCCL_FLOAT32:  return "float";
        case CCCL_FLOAT64:  return "double";
        case CCCL_FLOAT16:  return "__half";
        default:            return "int32_t";
    }
}

// Get size of type
inline size_t type_size(cccl_type_enum type) {
    switch (type) {
        case CCCL_INT8:
        case CCCL_UINT8:    return 1;
        case CCCL_INT16:
        case CCCL_UINT16:
        case CCCL_FLOAT16:  return 2;
        case CCCL_INT32:
        case CCCL_UINT32:
        case CCCL_FLOAT32:  return 4;
        case CCCL_INT64:
        case CCCL_UINT64:
        case CCCL_FLOAT64:  return 8;
        default:            return 4;
    }
}

// Get alignment of type
inline size_t type_alignment(cccl_type_enum type) {
    return type_size(type);  // For primitive types, alignment == size
}

} // namespace jit
} // namespace cccl
