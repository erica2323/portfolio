//==============================================================================
// Operation Mapping for JIT Templates
// Maps CCCL operations to CUB binary operators
//==============================================================================

#pragma once

#include <cccl/c/types_official.h>
#include <string>
#include <sstream>

namespace cccl {
namespace jit {

//==============================================================================
// Operation Type Detection
//==============================================================================

struct operation_info {
    bool is_builtin;
    bool has_custom_code;
    std::string builtin_name;  // e.g., "cub::Sum", "cub::Min"
    std::string custom_code;
};

inline operation_info get_operation_info(const cccl_op_t& op) {
    operation_info info;

    switch (op.type) {
        case CCCL_PLUS:
            info.is_builtin = true;
            info.has_custom_code = false;
            info.builtin_name = "::cub::Sum";
            break;

        case CCCL_MINIMUM:
            info.is_builtin = true;
            info.has_custom_code = false;
            info.builtin_name = "::cub::Min";
            break;

        case CCCL_MAXIMUM:
            info.is_builtin = true;
            info.has_custom_code = false;
            info.builtin_name = "::cub::Max";
            break;

        case CCCL_CUSTOM:
            info.is_builtin = false;
            info.has_custom_code = (op.code != nullptr && strlen(op.code) > 0);
            info.custom_code = info.has_custom_code ? op.code : "";
            break;

        default:
            // Unknown - default to Sum
            info.is_builtin = true;
            info.has_custom_code = false;
            info.builtin_name = "::cub::Sum";
            break;
    }

    return info;
}

//==============================================================================
// Operation Name Generation
//==============================================================================

// Get operation name for display
inline std::string get_operation_name(cccl_op_kind_t op_type) {
    switch (op_type) {
        case CCCL_PLUS:     return "Sum";
        case CCCL_MINIMUM:  return "Min";
        case CCCL_MAXIMUM:  return "Max";
        case CCCL_CUSTOM:   return "Custom";
        default:            return "Unknown";
    }
}

} // namespace jit
} // namespace cccl
