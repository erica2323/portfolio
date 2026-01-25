//==============================================================================
// MACA CCCL - JIT Template Mapping: Type Info
// Converts cccl_type_info to template-compatible strings
//==============================================================================

#ifndef CCCL_JIT_MAPPINGS_TYPE_INFO_H
#define CCCL_JIT_MAPPINGS_TYPE_INFO_H

#include <cccl/c/types_official.h>
#include <string>
#include <format>
#include <stdexcept>

namespace cccl {
namespace jit {

/**
 * Mapping for cccl_type_info to C++ type names and template arguments
 */
struct cccl_type_info_mapping {
    /**
     * Get the C++ type name for a given type enum
     */
    static const char* get_type_name(cccl_type_enum type) {
        switch (type) {
            case CCCL_INT8:    return "int8_t";
            case CCCL_INT16:   return "int16_t";
            case CCCL_INT32:   return "int32_t";
            case CCCL_INT64:   return "int64_t";
            case CCCL_UINT8:   return "uint8_t";
            case CCCL_UINT16:  return "uint16_t";
            case CCCL_UINT32:  return "uint32_t";
            case CCCL_UINT64:  return "uint64_t";
            case CCCL_FLOAT16: return "__half";     // MACA half precision
            case CCCL_FLOAT32: return "float";
            case CCCL_FLOAT64: return "double";
            case CCCL_BOOLEAN: return "bool";
            case CCCL_STORAGE: return "void";       // Opaque storage
            default:
                throw std::runtime_error("Unknown type enum: " +
                                        std::to_string(static_cast<int>(type)));
        }
    }

    /**
     * Check if the type is a builtin numeric type
     */
    static bool is_builtin(cccl_type_enum type) {
        switch (type) {
            case CCCL_INT8:
            case CCCL_INT16:
            case CCCL_INT32:
            case CCCL_INT64:
            case CCCL_UINT8:
            case CCCL_UINT16:
            case CCCL_UINT32:
            case CCCL_UINT64:
            case CCCL_FLOAT16:
            case CCCL_FLOAT32:
            case CCCL_FLOAT64:
            case CCCL_BOOLEAN:
                return true;
            default:
                return false;
        }
    }

    /**
     * Generate a typedef for a type with given name
     * Example: "using InputT = int32_t;"
     */
    static std::string map(const cccl_type_info& info, const char* alias_name) {
        if (is_builtin(info.type)) {
            return std::format("using {} = {};", alias_name, get_type_name(info.type));
        } else {
            // For custom/storage types, generate an aligned storage type
            return std::format(
                "struct alignas({}) {} {{ char data[{}]; }};",
                info.alignment,
                alias_name,
                info.size
            );
        }
    }

    /**
     * Generate a storage_t type for aligned byte storage
     * Used when we need to hold a value of unknown type
     */
    static std::string map_storage(const cccl_type_info& info, const char* alias_name) {
        return std::format(
            "struct alignas({}) {} {{ char data[{}]; }};",
            info.alignment,
            alias_name,
            info.size
        );
    }

    /**
     * Generate initialization value for a type
     */
    static std::string get_init_value(cccl_type_enum type) {
        switch (type) {
            case CCCL_BOOLEAN:
                return "false";
            case CCCL_FLOAT16:
            case CCCL_FLOAT32:
            case CCCL_FLOAT64:
                return "0.0";
            default:
                return "0";
        }
    }

    /**
     * Generate a literal suffix for a value
     */
    static const char* get_literal_suffix(cccl_type_enum type) {
        switch (type) {
            case CCCL_INT64:   return "LL";
            case CCCL_UINT32:  return "U";
            case CCCL_UINT64:  return "ULL";
            case CCCL_FLOAT32: return "f";
            case CCCL_FLOAT64: return "";
            default:           return "";
        }
    }

    /**
     * Generate an initial value expression from a pointer
     * This reinterprets the bytes at the pointer as the given type
     */
    static std::string format_init_value(const cccl_type_info& info, const void* value_ptr) {
        if (!value_ptr) {
            return get_init_value(info.type);
        }

        // For builtin types, we can cast and format directly
        switch (info.type) {
            case CCCL_INT8:
                return std::to_string(*static_cast<const int8_t*>(value_ptr));
            case CCCL_INT16:
                return std::to_string(*static_cast<const int16_t*>(value_ptr));
            case CCCL_INT32:
                return std::to_string(*static_cast<const int32_t*>(value_ptr));
            case CCCL_INT64:
                return std::to_string(*static_cast<const int64_t*>(value_ptr)) + "LL";
            case CCCL_UINT8:
                return std::to_string(*static_cast<const uint8_t*>(value_ptr));
            case CCCL_UINT16:
                return std::to_string(*static_cast<const uint16_t*>(value_ptr));
            case CCCL_UINT32:
                return std::to_string(*static_cast<const uint32_t*>(value_ptr)) + "U";
            case CCCL_UINT64:
                return std::to_string(*static_cast<const uint64_t*>(value_ptr)) + "ULL";
            case CCCL_FLOAT32: {
                float val = *static_cast<const float*>(value_ptr);
                return std::format("{}f", val);
            }
            case CCCL_FLOAT64: {
                double val = *static_cast<const double*>(value_ptr);
                return std::format("{}", val);
            }
            case CCCL_BOOLEAN:
                return *static_cast<const bool*>(value_ptr) ? "true" : "false";
            default:
                // For unknown types, return zero-initialized
                return "{}";
        }
    }
};

} // namespace jit
} // namespace cccl

#endif // CCCL_JIT_MAPPINGS_TYPE_INFO_H
