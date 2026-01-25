//==============================================================================
// MACA CCCL - JIT Template Mapping: Iterator
// Converts cccl_iterator_t to template-compatible strings
//==============================================================================

#ifndef CCCL_JIT_MAPPINGS_ITERATOR_H
#define CCCL_JIT_MAPPINGS_ITERATOR_H

#include <cccl/c/types_official.h>
#include "type_info.h"
#include "operation.h"
#include <string>
#include <format>

namespace cccl {
namespace jit {

/**
 * Mapping for cccl_iterator_t to C++ iterator types and template arguments
 */
struct cccl_iterator_t_mapping {
    /**
     * Check if iterator is a simple pointer
     */
    static bool is_pointer(const cccl_iterator_t& it) {
        return it.type == CCCL_POINTER;
    }

    /**
     * Generate the iterator type definition
     *
     * For pointers: "using InputIt = const int32_t*;"
     * For custom iterators: "using InputIt = input_iterator_t<tag, size, align, ValueT>;"
     */
    static std::string map(const cccl_iterator_t& it, const char* tag_name, bool is_const = true) {
        if (is_pointer(it)) {
            // Simple pointer iterator
            const char* type_name = cccl_type_info_mapping::get_type_name(it.value_type.type);
            if (is_const) {
                return std::format("using {} = const {}*;", tag_name, type_name);
            } else {
                return std::format("using {} = {}*;", tag_name, type_name);
            }
        }

        // Custom iterator - use input_iterator_t template
        const char* value_type = cccl_type_info_mapping::get_type_name(it.value_type.type);

        return std::format(
            "using {} = input_iterator_t<{}_tag, {}, {}, {}>;",
            tag_name,
            tag_name,
            it.size,
            it.alignment,
            value_type
        );
    }

    /**
     * Generate auxiliary declarations for custom iterators
     *
     * For pointers: returns empty string
     * For custom: generates tag type and extern device function declarations
     */
    static std::string aux(const cccl_iterator_t& it, const char* tag_name) {
        if (is_pointer(it)) {
            return "";  // Pointers don't need declarations
        }

        std::string result;

        // Generate tag type
        result += std::format("struct {}_tag {{}};\n", tag_name);

        // Generate state type for aligned storage
        result += std::format(
            "struct alignas({}) {}_state_t {{ char data[{}]; }};\n",
            it.alignment,
            tag_name,
            it.size
        );

        // Generate extern device function declarations for advance and dereference
        const char* value_type = cccl_type_info_mapping::get_type_name(it.value_type.type);

        // Dereference: reads value from iterator state
        result += std::format(
            "extern \"C\" __device__ void {}_dereference("
            "{}* __restrict__ result, "
            "const void* __restrict__ state);\n",
            tag_name,
            value_type
        );

        // Advance: modifies iterator state by offset
        result += std::format(
            "extern \"C\" __device__ void {}_advance("
            "void* __restrict__ state, "
            "int64_t offset);\n",
            tag_name
        );

        return result;
    }

    /**
     * Generate code to create an iterator from state pointer
     *
     * For pointers: "static_cast<const int32_t*>(state_ptr)"
     * For custom: "InputIt::from_state(state_ptr)"
     */
    static std::string instantiate(const cccl_iterator_t& it, const char* tag_name,
                                   const char* state_var) {
        if (is_pointer(it)) {
            const char* type_name = cccl_type_info_mapping::get_type_name(it.value_type.type);
            return std::format("static_cast<const {}*>({})", type_name, state_var);
        }

        return std::format("{}::from_state({})", tag_name, state_var);
    }

    /**
     * Generate code to create an output iterator from state pointer
     */
    static std::string instantiate_output(const cccl_iterator_t& it, const char* tag_name,
                                          const char* state_var) {
        if (is_pointer(it)) {
            const char* type_name = cccl_type_info_mapping::get_type_name(it.value_type.type);
            return std::format("static_cast<{}*>({})", type_name, state_var);
        }

        return std::format("{}::from_state({})", tag_name, state_var);
    }

    /**
     * Get the LTOIR data for advance operation (if custom iterator)
     */
    static std::pair<const char*, size_t> get_advance_ltoir(const cccl_iterator_t& it) {
        if (is_pointer(it) || it.advance.code == nullptr) {
            return {nullptr, 0};
        }
        return {it.advance.code, it.advance.code_size};
    }

    /**
     * Get the LTOIR data for dereference operation (if custom iterator)
     */
    static std::pair<const char*, size_t> get_dereference_ltoir(const cccl_iterator_t& it) {
        if (is_pointer(it) || it.dereference.code == nullptr) {
            return {nullptr, 0};
        }
        return {it.dereference.code, it.dereference.code_size};
    }

    /**
     * Check if iterator requires LTOIR linking
     */
    static bool requires_linking(const cccl_iterator_t& it) {
        if (is_pointer(it)) {
            return false;
        }
        return it.advance.code != nullptr || it.dereference.code != nullptr;
    }

    /**
     * Generate a pointer cast expression for simple pointer access
     */
    static std::string pointer_cast(const cccl_iterator_t& it, const char* ptr_var,
                                    bool is_const = true) {
        const char* type_name = cccl_type_info_mapping::get_type_name(it.value_type.type);
        if (is_const) {
            return std::format("static_cast<const {}*>({})", type_name, ptr_var);
        }
        return std::format("static_cast<{}*>({})", type_name, ptr_var);
    }
};

} // namespace jit
} // namespace cccl

#endif // CCCL_JIT_MAPPINGS_ITERATOR_H
