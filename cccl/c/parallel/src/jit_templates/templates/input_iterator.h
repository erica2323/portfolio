//==============================================================================
// Input Iterator Template - Complete Implementation
// Generates complete iterator code matching NVIDIA CCCL structure
//==============================================================================

#pragma once

#include <cccl/c/types_official.h>
#include <string>
#include <sstream>
#include "../mappings/iterator.h"
#include "../mappings/type_info.h"

namespace cccl {
namespace jit {

//==============================================================================
// Input Iterator Code Generation
//==============================================================================

inline std::string generate_input_iterator(
    const cccl_iterator_t& iterator,
    const std::string& iterator_name = "InputIterator"
) {
    std::stringstream ss;

    auto spec = detect_specialization(iterator);
    auto state = get_iterator_state_info(iterator);
    std::string value_type = type_to_name(iterator.value_type.type);

    // Check for simple pointer optimization
    if (spec.is_pointer && !spec.has_custom_code) {
        // Optimization: use raw pointer directly
        ss << "// Optimized path: raw pointer\n";
        ss << "typedef " << value_type << "* " << iterator_name << ";\n";
        return ss.str();
    }

    // Custom iterator with full implementation
    ss << "//==============================================================================\n";
    ss << "// Custom Input Iterator: " << iterator_name << "\n";
    ss << "//==============================================================================\n\n";

    // If user provided custom dereference code, include it first
    if (spec.has_custom_code) {
        ss << "// User-provided dereference function\n";
        ss << spec.code << "\n\n";
    }

    // Iterator state struct
    ss << "// Iterator state\n";
    ss << "struct " << iterator_name << "_state_t {\n";
    ss << "    " << value_type << "* ptr;\n";
    if (spec.has_custom_code) {
        ss << "    // Additional state for custom iterator\n";
        ss << "    // (can be extended based on iterator requirements)\n";
    }
    ss << "};\n\n";

    // Iterator struct with full traits and operators
    ss << "// Iterator implementation\n";
    ss << "struct " << iterator_name << " {\n";
    ss << "    using iterator_type = " << iterator_name << ";\n";
    ss << "\n";

    // Iterator traits
    ss << generate_iterator_traits(iterator_name, value_type);
    ss << "\n";

    // State
    ss << "    // Iterator state\n";
    ss << "    " << value_type << "* ptr;\n";
    ss << "\n";

    // Constructor
    ss << "    // Constructor\n";
    ss << "    __device__ __forceinline__\n";
    ss << "    " << iterator_name << "(" << value_type << "* p = nullptr) : ptr(p) {}\n";
    ss << "\n";

    // Operators
    std::string deref_impl;
    if (spec.has_custom_code) {
        // Call user's dereference function
        deref_impl = "return dereference(ptr, 0);";
    } else {
        deref_impl = "return *ptr;";
    }

    ss << "    // Iterator operators\n";
    ss << generate_iterator_operators(value_type, spec.has_custom_code, deref_impl);

    ss << "};\n";

    return ss.str();
}

//==============================================================================
// Convenience Functions
//==============================================================================

// Generate input iterator with automatic naming
inline std::string generate_input_iterator_code(
    const cccl_iterator_t& iterator,
    const std::string& value_type_name,
    const std::string& iterator_name
) {
    return generate_input_iterator(iterator, iterator_name);
}

} // namespace jit
} // namespace cccl
