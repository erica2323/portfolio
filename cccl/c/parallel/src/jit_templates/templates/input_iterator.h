//==============================================================================
// JIT Template: Input Iterator Code Generation
// Generates code for custom input iterators
//==============================================================================

#pragma once

#include <string>
#include <sstream>
#include <cccl/c/types_official.h>

namespace cccl {
namespace jit {

// Generate input iterator code from cccl_iterator_t
inline std::string generate_input_iterator_code(
    const cccl_iterator_t& iterator,
    const std::string& value_type_name,
    const std::string& iterator_name = "InputIterator"
) {
    std::stringstream ss;

    switch (iterator.type) {
        case CCCL_POINTER: {
            // Simple pointer - no custom code needed
            ss << "// Pointer iterator - using raw pointer directly\n";
            ss << "typedef " << value_type_name << "* " << iterator_name << ";\n";
            break;
        }

        case CCCL_ITERATOR: {
            // Custom iterator with dereference code
            if (iterator.dereference.code != nullptr) {
                ss << "// Custom iterator dereference function\n";
                ss << iterator.dereference.code << "\n\n";

                ss << "// Custom iterator wrapper struct\n";
                ss << "struct " << iterator_name << " {\n";
                ss << "    " << value_type_name << "* data;\n";
                ss << "    \n";
                ss << "    __device__ __forceinline__\n";
                ss << "    " << value_type_name << " operator[](size_t idx) const {\n";
                ss << "        return dereference(data, idx);\n";
                ss << "    }\n";
                ss << "    \n";
                ss << "    __device__ __forceinline__\n";
                ss << "    " << iterator_name << " operator+(size_t offset) const {\n";
                ss << "        return {data + offset};\n";
                ss << "    }\n";
                ss << "};\n";
            } else {
                // No custom code - fall back to pointer
                ss << "// Iterator without custom code - using pointer\n";
                ss << "typedef " << value_type_name << "* " << iterator_name << ";\n";
            }
            break;
        }

        default:
            // Unknown type - use pointer as fallback
            ss << "// Unknown iterator type - using pointer\n";
            ss << "typedef " << value_type_name << "* " << iterator_name << ";\n";
            break;
    }

    return ss.str();
}

} // namespace jit
} // namespace cccl
