//==============================================================================
// Operation Template - Complete Implementation
// Generates reduction operation code
//==============================================================================

#pragma once

#include <cccl/c/types_official.h>
#include <string>
#include <sstream>
#include "../mappings/operation.h"
#include "../mappings/type_info.h"

namespace cccl {
namespace jit {

//==============================================================================
// Operation Code Generation
//==============================================================================

inline std::string generate_operation(
    const cccl_op_t& op,
    const std::string& value_type,
    const std::string& op_name = "ReductionOp"
) {
    std::stringstream ss;

    auto info = get_operation_info(op);

    if (info.is_builtin) {
        // Use CUB built-in operators
        ss << "// Using CUB built-in operator: " << get_operation_name(op.type) << "\n";
        ss << "typedef " << info.builtin_name << " " << op_name << ";\n";
    }
    else if (info.has_custom_code) {
        // Custom operation with user code
        ss << "//==============================================================================\n";
        ss << "// Custom Reduction Operation\n";
        ss << "//==============================================================================\n\n";

        // Include user's operation code
        ss << "// User-provided operation function\n";
        ss << info.custom_code << "\n\n";

        // Wrapper struct
        ss << "// Operation wrapper\n";
        ss << "struct " << op_name << " {\n";
        ss << "    __device__ __forceinline__\n";
        ss << "    " << value_type << " operator()(\n";
        ss << "        const " << value_type << "& a,\n";
        ss << "        const " << value_type << "& b) const {\n";
        ss << "        " << value_type << " result;\n";
        ss << "        custom_op(const_cast<" << value_type << "*>(&a),\n";
        ss << "                  const_cast<" << value_type << "*>(&b),\n";
        ss << "                  &result);\n";
        ss << "        return result;\n";
        ss << "    }\n";
        ss << "};\n";
    }
    else {
        // No code provided - fall back to Sum
        ss << "// No custom code - using default Sum operator\n";
        ss << "typedef ::cub::Sum " << op_name << ";\n";
    }

    return ss.str();
}

} // namespace jit
} // namespace cccl
