//==============================================================================
// JIT Template: Operation Code Generation
// Generates code for custom reduction operations
//==============================================================================

#pragma once

#include <string>
#include <sstream>
#include <cccl/c/types_official.h>

namespace cccl {
namespace jit {

// Generate operation code from cccl_op_t
inline std::string generate_operation_code(
    const cccl_op_t& op,
    const std::string& value_type_name,
    const std::string& op_struct_name = "ReductionOp"
) {
    std::stringstream ss;

    switch (op.type) {
        case CCCL_PLUS: {
            // Use CUB's built-in Sum operator
            ss << "// Using CUB built-in Sum operator\n";
            ss << "typedef cub::Sum " << op_struct_name << ";\n";
            break;
        }

        case CCCL_MINIMUM: {
            // Use CUB's built-in Min operator
            ss << "// Using CUB built-in Min operator\n";
            ss << "typedef cub::Min " << op_struct_name << ";\n";
            break;
        }

        case CCCL_MAXIMUM: {
            // Use CUB's built-in Max operator
            ss << "// Using CUB built-in Max operator\n";
            ss << "typedef cub::Max " << op_struct_name << ";\n";
            break;
        }

        case CCCL_CUSTOM: {
            // Custom operation with user-provided code
            if (op.code != nullptr) {
                ss << "// Custom operation function\n";
                ss << op.code << "\n\n";

                ss << "// Custom operation wrapper struct\n";
                ss << "struct " << op_struct_name << " {\n";
                ss << "    __device__ __forceinline__\n";
                ss << "    " << value_type_name << " operator()(\n";
                ss << "        const " << value_type_name << "& a,\n";
                ss << "        const " << value_type_name << "& b) const {\n";
                ss << "        " << value_type_name << " result;\n";
                ss << "        custom_op(&a, &b, &result);\n";
                ss << "        return result;\n";
                ss << "    }\n";
                ss << "};\n";
            } else {
                // No custom code - fall back to Sum
                ss << "// Custom op without code - using Sum\n";
                ss << "typedef cub::Sum " << op_struct_name << ";\n";
            }
            break;
        }

        default:
            // Unknown type - use Sum as fallback
            ss << "// Unknown operation type - using Sum\n";
            ss << "typedef cub::Sum " << op_struct_name << ";\n";
            break;
    }

    return ss.str();
}

} // namespace jit
} // namespace cccl
