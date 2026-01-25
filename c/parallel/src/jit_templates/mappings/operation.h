//==============================================================================
// MACA CCCL - JIT Template Mapping: Operation
// Converts cccl_op_t to template-compatible strings
//==============================================================================

#ifndef CCCL_JIT_MAPPINGS_OPERATION_H
#define CCCL_JIT_MAPPINGS_OPERATION_H

#include <cccl/c/types_official.h>
#include <string>
#include <format>
#include <stdexcept>

namespace cccl {
namespace jit {

/**
 * Mapping for cccl_op_t to C++ operation types and template arguments
 */
struct cccl_op_t_mapping {
    /**
     * Get the CUB/mcCub functor name for a builtin operation
     */
    static const char* get_builtin_name(cccl_op_kind_t kind) {
        switch (kind) {
            case CCCL_PLUS:          return "cub::Sum";
            case CCCL_MINUS:         return "cub::Difference";  // Custom
            case CCCL_MULTIPLIES:    return "cub::Product";     // Custom
            case CCCL_DIVIDES:       return "cub::Quotient";    // Custom
            case CCCL_MINIMUM:       return "cub::Min";
            case CCCL_MAXIMUM:       return "cub::Max";
            case CCCL_BIT_AND:       return "cub::BitwiseAnd";  // Custom
            case CCCL_BIT_OR:        return "cub::BitwiseOr";   // Custom
            case CCCL_BIT_XOR:       return "cub::BitwiseXor";  // Custom
            case CCCL_LOGICAL_AND:   return "cub::LogicalAnd";  // Custom
            case CCCL_LOGICAL_OR:    return "cub::LogicalOr";   // Custom
            case CCCL_EQUAL_TO:      return "cub::Equality";    // Custom
            case CCCL_NOT_EQUAL_TO:  return "cub::Inequality";  // Custom
            case CCCL_GREATER:       return "cub::Max";         // Fallback
            case CCCL_LESS:          return "cub::Min";         // Fallback
            case CCCL_IDENTITY:      return "cub::Identity";    // Custom
            default:
                return nullptr;  // Not a builtin
        }
    }

    /**
     * Check if operation is a builtin (uses standard CUB functor)
     */
    static bool is_builtin(cccl_op_kind_t kind) {
        return kind != CCCL_STATELESS && kind != CCCL_STATEFUL;
    }

    /**
     * Generate the operation type definition
     *
     * For builtins: "using OpT = cub::Sum;"
     * For user-defined: "using OpT = stateless_user_operation<tag, size, align>;"
     */
    static std::string map(const cccl_op_t& op, const char* tag_name) {
        if (is_builtin(op.type)) {
            const char* builtin = get_builtin_name(op.type);
            if (builtin) {
                return std::format("using {} = {};", tag_name, builtin);
            }
            throw std::runtime_error("Unknown builtin operation type: " +
                                    std::to_string(static_cast<int>(op.type)));
        }

        // User-defined operation
        if (op.type == CCCL_STATELESS) {
            return std::format(
                "using {} = stateless_user_operation<{}_tag, {}, {}>;",
                tag_name,
                tag_name,
                op.size,
                op.alignment
            );
        } else {
            // CCCL_STATEFUL
            return std::format(
                "using {} = stateful_user_operation<{}_tag, {}, {}, {}, {}>;",
                tag_name,
                tag_name,
                op.size,
                op.alignment,
                op.size,      // state_size (assume same as op size)
                op.alignment  // state_alignment
            );
        }
    }

    /**
     * Generate auxiliary declarations needed for the operation
     *
     * For builtins: returns empty string
     * For user-defined: generates extern device function declarations
     */
    static std::string aux(const cccl_op_t& op, const char* tag_name) {
        if (is_builtin(op.type)) {
            return "";  // Builtins don't need external declarations
        }

        // Generate tag type
        std::string result = std::format("struct {}_tag {{}};\n", tag_name);

        // Generate extern device function declaration
        if (op.type == CCCL_STATELESS) {
            result += std::format(
                "extern \"C\" __device__ void {}_device_fn("
                "void* __restrict__ result, "
                "const void* __restrict__ arg0, "
                "const void* __restrict__ arg1);\n",
                tag_name
            );
        } else {
            // Stateful operation has additional state parameter
            result += std::format(
                "extern \"C\" __device__ void {}_device_fn("
                "void* __restrict__ result, "
                "const void* __restrict__ state, "
                "const void* __restrict__ arg0, "
                "const void* __restrict__ arg1);\n",
                tag_name
            );
        }

        return result;
    }

    /**
     * Generate the operation instantiation code
     *
     * For builtins: "OpT{}"
     * For user-defined with state: "OpT{state_ptr}"
     */
    static std::string instantiate(const cccl_op_t& op, const char* tag_name,
                                   const char* state_var = nullptr) {
        if (is_builtin(op.type)) {
            return std::format("{}{{}}",  tag_name);
        }

        if (op.type == CCCL_STATEFUL && state_var) {
            return std::format("{}{{*static_cast<const typename {}::state_type*>({})}}",
                              tag_name, tag_name, state_var);
        }

        return std::format("{}{{}}", tag_name);
    }

    /**
     * Get the arity (number of arguments) for an operation
     */
    static int get_arity(cccl_op_kind_t kind) {
        switch (kind) {
            case CCCL_IDENTITY:
            case CCCL_NEGATE:
            case CCCL_LOGICAL_NOT:
            case CCCL_BIT_NOT:
                return 1;  // Unary operations
            default:
                return 2;  // Binary operations
        }
    }

    /**
     * Check if operation is commutative (can reorder arguments)
     */
    static bool is_commutative(cccl_op_kind_t kind) {
        switch (kind) {
            case CCCL_PLUS:
            case CCCL_MULTIPLIES:
            case CCCL_MINIMUM:
            case CCCL_MAXIMUM:
            case CCCL_BIT_AND:
            case CCCL_BIT_OR:
            case CCCL_BIT_XOR:
            case CCCL_LOGICAL_AND:
            case CCCL_LOGICAL_OR:
            case CCCL_EQUAL_TO:
            case CCCL_NOT_EQUAL_TO:
                return true;
            default:
                return false;
        }
    }

    /**
     * Check if operation is associative (can regroup)
     */
    static bool is_associative(cccl_op_kind_t kind) {
        switch (kind) {
            case CCCL_PLUS:
            case CCCL_MULTIPLIES:
            case CCCL_MINIMUM:
            case CCCL_MAXIMUM:
            case CCCL_BIT_AND:
            case CCCL_BIT_OR:
            case CCCL_BIT_XOR:
            case CCCL_LOGICAL_AND:
            case CCCL_LOGICAL_OR:
                return true;
            default:
                return false;
        }
    }
};

} // namespace jit
} // namespace cccl

#endif // CCCL_JIT_MAPPINGS_OPERATION_H
