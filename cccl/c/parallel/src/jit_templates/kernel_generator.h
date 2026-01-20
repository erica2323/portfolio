//==============================================================================
// Kernel Source Generator
// Generates complete kernel source code for JIT compilation
//==============================================================================

#pragma once

#include <string>
#include <sstream>
#include "templates/input_iterator.h"
#include "templates/operation.h"
#include <cccl/c/types_official.h>

namespace cccl {
namespace jit {

// Generate complete reduce kernel source
inline std::string generate_reduce_kernel_source(
    const cccl_iterator_t& input_iter,
    const cccl_op_t& op,
    const std::string& value_type_name,
    const std::string& init_value_str
) {
    std::stringstream ss;

    // Headers
    ss << "//==============================================================================\n";
    ss << "// JIT-Compiled Reduce Kernel\n";
    ss << "// Generated at runtime by MACA CCCL\n";
    ss << "//==============================================================================\n\n";

    ss << "#include <mccub/block/block_reduce.cuh>\n";
    ss << "#include <mccub/device/dispatch/dispatch_reduce.cuh>\n\n";

    ss << "using namespace cub;\n\n";

    // Generate input iterator code
    ss << "//==============================================================================\n";
    ss << "// Input Iterator\n";
    ss << "//==============================================================================\n\n";
    ss << generate_input_iterator_code(input_iter, value_type_name, "InputIterator");
    ss << "\n";

    // Generate operation code
    ss << "//==============================================================================\n";
    ss << "// Reduction Operation\n";
    ss << "//==============================================================================\n\n";
    ss << generate_operation_code(op, value_type_name, "ReductionOp");
    ss << "\n";

    // Generate kernel
    ss << "//==============================================================================\n";
    ss << "// Reduction Kernel\n";
    ss << "//==============================================================================\n\n";

    ss << "extern \"C\" __global__ void cccl_reduce_kernel(\n";
    ss << "    " << value_type_name << "* d_in,\n";
    ss << "    " << value_type_name << "* d_out,\n";
    ss << "    int num_items,\n";
    ss << "    " << value_type_name << " init_value\n";
    ss << ") {\n";
    ss << "    // Use CUB BlockReduce\n";
    ss << "    typedef cub::BlockReduce<" << value_type_name << ", 256> BlockReduce;\n";
    ss << "    __shared__ typename BlockReduce::TempStorage temp_storage;\n\n";

    ss << "    InputIterator input{d_in};\n";
    ss << "    ReductionOp reduction_op;\n\n";

    ss << "    int tid = threadIdx.x + blockIdx.x * blockDim.x;\n";
    ss << "    " << value_type_name << " thread_data = " << init_value_str << ";\n\n";

    ss << "    if (tid < num_items) {\n";
    ss << "        thread_data = input[tid];\n";
    ss << "    }\n\n";

    ss << "    " << value_type_name << " block_aggregate = \n";
    ss << "        BlockReduce(temp_storage).Reduce(thread_data, reduction_op);\n\n";

    ss << "    if (threadIdx.x == 0) {\n";
    ss << "        atomicAdd(d_out, block_aggregate);\n";
    ss << "    }\n";
    ss << "}\n";

    return ss.str();
}

} // namespace jit
} // namespace cccl
