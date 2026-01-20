//==============================================================================
// Complete Kernel Source Generator
// Generates full reduction kernel code with proper iterator/operation support
//==============================================================================

#pragma once

#include <string>
#include <sstream>
#include "templates/input_iterator.h"
#include "templates/output_iterator.h"
#include "templates/operation.h"
#include "mappings/type_info.h"
#include <cccl/c/types_official.h>

namespace cccl {
namespace jit {

//==============================================================================
// Kernel Source Generation
//==============================================================================

struct kernel_generation_params {
    cccl_iterator_t input_iterator;
    cccl_iterator_t output_iterator;
    cccl_op_t operation;
    std::string value_type_name;
    std::string init_value_expr;
    int block_threads = 256;
};

inline std::string generate_reduce_kernel(const kernel_generation_params& params) {
    std::stringstream ss;

    // Header
    ss << "//==============================================================================\n";
    ss << "// JIT-Compiled Reduction Kernel\n";
    ss << "// Generated at runtime by MACA CCCL\n";
    ss << "//==============================================================================\n\n";

    // Includes
    ss << "#include <mccub/block/block_reduce.cuh>\n";
    ss << "#include <mccub/device/dispatch/dispatch_reduce.cuh>\n";
    ss << "#include <mccub/util_type.cuh>\n\n";

    ss << "using namespace cub;\n\n";

    // Generate iterator types
    ss << "//==============================================================================\n";
    ss << "// Iterator Types\n";
    ss << "//==============================================================================\n\n";

    ss << generate_input_iterator(params.input_iterator, "InputIterator");
    ss << "\n";
    ss << generate_output_iterator(params.output_iterator, "OutputIterator");
    ss << "\n";

    // Generate operation
    ss << "//==============================================================================\n";
    ss << "// Reduction Operation\n";
    ss << "//==============================================================================\n\n";

    ss << generate_operation(params.operation, params.value_type_name, "ReductionOp");
    ss << "\n";

    // Generate kernel
    ss << "//==============================================================================\n";
    ss << "// Reduction Kernel\n";
    ss << "//==============================================================================\n\n";

    ss << "extern \"C\" __global__ void cccl_reduce_kernel(\n";
    ss << "    " << params.value_type_name << "* d_in_raw,\n";
    ss << "    " << params.value_type_name << "* d_out_raw,\n";
    ss << "    int num_items,\n";
    ss << "    " << params.value_type_name << " init_value)\n";
    ss << "{\n";

    // Kernel body using CUB BlockReduce
    ss << "    // Type definitions\n";
    ss << "    typedef " << params.value_type_name << " T;\n";
    ss << "    typedef cub::BlockReduce<T, " << params.block_threads << "> BlockReduce;\n\n";

    ss << "    // Shared memory\n";
    ss << "    __shared__ typename BlockReduce::TempStorage temp_storage;\n\n";

    ss << "    // Setup iterators\n";
    ss << "    InputIterator input(d_in_raw);\n";
    ss << "    OutputIterator output(d_out_raw);\n";
    ss << "    ReductionOp reduction_op;\n\n";

    ss << "    // Thread index\n";
    ss << "    int tid = threadIdx.x + blockIdx.x * blockDim.x;\n\n";

    ss << "    // Load data\n";
    ss << "    T thread_data = " << params.init_value_expr << ";\n";
    ss << "    if (tid < num_items) {\n";
    ss << "        thread_data = input[tid];\n";
    ss << "    }\n\n";

    ss << "    // Block-level reduction\n";
    ss << "    T block_aggregate = BlockReduce(temp_storage).Reduce(thread_data, reduction_op);\n\n";

    ss << "    // Output result (first thread of each block)\n";
    ss << "    if (threadIdx.x == 0) {\n";
    ss << "        atomicAdd(d_out_raw, block_aggregate);\n";
    ss << "    }\n";

    ss << "}\n";

    return ss.str();
}

//==============================================================================
// Convenience Function
//==============================================================================

inline std::string generate_reduce_kernel_source(
    const cccl_iterator_t& input_iter,
    const cccl_iterator_t& output_iter,
    const cccl_op_t& op,
    const std::string& value_type_name,
    const std::string& init_value_expr
) {
    kernel_generation_params params;
    params.input_iterator = input_iter;
    params.output_iterator = output_iter;
    params.operation = op;
    params.value_type_name = value_type_name;
    params.init_value_expr = init_value_expr;

    return generate_reduce_kernel(params);
}

} // namespace jit
} // namespace cccl
