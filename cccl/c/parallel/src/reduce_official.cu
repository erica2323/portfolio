//==============================================================================
//
// MACA CCCL C Binding - Reduce Implementation
// Adapted for MACA - Phase 2 (Pointer + PLUS/MIN/MAX)
//
//==============================================================================

#include <cccl/c/reduce_official.h>
#include <mcrtc.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <limits>

//==============================================================================
// Helper Functions
//==============================================================================

static std::string get_type_name(cccl_type_enum type) {
    switch (type) {
        case CCCL_INT8:     return "int8_t";
        case CCCL_INT16:    return "int16_t";
        case CCCL_INT32:    return "int32_t";
        case CCCL_INT64:    return "int64_t";
        case CCCL_UINT8:    return "uint8_t";
        case CCCL_UINT16:   return "uint16_t";
        case CCCL_UINT32:   return "uint32_t";
        case CCCL_UINT64:   return "uint64_t";
        case CCCL_FLOAT16:  return "__half";
        case CCCL_FLOAT32:  return "float";
        case CCCL_FLOAT64:  return "double";
        default:            return "int32_t";
    }
}

static std::string get_identity_value(cccl_type_enum type, cccl_op_kind_t op) {
    // Identity value depends on the operation
    if (op == CCCL_PLUS) {
        // Identity for addition is 0
        if (type == CCCL_FLOAT32) return "0.0f";
        if (type == CCCL_FLOAT64) return "0.0";
        return "0";
    } else if (op == CCCL_MINIMUM) {
        // Identity for min is max possible value
        switch (type) {
            case CCCL_INT8:     return "127";
            case CCCL_INT16:    return "32767";
            case CCCL_INT32:    return "2147483647";
            case CCCL_INT64:    return "9223372036854775807LL";
            case CCCL_UINT8:    return "255";
            case CCCL_UINT16:   return "65535";
            case CCCL_UINT32:   return "4294967295U";
            case CCCL_UINT64:   return "18446744073709551615ULL";
            case CCCL_FLOAT32:  return "3.402823466e+38f";  // FLT_MAX
            case CCCL_FLOAT64:  return "1.7976931348623158e+308";  // DBL_MAX
            default:            return "2147483647";
        }
    } else if (op == CCCL_MAXIMUM) {
        // Identity for max is min possible value
        switch (type) {
            case CCCL_INT8:     return "-128";
            case CCCL_INT16:    return "-32768";
            case CCCL_INT32:    return "-2147483648";
            case CCCL_INT64:    return "-9223372036854775807LL - 1";
            case CCCL_UINT8:    return "0";
            case CCCL_UINT16:   return "0";
            case CCCL_UINT32:   return "0U";
            case CCCL_UINT64:   return "0ULL";
            case CCCL_FLOAT32:  return "-3.402823466e+38f";  // -FLT_MAX
            case CCCL_FLOAT64:  return "-1.7976931348623158e+308";  // -DBL_MAX
            default:            return "-2147483648";
        }
    }
    return "0";
}

static std::string get_reduce_op(cccl_op_kind_t op) {
    switch (op) {
        case CCCL_PLUS:     return "+=";
        case CCCL_MINIMUM:  return "= min(sdata[tid], sdata[tid + s])";
        case CCCL_MAXIMUM:  return "= max(sdata[tid], sdata[tid + s])";
        default:            return "+=";
    }
}

static std::string get_accumulate_op(cccl_op_kind_t op) {
    switch (op) {
        case CCCL_PLUS:     return "sum += d_in[i];";
        case CCCL_MINIMUM:  return "sum = min(sum, d_in[i]);";
        case CCCL_MAXIMUM:  return "sum = max(sum, d_in[i]);";
        default:            return "sum += d_in[i];";
    }
}

static std::string get_final_op(cccl_op_kind_t op) {
    switch (op) {
        case CCCL_PLUS:     return "sdata[0] + init_value";
        case CCCL_MINIMUM:  return "min(sdata[0], init_value)";
        case CCCL_MAXIMUM:  return "max(sdata[0], init_value)";
        default:            return "sdata[0] + init_value";
    }
}

// Generate reduction kernel source code
static std::string generate_reduce_kernel(
    cccl_op_t op,
    cccl_type_info type
) {
    std::string type_name = get_type_name(type.type);
    std::string identity_value = get_identity_value(type.type, op.type);
    std::string reduce_op = get_reduce_op(op.type);
    std::string accumulate_op = get_accumulate_op(op.type);
    std::string final_op = get_final_op(op.type);

    std::stringstream ss;

    // Add necessary headers
    ss << "#include <stdint.h>\n";
    ss << "\n";

    if (op.type == CCCL_PLUS || op.type == CCCL_MINIMUM || op.type == CCCL_MAXIMUM) {
        // Single tile kernel with initial value parameter
        ss << "// Single tile kernel: for small arrays that fit in one block\n";
        ss << "extern \"C\" __global__\n";
        ss << "void reduce_single_tile_kernel(\n";
        ss << "    const " << type_name << "* __restrict__ d_in,\n";
        ss << "    " << type_name << "* __restrict__ d_out,\n";
        ss << "    unsigned long long n,\n";
        ss << "    " << type_name << " init_value\n";
        ss << ") {\n";
        ss << "    __shared__ " << type_name << " sdata[256];\n";
        ss << "    unsigned int tid = threadIdx.x;\n";
        ss << "    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "\n";
        ss << "    // Load data into shared memory\n";
        ss << "    " << type_name << " val = " << identity_value << ";\n";
        ss << "    if (idx < n) {\n";
        ss << "        val = d_in[idx];\n";
        ss << "    }\n";
        ss << "    sdata[tid] = val;\n";
        ss << "    __syncthreads();\n";
        ss << "\n";
        ss << "    // Reduce in shared memory\n";
        ss << "    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n";
        ss << "        if (tid < s) {\n";
        ss << "            sdata[tid] " << reduce_op << ";\n";
        ss << "        }\n";
        ss << "        __syncthreads();\n";
        ss << "    }\n";
        ss << "\n";
        ss << "    // Write result (combine with initial value)\n";
        ss << "    if (tid == 0) {\n";
        ss << "        d_out[0] = " << final_op << ";\n";
        ss << "    }\n";
        ss << "}\n";
        ss << "\n";

        // Multi-block reduction kernel
        ss << "// Reduction kernel: for large arrays (multi-block)\n";
        ss << "extern \"C\" __global__\n";
        ss << "void reduce_kernel(\n";
        ss << "    const " << type_name << "* __restrict__ d_in,\n";
        ss << "    " << type_name << "* __restrict__ d_out,\n";
        ss << "    unsigned long long n\n";
        ss << ") {\n";
        ss << "    __shared__ " << type_name << " sdata[256];\n";
        ss << "    unsigned int tid = threadIdx.x;\n";
        ss << "    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "    unsigned long long gridSize = blockDim.x * gridDim.x;\n";
        ss << "\n";
        ss << "    // Grid-stride loop to accumulate values\n";
        ss << "    " << type_name << " sum = " << identity_value << ";\n";
        ss << "    for (unsigned long long i = idx; i < n; i += gridSize) {\n";
        ss << "        " << accumulate_op << "\n";
        ss << "    }\n";
        ss << "    sdata[tid] = sum;\n";
        ss << "    __syncthreads();\n";
        ss << "\n";
        ss << "    // Block-level reduction in shared memory\n";
        ss << "    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n";
        ss << "        if (tid < s) {\n";
        ss << "            sdata[tid] " << reduce_op << ";\n";
        ss << "        }\n";
        ss << "        __syncthreads();\n";
        ss << "    }\n";
        ss << "\n";
        ss << "    // Write block result\n";
        ss << "    if (tid == 0) {\n";
        ss << "        d_out[blockIdx.x] = sdata[0];\n";
        ss << "    }\n";
        ss << "}\n";
    } else {
        // Unsupported operation
        ss << "// Error: Only CCCL_PLUS, CCCL_MINIMUM, CCCL_MAXIMUM are supported\n";
        ss << "extern \"C\" __global__ void reduce_single_tile_kernel() {}\n";
        ss << "extern \"C\" __global__ void reduce_kernel() {}\n";
    }

    return ss.str();
}

//==============================================================================
// Build Function
//==============================================================================

mcError_t cccl_device_reduce_build(
    cccl_device_reduce_build_result_t* build,
    cccl_op_t op,
    cccl_type_info type,
    void* initial_value,
    cccl_build_config* build_config
) {
    if (build == nullptr) {
        return mcErrorInvalidValue;
    }

    // Phase 2: Support PLUS, MINIMUM, MAXIMUM
    if (op.type != CCCL_PLUS && op.type != CCCL_MINIMUM && op.type != CCCL_MAXIMUM) {
        std::cerr << "Error: Only CCCL_PLUS, CCCL_MINIMUM, CCCL_MAXIMUM are supported" << std::endl;
        return mcErrorInvalidValue;
    }

    try {
        // 1. Generate kernel source code
        std::string kernel_src = generate_reduce_kernel(op, type);
        std::cout << "Generated reduce kernel:\n" << kernel_src << std::endl;

        // 2. Create MCRTC program
        mcrtcProgram prog;
        mcrtcResult result = mcrtcCreateProgram(
            &prog,
            kernel_src.c_str(),
            "reduce_official.cu",
            0, nullptr, nullptr
        );

        if (result != MCRTC_SUCCESS) {
            std::cerr << "Error: mcrtcCreateProgram failed" << std::endl;
            return mcErrorUnknown;
        }

        // 3. Prepare compile options
        std::vector<const char*> opts;
        opts.push_back("-xmaca");

        // Add extra compile flags
        if (build_config != nullptr && build_config->num_extra_compile_flags > 0) {
            for (size_t i = 0; i < build_config->num_extra_compile_flags; ++i) {
                opts.push_back(build_config->extra_compile_flags[i]);
            }
        }

        // Add extra include directories
        std::vector<std::string> include_flags;
        if (build_config != nullptr && build_config->num_extra_include_dirs > 0) {
            for (size_t i = 0; i < build_config->num_extra_include_dirs; ++i) {
                include_flags.push_back(
                    std::string("-I") + build_config->extra_include_dirs[i]
                );
                opts.push_back(include_flags.back().c_str());
            }
        }

        // 4. Compile program
        result = mcrtcCompileProgram(prog, opts.size(), opts.data());
        if (result != MCRTC_SUCCESS) {
            size_t logSize;
            mcrtcGetProgramLogSize(prog, &logSize);
            if (logSize > 1) {
                char* log = new char[logSize];
                mcrtcGetProgramLog(prog, log);
                std::cerr << "Compilation failed:\n" << log << std::endl;
                delete[] log;
            }
            mcrtcDestroyProgram(&prog);
            return mcErrorUnknown;
        }

        std::cout << "Compilation successful" << std::endl;

        // 5. Get bitcode
        size_t codeSize;
        mcrtcGetBitcodeSize(prog, &codeSize);
        char* code = new char[codeSize];
        mcrtcGetBitcode(prog, code);

        std::cout << "Bitcode size: " << codeSize << " bytes" << std::endl;

        // 6. Destroy MCRTC program
        mcrtcDestroyProgram(&prog);

        // 7. Load module
        mcError_t err = mcModuleLoadData(&build->module, code);
        if (err != mcSuccess) {
            std::cerr << "Error: mcModuleLoadData failed with code " << err << std::endl;
            delete[] code;
            return err;
        }

        std::cout << "Module loaded successfully" << std::endl;

        // 8. Get kernel functions
        err = mcModuleGetFunction(&build->single_tile_kernel, build->module, "reduce_single_tile_kernel");
        if (err != mcSuccess) {
            std::cerr << "Error: mcModuleGetFunction (single_tile) failed with code " << err << std::endl;
            mcModuleUnload(build->module);
            delete[] code;
            return err;
        }

        err = mcModuleGetFunction(&build->reduction_kernel, build->module, "reduce_kernel");
        if (err != mcSuccess) {
            std::cerr << "Error: mcModuleGetFunction (reduction) failed with code " << err << std::endl;
            mcModuleUnload(build->module);
            delete[] code;
            return err;
        }

        std::cout << "Kernel functions obtained successfully" << std::endl;

        // 9. Save result
        build->bitcode = code;
        build->bitcode_size = codeSize;
        build->type = type;
        build->op = op;

        // Copy initial value
        build->initial_value_size = type.size;
        build->initial_value = malloc(type.size);
        memcpy(build->initial_value, initial_value, type.size);

        return mcSuccess;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return mcErrorUnknown;
    }
}

//==============================================================================
// Execute Function
//==============================================================================

mcError_t cccl_device_reduce(
    cccl_device_reduce_build_result_t build,
    void* d_in,
    void* d_out,
    uint64_t num_items,
    mcStream_t stream
) {
    if (num_items == 0) {
        return mcSuccess;
    }

    try {
        const unsigned int threads = 256;

        // Decide which kernel to use based on data size
        if (num_items <= threads) {
            // Small data: use single tile kernel
            std::cout << "Using single tile kernel for " << num_items << " items" << std::endl;

            void* args[] = { &d_in, &d_out, &num_items, build.initial_value };

            mcError_t err = mcModuleLaunchKernel(
                build.single_tile_kernel,
                1, 1, 1,           // 1 block
                threads, 1, 1,
                0,
                stream,
                args,
                nullptr
            );

            if (err != mcSuccess) {
                std::cerr << "Error: mcModuleLaunchKernel (single_tile) failed with code "
                          << err << std::endl;
                return err;
            }
        } else {
            // Large data: use multi-block reduction (two-phase)
            const unsigned int blocks = (num_items + threads - 1) / threads;
            const unsigned int max_blocks = 1024;  // Limit number of blocks
            const unsigned int actual_blocks = (blocks < max_blocks) ? blocks : max_blocks;

            std::cout << "Using multi-block reduction: "
                      << actual_blocks << " blocks, "
                      << threads << " threads, "
                      << num_items << " items" << std::endl;

            // Allocate temporary storage for block results
            void* d_temp;
            mcError_t err = mcMalloc(&d_temp, actual_blocks * build.type.size);
            if (err != mcSuccess) {
                std::cerr << "Error: mcMalloc failed for temp storage" << std::endl;
                return err;
            }

            // Phase 1: Reduce to per-block results
            void* args1[] = { &d_in, &d_temp, &num_items };

            err = mcModuleLaunchKernel(
                build.reduction_kernel,
                actual_blocks, 1, 1,
                threads, 1, 1,
                0,
                stream,
                args1,
                nullptr
            );

            if (err != mcSuccess) {
                std::cerr << "Error: mcModuleLaunchKernel (phase 1) failed with code "
                          << err << std::endl;
                mcFree(d_temp);
                return err;
            }

            // Phase 2: Final reduction of block results (with initial value)
            uint64_t temp_items = actual_blocks;
            void* args2[] = { &d_temp, &d_out, &temp_items, build.initial_value };

            err = mcModuleLaunchKernel(
                build.single_tile_kernel,
                1, 1, 1,
                threads, 1, 1,
                0,
                stream,
                args2,
                nullptr
            );

            if (err != mcSuccess) {
                std::cerr << "Error: mcModuleLaunchKernel (phase 2) failed with code "
                          << err << std::endl;
                mcFree(d_temp);
                return err;
            }

            // Clean up temporary storage
            mcFree(d_temp);
        }

        std::cout << "Kernel launched successfully" << std::endl;
        return mcSuccess;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return mcErrorUnknown;
    }
}

//==============================================================================
// Cleanup Function
//==============================================================================

mcError_t cccl_device_reduce_cleanup(
    cccl_device_reduce_build_result_t* build
) {
    if (build == nullptr) {
        return mcErrorInvalidValue;
    }

    try {
        if (build->module != nullptr) {
            mcModuleUnload(build->module);
            build->module = nullptr;
        }

        if (build->bitcode != nullptr) {
            delete[] (char*)build->bitcode;
            build->bitcode = nullptr;
        }

        if (build->initial_value != nullptr) {
            free(build->initial_value);
            build->initial_value = nullptr;
        }

        build->bitcode_size = 0;
        build->initial_value_size = 0;
        return mcSuccess;
    }
    catch (...) {
        return mcErrorUnknown;
    }
}
