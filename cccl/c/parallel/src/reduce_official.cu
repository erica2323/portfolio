//==============================================================================
//
// MACA CCCL C Binding - Reduce Implementation
// Adapted for MACA - Phase 3 (Iterator Support)
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
    if (op == CCCL_PLUS) {
        if (type == CCCL_FLOAT32) return "0.0f";
        if (type == CCCL_FLOAT64) return "0.0";
        return "0";
    } else if (op == CCCL_MINIMUM) {
        switch (type) {
            case CCCL_INT8:     return "127";
            case CCCL_INT16:    return "32767";
            case CCCL_INT32:    return "2147483647";
            case CCCL_INT64:    return "9223372036854775807LL";
            case CCCL_UINT8:    return "255";
            case CCCL_UINT16:   return "65535";
            case CCCL_UINT32:   return "4294967295U";
            case CCCL_UINT64:   return "18446744073709551615ULL";
            case CCCL_FLOAT32:  return "3.402823466e+38f";
            case CCCL_FLOAT64:  return "1.7976931348623158e+308";
            default:            return "2147483647";
        }
    } else if (op == CCCL_MAXIMUM) {
        switch (type) {
            case CCCL_INT8:     return "-128";
            case CCCL_INT16:    return "-32768";
            case CCCL_INT32:    return "-2147483648";
            case CCCL_INT64:    return "-9223372036854775807LL - 1";
            case CCCL_UINT8:    return "0";
            case CCCL_UINT16:   return "0";
            case CCCL_UINT32:   return "0U";
            case CCCL_UINT64:   return "0ULL";
            case CCCL_FLOAT32:  return "-3.402823466e+38f";
            case CCCL_FLOAT64:  return "-1.7976931348623158e+308";
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

static std::string get_accumulate_op_template(cccl_op_kind_t op) {
    switch (op) {
        case CCCL_PLUS:     return "sum += {VAL};";
        case CCCL_MINIMUM:  return "sum = min(sum, {VAL});";
        case CCCL_MAXIMUM:  return "sum = max(sum, {VAL});";
        default:            return "sum += {VAL};";
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

// Generate iterator access code
static std::string generate_iterator_access(
    cccl_iterator_t iter,
    const std::string& index_expr
) {
    if (iter.type == CCCL_POINTER) {
        // Simple pointer access
        return "d_in_base[" + index_expr + "]";
    } else {
        // Custom iterator - call dereference function
        // For now, we'll support strided iterator as example
        // User would provide stride in state
        return "d_in_base[" + index_expr + " * stride]";
    }
}

// Generate reduction kernel source code
static std::string generate_reduce_kernel(
    cccl_op_t op,
    cccl_iterator_t d_in_iter
) {
    std::string type_name = get_type_name(d_in_iter.value_type.type);
    std::string identity_value = get_identity_value(d_in_iter.value_type.type, op.type);
    std::string reduce_op = get_reduce_op(op.type);
    std::string accumulate_template = get_accumulate_op_template(op.type);
    std::string final_op = get_final_op(op.type);

    std::stringstream ss;

    // Add necessary headers
    ss << "#include <stdint.h>\n";
    ss << "\n";

    // Generate iterator-specific code
    bool is_pointer = (d_in_iter.type == CCCL_POINTER);

    if (!is_pointer && d_in_iter.dereference.code != nullptr) {
        // Custom iterator: include user's dereference code
        ss << "// User-provided iterator code\n";
        ss << d_in_iter.dereference.code << "\n";
        ss << "\n";
    }

    if (op.type == CCCL_PLUS || op.type == CCCL_MINIMUM || op.type == CCCL_MAXIMUM) {
        // Single tile kernel
        ss << "// Single tile kernel\n";
        ss << "extern \"C\" __global__\n";
        ss << "void reduce_single_tile_kernel(\n";
        ss << "    const " << type_name << "* __restrict__ d_in_base,\n";

        if (!is_pointer) {
            ss << "    unsigned long long stride,\n";
        }

        ss << "    " << type_name << "* __restrict__ d_out,\n";
        ss << "    unsigned long long n,\n";
        ss << "    " << type_name << " init_value\n";
        ss << ") {\n";
        ss << "    __shared__ " << type_name << " sdata[256];\n";
        ss << "    unsigned int tid = threadIdx.x;\n";
        ss << "    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "\n";
        ss << "    // Load data\n";
        ss << "    " << type_name << " val = " << identity_value << ";\n";
        ss << "    if (idx < n) {\n";

        std::string access = generate_iterator_access(d_in_iter, "idx");
        ss << "        val = " << access << ";\n";

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
        ss << "    // Write result\n";
        ss << "    if (tid == 0) {\n";
        ss << "        d_out[0] = " << final_op << ";\n";
        ss << "    }\n";
        ss << "}\n";
        ss << "\n";

        // Multi-block reduction kernel
        ss << "// Multi-block reduction kernel\n";
        ss << "extern \"C\" __global__\n";
        ss << "void reduce_kernel(\n";
        ss << "    const " << type_name << "* __restrict__ d_in_base,\n";

        if (!is_pointer) {
            ss << "    unsigned long long stride,\n";
        }

        ss << "    " << type_name << "* __restrict__ d_out,\n";
        ss << "    unsigned long long n\n";
        ss << ") {\n";
        ss << "    __shared__ " << type_name << " sdata[256];\n";
        ss << "    unsigned int tid = threadIdx.x;\n";
        ss << "    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        ss << "    unsigned long long gridSize = blockDim.x * gridDim.x;\n";
        ss << "\n";
        ss << "    // Grid-stride loop\n";
        ss << "    " << type_name << " sum = " << identity_value << ";\n";
        ss << "    for (unsigned long long i = idx; i < n; i += gridSize) {\n";

        std::string access_loop = generate_iterator_access(d_in_iter, "i");
        std::string accumulate = accumulate_template;
        size_t pos = accumulate.find("{VAL}");
        if (pos != std::string::npos) {
            accumulate.replace(pos, 5, access_loop);
        }
        ss << "        " << accumulate << "\n";

        ss << "    }\n";
        ss << "    sdata[tid] = sum;\n";
        ss << "    __syncthreads();\n";
        ss << "\n";
        ss << "    // Block-level reduction\n";
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
        ss << "// Error: Unsupported operation\n";
        ss << "extern \"C\" __global__ void reduce_single_tile_kernel() {}\n";
        ss << "extern \"C\" __global__ void reduce_kernel() {}\n";
    }

    return ss.str();
}

//==============================================================================
// Build Functions
//==============================================================================

mcError_t cccl_device_reduce_build_ex(
    cccl_device_reduce_build_result_t* build,
    cccl_op_t op,
    cccl_iterator_t d_in,
    void* initial_value,
    cccl_build_config* build_config
) {
    if (build == nullptr) {
        return mcErrorInvalidValue;
    }

    if (op.type != CCCL_PLUS && op.type != CCCL_MINIMUM && op.type != CCCL_MAXIMUM) {
        std::cerr << "Error: Only CCCL_PLUS, CCCL_MINIMUM, CCCL_MAXIMUM are supported" << std::endl;
        return mcErrorInvalidValue;
    }

    try {
        // Generate kernel source
        std::string kernel_src = generate_reduce_kernel(op, d_in);
        std::cout << "Generated reduce kernel:\n" << kernel_src << std::endl;

        // Create MCRTC program
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

        // Compile options
        std::vector<const char*> opts;
        opts.push_back("-xmaca");

        if (build_config != nullptr && build_config->num_extra_compile_flags > 0) {
            for (size_t i = 0; i < build_config->num_extra_compile_flags; ++i) {
                opts.push_back(build_config->extra_compile_flags[i]);
            }
        }

        std::vector<std::string> include_flags;
        if (build_config != nullptr && build_config->num_extra_include_dirs > 0) {
            for (size_t i = 0; i < build_config->num_extra_include_dirs; ++i) {
                include_flags.push_back(
                    std::string("-I") + build_config->extra_include_dirs[i]
                );
                opts.push_back(include_flags.back().c_str());
            }
        }

        // Compile
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

        // Get bitcode
        size_t codeSize;
        mcrtcGetBitcodeSize(prog, &codeSize);
        char* code = new char[codeSize];
        mcrtcGetBitcode(prog, code);

        std::cout << "Bitcode size: " << codeSize << " bytes" << std::endl;

        mcrtcDestroyProgram(&prog);

        // Load module
        mcError_t err = mcModuleLoadData(&build->module, code);
        if (err != mcSuccess) {
            std::cerr << "Error: mcModuleLoadData failed" << std::endl;
            delete[] code;
            return err;
        }

        std::cout << "Module loaded successfully" << std::endl;

        // Get kernel functions
        err = mcModuleGetFunction(&build->single_tile_kernel, build->module, "reduce_single_tile_kernel");
        if (err != mcSuccess) {
            std::cerr << "Error: mcModuleGetFunction (single_tile) failed" << std::endl;
            mcModuleUnload(build->module);
            delete[] code;
            return err;
        }

        err = mcModuleGetFunction(&build->reduction_kernel, build->module, "reduce_kernel");
        if (err != mcSuccess) {
            std::cerr << "Error: mcModuleGetFunction (reduction) failed" << std::endl;
            mcModuleUnload(build->module);
            delete[] code;
            return err;
        }

        std::cout << "Kernel functions obtained successfully" << std::endl;

        // Save result
        build->bitcode = code;
        build->bitcode_size = codeSize;
        build->type = d_in.value_type;
        build->op = op;
        build->d_in_iterator = d_in;

        // Copy initial value
        build->initial_value_size = d_in.value_type.size;
        build->initial_value = malloc(d_in.value_type.size);
        memcpy(build->initial_value, initial_value, d_in.value_type.size);

        return mcSuccess;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return mcErrorUnknown;
    }
}

// Backward compatible pointer version
mcError_t cccl_device_reduce_build(
    cccl_device_reduce_build_result_t* build,
    cccl_op_t op,
    cccl_type_info type,
    void* initial_value,
    cccl_build_config* build_config
) {
    // Create a pointer iterator
    cccl_iterator_t iter = cccl_make_pointer_iterator(nullptr, type);

    // Call the extended version
    return cccl_device_reduce_build_ex(build, op, iter, initial_value, build_config);
}

//==============================================================================
// Execute Functions
//==============================================================================

mcError_t cccl_device_reduce_ex(
    cccl_device_reduce_build_result_t build,
    cccl_iterator_t d_in,
    void* d_out,
    uint64_t num_items,
    mcStream_t stream
) {
    if (num_items == 0) {
        return mcSuccess;
    }

    try {
        const unsigned int threads = 256;
        bool is_pointer = (d_in.type == CCCL_POINTER);

        // Extract base pointer and stride
        void* d_in_base = d_in.state;
        uint64_t stride = 1;  // Default stride for pointer

        if (!is_pointer && d_in.size > 0) {
            // Custom iterator with stride (stored in first 8 bytes of state)
            stride = *(uint64_t*)d_in.state;
        }

        // Decide kernel
        if (num_items <= threads) {
            // Small data: single tile
            std::cout << "Using single tile kernel for " << num_items << " items" << std::endl;

            std::vector<void*> args;
            args.push_back(&d_in_base);
            if (!is_pointer) args.push_back(&stride);
            args.push_back(&d_out);
            args.push_back(&num_items);
            args.push_back(build.initial_value);

            mcError_t err = mcModuleLaunchKernel(
                build.single_tile_kernel,
                1, 1, 1,
                threads, 1, 1,
                0,
                stream,
                args.data(),
                nullptr
            );

            if (err != mcSuccess) {
                std::cerr << "Error: mcModuleLaunchKernel (single_tile) failed" << std::endl;
                return err;
            }
        } else {
            // Large data: multi-block
            const unsigned int blocks = (num_items + threads - 1) / threads;
            const unsigned int max_blocks = 1024;
            const unsigned int actual_blocks = (blocks < max_blocks) ? blocks : max_blocks;

            std::cout << "Using multi-block reduction: "
                      << actual_blocks << " blocks, "
                      << threads << " threads, "
                      << num_items << " items" << std::endl;

            // Allocate temp storage
            void* d_temp;
            mcError_t err = mcMalloc(&d_temp, actual_blocks * build.type.size);
            if (err != mcSuccess) {
                std::cerr << "Error: mcMalloc failed" << std::endl;
                return err;
            }

            // Phase 1
            std::vector<void*> args1;
            args1.push_back(&d_in_base);
            if (!is_pointer) args1.push_back(&stride);
            args1.push_back(&d_temp);
            args1.push_back(&num_items);

            err = mcModuleLaunchKernel(
                build.reduction_kernel,
                actual_blocks, 1, 1,
                threads, 1, 1,
                0,
                stream,
                args1.data(),
                nullptr
            );

            if (err != mcSuccess) {
                std::cerr << "Error: mcModuleLaunchKernel (phase 1) failed" << std::endl;
                mcFree(d_temp);
                return err;
            }

            // Phase 2
            uint64_t temp_items = actual_blocks;
            uint64_t temp_stride = 1;  // Temp is always contiguous

            std::vector<void*> args2;
            args2.push_back(&d_temp);
            if (!is_pointer) args2.push_back(&temp_stride);
            args2.push_back(&d_out);
            args2.push_back(&temp_items);
            args2.push_back(build.initial_value);

            err = mcModuleLaunchKernel(
                build.single_tile_kernel,
                1, 1, 1,
                threads, 1, 1,
                0,
                stream,
                args2.data(),
                nullptr
            );

            if (err != mcSuccess) {
                std::cerr << "Error: mcModuleLaunchKernel (phase 2) failed" << std::endl;
                mcFree(d_temp);
                return err;
            }

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

// Backward compatible pointer version
mcError_t cccl_device_reduce(
    cccl_device_reduce_build_result_t build,
    void* d_in,
    void* d_out,
    uint64_t num_items,
    mcStream_t stream
) {
    // Create pointer iterator
    cccl_iterator_t iter = cccl_make_pointer_iterator(d_in, build.type);

    // Call extended version
    return cccl_device_reduce_ex(build, iter, d_out, num_items, stream);
}

//==============================================================================
// Cleanup
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
