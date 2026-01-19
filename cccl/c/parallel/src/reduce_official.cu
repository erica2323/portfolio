//==============================================================================
//
// MACA CCCL C Binding - Reduce Implementation
// Adapted for MACA - Phase 5 (JIT + mcCub Integration - True CCCL Style)
//
// Architecture: JIT compile kernels that #include mcCub, just like NVIDIA CCCL
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

static std::string get_mccub_op_name(cccl_op_kind_t op) {
    switch (op) {
        case CCCL_PLUS:     return "Sum";
        case CCCL_MINIMUM:  return "Min";
        case CCCL_MAXIMUM:  return "Max";
        default:            return "Sum";
    }
}

//==============================================================================
// JIT Kernel Source Generation - The CCCL Way
//==============================================================================

// Generate kernel source that #includes mcCub and calls its API
// This matches NVIDIA CCCL's approach of including CUB headers
static std::string generate_reduce_kernel_with_mccub(
    cccl_op_t op,
    cccl_iterator_t d_in_iter
) {
    std::string type_name = get_type_name(d_in_iter.value_type.type);
    std::string op_name = get_mccub_op_name(op.type);

    std::stringstream ss;

    // Include mcCub headers - just like NVIDIA CCCL includes CUB headers!
    ss << "//==============================================================================\n";
    ss << "// JIT-compiled Reduce Kernel with mcCub Integration\n";
    ss << "// Generated at runtime - matches NVIDIA CCCL architecture\n";
    ss << "//==============================================================================\n";
    ss << "\n";
    ss << "#include <stdint.h>\n";
    ss << "#include <mccub/device/device_reduce.cuh>\n";  // KEY: Include mcCub!
    ss << "#include <thrust/mccub.h>\n";
    ss << "\n";

    // Generate iterator specialization if needed
    bool is_pointer = (d_in_iter.type == CCCL_POINTER);
    if (!is_pointer && d_in_iter.dereference.code != nullptr) {
        ss << "// Custom iterator code\n";
        ss << d_in_iter.dereference.code << "\n";
        ss << "\n";
    }

    // Generate the wrapper kernel that calls mcCub
    // This kernel bridges our C API to mcCub's C++ API
    ss << "//==============================================================================\n";
    ss << "// Wrapper Kernel - Calls mcCub::" << op_name << " internally\n";
    ss << "//==============================================================================\n";
    ss << "\n";
    ss << "extern \"C\" __global__ void cccl_reduce_kernel_wrapper(\n";
    ss << "    const " << type_name << "* d_in,\n";
    ss << "    " << type_name << "* d_out,\n";
    ss << "    void* d_temp_storage,\n";
    ss << "    size_t* d_temp_storage_bytes,\n";
    ss << "    unsigned long long num_items,\n";
    ss << "    " << type_name << " init_value,\n";
    ss << "    int phase  // 0 = query size, 1 = execute\n";
    ss << ") {\n";
    ss << "    // Only thread 0 does the work (host-like kernel)\n";
    ss << "    if (threadIdx.x == 0 && blockIdx.x == 0) {\n";
    ss << "        // Call mcCub DeviceReduce::" << op_name << "\n";
    ss << "        // This uses the highly optimized CUB kernels!\n";
    ss << "        thrust::mccub::DeviceReduce::" << op_name << "(\n";
    ss << "            d_temp_storage,\n";
    ss << "            *d_temp_storage_bytes,\n";
    ss << "            d_in,\n";
    ss << "            d_out,\n";
    ss << "            (int)num_items\n";
    ss << "        );\n";
    ss << "        \n";
    ss << "        // Note: init_value handling would go here if mcCub supports it\n";
    ss << "        // For now we assume init_value=0 for sum, INT_MAX for min, etc.\n";
    ss << "    }\n";
    ss << "}\n";
    ss << "\n";

    return ss.str();
}

//==============================================================================
// Build Functions - JIT Compilation
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
        std::cout << "\n======================================" << std::endl;
        std::cout << "CCCL Reduce Build Phase (JIT + mcCub)" << std::endl;
        std::cout << "======================================" << std::endl;

        // Step 1: Generate kernel source with mcCub includes
        std::string kernel_src = generate_reduce_kernel_with_mccub(op, d_in);
        std::cout << "\n[1/4] Generated kernel source with mcCub integration:\n";
        std::cout << "--------------------------------------\n";
        std::cout << kernel_src;
        std::cout << "--------------------------------------\n";

        // Step 2: Create MCRTC program
        std::cout << "\n[2/4] Creating MCRTC program..." << std::endl;
        mcrtcProgram prog;
        mcrtcResult result = mcrtcCreateProgram(
            &prog,
            kernel_src.c_str(),
            "reduce_with_mccub.cu",
            0, nullptr, nullptr
        );

        if (result != MCRTC_SUCCESS) {
            std::cerr << "Error: mcrtcCreateProgram failed" << std::endl;
            return mcErrorUnknown;
        }

        // Step 3: Prepare compile options
        std::vector<const char*> opts;
        opts.push_back("-xmaca");
        opts.push_back("-std=c++17");

        // Add mcCub include path
        opts.push_back("-I/mnt/data/minxi/1_15/mcCub");

        // Add MACA include path
        const char* maca_path = std::getenv("MACA_PATH");
        if (maca_path) {
            std::string maca_include = std::string("-I") + maca_path + "/include";
            opts.push_back(maca_include.c_str());
        }

        // Add user config
        std::vector<std::string> include_flags;
        if (build_config != nullptr) {
            if (build_config->num_extra_compile_flags > 0) {
                for (size_t i = 0; i < build_config->num_extra_compile_flags; ++i) {
                    opts.push_back(build_config->extra_compile_flags[i]);
                }
            }
            if (build_config->num_extra_include_dirs > 0) {
                for (size_t i = 0; i < build_config->num_extra_include_dirs; ++i) {
                    include_flags.push_back(
                        std::string("-I") + build_config->extra_include_dirs[i]
                    );
                    opts.push_back(include_flags.back().c_str());
                }
            }
        }

        std::cout << "[3/4] Compiling with MCRTC (including mcCub headers)..." << std::endl;

        // Step 4: Compile
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

        std::cout << "✅ Compilation successful!" << std::endl;

        // Step 5: Get bitcode
        size_t codeSize;
        mcrtcGetBitcodeSize(prog, &codeSize);
        char* code = new char[codeSize];
        mcrtcGetBitcode(prog, code);

        std::cout << "[4/4] Generated bitcode: " << codeSize << " bytes" << std::endl;

        mcrtcDestroyProgram(&prog);

        // Step 6: Load module
        mcError_t err = mcModuleLoadData(&build->module, code);
        if (err != mcSuccess) {
            std::cerr << "Error: mcModuleLoadData failed" << std::endl;
            delete[] code;
            return err;
        }

        // Step 7: Get kernel function
        err = mcModuleGetFunction(&build->reduce_kernel, build->module, "cccl_reduce_kernel_wrapper");
        if (err != mcSuccess) {
            std::cerr << "Error: mcModuleGetFunction failed" << std::endl;
            mcModuleUnload(build->module);
            delete[] code;
            return err;
        }

        std::cout << "✅ Module loaded, kernel ready!" << std::endl;

        // Step 8: Save build result
        build->bitcode = code;
        build->bitcode_size = codeSize;
        build->type = d_in.value_type;
        build->op = op;
        build->d_in_iterator = d_in;

        // Copy initial value
        build->initial_value_size = d_in.value_type.size;
        build->initial_value = malloc(d_in.value_type.size);
        memcpy(build->initial_value, initial_value, d_in.value_type.size);

        std::cout << "\n✅ Build phase complete! Kernel is JIT-compiled with mcCub.\n" << std::endl;

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
    cccl_iterator_t iter = cccl_make_pointer_iterator(nullptr, type);
    return cccl_device_reduce_build_ex(build, op, iter, initial_value, build_config);
}

//==============================================================================
// Execute Functions - Using JIT-compiled kernel
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
        std::cout << "\n======================================" << std::endl;
        std::cout << "CCCL Reduce Execute Phase" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "Items: " << num_items << std::endl;

        void* d_in_ptr = d_in.state;

        // Allocate device memory for temp storage size query
        size_t* d_temp_storage_bytes;
        mcError_t err = mcMalloc((void**)&d_temp_storage_bytes, sizeof(size_t));
        if (err != mcSuccess) {
            std::cerr << "Error: mcMalloc failed for temp storage size" << std::endl;
            return err;
        }

        // Phase 1: Query temp storage size
        std::cout << "[1/3] Querying temp storage size..." << std::endl;
        void* d_temp_storage = nullptr;
        int phase = 0;  // Query phase

        void* args_query[] = {
            &d_in_ptr,
            &d_out,
            &d_temp_storage,
            &d_temp_storage_bytes,
            &num_items,
            build.initial_value,
            &phase
        };

        err = mcModuleLaunchKernel(
            build.reduce_kernel,
            1, 1, 1,
            1, 1, 1,
            0, stream,
            args_query,
            nullptr
        );

        if (err != mcSuccess) {
            std::cerr << "Error: Query phase kernel launch failed" << std::endl;
            mcFree(d_temp_storage_bytes);
            return err;
        }

        // Get the temp storage size
        size_t temp_storage_bytes = 0;
        mcMemcpy(&temp_storage_bytes, d_temp_storage_bytes, sizeof(size_t), mcMemcpyDeviceToHost);
        mcDeviceSynchronize();

        std::cout << "    Temp storage needed: " << temp_storage_bytes << " bytes" << std::endl;

        // Phase 2: Allocate temp storage
        if (temp_storage_bytes > 0) {
            std::cout << "[2/3] Allocating temp storage..." << std::endl;
            err = mcMalloc(&d_temp_storage, temp_storage_bytes);
            if (err != mcSuccess) {
                std::cerr << "Error: mcMalloc failed for temp storage" << std::endl;
                mcFree(d_temp_storage_bytes);
                return err;
            }
        }

        // Phase 3: Execute reduce
        std::cout << "[3/3] Executing reduction with mcCub..." << std::endl;
        phase = 1;  // Execute phase

        void* args_exec[] = {
            &d_in_ptr,
            &d_out,
            &d_temp_storage,
            &d_temp_storage_bytes,
            &num_items,
            build.initial_value,
            &phase
        };

        err = mcModuleLaunchKernel(
            build.reduce_kernel,
            1, 1, 1,
            1, 1, 1,
            0, stream,
            args_exec,
            nullptr
        );

        if (err != mcSuccess) {
            std::cerr << "Error: Execute phase kernel launch failed" << std::endl;
            if (d_temp_storage) mcFree(d_temp_storage);
            mcFree(d_temp_storage_bytes);
            return err;
        }

        mcDeviceSynchronize();

        // Cleanup
        if (d_temp_storage) mcFree(d_temp_storage);
        mcFree(d_temp_storage_bytes);

        std::cout << "✅ Reduction complete!\n" << std::endl;

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
    cccl_iterator_t iter = cccl_make_pointer_iterator(d_in, build.type);
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
