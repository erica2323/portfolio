//==============================================================================
// MACA CCCL C Binding - Reduce Implementation (Phase B: with JIT support)
//==============================================================================

#include <cccl/c/reduce_official.h>
#include <cccl/c/jit_compiler.h>  // 新增
#include <cstring>
#include <iostream>

// Include mcCub internal dispatch layer
#include <mccub/device/dispatch/dispatch_reduce.cuh>

using namespace cub;

//==============================================================================
// Helper Functions
//==============================================================================

static std::string get_type_name(cccl_type_enum type) {
    switch (type) {
        case CCCL_INT32: return "int32_t";
        case CCCL_INT64: return "int64_t";
        case CCCL_FLOAT32: return "float";
        case CCCL_FLOAT64: return "double";
        default: return "unknown";
    }
}

static std::string get_op_name(cccl_op_kind_t op) {
    switch (op) {
        case CCCL_PLUS: return "Sum";
        case CCCL_MINIMUM: return "Min";
        case CCCL_MAXIMUM: return "Max";
        case CCCL_STATELESS: return "CustomStateless";
        case CCCL_STATEFUL: return "CustomStateful";
        default: return "Unknown";
    }
}

//==============================================================================
// Builtin Operators Dispatch (Pre-compiled mcCub templates)
//==============================================================================

template<typename T>
static mcError_t dispatch_reduce_builtin(
    void* temp_storage,
    size_t* temp_storage_bytes,
    cccl_op_kind_t op_type,
    const T* d_in,
    T* d_out,
    int num_items,
    T init_value,
    mcStream_t stream
) {
    mcError_t err = mcSuccess;

    try {
        switch (op_type) {
            case CCCL_PLUS: {
                err = DispatchReduce<const T*, T*, int, cub::Sum>::Dispatch(
                    temp_storage,
                    *temp_storage_bytes,
                    d_in,
                    d_out,
                    num_items,
                    cub::Sum(),
                    init_value,
                    stream,
                    false
                );
                break;
            }

            case CCCL_MINIMUM: {
                err = DispatchReduce<const T*, T*, int, cub::Min>::Dispatch(
                    temp_storage,
                    *temp_storage_bytes,
                    d_in,
                    d_out,
                    num_items,
                    cub::Min(),
                    init_value,
                    stream,
                    false
                );
                break;
            }

            case CCCL_MAXIMUM: {
                err = DispatchReduce<const T*, T*, int, cub::Max>::Dispatch(
                    temp_storage,
                    *temp_storage_bytes,
                    d_in,
                    d_out,
                    num_items,
                    cub::Max(),
                    init_value,
                    stream,
                    false
                );
                break;
            }

            default:
                err = mcErrorInvalidValue;
        }

        return err;
    }
    catch (...) {
        return mcErrorUnknown;
    }
}

//==============================================================================
// JIT Path: Custom Operators
//==============================================================================

template<typename T>
static mcError_t dispatch_reduce_jit(
    void* temp_storage,
    size_t* temp_storage_bytes,
    cccl_jit_kernel_t* jit_kernel,
    const T* d_in,
    T* d_out,
    int num_items,
    T init_value,
    mcStream_t stream
) {
    std::cout << "[JIT Reduce] Launching JIT-compiled kernel" << std::endl;

    // Simple kernel launch (for now, this is a placeholder)
    // In production, you'd integrate this with CUB's DispatchReduce
    // by passing the JIT-compiled operator functor

    // Phase 1: Query temp storage (use heuristic for now)
    if (temp_storage == nullptr) {
        // Rough estimate: num_items * sizeof(T) for intermediate reductions
        *temp_storage_bytes = num_items * sizeof(T);
        std::cout << "[JIT Reduce] Query phase: " << *temp_storage_bytes << " bytes" << std::endl;
        return mcSuccess;
    }

    // Phase 2: Launch kernel
    // This is a simplified example - in practice you'd use CUB's infrastructure

    // For demonstration, use mcLaunchKernel (or equivalent MACA API)
    // void* args[] = { &d_in, &d_out, &num_items, &init_value };
    //
    // mcError_t err = mcLaunchKernel(
    //     jit_kernel->kernel_func,
    //     1, 1, 1,    // grid
    //     256, 1, 1,  // block
    //     jit_kernel->shared_mem_bytes,
    //     stream,
    //     args,
    //     nullptr
    // );

    // TODO: Integrate JIT kernel with CUB's block-level reduction primitives
    std::cout << "[JIT Reduce] ⚠️  JIT execution not fully implemented yet" << std::endl;
    std::cout << "[JIT Reduce] Kernel function: " << jit_kernel->kernel_func << std::endl;

    return mcErrorNotSupported;  // Placeholder until full integration
}

//==============================================================================
// Build Function (Extended for JIT)
//==============================================================================

mcError_t cccl_device_reduce_build(
    cccl_device_reduce_build_result_t* build,
    cccl_iterator_t d_in,
    cccl_iterator_t d_out,
    cccl_op_t op,
    cccl_value_t h_init,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path
) {
    if (build == nullptr) {
        return mcErrorInvalidValue;
    }

    try {
        // Store configuration
        build->type = d_in.value_type;
        build->op = op;
        build->d_in_iterator = d_in;

        // Copy initial value
        build->initial_value_size = h_init.type.size;
        build->initial_value = malloc(h_init.type.size);
        if (build->initial_value == nullptr) {
            return mcErrorMemoryAllocation;
        }
        memcpy(build->initial_value, h_init.state, h_init.type.size);

        std::cout << "[CCCL Build] Op=" << get_op_name(op.type)
                  << ", Type=" << get_type_name(d_in.value_type.type) << std::endl;

        // 🔥 NEW: JIT compilation for custom operators
        if (op.type == CCCL_STATELESS || op.type == CCCL_STATEFUL) {
            if (op.code != nullptr && op.code_size > 0) {
                std::cout << "[CCCL Build] 🚀 Compiling custom operator via JIT..." << std::endl;

                cccl_jit_kernel_t* jit_kernel = nullptr;
                mcError_t err = cccl_jit_compile_reduce_op(
                    &op,
                    &d_in.value_type,
                    cc_major,
                    cc_minor,
                    cub_path,
                    thrust_path,
                    libcudacxx_path,
                    &jit_kernel
                );

                if (err != mcSuccess) {
                    std::cerr << "[CCCL Build] ❌ JIT compilation failed: " << err << std::endl;
                    free(build->initial_value);
                    return err;
                }

                // Store JIT kernel in build result (extend struct if needed)
                // For now, store in op.state
                build->op.state = jit_kernel;

                std::cout << "[CCCL Build] ✅ JIT compilation successful" << std::endl;
            } else {
                std::cerr << "[CCCL Build] ❌ Custom operator requires code" << std::endl;
                free(build->initial_value);
                return mcErrorInvalidValue;
            }
        } else {
            // Validate builtin operators
            if (op.type != CCCL_PLUS && op.type != CCCL_MINIMUM && op.type != CCCL_MAXIMUM) {
                std::cerr << "[CCCL Build] ❌ Unsupported builtin operator" << std::endl;
                free(build->initial_value);
                return mcErrorInvalidValue;
            }
        }

        return mcSuccess;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return mcErrorUnknown;
    }
}

//==============================================================================
// Compute Function (Extended for JIT)
//==============================================================================

mcError_t cccl_device_reduce(
    cccl_device_reduce_build_result_t build,
    void* temp_storage,
    size_t* temp_storage_bytes,
    cccl_iterator_t d_in,
    cccl_iterator_t d_out,
    uint64_t num_items,
    cccl_op_t op,
    cccl_value_t h_init,
    mcStream_t stream
) {
    if (num_items == 0) {
        if (temp_storage_bytes != nullptr) {
            *temp_storage_bytes = 0;
        }
        return mcSuccess;
    }

    try {
        void* d_in_ptr = d_in.state;
        void* d_out_ptr = d_out.state;

        if (temp_storage == nullptr) {
            std::cout << "[CCCL Reduce] Phase 1: Query temp storage bytes" << std::endl;
        } else {
            std::cout << "[CCCL Reduce] Phase 2: Execute (items=" << num_items << ")" << std::endl;
        }

        mcError_t err = mcSuccess;

        // 🔥 NEW: Check if this is a JIT-compiled operator
        if (build.op.state != nullptr &&
            (build.op.type == CCCL_STATELESS || build.op.type == CCCL_STATEFUL)) {

            std::cout << "[CCCL Reduce] Using JIT path" << std::endl;
            cccl_jit_kernel_t* jit_kernel = static_cast<cccl_jit_kernel_t*>(build.op.state);

            // Dispatch to JIT kernel
            switch (build.type.type) {
                case CCCL_INT32:
                    err = dispatch_reduce_jit<int32_t>(
                        temp_storage, temp_storage_bytes, jit_kernel,
                        static_cast<const int32_t*>(d_in_ptr),
                        static_cast<int32_t*>(d_out_ptr),
                        static_cast<int>(num_items),
                        *static_cast<int32_t*>(build.initial_value),
                        stream
                    );
                    break;

                case CCCL_FLOAT32:
                    err = dispatch_reduce_jit<float>(
                        temp_storage, temp_storage_bytes, jit_kernel,
                        static_cast<const float*>(d_in_ptr),
                        static_cast<float*>(d_out_ptr),
                        static_cast<int>(num_items),
                        *static_cast<float*>(build.initial_value),
                        stream
                    );
                    break;

                // Add other types...

                default:
                    std::cerr << "[CCCL Reduce] ❌ Unsupported JIT type" << std::endl;
                    return mcErrorInvalidValue;
            }
        } else {
            // Builtin operator path
            std::cout << "[CCCL Reduce] Using builtin path" << std::endl;

            switch (build.type.type) {
                case CCCL_INT32:
                    err = dispatch_reduce_builtin<int32_t>(
                        temp_storage,
                        temp_storage_bytes,
                        build.op.type,
                        static_cast<const int32_t*>(d_in_ptr),
                        static_cast<int32_t*>(d_out_ptr),
                        static_cast<int>(num_items),
                        *static_cast<int32_t*>(build.initial_value),
                        stream
                    );
                    break;

                case CCCL_INT64:
                    err = dispatch_reduce_builtin<int64_t>(
                        temp_storage,
                        temp_storage_bytes,
                        build.op.type,
                        static_cast<const int64_t*>(d_in_ptr),
                        static_cast<int64_t*>(d_out_ptr),
                        static_cast<int>(num_items),
                        *static_cast<int64_t*>(build.initial_value),
                        stream
                    );
                    break;

                case CCCL_FLOAT32:
                    err = dispatch_reduce_builtin<float>(
                        temp_storage,
                        temp_storage_bytes,
                        build.op.type,
                        static_cast<const float*>(d_in_ptr),
                        static_cast<float*>(d_out_ptr),
                        static_cast<int>(num_items),
                        *static_cast<float*>(build.initial_value),
                        stream
                    );
                    break;

                case CCCL_FLOAT64:
                    err = dispatch_reduce_builtin<double>(
                        temp_storage,
                        temp_storage_bytes,
                        build.op.type,
                        static_cast<const double*>(d_in_ptr),
                        static_cast<double*>(d_out_ptr),
                        static_cast<int>(num_items),
                        *static_cast<double*>(build.initial_value),
                        stream
                    );
                    break;

                default:
                    std::cerr << "[CCCL Reduce] ❌ Unsupported builtin type" << std::endl;
                    return mcErrorInvalidValue;
            }
        }

        if (err == mcSuccess) {
            if (temp_storage == nullptr) {
                std::cout << "[CCCL Reduce] Required temp bytes: " << *temp_storage_bytes << std::endl;
            } else {
                std::cout << "[CCCL Reduce] ✅ Execution complete" << std::endl;
            }
        } else {
            std::cerr << "[CCCL Reduce] ❌ Failed with error: " << err << std::endl;
        }

        return err;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return mcErrorUnknown;
    }
}

//==============================================================================
// Cleanup (Extended for JIT)
//==============================================================================

mcError_t cccl_device_reduce_cleanup(
    cccl_device_reduce_build_result_t* build
) {
    if (build == nullptr) {
        return mcErrorInvalidValue;
    }

    try {
        // 🔥 NEW: Clean up JIT kernel (if cached, it's managed by cache)
        if (build->op.state != nullptr) {
            cccl_jit_kernel_t* jit_kernel = static_cast<cccl_jit_kernel_t*>(build->op.state);
            // Don't destroy if cached (cache manages lifetime)
            if (!jit_kernel->is_cached) {
                cccl_jit_kernel_destroy(jit_kernel);
            }
            build->op.state = nullptr;
        }

        if (build->initial_value != nullptr) {
            free(build->initial_value);
            build->initial_value = nullptr;
        }

        build->initial_value_size = 0;
        return mcSuccess;
    }
    catch (...) {
        return mcErrorUnknown;
    }
}
