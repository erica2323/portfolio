//==============================================================================
// MACA CCCL C Binding - Reduce Implementation
// Supports both static dispatch (Phase A) and JIT compilation (Phase B)
//==============================================================================

#include <cccl/c/reduce.h>
#include <cstring>
#include <iostream>
#include <string>
#include <sstream>

// mcCub dispatch layer
#include <mccub/device/dispatch/dispatch_reduce.cuh>

// JIT infrastructure
#include "mcrtc/mcrtc_helper.h"
#include "jit_templates/mappings/type_info.h"
#include "jit_templates/mappings/operation.h"
#include "jit_templates/mappings/iterator.h"

using namespace cub;

//==============================================================================
// Debug Helpers
//==============================================================================

static std::string get_type_name(cccl_type_enum type) {
    switch (type) {
        case CCCL_INT8:    return "int8_t";
        case CCCL_INT16:   return "int16_t";
        case CCCL_INT32:   return "int32_t";
        case CCCL_INT64:   return "int64_t";
        case CCCL_UINT8:   return "uint8_t";
        case CCCL_UINT16:  return "uint16_t";
        case CCCL_UINT32:  return "uint32_t";
        case CCCL_UINT64:  return "uint64_t";
        case CCCL_FLOAT32: return "float";
        case CCCL_FLOAT64: return "double";
        default:           return "unknown";
    }
}

static std::string get_op_name(cccl_op_kind_t op) {
    switch (op) {
        case CCCL_STATELESS: return "Stateless(JIT)";
        case CCCL_STATEFUL:  return "Stateful(JIT)";
        case CCCL_PLUS:      return "Sum";
        case CCCL_MINIMUM:   return "Min";
        case CCCL_MAXIMUM:   return "Max";
        case CCCL_MULTIPLIES: return "Product";
        default:             return "Unknown";
    }
}

//==============================================================================
// JIT Code Generation
//==============================================================================

static std::string generate_reduce_kernel_source(
    const cccl_iterator_t& d_in,
    const cccl_iterator_t& d_out,
    const cccl_op_t& op,
    const cccl_value_t& h_init
) {
    std::ostringstream ss;

    // Headers
    ss << R"(
#include <mccub/device/device_reduce.cuh>
#include <cstdint>

)";

    // Generate auxiliary declarations (extern device functions)
    if (!cccl::jit::cccl_op_t_mapping::is_builtin(op.type)) {
        ss << cccl::jit::cccl_op_t_mapping::aux(op, "reduce_op");
        ss << "\n";
    }

    // Generate type definitions
    ss << "// Type definitions\n";
    ss << cccl::jit::cccl_type_info_mapping::map(d_in.value_type, "ValueT") << "\n";

    // Generate operation type
    if (cccl::jit::cccl_op_t_mapping::is_builtin(op.type)) {
        ss << cccl::jit::cccl_op_t_mapping::map(op, "OpT") << "\n";
    } else {
        // Include user operation template
        ss << R"(
// User operation wrapper
template <typename Tag, size_t Size, size_t Alignment>
struct stateless_user_operation {
    template <typename T>
    __device__ __forceinline__
    T operator()(const T& lhs, const T& rhs) const {
        alignas(Alignment) char result_buf[Size];
        extern "C" __device__ void reduce_op_device_fn(
            void* __restrict__ result,
            const void* __restrict__ arg0,
            const void* __restrict__ arg1
        );
        reduce_op_device_fn(result_buf, &lhs, &rhs);
        return *reinterpret_cast<T*>(result_buf);
    }
};

struct reduce_op_tag {};
using OpT = stateless_user_operation<reduce_op_tag, )"
           << op.size << ", " << op.alignment << ">;\n";
    }

    // Generate kernel
    ss << R"(
extern "C" __global__ void cccl_reduce_kernel(
    const void* __restrict__ d_in_ptr,
    void* __restrict__ d_out_ptr,
    uint64_t num_items,
    void* __restrict__ temp_storage,
    size_t temp_storage_bytes,
    const void* __restrict__ init_ptr
) {
    using T = ValueT;

    const T* d_in = static_cast<const T*>(d_in_ptr);
    T* d_out = static_cast<T*>(d_out_ptr);
    T init = *static_cast<const T*>(init_ptr);

    // Use CUB block reduce
    cub::DeviceReduce::Reduce(
        temp_storage, temp_storage_bytes,
        d_in, d_out,
        static_cast<int>(num_items),
        OpT{},
        init
    );
}
)";

    return ss.str();
}

//==============================================================================
// Static Dispatch (Phase A - Builtin Operations)
//==============================================================================

template<typename T>
static mcError_t dispatch_reduce_static(
    void*           temp_storage,
    size_t*         temp_storage_bytes,
    cccl_op_kind_t  op_type,
    const T*        d_in,
    T*              d_out,
    int             num_items,
    T               init_value,
    mcStream_t      stream
) {
    mcError_t err = mcSuccess;

    try {
        switch (op_type) {
            case CCCL_PLUS: {
                err = DispatchReduce<const T*, T*, int, cub::Sum>::Dispatch(
                    temp_storage, *temp_storage_bytes,
                    d_in, d_out, num_items,
                    cub::Sum(), init_value,
                    stream, false
                );
                break;
            }
            case CCCL_MINIMUM: {
                err = DispatchReduce<const T*, T*, int, cub::Min>::Dispatch(
                    temp_storage, *temp_storage_bytes,
                    d_in, d_out, num_items,
                    cub::Min(), init_value,
                    stream, false
                );
                break;
            }
            case CCCL_MAXIMUM: {
                err = DispatchReduce<const T*, T*, int, cub::Max>::Dispatch(
                    temp_storage, *temp_storage_bytes,
                    d_in, d_out, num_items,
                    cub::Max(), init_value,
                    stream, false
                );
                break;
            }
            case CCCL_MULTIPLIES: {
                // For product, we'd need a custom functor or use Sum with log trick
                // For now, return error
                err = mcErrorInvalidValue;
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
// Check if JIT is needed
//==============================================================================

static bool needs_jit_compilation(
    const cccl_iterator_t& d_in,
    const cccl_iterator_t& d_out,
    const cccl_op_t& op
) {
    // Need JIT if:
    // 1. Operation is user-defined (STATELESS or STATEFUL)
    // 2. Input iterator is custom (not pointer)
    // 3. Output iterator is custom (not pointer)

    if (op.type == CCCL_STATELESS || op.type == CCCL_STATEFUL) {
        return true;
    }

    if (d_in.type == CCCL_ITERATOR) {
        return true;
    }

    if (d_out.type == CCCL_ITERATOR) {
        return true;
    }

    return false;
}

//==============================================================================
// Build Function
//==============================================================================

mcError_t cccl_device_reduce_build(
    cccl_device_reduce_build_result_t* build,
    cccl_iterator_t                    d_in,
    cccl_iterator_t                    d_out,
    cccl_op_t                          op,
    cccl_value_t                       h_init,
    int                                cc_major,
    int                                cc_minor,
    const char*                        cub_path,
    const char*                        thrust_path,
    const char*                        libcudacxx_path,
    const char*                        ctk_path
) {
    if (build == nullptr) {
        return mcErrorInvalidValue;
    }

    // Initialize build result
    memset(build, 0, sizeof(cccl_device_reduce_build_result_t));

    try {
        // Store configuration
        build->type          = d_in.value_type;
        build->op            = op;
        build->d_in_iterator = d_in;

        // Copy initial value
        build->initial_value_size = h_init.type.size;
        build->initial_value = malloc(h_init.type.size);
        if (build->initial_value == nullptr) {
            return mcErrorMemoryAllocation;
        }
        memcpy(build->initial_value, h_init.state, h_init.type.size);

        // Check if we need JIT compilation
        if (needs_jit_compilation(d_in, d_out, op)) {
            // Phase B: JIT compilation path
            std::cout << "[CCCL Build] JIT compilation for Op="
                      << get_op_name(op.type)
                      << ", Type=" << get_type_name(d_in.value_type.type)
                      << std::endl;

            // Generate kernel source
            std::string source = generate_reduce_kernel_source(d_in, d_out, op, h_init);

            // Build compile options
            std::vector<std::string> options = {"-x maca"};
            if (cub_path && strlen(cub_path) > 0) {
                options.push_back(std::string("-I") + cub_path);
            }
            if (ctk_path && strlen(ctk_path) > 0) {
                options.push_back(std::string("-I") + ctk_path + "/include");
            }

            // Compile and load
            auto result = cccl::mcrtc::Compiler::compileAndLoad(
                source,
                "cccl_reduce_kernel",
                options
            );

            if (!result.success) {
                std::cerr << "[CCCL Build] JIT compilation failed" << std::endl;
                free(build->initial_value);
                build->initial_value = nullptr;
                return mcErrorInvalidValue;
            }

            build->jit_module = result.module;
            build->jit_kernel = result.kernel;
            build->uses_jit   = 1;

            std::cout << "[CCCL Build] JIT compilation successful" << std::endl;
        } else {
            // Phase A: Static dispatch path
            std::cout << "[CCCL Build] Static dispatch for Op="
                      << get_op_name(op.type)
                      << ", Type=" << get_type_name(d_in.value_type.type)
                      << std::endl;

            build->uses_jit = 0;
        }

        return mcSuccess;
    }
    catch (const std::exception& e) {
        std::cerr << "[CCCL Build] Exception: " << e.what() << std::endl;
        if (build->initial_value) {
            free(build->initial_value);
            build->initial_value = nullptr;
        }
        return mcErrorUnknown;
    }
}

//==============================================================================
// Compute Function
//==============================================================================

mcError_t cccl_device_reduce(
    cccl_device_reduce_build_result_t build,
    void*                             temp_storage,
    size_t*                           temp_storage_bytes,
    cccl_iterator_t                   d_in,
    cccl_iterator_t                   d_out,
    uint64_t                          num_items,
    cccl_op_t                         op,
    cccl_value_t                      h_init,
    mcStream_t                        stream
) {
    if (num_items == 0) {
        if (temp_storage_bytes != nullptr) {
            *temp_storage_bytes = 0;
        }
        return mcSuccess;
    }

    try {
        if (temp_storage == nullptr) {
            std::cout << "[CCCL Reduce] Phase 1: Query temp storage bytes" << std::endl;
        } else {
            std::cout << "[CCCL Reduce] Phase 2: Execute (items=" << num_items << ")" << std::endl;
        }

        mcError_t err = mcSuccess;

        if (build.uses_jit) {
            // JIT path - launch compiled kernel
            mcFunction_t kernel = static_cast<mcFunction_t>(build.jit_kernel);

            if (temp_storage == nullptr) {
                // Query temp storage - for simplicity, use a fixed size
                // In production, you'd call a separate query kernel
                *temp_storage_bytes = 1024 * 1024;  // 1MB default
                return mcSuccess;
            }

            // Launch kernel
            void* d_in_ptr  = d_in.state;
            void* d_out_ptr = d_out.state;
            size_t ts_bytes = *temp_storage_bytes;

            void* args[] = {
                &d_in_ptr,
                &d_out_ptr,
                &num_items,
                &temp_storage,
                &ts_bytes,
                &build.initial_value
            };

            // Calculate grid/block dimensions
            // This is simplified - production code would tune these
            unsigned int blockSize = 256;
            unsigned int gridSize  = 1;  // Single block for reduce final stage

            err = cccl::mcrtc::launchKernel(
                kernel,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0,  // shared memory
                stream,
                args,
                nullptr
            );

            if (err != mcSuccess) {
                std::cerr << "[CCCL Reduce] JIT kernel launch failed: " << err << std::endl;
                return err;
            }

        } else {
            // Static dispatch path
            void* d_in_ptr  = d_in.state;
            void* d_out_ptr = d_out.state;

            switch (build.type.type) {
                case CCCL_INT32:
                    err = dispatch_reduce_static<int32_t>(
                        temp_storage, temp_storage_bytes,
                        build.op.type,
                        static_cast<const int32_t*>(d_in_ptr),
                        static_cast<int32_t*>(d_out_ptr),
                        static_cast<int>(num_items),
                        *static_cast<int32_t*>(build.initial_value),
                        stream
                    );
                    break;

                case CCCL_INT64:
                    err = dispatch_reduce_static<int64_t>(
                        temp_storage, temp_storage_bytes,
                        build.op.type,
                        static_cast<const int64_t*>(d_in_ptr),
                        static_cast<int64_t*>(d_out_ptr),
                        static_cast<int>(num_items),
                        *static_cast<int64_t*>(build.initial_value),
                        stream
                    );
                    break;

                case CCCL_FLOAT32:
                    err = dispatch_reduce_static<float>(
                        temp_storage, temp_storage_bytes,
                        build.op.type,
                        static_cast<const float*>(d_in_ptr),
                        static_cast<float*>(d_out_ptr),
                        static_cast<int>(num_items),
                        *static_cast<float*>(build.initial_value),
                        stream
                    );
                    break;

                case CCCL_FLOAT64:
                    err = dispatch_reduce_static<double>(
                        temp_storage, temp_storage_bytes,
                        build.op.type,
                        static_cast<const double*>(d_in_ptr),
                        static_cast<double*>(d_out_ptr),
                        static_cast<int>(num_items),
                        *static_cast<double*>(build.initial_value),
                        stream
                    );
                    break;

                default:
                    std::cerr << "[CCCL Reduce] Unsupported type: "
                              << static_cast<int>(build.type.type) << std::endl;
                    return mcErrorInvalidValue;
            }
        }

        if (err == mcSuccess) {
            if (temp_storage == nullptr) {
                std::cout << "[CCCL Reduce] Required temp bytes: "
                          << *temp_storage_bytes << std::endl;
            } else {
                std::cout << "[CCCL Reduce] Execution complete" << std::endl;
            }
        } else {
            std::cerr << "[CCCL Reduce] Failed with error: " << err << std::endl;
        }

        return err;
    }
    catch (const std::exception& e) {
        std::cerr << "[CCCL Reduce] Exception: " << e.what() << std::endl;
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
        // Free initial value
        if (build->initial_value != nullptr) {
            free(build->initial_value);
            build->initial_value = nullptr;
        }

        // Unload JIT module
        if (build->uses_jit && build->jit_module != nullptr) {
            mcError_t err = cccl::mcrtc::unloadModule(
                static_cast<mcModule_t>(build->jit_module)
            );
            if (err != mcSuccess) {
                std::cerr << "[CCCL Cleanup] Failed to unload module: " << err << std::endl;
            }
            build->jit_module = nullptr;
            build->jit_kernel = nullptr;
        }

        // Free cached bitcode
        if (build->jit_bitcode != nullptr) {
            free(build->jit_bitcode);
            build->jit_bitcode = nullptr;
        }

        build->initial_value_size = 0;
        build->jit_bitcode_size   = 0;
        build->uses_jit           = 0;

        return mcSuccess;
    }
    catch (...) {
        return mcErrorUnknown;
    }
}
