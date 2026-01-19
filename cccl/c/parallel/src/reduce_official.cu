//==============================================================================
//
// MACA CCCL C Binding - Reduce Implementation
// Adapted for MACA - Phase 5 (CCCL-Style Architecture)
//
// Architecture Overview (matching NVIDIA CCCL):
// ============================================
//
// NVIDIA CCCL:
//   - reduce.cu is host-side C++ orchestration code
//   - Includes CUB headers: #include <cub/device/device_reduce.cuh>
//   - Dispatches to CUB's internal APIs (DeviceReduce::Sum/Min/Max)
//   - Uses NVRTC for JIT compilation of custom iterators/operations
//   - Two-phase execution: query temp storage, then execute
//
// MACA CCCL (this file):
//   - reduce_official.cu is host-side C++ orchestration code
//   - Includes mcCub headers: #include <mccub/device/device_reduce.cuh>
//   - Dispatches to mcCub's APIs (DeviceReduce::Sum/Min/Max)
//   - Uses MCRTC for JIT compilation (ready for custom iterators/ops)
//   - Two-phase execution: query temp storage, then execute
//
// Key Design Decisions:
// ====================
// 1. Host-side dispatch: mcCub::DeviceReduce is called from HOST code,
//    not from inside GPU kernels (this matches CCCL/CUB design)
//
// 2. Build/Execute separation: Maintains CCCL's API pattern of separating
//    compilation/configuration (build) from execution
//
// 3. Type-specific dispatch: Template dispatch based on data type
//    (matches CCCL's type handling)
//
// 4. Future: JIT compilation will be added for custom iterators/operations
//    using MCRTC (analogous to CCCL's NVRTC usage)
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

// Include mcCub for host-side API (like CCCL includes CUB)
#include <mccub/device/device_reduce.cuh>

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
// mcCub Reduce Dispatcher - CCCL-Style
//==============================================================================

// This is analogous to NVIDIA CCCL's reduce dispatcher
// In CCCL: host code calls CUB's DeviceReduce APIs
// In MACA: host code calls mcCub's DeviceReduce APIs
//
// Note: Full CCCL architecture uses JIT compilation for custom iterators/ops
// For now, we use mcCub directly for builtin operations (PLUS, MIN, MAX)

template<typename T>
static mcError_t dispatch_mccub_reduce(
    cccl_op_kind_t op_type,
    const T* d_in,
    T* d_out,
    uint64_t num_items,
    T init_value,
    mcStream_t stream
) {
    // Allocate temp storage
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    mcError_t err = mcSuccess;

    try {
        // Dispatch to appropriate mcCub function based on operation type
        switch (op_type) {
            case CCCL_PLUS:
                // Phase 1: Query temp storage size
                mccub::DeviceReduce::Sum(
                    d_temp_storage, temp_storage_bytes,
                    d_in, d_out, static_cast<int>(num_items), stream
                );

                // Allocate temp storage
                mcMalloc(&d_temp_storage, temp_storage_bytes);

                // Phase 2: Execute reduction
                mccub::DeviceReduce::Sum(
                    d_temp_storage, temp_storage_bytes,
                    d_in, d_out, static_cast<int>(num_items), stream
                );
                break;

            case CCCL_MINIMUM:
                // Phase 1: Query
                mccub::DeviceReduce::Min(
                    d_temp_storage, temp_storage_bytes,
                    d_in, d_out, static_cast<int>(num_items), stream
                );

                // Allocate
                mcMalloc(&d_temp_storage, temp_storage_bytes);

                // Phase 2: Execute
                mccub::DeviceReduce::Min(
                    d_temp_storage, temp_storage_bytes,
                    d_in, d_out, static_cast<int>(num_items), stream
                );
                break;

            case CCCL_MAXIMUM:
                // Phase 1: Query
                mccub::DeviceReduce::Max(
                    d_temp_storage, temp_storage_bytes,
                    d_in, d_out, static_cast<int>(num_items), stream
                );

                // Allocate
                mcMalloc(&d_temp_storage, temp_storage_bytes);

                // Phase 2: Execute
                mccub::DeviceReduce::Max(
                    d_temp_storage, temp_storage_bytes,
                    d_in, d_out, static_cast<int>(num_items), stream
                );
                break;

            default:
                err = mcErrorInvalidValue;
        }

        // Cleanup temp storage
        if (d_temp_storage != nullptr) {
            mcFree(d_temp_storage);
        }

        return err;
    }
    catch (...) {
        if (d_temp_storage != nullptr) {
            mcFree(d_temp_storage);
        }
        return mcErrorUnknown;
    }
}

//==============================================================================
// Build Functions - CCCL-Style Configuration Phase
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
        std::cout << "CCCL Reduce Build Phase (CCCL-Style)" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "Operation: " << get_mccub_op_name(op.type) << std::endl;
        std::cout << "Data type: " << get_type_name(d_in.value_type.type) << std::endl;

        // Initialize build result
        build->bitcode = nullptr;
        build->bitcode_size = 0;
        build->module = nullptr;
        build->reduce_kernel = nullptr;
        build->type = d_in.value_type;
        build->op = op;
        build->d_in_iterator = d_in;

        // Copy initial value
        build->initial_value_size = d_in.value_type.size;
        build->initial_value = malloc(d_in.value_type.size);
        if (build->initial_value == nullptr) {
            return mcErrorMemoryAllocation;
        }
        memcpy(build->initial_value, initial_value, d_in.value_type.size);

        std::cout << "\n✅ Build phase complete!" << std::endl;
        std::cout << "    Architecture: Direct mcCub dispatch (matches CCCL using CUB)" << std::endl;
        std::cout << "    Ready to execute reduction.\n" << std::endl;

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
// Execute Functions - mcCub Dispatch (CCCL-Style)
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
        std::cout << "Operation: " << get_mccub_op_name(build.op.type) << std::endl;

        void* d_in_ptr = d_in.state;
        mcError_t err = mcSuccess;

        // Dispatch to mcCub based on data type
        // This matches CCCL's pattern of dispatching to CUB
        switch (build.type.type) {
            case CCCL_INT32:
                err = dispatch_mccub_reduce<int32_t>(
                    build.op.type,
                    static_cast<const int32_t*>(d_in_ptr),
                    static_cast<int32_t*>(d_out),
                    num_items,
                    *static_cast<int32_t*>(build.initial_value),
                    stream
                );
                break;

            case CCCL_INT64:
                err = dispatch_mccub_reduce<int64_t>(
                    build.op.type,
                    static_cast<const int64_t*>(d_in_ptr),
                    static_cast<int64_t*>(d_out),
                    num_items,
                    *static_cast<int64_t*>(build.initial_value),
                    stream
                );
                break;

            case CCCL_FLOAT32:
                err = dispatch_mccub_reduce<float>(
                    build.op.type,
                    static_cast<const float*>(d_in_ptr),
                    static_cast<float*>(d_out),
                    num_items,
                    *static_cast<float*>(build.initial_value),
                    stream
                );
                break;

            case CCCL_FLOAT64:
                err = dispatch_mccub_reduce<double>(
                    build.op.type,
                    static_cast<const double*>(d_in_ptr),
                    static_cast<double*>(d_out),
                    num_items,
                    *static_cast<double*>(build.initial_value),
                    stream
                );
                break;

            default:
                std::cerr << "Error: Unsupported data type" << std::endl;
                return mcErrorInvalidValue;
        }

        if (err == mcSuccess) {
            std::cout << "✅ Reduction complete!\n" << std::endl;
        } else {
            std::cerr << "❌ Reduction failed with error: " << err << std::endl;
        }

        return err;
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
        // Note: In this version we don't use JIT, so no module to unload
        // This is kept for API compatibility
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
