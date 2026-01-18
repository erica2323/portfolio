//==============================================================================
//
// MACA CCCL C Binding - Reduce Implementation
// Adapted for MACA - Phase 4 (mcCub Integration)
//
//==============================================================================

#include <cccl/c/reduce_official.h>
#include <cstring>
#include <iostream>

// Include mcCub
#include <mccub/device/device_reduce.cuh>

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
        // Save configuration (no JIT compilation needed with mcCub!)
        build->type = d_in.value_type;
        build->op = op;
        build->d_in_iterator = d_in;

        // Copy initial value
        build->initial_value_size = d_in.value_type.size;
        build->initial_value = malloc(d_in.value_type.size);
        memcpy(build->initial_value, initial_value, d_in.value_type.size);

        std::cout << "Reduce configuration saved (mcCub mode - no JIT compilation)" << std::endl;

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
        // Determine temporary storage size
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        mcError_t err = mcSuccess;

        // Extract base pointer
        void* d_in_ptr = d_in.state;

        // Call appropriate mcCub function based on operation type
        switch (build.op.type) {
            case CCCL_PLUS: {
                std::cout << "Calling mcCub DeviceReduce::Sum for " << num_items << " items" << std::endl;

                // Dispatch based on data type
                switch (build.type.type) {
                    case CCCL_INT32: {
                        int32_t* d_in_typed = (int32_t*)d_in_ptr;
                        int32_t* d_out_typed = (int32_t*)d_out;
                        int32_t init_val = *(int32_t*)build.initial_value;

                        // Query temp storage size
                        err = mccub::DeviceReduce::Sum(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );
                        if (err != mcSuccess) return err;

                        // Allocate temp storage
                        err = mcMalloc(&d_temp_storage, temp_storage_bytes);
                        if (err != mcSuccess) return err;

                        // Run reduction
                        err = mccub::DeviceReduce::Sum(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );

                        // Add initial value if non-zero
                        if (init_val != 0) {
                            std::cerr << "Warning: mcCub Sum doesn't support non-zero initial value directly" << std::endl;
                            // TODO: Add init_val to result on device
                        }

                        break;
                    }
                    case CCCL_FLOAT32: {
                        float* d_in_typed = (float*)d_in_ptr;
                        float* d_out_typed = (float*)d_out;
                        float init_val = *(float*)build.initial_value;

                        err = mccub::DeviceReduce::Sum(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );
                        if (err != mcSuccess) return err;

                        err = mcMalloc(&d_temp_storage, temp_storage_bytes);
                        if (err != mcSuccess) return err;

                        err = mccub::DeviceReduce::Sum(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );

                        if (init_val != 0.0f) {
                            std::cerr << "Warning: mcCub Sum doesn't support non-zero initial value directly" << std::endl;
                        }

                        break;
                    }
                    default:
                        std::cerr << "Error: Unsupported type for PLUS operation" << std::endl;
                        return mcErrorInvalidValue;
                }
                break;
            }

            case CCCL_MINIMUM: {
                std::cout << "Calling mcCub DeviceReduce::Min for " << num_items << " items" << std::endl;

                switch (build.type.type) {
                    case CCCL_INT32: {
                        int32_t* d_in_typed = (int32_t*)d_in_ptr;
                        int32_t* d_out_typed = (int32_t*)d_out;

                        err = mccub::DeviceReduce::Min(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );
                        if (err != mcSuccess) return err;

                        err = mcMalloc(&d_temp_storage, temp_storage_bytes);
                        if (err != mcSuccess) return err;

                        err = mccub::DeviceReduce::Min(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );

                        break;
                    }
                    case CCCL_FLOAT32: {
                        float* d_in_typed = (float*)d_in_ptr;
                        float* d_out_typed = (float*)d_out;

                        err = mccub::DeviceReduce::Min(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );
                        if (err != mcSuccess) return err;

                        err = mcMalloc(&d_temp_storage, temp_storage_bytes);
                        if (err != mcSuccess) return err;

                        err = mccub::DeviceReduce::Min(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );

                        break;
                    }
                    default:
                        std::cerr << "Error: Unsupported type for MIN operation" << std::endl;
                        return mcErrorInvalidValue;
                }
                break;
            }

            case CCCL_MAXIMUM: {
                std::cout << "Calling mcCub DeviceReduce::Max for " << num_items << " items" << std::endl;

                switch (build.type.type) {
                    case CCCL_INT32: {
                        int32_t* d_in_typed = (int32_t*)d_in_ptr;
                        int32_t* d_out_typed = (int32_t*)d_out;

                        err = mccub::DeviceReduce::Max(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );
                        if (err != mcSuccess) return err;

                        err = mcMalloc(&d_temp_storage, temp_storage_bytes);
                        if (err != mcSuccess) return err;

                        err = mccub::DeviceReduce::Max(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );

                        break;
                    }
                    case CCCL_FLOAT32: {
                        float* d_in_typed = (float*)d_in_ptr;
                        float* d_out_typed = (float*)d_out;

                        err = mccub::DeviceReduce::Max(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );
                        if (err != mcSuccess) return err;

                        err = mcMalloc(&d_temp_storage, temp_storage_bytes);
                        if (err != mcSuccess) return err;

                        err = mccub::DeviceReduce::Max(
                            d_temp_storage, temp_storage_bytes,
                            d_in_typed, d_out_typed, (int)num_items, stream
                        );

                        break;
                    }
                    default:
                        std::cerr << "Error: Unsupported type for MAX operation" << std::endl;
                        return mcErrorInvalidValue;
                }
                break;
            }

            default:
                std::cerr << "Error: Unsupported operation" << std::endl;
                return mcErrorInvalidValue;
        }

        // Free temp storage
        if (d_temp_storage != nullptr) {
            mcFree(d_temp_storage);
        }

        if (err != mcSuccess) {
            std::cerr << "Error: mcCub DeviceReduce failed" << std::endl;
            return err;
        }

        std::cout << "mcCub reduction completed successfully" << std::endl;
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
