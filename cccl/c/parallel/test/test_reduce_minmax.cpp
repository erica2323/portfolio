//==============================================================================
//
// MACA CCCL Reduce Test Suite - MIN/MAX Operations
// Phase 2: Testing MINIMUM and MAXIMUM
//
//==============================================================================

#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <limits>

#include "cccl/c/types_official.h"
#include "cccl/c/reduce_official.h"

//==============================================================================
// Helper Functions
//==============================================================================

template<typename T>
void print_test_header(const char* test_name, size_t n) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << test_name << " (n=" << n << ")" << std::endl;
    std::cout << "==========================================" << std::endl;
}

template<typename T>
bool check_result(T result, T expected, const char* test_name, double epsilon = 1e-5) {
    bool passed = false;

    if constexpr (std::is_floating_point<T>::value) {
        passed = std::abs(result - expected) < epsilon;
    } else {
        passed = (result == expected);
    }

    std::cout << "Result:   " << result << std::endl;
    std::cout << "Expected: " << expected << std::endl;

    if (passed) {
        std::cout << "✅ " << test_name << " PASSED" << std::endl;
    } else {
        std::cout << "❌ " << test_name << " FAILED" << std::endl;
    }

    return passed;
}

//==============================================================================
// Test: Minimum (small array)
//==============================================================================

bool test_min_small() {
    print_test_header<int32_t>("Test 1: MIN (Small Array)", 100);

    const size_t n = 100;
    std::vector<int32_t> h_in(n);

    // Input: [100, 99, 98, ..., 1]
    for (size_t i = 0; i < n; i++) {
        h_in[i] = static_cast<int32_t>(n - i);
    }

    // Expected minimum: 1
    int32_t expected = 1;

    // Allocate device memory
    int32_t* d_in;
    int32_t* d_out;
    mcMalloc((void**)&d_in, n * sizeof(int32_t));
    mcMalloc((void**)&d_out, sizeof(int32_t));

    // Copy input to device
    mcMemcpy(d_in, h_in.data(), n * sizeof(int32_t), mcMemcpyHostToDevice);

    // Setup type info
    cccl_type_info type;
    type.type = CCCL_INT32;
    type.size = sizeof(int32_t);
    type.alignment = alignof(int32_t);

    // Setup operation (MINIMUM)
    cccl_op_t op;
    op.type = CCCL_MINIMUM;
    op.name = "min";
    op.code = nullptr;
    op.code_size = 0;
    op.code_type = CCCL_OP_LTOIR;
    op.size = 0;
    op.alignment = 0;
    op.state = nullptr;

    // Initial value: INT_MAX (identity for min)
    int32_t init_value = std::numeric_limits<int32_t>::max();

    // Build
    cccl_device_reduce_build_result_t build;
    mcError_t result = cccl_device_reduce_build(&build, op, type, &init_value, nullptr);

    bool passed = false;
    if (result == mcSuccess) {
        // Execute
        result = cccl_device_reduce(build, d_in, d_out, n, nullptr);
        if (result == mcSuccess) {
            // Copy result back
            int32_t h_out;
            mcMemcpy(&h_out, d_out, sizeof(int32_t), mcMemcpyDeviceToHost);

            // Synchronize
            mcDeviceSynchronize();

            // Check result
            passed = check_result(h_out, expected, "MIN (Small)");
        } else {
            std::cerr << "cccl_device_reduce failed: " << result << std::endl;
        }

        // Cleanup
        cccl_device_reduce_cleanup(&build);
    } else {
        std::cerr << "cccl_device_reduce_build failed: " << result << std::endl;
    }

    // Free device memory
    mcFree(d_in);
    mcFree(d_out);

    return passed;
}

//==============================================================================
// Test: Maximum (small array)
//==============================================================================

bool test_max_small() {
    print_test_header<int32_t>("Test 2: MAX (Small Array)", 100);

    const size_t n = 100;
    std::vector<int32_t> h_in(n);

    // Input: [1, 2, 3, ..., 100]
    for (size_t i = 0; i < n; i++) {
        h_in[i] = static_cast<int32_t>(i + 1);
    }

    // Expected maximum: 100
    int32_t expected = 100;

    // Allocate device memory
    int32_t* d_in;
    int32_t* d_out;
    mcMalloc((void**)&d_in, n * sizeof(int32_t));
    mcMalloc((void**)&d_out, sizeof(int32_t));

    // Copy input to device
    mcMemcpy(d_in, h_in.data(), n * sizeof(int32_t), mcMemcpyHostToDevice);

    // Setup type info
    cccl_type_info type;
    type.type = CCCL_INT32;
    type.size = sizeof(int32_t);
    type.alignment = alignof(int32_t);

    // Setup operation (MAXIMUM)
    cccl_op_t op;
    op.type = CCCL_MAXIMUM;
    op.name = "max";
    op.code = nullptr;
    op.code_size = 0;
    op.code_type = CCCL_OP_LTOIR;
    op.size = 0;
    op.alignment = 0;
    op.state = nullptr;

    // Initial value: INT_MIN (identity for max)
    int32_t init_value = std::numeric_limits<int32_t>::min();

    // Build
    cccl_device_reduce_build_result_t build;
    mcError_t result = cccl_device_reduce_build(&build, op, type, &init_value, nullptr);

    bool passed = false;
    if (result == mcSuccess) {
        // Execute
        result = cccl_device_reduce(build, d_in, d_out, n, nullptr);
        if (result == mcSuccess) {
            // Copy result back
            int32_t h_out;
            mcMemcpy(&h_out, d_out, sizeof(int32_t), mcMemcpyDeviceToHost);

            // Synchronize
            mcDeviceSynchronize();

            // Check result
            passed = check_result(h_out, expected, "MAX (Small)");
        } else {
            std::cerr << "cccl_device_reduce failed: " << result << std::endl;
        }

        // Cleanup
        cccl_device_reduce_cleanup(&build);
    } else {
        std::cerr << "cccl_device_reduce_build failed: " << result << std::endl;
    }

    // Free device memory
    mcFree(d_in);
    mcFree(d_out);

    return passed;
}

//==============================================================================
// Test: Minimum (large array)
//==============================================================================

bool test_min_large() {
    print_test_header<int32_t>("Test 3: MIN (Large Array - Multi-Tile)", 10000);

    const size_t n = 10000;
    std::vector<int32_t> h_in(n);

    // Input: [10000, 9999, ..., 1], with minimum at the end
    for (size_t i = 0; i < n; i++) {
        h_in[i] = static_cast<int32_t>(n - i);
    }

    // Expected minimum: 1
    int32_t expected = 1;

    // Allocate device memory
    int32_t* d_in;
    int32_t* d_out;
    mcMalloc((void**)&d_in, n * sizeof(int32_t));
    mcMalloc((void**)&d_out, sizeof(int32_t));

    // Copy input to device
    mcMemcpy(d_in, h_in.data(), n * sizeof(int32_t), mcMemcpyHostToDevice);

    // Setup type info
    cccl_type_info type;
    type.type = CCCL_INT32;
    type.size = sizeof(int32_t);
    type.alignment = alignof(int32_t);

    // Setup operation (MINIMUM)
    cccl_op_t op;
    op.type = CCCL_MINIMUM;
    op.name = "min";
    op.code = nullptr;
    op.code_size = 0;
    op.code_type = CCCL_OP_LTOIR;
    op.size = 0;
    op.alignment = 0;
    op.state = nullptr;

    // Initial value: INT_MAX (identity for min)
    int32_t init_value = std::numeric_limits<int32_t>::max();

    // Build
    cccl_device_reduce_build_result_t build;
    mcError_t result = cccl_device_reduce_build(&build, op, type, &init_value, nullptr);

    bool passed = false;
    if (result == mcSuccess) {
        // Execute
        result = cccl_device_reduce(build, d_in, d_out, n, nullptr);
        if (result == mcSuccess) {
            // Copy result back
            int32_t h_out;
            mcMemcpy(&h_out, d_out, sizeof(int32_t), mcMemcpyDeviceToHost);

            // Synchronize
            mcDeviceSynchronize();

            // Check result
            passed = check_result(h_out, expected, "MIN (Large)");
        } else {
            std::cerr << "cccl_device_reduce failed: " << result << std::endl;
        }

        // Cleanup
        cccl_device_reduce_cleanup(&build);
    } else {
        std::cerr << "cccl_device_reduce_build failed: " << result << std::endl;
    }

    // Free device memory
    mcFree(d_in);
    mcFree(d_out);

    return passed;
}

//==============================================================================
// Test: Maximum with floats
//==============================================================================

bool test_max_float() {
    print_test_header<float>("Test 4: MAX (Float32)", 1000);

    const size_t n = 1000;
    std::vector<float> h_in(n);

    // Input: [0.5, 1.5, ..., 999.5]
    for (size_t i = 0; i < n; i++) {
        h_in[i] = i + 0.5f;
    }

    // Expected maximum: 999.5
    float expected = 999.5f;

    // Allocate device memory
    float* d_in;
    float* d_out;
    mcMalloc((void**)&d_in, n * sizeof(float));
    mcMalloc((void**)&d_out, sizeof(float));

    // Copy input to device
    mcMemcpy(d_in, h_in.data(), n * sizeof(float), mcMemcpyHostToDevice);

    // Setup type info
    cccl_type_info type;
    type.type = CCCL_FLOAT32;
    type.size = sizeof(float);
    type.alignment = alignof(float);

    // Setup operation (MAXIMUM)
    cccl_op_t op;
    op.type = CCCL_MAXIMUM;
    op.name = "max";
    op.code = nullptr;
    op.code_size = 0;
    op.code_type = CCCL_OP_LTOIR;
    op.size = 0;
    op.alignment = 0;
    op.state = nullptr;

    // Initial value: -FLT_MAX (identity for max)
    float init_value = -std::numeric_limits<float>::max();

    // Build
    cccl_device_reduce_build_result_t build;
    mcError_t result = cccl_device_reduce_build(&build, op, type, &init_value, nullptr);

    bool passed = false;
    if (result == mcSuccess) {
        // Execute
        result = cccl_device_reduce(build, d_in, d_out, n, nullptr);
        if (result == mcSuccess) {
            // Copy result back
            float h_out;
            mcMemcpy(&h_out, d_out, sizeof(float), mcMemcpyDeviceToHost);

            // Synchronize
            mcDeviceSynchronize();

            // Check result
            passed = check_result(h_out, expected, "MAX (Float)", 0.01f);
        } else {
            std::cerr << "cccl_device_reduce failed: " << result << std::endl;
        }

        // Cleanup
        cccl_device_reduce_cleanup(&build);
    } else {
        std::cerr << "cccl_device_reduce_build failed: " << result << std::endl;
    }

    // Free device memory
    mcFree(d_in);
    mcFree(d_out);

    return passed;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "MACA CCCL Reduce MIN/MAX Test Suite" << std::endl;
    std::cout << "==========================================" << std::endl;

    int test_count = 0;
    int passed_count = 0;

    // Run tests
    if (test_min_small()) passed_count++;
    test_count++;

    if (test_max_small()) passed_count++;
    test_count++;

    if (test_min_large()) passed_count++;
    test_count++;

    if (test_max_float()) passed_count++;
    test_count++;

    // Summary
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Passed: " << passed_count << "/" << test_count << std::endl;
    std::cout << "==========================================" << std::endl;

    return (passed_count == test_count) ? 0 : 1;
}
