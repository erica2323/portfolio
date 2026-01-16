//==============================================================================
//
// MACA CCCL Reduce Test Suite
// Phase 1: Pointer + PLUS only
//
//==============================================================================

#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>

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
// Test: Sum of integers (small array, single tile)
//==============================================================================

bool test_sum_small() {
    print_test_header<int32_t>("Test 1: Sum (Small Array - Single Tile)", 100);

    const size_t n = 100;
    std::vector<int32_t> h_in(n);

    // Input: [1, 2, 3, ..., 100]
    for (size_t i = 0; i < n; i++) {
        h_in[i] = static_cast<int32_t>(i + 1);
    }

    // Expected: 1 + 2 + ... + 100 = 100 * 101 / 2 = 5050
    int32_t expected = n * (n + 1) / 2;

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

    // Setup operation (PLUS)
    cccl_op_t op;
    op.type = CCCL_PLUS;
    op.name = "sum";
    op.code = nullptr;
    op.code_size = 0;
    op.code_type = CCCL_OP_LTOIR;
    op.size = 0;
    op.alignment = 0;
    op.state = nullptr;

    // Initial value: 0
    int32_t init_value = 0;

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
            passed = check_result(h_out, expected, "Sum (Small)");
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
// Test: Sum of integers (large array, multi-tile)
//==============================================================================

bool test_sum_large() {
    print_test_header<int32_t>("Test 2: Sum (Large Array - Multi-Tile)", 10000);

    const size_t n = 10000;
    std::vector<int32_t> h_in(n);

    // Input: [1, 2, 3, ..., 10000]
    for (size_t i = 0; i < n; i++) {
        h_in[i] = static_cast<int32_t>(i + 1);
    }

    // Expected: 1 + 2 + ... + 10000 = 10000 * 10001 / 2 = 50005000
    int64_t expected = (int64_t)n * (n + 1) / 2;

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

    // Setup operation (PLUS)
    cccl_op_t op;
    op.type = CCCL_PLUS;
    op.name = "sum";
    op.code = nullptr;
    op.code_size = 0;
    op.code_type = CCCL_OP_LTOIR;
    op.size = 0;
    op.alignment = 0;
    op.state = nullptr;

    // Initial value: 0
    int32_t init_value = 0;

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
            passed = check_result<int64_t>(h_out, expected, "Sum (Large)");
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
// Test: Sum of floats
//==============================================================================

bool test_sum_float() {
    print_test_header<float>("Test 3: Sum (Float32)", 1000);

    const size_t n = 1000;
    std::vector<float> h_in(n);

    // Input: [0.5, 1.5, 2.5, ..., 999.5]
    for (size_t i = 0; i < n; i++) {
        h_in[i] = i + 0.5f;
    }

    // Expected: sum = 0.5 + 1.5 + 2.5 + ... + 999.5
    //         = (0 + 1 + 2 + ... + 999) + 1000 * 0.5
    //         = 999 * 1000 / 2 + 500
    //         = 499500 + 500 = 500000
    float expected = 500000.0f;

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

    // Setup operation (PLUS)
    cccl_op_t op;
    op.type = CCCL_PLUS;
    op.name = "sum";
    op.code = nullptr;
    op.code_size = 0;
    op.code_type = CCCL_OP_LTOIR;
    op.size = 0;
    op.alignment = 0;
    op.state = nullptr;

    // Initial value: 0.0f
    float init_value = 0.0f;

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

            // Check result (with tolerance for floating point)
            passed = check_result(h_out, expected, "Sum (Float)", 1.0);
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
// Test: Sum with non-zero initial value
//==============================================================================

bool test_sum_with_init() {
    print_test_header<int32_t>("Test 4: Sum with Initial Value", 50);

    const size_t n = 50;
    std::vector<int32_t> h_in(n);

    // Input: [1, 1, 1, ..., 1] (50 ones)
    for (size_t i = 0; i < n; i++) {
        h_in[i] = 1;
    }

    // Initial value: 100
    int32_t init_value = 100;

    // Expected: 100 + 50 = 150
    int32_t expected = init_value + n;

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

    // Setup operation (PLUS)
    cccl_op_t op;
    op.type = CCCL_PLUS;
    op.name = "sum";
    op.code = nullptr;
    op.code_size = 0;
    op.code_type = CCCL_OP_LTOIR;
    op.size = 0;
    op.alignment = 0;
    op.state = nullptr;

    // Build (with initial value = 100)
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
            passed = check_result(h_out, expected, "Sum with Init");
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
    std::cout << "MACA CCCL Reduce Test Suite (Phase 1)" << std::endl;
    std::cout << "==========================================" << std::endl;

    int test_count = 0;
    int passed_count = 0;

    // Run tests
    if (test_sum_small()) passed_count++;
    test_count++;

    if (test_sum_large()) passed_count++;
    test_count++;

    if (test_sum_float()) passed_count++;
    test_count++;

    if (test_sum_with_init()) passed_count++;
    test_count++;

    // Summary
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Passed: " << passed_count << "/" << test_count << std::endl;
    std::cout << "==========================================" << std::endl;

    return (passed_count == test_count) ? 0 : 1;
}
