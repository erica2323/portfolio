//==============================================================================
//
// MACA CCCL Reduce Test - Phase 5 (True CCCL Style: JIT + mcCub)
//
//==============================================================================

#include <iostream>
#include <cstdint>
#include <vector>
#include <limits>

#include "cccl/c/types_official.h"
#include "cccl/c/reduce_official.h"

//==============================================================================
// Test 1: SUM with JIT + mcCub
//==============================================================================

bool test_sum() {
    std::cout << "\n==========================================\n";
    std::cout << "Test 1: SUM (JIT + mcCub Integration)\n";
    std::cout << "==========================================\n";

    const size_t n = 100;
    std::vector<int32_t> h_in(n);

    // Input: [1, 2, 3, ..., 100]
    for (size_t i = 0; i < n; i++) {
        h_in[i] = static_cast<int32_t>(i + 1);
    }

    // Expected: 1 + 2 + ... + 100 = 5050
    int32_t expected = 5050;

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

    // Initial value
    int32_t init_value = 0;

    // Build phase - JIT compile kernel with mcCub
    cccl_device_reduce_build_result_t build;
    mcError_t result = cccl_device_reduce_build(&build, op, type, &init_value, nullptr);

    bool passed = false;
    if (result == mcSuccess) {
        // Execute phase - run JIT-compiled kernel
        result = cccl_device_reduce(build, d_in, d_out, n, nullptr);
        if (result == mcSuccess) {
            // Copy result back
            int32_t h_out;
            mcMemcpy(&h_out, d_out, sizeof(int32_t), mcMemcpyDeviceToHost);
            mcDeviceSynchronize();

            std::cout << "\n------------------------------------------\n";
            std::cout << "Result:   " << h_out << std::endl;
            std::cout << "Expected: " << expected << std::endl;

            if (h_out == expected) {
                std::cout << "✅ Test PASSED!\n";
                passed = true;
            } else {
                std::cout << "❌ Test FAILED!\n";
            }
        } else {
            std::cerr << "❌ Execute failed: " << result << std::endl;
        }

        // Cleanup
        cccl_device_reduce_cleanup(&build);
    } else {
        std::cerr << "❌ Build failed: " << result << std::endl;
    }

    // Free device memory
    mcFree(d_in);
    mcFree(d_out);

    return passed;
}

//==============================================================================
// Test 2: MIN with JIT + mcCub
//==============================================================================

bool test_min() {
    std::cout << "\n==========================================\n";
    std::cout << "Test 2: MIN (JIT + mcCub Integration)\n";
    std::cout << "==========================================\n";

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

    // Initial value: INT_MAX
    int32_t init_value = std::numeric_limits<int32_t>::max();

    // Build phase
    cccl_device_reduce_build_result_t build;
    mcError_t result = cccl_device_reduce_build(&build, op, type, &init_value, nullptr);

    bool passed = false;
    if (result == mcSuccess) {
        // Execute phase
        result = cccl_device_reduce(build, d_in, d_out, n, nullptr);
        if (result == mcSuccess) {
            // Copy result back
            int32_t h_out;
            mcMemcpy(&h_out, d_out, sizeof(int32_t), mcMemcpyDeviceToHost);
            mcDeviceSynchronize();

            std::cout << "\n------------------------------------------\n";
            std::cout << "Result:   " << h_out << std::endl;
            std::cout << "Expected: " << expected << std::endl;

            if (h_out == expected) {
                std::cout << "✅ Test PASSED!\n";
                passed = true;
            } else {
                std::cout << "❌ Test FAILED!\n";
            }
        } else {
            std::cerr << "❌ Execute failed: " << result << std::endl;
        }

        // Cleanup
        cccl_device_reduce_cleanup(&build);
    } else {
        std::cerr << "❌ Build failed: " << result << std::endl;
    }

    // Free device memory
    mcFree(d_in);
    mcFree(d_out);

    return passed;
}

//==============================================================================
// Test 3: MAX with JIT + mcCub
//==============================================================================

bool test_max() {
    std::cout << "\n==========================================\n";
    std::cout << "Test 3: MAX (JIT + mcCub Integration)\n";
    std::cout << "==========================================\n";

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

    // Initial value: INT_MIN
    int32_t init_value = std::numeric_limits<int32_t>::min();

    // Build phase
    cccl_device_reduce_build_result_t build;
    mcError_t result = cccl_device_reduce_build(&build, op, type, &init_value, nullptr);

    bool passed = false;
    if (result == mcSuccess) {
        // Execute phase
        result = cccl_device_reduce(build, d_in, d_out, n, nullptr);
        if (result == mcSuccess) {
            // Copy result back
            int32_t h_out;
            mcMemcpy(&h_out, d_out, sizeof(int32_t), mcMemcpyDeviceToHost);
            mcDeviceSynchronize();

            std::cout << "\n------------------------------------------\n";
            std::cout << "Result:   " << h_out << std::endl;
            std::cout << "Expected: " << expected << std::endl;

            if (h_out == expected) {
                std::cout << "✅ Test PASSED!\n";
                passed = true;
            } else {
                std::cout << "❌ Test FAILED!\n";
            }
        } else {
            std::cerr << "❌ Execute failed: " << result << std::endl;
        }

        // Cleanup
        cccl_device_reduce_cleanup(&build);
    } else {
        std::cerr << "❌ Build failed: " << result << std::endl;
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
    std::cout << "====================================================\n";
    std::cout << " MACA CCCL Reduce Test Suite\n";
    std::cout << " Phase 5: True CCCL Style (JIT + mcCub)\n";
    std::cout << "====================================================\n";
    std::cout << "\nArchitecture:\n";
    std::cout << "  • JIT compile kernels at runtime\n";
    std::cout << "  • Generated kernels #include <mccub/...>\n";
    std::cout << "  • Wrapper kernel calls mcCub internally\n";
    std::cout << "  • Matches NVIDIA CCCL's design!\n";

    int test_count = 0;
    int passed_count = 0;

    if (test_sum()) passed_count++;
    test_count++;

    if (test_min()) passed_count++;
    test_count++;

    if (test_max()) passed_count++;
    test_count++;

    // Summary
    std::cout << "\n====================================================\n";
    std::cout << " Test Summary\n";
    std::cout << "====================================================\n";
    std::cout << "Passed: " << passed_count << "/" << test_count << std::endl;
    std::cout << "====================================================\n";

    return (passed_count == test_count) ? 0 : 1;
}
