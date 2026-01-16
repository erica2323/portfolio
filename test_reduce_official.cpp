#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>
#include "cccl/c/types_official.h"
#include "cccl/c/reduce_official.h"

// Helper function to print array
void print_array(const char* label, const int32_t* arr, size_t n, size_t max_print = 16) {
    std::cout << label << ": [";
    size_t print_count = (n < max_print) ? n : max_print;
    for (size_t i = 0; i < print_count; i++) {
        std::cout << arr[i];
        if (i < print_count - 1) std::cout << ", ";
    }
    if (n > max_print) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}

// Helper function to check results
bool check_result(int32_t output, int32_t expected, const char* test_name) {
    bool passed = (output == expected);

    if (passed) {
        std::cout << "✅ " << test_name << " PASSED" << std::endl;
        std::cout << "   Result: " << output << std::endl;
    } else {
        std::cout << "❌ " << test_name << " FAILED" << std::endl;
        std::cout << "   Expected: " << expected << std::endl;
        std::cout << "   Got:      " << output << std::endl;
    }

    return passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MACA CCCL Reduce Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    const size_t n = 16;
    int test_count = 0;
    int passed_count = 0;

    // ========================================
    // Test 1: Built-in SUM operation
    // ========================================
    {
        test_count++;
        std::cout << "\nTest 1: Built-in SUM operation" << std::endl;
        std::cout << "========================================" << std::endl;

        // Prepare input: [0, 1, 2, 3, ..., 15]
        std::vector<int32_t> h_in(n);
        for (size_t i = 0; i < n; i++) {
            h_in[i] = static_cast<int32_t>(i);
        }
        print_array("Input", h_in.data(), n);

        // Allocate device memory
        int32_t *d_in, *d_out;
        mcMalloc((void**)&d_in, n * sizeof(int32_t));
        mcMalloc((void**)&d_out, sizeof(int32_t));

        // Copy input to device
        mcMemcpy(d_in, h_in.data(), n * sizeof(int32_t), mcMemcpyHostToDevice);

        // Create type info
        cccl_type_info in_type;
        in_type.type = CCCL_INT32;
        in_type.size = sizeof(int32_t);
        in_type.alignment = alignof(int32_t);

        cccl_type_info out_type;
        out_type.type = CCCL_INT32;
        out_type.size = sizeof(int32_t);
        out_type.alignment = alignof(int32_t);

        // Create operation
        cccl_op_t op;
        op.type = CCCL_PLUS;
        op.name = "sum";
        op.code = nullptr;
        op.code_size = 0;
        op.code_type = CCCL_OP_LTOIR;
        op.size = 0;
        op.alignment = 0;
        op.state = nullptr;

        std::cout << "Operation: SUM(array)" << std::endl;

        // Build reduce
        cccl_reduce_build_result_t build;
        mcError_t result = cccl_reduce_build(&build, in_type, out_type, op);
        if (result != mcSuccess) {
            std::cerr << "cccl_reduce_build failed: " << result << std::endl;
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Execute reduce
        result = cccl_reduce_dispatch(build, d_in, d_out, n, nullptr, in_type, op, nullptr);
        if (result != mcSuccess) {
            std::cerr << "cccl_reduce_dispatch failed: " << result << std::endl;
            cccl_reduce_cleanup(build);
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Copy result back
        int32_t h_out;
        mcMemcpy(&h_out, d_out, sizeof(int32_t), mcMemcpyDeviceToHost);

        // Verify result: sum of 0..15 = 120
        int32_t expected = 0;
        for (size_t i = 0; i < n; i++) {
            expected += h_in[i];
        }

        if (check_result(h_out, expected, "SUM")) {
            passed_count++;
        }

        // Cleanup
        cccl_reduce_cleanup(build);
        mcFree(d_in);
        mcFree(d_out);
    }

    // ========================================
    // Test 2: Built-in MAX operation
    // ========================================
    {
        test_count++;
        std::cout << "\nTest 2: Built-in MAX operation" << std::endl;
        std::cout << "========================================" << std::endl;

        // Prepare input: [15, 3, 7, 1, 9, 12, 4, 8, 2, 14, 6, 11, 5, 10, 13, 0]
        std::vector<int32_t> h_in = {15, 3, 7, 1, 9, 12, 4, 8, 2, 14, 6, 11, 5, 10, 13, 0};
        print_array("Input", h_in.data(), n);

        // Allocate device memory
        int32_t *d_in, *d_out;
        mcMalloc((void**)&d_in, n * sizeof(int32_t));
        mcMalloc((void**)&d_out, sizeof(int32_t));

        // Copy input to device
        mcMemcpy(d_in, h_in.data(), n * sizeof(int32_t), mcMemcpyHostToDevice);

        // Create type info
        cccl_type_info in_type;
        in_type.type = CCCL_INT32;
        in_type.size = sizeof(int32_t);
        in_type.alignment = alignof(int32_t);

        cccl_type_info out_type;
        out_type.type = CCCL_INT32;
        out_type.size = sizeof(int32_t);
        out_type.alignment = alignof(int32_t);

        // Create operation
        cccl_op_t op;
        op.type = CCCL_MAXIMUM;
        op.name = "max";
        op.code = nullptr;
        op.code_size = 0;
        op.code_type = CCCL_OP_LTOIR;
        op.size = 0;
        op.alignment = 0;
        op.state = nullptr;

        std::cout << "Operation: MAX(array)" << std::endl;

        // Build reduce
        cccl_reduce_build_result_t build;
        mcError_t result = cccl_reduce_build(&build, in_type, out_type, op);
        if (result != mcSuccess) {
            std::cerr << "cccl_reduce_build failed: " << result << std::endl;
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Execute reduce
        result = cccl_reduce_dispatch(build, d_in, d_out, n, nullptr, in_type, op, nullptr);
        if (result != mcSuccess) {
            std::cerr << "cccl_reduce_dispatch failed: " << result << std::endl;
            cccl_reduce_cleanup(build);
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Copy result back
        int32_t h_out;
        mcMemcpy(&h_out, d_out, sizeof(int32_t), mcMemcpyDeviceToHost);

        // Verify result: max should be 15
        int32_t expected = 15;

        if (check_result(h_out, expected, "MAX")) {
            passed_count++;
        }

        // Cleanup
        cccl_reduce_cleanup(build);
        mcFree(d_in);
        mcFree(d_out);
    }

    // ========================================
    // Test 3: Built-in MIN operation
    // ========================================
    {
        test_count++;
        std::cout << "\nTest 3: Built-in MIN operation" << std::endl;
        std::cout << "========================================" << std::endl;

        // Prepare input: [15, 3, 7, 1, 9, 12, 4, 8, 2, 14, 6, 11, 5, 10, 13, 0]
        std::vector<int32_t> h_in = {15, 3, 7, 1, 9, 12, 4, 8, 2, 14, 6, 11, 5, 10, 13, 0};
        print_array("Input", h_in.data(), n);

        // Allocate device memory
        int32_t *d_in, *d_out;
        mcMalloc((void**)&d_in, n * sizeof(int32_t));
        mcMalloc((void**)&d_out, sizeof(int32_t));

        // Copy input to device
        mcMemcpy(d_in, h_in.data(), n * sizeof(int32_t), mcMemcpyHostToDevice);

        // Create type info
        cccl_type_info in_type;
        in_type.type = CCCL_INT32;
        in_type.size = sizeof(int32_t);
        in_type.alignment = alignof(int32_t);

        cccl_type_info out_type;
        out_type.type = CCCL_INT32;
        out_type.size = sizeof(int32_t);
        out_type.alignment = alignof(int32_t);

        // Create operation
        cccl_op_t op;
        op.type = CCCL_MINIMUM;
        op.name = "min";
        op.code = nullptr;
        op.code_size = 0;
        op.code_type = CCCL_OP_LTOIR;
        op.size = 0;
        op.alignment = 0;
        op.state = nullptr;

        std::cout << "Operation: MIN(array)" << std::endl;

        // Build reduce
        cccl_reduce_build_result_t build;
        mcError_t result = cccl_reduce_build(&build, in_type, out_type, op);
        if (result != mcSuccess) {
            std::cerr << "cccl_reduce_build failed: " << result << std::endl;
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Execute reduce
        result = cccl_reduce_dispatch(build, d_in, d_out, n, nullptr, in_type, op, nullptr);
        if (result != mcSuccess) {
            std::cerr << "cccl_reduce_dispatch failed: " << result << std::endl;
            cccl_reduce_cleanup(build);
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Copy result back
        int32_t h_out;
        mcMemcpy(&h_out, d_out, sizeof(int32_t), mcMemcpyDeviceToHost);

        // Verify result: min should be 0
        int32_t expected = 0;

        if (check_result(h_out, expected, "MIN")) {
            passed_count++;
        }

        // Cleanup
        cccl_reduce_cleanup(build);
        mcFree(d_in);
        mcFree(d_out);
    }

    // ========================================
    // Summary
    // ========================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << passed_count << "/" << test_count << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed_count == test_count) ? 0 : 1;
}
