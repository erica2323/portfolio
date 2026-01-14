#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>
#include "cccl/c/types_official.h"
#include "cccl/c/transform_official.h"

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
bool check_results(const int32_t* output, const int32_t* expected, size_t n, const char* test_name) {
    bool passed = true;
    for (size_t i = 0; i < n; i++) {
        if (output[i] != expected[i]) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "✅ " << test_name << " PASSED" << std::endl;
    } else {
        std::cout << "❌ " << test_name << " FAILED" << std::endl;
        std::cout << "Expected: [";
        for (size_t i = 0; i < n && i < 16; i++) {
            std::cout << expected[i];
            if (i < n - 1 && i < 15) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    return passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MACA CCCL Transform Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    const size_t n = 16;
    int test_count = 0;
    int passed_count = 0;

    // ========================================
    // Test 1: Built-in PLUS operation
    // ========================================
    {
        test_count++;
        std::cout << "\nTest 1: Built-in PLUS operation (add 100)" << std::endl;
        std::cout << "========================================" << std::endl;

        // Prepare input
        std::vector<int32_t> h_in(n);
        for (size_t i = 0; i < n; i++) {
            h_in[i] = static_cast<int32_t>(i);
        }
        print_array("Input", h_in.data(), n);

        // Allocate device memory
        int32_t *d_in, *d_out;
        mcMalloc((void**)&d_in, n * sizeof(int32_t));
        mcMalloc((void**)&d_out, n * sizeof(int32_t));

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

        // Create operation with addend = 100
        int addend = 100;
        cccl_op_t op;
        op.type = CCCL_PLUS;
        op.name = "add_100";
        op.code = nullptr;
        op.code_size = 0;
        op.code_type = CCCL_OP_LTOIR;
        op.size = 0;
        op.alignment = 0;
        op.state = &addend;

        std::cout << "Operation: x + 100" << std::endl;

        // Build transform
        cccl_transform_build_result_t build;
        mcError_t result = cccl_transform_build(&build, in_type, out_type, op);
        if (result != mcSuccess) {
            std::cerr << "cccl_transform_build failed: " << result << std::endl;
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Execute transform
        result = cccl_transform(build, d_in, d_out, n, nullptr);
        if (result != mcSuccess) {
            std::cerr << "cccl_transform failed: " << result << std::endl;
            cccl_transform_cleanup(build);
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Copy result back
        std::vector<int32_t> h_out(n);
        mcMemcpy(h_out.data(), d_out, n * sizeof(int32_t), mcMemcpyDeviceToHost);

        print_array("Output", h_out.data(), n);

        // Verify results
        std::vector<int32_t> expected(n);
        for (size_t i = 0; i < n; i++) {
            expected[i] = h_in[i] + 100;
        }
        print_array("Expected", expected.data(), n);

        if (check_results(h_out.data(), expected.data(), n, "PLUS")) {
            passed_count++;
        }

        // Cleanup
        cccl_transform_cleanup(build);
        mcFree(d_in);
        mcFree(d_out);
    }

    // ========================================
    // Test 2: User-provided C++ source
    // ========================================
    {
        test_count++;
        std::cout << "\nTest 2: User C++ source (x^3 + 1)" << std::endl;
        std::cout << "========================================" << std::endl;

        // Prepare input
        std::vector<int32_t> h_in(n);
        for (size_t i = 0; i < n; i++) {
            h_in[i] = static_cast<int32_t>(i);
        }
        print_array("Input", h_in.data(), n);

        // Allocate device memory
        int32_t *d_in, *d_out;
        mcMalloc((void**)&d_in, n * sizeof(int32_t));
        mcMalloc((void**)&d_out, n * sizeof(int32_t));

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

        // User-defined operation: x^3 + 1
        const char* user_code = R"(
__device__ int32_t user_op(int32_t x) {
    return x * x * x + 1;
}
)";

        std::cout << "Operation: x * x * x + 1" << std::endl;

        cccl_op_t op;
        op.type = CCCL_STATELESS;
        op.name = "cube_plus_one";
        op.code = user_code;
        op.code_size = strlen(user_code);
        op.code_type = CCCL_OP_CPP_SOURCE;
        op.size = 0;
        op.alignment = 0;
        op.state = nullptr;

        // Build transform
        cccl_transform_build_result_t build;
        mcError_t result = cccl_transform_build(&build, in_type, out_type, op);
        if (result != mcSuccess) {
            std::cerr << "cccl_transform_build failed: " << result << std::endl;
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Execute transform
        result = cccl_transform(build, d_in, d_out, n, nullptr);
        if (result != mcSuccess) {
            std::cerr << "cccl_transform failed: " << result << std::endl;
            cccl_transform_cleanup(build);
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Copy result back
        std::vector<int32_t> h_out(n);
        mcMemcpy(h_out.data(), d_out, n * sizeof(int32_t), mcMemcpyDeviceToHost);

        print_array("Output", h_out.data(), n);

        // Verify results
        std::vector<int32_t> expected(n);
        for (size_t i = 0; i < n; i++) {
            expected[i] = h_in[i] * h_in[i] * h_in[i] + 1;
        }
        print_array("Expected", expected.data(), n);

        if (check_results(h_out.data(), expected.data(), n, "User source (x^3 + 1)")) {
            passed_count++;
        }

        // Cleanup
        cccl_transform_cleanup(build);
        mcFree(d_in);
        mcFree(d_out);
    }

    // ========================================
    // Test 3: Built-in MULTIPLIES operation
    // ========================================
    {
        test_count++;
        std::cout << "\nTest 3: Built-in MULTIPLIES operation (multiply by 10)" << std::endl;
        std::cout << "========================================" << std::endl;

        // Prepare input
        std::vector<int32_t> h_in(n);
        for (size_t i = 0; i < n; i++) {
            h_in[i] = static_cast<int32_t>(i);
        }
        print_array("Input", h_in.data(), n);

        // Allocate device memory
        int32_t *d_in, *d_out;
        mcMalloc((void**)&d_in, n * sizeof(int32_t));
        mcMalloc((void**)&d_out, n * sizeof(int32_t));

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

        // Create operation with multiplier = 10
        int multiplier = 10;
        cccl_op_t op;
        op.type = CCCL_MULTIPLIES;
        op.name = "multiply_10";
        op.code = nullptr;
        op.code_size = 0;
        op.code_type = CCCL_OP_LTOIR;
        op.size = 0;
        op.alignment = 0;
        op.state = &multiplier;

        std::cout << "Operation: x * 10" << std::endl;

        // Build transform
        cccl_transform_build_result_t build;
        mcError_t result = cccl_transform_build(&build, in_type, out_type, op);
        if (result != mcSuccess) {
            std::cerr << "cccl_transform_build failed: " << result << std::endl;
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Execute transform
        result = cccl_transform(build, d_in, d_out, n, nullptr);
        if (result != mcSuccess) {
            std::cerr << "cccl_transform failed: " << result << std::endl;
            cccl_transform_cleanup(build);
            mcFree(d_in);
            mcFree(d_out);
            continue;
        }

        // Copy result back
        std::vector<int32_t> h_out(n);
        mcMemcpy(h_out.data(), d_out, n * sizeof(int32_t), mcMemcpyDeviceToHost);

        print_array("Output", h_out.data(), n);

        // Verify results
        std::vector<int32_t> expected(n);
        for (size_t i = 0; i < n; i++) {
            expected[i] = h_in[i] * 10;
        }
        print_array("Expected", expected.data(), n);

        if (check_results(h_out.data(), expected.data(), n, "MULTIPLIES")) {
            passed_count++;
        }

        // Cleanup
        cccl_transform_cleanup(build);
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
