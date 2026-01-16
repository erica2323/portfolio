// test_mccub_reduce_api.cpp
// 测试 mcCub 的 DeviceReduce API 是否可用

#include <mc_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <iostream>

int main() {
    std::cout << "Testing mcCub DeviceReduce API availability..." << std::endl;

    const int N = 10;

    // 分配内存
    int *d_in, *d_out;
    mcMalloc((void**)&d_in, N * sizeof(int));
    mcMalloc((void**)&d_out, sizeof(int));

    // 初始化输入数据
    int h_in[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = i + 1;  // 1, 2, 3, ..., 10
    }
    mcMemcpy(d_in, h_in, N * sizeof(int), mcMemcpyHostToDevice);

    std::cout << "Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]" << std::endl;
    std::cout << "Expected sum: 55" << std::endl;

    // 获取临时存储大小
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    std::cout << "Getting temp storage size..." << std::endl;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);

    // 分配临时存储
    mcMalloc(&d_temp_storage, temp_storage_bytes);
    std::cout << "Temp storage size: " << temp_storage_bytes << " bytes" << std::endl;

    // 执行求和
    std::cout << "Calling cub::DeviceReduce::Sum..." << std::endl;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);

    // 获取结果
    int h_out;
    mcMemcpy(&h_out, d_out, sizeof(int), mcMemcpyDeviceToHost);

    std::cout << "Result: " << h_out << std::endl;

    if (h_out == 55) {
        std::cout << "✅ cub::DeviceReduce::Sum works correctly!" << std::endl;
    } else {
        std::cout << "❌ cub::DeviceReduce::Sum failed! Expected 55, got " << h_out << std::endl;
    }

    // 清理
    mcFree(d_temp_storage);
    mcFree(d_in);
    mcFree(d_out);

    return (h_out == 55) ? 0 : 1;
}
