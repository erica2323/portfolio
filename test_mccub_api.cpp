// test_mccub_api.cpp
// 测试 mcCub 的 DeviceTransform API 是否可用

#include <mc_runtime.h>
#include <cub/device/device_transform.cuh>
#include <iostream>

int main() {
    std::cout << "Testing mcCub API availability..." << std::endl;

    const int N = 10;

    // 分配内存
    int *d_in, *d_out;
    mcMalloc((void**)&d_in, N * sizeof(int));
    mcMalloc((void**)&d_out, N * sizeof(int));

    // 定义操作符
    auto plus_op = [] __device__ (int x) { return x + 5; };

    std::cout << "Calling cub::DeviceTransform::Transform..." << std::endl;

    // 测试：能否调用 cub::DeviceTransform?
    cub::DeviceTransform::Transform(d_in, d_out, N, plus_op);

    std::cout << "✅ cub::DeviceTransform::Transform works!" << std::endl;

    // 清理
    mcFree(d_in);
    mcFree(d_out);

    return 0;
}
