# CCCL MACA JIT Integration Guide

## 概述

本指南说明如何为 CCCL MACA 适配添加 JIT (Just-In-Time) 编译支持，使其能够运行时编译自定义操作符。

## 架构设计

### 三层架构

```
┌─────────────────────────────────────────────────────┐
│  Python Layer (_reduce_simplified.py)              │
│  - reduce_with_custom_op()                          │
│  - 用户友好的 API                                    │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│  Cython Bindings (_bindings_maca.pyx)              │
│  - reduce_build_with_custom_op()                    │
│  - 传递操作符源代码到 C 层                            │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│  C/C++ Layer (reduce_official.cu + jit_compiler.cu)│
│  - JIT 编译管道 (MCRTC)                              │
│  - 内核缓存 (SHA256)                                 │
│  - mcCub 集成                                        │
└─────────────────────────────────────────────────────┘
```

### JIT 编译流程

```
用户自定义操作符 (C++ lambda)
    ↓
[1. 缓存查找]
    ├─ HIT → 使用已缓存内核
    └─ MISS ↓
[2. 代码生成]
    - 包装用户代码
    - 生成完整内核源码
    - 添加 CUB 头文件
    ↓
[3. MCRTC 编译]
    - mcrtcCreateProgram()
    - mcrtcCompileProgram()
    - mcrtcGetPTX()
    ↓
[4. 模块加载]
    - mcModuleLoadData()
    - mcModuleGetFunction()
    ↓
[5. 缓存存储]
    - SHA256 哈希作为 key
    - 存入全局缓存
    ↓
[6. 执行]
    - 与 mcCub DispatchReduce 集成
    - 两阶段执行（查询临时内存 + 执行）
```

## 文件清单

### 新增文件

| 文件 | 作用 |
|------|------|
| `parallel/include/cccl/c/jit_compiler.h` | JIT 编译器接口定义 |
| `parallel/src/jit_compiler.cu` | JIT 编译器实现（MCRTC 封装） |
| `_bindings_maca_jit_extension.pyx` | Cython 绑定扩展 |
| `_reduce_simplified_with_jit.py` | Python API 扩展 |
| `test_jit_reduce.py` | JIT 功能测试 |
| `build_with_jit.sh` | 支持 JIT 的构建脚本 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `parallel/src/reduce_official.cu` | 添加 JIT 路径分发逻辑 |
| `parallel/include/cccl/c/reduce_official.h` | (可选) 扩展 build_result 结构体 |

## 安装步骤

### 1. 复制新文件到项目

```bash
# 复制头文件
cp jit_compiler.h /path/to/mcCub/c/parallel/include/cccl/c/

# 复制实现文件
cp jit_compiler.cu /path/to/mcCub/c/parallel/src/

# 替换 reduce 实现
cp reduce_official_with_jit.cu /path/to/mcCub/c/parallel/src/reduce_official.cu
```

### 2. 构建 C++ 库

```bash
chmod +x build_with_jit.sh
./build_with_jit.sh
```

**注意事项：**
- 需要链接 `-lcrypto`（用于 SHA256 哈希）
- 需要链接 MCRTC 运行时库（可能需要根据 MACA SDK 调整）
- 确认 MCRTC 头文件路径（可能需要调整 `#include` 语句）

### 3. 扩展 Cython 绑定

将 `_bindings_maca_jit_extension.pyx` 的内容合并到现有的 `_bindings_maca.pyx`：

```python
# _bindings_maca.pyx

# ... (existing code)

# ============ 添加以下部分 ============

def reduce_build_with_custom_op(...):
    # ... (见 _bindings_maca_jit_extension.pyx)

def clear_jit_cache():
    # ... (见 _bindings_maca_jit_extension.pyx)
```

### 4. 更新 Python API

将 `_reduce_simplified_with_jit.py` 的内容合并到 `_reduce_simplified.py`：

```python
# 添加新函数
def reduce_with_custom_op(...):
    # ... (见 _reduce_simplified_with_jit.py)
```

### 5. 重新编译 Python 扩展

```bash
python setup_maca.py build_ext --inplace
```

### 6. 运行测试

```bash
# 测试内置操作符（应该保持正常工作）
python test_reduce_simplified.py

# 测试 JIT 自定义操作符
python test_jit_reduce.py
```

## 使用示例

### 内置操作符（无 JIT）

```python
from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind

# 简单求和
reduce(d_in, d_out, op=OpKind.SUM, init=0)

# 最小值
reduce(d_in, d_out, op=OpKind.MIN, init=float('inf'))
```

### 自定义操作符（JIT）

#### 示例 1：乘法规约（product）

```python
from cuda_cccl.cuda.cccl.parallel.experimental import reduce_with_custom_op

# 定义自定义操作符
multiply_op = "[](const auto& a, const auto& b) { return a * b; }"

# 使用 JIT 编译的操作符
reduce_with_custom_op(
    d_in,
    d_out,
    op_code=multiply_op,
    op_name="multiply",
    init=1
)
# 结果: 所有元素相乘
```

#### 示例 2：平方和

```python
squared_sum_op = """
[](const auto& a, const auto& b) {
    return a + (b * b);
}
"""

reduce_with_custom_op(
    d_in,
    d_out,
    op_code=squared_sum_op,
    op_name="squared_sum",
    init=0.0
)
# 结果: Σ(x²)
```

#### 示例 3：加权和

```python
# 假设权重编码在数据的高位
weighted_sum_op = """
[](const auto& a, const auto& b) {
    float value = static_cast<float>(b & 0xFFFF);
    float weight = static_cast<float>(b >> 16) / 100.0f;
    return a + (value * weight);
}
"""

reduce_with_custom_op(
    d_in,
    d_out,
    op_code=weighted_sum_op,
    op_name="weighted_sum",
    init=0.0
)
```

## 性能优化

### 内核缓存

JIT 编译结果会自动缓存：

```python
# 第一次调用：触发 JIT 编译（慢）
reduce_with_custom_op(d_in, d_out, op_code=my_op, op_name="my_op")

# 后续调用：使用缓存内核（快）
reduce_with_custom_op(d_in2, d_out2, op_code=my_op, op_name="my_op")
```

缓存键基于：
- 操作符源代码 (SHA256)
- 数据类型
- 计算能力版本

清空缓存：

```python
from cuda_cccl.cuda.cccl.parallel.experimental._bindings_maca import clear_jit_cache

clear_jit_cache()
```

### 编译选项调优

```python
reduce_with_custom_op(
    d_in, d_out,
    op_code=my_op,
    cc_major=9,           # 匹配你的 GPU 架构
    cc_minor=0,
    cub_path="/path/to/cub",
    thrust_path="/path/to/thrust",
    libcudacxx_path="/path/to/libcudacxx"
)
```

## 调试技巧

### 1. 查看编译日志

修改 `jit_compiler.cu`，增加详细输出：

```cpp
std::cout << "[MCRTC] Source code:\n" << source_code << std::endl;
```

### 2. 保存生成的 PTX

```cpp
// In cccl_jit_compile_reduce_op()
std::ofstream ptx_file("debug_kernel.ptx");
ptx_file << ptx_out;
ptx_file.close();
```

### 3. 验证操作符语法

在 CPU 上先测试操作符：

```cpp
// test.cpp
#include <iostream>

auto my_op = [](const auto& a, const auto& b) {
    return a + (b * b);
};

int main() {
    int result = my_op(0, 5);
    std::cout << "Result: " << result << std::endl;  // Should be 25
}
```

## 已知限制

### Phase B（当前实现）

1. **JIT 内核尚未与 CUB DispatchReduce 完全集成**
   - 当前 `dispatch_reduce_jit()` 返回 `mcErrorNotSupported`
   - 需要将 JIT 编译的 functor 传递给 CUB 模板

2. **MCRTC API 假设**
   - 代码中使用了假设的 MCRTC 函数签名
   - 需要根据实际 MACA SDK 文档调整

3. **支持的操作符类型**
   - 当前仅支持 C++ lambda 和 functor
   - 不支持有状态操作符（CCCL_STATEFUL）

### 未来改进 (Phase C)

1. **完整的 CUB 集成**
   - 使用 JIT 编译的操作符实例化 CUB 模板
   - 支持高级 CUB 优化（block-level reduction）

2. **更多算法**
   - Scan (prefix sum)
   - Sort
   - Segmented reduce

3. **LTOIR 支持**
   - 接受预编译的 LLVM bitcode
   - 跨语言互操作（Python → Numba LTOIR → CCCL）

## MCRTC API 适配检查清单

根据你的 MACA SDK，可能需要调整以下部分：

### ✅ 需要确认的 API

```cpp
// jit_compiler.cu 中需要验证的函数：

□ mcrtcCreateProgram() - 创建编译单元
  → 检查参数顺序和类型
  → 确认是否支持 headers 参数

□ mcrtcCompileProgram() - 编译源代码
  → 确认编译选项格式 (例如: "-arch=compute_90")
  → 检查是否使用 -xmaca 标志

□ mcrtcGetPTXSize() / mcrtcGetPTX() - 获取编译结果
  → MACA 可能返回 LLVM-IR 而非 PTX
  → 可能需要使用不同的函数名

□ mcModuleLoadData() - 加载编译后的模块
  → 确认输入格式（PTX 字符串 vs. 二进制）

□ mcModuleGetFunction() - 获取内核函数指针
  → 确认返回类型（mcFunction_t）

□ mcLaunchKernel() - 启动内核
  → 确认参数传递方式
```

### 🔍 查找实际 API 方法

```bash
# 方法 1: 查找头文件
find $MACA_PATH/include -name "*rtc*.h" -o -name "*jit*.h"

# 方法 2: 查找库符号
nm -D $MACA_PATH/lib64/libmcrtc.so 2>/dev/null | grep -i compile

# 方法 3: 查看示例代码
find $MACA_PATH -name "*.cu" -exec grep -l "rtc\|jit" {} \;
```

### 📝 常见替换模式

| CUDA/NVRTC | 可能的 MACA 等价物 |
|------------|-------------------|
| `nvrtcCreateProgram` | `mcrtcCreateProgram` |
| `nvrtcCompileProgram` | `mcrtcCompileProgram` |
| `nvrtcGetPTX` | `mcrtcGetLLVMIR` 或 `mcrtcGetPTX` |
| `cuModuleLoadData` | `mcModuleLoadData` |
| `cuModuleGetFunction` | `mcModuleGetFunction` |
| `cuLaunchKernel` | `mcLaunchKernel` |

## 故障排除

### 编译错误：找不到 mcrtc.h

```bash
# 解决方案 1: 查找正确的头文件
find $MACA_PATH -name "*rtc*.h"

# 解决方案 2: 如果没有 MCRTC，可能需要使用替代方法
# 例如: mcModule API 直接加载预编译的 LLVM-IR
```

### 链接错误：undefined reference to mcrtcCreateProgram

```bash
# 添加正确的库到链接器
-lmcrtc  # 或者 -lmaca_rtc, -lmaca_jit 等

# 查找库名称
ls $MACA_PATH/lib64/ | grep -i rtc
```

### 运行时错误：mcErrorNotSupported

当前 `dispatch_reduce_jit()` 是占位实现。要完全启用：

1. 创建包装 functor 封装 JIT 内核
2. 传递给 CUB 的 `DispatchReduce` 模板
3. 或者直接使用自定义内核启动逻辑

## 总结

本 JIT 集成方案提供了：

✅ **已完成**
- 完整的 JIT 编译架构设计
- 内核缓存机制（SHA256 哈希）
- Python → Cython → C++ 的完整调用链
- 用户友好的 API 设计
- 内置操作符兼容性（无回退）

⚠️ **需要适配**
- MCRTC API 函数调用（根据实际 MACA SDK）
- JIT 内核与 CUB 的完整集成
- 完整的错误处理和日志

🚀 **未来扩展**
- 更多算法（scan, sort, etc.）
- LTOIR 支持
- 有状态操作符
- 性能剖析工具

## 参考资料

- [CUDA CCCL C API 文档](https://nvidia.github.io/cccl/)
- [NVRTC 用户指南](https://docs.nvidia.com/cuda/nvrtc/)
- [CUB 库文档](https://nvlabs.github.io/cub/)
- MACA SDK 文档（请参考你的供应商文档）

---

**作者**: Claude AI
**版本**: Phase B - JIT Framework
**日期**: 2026-01-23
