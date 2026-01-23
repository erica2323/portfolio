# CCCL MACA JIT - 快速开始

## 5 分钟快速集成

### Step 1: 添加文件 (5个新文件)

```bash
# C++ 层
parallel/include/cccl/c/jit_compiler.h      ← JIT 编译器接口
parallel/src/jit_compiler.cu                ← JIT 编译器实现

# 替换现有文件
parallel/src/reduce_official.cu             ← 添加 JIT 分发逻辑

# Python 层（合并到现有文件）
_bindings_maca_jit_extension.pyx            → 合并到 _bindings_maca.pyx
_reduce_simplified_with_jit.py              → 合并到 _reduce_simplified.py
```

### Step 2: 修改构建脚本

```bash
# build.sh - 添加 jit_compiler.cu
SOURCES="parallel/src/reduce_official.cu parallel/src/jit_compiler.cu"

# 添加 crypto 库（用于 SHA256）
LDFLAGS="-lmcruntime -lcrypto"
```

### Step 3: 确认 MCRTC API

⚠️ **最重要的一步！**

检查你的 MACA SDK 中的 RTC API：

```bash
# 查找 RTC 头文件
find $MACA_PATH/include -name "*rtc*.h"

# 查找 RTC 库
ls $MACA_PATH/lib64/ | grep rtc
```

然后修改 `jit_compiler.cu` 中的：

```cpp
// 行 20-30 左右
#ifdef __MACA__
  #include <mcrtc/mcrtc.h>  // ← 调整为实际路径
  // ...
#endif
```

### Step 4: 编译

```bash
# C++ 库
./build_with_jit.sh

# Python 扩展
python setup_maca.py build_ext --inplace
```

### Step 5: 测试

```python
# test_basic.py
from cuda_cccl.cuda.cccl.parallel.experimental import reduce_with_custom_op

# 自定义操作符：乘法
multiply_op = "[](const auto& a, const auto& b) { return a * b; }"

reduce_with_custom_op(
    d_in, d_out,
    op_code=multiply_op,
    op_name="multiply",
    init=1
)
```

---

## 核心代码片段

### 在 Python 中使用 JIT

```python
from cuda_cccl.cuda.cccl.parallel.experimental import reduce_with_custom_op

# 定义自定义操作符 (C++ lambda)
custom_op = """
[](const float& a, const float& b) {
    return a + (b * b);  // 平方和
}
"""

# 调用（第一次会 JIT 编译）
reduce_with_custom_op(
    d_in=my_gpu_array,
    d_out=result_gpu,
    op_code=custom_op,
    op_name="squared_sum",
    init=0.0
)
```

### JIT 编译流程（自动）

```
用户调用 reduce_with_custom_op()
    ↓
Python → Cython → C++
    ↓
[缓存查找: SHA256(op_code + type + arch)]
    ├─ 命中 → 使用缓存内核 ✅
    └─ 未命中 ↓
[代码生成: 包装用户操作符]
    ↓
[MCRTC 编译: 源码 → PTX/LLVM-IR]
    ↓
[模块加载: PTX → GPU 内核]
    ↓
[缓存存储: 下次调用更快]
    ↓
[执行: 两阶段 reduce]
```

---

## 常见问题

### Q1: 编译失败 - 找不到 mcrtc.h

**原因**: MACA SDK 可能使用不同的 RTC API 名称

**解决**:
1. 查找实际的头文件: `find $MACA_PATH -name "*rtc*.h"`
2. 查看 MACA 示例代码
3. 可能需要使用替代 API（如 `mcModule` 直接加载）

### Q2: 运行时错误 - mcErrorNotSupported

**原因**: 当前 `dispatch_reduce_jit()` 是占位实现

**解决**:
- Phase B: JIT 编译工作，但内核启动是简化版
- 完整版需要将 JIT functor 集成到 CUB 的 `DispatchReduce`

### Q3: 性能比内置操作符慢

**原因**: JIT 编译开销（第一次调用）

**优化**:
- 使用 `op_name` 启用缓存
- 预热：在性能关键路径前先调用一次
- 批量处理：多次 reduce 使用同一操作符

---

## MCRTC API 检查清单

根据你的 MACA SDK，需要验证以下函数存在：

```cpp
// ✅ 必需的函数
□ mcrtcCreateProgram()    // 或类似名称
□ mcrtcCompileProgram()   // 或类似名称
□ mcrtcGetPTX() / mcrtcGetLLVMIR()
□ mcModuleLoadData()
□ mcModuleGetFunction()
□ mcLaunchKernel()        // (未来完整集成需要)
```

**如果没有 MCRTC**:
- 选项 1: 使用离线编译 → 生成 `.o` 文件 → 链接时加载
- 选项 2: 使用 `mcModule` API 直接加载预编译的 LLVM-IR
- 选项 3: 等待 MACA SDK 更新

---

## 架构总览

```
┌──────────────────────────────────────────────────┐
│ Python 用户代码                                   │
│ reduce_with_custom_op(d_in, d_out, op_code=...) │
└─────────────────┬────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────┐
│ _reduce_simplified.py                            │
│ - 参数验证                                        │
│ - 类型转换 (NumPy → CCCL)                        │
│ - 创建 init value 指针                            │
└─────────────────┬────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────┐
│ _bindings_maca.pyx (Cython)                      │
│ - reduce_build_with_custom_op()                  │
│ - 传递 op.code 到 C 层                            │
└─────────────────┬────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────┐
│ reduce_official.cu                               │
│ - cccl_device_reduce_build()                     │
│   ├─ 检测到 custom op → 调用 JIT 编译器           │
│   └─ 内置 op → 直接使用 mcCub                     │
└─────────────────┬────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────┐
│ jit_compiler.cu                                  │
│ [1] KernelCache::get() - 查缓存                   │
│ [2] ReduceKernelGenerator - 生成源码              │
│ [3] MCRTCCompiler::compile() - 调用 MCRTC        │
│ [4] MCRTCCompiler::load_module() - 加载内核      │
│ [5] KernelCache::put() - 存缓存                   │
└─────────────────┬────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────┐
│ MCRTC (MACA Runtime Compiler)                    │
│ - 源码 → LLVM-IR/PTX                              │
│ - 优化                                            │
│ - 代码生成                                        │
└──────────────────────────────────────────────────┘
```

---

## 下一步

### 立即可用
- ✅ 内置操作符（SUM, MIN, MAX）- 已测试
- ✅ JIT 编译框架 - 已实现
- ✅ 内核缓存 - 已实现

### 需要完成
- ⚠️ 验证 MCRTC API（根据你的 MACA SDK）
- ⚠️ 集成 JIT 内核到 CUB DispatchReduce
- ⚠️ 完整的错误处理

### 未来增强
- 🔮 Scan (prefix sum) JIT 支持
- 🔮 Sort JIT 支持
- 🔮 LTOIR 输入支持
- 🔮 性能剖析工具

---

## 获取帮助

1. **查看详细文档**: `JIT_INTEGRATION_GUIDE.md`
2. **运行测试**: `python test_jit_reduce.py`
3. **调试 JIT 编译**: 在 `jit_compiler.cu` 中启用详细日志
4. **检查 MACA 文档**: 查找 RTC/JIT 相关章节

---

**快速开始完成！接下来验证 MCRTC API 并运行测试。**
