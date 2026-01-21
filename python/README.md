# CCCL Python Bindings for MACA (Simplified)

这是 NVIDIA CCCL 的 MACA 移植版本的 Python 绑定，**简化版本（无 Numba 支持）**。

## 特性

✅ **支持的功能**：
- 内置归约算子：`OpKind.SUM`, `OpKind.MIN`, `OpKind.MAX`, `OpKind.PROD`
- 与 NumPy 类型兼容
- 支持 `__cuda_array_interface__` 协议（兼容 CuPy 等）
- 两阶段执行模型（build + execute）
- 异步流支持

❌ **不支持的功能**（需要 Numba）：
- 自定义 Python lambda：`lambda a, b: a * b`
- 自定义算子函数
- TransformIterator 等高级迭代器
- JIT 编译 Python 代码到 GPU

## 架构

```
python/
├── cuda_cccl/
│   └── cuda/
│       └── cccl/
│           └── parallel/
│               └── experimental/
│                   ├── __init__.py
│                   ├── _cccl_interop.py        # OpKind enum, 类型转换
│                   ├── _bindings_maca.pyx      # Cython wrapper → C API
│                   └── algorithms/
│                       ├── __init__.py
│                       └── _reduce.py          # 高层 reduce() API
├── setup.py                                    # 构建脚本
└── test_reduce_simplified.py                   # 测试示例
```

## 依赖关系

```
Python API (algorithms/_reduce.py)
    ↓
Cython Bindings (_bindings_maca.pyx)
    ↓
C API (libcccl_maca.so)
    ↓
mcCub (MACA's CUB library)
```

## 构建步骤

### 1. 环境变量设置

```bash
export MACA_PATH=/usr/local/maca
export CCCL_PATH=/path/to/your/cccl/repo
export LD_LIBRARY_PATH=$MACA_PATH/lib64:$CCCL_PATH/build:$LD_LIBRARY_PATH
```

### 2. 确保 C 层已构建

你应该已经有：
- `libcccl_maca.so` （Phase A 的 C 层）
- 头文件：
  - `parallel/include/cccl/c/reduce_official.h`
  - `parallel/include/cccl/c/types_official.h`

### 3. 修改 setup.py 路径

编辑 `setup.py`，修改这两个路径：

```python
MACA_ROOT = os.environ.get('MACA_PATH', '/usr/local/maca')
CCCL_ROOT = os.environ.get('CCCL_PATH', '/path/to/cccl')  # ← 改成你的 CCCL 路径
```

### 4. 构建 Cython 扩展

```bash
cd python
python setup.py build_ext --inplace
```

这会生成：
```
cuda_cccl/cuda/cccl/parallel/experimental/_bindings_maca.cpython-*.so
```

### 5. 安装（可选）

```bash
pip install -e .
```

或者直接在 `python/` 目录下使用（通过 `PYTHONPATH`）。

## 使用示例

### 基本用法

```python
from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind
import numpy as np

# 假设你有 GPU 数组 (CuPy, PyTorch, 或者自定义 MACA wrapper)
import cupy as cp

# 创建输入数组
d_in = cp.ones(1000, dtype=cp.int32)
d_out = cp.zeros(1, dtype=cp.int32)

# Sum reduction
reduce(d_in, d_out, op=OpKind.SUM, init=10)
print(d_out)  # 输出: [1010]  (1000 个 1 + init 值 10)

# Min reduction
d_in = cp.array([5, 2, 8, 1, 9], dtype=cp.int32)
reduce(d_in, d_out, op=OpKind.MIN, init=0)
print(d_out)  # 输出: [0]  (0 是最小值)

# Max reduction
reduce(d_in, d_out, op=OpKind.MAX, init=100)
print(d_out)  # 输出: [100]  (100 是最大值)
```

### 使用原始指针

```python
from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind

# 如果你有原始 device 指针
d_in_ptr = 0x7f1234567890  # 某个 device 指针
d_out_ptr = 0x7f9876543210

reduce(
    d_in=d_in_ptr,
    d_out=d_out_ptr,
    op=OpKind.SUM,
    init=0,
    num_items=1000  # 必须指定 num_items
)
```

### 完整测试示例

运行提供的测试：

```bash
cd python
python test_reduce_simplified.py
```

## 支持的算子

| OpKind      | 描述         | C API 对应    |
|-------------|--------------|---------------|
| OpKind.SUM  | 求和归约     | CCCL_SUM      |
| OpKind.MIN  | 最小值归约   | CCCL_MIN      |
| OpKind.MAX  | 最大值归约   | CCCL_MAX      |
| OpKind.PROD | 乘积归约     | CCCL_PROD     |

## 与 NVIDIA CCCL 的区别

### NVIDIA CCCL (有 Numba)
```python
# ✅ 支持自定义 lambda
reduce(d_in, d_out, op=lambda a, b: a * b)

# ✅ 支持自定义函数
def custom_op(a, b):
    return a + b * 2

reduce(d_in, d_out, op=custom_op)

# ✅ 支持 TransformIterator
reduce(TransformIterator(d_in, lambda x: x**2), d_out, op=OpKind.SUM)
```

### MACA Port (无 Numba)
```python
# ❌ 不支持自定义 lambda
# reduce(d_in, d_out, op=lambda a, b: a * b)  # 会失败

# ✅ 只支持内置算子
reduce(d_in, d_out, op=OpKind.SUM)   # OK
reduce(d_in, d_out, op=OpKind.MIN)   # OK
reduce(d_in, d_out, op=OpKind.MAX)   # OK
```

## 为什么不支持自定义算子？

NVIDIA CCCL 的 `_cccl_interop.py` 使用 **Numba** 来实现：

```python
# NVIDIA 的代码
from numba import cuda

def to_cccl_op(op: Callable, sig: Signature) -> Op:
    wrapped_op, wrapper_sig = _create_void_ptr_wrapper(op, sig)
    ltoir, _ = cuda.compile(wrapped_op, sig=wrapper_sig, output="ltoir")  # 🔥 JIT 编译
    return Op(name=wrapped_op.__name__, ltoir=ltoir, ...)
```

**问题**：
- Numba 只支持 NVIDIA CUDA
- MACA 没有 Numba 支持
- 没有 JIT 编译器把 Python lambda 转成 GPU 代码

**解决方案**：
1. **当前方案**：只支持内置算子（已实现）
2. **未来方案**：如果 MACA 提供 Python JIT 编译器，可以适配

## 测试覆盖

| NVIDIA test_reduce.py 测试               | MACA Port 支持? |
|------------------------------------------|-----------------|
| `test_reduce_sum()`                      | ✅ 支持         |
| `test_reduce_min()`                      | ✅ 支持         |
| `test_reduce_max()`                      | ✅ 支持         |
| `test_reduce_custom_op(lambda a,b: ...)` | ❌ 不支持       |
| `test_transform_iterator()`              | ❌ 不支持       |

**预计可运行 ~60% 的 NVIDIA CCCL 测试**（所有使用内置算子的测试）。

## 故障排除

### 错误：`ImportError: No module named '_bindings_maca'`

**原因**：Cython 扩展未构建。

**解决**：
```bash
python setup.py build_ext --inplace
```

### 错误：`OSError: libcccl_maca.so: cannot open shared object file`

**原因**：找不到 C 层库。

**解决**：
```bash
export LD_LIBRARY_PATH=/path/to/cccl/build:$LD_LIBRARY_PATH
```

### 错误：`mcMalloc failed with error 999`

**原因**：MACA 运行时未正确初始化。

**解决**：
- 检查 MACA 驱动是否安装
- 确认 GPU 可用
- 尝试运行简单的 MACA 程序验证环境

## 下一步

1. **✅ 已完成**：
   - Phase A C 层（libcccl_maca.so）
   - Phase B Python 层（简化版）

2. **待测试**：
   - 在真实 MACA GPU 上运行测试
   - 验证所有内置算子

3. **未来工作**（如果需要）：
   - 研究 MACA 的 Python JIT 能力
   - 如果有，实现自定义算子支持
   - 添加更多算法（scan, sort 等）

## 文件清单

```
python/
├── README.md                                         # 本文件
├── setup.py                                          # 构建脚本
├── test_reduce_simplified.py                         # 测试示例
└── cuda_cccl/
    └── cuda/
        └── cccl/
            └── parallel/
                └── experimental/
                    ├── __init__.py                   # 包入口
                    ├── _cccl_interop.py              # 类型定义 + OpKind
                    ├── _bindings_maca.pyx            # Cython wrapper
                    └── algorithms/
                        ├── __init__.py
                        └── _reduce.py                # 高层 reduce() API
```

## 参考

- NVIDIA CCCL: https://github.com/NVIDIA/cccl
- MACA 文档: https://developer.metax-tech.com/maca
- Numba: https://numba.pydata.org/
