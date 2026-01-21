# 快速开始指南

## 📋 文件列表

已创建的 Python 层文件（无 Numba 版本）：

```
python/
├── README.md                          # 详细文档
├── QUICKSTART.md                      # 本文件
├── setup.py                           # 构建脚本
├── test_reduce_simplified.py          # 测试示例
└── cuda_cccl/
    └── cuda/
        └── cccl/
            └── parallel/
                └── experimental/
                    ├── __init__.py
                    ├── _cccl_interop.py          # OpKind enum + 类型转换
                    ├── _bindings_maca.pyx        # Cython → C API
                    └── algorithms/
                        ├── __init__.py
                        └── _reduce.py            # 高层 reduce() API
```

## 🚀 3 步使用

### 第 1 步：复制到你的 CCCL 仓库

```bash
# 假设你的 CCCL 在 ~/cccl
cd ~/cccl
cp -r /path/to/this/python ./
```

### 第 2 步：修改 setup.py 路径

编辑 `python/setup.py`：

```python
# 第 15-16 行
MACA_ROOT = os.environ.get('MACA_PATH', '/usr/local/maca')
CCCL_ROOT = os.environ.get('CCCL_PATH', '/home/yourname/cccl')  # ← 改这里
```

### 第 3 步：构建并测试

```bash
cd python

# 设置环境变量
export MACA_PATH=/usr/local/maca
export LD_LIBRARY_PATH=$MACA_PATH/lib64:../build:$LD_LIBRARY_PATH

# 构建 Cython 扩展
python setup.py build_ext --inplace

# 运行测试
python test_reduce_simplified.py
```

## ✅ 预期输出

```
============================================================
CCCL Reduce Tests (Simplified, No Numba)
============================================================

=== Test 1: Sum Reduction ===
Input: 1000 ones
Init value: 10
Result: 1010
Expected: 1010
✅ PASSED

=== Test 2: Min Reduction ===
Input: [5 2 8 1 9]
Init value: 0
Result: 0
Expected: 0
✅ PASSED

=== Test 3: Max Reduction ===
Input: [5 2 8 1 9]
Init value: 100
Result: 100
Expected: 100
✅ PASSED

============================================================
Results: 3/3 tests passed
============================================================
```

## 📝 API 使用示例

### 最简单的例子

```python
from cuda_cccl.cuda.cccl.parallel.experimental import reduce, OpKind
import cupy as cp  # 或者你的 MACA Python wrapper

# 创建数组
d_in = cp.ones(1000, dtype=cp.int32)
d_out = cp.zeros(1, dtype=cp.int32)

# 调用 reduce
reduce(d_in, d_out, op=OpKind.SUM, init=10)

print(d_out)  # [1010]
```

### 支持的算子

```python
reduce(d_in, d_out, op=OpKind.SUM)   # 求和
reduce(d_in, d_out, op=OpKind.MIN)   # 最小值
reduce(d_in, d_out, op=OpKind.MAX)   # 最大值
reduce(d_in, d_out, op=OpKind.PROD)  # 乘积（如果 C 层支持）
```

### 不支持的功能（需要 Numba）

```python
# ❌ 这些都不支持（NVIDIA 可以，MACA 不行）
reduce(d_in, d_out, op=lambda a, b: a * b)
reduce(d_in, d_out, op=custom_function)
reduce(TransformIterator(...), d_out, op=OpKind.SUM)
```

## 🔧 常见问题

### Q1: `ImportError: No module named '_bindings_maca'`

**A**: 忘记构建 Cython 扩展了。

```bash
python setup.py build_ext --inplace
```

### Q2: `OSError: libcccl_maca.so: cannot open`

**A**: 设置 LD_LIBRARY_PATH。

```bash
export LD_LIBRARY_PATH=/path/to/cccl/build:$LD_LIBRARY_PATH
```

### Q3: 我能用 CuPy/PyTorch 数组吗？

**A**: 可以！只要对象提供 `__cuda_array_interface__` 协议。

```python
# CuPy
import cupy as cp
d_in = cp.array([1, 2, 3])
reduce(d_in, d_out, op=OpKind.SUM)  # ✅

# PyTorch (GPU)
import torch
d_in = torch.tensor([1, 2, 3], device='cuda')
reduce(d_in, d_out, op=OpKind.SUM)  # ✅

# 原始指针
d_in_ptr = 0x7f1234567890
reduce(d_in_ptr, d_out_ptr, op=OpKind.SUM, num_items=1000)  # ✅
```

## 📊 对比：NVIDIA vs MACA

| 特性                    | NVIDIA CCCL | MACA Port |
|-------------------------|-------------|-----------|
| 内置算子 (SUM, MIN...)  | ✅          | ✅        |
| 自定义 lambda           | ✅          | ❌        |
| TransformIterator       | ✅          | ❌        |
| JIT 编译                | ✅ (Numba)  | ❌        |
| 性能                    | 100%        | 100% *    |

\* 对于内置算子，性能应该相同（都直接调用 CUB/mcCub）

## 🎯 接下来做什么？

1. **复制这些文件到你的 CCCL 仓库**
2. **修改 setup.py 中的路径**
3. **构建并运行测试**
4. **尝试跑 NVIDIA 的 test_reduce.py**（只跑使用内置算子的测试）

## 📚 更多信息

详见 `README.md`。
