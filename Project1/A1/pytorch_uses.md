# PyTorch 常用语句整理

> 整理自 `pytorch101.py`，按操作类型分类。

---

## 一、张量创建（Tensor Creation）

### 1. `torch.tensor()`
**标准格式：** `torch.tensor(data, dtype=None, device=None, requires_grad=False)`
**作用：** 从 Python 数据（列表、数组等）创建张量。
**示例：**
```python
# 创建一个2x3的张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 指定数据类型和设备
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64, device='cuda')

# 从数组创建并设置 requires_grad（用于自动求导）
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
```

---

### 2. `torch.ones()` / `torch.zeros()`
**标准格式：** `torch.ones(*shape, dtype=None, device=None)` / `torch.zeros(*shape, ...)`
**作用：** 创建全1 / 全0 的张量。
**示例：**
```python
# 创建形状为 (3, 4) 的全1张量
x = torch.ones(3, 4)

# 创建全0张量，指定数据类型
x = torch.zeros(2, 3, dtype=torch.float32)

# 创建与输入形状相同的全1/全0张量
y = torch.ones_like(x)   # 与 x 同shape的全1张量
z = torch.zeros_like(x)  # 与 x 同shape的全0张量
```

---

### 3. `torch.full()`
**标准格式：** `torch.full(size, fill_value, dtype=None, device=None)`
**作用：** 创建指定形状并填充指定值的张量。
**示例：**
```python
# 创建形状为 (2, 3)，全部填充 3.14 的张量
x = torch.full((2, 3), 3.14)

# 填充整数
y = torch.full((4, 4), 99, dtype=torch.int32)
```

---

### 4. `torch.arange()` / `torch.range()`
**标准格式：** `torch.arange(start=0, end, step=1, dtype=None)` / `torch.range(start, end, step=1, ...)`
**作用：** 创建等差序列张量。**注意：`torch.range` 已废弃，推荐使用 `torch.arange`**。
**示例：**
```python
# 创建 [0, 1, 2, ..., 9]
x = torch.arange(10)

# 创建 [5, 10, 15, 20]
x = torch.arange(5, 25, step=5)

# 创建 [2.0, 2.5, 3.0, 3.5]
x = torch.arange(2, 4, step=0.5, dtype=torch.float32)
```

---

### 5. `torch.linspace()`
**标准格式：** `torch.linspace(start, end, steps, dtype=None)`
**作用：** 创建在 start 和 end 之间等间距分布的张量。
**示例：**
```python
# 在 [0, 10] 之间创建5个等距点: [0, 2.5, 5, 7.5, 10]
x = torch.linspace(0, 10, steps=5)

# 创建 Pi 等分点
y = torch.linspace(0, torch.pi, steps=100)
```

---

### 6. `torch.eye()`
**标准格式：** `torch.eye(n, m=None, dtype=None, device=None)`
**作用：** 创建单位矩阵（对角线为1，其余为0）。
**示例：**
```python
# 创建 3x3 单位矩阵
x = torch.eye(3)

# 创建 2x4 的单位矩阵（行数=2，列数=4）
y = torch.eye(2, 4)
```

---

### 7. `torch.rand()` / `torch.randn()` / `torch.randint()`
**标准格式：**
- `torch.rand(*size, dtype=None)` — 均匀分布 [0, 1)
- `torch.randn(*size, dtype=None)` — 标准正态分布 N(0, 1)
- `torch.randint(low, high, size, dtype=torch.long)` — 整数 [low, high)

**示例：**
```python
# 均匀分布随机张量 [0, 1)
x = torch.rand(3, 4)

# 标准正态分布随机张量
y = torch.randn(2, 3)

# 随机整数张量 [1, 10)
z = torch.randint(1, 10, (5, 5))
```

---

### 8. `torch.from_numpy()`
**标准格式：** `torch.from_numpy(ndarray)`
**作用：** 从 NumPy 数组创建张量（共享内存）。
**示例：**
```python
import numpy as np

# NumPy 数组转 PyTorch 张量
np_array = np.array([[1, 2], [3, 4]])
x = torch.from_numpy(np_array)

# 修改 NumPy 数组，张量也会变（共享内存）
np_array[0, 0] = 99
print(x[0, 0])  # 输出 99
```

---

## 二、张量属性（Tensor Attributes）

### 1. `x.shape` / `x.size()`
**标准格式：** `tensor.shape` 或 `tensor.size(dim=None)`
**作用：** 获取张量的形状。
**示例：**
```python
x = torch.randn(3, 4, 5)

print(x.shape)       # torch.Size([3, 4, 5])
print(x.size())      # torch.Size([3, 4, 5])
print(x.size(0))     # 3（第一维的大小）
print(x.size(-1))    # 5（最后一维的大小）
```

---

### 2. `x.dtype` / `x.device`
**标准格式：** `tensor.dtype` / `tensor.device`
**作用：** 获取张量的数据类型 / 所在设备（CPU/CUDA）。
**示例：**
```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)

print(x.dtype)    # torch.float32
print(x.device)   # cpu

# 检查是否在 GPU 上
if x.is_cuda:
    print("在GPU上")
```

---

### 3. `x.ndimension()` / `x.ndim`
**标准格式：** `tensor.ndimension()` / `tensor.ndim`
**作用：** 获取张量的维度数。
**示例：**
```python
x = torch.randn(3, 4, 5)
print(x.ndimension())  # 3
print(x.ndim)          # 3
```

---

### 4. 类型转换
**标准格式：** `x.long()` / `x.float()` / `x.int()` / `x.double()` / `x.half()`
**作用：** 将张量转换为指定数据类型。
**示例：**
```python
x = torch.tensor([1.5, 2.7, 3.9])

y = x.long()    # torch.long: [1, 2, 3]
z = x.int()     # torch.int32: [1, 2, 3]
w = x.double()  # torch.float64: [1.5, 2.7, 3.9]
```

---

## 三、索引与切片（Indexing & Slicing）

### 1. 基本索引
**标准格式：** `x[i, j, k]` 或 `x[i][j][k]`
**作用：** 按位置索引获取单个元素。
**示例：**
```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(x[0, 1])   # 2（第一行，第二列）
print(x[1, -1])  # 6（第二行，最后一列）
```

---

### 2. 切片索引
**标准格式：** `x[start:stop:step, ...]`
**作用：** 按范围切片获取子张量。
**示例：**
```python
x = torch.arange(20).reshape(4, 5)

# 获取最后一行
last_row = x[-1, :]              # torch.Size([5])

# 获取第三列（保留维度）
third_col = x[:, 2:3]            # torch.Size([4, 1])

# 获取前两行前三列
first_two_rows_three_cols = x[:2, :3]  # torch.Size([2, 3])

# 步长切片：取偶数行奇数列
even_rows_odd_cols = x[0::2, 1::2]
# 等差: start=0, stop=end, step=2 → [0, 2]
# 奇数: start=1, stop=end, step=2 → [1, 3]
```

---

### 3. 切片赋值
**标准格式：** `x[start:stop:step] = value`
**作用：** 对切片的元素进行批量赋值（in-place 操作）。
**示例：**
```python
x = torch.zeros(4, 6)

# 对前两行第一列赋值0
x[0:2, 0] = 0

# 对前两行第二列赋值1
x[0:2, 1] = 1

# 对前两行第2-6列赋值2
x[0:2, 2:6] = 2

# 步长赋值：间隔为2的列
x[2:4, 0:4:2] = 3   # 第0,2列
x[2:4, 1:4:2] = 4   # 第1,3列
x[2:4, 4:6] = 5
```

---

### 4. 整数数组索引（Advanced Indexing）
**标准格式：** `x[indices]`，其中 `indices` 是张量
**作用：** 使用索引数组批量获取元素。
**示例：**
```python
x = torch.arange(12).reshape(3, 4)

# 获取指定位置的元素
indices = torch.tensor([0, 2])
print(x[indices])  # 获取第0行和第2行

# 复杂索引：y[0] = x[:, 0]（第一列），y[2] = x[:, 2]（第三列）
y = x[:, torch.tensor([0, 2])]
```

---

### 5. `torch.take()`
**标准格式：** `torch.take(input, index) → Tensor`
**作用：** 将张量视为扁平数组，按索引获取元素。
**示例：**
```python
x = torch.tensor([[1, 2], [3, 4]])
# 扁平化后: [1, 2, 3, 4]，索引0->1, 索引3->4
print(torch.take(x, torch.tensor([0, 3])))  # tensor([1, 4])
```

---

### 6. `torch.gather()`
**标准格式：** `torch.gather(input, dim, index) → Tensor`
**作用：** 沿指定维度收集元素。
**示例：**
```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# 沿 dim=1（列方向）收集，每行取索引 [0, 2] 对应的元素
index = torch.tensor([[0, 2], [1, 2]])
print(torch.gather(x, 1, index))
# 结果: [[1, 3], [4, 6]]
```

---

## 四、张量变形（Reshaping & Manipulation）

### 1. `x.view()` / `x.reshape()`
**标准格式：** `tensor.view(*shape)` / `tensor.reshape(*shape)`
**作用：** 改变张量的形状（元素总数不变）。`view` 可能不连续，`reshape` 更安全。
**示例：**
```python
x = torch.arange(24)

# 将 (24,) 变为 (3, 8)
y = x.view(3, 8)

# 将 (24,) 变为 (2, 3, 4)
z = x.reshape(2, 3, 4)

# 使用 -1 自动推断维度
w = x.view(-1, 6)   # (4, 6)
```

---

### 2. `x.t()` / `x.transpose()`
**标准格式：** `tensor.t()` / `tensor.transpose(dim0, dim1)`
**作用：** 转置（交换维度）。`t()` 仅用于2D，`transpose` 可交换任意两维。
**示例：**
```python
x = torch.randn(3, 4)

# 转置：4x3
y = x.t()

# 交换维度 0 和 2（3D张量）
z = torch.randn(2, 3, 4)
w = z.transpose(0, 2)  # 4x3x2
```

---

### 3. `x.permute()`
**标准格式：** `tensor.permute(*dims)`
**作用：** 按指定顺序重新排列所有维度。
**示例：**
```python
x = torch.randn(2, 3, 4, 5)

# 变成 (5, 4, 3, 2)
y = x.permute(3, 2, 1, 0)
```

---

### 4. `x.squeeze()` / `x.unsqueeze()`
**标准格式：** `tensor.squeeze(dim=None)` / `tensor.unsqueeze(dim)`
**作用：** 移除大小为1的维度 / 在指定位置添加大小为1的维度。
**示例：**
```python
x = torch.randn(1, 3, 1, 4, 1)

# 移除所有大小为1的维度
y = x.squeeze()       # (3, 4)

# 移除指定维度
z = x.squeeze(0)       # (3, 1, 4, 1)
z = x.squeeze(-1)     # (1, 3, 1, 4)

# 在指定位置添加维度
w = torch.randn(3, 4)
w_unsqueezed = w.unsqueeze(0)   # (1, 3, 4)
w_unsqueezed = w.unsqueeze(-1)  # (3, 4, 1)
```

---

### 5. `x.contiguous()`
**标准格式：** `tensor.contiguous()`
**作用：** 返回一个在内存中连续存储的张量拷贝。
**示例：**
```python
x = torch.randn(3, 4)
y = x.t()  # 转置后可能不连续
z = y.contiguous()  # 转为连续张量
```

---

### 6. `torch.cat()` / `torch.stack()`
**标准格式：** `torch.cat(tensors, dim=0)` / `torch.stack(tensors, dim=0)`
**作用：** 连接 / 堆叠张量。`cat` 不增加维度，`stack` 增加一个新维度。
**示例：**
```python
x = torch.randn(2, 3)
y = torch.randn(2, 3)

# 沿 dim=0 连接: (2,3)+(2,3) → (4,3)
z = torch.cat([x, y], dim=0)

# 沿 dim=1 连接: (2,3)+(2,3) → (2,6)
z = torch.cat([x, y], dim=1)

# 堆叠：增加维度 (2,3)+(2,3) → (2,2,3)
z = torch.stack([x, y], dim=0)  # (2, 2, 3)
z = torch.stack([x, y], dim=1)  # (2, 2, 3)
```

---

### 7. `torch.split()` / `torch.chunk()`
**标准格式：** `torch.split(tensor, split_size_or_sections, dim)` / `torch.chunk(num_chunks, dim)`
**作用：** 分割张量为多个子张量。
**示例：**
```python
x = torch.randn(6, 4)

# 按大小分割
a, b, c = torch.split(x, [2, 2, 2], dim=0)  # 分割为 (2,4), (2,4), (2,4)

# 按块数分割
p, q = torch.chunk(x, 2, dim=0)  # 分割为 (3,4), (3,4)
```

---

## 五、数学运算（Mathematical Operations）

### 1. 逐元素运算（Element-wise）
**标准格式：** `x + y`, `x - y`, `x * y`, `x / y`, `x ** y`, `x % y`
**作用：** 逐元素进行数学运算。
**示例：**
```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 2.0, 2.0])

print(x + y)    # [3, 4, 5]
print(x - y)    # [-1, 0, 1]
print(x * y)    # [2, 4, 6]
print(x / y)    # [0.5, 1.0, 1.5]
print(x ** 2)   # [1, 4, 9]
print(x % 2)    # [1, 0, 1]
```

---

### 2. `torch.mm()` / `torch.matmul()` / `@`
**标准格式：** `torch.mm(a, b)` / `torch.matmul(a, b)` / `a @ b`
**作用：** 矩阵乘法。`mm` 适用于2D，`matmul` 支持广播。
**示例：**
```python
a = torch.randn(3, 4)
b = torch.randn(4, 5)

# torch.mm: 3x4 @ 4x5 → 3x5
c = torch.mm(a, b)

# @ 运算符
d = a @ b

# matmul（对于batch矩阵更强大）
torch.matmul(a, b)  # 结果同上
```

---

### 3. `torch.bmm()`（批量矩阵乘法）
**标准格式：** `torch.bmm(batch1, batch2) → Tensor`
**作用：** 批量矩阵乘法，输入形状 `(B, N, M)` 和 `(B, M, P)`，输出 `(B, N, P)`。
**示例：**
```python
B, N, M, P = 2, 3, 4, 5
x = torch.randn(B, N, M)  # (2, 3, 4)
y = torch.randn(B, M, P)  # (2, 4, 5)

# 批量矩阵乘法
z = torch.bmm(x, y)  # (2, 3, 5)
# z[i] = x[i] @ y[i] for i in [0, 1]
```

---

### 4. `x.mm()`（实例方法矩阵乘法）
**标准格式：** `tensor.mm(other)`
**作用：** 张量的实例方法形式的矩阵乘法。
**示例：**
```python
x = torch.randn(3, 4)
w = torch.randn(4, 5)

# 实例方法调用
y = x.mm(w)
```

---

### 5. `torch.sum()` / `x.sum()`
**标准格式：** `torch.sum(input, dim=None, keepdim=False)` / `tensor.sum(...)`
**作用：** 求和。
**示例：**
```python
x = torch.randn(3, 4)

# 全部求和
print(torch.sum(x))    # scalar

# 按维度求和
print(torch.sum(x, dim=0))     # (4,) — 每列求和
print(torch.sum(x, dim=1))     # (3,) — 每行求和

# 保持维度
print(torch.sum(x, dim=0, keepdim=True))  # (1, 4)
```

---

### 6. `torch.mean()` / `x.mean()`
**标准格式：** `torch.mean(input, dim=None, keepdim=False)` / `tensor.mean(...)`
**作用：** 计算均值。
**示例：**
```python
x = torch.randn(3, 4)

print(torch.mean(x))                    # 全部均值
print(torch.mean(x, dim=0))             # 按列均值
print(torch.mean(x, dim=1, keepdim=True)) # 按行均值，保持维度
```

---

### 7. `torch.max()` / `torch.min()` / `x.max()` / `x.min()`
**标准格式：** `torch.max(input, dim=None)` / `tensor.max(dim)`
**作用：** 求最大值/最小值。
**示例：**
```python
x = torch.randn(3, 4)

# 全部最值
print(torch.max(x))    # scalar

# 按维度返回 (values, indices)
values, indices = torch.max(x, dim=1)  # 每行最大值
print(values)   # (3,)
print(indices)  # (3,) — 最大值的位置索引
```

---

### 8. `torch.std()` / `x.std()`
**标准格式：** `torch.std(input, dim=None, unbiased=True)` / `tensor.std(...)`
**作用：** 计算标准差。
**示例：**
```python
x = torch.randn(3, 4)

# 全部标准差
print(torch.std(x))

# 按维度标准差
print(torch.std(x, dim=0))  # 每列标准差
```

---

### 9. `torch.prod()` / `x.prod()`
**标准格式：** `torch.prod(input, dim=None)` / `tensor.prod(dim)`
**作用：** 计算元素乘积。
**示例：**
```python
x = torch.tensor([1, 2, 3, 4])
print(torch.prod(x))  # 24
```

---

### 10. `torch.abs()` / `torch.sqrt()` / `torch.pow()`
**标准格式：** `torch.abs(input)` / `torch.sqrt(input)` / `torch.pow(input, exponent)`
**作用：** 绝对值 / 平方根 / 幂运算。
**示例：**
```python
x = torch.tensor([-4.0, 1.0, 9.0])

print(torch.abs(x))      # [4, 1, 9]
print(torch.sqrt(x))      # [2, 1, 3]
print(torch.pow(x, 2))    # [16, 1, 81]
print(x ** 0.5)          # [nan, 1, 3]（sqrt 等价）
```

---

### 11. `torch.clamp()`
**标准格式：** `torch.clamp(input, min, max)`
**作用：** 将值限制在 [min, max] 范围内。
**示例：**
```python
x = torch.tensor([-1.0, 0.5, 2.0, 5.0])

# 限制在 [0, 1] 范围
y = torch.clamp(x, 0, 1)  # [0, 0.5, 1, 1]
```

---

## 六、归约操作（Reduction Operations）

### 1. 沿维度归约
**标准格式：** `x.sum(dim)`, `x.mean(dim)`, `x.max(dim)`, `x.min(dim)`, `x.std(dim)`, `x.prod(dim)`
**作用：** 沿指定维度进行归约操作，返回该维度上的聚合结果。
**示例：**
```python
x = torch.randn(3, 4)

# 每行的和/最大值
row_sums = x.sum(dim=1)        # (3,)
row_max, row_max_idx = x.max(dim=1)  # (3,), (3,)

# 每列的均值/标准差
col_means = x.mean(dim=0)      # (4,)
col_stds = x.std(dim=0)        # (4,)
```

---

### 2. `x.argmin()` / `x.argmax()`
**标准格式：** `tensor.argmin(dim=None)` / `tensor.argmax(dim=None)`
**作用：** 返回最小/最大值的**索引**。
**示例：**
```python
x = torch.tensor([3, 1, 4, 1, 5])

print(x.argmin())  # 1（最小值1在索引1的位置）
print(x.argmax())  # 4（最大值5在索引4的位置）

# 2D 按维度
y = torch.randn(3, 4)
print(y.argmin(dim=1))  # 每行最小值的列索引 (3,)
```

---

### 3. `torch.topk()`
**标准格式：** `torch.topk(input, k, dim=-1, largest=True)`
**作用：** 返回最大/最小的 k 个元素及其索引。
**示例：**
```python
x = torch.tensor([3, 1, 4, 1, 5, 9])

# 最大3个
values, indices = torch.topk(x, 3)
print(values)   # [9, 5, 4]
print(indices)  # [5, 4, 2]

# 最小2个
min_values, min_indices = torch.topk(x, 2, largest=False)
```

---

### 4. `torch.sort()`
**标准格式：** `torch.sort(input, dim=-1, descending=False)`
**作用：** 对张量排序。
**示例：**
```python
x = torch.tensor([3, 1, 4, 1, 5])

sorted_x, indices = torch.sort(x)
print(sorted_x)  # [1, 1, 3, 4, 5]
print(indices)   # [1, 3, 0, 2, 4]（排序后各元素的原始索引）

# 降序
sorted_desc, _ = torch.sort(x, descending=True)
```

---

## 七、条件与逻辑运算

### 1. `torch.where()`
**标准格式：** `torch.where(condition, x, y)`
**作用：** 条件选择，类似于 `x if condition else y` 的向量化版本。
**示例：**
```python
x = torch.randn(3, 4)
y = torch.zeros(3, 4)

# 将 x 中小于0的元素替换为0
result = torch.where(x < 0, 0, x)

# 复杂条件
result = torch.where(x > 0, x, y)
```

---

### 2. 比较运算
**标准格式：** `x > y`, `x < y`, `x >= y`, `x <= y`, `x == y`, `x != y`
**作用：** 元素级比较，返回布尔张量。
**示例：**
```python
x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([2, 2, 2, 2])

print(x > y)   # [False, False, True, True]
print(x == y)  # [False, True, False, False]
print(x != y)  # [True, False, True, True]
```

---

### 3. `torch.all()` / `torch.any()`
**标准格式：** `torch.all(input)` / `torch.any(input)`
**作用：** 判断是否全部/任一元素为 True。
**示例：**
```python
x = torch.tensor([True, False, True])

print(torch.all(x))  # False
print(torch.any(x))  # True

# 配合条件使用
y = torch.tensor([1, 2, 3, 4])
print(torch.all(y > 0))  # True
```

---

## 八、GPU 操作

### 1. `x.cuda()` / `x.cpu()`
**标准格式：** `tensor.cuda(device=None)` / `tensor.cpu()`
**作用：** 将张量在 GPU 和 CPU 之间移动。
**示例：**
```python
x = torch.randn(3, 4)

# 移到 GPU
if torch.cuda.is_available():
    x_gpu = x.cuda()
    print(x_gpu.device)  # cuda:0

    # 移回 CPU
    x_cpu = x_gpu.cpu()
    print(x_cpu.device)  # cpu

# .to(device) 更通用的写法
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_gpu = x.to(device)
```

---

### 2. `torch.device()`
**标准格式：** `torch.device('cuda')` / `torch.device('cuda:0')` / `torch.device('cpu')`
**作用：** 创建设备对象，指定张量存放位置。
**示例：**
```python
device = torch.device('cuda:0')

x = torch.randn(3, 4, device=device)  # 直接在 GPU 上创建
y = torch.randn(3, 4).to(device)     # 移动到 GPU
```

---

## 九、自动求导（Autograd）

### 1. `x.requires_grad`
**标准格式：** `tensor.requires_grad_(True)` / `x.requires_grad`
**作用：** 设置张量是否参与自动求导计算图。
**示例：**
```python
# 创建时指定
x = torch.tensor([1.0, 2.0], requires_grad=True)

# 后续修改
x.requires_grad_(True)

# 追踪操作
y = x ** 2
z = y.sum()  # z = 1 + 4 = 5
z.backward()  # 计算梯度 dz/dx = 2x = [2, 4]
print(x.grad)  # [2, 4]
```

---

### 2. `x.grad`
**标准格式：** `tensor.grad`
**作用：** 存储计算出的梯度值。
**示例：**
```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()
z.backward()
print(x.grad)  # tensor([4., 6.])  # dz/dx = 2x
```

---

### 3. `torch.no_grad()`
**标准格式：** `with torch.no_grad():`
**作用：** 禁用梯度计算，用于推理阶段以节省内存。
**示例：**
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)

# 不追踪梯度
with torch.no_grad():
    y = x ** 2  # y.requires_grad 仍为 False
    print(y.requires_grad)  # False
```

---

## 十、其他常用操作

### 1. `torch.flip()`
**标准格式：** `torch.flip(input, dims)`
**作用：** 沿指定维度翻转张量。
**示例：**
```python
x = torch.arange(8).reshape(2, 4)

# 翻转行（逆序）
y = torch.flip(x, dims=[0])  # 沿 dim=0 翻转

# 翻转列
z = torch.flip(x, dims=[1])

# 翻转多个维度
w = torch.flip(x, dims=[0, 1])
```

---

### 2. `torch.unique()`
**标准格式：** `torch.unique(input, sorted=True, return_inverse=False)`
**作用：** 返回张量中的唯一值。
**示例：**
```python
x = torch.tensor([2, 1, 3, 2, 1])

uniques = torch.unique(x)
print(uniques)  # tensor([1, 2, 3])

# 返回唯一值和对应的索引
uniques, indices = torch.unique(x, return_inverse=True)
print(indices)  # tensor([1, 0, 2, 1, 0]) — 唯一值在原始张量中的位置
```

---

### 3. `torch.clone()`
**标准格式：** `tensor.clone()`
**作用：** 创建张量的深拷贝。
**示例：**
```python
x = torch.randn(3, 4)
y = x.clone()  # y 是独立于 x 的新张量
y.fill_(0)     # 修改 y 不影响 x
```

---

### 4. `torch.detached()`
**标准格式：** `tensor.detach()`
**作用：** 断开计算图连接，用于从图中分离张量。
**示例：**
```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# 断开梯度追踪
z = y.detach()
print(z.requires_grad)  # False
```

---

### 5. `torch.item()`
**标准格式：** `tensor.item()`
**作用：** 将单元素张量转换为 Python 标量。
**示例：**
```python
x = torch.tensor([42.0])
print(x.item())   # 42.0（Python float）
print(x.item())   # 42（如果 dtype=int）

# 用于从 GPU 获取标量（必须先转到 CPU）
if x.is_cuda:
    print(x.cpu().item())
```

---

### 6. `torch.zeros_like()` / `torch.ones_like()`
**标准格式：** `torch.zeros_like(input)` / `torch.ones_like(input)`
**作用：** 创建与输入同 shape 的全 0 / 全 1 张量。
**示例：**
```python
x = torch.randn(3, 4)
zeros = torch.zeros_like(x)  # 同 shape，全 0
ones = torch.ones_like(x)    # 同 shape，全 1
```

---

## 十一、文件练习中出现的特殊模式

### 1. 批量操作中的 Python 循环
```python
# 在 batched_matrix_multiply_loop 中的模式
for k in range(len(indices)):
    i, j = indices[k]
    x[i, j] = values[k]
```

### 2. 列表推导式转张量
```python
# multiples_of_ten 函数
multiples = [i for i in range(start, stop + 1) if i % 10 == 0]
x = torch.tensor(multiples, dtype=torch.float64)
```

### 3. 切片与步长组合
```python
# slice_indexing_practice 函数
even_rows_odd_cols = x[0::2, 1::2]
# 0::2 表示从索引0开始，步长2 → [0, 2, 4, ...]
# 1::2 表示从索引1开始，步长2 → [1, 3, 5, ...]
```

---

## 附录：数据类型（dtype）速查

| dtype | 说明 | 常见用途 |
|-------|------|---------|
| `torch.float32` / `torch.float` | 32位浮点 | 默认，深度学习权重 |
| `torch.float64` / `torch.double` | 64位浮点 | 高精度计算 |
| `torch.float16` / `torch.half` | 16位浮点 | GPU加速，混合精度 |
| `torch.int32` / `torch.int` | 32位整数 | 通用整数 |
| `torch.int64` / `torch.long` | 64位整数 | 索引，Long tensor |
| `torch.int16` / `torch.short` | 16位整数 | 节省内存 |
| `torch.bool` | 布尔 | 条件掩码 |