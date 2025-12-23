# 为什么 mu 和 sigma 的 shape 是 64×100？

## 📊 Shape 解析

### 基本结构
```
mu.shape = (64, 100)
sigma.shape = (64, 100)
```

### 维度含义

#### 第一个维度：64 = batch_size（批次大小）
- **含义**：同时处理 **64 个样本**
- **为什么是这个数字？**：这是训练时的批次大小（batch size），通常在训练参数中设置
- **作用**：PyTorch 的 Linear 层可以并行处理多个样本，提高计算效率

#### 第二个维度：100 = z_dim（隐状态维度）
- **含义**：每个样本的隐状态有 **100 个特征维度**
- **定义位置**：在模型初始化时设置 `z_dim=100`
- **作用**：表示每个样本的状态信息由 100 个数值描述

---

## 🔍 Shape 变化过程

### 输入阶段

```python
# 在模型调用时（scandmm_integrated.py 第552-564行）
scanpaths.shape = (64, T, 3)  # 64个样本，每个有T个时间步，每步3个坐标

# 初始化 z_prev
z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
# z_prev.shape = (64, 100)  # 64个样本，每个100维

# 提取图像特征
img_features = self.cnn(images)
# img_features.shape = (64, 100)  # 64个样本，每个100维特征
```

### Forward 过程

```python
def forward(self, z_t_1, img_feature=None):
    # z_t_1.shape = (64, 100)  # 输入：64个样本，每个100维
    # img_feature.shape = (64, 100)  # 输入：64个样本，每个100维
    
    # 拼接操作：在特征维度上拼接
    z_t_1_img = torch.cat((z_t_1, img_feature), dim=1)
    # z_t_1_img.shape = (64, 200)  # 64个样本，200维（100+100）
    
    # 通过转移网络
    _z_t = self.lin_trans_hidden_to_z(...)
    # _z_t.shape = (64, 100)  # 64个样本，100维
    
    # 计算门控权重
    weight = torch.sigmoid(...)
    # weight.shape = (64, 100)  # 64个样本，100维权重
    
    # 计算均值（加权混合）
    mu = (1 - weight) * self.lin_z_to_mu(z_t_1) + weight * _z_t
    # mu.shape = (64, 100)  # 64个样本，100维
    
    # 计算标准差
    sigma = self.softplus(...)
    # sigma.shape = (64, 100)  # 64个样本，100维
    
    return mu, sigma
    # 返回：mu和sigma都是 (64, 100)
```

---

## 💡 关键点：PyTorch Linear 层的行为

### PyTorch Linear 层保持 batch 维度

```python
# Linear层的定义
self.lin_trans_hidden_to_z = nn.Linear(hidden_dim, z_dim)
# 这个层将 hidden_dim 维映射到 z_dim 维

# 但是！
# 如果输入是 (batch_size, hidden_dim)
# 输出就是 (batch_size, z_dim)
# batch维度保持不变！
```

**示例**：
```python
# 定义层
linear = nn.Linear(200, 100)

# 输入：batch_size=64, 特征维度=200
input_tensor = torch.randn(64, 200)

# 输出：batch_size=64, 特征维度=100
output_tensor = linear(input_tensor)
print(output_tensor.shape)  # torch.Size([64, 100])
```

---

## 🎯 为什么需要 batch 维度？

### 并行处理优势

1. **效率**：同时处理 64 个样本比逐个处理快得多
2. **GPU 加速**：GPU 擅长并行计算，batch 越大，利用率越高
3. **梯度稳定**：使用多个样本的平均梯度，训练更稳定

### 实际含义

```python
# mu[0, :] 表示第1个样本的预测均值（100维）
# mu[1, :] 表示第2个样本的预测均值（100维）
# ...
# mu[63, :] 表示第64个样本的预测均值（100维）

# 例如：
mu[0] = [0.5, 0.3, 0.8, ..., 0.2]  # 样本1的100维隐状态均值
mu[1] = [0.4, 0.6, 0.7, ..., 0.3]  # 样本2的100维隐状态均值
```

---

## 📐 完整的 Shape 流程图

```
输入:
├─ z_t_1: (64, 100)     [64个样本，每个100维隐状态]
└─ img_feature: (64, 100) [64个样本，每个100维图像特征]

步骤1: 拼接
└─ z_t_1_img: (64, 200)  [64个样本，200维（拼接结果）]

步骤2: 转移网络
├─ lin_trans_2z_to_hidden: (64, 200) -> (64, 200)
└─ lin_trans_hidden_to_z: (64, 200) -> (64, 100)
   └─ _z_t: (64, 100)

步骤3: 门控网络
├─ lin_gate_z_to_hidden_dim: (64, 100) -> (64, 200)
└─ lin_gate_hidden_dim_to_z: (64, 200) -> (64, 100)
   └─ weight: (64, 100)

步骤4: 混合计算
├─ lin_z_to_mu(z_t_1): (64, 100)
├─ _z_t: (64, 100)
└─ weight: (64, 100)
   └─ mu: (64, 100) = (1-weight) * lin_z_to_mu(z_t_1) + weight * _z_t

步骤5: 方差计算
├─ lin_sig(_z_t): (64, 100)
└─ sigma: (64, 100) = softplus(lin_sig(_z_t))

输出:
├─ mu: (64, 100)    [64个样本，每个100维均值]
└─ sigma: (64, 100) [64个样本，每个100维标准差]
```

---

## 🎓 总结

**为什么是 64×100？**

1. **64** = batch_size（批次大小）
   - 同时处理 64 个样本
   - 由训练配置决定，可以是其他值（如 32, 128）

2. **100** = z_dim（隐状态维度）
   - 每个样本有 100 个特征维度
   - 由模型初始化参数 `z_dim=100` 决定

3. **PyTorch Linear 层保持 batch 维度**
   - 输入 `(batch_size, in_features)` → 输出 `(batch_size, out_features)`
   - 所以最终输出的 shape 是 `(batch_size, z_dim) = (64, 100)`

**实际使用**：
- `mu[i, j]` 表示第 i 个样本的第 j 个特征维度的预测均值
- `sigma[i, j]` 表示第 i 个样本的第 j 个特征维度的预测标准差
- 每个样本都有自己独立的 100 维高斯分布参数
