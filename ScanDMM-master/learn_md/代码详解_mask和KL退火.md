# 代码详解：mask、KL退火和采样机制

## 📝 代码片段

```python
with poutine.scale(scale=annealing_factor):  # KL退火
    z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma)
                      .mask(mask[:, t - 1: t]).to_event(1))
```

---

## 🔍 逐部分详解

### 1. `poutine.scale(scale=annealing_factor)` - KL散度退火

#### 作用
**缩放KL散度项的权重**，这是变分推理中的训练技巧。

#### 工作原理

```python
# 在ELBO损失中：
# ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
#                    ↑              ↑
#               重构损失          KL散度项

# 使用poutine.scale后：
# ELBO = E_q[log p(x|z)] - annealing_factor * KL(q(z|x) || p(z))
#                                    ↑
#                            退火因子缩放KL项
```

#### 为什么需要退火？

**问题**：训练初期，模型参数随机初始化，如果直接最大化完整ELBO（KL项权重=1.0），可能会遇到：
- **模式崩塌（Mode Collapse）**：KL散度项会强制guide接近先验，导致隐状态变化太小
- **后验坍塌（Posterior Collapse）**：模型倾向于忽略隐状态，直接学习重构

**解决方案**：逐步增加KL项的权重

```python
# 训练初期：KL项权重小
annealing_factor = 0.2  # 只关注重构损失
# ELBO ≈ E_q[log p(x|z)] - 0.2 * KL(...)

# 训练中期：KL项权重逐渐增加
annealing_factor = 0.6  # 平衡重构和正则化

# 训练后期：KL项权重达到1.0
annealing_factor = 1.0  # 完整的ELBO
# ELBO = E_q[log p(x|z)] - 1.0 * KL(...)
```

#### 退火因子计算示例

```python
# 假设：min_af=0.2, annealing_epochs=10, 当前epoch=5, batch=50

# 计算当前步数
current_step = 50 + 5 * N_mini_batches + 1
total_steps = 10 * N_mini_batches
progress = current_step / total_steps  # 例如: 0.5

# 计算退火因子
annealing_factor = 0.2 + (1.0 - 0.2) * 0.5 = 0.6
```

#### 代码实现机制

```python
# Pyro的poutine.scale会包装sample语句
# 在计算损失时，自动将KL散度乘以scale因子

with poutine.scale(scale=0.5):
    z = pyro.sample("z", dist.Normal(0, 1))

# 等价于：
# KL损失 = 0.5 * 原始KL损失
```

---

### 2. `pyro.sample("z_%d" % t, ...)` - 概率采样

#### 作用
**从概率分布中采样隐状态** `z_t`。

#### 工作原理

```python
# 在model中（生成模型）：
z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma))
# 含义：从分布 N(z_mu, z_sigma²) 中采样 z_t
# 采样：z_t ~ N(z_mu, z_sigma²)

# 在guide中（变分后验）：
z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma))
# 含义：从变分后验 q(z_t | z_{t-1}, x_{t:T}) 中采样
# 注意：必须与model中的名称"z_%d"相同，这样Pyro才能匹配它们计算KL散度
```

#### 为什么需要名称匹配？

```python
# Pyro通过名称匹配model和guide中的采样节点

# Model中：
z_t = pyro.sample("z_1", dist.Normal(mu1, sigma1))  # 先验 p(z)

# Guide中：
z_t = pyro.sample("z_1", dist.Normal(mu2, sigma2))  # 后验 q(z|x)

# Pyro自动计算：KL(q(z|x) || p(z))
# 因为名称"z_1"相同，所以知道这两个分布需要匹配
```

#### 维度示例

```python
# 输入：
z_mu: (batch_size, 100)   # 例如: (64, 100)
z_sigma: (batch_size, 100)  # 例如: (64, 100)

# 采样：
z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma))
# z_t: (batch_size, 100)  # 例如: (64, 100)
```

---

### 3. `.mask(mask[:, t - 1: t])` - 掩码处理变长序列

#### 作用
**处理变长序列**：标记哪些样本在时间步t是有效的，哪些是填充的。

#### mask的形状和含义

```python
# mask的形状：(batch_size, T_max)
# 例如: (64, 100) - 64个样本，最多100个时间步

# mask的值：
# - 1：表示该时间步有效（有真实数据）
# - 0：表示该时间步无效（填充，序列已结束）

# 示例（batch_size=4, T_max=5）:
mask = [
    [1, 1, 1, 1, 1],  # 样本0：长度为5，全部有效
    [1, 1, 1, 1, 0],  # 样本1：长度为4，第5步是填充
    [1, 1, 1, 0, 0],  # 样本2：长度为3，第4-5步是填充
    [1, 1, 0, 0, 0],  # 样本3：长度为2，第3-5步是填充
]
```

#### `mask[:, t - 1: t]` 的含义

```python
# mask: (batch_size, T_max)  # 例如: (64, 100)
# t: 当前时间步（从1开始）   # 例如: t=1, 2, ..., 100

mask[:, t - 1: t]  # 提取时间步 t-1 的掩码
# shape: (batch_size, 1)  # 例如: (64, 1)

# 示例（t=3）:
mask[:, 2:3]  # 提取第2列（索引从0开始，所以是t-1=2）
# 结果: [[1], [1], [1], [0], ...]  # (64, 1)
#        ↑    ↑    ↑    ↑
#       有效 有效 有效 无效（序列已结束）
```

#### mask如何生成？

```python
# 在get_mini_batch中：
mini_batch_mask = poly.get_mini_batch_mask(mini_batch, sorted_seq_lengths)
# sorted_seq_lengths: (batch_size,) - 每个序列的长度
# 例如: [100, 95, 90, 85, ...]

# get_mini_batch_mask的逻辑（伪代码）:
def get_mini_batch_mask(sequences, lengths):
    batch_size, T_max = sequences.shape[:2]
    mask = torch.zeros(batch_size, T_max)
    
    for i in range(batch_size):
        seq_len = lengths[i]  # 例如: 85
        mask[i, :seq_len] = 1  # 前85个为1（有效）
        mask[i, seq_len:] = 0  # 后15个为0（填充）
    
    return mask
```

#### mask在采样中的作用

```python
# 当mask=1时（有效时间步）：
z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma).mask(True))
# 正常采样，参与损失计算

# 当mask=0时（无效时间步）：
z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma).mask(False))
# 采样结果被忽略，不参与损失计算
# Pyro会自动处理：只计算有效时间步的损失
```

#### 为什么需要mask？

**问题**：不同序列长度不同，但需要批量处理

```python
# 示例：一个batch有3个序列
序列1: [x1, x2, x3, x4, x5]           # 长度=5
序列2: [x1, x2, x3]                   # 长度=3
序列3: [x1, x2, x3, x4, x5, x6]       # 长度=6

# 为了批量处理，需要填充到相同长度（最长=6）:
序列1: [x1, x2, x3, x4, x5, PAD]      # 长度=6，第6步是填充
序列2: [x1, x2, x3, PAD, PAD, PAD]    # 长度=6，第4-6步是填充
序列3: [x1, x2, x3, x4, x5, x6]       # 长度=6，全部有效

# mask标记哪些是真实的，哪些是填充的:
mask = [
    [1, 1, 1, 1, 1, 0],  # 序列1：前5个有效
    [1, 1, 1, 0, 0, 0],  # 序列2：前3个有效
    [1, 1, 1, 1, 1, 1],  # 序列3：全部有效
]
```

**解决方案**：使用mask告诉模型哪些是有效数据

```python
# 在计算损失时：
# 只有mask=1的时间步会贡献损失
# mask=0的时间步被忽略（不会影响梯度更新）
```

---

### 4. `.to_event(1)` - 事件维度处理

#### 作用
**将最后一个维度标记为事件维度**，用于多变量分布。

#### 为什么需要？

```python
# 隐状态z_t是多变量（100维）
z_t: (batch_size, 100)
#     ↑          ↑
#   批次维度   事件维度（100个变量）

# 不使用to_event(1):
dist.Normal(z_mu, z_sigma)
# 这被解释为100个独立的单变量正态分布
# 每个分布独立采样

# 使用to_event(1):
dist.Normal(z_mu, z_sigma).to_event(1)
# 这被解释为1个100维的多变量正态分布
# 所有维度一起采样（虽然这里是独立的高斯，但语义上更清晰）
```

#### 维度说明

```python
# 输入分布参数：
z_mu: (batch_size, z_dim)      # (64, 100)
z_sigma: (batch_size, z_dim)   # (64, 100)

# 创建分布：
dist_normal = dist.Normal(z_mu, z_sigma)
# batch_shape: (64, 100)  # 64个批次，每个100个分布
# event_shape: ()         # 每个分布是标量

# 使用to_event(1):
dist_normal_event = dist.Normal(z_mu, z_sigma).to_event(1)
# batch_shape: (64,)      # 64个批次
# event_shape: (100,)     # 每个批次是一个100维的分布
```

#### 实际效果

```python
# 采样结果相同，但语义不同：

# 方式1：不使用to_event
z_t = pyro.sample("z_1", dist.Normal(z_mu, z_sigma))
# 语义：从100个独立的单变量分布采样

# 方式2：使用to_event(1)
z_t = pyro.sample("z_1", dist.Normal(z_mu, z_sigma).to_event(1))
# 语义：从1个100维的多变量分布采样
# 效果：相同（因为100个高斯是独立的）

# 但使用to_event(1)更符合多变量隐状态的语义
```

---

## 🔄 完整代码流程

```python
# 步骤1：计算分布参数
z_mu, z_sigma = self.trans(z_prev, img_features)
# z_mu: (64, 100), z_sigma: (64, 100)

# 步骤2：获取当前时间步的掩码
time_mask = mask[:, t - 1: t]  # (64, 1)
# 例如: [[1], [1], [1], [0], ...]  # 前3个有效，第4个无效

# 步骤3：创建分布并应用掩码
z_dist = dist.Normal(z_mu, z_sigma)  # 创建分布
z_dist_masked = z_dist.mask(time_mask)  # 应用掩码
z_dist_event = z_dist_masked.to_event(1)  # 标记事件维度

# 步骤4：在KL退火上下文中采样
with poutine.scale(scale=0.6):  # 假设退火因子=0.6
    z_t = pyro.sample("z_%d" % t, z_dist_event)
    # 采样结果: (64, 100)
    # 但是：
    #   - 对于mask=1的样本：正常采样，参与损失计算
    #   - 对于mask=0的样本：采样被忽略，不参与损失计算
    #   - KL散度会被乘以0.6（退火因子）
```

---

## 📊 维度变化图解

```
输入:
├─ z_prev: (batch_size, 100)           # 例如: (64, 100)
├─ img_features: (batch_size, 100)     # 例如: (64, 100)
└─ mask: (batch_size, T_max)           # 例如: (64, 100)

步骤1: GatedTransition计算分布参数
├─ z_mu: (batch_size, 100)             # (64, 100)
└─ z_sigma: (batch_size, 100)          # (64, 100)

步骤2: 提取掩码
└─ mask[:, t-1:t]: (batch_size, 1)     # (64, 1)

步骤3: 创建分布
├─ dist.Normal(z_mu, z_sigma)
│  └─ batch_shape: (64, 100)
│  └─ event_shape: ()
│
├─ .mask(mask[:, t-1:t])
│  └─ 应用掩码：标记哪些样本有效
│
└─ .to_event(1)
   └─ batch_shape: (64,)
   └─ event_shape: (100,)

步骤4: 采样（在poutine.scale上下文中）
└─ z_t: (batch_size, 100)              # (64, 100)
   ├─ mask=1的位置：正常采样
   └─ mask=0的位置：采样结果被忽略
```

---

## 🎯 实际运行示例

假设 `batch_size=4`, `z_dim=100`, `T_max=5`, `t=3`, `annealing_factor=0.6`:

```python
# 输入数据
mask = torch.tensor([
    [1, 1, 1, 1, 1],  # 样本0：长度5
    [1, 1, 1, 1, 0],  # 样本1：长度4
    [1, 1, 1, 0, 0],  # 样本2：长度3
    [1, 1, 0, 0, 0],  # 样本3：长度2
])

# 时间步t=3（索引从1开始，所以取mask的第2列，即t-1=2）
time_mask = mask[:, 2:3]  # shape: (4, 1)
# [[1],   # 样本0：有效
#  [1],   # 样本1：有效
#  [1],   # 样本2：有效
#  [0]]   # 样本3：无效（序列已结束）

# 分布参数
z_mu = torch.randn(4, 100)   # 随机示例
z_sigma = torch.abs(torch.randn(4, 100))  # 必须>0

# 采样
with poutine.scale(scale=0.6):
    z_t = pyro.sample("z_3", 
                     dist.Normal(z_mu, z_sigma)
                     .mask(time_mask)    # 样本3的采样被忽略
                     .to_event(1))

# 结果：
# z_t: (4, 100)
# - 样本0-2：正常采样，参与损失计算，KL项乘以0.6
# - 样本3：虽然采样了，但在损失计算时会被忽略（mask=0）
```

---

## 🔑 关键要点总结

1. **`poutine.scale`**：
   - 缩放KL散度项的权重
   - 实现KL退火，避免训练初期模式崩塌
   - 从 `min_af` 线性增长到 1.0

2. **`pyro.sample`**：
   - 从概率分布采样
   - model和guide中的名称必须匹配（用于计算KL散度）

3. **`.mask()`**：
   - 处理变长序列
   - 标记哪些样本在某个时间步有效（1=有效，0=填充）
   - 无效时间步不参与损失计算

4. **`.to_event(1)`**：
   - 标记事件维度
   - 将多变量分布正确解释为事件分布
   - 语义上更清晰（100维隐状态作为整体）

---

## 💡 为什么这样设计？

1. **批量处理变长序列**：
   - 不同序列长度不同，但需要批量处理以提高效率
   - mask标记填充位置，确保填充部分不影响训练

2. **KL退火**：
   - 训练初期更关注重构，后期更关注正则化
   - 避免后验坍塌和模式崩塌

3. **事件维度**：
   - 隐状态是多维的（100维），应该作为整体处理
   - 使用to_event(1)使语义更清晰
