# 详细流程对比：ScanDMM vs AdaptiveScanDMM

## 📊 完整流程对比（逐步骤详解）

---

## 🔴 原始 ScanDMM 流程

### 完整代码流程

```python
def model(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths, 
          images=None, annealing_factor=1.0, predict=False):
    """
    生成模型：p(x_{1:T} | z_{1:T}), p(z_{1:T})
    """
    T_max = scanpaths.size(1)  # 例如：20
    pyro.module("dmm", self)
    
    # ========== 步骤1：状态初始化 ==========
    z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
    # z_prev: (B, 100) - 初始隐状态
    
    z_prev = self.tanh(self.twoZ_to_z_dim(
        torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
    ))
    # 结合初始状态和第一个观测
    # z_prev: (B, 100)
    
    # ========== 步骤2：提取图像特征（关键：只执行一次）==========
    img_features = self.cnn(images)
    # 输入: images (B, 3, 128, 256) - 完整360度图像
    # 处理: Sphere_CNN完整前向传播
    # 输出: img_features (B, 100) - 100维全局特征向量
    # 计算成本: 100%（全图CNN）
    # 时间: T0（只在开始时计算一次）
    # 特点: 所有时间步使用相同的img_features
    
    # ========== 步骤3：序列生成循环（固定长度T_max）==========
    with pyro.plate("z_minibatch", len(scanpaths)):
        for t in pyro.markov(range(1, T_max + 1)):  # t = 1, 2, ..., 20
            
            # ---------- 步骤3.1：状态转移 ----------
            z_mu, z_sigma = self.trans(z_prev, img_features)
            #                                    ↑
            #                              固定不变！
            #                              所有t都使用相同的img_features
            # 输入: z_prev (B, 100), img_features (B, 100)
            # 输出: z_mu, z_sigma (B, 100)
            # 说明: GatedTransition根据上一步状态和图像特征预测当前状态
            
            # ---------- 步骤3.2：采样隐状态 ----------
            with poutine.scale(scale=annealing_factor):
                z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma)
                                  .mask(mask[:, t - 1: t]).to_event(1))
            # z_t: (B, 100) - 当前时间步的隐状态
            
            # ---------- 步骤3.3：生成注视点 ----------
            x_mu, x_sigma = self.emitter(z_t)
            # 输入: z_t (B, 100)
            # 输出: x_mu, x_sigma (B, 3) - 3D坐标的分布参数
            
            if not predict:
                # 训练模式：提供真实观测
                pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                            .mask(mask[:, t - 1: t]).to_event(1),
                            obs=scanpaths[:, t - 1, :])
            else:
                # 预测模式：生成新样本
                pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                            .mask(mask[:, t - 1: t]).to_event(1))
            
            # ---------- 步骤3.4：更新状态 ----------
            z_prev = z_t  # 为下一步准备
```

### 关键特点总结

| 特性 | 原始ScanDMM |
|------|------------|
| **图像特征提取** | 一次性，全图处理，100%成本 |
| **特征复用** | 所有时间步使用相同的img_features |
| **动态性** | ❌ 无，静态特征 |
| **计算成本** | 固定：100% + 序列生成成本 |
| **序列长度** | 固定：T_max=20 |
| **局部信息** | ❌ 无，只有全局特征 |

---

## 🟢 改进后 AdaptiveScanDMM 流程

### 完整代码流程

```python
def model(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths,
          images=None, annealing_factor=1.0, predict=False):
    """
    自适应生成模型：支持多尺度动态特征提取
    """
    T_max = scanpaths.size(1)  # 例如：20
    pyro.module("dmm", self)
    
    # ========== 步骤1：状态初始化（保持不变）==========
    z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
    z_prev = self.tanh(self.twoZ_to_z_dim(
        torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
    ))
    # z_prev: (B, 100)
    
    # ========== 步骤2：Glance阶段 - 提取粗特征（关键改进）==========
    s_coarse = self.cnn(images, mode='coarse')
    # 输入: images (B, 3, 128, 256) - 完整360度图像
    # 处理: 降采样或轻量级Sphere_CNN
    # 输出: s_coarse (B, 50) - 50维粗特征向量
    # 计算成本: ~30%（降采样或轻量级架构）
    # 时间: T0（只在开始时计算一次）
    # 特点: 快速提取全局概览，所有时间步复用
    
    # ========== 步骤3：序列生成循环（关键改进：动态特征）==========
    with pyro.plate("z_minibatch", len(scanpaths)):
        for t in pyro.markov(range(1, T_max + 1)):  # t = 1, 2, ..., 20
            
            # ---------- 步骤3.1：决定使用粗特征还是细特征 ----------
            if t <= self.coarse_threshold:  # 例如：前5步
                # 粗粒度阶段：使用全局粗特征
                current_img_features = F.pad(s_coarse, (0, 50))  # (B, 50) -> (B, 100)
                # 特点: 快速，使用预计算的粗特征
                # 计算成本: 0%（已预计算）
            
            else:
                # 细粒度阶段：根据当前注视点提取局部特征
                
                # ---------- 步骤3.1.1：获取上一个注视点 ----------
                if t == 1:
                    # 第一个时间步：使用初始注视点
                    prev_gaze_2d = xyz2plane(scanpaths[:, 0, :])  # (B, 2)
                else:
                    # 后续时间步：使用上一个生成的注视点
                    prev_gaze_2d = xyz2plane(x_t_prev)  # (B, 2)
                # prev_gaze_2d: (B, 2) - 归一化坐标 [0, 1]，格式 (y, x)
                
                # ---------- 步骤3.1.2：提取局部区域 ----------
                local_patches = self.region_extractor(images, prev_gaze_2d)
                # 输入: images (B, 3, 128, 256), prev_gaze_2d (B, 2)
                # 处理: 根据注视点坐标提取局部区域
                # 输出: local_patches (B, 3, 64, 64) - 局部图像patch
                # 计算成本: ~5%（区域提取，主要是坐标转换）
                # 关键: 处理360度图像的边界连续性
                
                # ---------- 步骤3.1.3：Focus阶段 - 提取细特征 ----------
                s_fine = self.fine_cnn(local_patches)
                # 输入: local_patches (B, 3, 64, 64) - 局部patch
                # 处理: Sphere_CNN（但输入更小）
                # 输出: s_fine (B, 50) - 50维细特征向量
                # 计算成本: ~20%（仅局部区域，比全图小得多）
                # 特点: 精细处理局部区域
                
                # ---------- 步骤3.1.4：特征融合 ----------
                current_img_features = self.feature_fusion(s_coarse, s_fine)
                # 输入: s_coarse (B, 50), s_fine (B, 50)
                # 处理: 融合粗特征和细特征
                # 输出: current_img_features (B, 100) - 融合后的特征
                # 方式: 拼接或学习融合
                # 特点: 结合全局上下文和局部细节
            
            # ---------- 步骤3.2：状态转移（使用动态特征）----------
            z_mu, z_sigma = self.trans(z_prev, current_img_features)
            #                                    ↑
            #                              每个t不同！
            #                              初始阶段: 粗特征
            #                              后续阶段: 粗+细特征融合
            # 输入: z_prev (B, 100), current_img_features (B, 100)
            # 输出: z_mu, z_sigma (B, 100)
            # 关键差异: 每个时间步使用不同的图像特征
            
            # ---------- 步骤3.3：采样隐状态（保持不变）----------
            with poutine.scale(scale=annealing_factor):
                z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma)
                                  .mask(mask[:, t - 1: t]).to_event(1))
            # z_t: (B, 100)
            
            # ---------- 步骤3.4：生成注视点（保持不变）----------
            x_mu, x_sigma = self.emitter(z_t)
            # x_mu, x_sigma: (B, 3)
            
            if not predict:
                pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                            .mask(mask[:, t - 1: t]).to_event(1),
                            obs=scanpaths[:, t - 1, :])
            else:
                x_t = pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                  .mask(mask[:, t - 1: t]).to_event(1))
                x_t_prev = x_t  # 保存用于下一步提取局部特征
            
            # ---------- 步骤3.5：更新状态 ----------
            z_prev = z_t
```

### 关键特点总结

| 特性 | AdaptiveScanDMM |
|------|---------------|
| **图像特征提取** | 初始粗特征（30%成本）+ 动态细特征（20%成本/步） |
| **特征复用** | 粗特征复用，细特征每个时间步动态提取 |
| **动态性** | ✅ 有，根据注视点动态提取局部特征 |
| **计算成本** | 动态：30% + 20%×T（T是实际注视次数） |
| **序列长度** | 固定：T_max=20（可扩展为自适应） |
| **局部信息** | ✅ 有，根据注视点提取局部细节 |

---

## 📊 逐步骤详细对比

### 步骤1：状态初始化

| 项目 | 原始ScanDMM | AdaptiveScanDMM |
|------|------------|----------------|
| **代码** | 完全相同 | 完全相同 |
| **输入** | z_0, scanpaths[:, 0, :] | z_0, scanpaths[:, 0, :] |
| **输出** | z_prev (B, 100) | z_prev (B, 100) |
| **变化** | ❌ 无变化 | ❌ 无变化 |

**结论**：这一步不需要修改。

---

### 步骤2：图像特征提取（关键差异）

#### 原始 ScanDMM

```python
# 位置：model()函数开始，只执行一次
img_features = self.cnn(images)
# 输入: (B, 3, 128, 256)
# 处理: Sphere_CNN完整前向传播
#   1. 添加坐标通道: (B, 3, 128, 256) -> (B, 5, 128, 256)
#   2. SphereConv2D层1: (B, 5, 128, 256) -> (B, 64, 64, 128)
#   3. SphereConv2D层2: (B, 64, 64, 128) -> (B, 128, 32, 64)
#   4. SphereConv2D层3: (B, 128, 32, 64) -> (B, 256, 16, 32)
#   5. SphereConv2D层4: (B, 256, 16, 32) -> (B, 512, 8, 16)
#   6. Conv2d层5: (B, 512, 8, 16) -> (B, 256, 4, 8)
#   7. Conv2d层6: (B, 256, 4, 8) -> (B, 64, 2, 4)
#   8. Flatten: (B, 64, 2, 4) -> (B, 512)
#   9. Linear: (B, 512) -> (B, 100)
# 输出: (B, 100)
# 计算成本: 100%（全图处理）
# 时间: T0（只计算一次）
# 复用: 所有时间步使用相同的img_features
```

**特点**：
- ✅ 一次性提取，后续复用
- ❌ 所有时间步使用相同的特征
- ❌ 无法根据注视点动态调整

#### 改进后 AdaptiveScanDMM

```python
# 位置：model()函数开始，只执行一次（粗特征）
s_coarse = self.cnn(images, mode='coarse')
# 输入: (B, 3, 128, 256)
# 处理: 降采样或轻量级Sphere_CNN
#   选项1：降采样图像到 (B, 3, 64, 128)，然后标准Sphere_CNN
#   选项2：使用轻量级架构（减少层数或通道数）
#   选项3：使用标准Sphere_CNN但输出50维
# 输出: (B, 50)
# 计算成本: ~30%（降采样或轻量级）
# 时间: T0（只计算一次）
# 复用: 所有时间步复用s_coarse

# 位置：循环中，每个时间步（细特征，仅在t > coarse_threshold时）
if t > self.coarse_threshold:
    # 步骤2.1：提取局部区域
    local_patches = self.region_extractor(images, prev_gaze_2d)
    # 输入: images (B, 3, 128, 256), prev_gaze_2d (B, 2)
    # 处理: 根据注视点坐标提取64×64的局部区域
    # 输出: (B, 3, 64, 64)
    # 计算成本: ~5%
    
    # 步骤2.2：提取细特征
    s_fine = self.fine_cnn(local_patches)
    # 输入: (B, 3, 64, 64)
    # 处理: Sphere_CNN（但输入更小）
    #   1. 添加坐标通道: (B, 3, 64, 64) -> (B, 5, 64, 64)
    #   2. SphereConv2D层1: (B, 5, 64, 64) -> (B, 64, 32, 32)
    #   3. SphereConv2D层2: (B, 64, 32, 32) -> (B, 128, 16, 16)
    #   4. SphereConv2D层3: (B, 128, 16, 16) -> (B, 256, 8, 8)
    #   5. SphereConv2D层4: (B, 256, 8, 8) -> (B, 512, 4, 4)
    #   6. Conv2d层5: (B, 512, 4, 4) -> (B, 256, 2, 2)
    #   7. Conv2d层6: (B, 256, 2, 2) -> (B, 64, 1, 1)
    #   8. Flatten: (B, 64, 1, 1) -> (B, 64)
    #   9. Linear: (B, 64) -> (B, 50)
    # 输出: (B, 50)
    # 计算成本: ~20%（仅局部区域）
    # 时间: 每个时间步计算一次
    # 动态: 根据当前注视点动态提取
    
    # 步骤2.3：特征融合
    current_img_features = self.feature_fusion(s_coarse, s_fine)
    # 输入: s_coarse (B, 50), s_fine (B, 50)
    # 处理: 融合（拼接或学习融合）
    # 输出: (B, 100)
```

**特点**：
- ✅ 初始阶段快速（粗特征，30%成本）
- ✅ 后续阶段精细（细特征，20%成本/步）
- ✅ 每个时间步使用不同的特征
- ✅ 根据注视点动态调整

---

### 步骤3：状态转移（关键差异）

#### 原始 ScanDMM

```python
for t in range(1, T_max + 1):
    z_mu, z_sigma = self.trans(z_prev, img_features)
    #                                    ↑
    #                              固定不变
    #                              所有t都使用相同的img_features (B, 100)
    
    # GatedTransition内部：
    # z_t_1_img = torch.cat((z_prev, img_features), dim=1)  # (B, 200)
    # _z_t = self.lin_trans_hidden_to_z(...)  # (B, 100)
    # weight = sigmoid(...)  # 门控权重
    # mu = (1-weight) * z_prev + weight * _z_t  # 加权融合
    # sigma = softplus(...)
```

**特点**：
- `img_features`在所有时间步都是相同的
- 状态转移只依赖隐状态z_{t-1}和固定的图像特征
- 无法利用当前注视点的局部信息

#### 改进后 AdaptiveScanDMM

```python
for t in range(1, T_max + 1):
    if t <= self.coarse_threshold:
        # 粗粒度阶段：使用粗特征
        current_img_features = F.pad(s_coarse, (0, 50))  # (B, 100)
    else:
        # 细粒度阶段：提取局部特征并融合
        local_patches = self.region_extractor(images, prev_gaze_2d)
        s_fine = self.fine_cnn(local_patches)
        current_img_features = self.feature_fusion(s_coarse, s_fine)  # (B, 100)
    
    z_mu, z_sigma = self.trans(z_prev, current_img_features)
    #                                    ↑
    #                              每个t不同！
    #                              初始: 粗特征
    #                              后续: 粗+细特征融合
```

**特点**：
- `current_img_features`每个时间步都不同
- 初始阶段：使用粗特征（全局上下文）
- 后续阶段：使用粗+细特征融合（全局+局部）
- 状态转移考虑了当前注视点的局部信息

---

### 步骤4：生成注视点

| 项目 | 原始ScanDMM | AdaptiveScanDMM |
|------|------------|----------------|
| **代码** | 完全相同 | 完全相同 |
| **输入** | z_t (B, 100) | z_t (B, 100) |
| **输出** | x_t (B, 3) | x_t (B, 3) |
| **变化** | ❌ 无变化 | ❌ 无变化 |

**结论**：这一步不需要修改，但生成的注视点会影响下一步的特征提取。

---

## 📈 计算成本对比（详细）

### 原始 ScanDMM

```
总计算成本 = CNN成本 + 序列生成成本

CNN成本：
- Sphere_CNN前向传播: 100%
- 只计算一次
- 总CNN成本: 100%

序列生成成本（每个时间步）：
- GatedTransition: ~0.5%
- Emitter: ~0.1%
- 采样和损失计算: ~0.1%
- 每个时间步: ~0.7%
- T_max=20步: ~14%

总成本: 100% + 14% = 114%
（CNN占主导）
```

### 改进后 AdaptiveScanDMM

```
总计算成本 = 粗特征成本 + 细特征成本 + 序列生成成本

粗特征成本（初始）：
- 降采样或轻量级CNN: ~30%
- 只计算一次
- 总粗特征成本: 30%

细特征成本（每个时间步，t > coarse_threshold）：
- 区域提取: ~5%
- 细粒度CNN（局部）: ~20%
- 特征融合: ~0.5%
- 每个时间步: ~25.5%
- 假设T=20，前5步用粗特征，后15步用细特征:
  - 细特征成本: 25.5% × 15 = 382.5%

序列生成成本（每个时间步）：
- 与原始相同: ~0.7%
- T_max=20步: ~14%

总成本（T=20）: 30% + 382.5% + 14% = 426.5%

但注意：
- 如果使用自适应长度，简单场景可以提前结束
- 例如T=10: 30% + 25.5%×5 + 14% = 171.5%
- 比原始版本高，但提供了更丰富的特征信息
```

**关键点**：
- 如果所有时间步都用细特征，总成本会更高
- 但前几个时间步用粗特征，可以节省成本
- 更重要的是：**提供了更丰富的特征信息**（全局+局部）

---

## 🔄 数据流对比

### 原始 ScanDMM 数据流

```
时间步 t=1:
  图像 I → [CNN] → img_features (B, 100) [固定]
  z_0 → [GatedTransition(z_0, img_features)] → z_1
  z_1 → [Emitter] → x_1

时间步 t=2:
  图像 I → [CNN] → img_features (B, 100) [固定，复用]
  z_1 → [GatedTransition(z_1, img_features)] → z_2
  z_2 → [Emitter] → x_2

时间步 t=3:
  图像 I → [CNN] → img_features (B, 100) [固定，复用]
  z_2 → [GatedTransition(z_2, img_features)] → z_3
  z_3 → [Emitter] → x_3

...（所有时间步使用相同的img_features）
```

### 改进后 AdaptiveScanDMM 数据流

```
初始阶段（Glance）:
  图像 I → [粗CNN] → s_coarse (B, 50) [固定，复用]

时间步 t=1 (粗粒度阶段):
  s_coarse (B, 50) → [Pad] → current_features (B, 100)
  z_0 → [GatedTransition(z_0, current_features)] → z_1
  z_1 → [Emitter] → x_1

时间步 t=2 (粗粒度阶段):
  s_coarse (B, 50) → [Pad] → current_features (B, 100)
  z_1 → [GatedTransition(z_1, current_features)] → z_2
  z_2 → [Emitter] → x_2

时间步 t=3 (粗粒度阶段):
  s_coarse (B, 50) → [Pad] → current_features (B, 100)
  z_2 → [GatedTransition(z_2, current_features)] → z_3
  z_3 → [Emitter] → x_3

时间步 t=6 (细粒度阶段):
  图像 I + x_5 → [提取局部] → patch (B, 3, 64, 64)
  patch → [细CNN] → s_fine (B, 50)
  s_coarse (B, 50) + s_fine (B, 50) → [融合] → current_features (B, 100)
  z_5 → [GatedTransition(z_5, current_features)] → z_6
  z_6 → [Emitter] → x_6

时间步 t=7 (细粒度阶段):
  图像 I + x_6 → [提取局部] → patch (B, 3, 64, 64)
  patch → [细CNN] → s_fine (B, 50) [新的，基于x_6]
  s_coarse (B, 50) + s_fine (B, 50) → [融合] → current_features (B, 100) [新的]
  z_6 → [GatedTransition(z_6, current_features)] → z_7
  z_7 → [Emitter] → x_7

...（每个时间步使用不同的current_features）
```

**关键差异**：
- ✅ 初始阶段：快速粗特征
- ✅ 后续阶段：根据注视点动态提取细特征
- ✅ 每个时间步的特征都不同

---

## 🎯 关键改进点总结

### 1. 特征提取策略

| 维度 | 原始ScanDMM | AdaptiveScanDMM |
|------|------------|----------------|
| **初始阶段** | 全图CNN (100%) | 粗特征CNN (30%) |
| **后续阶段** | 无（复用初始特征） | 局部细特征CNN (20%/步) |
| **特征类型** | 全局特征（固定） | 全局+局部特征（动态） |
| **计算成本** | 固定100% | 动态30% + 20%×T |

### 2. 状态转移

| 维度 | 原始ScanDMM | AdaptiveScanDMM |
|------|------------|----------------|
| **图像特征** | 固定不变 | 每个时间步不同 |
| **特征来源** | 全局特征 | 全局+局部特征 |
| **动态性** | ❌ 无 | ✅ 有 |

### 3. 信息利用

| 维度 | 原始ScanDMM | AdaptiveScanDMM |
|------|------------|----------------|
| **全局信息** | ✅ 有 | ✅ 有（粗特征） |
| **局部信息** | ❌ 无 | ✅ 有（细特征） |
| **注视点信息** | ❌ 未利用 | ✅ 用于提取局部特征 |

---

## 💡 实施建议

### MVP版本（简化实现）

**核心修改**：
1. ✅ 实现AdaptiveSphere_CNN（粗+细CNN）
2. ✅ 实现RegionExtractor（简化版，处理边界）
3. ✅ 修改DMM.model()，支持动态特征提取
4. ✅ 简单特征融合（拼接）

**不做的**：
- ❌ 不引入策略网络
- ❌ 不引入PPO
- ❌ 不做自适应序列长度
- ❌ 不做复杂球面几何处理

**预期效果**：
- 效率：初始阶段节省70%成本
- 精度：可能提升2-5%（粗细特征结合）
- 实现时间：2-3周

---

## 📝 总结

**核心改进**：
1. **从静态到动态**：图像特征从固定变为动态
2. **从全局到局部**：增加了局部特征提取
3. **从粗到细**：实现了渐进式感知

**关键代码修改点**：
1. 替换`self.cnn`为`AdaptiveSphere_CNN`
2. 修改`model()`方法，在循环中动态提取特征
3. 实现`RegionExtractor`处理360度图像

**预期收益**：
- 效率：初始阶段大幅节省
- 精度：可能提升（需要实验验证）
- 灵活性：可以根据注视点动态调整

