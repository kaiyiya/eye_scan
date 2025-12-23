# ScanDMM + AdaptiveNN：完整改进方案设计

## 📋 概述

本文档基于AdaptiveNN的"从粗到细"主动感知机制，设计一个完整的ScanDMM改进方案，实现**自适应、高效、可解释**的360度图像扫描路径预测。

---

## 🎯 AdaptiveNN核心思想回顾

### 关键机制

1. **Glance（快速扫描）**：降采样图像，获取整体概览
2. **Fixations（注视序列）**：顺序选择感兴趣区域进行精细观察
3. **策略网络（π）**：决定下一个注视位置
4. **价值网络（V*）**：评估是否继续观察（决策终止）
5. **自奖励强化学习**：端到端训练，无需额外标注

### 核心优势

- ✅ **效率**：计算成本降低最高28倍
- ✅ **灵活性**：在线调整计算需求
- ✅ **可解释性**：注视点可视化
- ✅ **广泛适用**：兼容多种架构和任务

---

## 🔍 ScanDMM当前架构分析

### 现有流程

```
输入：360度图像 I (3, H, W)
  ↓
[Sphere CNN] → 全局特征 s (100维)  [一次性提取，固定]
  ↓
[状态初始化] → z_0 = F(z_0, x_1)
  ↓
循环生成（固定长度T_max）：
  z_t ~ GatedTransition(z_{t-1}, s)  [所有t使用相同的s]
  x_t ~ Emitter(z_t)
```

### 主要问题

1. ❌ **静态特征**：所有时间步使用相同的全局特征
2. ❌ **固定计算**：无论场景复杂度，都执行相同计算
3. ❌ **固定长度**：所有扫描路径都是固定长度T_max
4. ❌ **缺乏主动决策**：无法根据信息量决定是否继续

---

## 💡 改进方案：AdaptiveScanDMM

### 整体架构设计

```
输入：360度图像 I
  ↓
[Glance阶段] → 粗粒度全局特征 s_coarse (50维)  [快速，低分辨率]
  ↓
[策略网络 π] → 初始注视位置 x_1
  ↓
循环注视（自适应长度）：
  [价值网络 V*] → 评估是否继续 (continue/stop)
    ↓ continue
  [局部特征提取] → 细粒度局部特征 s_fine (50维)  [基于当前注视点]
  [特征融合] → s = F(s_coarse, s_fine)  [100维]
  [状态转移] → z_t ~ GatedTransition(z_{t-1}, s)
  [眼动生成] → x_t ~ Emitter(z_t)
  [策略网络 π] → 下一个注视位置 x_{t+1}
    ↓ stop
  [输出扫描路径] → x_{1:T} (T ≤ T_max)
```

---

## 🏗️ 核心组件设计

### 1. Glance模块（快速扫描）

**功能**：快速获取360度图像的整体概览

```python
class GlanceModule(nn.Module):
    """
    Glance阶段：快速扫描，获取全局粗粒度特征
    """
    def __init__(self, z_dim=100):
        super().__init__()
        # 使用轻量级球面CNN，输入降采样图像
        self.coarse_cnn = Sphere_CNN(out_put_dim=z_dim // 2)  # 50维
        
    def forward(self, images):
        """
        Args:
            images: (batch, 3, H, W) - 原始360度图像
        
        Returns:
            coarse_features: (batch, 50) - 粗粒度全局特征
        """
        # 可选：对图像进行降采样（如H/2, W/2）
        # 或者使用更轻量的CNN架构
        coarse_features = self.coarse_cnn(images)
        return coarse_features
```

**设计要点**：
- 使用降采样图像或轻量级CNN
- 只提取全局语义信息，不关注细节
- 计算成本约为全分辨率CNN的20-30%

---

### 2. 局部特征提取模块

**功能**：根据当前注视点提取局部区域的细粒度特征

```python
class LocalFeatureExtractor(nn.Module):
    """
    局部特征提取器：基于注视点提取局部区域特征
    """
    def __init__(self, z_dim=100, region_size=64):
        super().__init__()
        self.region_size = region_size
        self.fine_cnn = Sphere_CNN(out_put_dim=z_dim // 2)  # 50维
        
    def extract_region(self, images, gaze_points_2d):
        """
        从360度图像中提取注视点周围的局部区域
        
        Args:
            images: (batch, 3, H, W) - 360度图像
            gaze_points_2d: (batch, 2) - 归一化坐标 (y, x) in [0, 1]
        
        Returns:
            local_regions: (batch, 3, region_size, region_size)
        """
        batch_size, _, H, W = images.shape
        
        # 转换为像素坐标
        y_pixel = (gaze_points_2d[:, 0] * H).long()
        x_pixel = (gaze_points_2d[:, 1] * W).long()
        
        # 考虑360度图像的左右边界连续性
        # 使用grid_sample提取局部区域
        # ... (实现细节)
        
        return local_regions
    
    def forward(self, images, gaze_points_2d):
        """
        Args:
            images: (batch, 3, H, W)
            gaze_points_2d: (batch, 2) - 当前注视点
        
        Returns:
            fine_features: (batch, 50) - 细粒度局部特征
        """
        local_regions = self.extract_region(images, gaze_points_2d)
        fine_features = self.fine_cnn(local_regions)
        return fine_features
```

**设计要点**：
- 考虑360度图像的球面几何特性
- 处理左右边界连续性（等距圆柱投影）
- 局部区域大小可调（如64x64像素）

---

### 3. 策略网络（π）

**功能**：决定下一个注视位置

```python
class PolicyNetwork(nn.Module):
    """
    策略网络：根据当前隐状态和图像特征，决定下一个注视位置
    类似于AdaptiveNN的策略网络π
    """
    def __init__(self, z_dim=100, hidden_dim=128):
        super().__init__()
        # 输入：隐状态z_t + 全局特征s_coarse
        self.network = nn.Sequential(
            nn.Linear(z_dim + 50, hidden_dim),  # z_t + s_coarse
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 输出2D坐标 (y, x)
            nn.Sigmoid()  # 归一化到[0, 1]
        )
        
    def forward(self, z_t, s_coarse):
        """
        Args:
            z_t: (batch, z_dim) - 当前隐状态
            s_coarse: (batch, 50) - 粗粒度全局特征
        
        Returns:
            next_gaze_2d: (batch, 2) - 下一个注视点的2D坐标
            log_prob: (batch,) - 策略的对数概率（用于强化学习）
        """
        # 结合隐状态和全局特征
        input_features = torch.cat([z_t, s_coarse], dim=1)
        next_gaze_2d = self.network(input_features)
        
        # 计算策略概率（用于强化学习）
        # 这里简化处理，实际可以使用更复杂的分布（如高斯混合）
        log_prob = torch.log(next_gaze_2d + 1e-8).sum(dim=1)
        
        return next_gaze_2d, log_prob
```

**设计要点**：
- 输入：当前隐状态 + 全局特征
- 输出：下一个注视位置的2D坐标
- 支持强化学习训练（输出概率分布）

---

### 4. 价值网络（V*）

**功能**：评估当前状态是否足够，决定是否继续观察

```python
class ValueNetwork(nn.Module):
    """
    价值网络：评估当前状态的信息量，决定是否继续注视
    类似于AdaptiveNN的价值网络V*
    """
    def __init__(self, z_dim=100, hidden_dim=128):
        super().__init__()
        # 输入：当前隐状态z_t + 累积特征
        self.network = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出标量值
            nn.Sigmoid()  # 归一化到[0, 1]，表示继续概率
        )
        
    def forward(self, z_t):
        """
        Args:
            z_t: (batch, z_dim) - 当前隐状态
        
        Returns:
            continue_prob: (batch, 1) - 继续观察的概率 [0, 1]
            value: (batch, 1) - 状态价值（用于强化学习）
        """
        continue_prob = self.network(z_t)
        value = continue_prob  # 简化：使用继续概率作为价值
        
        return continue_prob, value
```

**设计要点**：
- 输入：当前隐状态
- 输出：继续观察的概率（0-1）
- 可以结合注视次数、信息熵等额外特征

---

### 5. 特征融合模块

**功能**：融合粗粒度和细粒度特征

```python
class FeatureFusion(nn.Module):
    """
    特征融合：结合全局粗特征和局部细特征
    """
    def __init__(self, z_dim=100):
        super().__init__()
        # 自适应权重：根据注视次数调整粗细特征权重
        self.alpha_network = nn.Sequential(
            nn.Linear(1, 32),  # 输入：注视次数
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(z_dim, z_dim)
        
    def forward(self, s_coarse, s_fine, gaze_count):
        """
        Args:
            s_coarse: (batch, 50) - 粗粒度特征
            s_fine: (batch, 50) - 细粒度特征
            gaze_count: (batch, 1) - 当前注视次数
        
        Returns:
            fused_features: (batch, 100) - 融合后的特征
        """
        # 计算自适应权重：前期更依赖粗特征，后期更依赖细特征
        alpha = self.alpha_network(gaze_count.float())  # (batch, 1)
        
        # 加权融合
        combined = torch.cat([
            (1 - alpha) * s_coarse,  # 粗特征权重
            alpha * s_fine           # 细特征权重
        ], dim=1)  # (batch, 100)
        
        # 进一步融合
        fused_features = self.fusion(combined)
        return fused_features
```

**设计要点**：
- 自适应权重：根据注视次数调整
- 前期更依赖全局特征，后期更依赖局部特征
- 实现从粗到细的渐进式感知

---

## 🔄 改进后的AdaptiveScanDMM模型

### 完整模型架构

```python
class AdaptiveScanDMM(DMM):
    """
    自适应扫描路径预测模型：结合AdaptiveNN的主动感知机制
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 新增组件
        self.glance = GlanceModule(z_dim=self.z_0.size(0))
        self.local_extractor = LocalFeatureExtractor(z_dim=self.z_0.size(0))
        self.policy_net = PolicyNetwork(z_dim=self.z_0.size(0))
        self.value_net = ValueNetwork(z_dim=self.z_0.size(0))
        self.feature_fusion = FeatureFusion(z_dim=self.z_0.size(0))
        
        # 替换原始CNN（保留用于兼容性）
        # self.cnn = ...  # 可以移除或保留
        
        # 超参数
        self.stop_threshold = 0.5  # 停止阈值（可在线调整）
        self.max_length = kwargs.get('max_length', 20)  # 最大长度
        
    def model(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths,
              images=None, annealing_factor=1.0, predict=False):
        """
        改进后的生成模型：支持自适应长度和动态特征提取
        """
        T_max = scanpaths.size(1)
        pyro.module("adaptive_dmm", self)
        
        # ===== 步骤1：Glance阶段 =====
        s_coarse = self.glance(images)  # (batch, 50) - 快速获取全局特征
        
        # ===== 步骤2：状态初始化 =====
        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))
        
        # ===== 步骤3：自适应注视循环 =====
        with pyro.plate("z_minibatch", len(scanpaths)):
            for t in pyro.markov(range(1, T_max + 1)):
                # ===== 3.1：价值网络评估是否继续 =====
                continue_prob, value = self.value_net(z_prev)
                
                # 决定是否继续（训练时使用真实长度，预测时使用价值网络）
                if predict:
                    should_continue = continue_prob > self.stop_threshold
                    # 如果所有样本都停止，提前退出
                    if not should_continue.any():
                        break
                else:
                    # 训练时：使用真实序列长度
                    should_continue = mask[:, t-1:t].squeeze(1) > 0
                
                # ===== 3.2：提取局部特征（如果继续） =====
                if should_continue.any():
                    # 获取上一个注视点的2D坐标
                    prev_gaze_3d = scanpaths[:, t-2, :] if t > 1 else scanpaths[:, 0, :]
                    prev_gaze_2d = xyz2plane(prev_gaze_3d)
                    
                    # 提取局部细粒度特征
                    s_fine = self.local_extractor(images, prev_gaze_2d)  # (batch, 50)
                    
                    # 融合粗细特征
                    gaze_count = torch.ones(scanpaths.size(0), 1) * t  # 当前注视次数
                    s_fused = self.feature_fusion(s_coarse, s_fine, gaze_count)  # (batch, 100)
                else:
                    # 如果停止，使用粗特征（填充）
                    s_fused = torch.cat([s_coarse, torch.zeros_like(s_coarse)], dim=1)
                
                # ===== 3.3：状态转移（使用融合特征） =====
                z_mu, z_sigma = self.trans(z_prev, s_fused)
                
                # 采样隐状态
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma)
                                      .mask(mask[:, t - 1: t] if not predict else should_continue.unsqueeze(1)).to_event(1))
                
                # ===== 3.4：生成眼动观测 =====
                x_mu, x_sigma = self.emitter(z_t)
                
                if not predict:
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                              .mask(mask[:, t - 1: t]).to_event(1), 
                              obs=scanpaths[:, t - 1, :])
                else:
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                              .mask(should_continue.unsqueeze(1)).to_event(1))
                
                # ===== 3.5：策略网络预测下一个注视位置（可选） =====
                # 注意：在训练时，我们使用真实的下一个位置
                # 在预测时，可以使用策略网络生成
                if predict and should_continue.any():
                    next_gaze_2d, log_prob = self.policy_net(z_t, s_coarse)
                    # 将2D坐标转换为3D坐标（用于下一轮）
                    # ... (坐标转换)
                
                z_prev = z_t
```

---

## 🎓 训练策略

### 1. 联合训练策略

**挑战**：需要同时训练：
- 变分推断部分（DMM的model和guide）
- 策略网络（强化学习）
- 价值网络（强化学习）

**解决方案**：分阶段训练

#### 阶段1：预训练基础模型
- 使用原始ScanDMM的训练方式
- 固定使用粗特征，训练基础DMM
- 目标：学习基本的扫描路径生成能力

#### 阶段2：训练特征提取
- 固定DMM参数
- 训练Glance模块和LocalFeatureExtractor
- 使用多尺度损失：同时优化粗粒度和细粒度预测

#### 阶段3：训练策略和价值网络
- 使用自奖励强化学习
- 将扫描路径预测的损失作为奖励信号
- 训练策略网络选择能降低损失的注视位置
- 训练价值网络评估信息量

#### 阶段4：端到端微调
- 联合训练所有组件
- 使用混合损失：ELBO + 策略损失 + 价值损失

### 2. 损失函数设计

```python
def compute_loss(model_output, ground_truth):
    """
    计算总损失
    """
    # 1. ELBO损失（原始DMM损失）
    elbo_loss = compute_elbo(...)
    
    # 2. 策略损失（强化学习）
    # 奖励 = -预测误差（误差越小，奖励越大）
    reward = -compute_prediction_error(model_output, ground_truth)
    policy_loss = -log_prob * reward  # REINFORCE算法
    
    # 3. 价值损失（TD误差）
    value_loss = compute_td_error(value_net_output, reward)
    
    # 4. 特征融合损失（可选）
    fusion_loss = compute_feature_fusion_loss(...)
    
    # 总损失
    total_loss = (
        elbo_loss + 
        lambda_policy * policy_loss + 
        lambda_value * value_loss +
        lambda_fusion * fusion_loss
    )
    
    return total_loss
```

### 3. 自奖励机制

**核心思想**：将任务损失作为奖励信号

```python
def compute_reward(predicted_path, ground_truth_path):
    """
    计算奖励：预测误差越小，奖励越大
    """
    # 计算预测误差（如DTW距离、Levenshtein距离等）
    error = compute_path_error(predicted_path, ground_truth_path)
    
    # 奖励 = -误差（归一化到合理范围）
    reward = -error / max_error
    
    return reward
```

---

## 📊 预期效果

### 效率提升

| 场景类型 | 原始ScanDMM | AdaptiveScanDMM | 提升倍数 |
|---------|------------|-----------------|---------|
| 简单场景 | 100% | 30-40% | 2.5-3.3x |
| 中等场景 | 100% | 50-60% | 1.7-2.0x |
| 复杂场景 | 100% | 70-80% | 1.25-1.43x |
| **平均** | **100%** | **50-60%** | **1.7-2.0x** |

**计算成本分析**：
- Glance阶段：~20%成本（粗特征提取）
- 每次注视：~5-10%成本（局部特征提取）
- 总成本 = 20% + N × 5-10%（N为注视次数）
- 简单场景N=2-3，复杂场景N=8-10

### 准确性影响

- **预期**：准确率保持或略有提升（1-2%）
- **原因**：
  - 粗特征提供全局上下文
  - 细特征提供局部细节
  - 自适应长度避免过度注视

### 可解释性提升

- ✅ 可视化Glance阶段的全局注意力
- ✅ 可视化每个注视点的局部区域
- ✅ 可视化价值网络的继续/停止决策
- ✅ 可视化策略网络的选择策略

---

## 🛠️ 实施路线图

### Phase 1：基础架构（2-3周）

1. **实现Glance模块**
   - 设计轻量级球面CNN
   - 实现降采样策略
   - 验证特征提取质量

2. **实现局部特征提取**
   - 设计RegionExtractor
   - 处理360度图像边界连续性
   - 实现局部区域提取

3. **实现特征融合**
   - 设计FeatureFusion模块
   - 实现自适应权重机制

### Phase 2：策略和价值网络（2-3周）

1. **实现策略网络**
   - 设计PolicyNetwork架构
   - 实现注视位置预测

2. **实现价值网络**
   - 设计ValueNetwork架构
   - 实现继续/停止决策

3. **集成到DMM**
   - 修改model()方法
   - 支持自适应长度序列

### Phase 3：训练策略（3-4周）

1. **实现分阶段训练**
   - 预训练基础模型
   - 训练特征提取
   - 训练策略和价值网络

2. **实现自奖励机制**
   - 设计奖励函数
   - 实现REINFORCE算法
   - 实现TD学习

3. **端到端微调**
   - 联合训练所有组件
   - 调优超参数

### Phase 4：评估和优化（2-3周）

1. **性能评估**
   - 效率指标（FLOPs、推理时间）
   - 准确性指标（DTW、LEV、REC）
   - 可解释性可视化

2. **消融实验**
   - 各组件贡献分析
   - 超参数敏感性分析

3. **对比实验**
   - 与原始ScanDMM对比
   - 与其他方法对比

---

## 🎯 关键技术挑战与解决方案

### 挑战1：360度图像的局部区域提取

**问题**：等距圆柱投影的边界连续性和球面几何特性

**解决方案**：
- 使用`F.grid_sample`进行双线性插值
- 考虑左右边界wrap-around
- 使用球面坐标转换

### 挑战2：可变长度序列训练

**问题**：Pyro的mask机制需要处理可变长度

**解决方案**：
- 使用mask标记有效时间步
- 在价值网络决定停止时，设置mask=0
- 确保梯度正确传播

### 挑战3：强化学习训练不稳定

**问题**：策略梯度方差大，训练不稳定

**解决方案**：
- 使用基线（baseline）减少方差
- 使用价值网络作为基线
- 实现PPO或A2C算法

### 挑战4：计算成本平衡

**问题**：局部特征提取可能增加计算成本

**解决方案**：
- 使用轻量级CNN提取局部特征
- 缓存粗特征，避免重复计算
- 批量处理多个注视点

---

## 🔮 未来扩展方向

### 1. 多模态融合
- 结合语言提示（如"找数字2和5"）
- 任务导向的注视策略
- 结合音频、文本等多模态信息

### 2. 在线适应
- 根据用户反馈在线调整策略
- 自适应学习最优注视次数
- 个性化注视模式

### 3. 跨任务应用
- 扩展到视频扫描路径预测
- 应用到显著性检测
- 应用到图像质量评估

### 4. 认知科学结合
- 建模工作记忆机制
- 结合注意力理论
- 模拟眼动生理机制

---

## 📝 总结

### 核心创新

1. **Glance阶段**：快速获取全局概览
2. **动态特征提取**：根据注视点提取局部特征
3. **自适应长度**：根据信息量决定是否继续
4. **策略网络**：学习最优注视策略
5. **价值网络**：评估信息量，决定终止

### 预期收益

- ✅ **效率提升**：计算成本降低40-50%
- ✅ **准确性保持**：准确率保持或略有提升
- ✅ **可解释性**：可视化注视策略
- ✅ **灵活性**：在线调整计算成本

### 实施建议

**优先实施顺序**：
1. Glance模块 + 局部特征提取（最大收益）
2. 特征融合机制（提升准确性）
3. 价值网络（自适应长度）
4. 策略网络（优化注视策略）

---

**文档创建日期**：2024年  
**最后更新**：2024年

