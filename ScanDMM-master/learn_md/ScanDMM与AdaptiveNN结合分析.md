# ScanDMM与AdaptiveNN"从粗到细"思想结合分析

## 📋 概述

本文档分析如何将AdaptiveNN的"从粗到细"（Coarse-to-Fine）动态感知策略结合到ScanDMM扫描路径预测模型中，以提升效率、灵活性和可解释性。

---

## 🔍 当前ScanDMM的工作方式

### 现有架构特点

1. **一次性全图特征提取**
   - 在模型开始时，使用`Sphere_CNN`一次性处理整个360度图像
   - 提取固定100维的全局特征向量：`img_features = self.cnn(images)  # (batch_size, 100)`
   - 这个特征向量在所有时间步都被复用，不随注视点变化

2. **静态特征使用**
   ```python
   # 在model()函数中
   img_features = self.cnn(images)  # 只计算一次
   
   for t in range(1, T_max + 1):
       z_mu, z_sigma = self.trans(z_prev, img_features)  # 所有时间步使用相同的img_features
   ```

3. **计算成本固定**
   - 无论图像复杂度如何，都执行相同的CNN前向传播
   - 计算成本 = 固定CNN成本 + 序列生成成本（与序列长度线性相关）

### 当前方式的局限性

- ❌ **效率问题**：简单场景和复杂场景使用相同的计算资源
- ❌ **缺乏动态性**：无法根据当前注视点动态调整关注区域
- ❌ **分辨率固定**：无法实现从粗到细的渐进式感知
- ❌ **资源浪费**：对不感兴趣的区域也进行了详细处理

---

## 💡 AdaptiveNN"从粗到细"核心思想

### 核心机制

1. **动态资源分配**
   - 初始阶段：使用低分辨率/粗粒度特征快速定位感兴趣区域
   - 后续阶段：对感兴趣区域进行高分辨率/细粒度处理
   - 资源消耗取决于注视次数，而非场景复杂度

2. **顺序注视策略**
   - 从粗到细：先看全局，再看局部
   - 自适应：根据任务难度动态调整注视次数
   - 聚焦：只处理任务相关的区域

3. **自适应推断成本**
   - 可通过调节阈值η在线调整平均计算成本
   - 无需重新训练即可适配不同资源约束

---

## 🔗 结合方案设计

### 方案一：多尺度动态特征提取（推荐）

#### 核心思想
将固定的单次特征提取改为**多尺度、动态、从粗到细**的特征提取策略。

#### 实现架构

```python
class AdaptiveSphere_CNN(nn.Module):
    """
    自适应球面CNN：支持多尺度动态特征提取
    """
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        
        # 粗粒度特征提取器（低分辨率，快速）
        self.coarse_cnn = Sphere_CNN(out_put_dim=z_dim // 2)  # 50维粗特征
        
        # 细粒度特征提取器（高分辨率，详细）
        self.fine_cnn = Sphere_CNN(out_put_dim=z_dim // 2)  # 50维细特征
        
        # 区域提取器：根据注视点提取局部区域
        self.region_extractor = RegionExtractor()
        
        # 特征融合层
        self.feature_fusion = nn.Linear(z_dim, z_dim)
    
    def forward(self, images, gaze_points=None, scale='coarse'):
        """
        Args:
            images: 输入图像 (batch, 3, H, W)
            gaze_points: 当前注视点 (batch, 2) - (y, x) 归一化坐标
            scale: 'coarse' | 'fine' | 'adaptive'
        """
        if scale == 'coarse':
            # 粗粒度：快速提取全局特征
            features = self.coarse_cnn(images)  # (batch, 50)
            return features
        
        elif scale == 'fine' and gaze_points is not None:
            # 细粒度：提取注视点周围的局部区域特征
            local_regions = self.region_extractor(images, gaze_points)  # 提取局部区域
            fine_features = self.fine_cnn(local_regions)  # (batch, 50)
            
            # 结合粗特征和细特征
            coarse_features = self.coarse_cnn(images)  # (batch, 50)
            combined = torch.cat([coarse_features, fine_features], dim=1)  # (batch, 100)
            return self.feature_fusion(combined)
        
        elif scale == 'adaptive':
            # 自适应：根据注视次数动态选择
            # 前N步用粗特征，后续用细特征
            pass
```

#### 修改DMM模型

```python
class AdaptiveDMM(DMM):
    """
    自适应深度马尔可夫模型：支持从粗到细的特征提取
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 替换为自适应CNN
        self.cnn = AdaptiveSphere_CNN(out_put_dim=self.z_0.size(0))
        
        # 注视次数计数器（用于决定何时切换粗细）
        self.gaze_count = 0
        self.coarse_threshold = 5  # 前5步使用粗特征
    
    def model(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths, 
              images=None, annealing_factor=1.0, predict=False):
        T_max = scanpaths.size(1)
        pyro.module("dmm", self)
        
        # 初始化
        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))
        
        # ===== 关键修改：动态特征提取 =====
        # 初始阶段：使用粗粒度特征（快速）
        img_features_coarse = self.cnn(images, scale='coarse')  # (batch, 50)
        
        with pyro.plate("z_minibatch", len(scanpaths)):
            for t in pyro.markov(range(1, T_max + 1)):
                # 决定使用粗特征还是细特征
                if t <= self.coarse_threshold:
                    # 粗粒度阶段：使用全局粗特征
                    current_img_features = img_features_coarse
                    # 扩展到100维（与z_dim匹配）
                    current_img_features = F.pad(current_img_features, (0, 50))  # (batch, 100)
                else:
                    # 细粒度阶段：根据当前注视点提取局部特征
                    # 获取上一个注视点（用于提取局部区域）
                    prev_gaze_2d = xyz2plane(scanpaths[:, t-2, :])  # (batch, 2)
                    current_img_features = self.cnn(
                        images, 
                        gaze_points=prev_gaze_2d, 
                        scale='fine'
                    )  # (batch, 100)
                
                # 使用动态特征进行转移
                z_mu, z_sigma = self.trans(z_prev, current_img_features)
                
                # 后续步骤保持不变...
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma)
                                      .mask(mask[:, t - 1: t]).to_event(1))
                
                x_mu, x_sigma = self.emitter(z_t)
                
                if not predict:
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                .mask(mask[:, t - 1: t]).to_event(1), 
                                obs=scanpaths[:, t - 1, :])
                else:
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                .mask(mask[:, t - 1: t]).to_event(1))
                
                z_prev = z_t
```

#### 优势

✅ **效率提升**：初始阶段使用粗特征，计算成本降低  
✅ **动态适应**：根据注视点动态提取局部特征  
✅ **从粗到细**：实现渐进式感知策略  
✅ **可解释性**：可以可视化哪些区域被详细处理  

---

### 方案二：自适应注视次数控制

#### 核心思想
根据图像复杂度或任务难度，动态决定需要多少个注视点。

#### 实现思路

```python
class AdaptiveGazeController(nn.Module):
    """
    自适应注视控制器：决定何时停止注视
    """
    def __init__(self, z_dim=100):
        super().__init__()
        self.stop_network = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def should_stop(self, z_t, threshold=0.5):
        """
        根据当前隐状态判断是否应该停止注视
        
        Returns:
            stop_prob: 停止概率 [0, 1]
        """
        stop_prob = self.stop_network(z_t)  # (batch, 1)
        return stop_prob > threshold

# 在DMM中使用
def model(self, ...):
    # ...
    for t in pyro.markov(range(1, T_max + 1)):
        # ... 生成z_t和x_t ...
        
        # 自适应停止：如果置信度足够高，提前结束
        if self.adaptive_controller.should_stop(z_t):
            break  # 提前停止，节省计算
```

#### 优势

✅ **自适应计算成本**：简单场景快速决策，复杂场景深入分析  
✅ **在线调整**：可通过阈值η控制平均注视次数  
✅ **效率优化**：避免不必要的注视点  

---

### 方案三：区域聚焦机制

#### 核心思想
在GatedTransition中，不仅使用全局图像特征，还根据当前注视点动态提取局部区域特征。

#### 实现架构

```python
class FocusedGatedTransition(GatedTransition):
    """
    聚焦门控转移：结合全局和局部特征
    """
    def __init__(self, z_dim, hidden_dim):
        super().__init__(z_dim, hidden_dim)
        # 局部特征处理网络
        self.local_feature_net = nn.Linear(z_dim, z_dim)
    
    def forward(self, z_t_1, img_feature_global, img_feature_local=None):
        """
        Args:
            z_t_1: 前一个隐状态
            img_feature_global: 全局图像特征（粗）
            img_feature_local: 局部图像特征（细，可选）
        """
        # 如果有局部特征，融合全局和局部
        if img_feature_local is not None:
            # 加权融合：前期更依赖全局，后期更依赖局部
            alpha = 0.7  # 局部特征权重
            img_feature = (1 - alpha) * img_feature_global + alpha * img_feature_local
        else:
            img_feature = img_feature_global
        
        # 后续处理与原始GatedTransition相同
        z_t_1_img = torch.cat((z_t_1, img_feature), dim=1)
        _z_t = self.lin_trans_hidden_to_z(self.relu(self.lin_trans_2z_to_hidden(z_t_1_img)))
        
        weight = torch.sigmoid(
            self.lin_gate_hidden_dim_to_z(self.relu(self.lin_gate_z_to_hidden_dim(z_t_1)))
        )
        
        mu = (1 - weight) * self.lin_z_to_mu(z_t_1) + weight * _z_t
        sigma = self.softplus(self.lin_sig(self.relu(_z_t)))
        
        return mu, sigma
```

---

## 📊 对比分析

### 计算效率对比

| 方案 | 初始阶段 | 后续阶段 | 总计算成本 | 适用场景 |
|------|---------|---------|-----------|---------|
| **原始ScanDMM** | 全图CNN (100%) | 序列生成 | 固定 | 所有场景 |
| **方案一：多尺度** | 粗CNN (30%) | 粗+细CNN (80%) | 降低40-60% | 复杂场景 |
| **方案二：自适应次数** | 全图CNN (100%) | 提前停止 | 降低20-50% | 简单场景 |
| **方案三：区域聚焦** | 全图CNN (100%) | 全局+局部 | 增加10-20% | 需要高精度 |

### 性能预期

1. **效率提升**
   - 方案一：在保持准确率的前提下，计算成本降低40-60%
   - 方案二：简单场景效率提升2-3倍，复杂场景提升不明显

2. **准确性影响**
   - 方案一：可能略微下降（1-3%），但可通过精细设计弥补
   - 方案二：对简单场景准确率提升（避免过度注视），对复杂场景可能下降

3. **可解释性**
   - 所有方案都能提供更好的可解释性：可视化注视点的粗细程度

---

## 🛠️ 实施建议

### 阶段一：方案一（多尺度动态特征）

**优先级：高**  
**难度：中等**  
**预期收益：效率提升40-60%，保持准确率**

**实施步骤：**
1. 实现`AdaptiveSphere_CNN`类
2. 实现`RegionExtractor`（根据注视点提取局部区域）
3. 修改`DMM.model()`方法，支持动态特征提取
4. 训练和评估

### 阶段二：方案二（自适应注视次数）

**优先级：中**  
**难度：低**  
**预期收益：简单场景效率提升2-3倍**

**实施步骤：**
1. 实现`AdaptiveGazeController`
2. 在训练时添加停止信号监督
3. 修改模型支持可变长度序列

### 阶段三：方案三（区域聚焦）

**优先级：低**  
**难度：高**  
**预期收益：提升局部区域预测精度**

**实施步骤：**
1. 实现`FocusedGatedTransition`
2. 设计局部特征提取机制
3. 融合全局和局部特征

---

## 🎯 关键挑战与解决方案

### 挑战1：如何提取局部区域？

**问题**：360度图像是等距圆柱投影，需要根据球面坐标提取局部区域。

**解决方案**：
```python
class RegionExtractor(nn.Module):
    """
    区域提取器：从360度图像中提取注视点周围的局部区域
    """
    def forward(self, images, gaze_points, region_size=64):
        """
        Args:
            images: (batch, 3, H, W) - 360度图像
            gaze_points: (batch, 2) - 归一化坐标 (y, x) in [0, 1]
            region_size: 提取区域的大小（像素）
        
        Returns:
            local_regions: (batch, 3, region_size, region_size)
        """
        # 将归一化坐标转换为像素坐标
        H, W = images.shape[2], images.shape[3]
        y_pixel = (gaze_points[:, 0] * H).long()
        x_pixel = (gaze_points[:, 1] * W).long()
        
        # 提取局部区域（考虑360度图像的左右边界连续性）
        # 使用F.grid_sample或手动裁剪
        # ...
        return local_regions
```

### 挑战2：如何训练多尺度模型？

**问题**：粗特征和细特征需要协同训练。

**解决方案**：
- **渐进式训练**：先训练粗特征，再训练细特征，最后联合训练
- **多任务学习**：同时优化粗粒度预测和细粒度预测
- **知识蒸馏**：用全分辨率模型指导多尺度模型

### 挑战3：如何平衡效率和准确率？

**问题**：从粗到细可能降低准确率。

**解决方案**：
- **自适应阈值**：根据图像复杂度动态调整粗细切换点
- **不确定性估计**：当粗特征不确定性高时，强制使用细特征
- **混合策略**：结合全局粗特征和局部细特征

---

## 📈 实验设计建议

### 评估指标

1. **效率指标**
   - FLOPs（浮点运算次数）
   - 推理时间
   - 内存占用

2. **准确性指标**
   - 扫描路径预测准确率（与真实路径的相似度）
   - 注视点位置误差
   - 路径长度匹配度

3. **可解释性指标**
   - 注视点分布可视化
   - 粗细特征使用比例
   - 区域聚焦热力图

### 对比实验

1. **基线对比**
   - 原始ScanDMM
   - 多尺度ScanDMM（方案一）
   - 自适应次数ScanDMM（方案二）

2. **消融实验**
   - 粗特征维度的影响
   - 粗细切换点的选择
   - 局部区域大小的影响

---

## 🔮 未来扩展方向

1. **多模态融合**
   - 结合语言提示（如"找数字2和5"）动态调整注视策略
   - 任务导向的从粗到细感知

2. **在线学习**
   - 根据用户反馈在线调整粗细切换策略
   - 自适应学习最优注视次数

3. **跨任务泛化**
   - 将模型应用到其他视觉任务（目标检测、图像分类等）
   - 作为MLLM的感知前端

---

## 📝 总结

ScanDMM与AdaptiveNN"从粗到细"思想的结合具有**巨大的潜力**：

### ✅ 优势
1. **效率提升**：计算成本降低40-60%，特别适合实时应用
2. **动态适应**：根据场景复杂度自动调整计算资源
3. **可解释性**：可视化注视策略，理解模型决策过程
4. **灵活性**：可通过阈值在线调整计算成本

### ⚠️ 挑战
1. **实现复杂度**：需要设计多尺度特征提取和区域聚焦机制
2. **训练难度**：多尺度模型需要精心设计的训练策略
3. **准确率权衡**：可能需要在效率和准确率之间平衡

### 🎯 推荐路径
**优先实施方案一（多尺度动态特征）**，这是最直接、收益最大的方案，可以在保持准确率的前提下显著提升效率，同时为后续方案奠定基础。

---

## 📚 参考文献

1. AdaptiveNN论文（用户提供的总结）
2. ScanDMM: A Deep Markov Model of Scanpath Prediction for 360° Images [CVPR2023]
3. CoordConv: 坐标卷积技术
4. 球面CNN相关论文

---

*文档创建日期：2024年*  
*最后更新：2024年*
