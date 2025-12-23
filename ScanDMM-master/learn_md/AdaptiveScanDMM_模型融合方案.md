# AdaptiveScanDMM + some_models 模型融合方案

## 📋 概述

本文档分析`some_models`文件夹下的模型，找出可以结合到AdaptiveScanDMM中的组件，甚至替换DMM的架构。

---

## 🎯 核心发现：可融合的模型分类

### 类别1：序列建模替代方案（可替换DMM）

#### 1.1 VMRNN (5_3_VMRNN.py) ⭐⭐⭐⭐⭐
**核心特点**：
- 结合Mamba/SSM和RNN的混合架构
- 使用VSSBlock（Visual State Space Block）进行空间-时间建模
- 类似LSTM的cell结构，维护隐藏状态(h_t, c_t)

**优势**：
- ✅ **高效**：Mamba的线性复杂度，比RNN更高效
- ✅ **长程依赖**：SSM擅长建模长序列依赖
- ✅ **空间感知**：VSSBlock考虑2D空间结构
- ✅ **状态维护**：类似DMM的隐状态机制

**融合方案**：
```python
# 替换DMM的GatedTransition和RNN部分
class AdaptiveScanVMRNN(AdaptiveScanDMM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用VMRNNCell替换RNN和GatedTransition
        self.vmrnn_cell = VMRNNCell(
            hidden_dim=z_dim,
            input_resolution=(H, W),  # 360度图像分辨率
            depth=1,
            d_state=16
        )
    
    def forward(self, ...):
        # 使用VMRNNCell进行状态转移
        h_t, (h_t, c_t) = self.vmrnn_cell(z_t, (h_{t-1}, c_{t-1}))
```

**适用场景**：
- 需要长程依赖建模
- 需要空间-时间联合建模
- 希望提升效率

---

#### 1.2 Mamba (5_1_Mamba.py) ⭐⭐⭐⭐
**核心特点**：
- 纯状态空间模型（SSM）
- 线性复杂度O(N)
- 选择性扫描机制

**优势**：
- ✅ **极高效率**：线性复杂度，比Transformer快
- ✅ **长序列**：擅长处理长序列
- ✅ **选择性**：可以选择性关注重要信息

**融合方案**：
```python
# 用Mamba替换DMM的序列建模部分
class AdaptiveScanMamba(AdaptiveScanDMM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用Mamba替换RNN
        self.mamba = Mamba(
            d_model=z_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
```

**适用场景**：
- 需要极高效率
- 长扫描路径预测
- 资源受限场景

---

#### 1.3 TimesNet (3_10_TimesNet.py) ⭐⭐⭐⭐
**核心特点**：
- 使用FFT提取时序周期性
- 多周期建模
- 2D卷积处理时序模式

**优势**：
- ✅ **周期性建模**：捕捉扫描路径的周期性模式
- ✅ **多尺度**：同时建模多个时间尺度
- ✅ **2D表示**：将1D时序转换为2D表示

**融合方案**：
```python
# 用于捕捉扫描路径的周期性
class PeriodicScanPathModel(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.times_block = TimesBlock(
            seq_len=20,  # 扫描路径长度
            pred_len=20,
            top_k=3,  # 提取3个主要周期
            d_model=z_dim
        )
    
    def forward(self, scanpath_sequence):
        # scanpath_sequence: (B, T, z_dim)
        periodic_features = self.times_block(scanpath_sequence)
        return periodic_features
```

**适用场景**：
- 需要捕捉周期性模式
- 多时间尺度建模
- 作为辅助模块增强DMM

---

### 类别2：注意力机制增强（用于特征提取和注视点选择）

#### 2.1 HiLo Attention (2_7_HiLoAttention.py) ⭐⭐⭐⭐⭐
**核心特点**：
- **高低频分离**：将注意力分为高频（Hi-Fi）和低频（Lo-Fi）
- **窗口注意力**：高频使用局部窗口，低频使用全局
- **效率优化**：低频使用池化降采样

**优势**：
- ✅ **完美契合"从粗到细"**：低频=粗粒度，高频=细粒度
- ✅ **效率高**：低频降采样，计算成本低
- ✅ **多尺度感知**：同时捕捉全局和局部信息

**融合方案**：
```python
# 用于Glance阶段的特征提取
class HiLoGlanceModule(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        # 使用HiLo Attention提取多尺度特征
        self.hilo_attn = HiLo(
            dim=z_dim,
            num_heads=8,
            window_size=2,  # 窗口大小
            alpha=0.5  # 低频/高频比例
        )
    
    def forward(self, images):
        # 提取粗粒度（低频）和细粒度（高频）特征
        features = self.hilo_attn(images)
        return features
```

**适用场景**：
- **Glance阶段**：快速提取多尺度特征
- **特征融合**：结合全局和局部信息
- **替换Sphere CNN**：更高效的特征提取

---

#### 2.2 Agent Attention (4_3_AgentAttention.py) ⭐⭐⭐⭐
**核心特点**：
- **代理机制**：使用少量代理token代表全局信息
- **高效注意力**：通过代理减少计算复杂度
- **空间感知**：考虑2D空间结构

**优势**：
- ✅ **高效**：通过代理减少注意力计算
- ✅ **全局感知**：代理token捕捉全局信息
- ✅ **适合注视点选择**：可以用于策略网络

**融合方案**：
```python
# 用于策略网络，选择下一个注视点
class AgentPolicyNetwork(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        # 使用Agent Attention选择注视位置
        self.agent_attn = AgentAttention(
            dim=z_dim,
            num_heads=8,
            agent_num=49,  # 代理数量
            window=14
        )
    
    def forward(self, z_t, s_coarse):
        # 使用代理注意力选择下一个注视点
        attention_map = self.agent_attn(z_t)
        next_gaze = self.extract_gaze_from_attention(attention_map)
        return next_gaze
```

**适用场景**：
- **策略网络**：选择下一个注视位置
- **全局感知**：捕捉全局注意力分布
- **效率优化**：减少注意力计算

---

#### 2.3 Sea Attention (2_4_SeaAttention.py) ⭐⭐⭐
**核心特点**：
- **轴向注意力**：分别处理行和列
- **位置编码**：显式的位置嵌入
- **高效**：分离的行列注意力

**融合方案**：
```python
# 用于360度图像的特征提取（考虑球面结构）
class SeaSphereAttention(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.sea_attn = Sea_Attention(
            dim=z_dim,
            key_dim=32,
            num_heads=8
        )
```

**适用场景**：
- 360度图像的特征提取
- 考虑经纬度方向的特征

---

### 类别3：特征融合模块

#### 3.1 Gated Fusion (6_1_Gated_Fusion.py) ⭐⭐⭐⭐⭐
**核心特点**：
- **门控机制**：自适应融合两个特征
- **简单高效**：轻量级融合模块
- **可学习权重**：通过sigmoid生成融合权重

**优势**：
- ✅ **简单有效**：轻量级，计算成本低
- ✅ **自适应**：根据输入自动调整融合权重
- ✅ **直接可用**：可以直接替换FeatureFusion模块

**融合方案**：
```python
# 直接替换FeatureFusion模块
class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        # 使用Gated Fusion融合粗细特征
        self.gated_fusion = gatedFusion(dim=z_dim // 2)  # 50维
    
    def forward(self, s_coarse, s_fine):
        # s_coarse: (batch, 50)
        # s_fine: (batch, 50)
        fused = self.gated_fusion(s_coarse, s_fine)  # (batch, 50)
        return fused
```

**适用场景**：
- **特征融合**：融合粗粒度和细粒度特征
- **多模态融合**：融合不同来源的特征
- **简单高效**：需要轻量级融合时

---

#### 3.2 MSFblock (6_2_MSFblock.py) ⭐⭐⭐⭐
**核心特点**：
- **多尺度融合**：融合4个不同尺度的特征
- **SE注意力**：使用Squeeze-and-Excitation机制
- **自适应权重**：Softmax生成多尺度权重

**优势**：
- ✅ **多尺度**：同时融合多个尺度
- ✅ **注意力机制**：SE注意力增强重要特征
- ✅ **灵活**：可以融合任意数量的尺度

**融合方案**：
```python
# 用于多尺度特征融合
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        # 融合4个不同尺度的特征
        self.msf = MSFblock(in_channels=z_dim // 4)
    
    def forward(self, s_coarse, s_fine_1, s_fine_2, s_fine_3):
        # 融合多个尺度的特征
        fused = self.msf(s_coarse, s_fine_1, s_fine_2, s_fine_3)
        return fused
```

**适用场景**：
- **多尺度融合**：需要融合多个尺度的特征
- **复杂场景**：复杂场景需要多尺度信息

---

### 类别4：时序建模增强

#### 4.1 Temporal Conv (3_5_Temporal_conv.py) ⭐⭐⭐⭐
**核心特点**：
- **膨胀卷积**：多尺度时序卷积
- **门控机制**：Filter和Gate分离
- **因果卷积**：保持时序因果性

**优势**：
- ✅ **多尺度**：不同膨胀率的卷积捕捉不同时间尺度
- ✅ **门控**：类似LSTM的门控机制
- ✅ **高效**：卷积比RNN更高效

**融合方案**：
```python
# 用于时序特征提取
class TemporalFeatureExtractor(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.temporal_conv = temporal_conv(
            cin=z_dim,
            cout=z_dim,
            dilation_factor=1,
            seq_len=20
        )
    
    def forward(self, scanpath_sequence):
        # scanpath_sequence: (B, C, N, T)
        temporal_features = self.temporal_conv(scanpath_sequence)
        return temporal_features
```

**适用场景**：
- **时序特征提取**：提取扫描路径的时序模式
- **多时间尺度**：捕捉不同时间尺度的依赖

---

## 🏗️ 完整融合架构设计

### 方案A：VMRNN + HiLo Attention + Gated Fusion（推荐）⭐⭐⭐⭐⭐

**架构**：
```
输入：360度图像 I
  ↓
[HiLo Glance] → 粗粒度特征 s_coarse (50维) + 细粒度特征 s_fine_init (50维)
  ↓
[Gated Fusion] → 初始融合特征 s_0 (50维)
  ↓
[状态初始化] → z_0 = F(z_0, x_1, s_0)
  ↓
循环注视（自适应长度）：
  [价值网络 V*] → 评估是否继续
    ↓ continue
  [局部HiLo Attention] → 提取局部细粒度特征 s_fine (50维)
  [Gated Fusion] → 融合 s_coarse + s_fine → s_fused (50维)
  [VMRNN Cell] → z_t = VMRNN(z_{t-1}, s_fused, (h_{t-1}, c_{t-1}))
  [Emitter] → x_t ~ p(x_t | z_t)
  [Agent Policy] → 下一个注视位置 x_{t+1}
    ↓ stop
  [输出扫描路径]
```

**优势**：
- ✅ **高效序列建模**：VMRNN比RNN更高效
- ✅ **多尺度感知**：HiLo Attention实现从粗到细
- ✅ **灵活融合**：Gated Fusion自适应融合特征
- ✅ **智能选择**：Agent Attention选择注视点

**代码框架**：
```python
class AdaptiveScanVMRNN_HiLo(AdaptiveScanDMM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. HiLo Glance模块
        self.hilo_glance = HiLoGlanceModule(z_dim=kwargs['z_dim'])
        
        # 2. VMRNN Cell（替换GatedTransition和RNN）
        self.vmrnn_cell = VMRNNCell(
            hidden_dim=kwargs['z_dim'],
            input_resolution=(128, 256),  # 360度图像分辨率
            depth=1,
            d_state=16
        )
        
        # 3. Gated Fusion（替换FeatureFusion）
        self.gated_fusion = gatedFusion(dim=kwargs['z_dim'] // 2)
        
        # 4. Agent Policy Network（替换PolicyNetwork）
        self.agent_policy = AgentPolicyNetwork(z_dim=kwargs['z_dim'])
    
    def model(self, ...):
        # 使用新组件进行扫描路径生成
        ...
```

---

### 方案B：Mamba + HiLo Attention + Gated Fusion ⭐⭐⭐⭐

**架构**：类似方案A，但用Mamba替换VMRNN

**优势**：
- ✅ **极高效率**：Mamba的线性复杂度
- ✅ **长序列**：擅长处理长扫描路径
- ✅ **选择性**：选择性关注重要信息

**适用场景**：
- 需要极高效率
- 长扫描路径预测
- 资源受限场景

---

### 方案C：TimesNet增强版 ⭐⭐⭐

**架构**：
```
输入：360度图像 I
  ↓
[Glance] → s_coarse
  ↓
循环注视：
  [TimesNet] → 提取周期性特征（捕捉扫描路径的周期性模式）
  [DMM/VMRNN] → 状态转移（结合周期性特征）
  [Emitter] → 生成注视点
```

**优势**：
- ✅ **周期性建模**：捕捉扫描路径的周期性
- ✅ **多时间尺度**：同时建模多个时间尺度

**适用场景**：
- 需要捕捉周期性模式
- 作为辅助模块增强主模型

---

## 📊 模型对比分析

| 模型 | 复杂度 | 效率 | 长程依赖 | 空间感知 | 适用场景 | 推荐度 |
|------|--------|------|----------|----------|----------|--------|
| **VMRNN** | O(N) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 通用，需要空间-时间建模 | ⭐⭐⭐⭐⭐ |
| **Mamba** | O(N) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 极高效率需求 | ⭐⭐⭐⭐ |
| **HiLo Attention** | O(N) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 从粗到细特征提取 | ⭐⭐⭐⭐⭐ |
| **Agent Attention** | O(N) | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 注视点选择 | ⭐⭐⭐⭐ |
| **Gated Fusion** | O(1) | ⭐⭐⭐⭐⭐ | - | - | 特征融合 | ⭐⭐⭐⭐⭐ |
| **TimesNet** | O(N log N) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 周期性建模 | ⭐⭐⭐ |

---

## 🎯 推荐融合方案

### 方案1：VMRNN + HiLo + Gated Fusion（最佳平衡）⭐⭐⭐⭐⭐

**组件**：
- **序列建模**：VMRNN（替换DMM）
- **特征提取**：HiLo Attention（替换Sphere CNN）
- **特征融合**：Gated Fusion（替换FeatureFusion）
- **策略网络**：Agent Attention（可选）

**优势**：
- ✅ 效率提升40-50%
- ✅ 准确性保持或提升
- ✅ 多尺度感知
- ✅ 空间-时间联合建模

**实施难度**：中等

---

### 方案2：Mamba + HiLo + Gated Fusion（最高效）⭐⭐⭐⭐

**组件**：
- **序列建模**：Mamba（替换DMM）
- **特征提取**：HiLo Attention
- **特征融合**：Gated Fusion

**优势**：
- ✅ 效率提升50-60%
- ✅ 线性复杂度
- ✅ 长序列建模

**实施难度**：中等

---

### 方案3：DMM + HiLo + Gated Fusion（最小改动）⭐⭐⭐⭐

**组件**：
- **序列建模**：保留DMM
- **特征提取**：HiLo Attention（替换Sphere CNN）
- **特征融合**：Gated Fusion

**优势**：
- ✅ 最小改动
- ✅ 快速实施
- ✅ 效率提升30-40%

**实施难度**：低

---

## 🛠️ 实施建议

### Phase 1：最小改动方案（1-2周）
1. 集成Gated Fusion替换FeatureFusion
2. 集成HiLo Attention到Glance模块
3. 测试和评估

### Phase 2：序列建模替换（2-3周）
1. 集成VMRNN或Mamba
2. 替换DMM的序列建模部分
3. 调整训练策略
4. 测试和评估

### Phase 3：完整优化（2-3周）
1. 集成Agent Attention到策略网络
2. 集成TimesNet作为辅助模块
3. 端到端优化
4. 全面评估

---

## 📝 总结

### 核心发现

1. **VMRNN**是最佳DMM替代方案：结合Mamba和RNN，高效且适合空间-时间建模
2. **HiLo Attention**完美契合"从粗到细"：低频=粗粒度，高频=细粒度
3. **Gated Fusion**简单高效：可以直接替换FeatureFusion模块
4. **Agent Attention**适合策略网络：可以用于选择下一个注视点

### 推荐路径

**优先实施**：
1. ✅ **Gated Fusion**（最简单，直接收益）
2. ✅ **HiLo Attention**（完美契合，效率提升大）
3. ✅ **VMRNN**（替换DMM，提升效率和性能）

**可选实施**：
- Agent Attention（策略网络优化）
- TimesNet（周期性建模）
- Mamba（极高效率需求）

---

**文档创建日期**：2024年  
**最后更新**：2024年

