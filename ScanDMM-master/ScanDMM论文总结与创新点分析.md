# ScanDMM论文总结与创新点分析

## 一、论文基本信息

**论文标题**：ScanDMM: A Deep Markov Model of Scanpath Prediction for 360° Images

**作者**：Xiangjie Sui¹, Yuming Fang¹*, Hanwei Zhu², Shiqi Wang², Zhou Wang³  
¹ 江西财经大学，² 香港城市大学，³ 滑铁卢大学

**会议**：CVPR 2023

**项目地址**：https://github.com/xiangjieSui/ScanDMM

---

## 二、研究背景与问题

### 2.1 研究背景

360度图像（全向图像、球面图像、VR图像）在虚拟现实、沉浸式体验等应用中越来越重要。理解人类如何在360度图像中探索虚拟环境，对于VR渲染、显示、压缩和传输具有重要意义。

### 2.2 核心问题

**现有方法的局限性**：
1. **时间依赖性处理不完整**：大多数方法没有完整地处理观看行为的时间依赖性
2. **确定性模型过拟合**：RNN等确定性模型容易过拟合，特别是在小型360度数据库上
3. **简化假设不合理**：将显著性图直接连接到隐藏状态，忽略了场景语义和历史信息的重要性
4. **起始点影响被忽略**：观看的起始点对扫描路径有重要影响，但现有方法未充分考虑

### 2.3 神经科学启发

根据神经科学研究：
- 除了自下而上和自上而下的特征外，**历史信息和场景语义**是引导视觉注意力的重要来源
- 要被识别为兴趣或被拒绝为干扰物，项目必须与**记忆中保存的目标模板**进行比较
- 人类在360度场景中的扫描路径是**复杂的非线性动态注意力景观**，作为场景语义对视觉工作记忆干预的函数

---

## 三、方法概述

### 3.1 核心思想

ScanDMM采用**深度马尔可夫模型（DMM）**框架，通过概率方法学习编码时间依赖注意力景观的视觉状态，建模这些状态如何在场景语义和视觉工作记忆的指导下演化。

### 3.2 生成过程

ScanDMM使用以下生成过程预测扫描路径：

**状态转移**：
$$\mathbf{z}_t \sim p_{\theta_t}(\mathbf{z}_t | \mathbf{z}_{t-1}) $$

**眼动生成**：
$$\widetilde{\mathbf{x}}_t \sim p_{\theta_e}(\mathbf{x}_t | \mathbf{z}_t)$$

其中：
- $\mathbf{z}_t$：视觉状态（编码动态注意力景观）
- $\mathbf{x}_t$：注视点（三维坐标）
- $p_{\theta_t}$：转移概率（状态演化）
- $p_{\theta_e}$：发射概率（从状态生成注视点）

### 3.3 训练目标（ELBO）

$$\mathcal{L}(\theta; \phi; \mathbf{x}) = \underbrace{\mathbb{E}_{q_{\phi}} [\log p_{\theta_{e}}(\mathbf{x}|\mathbf{z})]}_{\text{重构项}} - \underbrace{\text{KL}(q_{\phi}(\mathbf{z}|\mathbf{x})||p_{\theta_{t}}(\mathbf{z}))}_{\text{正则化项}}$$

- **重构项**：评估模型的准确性，让预测的眼动位置接近真实值
- **正则化项**：强制变分分布接近先验分布

---

## 四、核心创新点

### 4.1 创新点1：语义引导的转移函数

**问题**：如何建模场景语义对视觉工作记忆的干预？

**解决方案**：
- 使用**SphereCNN**提取场景语义特征 $\hat{\mathbf{s}}$
- 使用**CoordConv**让卷积访问坐标信息
- 设计**门控机制**（不确定性加权）自适应地确定要更新多少先前视觉状态

**数学表达**：
$$\hat{\mathbf{z}}_t = \mathbf{W}_z^t(\mathbf{z}_{t-1} \oplus \hat{\mathbf{s}}) + \mathbf{b}_z^t$$
$$\alpha_t = \sigma(\mathbf{W}_{\alpha}^t \mathbf{z}_{t-1} + \mathbf{b}_{\alpha}^t)$$
$$\mu_t^z = \alpha_t \hat{\mathbf{z}}_t + (1 - \alpha_t) \mathbf{z}_{t-1}$$

**代码实现**：
```30:56:models.py
class GatedTransition(nn.Module):
    """ p(z_t | z_{t-1}) """

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.lin_gate_z_to_hidden_dim = nn.Linear(z_dim, hidden_dim)
        self.lin_gate_hidden_dim_to_z = nn.Linear(hidden_dim, z_dim)
        self.lin_trans_2z_to_hidden = nn.Linear(2 * z_dim, hidden_dim)
        self.lin_trans_hidden_to_z = nn.Linear(hidden_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_mu = nn.Linear(z_dim, z_dim)
        self.lin_z_to_mu.weight.data = torch.eye(z_dim)
        self.lin_z_to_mu.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, img_feature=None):
        """ Compute _z_t """
        z_t_1_img = torch.cat((z_t_1, img_feature), dim=1)
        _z_t = self.lin_trans_hidden_to_z(self.relu(self.lin_trans_2z_to_hidden(z_t_1_img)))

        ' Uncertainty weighting '
        weight = torch.sigmoid(self.lin_gate_hidden_dim_to_z(self.relu(self.lin_gate_z_to_hidden_dim(z_t_1))))
        ' Gaussian parameters '
        mu = (1 - weight) * self.lin_z_to_mu(z_t_1) + weight * _z_t
        sigma = self.softplus(self.lin_sig(self.relu(_z_t)))
        return mu, sigma
```

### 4.2 创新点2：状态初始化策略

**问题**：如何从正确的"启动器"学习状态的动态？

**解决方案**：
- 考虑观看的起始点 $\mathbf{x}_1$
- 使用起始点初始化初始状态 $\mathbf{z}_0$，而不是简单的零向量或随机向量

**数学表达**：
$$\mathbf{z}_0 = \mathbf{F}(\hat{\mathbf{z}}_0, \mathbf{x}_1)$$

其中 $\hat{\mathbf{z}}_0$ 是可学习参数，$\mathbf{F}$ 是线性神经网络。

**代码实现**：
```125:129:models.py
        # state initialization
        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))
```

**优势**：
- 使模型能够为扫描路径生成分配特定的起始点
- 在图像质量评估等任务中灵活且至关重要
- 让模型专注于学习动态，而不是随机起点

### 4.3 创新点3：球面卷积特征提取

**问题**：360度图像在等距圆柱投影下有几何扭曲，普通CNN无法正确处理。

**解决方案**：
- **CoordConv**：添加坐标信息（x, y坐标通道）
- **SphereConv2D**：考虑球面几何，计算正确的采样位置
- **坐标变换**：平面坐标 ↔ 球面坐标（经纬度）

**代码实现**：
```154:215:sphere_cnn.py
class Sphere_CNN(nn.Module):
    def __init__(self, out_put_dim):
        super(Sphere_CNN, self).__init__()
        self.output_dim = out_put_dim
        self.coord_conv = AddCoordsTh(x_dim=128, y_dim=256, with_r=False)

        # Image pipeline
        self.image_conv1 = SphereConv2D(5, 64, stride=2, bias=False)
        self.image_norm1 = nn.BatchNorm2d(64)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv2 = SphereConv2D(64, 128, stride=2, bias=False)
        self.image_norm2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv3 = SphereConv2D(128, 256, stride=2, bias=False)
        self.image_norm3 = nn.BatchNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv3_5 = SphereConv2D(256, 512, stride=2, bias=False)
        self.image_norm3_5 = nn.BatchNorm2d(512)
        self.leaky_relu3_5 = nn.LeakyReLU(0.2, inplace=True)

        # Joint pipeline

        self.image_conv4 = nn.Conv2d(512, 256, 4, 2, 1, bias=False)
        self.image_norm4 = nn.BatchNorm2d(256)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv5 = nn.Conv2d(256, 64, 4, 2, 1, bias=False)
        self.image_norm5 = nn.BatchNorm2d(64)
        self.leaky_relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.fc1 = nn.Linear(64 * 4 * 2, self.output_dim)
        self.flatten = nn.Flatten()
        self.activation = nn.Tanh()

    def forward(self, image):
        x = image

        x = self.coord_conv(x)

        x = self.leaky_relu1(self.image_norm1(self.image_conv1(x)))

        x = self.leaky_relu2(self.image_norm2(self.image_conv2(x)))

        x = self.leaky_relu3(self.image_norm3(self.image_conv3(x)))

        x = self.leaky_relu3_5(self.image_norm3_5(self.image_conv3_5(x)))

        # y = torch.reshape(y, (batch_size, 2, 8, 16))
        #
        # # Joint operations
        # x = torch.cat((y, x), dim=1)
        #
        x = self.leaky_relu4(self.image_norm4(self.image_conv4(x)))

        x = self.leaky_relu5(self.image_norm5(self.image_conv5(x)))

        x = self.activation(self.fc1(self.flatten(x)))

        return x
```

### 4.4 创新点4：变分推理框架

**问题**：真实后验 $p(\mathbf{z}|\mathbf{x})$ 难以直接计算。

**解决方案**：
- 使用变分推理，用可学习的分布 $q_{\phi}(\mathbf{z}|\mathbf{x})$ 近似真实后验
- 使用RNN编码观测序列，得到对隐状态的猜测
- 通过ELBO损失函数优化模型

**代码实现**：
```156:188:models.py
    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths, images=None, annealing_factor=1.0):

        T_max = scanpaths.size(1)
        pyro.module("dmm", self)

        h_0_contig = self.h_0.expand(1, scanpaths.size(0), self.rnn.hidden_size).contiguous()

        rnn_output, _ = self.rnn(scanpaths_reversed, h_0_contig)
        rnn_output = poly.pad_and_reverse(rnn_output, scanpath_lengths)

        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))

        with pyro.plate("z_minibatch", len(scanpaths)):
            for t in pyro.markov(range(1, T_max + 1)):

                # assemble the distribution q(z_t | z_{t-1}, x_{t:T})

                z_mu, z_sigma = self.combiner(z_prev, rnn_output[:, t - 1, :])

                z_dist = dist.Normal(z_mu, z_sigma)
                assert z_dist.event_shape == ()
                assert z_dist.batch_shape[-2:] == (len(scanpaths), self.z_0.size(0))

                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        z_dist.mask(mask[:, t - 1: t]).to_event(1),
                    )
                z_prev = z_t
```

---

## 五、实验验证

### 5.1 数据集

- **Sitzmann**：22张图像，1,980条扫描路径
- **Salient360!**：85张图像，3,036条扫描路径
- **AOI**：600张图像，18,000条扫描路径
- **JUFE**：1,032张图像，30,960条扫描路径

### 5.2 评估指标

- **LEV**（Levenshtein距离）：越小越好
- **DTW**（动态时间规整）：越小越好
- **REC**（重复性度量）：越大越好

### 5.3 主要结果

1. **准确性**：在四个数据库上达到最先进的性能
2. **效率**：模型大小18.7MB（约为SaltiNet的1/5），推理速度0.737秒/1000条路径
3. **泛化能力**：成功应用于显著性检测和图像质量评估任务

### 5.4 消融实验

- **ScanDMM_(I)**：不用起始点初始化 → 性能下降
- **ScanDMM_(S)**：不用场景语义 → 性能下降
- **ScanDMM_(IS)**：两者都不用 → 性能最差

证明了两个创新的必要性。

---

## 六、可能的创新点

基于对ScanDMM论文的深入理解，以下是可能的改进方向和创新点：

### 6.1 注意力机制增强特征提取

**方向**：在SphereCNN中引入注意力机制

**具体方案**：
- **CoordAttention**：同时关注通道和空间位置信息，与现有的CoordConv形成互补
- **CBAM**：结合通道注意力和空间注意力
- **ECA-Net**：高效的通道注意力机制

**优势**：
- 增强对360度图像坐标信息的利用
- 提升特征提取能力
- 与现有的CoordConv形成互补

**实现位置**：
```python
# 在Sphere_CNN的卷积层后添加注意力模块
class Sphere_CNN(nn.Module):
    def __init__(self, out_put_dim):
        # ... 现有代码 ...
        self.coord_att = CoordAtt(inp=512, oup=512)  # 在conv3_5后添加
    
    def forward(self, image):
        # ... 现有代码 ...
        x = self.leaky_relu3_5(self.image_norm3_5(self.image_conv3_5(x)))
        x = self.coord_att(x)  # 添加坐标注意力
        # ... 后续代码 ...
```

### 6.2 多尺度特征融合

**方向**：融合不同尺度的特征

**具体方案**：
- **MSFblock**：多尺度融合模块
- **MSTF**：多尺度时序融合
- **FPN**：特征金字塔网络

**优势**：
- 捕获不同尺度的语义信息
- 提升对复杂场景的理解能力

### 6.3 改进的状态转移机制

**方向**：增强门控机制和状态转移

**具体方案**：
- **多头注意力**：在状态转移中引入注意力机制
- **残差连接**：改进状态转移的梯度流动
- **自适应门控**：更复杂的门控机制

**优势**：
- 更好地建模状态之间的依赖关系
- 提升模型的表达能力

### 6.4 改进的变分推理方法

**方向**：使用更先进的序列建模方法

**具体方案**：
- **Mamba**：状态空间模型，在长序列建模上比RNN更有效
- **Transformer**：自注意力机制
- **LSTM/GRU**：替代简单的RNN

**优势**：
- 更好地处理长序列
- 提升变分推理的准确性

**实现位置**：
```python
# 在guide()函数中替换RNN
class DMM(nn.Module):
    def __init__(self, ...):
        # ... 现有代码 ...
        self.mamba = Mamba(...)  # 替换RNN
    
    def guide(self, ...):
        # ... 现有代码 ...
        rnn_output, _ = self.mamba(scanpaths_reversed, h_0_contig)  # 使用Mamba
        # ... 后续代码 ...
```

### 6.5 多模态信息融合

**方向**：结合其他模态信息

**具体方案**：
- **音频信息**：如果360度视频有音频，可以结合音频特征
- **深度信息**：如果有深度图，可以结合深度信息
- **语义分割**：结合语义分割结果

**优势**：
- 更丰富的特征表示
- 提升预测准确性

### 6.6 改进的损失函数

**方向**：设计更适合360度图像的损失函数

**具体方案**：
- **球面距离损失**：考虑球面几何的距离度量
- **时序一致性损失**：保证生成的扫描路径在时序上的一致性
- **多样性损失**：鼓励生成多样化的扫描路径

**优势**：
- 更好地适应360度图像的特性
- 提升生成质量

### 6.7 自适应状态维度

**方向**：根据场景复杂度自适应调整状态维度

**具体方案**：
- **动态状态维度**：简单场景用较小维度，复杂场景用较大维度
- **注意力机制**：动态选择重要的状态维度

**优势**：
- 提高计算效率
- 适应不同复杂度的场景

### 6.8 对比学习

**方向**：引入对比学习提升特征表示

**具体方案**：
- **SimCLR**：自监督对比学习
- **MoCo**：动量对比学习
- **CLIP**：图文对比学习

**优势**：
- 学习更好的特征表示
- 提升模型的泛化能力

### 6.9 知识蒸馏

**方向**：使用更大的模型作为教师模型

**具体方案**：
- **教师-学生框架**：用大模型指导小模型
- **特征蒸馏**：在特征层面进行知识传递

**优势**：
- 提升小模型的性能
- 保持模型的效率

### 6.10 元学习

**方向**：快速适应新场景

**具体方案**：
- **MAML**：模型无关的元学习
- **Few-shot学习**：少样本学习

**优势**：
- 快速适应新场景
- 提升模型的泛化能力

---

## 七、推荐优先级

### 高优先级（容易实现，效果明显）

1. **注意力机制增强特征提取**（CoordAttention、CBAM）
   - 实现简单，与现有架构兼容
   - 可能带来明显提升

2. **改进的变分推理方法**（Mamba、LSTM）
   - 替换RNN，提升长序列建模能力
   - 实现相对简单

### 中优先级（需要一定工作量）

3. **多尺度特征融合**（MSFblock）
   - 需要设计融合策略
   - 可能带来性能提升

4. **改进的损失函数**（球面距离损失）
   - 需要设计新的损失函数
   - 可能提升生成质量

### 低优先级（需要大量工作，但潜力大）

5. **多模态信息融合**
   - 需要额外的数据和特征提取
   - 可能带来显著提升

6. **元学习/对比学习**
   - 需要大量实验和调参
   - 可能带来突破性提升

---

## 八、总结

ScanDMM是一个优秀的360度图像扫描路径预测模型，主要创新点包括：
1. 语义引导的转移函数
2. 状态初始化策略
3. 球面卷积特征提取
4. 变分推理框架

**优势**：
- 在四个数据库上达到最先进的性能
- 模型小、速度快
- 具有良好的泛化能力

**可能的改进方向**：
- 注意力机制增强特征提取
- 改进的序列建模方法
- 多尺度特征融合
- 改进的损失函数

这些改进方向都有潜力进一步提升模型的性能，可以根据实际需求和资源选择合适的方向进行深入研究。


