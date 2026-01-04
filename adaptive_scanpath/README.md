# AdaptiveScanPath - 基于AdaptiveNN架构的眼动路径预测模型

基于AdaptiveNN架构的360度全景图像眼动扫描路径预测模型，采用自适应���视机制和提前停止策略。

## 🌟 核心特性

- **自适应注视机制**: 模拟人类视觉的"扫视-注视"过程
- **智能提前停止**: 根据场景复杂度自适应决定扫描长度
- **球面几何感知**: 专门处理360度全景图像的球面CNN
- **序列建模**: 可选GRU建模时序依赖关系
- **高效训练**: 端到端监督学习，训练稳定

## 📁 项目结构

```
adaptive_scanpath/
├── models/                    # 模型定义
│   ├── sphere_cnn.py         # 球面CNN特征提取器
│   ├── policy_network.py     # 策略网络和停止网络
│   └── adaptive_scanpath.py  # 完整模型
├── utils/                     # 工具函数
│   └── losses.py             # 损失函数和评估指标
├── data/                      # 数据加载
│   └── dataset.py            # 数据集类
├── train.py                   # 训练脚本
├── eval.py                    # 评估脚本
├── config.py                  # 配置文件
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 环境要求

```bash
# Python 3.8+
torch>=1.10.0
torchvision
numpy
matplotlib
opencv-python
tqdm
Pillow
# tensorboard (可选，用于可视化训练)
```

### 2. 准备数据 ⭐

**方案A: 使用合成数据（推荐新手）**
```bash
# 自动生成100个合成样本（1分钟内完成）
python prepare_data.py --num_samples 100
```

**方案B: 使用开源数据集**
- **Salient360!**: 360度图像眼动数据集（推荐）
  - 官网: https://salient360.di.fc.ul.pt/
  - 详细指南: 见 [DATA_GUIDE.md](DATA_GUIDE.md)
- **SALICON**: 大规模2D眼动数据集
  - 官网: http://salicon.net/

📖 **完整数据指南**: 查看 [DATA_GUIDE.md](DATA_GUIDE.md)

数据集格式：
```
data/
├── train/
│   ├── images/              # 全景图像文件夹
│   └── annotations.json     # 标注文件
└── val/
    ├── images/
    └── annotations.json
```

标注文件格式 (`annotations.json`):
```json
[
  {
    "image": "image_001.jpg",
    "scanpath": [
      [theta_1, phi_1, duration_1],
      [theta_2, phi_2, duration_2],
      ...
    ]
  },
  ...
]
```

**坐标说明**:
- `theta`: 水平角度，范围 [-π, π]
- `phi`: 垂直角度，范围 [-π/2, π/2]
- `duration`: 注视持续时间（秒），范围 [0, 3]

### 3. 一键开始 🎉

```bash
# 步骤1: 准备数据（合成数据，100个样本）
python prepare_data.py --num_samples 100

# 步骤2: 快速训练（5个epoch，约5分钟）
python quickstart.py

# 完成！模型保存在 checkpoints/quickstart_model.pth
```

**或者使用完整训练流程**：

```bash
# 完整训练（需要更多数据）
python train.py
```

---

### 4. 配置（可选）

如果需要自定义训练参数，编辑 `config.py`：

```python
# 数据配置
image_height = 256
image_width = 512
max_seq_len = 12

# 训练配置
batch_size = 16
num_epochs = 100
learning_rate = 1e-4

# 模型配置
feature_dim = 384
use_rnn = True
```

### 4. 训练

```bash
# 开始训练
python train.py

# 使用TensorBoard监控训练
tensorboard --logdir=logs
```

### 5. 评估

```bash
# 评估模型
python eval.py --checkpoint checkpoints/best_model.pth \
               --data_path data/val \
               --visualize \
               --export
```

## 🎯 模型架构

```
输入: 360度全景图像 (3, 256, 512)
    ↓
[球面CNN] → 全局特征 (384维)
    ↓
循环生成注视点序列:
    ├─ [上下文RNN] → 时序上下文 (256维)
    ├─ [策略网络] → 预测注视点 [theta, phi, duration]
    ├─ [停止网络] → 决策是否继续扫描
    └─ [特征更新] → 累积视觉信息
    ↓
输出: 扫描路径 (T, 3) + 停止位置
```

## 📊 损失函数

模型使用多任务损失：

```python
total_loss = w₁·coord_loss +          # 坐标回归损失 (MSE)
             w₂·duration_loss +       # 持续时间损失
             w₃·smoothness_loss +     # 平滑性约束
             w₄·stopping_loss         # 停止策略损失
```

默认权重：
- `w₁ = 1.0` (坐标)
- `w₂ = 0.1` (持续时间)
- `w₃ = 0.05` (平滑性)
- `w₄ = 0.2` (停止策略)

## 🔧 关键技术

### 1. 球面CNN (SphereCNN)

- **CoordConv**: 添加坐标信息，增强位置感知
- **球面卷积**: 正确处理360度图像的几何特性
- **多尺度特征**: 提取不同抽象层次的特征

### 2. 策略网络 (PolicyNetwork)

- **输入**: 图像特征 + 时序上下文
- **输出**: 注视点坐标 [theta, phi, duration]
- **激活**: tanh (坐标) + sigmoid (持续时间)

### 3. 停止网络 (StoppingNetwork)

- **输入**: 当前特征 + 最后一个注视点
- **输出**: 停止概率 [0, 1]
- **训练**: 二元交叉熵损失

### 4. 特征更新 (FeatureUpdater)

- **门控机制**: 自适应融合新旧特征
- **信息累积**: 模拟视觉信息的逐步积累

## 📈 训练技巧

### 1. 学习率调度

推荐使用余弦退火：
```python
scheduler_type = 'cosine'
scheduler_params = {'T_max': 100, 'eta_min': 1e-6}
```

### 2. Teacher Forcing

训练时使用50%的teacher forcing：
```python
pred_paths, stop_probs = model(images, gt_paths, teacher_forcing_ratio=0.5)
```

### 3. 梯度裁剪

防止梯度爆炸：
```python
max_grad_norm = 1.0
```

### 4. 早停

监控验证损失，防止过拟合：
```python
use_early_stopping = True
patience = 15
```

## 📝 评估指标

- **MSE**: 均方误差（坐标）
- **MAE**: 平均绝对误差（坐标）
- **RMSE**: 均方根误差
- **Length Accuracy**: 序列长度预测准确率（±1步）

## 🎨 可视化

模型支持多种可视化方式：

1. **路径可视化**: 在全景图上绘制预测和真实路径
2. **TensorBoard**: 实时监控训练曲线
3. **导出JSON**: 保存预测结果用于分析

## 🔍 模型对比

| 特性 | DMM | AdaptiveScanPath |
|------|-----|-----------------|
| 训练复杂度 | 高（变分推断） | **低**（监督学习） |
| 生成多样性 | 高 | 中等 |
| 训练稳定性 | 中等 | **高** |
| 推理速度 | 中等 | **快** |
| 可解释性 | 中等 | **高** |
| 自适应长度 | ❌ | **✅** |

## 💡 使用建议

### 训练阶段

1. **预训练**: 先训练特征提取器
2. **联合训练**: 端到端微调
3. **超参调优**: 调整损失权重

### 推理阶段

1. **单次预测**: 快速生成单个路径
2. **多样化采样**: 生成多个候选路径
3. **后处理**: 平滑路径、过滤异常点

## 🐛 常见问题

### Q1: 训练损失不下降？

- 检查数据是否正确加载
- 调整学习率（尝试更小或更大）
- 检查损失权重是否合理

### Q2: 预测路径长度固定？

- 确保停止网络训练正常
- 检查停止损失是否下降
- 调整停止阈值

### Q3: 显存不足？

- 减小batch_size
- 减小图像分辨率
- 减小模型特征维度

## 📞 联系方式

如有问题，请提交Issue或Pull Request。

## 📄 许可证

MIT License

## 🙏 致谢

- AdaptiveNN: 原始论文和代码
- ScanDMM: 360度图像处理技术

---

**Happy Training! 🚀**
