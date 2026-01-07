# 数据准备指南

## 🎯 快速开始（推荐）

直接使用合成数据快速测试和训练：

```bash
# 1. 准备数据（100个样本，1分钟内完成）
python prepare_data.py --num_samples 100

# 2. 开始训练
python quickstart.py
```

**合成数据特点**：
- ✅ 无需下载，秒级生成
- ✅ 模拟真实眼动模式（中心偏差、聚集性）
- ✅ 包含多样化的场景（不同颜色和布局）
- ✅ 适合快速验证和原型开发

---

## 📚 开源数据集选项

### 选项1: Salient360! ⭐⭐⭐⭐⭐
**专门针对360度图像的眼动数据集**

**优点**：
- ✅ 360度全景图像
- ✅ 真实的人类眼动数据
- ✅ 多个人工标注
- ✅ 学术研究免费

**获取步骤**：
1. 访问官网：https://salient360.di.fc.ul.pt/
2. 注册账号（免费）
3. 下载数据集
4. 解压到 `data/salient360/`

**数据格式**：
```
data/salient360/
├── images/          # 全景图像
│   ├── img001.jpg
│   └── ...
└── annotations/     # 眼动标注
    ├── img001.json
    └── ...
```

---

### 选项2: SALICON ⭐⭐⭐⭐
**最常用的2D眼动数据集**

**优点**：
- ✅ 大规模（10,000张图像）
- ✅ 高质量标注
- ✅ 易于下载

**缺点**：
- ❌ 2D图像（非360度）
- ❌ 需要��配

**获取**：
- 官网：http://salicon.net/
- 下载链接：http://salicon.net/download/

---

### 选项3: 其他数据集

#### OSIE
- 700张高质量图像
- GitHub: https://github.com/NUS-VIP/osie

#### CAT2000
- 2000张图像
- 多种类别

---

## 🔄 数据格式转换

我们的项目使用以下JSON格式：

```json
[
  {
    "image": "panorama_0001.jpg",
    "scanpath": [
      [theta_1, phi_1, duration_1],
      [theta_2, phi_2, duration_2],
      ...
    ]
  },
  ...
]
```

**坐标说明**：
- `theta`: 水平角度，范围 [-π, π]
- `phi`: 垂直角度，范围 [-π/2, π/2]
- `duration`: 持续时间（秒），范围 [0, 3]

---

## 💡 推荐使用方案

### 阶段1: 快速验证（1-2天）
```bash
# 使用合成数据快速验证代码
python prepare_data.py --num_samples 100
python quickstart.py
```

### 阶段2: 小规模真实数据（1周）
- ���载Salient360!数据集（约40张图像）
- 在真实数据上微调模型
- 验证模型性能

### 阶段3: 大规模训练
- 使用SALICON等大数据集预训练
- 在360度数据上微调

---

## 🛠️ 自定义数据集

如果你想使用自己的数据：

### 1. 准备图像
- 格式：JPG或PNG
- 分辨率：建议至少 512x1024
- 投影：等距圆柱投影（equirectangular）

### 2. 准备标注
创建 `annotations.json`：
```json
[
  {
    "image": "your_image_001.jpg",
    "scanpath": [
      [0.5, 0.3, 0.5],
      [-0.8, 0.2, 0.3],
      [1.2, -0.4, 0.6]
    ]
  }
]
```

### 3. 组织目录
```
data/
└── your_dataset/
    ├── images/
    │   └── your_image_001.jpg
    └── annotations.json
```

### 4. 修改配置
```python
# config.py
train_data_path = 'data/your_dataset'
val_data_path = 'data/your_dataset_val'
```

---

## 📊 数据增强

代码已内置数据增强（在训练时自动启用）：

- ✅ 亮度调整
- ✅ 对比度调整
- ✅ 饱和度调整
- ⚠️ 不建议水平翻转（会破坏360度图像的连续性）

---

## 🚀 快速命令

```bash
# 1. 准备合成数据（100个样本）
python prepare_data.py --num_samples 100

# 2. 准备更多合成数据（1000个样本）
python prepare_data.py --num_samples 1000

# 3. 查看数据集信息
python prepare_data.py --mode info

# 4. 开始训练
python quickstart.py
```

---

## ❓ 常见问题

### Q: 合成数据能训练出好的模型吗？
A: 合成数据主要用于快速验证。为了获得最佳性能，建议使用真实眼动数据。

### Q: 必须使用360度图像吗？
A: 项目专门为360度图像设计，但也可以使用普通图像（只是球面卷积的优势体现不出来）。

### Q: 数据集需要多大？
A:
- 最小验证：50-100个样本
- 小型训练：500-1000个样本
- 完整训练：5000+ 个样本

### Q: 如何评估模型好坏？
A: 运行评估脚本：
```bash
python eval.py --checkpoint checkpoints/best_model.pth --visualize
```

---

**准备好了吗？运行 `python prepare_data.py` 开始吧！** 🚀
