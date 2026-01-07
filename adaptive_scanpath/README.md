# AdaptiveScanPath - 基于AdaptiveNN的360度全景图像眼动路径预测模型

## 项目简介

AdaptiveScanPath是基于AdaptiveNN架构在ScanDMM模型上的迁移创新，用于预测360度全景图像的眼动扫描路径。

## 核心特性

- **球面CNN特征提取器**: 处理360度全景图像，保持球面几何特性
- **策略网络**: 预测下一个注视点
- **停止网络**: 决策是否继续扫描
- **RNN序列建模**: 建模序列依赖关系
- **特征更新机制**: 累积视觉信息

## 数据集

本项目使用 **ScanDMM Sitzmann数据集** 进行训练和评估。

### 数据准备步骤

**第一步：处理原始数据集**

运行数据处理脚本，将pkl格式转换为清晰的文件夹结构：

```bash
cd adaptive_scanpath
python prepare_scandmm_data.py --input_path ../ScanDMM-master/Datasets/Sitzmann.pkl --output_dir data/scandmm
```

这将创建以下目录结构：
```
data/scandmm/
├── train/
│   ├── images/          # 训练图像
│   └── annotations.json # 训练标注
├── val/
│   ├── images/          # 验证图像
│   └── annotations.json # 验证标注
└── test/
    ├── images/          # 测试图像
    └── annotations.json # 测试标注
```

**数据划分说明：**
- **训练集**: 从原始train split中随机选择85%的样本
- **验证集**: 从原始train split中随机选择15%的样本
- **测试集**: 使用原始test split的所有样本

**第二步：开始训练**

数据处理完成后，直接运行训练脚本即可。

## 项目结构

```
adaptive_scanpath/
├── models/                  # 模型定义
│   ├── adaptive_scanpath.py  # 主模型
│   ├── policy_network.py     # 策略网络和停止网络
│   └── sphere_cnn.py          # 球面CNN特征提取器
├── data/                     # 数据加载器
│   └── load_scandmm.py       # ScanDMM数据集加载器
├── utils/                    # 工具函数
│   └── losses.py             # 损失函数和评估指标
├── config.py                 # 配置文件
├── train.py                  # 训练脚本
├── eval.py                   # 评估脚本
└── requirements.txt          # 依赖包
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练模型

### 基本训练

**确保已完成数据准备步骤！**

```bash
cd adaptive_scanpath
python train.py
```

训练脚本会自动：
1. 加载处理好的数据集 (`data/scandmm/train` 和 `data/scandmm/val`)
2. 创建模型
3. 开始训练
4. 保存检查点到 `checkpoints/adaptive_scanpath/{timestamp}/`

### 训练配置

在 `config.py` 中可以调整训练参数：
- `batch_size`: 批次大小（默认16）
- `num_epochs`: 训练轮数（默认100）
- `learning_rate`: 学习率（默认1e-4）
- `image_height`, `image_width`: 图像尺寸（默认256x512）
- `max_seq_len`: 最大序列长度（默认12）

## 评估模型

```bash
python eval.py --checkpoint checkpoints/adaptive_scanpath/{timestamp}/checkpoints/best_model.pth
```

可选参数：
- `--visualize`: 生成可视化结果
- `--export`: 导出预测结果到JSON

## 模型架构

### AdaptiveScanPath模型

1. **特征提取**: 球面CNN提取全局特征
2. **序列生成**: 自回归生成注视点序列
   - 策略网络预测下一个注视点
   - 停止网络决策是否继续
   - RNN建模序列依赖
   - 特征更新累积信息

### 损失函数

- **坐标损失**: MSE损失，预测注视点坐标
- **持续时间损失**: MSE损失，预测注视持续时间
- **平滑性损失**: 鼓励路径连贯
- **停止策略损失**: 预测序列长度

## 检查点

训练过程中会保存：
- `checkpoint_epoch_{N}.pth`: 定期检查点
- `best_model.pth`: 最佳验证损失模型
- `final_model.pth`: 最终模型

## 注意事项

1. **数据准备**: 首次使用前必须先运行 `prepare_scandmm_data.py` 处理数据集
2. **数据集路径**: 确保 `ScanDMM-master/Datasets/Sitzmann.pkl` 存在
3. **GPU内存**: 如果GPU内存不足，可以减小 `batch_size` 或图像尺寸
4. **训练时间**: ScanDMM数据集较小，建议训练50-100个epoch
5. **数据格式**: 处理后的数据使用JSON格式，包含图像路径和扫描路径标注

## 引用

如果使用本项目，请引用相关论文：
- AdaptiveNN: [相关论文]
- ScanDMM: [相关论文]

## 许可证

[添加许可证信息]

