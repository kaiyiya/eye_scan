"""
简单测试脚本 - 验证导入是否正常
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("测试导入...")
print("=" * 60)

# 测试PyTorch
try:
    import torch
    print(f"✓ PyTorch版本: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch导入失败: {e}")
    sys.exit(1)

# 测试TensorBoard（可选）
try:
    from torch.utils.tensorboard import SummaryWriter
    print("✓ TensorBoard已安装")
except ImportError:
    print("⚠ TensorBoard未安装（可选）")

# 测试配置
try:
    from config import Config
    print("✓ 配置模块导入成功")
except Exception as e:
    print(f"✗ 配置模块导入失败: {e}")
    sys.exit(1)

# 测试模型
try:
    from models import AdaptiveScanPath
    print("✓ 模型模块导入成功")
except Exception as e:
    print(f"✗ 模型模块导入失败: {e}")
    sys.exit(1)

# 测试损失函数
try:
    from utils.losses import ScanPathLoss, ScanPathMetrics
    print("✓ 损失函数模块导入成功")
except Exception as e:
    print(f"✗ 损失函数模块导入失败: {e}")
    sys.exit(1)

# 测试数据集
try:
    from data.dataset import ScanpathDataset
    print("✓ 数据集模块导入成功")
except Exception as e:
    print(f"✗ 数据集模块导入失败: {e}")
    sys.exit(1)

# 测试训练脚本导入
try:
    import train
    print("✓ 训练脚本导入成功")
except Exception as e:
    print(f"✗ 训练脚本导入失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有导入测试通过！")
print("=" * 60)
print("\n可以开始训练了:")
print("  python quickstart.py  # 快速开始（模拟数据）")
print("  python train.py        # 完整训练")
print("=" * 60)
