"""
测试脚本 - 验证所有模块是否正常工作
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("AdaptiveScanPath 完整测试")
print("=" * 60)

# 测试配置
print("\n1. 测试配置...")
try:
    from config import Config, print_config
    print("✓ 配置模块导入成功")
    print_config()
except Exception as e:
    print(f"✗ 配置模块测试失败: {e}")
    sys.exit(1)

# 测试球面CNN
print("\n2. 测试球面CNN...")
try:
    from models.sphere_cnn import SphericalCNN
    model = SphericalCNN(in_channels=3, channels=[64, 128, 256, 384], feature_dim=384)
    x = torch.randn(2, 3, 256, 512)
    features = model(x)
    print(f"✓ 球面CNN测试通过")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {features.shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ 球面CNN测试失败: {e}")
    sys.exit(1)

# 测试策略网络
print("\n3. 测试策略网络...")
try:
    from models.policy_network import PolicyNetwork, StoppingNetwork
    policy_net = PolicyNetwork(input_dim=384, hidden_dim=256)
    stopping_net = StoppingNetwork(feature_dim=384)

    context = torch.randn(4, 384)
    fixation = policy_net(context)
    stop_prob, stop_decision = stopping_net(context, fixation)

    print(f"✓ 策略网络测试通过")
    print(f"  注视点形状: {fixation.shape}")
    print(f"  停止概率: {stop_prob}")
except Exception as e:
    print(f"✗ 策略网络测试失败: {e}")
    sys.exit(1)

# 测试完整模型
print("\n4. 测试完整模型...")
try:
    from models.adaptive_scanpath import AdaptiveScanPath
    model = AdaptiveScanPath(
        image_channels=3,
        image_height=256,
        image_width=512,
        feature_dim=384,
        max_seq_len=12,
        use_rnn=True
    )

    images = torch.randn(2, 3, 256, 512)
    gt_paths = torch.randn(2, 12, 3)

    # 训练模式
    model.train()
    pred_paths, stop_probs = model(images, gt_paths)

    print(f"✓ 完整模型测试通过")
    print(f"  训练模式 - 预测路径形状: {pred_paths.shape}")
    print(f"  训练模式 - 停止概率形状: {stop_probs.shape}")

    # 推理模式
    model.eval()
    with torch.no_grad():
        pred_paths, actual_lengths = model.generate(images)

    print(f"  推理模式 - 预测路径形状: {pred_paths.shape}")
    print(f"  推理模式 - 实际长度: {actual_lengths}")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ 完整模型测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试损失函数
print("\n5. 测试损失函数...")
try:
    from utils.losses import ScanPathLoss, ScanPathMetrics
    criterion = ScanPathLoss()

    B, T = 4, 12
    pred_paths = torch.randn(B, T, 3)
    gt_paths = torch.randn(B, T, 3)
    stop_probs = torch.rand(B, T)
    gt_lengths = torch.tensor([5, 7, 9, 12])

    losses = criterion(pred_paths, gt_paths, stop_probs, gt_lengths)

    print(f"✓ 损失函数测试通过")
    print(f"  总损失: {losses['total'].item():.6f}")
    print(f"  坐标损失: {losses['coord'].item():.6f}")
    print(f"  停止损失: {losses['stopping'].item():.6f}")
except Exception as e:
    print(f"✗ 损失函数测试失败: {e}")
    sys.exit(1)

# 测试数据集
print("\n6. 测试数据集...")
try:
    from data.dataset import ScanpathDataset, create_dataloaders

    # 创建测试数据集（会使用模拟数据）
    dataset = ScanpathDataset(
        data_path='data/test_temp',
        image_height=256,
        image_width=512,
        max_seq_len=12,
        is_train=True
    )

    print(f"✓ 数据集测试通过")
    print(f"  数据集大小: {len(dataset)}")

    sample = dataset[0]
    print(f"  图像形状: {sample['image'].shape}")
    print(f"  扫描路径形状: {sample['scanpath'].shape}")
    print(f"  序列长度: {sample['length']}")

except Exception as e:
    print(f"✗ 数据集测试失败: {e}")
    sys.exit(1)

# 测试训练流程
print("\n7. 测试训练流程（单步）...")
try:
    from config import Config
    from models import AdaptiveScanPath
    from utils.losses import ScanPathLoss
    import torch.optim as optim

    config = Config()
    config.batch_size = 2  # 小batch测试

    # 创建模型
    model = AdaptiveScanPath(
        image_channels=3,
        image_height=256,
        image_width=512,
        feature_dim=384,
        max_seq_len=12,
        use_rnn=True
    )

    # 创建优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = ScanPathLoss()

    # 模拟数据
    images = torch.randn(2, 3, 256, 512)
    gt_paths = torch.randn(2, 12, 3)
    gt_lengths = torch.tensor([5, 7])

    # 训练步骤
    model.train()
    optimizer.zero_grad()

    pred_paths, stop_probs = model(images, gt_paths)
    losses = criterion(pred_paths, gt_paths, stop_probs, gt_lengths)
    loss = losses['total']

    loss.backward()
    optimizer.step()

    print(f"✓ 训练流程测试通过")
    print(f"  损失值: {loss.item():.6f}")
    print(f"  梯度已更新")

except Exception as e:
    print(f"✗ 训练流程测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有测试通过！")
print("=" * 60)
print("\n系统已准备就绪，可以开始训练！")
print("\n快速开始:")
print("  1. 训练模型: python train.py")
print("  2. 评估模型: python eval.py --checkpoint checkpoints/best_model.pth")
print("  3. 查看日志: tensorboard --logdir=logs")
print("=" * 60)
