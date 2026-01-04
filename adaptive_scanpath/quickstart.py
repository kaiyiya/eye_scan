"""
快速开始脚本 - 在模拟数据上训练一个简单模型
用于验证系统是否正常工作
"""

import torch
import torch.optim as optim
import sys
import os
from tqdm import tqdm

from adaptive_scanpath.models import AdaptiveScanPath

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.losses import ScanPathLoss, ScanPathMetrics
from data.dataset import ScanpathDataset


def quickstart_train():
    """快速训练示例"""
    print("=" * 60)
    print("AdaptiveScanPath 快速开始")
    print("=" * 60)

    # 配置
    config = Config()
    config.batch_size = 4
    config.num_epochs = 5
    config.learning_rate = 1e-4

    # 创建模拟数据集
    print("\n创建模拟数据集...")
    train_dataset = ScanpathDataset(
        data_path='data/train_dummy',
        image_height=256,
        image_width=512,
        max_seq_len=12,
        is_train=True
    )

    val_dataset = ScanpathDataset(
        data_path='data/val_dummy',
        image_height=256,
        image_width=512,
        max_seq_len=12,
        is_train=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            'images': torch.stack([item['image'] for item in x]),
            'scanpaths': torch.stack([item['scanpath'] for item in x]),
            'lengths': torch.tensor([item['length'] for item in x])
        }
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            'images': torch.stack([item['image'] for item in x]),
            'scanpaths': torch.stack([item['scanpath'] for item in x]),
            'lengths': torch.tensor([item['length'] for item in x])
        }
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建模型
    print("\n创建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = AdaptiveScanPath(
        image_channels=3,
        image_height=256,
        image_width=512,
        feature_dim=256,  # 小一点
        cnn_channels=[64, 128, 256, 256],  # 小一点
        policy_hidden_dim=128,
        stopping_hidden_dim=64,
        max_seq_len=12,
        use_rnn=True,
        rnn_hidden_dim=128
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = ScanPathLoss()

    # 训练
    print("\n开始训练...")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        # 训练
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            images = batch['images'].to(device)
            gt_paths = batch['scanpaths'].to(device)
            gt_lengths = batch['lengths']

            # 前向传播
            pred_paths, stop_probs = model(images, gt_paths, teacher_forcing_ratio=0.5)

            # 计算损失
            losses = criterion(pred_paths, gt_paths, stop_probs, gt_lengths)
            loss = losses['total']

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        val_metrics = ScanPathMetrics()

        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                gt_paths = batch['scanpaths'].to(device)
                gt_lengths = batch['lengths']

                pred_paths, stop_probs = model(images, gt_paths)

                losses = criterion(pred_paths, gt_paths, stop_probs, gt_lengths)
                loss = losses['total']

                val_loss += loss.item()
                val_metrics.update(pred_paths, gt_paths, gt_lengths.tolist(), gt_lengths)

        val_loss /= len(val_loader)
        metrics_dict = val_metrics.compute()

        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  验证MSE: {metrics_dict['mse']:.6f}")
        print(f"  验证MAE: {metrics_dict['mae']:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/quickstart_model.pth')
            print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.4f})")

        print("=" * 60)

    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print("\n下一步:")
    print("  1. 查看保存的模型: checkpoints/quickstart_model.pth")
    print("  2. 在真实数据上训练: python train.py")
    print("  3. 评估模型: python eval.py --checkpoint checkpoints/quickstart_model.pth")


if __name__ == "__main__":
    quickstart_train()
