"""
使用ScanDMM数据集训练AdaptiveScanPath模型
"""

import torch
import torch.optim as optim
import sys
import os
from datetime import datetime

from adaptive_scanpath.models import AdaptiveScanPath

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.losses import ScanPathLoss, ScanPathMetrics
from data.load_scandmm import create_scandmm_dataloaders


def train_with_scandmm():
    """使用ScanDMM数据训练模型"""
    print("=" * 60)
    print("使用ScanDMM数据集训练AdaptiveScanPath")
    print("=" * 60)

    # 配置
    config = Config()

    # ScanDMM数据路径
    scandmm_pkl_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'ScanDMM-master', 'Datasets', 'Sitzmann.pkl'
    )

    if not os.path.exists(scandmm_pkl_path):
        print(f"\n错误: 未找到ScanDMM数据集")
        print(f"路径: {scandmm_pkl_path}")
        print("\n请确认:")
        print("  1. ScanDMM-master/Datasets/Sitzmann.pkl 存在")
        print("  2. 或使用合成数据: python prepare_data.py")
        return

    print(f"\n✓ 找到ScanDMM数据集")
    print(f"  路径: {scandmm_pkl_path}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    # 创建数据加载器
    print("\n加载数据...")
    train_loader, val_loader = create_scandmm_dataloaders(
        pkl_path=scandmm_pkl_path,
        batch_size=8,  # ScanDMM数据较少，使用小batch
        num_workers=2,
        image_size=(config.image_height, config.image_width),
        max_seq_len=config.max_seq_len
    )

    # 创建模型
    print("\n创建模型...")
    model = AdaptiveScanPath(
        image_channels=3,
        image_height=config.image_height,
        image_width=config.image_width,
        feature_dim=config.feature_dim,
        cnn_channels=config.cnn_channels,
        policy_hidden_dim=config.policy_hidden_dim,
        policy_dropout=config.policy_dropout,
        stopping_hidden_dim=config.stopping_hidden_dim,
        max_seq_len=config.max_seq_len,
        use_rnn=config.use_rnn,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        use_feature_update=True
    ).to(device)

    print(f"✓ 模型已创建")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = ScanPathLoss()

    # 训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    num_epochs = 50  # ScanDMM数据较少，训练更多epoch
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            gt_paths = batch['scanpath'].to(device)
            gt_lengths = batch['length']

            # 调试：打印形状
            if batch_idx == 0:
                print(f"\n第一个batch:")
                print(f"  images形状: {images.shape}")
                print(f"  gt_paths形状: {gt_paths.shape}")
                print(f"  gt_lengths: {gt_lengths}")

            # 前向传播
            try:
                pred_paths, stop_probs = model(images, gt_paths, teacher_forcing_ratio=0.5)
            except Exception as e:
                print(f"\n错误在batch {batch_idx}")
                print(f"  images形状: {images.shape}")
                print(f"  gt_paths形状: {gt_paths.shape}")
                print(f"  gt_lengths: {gt_lengths}")
                print(f"  错误: {e}")
                raise

            # 计算损失
            losses = criterion(pred_paths, gt_paths, stop_probs, gt_lengths)
            loss = losses['total']

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        val_metrics = ScanPathMetrics()

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                gt_paths = batch['scanpath'].to(device)
                gt_lengths = batch['length']

                pred_paths, stop_probs = model(images, gt_paths)

                losses = criterion(pred_paths, gt_paths, stop_probs, gt_lengths)
                loss = losses['total']

                val_loss += loss.item()
                val_metrics.update(pred_paths, gt_paths, gt_lengths.tolist(), gt_lengths)

        val_loss /= len(val_loader)
        metrics_dict = val_metrics.compute()

        print(f"\nEpoch {epoch+1}/{num_epochs}")
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
            }, 'checkpoints/scandmm_model.pth')
            print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.4f})")

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print("=" * 60)

    print("\n下一步:")
    print("  1. 模型保存在: checkpoints/scandmm_model.pth")
    print("  2. 评估模型: python eval.py --checkpoint checkpoints/scandmm_model.pth")
    print("  3. 对比ScanDMM原始性能")


if __name__ == "__main__":
    train_with_scandmm()
