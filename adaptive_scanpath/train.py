"""
训练脚本 - AdaptiveScanPath模型
完整的训练流程，包括验证、保存和日志记录
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 尝试导入TensorBoard（可选）
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("警告: TensorBoard未安装，将跳过TensorBoard日志记录")
    print("安装命令: pip install tensorboard")

from adaptive_scanpath.models import AdaptiveScanPath

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.losses import ScanPathLoss, ScanPathMetrics, LengthAccuracy
from data.dataset import create_dataloaders


class Trainer:
    """
    训练器类
    封装完整的训练流程
    """
    def __init__(self, config):
        """
        初始化训练器

        Args:
            config: 配置对象
        """
        self.config = config
        self.device = torch.device(config.device)

        # 创建输出目录
        self.experiment_dir = os.path.join(
            config.output_dir,
            config.experiment_name,
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'checkpoints'), exist_ok=True)

        # TensorBoard（可选）
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=os.path.join(config.log_dir, config.experiment_name))
        else:
            self.writer = None

        # 初始化
        self._setup_model()
        self._setup_data()
        self._setup_criterion()
        self._setup_optimizer()

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        print("=" * 60)
        print("训练器初始化完成")
        print(f"实验目录: {self.experiment_dir}")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)

    def _setup_model(self):
        """设置模型"""
        self.model = AdaptiveScanPath(
            image_channels=self.config.image_channels,
            image_height=self.config.image_height,
            image_width=self.config.image_width,
            feature_dim=self.config.feature_dim,
            cnn_channels=self.config.cnn_channels,
            policy_hidden_dim=self.config.policy_hidden_dim,
            policy_dropout=self.config.policy_dropout,
            stopping_hidden_dim=self.config.stopping_hidden_dim,
            max_seq_len=self.config.max_seq_len,
            use_rnn=self.config.use_rnn,
            rnn_hidden_dim=self.config.rnn_hidden_dim,
            rnn_num_layers=self.config.rnn_num_layers,
            use_feature_update=True
        ).to(self.device)

        print(f"模型已加载到设备: {self.device}")

    def _setup_data(self):
        """设置数据加载器"""
        print("加载数据集...")

        self.train_loader, self.val_loader = create_dataloaders(
            train_data_path=self.config.train_data_path,
            val_data_path=self.config.val_data_path,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            image_height=self.config.image_height,
            image_width=self.config.image_width,
            max_seq_len=self.config.max_seq_len,
            min_seq_len=self.config.min_seq_len
        )

        print(f"训练集批次数: {len(self.train_loader)}")
        print(f"验证集批次数: {len(self.val_loader)}")

    def _setup_criterion(self):
        """设置损失函数"""
        self.criterion = ScanPathLoss(
            coord_weight=self.config.loss_weights['coord'],
            duration_weight=self.config.loss_weights['duration'],
            smoothness_weight=self.config.loss_weights['smoothness'],
            stopping_weight=self.config.loss_weights['stopping'],
            use_smoothness=True,
            use_coverage=False
        )

        self.train_metrics = ScanPathMetrics()
        self.val_metrics = ScanPathMetrics()
        self.length_acc = LengthAccuracy(tolerance=1)

    def _setup_optimizer(self):
        """设置优化器和学习率调度器"""
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 学习率调度器
        if self.config.use_scheduler:
            if self.config.scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    **self.config.scheduler_params['cosine']
                )
            elif self.config.scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    **self.config.scheduler_params['step']
                )
            elif self.config.scheduler_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    **self.config.scheduler_params['plateau']
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()
        self.length_acc.reset()

        epoch_loss = 0.0
        epoch_start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            images = batch['images'].to(self.device)
            gt_paths = batch['scanpaths'].to(self.device)
            gt_lengths = batch['lengths']

            # 前向传播
            pred_paths, stop_probs = self.model(images, gt_paths, teacher_forcing_ratio=0.5)

            # 计算损失
            losses = self.criterion(pred_paths, gt_paths, stop_probs, gt_lengths)
            loss = losses['total']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()

            # 更新指标
            epoch_loss += loss.item()
            self.train_metrics.update(pred_paths.detach(), gt_paths, gt_lengths.tolist(), gt_lengths)

            # 日志
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

                # TensorBoard（可选）
                if self.writer is not None:
                    global_step = self.current_epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('train/loss', loss.item(), global_step)
                    self.writer.add_scalar('train/coord_loss', losses['coord'].item(), global_step)
                    self.writer.add_scalar('train/stopping_loss', losses['stopping'].item(), global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)

        # 平均损失
        avg_loss = epoch_loss / len(self.train_loader)
        epoch_time = time.time() - epoch_start_time

        # 计算指标
        train_metrics_dict = self.train_metrics.compute()

        print(f"\n训练完成 - 损失: {avg_loss:.4f}, 时间: {epoch_time:.2f}s")
        print(f"训练指标: MSE={train_metrics_dict['mse']:.6f}, MAE={train_metrics_dict['mae']:.6f}")

        return avg_loss, train_metrics_dict

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        self.val_metrics.reset()
        self.length_acc.reset()

        val_loss = 0.0

        for batch in tqdm(self.val_loader, desc="验证中..."):
            images = batch['images'].to(self.device)
            gt_paths = batch['scanpaths'].to(self.device)
            gt_lengths = batch['lengths']

            # 前向传播
            pred_paths, stop_probs = self.model(images, gt_paths, teacher_forcing_ratio=0.0)

            # 计算损失
            losses = self.criterion(pred_paths, gt_paths, stop_probs, gt_lengths)
            loss = losses['total']

            val_loss += loss.item()

            # 更新指标
            self.val_metrics.update(pred_paths, gt_paths, gt_lengths.tolist(), gt_lengths)

        # 平均损失
        avg_val_loss = val_loss / len(self.val_loader)

        # 计算指标
        val_metrics_dict = self.val_metrics.compute()

        print(f"\n验证完成 - 损失: {avg_val_loss:.4f}")
        print(f"验证指标: MSE={val_metrics_dict['mse']:.6f}, MAE={val_metrics_dict['mae']:.6f}")

        return avg_val_loss, val_metrics_dict

    def save_checkpoint(self, filename, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filepath = os.path.join(self.experiment_dir, 'checkpoints', filename)
        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath}")

        if is_best:
            best_filepath = os.path.join(self.experiment_dir, 'checkpoints', 'best_model.pth')
            torch.save(checkpoint, best_filepath)
            print(f"最佳模型已保存: {best_filepath}")

    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"检查点已加载: {filepath}")
        print(f"从epoch {self.current_epoch}继续训练")

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # 训练
            train_loss, train_metrics = self.train_epoch()

            # 验证
            if (epoch + 1) % self.config.val_interval == 0:
                val_loss, val_metrics = self.validate()

                # TensorBoard（可选）
                if self.writer is not None:
                    self.writer.add_scalar('val/loss', val_loss, epoch)
                    self.writer.add_scalar('val/mse', val_metrics['mse'], epoch)
                    self.writer.add_scalar('val/mae', val_metrics['mae'], epoch)

                # 学习率调度
                if self.scheduler is not None:
                    if self.config.scheduler_type == 'plateau':
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # 保存检查点
                is_best = val_loss < self.best_val_loss

                if is_best:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1

                if (epoch + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', is_best)

                # 早停
                if self.config.use_early_stopping and self.epochs_no_improve >= self.config.patience:
                    print(f"\n早停触发！{self.config.patience}个epoch没有改进")
                    break

        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print("=" * 60)

        if self.writer is not None:
            self.writer.close()


def main():
    """主函数"""
    # 打印配置
    from config import print_config
    print_config()

    # 创建训练器
    trainer = Trainer(Config())

    # 开始训练
    trainer.train()

    # 保存最终模型
    trainer.save_checkpoint('final_model.pth')


if __name__ == "__main__":
    main()
