"""
损失函数
用于训练AdaptiveScanPath模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ScanPathLoss(nn.Module):
    """
    眼动路径预测损失函数

    包含多个损失项：
    1. 坐标回归损失（MSE）
    2. 持续时间损失
    3. 平滑性损失（鼓励路径连贯）
    4. 停止策略损失（预测序列长度）
    """
    def __init__(
        self,
        coord_weight=1.0,
        duration_weight=0.1,
        smoothness_weight=0.05,
        stopping_weight=0.2,
        use_smoothness=True,
        use_coverage=False
    ):
        super().__init__()

        self.coord_weight = coord_weight
        self.duration_weight = duration_weight
        self.smoothness_weight = smoothness_weight
        self.stopping_weight = stopping_weight
        self.use_smoothness = use_smoothness
        self.use_coverage = use_coverage

    def forward(
        self,
        pred_paths: torch.Tensor,
        gt_paths: torch.Tensor,
        stop_probs: torch.Tensor,
        gt_lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失

        Args:
            pred_paths: (B, T, 3) 预测路径
            gt_paths: (B, T, 3) 真实路径
            stop_probs: (B, T) 停止概率
            gt_lengths: (B,) 真实序列长度
            mask: (B, T) 有效长度掩码（可选）

        Returns:
            losses: dict containing all loss terms
        """
        B, T, _ = pred_paths.shape
        device = pred_paths.device

        # 1. 构建掩码（标记有效位置）
        if mask is None:
            mask = torch.zeros(B, T, device=device)
            for i, length in enumerate(gt_lengths):
                mask[i, :length] = 1.0

        mask = mask.unsqueeze(-1)  # (B, T, 1)

        # ============ 损失1: 坐标回归损失 ============
        coord_pred = pred_paths[:, :, :2]  # (B, T, 2) [theta, phi]
        coord_gt = gt_paths[:, :, :2]
        coord_loss = F.mse_loss(coord_pred * mask, coord_gt * mask, reduction='sum')
        coord_loss = coord_loss / mask.sum()  # 平均

        # ============ 损失2: 持续时间损失 ============
        duration_pred = pred_paths[:, :, 2]  # (B, T)
        duration_gt = gt_paths[:, :, 2]
        duration_loss = F.mse_loss(duration_pred * mask.squeeze(-1), duration_gt * mask.squeeze(-1), reduction='sum')
        duration_loss = duration_loss / mask.sum()

        # ============ 损失3: 平滑性损失 ============
        smoothness_loss = torch.tensor(0.0, device=device)
        if self.use_smoothness:
            # 相邻注视点之间的距离应该合理
            diff = pred_paths[:, 1:, :2] - pred_paths[:, :-1, :2]  # (B, T-1, 2)
            diff_mask = mask[:, 1:, :] * mask[:, :-1, :]  # (B, T-1, 1)

            # 计算距离的L2范数
            distances = torch.norm(diff, dim=-1, keepdim=True)  # (B, T-1, 1)

            # 鼓励适中的距离（不要太远也不要太近）
            # 使用高斯核，鼓励距离在合理范围内
            target_distance = 0.5  # 弧度
            smoothness_loss = F.mse_loss(distances * diff_mask, torch.full_like(distances, target_distance) * diff_mask, reduction='sum')
            smoothness_loss = smoothness_loss / (diff_mask.sum() + 1e-8)

        # ============ 损失4: 停止策略损失 ============
        # 构建停止目标：在真实长度之后应该停止
        stopping_targets = torch.arange(T, device=device).float().unsqueeze(0).expand(B, T)  # (B, T)
        stopping_targets = (stopping_targets >= gt_lengths.unsqueeze(1)).float()  # (B, T)

        stopping_loss = F.binary_cross_entropy(stop_probs, stopping_targets, reduction='none')  # (B, T)
        stopping_loss = stopping_loss.mean()

        # ============ 损失5: 覆盖度损失（可选）============
        coverage_loss = torch.tensor(0.0, device=device)
        if self.use_coverage:
            # 鼓励探索不同区域（最大化标准差）
            valid_pred = pred_paths * mask  # (B, T, 3)
            std = torch.std(valid_pred[:, :, :2], dim=1).mean(dim=-1)  # (B,)
            coverage_loss = -std.mean()  # 最大化标准差 = 最小化负标准差

        # ============ 总损失 ============
        total_loss = (
            self.coord_weight * coord_loss +
            self.duration_weight * duration_loss +
            self.smoothness_weight * smoothness_loss +
            self.stopping_weight * stopping_loss +
            0.01 * coverage_loss
        )

        return {
            'total': total_loss,
            'coord': coord_loss,
            'duration': duration_loss,
            'smoothness': smoothness_loss,
            'stopping': stopping_loss,
            'coverage': coverage_loss
        }


class ScanPathMetrics:
    """
    评估指标
    用于评估模型性能
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.metrics = {
            'mse': [],
            'mae': [],
            'length_error': [],
            'num_samples': 0
        }

    def update(
        self,
        pred_paths: torch.Tensor,
        gt_paths: torch.Tensor,
        pred_lengths: list,
        gt_lengths: torch.Tensor
    ):
        """
        更新指标

        Args:
            pred_paths: (B, T, 3) 预测路径
            gt_paths: (B, T, 3) 真实路径
            pred_lengths: list of int 预测长度
            gt_lengths: (B,) 真实长度
        """
        B = pred_paths.shape[0]
        device = pred_paths.device

        # 1. 构建掩码
        mask = torch.zeros(B, pred_paths.shape[1], device=device)
        for i, length in enumerate(gt_lengths):
            mask[i, :length] = 1.0
        mask = mask.unsqueeze(-1)

        # 2. MSE（均方误差）
        mse = F.mse_loss(pred_paths[:, :, :2] * mask, gt_paths[:, :, :2] * mask, reduction='sum')
        mse = mse / mask.sum()
        self.metrics['mse'].append(mse.item())

        # 3. MAE（平均绝对误差）
        mae = F.l1_loss(pred_paths[:, :, :2] * mask, gt_paths[:, :, :2] * mask, reduction='sum')
        mae = mae / mask.sum()
        self.metrics['mae'].append(mae.item())

        # 4. 长度预测误差
        pred_lengths_tensor = torch.tensor(pred_lengths, device=device, dtype=torch.float32)
        length_error = F.l1_loss(pred_lengths_tensor, gt_lengths.float())
        self.metrics['length_error'].append(length_error.item())

        self.metrics['num_samples'] += B

    def compute(self) -> Dict[str, float]:
        """
        计算平均指标

        Returns:
            metrics_dict: dict of metric names and values
        """
        result = {}

        if len(self.metrics['mse']) > 0:
            result['mse'] = sum(self.metrics['mse']) / len(self.metrics['mse'])
            result['mae'] = sum(self.metrics['mae']) / len(self.metrics['mae'])
            result['length_error'] = sum(self.metrics['length_error']) / len(self.metrics['length_error'])
            result['rmse'] = result['mse'] ** 0.5
            result['num_samples'] = self.metrics['num_samples']

        return result

    def print_metrics(self):
        """打印指标"""
        metrics = self.compute()
        print("=" * 60)
        print("评估指标")
        print("=" * 60)
        for key, value in metrics.items():
            if key != 'num_samples':
                print(f"{key}: {value:.6f}")
        print(f"样本数: {metrics.get('num_samples', 0)}")
        print("=" * 60)


class LengthAccuracy:
    """
    长度预测准确率
    """
    def __init__(self, tolerance=1):
        """
        Args:
            tolerance: 容忍误差（步数）
        """
        self.tolerance = tolerance
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, pred_lengths: list, gt_lengths: torch.Tensor):
        """
        更新准确率

        Args:
            pred_lengths: list of int
            gt_lengths: (B,) tensor
        """
        gt_lengths_list = gt_lengths.cpu().tolist()

        for pred, gt in zip(pred_lengths, gt_lengths_list):
            if abs(pred - gt) <= self.tolerance:
                self.correct += 1
            self.total += 1

    def compute(self) -> float:
        """计算准确率"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def print_accuracy(self):
        """打印准确率"""
        acc = self.compute()
        print(f"长度预测准确率（±{self.tolerance}步）: {acc*100:.2f}% ({self.correct}/{self.total})")


def test_loss():
    """测试损失函数"""
    print("测试损失函数...")

    # 创建损失函数
    criterion = ScanPathLoss()

    # 模拟数据
    B, T = 4, 12
    pred_paths = torch.randn(B, T, 3)
    gt_paths = torch.randn(B, T, 3)
    stop_probs = torch.rand(B, T)
    gt_lengths = torch.tensor([5, 7, 9, 12])

    # 计算损失
    losses = criterion(pred_paths, gt_paths, stop_probs, gt_lengths)

    print(f"总损失: {losses['total'].item():.6f}")
    print(f"坐标损失: {losses['coord'].item():.6f}")
    print(f"持续时间损失: {losses['duration'].item():.6f}")
    print(f"平滑性损失: {losses['smoothness'].item():.6f}")
    print(f"停止策略损失: {losses['stopping'].item():.6f}")

    print("\n损失函数测试通过！")


def test_metrics():
    """测试评估指标"""
    print("\n测试评估指标...")

    # 创建指标计算器
    metrics = ScanPathMetrics()
    length_acc = LengthAccuracy(tolerance=1)

    # 模拟数据
    B, T = 4, 12
    for _ in range(3):
        pred_paths = torch.randn(B, T, 3)
        gt_paths = torch.randn(B, T, 3)
        pred_lengths = [5, 7, 9, 12]
        gt_lengths = torch.tensor([5, 8, 9, 11])

        metrics.update(pred_paths, gt_paths, pred_lengths, gt_lengths)
        length_acc.update(pred_lengths, gt_lengths)

    # 打印结果
    metrics.print_metrics()
    length_acc.print_accuracy()

    print("\n评估指标测试通过！")


if __name__ == "__main__":
    test_loss()
    test_metrics()
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
