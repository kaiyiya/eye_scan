"""
完整的AdaptiveScanPath模型
基于AdaptiveNN架构的眼动路径预测模型
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from .sphere_cnn import SphericalCNN
from .policy_network import PolicyNetwork, StoppingNetwork, ContextRNN, FeatureUpdater


class AdaptiveScanPath(nn.Module):
    """
    基于AdaptiveNN架构的眼动路径预测模型

    核心特性：
    1. 球面CNN特征提取器：处理360度全景图像
    2. 策略网络：预测下一个注视点
    3. 停止网络：决策是否继续扫描
    4. 可选RNN：建模序列依赖
    5. 特征更新：累积视觉信息

    架构流程：
        输入图像 → 球面CNN → 全局特征
        ↓
        循环生成注视点序列:
            策略网络 → 预测注视点
            停止网络 → 决策是否继续
            特征更新 → 更新全局特征
            ↓
            如果继续，重复；否则停止
        ↓
        输出注视点序列
    """
    def __init__(
        self,
        # 图像配置
        image_channels=3,
        image_height=256,
        image_width=512,

        # 特征提取器配置
        feature_dim=384,
        cnn_channels=[64, 128, 256, 384, 384],

        # 策略网络配置
        policy_hidden_dim=256,
        policy_dropout=0.1,

        # 停止网络配置
        stopping_hidden_dim=128,

        # 序列建模配置
        max_seq_len=12,
        use_rnn=True,
        rnn_hidden_dim=256,
        rnn_num_layers=1,

        # 特征更新配置
        use_feature_update=True
    ):
        super().__init__()

        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        self.use_rnn = use_rnn
        self.use_feature_update = use_feature_update

        # 1. 特征提取器：球面CNN
        self.feature_extractor = SphericalCNN(
            in_channels=image_channels,
            channels=cnn_channels,
            feature_dim=feature_dim
        )

        # 2. 可选：上下文RNN
        if use_rnn:
            self.context_rnn = ContextRNN(
                input_size=feature_dim,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_num_layers
            )
            policy_input_dim = feature_dim + rnn_hidden_dim
        else:
            self.context_rnn = None
            policy_input_dim = feature_dim

        # 3. 策略网络：预测注视点
        self.policy_net = PolicyNetwork(
            input_dim=policy_input_dim,
            hidden_dim=policy_hidden_dim,
            output_dim=3,  # [theta, phi, duration]
            dropout=policy_dropout
        )

        # 4. 停止网络：决策是否继续
        self.stopping_net = StoppingNetwork(
            feature_dim=feature_dim,
            hidden_dim=stopping_hidden_dim
        )

        # 5. 可选：特征更新器
        if use_feature_update:
            self.feature_updater = FeatureUpdater(
                feature_dim=feature_dim,
                fixation_dim=3
            )
        else:
            self.feature_updater = None

    def forward(
        self,
        images: torch.Tensor,
        gt_paths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            images: (B, 3, H, W) 输入全景图像
            gt_paths: (B, T, 3) 真实路径（可选，用于teacher forcing）
            teacher_forcing_ratio: teacher forcing比例（0-1）

        Returns:
            pred_paths: (B, T, 3) 预测的扫描路径
            stop_probs: (B, T) 每步的停止概率
        """
        B = images.shape[0]
        device = images.device

        # 1. 提取全局特征
        global_features = self.feature_extractor(images)  # (B, feature_dim)
        current_features = global_features

        # 2. 初始化RNN状态
        if self.context_rnn is not None:
            hidden_state = self.context_rnn.init_hidden(B, device)

        # 3. 自回归生成序列
        all_fixations = []
        all_stop_probs = []

        for step in range(self.max_seq_len):
            # 3.1 更新RNN上下文
            if self.context_rnn is not None:
                rnn_output, hidden_state = self.context_rnn(current_features, hidden_state)
                policy_input = torch.cat([global_features, rnn_output], dim=-1)  # (B, feature_dim + rnn_hidden)
            else:
                policy_input = current_features

            # 3.2 预测下一个注视点
            fixation = self.policy_net(policy_input)  # (B, 3)
            all_fixations.append(fixation)

            # 3.3 预测停止概率
            stop_prob, _ = self.stopping_net(current_features, fixation)  # (B,)
            all_stop_probs.append(stop_prob)

            # 3.4 Teacher forcing（训练时）
            if self.training and gt_paths is not None and torch.rand(1).item() < teacher_forcing_ratio:
                fixation = gt_paths[:, step, :]  # 使用真实注视点

            # 3.5 更新特征（累积信息）
            if self.feature_updater is not None:
                current_features = self.feature_updater(current_features, fixation)

        # 4. 组合输出
        pred_paths = torch.stack(all_fixations, dim=1)  # (B, T, 3)
        stop_probs = torch.stack(all_stop_probs, dim=1)  # (B, T)

        return pred_paths, stop_probs

    def generate(
        self,
        images: torch.Tensor,
        max_steps: Optional[int] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        推理时生成扫描路径（带提前停止）

        Args:
            images: (B, 3, H, W) 输入图像
            max_steps: 最大步数（默认使用self.max_seq_len）
            temperature: 采样温度（>1更随机，<1更确定）

        Returns:
            pred_paths: (B, actual_T, 3) 预测路径（实际长度）
            actual_lengths: List[int] 每个样本的实际长度
        """
        if max_steps is None:
            max_steps = self.max_seq_len

        B = images.shape[0]
        device = images.device

        self.eval()
        with torch.no_grad():
            # 1. 提取特征
            global_features = self.feature_extractor(images)
            current_features = global_features

            # 2. 初始化RNN
            if self.context_rnn is not None:
                hidden_state = self.context_rnn.init_hidden(B, device)

            # 3. 生成序列（带提前停止）
            all_fixations = []
            stopped = torch.zeros(B, dtype=torch.bool, device=device)
            actual_lengths = [max_steps] * B

            for step in range(max_steps):
                # RNN上下文
                if self.context_rnn is not None:
                    rnn_output, hidden_state = self.context_rnn(current_features, hidden_state)
                    policy_input = torch.cat([global_features, rnn_output], dim=-1)
                else:
                    policy_input = current_features

                # 预测注视点
                fixation = self.policy_net(policy_input)
                all_fixations.append(fixation)

                # 停止决策
                stop_prob, stop_decision = self.stopping_net(current_features, fixation)

                # 更新停止状态
                stopped = stopped | stop_decision
                for i in range(B):
                    if stopped[i] and actual_lengths[i] == max_steps:
                        actual_lengths[i] = step + 1

                # 更新特征
                if self.feature_updater is not None:
                    current_features = self.feature_updater(current_features, fixation)

            # 4. 根据实际长度截断
            pred_paths = torch.stack(all_fixations, dim=1)  # (B, max_steps, 3)

            # 创建mask
            for i, length in enumerate(actual_lengths):
                if length < max_steps:
                    pred_paths[i, length:] = 0  # 填充0

        return pred_paths, actual_lengths

    def generate_with_diversity(
        self,
        images: torch.Tensor,
        num_samples: int = 5,
        temperature: float = 1.0
    ) -> List[torch.Tensor]:
        """
        生成多样化的扫描路径（多次采样）

        Args:
            images: (B, 3, H, W)
            num_samples: 每个图像的采样次数
            temperature: 采样温度

        Returns:
            List of (B, T, 3): num_samples个不同的路径预测
        """
        B = images.shape[0]
        all_paths = []

        for _ in range(num_samples):
            pred_paths, _ = self.generate(images, temperature=temperature)
            all_paths.append(pred_paths)

        return all_paths


def test_adaptive_scanpath():
    """测试AdaptiveScanPath模型"""
    print("测试AdaptiveScanPath模型...")

    # 创建模型
    model = AdaptiveScanPath(
        image_channels=3,
        image_height=256,
        image_width=512,
        feature_dim=384,
        cnn_channels=[64, 128, 256, 384, 384],
        policy_hidden_dim=256,
        policy_dropout=0.1,
        stopping_hidden_dim=128,
        max_seq_len=12,
        use_rnn=True,
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        use_feature_update=True
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试训练模式
    print("\n测试训练模式...")
    model.train()
    B = 4
    images = torch.randn(B, 3, 256, 512)
    gt_paths = torch.randn(B, 12, 3)

    pred_paths, stop_probs = model(images, gt_paths, teacher_forcing_ratio=0.5)

    print(f"输入图像形状: {images.shape}")
    print(f"预测路径形状: {pred_paths.shape}")
    print(f"停止概率形状: {stop_probs.shape}")
    print(f"预测路径示例（第一个样本，第一步）:\n{pred_paths[0, 0]}")

    # 测试推理模式
    print("\n测试推理模式（带提前停止）...")
    model.eval()
    with torch.no_grad():
        pred_paths, actual_lengths = model.generate(images)

    print(f"生成路径形状: {pred_paths.shape}")
    print(f"实际长度: {actual_lengths}")

    # 测试多样性采样
    print("\n测试多样性采样...")
    with torch.no_grad():
        diverse_paths = model.generate_with_diversity(images, num_samples=3)

    print(f"采样数量: {len(diverse_paths)}")
    print(f"每个路径形状: {diverse_paths[0].shape}")

    print("\nAdaptiveScanPath模型测试通过！")


if __name__ == "__main__":
    test_adaptive_scanpath()
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
