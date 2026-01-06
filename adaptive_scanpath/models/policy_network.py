"""
策略网络和停止网络
- PolicyNetwork: 预测下一个注视点
- StoppingNetwork: 决策是否停止扫描
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PolicyNetwork(nn.Module):
    """
    策略网络：预测下一个注视点

    输入：图像特征 + 可选的上下文
    输出：注视点坐标 [theta, phi, duration]
    """
    def __init__(
        self,
        input_dim=384,
        hidden_dim=256,
        output_dim=3,
        dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 策略网络主体
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, output_dim)
        )

        # 坐标范围参数（可学习）
        self.coord_scale = nn.Parameter(torch.ones(2))  # theta, phi的缩放因子

    def forward(self, context):
        """
        Args:
            context: (B, input_dim) 上下文特征
        Returns:
            fixation: (B, 3) [theta, phi, duration]
                - theta: [-π, π] 水平角度
                - phi: [-π/2, π/2] 垂直角度
                - duration: [0, 3] 持续时间（秒）
        """
        fixation = self.net(context)

        # 归一化坐标到合理范围
        # theta ∈ [-π, π]
        fixation[:, 0] = torch.tanh(fixation[:, 0]) * math.pi * self.coord_scale[0]

        # phi ∈ [-π/2, π/2]
        fixation[:, 1] = torch.tanh(fixation[:, 1]) * (math.pi / 2) * self.coord_scale[1]

        # duration ∈ [0, 3]
        fixation[:, 2] = torch.sigmoid(fixation[:, 2]) * 3.0

        return fixation

    def sample(self, context, num_samples=1):
        """
        采样多个注视点（用于推理时增加多样性）

        Args:
            context: (B, input_dim)
            num_samples: 采样数量
        Returns:
            fixations: (B, num_samples, 3)
        """
        B = context.shape[0]

        # 重复上下文
        context_repeated = context.unsqueeze(1).expand(B, num_samples, -1)
        context_repeated = context_repeated.reshape(B * num_samples, -1)

        # 添加噪声探索
        noise = torch.randn_like(context_repeated) * 0.1
        fixations = self.forward(context_repeated + noise)

        fixations = fixations.view(B, num_samples, 3)

        return fixations


class StoppingNetwork(nn.Module):
    """
    停止策略网络：决策是否继续扫描

    输入：当前特征 + 最后一个注视点
    输出：停止概率
    """
    def __init__(
        self,
        feature_dim=384,
        hidden_dim=128,
        dropout=0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 停止策略网络
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),  # +3 for fixation [theta, phi, duration]
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出停止概率 [0, 1]
        )

    def forward(self, features, last_fixation):
        """
        Args:
            features: (B, feature_dim) 当前特征
            last_fixation: (B, 3) 最后一个注视点 [theta, phi, duration]
        Returns:
            stop_prob: (B,) 停止概率
            stop_decision: (B,) 布尔决策（是否停止）
        """
        # 确保last_fixation是正确的形状
        if last_fixation.dim() == 1:
            last_fixation = last_fixation.unsqueeze(0)
        elif last_fixation.dim() == 3:
            last_fixation = last_fixation.squeeze(1)

        # 拼接特征和注视点
        context = torch.cat([features, last_fixation], dim=-1)  # (B, feature_dim + 3)

        # 预测停止概率
        stop_prob = self.net(context).squeeze(-1)  # (B,)

        # 训练时：直接使用概率
        # 推理时：采样决策
        if self.training:
            # 训练时使用确定性决策（阈值0.5）
            stop_decision = stop_prob > 0.5
        else:
            # 推理时使用采样增加多样性
            random_samples = torch.rand_like(stop_prob)
            stop_decision = random_samples < stop_prob

        return stop_prob, stop_decision


class ContextRNN(nn.Module):
    """
    上下文RNN：建模序列依赖关系

    维护一个隐状态，累积历史信息
    """
    def __init__(
        self,
        input_size=384,
        hidden_size=256,
        num_layers=1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, input_features, hidden_state=None):
        """
        Args:
            input_features: (B, input_size) 当前输入特征
            hidden_state: (num_layers, B, hidden_size) 之前的状态
        Returns:
            output: (B, hidden_size) RNN输出
            new_hidden: (num_layers, B, hidden_size) 更新后的状态
        """
        # 添加序列维度
        input_features = input_features.unsqueeze(1)  # (B, 1, input_size)

        # RNN前向传播
        output, new_hidden = self.rnn(input_features, hidden_state)

        # 移除序列维度
        output = output.squeeze(1)  # (B, hidden_size)

        return output, new_hidden

    def init_hidden(self, batch_size, device):
        """初始化隐状态"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class FeatureUpdater(nn.Module):
    """
    特征更新器：根据新的注视点更新全局特征

    模拟人类视觉中，新注视点提供的信息会累积到整体理解中
    """
    def __init__(
        self,
        feature_dim=384,
        fixation_dim=3
    ):
        super().__init__()

        # 注视点编码器
        self.fixation_encoder = nn.Sequential(
            nn.Linear(fixation_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim)
        )

        # 特征融合门控
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        # 特征更新
        self.update = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, global_features, fixation):
        """
        Args:
            global_features: (B, feature_dim) 全局特征
            fixation: (B, 3) 当前注视点
        Returns:
            updated_features: (B, feature_dim) 更新后的特征
        """
        # 确��fixation是正确的形状
        if fixation.dim() == 1:
            fixation = fixation.unsqueeze(0)  # (3,) -> (1, 3)
        elif fixation.dim() == 3:
            # (B, 1, 3) -> (B, 3)
            fixation = fixation.squeeze(1)

        # 编码注视点
        fixation_feat = self.fixation_encoder(fixation)  # (B, feature_dim)

        # 计算门控权重
        concat = torch.cat([global_features, fixation_feat], dim=-1)  # (B, feature_dim * 2)
        gate = self.gate(concat)  # (B, feature_dim)

        # 融合特征
        updated_features = gate * global_features + (1 - gate) * self.update(concat)

        return updated_features


def test_policy_network():
    """测试策略网络"""
    print("测试策略网络...")

    # 创建网络
    policy_net = PolicyNetwork(
        input_dim=384,
        hidden_dim=256,
        output_dim=3
    )

    # 测试前向传播
    B = 4
    context = torch.randn(B, 384)
    fixation = policy_net(context)

    print(f"输入特征形状: {context.shape}")
    print(f"输出注视点形状: {fixation.shape}")
    print(f"注视点示例:\n{fixation[0]}")

    # 测试采样
    fixations = policy_net.sample(context, num_samples=5)
    print(f"\n采样注视点形状: {fixations.shape}")

    print("\n策略网络测试通过！")


def test_stopping_network():
    """测试停止网络"""
    print("\n测试停止网络...")

    # 创建网络
    stopping_net = StoppingNetwork(
        feature_dim=384,
        hidden_dim=128
    )

    # 测试前向传播
    B = 4
    features = torch.randn(B, 384)
    last_fixation = torch.randn(B, 3)
    stop_prob, stop_decision = stopping_net(features, last_fixation)

    print(f"输入特征形状: {features.shape}")
    print(f"最后注视点形状: {last_fixation.shape}")
    print(f"停止概率: {stop_prob}")
    print(f"停止决策: {stop_decision}")

    print("\n停止网络测试通过！")


def test_context_rnn():
    """测试上下文RNN"""
    print("\n测试上下文RNN...")

    # 创建RNN
    context_rnn = ContextRNN(
        input_size=384,
        hidden_size=256,
        num_layers=1
    )

    # 测试前向传播
    B = 4
    input_features = torch.randn(B, 384)
    hidden = context_rnn.init_hidden(B, 'cpu')

    output, new_hidden = context_rnn(input_features, hidden)

    print(f"输入特征形状: {input_features.shape}")
    print(f"输出形状: {output.shape}")
    print(f"隐状态形状: {new_hidden.shape}")

    # 测试多步传播
    print("\n测试多步传播...")
    for step in range(3):
        output, hidden = context_rnn(input_features, hidden)
        print(f"步骤 {step+1}: 输出形状 {output.shape}")

    print("\n上下文RNN测试通过！")


def test_feature_updater():
    """测试特征更新器"""
    print("\n测试特征更新器...")

    # 创建更新器
    updater = FeatureUpdater(
        feature_dim=384,
        fixation_dim=3
    )

    # 测试前向传播
    B = 4
    global_features = torch.randn(B, 384)
    fixation = torch.randn(B, 3)

    updated_features = updater(global_features, fixation)

    print(f"全局特征形状: {global_features.shape}")
    print(f"注视点形状: {fixation.shape}")
    print(f"更新后特征形状: {updated_features.shape}")

    print("\n特征更新器测试通过！")


if __name__ == "__main__":
    test_policy_network()
    test_stopping_network()
    test_context_rnn()
    test_feature_updater()
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
