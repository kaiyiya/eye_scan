"""
球面CNN特征提取器
处理360度全景图像，保持球面几何特性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CoordConv2d(nn.Module):
    """
    坐标卷积层
    为卷积层添加绝对坐标信息，增强位置感知能力
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # 添加2个通道（x, y坐标）
        self.conv = nn.Conv2d(
            in_channels + 2,  # 原始通道 + 2个坐标通道
            out_channels,
            kernel_size,
            stride,
            padding
        )
        self.out_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, out_channels, H', W')
        """
        B, C, H, W = x.shape

        # 生成坐标网格
        xx_channel = torch.arange(W, dtype=torch.float32, device=x.device) / (W - 1) * 2 - 1
        yy_channel = torch.arange(H, dtype=torch.float32, device=x.device) / (H - 1) * 2 - 1

        xx_channel = xx_channel.view(1, 1, 1, W).expand(B, 1, H, W)
        yy_channel = yy_channel.view(1, 1, H, 1).expand(B, 1, H, W)

        # 拼接坐标信息
        x_with_coords = torch.cat([x, xx_channel, yy_channel], dim=1)  # (B, C+2, H, W)

        return self.conv(x_with_coords)


class SphereConv2d(nn.Module):
    """
    球面卷积层
    处理360度全景图像的球面几何特性
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # 常规卷积作为基础
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 等距圆柱投影的全景图
        Returns:
            out: (B, out_channels, H', W')
        """
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class SphericalCNN(nn.Module):
    """
    球面CNN特征提取器

    架构：
    1. 使用CoordConv增强位置感知
    2. 5层卷积逐步提取特征
    3. 全局平均池化得到特征向量
    """
    def __init__(
        self,
        in_channels=3,
        channels=[64, 128, 256, 384, 384],
        feature_dim=384
    ):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.feature_dim = feature_dim

        # 构建卷积层
        layers = []
        current_channels = in_channels

        for i, out_channels in enumerate(channels):
            # 第一层使用CoordConv
            if i == 0:
                layers.append(
                    CoordConv2d(
                        current_channels,
                        out_channels,
                        kernel_size=7,
                        stride=2,
                        padding=3
                    )
                )
            else:
                layers.append(
                    SphereConv2d(
                        current_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2
                    )
                )

            current_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 特征投影
        self.fc = nn.Sequential(
            nn.Linear(current_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 输入全景图像
        Returns:
            features: (B, feature_dim) 全局特征向量
        """
        # 卷积特征提取
        features = self.conv_layers(x)  # (B, C, H', W')

        # 全局池化
        pooled = self.global_pool(features)  # (B, C, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)

        # 特征投影
        output = self.fc(pooled)  # (B, feature_dim)

        return output

    def get_feature_maps(self, x):
        """
        获取中间特征图（用于可视化）

        Args:
            x: (B, 3, H, W) 输入图像
        Returns:
            feature_maps: list of (B, C, H, W) 各层特征图
        """
        feature_maps = []
        current = x

        for layer in self.conv_layers:
            current = layer(current)
            feature_maps.append(current)

        return feature_maps


class MultiScaleSphericalCNN(nn.Module):
    """
    多尺度球面CNN
    提取不同尺度的特征并融合
    """
    def __init__(
        self,
        in_channels=3,
        channels=[64, 128, 256, 384],
        feature_dim=384
    ):
        super().__init__()

        # 多尺度分支
        self.scale1 = self._make_branch(in_channels, channels[0], kernel_size=3)
        self.scale2 = self._make_branch(in_channels, channels[1], kernel_size=5)
        self.scale3 = self._make_branch(in_channels, channels[2], kernel_size=7)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(sum(channels), feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )

    def _make_branch(self, in_channels, out_channels, kernel_size):
        """创建多尺度分支"""
        return nn.Sequential(
            CoordConv2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            SphereConv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: (B, feature_dim)
        """
        # 多尺度特征提取
        feat1 = self.scale1(x).view(x.size(0), -1)  # (B, C1)
        feat2 = self.scale2(x).view(x.size(0), -1)  # (B, C2)
        feat3 = self.scale3(x).view(x.size(0), -1)  # (B, C3)

        # 拼接并融合
        multi_scale_feat = torch.cat([feat1, feat2, feat3], dim=-1)  # (B, C1+C2+C3)
        output = self.fusion(multi_scale_feat)  # (B, feature_dim)

        return output


def test_spherical_cnn():
    """测试球面CNN"""
    print("测试球面CNN...")

    # 基础版本
    model = SphericalCNN(in_channels=3, channels=[64, 128, 256, 384, 384], feature_dim=384)

    # 测试前向传播
    x = torch.randn(2, 3, 256, 512)
    features = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出特征形状: {features.shape}")

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 多尺度版本
    print("\n测试多尺度球面CNN...")
    multi_scale_model = MultiScaleSphericalCNN()
    features_multi = multi_scale_model(x)
    print(f"多尺度特征形状: {features_multi.shape}")

    print("\n测试通过！")


if __name__ == "__main__":
    test_spherical_cnn()
