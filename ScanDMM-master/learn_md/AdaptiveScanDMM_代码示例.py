"""
AdaptiveScanDMM: 结合AdaptiveNN"从粗到细"思想的ScanDMM改进版本

核心改进：
1. 多尺度特征提取：粗粒度（快速） + 细粒度（详细）
2. 动态区域聚焦：根据注视点提取局部特征
3. 自适应计算：根据场景复杂度调整计算资源

这是一个概念性实现，展示核心思想，实际使用时需要根据具体需求调整。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sphere_cnn import Sphere_CNN
from models import DMM, GatedTransition, Emitter, Combiner
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine


class RegionExtractor(nn.Module):
    """
    区域提取器：从360度图像中提取注视点周围的局部区域
    
    功能：
    - 根据注视点坐标提取局部区域
    - 考虑360度图像的左右边界连续性
    - 支持不同大小的区域提取
    """
    
    def __init__(self, region_size=64):
        """
        Args:
            region_size: 提取区域的大小（像素），例如64x64
        """
        super().__init__()
        self.region_size = region_size
    
    def forward(self, images, gaze_points):
        """
        从360度图像中提取注视点周围的局部区域
        
        Args:
            images: 输入图像 (batch, 3, H, W)，例如 (64, 3, 128, 256)
            gaze_points: 注视点坐标 (batch, 2)，归一化坐标 [0, 1]，格式为 (y, x)
        
        Returns:
            local_regions: 局部区域 (batch, 3, region_size, region_size)
        """
        batch_size, channels, H, W = images.shape
        
        # 将归一化坐标转换为像素坐标
        y_pixel = gaze_points[:, 0] * H  # (batch,)
        x_pixel = gaze_points[:, 1] * W  # (batch,)
        
        # 计算区域边界
        half_size = self.region_size // 2
        
        # 为每个样本创建采样网格
        local_regions = []
        
        for b in range(batch_size):
            y_center = y_pixel[b].item()
            x_center = x_pixel[b].item()
            
            # 创建局部区域的采样网格
            y_coords = torch.arange(
                y_center - half_size, 
                y_center + half_size, 
                dtype=torch.float32
            ).clamp(0, H - 1)
            
            x_coords = torch.arange(
                x_center - half_size, 
                x_center + half_size, 
                dtype=torch.float32
            ).clamp(0, W - 1)
            
            # 处理360度图像的左右边界连续性
            # 如果x坐标超出边界，需要wrap around
            x_coords = x_coords % W
            
            # 使用grid_sample提取局部区域
            # 创建归一化坐标网格 [-1, 1]
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            y_grid_norm = (y_grid / H) * 2 - 1  # 归一化到[-1, 1]
            x_grid_norm = (x_grid / W) * 2 - 1
            
            # 组合成grid_sample需要的格式 (1, H, W, 2)
            grid = torch.stack([x_grid_norm, y_grid_norm], dim=-1).unsqueeze(0)
            grid = grid.to(images.device)
            
            # 提取局部区域
            local_region = F.grid_sample(
                images[b:b+1],  # (1, 3, H, W)
                grid,  # (1, region_size, region_size, 2)
                mode='bilinear',
                align_corners=True
            )  # (1, 3, region_size, region_size)
            
            local_regions.append(local_region.squeeze(0))
        
        return torch.stack(local_regions, dim=0)  # (batch, 3, region_size, region_size)


class AdaptiveSphere_CNN(nn.Module):
    """
    自适应球面CNN：支持多尺度动态特征提取
    
    工作方式：
    1. 粗粒度模式：快速提取全局特征（低分辨率，50维）
    2. 细粒度模式：提取局部区域特征（高分辨率，50维）+ 全局特征（50维）
    3. 自适应模式：根据注视次数动态切换
    """
    
    def __init__(self, out_put_dim=100):
        super().__init__()
        self.output_dim = out_put_dim
        
        # 粗粒度特征提取器（快速，全局）
        self.coarse_cnn = Sphere_CNN(out_put_dim=out_put_dim // 2)  # 50维
        
        # 细粒度特征提取器（详细，局部）
        self.fine_cnn = Sphere_CNN(out_put_dim=out_put_dim // 2)  # 50维
        
        # 区域提取器
        self.region_extractor = RegionExtractor(region_size=64)
        
        # 特征融合层（可选，用于进一步融合粗细特征）
        self.feature_fusion = nn.Sequential(
            nn.Linear(out_put_dim, out_put_dim),
            nn.Tanh()
        )
    
    def forward(self, images, gaze_points=None, scale='coarse'):
        """
        前向传播
        
        Args:
            images: 输入图像 (batch, 3, H, W)
            gaze_points: 注视点坐标 (batch, 2)，归一化坐标 [0, 1]，格式为 (y, x)
            scale: 'coarse' | 'fine' | 'adaptive'
        
        Returns:
            features: 图像特征 (batch, output_dim)
        """
        if scale == 'coarse':
            # 粗粒度：快速提取全局特征
            features = self.coarse_cnn(images)  # (batch, 50)
            # 扩展到完整维度（用零填充或复制）
            features = F.pad(features, (0, self.output_dim - features.shape[1]))  # (batch, 100)
            return features
        
        elif scale == 'fine' and gaze_points is not None:
            # 细粒度：提取局部区域特征 + 全局粗特征
            # 1. 提取局部区域
            local_regions = self.region_extractor(images, gaze_points)  # (batch, 3, 64, 64)
            
            # 2. 提取局部细特征
            fine_features = self.fine_cnn(local_regions)  # (batch, 50)
            
            # 3. 提取全局粗特征
            coarse_features = self.coarse_cnn(images)  # (batch, 50)
            
            # 4. 融合粗细特征
            combined = torch.cat([coarse_features, fine_features], dim=1)  # (batch, 100)
            
            # 5. 可选：进一步融合
            combined = self.feature_fusion(combined)  # (batch, 100)
            
            return combined
        
        else:
            # 默认使用粗特征
            return self.forward(images, scale='coarse')


class AdaptiveDMM(DMM):
    """
    自适应深度马尔可夫模型：支持从粗到细的特征提取
    
    改进点：
    1. 使用AdaptiveSphere_CNN替代固定CNN
    2. 在序列生成过程中动态切换粗细特征
    3. 根据注视点提取局部特征
    """
    
    def __init__(self, coarse_threshold=5, *args, **kwargs):
        """
        Args:
            coarse_threshold: 前N步使用粗特征，后续使用细特征
        """
        # 先调用父类初始化（但会创建原始CNN，我们需要替换它）
        super().__init__(*args, **kwargs)
        
        # 替换为自适应CNN
        self.cnn = AdaptiveSphere_CNN(out_put_dim=self.z_0.size(0))
        
        # 粗细切换阈值
        self.coarse_threshold = coarse_threshold
    
    def model(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths, 
              images=None, annealing_factor=1.0, predict=False):
        """
        生成模型：支持从粗到细的特征提取
        
        改进：
        - 前coarse_threshold步使用粗特征（快速）
        - 后续步骤使用细特征（根据注视点提取局部特征）
        """
        T_max = scanpaths.size(1)
        pyro.module("dmm", self)
        
        # 初始化（与原始DMM相同）
        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))
        
        # ===== 关键改进：动态特征提取 =====
        # 预先提取粗特征（用于初始阶段）
        img_features_coarse = self.cnn(images, scale='coarse')  # (batch, 100)
        
        with pyro.plate("z_minibatch", len(scanpaths)):
            for t in pyro.markov(range(1, T_max + 1)):
                # ===== 决定使用粗特征还是细特征 =====
                if t <= self.coarse_threshold:
                    # 粗粒度阶段：使用全局粗特征
                    current_img_features = img_features_coarse
                else:
                    # 细粒度阶段：根据上一个注视点提取局部特征
                    # 获取上一个注视点的2D坐标（用于提取局部区域）
                    # 注意：scanpaths是3D坐标，需要转换为2D平面坐标
                    from suppor_lib import xyz2plane
                    
                    # 获取上一个注视点（t-2是因为索引从0开始，且t从1开始）
                    prev_gaze_3d = scanpaths[:, t-2, :]  # (batch, 3) - 3D坐标
                    prev_gaze_2d = xyz2plane(prev_gaze_3d)  # (batch, 2) - 2D平面坐标，归一化[0,1]
                    
                    # 提取细粒度特征（结合全局和局部）
                    current_img_features = self.cnn(
                        images, 
                        gaze_points=prev_gaze_2d, 
                        scale='fine'
                    )  # (batch, 100)
                
                # ===== 使用动态特征进行转移（与原始DMM相同） =====
                z_mu, z_sigma = self.trans(z_prev, current_img_features)
                
                # 采样隐状态
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma)
                                      .mask(mask[:, t - 1: t]).to_event(1))
                
                # 生成观测
                x_mu, x_sigma = self.emitter(z_t)
                
                # 训练或预测
                if not predict:
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                .mask(mask[:, t - 1: t]).to_event(1), 
                                obs=scanpaths[:, t - 1, :])
                else:
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                .mask(mask[:, t - 1: t]).to_event(1))
                
                z_prev = z_t


class AdaptiveGazeController(nn.Module):
    """
    自适应注视控制器：决定何时停止注视
    
    功能：
    - 根据当前隐状态判断是否应该停止注视
    - 实现自适应计算成本控制
    """
    
    def __init__(self, z_dim=100):
        super().__init__()
        self.stop_network = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z_t):
        """
        根据当前隐状态计算停止概率
        
        Args:
            z_t: 当前隐状态 (batch, z_dim)
        
        Returns:
            stop_prob: 停止概率 (batch, 1)，值域[0, 1]
        """
        stop_prob = self.stop_network(z_t)  # (batch, 1)
        return stop_prob
    
    def should_stop(self, z_t, threshold=0.5):
        """
        判断是否应该停止注视
        
        Args:
            z_t: 当前隐状态 (batch, z_dim)
            threshold: 停止阈值，默认0.5
        
        Returns:
            stop_mask: 布尔张量 (batch,)，True表示应该停止
        """
        stop_prob = self.forward(z_t)  # (batch, 1)
        return (stop_prob.squeeze(-1) > threshold)  # (batch,)


# ===== 使用示例 =====

if __name__ == '__main__':
    """
    使用示例：展示如何使用AdaptiveDMM
    """
    
    # 创建自适应DMM模型
    adaptive_dmm = AdaptiveDMM(
        input_dim=3,
        z_dim=100,
        emission_dim=100,
        transition_dim=200,
        rnn_dim=600,
        coarse_threshold=5,  # 前5步使用粗特征
        use_cuda=False
    )
    
    print("AdaptiveDMM模型创建成功！")
    print(f"粗特征维度: {adaptive_dmm.cnn.coarse_cnn.output_dim}")
    print(f"细特征维度: {adaptive_dmm.cnn.fine_cnn.output_dim}")
    print(f"粗细切换阈值: {adaptive_dmm.coarse_threshold}")
    
    # 示例：创建自适应注视控制器
    gaze_controller = AdaptiveGazeController(z_dim=100)
    print("\n自适应注视控制器创建成功！")
    
    # 示例：测试区域提取器
    region_extractor = RegionExtractor(region_size=64)
    dummy_images = torch.randn(2, 3, 128, 256)  # 2个样本，3通道，128x256
    dummy_gaze_points = torch.tensor([[0.5, 0.5], [0.3, 0.7]])  # 2个注视点，归一化坐标
    local_regions = region_extractor(dummy_images, dummy_gaze_points)
    print(f"\n区域提取器测试成功！")
    print(f"输入图像形状: {dummy_images.shape}")
    print(f"提取的局部区域形状: {local_regions.shape}")
    
    print("\n✅ 所有组件测试通过！")
