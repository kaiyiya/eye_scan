"""
ScanDMM: 360度图像扫描路径预测的深度马尔可夫模型
整合版本 - 将所有模块合并到一个文件中，便于理解和学习

模型概述：
    这是一个用于预测人们在观看360度全景图像时的视线扫描路径的深度学习模型。
    模型使用深度马尔可夫模型（DMM）来建模隐状态序列，并通过变分推断进行训练。

主要组件：
1. 配置参数 (Config)
   - 数据集路径、训练超参数、模型配置等

2. 添加坐标通道 (AddCoordsTh)
   - CoordConv技术，在特征图上添加x和y坐标通道，帮助CNN学习空间位置特征

3. 球面CNN (Sphere CNN)
   - 专门为360度等距圆柱投影图像设计的卷积层
   - 考虑球面几何特性，使用grid_sample进行采样
   - 包括SphereConv2D和Sphere_CNN类

4. 深度马尔可夫模型 (DMM)
   - GatedTransition: 门控转移机制，建模隐状态转移（生成模型）
   - Combiner: 组合器，变分后验推断（训练模型）
   - Emitter: 发射器，从隐状态生成观测
   - model和guide两个方法：生成模型和变分后验

5. 训练类 (Train)
   - 封装完整的训练流程
   - 使用SVI（随机变分推断）和KL散度退火
   - 支持检查点保存和加载

6. 推理类 (Inference)
   - 从训练好的模型生成扫描路径
   - 使用随机起始点
   - 输出2D平面坐标

7. 数据处理 (Data Processing)
   - Sitzmann_Dataset: 数据集加载和处理
   - 坐标转换函数：球面坐标、3D坐标、平面坐标之间的转换
   - 数据增强：通过旋转生成更多训练样本

文件结构：
    - 第一部分：配置参数
    - 第二部分：添加坐标通道
    - 第三部分：球面CNN
    - 第四部分：深度马尔可夫模型
    - 第五部分：工具函数（坐标转换）
    - 第六部分：训练类
    - 第七部分：推理类
    - 第八部分：数据处理
    - 主函数：训练入口

使用示例：
    训练：
        python scandmm_integrated.py --dataset ./Datasets/Sitzmann.pkl --lr 0.0003 --bs 64 --epochs 500
    
    数据集预处理：
        dataset = Sitzmann_Dataset()
        data_dict = dataset.run()
        save_file('Sitzmann.pkl', data_dict)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import pickle
import pickle as pck
import cv2
import torchvision.transforms as transforms
from argparse import ArgumentParser
from functools import lru_cache
from os.path import exists

# Pyro相关导入
import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import *
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from numpy import sin, cos, tan, arcsin, arctan
from torch.nn import Parameter

# 使用numpy的pi，因为球面计算函数需要numpy的pi
import numpy as np

pi = np.pi


# ============================================================================
# 第一部分：配置参数 (Configuration)
# ============================================================================

class Config:
    """
    训练和模型配置参数类
    
    包含所有训练和模型相关的配置参数，集中管理便于修改
    """

    # ===== 数据集路径配置 =====
    DATABASE_ROOT = '/home/......'  # 数据库根目录（需要根据实际情况修改）

    dic_Sitzmann = {
        'IMG_PATH': DATABASE_ROOT + '/Sitzmann/rotation_imgs/imgs/',  # 图像文件夹路径
        'GAZE_PATH': DATABASE_ROOT + '/Sitzmann/vr/',  # 眼动数据文件夹路径
        'TEST_SET': ['cubemap_0000.png', 'cubemap_0006.png', 'cubemap_0009.png']  # 测试集图像文件名列表
    }

    # ===== 图像处理参数 =====
    image_size = [128, 256]  # 图像尺寸 [高度, 宽度]（像素）
    # 所有图像会被调整到这个尺寸
    # 高度128，宽度256（宽高比2:1，符合等距圆柱投影的宽高比）

    # ===== 训练超参数 =====
    seed = 1234  # 随机种子（保证结果可复现）
    num_epochs = 500  # 训练轮数（整个数据集遍历500次）
    learning_rate = 0.0003  # 初始学习率
    lr_decay = 0.99998  # 学习率衰减率（每个训练step衰减，公式: lr = lr * lr_decay）
    weight_decay = 2.0  # L2正则化权重（防止过拟合，值越大正则化越强）
    mini_batch_size = 64  # 批次大小（每个小批次包含64个样本）

    # ===== KL散度退火参数 =====
    # KL散度退火：训练初期逐渐增加KL项的权重，避免模式崩塌
    annealing_epochs = 10  # KL散度退火轮数（前10个epoch进行退火）
    minimum_annealing_factor = 0.2  # 最小退火因子（退火开始时的KL项权重，例如0.2表示只使用20%的KL项）

    # ===== 模型加载和保存配置 =====
    load_model = None  # 预训练模型路径（用于继续训练或推理，None表示从头训练）
    load_opt = None  # 优化器状态路径（用于继续训练，None表示重新初始化优化器）
    save_root = './model/'  # 模型保存根目录（训练过程中会在此目录保存检查点）

    # ===== CUDA配置 =====
    use_cuda = torch.cuda.is_available()  # 是否使用GPU（自动检测，如果CUDA可用则使用GPU）


# ============================================================================
# 第二部分：添加通道
# ============================================================================

class AddCoordsTh(nn.Module):
    """
    添加坐标通道到输入张量
    这是CoordConv的核心组件，通过在特征图上添加x和y坐标信息，
    帮助CNN学习空间位置相关的特征
    """

    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim  # [128]
        self.y_dim = y_dim  # [256]
        self.with_r = with_r  # 是否添加径向距离r

    def forward(self, input_tensor):
        """
        在输入张量上添加x和y坐标通道（CoordConv技术）
        
        Args:
            input_tensor: 输入特征图, shape = (batch_size, channels, height, width)
                        例如: (64, 3, 128, 256) - 64个样本, 3通道(RGB), 128x256像素
        
        Returns:
            ret: 添加了坐标通道的特征图
                如果with_r=False: shape = (batch_size, channels+2, height, width)
                例如: (64, 5, 128, 256) - 3通道(RGB) + 2通道(x坐标, y坐标)
                如果with_r=True: shape = (batch_size, channels+3, height, width)
                例如: (64, 6, 128, 256) - 额外添加了径向距离r通道
        
        维度变化示例（batch_size=64, 输入3通道）:
            输入: (64, 3, 128, 256)
            -> 添加x坐标通道: (64, 1, 128, 256)
            -> 添加y坐标通道: (64, 1, 128, 256)
            -> 拼接: (64, 3+1+1, 128, 256) = (64, 5, 128, 256)
        """
        batch_size_tensor = input_tensor.shape[0]  # 批次大小, 例如: 64

        # ===== 步骤1: 生成x坐标通道 =====
        # 创建x坐标矩阵，每一列的值相同（列索引）
        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).unsqueeze(-1)  # shape: (1, 256, 1)
        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).unsqueeze(1)  # shape: (1, 1, 128)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(-1)  # shape: (1, 256, 128, 1)
        # 说明: matmul生成256x128的矩阵，每个位置的值是其列索引

        # ===== 步骤2: 生成y坐标通道 =====
        # 创建y坐标矩阵，每一行的值相同（行索引）
        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).unsqueeze(1)  # shape: (1, 128, 1)
        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).unsqueeze(-1)  # shape: (1, 1, 256)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(-1)  # shape: (1, 256, 128, 1)
        # 说明: matmul生成256x128的矩阵，每个位置的值是其行索引

        # ===== 步骤3: 调整维度顺序 =====
        # 将维度从(height, width)调整为(channel, height, width)
        xx_channel = xx_channel.permute(0, 3, 2, 1)  # shape: (1, 1, 128, 256) - [batch, channel, height, width]
        yy_channel = yy_channel.permute(0, 3, 2, 1)  # shape: (1, 1, 128, 256)

        # ===== 步骤4: 归一化坐标到[-1, 1]范围 =====
        # 将坐标从[0, dim-1]归一化到[-1, 1]
        xx_channel = xx_channel.float() / (self.x_dim - 1) * 2 - 1  # x坐标归一化
        yy_channel = yy_channel.float() / (self.y_dim - 1) * 2 - 1  # y坐标归一化
        # 例如: x_dim=128, 坐标0 -> -1.0, 坐标63 -> 0.0, 坐标127 -> 1.0

        # ===== 步骤5: 扩展到批次大小并移动到相同设备 =====
        # 将单样本的坐标通道扩展到整个批次，并移动到输入张量的设备（CPU/GPU）
        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1).to(input_tensor.device)
        # shape: (batch_size, 1, 128, 256), 例如: (64, 1, 128, 256)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1).to(input_tensor.device)
        # shape: (batch_size, 1, 128, 256), 例如: (64, 1, 128, 256)

        # ===== 步骤6: 拼接坐标通道到输入张量 =====
        # 在通道维度上拼接: [原始通道, x坐标, y坐标]
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        # shape: (batch_size, channels+2, height, width)
        # 例如: (64, 3, 128, 256) + (64, 1, 128, 256) + (64, 1, 128, 256) -> (64, 5, 128, 256)

        # ===== 可选步骤: 添加径向距离r =====
        if self.with_r:
            # 计算每个位置到中心(0.5, 0.5)的径向距离
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            # shape: (batch_size, 1, height, width)
            ret = torch.cat([ret, rr], dim=1)
            # shape: (batch_size, channels+3, height, width), 例如: (64, 6, 128, 256)

        return ret


# ============================================================================
# 第三部分：球面CNN (Sphere CNN)
# ============================================================================

@lru_cache(None)
def get_xy(delta_phi, delta_theta):
    """
    计算球面卷积核的采样模式（在切平面上的偏移量）
    
    Args:
        delta_phi: 纬度的步长（弧度）
        delta_theta: 经度的步长（弧度）
    
    Returns:
        采样模式, shape = (3, 3, 2)
        表示3x3卷积核的9个采样点在切平面上的偏移量(x, y)
    
    说明：
    - 这是球面卷积的核心：在球面上进行卷积时，需要计算卷积核的采样位置
    - 由于球面的几何特性，不能直接使用欧几里得空间的坐标
    - 先在切平面上计算偏移量，然后投影回球面
    - 中心点(1,1)是占位符，后面会被实际位置覆盖
    
    几何原理：
    - 对于球面上的一个点，计算其切平面
    - 在切平面上计算3x3卷积核的偏移量
    - 这些偏移量表示从中心点出发的9个方向
    
    返回的数组结构：
    [
        [左上, 上, 右上],
        [左,   中, 右],
        [左下, 下, 右下]
    ]
    每个位置是(x, y)偏移量
    """
    return np.array([
        [
            # 第一行（上方）
            (-tan(delta_theta), 1 / cos(delta_theta) * tan(delta_phi)),  # 左上
            (0, tan(delta_phi)),  # 上
            (tan(delta_theta), 1 / cos(delta_theta) * tan(delta_phi)),  # 右上
        ],
        [
            # 第二行（中间）
            (-tan(delta_theta), 0),  # 左
            (1, 1),  # 中心（占位符，会被实际位置覆盖）
            (tan(delta_theta), 0),  # 右
        ],
        [
            # 第三行（下方）
            (-tan(delta_theta), -1 / cos(delta_theta) * tan(delta_phi)),  # 左下
            (0, -tan(delta_phi)),  # 下
            (tan(delta_theta), -1 / cos(delta_theta) * tan(delta_phi)),  # 右下
        ]
    ])
    # shape: (3, 3, 2) - 3x3卷积核，每个位置有2个坐标(x, y)


@lru_cache(None)
def cal_index(h, w, img_r, img_c):
    """
    计算球面卷积核的采样索引
    
    功能：为等距圆柱投影图像上的一个像素点，计算3x3卷积核的9个采样位置
    
    Args:
        h: 图像高度（像素），例如: 128
        w: 图像宽度（像素），例如: 256
        img_r: 当前像素的行位置（0到h-1）
        img_c: 当前像素的列位置（0到w-1）
    
    Returns:
        new_result: 采样位置的像素坐标, shape = (3, 3, 2)
                    每个位置是(row, col)像素坐标
                    例如: (3, 3, 2) - 3x3卷积核的9个采样位置
    
    处理流程：
    1. 将像素坐标转换为球面坐标（纬度phi、经度theta）
    2. 计算切平面上的3x3偏移量
    3. 将偏移量投影回球面，得到新的球面坐标
    4. 将新球面坐标转换回像素坐标
    5. 处理边界连续性（360度图像左右边界相邻）
    
    维度变化:
        输入: h, w, img_r, img_c（标量）
        中间计算: 
            phi, theta: 标量（弧度）
            xys: (3, 3, 2) - 切平面偏移量
            new_phi, new_theta: (3, 3) - 新球面坐标
            new_r, new_c: (3, 3) - 新像素坐标
        输出: (3, 3, 2) - 采样位置的像素坐标
    """
    # ===== 步骤1: 像素坐标转换为球面角度（弧度） =====
    # 等距圆柱投影的逆变换：像素坐标 -> 球面坐标
    phi = -((img_r + 0.5) / h * pi - pi / 2)  # 纬度（弧度），范围[-π/2, π/2]
    # 公式: phi = -((r + 0.5) / h * π - π/2)
    # 例如: h=128, img_r=0 -> phi ≈ π/2（北极）
    #      h=128, img_r=64 -> phi ≈ 0（赤道）
    #      h=128, img_r=127 -> phi ≈ -π/2（南极）

    theta = (img_c + 0.5) / w * 2 * pi - pi  # 经度（弧度），范围[-π, π]
    # 公式: theta = (c + 0.5) / w * 2π - π
    # 例如: w=256, img_c=0 -> theta ≈ -π（左边界）
    #      w=256, img_c=128 -> theta ≈ 0（中央子午线）
    #      w=256, img_c=255 -> theta ≈ π（右边界）

    # ===== 步骤2: 计算步长（相邻像素之间的角度差） =====
    delta_phi = pi / h  # 纬度步长（弧度）
    # 例如: h=128 -> delta_phi ≈ 0.02454弧度 ≈ 1.406度

    delta_theta = 2 * pi / w  # 经度步长（弧度）
    # 例如: w=256 -> delta_theta ≈ 0.02454弧度 ≈ 1.406度

    # ===== 步骤3: 获取3x3采样模式（切平面上的偏移量） =====
    xys = get_xy(delta_phi, delta_theta)
    # shape: (3, 3, 2) - 3x3卷积核在切平面上的偏移量(x, y)

    x = xys[..., 0]  # x偏移量, shape: (3, 3)
    y = xys[..., 1]  # y偏移量, shape: (3, 3)

    # ===== 步骤4: 将切平面偏移量投影回球面，计算新的球面坐标 =====
    # 这是球面几何的核心：在切平面上计算偏移，然后投影回球面
    rho = np.sqrt(x ** 2 + y ** 2)  # 偏移量的距离, shape: (3, 3)
    v = arctan(rho)  # 角度, shape: (3, 3)

    # 计算新的纬度（考虑球面几何）
    new_phi = arcsin(cos(v) * sin(phi) + y * sin(v) * cos(phi) / rho)
    # shape: (3, 3) - 新纬度的数组

    # 计算新的经度（考虑球面几何）
    new_theta = theta + arctan(x * sin(v) / (rho * cos(phi) * cos(v) - y * sin(phi) * sin(v)))
    # shape: (3, 3) - 新经度的数组

    # ===== 步骤5: 球面坐标转回像素坐标 =====
    # 等距圆柱投影：球面坐标 -> 像素坐标
    new_r = (-new_phi + pi / 2) * h / pi - 0.5
    # 纬度 -> 行坐标
    # shape: (3, 3)

    new_c = (new_theta + pi) * w / 2 / pi - 0.5
    # 经度 -> 列坐标
    # shape: (3, 3)

    # ===== 步骤6: 处理等距圆柱投影的左右边界连续性 =====
    # 360度图像的左右边界是相邻的（经度-180和+180是同一个位置）
    # 使用模运算处理边界越界
    new_c = (new_c + w) % w
    # 例如: new_c = -1 -> (256-1) % 256 = 255（转到右边界）
    #      new_c = 257 -> (256+257) % 256 = 1（转到左边界）

    # ===== 步骤7: 组合并修正中心点 =====
    new_result = np.stack([new_r, new_c], axis=-1)
    # shape: (3, 3, 2) - 每个位置是(row, col)

    new_result[1, 1] = (img_r, img_c)  # 中心位置保持原位置（覆盖占位符(1,1)）
    # 确保中心点就是原始位置

    return new_result  # shape: (3, 3, 2) - 9个采样位置的像素坐标


@lru_cache(None)
def _gen_filters_coordinates(h, w, stride):
    """
    生成所有位置的卷积核采样坐标（内部函数，使用缓存）
    
    Args:
        h: 图像高度
        w: 图像宽度
        stride: 步长（每隔stride个像素计算一次）
    
    Returns:
        co: 所有位置的采样坐标, shape = (2, H/stride, W/stride, 3, 3)
            维度0: 2表示(row, col)坐标
            维度1-2: (H/stride, W/stride)表示输出位置
            维度3-4: (3, 3)表示3x3卷积核的采样位置
    
    功能：
    - 为图像的每个输出位置（按stride采样），计算3x3卷积核的采样坐标
    - 使用lru_cache缓存结果，避免重复计算
    
    维度变化:
        遍历: H/stride × W/stride 个位置
        每个位置: cal_index返回 (3, 3, 2)
        组合: (H/stride, W/stride, 3, 3, 2)
        转置: (2, H/stride, W/stride, 3, 3)
    """
    # 为每个输出位置（按stride采样）计算3x3卷积核的采样坐标
    co = np.array([[cal_index(h, w, i, j) for j in range(0, w, stride)]
                   for i in range(0, h, stride)])
    # 列表推导式遍历所有位置: i从0到h（步长stride），j从0到w（步长stride）
    # 每个位置调用cal_index计算3x3采样坐标
    # shape: (H/stride, W/stride, 3, 3, 2)
    # 例如: h=128, w=256, stride=2 -> (64, 128, 3, 3, 2)

    # 转置维度顺序，方便grid_sample使用
    # 从 (H/stride, W/stride, 3, 3, 2) 转置为 (2, H/stride, W/stride, 3, 3)
    # 这样第一维是坐标(row, col)，后面是位置和卷积核
    return np.ascontiguousarray(co.transpose([4, 0, 1, 2, 3]))
    # shape: (2, H/stride, W/stride, 3, 3)
    # 例如: (2, 64, 128, 3, 3)


def gen_filters_coordinates(h, w, stride=1):
    """
    生成卷积核采样坐标（对外接口）
    
    Args:
        h: 图像高度（整数）
        w: 图像宽度（整数）
        stride: 步长，默认为1
    
    Returns:
        coordinates: 采样坐标, shape = (2, H/stride, W/stride, 3, 3)
                     维度0: 2表示(row, col)坐标
                     维度1-2: (H/stride, W/stride)表示输出位置
                     维度3-4: (3, 3)表示3x3卷积核
    
    用途：
    - 计算球面卷积需要的采样坐标
    - 结果会被缓存（通过lru_cache），提高效率
    
    示例（h=128, w=256, stride=2）:
        输出: (2, 64, 128, 3, 3)
        - 2: 坐标维度（行和列）
        - 64: 输出高度位置数（128/2）
        - 128: 输出宽度位置数（256/2）
        - 3, 3: 每个位置的3x3卷积核采样点
    """
    assert (isinstance(h, int) and isinstance(w, int))
    return _gen_filters_coordinates(h, w, stride).copy()


def gen_grid_coordinates(h, w, stride=1):
    """
    生成用于grid_sample的采样网格坐标
    
    Args:
        h: 图像高度
        w: 图像宽度
        stride: 步长
    
    Returns:
        coordinates: grid_sample需要的网格坐标, shape = (1, H/stride*3, W/stride*3, 2)
                     归一化到[-1, 1]范围，格式符合F.grid_sample的要求
    
    处理流程：
    1. 获取所有位置的采样坐标
    2. 归一化到[-1, 1]范围（grid_sample要求）
    3. 调整维度顺序（grid_sample要求坐标在最后）
    4. 重塑为grid_sample需要的格式
    
    维度变化:
        输入: h, w, stride
        步骤1: gen_filters_coordinates -> (2, H/stride, W/stride, 3, 3)
        步骤2: 归一化 -> (2, H/stride, W/stride, 3, 3)
        步骤3: 反转+转置 -> (H/stride, 3, W/stride, 3, 2)
        步骤4: reshape -> (1, H/stride*3, W/stride*3, 2)
    
    示例（h=128, w=256, stride=2）:
        输入: h=128, w=256, stride=2
        输出: (1, 192, 384, 2)
        - 1: batch维度（单样本）
        - 192: 高度 = 64*3 = 192（64个输出位置，每个3个采样点）
        - 384: 宽度 = 128*3 = 384（128个输出位置，每个3个采样点）
        - 2: 坐标(x, y)，值域[-1, 1]
    """
    # ===== 步骤1: 获取所有位置的采样坐标 =====
    coordinates = gen_filters_coordinates(h, w, stride).copy()
    # shape: (2, H/stride, W/stride, 3, 3)
    # 例如: (2, 64, 128, 3, 3)
    # 维度0: 2表示(row, col)坐标
    # 维度1-2: 输出位置
    # 维度3-4: 3x3卷积核

    # ===== 步骤2: 归一化坐标到[-1, 1]范围 =====
    # grid_sample要求坐标在[-1, 1]范围
    coordinates[0] = (coordinates[0] * 2 / h) - 1  # 行坐标归一化
    # 从[0, h-1]映射到[-1, 1]
    # 例如: row=0 -> -1, row=h/2 -> 0, row=h-1 -> 1

    coordinates[1] = (coordinates[1] * 2 / w) - 1  # 列坐标归一化
    # 从[0, w-1]映射到[-1, 1]

    # ===== 步骤3: 调整维度顺序 =====
    coordinates = coordinates[::-1]  # 反转第一维：从(row, col)变为(col, row)
    # shape: (2, H/stride, W/stride, 3, 3) -> (2, H/stride, W/stride, 3, 3)（但顺序变了）

    coordinates = coordinates.transpose(1, 3, 2, 4, 0)
    # 转置: (2, H/stride, W/stride, 3, 3) -> (H/stride, 3, W/stride, 3, 2)
    # 重新组织维度，将坐标移到最后
    sz = coordinates.shape
    # shape: (H/stride, 3, W/stride, 3, 2)
    # 例如: (64, 3, 128, 3, 2)

    # ===== 步骤4: 重塑为grid_sample需要的格式 =====
    # 将(H/stride, 3, W/stride, 3, 2)重塑为(1, H/stride*3, W/stride*3, 2)
    coordinates = coordinates.reshape(1, sz[0] * sz[1], sz[2] * sz[3], sz[4])
    # (H/stride, 3, W/stride, 3, 2) -> (1, H/stride*3, W/stride*3, 2)
    # 例如: (64, 3, 128, 3, 2) -> (1, 192, 384, 2)
    # 每个输出位置的3x3采样点被展平成一维

    return coordinates.copy()
    # shape: (1, H/stride*3, W/stride*3, 2)
    # 这个格式可以直接用于F.grid_sample


class SphereConv2D(nn.Module):
    """
    球面卷积层
    专门为360度等距圆柱投影图像设计的卷积层，考虑了球面的几何特性
    注意：只支持3x3卷积核
    """

    def __init__(self, in_c, out_c, stride=1, bias=True, mode='bilinear'):
        super(SphereConv2D, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.mode = mode  # 插值模式：'bilinear'或'nearest'

        # 卷积权重参数
        self.weight = Parameter(torch.Tensor(out_c, in_c, 3, 3))
        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)

        # 缓存采样网格，避免重复计算
        self.grid_shape = None
        self.grid = None

        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重参数"""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        """
        前向传播：执行球面卷积
        
        Args:
            x: 输入特征图, shape = (batch_size, in_channels, height, width)
               例如: (64, 5, 128, 256) - 64个样本, 5通道, 128x256像素
        
        Returns:
            x: 输出特征图, shape = (batch_size, out_channels, height/stride, width/stride)
               例如: stride=2时, (64, 64, 64, 128) - 高度和宽度各减半
        
        工作原理：
            1. 根据球面几何计算3x3卷积核的采样位置
            2. 使用grid_sample根据这些位置进行采样
            3. 对采样结果应用3x3卷积（通过stride=3实现）
        
        维度变化示例（假设stride=2）:
            输入: (64, 5, 128, 256)
            -> grid_sample采样: (64, 5, 64*3, 128*3) = (64, 5, 192, 384)
               (因为每个输出位置需要3x3=9个采样点，所以尺寸扩大了3倍)
            -> conv2d(stride=3): (64, 64, 192/3, 384/3) = (64, 64, 64, 128)
               (stride=3将每个3x3区域压缩为1个点)
        """
        # ===== 步骤1: 生成/更新采样网格 =====
        # 如果输入尺寸改变，需要重新生成采样网格（缓存在self.grid中）
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])  # 记录当前输入尺寸, 例如: (128, 256)
            # 生成采样网格坐标，考虑球面几何
            coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
            # shape: (1, H/stride*3, W/stride*3, 2) - 2表示(x,y)坐标
            # 例如: stride=2时, (1, 192, 384, 2)

            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        # ===== 步骤2: 扩展到批次大小 =====
        # 将单样本的网格扩展到整个批次
        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)
            # shape: (batch_size, H/stride*3, W/stride*3, 2)
            # 例如: (64, 192, 384, 2)

        # ===== 步骤3: 使用grid_sample进行球面采样 =====
        # 根据球面几何计算的采样位置，从输入特征图中采样
        x = F.grid_sample(x, grid, mode=self.mode, align_corners=True)
        # 输入x: (batch_size, in_channels, height, width), 例如: (64, 5, 128, 256)
        # 输出x: (batch_size, in_channels, H/stride*3, W/stride*3)
        # 例如: (64, 5, 192, 384) - 每个输出位置对应3x3=9个采样点

        # ===== 步骤4: 应用3x3卷积 =====
        # 对采样结果应用3x3卷积，stride=3将每个3x3区域压缩为1个输出点
        x = F.conv2d(x, self.weight, self.bias, stride=3)
        # 输入x: (batch_size, in_channels, H/stride*3, W/stride*3), 例如: (64, 5, 192, 384)
        # weight: (out_channels, in_channels, 3, 3), 例如: (64, 5, 3, 3)
        # 输出x: (batch_size, out_channels, H/stride, W/stride)
        # 例如: (64, 64, 192/3, 384/3) = (64, 64, 64, 128)

        return x


class Sphere_CNN(nn.Module):
    """
    球面CNN特征提取器
    用于从360度图像中提取特征，输出固定维度的特征向量 100维
    """

    def __init__(self, out_put_dim):
        super(Sphere_CNN, self).__init__()
        self.output_dim = out_put_dim  # [100]

        # 添加坐标通道（CoordConv）
        self.coord_conv = AddCoordsTh(x_dim=128, y_dim=256, with_r=False)

        # 图像特征提取管道
        # 第一层：5通道输入（RGB 3通道 + x坐标 + y坐标）-> 64通道
        self.image_conv1 = SphereConv2D(5, 64, stride=2, bias=False)
        self.image_norm1 = nn.BatchNorm2d(64)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        # 第二层：64 -> 128通道
        self.image_conv2 = SphereConv2D(64, 128, stride=2, bias=False)
        self.image_norm2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        # 第三层：128 -> 256通道
        self.image_conv3 = SphereConv2D(128, 256, stride=2, bias=False)
        self.image_norm3 = nn.BatchNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        # 第四层：256 -> 512通道
        self.image_conv3_5 = SphereConv2D(256, 512, stride=2, bias=False)
        self.image_norm3_5 = nn.BatchNorm2d(512)
        self.leaky_relu3_5 = nn.LeakyReLU(0.2, inplace=True)

        # 使用标准卷积进行进一步特征提取
        self.image_conv4 = nn.Conv2d(512, 256, 4, 2, 1, bias=False)
        self.image_norm4 = nn.BatchNorm2d(256)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv5 = nn.Conv2d(256, 64, 4, 2, 1, bias=False)
        self.image_norm5 = nn.BatchNorm2d(64)
        self.leaky_relu5 = nn.LeakyReLU(0.2, inplace=True)

        # 全连接层：将特征图展平并映射到输出维度
        self.fc1 = nn.Linear(64 * 4 * 2, self.output_dim)
        self.flatten = nn.Flatten()
        self.activation = nn.Tanh()

    def forward(self, image):
        """
        前向传播：从360度图像提取特征向量
        
        Args:
            image: 输入图像, shape = (batch_size, 3, height, width)
                   例如: (64, 3, 128, 256) - 64个样本, RGB 3通道, 128x256像素
        
        Returns:
            x: 图像特征向量, shape = (batch_size, output_dim)
               例如: (64, 100) - 64个样本, 每个样本100维特征
        
        维度变化详细过程（假设batch_size=64）:
            输入: (64, 3, 128, 256)
            
            1. 添加坐标通道: (64, 3, 128, 256) -> (64, 5, 128, 256)  [+2通道: x, y坐标]
            
            2. SphereConv2D(5->64, stride=2): (64, 5, 128, 256) -> (64, 64, 64, 128)
               BatchNorm + LeakyReLU: (64, 64, 64, 128) -> (64, 64, 64, 128)
            
            3. SphereConv2D(64->128, stride=2): (64, 64, 64, 128) -> (64, 128, 32, 64)
               BatchNorm + LeakyReLU: (64, 128, 32, 64) -> (64, 128, 32, 64)
            
            4. SphereConv2D(128->256, stride=2): (64, 128, 32, 64) -> (64, 256, 16, 32)
               BatchNorm + LeakyReLU: (64, 256, 16, 32) -> (64, 256, 16, 32)
            
            5. SphereConv2D(256->512, stride=2): (64, 256, 16, 32) -> (64, 512, 8, 16)
               BatchNorm + LeakyReLU: (64, 512, 8, 16) -> (64, 512, 8, 16)
            
            6. Conv2d(512->256, 4x4, stride=2, pad=1): (64, 512, 8, 16) -> (64, 256, 4, 8)
               BatchNorm + LeakyReLU: (64, 256, 4, 8) -> (64, 256, 4, 8)
            
            7. Conv2d(256->64, 4x4, stride=2, pad=1): (64, 256, 4, 8) -> (64, 64, 2, 4)
               BatchNorm + LeakyReLU: (64, 64, 2, 4) -> (64, 64, 2, 4)
            
            8. Flatten: (64, 64, 2, 4) -> (64, 64*2*4) = (64, 512)
            
            9. Linear(512->100) + Tanh: (64, 512) -> (64, 100)
            
            输出: (64, 100)
        """
        x = image  # 输入: (batch_size, 3, 128, 256), 例如: (64, 3, 128, 256)

        # ===== 步骤1: 添加坐标通道 =====
        # 使用CoordConv技术，添加x和y坐标通道，帮助CNN学习空间位置特征
        x = self.coord_conv(x)
        # 维度变化: (batch_size, 3, 128, 256) -> (batch_size, 5, 128, 256)
        # 例如: (64, 3, 128, 256) -> (64, 5, 128, 256)

        # ===== 步骤2-5: 球面CNN特征提取 =====
        # 使用球面卷积（考虑360度图像的几何特性）逐层提取特征

        # 第一层: 5通道 -> 64通道, stride=2 (尺寸减半)
        x = self.leaky_relu1(self.image_norm1(self.image_conv1(x)))
        # image_conv1: (64, 5, 128, 256) -> (64, 64, 64, 128)
        # BatchNorm: (64, 64, 64, 128) -> (64, 64, 64, 128) (归一化)
        # LeakyReLU: (64, 64, 64, 128) -> (64, 64, 64, 128) (激活)

        # 第二层: 64通道 -> 128通道, stride=2
        x = self.leaky_relu2(self.image_norm2(self.image_conv2(x)))
        # (64, 64, 64, 128) -> (64, 128, 32, 64)

        # 第三层: 128通道 -> 256通道, stride=2
        x = self.leaky_relu3(self.image_norm3(self.image_conv3(x)))
        # (64, 128, 32, 64) -> (64, 256, 16, 32)

        # 第四层: 256通道 -> 512通道, stride=2
        x = self.leaky_relu3_5(self.image_norm3_5(self.image_conv3_5(x)))
        # (64, 256, 16, 32) -> (64, 512, 8, 16)

        # ===== 步骤6-7: 标准CNN进一步提取特征 =====
        # 使用标准卷积（不再是球面卷积）进一步压缩特征

        # 第五层: 512通道 -> 256通道, 4x4卷积, stride=2
        x = self.leaky_relu4(self.image_norm4(self.image_conv4(x)))
        # (64, 512, 8, 16) -> (64, 256, 4, 8)

        # 第六层: 256通道 -> 64通道, 4x4卷积, stride=2
        x = self.leaky_relu5(self.image_norm5(self.image_conv5(x)))
        # (64, 256, 4, 8) -> (64, 64, 2, 4)

        # ===== 步骤8-9: 展平并映射到输出维度 =====
        # 将特征图展平为一维向量，然后通过全连接层映射到目标维度
        x = self.activation(self.fc1(self.flatten(x)))
        # flatten: (64, 64, 2, 4) -> (64, 64*2*4) = (64, 512)
        # fc1 (Linear): (64, 512) -> (64, 100)
        # activation (Tanh): (64, 100) -> (64, 100) (值域[-1, 1])

        return x  # 输出: (batch_size, output_dim), 例如: (64, 100)


# ============================================================================
# 第四部分：深度马尔可夫模型 (Deep Markov Model)
# ============================================================================

class Emitter(nn.Module):
    """
    发射器：p(x_t | z_t)
    从隐状态z_t生成观测值x_t（扫描路径点，即眼睛位置的3D坐标）
    
    作用：将隐藏的注意力状态（隐状态）转换为可观测的眼睛位置坐标
    比喻：根据"你在想什么"（隐状态），预测"你的眼睛在看哪里"（观测）
    """

    def __init__(self, gaze_dim, z_dim, hidden_dim):
        """
        初始化发射器网络
        
        Args:
            gaze_dim: 观测维度，即眼睛位置的坐标维度 = 3 (x, y, z)
            z_dim: 隐状态维度 = 100
            hidden_dim: 隐藏层维度 = 100
        """
        super().__init__()
        # 网络结构：z_dim -> hidden_dim -> gaze_dim
        # 将隐状态（100维）映射到观测空间（3维）

        # 第一层：隐状态空间 -> 隐藏空间
        self.lin_em_z_to_hidden = nn.Linear(z_dim, hidden_dim)  # (100) -> (100)

        # 第二层：隐藏空间 -> 观测空间（均值）
        self.lin_hidden_to_gaze = nn.Linear(hidden_dim, gaze_dim)  # (100) -> (3)

        # 标准差网络：从均值计算标准差
        self.lin_gaze_sig = nn.Linear(gaze_dim, gaze_dim)  # (3) -> (3)

        # 激活函数
        self.relu = nn.ReLU()  # ReLU激活
        self.softplus = nn.Softplus()  # Softplus确保输出>0（用于标准差）
        self.sigmod = nn.Sigmoid()  # Sigmoid将输出映射到[0,1]

    def forward(self, z_t):
        """
        从隐状态z_t生成观测值的均值和方差
        
        Args:
            z_t: 隐状态, shape = (batch_size, z_dim)
                 例如: (64, 100) - 64个样本，每个100维隐状态
        
        Returns:
            mu: 观测的均值（眼睛位置的3D坐标均值）, shape = (batch_size, gaze_dim)
                例如: (64, 3) - 值域[-1, 1]（归一化的坐标）
            sigma: 观测的标准差, shape = (batch_size, gaze_dim)
                   例如: (64, 3) - 值域>0
        
        维度变化详细过程（假设batch_size=64）:
            输入 z_t: (64, 100)
            
            计算均值 mu:
                lin_em_z_to_hidden: (64, 100) -> (64, 100)
                relu: (64, 100) -> (64, 100)
                lin_hidden_to_gaze: (64, 100) -> (64, 3)
                sigmod: (64, 3) -> (64, 3) [值域: 0~1]
                * 2 - 1: (64, 3) -> (64, 3) [值域: -1~1]
                输出 mu: (64, 3)
            
            计算标准差 sigma:
                relu(mu): (64, 3) -> (64, 3)
                lin_gaze_sig: (64, 3) -> (64, 3)
                softplus: (64, 3) -> (64, 3) [值域: >0]
                输出 sigma: (64, 3)
        """
        # ===== 步骤1: 计算均值mu（眼睛位置的3D坐标均值） =====
        # 将隐状态映射到观测空间，输出归一化到[-1, 1]
        hidden = self.relu(self.lin_em_z_to_hidden(z_t))
        # z_t: (batch_size, 100) -> hidden: (batch_size, 100)
        # 例如: (64, 100) -> (64, 100)

        mu = self.sigmod(self.lin_hidden_to_gaze(hidden)) * 2 - 1
        # hidden: (batch_size, 100) -> (batch_size, 3)
        # sigmod: (batch_size, 3) -> (batch_size, 3) [0~1]
        # * 2 - 1: (batch_size, 3) -> (batch_size, 3) [-1~1]
        # 例如: (64, 100) -> (64, 3)
        # mu的每个元素表示眼睛在x, y, z方向的归一化坐标（范围[-1, 1]）

        # ===== 步骤2: 计算标准差sigma（预测的不确定度） =====
        # 基于均值计算标准差，表示预测的置信度
        sigma = self.softplus(self.lin_gaze_sig(self.relu(mu)))
        # mu: (batch_size, 3) -> relu(mu): (batch_size, 3)
        # lin_gaze_sig: (batch_size, 3) -> (batch_size, 3)
        # softplus: (batch_size, 3) -> (batch_size, 3) [>0]
        # 例如: (64, 3) -> (64, 3)
        # sigma的每个元素表示对应坐标方向预测的不确定度

        return mu, sigma
        # mu: (batch_size, 3) - 眼睛位置的3D坐标均值
        # sigma: (batch_size, 3) - 对应坐标的标准差
        # 例如: mu=(64,3), sigma=(64,3)
        # 最终返回高斯分布 N(mu, sigma^2)，用于采样眼睛位置


class GatedTransition(nn.Module):
    """
    门控转移：p(z_t | z_{t-1})
    定义隐状态之间的转移分布，使用门控机制控制转移强度
    """

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # 门控网络：决定转移的权重
        self.lin_gate_z_to_hidden_dim = nn.Linear(z_dim, hidden_dim)  # [100]=>[200]
        self.lin_gate_hidden_dim_to_z = nn.Linear(hidden_dim, z_dim)  # [200]=>[100] 输出的100维,每个代表一个权重

        # 转移网络：计算新的隐状态
        self.lin_trans_2z_to_hidden = nn.Linear(2 * z_dim, hidden_dim)  # [200]=>[200] 处理合并信息
        self.lin_trans_hidden_to_z = nn.Linear(hidden_dim, z_dim)  # [200]=>[100] 得到新的状态

        # 方差网络
        self.lin_sig = nn.Linear(z_dim, z_dim)  # [100]=>[100] 评估预测的可靠程度

        # 恒等映射（用于门控）
        self.lin_z_to_mu = nn.Linear(z_dim, z_dim)  # [100]=>[100] 这几行保持张量不变,因为y=kx+b,k=1,b=0
        self.lin_z_to_mu.weight.data = torch.eye(z_dim)
        self.lin_z_to_mu.bias.data = torch.zeros(z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, img_feature=None):
        """
        计算从z_{t-1}到z_t的转移分布
        使用门控机制在恒等映射和转移网络之间进行插值
        
        Args:
            z_t_1: 前一个隐状态, shape = (batch_size, z_dim), 例如 (64, 100)
            img_feature: 图像特征, shape = (batch_size, z_dim), 例如 (64, 100)
            
        Returns:
            mu: 下一个状态的均值, shape = (batch_size, z_dim), 例如 (64, 100)
            sigma: 下一个状态的标准差, shape = (batch_size, z_dim), 例如 (64, 100)
            
        注意: 
            - 第一个维度64是batch_size(批次大小),表示同时处理64个样本
            - 第二个维度100是z_dim(隐状态维度),表示每个样本有100个特征维度
            - PyTorch的Linear层会保持batch维度不变,所以输出shape与输入相同
        """
        # 将前一个隐状态和图像特征拼接
        # z_t_1: (64, 100), img_feature: (64, 100) -> z_t_1_img: (64, 200)
        z_t_1_img = torch.cat((z_t_1, img_feature), dim=1)  # 拼接两个100维信息
        # _z_t: (64, 200) -> (64, 200) -> (64, 100) 转移网络预测的新状态 只是一个简单的MLP
        _z_t = self.lin_trans_hidden_to_z(self.relu(self.lin_trans_2z_to_hidden(z_t_1_img)))

        # 不确定性加权：计算门控权重
        # z_t_1: (64, 100) -> (64, 200) -> (64, 100) -> weight: (64, 100)
        weight = torch.sigmoid(
            self.lin_gate_hidden_dim_to_z(self.relu(self.lin_gate_z_to_hidden_dim(z_t_1))))  # 计算激进策略的预测 "门控机制灵魂所在"

        # 高斯参数：在恒等映射和转移网络之间插值
        # z_t_1: (64, 100) -> (64, 100), _z_t: (64, 100), weight: (64, 100)
        # mu: (64, 100) 算是一个小创新点吧,保守部分+激进部分
        mu = (1 - weight) * self.lin_z_to_mu(z_t_1) + weight * _z_t  # 也很重要
        # _z_t: (64, 100) -> (64, 100) -> sigma: (64, 100) 计算标准差,因为返回的是一个概率分布,sigma越大约不确定
        sigma = self.softplus(self.lin_sig(self.relu(_z_t)))
        return mu, sigma  # mu标识下一个状态的平均值,sigma表示预测的不确定度, shape都是(64, 100)


class Combiner(nn.Module):
    """
    组合器：q(z_t | z_{t-1}, x_{t:T})
    变分后验分布，结合前一个隐状态和未来观测信息
    
    与GatedTransition的区别：
    - GatedTransition：在model中使用，只能看过去（z_{t-1} + img_feature）
    - Combiner：在guide中使用，训练时可以看到未来（z_{t-1} + RNN编码的未来信息）
    
    比喻：GatedTransition像普通人预测，Combiner像能看未来的时间旅行者预测
    """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        # 将隐状态转换到RNN的隐藏空间（用于与RNN输出融合）
        self.lin_comb_z_to_hidden = nn.Linear(z_dim, rnn_dim)  # (100) -> (600)
        # 将融合后的信息转换回隐状态空间（得到均值）
        self.lin_hidden_to_mu = nn.Linear(rnn_dim, z_dim)  # (600) -> (100)
        # 将融合后的信息转换回隐状态空间（得到标准差）
        self.lin_hidden_to_sig = nn.Linear(rnn_dim, z_dim)  # (600) -> (100)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        结合前一个隐状态z_{t-1}和RNN隐藏状态h_rnn
        输出变分后验的均值和方差
        
        Args:
            z_t_1: 前一个隐状态, shape = (batch_size, z_dim), 例如 (64, 100)
                   代表"过去的信息"（刚才的状态是什么）
            h_rnn: RNN编码的未来观测信息, shape = (batch_size, rnn_dim), 例如 (64, 600)
                   代表"未来的信息"（RNN已经处理了未来所有时间步的观测）
        
        Returns:
            mu: 变分后验的均值, shape = (batch_size, z_dim), 例如 (64, 100)
            sigma: 变分后验的标准差, shape = (batch_size, z_dim), 例如 (64, 100)
        
        工作原理：
            1. 将z_t_1转换到RNN空间：z_dim (100) -> rnn_dim (600)
            2. 简单平均融合：0.5 * (过去的转换结果 + 未来的RNN信息)
            3. 从融合结果计算均值和标准差
        """
        # 步骤1：将过去的状态转换到RNN的隐藏空间
        # z_t_1: (64, 100) -> (64, 600) -> tanh -> h_past: (64, 600)
        h_past = self.tanh(self.lin_comb_z_to_hidden(z_t_1))

        # 步骤2：平均融合两种信息（50%过去 + 50%未来）
        # h_past: (64, 600), h_rnn: (64, 600) -> h_combined: (64, 600)
        # 为什么用0.5？给过去和未来相等的权重，简单有效
        h_combined = 0.5 * (h_past + h_rnn)

        # 步骤3：计算均值（将融合后的信息转换回隐状态空间）
        # h_combined: (64, 600) -> mu: (64, 100)
        mu = self.lin_hidden_to_mu(h_combined)

        # 步骤4：计算标准差（预测的不确定度）
        # h_combined: (64, 600) -> (64, 100) -> softplus -> sigma: (64, 100)
        # softplus确保sigma > 0（标准差必须为正）
        sigma = self.softplus(self.lin_hidden_to_sig(h_combined))

        return mu, sigma


class DMM(nn.Module):
    """
    深度马尔可夫模型 (Deep Markov Model)
    用于建模360度图像上的扫描路径序列
    
    模型结构：
    - 使用球面CNN提取图像特征
    - 使用RNN处理观测序列（用于变分推断）
    - 使用门控转移建模隐状态转移（生成模型）
    - 使用组合器建模变分后验（推断模型）
    - 使用发射器从隐状态生成观测
    
    核心思想：
    - 隐状态z_t（隐藏的注意力状态）驱动观测x_t（眼睛位置）
    - model：正向生成，使用GatedTransition（只能看过去）
    - guide：反向推断，使用Combiner（训练时可以看未来）
    """

    def __init__(
            self,
            input_dim=3,  # 输入维度（3D坐标：x, y, z）
            z_dim=100,  # 隐状态维度（每个隐状态有100个特征）
            emission_dim=100,  # 发射器隐藏层维度
            transition_dim=200,  # 转移网络隐藏层维度
            rnn_dim=600,  # RNN隐藏层维度（用于编码未来信息）
            num_layers=1,  # RNN层数
            rnn_dropout_rate=0.1,  # RNN dropout率
            use_cuda=False,
    ):
        super().__init__()

        # ===== 1. 图像特征提取 =====
        # 球面CNN：从360度图像中提取特征（图像 → 100维特征向量）
        self.cnn = Sphere_CNN(out_put_dim=z_dim)

        # ===== 2. 核心模型组件 =====
        # Emitter：从隐状态生成观测 p(x_t | z_t)
        # 输入：隐状态z_t (100维) → 输出：眼睛位置x_t的分布参数 (3维坐标的mu和sigma)
        self.emitter = Emitter(input_dim, z_dim, emission_dim)

        # GatedTransition：建模隐状态转移 p(z_t | z_{t-1})（用于生成模型）
        # 输入：前一个隐状态z_{t-1} + 图像特征 → 输出：下一个隐状态的分布参数
        # 特点：只能看"过去"和"图像"，不能看未来
        self.trans = GatedTransition(z_dim, transition_dim)

        # Combiner：变分后验 q(z_t | z_{t-1}, x_{t:T})（用于推断模型）
        # 输入：前一个隐状态z_{t-1} + RNN编码的未来信息 → 输出：当前隐状态的分布参数
        # 特点：训练时可以看"未来"信息（通过RNN）
        self.combiner = Combiner(z_dim, rnn_dim)

        # ===== 3. 输入处理层 =====
        # 将3D坐标(x,y,z)转换为隐状态空间(100维)
        self.input_to_z_dim = nn.Linear(input_dim, z_dim)
        # 将两个100维向量合并为一个100维向量（用于初始化）
        self.twoZ_to_z_dim = nn.Linear(2 * z_dim, z_dim)
        self.tanh = nn.Tanh()

        # ===== 4. RNN：用于变分推断 =====
        # 处理反向序列，编码未来观测信息
        # 注意：RNN只用于guide（变分后验），不用于model（生成模型）
        rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(
            input_size=input_dim,  # 输入：3D坐标
            hidden_size=rnn_dim,  # 隐藏层：600维（用于编码丰富的信息）
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,  # 单向RNN（因为处理的是反向序列）
            num_layers=num_layers,
            dropout=rnn_dropout_rate,
        )

        # ===== 5. 初始状态参数 =====
        # 初始隐状态：z_0（100维），可学习的参数
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        # RNN初始隐藏状态：h_0（600维），可学习的参数
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def model(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths, images=None, annealing_factor=1.0,
              predict=False):
        """
        生成模型：p(x_{1:T} | z_{1:T}), p(z_{1:T})
        
        这个函数定义了如何"生成"一条扫描路径：
        1. 从初始状态开始
        2. 每一步：根据上一步状态和图像特征，预测下一步状态（GatedTransition）
        3. 每一步：根据当前状态，生成眼睛位置（Emitter）
        
        参数:
            scanpaths: 扫描路径序列 (batch, T, 3) - 每个样本有T个时间步，每步3个坐标
            scanpaths_reversed: 反向序列（未使用，为兼容性保留）
            mask: 序列掩码，处理变长序列（标记哪些时间步是有效的）
            scanpath_lengths: 每个序列的长度（因为序列长度可能不同）
            images: 输入图像（360度图像）
            annealing_factor: KL散度退火因子（训练技巧，逐渐增加KL项的权重）
            predict: 是否为预测模式
                - False（训练模式）：提供真实观测值，计算重构损失
                - True（预测模式）：不提供观测值，用于生成新序列
        """
        T_max = scanpaths.size(1)  # 序列的最大长度
        pyro.module("dmm", self)  # 注册模型参数到Pyro

        # ===== 步骤1：初始化隐状态 =====
        # 将初始隐状态z_0扩展到batch_size
        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))  # (batch_size, 100)
        # 结合初始状态和第一个观测（将3D坐标转换为100维，然后与z_0合并）
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))
        # 现在z_prev包含了初始状态和第一个观测的信息

        # ===== 步骤2：提取图像特征 =====
        # 使用球面CNN从360度图像中提取特征（图像 → 100维特征向量）
        img_features = self.cnn(images)  # (batch_size, 100)

        # ===== 步骤3：对每个时间步循环生成 =====
        with pyro.plate("z_minibatch", len(scanpaths)):  # 批次维度
            for t in pyro.markov(range(1, T_max + 1)):  # 从时间步1到T_max
                # 3.1 计算转移分布的参数
                # 根据上一步隐状态z_{t-1}和图像特征，预测当前隐状态z_t的分布
                # 使用GatedTransition（门控转移机制）
                z_mu, z_sigma = self.trans(z_prev, img_features)
                # z_mu, z_sigma: (batch_size, 100) - 隐状态分布的参数

                # 3.2 从转移分布采样隐状态z_t
                # 采样：z_t ~ N(z_mu, z_sigma^2)
                with poutine.scale(scale=annealing_factor):  # KL退火
                    z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma)
                                      .mask(mask[:, t - 1: t]).to_event(1))
                    # z_t: (batch_size, 100) - 当前时间步的隐状态

                # 3.3 计算发射分布的参数
                # 根据当前隐状态z_t，生成眼睛位置x_t的分布
                # 使用Emitter（发射器）
                x_mu, x_sigma = self.emitter(z_t)
                # x_mu, x_sigma: (batch_size, 3) - 3D坐标(x,y,z)的分布参数

                # 3.4 生成/观测眼睛位置
                if not predict:
                    # 训练模式：提供真实观测值（计算重构损失）
                    # obs=scanpaths[:, t - 1, :] 表示真实的眼睛位置
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                .mask(mask[:, t - 1: t]).to_event(1), obs=scanpaths[:, t - 1, :])
                else:
                    # 预测模式：不提供观测值，用于生成新序列
                    # 采样：x_t ~ N(x_mu, x_sigma^2)
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                .mask(mask[:, t - 1: t]).to_event(1))

                z_prev = z_t  # 更新：当前状态成为下一步的"上一步状态"

    def guide(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths, images=None, annealing_factor=1.0):
        """
        变分后验：q(z_{1:T} | x_{1:T})
        
        这个函数定义了如何从观测序列"推断"隐状态：
        1. 使用RNN处理反向序列（从未来往过去看）
        2. RNN编码每个时间步的"未来信息"
        3. 每一步：结合上一步状态和RNN编码的未来信息，推断当前状态（Combiner）
        
        与model的区别：
        - model：使用GatedTransition，只能看"过去"和"图像"
        - guide：使用Combiner，可以看到"未来"信息（训练时的优势）
        
        参数:
            scanpaths: 扫描路径序列 (batch, T, 3)
            scanpaths_reversed: 反向序列 (batch, T, 3) - 用于RNN处理
            mask: 序列掩码
            scanpath_lengths: 每个序列的长度
            images: 输入图像（未使用，为兼容性保留）
            annealing_factor: KL散度退火因子
        """
        T_max = scanpaths.size(1)
        pyro.module("dmm", self)

        # ===== 步骤1：初始化RNN隐藏状态 =====
        # 将RNN初始状态h_0扩展到(batch_size, rnn_dim)
        h_0_contig = self.h_0.expand(1, scanpaths.size(0), self.rnn.hidden_size).contiguous()
        # shape: (1, batch_size, 600) - RNN需要的格式

        # ===== 步骤2：RNN处理反向序列 =====
        # 关键：处理反向序列（从未来往过去看）
        # 输入：scanpaths_reversed = [x_T, x_{T-1}, ..., x_1]（反向顺序）
        # 这样在时间步t时，RNN已经"看到"了x_{t+1}到x_T的信息
        rnn_output, _ = self.rnn(scanpaths_reversed, h_0_contig)
        # rnn_output: (batch, T, 600) - 每个时间步的RNN隐藏状态

        # 将RNN输出反转回正向顺序（因为我们要按正向顺序推断隐状态）
        rnn_output = poly.pad_and_reverse(rnn_output, scanpath_lengths)
        # rnn_output[:, t-1, :] 现在包含了时间步t的未来信息

        # ===== 步骤3：初始化隐状态 =====
        # 与model相同：结合初始状态和第一个观测
        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))

        # ===== 步骤4：对每个时间步循环推断 =====
        with pyro.plate("z_minibatch", len(scanpaths)):
            for t in pyro.markov(range(1, T_max + 1)):
                # 4.1 组合前一个隐状态和RNN输出
                # 关键区别：这里使用Combiner（不是GatedTransition）
                # z_prev: 上一步隐状态 (batch_size, 100)
                # rnn_output[:, t-1, :]: RNN编码的未来信息 (batch_size, 600)
                # 注意：rnn_output[:, t-1, :]已经包含了x_t到x_T的信息（因为RNN处理的是反向序列）
                z_mu, z_sigma = self.combiner(z_prev, rnn_output[:, t - 1, :])
                # z_mu, z_sigma: (batch_size, 100) - 当前隐状态分布的参数

                # 验证分布的形状
                z_dist = dist.Normal(z_mu, z_sigma)
                assert z_dist.event_shape == ()
                assert z_dist.batch_shape[-2:] == (len(scanpaths), self.z_0.size(0))

                # 4.2 从变分后验采样隐状态z_t
                # 采样：z_t ~ q(z_t | z_{t-1}, x_{t:T})
                # 这里的q就是Combiner定义的分布
                with pyro.poutine.scale(scale=annealing_factor):  # KL退火
                    z_t = pyro.sample(
                        "z_%d" % t,  # 必须与model中的名称相同
                        z_dist.mask(mask[:, t - 1: t]).to_event(1),
                    )
                    # z_t: (batch_size, 100) - 当前时间步的隐状态

                z_prev = z_t  # 更新：当前状态成为下一步的"上一步状态"


# ============================================================================
# 第五部分：工具函数 (Utility Functions)
# ============================================================================

def image_process(path):
    """
    图像预处理：读取、调整大小、归一化
    
    Args:
        path: 图像文件路径
    
    Returns:
        image: 预处理后的图像张量, shape = (3, height, width)
               例如: (3, 128, 256) - RGB 3通道，128x256像素
    
    预处理流程：
    1. 读取图像（BGR格式）
    2. 转换为RGB格式
    3. 调整大小到指定尺寸（例如: 128x256）
    4. 归一化到[0, 1]范围
    5. 转换为PyTorch张量
    6. 归一化到[-1, 1]范围（使用mean=0.5, std=0.5）
    
    维度变化:
        输入: 文件路径（字符串）
        读取: (H_orig, W_orig, 3) - BGR格式，例如: (512, 1024, 3)
        BGR->RGB: (H_orig, W_orig, 3) - RGB格式
        调整大小: (128, 256, 3) - 调整到指定尺寸
        归一化[0,1]: (128, 256, 3) - 值域[0, 1]
        ToTensor: (3, 128, 256) - 转换为张量，通道在前
        Normalize: (3, 128, 256) - 值域[-1, 1]
        输出: (3, 128, 256)
    """
    # 步骤1: 读取图像（OpenCV默认BGR格式）
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    # shape: (H_orig, W_orig, 3) - BGR格式
    # 例如: (512, 1024, 3)

    # 步骤2: 转换为RGB格式（模型需要RGB）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # shape: (H_orig, W_orig, 3) - RGB格式
    # 例如: (512, 1024, 3)

    # 步骤3: 调整大小到指定尺寸
    image = cv2.resize(image, (Config.image_size[1], Config.image_size[0]), interpolation=cv2.INTER_AREA)
    # 参数: (width, height) = (256, 128)
    # shape: (128, 256, 3) - 调整后的尺寸
    # INTER_AREA: 使用区域插值，适合缩小图像

    # 步骤4: 归一化到[0, 1]范围
    image = image.astype(np.float32) / 255.0
    # 转换为float32，除以255得到[0, 1]范围
    # shape: (128, 256, 3) - 值域[0, 1]

    # 步骤5: 转换为PyTorch张量并归一化到[-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W)，值域[0, 1]
        transforms.Normalize([0.5], [0.5])  # 归一化: (x - 0.5) / 0.5 = 2*x - 1，值域[-1, 1]
    ])
    image = transform(image)
    # ToTensor: (128, 256, 3) -> (3, 128, 256)
    # Normalize: (3, 128, 256) -> (3, 128, 256)，值域从[0,1]变为[-1,1]

    return image  # shape: (3, 128, 256) - 值域[-1, 1]


def sphere2plane(sphere_cord, height_width=None):
    """
    球面坐标转平面坐标（等距圆柱投影）
    
    将球面坐标(lat, lon)转换为平面图像坐标(x, y)
    这是360度图像常用的等距圆柱投影（Equirectangular Projection）
    
    Args:
        sphere_cord: 球面坐标, shape = (n, 2)
                     列0: 纬度lat (度) - 范围[-90, 90]
                     列1: 经度lon (度) - 范围[-180, 180]
                     例如: (100, 2) - 100个点的球面坐标
        
        height_width: 可选，图像尺寸 [height, width]
                      如果提供，输出像素坐标；如果不提供，输出归一化坐标[0,1]
    
    Returns:
        plane_cord: 平面坐标, shape = (n, 2)
                    列0: y坐标（高度方向）
                    列1: x坐标（宽度方向）
                    - 如果height_width=None: 归一化坐标[0, 1]
                    - 如果提供了height_width: 像素坐标[0, height/width]
                    例如: (100, 2)
    
    转换公式:
        - 纬度: lat [-90, 90] -> y [0, 1] 或 [0, height]
          y = (lat + 90) / 180 * height (如果提供height_width)
          y = (lat + 90) / 180 (如果height_width=None)
        
        - 经度: lon [-180, 180] -> x [0, 1] 或 [0, width]
          x = (lon + 180) / 360 * width (如果提供height_width)
          x = (lon + 180) / 360 (如果height_width=None)
    
    示例:
        输入: lat=0, lon=0 (赤道和本初子午线交点)
        输出: y=0.5, x=0.5 (图像中心点)
    """
    lat, lon = sphere_cord[:, 0], sphere_cord[:, 1]  # 提取纬度和经度

    if height_width is None:
        # 归一化坐标 [0, 1]
        y = (lat + 90) / 180  # 纬度: [-90, 90] -> [0, 1]
        x = (lon + 180) / 360  # 经度: [-180, 180] -> [0, 1]
    else:
        # 像素坐标 [0, height/width]
        y = (lat + 90) / 180 * height_width[0]  # 纬度 -> y像素坐标
        x = (lon + 180) / 360 * height_width[1]  # 经度 -> x像素坐标

    return torch.cat((y.view(-1, 1), x.view(-1, 1)), 1)  # shape: (n, 2)


def plane2sphere(plane_cord, height_width=None):
    """
    平面坐标转球面坐标（等距圆柱投影的逆变换）
    
    将平面图像坐标(x, y)转换为球面坐标(lat, lon)
    这是sphere2plane的逆变换
    
    Args:
        plane_cord: 平面坐标, shape = (n, 2)
                    列0: y坐标（高度方向）
                    列1: x坐标（宽度方向）
                    - 如果height_width=None且坐标<=1: 归一化坐标[0, 1]
                    - 如果提供了height_width: 像素坐标
                    例如: (100, 2)
        
        height_width: 可选，图像尺寸 [height, width]
    
    Returns:
        sphere_cord: 球面坐标, shape = (n, 2)
                     列0: 纬度lat (度) - 范围[-90, 90]
                     列1: 经度lon (度) - 范围[-180, 180]
                     例如: (100, 2)
    
    转换公式（sphere2plane的逆）:
        - y -> lat: lat = (y - 0.5) * 180
        - x -> lon: lon = (x - 0.5) * 360
    """
    y, x = plane_cord[:, 0], plane_cord[:, 1]  # 提取y和x坐标

    if (height_width is None) & (torch.any(plane_cord <= 1).item()):
        # 归一化坐标 [0, 1] -> 球面坐标
        lat = (y - 0.5) * 180  # y: [0, 1] -> lat: [-90, 90]
        lon = (x - 0.5) * 360  # x: [0, 1] -> lon: [-180, 180]
    else:
        # 像素坐标 -> 球面坐标
        lat = (y / height_width[0] - 0.5) * 180  # 像素y -> lat
        lon = (x / height_width[1] - 0.5) * 360  # 像素x -> lon

    return torch.cat((lat.view(-1, 1), lon.view(-1, 1)), 1)  # shape: (n, 2)


def sphere2xyz(shpere_cord):
    """
    球面坐标转3D笛卡尔坐标（单位球面）
    
    将球面坐标(lat, lon)转换为3D笛卡尔坐标(x, y, z)
    假设点在单位球面上（半径为1）
    
    Args:
        shpere_cord: 球面坐标, shape = (n, 2)
                     列0: 纬度lat (度) - 范围[-90, 90]
                     列1: 经度lon (度) - 范围[-180, 180]
                     例如: (100, 2)
    
    Returns:
        xyz: 3D笛卡尔坐标, shape = (n, 3)
             列0: x坐标 - 范围[-1, 1]
             列1: y坐标 - 范围[-1, 1]
             列2: z坐标 - 范围[-1, 1]
             点在单位球面上，满足 x² + y² + z² = 1
             例如: (100, 3)
    
    转换公式:
        lat_rad = lat * π / 180  (转换为弧度)
        lon_rad = lon * π / 180
        
        x = cos(lat) * cos(lon)  (x轴方向)
        y = cos(lat) * sin(lon)  (y轴方向)
        z = sin(lat)             (z轴方向)
    
    坐标系定义:
        - z轴: 从南极点指向北极点
        - x轴: 在赤道平面上，指向本初子午线
        - y轴: 在赤道平面上，指向东经90度
    """
    lat, lon = shpere_cord[:, 0], shpere_cord[:, 1]  # 提取纬度和经度

    # 转换为弧度
    lat = lat / 180 * pi  # 纬度: 度 -> 弧度
    lon = lon / 180 * pi  # 经度: 度 -> 弧度

    # 转换为3D笛卡尔坐标（单位球面）
    x = torch.cos(lat) * torch.cos(lon)  # x坐标
    y = torch.cos(lat) * torch.sin(lon)  # y坐标
    z = torch.sin(lat)  # z坐标

    return torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), 1)  # shape: (n, 3)


def xyz2sphere(threeD_cord):
    """
    3D笛卡尔坐标转球面坐标
    
    将3D笛卡尔坐标(x, y, z)转换为球面坐标(lat, lon)
    这是sphere2xyz的逆变换
    
    Args:
        threeD_cord: 3D笛卡尔坐标, shape = (n, 3)
                     列0: x坐标
                     列1: y坐标
                     列2: z坐标
                     例如: (100, 3)
    
    Returns:
        sphere_cord: 球面坐标, shape = (n, 2)
                     列0: 纬度lat (度) - 范围[-90, 90]
                     列1: 经度lon (度) - 范围[-180, 180]
                     例如: (100, 2)
    
    转换公式:
        lon = atan2(y, x) * 180 / π  (经度，从x轴逆时针)
        lat = atan2(z, sqrt(x²+y²)) * 180 / π  (纬度，从赤道平面)
    """
    x, y, z = threeD_cord[:, 0], threeD_cord[:, 1], threeD_cord[:, 2]  # 提取x, y, z

    # 计算经度（从x轴逆时针，范围[-180, 180]度）
    lon = torch.atan2(y, x)
    # 计算纬度（从赤道平面，范围[-90, 90]度）
    lat = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))

    # 转换为度
    lat = lat / pi * 180  # 纬度: 弧度 -> 度
    lon = lon / pi * 180  # 经度: 弧度 -> 度

    return torch.cat((lat.view(-1, 1), lon.view(-1, 1)), 1)  # shape: (n, 2)


def xyz2plane(threeD_cord, height_width=None):
    """
    3D笛卡尔坐标转平面坐标（两步转换）
    
    将3D笛卡尔坐标(x, y, z)转换为平面图像坐标
    通过两步转换：3D坐标 -> 球面坐标 -> 平面坐标
    
    Args:
        threeD_cord: 3D笛卡尔坐标, shape = (n, 3)
                     例如: (100, 3)
        
        height_width: 可选，图像尺寸 [height, width]
    
    Returns:
        plane_cord: 平面坐标, shape = (n, 2)
                    例如: (100, 2)
    
    转换流程:
        1. xyz2sphere: (n, 3) -> (n, 2)  [3D坐标 -> 球面坐标]
        2. sphere2plane: (n, 2) -> (n, 2)  [球面坐标 -> 平面坐标]
    """
    # 步骤1: 3D坐标 -> 球面坐标
    sphere_cords = xyz2sphere(threeD_cord)  # (n, 3) -> (n, 2)

    # 步骤2: 球面坐标 -> 平面坐标
    plane_cors = sphere2plane(sphere_cords, height_width)  # (n, 2) -> (n, 2)

    return plane_cors  # shape: (n, 2)


# ============================================================================
# 第六部分：训练类 (Training)
# ============================================================================

class Train:
    """
    训练类：封装整个训练流程
    
    主要功能：
    1. 设置优化器（Adam优化器）
    2. 设置变分推断（SVI - Stochastic Variational Inference）
    3. 准备训练数据
    4. 执行训练循环
    5. 保存/加载模型检查点
    """

    def __init__(self, model, train_package, args):
        """
        初始化训练类
        
        Args:
            model: DMM模型实例
            train_package: 训练数据包，包含训练数据和元信息
            args: 训练参数（学习率、批次大小、epoch数等）
        """
        self.dmm = model  # DMM模型
        self.args = args  # 训练参数
        self.train_package = train_package  # 训练数据包

    def setup_adam(self):
        """
        设置Adam优化器（带梯度裁剪）
        
        优化器参数：
        - lr: 学习率（例如: 0.0003）
        - betas: Adam的动量参数 (beta1, beta2) = (0.96, 0.999)
        - clip_norm: 梯度裁剪阈值（防止梯度爆炸）
        - lrd: 学习率衰减率（每个step衰减）
        - weight_decay: L2正则化权重（防止过拟合）
        
        使用ClippedAdam是为了：
        1. 防止梯度爆炸（通过clip_norm）
        2. 稳定训练过程
        """
        params = {
            "lr": self.args.lr,  # 初始学习率
            "betas": (0.96, 0.999),  # Adam的动量参数
            "clip_norm": 10,  # 梯度裁剪阈值：将梯度范数限制在10以内
            "lrd": self.args.lr_decay,  # 学习率衰减率（每个step衰减）
            "weight_decay": self.args.weight_decay,  # L2正则化权重
        }
        self.adam = ClippedAdam(params)  # 创建带梯度裁剪的Adam优化器

    def setup_inference(self):
        """
        设置随机变分推断（SVI）
        
        SVI是Pyro框架中的变分推断方法，用于训练概率模型。
        
        组件：
        - model: 生成模型 p(x|z) * p(z)，定义如何从隐状态生成观测
        - guide: 变分后验 q(z|x)，用于近似真实后验 p(z|x)
        - optimizer: 优化器（Adam）
        - loss: ELBO（Evidence Lower BOund）损失函数
                  ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
                  包含重构损失（第一项）和KL散度（第二项）
        """
        elbo = Trace_ELBO()  # 创建ELBO损失函数
        # 创建SVI对象：将model、guide、优化器和损失函数组合
        self.svi = SVI(self.dmm.model, self.dmm.guide, self.adam, loss=elbo)

    def save_checkpoint(self, name):
        """
        保存模型检查点
        
        Args:
            name: 保存的文件名，例如: 'model_lr-0.0003_bs-64_epoch-0.pkl'
        
        保存内容：
        - 模型的state_dict：包含所有模型参数（权重和偏置）
        
        注意：
        - 只保存模型参数，不保存优化器状态（如果需要，可以单独保存）
        - 文件保存在args.save_root目录下
        """
        # 创建保存目录（如果不存在）
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

        # 保存模型参数（state_dict包含所有可学习参数）
        torch.save(self.dmm.state_dict(), self.args.save_root + name)
        # 保存的文件包含：
        # - CNN的权重和偏置
        # - GatedTransition的权重和偏置
        # - Combiner的权重和偏置
        # - Emitter的权重和偏置
        # - 初始状态参数z_0和h_0
        # - 等等所有nn.Parameter和nn.Module的参数

    def load_checkpoint(self):
        """
        加载模型检查点（从之前保存的状态恢复）
        
        加载内容：
        1. 模型参数（state_dict）
        2. 优化器状态（如果提供了load_opt路径）
        
        用途：
        - 从之前的训练状态继续训练
        - 加载训练好的模型进行推理
        
        注意：
        - 需要同时提供模型文件路径和优化器文件路径
        - 如果只做推理，可以只加载模型参数（不需要优化器）
        """
        # 检查文件路径是否存在
        assert exists(self.args.load_opt) and exists(self.args.load_model), "Load model: path error"

        # 加载模型参数
        self.dmm.load_state_dict(torch.load(self.args.load_model))
        # 将保存的参数加载到模型中

        # 加载优化器状态（用于继续训练）
        self.adam.load(self.args.load_opt)
        # 加载优化器的状态（学习率、动量等），这样可以无缝继续训练

    def prepare_train_data(self):
        """
        准备训练数据：将原始数据组织成训练所需的格式
        
        Returns:
            dic: 包含训练数据的字典
                - sequences: 所有扫描路径序列, shape = (num_scanpath, scanpath_length, 3)
                - sequence_lengths: 每个序列的长度, shape = (num_scanpath,)
                - image_index: 每个扫描路径对应的图像索引, shape = (num_scanpath,)
                - images: 所有图像, shape = (num_scanpath, 3, height, width)
        
        数据组织说明：
        - 每个图像可能有多个扫描路径（多个被试者观看同一图像）
        - 将所有扫描路径平铺成一维数组，同时记录对应的图像索引
        - 所有序列的长度相同（scanpath_length），用mask处理变长序列
        
        维度示例（假设num_scanpath=1000, scanpath_length=100, image_size=[128,256]）:
            sequences: (1000, 100, 3) - 1000个序列，每个100个时间步，每步3D坐标
            sequence_lengths: (1000,) - 每个序列的长度（都是100）
            image_index: (1000,) - 每个序列对应的图像索引
            images: (1000, 3, 128, 256) - 1000张图像（可能有重复，因为一图对应多路径）
        """
        _train = self.train_package['train']  # 原始训练数据
        _info = self.train_package['info']['train']  # 训练数据元信息
        dic = {'sequences': None, 'sequence_lengths': None, 'images': None}

        # 获取数据维度信息
        scanpath_length = _info['scanpath_length']  # 扫描路径长度（时间步数）
        num_scanpath = _info['num_scanpath']  # 扫描路径总数
        image_index = np.zeros((num_scanpath))  # 图像索引数组

        # ===== 初始化数据数组 =====
        # 扫描路径序列：所有路径的3D坐标序列
        scanpath_set = np.zeros([num_scanpath, scanpath_length, 3])
        # shape: (num_scanpath, scanpath_length, 3)
        # 例如: (1000, 100, 3) - 1000个路径，每个100步，每步3D坐标(x,y,z)

        # 序列长度：所有序列的长度（这里假设都相同）
        length_set = (np.ones(num_scanpath) * _info['scanpath_length']).astype(int)
        # shape: (num_scanpath,)
        # 例如: (1000,) - 每个序列都是100步

        # 图像集合：所有图像
        image_set = torch.zeros([num_scanpath, 3, Config.image_size[0], Config.image_size[1]])
        # shape: (num_scanpath, 3, height, width)
        # 例如: (1000, 3, 128, 256) - 1000张图像，RGB 3通道，128x256像素

        # ===== 填充数据 =====
        # 遍历所有图像实例，提取扫描路径和对应图像
        index, img_index = 0, 0  # index: 扫描路径索引, img_index: 图像索引
        for instance in _train:  # instance: 图像文件名
            scanpaths = _train[instance]['scanpaths']  # 该图像的所有扫描路径
            # scanpaths.shape = (num_paths_for_this_image, scanpath_length, 3)

            # 遍历该图像的所有扫描路径
            for j in range(scanpaths.shape[0]):
                scanpath_set[index] = scanpaths[j]  # 保存扫描路径
                image_index[index] = img_index  # 记录对应的图像索引
                image_set[index] = _train[instance]['image']  # 保存图像（可能有重复）
                index += 1
            img_index += 1  # 下一个图像

        # ===== 转换为PyTorch张量 =====
        dic['sequences'] = torch.from_numpy(scanpath_set).float()
        # shape: (num_scanpath, scanpath_length, 3)

        dic['sequence_lengths'] = torch.from_numpy(length_set.astype(int))
        # shape: (num_scanpath,)

        dic['image_index'] = torch.from_numpy(image_index.astype(int))
        # shape: (num_scanpath,)

        dic['images'] = image_set
        # shape: (num_scanpath, 3, height, width)

        return dic

    def get_mini_batch(self, mini_batch_indices, sequences, seq_lengths, images, cuda=False):
        """
        获取一个小批次的数据，并进行预处理
        
        Args:
            mini_batch_indices: 小批次中样本的索引, shape = (batch_size,)
                                例如: [5, 12, 8, 23, ...] (64个索引)
            sequences: 所有序列数据, shape = (N, T_max, 3)
            seq_lengths: 所有序列的长度, shape = (N,)
            images: 所有图像, shape = (N, 3, H, W)
            cuda: 是否使用GPU
        
        Returns:
            mini_batch: 小批次序列（正向）, shape = (batch_size, T_max, 3)
            mini_batch_reversed: 小批次序列（反向，已packed）, 用于RNN
            mini_batch_mask: 序列掩码, shape = (batch_size, T_max)
            sorted_seq_lengths: 排序后的序列长度, shape = (batch_size,)
            mini_batch_images: 小批次图像, shape = (batch_size, 3, H, W)
        
        预处理步骤：
        1. 根据batch_size提取对应索引的数据
        2. 按序列长度降序排序（RNN处理变长序列需要）
        3. 生成反向序列（用于guide中的RNN）
        4. 生成掩码（标记有效时间步）
        5. 移动到GPU（如果使用）
        
        维度示例（假设batch_size=64, T_max=100）:
            输入:
                mini_batch_indices: (64,) - 64个样本的索引
                sequences: (N, 100, 3) - N个序列
                images: (N, 3, 128, 256) - N张图像
            
            输出:
                mini_batch: (64, 100, 3) - 64个序列，最多100步
                mini_batch_reversed: PackedSequence - 反向序列（已pack）
                mini_batch_mask: (64, 100) - 掩码，标记有效步
                sorted_seq_lengths: (64,) - 排序后的长度
                mini_batch_images: (64, 3, 128, 256) - 64张图像
        """
        # ===== 步骤1: 提取小批次的序列长度 =====
        seq_lengths = seq_lengths[mini_batch_indices]
        # 从所有序列长度中提取小批次的序列长度
        # shape: (batch_size,), 例如: (64,)

        # ===== 步骤2: 按序列长度降序排序 =====
        # 这是为了RNN处理变长序列的效率（PyTorch RNN要求）
        _, sorted_seq_length_indices = torch.sort(seq_lengths)
        sorted_seq_length_indices = sorted_seq_length_indices.flip(0)  # 降序排序
        sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
        sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]
        # sorted_seq_length_indices: (batch_size,) - 排序后的索引
        # sorted_seq_lengths: (batch_size,) - 排序后的长度（从长到短）

        # ===== 步骤3: 提取小批次数据 =====
        T_max = torch.max(seq_lengths)  # 小批次中最长序列的长度
        # 提取序列（只取前T_max步，因为所有序列长度<=T_max）
        mini_batch = sequences[sorted_mini_batch_indices, 0:T_max, :]
        # shape: (batch_size, T_max, 3), 例如: (64, 100, 3)

        # 提取对应的图像
        mini_batch_images = images[sorted_mini_batch_indices]
        # shape: (batch_size, 3, H, W), 例如: (64, 3, 128, 256)

        # ===== 步骤4: 生成反向序列（用于guide中的RNN） =====
        # RNN需要处理反向序列来编码未来信息
        mini_batch_reversed = poly.reverse_sequences(mini_batch, sorted_seq_lengths)
        # shape: (batch_size, T_max, 3) - 反向的序列
        # 例如: 原序列 [x1, x2, x3] -> 反向 [x3, x2, x1]

        # ===== 步骤5: 生成掩码（标记有效时间步） =====
        # 掩码用于处理变长序列：1表示有效，0表示填充
        mini_batch_mask = poly.get_mini_batch_mask(mini_batch, sorted_seq_lengths)
        # shape: (batch_size, T_max)
        # 例如: 如果某个序列长度为80，则mask前80个为1，后20个为0

        # ===== 步骤6: 移动到GPU（如果使用） =====
        if cuda:
            mini_batch = mini_batch.cuda()
            mini_batch_mask = mini_batch_mask.cuda()
            mini_batch_reversed = mini_batch_reversed.cuda()
            mini_batch_images = mini_batch_images.cuda()

        # ===== 步骤7: 打包反向序列（用于RNN高效处理） =====
        # pack_padded_sequence将变长序列打包，提高RNN计算效率
        mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(
            mini_batch_reversed, sorted_seq_lengths, batch_first=True)
        # 返回PackedSequence对象，RNN可以直接处理

        return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths, mini_batch_images

    def process_minibatch(self, epoch, which_mini_batch, shuffled_indices):
        """
        处理一个小批次：计算损失并执行一步梯度更新
        
        Args:
            epoch: 当前epoch数（从0开始）
            which_mini_batch: 当前是第几个小批次（从0开始）
            shuffled_indices: 打乱后的样本索引
        
        Returns:
            loss: 当前批次的ELBO损失（负值，因为要最大化ELBO，但优化器最小化，所以取负）
        
        流程：
        1. 计算KL散度退火因子（训练初期逐渐增加KL项权重）
        2. 获取小批次数据的索引
        3. 提取小批次数据
        4. 执行SVI一步更新（前向传播 + 反向传播 + 优化器更新）
        
        KL散度退火说明：
        - 训练初期，先让模型学习重构观测（忽略KL项）
        - 逐渐增加KL项权重，让guide接近真实后验
        - 这样可以避免训练初期的模式崩塌问题
        """
        # ===== 步骤1: 计算KL散度退火因子 =====
        # KL散度退火：训练初期逐渐增加KL项的权重
        # 公式: annealing_factor = min_af + (1 - min_af) * (当前步数 / 总退火步数)
        if self.args.annealing_epochs > 0 and epoch < self.args.annealing_epochs:
            min_af = self.args.minimum_annealing_factor  # 最小退火因子（例如: 0.2）
            # 计算当前进度：[0, 1]之间，从min_af线性增长到1.0
            current_step = which_mini_batch + epoch * self.N_mini_batches + 1
            total_steps = self.args.annealing_epochs * self.N_mini_batches
            annealing_factor = min_af + (1.0 - min_af) * (float(current_step) / float(total_steps))
            # 例如: epoch=0, which_mini_batch=0 -> annealing_factor ≈ min_af
            #       epoch接近annealing_epochs -> annealing_factor ≈ 1.0
        else:
            annealing_factor = 1.0  # 退火完成后，KL项权重为1.0（正常训练）

        # ===== 步骤2: 获取小批次数据的索引 =====
        mini_batch_start = which_mini_batch * self.args.bs  # 小批次起始索引
        mini_batch_end = np.min([(which_mini_batch + 1) * self.args.bs, self.N_sequences])  # 结束索引
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        # shape: (batch_size,), 例如: (64,)
        # 例如: which_mini_batch=0, bs=64 -> indices[0:64]
        #      which_mini_batch=1, bs=64 -> indices[64:128]

        # ===== 步骤3: 提取小批次数据 =====
        (mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, mini_batch_images) = \
            self.get_mini_batch(mini_batch_indices, self.sequences, self.seq_lengths, self.images,
                                cuda=Config.use_cuda)
        # mini_batch: (batch_size, T_max, 3) - 正向序列
        # mini_batch_reversed: PackedSequence - 反向序列（用于RNN）
        # mini_batch_mask: (batch_size, T_max) - 掩码
        # mini_batch_seq_lengths: (batch_size,) - 序列长度
        # mini_batch_images: (batch_size, 3, H, W) - 图像

        # ===== 步骤4: 执行SVI一步更新 =====
        # SVI.step()执行：
        # 1. 前向传播：model和guide各运行一次
        # 2. 计算ELBO损失 = E_q[log p(x|z)] - annealing_factor * KL(q(z|x) || p(z))
        #    其中重构损失和KL散度通过annealing_factor加权
        # 3. 反向传播：计算梯度
        # 4. 优化器更新：更新模型参数
        loss = self.svi.step(
            scanpaths=mini_batch,  # 正向序列
            scanpaths_reversed=mini_batch_reversed,  # 反向序列（用于guide中的RNN）
            mask=mini_batch_mask,  # 掩码（处理变长序列）
            scanpath_lengths=mini_batch_seq_lengths,  # 序列长度
            images=mini_batch_images,  # 图像
            annealing_factor=annealing_factor,  # KL退火因子
        )
        # loss: 标量，当前批次的ELBO损失（返回的是负ELBO，因为优化器最小化）
        # loss越小，表示ELBO越大，模型越好

        return loss

    def run(self):
        """
        运行完整的训练流程
        
        训练流程：
        1. 初始化优化器和SVI
        2. 加载检查点（如果提供）
        3. 准备训练数据
        4. 计算训练统计信息
        5. 执行训练循环（多个epoch）
           每个epoch:
             - 打乱数据顺序
             - 遍历所有小批次
             - 对每个小批次执行梯度更新
             - 保存检查点
        
        训练统计信息：
        - N_sequences: 序列总数
        - N_time_slices: 时间步总数（所有序列的时间步之和）
        - N_mini_batches: 小批次数量 = ceil(N_sequences / batch_size)
        """
        # ===== 步骤1: 初始化优化器和SVI =====
        self.setup_adam()  # 设置Adam优化器
        self.setup_inference()  # 设置SVI（随机变分推断）

        # ===== 步骤2: 加载检查点（如果提供） =====
        # 用于从之前的训练状态继续训练
        if self.args.load_opt is not None and self.args.load_model is not None:
            self.load_checkpoint()

        # ===== 步骤3: 准备训练数据 =====
        train_data = self.prepare_train_data()
        self.sequences = train_data["sequences"]
        # shape: (N_sequences, T_max, 3) - 所有序列的3D坐标
        # 例如: (1000, 100, 3)

        self.seq_lengths = train_data["sequence_lengths"]
        # shape: (N_sequences,) - 每个序列的长度
        # 例如: (1000,)

        self.images = train_data["images"]
        # shape: (N_sequences, 3, H, W) - 所有图像
        # 例如: (1000, 3, 128, 256)

        # ===== 步骤4: 计算训练统计信息 =====
        self.N_sequences = len(self.seq_lengths)  # 序列总数
        # 例如: 1000

        self.N_time_slices = float(torch.sum(self.seq_lengths))  # 时间步总数
        # 例如: 1000 * 100 = 100000（如果所有序列长度都是100）

        # 计算小批次数量（向上取整）
        self.N_mini_batches = int(self.N_sequences / self.args.bs +
                                  int(self.N_sequences % self.args.bs > 0))
        # 例如: N_sequences=1000, bs=64 -> N_mini_batches = ceil(1000/64) = 16

        # ===== 步骤5: 训练循环 =====
        for epoch in range(self.args.epochs):
            epoch_nll = 0.0  # 当前epoch的累计损失（NLL = Negative Log Likelihood）
            # 每个epoch开始时打乱数据顺序（随机采样）
            shuffled_indices = torch.randperm(self.N_sequences)
            # shape: (N_sequences,)
            # 例如: [523, 12, 789, ..., 234]（随机排列的索引）

            # 遍历所有小批次
            for which_mini_batch in range(self.N_mini_batches):
                # 处理一个小批次：执行梯度更新
                batch_loss = self.process_minibatch(epoch, which_mini_batch, shuffled_indices)
                epoch_nll += batch_loss  # 累计损失

            # 每个epoch结束后保存检查点
            save_name = 'model_lr-{}_bs-{}_epoch-{}.pkl'.format(
                self.args.lr, self.args.bs, epoch)
            self.save_checkpoint(save_name)
            # 保存文件名示例: 'model_lr-0.0003_bs-64_epoch-0.pkl'

            # 注意：epoch_nll是负ELBO的累计，所以越大表示损失越大（模型越差）
            # 可以通过打印epoch_nll / self.N_mini_batches来查看平均损失


# ============================================================================
# 第七部分：推理类 (Inference)
# ============================================================================

class Inference:
    """
    推理类：用于从训练好的模型生成扫描路径
    
    主要功能：
    1. 为每张图像生成多个扫描路径
    2. 使用随机起始点
    3. 将生成的3D坐标转换为2D平面坐标
    4. 保存预测结果
    """

    def __init__(self, model, img_path, n_scanpaths, length, output_path):
        """
        初始化推理类
        
        Args:
            model: 训练好的DMM模型
            img_path: 图像文件夹路径
            n_scanpaths: 每张图像生成的扫描路径数量
            length: 每个扫描路径的长度（时间步数）
            output_path: 输出文件夹路径
        """
        self.dmm = model  # DMM模型（应该已经加载了训练好的权重）
        self.img_path = img_path  # 输入图像路径
        self.n_scanpaths = n_scanpaths  # 每张图像生成多少个扫描路径
        self.length = length  # 扫描路径长度（时间步数）
        self.output_path = output_path  # 输出路径

    def create_random_starting_points(self, num_points):
        """
        创建随机起始点（从赤道偏置分布采样）
        
        采样策略：
        - y坐标（纬度）：从N(0, 0.2²)采样，截断到[-1, 1]，然后*90度
          这意味着起始点更可能靠近赤道（y=0），符合人类观看习惯
        - x坐标（经度）：从U(-1, 1)均匀采样，然后*180度
        
        Args:
            num_points: 需要生成的起始点数量
        
        Returns:
            cords: 起始点的3D坐标, shape = (num_points, 3)
                   在单位球面上的3D坐标(x, y, z)
        
        维度变化:
            采样: num_points个(lat, lon)对
            -> 转换为度: (num_points, 2) - [纬度(-90~90), 经度(-180~180)]
            -> sphere2xyz: (num_points, 3) - 3D坐标(x, y, z)
        
        示例（num_points=10）:
            输出: (10, 3) - 10个起始点的3D坐标
        """
        y, x = [], []  # y: 纬度, x: 经度

        # 为每个起始点生成随机坐标
        for i in range(num_points):
            # 生成纬度（y坐标）：从正态分布采样，偏向赤道
            while True:
                temp = np.random.normal(loc=0, scale=0.2)  # N(0, 0.2²)
                if (temp <= 1) and (temp >= -1):  # 截断到[-1, 1]
                    y.append(temp)  # 归一化的纬度
                    break
            # 生成经度（x坐标）：均匀分布
            x.append(np.random.uniform(-1, 1))  # U(-1, 1)

        # 转换为球面坐标（度）
        # y * 90: 归一化坐标 -> 纬度（度）[-90, 90]
        # x * 180: 归一化坐标 -> 经度（度）[-180, 180]
        cords = np.vstack((np.array(y) * 90, np.array(x) * 180)).swapaxes(0, 1)
        # shape: (num_points, 2) - [纬度, 经度]（度）

        # 转换为3D坐标（单位球面）
        cords = sphere2xyz(torch.from_numpy(cords))
        # shape: (num_points, 3) - 3D坐标(x, y, z)

        return cords

    def summary(self, samples):
        """
        整理预测结果：将3D坐标转换为2D平面坐标
        
        Args:
            samples: Predictive对象返回的采样结果字典
                     包含所有时间步的观测值 'obs_x_1', 'obs_x_2', ..., 'obs_x_T'
        
        Returns:
            obs: 2D平面坐标序列, shape = (num_samples, T, 2)
                 例如: (1, 100, 2) - 1个样本，100个时间步，每步2D坐标(x, y)
        
        处理流程：
        1. 提取每个时间步的3D坐标
        2. 归一化到单位球面（确保在球面上）
        3. 转换为2D平面坐标（等距圆柱投影）
        4. 重新组织维度：从(T, num_samples, 2)转置为(num_samples, T, 2)
        
        维度变化示例（假设T=100, num_samples=1）:
            输入samples:
                'obs_x_1': (1, 3) - 时间步1的3D坐标
                'obs_x_2': (1, 3) - 时间步2的3D坐标
                ...
                'obs_x_100': (1, 3) - 时间步100的3D坐标
            
            处理过程:
                temp: (1, 3) -> 归一化 -> (1, 3) -> xyz2plane -> (1, 2)
                收集所有时间步: (100, 1, 2)
                转置: (1, 100, 2)
            
            输出: (1, 100, 2) - 1个样本，100步，每步2D坐标
        """
        obs = None

        # 遍历所有时间步（samples包含'obs_x_t'和'z_t'，所以除以2）
        for index in range(int(len(samples) / 2)):
            name = 'obs_x_' + str(index + 1)  # 例如: 'obs_x_1', 'obs_x_2', ...
            temp = samples[name].reshape([-1, 3])
            # shape: (num_samples, 3) - 当前时间步的3D坐标
            # 例如: (1, 3) - 1个样本的3D坐标(x, y, z)

            # ===== 步骤1: 归一化到单位球面 =====
            # 确保所有点都在单位球面上（满足x²+y²+z²=1）
            its_sum = torch.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
            # shape: (num_samples,) - 每个点到原点的距离
            temp = temp / torch.unsqueeze(its_sum, 1)
            # shape: (num_samples, 3) - 归一化后的3D坐标
            # 例如: (1, 3)

            # ===== 步骤2: 转换为2D平面坐标 =====
            # 使用等距圆柱投影将3D坐标转换为平面坐标
            plane_coords = xyz2plane(temp)  # (num_samples, 2)
            # 例如: (1, 2) - 2D坐标(x, y)

            # 收集所有时间步的结果
            if obs is not None:
                obs = torch.cat((obs, torch.unsqueeze(plane_coords, dim=0)), dim=0)
            else:
                obs = torch.unsqueeze(plane_coords, dim=0)
            # obs逐渐累积: (1, num_samples, 2) -> (2, num_samples, 2) -> ... -> (T, num_samples, 2)

        # ===== 步骤3: 转置维度 =====
        # 从 (T, num_samples, 2) 转置为 (num_samples, T, 2)
        obs = torch.transpose(obs, 0, 1)
        # shape: (num_samples, T, 2)
        # 例如: (1, 100, 2) - 1个样本，100个时间步，每步2D坐标

        return obs

    def predict(self):
        """
        执行预测：为每张图像生成扫描路径
        
        预测流程：
        1. 遍历图像文件夹中的所有图像
        2. 为每张图像生成n_scanpaths个扫描路径
        3. 使用随机起始点
        4. 调用model生成完整路径
        5. 转换坐标并保存结果
        
        注意：
        - 使用predict=True模式，只调用model（不调用guide）
        - 不需要提供真实观测，完全由模型生成
        - 使用torch.no_grad()节省内存（不需要计算梯度）
        """
        num_samples = 1  # 每次生成的样本数（为了内存效率，一次只生成1个）
        rep_num = self.n_scanpaths // num_samples  # 需要重复的次数
        # 例如: n_scanpaths=64, num_samples=1 -> rep_num=64

        # 创建Predictive对象（Pyro的预测工具）
        # 它会多次调用model，收集所有采样结果
        predictive = Predictive(self.dmm.model, num_samples=num_samples)

        # 遍历图像文件夹
        for _, _, files in os.walk(self.img_path):
            num_img = len(files)  # 图像总数

            # 为每张图像生成扫描路径
            for img in files:
                # ===== 步骤1: 加载并预处理图像 =====
                img_path = os.path.join(self.img_path, img)
                # 加载图像并转换为张量，然后扩展到rep_num份（因为要生成多个路径）
                image_tensor = torch.unsqueeze(image_process(img_path), dim=0).repeat([rep_num, 1, 1, 1])
                # image_process: 加载并预处理图像 -> (3, H, W)
                # unsqueeze: (3, H, W) -> (1, 3, H, W)
                # repeat: (1, 3, H, W) -> (rep_num, 3, H, W)
                # 例如: (64, 3, 128, 256) - 64份相同的图像

                # ===== 步骤2: 创建随机起始点 =====
                starting_points = torch.unsqueeze(
                    self.create_random_starting_points(rep_num), dim=1).to(torch.float32)
                # create_random_starting_points: (rep_num, 3) - 3D坐标
                # unsqueeze: (rep_num, 3) -> (rep_num, 1, 3) - 添加时间步维度
                # 例如: (64, 1, 3) - 64个起始点，每个是3D坐标

                # ===== 步骤3: 创建初始扫描路径（用起始点填充） =====
                # 将起始点复制length次，作为初始序列（model会从这些点开始生成）
                _scanpaths = starting_points.repeat([1, self.length, 1])
                # shape: (rep_num, length, 3)
                # 例如: (64, 100, 3) - 64个序列，每个100步，每步都是起始点的3D坐标
                # 注意：这只是初始值，model会生成新的路径覆盖它们

                # ===== 步骤4: 创建掩码（所有时间步都有效） =====
                test_mask = torch.ones([rep_num, self.length])
                # shape: (rep_num, length)
                # 例如: (64, 100) - 所有位置都是1（所有时间步都有效）

                # ===== 步骤5: 移动到模型所在的设备（CPU/GPU） =====
                device = next(self.dmm.parameters()).device  # 获取模型所在的设备
                test_batch = _scanpaths.to(device)
                # shape: (rep_num, length, 3)
                test_batch_mask = test_mask.to(device)
                # shape: (rep_num, length)
                test_batch_images = image_tensor.to(device)
                # shape: (rep_num, 3, H, W)

                # ===== 步骤6: 执行预测（不需要梯度） =====
                with torch.no_grad():  # 禁用梯度计算（推理阶段不需要）
                    # 调用model生成扫描路径
                    # predict=True表示预测模式（不提供真实观测）
                    samples = predictive(scanpaths=test_batch,  # 初始序列（会被覆盖）
                                         scanpaths_reversed=None,  # 预测模式不需要反向序列
                                         mask=test_batch_mask,  # 掩码
                                         scanpath_lengths=None,  # 预测模式不需要长度
                                         images=test_batch_images,  # 图像
                                         predict=True)  # 预测模式
                    # samples: 字典，包含所有时间步的采样结果
                    # 例如: {'obs_x_1': (rep_num, 3), 'obs_x_2': (rep_num, 3), ..., 'obs_x_100': (rep_num, 3)}

                    # ===== 步骤7: 整理结果（转换为2D坐标） =====
                    scanpaths = self.summary(samples).cpu().numpy()
                    # summary: 将3D坐标转换为2D平面坐标
                    # shape: (rep_num, length, 2)
                    # 例如: (64, 100, 2) - 64个扫描路径，每个100步，每步2D坐标
                    # cpu().numpy(): 转换为numpy数组（便于保存）

                    # ===== 步骤8: 保存结果 =====
                    save_name = img.split('.')[0] + '.npy'  # 例如: 'image001.npy'

                    if not os.path.exists(self.output_path):
                        os.makedirs(self.output_path)  # 创建输出文件夹（如果不存在）

                    # 保存为.npy文件
                    np.save(os.path.join(self.output_path, save_name), scanpaths)
                    # 保存的数组shape: (rep_num, length, 2)
                    # 例如: (64, 100, 2) - 64个扫描路径的2D坐标序列


# ============================================================================
# 第八部分：数据处理 (Data Processing)
# ============================================================================

def save_file(file_name, data):
    """
    保存数据到文件（使用pickle序列化）
    
    Args:
        file_name: 保存的文件路径
        data: 要保存的数据（可以是任意Python对象）
    
    用途：
    - 保存预处理后的数据集
    - 保存模型检查点等
    """
    with open(file_name, 'wb') as f:  # 'wb' = write binary
        pickle.dump(data, f)  # 使用pickle序列化数据
    f.close()


def load_logfile(path):
    """
    加载日志文件（pickle格式）
    
    Args:
        path: 日志文件路径
    
    Returns:
        log: 加载的数据（字典格式，包含原始眼动数据）
    
    用途：
    - 加载Sitzmann数据集的原始眼动数据文件
    - 文件包含多个被试者对图像的注视数据
    
    数据格式：
    log是一个字典，包含'data'键，data是一个列表，每个元素是一个字典：
    {
        'data': [
            {
                'gaze_lat_lon': ...  # 注视点的球面坐标
                ...
            },
            ...
        ]
    }
    """
    log = pck.load(open(path, 'rb'), encoding='latin1')  # 'rb' = read binary
    # encoding='latin1'用于兼容旧版本的pickle文件
    return log


def twoDict(pack, key_a, key_b, data):
    """
    辅助函数：更新嵌套字典
    
    Args:
        pack: 目标字典
        key_a: 第一层键（例如: 'train'或'test'）
        key_b: 第二层键（例如: 图像文件名）
        data: 要存储的数据（例如: {'image': ..., 'scanpaths': ...}）
    
    Returns:
        pack: 更新后的字典
    
    功能：
    实现嵌套字典的更新：pack[key_a][key_b] = data
    如果key_a不存在，则创建新的嵌套字典
    
    示例：
        输入:
            pack = {}
            key_a = 'train'
            key_b = 'image001'
            data = {'image': tensor, 'scanpaths': array}
        
        输出:
            pack = {
                'train': {
                    'image001': {'image': tensor, 'scanpaths': array}
                }
            }
    """
    if key_a in pack:
        # 如果第一层键存在，更新第二层
        pack[key_a].update({key_b: data})
    else:
        # 如果第一层键不存在，创建新的嵌套字典
        pack.update({key_a: {key_b: data}})
    return pack


def create_info():
    """
    创建数据集信息字典（用于统计数据集的基本信息）
    
    Returns:
        info: 数据集信息字典，包含训练集和测试集的统计信息
    
    数据结构：
    {
        'train': {
            'num_image': 0,          # 训练集图像数量
            'num_scanpath': 0,       # 训练集扫描路径总数
            'scanpath_length': 0,    # 扫描路径长度（时间步数）
            'max_scan_length': 0     # 最大扫描路径长度
        },
        'test': {
            'num_image': 0,          # 测试集图像数量
            'num_scanpath': 0,       # 测试集扫描路径总数
            'scanpath_length': 0,    # 扫描路径长度
            'max_scan_length': 0     # 最大扫描路径长度
        }
    }
    
    用途：
    - 记录数据集的基本统计信息
    - 用于训练时的数据组织和验证
    """
    info = {
        'train': {
            'num_image': 0,  # 训练集图像数量（每个图像可能有多个扫描路径）
            'num_scanpath': 0,  # 训练集扫描路径总数（所有图像的所有扫描路径）
            'scanpath_length': 0,  # 每个扫描路径的长度（时间步数，例如: 30）
            'max_scan_length': 0  # 最大扫描路径长度（通常等于scanpath_length）
        },
        'test': {
            'num_image': 0,  # 测试集图像数量
            'num_scanpath': 0,  # 测试集扫描路径总数
            'scanpath_length': 0,  # 扫描路径长度
            'max_scan_length': 0  # 最大扫描路径长度
        }
    }
    return info


class Sitzmann_Dataset:
    """
    Sitzmann数据集处理类
    
    功能：
    - 加载Sitzmann 360度图像眼动数据集
    - 处理原始眼动数据（注视点序列）
    - 组织训练集和测试集
    - 数据增强（旋转图像生成更多训练样本）
    
    数据集特点：
    - 360度全景图像
    - 多个被试者观看每张图像
    - 记录30秒的注视点序列
    - 使用球面坐标（纬度、经度）表示注视点
    """

    def __init__(self):
        """
        初始化数据集处理类
        
        初始化路径配置和数据存储变量
        """
        super().__init__()
        self.images_path = Config.dic_Sitzmann['IMG_PATH']  # 图像文件夹路径
        self.gaze_path = Config.dic_Sitzmann['GAZE_PATH']  # 眼动数据文件夹路径
        self.test_set = Config.dic_Sitzmann['TEST_SET']  # 测试集图像文件名列表
        self.duration = 30  # 注视序列持续时间（秒），也是序列长度
        self.info = create_info()  # 数据集信息字典（统计信息）
        self.images_test_list = []  # 测试集图像路径列表
        self.images_train_list = []  # 训练集图像路径列表
        self.image_and_scanpath_dict = {}  # 图像和扫描路径的字典 {train/test: {image_name: {image, scanpaths}}}

    def mod(self, a, b):
        """
        取模运算（用于处理经度的周期性）
        
        Args:
            a: 被除数
            b: 除数
        
        Returns:
            r: a mod b（余数）
        
        用途：
        - 处理经度的周期性（-180度到180度是连续的）
        - 例如: 179度 + 3度 = 182度，但在球面上应该是-178度
        
        示例：
            mod(182, 360) = 182
            mod(-178, 360) = 182 (因为 -178 = -1 * 360 + 182)
        """
        c = a // b  # 整数除法
        r = a - c * b  # 余数
        return r

    def rotate(self, lat_lon, angle):
        """
        旋转球面坐标（用于数据增强）
        
        Args:
            lat_lon: 球面坐标, shape = (n, 2)
                     列0: 纬度 (度) - 范围[-90, 90]
                     列1: 经度 (度) - 范围[-180, 180]
            angle: 旋转角度 (度)，正值表示逆时针旋转
        
        Returns:
            rotate_lat_lon: 旋转后的球面坐标, shape = (n, 2)
        
        用途：
        - 数据增强：通过旋转生成更多的训练样本
        - 每张图像旋转6次（每次60度），生成6个训练样本
        
        注意：
        - 只旋转经度（纬度不变，因为旋转是绕垂直轴）
        - 处理经度的周期性（-180和180度相邻）
        
        示例：
            输入: lat_lon = [[0, 0], [0, 90]]  # 赤道，0度和90度经度
            angle = 60度
            输出: [[0, 60], [0, 150]]  # 经度旋转60度
        
        维度变化:
            输入: (n, 2) - n个点的球面坐标
            输出: (n, 2) - 旋转后的球面坐标
        """
        # 旋转经度：加上旋转角度，然后处理周期性
        # 先加180是为了将范围从[-180, 180]转换为[0, 360]
        # 旋转后再减180转回[-180, 180]范围
        new_lon = self.mod(lat_lon[:, 1] + 180 - angle, 360) - 180
        rotate_lat_lon = lat_lon.copy()  # 复制以避免修改原数组
        rotate_lat_lon[:, 1] = new_lon  # 更新经度
        return rotate_lat_lon

    def handle_empty(self, sphere_coords):
        """
        处理空值：插值填充无效的注视点
        
        Args:
            sphere_coords: 球面坐标序列, shape = (duration, 2)
                           列0: 纬度 (度)
                           列1: 经度 (度)
                           -999表示无效值（缺失数据）
        
        Returns:
            sphere_coords: 填充后的球面坐标, shape = (duration, 2)
            throw: 布尔值，True表示该序列应该被丢弃（无法修复）
        
        处理策略：
        1. 如果第一个点缺失：用下一个点填充
        2. 如果最后一个点缺失：用前一个点填充
        3. 如果中间点缺失：用前后两个点的线性插值填充
           - 注意处理经度的周期性（-180和180度相邻）
           - 如果前后点都缺失，则丢弃该序列
        
        维度变化:
            输入: (duration, 2) - 例如: (30, 2)
            输出: (duration, 2) - 例如: (30, 2)
        """
        # 找到所有空值的位置（标记为-999）
        empty_index = np.where(sphere_coords[:, 0] == -999)[0]
        throw = False  # 标记是否应该丢弃该序列

        # 遍历所有空值位置
        for _index in range(empty_index.shape[0]):
            if not throw:
                idx = empty_index[_index]  # 当前空值的位置

                # ===== 情况1: 第一个点缺失 =====
                if idx == 0:
                    # 如果下一个点有效，用下一个点填充
                    if sphere_coords[idx + 1, 0] != -999:
                        sphere_coords[idx, 0] = sphere_coords[idx + 1, 0]  # 纬度
                        sphere_coords[idx, 1] = sphere_coords[idx + 1, 1]  # 经度
                    else:
                        throw = True  # 下一个点也缺失，无法修复

                # ===== 情况2: 最后一个点缺失 =====
                elif idx == (self.duration - 1):
                    # 用前一个点填充
                    sphere_coords[idx, 0] = sphere_coords[idx - 1, 0]  # 纬度
                    sphere_coords[idx, 1] = sphere_coords[idx - 1, 1]  # 经度

                # ===== 情况3: 中间点缺失 =====
                else:
                    # 获取前后两个点的坐标
                    prev_x = sphere_coords[idx - 1, 1]  # 前一个点的经度
                    prev_y = sphere_coords[idx - 1, 0]  # 前一个点的纬度
                    next_x = sphere_coords[idx + 1, 1]  # 下一个点的经度
                    next_y = sphere_coords[idx + 1, 0]  # 下一个点的纬度

                    # 如果前后点都缺失，无法修复
                    if prev_x == -999 or next_x == -999:
                        throw = True
                    else:
                        # 线性插值填充
                        # 纬度：直接取平均值
                        sphere_coords[idx, 0] = 0.5 * (prev_y + next_y)

                        # 经度：需要考虑周期性
                        if np.abs(next_x - prev_x) <= 180:
                            # 情况A: 前后点距离 <= 180度，直接取平均值
                            sphere_coords[idx, 1] = 0.5 * (prev_x + next_x)
                        else:
                            # 情况B: 前后点距离 > 180度，需要处理周期性
                            # 例如: prev_x=-170, next_x=170，实际距离是20度（不是340度）
                            true_distance = 360 - np.abs(next_x - prev_x)

                            if next_x > prev_x:
                                # 例如: prev_x=-170, next_x=170
                                # 真实中点应该是180度（跨越了180/-180边界）
                                _temp = prev_x - true_distance / 2
                                if _temp < -180:
                                    _temp = 360 + _temp  # 处理周期性
                            else:
                                # 例如: prev_x=170, next_x=-170
                                _temp = prev_x + true_distance / 2
                                if _temp > 180:
                                    _temp = _temp - 360  # 处理周期性

                            sphere_coords[idx, 1] = _temp

        return sphere_coords, throw

    def sample_gaze_points(self, raw_data):
        """
        采样注视点：从原始数据中提取每秒钟的注视点
        
        Args:
            raw_data: 原始眼动数据, shape = (num_samples, 2)
                      列0: 纬度 (像素坐标，范围[0, 180])
                      列1: 经度 (像素坐标，范围[0, 360])
                      包含整个30秒的采样点
        
        Returns:
            sphere_coords: 采样后的球面坐标, shape = (duration, 2)
                           列0: 纬度 (度) - 范围[-90, 90]
                           列1: 经度 (度) - 范围[-180, 180]
        
        采样策略：
        - 将30秒的数据分成30个时间窗口（每秒一个）
        - 每个窗口选择一个代表性的注视点
        - 如果某个窗口没有有效数据，标记为-999
        
        维度变化:
            输入: (num_samples, 2) - 例如: (900, 2) - 900个采样点（30秒*30Hz）
            处理: 分成30个窗口，每个窗口选择一个点
            输出: (duration, 2) = (30, 2) - 30个时间步的球面坐标
        """
        fixation_coords = []  # 存储每个时间窗口的代表性注视点

        # 计算每个时间窗口的采样点数
        samples_per_bin = raw_data.shape[0] // self.duration
        # 例如: 如果raw_data有900个点，duration=30 -> samples_per_bin = 30

        # 将数据分成duration个时间窗口
        bins = raw_data[:samples_per_bin * self.duration].reshape([self.duration, -1, 2])
        # shape: (duration, samples_per_bin, 2)
        # 例如: (30, 30, 2) - 30个窗口，每个窗口30个采样点

        # 遍历每个时间窗口
        for bin in range(self.duration):
            # 找到该窗口中的有效点（排除(0, 0)，可能表示无效数据）
            valid_mask = (bins[bin, :, 0] != 0) & (bins[bin, :, 1] != 0)
            _fixation_coords = bins[bin, np.where(valid_mask)]
            # shape: (num_valid_points, 2) - 该窗口的有效点

            if _fixation_coords.shape[1] == 0:
                # 如果没有有效点，标记为缺失
                fixation_coords.append([-999, -999])
            else:
                # 选择第一个有效点作为该窗口的代表
                sample_vale = _fixation_coords[0, 0]  # 获取第一个有效点
                fixation_coords.append(sample_vale)

        # 堆叠所有窗口的点
        sphere_coords = np.vstack(fixation_coords)
        # shape: (duration, 2) = (30, 2)

        # 转换坐标系统：从像素坐标[0, 180]x[0, 360]转换为度坐标[-90, 90]x[-180, 180]
        sphere_coords = sphere_coords - [90, 180]
        # 纬度: [0, 180] -> [-90, 90]
        # 经度: [0, 360] -> [-180, 180]

        return sphere_coords  # shape: (duration, 2) = (30, 2)

    def get_train_set(self):
        """
        获取训练集：加载和处理训练数据，并进行数据增强
        
        处理流程：
        1. 加载原始眼动数据文件
        2. 提取每个被试者的注视点序列
        3. 采样注视点（每秒一个点）
        4. 处理缺失值（插值填充）
        5. 数据增强：每个原始图像旋转6次（每次60度），生成6个训练样本
        6. 转换为3D坐标并保存
        
        数据增强说明：
        - 每张原始图像生成6个旋转版本（旋转角度: -180, -120, -60, 0, 60, 120度）
        - 对应地旋转注视点坐标
        - 这样可以大大增加训练数据量
        
        输出：
        - 更新self.image_and_scanpath_dict，包含所有训练样本
        - 更新self.info['train']，包含训练集统计信息
        
        数据结构：
        image_and_scanpath_dict['train'][image_name] = {
            'image': tensor,      # shape: (3, 128, 256) - 图像
            'scanpaths': array    # shape: (num_paths, 30, 3) - 多个扫描路径的3D坐标
        }
        """
        # ===== 步骤1: 构建眼动数据文件路径 =====
        # 每6个图像对应一个原始图像（因为数据增强生成了6个版本）
        # 所以每隔6个取一个，找到对应的原始眼动数据文件
        all_files = [os.path.join(self.gaze_path,
                                  self.images_train_list[i].split('/')[-1].split('.')[0][:-2] + '.pck')
                     for i in range(0, len(self.images_train_list), 6)]
        # 例如: images_train_list[0, 6, 12, ...] 对应原始图像
        # 文件名格式: image_name_xx.png -> image_name.pck

        # ===== 步骤2: 加载所有眼动数据文件 =====
        runs_files = [load_logfile(logfile) for logfile in all_files]
        # runs_files: 列表，每个元素是一个字典，包含一个图像的所有被试者数据

        image_id = 0  # 当前处理的图像ID（包括所有旋转版本）
        original_image_id = 0  # 原始图像ID（不包括旋转版本）

        # ===== 步骤3: 遍历每个原始图像 =====
        for run in runs_files:
            # 为当前图像的所有扫描路径创建数组
            temple_gaze = np.zeros((len(run['data']), 30, 2))
            # shape: (num_subjects, 30, 2) - 每个被试者一个扫描路径，30个时间步，2个坐标(纬度,经度)
            scanpath_id = 0  # 有效扫描路径的计数

            # ===== 步骤4: 处理每个被试者的数据 =====
            for data in run['data']:
                relevant_fixations = data['gaze_lat_lon']
                # 原始注视点数据，shape可能为 (num_samples, 2) 或 (num_samples,)

                # 检查数据有效性
                if len(relevant_fixations.shape) > 1:
                    # 采样注视点：从原始数据中提取每秒钟的点
                    sphere_coords = self.sample_gaze_points(relevant_fixations)
                    # shape: (30, 2) - 30个时间步的球面坐标(纬度,经度)
                else:
                    continue  # 数据无效，跳过

                # 处理缺失值（插值填充）
                sphere_coords, throw = self.handle_empty(sphere_coords)
                # sphere_coords: (30, 2) - 处理后的坐标
                # throw: True表示该序列无法修复，应该丢弃

                if throw:
                    continue  # 丢弃无法修复的序列
                else:
                    # 保存有效的扫描路径
                    temple_gaze[scanpath_id] = torch.from_numpy(sphere_coords)
                    # shape: (30, 2) - 球面坐标(纬度,经度)
                    scanpath_id += 1

            # 只保留有效的扫描路径
            temple_gaze = temple_gaze[:scanpath_id]
            # shape: (num_valid_paths, 30, 2)
            original_image_id += 1

            # ===== 步骤5: 数据增强 - 生成6个旋转版本 =====
            for rotation_id in range(6):
                # 加载图像（6个旋转版本共用同一张图像，但注视点会旋转）
                image = image_process(self.images_train_list[image_id])
                # shape: (3, 128, 256) - 预处理后的图像

                # 创建3D坐标数组
                gaze_ = np.zeros((temple_gaze.shape[0], 30, 3))
                # shape: (num_valid_paths, 30, 3) - 用于存储3D坐标

                # 计算旋转角度（6个版本：-180, -120, -60, 0, 60, 120度）
                rotation_angle = rotation_id * 60 - 180
                # rotation_id=0 -> -180度
                # rotation_id=1 -> -120度
                # rotation_id=2 -> -60度
                # rotation_id=3 -> 0度
                # rotation_id=4 -> 60度
                # rotation_id=5 -> 120度

                # ===== 步骤6: 旋转每个扫描路径并转换为3D坐标 =====
                for scanpath_id in range(0, temple_gaze.shape[0]):
                    # 旋转球面坐标
                    rotated_coords = self.rotate(temple_gaze[scanpath_id], rotation_angle)
                    # temple_gaze[scanpath_id]: (30, 2) - 球面坐标(纬度,经度)
                    # rotated_coords: (30, 2) - 旋转后的球面坐标

                    # 转换为3D坐标
                    gaze_[scanpath_id] = sphere2xyz(torch.from_numpy(rotated_coords))
                    # sphere2xyz: (30, 2) -> (30, 3) - 转换为3D坐标(x, y, z)
                    # gaze_[scanpath_id]: (30, 3)

                    self.info['train']['num_scanpath'] += 1  # 更新统计信息

                # ===== 步骤7: 保存图像和扫描路径 =====
                dic = {"image": image, "scanpaths": gaze_}
                # image: (3, 128, 256) - 图像
                # scanpaths: (num_valid_paths, 30, 3) - 所有扫描路径的3D坐标

                # 保存到字典
                twoDict(self.image_and_scanpath_dict, "train",
                        self.images_train_list[image_id].split('/')[-1].split('.')[0],
                        dic)
                # 字典结构: {'train': {image_name: {'image': ..., 'scanpaths': ...}}}

                image_id += 1  # 下一个图像（旋转版本）

        # ===== 步骤8: 更新训练集统计信息 =====
        self.info['train']['num_image'] = image_id  # 图像总数（包括所有旋转版本）
        self.info['train']['scanpath_length'] = self.duration  # 扫描路径长度（30步）

    def get_test_set(self):
        """
        获取测试集：加载和处理测试数据
        
        处理流程：
        1. 加载原始眼动数据文件
        2. 提取每个被试者的注视点序列
        3. 采样注视点（每秒一个点）
        4. 处理缺失值（插值填充）
        5. 转换为3D坐标并保存
        
        注意：
        - 测试集不做数据增强（不旋转）
        - 处理方式与训练集类似，但更简单
        
        输出：
        - 更新self.image_and_scanpath_dict，包含所有测试样本
        - 更新self.info['test']，包含测试集统计信息
        
        数据结构：
        image_and_scanpath_dict['test'][image_name] = {
            'image': tensor,      # shape: (3, 128, 256) - 图像
            'scanpaths': array    # shape: (num_paths, 30, 3) - 多个扫描路径的3D坐标
        }
        """
        # ===== 步骤1: 构建眼动数据文件路径 =====
        # 测试集不需要考虑旋转，所以每个图像对应一个数据文件
        all_files = [os.path.join(self.gaze_path,
                                  self.images_test_list[i].split('/')[-1].split('.')[0] + '.pck')
                     for i in range(len(self.images_test_list))]
        # 例如: image001.png -> image001.pck

        # ===== 步骤2: 加载所有眼动数据文件 =====
        runs_files = [load_logfile(logfile) for logfile in all_files]
        # runs_files: 列表，每个元素包含一个图像的注视数据

        image_id = 0  # 当前处理的图像ID

        # ===== 步骤3: 遍历每个测试图像 =====
        for run in runs_files:
            scanpath_id = 0  # 有效扫描路径的计数

            # 创建3D坐标数组
            gaze_ = np.zeros((len(run['data']), 30, 3))
            # shape: (num_subjects, 30, 3) - 最多num_subjects个扫描路径，每个30步，3D坐标

            # 加载图像
            image = image_process(self.images_test_list[image_id])
            # shape: (3, 128, 256) - 预处理后的图像

            # ===== 步骤4: 处理每个被试者的数据 =====
            for data in run['data']:
                relevant_fixations = data['gaze_lat_lon']
                # 原始注视点数据

                # 检查数据有效性
                if len(relevant_fixations.shape) > 1:
                    # 采样注视点
                    sphere_coords = self.sample_gaze_points(relevant_fixations)
                    # shape: (30, 2) - 30个时间步的球面坐标
                else:
                    continue

                # 处理缺失值
                sphere_coords, throw = self.handle_empty(sphere_coords)

                if throw:
                    continue  # 丢弃无法修复的序列
                else:
                    # 转换为3D坐标
                    sphere_coords = torch.from_numpy(sphere_coords.copy())
                    # shape: (30, 2) - 球面坐标

                    gaze_[scanpath_id] = sphere2xyz(sphere_coords)
                    # sphere2xyz: (30, 2) -> (30, 3) - 转换为3D坐标
                    # gaze_[scanpath_id]: (30, 3)

                    scanpath_id += 1
                    self.info['test']['num_scanpath'] += 1  # 更新统计信息

            # ===== 步骤5: 只保留有效的扫描路径 =====
            gaze = gaze_[:scanpath_id]
            # shape: (num_valid_paths, 30, 3)

            # ===== 步骤6: 保存图像和扫描路径 =====
            dic = {"image": image, "scanpaths": gaze}
            # image: (3, 128, 256) - 图像
            # scanpaths: (num_valid_paths, 30, 3) - 所有扫描路径的3D坐标

            twoDict(self.image_and_scanpath_dict, "test",
                    self.images_test_list[image_id].split('/')[-1].split('.')[0],
                    dic)

            image_id += 1

        # ===== 步骤7: 更新测试集统计信息 =====
        self.info['test']['num_image'] = image_id  # 图像总数
        self.info['test']['scanpath_length'] = self.duration  # 扫描路径长度（30步）

    def run(self):
        """
        运行完整的数据处理流程
        
        流程：
        1. 扫描图像文件夹，分离训练集和测试集
        2. 处理训练集（包括数据增强）
        3. 处理测试集
        4. 添加统计信息到字典
        5. 返回完整的数据字典
        
        Returns:
            image_and_scanpath_dict: 完整的数据字典
            {
                'train': {
                    'image001': {'image': tensor, 'scanpaths': array},
                    'image001_rot0': {'image': tensor, 'scanpaths': array},
                    ...
                },
                'test': {
                    'image001': {'image': tensor, 'scanpaths': array},
                    ...
                },
                'info': {
                    'train': {...统计信息...},
                    'test': {...统计信息...}
                }
            }
        """
        # ===== 步骤1: 扫描图像文件夹，分离训练集和测试集 =====
        for file_name in os.listdir(self.images_path):
            if ".png" in file_name:
                if file_name in self.test_set:
                    # 测试集图像
                    self.images_test_list.append(os.path.join(self.images_path, file_name))
                else:
                    # 训练集图像（会通过数据增强生成多个版本）
                    self.images_train_list.append(os.path.join(self.images_path, file_name))

        # ===== 步骤2: 处理训练集（包括数据增强） =====
        self.get_train_set()
        # 每个原始图像生成6个旋转版本，所以训练集图像数量会增加

        # ===== 步骤3: 处理测试集 =====
        self.get_test_set()
        # 测试集不做数据增强

        # ===== 步骤4: 添加统计信息 =====
        self.image_and_scanpath_dict['info'] = self.info
        # 添加统计信息到字典，包含训练集和测试集的各种统计

        return self.image_and_scanpath_dict


# ============================================================================
# 主函数：训练入口
# ============================================================================

if __name__ == '__main__':
    """
    主函数：ScanDMM模型的训练入口
    
    使用方式：
    python scandmm_integrated.py --dataset ./Datasets/Sitzmann.pkl --lr 0.0003 --bs 64 --epochs 500
    
    主要步骤：
    1. 解析命令行参数
    2. 设置随机种子（保证可复现性）
    3. 创建DMM模型
    4. 加载数据集
    5. 创建训练器并开始训练
    """
    # ===== 步骤1: 创建参数解析器 =====
    parser = ArgumentParser(description='ScanDMM - 360度图像扫描路径预测模型')

    # 数据集路径
    parser.add_argument('--dataset', default='./Datasets/Sitzmann.pkl', type=str,
                        help='数据集路径（预处理后的pickle文件）, default = ./Datasets/Sitzmann.pkl')

    # 优化器参数
    parser.add_argument('--lr', default=Config.learning_rate, type=float,
                        help='学习率, default = 0.0003')
    parser.add_argument('--lr_decay', default=Config.lr_decay, type=float,
                        help='学习率衰减率（每个step衰减）, default = 0.99998')

    # 训练参数
    parser.add_argument('--seed', default=Config.seed, type=int,
                        help='随机种子（保证可复现性）, default = 1234')
    parser.add_argument('--bs', default=Config.mini_batch_size, type=int,
                        help='批次大小（batch size）, default = 64')
    parser.add_argument('--epochs', default=Config.num_epochs, type=int,
                        help='训练轮数, default = 500')
    parser.add_argument('--weight_decay', default=Config.weight_decay, type=float,
                        help='L2正则化权重（防止过拟合）, default = 2.0')

    # KL散度退火参数
    parser.add_argument('--annealing_epochs', default=Config.annealing_epochs, type=int,
                        help='KL散度退火轮数（训练初期逐渐增加KL项权重）, default = 10')
    parser.add_argument('--minimum_annealing_factor', default=Config.minimum_annealing_factor, type=float,
                        help='最小KL退火因子（退火开始时的权重）, default = 0.2')

    # 模型加载和保存
    parser.add_argument('--load_model', default=None, type=str,
                        help='预训练模型路径（用于继续训练）, default = None')
    parser.add_argument('--load_opt', default=None, type=str,
                        help='优化器状态路径（用于继续训练）, default = None')
    parser.add_argument('--save_root', default=Config.save_root, type=str,
                        help='模型保存路径, default = ./model/')

    # ===== 步骤2: 解析命令行参数 =====
    args = parser.parse_args()
    # 解析命令行传入的参数，未提供的参数使用默认值

    # ===== 步骤3: 设置随机种子（保证可复现性） =====
    torch.manual_seed(args.seed)
    # 设置PyTorch的随机种子，使结果可复现
    # 注意：如果使用GPU，还需要设置CUDA的随机种子

    # ===== 步骤4: 创建DMM模型 =====
    dmm = DMM(use_cuda=Config.use_cuda)
    # 创建深度马尔可夫模型实例
    # use_cuda: 是否使用GPU（如果可用）
    # 模型会自动初始化所有组件：CNN、GatedTransition、Combiner、Emitter、RNN等

    # ===== 步骤5: 加载数据集 =====
    train_dict = pickle.load(open(args.dataset, 'rb'))
    # 加载预处理后的数据集（pickle格式）
    # train_dict包含：
    # - 'train': {image_name: {'image': tensor, 'scanpaths': array}, ...}
    # - 'test': {image_name: {'image': tensor, 'scanpaths': array}, ...}
    # - 'info': {'train': {...}, 'test': {...}}

    # ===== 步骤6: 创建训练器并开始训练 =====
    trainer = Train(dmm, train_dict, args)
    # 创建训练器，传入模型、数据集和训练参数
    trainer.run()
    # 开始训练：
    # 1. 初始化优化器和SVI
    # 2. 准备训练数据
    # 3. 执行训练循环（多个epoch）
    # 4. 每个epoch：遍历所有小批次，执行梯度更新
    # 5. 每个epoch结束后保存检查点
