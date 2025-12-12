"""
ScanDMM: 360度图像扫描路径预测的深度马尔可夫模型
整合版本 - 将所有模块合并到一个文件中，便于理解和学习

主要组件：
1. 配置参数 (Config)
2. 坐标卷积层 (CoordConv)
3. 球面CNN (Sphere CNN)
4. 深度马尔可夫模型 (DMM)
5. 训练和推理功能
6. 数据处理和工具函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import time
import logging
import pickle
import pickle as pck
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import torchvision.transforms as transforms
from argparse import ArgumentParser
from datetime import datetime
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
from numpy import sin, cos, tan, pi, arcsin, arctan
from torch.nn import Parameter

pi = math.pi


# ============================================================================
# 第一部分：配置参数 (Configuration)
# ============================================================================

class Config:
    """训练和模型配置参数"""
    # 数据集路径配置
    DATABASE_ROOT = '/home/......'

    dic_Sitzmann = {
        'IMG_PATH': DATABASE_ROOT + '/Sitzmann/rotation_imgs/imgs/',
        'GAZE_PATH': DATABASE_ROOT + '/Sitzmann/vr/',
        'TEST_SET': ['cubemap_0000.png', 'cubemap_0006.png', 'cubemap_0009.png']
    }

    # 训练参数
    image_size = [128, 256]  # 图像尺寸 [高度, 宽度]
    seed = 1234  # 随机种子
    num_epochs = 500  # 训练轮数
    learning_rate = 0.0003  # 学习率
    lr_decay = 0.99998  # 学习率衰减
    weight_decay = 2.0  # L2正则化权重
    mini_batch_size = 64  # 批次大小
    annealing_epochs = 10  # KL散度退火轮数
    minimum_annealing_factor = 0.2  # 最小退火因子

    # 模型加载和保存
    load_model = None
    load_opt = None
    save_root = './model/'

    # CUDA配置
    use_cuda = torch.cuda.is_available()


# ============================================================================
# 第二部分：坐标卷积层 (CoordConv)
# ============================================================================

class AddCoordsTh(nn.Module):
    """
    添加坐标通道到输入张量
    这是CoordConv的核心组件，通过在特征图上添加x和y坐标信息，
    帮助CNN学习空间位置相关的特征
    """

    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r  # 是否添加径向距离r

    def forward(self, input_tensor):
        """
        输入: (batch, c, x_dim, y_dim)
        输出: (batch, c+2(+1), x_dim, y_dim) - 添加了x和y坐标通道
        """
        batch_size_tensor = input_tensor.shape[0]

        # 生成x坐标通道
        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).unsqueeze(-1)
        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(-1)

        # 生成y坐标通道
        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).unsqueeze(1)
        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(-1)

        # 调整维度顺序
        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        # 归一化到[-1, 1]
        xx_channel = xx_channel.float() / (self.x_dim - 1) * 2 - 1
        yy_channel = yy_channel.float() / (self.y_dim - 1) * 2 - 1

        # 扩展到批次大小并移动到相同设备
        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1).to(input_tensor.device)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1).to(input_tensor.device)

        # 拼接坐标通道
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        # 可选：添加径向距离r
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class AddCoords(nn.Module):
    """自动推断尺寸的坐标添加层（替代实现）"""

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """自动从输入张量推断尺寸"""
        batch_size, _, x_dim, y_dim = input_tensor.size()

        # 生成坐标通道
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        # 归一化到[-1, 1]
        xx_channel = xx_channel.float() / (x_dim - 1) * 2 - 1
        yy_channel = yy_channel.float() / (y_dim - 1) * 2 - 1

        # 调整维度并扩展到批次大小
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) +
                torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """坐标卷积层：先添加坐标，再进行卷积"""

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


# ============================================================================
# 第三部分：球面CNN (Sphere CNN)
# ============================================================================

@lru_cache(None)
def get_xy(delta_phi, delta_theta):
    """
    计算球面卷积核的采样模式
    返回3x3卷积核在球面上的采样位置
    """
    return np.array([
        [
            (-tan(delta_theta), 1 / cos(delta_theta) * tan(delta_phi)),
            (0, tan(delta_phi)),
            (tan(delta_theta), 1 / cos(delta_theta) * tan(delta_phi)),
        ],
        [
            (-tan(delta_theta), 0),
            (1, 1),
            (tan(delta_theta), 0),
        ],
        [
            (-tan(delta_theta), -1 / cos(delta_theta) * tan(delta_phi)),
            (0, -tan(delta_phi)),
            (tan(delta_theta), -1 / cos(delta_theta) * tan(delta_phi)),
        ]
    ])


@lru_cache(None)
def cal_index(h, w, img_r, img_c):
    """
    计算球面卷积核的采样索引
    将等距圆柱投影(equirectangular)图像上的像素位置转换为球面坐标，
    然后计算3x3卷积核的采样位置
    
    参数:
        h, w: 图像高度和宽度
        img_r, img_c: 当前像素的行和列位置
    返回:
        9个采样位置的坐标 (3, 3, 2)
    """
    # 像素坐标转换为球面角度（弧度）
    phi = -((img_r + 0.5) / h * pi - pi / 2)  # 纬度
    theta = (img_c + 0.5) / w * 2 * pi - pi  # 经度

    delta_phi = pi / h  # 纬度步长
    delta_theta = 2 * pi / w  # 经度步长

    # 获取3x3采样模式
    xys = get_xy(delta_phi, delta_theta)
    x = xys[..., 0]
    y = xys[..., 1]

    # 计算球面上的新位置
    rho = np.sqrt(x ** 2 + y ** 2)
    v = arctan(rho)
    new_phi = arcsin(cos(v) * sin(phi) + y * sin(v) * cos(phi) / rho)
    new_theta = theta + arctan(x * sin(v) / (rho * cos(phi) * cos(v) - y * sin(phi) * sin(v)))

    # 球面坐标转回像素坐标
    new_r = (-new_phi + pi / 2) * h / pi - 0.5
    new_c = (new_theta + pi) * w / 2 / pi - 0.5

    # 处理等距圆柱投影的左右边界连续性（360度图像左右边界相邻）
    new_c = (new_c + w) % w
    new_result = np.stack([new_r, new_c], axis=-1)
    new_result[1, 1] = (img_r, img_c)  # 中心位置保持原位置

    return new_result


@lru_cache(None)
def _gen_filters_coordinates(h, w, stride):
    """生成所有位置的卷积核采样坐标"""
    co = np.array([[cal_index(h, w, i, j) for j in range(0, w, stride)]
                   for i in range(0, h, stride)])
    return np.ascontiguousarray(co.transpose([4, 0, 1, 2, 3]))


def gen_filters_coordinates(h, w, stride=1):
    """
    生成卷积核采样坐标
    返回: (2, H/stride, W/stride, 3, 3) - 2表示行和列坐标
    """
    assert (isinstance(h, int) and isinstance(w, int))
    return _gen_filters_coordinates(h, w, stride).copy()


def gen_grid_coordinates(h, w, stride=1):
    """
    生成用于grid_sample的采样网格坐标
    将采样坐标归一化到[-1, 1]范围，并调整为grid_sample需要的格式
    """
    coordinates = gen_filters_coordinates(h, w, stride).copy()
    # 归一化到[-1, 1]
    coordinates[0] = (coordinates[0] * 2 / h) - 1
    coordinates[1] = (coordinates[1] * 2 / w) - 1
    coordinates = coordinates[::-1]  # 反转维度顺序
    coordinates = coordinates.transpose(1, 3, 2, 4, 0)
    sz = coordinates.shape
    coordinates = coordinates.reshape(1, sz[0] * sz[1], sz[2] * sz[3], sz[4])
    return coordinates.copy()


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
        前向传播：
        1. 使用grid_sample根据球面几何进行采样
        2. 对采样结果应用3x3卷积
        """
        # 如果输入尺寸改变，重新生成采样网格
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        # 扩展到批次大小
        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        # 使用grid_sample进行球面采样
        x = F.grid_sample(x, grid, mode=self.mode, align_corners=True)
        # 应用3x3卷积（stride=3是因为grid_sample已经做了下采样）
        x = F.conv2d(x, self.weight, self.bias, stride=3)
        return x


class SphereMaxPool2D(nn.Module):
    """
    球面最大池化层
    在球面几何上进行最大池化操作
    注意：只支持3x3池化核
    """

    def __init__(self, stride=1, mode='bilinear'):
        super(SphereMaxPool2D, self).__init__()
        self.stride = stride
        self.mode = mode
        self.grid_shape = None
        self.grid = None
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

    def forward(self, x):
        """前向传播：先进行球面采样，再进行最大池化"""
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        return self.pool(F.grid_sample(x, grid, mode=self.mode, align_corners=True))


class Sphere_CNN(nn.Module):
    """
    球面CNN特征提取器
    用于从360度图像中提取特征，输出固定维度的特征向量
    """

    def __init__(self, out_put_dim):
        super(Sphere_CNN, self).__init__()
        self.output_dim = out_put_dim

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
        """前向传播：提取图像特征"""
        x = image

        # 添加坐标通道
        x = self.coord_conv(x)

        # 通过球面CNN提取特征
        x = self.leaky_relu1(self.image_norm1(self.image_conv1(x)))
        x = self.leaky_relu2(self.image_norm2(self.image_conv2(x)))
        x = self.leaky_relu3(self.image_norm3(self.image_conv3(x)))
        x = self.leaky_relu3_5(self.image_norm3_5(self.image_conv3_5(x)))

        # 标准卷积进一步提取特征
        x = self.leaky_relu4(self.image_norm4(self.image_conv4(x)))
        x = self.leaky_relu5(self.image_norm5(self.image_conv5(x)))

        # 展平并映射到输出维度
        x = self.activation(self.fc1(self.flatten(x)))

        return x


# ============================================================================
# 第四部分：深度马尔可夫模型 (Deep Markov Model)
# ============================================================================

class Emitter(nn.Module):
    """
    发射器：p(x_t | z_t)
    从隐状态z_t生成观测值x_t（扫描路径点）
    """

    def __init__(self, gaze_dim, z_dim, hidden_dim):
        super().__init__()
        self.lin_em_z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.lin_hidden_to_gaze = nn.Linear(hidden_dim, gaze_dim)
        self.lin_gaze_sig = nn.Linear(gaze_dim, gaze_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmod = nn.Sigmoid()

    def forward(self, z_t):
        """
        从隐状态z_t生成观测值的均值和方差
        输出归一化到[-1, 1]范围
        """
        mu = self.sigmod(self.lin_hidden_to_gaze(self.relu(self.lin_em_z_to_hidden(z_t)))) * 2 - 1
        sigma = self.softplus(self.lin_gaze_sig(self.relu(mu)))
        return mu, sigma


class GatedTransition(nn.Module):
    """
    门控转移：p(z_t | z_{t-1})
    定义隐状态之间的转移分布，使用门控机制控制转移强度
    """

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # 门控网络：决定转移的权重
        self.lin_gate_z_to_hidden_dim = nn.Linear(z_dim, hidden_dim)
        self.lin_gate_hidden_dim_to_z = nn.Linear(hidden_dim, z_dim)

        # 转移网络：计算新的隐状态
        self.lin_trans_2z_to_hidden = nn.Linear(2 * z_dim, hidden_dim)
        self.lin_trans_hidden_to_z = nn.Linear(hidden_dim, z_dim)

        # 方差网络
        self.lin_sig = nn.Linear(z_dim, z_dim)

        # 恒等映射（用于门控）
        self.lin_z_to_mu = nn.Linear(z_dim, z_dim)
        self.lin_z_to_mu.weight.data = torch.eye(z_dim)
        self.lin_z_to_mu.bias.data = torch.zeros(z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, img_feature=None):
        """
        计算从z_{t-1}到z_t的转移分布
        使用门控机制在恒等映射和转移网络之间进行插值
        """
        # 将前一个隐状态和图像特征拼接
        z_t_1_img = torch.cat((z_t_1, img_feature), dim=1)
        _z_t = self.lin_trans_hidden_to_z(self.relu(self.lin_trans_2z_to_hidden(z_t_1_img)))

        # 不确定性加权：计算门控权重
        weight = torch.sigmoid(self.lin_gate_hidden_dim_to_z(self.relu(self.lin_gate_z_to_hidden_dim(z_t_1))))

        # 高斯参数：在恒等映射和转移网络之间插值
        mu = (1 - weight) * self.lin_z_to_mu(z_t_1) + weight * _z_t
        sigma = self.softplus(self.lin_sig(self.relu(_z_t)))
        return mu, sigma


class Combiner(nn.Module):
    """
    组合器：q(z_t | z_{t-1}, x_{t:T})
    变分后验分布，结合前一个隐状态和未来观测信息
    """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_comb_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_mu = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_sig = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        结合前一个隐状态z_{t-1}和RNN隐藏状态h_rnn
        输出变分后验的均值和方差
        """
        h_combined = 0.5 * (self.tanh(self.lin_comb_z_to_hidden(z_t_1)) + h_rnn)
        mu = self.lin_hidden_to_mu(h_combined)
        sigma = self.softplus(self.lin_hidden_to_sig(h_combined))
        return mu, sigma


class DMM(nn.Module):
    """
    深度马尔可夫模型 (Deep Markov Model)
    用于建模360度图像上的扫描路径序列
    
    模型结构：
    - 使用球面CNN提取图像特征
    - 使用RNN处理观测序列（用于变分推断）
    - 使用门控转移建模隐状态转移
    - 使用发射器从隐状态生成观测
    """

    def __init__(
            self,
            input_dim=3,  # 输入维度（3D坐标：x, y, z）
            z_dim=100,  # 隐状态维度
            emission_dim=100,  # 发射器隐藏层维度
            transition_dim=200,  # 转移网络隐藏层维度
            rnn_dim=600,  # RNN隐藏层维度
            num_layers=1,  # RNN层数
            rnn_dropout_rate=0.1,  # RNN dropout率
            use_cuda=False,
    ):
        super().__init__()

        # 球面CNN：提取图像特征
        self.cnn = Sphere_CNN(out_put_dim=z_dim)

        # 模型组件
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)

        # 输入处理层
        self.input_to_z_dim = nn.Linear(input_dim, z_dim)
        self.twoZ_to_z_dim = nn.Linear(2 * z_dim, z_dim)
        self.tanh = nn.Tanh()

        # RNN：用于变分推断，处理反向序列
        rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=rnn_dropout_rate,
        )

        # 初始状态参数
        self.z_0 = nn.Parameter(torch.zeros(z_dim))  # 初始隐状态
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))  # RNN初始隐藏状态

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def model(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths, images=None, annealing_factor=1.0,
              predict=False):
        """
        生成模型：p(x_{1:T} | z_{1:T}), p(z_{1:T})
        
        参数:
            scanpaths: 扫描路径序列 (batch, T, 3)
            scanpaths_reversed: 反向序列（未使用，为兼容性保留）
            mask: 序列掩码，处理变长序列
            scanpath_lengths: 每个序列的长度
            images: 输入图像
            annealing_factor: KL散度退火因子
            predict: 是否为预测模式（不提供观测值）
        """
        T_max = scanpaths.size(1)
        pyro.module("dmm", self)

        # 初始化隐状态：结合初始状态和第一个观测
        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))

        # 提取图像特征
        img_features = self.cnn(images)

        # 对每个时间步进行采样
        with pyro.plate("z_minibatch", len(scanpaths)):
            for t in pyro.markov(range(1, T_max + 1)):
                # 计算转移分布的参数
                z_mu, z_sigma = self.trans(z_prev, img_features)

                # 从转移分布采样z_t
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, dist.Normal(z_mu, z_sigma)
                                      .mask(mask[:, t - 1: t]).to_event(1))

                # 计算发射分布的参数
                x_mu, x_sigma = self.emitter(z_t)

                # 训练模式：提供观测值
                if not predict:
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                .mask(mask[:, t - 1: t]).to_event(1), obs=scanpaths[:, t - 1, :])
                # 预测模式：不提供观测值，用于生成
                else:
                    pyro.sample("obs_x_%d" % t, dist.Normal(x_mu, x_sigma)
                                .mask(mask[:, t - 1: t]).to_event(1))
                z_prev = z_t

    def guide(self, scanpaths, scanpaths_reversed, mask, scanpath_lengths, images=None, annealing_factor=1.0):
        """
        变分后验：q(z_{1:T} | x_{1:T})
        
        使用RNN处理反向序列，结合前一个隐状态和未来观测信息
        """
        T_max = scanpaths.size(1)
        pyro.module("dmm", self)

        # 初始化RNN隐藏状态
        h_0_contig = self.h_0.expand(1, scanpaths.size(0), self.rnn.hidden_size).contiguous()

        # RNN处理反向序列
        rnn_output, _ = self.rnn(scanpaths_reversed, h_0_contig)
        rnn_output = poly.pad_and_reverse(rnn_output, scanpath_lengths)

        # 初始化隐状态
        z_prev = self.z_0.expand(scanpaths.size(0), self.z_0.size(0))
        z_prev = self.tanh(self.twoZ_to_z_dim(
            torch.cat((z_prev, self.tanh(self.input_to_z_dim(scanpaths[:, 0, :]))), dim=1)
        ))

        # 对每个时间步进行采样
        with pyro.plate("z_minibatch", len(scanpaths)):
            for t in pyro.markov(range(1, T_max + 1)):
                # 组合前一个隐状态和RNN输出
                z_mu, z_sigma = self.combiner(z_prev, rnn_output[:, t - 1, :])

                z_dist = dist.Normal(z_mu, z_sigma)
                assert z_dist.event_shape == ()
                assert z_dist.batch_shape[-2:] == (len(scanpaths), self.z_0.size(0))

                # 从变分后验采样z_t
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        z_dist.mask(mask[:, t - 1: t]).to_event(1),
                    )
                z_prev = z_t


# ============================================================================
# 第五部分：工具函数 (Utility Functions)
# ============================================================================

def rotate_images(input_path, output_path):
    """旋转360度图像进行数据增强"""
    for _, _, files in os.walk(input_path):
        for name in files:
            for i in range(6):
                angle = str(-180 + i * 60)
                cmd = 'ffmpeg -i ' + input_path + name + ' -vf v360=e:e:yaw=' + angle + ' ' + \
                      output_path + name.split('.')[0] + '_' + str(i) + '.png'
                os.system(cmd)


def image_process(path):
    """图像预处理：读取、调整大小、归一化"""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (Config.image_size[1], Config.image_size[0]), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    image = transform(image)
    return image


def sphere2plane(sphere_cord, height_width=None):
    """
    球面坐标转平面坐标
    输入: (lat, lon) shape = (n, 2)
    输出: (x, y) shape = (n, 2) - 归一化坐标
    """
    lat, lon = sphere_cord[:, 0], sphere_cord[:, 1]
    if height_width is None:
        y = (lat + 90) / 180
        x = (lon + 180) / 360
    else:
        y = (lat + 90) / 180 * height_width[0]
        x = (lon + 180) / 360 * height_width[1]
    return torch.cat((y.view(-1, 1), x.view(-1, 1)), 1)


def plane2sphere(plane_cord, height_width=None):
    """
    平面坐标转球面坐标
    输入: (x, y) shape = (n, 2)
    输出: (lat, lon) shape = (n, 2)
    """
    y, x = plane_cord[:, 0], plane_cord[:, 1]
    if (height_width is None) & (torch.any(plane_cord <= 1).item()):
        lat = (y - 0.5) * 180
        lon = (x - 0.5) * 360
    else:
        lat = (y / height_width[0] - 0.5) * 180
        lon = (x / height_width[1] - 0.5) * 360
    return torch.cat((lat.view(-1, 1), lon.view(-1, 1)), 1)


def sphere2xyz(shpere_cord):
    """
    球面坐标转3D笛卡尔坐标
    输入: (lat, lon) shape = (n, 2)
    输出: (x, y, z) shape = (n, 3) - 单位球面上的点
    """
    lat, lon = shpere_cord[:, 0], shpere_cord[:, 1]
    lat = lat / 180 * pi
    lon = lon / 180 * pi
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), 1)


def xyz2sphere(threeD_cord):
    """
    3D笛卡尔坐标转球面坐标
    输入: (x, y, z) shape = (n, 3)
    输出: (lat, lon) shape = (n, 2)
    """
    x, y, z = threeD_cord[:, 0], threeD_cord[:, 1], threeD_cord[:, 2]
    lon = torch.atan2(y, x)
    lat = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))
    lat = lat / pi * 180
    lon = lon / pi * 180
    return torch.cat((lat.view(-1, 1), lon.view(-1, 1)), 1)


def xyz2plane(threeD_cord, height_width=None):
    """
    3D坐标转平面坐标
    输入: (x, y, z) shape = (n, 3)
    输出: (x, y) shape = (n, 2)
    """
    sphere_cords = xyz2sphere(threeD_cord)
    plane_cors = sphere2plane(sphere_cords, height_width)
    return plane_cors


def plot_scanpaths(scanpaths, img_path, lengths, save_path, img_height=256, img_witdth=512):
    """
    可视化扫描路径
    在360度图像上绘制预测的扫描路径
    """
    image = cv2.resize(matplotlib.image.imread(img_path), (img_witdth, img_height))
    image_name = os.path.basename(img_path).split('.')[0]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, axs = plt.subplots(4, 5, constrained_layout=True)
    fig.set_size_inches(48, 24)

    # 绘制20条扫描路径
    for user_i in range(20):
        idx1 = int(user_i / 5)
        idx2 = user_i % 5

        points_x = []
        points_y = []
        for sample_i in range(lengths[user_i]):
            points_x.append(scanpaths[user_i][sample_i][1])
            points_y.append(scanpaths[user_i][sample_i][0])

        colors = cm.rainbow(np.linspace(0, 1, len(points_x)))

        previous_point = None
        for num, x, y, c in zip(range(0, len(points_x)), points_x, points_y, colors):
            x = x * img_witdth
            y = y * img_height
            markersize = 28.

            # 绘制连接线（处理360度图像的边界连续性）
            if previous_point is not None:
                if abs(previous_point[0] - x) < (img_witdth / 2):
                    axs[idx1, idx2].plot([x, previous_point[0]], [y, previous_point[1]],
                                         color='blue', linewidth=8., alpha=0.35)
                else:
                    h_diff = (y - previous_point[1]) / 2
                    if (x > previous_point[0]):
                        axs[idx1, idx2].plot([previous_point[0], 0],
                                             [previous_point[1], previous_point[1] + h_diff],
                                             color='blue', linewidth=8., alpha=0.35)
                        axs[idx1, idx2].plot([img_witdth, x], [previous_point[1] + h_diff, y],
                                             color='blue', linewidth=8., alpha=0.35)
                    else:
                        axs[idx1, idx2].plot([previous_point[0], img_witdth],
                                             [previous_point[1], previous_point[1] + h_diff],
                                             color='blue', linewidth=8., alpha=0.35)
                        axs[idx1, idx2].plot([0, x], [previous_point[1] + h_diff, y],
                                             color='blue', linewidth=8., alpha=0.35)
            previous_point = [x, y]
            axs[idx1, idx2].plot(x, y, marker='o', markersize=markersize, color=c, alpha=.8)
        axs[idx1, idx2].imshow(image)
        axs[idx1, idx2].axis('off')

    plt.savefig(os.path.join(save_path, 'sp_' + image_name + ".png"))
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.clf()
    plt.close('all')


# ============================================================================
# 第六部分：训练类 (Training)
# ============================================================================

class Train:
    """训练类：封装训练流程"""

    def __init__(self, model, train_package, args, log_path):
        self.dmm = model
        self.args = args
        self.log_path = log_path
        self.train_package = train_package

    def setup_logging(self):
        """设置日志"""
        if not os.path.exists('./Log'):
            os.makedirs('./Log')
        logging.basicConfig(level=logging.DEBUG, format="%(message)s", filename=self.log_path, filemode="w")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger("").addHandler(console)
        logging.info('Train_set:{}\nLearning Rate:{}\nBatch Size:{}\nEpochs:{}\n'.format(
            self.args.dataset, self.args.lr, self.args.bs, self.args.epochs
        ))

    def setup_adam(self):
        """设置优化器"""
        params = {
            "lr": self.args.lr,
            "betas": (0.96, 0.999),
            "clip_norm": 10,
            "lrd": self.args.lr_decay,
            "weight_decay": self.args.weight_decay,
        }
        self.adam = ClippedAdam(params)

    def setup_inference(self):
        """设置变分推断"""
        elbo = Trace_ELBO()
        self.svi = SVI(self.dmm.model, self.dmm.guide, self.adam, loss=elbo)

    def save_checkpoint(self, name):
        """保存模型检查点"""
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)
        torch.save(self.dmm.state_dict(), self.args.save_root + name)

    def load_checkpoint(self):
        """加载模型检查点"""
        assert exists(self.args.load_opt) and exists(self.args.load_model), "Load model: path error"
        logging.info("Loading model from %s" % self.args.load_model)
        self.dmm.load_state_dict(torch.load(self.args.load_model))
        self.adam.load(self.args.load_opt)

    def prepare_train_data(self):
        """准备训练数据"""
        _train = self.train_package['train']
        _info = self.train_package['info']['train']
        dic = {'sequences': None, 'sequence_lengths': None, 'images': None}
        scanpath_length = _info['scanpath_length']
        num_scanpath = _info['num_scanpath']
        image_index = np.zeros((num_scanpath))

        scanpath_set = np.zeros([num_scanpath, scanpath_length, 3])
        length_set = (np.ones(num_scanpath) * _info['scanpath_length']).astype(int)
        image_set = torch.zeros([num_scanpath, 3, Config.image_size[0], Config.image_size[1]])

        index, img_index = 0, 0
        for instance in _train:
            scanpaths = _train[instance]['scanpaths']
            for j in range(scanpaths.shape[0]):
                scanpath_set[index] = scanpaths[j]
                image_index[index] = img_index
                image_set[index] = _train[instance]['image']
                index += 1
            img_index += 1

        dic['sequences'] = torch.from_numpy(scanpath_set).float()
        dic['sequence_lengths'] = torch.from_numpy(length_set.astype(int))
        dic['image_index'] = torch.from_numpy(image_index.astype(int))
        dic['images'] = image_set

        return dic

    def get_mini_batch(self, mini_batch_indices, sequences, seq_lengths, images, cuda=False):
        """获取小批次数据"""
        seq_lengths = seq_lengths[mini_batch_indices]
        _, sorted_seq_length_indices = torch.sort(seq_lengths)
        sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
        sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
        sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]

        T_max = torch.max(seq_lengths)
        mini_batch = sequences[sorted_mini_batch_indices, 0:T_max, :]
        mini_batch_images = images[sorted_mini_batch_indices]
        mini_batch_reversed = poly.reverse_sequences(mini_batch, sorted_seq_lengths)
        mini_batch_mask = poly.get_mini_batch_mask(mini_batch, sorted_seq_lengths)

        if cuda:
            mini_batch = mini_batch.cuda()
            mini_batch_mask = mini_batch_mask.cuda()
            mini_batch_reversed = mini_batch_reversed.cuda()
            mini_batch_images = mini_batch_images.cuda()

        mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(mini_batch_reversed, sorted_seq_lengths,
                                                                batch_first=True)

        return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths, mini_batch_images

    def process_minibatch(self, epoch, which_mini_batch, shuffled_indices):
        """处理一个小批次"""
        # KL散度退火
        if self.args.annealing_epochs > 0 and epoch < self.args.annealing_epochs:
            min_af = self.args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * (
                    float(which_mini_batch + epoch * self.N_mini_batches + 1)
                    / float(self.args.annealing_epochs * self.N_mini_batches)
            )
        else:
            annealing_factor = 1.0

        mini_batch_start = which_mini_batch * self.args.bs
        mini_batch_end = np.min([(which_mini_batch + 1) * self.args.bs, self.N_sequences])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]

        (mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, mini_batch_images) = \
            self.get_mini_batch(mini_batch_indices, self.sequences, self.seq_lengths, self.images,
                                cuda=Config.use_cuda)

        # 执行一步梯度更新
        loss = self.svi.step(
            scanpaths=mini_batch,
            scanpaths_reversed=mini_batch_reversed,
            mask=mini_batch_mask,
            scanpath_lengths=mini_batch_seq_lengths,
            images=mini_batch_images,
            annealing_factor=annealing_factor,
        )
        return loss

    def run(self):
        """运行训练"""
        self.setup_adam()
        self.setup_inference()
        self.setup_logging()

        if self.args.load_opt is not None and self.args.load_model is not None:
            self.load_checkpoint()

        train_data = self.prepare_train_data()
        self.sequences = train_data["sequences"]
        self.seq_lengths = train_data["sequence_lengths"]
        self.images = train_data["images"]
        self.N_sequences = len(self.seq_lengths)
        self.N_time_slices = float(torch.sum(self.seq_lengths))
        self.N_mini_batches = int(self.N_sequences / self.args.bs +
                                  int(self.N_sequences % self.args.bs > 0))

        logging.info("N_train_data: %d\t avg. training seq. length: %.2f\t N_mini_batches: %d"
                     % (self.N_sequences, self.seq_lengths.float().mean(), self.N_mini_batches))

        times = [time.time()]

        for epoch in range(self.args.epochs):
            epoch_nll = 0.0
            shuffled_indices = torch.randperm(self.N_sequences)

            for which_mini_batch in range(self.N_mini_batches):
                epoch_nll += self.process_minibatch(epoch, which_mini_batch, shuffled_indices)

            times.append(time.time())
            epoch_time = times[-1] - times[-2]
            logging.info("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)"
                         % (epoch, epoch_nll / self.N_time_slices, epoch_time))

            save_name = 'model_lr-{}_bs-{}_epoch-{}.pkl'.format(
                self.args.lr, self.args.bs, epoch)

            self.save_checkpoint(save_name)


# ============================================================================
# 第七部分：推理类 (Inference)
# ============================================================================

class Inference:
    """推理类：用于生成扫描路径"""

    def __init__(self, model, img_path, n_scanpaths, length, output_path, if_plot=False):
        self.dmm = model
        self.img_path = img_path
        self.n_scanpaths = n_scanpaths
        self.length = length
        self.output_path = output_path
        self.if_plot = if_plot

    def create_random_starting_points(self, num_points):
        """创建随机起始点（从赤道偏置分布采样）"""
        y, x = [], []

        for i in range(num_points):
            while True:
                temp = np.random.normal(loc=0, scale=0.2)
                if (temp <= 1) and (temp >= -1):
                    y.append(temp)
                    break
            x.append(np.random.uniform(-1, 1))

        cords = np.vstack((np.array(y) * 90, np.array(x) * 180)).swapaxes(0, 1)
        cords = sphere2xyz(torch.from_numpy(cords))
        return cords

    def summary(self, samples):
        """整理预测结果：将3D坐标转换为2D平面坐标"""
        obs = None

        for index in range(int(len(samples) / 2)):
            name = 'obs_x_' + str(index + 1)
            temp = samples[name].reshape([-1, 3])
            # 归一化到单位球面
            its_sum = torch.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
            temp = temp / torch.unsqueeze(its_sum, 1)

            # 转换为2D坐标
            if obs is not None:
                obs = torch.cat((obs, torch.unsqueeze(xyz2plane(temp), dim=0)), dim=0)
            else:
                obs = torch.unsqueeze(xyz2plane(temp), dim=0)

        obs = torch.transpose(obs, 0, 1)
        return obs

    def predict(self):
        """执行预测"""
        num_samples = 1
        rep_num = self.n_scanpaths // num_samples
        predictive = Predictive(self.dmm.model, num_samples=num_samples)

        for _, _, files in os.walk(self.img_path):
            num_img = len(files)
            count = 0
            for img in files:
                count += 1
                img_path = os.path.join(self.img_path, img)
                image_tensor = torch.unsqueeze(image_process(img_path), dim=0).repeat([rep_num, 1, 1, 1])
                starting_points = torch.unsqueeze(
                    self.create_random_starting_points(rep_num), dim=1).to(torch.float32)
                _scanpaths = starting_points.repeat([1, self.length, 1])

                test_mask = torch.ones([rep_num, self.length])

                device = next(self.dmm.parameters()).device
                test_batch = _scanpaths.to(device)
                test_batch_mask = test_mask.to(device)
                test_batch_images = image_tensor.to(device)

                with torch.no_grad():
                    samples = predictive(scanpaths=test_batch,
                                         scanpaths_reversed=None,
                                         mask=test_batch_mask,
                                         scanpath_lengths=None,
                                         images=test_batch_images,
                                         predict=True)

                    scanpaths = self.summary(samples).cpu().numpy()

                    print('[{}]/[{}]:{} {} scanpaths are produced\nSaving to {}'
                          .format(count, num_img, img, scanpaths.shape[0], self.output_path))
                    save_name = img.split('.')[0] + '.npy'

                    if not os.path.exists(self.output_path):
                        os.makedirs(self.output_path)

                    np.save(os.path.join(self.output_path, save_name), scanpaths)

                    if self.if_plot:
                        print('Begin to plot scanpaths')
                        length_tensor = (torch.ones(self.n_scanpaths) * self.length).int()
                        plot_scanpaths(scanpaths, img_path, length_tensor.numpy(), save_path=self.output_path)


# ============================================================================
# 第八部分：数据处理 (Data Processing)
# ============================================================================

def save_file(file_name, data):
    """保存文件"""
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def load_logfile(path):
    """加载日志文件"""
    log = pck.load(open(path, 'rb'), encoding='latin1')
    return log


def twoDict(pack, key_a, key_b, data):
    """辅助函数：更新嵌套字典"""
    if key_a in pack:
        pack[key_a].update({key_b: data})
    else:
        pack.update({key_a: {key_b: data}})
    return pack


def create_info():
    """创建数据集信息字典"""
    info = {
        'train': {
            'num_image': 0,
            'num_scanpath': 0,
            'scanpath_length': 0,
            'max_scan_length': 0
        },
        'test': {
            'num_image': 0,
            'num_scanpath': 0,
            'scanpath_length': 0,
            'max_scan_length': 0
        }
    }
    return info


def summary(info):
    """打印数据集摘要信息"""
    print("\n============================================")
    print("Train_set:   {} images, {} scanpaths,  length ={}".
          format(info['train']['num_image'], info['train']['num_scanpath'], info['train']['scanpath_length']))
    print("Test_set:    {} images, {} scanpaths,  length ={}".
          format(info['test']['num_image'], info['test']['num_scanpath'], info['test']['scanpath_length']))
    print("============================================\n")


class Sitzmann_Dataset:
    """Sitzmann数据集处理类"""

    def __init__(self):
        super().__init__()
        self.images_path = Config.dic_Sitzmann['IMG_PATH']
        self.gaze_path = Config.dic_Sitzmann['GAZE_PATH']
        self.test_set = Config.dic_Sitzmann['TEST_SET']
        self.duration = 30
        self.info = create_info()
        self.images_test_list = []
        self.images_train_list = []
        self.image_and_scanpath_dict = {}

    def mod(self, a, b):
        """取模运算"""
        c = a // b
        r = a - c * b
        return r

    def rotate(self, lat_lon, angle):
        """旋转球面坐标"""
        new_lon = self.mod(lat_lon[:, 1] + 180 - angle, 360) - 180
        rotate_lat_lon = lat_lon
        rotate_lat_lon[:, 1] = new_lon
        return rotate_lat_lon

    def handle_empty(self, sphere_coords):
        """处理空值：插值填充无效的注视点"""
        empty_index = np.where(sphere_coords[:, 0] == -999)[0]
        throw = False
        for _index in range(empty_index.shape[0]):
            if not throw:
                if empty_index[_index] == 0:
                    if sphere_coords[empty_index[_index] + 1, 0] != -999:
                        sphere_coords[empty_index[_index], 0] = sphere_coords[empty_index[_index] + 1, 0]
                        sphere_coords[empty_index[_index], 1] = sphere_coords[empty_index[_index] + 1, 1]
                    else:
                        throw = True
                elif empty_index[_index] == (self.duration - 1):
                    sphere_coords[empty_index[_index], 0] = sphere_coords[empty_index[_index] - 1, 0]
                    sphere_coords[empty_index[_index], 1] = sphere_coords[empty_index[_index] - 1, 1]
                else:
                    prev_x = sphere_coords[empty_index[_index] - 1, 1]
                    prev_y = sphere_coords[empty_index[_index] - 1, 0]
                    next_x = sphere_coords[empty_index[_index] + 1, 1]
                    next_y = sphere_coords[empty_index[_index] + 1, 0]

                    if prev_x == -999 or next_x == -999:
                        throw = True
                    else:
                        sphere_coords[empty_index[_index], 0] = 0.5 * (prev_y + next_y)
                        if np.abs(next_x - prev_x) <= 180:
                            sphere_coords[empty_index[_index], 1] = 0.5 * (prev_x + next_x)
                        else:
                            true_distance = 360 - np.abs(next_x - prev_x)
                            if next_x > prev_x:
                                _temp = prev_x - true_distance / 2
                                if _temp < -180:
                                    _temp = 360 + _temp
                            else:
                                _temp = prev_x + true_distance / 2
                                if _temp > 180:
                                    _temp = _temp - 360
                            sphere_coords[empty_index[_index], 1] = _temp

        return sphere_coords, throw

    def sample_gaze_points(self, raw_data):
        """采样注视点：从原始数据中提取每秒钟的注视点"""
        fixation_coords = []
        samples_per_bin = raw_data.shape[0] // self.duration
        bins = raw_data[:samples_per_bin * self.duration].reshape([self.duration, -1, 2])
        for bin in range(self.duration):
            _fixation_coords = bins[bin, np.where((bins[bin, :, 0] != 0) & (bins[bin, :, 1] != 0))]
            if _fixation_coords.shape[1] == 0:
                fixation_coords.append([-999, -999])
            else:
                sample_vale = _fixation_coords[0, 0]
                fixation_coords.append(sample_vale)
        sphere_coords = np.vstack(fixation_coords) - [90, 180]
        return sphere_coords

    def get_train_set(self):
        """获取训练集"""
        all_files = [os.path.join(self.gaze_path, self.images_train_list[i].split('/')[-1].split('.')[0][:-2] + '.pck')
                     for i in range(0, len(self.images_train_list), 6)]

        runs_files = [load_logfile(logfile) for logfile in all_files]

        image_id = 0
        original_image_id = 0

        for run in runs_files:
            temple_gaze = np.zeros((len(run['data']), 30, 2))
            scanpath_id = 0

            for data in run['data']:
                relevant_fixations = data['gaze_lat_lon']

                if len(relevant_fixations.shape) > 1:
                    sphere_coords = self.sample_gaze_points(relevant_fixations)
                else:
                    continue

                sphere_coords, throw = self.handle_empty(sphere_coords)

                if throw:
                    continue
                else:
                    temple_gaze[scanpath_id] = torch.from_numpy(sphere_coords)
                    scanpath_id += 1

            temple_gaze = temple_gaze[:scanpath_id]
            original_image_id += 1

            for rotation_id in range(6):
                image = image_process(self.images_train_list[image_id])
                gaze_ = np.zeros((temple_gaze.shape[0], 30, 3))
                rotation_angle = rotation_id * 60 - 180

                for scanpath_id in range(0, temple_gaze.shape[0]):
                    gaze_[scanpath_id] = sphere2xyz(
                        torch.from_numpy(self.rotate(temple_gaze[scanpath_id], rotation_angle)))

                    self.info['train']['num_scanpath'] += 1

                dic = {"image": image, "scanpaths": gaze_}

                twoDict(self.image_and_scanpath_dict, "train",
                        self.images_train_list[image_id].split('/')[-1].split('.')[0],
                        dic)

                print('Processing - {}. [Filter out {} scanpaths]'
                      .format(self.images_train_list[image_id].split('/')[-1],
                              len(run['data']) - scanpath_id - 1))

                image_id += 1

        self.info['train']['num_image'] = image_id
        self.info['train']['scanpath_length'] = self.duration

    def get_test_set(self):
        """获取测试集"""
        all_files = [os.path.join(self.gaze_path, self.images_test_list[i].split('/')[-1].split('.')[0] + '.pck')
                     for i in range(len(self.images_test_list))]

        runs_files = [load_logfile(logfile) for logfile in all_files]

        image_id = 0

        for run in runs_files:
            scanpath_id = 0
            gaze_ = np.zeros((len(run['data']), 30, 3))
            image = image_process(self.images_test_list[image_id])

            for data in run['data']:
                relevant_fixations = data['gaze_lat_lon']

                if len(relevant_fixations.shape) > 1:
                    sphere_coords = self.sample_gaze_points(relevant_fixations)
                else:
                    continue

                sphere_coords, throw = self.handle_empty(sphere_coords)

                if throw:
                    continue
                else:
                    sphere_coords = torch.from_numpy(sphere_coords.copy())
                    gaze_[scanpath_id] = sphere2xyz(sphere_coords)
                    scanpath_id += 1
                    self.info['test']['num_scanpath'] += 1

            gaze = gaze_[:scanpath_id]

            dic = {"image": image, "scanpaths": gaze}

            twoDict(self.image_and_scanpath_dict, "test",
                    self.images_test_list[image_id].split('/')[-1].split('.')[0],
                    dic)

            print('Processing - {}. [Filter out {} scanpaths]'
                  .format(self.images_test_list[image_id].split('/')[-1], gaze_.shape[0] - scanpath_id))

            image_id += 1

        self.info['test']['num_image'] = image_id
        self.info['test']['scanpath_length'] = self.duration

    def run(self):
        """运行数据处理"""
        for file_name in os.listdir(self.images_path):
            if ".png" in file_name:
                if file_name in self.test_set:
                    self.images_test_list.append(os.path.join(self.images_path, file_name))
                else:
                    self.images_train_list.append(os.path.join(self.images_path, file_name))

        print('\nProcessing [Training Set]\n')
        self.get_train_set()

        print('\nProcessing [Test Set]\n')
        self.get_test_set()

        self.image_and_scanpath_dict['info'] = self.info
        return self.image_and_scanpath_dict


# ============================================================================
# 主函数示例
# ============================================================================

if __name__ == '__main__':
    # 训练示例
    parser = ArgumentParser(description='ScanDMM')
    parser.add_argument('--seed', default=Config.seed, type=int, help='seed, default = 1234')
    parser.add_argument('--dataset', default='./Datasets/Sitzmann.pkl', type=str,
                        help='dataset path, default = ./Datasets/Sitzmann.pkl')
    parser.add_argument('--lr', default=Config.learning_rate, type=float, help='learning rate, default = 0.0003')
    parser.add_argument('--bs', default=Config.mini_batch_size, type=int, help='mini batch size, default = 64')
    parser.add_argument('--lr_decay', default=Config.lr_decay, type=float,
                        help='learning rate decay, default = 0.99998')
    parser.add_argument('--epochs', default=Config.num_epochs, type=int, help='num_epochs, default = 500')
    parser.add_argument('--weight_decay', default=Config.weight_decay, type=float,
                        help='L2 regularization term, default = 2.0')
    parser.add_argument('--annealing_epochs', default=Config.annealing_epochs, type=int,
                        help='KL annealing, default = 10')
    parser.add_argument('--minimum_annealing_factor', default=Config.minimum_annealing_factor, type=float,
                        help='minimum KL annealing factor, default = 0.2')
    parser.add_argument('--load_model', default=None, type=str,
                        help='path of pre-trained model, default = None')
    parser.add_argument('--load_opt', default=None, type=str,
                        help='path of optimizer state, default = None')
    parser.add_argument('--save_root', default=Config.save_root, type=str,
                        help='model save path, default = ./model/')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dmm = DMM(use_cuda=Config.use_cuda)

    train_log = './Log/lr-{}_bs-{}_dy-{}_epo-{}_{}.txt'.format(
        args.lr, args.bs, args.lr_decay, args.epochs,
        datetime.now().strftime("%I:%M%p on %B %d, %Y"))

    train_dict = pickle.load(open(args.dataset, 'rb'))

    trainer = Train(dmm, train_dict, args, train_log)
    trainer.run()
