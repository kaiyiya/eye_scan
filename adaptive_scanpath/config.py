"""
配置文件 - AdaptiveScanPath模型训练配置
"""

import torch


class Config:
    """全局配置类"""

    # ============ 基础配置 ============
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4  # 数据加载线程数
    pin_memory = True  # 是否锁页内存

    # ============ 数据配置 ============
    # 图像配置
    image_height = 256  # 全景图像高度
    image_width = 512   # 全景图像宽度
    image_channels = 3

    # 扫描路径配置
    min_seq_len = 3     # 最短序列长度
    max_seq_len = 12    # 最长序列长度
    coord_dim = 3       # 坐标维度 [theta, phi, duration]

    # 数据集路径
    train_data_path = 'data/train'
    val_data_path = 'data/val'
    test_data_path = 'data/test'

    # ============ 模型配置 ============
    # 特征提取器
    feature_dim = 384       # 特征维度
    cnn_channels = [64, 128, 256, 384, 384]  # CNN各层通道数

    # 策略网络
    policy_hidden_dim = 256      # 策略网络隐藏层维度
    policy_dropout = 0.1         # Dropout比例

    # 停止网络
    stopping_hidden_dim = 128    # 停止网络隐藏层维度
    stopping_threshold = 0.5     # 停止阈值

    # 序列建模
    use_rnn = True              # 是否使用RNN建模时序
    rnn_hidden_dim = 256        # RNN隐藏层维度
    rnn_num_layers = 1          # RNN层数

    # ============ 训练配置 ============
    # 训练参数
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5

    # 学习率调度
    use_scheduler = True
    scheduler_type = 'cosine'  # 'cosine', 'step', 'plateau'
    scheduler_params = {
        'cosine': {'T_max': 100, 'eta_min': 1e-6},
        'step': {'step_size': 30, 'gamma': 0.1},
        'plateau': {'patience': 10, 'factor': 0.5}
    }

    # 早停配置
    use_early_stopping = True
    patience = 15
    min_delta = 1e-4

    # 梯度裁剪
    max_grad_norm = 1.0

    # ============ 损失函数配置 ============
    # 损失权重
    loss_weights = {
        'coord': 1.0,          # 坐标损失权重
        'duration': 0.1,       # 持续时间损失权重
        'smoothness': 0.05,    # 平滑性损失权重
        'stopping': 0.2        # 停止策略损失权重
    }

    # ============ 验证和测试配置 ============
    val_interval = 1           # 验证间隔（epoch）
    save_interval = 5          # 保存间隔（epoch）
    log_interval = 10          # 日志间隔（batch）

    # ============ 输出配置 ============
    output_dir = 'checkpoints'
    log_dir = 'logs'
    experiment_name = 'adaptive_scanpath'

    # 可视化配置
    vis_num_samples = 4        # 可视化样本数量
    save_predictions = True    # 是否保存预测结果

    # ============ 高级配置 ============
    # 混合精度训练
    use_amp = False           # 是否使用自动混合精度

    # 数据增强
    use_augmentation = True
    augmentation_params = {
        'horizontal_flip': 0.0,  # 球面图像不建议翻转
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.1
    }

    # 随机种子
    seed = 42

    @classmethod
    def update(cls, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise ValueError(f"Config没有属性: {key}")

    @classmethod
    def to_dict(cls):
        """转换为字典"""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }


def print_config():
    """打印配置信息"""
    print("=" * 60)
    print("配置信息")
    print("=" * 60)
    config_dict = Config.to_dict()
    for key, value in sorted(config_dict.items()):
        print(f"{key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
