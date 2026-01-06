"""
加载ScanDMM的Sitzmann数据集
将3D坐标转换为球面坐标，适配我们的模型
"""

import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os


class ScanDMMDataset(Dataset):
    """
    ScanDMM Sitzmann数据集加载器

    数据格式：
    - 图像: Tensor[3, H, W]
    - 扫描路径: Tensor[n_scanpaths, T, 3] (x, y, z) -> 转换为 (theta, phi, duration)
    """

    def __init__(
        self,
        pkl_path,
        split='train',
        image_size=(256, 512),  # 调整图像大小
        max_seq_len=12,
        transform=None
    ):
        """
        Args:
            pkl_path: Sitzmann.pkl文件路径
            split: 'train' 或 'test'
            image_size: 目标图像大小 (height, width)
            max_seq_len: 最大序列长度
            transform: 图像变换（可选）
        """
        self.pkl_path = pkl_path
        self.split = split
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.transform = transform

        # 加载数据
        print(f"加载ScanDMM {split}数据集: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)

        print(f"数据类型: {type(data_dict)}")
        print(f"顶层键: {list(data_dict.keys()) if isinstance(data_dict, dict) else 'N/A'}")

        # 检查split是否存在
        if split not in data_dict:
            print(f"错误: split '{split}' 不在数据中")
            print(f"可用的键: {list(data_dict.keys())}")
            raise ValueError(f"split '{split}' not found in data")

        # 提取对应split的数据
        self.samples = []
        split_data = data_dict[split]
        print(f"{split}数据类型: {type(split_data)}")

        if not isinstance(split_data, dict):
            print(f"错误: {split} 数据不是字典类型")
            raise TypeError(f"{split} data is not a dictionary")

        print(f"处理 {len(split_data)} 个图像...")

        for idx, (img_name, img_data) in enumerate(split_data.items()):
            try:
                if not isinstance(img_data, dict):
                    print(f"  警告: {img_name} 的数据不是字典，跳过")
                    continue

                if 'image' not in img_data or 'scanpaths' not in img_data:
                    print(f"  警告: {img_name} 缺少 'image' 或 'scanpaths' 键")
                    print(f"    可用的键: {list(img_data.keys())}")
                    continue

                image = img_data['image']  # Tensor[3, H, W]
                scanpaths = img_data['scanpaths']  # Tensor[n, T, 3]

                print(f"  [{idx+1}/{len(split_data)}] {img_name}")
                print(f"    图像形状: {image.shape if hasattr(image, 'shape') else type(image)}")
                print(f"    扫描路径形状: {scanpaths.shape if hasattr(scanpaths, 'shape') else type(scanpaths)}")

                # 转换为tensor（如果还不是）
                if not isinstance(image, torch.Tensor):
                    image = torch.tensor(image)

                if not isinstance(scanpaths, torch.Tensor):
                    scanpaths = torch.tensor(scanpaths)

                # 调整图像大小
                image = self._resize_image(image, image_size)

                # 转换扫描路径坐标
                scanpaths_spherical = self._convert_scanpaths(scanpaths)

                self.samples.append({
                    'image': image,
                    'scanpaths': scanpaths_spherical,
                    'original_name': img_name
                })

            except Exception as e:
                print(f"  错误处理 {img_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"✓ 加载完成: {len(self.samples)} 张图像")

    def _resize_image(self, image, target_size):
        """调整图像大小"""
        _, h, w = image.shape
        target_h, target_w = target_size

        # 使用双线性插值调整大小
        image_pil = transforms.ToPILImage()(image)
        image_resized = transforms.Resize(target_size)(image_pil)
        image_tensor = transforms.ToTensor()(image_resized)

        return image_tensor

    def _convert_scanpaths(self, scanpaths):
        """
        将3D笛卡尔坐标转换为球面坐标

        Args:
            scanpaths: Tensor[n, T, 3] (x, y, z)

        Returns:
            Tensor[n, T, 3] (theta, phi, duration)
        """
        n_scanpaths, T, _ = scanpaths.shape

        # 转换每个扫描路径
        converted = []
        for i in range(n_scanpaths):
            scanpath = scanpaths[i]  # (T, 3)

            # 提取3D坐标
            x = scanpath[:, 0]
            y = scanpath[:, 1]
            z = scanpath[:, 2]

            # 计算球面坐标
            r = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
            theta = torch.atan2(x, z)  # [-π, π]
            phi = torch.asin(y / (r + 1e-8))  # [-π/2, π/2]

            # 计算持续时间（简单估计：假设1Hz采样）
            duration = torch.ones(T) * 0.5  # 默认0.5秒

            # 组合
            scanpath_spherical = torch.stack([theta, phi, duration], dim=1)
            converted.append(scanpath_spherical)

        return torch.stack(converted)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回一个样本

        Returns:
            image: (3, H, W)
            scanpath: (max_seq_len, 3) 随机选择一条扫描路径
            length: int 实际序列长度
        """
        sample = self.samples[idx]
        image = sample['image']
        scanpaths = sample['scanpaths']  # (n, T, 3)

        # 随机选择一条扫描路径
        n_scanpaths = scanpaths.shape[0]
        selected_idx = np.random.randint(0, n_scanpaths)
        scanpath = scanpaths[selected_idx]  # (T, 3)

        # 获取实际长度
        T = scanpath.shape[0]
        length = min(T, self.max_seq_len)

        # 截断或填充
        if T >= self.max_seq_len:
            scanpath = scanpath[:self.max_seq_len]
        else:
            # 填充
            padding = torch.zeros(self.max_seq_len - T, 3)
            scanpath = torch.cat([scanpath, padding], dim=0)

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'scanpath': scanpath,
            'length': length
        }


def create_scandmm_dataloaders(
    pkl_path,
    batch_size=16,
    num_workers=4,
    image_size=(256, 512),
    max_seq_len=12
):
    """
    创建ScanDMM数据加载器

    Args:
        pkl_path: Sitzmann.pkl文件路径
        batch_size: batch大小
        num_workers: 数据加载线程数
        image_size: 图像大小
        max_seq_len: 最大序列长度

    Returns:
        train_loader, val_loader
    """
    # 训练集
    train_dataset = ScanDMMDataset(
        pkl_path=pkl_path,
        split='train',
        image_size=image_size,
        max_seq_len=max_seq_len
    )

    # 验证集
    val_dataset = ScanDMMDataset(
        pkl_path=pkl_path,
        split='test',
        image_size=image_size,
        max_seq_len=max_seq_len
    )

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")

    return train_loader, val_loader


def test_scandmm_dataset():
    """测试ScanDMM数据集加载"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ScanDMM数据路径
    pkl_path = '../ScanDMM-master/Datasets/Sitzmann.pkl'

    if not os.path.exists(pkl_path):
        print(f"错误: 未找到数据集 {pkl_path}")
        print("请确认ScanDMM-master/Datasets/Sitzmann.pkl存在")
        return

    print("=" * 60)
    print("测试ScanDMM数据集加载")
    print("=" * 60)

    # 创建数据集
    try:
        train_dataset = ScanDMMDataset(
            pkl_path=pkl_path,
            split='train',
            image_size=(256, 512),
            max_seq_len=12
        )

        # 测试获取样本
        sample = train_dataset[0]
        print(f"\n✓ 数据集加载成功")
        print(f"  图像形状: {sample['image'].shape}")
        print(f"  扫描路径形状: {sample['scanpath'].shape}")
        print(f"  序列长度: {sample['length']}")

        # 打印第一条扫描路径
        scanpath = sample['scanpath'][:sample['length']]
        print(f"\n扫描路径示例（前3个注视点）:")
        for i, (theta, phi, duration) in enumerate(scanpath[:3]):
            print(f"  注视点 {i+1}: theta={theta:.3f}, phi={phi:.3f}, duration={duration:.3f}s")

        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)

        # 创建数据加载器
        print("\n创建数据加载器...")
        train_loader, val_loader = create_scandmm_dataloaders(
            pkl_path=pkl_path,
            batch_size=4,
            num_workers=0,
            image_size=(256, 512),
            max_seq_len=12
        )

        # 测试batch
        batch = next(iter(train_loader))
        print(f"\n✓ 数据加载器测试成功")
        print(f"  Batch图像形状: {batch['image'].shape}")
        print(f"  Batch扫描路径形状: {batch['scanpath'].shape}")
        print(f"  Batch长度: {batch['length']}")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_scandmm_dataset()
