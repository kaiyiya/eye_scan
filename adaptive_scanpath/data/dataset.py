"""
数据加载器
用于加载360度全景图像和眼动扫描路径数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import json
from typing import Optional, Tuple, List
import random


class ScanpathDataset(Dataset):
    """
    眼动扫描路径数据集

    数据格式：
    - images/: 全景图像文件夹
    - annotations.json: 标注文件，包含扫描路径信息
    """
    def __init__(
        self,
        data_path: str,
        image_height: int = 256,
        image_width: int = 512,
        max_seq_len: int = 12,
        min_seq_len: int = 3,
        transform=None,
        is_train: bool = True
    ):
        """
        Args:
            data_path: 数据集路径
            image_height: 图像高度
            image_width: 图像宽度
            max_seq_len: 最大序列长度
            min_seq_len: 最小序列长度
            transform: 图像变换
            is_train: 是否为训练集
        """
        self.data_path = data_path
        self.image_height = image_height
        self.image_width = image_width
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.transform = transform
        self.is_train = is_train

        # 加载标注
        self.samples = self._load_annotations()

        print(f"加载{'训练' if is_train else '测试'}数据集: {len(self.samples)} 个样本")

    def _load_annotations(self):
        """加载标注文件"""
        annotation_file = os.path.join(self.data_path, 'annotations.json')

        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            # 过滤并处理样本
            samples = []
            for ann in annotations:
                # 获取序列长度
                scanpath = ann.get('scanpath', [])
                seq_len = len(scanpath)

                # 过滤长度
                if self.min_seq_len <= seq_len <= self.max_seq_len:
                    samples.append({
                        'image_path': os.path.join(self.data_path, 'images', ann['image']),
                        'scanpath': scanpath,
                        'seq_len': seq_len
                    })

            return samples
        else:
            print(f"警告: 标注文件不存在 {annotation_file}")
            print("使用模拟数据...")
            return self._create_dummy_data()

    def _create_dummy_data(self):
        """创建模拟数据（用于测试）"""
        num_samples = 100 if self.is_train else 20
        samples = []

        for i in range(num_samples):
            # 随机序列长度
            seq_len = random.randint(self.min_seq_len, self.max_seq_len)

            # 随机扫描路径
            scanpath = []
            for _ in range(seq_len):
                # theta ∈ [-π, π]
                theta = random.uniform(-np.pi, np.pi)
                # phi ∈ [-π/2, π/2]
                phi = random.uniform(-np.pi/2, np.pi/2)
                # duration ∈ [0.1, 3.0]
                duration = random.uniform(0.1, 3.0)

                scanpath.append([theta, phi, duration])

            samples.append({
                'image_path': f'dummy_image_{i}.jpg',
                'scanpath': scanpath,
                'seq_len': seq_len
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本

        Returns:
            image: (3, H, W) 图像tensor
            scanpath: (max_seq_len, 3) 扫描路径（填充）
            length: int 实际序列长度
        """
        sample = self.samples[idx]

        # 1. 加载图像
        if sample['image_path'].startswith('dummy'):
            # 模拟图像
            image = torch.randn(3, self.image_height, self.image_width)
        else:
            try:
                image = Image.open(sample['image_path']).convert('RGB')
                # 调整大小
                image = image.resize((self.image_width, self.image_height), Image.BICUBIC)
                # 转为tensor
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            except Exception as e:
                print(f"加载图像失败: {sample['image_path']}, 错误: {e}")
                image = torch.randn(3, self.image_height, self.image_width)

        # 2. 处理扫描路径
        scanpath = np.array(sample['scanpath'], dtype=np.float32)  # (seq_len, 3)
        seq_len = sample['seq_len']

        # 填充到max_seq_len
        padded_scanpath = np.zeros((self.max_seq_len, 3), dtype=np.float32)
        padded_scanpath[:seq_len] = scanpath

        # 3. 应用变换（可选）
        if self.transform is not None:
            image = self.transform(image)

        return {
            'image': image,  # (3, H, W)
            'scanpath': torch.from_numpy(padded_scanpath),  # (max_seq_len, 3)
            'length': seq_len
        }


def collate_fn(batch):
    """
    自定义batch整理函数

    Args:
        batch: list of dicts

    Returns:
        batch: dict of batched tensors
    """
    images = torch.stack([item['image'] for item in batch])  # (B, 3, H, W)
    scanpaths = torch.stack([item['scanpath'] for item in batch])  # (B, T, 3)
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)  # (B,)

    return {
        'images': images,
        'scanpaths': scanpaths,
        'lengths': lengths
    }


def create_dataloaders(
    train_data_path: str,
    val_data_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_height: int = 256,
    image_width: int = 512,
    max_seq_len: int = 12,
    min_seq_len: int = 3
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        batch_size: batch大小
        num_workers: 数据加载线程数
        image_height: 图像高度
        image_width: 图像宽度
        max_seq_len: 最大序列长度
        min_seq_len: 最小序列长度

    Returns:
        train_loader, val_loader
    """
    # 训练数据集
    train_dataset = ScanpathDataset(
        data_path=train_data_path,
        image_height=image_height,
        image_width=image_width,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        is_train=True
    )

    # 验证数据集
    val_dataset = ScanpathDataset(
        data_path=val_data_path,
        image_height=image_height,
        image_width=image_width,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        is_train=False
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    return train_loader, val_loader


def test_dataset():
    """测试数据集"""
    print("测试数据集...")

    # 创建模拟数据集
    dataset = ScanpathDataset(
        data_path='data/train',  # 不存在，会创建模拟数据
        image_height=256,
        image_width=512,
        max_seq_len=12,
        min_seq_len=3,
        is_train=True
    )

    print(f"数据集大小: {len(dataset)}")

    # 获取一个样本
    sample = dataset[0]
    print(f"图像形状: {sample['image'].shape}")
    print(f"扫描路径形状: {sample['scanpath'].shape}")
    print(f"序列长度: {sample['length']}")

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        train_data_path='data/train',
        val_data_path='data/val',
        batch_size=4,
        num_workers=0
    )

    # 测试迭代
    print("\n测试训练数据加载器...")
    for batch in train_loader:
        print(f"图像batch形状: {batch['images'].shape}")
        print(f"扫描路径batch形状: {batch['scanpaths'].shape}")
        print(f"长度batch: {batch['lengths']}")
        break

    print("\n数据集测试通过！")


if __name__ == "__main__":
    test_dataset()
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
