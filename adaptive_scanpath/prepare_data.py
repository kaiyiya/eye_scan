"""
自动准备训练数据
支持多种开源数据集
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class DataPreparer:
    """数据准备器"""

    def __init__(self, data_root='data'):
        self.data_root = data_root
        os.makedirs(data_root, exist_ok=True)

    def create_synthetic_data(self, num_samples=100, img_size=(512, 1024)):
        """
        创建合成数据用于快速测试

        Args:
            num_samples: 样本数量
            img_size: 图像大小 (height, width)
        """
        print("=" * 60)
        print("创建合成训练数据")
        print("=" * 60)

        # 创建目录
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'val')
        os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)

        # 分割比例
        train_split = int(0.8 * num_samples)
        val_split = num_samples - train_split

        print(f"\n生成训练集 ({train_split} 样本)...")
        train_annotations = self._generate_samples(
            train_dir, train_split, img_size, start_idx=0
        )

        print(f"生成验证集 ({val_split} 样本)...")
        val_annotations = self._generate_samples(
            val_dir, val_split, img_size, start_idx=train_split
        )

        # 保存标注
        with open(os.path.join(train_dir, 'annotations.json'), 'w') as f:
            json.dump(train_annotations, f, indent=2)

        with open(os.path.join(val_dir, 'annotations.json'), 'w') as f:
            json.dump(val_annotations, f, indent=2)

        print("\n" + "=" * 60)
        print("✓ 数据准备完成！")
        print("=" * 60)
        print(f"训练集: {len(train_annotations)} 样本")
        print(f"验证集: {len(val_annotations)} 样本")
        print(f"数据目录: {self.data_root}")

        return train_dir, val_dir

    def _generate_samples(self, output_dir, num_samples, img_size, start_idx=0):
        """生成样本数据"""
        annotations = []

        for i in range(num_samples):
            idx = start_idx + i

            # 生成图像（使用渐变和噪声模拟真实场景）
            img = self._generate_synthetic_image(img_size)
            img_name = f'panorama_{idx:04d}.jpg'
            img_path = os.path.join(output_dir, 'images', img_name)
            Image.fromarray(img).save(img_path)

            # 生成扫描路径（模拟真实眼动模式）
            scanpath = self._generate_synthetic_scanpath()

            annotations.append({
                'image': img_name,
                'scanpath': scanpath
            })

            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{num_samples}")

        return annotations

    def _generate_synthetic_image(self, img_size):
        """
        生成合成的全景图像

        模拟真实场景：
        - 地平线区域（中间）较亮
        - 天空和地面有渐变
        - 添加一些"兴趣点"（明亮的区域）
        """
        H, W = img_size
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # 1. 创建垂直渐变（天空到地面）
        for y in range(H):
            # 天空（蓝色调）
            if y < H // 3:
                blue_intensity = int(200 - (y / (H // 3)) * 100)
                img[y, :, 2] = blue_intensity
                img[y, :, 1] = int(blue_intensity * 0.5)
                img[y, :, 0] = int(blue_intensity * 0.3)
            # 地平线区域（绿色/棕色）
            elif y < 2 * H // 3:
                green_intensity = 150 + int(np.sin(y / H * np.pi) * 50)
                img[y, :, 1] = green_intensity
                img[y, :, 0] = int(green_intensity * 0.6)
                img[y, :, 2] = int(green_intensity * 0.3)
            # 地面（灰色调）
            else:
                gray = int(100 + (y - 2 * H // 3) / (H // 3) * 50)
                img[y, :, :] = gray

        # 2. 添加一些"兴趣点"（模拟场景中的物体）
        num_interest_points = np.random.randint(3, 8)
        for _ in range(num_interest_points):
            x = np.random.randint(0, W)
            y = np.random.randint(H // 4, 3 * H // 4)
            radius = np.random.randint(20, 60)

            # 创建圆形区域
            Y, X = np.ogrid[:H, :W]
            dist = np.sqrt((X - x)**2 + (Y - y)**2)
            mask = dist <= radius

            # 随机颜色
            color = np.random.randint(150, 255, 3)
            for c in range(3):
                img[:, :, c] = np.where(mask, color[c], img[:, :, c])

        # 3. 添加噪声
        noise = np.random.randint(-30, 30, (H, W, 3))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def _generate_synthetic_scanpath(self):
        """
        生成合成的扫描路径

        模拟真实眼动模式：
        - 初始注视点靠近中心
        - 后续注视点有聚集倾向（中心偏差）
        - 扫描长度5-12步
        """
        seq_len = np.random.randint(5, 13)
        scanpath = []

        # 初始注视点（靠近中心）
        theta_0 = np.random.normal(0, 0.5)  # [-π, π]
        phi_0 = np.random.normal(0, 0.3)    # [-π/2, π/2]
        duration_0 = np.random.uniform(0.3, 0.8)

        # 限制范围
        theta_0 = np.clip(theta_0, -np.pi, np.pi)
        phi_0 = np.clip(phi_0, -np.pi/2, np.pi/2)

        scanpath.append([float(theta_0), float(phi_0), float(duration_0)])

        # 后续注视点（有聚集倾向）
        for i in range(1, seq_len):
            # 从上一个注视点附近开始
            last_theta, last_phi, _ = scanpath[-1]

            # 随机偏移（有向中心回归的倾向）
            theta = last_theta + np.random.normal(-0.1 * last_theta, 0.3)
            phi = last_phi + np.random.normal(-0.1 * last_phi, 0.2)
            duration = np.random.uniform(0.2, 0.6)

            # 限制范围
            theta = np.clip(theta, -np.pi, np.pi)
            phi = np.clip(phi, -np.pi/2, np.pi/2)

            scanpath.append([float(theta), float(phi), float(duration)])

        return scanpath

    def download_salient360_info(self):
        """显示如何下载Salient360!数据集"""
        print("\n" + "=" * 60)
        print("Salient360! 数据集下载指南")
        print("=" * 60)
        print("\n1. 访问官网: https://salient360.di.fc.ul.pt/")
        print("2. 下载数据集（需要注册，免费）")
        print("3. 解压到:", self.data_root)
        print("4. 运行转换脚本")
        print("\n或者现在就使用合成数据开始训练！")


def main():
    parser = argparse.ArgumentParser(description='准备训练数据')
    parser.add_argument('--mode', type=str, default='synthetic',
                       choices=['synthetic', 'info'],
                       help='数据准备模式: synthetic(合成数据) 或 info(下载数据集信息)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='合成数据样本数量')
    parser.add_argument('--data_root', type=str, default='data',
                       help='数据根目录')

    args = parser.parse_args()

    preparer = DataPreparer(data_root=args.data_root)

    if args.mode == 'synthetic':
        # 创建合成数据
        train_dir, val_dir = preparer.create_synthetic_data(
            num_samples=args.num_samples
        )

        print("\n✓ 数据已准备完成！")
        print(f"  训练数据: {train_dir}")
        print(f"  验证数据: {val_dir}")
        print("\n现在可以开始训练:")
        print("  python quickstart.py")
        print("  python train.py")

    elif args.mode == 'info':
        # 显示下载信息
        preparer.download_salient360_info()


if __name__ == "__main__":
    main()
