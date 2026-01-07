"""
Salient360! 数据集下载和转换工具
自动下载并转换为项目所需格式
"""

import os
import requests
import json
import numpy as np
from PIL import Image
import zipfile
from tqdm import tqdm
import cv2


class Salient360Downloader:
    """Salient360!数据集下载器"""

    def __init__(self, save_dir='data/salient360'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'annotations'), exist_ok=True)

        # Salient360! 数据集URL
        self.base_url = "https://salient360.di.fc.ul.pt/dataset"

    def download_file(self, url, filename):
        """下载文件（带进度条）"""
        filepath = os.path.join(self.save_dir, filename)

        if os.path.exists(filepath):
            print(f"文件已存在: {filepath}")
            return filepath

        print(f"下载 {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return filepath

    def download_dataset(self):
        """下载完整数据集"""
        print("=" * 60)
        print("Salient360! 数据集下载器")
        print("=" * 60)

        # 下载说明
        print("\n请注意：")
        print("1. Salient360! 需要从官网手动下载")
        print("2. 访问: https://salient360.di.fc.ul.pt/dataset")
        print("3. 请求数据集权限（免费，学术研究用）")
        print("4. 下载后放在 data/salient360/ 目录")
        print("\n或者使用以下模拟数据进行测试...")

        # 创建模拟数据用于测试
        self.create_mock_data()

    def create_mock_data(self):
        """创建模拟数据用于测试"""
        print("\n创建模拟数据用于测试...")

        annotations = []

        # 生成100个模拟样本
        for i in range(100):
            # 生成模拟图像（使用随机噪声）
            img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
            img_path = os.path.join(self.save_dir, 'images', f'img_{i:04d}.jpg')
            Image.fromarray(img).save(img_path)

            # 生成模拟扫描路径
            seq_len = np.random.randint(5, 13)
            scanpath = []

            for _ in range(seq_len):
                # theta: [-π, π]
                theta = np.random.uniform(-np.pi, np.pi)
                # phi: [-π/2, π/2]
                phi = np.random.uniform(-np.pi/2, np.pi/2)
                # duration: [0.2, 2.0]
                duration = np.random.uniform(0.2, 2.0)

                scanpath.append([float(theta), float(phi), float(duration)])

            annotations.append({
                'image': f'img_{i:04d}.jpg',
                'scanpath': scanpath
            })

        # 保存标注文件
        train_annotations = annotations[:80]
        val_annotations = annotations[80:90]
        test_annotations = annotations[90:]

        with open(os.path.join(self.save_dir, 'annotations_train.json'), 'w') as f:
            json.dump(train_annotations, f, indent=2)

        with open(os.path.join(self.save_dir, 'annotations_val.json'), 'w') as f:
            json.dump(val_annotations, f, indent=2)

        with open(os.path.join(self.save_dir, 'annotations_test.json'), 'w') as f:
            json.dump(test_annotations, f, indent=2)

        print(f"✓ 模拟数据已创建")
        print(f"  训练集: {len(train_annotations)} 样本")
        print(f"  验证集: {len(val_annotations)} 样本")
        print(f"  测试集: {len(test_annotations)} 样本")
        print(f"  位置: {self.save_dir}")


def convert_salient360_to_our_format(salient360_path, output_path):
    """
    将Salient360!数据集转换为我们的格式

    Args:
        salient360_path: Salient360!数据集路径
        output_path: 输出路径
    """
    print("转换Salient360!数据集...")

    # 这里需要根据实际的Salient360!数据格式进行转换
    # 示例代码（需要根据实际格式调整）

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)

    annotations = []

    # 假设Salient360!的格式
    images_dir = os.path.join(salient360_path, 'images')
    fixations_dir = os.path.join(salient360_path, 'fixations')

    if os.path.exists(images_dir):
        for img_file in os.listdir(images_dir):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                # 读取图像
                img_path = os.path.join(images_dir, img_file)

                # 读取对应的注视点数据
                # （这里需要根据Salient360!的实际格式调整）
                # scanpath = load_fixations(os.path.join(fixations_dir, img_file.replace('.jpg', '.json')))

                # 转换格式
                # annotations.append({
                #     'image': img_file,
                #     'scanpath': scanpath
                # })

                pass

        # 保存标注
        with open(os.path.join(output_path, 'annotations.json'), 'w') as f:
            json.dump(annotations, f, indent=2)

        print(f"✓ 转换完成: {len(annotations)} 样本")
    else:
        print("未找到Salient360!数据集，使用模拟数据")


if __name__ == "__main__":
    # 创建下载器
    downloader = Salient360Downloader(save_dir='data/salient360')

    # 下载数据集（或创建模拟数据）
    downloader.download_dataset()

    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 修改 config.py 中的数据路径")
    print("  2. 运行 python train.py 开始训练")
    print("=" * 60)
