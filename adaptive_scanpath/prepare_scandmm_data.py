"""
处理ScanDMM数据集
将pkl格式转换为清晰的文件夹结构：train/val/test
"""

import pickle
import torch
import numpy as np
import os
import json
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import shutil


def xyz_to_sphere(xyz):
    """
    将3D坐标(x,y,z)转换为球面坐标(theta, phi)
    
    Args:
        xyz: (N, 3) 3D坐标
    
    Returns:
        sphere_coords: (N, 2) 球面坐标 [theta, phi]
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    # 计算theta (水平角度, 经度) [-π, π]
    theta = np.arctan2(x, z)
    
    # 计算phi (垂直角度, 纬度) [-π/2, π/2]
    r_xy = np.sqrt(x**2 + z**2)
    phi = np.arctan2(y, r_xy)
    
    return np.stack([theta, phi], axis=1)


def process_scandmm_dataset(
    pkl_path,
    output_dir='data/scandmm',
    train_ratio=0.85,
    image_size=(256, 512),
    max_seq_len=12,
    min_seq_len=3
):
    """
    处理ScanDMM数据集，转换为清晰的文件夹结构
    
    Args:
        pkl_path: Sitzmann.pkl文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例（剩余为验证集）
        image_size: 图像大小 (height, width)
        max_seq_len: 最大序列长度
        min_seq_len: 最小序列长度
    """
    print("=" * 60)
    print("处理ScanDMM数据集")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"未找到数据集文件: {pkl_path}")
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    
    # 加载pkl数据
    print(f"\n加载数据: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    print(f"数据类型: {type(data)}")
    print(f"顶层键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    
    # 处理训练集（进一步划分为train和val）
    print("\n" + "=" * 60)
    print("处理训练集（train split）")
    print("=" * 60)
    
    if 'train' not in data:
        raise ValueError("数据中缺少'train'键")
    
    train_data = data['train']
    print(f"训练集图像数: {len(train_data)}")
    
    # 收集所有样本（每个图像的所有扫描路径）
    all_train_samples = []
    
    for img_idx, (img_name, img_data) in enumerate(tqdm(train_data.items(), desc="处理训练图像")):
        try:
            if not isinstance(img_data, dict):
                print(f"  警告: {img_name} 的数据不是字典，跳过")
                continue
            
            if 'image' not in img_data or 'scanpaths' not in img_data:
                print(f"  警告: {img_name} 缺少 'image' 或 'scanpaths' 键")
                continue
            
            image_tensor = img_data['image']  # Tensor[3, H, W]
            scanpaths = img_data['scanpaths']  # Tensor[n_scanpath, T, 3] 3D坐标
            
            # 转换为tensor（如果还不是）
            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.tensor(image_tensor)
            if not isinstance(scanpaths, torch.Tensor):
                scanpaths = torch.tensor(scanpaths)
            
            # 调整图像大小
            image_pil = transforms.ToPILImage()(image_tensor)
            image_resized = image_pil.resize((image_size[1], image_size[0]), Image.Resampling.LANCZOS)
            
            # 保存图像
            img_filename = f"{img_name}.jpg"
            img_path_train = os.path.join(train_dir, 'images', img_filename)
            image_resized.save(img_path_train)
            
            # 处理每个扫描路径
            n_scanpaths = scanpaths.shape[0]
            for scanpath_idx in range(n_scanpaths):
                scanpath_xyz = scanpaths[scanpath_idx].numpy()  # (T, 3)
                
                # 转换为球面坐标
                sphere_coords = xyz_to_sphere(scanpath_xyz)  # (T, 2)
                
                # 转换为我们的格式 [theta, phi, duration]
                scanpath_our_format = []
                for i in range(sphere_coords.shape[0]):
                    theta = float(sphere_coords[i, 0])
                    phi = float(sphere_coords[i, 1])
                    duration = 0.5  # 默认0.5秒
                    scanpath_our_format.append([theta, phi, duration])
                
                # 过滤长度
                seq_len = len(scanpath_our_format)
                if min_seq_len <= seq_len <= max_seq_len:
                    all_train_samples.append({
                        'image': img_filename,
                        'scanpath': scanpath_our_format,
                        'seq_len': seq_len
                    })
        
        except Exception as e:
            print(f"  错误处理 {img_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n训练集总样本数: {len(all_train_samples)}")
    
    # 划分训练集和验证集
    np.random.seed(42)
    np.random.shuffle(all_train_samples)
    
    n_train = int(len(all_train_samples) * train_ratio)
    train_samples = all_train_samples[:n_train]
    val_samples = all_train_samples[n_train:]
    
    print(f"训练集样本数: {len(train_samples)}")
    print(f"验证集样本数: {len(val_samples)}")
    
    # 保存训练集标注
    train_annotations = []
    train_images = set()
    for sample in train_samples:
        train_annotations.append({
            'image': sample['image'],
            'scanpath': sample['scanpath']
        })
        train_images.add(sample['image'])
    
    with open(os.path.join(train_dir, 'annotations.json'), 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    # 保存验证集标注（需要复制图像）
    val_annotations = []
    val_images = set()
    for sample in val_samples:
        val_annotations.append({
            'image': sample['image'],
            'scanpath': sample['scanpath']
        })
        val_images.add(sample['image'])
    
    # 复制验证集图像
    for img_file in val_images:
        src = os.path.join(train_dir, 'images', img_file)
        dst = os.path.join(val_dir, 'images', img_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    with open(os.path.join(val_dir, 'annotations.json'), 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    # 处理测试集
    print("\n" + "=" * 60)
    print("处理测试集（test split）")
    print("=" * 60)
    
    if 'test' not in data:
        print("警告: 数据中缺少'test'键，跳过测试集")
        test_annotations = []
    else:
        test_data = data['test']
        print(f"测试集图像数: {len(test_data)}")
        
        test_annotations = []
        
        for img_idx, (img_name, img_data) in enumerate(tqdm(test_data.items(), desc="处理测试图像")):
            try:
                if not isinstance(img_data, dict):
                    continue
                
                if 'image' not in img_data or 'scanpaths' not in img_data:
                    continue
                
                image_tensor = img_data['image']
                scanpaths = img_data['scanpaths']
                
                if not isinstance(image_tensor, torch.Tensor):
                    image_tensor = torch.tensor(image_tensor)
                if not isinstance(scanpaths, torch.Tensor):
                    scanpaths = torch.tensor(scanpaths)
                
                # 调整图像大小
                image_pil = transforms.ToPILImage()(image_tensor)
                image_resized = image_pil.resize((image_size[1], image_size[0]), Image.Resampling.LANCZOS)
                
                # 保存图像
                img_filename = f"{img_name}.jpg"
                img_path_test = os.path.join(test_dir, 'images', img_filename)
                image_resized.save(img_path_test)
                
                # 处理每个扫描路径
                n_scanpaths = scanpaths.shape[0]
                for scanpath_idx in range(n_scanpaths):
                    scanpath_xyz = scanpaths[scanpath_idx].numpy()
                    sphere_coords = xyz_to_sphere(scanpath_xyz)
                    
                    scanpath_our_format = []
                    for i in range(sphere_coords.shape[0]):
                        theta = float(sphere_coords[i, 0])
                        phi = float(sphere_coords[i, 1])
                        duration = 0.5
                        scanpath_our_format.append([theta, phi, duration])
                    
                    seq_len = len(scanpath_our_format)
                    if min_seq_len <= seq_len <= max_seq_len:
                        test_annotations.append({
                            'image': img_filename,
                            'scanpath': scanpath_our_format
                        })
            
            except Exception as e:
                print(f"  错误处理 {img_name}: {e}")
                continue
        
        print(f"测试集样本数: {len(test_annotations)}")
    
    # 保存测试集标注
    with open(os.path.join(test_dir, 'annotations.json'), 'w') as f:
        json.dump(test_annotations, f, indent=2)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据处理完成！")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"\n训练集:")
    print(f"  图像数: {len(train_images)}")
    print(f"  样本数: {len(train_annotations)}")
    print(f"  路径: {train_dir}")
    print(f"\n验证集:")
    print(f"  图像数: {len(val_images)}")
    print(f"  样本数: {len(val_annotations)}")
    print(f"  路径: {val_dir}")
    print(f"\n测试集:")
    print(f"  图像数: {len(set(ann['image'] for ann in test_annotations))}")
    print(f"  样本数: {len(test_annotations)}")
    print(f"  路径: {test_dir}")
    print("\n下一步:")
    print("  1. 修改 config.py 中的数据路径")
    print("  2. 运行训练: python train.py")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='处理ScanDMM数据集')
    parser.add_argument('--input_path', type=str,
                       default='../ScanDMM-master/Datasets/Sitzmann.pkl',
                       help='Sitzmann.pkl文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='data/scandmm',
                       help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.85,
                       help='训练集比例（剩余为验证集）')
    parser.add_argument('--image_height', type=int, default=256,
                       help='图像高度')
    parser.add_argument('--image_width', type=int, default=512,
                       help='图像宽度')
    parser.add_argument('--max_seq_len', type=int, default=12,
                       help='最大序列长度')
    parser.add_argument('--min_seq_len', type=int, default=3,
                       help='最小序列长度')
    
    args = parser.parse_args()
    
    process_scandmm_dataset(
        pkl_path=args.input_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        image_size=(args.image_height, args.image_width),
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len
    )


if __name__ == "__main__":
    main()

