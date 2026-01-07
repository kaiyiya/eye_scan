# -*- coding: utf-8 -*-
"""
检查原始Sitzmann.pkl数据格式
"""

import pickle
import sys
import io

# 设置UTF-8编码输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def inspect_raw_data():
    """检查原始数据"""
    print("=" * 60)
    print("检查原始Sitzmann.pkl数据")
    print("=" * 60)

    pkl_path = '../ScanDMM-master/Datasets/Sitzmann.pkl'

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    print(f"\n数据类型: {type(data)}")
    print(f"顶层键: {list(data.keys())}")

    # 检查训练集
    if 'train' in data:
        train_data = data['train']
        print(f"\n训练集图像数: {len(train_data)}")

        # 获取第一个图像
        first_img_name = list(train_data.keys())[0]
        first_img_data = train_data[first_img_name]

        print(f"\n第一个图像名称: {first_img_name}")
        print(f"图像数据键: {list(first_img_data.keys()) if isinstance(first_img_data, dict) else 'N/A'}")

        if isinstance(first_img_data, dict):
            # 检查图像
            if 'image' in first_img_data:
                image_tensor = first_img_data['image']
                print(f"\n图像形状: {image_tensor.shape if hasattr(image_tensor, 'shape') else type(image_tensor)}")

            # 检查扫描路径
            if 'scanpaths' in first_img_data:
                scanpaths = first_img_data['scanpaths']
                print(f"\n扫描路径数量: {scanpaths.shape[0] if hasattr(scanpaths, 'shape') else len(scanpaths)}")

                # 检查第一个扫描路径
                first_scanpath = scanpaths[0]
                print(f"第一个扫描路径形状: {first_scanpath.shape if hasattr(first_scanpath, 'shape') else len(first_scanpath)}")
                print(f"前3个点: {first_scanpath[:3] if hasattr(first_scanpath, '__getitem__') else 'N/A'}")

        # 统计所有扫描路径的长度分布
        print(f"\n扫描路径长度分布统计:")
        lengths = []
        for img_name, img_data in list(train_data.items())[:10]:  # 检查前10个
            if isinstance(img_data, dict) and 'scanpaths' in img_data:
                scanpaths = img_data['scanpaths']
                if hasattr(scanpaths, 'shape'):
                    num_paths, seq_len, _ = scanpaths.shape
                    lengths.append(seq_len)

        if lengths:
            print(f"  序列长度范围: {min(lengths)} - {max(lengths)}")
            print(f"  所有检查的序列长度: {set(lengths)}")

    # 检查info
    if 'info' in data:
        print(f"\n数据集信息: {data['info']}")

if __name__ == "__main__":
    inspect_raw_data()
