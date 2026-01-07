"""
调试ScanDMM数据集结构
检查Sitzmann.pkl的实际内容
"""

import pickle
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
pkl_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'ScanDMM-master', 'Datasets', 'Sitzmann.pkl'
)

print("=" * 60)
print("���查ScanDMM数据集结构")
print("=" * 60)

if not os.path.exists(pkl_path):
    print(f"\n错误: 未找到数据集")
    print(f"路径: {pkl_path}")
    sys.exit(1)

print(f"\n✓ 找到文件: {pkl_path}")
print(f"文件大小: {os.path.getsize(pkl_path) / 1024 / 1024:.2f} MB")

try:
    print("\n加载数据...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"✓ 数据加载成功")

    # 检查数据类型
    print(f"\n数据类型: {type(data)}")

    # 如果是字典，打印键
    if isinstance(data, dict):
        print(f"\n顶层键: {list(data.keys())}")

        # 检查每个键
        for key in data.keys():
            print(f"\n{'='*60}")
            print(f"键: {key}")
            print(f"类型: {type(data[key])}")

            if isinstance(data[key], dict):
                print(f"子键数量: {len(data[key])}")
                sub_keys = list(data[key].keys())
                print(f"前5个子键: {sub_keys[:5]}")

                # 检查第一个样本
                if len(sub_keys) > 0:
                    first_key = sub_keys[0]
                    first_sample = data[key][first_key]
                    print(f"\n第一个样本 ({first_key}):")
                    print(f"  类型: {type(first_sample)}")

                    if isinstance(first_sample, dict):
                        for sample_key, sample_value in first_sample.items():
                            print(f"  {sample_key}: {type(sample_value)}", end='')
                            if hasattr(sample_value, 'shape'):
                                print(f", 形状: {sample_value.shape}")
                            elif hasattr(sample_value, '__len__'):
                                print(f", 长度: {len(sample_value)}")
                            else:
                                print(f", 值: {sample_value}")

                            # 如果是tensor，打印一些值
                            if isinstance(sample_value, torch.Tensor):
                                print(f"    数据类型: {sample_value.dtype}")
                                print(f"    最小值: {sample_value.min():.4f}")
                                print(f"    最大值: {sample_value.max():.4f}")
                                print(f"    均值: {sample_value.mean():.4f}")

                                # 打印前几个元素
                                if sample_value.ndim <= 2:
                                    print(f"    前3个元素: {sample_value.flatten()[:3]}")

            elif isinstance(data[key], list):
                print(f"列表长度: {len(data[key])}")
                if len(data[key]) > 0:
                    print(f"第一个元素类型: {type(data[key][0])}")

    # 检查是否有train/test
    print("\n" + "=" * 60)
    print("数据结构总结:")
    print("=" * 60)

    if 'train' in data:
        print(f"✓ 包含 'train' 数据")
        train_data = data['train']
        if isinstance(train_data, dict):
            print(f"  训练样本数: {len(train_data)}")

    if 'test' in data:
        print(f"✓ 包含 'test' 数据")
        test_data = data['test']
        if isinstance(test_data, dict):
            print(f"  测试样本数: {len(test_data)}")

    if 'info' in data:
        print(f"✓ 包含 'info' 数据")
        print(f"  内容: {data['info']}")

except Exception as e:
    print(f"\n✗ 加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
