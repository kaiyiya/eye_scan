"""
快速测试ScanDMM数据加载
"""

import pickle
import torch
import os

# 数据路径
pkl_path = r"E:\eye_scan\eye_scan\ScanDMM-master\Datasets\Sitzmann.pkl"

print("=" * 60)
print("快速测试ScanDMM数据")
print("=" * 60)

# 检查文件
if not os.path.exists(pkl_path):
    print(f"\n错误: 文件不存在")
    print(f"路径: {pkl_path}")
else:
    print(f"\n✓ 文件存在")
    print(f"大小: {os.path.getsize(pkl_path) / 1024 / 1024:.2f} MB")

    # 加载数据
    print("\n加载数据...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✓ 加载成功")
        print(f"数据类型: {type(data)}")

        # 检查键
        if isinstance(data, dict):
            print(f"\n顶层键: {list(data.keys())}")

            # 检查train
            if 'train' in data:
                print(f"\n✓ 包含 'train'")
                train = data['train']
                print(f"  train类型: {type(train)}")
                print(f"  train长度: {len(train) if hasattr(train, '__len__') else 'N/A'}")

                if isinstance(train, dict):
                    train_keys = list(train.keys())
                    print(f"  前3个键: {train_keys[:3]}")

                    if len(train_keys) > 0:
                        first_key = train_keys[0]
                        first_item = train[first_key]
                        print(f"\n  第一个样本 ({first_key}):")
                        print(f"    类型: {type(first_item)}")

                        if isinstance(first_item, dict):
                            for k, v in first_item.items():
                                if hasattr(v, 'shape'):
                                    print(f"    {k}: Tensor, shape={v.shape}, dtype={v.dtype}")
                                elif hasattr(v, '__len__'):
                                    print(f"    {k}: 长度={len(v)}")
                                else:
                                    print(f"    {k}: {v}")

            # 检查test
            if 'test' in data:
                print(f"\n✓ 包含 'test'")
                test = data['test']
                print(f"  test类型: {type(test)}")
                print(f"  test长度: {len(test) if hasattr(test, '__len__') else 'N/A'}")

            # 检查info
            if 'info' in data:
                print(f"\n✓ 包含 'info'")
                print(f"  内容: {data['info']}")

    except Exception as e:
        print(f"\n✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
