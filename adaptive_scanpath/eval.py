"""
评估和推理脚本
用于评估训练好的模型和生成预测结果
"""

import torch
import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2

from adaptive_scanpath.models import AdaptiveScanPath

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.losses import ScanPathMetrics, LengthAccuracy
from data.dataset import create_dataloaders


class Evaluator:
    """
    评估器类
    用于模型评估和推理
    """
    def __init__(self, checkpoint_path, config=None):
        """
        初始化评估器

        Args:
            checkpoint_path: 模型检查点路径
            config: 配置对象（可选，如果为None则从检查点加载）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载配置
        if config is None:
            config_dict = checkpoint['config']
            config = Config()
            config.update(**config_dict)

        self.config = config

        # 创建模型
        self.model = AdaptiveScanPath(
            image_channels=config.image_channels,
            image_height=config.image_height,
            image_width=config.image_width,
            feature_dim=config.feature_dim,
            cnn_channels=config.cnn_channels,
            policy_hidden_dim=config.policy_hidden_dim,
            policy_dropout=config.policy_dropout,
            stopping_hidden_dim=config.stopping_hidden_dim,
            max_seq_len=config.max_seq_len,
            use_rnn=config.use_rnn,
            rnn_hidden_dim=config.rnn_hidden_dim,
            rnn_num_layers=config.rnn_num_layers,
            use_feature_update=True
        ).to(self.device)

        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"模型已加载: {checkpoint_path}")
        print(f"设备: {self.device}")

    @torch.no_grad()
    def evaluate(self, data_loader):
        """
        评估模型

        Args:
            data_loader: 数据加载器

        Returns:
            metrics_dict: 评估指标字典
        """
        metrics = ScanPathMetrics()
        length_acc = LengthAccuracy(tolerance=1)

        for batch in tqdm(data_loader, desc="评估中..."):
            images = batch['images'].to(self.device)
            gt_paths = batch['scanpaths'].to(self.device)
            gt_lengths = batch['lengths']

            # 生成预测
            pred_paths, pred_lengths = self.model.generate(images)

            # 更新指标
            metrics.update(pred_paths, gt_paths, pred_lengths, gt_lengths)
            length_acc.update(pred_lengths, gt_lengths)

        # 计算指标
        metrics_dict = metrics.compute()
        metrics_dict['length_accuracy'] = length_acc.compute()

        return metrics_dict

    @torch.no_grad()
    def predict(self, images, num_samples=1):
        """
        预测单个或多个图像的扫描路径

        Args:
            images: (B, 3, H, W) 或 (3, H, W)
            num_samples: 每个图像的采样次数

        Returns:
            predictions: list of predictions
        """
        self.model.eval()

        # 确保是batch格式
        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device)

        if num_samples == 1:
            pred_paths, pred_lengths = self.model.generate(images)
            return pred_paths.cpu(), pred_lengths
        else:
            # 多样化采样
            all_paths = self.model.generate_with_diversity(images, num_samples=num_samples)
            return [paths.cpu() for paths in all_paths]

    def visualize_prediction(
        self,
        image,
        pred_path,
        gt_path=None,
        save_path=None,
        title="预测结果"
    ):
        """
        可视化预测结果

        Args:
            image: (H, W, 3) numpy数组
            pred_path: (T, 3) 预测路径
            gt_path: (T, 3) 真实路径（可选）
            save_path: 保存路径（可选）
            title: 图标题
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # 显示图像
        ax.imshow(image)

        # 绘制预测路径
        if pred_path is not None and len(pred_path) > 0:
            # 转换球面坐标到像素坐标
            H, W = image.shape[:2]
            pred_pixels = []
            for i, (theta, phi, duration) in enumerate(pred_path):
                # 球面坐标 -> 像素坐标
                x = int((theta + np.pi) / (2 * np.pi) * W)
                y = int((phi + np.pi/2) / np.pi * H)
                pred_pixels.append((x, y))

            pred_pixels = np.array(pred_pixels)

            # 绘制路径
            ax.plot(pred_pixels[:, 0], pred_pixels[:, 1], 'r-o',
                   linewidth=2, markersize=8, label='预测路径', alpha=0.7)

            # 标注数字
            for i, (x, y) in enumerate(pred_pixels):
                ax.text(x, y, str(i+1), color='white', fontsize=10,
                       fontweight='bold', ha='center', va='center')

        # 绘制真实路径
        if gt_path is not None and len(gt_path) > 0:
            gt_pixels = []
            for i, (theta, phi, duration) in enumerate(gt_path):
                x = int((theta + np.pi) / (2 * np.pi) * W)
                y = int((phi + np.pi/2) / np.pi * H)
                gt_pixels.append((x, y))

            gt_pixels = np.array(gt_pixels)

            ax.plot(gt_pixels[:, 0], gt_pixels[:, 1], 'g--s',
                   linewidth=2, markersize=8, label='真实路径', alpha=0.7)

        ax.legend(loc='upper right')
        ax.set_title(title)
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化已保存: {save_path}")

        plt.show()
        plt.close()

    def visualize_equirectangular(self, image, scanpath, save_path=None):
        """
        在等距圆柱投影图上可视化扫描路径

        Args:
            image: (H, W, 3) numpy数组
            scanpath: (T, 3) 扫描路径
            save_path: 保存路径（可选）
        """
        H, W = image.shape[:2]

        # 转换图像到RGB
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # 绘制路径
        overlay = image.copy()

        for i, (theta, phi, duration) in enumerate(scanpath):
            # 球面坐标 -> 像素坐标
            x = int((theta + np.pi) / (2 * np.pi) * W)
            y = int((phi + np.pi/2) / np.pi * H)

            # 绘制圆圈
            cv2.circle(overlay, (x, y), 10, (0, 0, 255), -1)
            cv2.circle(overlay, (x, y), 12, (255, 255, 255), 2)

            # 标注数字
            cv2.putText(overlay, str(i+1), (x-5, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 混合
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        if save_path:
            cv2.imwrite(save_path, result)
            print(f"可视化已保存: {save_path}")

        return result

    def export_predictions(self, data_loader, output_path):
        """
        导出预测结果到JSON文件

        Args:
            data_loader: 数据加载器
            output_path: 输出JSON文件路径
        """
        predictions = []

        for batch_idx, batch in enumerate(tqdm(data_loader, desc="生成预测...")):
            images = batch['images'].to(self.device)
            gt_paths = batch['scanpaths'].cpu().numpy()
            gt_lengths = batch['lengths'].cpu().numpy()

            # 生成预测
            pred_paths, pred_lengths = self.model.generate(images)
            pred_paths = pred_paths.cpu().numpy()

            # 保存每个样本
            for i in range(images.shape[0]):
                sample_pred = {
                    'sample_id': f"{batch_idx}_{i}",
                    'pred_path': pred_paths[i, :pred_lengths[i]].tolist(),
                    'pred_length': int(pred_lengths[i]),
                    'gt_path': gt_paths[i, :gt_lengths[i]].tolist(),
                    'gt_length': int(gt_lengths[i])
                }
                predictions.append(sample_pred)

        # 保存到JSON
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"预测结果已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估AdaptiveScanPath模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data_path', type=str, default='data/val',
                       help='评估数据路径')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    parser.add_argument('--export', action='store_true',
                       help='是否导出预测结果')

    args = parser.parse_args()

    # 创建评估器
    evaluator = Evaluator(args.checkpoint)

    # 创建数据加载器
    _, val_loader = create_dataloaders(
        train_data_path='data/train',
        val_data_path=args.data_path,
        batch_size=8,
        num_workers=4
    )

    # 评估
    print("\n开始评估...")
    metrics = evaluator.evaluate(val_loader)

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"长度预测准确率: {metrics['length_accuracy']*100:.2f}%")
    print("=" * 60)

    # 保存评估结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"评估结果已保存: {results_path}")

    # 可视化
    if args.visualize:
        print("\n生成可视化...")
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        for i, batch in enumerate(val_loader):
            if i >= 10:  # 只可视化前10个batch
                break

            images = batch['images']
            gt_paths = batch['scanpaths'].numpy()
            gt_lengths = batch['lengths'].numpy()

            # 预测
            images_gpu = images.to(evaluator.device)
            pred_paths, pred_lengths = evaluator.model.generate(images_gpu)
            pred_paths = pred_paths.cpu().numpy()

            # 可视化每个样本
            for j in range(min(4, images.shape[0])):
                image = images[j].permute(1, 2, 0).numpy()
                pred_path = pred_paths[j, :pred_lengths[j]]
                gt_path = gt_paths[j, :gt_lengths[j]]

                save_path = os.path.join(vis_dir, f'sample_{i}_{j}.png')
                evaluator.visualize_prediction(
                    image, pred_path, gt_path,
                    save_path=save_path,
                    title=f"样本 {i}_{j}"
                )

    # 导出预测
    if args.export:
        print("\n导出预测结果...")
        export_path = os.path.join(args.output_dir, 'predictions.json')
        evaluator.export_predictions(val_loader, export_path)


if __name__ == "__main__":
    main()
