#!/usr/bin/env python3
"""
通用 nnU-Net 预测 Mask 动态彩色叠加与可视化脚本
支持任意数据集、任意数量的前景类别。自动生成对比色与图例 (Legend)。
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def get_color_palette(num_classes: int):
    """动态获取任意类别数量的高对比度 RGB 颜色盘"""
    cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'tab20')
    palette = {}
    for i in range(1, num_classes + 1):
        color_rgb = [int(c * 255) for c in cmap((i - 1) % cmap.N)[:3]]
        palette[i] = color_rgb
    return palette


def load_dataset_labels(dataset_json_path: str):
    """尝试从 dataset.json 读取类别名称字典"""
    if os.path.exists(dataset_json_path):
        try:
            with open(dataset_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'labels' in data:
                    # 将 {"background": 0, "benti": 1} 转换为 {1: "benti", 2: "luosi"}
                    labels_map = {}
                    for name, idx in data['labels'].items():
                        if idx != 0:
                            labels_map[int(idx)] = str(name)
                    return labels_map
        except Exception:
            pass
    return {}


def visualize(
    pred_dir: str,
    raw_img_dir: str,
    gt_mask_dir: str = None,
    output_vis_dir: str = "./vis_results",
    dataset_json_path: str = None,
    dpi: int = 150
):
    os.makedirs(output_vis_dir, exist_ok=True)

    # 尝试加载类别名称
    class_names = {}
    if dataset_json_path and os.path.exists(dataset_json_path):
        class_names = load_dataset_labels(dataset_json_path)
    else:
        # 在 pred_dir 或上级目录自动寻找 dataset.json
        possible_json = os.path.join(pred_dir, 'dataset.json')
        if os.path.exists(possible_json):
            class_names = load_dataset_labels(possible_json)

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    if not pred_files:
        print(f"⚠️ 在 {pred_dir} 中未找到任何 .png 预测 Mask 文件！")
        return

    print(f"找到 {len(pred_files)} 张预测结果 Mask，准备进行动态可视化...")

    # 扫描获取最大类别 ID 确定调色板大小
    max_label_id = 1
    for f in pred_files[:5]:
        mask_arr = np.array(Image.open(os.path.join(pred_dir, f)))
        unique_vals = np.unique(mask_arr)
        if len(unique_vals) > 0:
            max_label_id = max(max_label_id, int(unique_vals.max()))

    colors = get_color_palette(max(max_label_id, 10))

    count = 0
    for f in pred_files:
        base_name = f[:-4]
        # 支持 filename_0000.png 或 filename.png
        img_path_0000 = os.path.join(raw_img_dir, f'{base_name}_0000.png')
        img_path_normal = os.path.join(raw_img_dir, f)
        img_path = img_path_0000 if os.path.exists(img_path_0000) else (img_path_normal if os.path.exists(img_path_normal) else None)

        gt_candidates = [f]
        if f.endswith('_0000.png'):
            gt_candidates.append(f.replace('_0000.png', '.png'))
        gt_path = None
        if gt_mask_dir:
            for gt_f in gt_candidates:
                candidate = os.path.join(gt_mask_dir, gt_f)
                if os.path.exists(candidate):
                    gt_path = candidate
                    break
        pred_path = os.path.join(pred_dir, f)

        if not img_path:
            continue

        img = Image.open(img_path).convert('RGB')
        pred = np.array(Image.open(pred_path))

        # 叠加预测彩色图层
        overlay = np.array(img).copy()
        present_classes = [c for c in np.unique(pred) if c != 0]

        legend_patches = []
        for class_id in present_classes:
            color = colors.get(class_id, [255, 0, 0])
            mask = (pred == class_id)
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(color)).astype(np.uint8)

            c_name = class_names.get(class_id, f"Class {class_id}")
            norm_color = [c / 255.0 for c in color]
            legend_patches.append(mpatches.Patch(color=norm_color, label=f"{c_name} (ID:{class_id})"))

        has_gt = gt_path is not None
        cols = 3 if has_gt else 2

        fig = plt.figure(figsize=(6 * cols, 6))

        # 1. 原图
        ax1 = fig.add_subplot(1, cols, 1)
        ax1.set_title('Original Image', fontsize=14)
        ax1.imshow(img)
        ax1.axis('off')

        # 2. GT (如果存在)
        if has_gt:
            gt = np.array(Image.open(gt_path))
            gt_overlay = np.array(img).copy()
            gt_classes = [c for c in np.unique(gt) if c != 0]
            for class_id in gt_classes:
                color = colors.get(class_id, [255, 0, 0])
                mask = (gt == class_id)
                gt_overlay[mask] = (0.5 * gt_overlay[mask] + 0.5 * np.array(color)).astype(np.uint8)

            ax2 = fig.add_subplot(1, cols, 2)
            ax2.set_title('Ground Truth', fontsize=14)
            ax2.imshow(gt_overlay)
            ax2.axis('off')

        # 3. Prediction
        ax3 = fig.add_subplot(1, cols, cols)
        ax3.set_title('Prediction Overlay', fontsize=14)
        ax3.imshow(overlay)
        ax3.axis('off')

        if legend_patches:
            ax3.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(output_vis_dir, f'{base_name}_vis.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
        count += 1

    print(f"✅ 通用可视化完成！已处理 {count} 张图像，保存结果至: {output_vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通用 nnU-Net 预测 Mask 动态彩色叠加与可视化")
    parser.add_argument("--pred_dir", type=str, required=True, help="预测 Mask 图像所在目录")
    parser.add_argument("--raw_img_dir", type=str, required=True, help="原始图像所在目录")
    parser.add_argument("--gt_mask_dir", type=str, default=None, help="Ground Truth 标注 Mask 目录 (可选)")
    parser.add_argument("--output_vis_dir", type=str, default="./vis_results", help="可视化结果输出目录")
    parser.add_argument("--dataset_json", type=str, default=None, help="dataset.json 路径 (用于自动解析类别名称，可选)")

    args = parser.parse_args()
    visualize(args.pred_dir, args.raw_img_dir, args.gt_mask_dir, args.output_vis_dir, args.dataset_json)
