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


def compute_dice(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    """计算单个类别的 Dice 系数"""
    pred_c = (pred == class_id)
    gt_c = (gt == class_id)
    intersection = np.logical_and(pred_c, gt_c).sum()
    total = pred_c.sum() + gt_c.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / (total + 1e-7))


def find_gt_mask(
    pred_file_name: str,
    pred_idx: int,
    gt_mask_dir: str = None,
    raw_img_dir: str = None
) -> str:
    """智能查找对应的 GT 标注文件路径"""
    base_name = os.path.splitext(pred_file_name)[0]
    
    # 候选文件名列表
    candidates = [
        pred_file_name,
        f"{base_name}.png",
        f"{base_name.replace('_0000', '')}.png",
    ]

    search_dirs = []
    if gt_mask_dir and os.path.exists(gt_mask_dir):
        search_dirs.append(gt_mask_dir)
    
    # 自动探测 raw_img_dir 临近的 labels 目录 (如 labelsTs, labelsTr, masks)
    if raw_img_dir and os.path.exists(raw_img_dir):
        parent_dir = os.path.dirname(raw_img_dir)
        auto_dirs = [
            raw_img_dir.replace("imagesTs", "labelsTs").replace("imagesTr", "labelsTr"),
            os.path.join(parent_dir, "labelsTs"),
            os.path.join(parent_dir, "labelsTr"),
            os.path.join(parent_dir, "labels"),
            os.path.join(parent_dir, "masks"),
        ]
        for ad in auto_dirs:
            if os.path.exists(ad) and ad not in search_dirs:
                search_dirs.append(ad)

    # 1. 尝试直接文件名匹配
    for s_dir in search_dirs:
        for c in candidates:
            p = os.path.join(s_dir, c)
            if os.path.exists(p) and not os.path.isdir(p):
                return p

    # 2. 如果文件名是 case_val_000x / case_train_000x 格式，按序号匹配
    for s_dir in search_dirs:
        all_gt_files = sorted([f for f in os.listdir(s_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))])
        if not all_gt_files:
            continue
        
        # 尝试提取 case 中的编号
        if "case_" in base_name:
            try:
                idx = int(base_name.split("_")[-1].replace("_0000", ""))
                if 0 <= idx < len(all_gt_files):
                    return os.path.join(s_dir, all_gt_files[idx])
            except Exception:
                pass
        
        # 兜底按列表遍历顺序索引
        if 0 <= pred_idx < len(all_gt_files):
            return os.path.join(s_dir, all_gt_files[pred_idx])

    return None


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
        # 在 pred_dir, raw_img_dir 或其父目录自动寻找 dataset.json
        for search_base in [pred_dir, raw_img_dir, os.path.dirname(raw_img_dir)]:
            possible_json = os.path.join(search_base, 'dataset.json')
            if os.path.exists(possible_json):
                class_names = load_dataset_labels(possible_json)
                break

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    if not pred_files:
        print(f"⚠️ 在 {pred_dir} 中未找到任何 .png 预测 Mask 文件！")
        return

    print(f"找到 {len(pred_files)} 张预测结果 Mask，开始执行可视化对比...")

    # 扫描获取最大类别 ID 确定调色板大小
    max_label_id = 1
    for f in pred_files[:5]:
        mask_arr = np.array(Image.open(os.path.join(pred_dir, f)))
        unique_vals = np.unique(mask_arr)
        if len(unique_vals) > 0:
            max_label_id = max(max_label_id, int(unique_vals.max()))

    colors = get_color_palette(max(max_label_id, 10))

    count = 0
    all_dices = {}

    for idx, f in enumerate(pred_files):
        base_name = os.path.splitext(f)[0]
        # 支持 filename_0000.png, filename.png, filename.jpg 等
        img_candidates = [
            os.path.join(raw_img_dir, f'{base_name}_0000.png'),
            os.path.join(raw_img_dir, f'{base_name}.png'),
            os.path.join(raw_img_dir, f'{base_name}_0000.jpg'),
            os.path.join(raw_img_dir, f'{base_name}.jpg'),
        ]
        img_path = None
        for cand in img_candidates:
            if os.path.exists(cand):
                img_path = cand
                break

        if not img_path:
            # 尝试在 raw_img_dir 中按索引查找
            all_raws = sorted([rf for rf in os.listdir(raw_img_dir) if rf.lower().endswith(('.png', '.jpg', '.tif'))])
            if idx < len(all_raws):
                img_path = os.path.join(raw_img_dir, all_raws[idx])

        if not img_path or not os.path.exists(img_path):
            continue

        gt_path = find_gt_mask(f, idx, gt_mask_dir, raw_img_dir)
        pred_path = os.path.join(pred_dir, f)

        img = Image.open(img_path).convert('RGB')
        pred = np.array(Image.open(pred_path))

        # 叠加预测彩色图层
        overlay = np.array(img).copy()
        pred_classes = [int(c) for c in np.unique(pred) if c != 0]

        legend_patches = []
        for class_id in sorted(list(set(pred_classes + (list(class_names.keys()) if class_names else [])))):
            color = colors.get(class_id, [255, 0, 0])
            if class_id in pred_classes:
                mask = (pred == class_id)
                overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array(color)).astype(np.uint8)

            c_name = class_names.get(class_id, f"Class {class_id}")
            norm_color = [c / 255.0 for c in color]
            legend_patches.append(mpatches.Patch(color=norm_color, label=f"{c_name} (ID:{class_id})"))

        has_gt = gt_path is not None and os.path.exists(gt_path)
        cols = 3 if has_gt else 2

        fig = plt.figure(figsize=(6.5 * cols, 6))

        # 1. 原图
        ax1 = fig.add_subplot(1, cols, 1)
        ax1.set_title(f'Original Image\n({os.path.basename(img_path)})', fontsize=13)
        ax1.imshow(img)
        ax1.axis('off')

        # 2. GT (如果找到)
        dice_info = []
        if has_gt:
            gt = np.array(Image.open(gt_path))
            gt_overlay = np.array(img).copy()
            gt_classes = [int(c) for c in np.unique(gt) if c != 0]
            for class_id in gt_classes:
                color = colors.get(class_id, [255, 0, 0])
                mask = (gt == class_id)
                gt_overlay[mask] = (0.45 * gt_overlay[mask] + 0.55 * np.array(color)).astype(np.uint8)

            ax2 = fig.add_subplot(1, cols, 2)
            ax2.set_title(f'Ground Truth\n({os.path.basename(gt_path)})', fontsize=13)
            ax2.imshow(gt_overlay)
            ax2.axis('off')

            # 计算各类别 Dice
            eval_classes = sorted(list(set(gt_classes + pred_classes)))
            for cid in eval_classes:
                c_name = class_names.get(cid, f"Class_{cid}")
                dice_val = compute_dice(pred, gt, cid)
                dice_info.append(f"{c_name}: {dice_val:.3f}")
                all_dices.setdefault(c_name, []).append(dice_val)

        # 3. Prediction
        ax3 = fig.add_subplot(1, cols, cols)
        dice_title = f"\n[Dice: {', '.join(dice_info)}]" if dice_info else ""
        ax3.set_title(f'Prediction Overlay{dice_title}', fontsize=13)
        ax3.imshow(overlay)
        ax3.axis('off')

        if legend_patches:
            ax3.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(output_vis_dir, f'{base_name}_vis.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
        count += 1

    print(f"✅ 可视化完成！共生成 {count} 张对比图，保存至: {output_vis_dir}")
    if all_dices:
        print("\n📊 各类别平均 Dice 评估统计:")
        for c_name, d_list in all_dices.items():
            print(f"   - {c_name}: {np.mean(d_list):.4f} (基于 {len(d_list)} 张测试样本)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通用 nnU-Net 预测 Mask 动态彩色叠加与可视化")
    parser.add_argument("--pred_dir", type=str, required=True, help="预测 Mask 图像所在目录")
    parser.add_argument("--raw_img_dir", type=str, required=True, help="原始图像所在目录")
    parser.add_argument("--gt_mask_dir", type=str, default=None, help="Ground Truth 标注 Mask 目录 (可选，不填将自动探测)")
    parser.add_argument("--output_vis_dir", type=str, default="./vis_results", help="可视化结果输出目录")
    parser.add_argument("--dataset_json", type=str, default=None, help="dataset.json 路径 (用于自动解析类别名称，可选)")

    args = parser.parse_args()
    visualize(args.pred_dir, args.raw_img_dir, args.gt_mask_dir, args.output_vis_dir, args.dataset_json)
