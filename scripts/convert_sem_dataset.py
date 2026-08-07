#!/usr/bin/env python3
"""
通用 2D/3D UNet 格式数据集转 nnU-Net v2 格式脚本
动态解析类别定义 (通过 names.txt 或命令行参数)，适用于任何医学/工业分割数据集
"""

import os
import shutil
import json
import argparse
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def parse_names_txt(names_path: str) -> dict:
    """尝试解析 unet_format 中的 names.txt 文件获取类别名称映射"""
    labels = {"background": 0}
    if not os.path.exists(names_path):
        return labels

    try:
        import yaml
        with open(names_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "names" in data:
                names = data["names"]
                labels = {}
                for k, v in names.items():
                    labels[str(v)] = int(k)
                return labels
    except Exception:
        pass

    # 简易文本解析后备逻辑
    with open(names_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line and not line.startswith("names"):
                parts = line.split(":")
                idx = int(parts[0].strip())
                name = parts[1].strip()
                labels[name] = idx
    return labels


def convert_dataset(
    src_dir: str,
    dataset_id: int = 1,
    dataset_name_str: str = "CustomDataset",
    labels_dict: dict = None
):
    nnUNet_raw = os.environ.get("nnUNet_raw", "/home/xsy/pythoncode/nnUNet_raw")
    dataset_name = f"Dataset{dataset_id:03d}_{dataset_name_str}"
    target_dir = join(nnUNet_raw, dataset_name)

    imagesTr = join(target_dir, "imagesTr")
    labelsTr = join(target_dir, "labelsTr")
    imagesTs = join(target_dir, "imagesTs")
    labelsTs = join(target_dir, "labelsTs")

    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)
    maybe_mkdir_p(imagesTs)
    maybe_mkdir_p(labelsTs)

    # 自动解析类别标签
    if labels_dict is None:
        names_path = join(src_dir, "masks", "train", "names.txt")
        if not os.path.exists(names_path):
            names_path = join(src_dir, "masks", "names.txt")
        labels_dict = parse_names_txt(names_path)

    print(f"📋 使用的类别标签定义: {labels_dict}")

    # 1. 转换训练集
    train_img_dir = join(src_dir, "images", "train")
    train_mask_dir = join(src_dir, "masks", "train")

    train_count = 0
    if os.path.exists(train_mask_dir):
        train_files = [f for f in os.listdir(train_mask_dir) if f.endswith(".png")]
        print(f"正在转换训练集，共 {len(train_files)} 张图像...")

        for idx, mask_name in enumerate(sorted(train_files)):
            base_name = os.path.splitext(mask_name)[0]
            case_id = f"case_train_{idx:04d}"

            # 支持 jpg, png, tif
            jpg_path = join(train_img_dir, base_name + ".jpg")
            png_path = join(train_img_dir, base_name + ".png")
            img_src = jpg_path if os.path.exists(jpg_path) else (png_path if os.path.exists(png_path) else None)

            if img_src:
                img = Image.open(img_src).convert("RGB")
                img.save(join(imagesTr, f"{case_id}_0000.png"))
                shutil.copy(join(train_mask_dir, mask_name), join(labelsTr, f"{case_id}.png"))
                train_count += 1

    # 2. 转换验证集/测试集 (可选)
    val_img_dir = join(src_dir, "images", "val")
    val_mask_dir = join(src_dir, "masks", "val")

    if os.path.exists(val_mask_dir):
        val_files = [f for f in os.listdir(val_mask_dir) if f.endswith(".png")]
        print(f"正在转换验证/测试集，共 {len(val_files)} 张图像...")
        for idx, mask_name in enumerate(sorted(val_files)):
            base_name = os.path.splitext(mask_name)[0]
            case_id = f"case_val_{idx:04d}"

            jpg_path = join(val_img_dir, base_name + ".jpg")
            png_path = join(val_img_dir, base_name + ".png")
            img_src = jpg_path if os.path.exists(jpg_path) else (png_path if os.path.exists(png_path) else None)

            if img_src:
                img = Image.open(img_src).convert("RGB")
                img.save(join(imagesTs, f"{case_id}_0000.png"))
                shutil.copy(join(val_mask_dir, mask_name), join(labelsTs, f"{case_id}.png"))

    # 3. 生成 dataset.json 元数据
    generate_dataset_json(
        output_folder=target_dir,
        channel_names={"0": "R", "1": "G", "2": "B"},
        labels=labels_dict,
        num_training_cases=train_count,
        file_ending=".png",
        dataset_name=dataset_name
    )

    print(f"✅ 数据集转换完成！格式化数据集保存在: {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通用 UNet 格式数据集转 nnU-Net v2 格式")
    parser.add_argument("--src_dir", type=str, required=True, help="原始数据集根目录")
    parser.add_argument("--dataset_id", type=int, default=1, help="nnU-Net Dataset ID (默认: 1)")
    parser.add_argument("--dataset_name", type=str, default="SEM", help="nnU-Net 数据集名称")
    parser.add_argument("--labels", type=str, default=None, help='JSON 格式类别映射，如 \'{"background":0,"cat":1,"dog":2}\'')

    args = parser.parse_args()
    labels = json.loads(args.labels) if args.labels else None

    convert_dataset(args.src_dir, args.dataset_id, args.dataset_name, labels)
