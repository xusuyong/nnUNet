#!/usr/bin/env python3
"""
nnU-Net 一键管道运行辅助脚本
包含：
1. dataset_convert : 转换原始数据集
2. preprocess      : 预处理 (plan & preprocess)
3. train           : 启动模型训练 (自动设置 nnUNet_compile=False 避免 GCC 编译错误)
4. predict         : 启动预测推理 (默认加载 checkpoint_best.pth)
"""

import os
import sys
import subprocess
import argparse

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nnunet_config.yaml")

ENV_DEFAULTS = {
    "nnUNet_raw": "/home/xsy/pythoncode/nnUNet_raw",
    "nnUNet_preprocessed": "/home/xsy/pythoncode/nnUNet_preprocessed",
    "nnUNet_results": "/home/xsy/pythoncode/nnUNet_results",
    "nnUNet_compile": "False",
    "nnUNet_n_proc_DA": "4",
}


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get("NNUNET_CONFIG", CONFIG_PATH)
    if not os.path.exists(config_path):
        if config_path != CONFIG_PATH:
            print(f"⚠️ 未找到配置文件: {config_path}，使用默认配置")
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("⚠️ 未安装 PyYAML，无法读取配置文件，使用默认配置")
        return {}


def set_env(config_path=None):
    # 优先使用 yaml 配置文件中的路径，其次使用已存在的环境变量，最后使用默认值
    cfg = load_config(config_path)
    for key, default in ENV_DEFAULTS.items():
        os.environ[key] = str(cfg.get(key, os.environ.get(key, default)))

    for path_var in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        os.makedirs(os.environ[path_var], exist_ok=True)


def run_command(cmd_list):
    cmd_str = " ".join(cmd_list)
    print(f"\n🚀 执行命令: {cmd_str}\n")
    ret = subprocess.run(cmd_list, env=os.environ)
    if ret.returncode != 0:
        print(f"❌ 命令执行失败 (错误码 {ret.returncode}): {cmd_str}")
        sys.exit(ret.returncode)


def main():
    parser = argparse.ArgumentParser(description="nnU-Net 快捷管道工具")
    parser.add_argument("--config_file", type=str, default=None, help="nnU-Net 配置文件 (yaml)，默认读取项目根目录 nnunet_config.yaml")
    subparsers = parser.add_subparsers(dest="action", required=True, help="可执行的操作")

    # 1. convert
    parser_convert = subparsers.add_parser("convert", help="转换数据集为 nnU-Net 规范")
    parser_convert.add_argument("--src_dir", type=str, default="/home/xsy/pythoncode/xsy_datasets/3455-sem/unet_format")
    parser_convert.add_argument("--dataset_id", type=int, default=1)
    parser_convert.add_argument("--out_dir", type=str, default=None, help="输出目录，默认在原数据集同目录下")

    # 2. preprocess
    parser_prep = subparsers.add_parser("preprocess", help="预处理 dataset (plan & preprocess)")
    parser_prep.add_argument("--dataset_id", type=int, default=1)

    # 3. train
    parser_train = subparsers.add_parser("train", help="模型训练")
    parser_train.add_argument("--dataset_id", type=int, default=1)
    parser_train.add_argument("--config", type=str, default="2d")
    parser_train.add_argument("--fold", type=str, default="0")
    parser_train.add_argument("--gpu", type=str, default="0")
    parser_train.add_argument("--tr", type=str, default=None, help="指定 Trainer 类名，例如 nnUNetTrainer_250epochs")
    parser_train.add_argument("--epochs", type=int, default=None, help="快速指定训练轮数 (如 10, 50, 100, 250, 500)")

    # 4. predict
    parser_pred = subparsers.add_parser("predict", help="预测/推理")
    parser_pred.add_argument("--input_dir", type=str, default="E:/xsy/pythoncode/xsy_datasets/nnUNet_data/nnUNet_raw/Dataset001_CustomDataset/imagesTs")
    parser_pred.add_argument(
        "--output_dir", type=str, default="E:/xsy/pythoncode/githubproj/nnUNet/save_predictions"
    )
    parser_pred.add_argument("--dataset_id", type=int, default=1)
    parser_pred.add_argument("--config", type=str, default="2d")
    parser_pred.add_argument("--fold", type=str, default="0")
    parser_pred.add_argument("--chk", type=str, default="checkpoint_best.pth")
    parser_pred.add_argument("--tr", type=str, default=None, help="指定 Trainer 类名，例如 nnUNetTrainer_5epochs")
    parser_pred.add_argument("--vis", action="store_true", help="预测完成后自动生成彩色叠加可视化对比图")
    parser_pred.add_argument("--vis_dir", type=str, default=None, help="可视化输出目录 (配合 --vis)")

    args = parser.parse_args()
    set_env(args.config_file)

    if args.action == "convert":
        sys.path.insert(0, os.path.dirname(__file__))
        from convert_sem_dataset import convert_dataset

        convert_dataset(args.src_dir, args.dataset_id, out_dir=args.out_dir)

    elif args.action == "preprocess":
        run_command(["nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id), "--verify_dataset_integrity"])

    elif args.action == "train":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        cmd = ["nnUNetv2_train", str(args.dataset_id), args.config, str(args.fold)]

        trainer_name = args.tr
        if not trainer_name and args.epochs:
            trainer_name = f"nnUNetTrainer_{args.epochs}epochs"

        if trainer_name:
            cmd.extend(["-tr", trainer_name])

        run_command(cmd)

    elif args.action == "predict":
        cmd = [
            "nnUNetv2_predict",
            "-i",
            args.input_dir,
            "-o",
            args.output_dir,
            "-d",
            str(args.dataset_id),
            "-c",
            args.config,
            "-f",
            str(args.fold),
            "-chk",
            args.chk,
        ]

        if args.tr:
            cmd.extend(["-tr", args.tr])

        run_command(cmd)

        if args.vis:
            sys.path.insert(0, os.path.dirname(__file__))
            from visualize_predictions import visualize

            dataset_folder = os.path.dirname(args.input_dir)
            gt_mask_dir = os.path.join(dataset_folder, "labelsTs")
            if not os.path.exists(gt_mask_dir):
                gt_mask_dir = None
            dataset_json = os.path.join(dataset_folder, "dataset.json")
            vis_dir = args.vis_dir or (args.output_dir + "_vis")
            visualize(
                pred_dir=args.output_dir,
                raw_img_dir=args.input_dir,
                gt_mask_dir=gt_mask_dir,
                output_vis_dir=vis_dir,
                dataset_json_path=dataset_json,
            )


if __name__ == "__main__":
    main()
