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


def set_env():
    # 自动设置并创建默认环境变量目录
    os.environ["nnUNet_raw"] = os.environ.get("nnUNet_raw", "/home/xsy/pythoncode/nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.environ.get(
        "nnUNet_preprocessed", "/home/xsy/pythoncode/nnUNet_preprocessed"
    )
    os.environ["nnUNet_results"] = os.environ.get("nnUNet_results", "/home/xsy/pythoncode/nnUNet_results")
    # 禁用 torch.compile 避免 Triton C 编译器报错
    os.environ["nnUNet_compile"] = "False"

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
    subparsers = parser.add_subparsers(dest="action", required=True, help="可执行的操作")

    # 1. convert
    parser_convert = subparsers.add_parser("convert", help="转换数据集为 nnU-Net 规范")
    parser_convert.add_argument("--src_dir", type=str, default="/home/xsy/pythoncode/xsy_datasets/3455-sem/unet_format")
    parser_convert.add_argument("--dataset_id", type=int, default=1)

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
    parser_pred.add_argument("--input_dir", type=str, default="/home/xsy/pythoncode/nnUNet_raw/Dataset001_SEM/imagesTs")
    parser_pred.add_argument(
        "--output_dir", type=str, default="/home/xsy/pythoncode/githubproj/nnUNet/save_predictions"
    )
    parser_pred.add_argument("--dataset_id", type=int, default=1)
    parser_pred.add_argument("--config", type=str, default="2d")
    parser_pred.add_argument("--fold", type=str, default="0")
    parser_pred.add_argument("--chk", type=str, default="checkpoint_best.pth")

    args = parser.parse_args()
    set_env()

    if args.action == "convert":
        sys.path.insert(0, os.path.dirname(__file__))
        from convert_sem_dataset import convert_dataset

        convert_dataset(args.src_dir, args.dataset_id)

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
        run_command(cmd)


if __name__ == "__main__":
    main()
