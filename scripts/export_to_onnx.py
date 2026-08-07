#!/usr/bin/env python3
"""
将 nnU-Net 训练好的 PyTorch 权重 (checkpoint_best.pth) 导出为 ONNX 格式
以便进一步转换为 TensorRT Engine (TensorRT .engine/.plan)
"""

import os
import argparse
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


def export_onnx(
    model_folder: str,
    output_onnx_path: str,
    fold: int = 0,
    checkpoint_name: str = "checkpoint_best.pth",
    opset_version: int = 17,
    batch_size: int = 1
):
    checkpoint_path = join(model_folder, f"fold_{fold}", checkpoint_name)
    dataset_json_path = join(model_folder, "dataset.json")
    plans_path = join(model_folder, "plans.json")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")

    dataset_json = load_json(dataset_json_path)
    plans = load_json(plans_path)
    plans_manager = PlansManager(plans)

    print(f"正在加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)

    trainer_name = checkpoint["trainer_name"]
    configuration_name = checkpoint["init_args"]["configuration"]

    configuration_manager = plans_manager.get_configuration(configuration_name)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_trainer_class_by_name(trainer_name)
    num_output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads

    print(f"创建网络结构 (Trainer: {trainer_name}, Configuration: {configuration_name})...")
    network = trainer_class.build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=False,  # 导出推理时禁用深层监督
    )
    network.load_state_dict(checkpoint["network_weights"])
    network.eval()

    # 获取图像通道数与建议的补丁尺寸 (Patch Size)
    patch_size = configuration_manager.patch_size
    print(f"输入通道数: {num_input_channels}, Patch Size: {patch_size}")

    if len(patch_size) == 2:
        dummy_input = torch.randn(batch_size, num_input_channels, patch_size[0], patch_size[1])
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }
    else:
        dummy_input = torch.randn(batch_size, num_input_channels, patch_size[0], patch_size[1], patch_size[2])
        dynamic_axes = {
            "input": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    print(f"正在导出模型到 ONNX 格式: {output_onnx_path} ...")
    torch.onnx.export(
        network,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print(f"✅ ONNX 导出成功！文件位置: {output_onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出 nnU-Net 训练模型至 ONNX 格式")
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="模型训练输出根目录 (例如: /path/to/nnUNet_results/Dataset001_SEM/nnUNetTrainer__nnUNetPlans__2d)",
    )
    parser.add_argument(
        "--output_onnx",
        type=str,
        default="./model.onnx",
        help="导出的 ONNX 文件保存路径 (默认: ./model.onnx)",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold 编号 (默认: 0)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth", help="权重文件名 (默认: checkpoint_best.pth)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本 (默认: 17)")

    args = parser.parse_args()
    export_onnx(args.model_folder, args.output_onnx, args.fold, args.checkpoint, args.opset)
