# nnU-Net 辅助工具脚本集 (Scripts)

本目录包含适用于任何 **2D/3D 医学与工业分割数据集** 在 **nnU-Net v2** 框架下运行的通用工具链脚本。

---

## 🛠️ 脚本清单

| 脚本文件名 | 说明 |
| :--- | :--- |
| [`convert_sem_dataset.py`](convert_sem_dataset.py) | **通用数据集转换脚本**：自动解析 `names.txt` 或 JSON 参数，将任意 UNet 格式数据集转为 nnU-Net v2 标准格式 |
| [`visualize_predictions.py`](visualize_predictions.py) | **通用动态彩图叠加可视化脚本**：自动适配任意数量前景类别，动态匹配高对比度 Palette 调色板并自动绘制图例 Legend |
| [`export_to_onnx.py`](export_to_onnx.py) | **ONNX 导出脚本**：将 `checkpoint_best.pth` 转换为 `.onnx` 中间格式，为转 TensorRT 做准备 |
| [`run_nnunet.py`](run_nnunet.py) | **一键式管道辅助脚本**：集成转换、预处理、自定义 Epoch 训练、自动加载最佳权重推理 |

---

## 📖 使用指南

### 1. 通用数据集转换 (`convert_sem_dataset.py`)
```bash
python scripts/convert_sem_dataset.py --src_dir /path/to/any_unet_dataset --dataset_id 1 --dataset_name SEM
```

### 2. 管道快捷工具 (`run_nnunet.py`)
* **预处理**：
  ```bash
  python scripts/run_nnunet.py preprocess --dataset_id 1
  ```
* **训练**（支持 `--epochs` 自定义训练轮数）：
  ```bash
  python scripts/run_nnunet.py train --dataset_id 1 --epochs 250 --gpu 0
  ```
* **推理**：
  ```bash
  python scripts/run_nnunet.py predict --input_dir /path/to/imagesTs --output_dir ./save_predictions --dataset_id 1
  ```

### 3. 转换模型至 ONNX 与 TensorRT (`export_to_onnx.py`)

#### 第一步：导出至 ONNX
```bash
python scripts/export_to_onnx.py \
  --model_folder $nnUNet_results/Dataset001_SEM/nnUNetTrainer__nnUNetPlans__2d \
  --checkpoint checkpoint_best.pth \
  --output_onnx ./model_best.onnx
```

#### 第二步：使用 TensorRT 官方 `trtexec` 转为 TensorRT Engine (`.engine`)
```bash
trtexec --onnx=model_best.onnx \
        --saveEngine=model_best.engine \
        --fp16 \
        --minShapes=input:1x3x512x512 \
        --optShapes=input:1x3x768x2048 \
        --maxShapes=input:4x3x1024x2048
```

### 4. 通用动态可视化 (`visualize_predictions.py`)
```bash
python scripts/visualize_predictions.py \
  --pred_dir ./save_predictions \
  --raw_img_dir /path/to/raw_images \
  --gt_mask_dir /path/to/gt_masks \
  --output_vis_dir ./vis_results
```
