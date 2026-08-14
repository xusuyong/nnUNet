# nnU-Net 辅助工具脚本集 (Scripts)

本目录包含适用于任何 **2D/3D 医学与工业分割数据集** 在 **nnU-Net v2** 框架下运行的通用工具链脚本。

---

## 🛠️ 脚本清单

| 脚本文件名 | 说明 |
| :--- | :--- |
| [`convert_sem_dataset.py`](convert_sem_dataset.py) | **通用数据集转换脚本**：自动解析 `names.txt` 或 JSON 参数，将任意 UNet 格式数据集转为 nnU-Net v2 标准格式 |
| [`visualize_predictions.py`](visualize_predictions.py) | **通用动态彩图叠加可视化脚本**：自动适配任意数量前景类别，动态匹配高对比度 Palette 调色板并自动绘制图例 Legend |
| [`export_to_onnx.py`](export_to_onnx.py) | **ONNX 导出脚本**：将 `checkpoint_best.pth` 转换为 `.onnx` 中间格式，为转 TensorRT 做准备 |
| [`run_nnunet.py`](run_nnunet.py) | **一键式管道辅助脚本**：集成转换、预处理、自定义 Epoch 训练、推理、自动可视化 |

---

## ⚙️ 环境配置 (`nnunet_config.yaml`)

`run_nnunet.py` 启动时自动读取项目根目录下的 **`nnunet_config.yaml`**，统一管理 nnU-Net 的全部路径与运行参数，无需再手动设置环境变量：

```yaml
# nnunet_config.yaml
nnUNet_raw: E:/xsy/pythoncode/xsy_datasets/nnUNet_data/nnUNet_raw
nnUNet_preprocessed: E:/xsy/pythoncode/xsy_datasets/nnUNet_data/nnUNet_preprocessed
nnUNet_results: E:/xsy/pythoncode/xsy_datasets/nnUNet_data/nnUNet_results
nnUNet_compile: False                    # 禁用 torch.compile 避免 Triton 编译器报错
nnUNet_n_proc_DA: 4                      # 数据增强进程数（图像很大时调小，避免内存不足）
```

- 修改 yaml 后重新运行命令即生效，无需改代码。
- 优先级：`yaml 配置 > 已有环境变量 > 内置默认值`。
- 也可用 `--config_file <路径>` 指定其它配置文件（注意：`--config` 是 nnU-Net 网络配置名，如 `2d`）。

---

## 📖 使用指南

### 1. 通用数据集转换 (`convert`)

```bash
python scripts/run_nnunet.py convert --src_dir /path/to/any_unet_dataset --dataset_id 1
```

- 默认输出到 `nnUNet_raw`（中央目录）；可用 `--out_dir` 指定其它输出位置。
- 也可直接调用底层脚本：
  ```bash
  python scripts/convert_sem_dataset.py --src_dir /path/to/any_unet_dataset --dataset_id 1 --dataset_name SEM
  ```

### 2. 预处理 (`preprocess`)

```bash
python scripts/run_nnunet.py preprocess --dataset_id 1
```

### 3. 训练 (`train`)

```bash
# 支持 --epochs 自定义训练轮数（自动映射为 nnUNetTrainer_Nepochs）
python scripts/run_nnunet.py train --dataset_id 1 --epochs 250 --gpu 0

# 指定自定义 Trainer 类名
python scripts/run_nnunet.py train --dataset_id 1 --config 2d --fold 0 --gpu 0 --tr nnUNetTrainer_250epochs
```

### 4. 推理 (`predict`)

```bash
python scripts/run_nnunet.py predict --input_dir /path/to/imagesTs --output_dir ./save_predictions --dataset_id 1
```

**配合可视化：**

```bash
# 预测完成后自动生成 原图 / GT / 预测 彩色叠加对比图（带图例）
python scripts/run_nnunet.py predict --dataset_id 1 --tr nnUNetTrainer_5epochs --vis

# 自定义可视化输出目录
python scripts/run_nnunet.py predict --dataset_id 1 --tr nnUNetTrainer_5epochs --vis --vis_dir ./my_vis
```

> 说明：`--tr` 需与训练时一致（如 `nnUNetTrainer_5epochs`），否则找不到模型目录。`--vis` 会自动从数据集目录读取 `labelsTs` 作为 GT、`dataset.json` 解析类别名；无 GT 时自动退化为两图对比。

### 5. 转换模型至 ONNX 与 TensorRT (`export_to_onnx.py`)

#### 第一步：导出至 ONNX
```bash
python scripts/export_to_onnx.py \
  --model_folder $nnUNet_results/Dataset001_CustomDataset/nnUNetTrainer__nnUNetPlans__2d \
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

### 6. 通用动态可视化 (`visualize_predictions.py`，可独立使用)

```bash
python scripts/visualize_predictions.py \
  --pred_dir ./save_predictions \
  --raw_img_dir /path/to/raw_images \
  --gt_mask_dir /path/to/gt_masks \
  --output_vis_dir ./vis_results
```

---

## 📁 目录约定（Windows 示例）

```
E:/xsy/pythoncode/xsy_datasets/nnUNet_data/
├── nnUNet_raw/               # 转换后的 DatasetXXX 原始数据
│   └── Dataset001_CustomDataset/{imagesTr, labelsTr, imagesTs, labelsTs, dataset.json}
├── nnUNet_preprocessed/      # 预处理结果
│   └── Dataset001_CustomDataset/
└── nnUNet_results/           # 训练模型
    └── Dataset001_CustomDataset/nnUNetTrainer_Nepochs__nnUNetPlans__2d/fold_0/
```
