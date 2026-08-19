import torch
import torch.nn as nn
import onnx
import onnxslim


class SimpleNetWithIN(nn.Module):
    def __init__(self):
        super().__init__()

        # 保持与 SimpleNet 一致，只增加 InstanceNorm
        self.conv = nn.Conv2d(
            3,
            2,
            kernel_size=1,
            bias=True
        )

        self.inorm = nn.InstanceNorm2d(
            2,
            affine=True,
            track_running_stats=False
        )

    def forward(self, images):
        x = self.conv(images)
        x = self.inorm(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            3,
            2,
            kernel_size=1,
            bias=True
        )

    def forward(self, images):
        return self.conv(images)


def export_and_slim(model, dummy_input, output_name):
    model.eval()

    onnx_path = f"{output_name}.onnx"
    slim_path = f"{output_name}_slim.onnx"

    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["images"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None
    )

    print(f"✅ 已导出: {onnx_path}")

    # 加载 ONNX
    onnx_model = onnx.load(onnx_path)

    # 检查原始模型
    onnx.checker.check_model(onnx_model)

    # 优化
    optimized_model = onnxslim.slim(onnx_model)

    # 检查优化后的模型
    onnx.checker.check_model(optimized_model)

    # 保存
    onnx.save(optimized_model, slim_path)

    print(f"✅ 已优化: {slim_path}")


# 创建模型
model_with_in = SimpleNetWithIN()
model_without_in = SimpleNet()

# Dummy 输入
dummy_input = torch.randn(1, 3, 1024, 1024)

# 导出并优化
export_and_slim(
    model_with_in,
    dummy_input,
    "model_with_in"
)

export_and_slim(
    model_without_in,
    dummy_input,
    "model_without_in"
)

print("🎉 所有任务完成！")