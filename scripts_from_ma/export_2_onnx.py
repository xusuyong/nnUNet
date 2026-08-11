import torch
import onnxsim, onnx
from nnunetv2.experiment_planning.experiment_planners.custom_planner.model import make_network


def convert_2_onnx_custom():

    pt_ = r"E:\TrainingFramework\nnUNet-master\DATASET\nnUNet_trained_models\Dataset502_void_huafeng_20260805_tianchong128\CustomTrainer__custom_nnUNetResEncUNetPlans__2d\fold_0/checkpoint_best.pth"

    onnx_f = pt_.replace(".pth", ".onnx")

    input_channel = 3
    class_num = 3
    input_feature_size = (1024, 1024)

    # model_type = "ResEnc" or "Plain"
    model = make_network(input_channel, class_num, input_feature_size, deep_supervision=False, model_type="ResEnc")
    model.eval()

    msd = torch.load(pt_, map_location="cpu", weights_only=False)
    model.load_state_dict(msd["network_weights"])

    # input
    dummy_input = torch.randn((1, input_channel, input_feature_size[0], input_feature_size[1]), dtype=torch.float)

    # input_names = 'images'
    dynamic = {
        "images": {0: "batch"},  # , 2:'height', 3:'width'},
        "outputs": {0: "batch"},  # , 2:'height', 3:'width'}
    }

    onnx_program = torch.onnx.export(
        model,
        dummy_input,
        onnx_f,
        input_names=["images"],
        output_names=["outputs"],
        opset_version=15,  # only support opset > 18 if dynamo = True
        dynamic_axes=None,
        # dynamo=True
    )

    # onnx_program.optimize()
    # onnx_program.save(onnx_f)

    # simplifier
    onnx_model = onnx.load(onnx_f)
    onnx_sim, succ = onnxsim.simplify(onnx_model)
    onnx_sim_f = pt_.replace(".pth", "_sim.onnx")
    onnx.save(onnx_sim, onnx_sim_f)


if __name__ == "__main__":
    convert_2_onnx_custom()
