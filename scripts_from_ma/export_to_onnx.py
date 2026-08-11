import inspect
import warnings
from copy import deepcopy
from time import sleep
from typing import Tuple, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import (
    load_json,
    join,
    isfile,
    maybe_mkdir_p,
    isdir,
    subdirs,
    save_json,
)

from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


class Exporter(object):
    def __init__(
        self,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_device: bool = True,
        device: torch.device = torch.device("cuda"),
        verbose: bool = False,
        verbose_preprocessing: bool = False,
        allow_tqdm: bool = True,
    ):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        (
            self.plans_manager,
            self.configuration_manager,
            self.list_of_parameters,
            self.network,
            self.dataset_json,
            self.trainer_name,
            self.allowed_mirroring_axes,
            self.label_manager,
        ) = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        else:
            print("perform_everything_on_device=True is only supported for cuda devices! Setting this to False")
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: Union[Tuple[Union[int, str]], None],
        checkpoint_name: str = "checkpoint_final.pth",
    ):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = Exporter.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, "dataset.json"))
        plans = load_json(join(model_training_output_dir, "plans.json"))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != "all" else f
            checkpoint = torch.load(
                join(model_training_output_dir, f"fold_{f}", checkpoint_name),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            if i == 0:
                trainer_name = checkpoint["trainer_name"]
                configuration_name = checkpoint["init_args"]["configuration"]
                inference_allowed_mirroring_axes = (
                    checkpoint["inference_allowed_mirroring_axes"]
                    if "inference_allowed_mirroring_axes" in checkpoint.keys()
                    else None
                )

            parameters.append(checkpoint["network_weights"])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_trainer_class_by_name(trainer_name)
        num_output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
        sig = inspect.signature(trainer_class.build_network_architecture)
        if "plans_manager" in sig.parameters:
            network = trainer_class.build_network_architecture(
                plans_manager,
                configuration_manager,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision=False,
            )
        else:
            warnings.warn(
                f"Trainer {trainer_name} uses the old build_network_architecture signature. "
                "Please update to the new signature: "
                "build_network_architecture(plans_manager, configuration_manager, "
                "num_input_channels, num_output_channels, enable_deep_supervision). "
                "The old signature will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision=False,
            )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.network = network

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print("use_folds is None, attempting to auto detect available folds")
        fold_folders = subdirs(model_training_output_dir, prefix="fold_", join=False)
        fold_folders = [i for i in fold_folders if i != "fold_all"]
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split("_")[-1]) for i in fold_folders]
        print(f"found the following folds: {use_folds}")
        return use_folds


if __name__ == "__main__":
    dataset_name = "Dataset512_void"
    nnUNet_plans = "nnUNetTrainer__nnUNetResEncUNetMPlans__2d"
    use_folds = (0,)

    ########################## predict a bunch of files
    from nnunetv2.paths import nnUNet_results
    import onnx, onnxslim

    predictor = Exporter(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, rf"{dataset_name}/{nnUNet_plans}"),
        use_folds=use_folds,
        checkpoint_name="checkpoint_best.pth",
    )

    dummy_input = torch.randn((1, 3, 1024, 1024), dtype=torch.float)

    model = predictor.network
    model.eval()

    onnx_f = f"./{dataset_name}.onnx"

    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=onnx_f,
        input_names=["images"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,
    )

    # simplifier
    onnx_model = onnx.load(onnx_f)
    onnx_slim = onnxslim.slim(onnx_model)
    onnx_slim_f = onnx_f.replace(".onnx", "_slim.onnx")
    onnx.save(onnx_slim, onnx_slim_f)
