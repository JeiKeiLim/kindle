"""Kindle Model parser and model.

This module parses model configuration yaml file
and generates PyTorch model accordingly.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import inspect
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import yaml
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract, ModuleGenerator
from kindle.generator.flatten import FlattenGenerator
from kindle.generator.yolo_head import YOLOHeadGenerator
from kindle.modules import Conv, DWConv, Focus, YOLOHead
from kindle.utils.model_utils import ModelInfoLogger, ModelProfiler
from kindle.utils.torch_utils import fuse_conv_and_batch_norm


class Model(nn.Module):
    """PyTorch model class."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, type]],
        verbose: bool = False,
    ) -> None:
        """Parse model from the model config file.

        Args:
            cfg: yaml file path or dictionary type of the model.
            verbose: print the model parsing information.
        """
        super().__init__()
        self.model_parser = ModelParser(cfg=cfg, verbose=verbose)
        self.model = self.model_parser.model
        self.output_save = self.model_parser.output_save

    def forward(
        self,
        x: Union[torch.Tensor, np.ndarray],
        augment_func: Optional[Union[List[Callable], Callable]] = None,
        n_augment: int = 3,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward the model.

        Args:
            x: model input. When augment_func is given,
                {X} can be either torch.Tensor or np.ndarray.
                However, output of augmentation function must emit torch.Tensor.

            augment_func: augmentation function.
                        If list of functions is given, TTA will be applied by original input {X} and
                        {X} with each augmentation functions.
                        Output shape will be (len({AUGMENT_FUNC})+1, batch_size, output_size).

                        If single function is given, TTA will be applied by original input {X} and
                        {X} {AUGMENT_FUNC} applied by {N_AUGMENT} times.
                        Output shape will be ({N_AUGMENT}+1, batch_size, output_size)

            n_augment: number of augmentations to apply on TTA augmentation.
                       Only applies when single augmentation function is given.

        Returns:
            Model output.
        """
        if augment_func is not None:
            if isinstance(augment_func, list):
                funcs = augment_func
            else:
                funcs = [augment_func for _ in range(n_augment)]

            y_list = [self.forward_once(aug_func(x)) for aug_func in funcs]
            if isinstance(x, torch.Tensor):
                y_list.append(self.forward_once(x))

            y = torch.stack(y_list)
        else:
            assert isinstance(x, torch.Tensor), "Input must be torch tensor for non-TTA"
            y = self.forward_once(x)

        return y

    def export(
        self, export_activations=("SiLU", "Hardswish", "Mish"), verbose: bool = False
    ) -> nn.Module:
        """Make model to export friendly version.

        This method will call export() method in every child modules,
        convert activations into export-friendly version,
        and fuse conv-batch_norm itself.

        Args:
            export_activations: Activation name list to convert themselves
                into export-friendly version.
            verbose: print logs

        Return:
            self
        """
        self.fuse(verbose=verbose)
        for module in self.model.modules():
            if hasattr(module, "export") and inspect.ismethod(module.export):
                # Check if the module has export method
                if verbose:
                    print(f"Calling {module.name} export()")
                module.export()  # type: ignore

            if hasattr(module, "activation"):
                for activation in export_activations:
                    if hasattr(nn, activation) and isinstance(
                        module.activation, getattr(nn, activation)
                    ):
                        if verbose:
                            print(
                                f"Converting {module.activation} activation"
                                f" in {module.__class__.__name__} to export friendly version."
                            )
                        module.activation = getattr(
                            __import__("kindle.modules.activation", fromlist=[""]),
                            activation,
                        )()
                        break

        return self

    def fuse(self, verbose: bool = False) -> nn.Module:
        """Fuse Conv - BatchNorm2d layers."""

        if verbose:
            print("Fusing layers ", end="")

        for module in self.model.modules():
            dot_str = "."

            if isinstance(module, (Conv, DWConv, Focus)) and hasattr(
                module, "batch_norm"
            ):
                if isinstance(
                    module, nn.Sequential
                ):  # Tensor decomposed conv will be converted into nn.Sequential of 3 convs
                    module.conv[-1] = fuse_conv_and_batch_norm(  # type: ignore
                        module.conv[-1], module.batch_norm  # type: ignore
                    )
                else:
                    module.conv = fuse_conv_and_batch_norm(
                        module.conv, module.batch_norm
                    )
                delattr(module, "batch_norm")
                module.forward = module.fuseforward  # type: ignore

                dot_str = ","

            if verbose:
                print(dot_str, end="", flush=True)

        if verbose:
            print(" Done!")

        return self

    def profile(self, verbose: bool = True, **kwargs) -> ModelProfiler:
        """Run model profiler.

        Args:
            verbose: print profiling result.

        Returns:
            ModelProfiler instance which contains profiling result.
        """
        profiler = ModelProfiler(self, **kwargs)
        profiler.run(verbose=verbose)

        return profiler

    def forward_once(
        self, x: torch.Tensor, profile_func: Optional[Callable] = None
    ) -> torch.Tensor:
        """Forward one time only."""
        y: List[Union[torch.Tensor, None]] = []

        for i, module in enumerate(self.model):  # type: ignore
            if module.from_idx != -1:
                x = (
                    y[module.from_idx]  # type: ignore
                    if isinstance(module.from_idx, int)
                    else [x if j == -1 else y[j] for j in module.from_idx]
                )
            if profile_func is not None:
                x = profile_func(module, x, i)
            else:
                x = module(x)
            y.append(x if module.module_idx in self.output_save else None)

        return x


class YOLOModel(Model):
    """YOLO model."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, type]],
        verbose: bool = False,
        init_bias=False,
        **kwargs,
    ) -> None:
        """Parse YOLOModel from the model config file.

        Args:
            cfg: yaml file path or dictionary type of the model.
            verbose: print the model parsing information.
            init_bias: Initialize bias.
            kwargs: Keyword arguments for the YOLOHead.initialize_biases
        """
        super().__init__(cfg, verbose=verbose)

        assert isinstance(self.model[-1], YOLOHead), (
            "YOLOHead must have YOLOHead at the end! "
            f"Current last layer: {self.model[-1].name} "
        )

        self._yolo_init()
        self.initialize_biases = self.model[-1].initialize_biases

        if init_bias:
            self.initialize_biases(**kwargs)

        # YOLOv5 compatability
        self.stride = self.model[-1].stride

    def _yolo_init(self) -> None:
        """Initialize model for YOLO training."""

        # Initialize batch norm
        for module in self.model.modules():
            module_type = type(module)

            if module_type is nn.Conv2d:
                pass
            elif module_type is nn.BatchNorm2d:
                module.eps = 1e-3  # type: ignore
                module.momentum = 0.03  # type: ignore
            elif module_type in (nn.LeakyReLU, nn.ReLU, nn.ReLU6):
                module.inplace = True  # type: ignore


class ModelParser:
    """Generate PyTorch model from the model yaml file."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]] = "./model_configs/show_case.yaml",
        verbose: bool = False,
    ) -> None:
        """Generate PyTorch model from the model yaml file.

        Args:
            cfg: model config file or dict values read from the model config file.
            verbose: print the parsed model information.
        """

        self.verbose = verbose
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            with open(cfg, encoding="utf-8") as f:
                self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.input_size = None

        if (
            self.cfg
            and "input_size" in self.cfg
            and len(self.cfg["input_size"]) == 2  # type: ignore
        ):
            self.input_size = self.cfg["input_size"]

        self.custom_module_paths: Optional[Union[List[str], str]]
        if "custom_module_paths" in self.cfg:
            self.custom_module_paths = self.cfg["custom_module_paths"]  # type: ignore
        else:
            self.custom_module_paths = None

        self.in_channel = self.cfg["input_channel"]

        self.depth_multiply = self.cfg["depth_multiple"]
        self.width_multiply = self.cfg["width_multiple"]

        self.backbone_cfg: List[List] = self.cfg["backbone"]  # type: ignore
        if "head" in self.cfg:
            self.head_cfg: Optional[List[List]] = self.cfg["head"]  # type: ignore
        else:
            self.head_cfg = None

        if "channel_divisor" in self.cfg:
            self.channel_divisor: int = self.cfg["channel_divisor"]  # type: ignore
        else:
            self.channel_divisor = GeneratorAbstract.CHANNEL_DIVISOR

        self.model, self.output_save = self._parse_model()

    def log(self, msg: str):
        """Log."""
        if self.verbose:
            print(msg)

    def _parse_model(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self,
    ) -> Tuple[nn.Sequential, List[int]]:
        """Parse model."""
        in_channels: List[int] = []
        in_sizes: List[int] = []
        layers: List[nn.Module] = []
        output_save: List[int] = []
        logger = ModelInfoLogger(self.input_size is not None)

        if self.head_cfg is not None:
            model_cfg = self.backbone_cfg + self.head_cfg
        else:
            model_cfg = self.backbone_cfg

        channel_divisor = GeneratorAbstract.CHANNEL_DIVISOR
        GeneratorAbstract.CHANNEL_DIVISOR = self.channel_divisor
        for i, module_cfg in enumerate(model_cfg):  # type: ignore
            if len(module_cfg) > 4:
                idx, repeat, module, args, kwargs = module_cfg
            else:
                idx, repeat, module, args = module_cfg
                kwargs = None

            if i >= len(self.backbone_cfg):
                GeneratorAbstract.CHANNEL_DIVISOR = 1
                width_multiply = 1.0
                depth_multiply = 1.0
            else:
                width_multiply = float(self.width_multiply)  # type: ignore
                depth_multiply = float(self.depth_multiply)  # type: ignore

            module_generator = ModuleGenerator(
                module, custom_module_paths=self.custom_module_paths
            )(
                *args,
                keyword_args=kwargs,
                from_idx=idx,
                in_channels=tuple(in_channels) if i > 0 else (self.in_channel,),  # type: ignore
                width_multiply=width_multiply,
            )
            repeat = max(round(repeat * depth_multiply), 1) if repeat > 1 else repeat

            if isinstance(module_generator, FlattenGenerator):
                if self.input_size is not None:
                    module_generator.in_shape = in_sizes[idx]  # type: ignore
                else:
                    module_generator.in_shape = [in_channels[idx], 1, 1]  # type: ignore
            elif isinstance(module_generator, YOLOHeadGenerator):
                module_generator.in_shape = [in_sizes[idx_shape] for idx_shape in idx]
                module_generator.input_size = self.input_size  # type: ignore

            module = module_generator(repeat=repeat)
            module.module_idx, module.from_idx = i, idx

            if self.input_size is not None:
                in_size = (
                    np.array(in_sizes, dtype=np.object_)[idx]
                    if i > 0
                    else [self.in_channel] + self.input_size
                )
                out_size = module_generator.compute_out_shape(in_size, repeat=repeat)
                in_sizes.append(out_size)
            else:
                in_size, out_size = None, None

            in_channels.append(module_generator.out_channel)
            layers.append(module)

            output_save.extend(
                [x % i for x in ([idx] if isinstance(idx, int) else idx) if x != -1]
            )

            logger.add(
                (i, idx, repeat),
                module,
                module_generator,
                args,
                kwargs=kwargs,
                in_size=in_size,
                out_size=out_size,
            )

        GeneratorAbstract.CHANNEL_DIVISOR = channel_divisor
        parsed_model = nn.Sequential(*layers)
        n_param = sum([x.numel() for x in parsed_model.parameters()])
        n_grad = sum([x.numel() for x in parsed_model.parameters() if x.requires_grad])

        self.log(logger.info)
        self.log(
            f"Model Summary: {len(list(parsed_model.modules())):,d} "
            f"layers, {n_param:,d} parameters, {n_grad:,d} gradients"
            f"\n"
        )

        return parsed_model, output_save
