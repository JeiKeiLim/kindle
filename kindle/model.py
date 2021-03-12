"""Kindle Model parser and model.

This module parses model configuration yaml file
and generates PyTorch model accordingly.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

from kindle.generator.base_generator import GeneratorAbstract, ModuleGenerator
from kindle.generator.flatten import FlattenGenerator


class Model(nn.Module):
    """PyTorch model class."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]],
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the model.

        For the time being, this method will only call self.forward_once. Later, we plan
        to add Test Time Augment.
        """
        return self.forward_once(x)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Forward one time only."""
        y: List[Union[torch.Tensor, None]] = []

        for module in self.model:  # type: ignore
            if module.from_idx != -1:
                x = (
                    y[module.from_idx]  # type: ignore
                    if isinstance(module.from_idx, int)
                    else [x if j == -1 else y[j] for j in module.from_idx]
                )

            x = module(x)
            y.append(x if module.module_idx in self.output_save else None)

        return x


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
            with open(cfg) as f:
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

        self.backbone_cfg: List[Union[int, str, float]] = self.cfg["backbone"]  # type: ignore
        if "head" in self.cfg:
            self.head_cfg: Optional[List[Union[int, str, float]]] = self.cfg["head"]  # type: ignore
        else:
            self.head_cfg = None

        self.model, self.output_save = self._parse_model()

    def log(self, msg: str):
        """Log."""
        if self.verbose:
            print(msg)

    def _log_parse(
        self,
        info: Optional[Tuple[int, int, int]] = None,
        module: Optional[nn.Module] = None,
        module_generator: Optional[GeneratorAbstract] = None,
        args: Optional[List[Any]] = None,
        in_size: Optional[Union[np.ndarray, List]] = None,
        out_size: Optional[List[int]] = None,
        head: bool = False,
    ) -> None:
        """Print log message for parsing modules.

        Args:
            info: (i, idx, repeat) Current parsing information.
            module: Parsed module.
            module_generator: Module generator used to generate module.
            args: Arguments of the module.
            in_size: Input size of the module.
                    Only required when {self.input_size} is not None.
            out_size: Output size of the module
                    Only required when {self.input_size} is not None.
            head: Print head(Column names) message only.
        """
        if head:
            log = (
                f"{'idx':>3} | {'from':>10} | {'n':>3} | {'params':>8} |"
                f" {'module':>15} | {'arguments':>20} |"
                f" {'in_channel':>10} | {'out_channel':>11} |"
            )
            if self.input_size is not None:
                log += f" {'in shape':>30} | {'out shape':>15} |"
            log += f"\n{len(log) * '-'}"
        else:
            assert (
                info is not None
                and module is not None
                and module_generator is not None
                and args is not None
            ), "info, module, and module_generator must be provided to generate log string."
            i, idx, repeat = info

            args = args.copy()
            if module.type == "YamlModule":
                args[0] = args[0].split(os.sep)[-1].split(".")[0]

            args_str = str(args)
            args_str_list = []
            for j in range(0, len(args_str), 20):
                end_idx = j + 20
                args_str_list.append(args_str[j:end_idx])
            args_str = args_str_list[0]
            log = (
                f"{i:3d} | {str(idx):>10} | {repeat:3d} |"
                f" {module.n_params:8,d} | {module.type:>15} | {args_str:>20} |"
                f" {module_generator.in_channel:>10} | {module_generator.out_channel:>11} |"
            )
            for j in range(1, len(args_str_list)):
                log += (
                    f"\n{'':>3} | {'':>10} | {'':>3} |"
                    f" {'':>8} | {'':>15} | {args_str_list[j]:>20} |"
                    f" {'':>10} | {'':>11} |"
                )

            if (
                self.input_size is not None
                and in_size is not None
                and out_size is not None
            ):
                in_size_str = str(in_size).replace("\n", ",")
                log += f" {in_size_str:>30} | {str(out_size):>15} |"

        self.log(log)

    def _parse_model(  # pylint: disable=too-many-locals
        self,
    ) -> Tuple[nn.Sequential, List[int]]:
        """Parse model."""
        in_channels: List[int] = []
        in_sizes: List[int] = []
        layers: List[nn.Module] = []
        output_save: List[int] = []
        self._log_parse(head=True)

        if self.head_cfg is not None:
            model_cfg = self.backbone_cfg + self.head_cfg
        else:
            model_cfg = self.backbone_cfg

        channel_divisor = GeneratorAbstract.CHANNEL_DIVISOR
        for i, (idx, repeat, module, args) in enumerate(model_cfg):  # type: ignore
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
                from_idx=idx,
                in_channels=tuple(in_channels) if i > 0 else (self.in_channel,),  # type: ignore
                width_multiply=width_multiply,
            )
            repeat = max(round(repeat * depth_multiply), 1) if repeat > 1 else repeat

            if isinstance(module_generator, FlattenGenerator):
                if self.input_size is not None:
                    module_generator.args = in_sizes[idx]  # type: ignore
                else:
                    module_generator.args = [in_channels[idx]]  # type: ignore

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

            self._log_parse(
                info=(i, idx, repeat),
                module=module,
                module_generator=module_generator,
                args=args,
                in_size=in_size,
                out_size=out_size,
            )

        GeneratorAbstract.CHANNEL_DIVISOR = channel_divisor
        parsed_model = nn.Sequential(*layers)
        n_param = sum([x.numel() for x in parsed_model.parameters()])
        n_grad = sum([x.numel() for x in parsed_model.parameters() if x.requires_grad])

        self.log(
            f"Model Summary: {len(list(parsed_model.modules())):,d} "
            f"layers, {n_param:,d} parameters, {n_grad:,d} gradients"
            f"\n"
        )

        return parsed_model, output_save
