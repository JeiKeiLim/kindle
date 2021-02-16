"""Kindle Model parser and model.

This module parses model configuration yaml file
and generates PyTorch model accordingly.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

from kindle.generator.base_generator import ModuleGenerator
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

        self.model_cfg: List[Union[int, str, float]] = self.cfg["backbone"]  # type: ignore

        self.model, self.output_save = self._parse_model()

    def log(self, msg: str):
        """Log."""
        if self.verbose:
            print(msg)

    def _parse_model(self) -> Tuple[nn.Sequential, List[int]]:
        """Parse model."""
        in_channels: List[int] = []
        in_sizes: List[int] = []
        layers: List[nn.Module] = []
        output_save: List[int] = []
        log: str = (
            f"{'idx':>3} | {'from':>10} | {'n':>3} | {'params':>10} "
            f"| {'module':>15} | {'arguments':>20} |"
        )
        if self.input_size is not None:
            log += f" {'in shape':>30} | {'out shape':>15} |"
        self.log(log)
        self.log(len(log) * "-")

        for i, (idx, repeat, module, args) in enumerate(self.model_cfg):  # type: ignore
            module_generator = ModuleGenerator(
                module, custom_module_paths=self.custom_module_paths
            )(
                *args,
                from_idx=idx,
                in_channels=tuple(in_channels) if i > 0 else (self.in_channel,),  # type: ignore
                width_multiply=self.width_multiply,
            )
            repeat = (
                max(round(repeat * self.depth_multiply), 1) if repeat > 1 else repeat
            )

            if isinstance(module_generator, FlattenGenerator):
                module_generator.args = in_sizes[idx]  # type: ignore

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

            in_channels.append(module_generator.out_channel)
            layers.append(module)

            args_str = args.copy()
            if module.type == "YamlModule":
                args_str[0] = args_str[0].split(os.sep)[-1].split(".")[0]

            args_str = str(args_str)

            log = (
                f"{i:3d} | {str(idx):>10} | {repeat:3d} | "
                f"{module.n_params:10,d} | {module.type:>15} | {args_str:>20} |"
            )
            if self.input_size is not None:
                in_size_str = str(in_size).replace("\n", ",")
                log += f" {in_size_str:>30} | {str(out_size):>15} |"
            self.log(log)

            output_save.extend(
                [x % i for x in ([idx] if isinstance(idx, int) else idx) if x != -1]
            )

        parsed_model = nn.Sequential(*layers)
        n_param = sum([x.numel() for x in parsed_model.parameters()])
        n_grad = sum([x.numel() for x in parsed_model.parameters() if x.requires_grad])

        self.log(
            f"Model Summary: {len(list(parsed_model.modules())):,d} "
            f"layers, {n_param:,d} parameters, {n_grad:,d} gradients"
        )

        return parsed_model, output_save
