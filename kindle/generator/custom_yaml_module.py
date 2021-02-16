"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import List, Union

import numpy as np
import torch
import yaml
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract
from kindle.model import Model


class YamlModuleGenerator(GeneratorAbstract):
    """Custom yaml module generator."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize YamlModuleGenerator."""
        super().__init__(*args, **kwargs)
        with open(self.args[0], "r") as f:
            self.cfg = yaml.load(f, yaml.FullLoader)

        self.cfg.update(
            {
                "input_channel": self.in_channel,
                "depth_multiple": 1.0,
                "width_multiple": self.width_multiply,
                "backbone": self.cfg.pop("module"),
            }
        )
        self.module = Model(self.cfg, verbose=False)

    @property
    def out_channel(self) -> int:
        temp_in_shape = np.array([self.cfg["input_channel"], 128, 128])
        out_shape = self.compute_out_shape(temp_in_shape)
        return out_shape[0]

    @property
    def in_channel(self) -> int:
        if isinstance(self.from_idx, int):
            return self.in_channels[self.from_idx]
        return sum([self.in_channels[idx] for idx in self.from_idx])

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))

        return list(module_out.shape[-3:])

    def __call__(self, repeat: int = 1) -> nn.Module:
        module: Union[List[nn.Module], nn.Module]

        if repeat > 1:
            # Currently, yaml module must have same in and out channel in order to apply repeat.
            module = [Model(self.cfg, verbose=False) for _ in range(repeat)]
        else:
            module = self.module

        return self._get_module(module)
