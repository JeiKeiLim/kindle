"""Identity module that returns raw input.

This module can be used for the reusability of the input layer.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import List

import numpy as np
import torch.nn as nn

from kindle.generator.base_generator import GeneratorAbstract


class IdentityGenerator(GeneratorAbstract):
    """Identity module."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self.in_channel

    @property
    def in_channel(self) -> int:
        return self.in_channels[self.from_idx]  # type: ignore

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return list(size)

    def __call__(self, repeat: int = 1) -> nn.Module:
        return self._get_module(nn.Identity())
