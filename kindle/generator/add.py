"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import List

import numpy as np
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract
from kindle.modules.add import Add


class AddGenerator(GeneratorAbstract):
    """Add module generator."""

    def __init__(self, *args, **kwargs) -> None:
        """Add module generator."""
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        if isinstance(self.from_idx, int):
            raise Exception("Add must have more than 2 inputs.")

        return self.in_channels[self.from_idx[0]]

    @property
    def in_channel(self) -> int:
        return self.out_channel

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return list(size)

    def __call__(self, repeat: int = 1) -> nn.Module:
        module = Add()

        return self._get_module(module)
