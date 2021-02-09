"""Flatten module generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import math
from typing import List

import numpy as np
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract


class FlattenGenerator(GeneratorAbstract):
    """Flatten module generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return math.prod(self.args)

    @property
    def in_channel(self) -> int:

        return -1

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return [self.out_channel]

    def __call__(self, repeat: int = 1):
        return self._get_module(nn.Flatten())
