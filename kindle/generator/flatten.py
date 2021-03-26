"""Flatten module generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import Any, Dict, List

import numpy as np
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract


class FlattenGenerator(GeneratorAbstract):
    """Flatten module generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_shape = [
            -1,
        ]

    @property
    def out_channel(self) -> int:
        return np.prod(self.in_shape)

    @property
    def in_channel(self) -> int:
        return -1

    @property
    def kwargs(self) -> Dict[str, Any]:
        kwargs = self._get_kwargs(nn.Flatten, self.args)
        return kwargs

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return [self.out_channel]

    def __call__(self, repeat: int = 1):
        return self._get_module(nn.Flatten(**self.kwargs))
