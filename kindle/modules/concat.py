"""Concat module, generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
# pylint: disable=useless-super-delegation

from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from kindle.modules.base_generator import GeneratorAbstract


class Concat(nn.Module):
    """Concatenation module."""

    def __init__(self, dimension: int = 1) -> None:
        """Concatenation module.

        Args:
            dimension: concatenation axis.
        """
        super().__init__()
        self.dimension = dimension

    def forward(self, x: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]):
        """Forward."""
        return torch.cat(x, self.dimension)


class ConcatGenerator(GeneratorAbstract):
    """Concatenation module generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        # error: Value of type "Optional[List[int]]" is not indexable
        # error: Item "int" of "Union[int, List[int]]" has no attribute
        # "__iter__" (not iterable)
        return sum([self.in_channels[i] for i in self.from_idx])  # type: ignore

    @property
    def in_channel(self) -> int:
        """Get in channel size."""
        return -1

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        """Compute out shape."""
        return [self.out_channel] + list(size[0][1:])

    def __call__(self, repeat: int = 1):
        module = Concat(*self.args)

        return self._get_module(module)
