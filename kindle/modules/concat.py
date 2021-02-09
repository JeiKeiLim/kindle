"""Concat module.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
# pylint: disable=useless-super-delegation

from typing import List, Tuple, Union

import torch
from torch import nn


class Concat(nn.Module):
    """Concatenation module.

    Arguments: [dimension]
    """

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
