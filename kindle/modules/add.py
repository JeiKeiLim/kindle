"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import List, Tuple, Union

import torch
from torch import nn


class Add(nn.Module):
    """Add module for Kindle."""

    def __init_(self):
        """Initialize module."""
        super().__init__()

    @classmethod
    def forward(
        cls, x: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
    ) -> torch.Tensor:
        """Add inputs.

        Args:
            x: list of torch tensors

        Returns:
            sum of all x's
        """
        result = x[0]
        for i in range(1, len(x)):
            result += x[i]

        return result
