"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import ast
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn


class Activation:
    """Convert string activation name to the activation class."""

    def __init__(self, act_type: Union[str, None]) -> None:
        """Convert string activation name to the activation class.

        Args:
            act_type: Activation name.

        Returns:
            nn.Identity if {type} is None.
        """
        self.name = act_type
        self.args = [1] if self.name == "Softmax" else []

    def __call__(self) -> nn.Module:
        if self.name is None:
            return nn.Identity()
        if hasattr(nn, self.name):
            return getattr(nn, self.name)(*self.args)

        return ast.literal_eval(self.name)()


class SiLU(nn.Module):
    """Export-friendly version of nn.SiLU()

    SiLU https://arxiv.org/pdf/1606.08415.pdf
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Forward SiLU activation."""
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """Export-friendly version of nn.Hardswish()"""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Forward Hardswish activation."""
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class Mish(nn.Module):
    """Mish https://github.com/digantamisra98/Mish."""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Forward Mish activation."""
        return x * F.softplus(x).tanh()
