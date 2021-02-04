"""Bottleneck(ResNet) module, generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import List, Union

import numpy as np
import torch
from torch import nn

from kindle.modules.base_generator import GeneratorAbstract
from kindle.modules.conv import Conv


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut=True,
        groups: int = 1,
        expansion: float = 0.5,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Initialize."""
        super().__init__()
        expansion_channel = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, expansion_channel, 1, 1, activation=activation)
        self.conv2 = Conv(expansion_channel, out_channels, 3, 1, groups=groups)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv2(self.conv1(x))

        if self.shortcut:
            out = out + x

        return out


class BottleneckGenerator(GeneratorAbstract):
    """Bottleneck block generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def in_channel(self) -> int:
        """Get in channel size."""
        # error: Value of type "Optional[List[int]]" is not indexable
        return self.in_channels[self.from_idx]  # type: ignore

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from kindle.common_modules based on the class name."""
        return getattr(__import__("kindle.modules", fromlist=[""]), self.name)

    def compute_out_shape(
        self, size: Union[list, np.ndarray], repeat: int = 1
    ) -> List[int]:
        """Compute output shape."""
        with torch.no_grad():
            module: nn.Module = self(repeat=repeat)
            module_out: torch.Tensor = module(torch.zeros([1, *list(size)]))
            return list(module_out.shape[-3:])

    def __call__(self, repeat: int = 1):
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        module = self.base_module(*args)
        return self._get_module(module)
