"""Bottleneck(ResNet) module.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import Union

import torch
from torch import nn

from kindle.modules.conv import Conv


class Bottleneck(nn.Module):
    """Standard bottleneck block.

    Arguments: [channel, shortcut, groups, expansion, activation]
    """

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
