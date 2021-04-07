"""Linear module.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import Union

import torch
from torch import nn

from kindle.modules.activation import Activation


class Linear(nn.Module):
    """Linear module.

    Arguments: [channel, activation]
    """

    def __init__(
        self, in_channels: int, out_channels: int, activation: Union[str, None]
    ):
        """

        Args:
            in_channels: input channels.
            out_channels: output channels.
            activation: activation name. If None is given, nn.Identity is applied
                which is no activation.
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.activation = Activation(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.activation(self.linear(x))
