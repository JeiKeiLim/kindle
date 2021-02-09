"""DWConv module.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import math
# pylint: disable=useless-super-delegation
from typing import Union

import torch
from torch import nn

from kindle.torch_utils import Activation, autopad


class DWConv(nn.Module):
    """Depthwise convolution with batch normalization and activation.

    Arguments: [channel, kernel_size, stride, padding, activation]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, None] = None,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Depthwise convolution with batch normalization and activation.

        Args:
            in_channels: input channels.
            out_channels: output channels.
            kernel_size: kernel size.
            stride: stride.
            padding: input padding. If None is given, autopad is applied
                which is identical to padding='SAME' in TensorFlow.
            activation: activation name. If None is given, nn.Identity is applied
                which is no activation.
        """
        super().__init__()
        # error: Argument "padding" to "Conv2d" has incompatible type "Union[int, List[int]]";
        # expected "Union[int, Tuple[int, int]]"
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=autopad(kernel_size, padding),  # type: ignore
            groups=math.gcd(in_channels, out_channels),
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = Activation(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.activation(self.batch_norm(self.conv(x)))

    def fusefoward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse forward."""
        return self.activation(self.conv(x))
