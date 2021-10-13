"""Module generator related to pooling operations.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import Tuple, Union

import torch
from torch import nn

from kindle.modules.conv import Conv


class GlobalAvgPool(nn.AdaptiveAvgPool2d):
    """Global average pooling module.

    Arguments: []
    """

    def __init__(self):
        """Initialize."""
        super().__init__(output_size=1)


class SPP(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Arguments: [channel, [kernel_size1, kernel_size2, ...], activation]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int, ...] = (5, 9, 13),
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Initialize Spatical Pyramid Pooling module.

        Args:
            in_channels: number of incoming channels
            out_channels: number of outgoing channels
            kernel_sizes: kernel sizes to use
            activation: Name of the activation to use in convolution.
        """
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = Conv(
            hidden_channels * (len(kernel_sizes) + 1),
            out_channels,
            1,
            1,
            activation=activation,
        )
        self.pooling_modules = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=k_size, stride=1, padding=k_size // 2)
                for k_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward."""
        x = self.conv1(x)
        return self.conv2(
            torch.cat([x] + [module(x) for module in self.pooling_modules], 1)
        )


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Initialize SPPF module.

        Args:
            in_channels: number of incoming channels
            out_channels: number of outgoing channels
            kernel_size: kernel sizes to use. 5 is equivalent to kernel_sizes=(5, 9, 13) in SPP
            activation: Name of the activation to use in convolution.
        """
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.pooling_module = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fee forward."""
        x = self.conv1(x)
        y_1 = self.pooling_module(x)
        y_2 = self.pooling_module(y_1)
        return self.conv2(torch.cat([x, y_1, y_2, self.pooling_module(y_2)], 1))
