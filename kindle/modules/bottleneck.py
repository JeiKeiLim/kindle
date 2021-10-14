"""Bottleneck(ResNet) module.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import List, Union

import torch
from torch import nn

from kindle.modules.activation import Activation
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
        """Initialize Bottleneck instance.

        Args:
            in_channels: number of incoming channels
            out_channels: number of outgoing channels
            shortcut: whether to use shortcut connection.
                This only works when in_channels and out_channels are identical.
            groups: number of group convolution number.
            expansion: expansion ratio.
            activation: Name of the activation to use in convolution.
        """
        super().__init__()
        expansion_channel = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, expansion_channel, 1, 1, activation=activation)
        self.conv2 = Conv(
            expansion_channel, out_channels, 3, 1, groups=groups, activation=activation
        )
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv2(self.conv1(x))

        if self.shortcut:
            out = out + x

        return out


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks.

    Arguments: [channel, shortcut, groups, expansion, activation]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_repeat: int = 1,
        shortcut=True,
        groups: int = 1,
        expansion: float = 0.5,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Initialize BottleneckCSP instance.

        Args:
            in_channels: number of incoming channels
            out_channels: number of outgoing channels
            n_repeat: repeat number of bottleneck.
            shortcut: whether to use shortcut connection.
                This only works when in_channels and out_channels are identical.
            groups: number of group convolution number.
            expansion: expansion ratio.
            activation: Name of the activation to use in convolution.
        """
        super().__init__()

        expansion_channel = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, expansion_channel, 1, 1, activation=activation)
        self.conv2 = nn.Conv2d(in_channels, expansion_channel, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(expansion_channel, expansion_channel, 1, 1, bias=False)
        self.conv4 = Conv(
            2 * expansion_channel, out_channels, 1, 1, activation=activation
        )
        self.batch_norm = nn.BatchNorm2d(2 * expansion_channel)
        self.activation = Activation(activation)()

        self.bottleneck_csp = nn.Sequential(
            *[
                Bottleneck(
                    expansion_channel,
                    expansion_channel,
                    shortcut=shortcut,
                    groups=groups,
                    expansion=1.0,
                    activation=activation,
                )
                for _ in range(n_repeat)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward bottleneck CSP.

        Args:
            x: input tensor.
        """
        conv_y1 = self.conv3(self.bottleneck_csp(self.conv1(x)))
        conv_y2 = self.conv2(x)

        return self.conv4(
            self.activation(self.batch_norm(torch.cat((conv_y1, conv_y2), dim=1)))
        )


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_repeat: int = 1,
        shortcut=True,
        groups: int = 1,
        expansion: float = 0.5,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Initialize BottleneckCSP instance.

        Args:
            in_channels: number of incoming channels
            out_channels: number of outgoing channels
            n_repeat: repeat number of bottleneck.
            shortcut: whether to use shortcut connection.
                This only works when in_channels and out_channels are identical.
            groups: number of group convolution number.
            expansion: expansion ratio.
            activation: Name of the activation to use in convolution.
        """
        super().__init__()
        expansion_channel = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, expansion_channel, 1, 1, activation=activation)
        self.conv2 = Conv(in_channels, expansion_channel, 1, 1, activation=activation)
        self.conv3 = Conv(
            2 * expansion_channel, out_channels, 1, activation=activation
        )  # act=FReLU(out_channels)

        self.bottleneck_c3 = nn.Sequential(
            *[
                Bottleneck(
                    expansion_channel,
                    expansion_channel,
                    shortcut=shortcut,
                    groups=groups,
                    expansion=1.0,
                    activation=activation,
                )
                for _ in range(n_repeat)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Bottleneck C3.

        Args:
            x: input tensor
        """
        return self.conv3(
            torch.cat((self.bottleneck_c3(self.conv1(x)), self.conv2(x)), dim=1)
        )


class MV2Block(nn.Module):
    """MobileNet v2 block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 4,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """MobileNet v2 block.

        https://github.com/pytorch/vision/blob/972ca657416c5896ba6e0f5fe7c0d0f3e99e1087/torchvision/models/mobilenetv2.py

        Args:
            in_channels: number of incoming channels
            out_channels: number of outgoing channels
            stride: stride.
            expand_ratio: expansion ratio.
            activation: Name of the activation to use in convolution.
        """
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv(in_channels, hidden_dim, kernel_size=1, activation=activation)
            )

        layers.extend(
            [
                # dw
                Conv(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_dim,
                    activation=activation,
                ),
                # pw-linear
                # TODO(jeikeilim): Add fuse forward here.
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward MobileNet v2 block."""
        if self.use_res_connect:
            return x + self.conv(x)

        return self.conv(x)
