"""Conv module, generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import List, Union

import numpy as np
import torch
from torch import nn

from kindle.modules.base_generator import GeneratorAbstract
from kindle.torch_utils import Activation, autopad


class Conv(nn.Module):
    """Standard convolution with batch normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, None] = None,
        groups: int = 1,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Standard convolution with batch normalization and activation.

        Args:
            in_channels: input channels.
            out_channels: output channels.
            kernel_size: kernel size.
            stride: stride.
            padding: input padding. If None is given, autopad is applied
                which is identical to padding='SAME' in TensorFlow.
            groups: group convolution.
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
            groups=groups,
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


class ConvGenerator(GeneratorAbstract):
    """Conv2d generator for parsing module."""

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

    @torch.no_grad()
    def compute_out_shape(
        self, size: Union[list, np.ndarray], repeat: int = 1
    ) -> List[int]:
        module: nn.Module = self(repeat=repeat)
        module.eval()
        module_out: torch.Tensor = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    def __call__(self, repeat: int = 1):
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        if repeat > 1:
            stride = 1
            # Important!: stride only applies at the end of the repeat.
            if len(args) > 2:
                stride = args[3]
                args[3] = 1

            module = []
            for i in range(repeat):
                if len(args) > 1 and stride > 1 and i == repeat - 1:
                    args[3] = stride

                module.append(self.base_module(*args))
                args[0] = self.out_channel
        else:
            module = self.base_module(*args)

        return self._get_module(module)
