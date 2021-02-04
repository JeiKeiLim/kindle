"""Linear module, generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import List, Union

import numpy as np
import torch
from torch import nn

from kindle.modules.base_generator import GeneratorAbstract
from kindle.torch_utils import Activation


class Linear(nn.Module):
    """Linear module."""

    def __init__(self, in_channel: int, out_channel: int, activation: Union[str, None]):
        """

        Args:
            in_channels: input channels.
            out_channels: output channels.
            activation: activation name. If None is given, nn.Identity is applied
                which is no activation.
        """
        super().__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.activation = Activation(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.activation(self.linear(x))


class LinearGenerator(GeneratorAbstract):
    """Linear (fully connected) module generator for parsing."""

    def __init__(self, *args, **kwargs):
        """Initailize."""
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.args[0]

    @property
    def in_channel(self) -> int:
        """Get in channel size."""
        # error: Value of type "Optional[List[int]]" is not indexable
        return self.in_channels[self.from_idx]  # type: ignore

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        """Compute output shape."""
        return [self.out_channel]

    def __call__(self, repeat: int = 1):
        # TODO: Apply repeat
        act = self.args[1] if len(self.args) > 1 else None

        return self._get_module(
            Linear(self.in_channel, self.out_channel, activation=act)
        )
