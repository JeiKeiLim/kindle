"""MaxPool, AvgPool, and GlobalAvgPool modules generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract
from kindle.modules.poolings import GlobalAvgPool


class MaxPoolGenerator(GeneratorAbstract):
    """Max pooling module generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.in_channels[self.from_idx]  # type: ignore

    @property
    def in_channel(self) -> int:
        """Get in channel size."""
        return self.in_channels[self.from_idx]  # type: ignore

    @property
    def base_module(self) -> nn.Module:
        """Base module."""
        return getattr(nn, f"{self.name}2d")

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        """Compute out shape."""
        with torch.no_grad():
            module: nn.Module = self(repeat=repeat)
            module_out: torch.Tensor = module(torch.zeros([1, *list(size)]))
            return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        kwargs = self._get_kwargs(self.base_module, self.args)

        return kwargs

    def __call__(self, repeat: int = 1):

        module = (
            [self.base_module(**self.kwargs) for _ in range(repeat)]
            if repeat > 1
            else self.base_module(**self.kwargs)
        )
        return self._get_module(module)


class AvgPoolGenerator(MaxPoolGenerator):
    """Average pooling module generator."""


class SPPGenerator(GeneratorAbstract):
    """Spatial Pyramid Pooling module generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def in_channel(self) -> int:
        """Get in channel size."""
        # error: Value of type "Optional[List[int]]" is not indexable
        return self.in_channels[self.from_idx]  # type: ignore

    @property
    def out_channel(self) -> int:
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from kindle.modules based on the class name."""
        return getattr(__import__("kindle.modules", fromlist=[""]), self.name)

    @property
    def kwargs(self) -> Dict[str, Any]:
        out_channels = self._get_divisible_channel(self.args[0] * self.width_multiply)
        args = [self.in_channel, out_channels, *self.args[1:]]
        kwargs = self._get_kwargs(self.base_module, args)

        return kwargs

    @torch.no_grad()
    def compute_out_shape(
        self, size: Union[list, np.ndarray], repeat: int = 1
    ) -> List[int]:
        module: nn.Module = self(repeat=repeat)
        module.eval()
        module_out: torch.Tensor = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    def __call__(self, repeat: int = 1):
        kwargs = self.kwargs
        module = self.base_module(**kwargs)

        return self._get_module(module)


class SPPFGenerator(SPPGenerator):
    """Spatial Pyramid Pooling - Fast module generator."""


class GlobalAvgPoolGenerator(GeneratorAbstract):
    """Global average pooling module generator."""

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        if isinstance(self.from_idx, int):
            return self.in_channels[self.from_idx]

        raise Exception()

    @property
    def in_channel(self) -> int:
        """Get in channel size."""
        if isinstance(self.from_idx, int):
            return self.in_channels[self.from_idx]

        raise Exception()

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._get_kwargs(GlobalAvgPool, self.args)

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        """Compute out shape."""
        return [self.out_channel, 1, 1]

    def __call__(self, repeat: int = 1):
        return self._get_module(GlobalAvgPool())
