"""Bottleneck module generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract


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

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(self.base_module, args)

        return kwargs

    def __call__(self, repeat: int = 1):
        module = self.base_module(**self.kwargs)
        return self._get_module(module)


class BottleneckCSPGenerator(BottleneckGenerator):
    """BottleneckCSP block generator."""

    def __call__(self, repeat: int = 1):
        kwargs = self.kwargs
        kwargs["n_repeat"] = repeat
        module = self.base_module(**kwargs)
        return self._get_module(module)


class C3Generator(BottleneckCSPGenerator):
    """BottleneckC3 block generator."""


class MV2BlockGenerator(BottleneckGenerator):
    """MobileNet v2 block generator."""

    def __call__(self, repeat: int = 1):
        kwargs = self.kwargs
        if repeat > 1:
            module = []
            out_channels = kwargs["out_channels"]

            for _ in range(repeat):
                module.append(self.base_module(**kwargs))
                kwargs["stride"] = 1
                kwargs["in_channels"] = out_channels
                kwargs["out_channels"] = out_channels
        else:
            module = self.base_module(**kwargs)

        return self._get_module(module)
