"""Conv module generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract


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
        """Returns module class from kindle.modules based on the class name."""
        return getattr(__import__("kindle.modules", fromlist=[""]), self.name)

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
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

        if repeat > 1:
            stride = 1
            # Important!: stride only applies at the end of the repeat.
            if kwargs["stride"] > 1:
                stride = kwargs["stride"]
                kwargs["stride"] = 1

            module = []
            for i in range(repeat):
                if stride > 1 and i == repeat - 1:
                    kwargs["stride"] = stride

                module.append(self.base_module(**kwargs))
                kwargs["in_channels"] = self.out_channel
        else:
            module = self.base_module(**kwargs)

        return self._get_module(module)


class DWConvGenerator(ConvGenerator):
    """Depth-wise convolution generator for parsing module."""


class FocusGenerator(ConvGenerator):
    """Focus convolution generator."""

    @property
    def in_channel(self) -> int:
        """Get in channel size."""
        return self.in_channels[self.from_idx] * 4  # type: ignore
