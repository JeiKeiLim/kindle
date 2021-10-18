"""MobileViTBlock module generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract


class MobileViTBlockGenerator(GeneratorAbstract):
    """MobileViTBlock generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def in_channel(self) -> int:
        """Get in channel size."""
        # error: Value of type "Optional[List[int]]" is not indexable
        return self.in_channels[self.from_idx]  # type: ignore

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.in_channel

    def compute_out_shape(
        self, size: Union[list, np.ndarray], repeat: int = 1
    ) -> List[int]:
        """Compute output shape."""
        with torch.no_grad():
            module: nn.Module = self(repeat=repeat)
            module_out: torch.Tensor = module(torch.zeros([1, *list(size)]))
            return list(module_out.shape[-3:])

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from kindle.common_modules based on the class name."""
        return getattr(__import__("kindle.modules", fromlist=[""]), self.name)

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, *self.args]
        kwargs = self._get_kwargs(self.base_module, args)

        return kwargs

    def __call__(self, repeat: int = 1):
        module = self.base_module(**self.kwargs)
        return self._get_module(module)
