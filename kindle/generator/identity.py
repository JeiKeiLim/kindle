"""Identity module that returns raw input.

This module can be used for the reusability of the input layer.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import Any, Dict, List

import numpy as np
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract


class IdentityGenerator(GeneratorAbstract):
    """Identity module."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self.in_channel

    @property
    def in_channel(self) -> int:
        return self.in_channels[self.from_idx]  # type: ignore

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._get_kwargs(nn.Identity, self.args)

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return list(size)

    def __call__(self, repeat: int = 1) -> nn.Module:
        return self._get_module(nn.Identity(**self.kwargs))
