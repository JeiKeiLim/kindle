"""UpSample module generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn

from kindle.generator.base_generator import GeneratorAbstract


class UpSampleGenerator(GeneratorAbstract):
    """UpSample module generator."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self.in_channel

    @property
    def in_channel(self) -> int:
        return self.in_channels[self.from_idx]  # type: ignore

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module_out = module(torch.zeros([1, *list(size)]))

        return list(module_out.shape[1:])

    def __call__(self, repeat: int = 1) -> nn.Module:
        return self._get_module(nn.Upsample(scale_factor=2))
