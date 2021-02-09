"""Linear module generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import List

import numpy as np

from kindle.generator.base_generator import GeneratorAbstract
from kindle.modules import Linear


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
