"""torch.nn module generator.

!!! Note
    Experimental feature.
    This might change in future release.
    Use with caution.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import inspect
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract


class TorchNNModuleGenerator(GeneratorAbstract):
    """torch.nn module generator."""

    def __init__(self, *args, module_name: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module_name = module_name
        self.base_module: nn.Module = getattr(
            __import__("torch.nn", fromlist=[""]), self.module_name[3:]
        )

    @property
    def out_channel(self) -> int:
        in_sizes = (self.in_channel, 128, 128)
        for i in range(len(in_sizes), 0, -1):
            try:
                out_shape = self.compute_out_shape(np.array(in_sizes[:i]))
                break
            except RuntimeError:
                continue
            except ValueError:
                continue

        return out_shape[0]

    @property
    def in_channel(self) -> int:
        return self.in_channels[self.from_idx]  # type: ignore

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))

        return list(module_out.shape[1:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        kwarg_name_0 = inspect.getfullargspec(self.base_module).args[1]
        if kwarg_name_0.startswith("in") or kwarg_name_0 in ["num_features"]:
            args = (self.in_channel, *self.args)
        else:
            args = self.args

        kwargs = self._get_kwargs(self.base_module, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [self.base_module(**self.kwargs) for _ in range(repeat)]
        else:
            module = self.base_module(**self.kwargs)

        out_module = self._get_module(module)
        out_module.name = self.module_name  # type: ignore
        return out_module
