"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch import nn

from kindle import Model
from kindle.generator import GeneratorAbstract
from kindle.modules import Activation
from kindle.utils.torch_utils import autopad, count_model_params


class MyConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n: int,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        super().__init__()
        convs = []
        for i in range(n):
            convs.append(
                nn.Conv2d(
                    in_channels,
                    in_channels if (i + 1) != n else out_channels,
                    kernel_size,
                    padding=autopad(kernel_size),
                    bias=False,
                )
            )

        self.convs = nn.Sequential(*convs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = Activation(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batch_norm(self.convs(x)))


class MyConvGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def in_channel(self) -> int:
        if isinstance(self.from_idx, list):
            raise Exception("from_idx can not be a list.")
        return self.in_channels[self.from_idx]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(MyConv, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [MyConv(**self.kwargs) for _ in range(repeat)]
        else:
            module = MyConv(**self.kwargs)

        return self._get_module(module)


def test_custom_module(verbose: bool = False):
    model = Model(
        os.path.join("tests", "test_configs", "custom_module_model.yaml"),
        verbose=verbose,
    )

    assert model(torch.rand(1, 3, 32, 32)).shape == torch.Size([1, 10])
    assert count_model_params(model) == 138568


if __name__ == "__main__":
    test_custom_module(verbose=True)
