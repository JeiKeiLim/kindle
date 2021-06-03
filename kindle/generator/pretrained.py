"""Pretrained generator module.

Generate pre-trained module that is served from https://github.com/rwightman/pytorch-image-models

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import Any, Dict, List, Union

import numpy as np
import timm
import torch
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract
from kindle.modules.pretrained import Pretrained, PretrainedFeatureMap


class PreTrainedGenerator(GeneratorAbstract):
    """Generate Pretrained module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> Union[int, List[int]]:
        temp_in_shape = np.array([self.in_channel, 128, 128])
        out_shape = self.compute_out_shape(temp_in_shape)

        if isinstance(out_shape[0], list):
            return [o_shape[0] for o_shape in out_shape if isinstance(o_shape, list)]

        return out_shape[0]

    @property
    def in_channel(self) -> int:
        return self.in_channels[self.from_idx]  # type: ignore

    def compute_out_shape(
        self, size: np.ndarray, repeat: int = 1
    ) -> Union[List[int], List[List[int]]]:
        module = self(repeat=repeat)
        module_out = module(torch.zeros([1, *list(size)]))

        if isinstance(module_out, torch.Tensor):
            return list(module_out.shape[1:])

        return [list(m_out.shape[1:]) for m_out in module_out]

    @property
    def kwargs(self) -> Dict[str, Any]:
        kwargs = self._get_kwargs(Pretrained, self.args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        module = Pretrained(**self.kwargs)
        return self._get_module(module)


class PreTrainedFeatureMapGenerator(GeneratorAbstract):
    """Generate PreTrainedFeatureMap."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self.in_channels[self.from_idx][self.kwargs["feature_idx"]]  # type: ignore

    @property
    def in_channel(self) -> int:
        return self.in_channels[self.from_idx][self.kwargs["feature_idx"]]  # type: ignore

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return list(size[self.kwargs["feature_idx"]])

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._get_kwargs(PretrainedFeatureMap, self.args)

    def __call__(self, repeat: int = 1) -> nn.Module:
        module = PretrainedFeatureMap(**self.kwargs)
        return self._get_module(module)


if __name__ == "__main__":
    m = timm.create_model("mobilenetv3_large_100", pretrained=True, features_only=True)
