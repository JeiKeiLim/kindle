"""YOLOv5 head generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch import nn

from kindle.generator.base_generator import GeneratorAbstract
from kindle.modules.yolo_head import YOLOHead


class YOLOHeadGenerator(GeneratorAbstract):
    """Generate YOLO Head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_shape: Optional[Union[np.ndarray, List[List[int]]]] = None
        self.input_size: Optional[List[int]] = None

    @property
    def out_channel(self) -> List[int]:
        kwargs = self.kwargs
        n_layers = len(kwargs["anchors"])

        return [kwargs["n_classes"] + 5 for _ in range(n_layers)]

    @property
    def in_channel(self) -> List[int]:
        assert self.in_shape is not None

        return [s[0] for s in self.in_shape]

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[List[int]]:
        self.in_shape = size
        out_channels = self.out_channel
        return [[-1, o_channel] for o_channel in out_channels]

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = self.args

        out_xyxy = False
        if len(args) == 3:
            out_xyxy = args[2]
            args = args[:2]

        kwargs = self._get_kwargs(YOLOHead, args)
        kwargs["out_xyxy"] = out_xyxy
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        assert self.in_shape is not None and self.input_size is not None
        in_shape = self.in_shape
        input_size = self.input_size
        kwargs = self.kwargs

        n_channels = [s[0] for s in in_shape]
        strides = [input_size[0] / s[1] for s in in_shape]

        kwargs["n_channels"] = n_channels
        kwargs["strides"] = strides

        module = YOLOHead(**kwargs)
        return self._get_module(module)
