"""PyTorch Module and ModuleGenerator."""

from kindle.modules.base_generator import GeneratorAbstract, ModuleGenerator
from kindle.modules.bottleneck import Bottleneck, BottleneckGenerator
from kindle.modules.concat import Concat, ConcatGenerator
from kindle.modules.conv import Conv, ConvGenerator
from kindle.modules.dwconv import DWConv, DWConvGenerator
from kindle.modules.flatten import FlattenGenerator
from kindle.modules.linear import Linear, LinearGenerator
from kindle.modules.poolings import (AvgPoolGenerator, GlobalAvgPool,
                                     GlobalAvgPoolGenerator, MaxPoolGenerator)

__all__ = [
    "ModuleGenerator",
    "GeneratorAbstract",
    "Bottleneck",
    "Concat",
    "Conv",
    "DWConv",
    "Linear",
    "GlobalAvgPool",
    "BottleneckGenerator",
    "ConcatGenerator",
    "ConvGenerator",
    "LinearGenerator",
    "DWConvGenerator",
    "FlattenGenerator",
    "MaxPoolGenerator",
    "AvgPoolGenerator",
    "GlobalAvgPoolGenerator",
]
