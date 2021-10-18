"""PyTorch Module Generator for parsing model yaml file.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from kindle.generator.add import AddGenerator
from kindle.generator.base_generator import GeneratorAbstract, ModuleGenerator
from kindle.generator.bottleneck import (BottleneckCSPGenerator,
                                         BottleneckGenerator, C3Generator,
                                         MV2BlockGenerator)
from kindle.generator.concat import ConcatGenerator
from kindle.generator.conv import (ConvGenerator, DWConvGenerator,
                                   FocusGenerator)
from kindle.generator.custom_yaml_module import YamlModuleGenerator
from kindle.generator.flatten import FlattenGenerator
from kindle.generator.identity import IdentityGenerator
from kindle.generator.linear import LinearGenerator
from kindle.generator.mobilevit import MobileViTBlockGenerator
from kindle.generator.nn import TorchNNModuleGenerator
from kindle.generator.poolings import (AvgPoolGenerator,
                                       GlobalAvgPoolGenerator,
                                       MaxPoolGenerator, SPPFGenerator,
                                       SPPGenerator)
from kindle.generator.pretrained import (PreTrainedFeatureMapGenerator,
                                         PreTrainedGenerator)
from kindle.generator.upsample import UpSampleGenerator
from kindle.generator.yolo_head import YOLOHeadGenerator

__all__ = [
    "ModuleGenerator",
    "GeneratorAbstract",
    "BottleneckGenerator",
    "BottleneckCSPGenerator",
    "C3Generator",
    "MV2BlockGenerator",
    "ConcatGenerator",
    "ConvGenerator",
    "DWConvGenerator",
    "FocusGenerator",
    "FlattenGenerator",
    "LinearGenerator",
    "AvgPoolGenerator",
    "GlobalAvgPoolGenerator",
    "MaxPoolGenerator",
    "SPPGenerator",
    "SPPFGenerator",
    "YamlModuleGenerator",
    "AddGenerator",
    "UpSampleGenerator",
    "IdentityGenerator",
    "TorchNNModuleGenerator",
    "PreTrainedGenerator",
    "PreTrainedFeatureMapGenerator",
    "YOLOHeadGenerator",
    "MobileViTBlockGenerator",
]
