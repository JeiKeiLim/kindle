"""Kindle is an easy model build package for PyTorch.

Building a deep learning model became so simple that
almost all model can be made by copy and paste from
other existing model codes.

So why code? when we can simply build a model with yaml markup file.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
- URL: https://github.com/JeiKeiLim/kindle
"""

from kindle import generator, modules
from kindle.model import Model, YOLOModel
from kindle.trainer import TorchTrainer
from kindle.utils import ModelProfiler

from .version import __version__

__all__ = [
    "modules",
    "generator",
    "Model",
    "YOLOModel",
    "TorchTrainer",
    "ModelProfiler",
    "__version__",
]
