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
from kindle.model import Model
from kindle.trainer import TorchTrainer

__all__ = ["modules", "generator", "Model", "TorchTrainer"]
