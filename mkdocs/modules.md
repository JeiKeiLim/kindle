# Kindle Modules
## Supported Modules Summary

|Module|Components|Arguments|
|-|-|-|
|Conv|Conv -> BatchNorm -> Activation|[channel, kernel size, stride, padding, activation]|
|DWConv|DWConv -> BatchNorm -> Activation|[channel, kernel_size, stride, padding, activation]|
|Bottleneck|Expansion ConvBNAct -> ConvBNAct|[channel, shortcut, groups, expansion, activation]
|AvgPool|Average pooling|[kernel_size, stride, padding]|
|MaxPool|Max pooling|[kernel_size, stride, padding]|
|GlobalAvgPool|Global Average Pooling|[]|
|Flatten|Flatten|[]|
|Concat|Concatenation|[dimension]|
|Linear|Linear|[channel, activation]|

## Design Custom Module with YAML
You can make your own custom module with yaml file.

**1. custom_module.yaml**
```yaml
args: [96, 32]

module:
    # [from, repeat, module, args]
    [
        [-1, 1, Conv, [arg0, 1, 1]],
        [0, 1, Conv, [arg1, 3, 1]],
        [0, 1, Conv, [arg1, 5, 1]],
        [0, 1, Conv, [arg1, 7, 1]],
        [[1, 2, 3], 1, Concat, [1]],
        [[0, 4], 1, Add, []],
    ]
```

* Arguments of yaml module can be defined as arg0, arg1 ...

**2. model_with_custom_module.yaml**
```yaml
input_size: [32, 32]
input_channel: 3

depth_multiple: 1.0
width_multiple: 1.0

backbone:
    [
        [-1, 1, Conv, [6, 5, 1, 0]],
        [-1, 1, MaxPool, [2]],
        [-1, 1, YamlModule, ["custom_module.yaml", 48, 16]],
        [-1, 1, MaxPool, [2]],
        [-1, 1, Flatten, []],
        [-1, 1, Linear, [120, ReLU]],
        [-1, 1, Linear, [84, ReLU]],
        [-1, 1, Linear, [10]]
    ]
```
* Note that argument of yaml module can be provided.

**3. Build model**
```python
from kindle import Model

model = Model("model_with_custom_module.yaml"), verbose=True)
```
```shell
idx |       from |   n |     params |          module |            arguments |                       in shape |       out shape |
---------------------------------------------------------------------------------------------------------------------------------
  0 |         -1 |   1 |        616 |            Conv |         [6, 5, 1, 0] |                    [3, 32, 32] |     [8, 32, 32] |
  1 |         -1 |   1 |          0 |         MaxPool |                  [2] |                      [8 32 32] |     [8, 16, 16] |
  2 |         -1 |   1 |     10,832 |      YamlModule |    ['custom_module'] |                      [8 16 16] |    [24, 16, 16] |
  3 |         -1 |   1 |          0 |         MaxPool |                  [2] |                     [24 16 16] |      [24, 8, 8] |
  4 |         -1 |   1 |          0 |         Flatten |                   [] |                       [24 8 8] |          [1536] |
  5 |         -1 |   1 |    184,440 |          Linear |        [120, 'ReLU'] |                         [1536] |           [120] |
  6 |         -1 |   1 |     10,164 |          Linear |         [84, 'ReLU'] |                          [120] |            [84] |
  7 |         -1 |   1 |        850 |          Linear |                 [10] |                           [84] |            [10] |
Model Summary: 36 layers, 206,902 parameters, 206,902 gradients
```

## Design Custom Module from Source
You can make your own custom module from the source.

**1. custom_module_model.yaml**
```yaml
input_size: [32, 32]
input_channel: 3

depth_multiple: 1.0
width_multiple: 1.0

custom_module_paths: ["tests.test_custom_module"]  # Paths to the custom modules of the source

backbone:
    # [from, repeat, module, args]
    [
        [-1, 1, MyConv, [6, 5, 3]],
        [-1, 1, MaxPool, [2]],
        [-1, 1, MyConv, [16, 3, 5, SiLU]],
        [-1, 1, MaxPool, [2]],
        [-1, 1, Flatten, []],
        [-1, 1, Linear, [120, ReLU]],
        [-1, 1, Linear, [84, ReLU]],
        [-1, 1, Linear, [10]]
    ]
```

**2. Write** ***PyTorch*** **module and** ***ModuleGenerator***

tests/test_custom_module.py
```python
from typing import List, Union

import numpy as np
import torch
from torch import nn

from kindle.generator import GeneratorAbstract
from kindle.torch_utils import Activation, autopad


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

    def __call__(self, repeat: int = 1) -> nn.Module:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        if repeat > 1:
            module = [MyConv(*args) for _ in range(repeat)]
        else:
            module = MyConv(*args)

        return self._get_module(module)
```

**3. Build a model**
```python
from kindle import Model

model = Model("custom_module_model.yaml"), verbose=True)
```
```shell
idx |       from |   n |     params |          module |            arguments |                       in shape |       out shape |
---------------------------------------------------------------------------------------------------------------------------------
  0 |         -1 |   1 |      1,066 |          MyConv |            [6, 5, 3] |                    [3, 32, 32] |     [8, 32, 32] |
  1 |         -1 |   1 |          0 |         MaxPool |                  [2] |                      [8 32 32] |     [8, 16, 16] |
  2 |         -1 |   1 |      3,488 |          MyConv |   [16, 3, 5, 'SiLU'] |                      [8 16 16] |    [16, 16, 16] |
  3 |         -1 |   1 |          0 |         MaxPool |                  [2] |                     [16 16 16] |      [16, 8, 8] |
  4 |         -1 |   1 |          0 |         Flatten |                   [] |                       [16 8 8] |          [1024] |
  5 |         -1 |   1 |    123,000 |          Linear |        [120, 'ReLU'] |                         [1024] |           [120] |
  6 |         -1 |   1 |     10,164 |          Linear |         [84, 'ReLU'] |                          [120] |            [84] |
  7 |         -1 |   1 |        850 |          Linear |                 [10] |                           [84] |            [10] |
Model Summary: 29 layers, 138,568 parameters, 138,568 gradients
```
