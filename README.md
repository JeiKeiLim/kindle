# Kindle - PyTorch no-code model builder

|`Documentation`|
|---------------|
|[Reference Document](https://limjk.ai/kindle)|

Kindle is an easy model build package for [PyTorch](https://pytorch.org). Building a deep learning model became so simple that almost all model can be made by copy and paste from other existing model codes. So why code? when we can simply build a model with yaml markup file.

Kindle builds a model with no code but yaml file which its method is inspired from [YOLOv5](https://github.com/ultralytics/yolov5).

## AutoML with Kindle
* [Kindle](https://github.com/JeiKeiLim/kindle) offers the easiest way to build your own deep learning architecture. Beyond building a model, AutoML became easier with [Kindle](https://github.com/JeiKeiLim/kindle) and [Optuna](https://optuna.org) or other optimization frameworks.
* For further information, please refer to [here](https://github.com/JeiKeiLim/kindle/wiki/AutoML-with-kindle-and-optuna)

## Working environment
* Other Python3 and PyTorch version should be working but we have not checked yet.

| Python | PyTorch |
|--------|---------|
| 3.8    | 1.7.1   |



# Install
**PyTorch** is required prior to install. Please visit [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install.

You can install `kindle` by pip.
```shell
$ pip install kindle
```

## Install from source
Please visit [Install from source wiki page](https://github.com/JeiKeiLim/kindle/wiki/Install-from-source)

## For contributors
Please visit [For contributors wiki page](https://github.com/JeiKeiLim/kindle/wiki/For-contributors)

# Usage

1. Make model yaml file
  - Example model https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


```yaml
input_size: [32, 32]
input_channel: 3

depth_multiple: 1.0
width_multiple: 1.0

backbone:
    # [from, repeat, module, args]
    [
        [-1, 1, Conv, [6, 5, 1, 0]],
        [-1, 1, MaxPool, [2]],
        [-1, 1, Conv, [16, 5, 1, 0]],
        [-1, 1, MaxPool, [2]],
        [-1, 1, Flatten, []],
        [-1, 1, Linear, [120, ReLU]],
        [-1, 1, Linear, [84, ReLU]],
        [-1, 1, Linear, [10]]
    ]
```

2. Build the model with **kindle**

```python
from kindle import Model

model = Model("model.yaml"), verbose=True)
```

```shell
idx |       from |   n |     params |          module |            arguments |                       in shape |       out shape |
---------------------------------------------------------------------------------------------------------------------------------
  0 |         -1 |   1 |        616 |            Conv |         [6, 5, 1, 0] |                    [3, 32, 32] |     [8, 32, 32] |
  1 |         -1 |   1 |          0 |         MaxPool |                  [2] |                      [8 32 32] |     [8, 16, 16] |
  2 |         -1 |   1 |      3,232 |            Conv |        [16, 5, 1, 0] |                      [8 16 16] |    [16, 16, 16] |
  3 |         -1 |   1 |          0 |         MaxPool |                  [2] |                     [16 16 16] |      [16, 8, 8] |
  4 |         -1 |   1 |          0 |         Flatten |                   [] |                       [16 8 8] |          [1024] |
  5 |         -1 |   1 |    123,000 |          Linear |        [120, 'ReLU'] |                         [1024] |           [120] |
  6 |         -1 |   1 |     10,164 |          Linear |         [84, 'ReLU'] |                          [120] |            [84] |
  7 |         -1 |   1 |        850 |          Linear |                 [10] |                           [84] |            [10] |
Model Summary: 21 layers, 137,862 parameters, 137,862 gradients
```

# Supported modules
* Detailed documents can be found [here](https://limjk.ai/kindle/modules/index.html)

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


# Custom module support
## Custom module with yaml
You can make your own custom module with yaml file.

**1. custom_module.yaml**
```yaml
module:
    # [from, repeat, module, args]
    [
        [-1, 1, Conv, [16, 1, 1]],
        [0, 1, Conv, [8, 3, 1]],
        [0, 1, Conv, [8, 5, 1]],
        [0, 1, Conv, [8, 7, 1]],
        [[1, 2, 3], 1, Concat, [1]],
    ]
```

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
        [-1, 1, YamlModule, ["custom_module.yaml"]],
        [-1, 1, MaxPool, [2]],
        [-1, 1, Flatten, []],
        [-1, 1, Linear, [120, ReLU]],
        [-1, 1, Linear, [84, ReLU]],
        [-1, 1, Linear, [10]]
    ]
```

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

## Custom module from source
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

# Planned features
* ~~Custom module support~~
* ~~Custom module with yaml support~~
* Use pre-trained model
* More modules!
