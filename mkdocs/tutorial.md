# Tutorial

## 1. Building a PyTorch model with yaml

Kindle builds a PyTorch model with yaml file.

### Components
- `input_size`: (Tuple[int, int]) (Optional) Model input image size(height, width).
- `input_channel`: (float) Model input channel size.

    !!! note
        ex) If `input_size`: [32, 32] and `input_channel`: 3 are given, input size of the model will be (batch_size, 3, 32, 32). 
        When `input_size` is not provided, Kindle assumes that the model can take any input size.

- `depth_multiple`: (float) Depth multiplication factor.
- `width_multiple`: (float) Width multiplication factor.
- `channel_divisor`: (int) (Optional) (Default: 8) Channel divisor. When `width_multiple` is adjusted, number of channel is changed to multiple of `channel_divisor`.
    
    !!! note
        ex) If `width_multiple` is 0.5 and the output channel of the module is assigned to 24, the actual output channel is `16` instead of `12`.

- `custom_module_paths`: (List[str]) (Optional) Custom module python script path list.
  
- `backbone`: (List[`module`]) Model layers. 
- `head`: (List[`module`]) (Optional) Model head. This section is same width `backbone` but `width_multiplier` is not considered which makes `head` to have fixed channel size.
   
    !!! note
        `backbone` and `head` consist of `module` list.
  
    - `module`: (List[(int or List[int]), int, str, List]) [`from index`, `repeat`, `module name`, `module arguments`]
        - `from index`: Index number of the input for the module. -1 represents a previous module. 
                        Index number of `head` is continued from `backbone`.
                        First module in `backbone` must have -1 `from index` value which represents input image.
          
        - `repeat`: Repeat number of the module. Ex) When Conv module has `repeat: 2`, this module will perform Conv operation twice (Input -> Conv -> Conv). 
        - `module_name`: Name of the module. Pre-built modules are descried [here](modules.md).
        - `module_arguments`: Arguments of the module. Each module takes pre-defined arguments. Pre-built module arguments are descried [here](modules.md).
        - `module_keyword_arguments`: Keyword argument of the module. Pre-built module keyword arguments are descried [here](modules.md).


### Example
```yaml
input_size: [32, 32]
input_channel: 3

depth_multiple: 1.0
width_multiple: 1.0

backbone:
    [
        [-1, 1, Conv, [6, 5, 1, 0], {activation: LeakyReLU}],
        [-1, 1, MaxPool, [2]],
        [-1, 1, nn.Conv2d, [16, 5, 1, 2], {bias: False}],
        [-1, 1, nn.BatchNorm2d, []],
        [-1, 1, nn.ReLU, []],
        [-1, 1, MaxPool, [2]],
        [-1, 1, Flatten, []],
        [-1, 1, Linear, [120, ReLU]],
        [-1, 1, Linear, [84, ReLU]],
    ]

head:
  [
        [-1, 1, Linear, [10]]
  ]
```

### Build a model
```python
from kindle import Model

model = Model("example.yaml"), verbose=True)
```

```shell
idx |       from |   n |   params |          module |                           arguments | in_channel | out_channel |        in shape |       out shape |
----------------------------------------------------------------------------------------------------------------------------------------------------------
  0 |         -1 |   1 |      616 |            Conv | [6, 5, 1, 0], activation: LeakyReLU |          3 |           8 |     [3, 32, 32] |     [8, 32, 32] |
  1 |         -1 |   1 |        0 |         MaxPool |                                 [2] |          8 |           8 |       [8 32 32] |     [8, 16, 16] |
  2 |         -1 |   1 |    3,200 |       nn.Conv2d |          [16, 5, 1, 2], bias: False |          8 |          16 |       [8 16 16] |    [16, 16, 16] |
  3 |         -1 |   1 |       32 |  nn.BatchNorm2d |                                  [] |         16 |          16 |      [16 16 16] |    [16, 16, 16] |
  4 |         -1 |   1 |        0 |         nn.ReLU |                                  [] |         16 |          16 |      [16 16 16] |    [16, 16, 16] |
  5 |         -1 |   1 |        0 |         MaxPool |                                 [2] |         16 |          16 |      [16 16 16] |      [16, 8, 8] |
  6 |         -1 |   1 |        0 |         Flatten |                                  [] |         -1 |        1024 |        [16 8 8] |          [1024] |
  7 |         -1 |   1 |  123,000 |          Linear |                       [120, 'ReLU'] |       1024 |         120 |          [1024] |           [120] |
  8 |         -1 |   1 |   10,164 |          Linear |                        [84, 'ReLU'] |        120 |          84 |           [120] |            [84] |
  9 |         -1 |   1 |      850 |          Linear |                                [10] |         84 |          10 |            [84] |            [10] |
Model Summary: 20 layers, 137,862 parameters, 137,862 gradients
```


## 2. Design Custom Module with YAML
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

## 3. Design Custom Module from Source
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
from typing import List, Union, Dict, Any

import numpy as np
import torch
from torch import nn

from kindle.generator import GeneratorAbstract
from kindle.utils.torch_utils import autopad
from kindle.modules.activation import Activation


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

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        return self._get_kwargs(MyConv, args)

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [MyConv(**self.kwargs) for _ in range(repeat)]
        else:
            module = MyConv(**self.kwargs)

        return self._get_module(module)
```

**3. Build a model**
```python
from kindle import Model

model = Model("custom_module_model.yaml"), verbose=True)
```
```shell
idx |       from |   n |   params |          module |                           arguments | in_channel | out_channel |        in shape |       out shape |
----------------------------------------------------------------------------------------------------------------------------------------------------------
  0 |         -1 |   1 |    1,066 |          MyConv |                           [6, 5, 3] |          3 |           8 |     [3, 32, 32] |     [8, 32, 32] |
  1 |         -1 |   1 |        0 |         MaxPool |                                 [2] |          8 |           8 |       [8 32 32] |     [8, 16, 16] |
  2 |         -1 |   1 |    3,488 |          MyConv |                  [16, 3, 5, 'SiLU'] |          8 |          16 |       [8 16 16] |    [16, 16, 16] |
  3 |         -1 |   1 |        0 |         MaxPool |                                 [2] |         16 |          16 |      [16 16 16] |      [16, 8, 8] |
  4 |         -1 |   1 |        0 |         Flatten |                                  [] |         -1 |        1024 |        [16 8 8] |          [1024] |
  5 |         -1 |   1 |  123,000 |          Linear |                       [120, 'ReLU'] |       1024 |         120 |          [1024] |           [120] |
  6 |         -1 |   1 |   10,164 |          Linear |                        [84, 'ReLU'] |        120 |          84 |           [120] |            [84] |
  7 |         -1 |   1 |      850 |          Linear |                                [10] |         84 |          10 |            [84] |            [10] |
Model Summary: 29 layers, 138,568 parameters, 138,568 gradients
```

## 4. Utilize pretrained model
Pre-trained model from [timm](https://github.com/rwightman/pytorch-image-models) can be loaded in kindle yaml config file.
Please refer to [https://rwightman.github.io/pytorch-image-models/results/](https://rwightman.github.io/pytorch-image-models/results/) for supported models.

### Example
* In this example, we load pretrained efficient-b0 model. Then we extract each feature map layer to apply convolution layer.  


**1. pretrained_model.yaml**
```yaml
input_size: [32, 32]
input_channel: 3

depth_multiple: 1.0
width_multiple: 1.0

pretrained: mobilenetv3_small_100

backbone:
    # [from, repeat, module, args]
    [
        [-1, 1, UpSample, []],
        [-1, 1, PreTrained, [efficientnet_b0, True]],
        [1, 1, PreTrainedFeatureMap, [-3]],
        [-1, 1, Conv, [8, 1], {activation: LeakyReLU}],
        [-1, 1, MaxPool, [2]],

        [1, 1, PreTrainedFeatureMap, [-2]],
        [-1, 1, Conv, [8, 1], {activation: LeakyReLU}],
        [[-1, -3], 1, Concat, []],
        [-1, 1, MaxPool, [2]],

        [1, 1, PreTrainedFeatureMap, [-1]],
        [-1, 1, Conv, [8, 1], {activation: LeakyReLU}],
        [[-1, -3], 1, Concat, []],

        [-1, 1, Flatten, []],
        [-1, 1, Linear, [120, ReLU]],
        [-1, 1, Linear, [84, ReLU]],
    ]

head:
  [
    [-1, 1, Linear, [10]]
  ]
```

* When `PreTrained` module has `features_only = True` argument, the output of the module will be list of each feature map.
* `PreTrainedFeatureMap` module simply bypass `feature_idx` output of `PreTrained`. 

**2. Build a model**
```python
from kindle import Model

model = Model("pretrained_model.yaml"), verbose=True)
```

```shell
   idx | from     |   n | params    | module               | arguments                     |   in_channel | out_channel            | in_shape                                                           | out_shape
-------+----------+-----+-----------+----------------------+-------------------------------+--------------+------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------
     0 | -1       |   1 | 0         | UpSample             | []                            |            3 | 3                      | [3, 32, 32]                                                        | [3, 64, 64]
     1 | -1       |   1 | 3,595,388 | PreTrained           | ['efficientnet_b0', True]     |            3 | [16, 24, 40, 112, 320] | [3 64 64]                                                          | [[16, 32, 32], [24, 16, 16], [40, 8, 8], [112, 4, 4], [320, 2, 2]]
     2 | 1        |   1 | 0         | PreTrainedFeatureMap | [-3]                          |           40 | 40                     | [[16, 32, 32], [24, 16, 16], [40, 8, 8], [112, 4, 4], [320, 2, 2]] | [40, 8, 8]
     3 | -1       |   1 | 336       | Conv                 | [8, 1], activation: LeakyReLU |           40 | 8                      | [40, 8, 8]                                                         | [8, 8, 8]
     4 | -1       |   1 | 0         | MaxPool              | [2]                           |            8 | 8                      | [8, 8, 8]                                                          | [8, 4, 4]
     5 | 1        |   1 | 0         | PreTrainedFeatureMap | [-2]                          |          112 | 112                    | [[16, 32, 32], [24, 16, 16], [40, 8, 8], [112, 4, 4], [320, 2, 2]] | [112, 4, 4]
     6 | -1       |   1 | 912       | Conv                 | [8, 1], activation: LeakyReLU |          112 | 8                      | [112, 4, 4]                                                        | [8, 4, 4]
     7 | [-1, -3] |   1 | 0         | Concat               | []                            |           -1 | 16                     | [list([8, 4, 4]) list([8, 4, 4])]                                  | [16, 4, 4]
     8 | -1       |   1 | 0         | MaxPool              | [2]                           |           16 | 16                     | [16, 4, 4]                                                         | [16, 2, 2]
     9 | 1        |   1 | 0         | PreTrainedFeatureMap | [-1]                          |          320 | 320                    | [[16, 32, 32], [24, 16, 16], [40, 8, 8], [112, 4, 4], [320, 2, 2]] | [320, 2, 2]
    10 | -1       |   1 | 2,576     | Conv                 | [8, 1], activation: LeakyReLU |          320 | 8                      | [320, 2, 2]                                                        | [8, 2, 2]
    11 | [-1, -3] |   1 | 0         | Concat               | []                            |           -1 | 24                     | [list([8, 2, 2]) list([16, 2, 2])]                                 | [24, 2, 2]
    12 | -1       |   1 | 0         | Flatten              | []                            |           -1 | 96                     | [24, 2, 2]                                                         | [96]
    13 | -1       |   1 | 11,640    | Linear               | [120, 'ReLU']                 |           96 | 120                    | [96]                                                               | [120]
    14 | -1       |   1 | 10,164    | Linear               | [84, 'ReLU']                  |          120 | 84                     | [120]                                                              | [84]
    15 | -1       |   1 | 850       | Linear               | [10]                          |           84 | 10                     | [84]                                                               | [10]
Model Summary: 250 layers, 3,621,866 parameters, 3,621,866 gradients
```

## 5. Make object detectiong model using YOLOHead
* You can build YOLO with simple configuration.
* In this example, we made a neck to bypass the feature maps to the YOLOHead but you can build your own detection neck layers.

!!! Note input_size 
    In order to compute stride size automatically, you will need to provide arbitrary `input_size`.
    However, the model can take any input sizes as the model is allowed to take.

**1. yolo_sample.yaml**
```yaml
input_size: [256, 256]
input_channel: 3

depth_multiple: 0.33
width_multiple: 0.5

anchors: &anchors
   - [10,13, 16,30, 33,23]  # P3/8
   - [30,61, 62,45, 59,119]  # P4
   - [116,90, 156,198, 373,326]  # P5/32

n_classes: &n_classes
  10

backbone:
    # [from, repeat, module, args]
    [
        [-1, 1, Focus, [64, 3]],
        [-1, 1, Conv, [128, 3, 2]],
        [-1, 3, BottleneckCSP, [128]],  # 2

        [-1, 1, Conv, [256, 3, 2]],
        [-1, 9, BottleneckCSP, [256]],  # 4

        [-1, 1, Conv, [512, 3, 2]],
        [-1, 9, BottleneckCSP, [512]],  # 6

        [-1, 1, Conv, [1024, 3, 2]],
        [-1, 1, SPP, [1024, [5, 9, 13]]],
        [-1, 3, BottleneckCSP, [1024, False]],  # 9

        [-1, 1, Conv, [512, 1, 1]],
        [-1, 1, UpSample, [null, 2]],
        [[-1, 6], 1, Concat, [1]],
        [-1, 3, BottleneckCSP, [512, False]],  # 13

        [-1, 1, Conv, [256, 1, 1]],
        [-1, 1, UpSample, [null, 2]],
        [[-1, 4], 1, Concat, [1]],
        [-1, 1, BottleneckCSP, [256, False]],  # 17

        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 14], 1, Concat, [1]],
        [-1, 3, BottleneckCSP, [512, False]],  # 20

        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 10], 1, Concat, [1]],
        [-1, 3, BottleneckCSP, [1024, False]]  # 23
    ]

head:
  [
    [[17, 20, 23], 1, YOLOHead, [*n_classes, *anchors]]
  ]
```

**2. Build a model**
```python
from kindle import Model

model = Model("yolo_sample.yaml", verbose=True)
```

```shell
   idx | from         |   n | params    | module        | arguments                                                                                  | in_channel      | out_channel   | in_shape                              | out_shape
-------+--------------+-----+-----------+---------------+--------------------------------------------------------------------------------------------+-----------------+---------------+---------------------------------------+--------------------------------
     0 | -1           |   1 | 3,520     | Focus         | [64, 3]                                                                                    | 12              | 32            | [3, 256, 256]                         | [32, 128, 128]
     1 | -1           |   1 | 18,560    | Conv          | [128, 3, 2]                                                                                | 32              | 64            | [32 128 128]                          | [64, 64, 64]
     2 | -1           |   1 | 19,904    | BottleneckCSP | [128]                                                                                      | 64              | 64            | [64 64 64]                            | [64, 64, 64]
     3 | -1           |   1 | 73,984    | Conv          | [256, 3, 2]                                                                                | 64              | 128           | [64 64 64]                            | [128, 32, 32]
     4 | -1           |   3 | 161,152   | BottleneckCSP | [256]                                                                                      | 128             | 128           | [128 32 32]                           | [128, 32, 32]
     5 | -1           |   1 | 295,424   | Conv          | [512, 3, 2]                                                                                | 128             | 256           | [128 32 32]                           | [256, 16, 16]
     6 | -1           |   3 | 641,792   | BottleneckCSP | [512]                                                                                      | 256             | 256           | [256 16 16]                           | [256, 16, 16]
     7 | -1           |   1 | 1,180,672 | Conv          | [1024, 3, 2]                                                                               | 256             | 512           | [256 16 16]                           | [512, 8, 8]
     8 | -1           |   1 | 656,896   | SPP           | [1024, [5, 9, 13]]                                                                         | 512             | 512           | [512 8 8]                             | [512, 8, 8]
     9 | -1           |   1 | 1,248,768 | BottleneckCSP | [1024, False]                                                                              | 512             | 512           | [512 8 8]                             | [512, 8, 8]
    10 | -1           |   1 | 131,584   | Conv          | [512, 1, 1]                                                                                | 512             | 256           | [512 8 8]                             | [256, 8, 8]
    11 | -1           |   1 | 0         | UpSample      | [None, 2]                                                                                  | 256             | 256           | [256 8 8]                             | [256, 16, 16]
    12 | [-1, 6]      |   1 | 0         | Concat        | [1]                                                                                        | -1              | 512           | [[256 16 16], [256 16 16]]            | [512, 16, 16]
    13 | -1           |   1 | 378,624   | BottleneckCSP | [512, False]                                                                               | 512             | 256           | [512 16 16]                           | [256, 16, 16]
    14 | -1           |   1 | 33,024    | Conv          | [256, 1, 1]                                                                                | 256             | 128           | [256 16 16]                           | [128, 16, 16]
    15 | -1           |   1 | 0         | UpSample      | [None, 2]                                                                                  | 128             | 128           | [128 16 16]                           | [128, 32, 32]
    16 | [-1, 4]      |   1 | 0         | Concat        | [1]                                                                                        | -1              | 256           | [[128 32 32], [128 32 32]]            | [256, 32, 32]
    17 | -1           |   1 | 95,104    | BottleneckCSP | [256, False]                                                                               | 256             | 128           | [256 32 32]                           | [128, 32, 32]
    18 | -1           |   1 | 147,712   | Conv          | [256, 3, 2]                                                                                | 128             | 128           | [128 32 32]                           | [128, 16, 16]
    19 | [-1, 14]     |   1 | 0         | Concat        | [1]                                                                                        | -1              | 256           | [[128 16 16], [128 16 16]]            | [256, 16, 16]
    20 | -1           |   1 | 313,088   | BottleneckCSP | [512, False]                                                                               | 256             | 256           | [256 16 16]                           | [256, 16, 16]
    21 | -1           |   1 | 590,336   | Conv          | [512, 3, 2]                                                                                | 256             | 256           | [256 16 16]                           | [256, 8, 8]
    22 | [-1, 10]     |   1 | 0         | Concat        | [1]                                                                                        | -1              | 512           | [[256 8 8], [256 8 8]]                | [512, 8, 8]
    23 | -1           |   1 | 1,248,768 | BottleneckCSP | [1024, False]                                                                              | 512             | 512           | [512 8 8]                             | [512, 8, 8]
    24 | [17, 20, 23] |   1 | 40,455    | YOLOHead      | [10, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]] | [128, 256, 512] | [15, 15, 15]  | [[128 32 32], [256 16 16], [512 8 8]] | [[-1, 15], [-1, 15], [-1, 15]]
Model Summary: 281 layers, 7,279,367 parameters, 7,279,367 gradients
```

**3. Initialize biases**

* Generally, object detection is better trained when biases is initialized with sample or class distribution.

```python
from kindle import Model

# Initialize biases if classs histogram exists and assume that generally 3 objects are shown up each bounding boxes in 100 images.
model = Model("yolo_sample.yaml", verbose=True)
model.model[-1].initialize_biases(class_probability=YOUR_CLASS_HISTOGRAM, n_object_per_image=(3, 100))

# Initialize biases if class histogram does not exists and assuming each class has 60% probability chance to show.
model = Model("yolo_sample.yaml", verbose=True)
model.model[-1].initialize_biases(class_frequency=0.6, n_object_per_image=(3, 100))
```

!!! Note
	Initializing bias method is currently experimental and prone to change in near future.
