# Kindle - PyTorch no-code model builder

|`Documentation`|
|---------------|
|[Reference Document](https://limjk.ai/kindle)|

Kindle is an easy model build package for [PyTorch](https://pytorch.org). Building a deep learning model became so simple that almost all model can be made by copy and paste from other existing model codes. So why code? when we can simply build a model with yaml markup file.

Kindle builds a model with no code but yaml file which its method is inspired from [YOLOv5](https://github.com/ultralytics/yolov5).

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


# Planned features
* Custom module support
* Custom module with yaml support
* Use pre-trained model
* More modules!
