# Kindle - Making a PyTorch model easier than ever!
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kindle)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.7.1-informational)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/JeiKeiLim/kindle/format-lint-test)
![PyPI](https://img.shields.io/pypi/v/kindle)
![LGTM Alerts](https://img.shields.io/lgtm/alerts/github/JeiKeiLim/kindle)

|`Documentation`|
|-------------|
|[![API reference](https://img.shields.io/badge/api-reference-informational)](https://limjk.ai/kindle/)|

Kindle is an easy model build package for [PyTorch](https://pytorch.org). Building a deep learning model became so simple that almost all model can be made by copy and paste from other existing model codes. So why code? when we can simply build a model with yaml markup file.

Kindle builds a model with yaml file which its method is inspired from [YOLOv5](https://github.com/ultralytics/yolov5).

# Contents
- [Installation](#installation)
  - [Install with pip](#install-with-pip)
  - [Install from source](#install-from-source)
  - [For contributors](#for-contributors)
- [AutoML with Kindle](#automl-with-kindle)
- [Usage](#usage)
- [Supported modules](#supported-modules)
- [Custom module support](#custom-module-support)
  - [Custom module with yaml](#custom-module-with-yaml)
  - [Custom module from source](#custom-module-from-source)
- [PretrainedModel support](#pretrained-model-support)
- [Model profiler](#model-profiler)
- [Test Time Augmentation](#test-time-augmentation)

# Installation
## Install with pip
**PyTorch** is required prior to install. Please visit [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install.

You can install `kindle` by pip.
```shell
$ pip install kindle
```

### Install `kindle` for PyTorch under 1.7.1 (not tested)
```shell
pip install kindle --no-deps
pip install tqdm ptflops timm tabulate einops
```

## Install from source
Please visit [Install from source wiki page](https://github.com/JeiKeiLim/kindle/wiki/Install-from-source)

## For contributors
Please visit [For contributors wiki page](https://github.com/JeiKeiLim/kindle/wiki/For-contributors)

# Usage
## Build a model

1. Make model yaml file
  - Example model https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


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

2. Build the model with **kindle**

```python
from kindle import Model

model = Model("model.yaml"), verbose=True)
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

## AutoML with Kindle
* [Kindle](https://github.com/JeiKeiLim/kindle) offers the easiest way to build your own deep learning architecture. Beyond building a model, AutoML became easier with [Kindle](https://github.com/JeiKeiLim/kindle) and [Optuna](https://optuna.org) or other optimization frameworks.
* For further information, please refer to [https://limjk.ai/kindle/usages/#automl-with-optuna](https://limjk.ai/kindle/usages/#automl-with-optuna)

# Supported modules
* Detailed documents can be found [https://limjk.ai/kindle/modules/](https://limjk.ai/kindle/modules/)

|Module|Components|Arguments|
|------|----------|---------|
|Conv|Conv -> BatchNorm -> Activation|[out_channels, kernel_size, stride, padding, groups, activation]|
|DWConv|DWConv -> BatchNorm -> Activation|[out_channels, kernel_size, stride, padding, activation]|
|Focus|Reshape x -> Conv -> Concat|[out_channels, kernel_size, stride, padding, activation]|
|Bottleneck|Expansion ConvBNAct -> ConvBNAct|[out_channels, shortcut, groups, expansion, activation]|
|BottleneckCSP|CSP Bottleneck|[out_channels, shortcut, groups, expansion, activation]
|C3|CSP Bottleneck with 3 Conv|[out_channels, shortcut, groups, expansion, activation]|
|MV2Block|MobileNet v2 block|[out_channels, stride, expand_ratio, activation]|
|AvgPool|Average pooling|[kernel_size, stride, padding]|
|MaxPool|Max pooling|[kernel_size, stride, padding]|
|GlobalAvgPool|Global Average Pooling|[]|
|SPP|Spatial Pyramid Pooling|[out_channels, [kernel_size1, kernel_size2, ...], activation]|
|SPPF|Spatial Pyramid Pooling - Fast|[out_channels, kernel_size, activation]|
|Flatten|Flatten|[]|
|Concat|Concatenation|[dimension]|
|Linear|Linear|[out_channels, activation]|
|Add|Add|[]|
|UpSample|UpSample|[]|
|Identity|Identity|[]|
|YamlModule|Custom module from yaml file|['yaml/file/path', arg0, arg1, ...]|
|nn.{module_name}|PyTorch torch.nn.* module|Please refer to [https://pytorch.org/docs/stable/nn.html](https://pytorch.org/docs/stable/nn.html)|
|Pretrained|timm.create_model|[model_name, use_feature_maps, features_only, pretrained]|
|PreTrainedFeatureMap|Bypass feature layer map from `Pretrained`|[feature_idx]|
|YOLOHead|YOLOv5 head module|[n_classes, anchors, out_xyxy]|
|MobileViTBlock|MobileVit Block(experimental)|[conv_channels, mlp_channels, depth, kernel_size, patch_size, dropout, activation]

* **nn.{module_name}** is currently experimental. This might change in the future release. Use with caution.
* For the supported model of **Pretrained** module, please refer to [https://rwightman.github.io/pytorch-image-models/results](https://rwightman.github.io/pytorch-image-models/results)


# Custom module support
## Custom module with yaml
* You can make your own custom module with yaml file. Please refer to [https://limjk.ai/kindle/tutorial/#2-design-custom-module-with-yaml](https://limjk.ai/kindle/tutorial/#2-design-custom-module-with-yaml) for further detail.


## Custom module from source code
* You can also make your own custom module from the source code. Please refer to https://limjk.ai/kindle/tutorial/#3-design-custom-module-from-source for further detail.

# Pretrained model support
* Pre-trained model from [timm](https://github.com/rwightman/pytorch-image-models) can be loaded in kindle yaml config file. Please refer to [https://limjk.ai/kindle/tutorial/#4-utilize-pretrained-model](https://limjk.ai/kindle/tutorial/#4-utilize-pretrained-model) for further detail.

# Model profiler
* Kindle provides model profiling option for each layers and calculating MACs.
* Please refer to https://limjk.ai/kindle/functionality/#1-model-profiling for further detail.

# Test Time Augmentation
* Kindle model supports TTA with easy usability. Just pass the model input and augmentation function.
* Please refer to https://limjk.ai/kindle/functionality/#3-test-time-augmentation for further detail.


# Recent changes
|Version|Description|Date|
|-------|-----------|----|
|0.4.15|Fix decomposed conv fuse|2021. 10. 25|
|0.4.14|Add MobileViTBlock module|2021. 10. 18|
|0.4.12|Add MV2Block module|2021. 10. 14|
|0.4.11|Add SPPF module in yolov5 v6.0|2021. 10. 13|
|0.4.10|Fix ONNX export padding issue.|2021. 10. 13|
|0.4.6|Add YOLOHead to choose coordinates format.|2021. 10. 09|
|0.4.5|Add C3 Module|2021. 10. 08|
|0.4.4|Fix YOLOHead module issue with anchor scaling|2021. 10. 08|
|0.4.2|Add YOLOModel, and ConvBN fusion, and Fix activation apply issue|2021. 09. 19|
|0.4.1|Add YOLOHead, SPP, BottleneckCSP, and Focus modules|2021. 09. 13|
|0.3.2|Fix PreTrained to work without PreTrainedFeatureMap|2021. 06. 03|
|0.3.1|Calculating MACs in profiler|2021. 05. 02|
|0.3.0|Add PreTrained support|2021. 04. 20|

# Planned features
* ~~Custom module support~~
* ~~Custom module with yaml support~~
* ~~Use pre-trained model~~
* Graphical model file generator
* Ensemble model
* More modules!
