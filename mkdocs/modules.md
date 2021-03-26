# Kindle Modules
## Supported Modules Summary

|Module|Components|Arguments|
|------|----------|---------|
|Conv|Conv -> BatchNorm -> Activation|[out_channels, kernel_size, stride, padding, groups, activation]|
|DWConv|DWConv -> BatchNorm -> Activation|[out_channels, kernel_size, stride, padding, activation]|
|Bottleneck|Expansion ConvBNAct -> ConvBNAct|[out_channels, shortcut, groups, expansion, activation]
|AvgPool|Average pooling|[kernel_size, stride, padding]|
|MaxPool|Max pooling|[kernel_size, stride, padding]|
|GlobalAvgPool|Global Average Pooling|[]|
|Flatten|Flatten|[]|
|Concat|Concatenation|[dimension]|
|Linear|Linear|[out_channels, activation]|
|Add|Add|[]|
|UpSample|UpSample|[]|
|Identity|Identity|[]|
|YamlModule|Custom module from yaml file|['yaml/file/path', arg0, arg1, ...]|
|nn.{module_name}|PyTorch torch.nn.* module|Please refer to [https://pytorch.org/docs/stable/nn.html](https://pytorch.org/docs/stable/nn.html)|

!!! Note
    nn.{module_name} is currently experimental. This might change in the future release. Use with caution.

## Conv
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|out_channels|int| |Conv channels|
|kernel_size|int| |(n, n) kernel size|
|stride|int|1|Conv stride|
|padding|int|None|Conv padding. If None, auto-padding will be applied which generates same width and height of the input|
|groups|int|1|Group convolution size. If 1, no group convolution|
|activation|str or None|"ReLU"|If None, no activation(Identity) is applied.|

* Please refer to [https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) for further detail.

## DWConv
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|out_channels|int| |Conv channels|
|kernel_size|int| |(n, n) kernel size|
|stride|int|1|Conv stride|
|padding|int|None|Conv padding. If None, auto-padding will be applied which generates same width and height of the input|
|activation|str or None|"ReLU"|If None, no activation(Identity) is applied.|

- DWConv is identical to Conv but with force grouped convolution.

## Bottleneck
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|out_channels|int| |Conv channels|
|shortcut|bool|True|Use shortcut. Only applied when in_channels and out_channels are same.
|groups|int|1|Group convolution size. If 1, no group convolution|
|expansion|int|0.5|Expansion(squeeze) ratio.|
|activation|str or None|"ReLU"|If None, no activation(Identity) is applied.|

## AvgPool
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|kernel_size|int|||
|stride|int or None|None||
|padding|int|0||
|ceil_mode|bool|False||
|count_include_pad|bool|True||
|divisor_override|bool or None|None|

* Please refer to [https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html) for further detail.

## MaxPool 
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|kernel_size|int|||
|stride|int or None|None||
|padding|int|0||
|dilation|int|1||
|return_indices|bool|False||
|ceil_mode|bool|False||

* Please refer to [https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) for further detail.

## Flatten
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|start_dim|int|1||
|end_dim|int|-1||

* Please refer to [https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) for further detail.

## Concat 
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|dimension|int|1||

## Linear
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|out_channels|int|||
|activation|str or None|None||

## UpSample 
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|size|int or None|None||
|scale_factor|int or None|2||
|mode|str|nearest||
|align_corners|bool or None|None||

* Please refer to [https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html) for further detail.

## YamlModule
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|verbose|bool|False||

* yaml file path and argument configured in yaml module can not be passed through keyword argument.
