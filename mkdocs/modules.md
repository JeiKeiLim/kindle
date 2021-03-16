# Kindle Modules
## Supported Modules Summary

|Module|Components|Arguments|
|-|-|-|
|Conv|Conv -> BatchNorm -> Activation|[channel, kernel size, stride, padding, groups, activation]|
|DWConv|DWConv -> BatchNorm -> Activation|[channel, kernel_size, stride, padding, activation]|
|Bottleneck|Expansion ConvBNAct -> ConvBNAct|[channel, shortcut, groups, expansion, activation]
|AvgPool|Average pooling|[kernel_size, stride, padding]|
|MaxPool|Max pooling|[kernel_size, stride, padding]|
|GlobalAvgPool|Global Average Pooling|[]|
|Flatten|Flatten|[]|
|Concat|Concatenation|[dimension]|
|Linear|Linear|[channel, activation]|
|Add|Add|[]|
|UpSample|UpSample|[]|
|Identity|Identity|[]|

