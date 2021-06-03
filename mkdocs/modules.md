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
|Pretrained|timm.create_model|[model_name, use_feature_maps, features_only, pretrained]|
|PreTrainedFeatureMap|Bypass feature layer map from `Pretrained`|[feature_idx]|

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


## Pretrained
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|model_name|str||Please refer to [https://rwightman.github.io/pytorch-image-models/results](https://rwightman.github.io/pytorch-image-models/results) for supported models.|
|ues_feature_maps|bool|False|If True, return value of the module will be list of each feature maps. `List[torch.Tensor]` (features_only must be True in this case). Otherwise, returns last feature map.|
|features_only|bool|True|If True, skip classification layer and use feature layers only. `torch.Tensor`|
|pretrained|bool|True|use pretrained weight|


??? Note "In case you are just lazy as I am, here is list of pretrained model names.  (`timm==0.4.5`)"
    'adv_inception_v3', 'cspdarknet53', 'cspresnet50', 'cspresnext50', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenetblur121d', 'dla34', 'dla46_c', 'dla46x_c', 'dla60', 'dla60_res2net', 'dla60_res2next', 'dla60x', 'dla60x_c', 'dla102', 'dla102x', 'dla102x2', 'dla169', 'dm_nfnet_f0', 'dm_nfnet_f1', 'dm_nfnet_f2', 'dm_nfnet_f3', 'dm_nfnet_f4', 'dm_nfnet_f5', 'dm_nfnet_f6', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'ecaresnet26t', 'ecaresnet50d', 'ecaresnet50d_pruned', 'ecaresnet50t', 'ecaresnet101d', 'ecaresnet101d_pruned', 'ecaresnet269d', 'ecaresnetlight', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b1_pruned', 'efficientnet_b2', 'efficientnet_b2_pruned', 'efficientnet_b2a', 'efficientnet_b3', 'efficientnet_b3_pruned', 'efficientnet_b3a', 'efficientnet_em', 'efficientnet_es', 'efficientnet_lite0', 'ens_adv_inception_resnet_v2', 'ese_vovnet19b_dw', 'ese_vovnet39b', 'fbnetc_100', 'gernet_l', 'gernet_m', 'gernet_s', 'gluon_inception_v3', 'gluon_resnet18_v1b', 'gluon_resnet34_v1b', 'gluon_resnet50_v1b', 'gluon_resnet50_v1c', 'gluon_resnet50_v1d', 'gluon_resnet50_v1s', 'gluon_resnet101_v1b', 'gluon_resnet101_v1c', 'gluon_resnet101_v1d', 'gluon_resnet101_v1s', 'gluon_resnet152_v1b', 'gluon_resnet152_v1c', 'gluon_resnet152_v1d', 'gluon_resnet152_v1s', 'gluon_resnext50_32x4d', 'gluon_resnext101_32x4d', 'gluon_resnext101_64x4d', 'gluon_senet154', 'gluon_seresnext50_32x4d', 'gluon_seresnext101_32x4d', 'gluon_seresnext101_64x4d', 'gluon_xception65', 'hrnet_w18', 'hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w30', 'hrnet_w32', 'hrnet_w40', 'hrnet_w44', 'hrnet_w48', 'hrnet_w64', 'ig_resnext101_32x8d', 'ig_resnext101_32x16d', 'ig_resnext101_32x32d', 'ig_resnext101_32x48d', 'inception_resnet_v2', 'inception_v3', 'inception_v4', 'legacy_senet154', 'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50', 'legacy_seresnet101', 'legacy_seresnet152', 'legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d', 'legacy_seresnext101_32x4d', 'mixnet_l', 'mixnet_m', 'mixnet_s', 'mixnet_xl', 'mnasnet_100', 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv2_140', 'mobilenetv3_large_100', 'mobilenetv3_rw', 'nasnetalarge', 'nf_regnet_b1', 'nf_resnet50', 'nfnet_l0c', 'pnasnet5large', 'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320', 'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008', 'regnety_016', 'regnety_032', 'regnety_040', 'regnety_064', 'regnety_080', 'regnety_120', 'regnety_160', 'regnety_320', 'repvgg_a2', 'repvgg_b0', 'repvgg_b1', 'repvgg_b1g4', 'repvgg_b2', 'repvgg_b2g4', 'repvgg_b3', 'repvgg_b3g4', 'res2net50_14w_8s', 'res2net50_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s', 'res2net50_48w_2s', 'res2net101_26w_4s', 'res2next50', 'resnest14d', 'resnest26d', 'resnest50d', 'resnest50d_1s4x24d', 'resnest50d_4s2x40d', 'resnest101e', 'resnest200e', 'resnest269e', 'resnet18', 'resnet18d', 'resnet26', 'resnet26d', 'resnet34', 'resnet34d', 'resnet50', 'resnet50d', 'resnet101d', 'resnet152d', 'resnet200d', 'resnetblur50', 'resnetv2_50x1_bitm', 'resnetv2_50x1_bitm_in21k', 'resnetv2_50x3_bitm', 'resnetv2_50x3_bitm_in21k', 'resnetv2_101x1_bitm', 'resnetv2_101x1_bitm_in21k', 'resnetv2_101x3_bitm', 'resnetv2_101x3_bitm_in21k', 'resnetv2_152x2_bitm', 'resnetv2_152x2_bitm_in21k', 'resnetv2_152x4_bitm', 'resnetv2_152x4_bitm_in21k', 'resnext50_32x4d', 'resnext50d_32x4d', 'resnext101_32x8d', 'rexnet_100', 'rexnet_130', 'rexnet_150', 'rexnet_200', 'selecsls42b', 'selecsls60', 'selecsls60b', 'semnasnet_100', 'seresnet50', 'seresnet152d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext50_32x4d', 'skresnet18', 'skresnet34', 'skresnext50_32x4d', 'spnasnet_100', 'ssl_resnet18', 'ssl_resnet50', 'ssl_resnext50_32x4d', 'ssl_resnext101_32x4d', 'ssl_resnext101_32x8d', 'ssl_resnext101_32x16d', 'swsl_resnet18', 'swsl_resnet50', 'swsl_resnext50_32x4d', 'swsl_resnext101_32x4d', 'swsl_resnext101_32x8d', 'swsl_resnext101_32x16d', 'tf_efficientnet_b0', 'tf_efficientnet_b0_ap', 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1', 'tf_efficientnet_b1_ap', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b3', 'tf_efficientnet_b3_ap', 'tf_efficientnet_b3_ns', 'tf_efficientnet_b4', 'tf_efficientnet_b4_ap', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b5_ns', 'tf_efficientnet_b6', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b6_ns', 'tf_efficientnet_b7', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b7_ns', 'tf_efficientnet_b8', 'tf_efficientnet_b8_ap', 'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_el', 'tf_efficientnet_em', 'tf_efficientnet_es', 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'tf_inception_v3', 'tf_mixnet_l', 'tf_mixnet_m', 'tf_mixnet_s', 'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100', 'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100', 'tresnet_l', 'tresnet_l_448', 'tresnet_m', 'tresnet_m_448', 'tresnet_xl', 'tresnet_xl_448', 'tv_densenet121', 'tv_resnet34', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152', 'tv_resnext50_32x4d', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_base_patch16_224', 'vit_base_patch16_224_in21k', 'vit_base_patch16_384', 'vit_base_patch32_224_in21k', 'vit_base_patch32_384', 'vit_base_resnet50_224_in21k', 'vit_base_resnet50_384', 'vit_deit_base_distilled_patch16_224', 'vit_deit_base_distilled_patch16_384', 'vit_deit_base_patch16_224', 'vit_deit_base_patch16_384', 'vit_deit_small_distilled_patch16_224', 'vit_deit_small_patch16_224', 'vit_deit_tiny_distilled_patch16_224', 'vit_deit_tiny_patch16_224', 'vit_large_patch16_224', 'vit_large_patch16_224_in21k', 'vit_large_patch16_384', 'vit_large_patch32_224_in21k', 'vit_large_patch32_384', 'vit_small_patch16_224', 'wide_resnet50_2', 'wide_resnet101_2', 'xception', 'xception41', 'xception65', 'xception71'


## PretrainedFeatureMap
|Argument name|Type|Default value|Description|
|-------------|----|-------------|-----------|
|feature_idx|int|-1|Index of the feature maps|
