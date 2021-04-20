"""Pretrained module.

Pre-trained model downloaded from https://github.com/rwightman/pytorch-image-models
Please refer to https://rwightman.github.io/pytorch-image-models/ for the supported model overview.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
from typing import List, Union

import timm
import torch
from torch import nn


class Pretrained(nn.Module):
    """Pretrained module.

    Please refer to https://rwightman.github.io/pytorch-image-models/ for the supported
    model overview.
    """

    def __init__(
        self, model_name: str, features_only: bool = False, pretrained: bool = True
    ) -> None:
        """Initialize Pretrained instance.

        Args:
            model_name: name of the model from timm.list_models(pretrained=True)
            features_only: if True, return value of the module will be list of each feature maps.
            pretrained: use pretrained weight.
        """
        super().__init__()

        self.module = timm.create_model(
            model_name=model_name, pretrained=pretrained, features_only=features_only
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pretrained network.

        Returns:
            if self.features_only is True, List of torch.Tensor will be returned.
            which each array contains corresponding feature map of pooling layer.
            Otherwise, torch.Tensor will be returned which is output of the last layer.
        """
        return self.module(x)


class PretrainedFeatureMap(nn.Module):
    """Placeholder of the output of Pretrained module when features_only = True"""

    def __init__(self, feature_idx: int = -1) -> None:
        """Initialize PretrainedFeatureMap.

        Args:
            feature_idx: index of the feature maps.
        """
        super().__init__()
        self.feature_idx = feature_idx

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Bypass input of x[self.feature_idx]."""
        return x[self.feature_idx]
