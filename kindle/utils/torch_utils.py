"""Common utility functions.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import math
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler


def split_dataset_index(
    n_data: int, split_ratio: float = 0.1
) -> Tuple[SubsetRandomSampler, SubsetRandomSampler]:
    """Split dataset indices with split_ratio.

    Args:
        n_data: number of total data
        split_ratio: split ratio (0.0 ~ 1.0)

    Returns:
        SubsetRandomSampler ({split_ratio} ~ 1.0)
        SubsetRandomSampler (0 ~ {split_ratio})
    """
    indices = np.arange(n_data)
    split = int(split_ratio * indices.shape[0])

    train_idx = indices[split:]
    valid_idx = indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler


def make_divisible(n_channel: Union[int, float], divisor: int = 8) -> int:
    """Convert {n_channel} to divisible by {divisor}

    Args:
        n_channel: number of channels.
        divisor: divisor to be used.

    Returns:
        Ex) n_channel=22, divisor=8
            ceil(22/8) * 8 = 24
    """
    return int(math.ceil(n_channel / divisor) * divisor)


def autopad(
    kernel_size: Union[int, List[int]], padding: Union[int, None] = None
) -> Union[int, List[int]]:
    """Auto padding calculation for pad='same' in TensorFlow."""
    # Pad to 'same'
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    return padding or [x // 2 for x in kernel_size]


def count_model_params(
    model: torch.nn.Module,
) -> int:
    """Count model's parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fuse_conv_and_batch_norm(conv: nn.Conv2d, batch_norm: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse convolution and batchnorm layers.

    https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    Args:
        conv: convolution module.
        batch_norm: Batch normalization module directly connected to the conv module.

    Return:
        Fused conv with batch norm.
    """

    fused_conv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,  # type: ignore
            stride=conv.stride,  # type: ignore
            padding=conv.padding,  # type: ignore
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Fusing weight
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_batch_norm = torch.diag(
        batch_norm.weight.div(
            torch.sqrt(batch_norm.eps + batch_norm.running_var)  # type: ignore
        )
    )
    fused_conv.weight.copy_(
        torch.mm(w_batch_norm, w_conv).view(fused_conv.weight.size())
    )

    # Fusing bias
    if conv.bias is None:
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device)
    else:
        b_conv = conv.bias

    b_batch_norm = batch_norm.bias - batch_norm.weight.mul(
        batch_norm.running_mean  # type: ignore
    ).div(
        torch.sqrt(batch_norm.running_var + batch_norm.eps)  # type: ignore
    )
    fused_conv.bias.copy_(  # type: ignore
        torch.mm(w_batch_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_batch_norm
    )

    return fused_conv
