"""Common utility functions.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import ast
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


def model_info(model, verbose=False):
    """Print out model info."""
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )  # number gradients
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s %10s"
            % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
        )
        for i, (name, param) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (
                    i,
                    name,
                    param.requires_grad,
                    param.numel(),
                    list(param.shape),
                    param.mean(),
                    param.std(),
                )
            )

    print(
        f"Model Summary: {len(list(model.modules()))} layers, "
        f"{n_p:,d} parameters, {n_g:,d} gradients"
    )


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


class Activation:
    """Convert string activation name to the activation class."""

    def __init__(self, act_type: Union[str, None]) -> None:
        """Convert string activation name to the activation class.

        Args:
            act_type: Activation name.

        Returns:
            nn.Identity if {type} is None.
        """
        self.type = act_type
        self.args = [1] if self.type == "Softmax" else []

    def __call__(self) -> nn.Module:
        if self.type is None:
            return nn.Identity()
        if hasattr(nn, self.type):
            return getattr(nn, self.type)(*self.args)

        return ast.literal_eval(self.type)()


def count_model_params(
    model: torch.nn.Module,
) -> int:
    """Count model's parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
