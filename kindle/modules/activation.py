"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import ast
from typing import Union

from torch import nn


class Activation:
    """Convert string activation name to the activation class."""

    def __init__(self, act_type: Union[str, None]) -> None:
        """Convert string activation name to the activation class.

        Args:
            act_type: Activation name.

        Returns:
            nn.Identity if {type} is None.
        """
        self.name = act_type
        self.args = [1] if self.name == "Softmax" else []

    def __call__(self) -> nn.Module:
        if self.name is None:
            return nn.Identity()
        if hasattr(nn, self.name):
            return getattr(nn, self.name)(*self.args)

        return ast.literal_eval(self.name)()
