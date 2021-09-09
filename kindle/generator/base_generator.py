"""Base Module Generator.

This module is responsible for GeneratorAbstract and ModuleGenerator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch import nn

from kindle.utils.torch_utils import make_divisible


class GeneratorAbstract(ABC):
    """Abstract Module Generator."""

    CHANNEL_DIVISOR: int = 8

    def __init__(
        self,
        *args,
        keyword_args: Optional[Dict[str, Any]] = None,
        from_idx: Union[int, List[int]] = -1,
        in_channels: Tuple[int] = (0,),
        width_multiply: float = 1.0,
    ):
        """Initialize module generator.

        Args:
            *args: Module arguments
            from_idx: Module input index
            in_channels: Number of input channel
            width_multiply: Channel width multiply
        """
        self.args = tuple(args)
        self.keyword_args = keyword_args
        self.from_idx = from_idx
        self.in_channels = in_channels
        self.width_multiply = width_multiply

    @property
    def name(self) -> str:
        """Module name."""
        return self.__class__.__name__.replace("Generator", "")

    def _get_module(self, module: Union[nn.Module, List[nn.Module]]) -> nn.Module:
        """Get module from __call__ function."""
        if isinstance(module, list):
            module = nn.Sequential(*module)

        # error: Incompatible types in assignment (expression has type "Union[Tensor, Module, int]",
        # variable has type "Union[Tensor, Module]")
        # error: List comprehension has incompatible type List[int];
        # expected List[Union[Tensor, Module]]
        module.n_params = sum([x.numel() for x in module.parameters()])  # type: ignore
        # error: Cannot assign to a method
        module.name = self.name  # type: ignore

        return module

    @classmethod
    def _get_divisible_channel(cls, n_channel: int) -> int:
        """Get divisible channel by default divisor.

        Args:
            n_channel: number of channel.

        Returns:
            Ex) given {n_channel} is 52 and {GeneratorAbstract.CHANNEL_DIVISOR} is 8.,
                return channel is 56 since ceil(52/8) = 7 and 7*8 = 56
        """
        return make_divisible(n_channel, divisor=cls.CHANNEL_DIVISOR)

    def _get_kwargs(
        self, module_type: Any, args: Union[Tuple[Any, ...], List[Any]]
    ) -> Dict[str, Any]:
        """Get keyword argument with default argument values.

        Args:
            module_type: module class. Ex) nn.Conv
            args: ordered argument values

        Returns:
            converted keyword argument dictionary
            from the ordered argument values.
        """
        arg_spec = inspect.getfullargspec(module_type)
        kwarg_names = arg_spec.args
        if "self" in kwarg_names:
            kwarg_names.remove("self")

        if arg_spec.defaults is None:  # No keyword arguments
            arg_values = [
                None,
            ] * len(kwarg_names)
        else:
            arg_values = [
                None,
            ] * (len(kwarg_names) - len(arg_spec.defaults))
            arg_values += list(arg_spec.defaults)
        arg_values[: len(args)] = args

        kwarg_dict = dict(zip(kwarg_names, arg_values))
        if self.keyword_args is not None:
            for key, val in self.keyword_args.items():
                kwarg_dict[key] = val

        return kwarg_dict

    @property
    @abstractmethod
    def out_channel(self) -> Union[int, List[int]]:
        """Out channel of the module."""

    @property
    @abstractmethod
    def in_channel(self) -> Union[int, List[int]]:
        """In channel of the module."""

    @abstractmethod
    def compute_out_shape(
        self, size: np.ndarray, repeat: int = 1
    ) -> Union[List[int], List[List[int]]]:
        """Compute output shape when {size} is given.

        Args: input size to compute output shape.
        """

    @abstractmethod
    def __call__(self, repeat: int = 1) -> nn.Module:
        """Returns nn.Module component."""

    @property
    @abstractmethod
    def kwargs(self) -> Dict[str, Any]:
        """Generate keyword argument of the module."""


class ModuleGenerator:
    """Module generator class."""

    def __init__(
        self, module_name: str, custom_module_paths: Optional[Union[str, List]] = None
    ):
        """Generate module based on the {module_name}

        Args:
            module_name: {module_name}Generator class must have been implemented.
            custom_module_paths: paths to find custom module generators.
                Default location to find module generator is 'kindle.generator'.
                If {custom_module_paths} is provided, ModuleGenerator will expand its search area.
        """
        self.module_name = module_name
        self.generator_paths = ["kindle.generator"]
        if custom_module_paths is not None:
            if isinstance(custom_module_paths, str):
                paths = [custom_module_paths]
            else:
                paths = custom_module_paths
            self.generator_paths += paths

    def __call__(self, *args, keyword_args: Optional[Dict[str, Any]] = None, **kwargs):
        if self.module_name.startswith("nn."):
            generator_name = "TorchNNModuleGenerator"
            kwargs["module_name"] = self.module_name
        else:
            generator_name = f"{self.module_name}Generator"

        kwargs["keyword_args"] = keyword_args

        for path in self.generator_paths:
            if hasattr(__import__(path, fromlist=[""]), generator_name):
                return getattr(__import__(path, fromlist=[""]), generator_name)(
                    *args, **kwargs
                )

        raise Exception(f"{generator_name} can not be found.")
