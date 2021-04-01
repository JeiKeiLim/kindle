"""Kindle Model parser and model.

This module parses model configuration yaml file
and generates PyTorch model accordingly.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from kindle.generator.base_generator import GeneratorAbstract, ModuleGenerator
from kindle.generator.flatten import FlattenGenerator


def split_str_line(msg: str, line_limit: int = 30) -> List[str]:
    """Split string with a maximum length of the line.

    Ex) split_str_line("hello world", line_limit=5)
        will return ["hello", " worl", "d"]

    Args:
        msg: message to split.
        line_limit: limit length of the line.

    Returns:
        list of the split message.
    """
    msg_list = []
    for j in range(0, len(msg), line_limit):
        end_idx = j + line_limit
        msg_list.append(msg[j:end_idx])

    return msg_list


class ModelProfiler:
    """Model time consumption profiler."""

    def __init__(
        self,
        model: "Model",
        n_run: int = 100,
        input_size: Tuple[int, int] = (128, 128),
        batch_size: int = 1,
    ) -> None:
        """Initialize model profiler.

        Args:
            model: kindle.Model instance for profiling.
            n_run: number of inference to run.
            input_size: input size to test.
                    If model config contains "input_size", this will be ignored.
            batch_size: batch size to test.
        """
        self.model = model
        self.n_run = n_run
        self.input_size: Tuple[int, ...] = (
            batch_size,
            self.model.model_parser.cfg["input_channel"],  # type: ignore
        )

        if "input_size" in self.model.model_parser.cfg:
            self.input_size += tuple(self.model.model_parser.cfg["input_size"])  # type: ignore
        else:
            self.input_size += input_size

        self.profile_result = [
            {"name": module.name, "time": np.zeros(self.n_run)}
            for module in self.model.model  # type: ignore
        ]
        self.n_running = 0

    @property
    def result_array(self) -> np.ndarray:
        """Profiling time result array."""
        return np.array([r["time"] for r in self.profile_result])

    @property
    def mean_run_time(self) -> float:
        """Mean time of the inference."""
        return float(self.result_array.sum(axis=0).mean())

    @property
    def std_run_time(self) -> float:
        """Standard deviation time of the inference."""
        return float(self.result_array.sum(axis=0).std())

    @property
    def total_run_time(self) -> float:
        """Total running time."""
        return float(self.result_array.sum())

    @property
    def sorted_index(self) -> np.ndarray:
        """Sorted indices by time consumption of the modules.

        Ex) First index element is the slowest module index.
        """
        return np.argsort(self.result_array.sum(axis=1))[::-1]

    @property
    def running_ratio(self) -> np.ndarray:
        """Running time consumption ratio of the modules."""
        result_array = self.result_array
        result = result_array.sum(axis=1) / result_array.sum()

        if isinstance(result, np.ndarray):
            result_out = result
        else:
            result_out = np.array(result)

        return result_out

    @torch.no_grad()
    def run(self, verbose: bool = True) -> None:
        """Run profiling.

        Args:
            verbose: print profiling result at the end.
        """
        self.profile_result = [
            {"name": module.name, "time": np.zeros(self.n_run)}
            for module in self.model.model  # type: ignore
        ]

        model_input = torch.zeros(self.input_size).to(
            list(self.model.parameters())[0].device
        )

        for run_idx in tqdm(range(self.n_run), desc="Profiling ..."):
            self.n_running = run_idx
            self.model.forward_once(model_input, profile_func=self._profile_func)

        if verbose:
            self.print_result()

    def print_result(  # pylint: disable=too-many-locals
        self, sort_by_rank: bool = False
    ) -> None:
        """Print profiling result.

        Args:
            sort_by_rank: print sorted by time consumption rank.
        """

        print(f"Profiling result by {self.n_run:,} times running.", end="")
        if sort_by_rank:
            print(" Sorted by time consumption.")
        else:
            print(" Sorted by running order.")

        msg_title = (
            f"{'idx':>4} | {'Name':>20} | {'Time(Mean)':>10} | "
            f"{'Time(Std)':>10} | {'Time(Total)':>10} | "
            f"{'Rank'} | {'Ratio':>7} | {'Params':>13} |"
        )

        print("-" * len(msg_title))
        print(msg_title)
        print("-" * len(msg_title))

        slow_index = {idx: i for i, idx in enumerate(self.sorted_index)}
        running_ratio = self.running_ratio

        log_msgs = []

        for i, result in enumerate(self.profile_result):
            name = result["name"]
            time_mean = result["time"].mean()
            time_std = result["time"].std()
            time_sum = result["time"].sum()

            time_mean, t_unit = self._time_convert(time_mean)
            time_std, t_unit_std = self._time_convert(time_std)
            time_sum, t_unit_sum = self._time_convert(time_sum)
            log_msg = (
                f"{i:4d} | {name:>20} | {time_mean:7.2f} {t_unit:<2} | "  # type: ignore
                f"{time_std:7.2f} {t_unit_std:<2} | "
                f"{time_sum:8.2f} {t_unit_sum:<2} | "
                f"{slow_index[i]+1:4d} | {running_ratio[i]*100:6.2f}% | "
                f"{self.model.model[i].n_params:13,d} |"
            )
            log_msgs.append(log_msg)

        if sort_by_rank:
            loop: Union[np.ndarray, range] = self.sorted_index
        else:
            loop = range(len(log_msgs))

        for i in loop:
            print(log_msgs[i])

        total_time_mean, t_unit_mean = self._time_convert(self.mean_run_time)
        total_time_std, t_unit_std = self._time_convert(self.std_run_time)
        total_time_sum, t_unit_sum = self._time_convert(self.total_run_time)
        print("-" * len(msg_title))
        print(
            f"Running time\n"
            f" - Total : {total_time_sum:8.2f} {t_unit_sum:<2}\n"
            f" - {'Mean':>5} : {total_time_mean:8.2f} {t_unit_mean:<2}\n"
            f" - {'STD':>5} : {total_time_std:8.2f} {t_unit_std:<2}\n"
        )

    @classmethod
    def _time_convert(cls, x: float) -> Tuple[float, str]:
        """Convert time units.

        Args:
            x: time to be converted (seconds).

        Returns:
            converted time value and its unit
            (seconds, milliseconds, microseconds, nanoseconds, picoseconds, femtoseconds)
        """
        time_units = ["s", "ms", "μs", "ns", "ps", "fs"]
        i = 0
        for i in range(len(time_units)):
            if x > 1.0:
                break
            x *= 1000

        return x, time_units[i]

    def _profile_func(self, module: Callable, x: torch.Tensor, i: int) -> torch.Tensor:
        """Profile callback function for kindle.Model.forward."""
        start_time = time.monotonic()
        y = module(x)
        time_took = time.monotonic() - start_time
        self.profile_result[i]["time"][self.n_running] = time_took

        return y


class Model(nn.Module):
    """PyTorch model class."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]],
        verbose: bool = False,
    ) -> None:
        """Parse model from the model config file.

        Args:
            cfg: yaml file path or dictionary type of the model.
            verbose: print the model parsing information.
        """
        super().__init__()
        self.model_parser = ModelParser(cfg=cfg, verbose=verbose)
        self.model = self.model_parser.model
        self.output_save = self.model_parser.output_save

    def forward(
        self,
        x: Union[torch.Tensor, np.ndarray],
        augment_func: Optional[Union[List[Callable], Callable]] = None,
        n_augment: int = 3,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward the model.

        For the time being, this method will only call self.forward_once. Later, we plan
        to add Test Time Augment.

        Args:
            x: model input. When augment_func is given,
                {X} can be either torch.Tensor or np.ndarray.
                However, output of augmentation function must emit torch.Tensor.

            augment_func: augmentation function.
                        If list of functions is given, TTA will be applied by original input {X} and
                        {X} with each augmentation functions.
                        Output shape will be (len({AUGMENT_FUNC})+1, batch_size, output_size).

                        If single function is given, TTA will be applied by original input {X} and
                        {X} {AUGMENT_FUNC} applied by {N_AUGMENT} times.
                        Output shape will be ({N_AUGMENT}+1, batch_size, output_size)

            n_augment: number of augmentations to apply on TTA augmentation.
                       Only applies when single augmentation function is given.

        Returns:
            Model output.
        """
        if augment_func is not None:
            if isinstance(augment_func, list):
                funcs = augment_func
            else:
                funcs = [augment_func for _ in range(n_augment)]

            y_list = [self.forward_once(aug_func(x)) for aug_func in funcs]
            if isinstance(x, torch.Tensor):
                y_list.append(self.forward_once(x))

            y = torch.stack(y_list)
        else:
            assert isinstance(x, torch.Tensor), "Input must be torch tensor for non-TTA"
            y = self.forward_once(x)

        return y

    def profile(self, verbose: bool = True, **kwargs) -> ModelProfiler:
        """Run model profiler.

        Args:
            verbose: print profiling result.

        Returns:
            ModelProfiler instance which contains profiling result.
        """
        profiler = ModelProfiler(self, **kwargs)
        profiler.run(verbose=verbose)

        return profiler

    def forward_once(
        self, x: torch.Tensor, profile_func: Optional[Callable] = None
    ) -> torch.Tensor:
        """Forward one time only."""
        y: List[Union[torch.Tensor, None]] = []

        for i, module in enumerate(self.model):  # type: ignore
            if module.from_idx != -1:
                x = (
                    y[module.from_idx]  # type: ignore
                    if isinstance(module.from_idx, int)
                    else [x if j == -1 else y[j] for j in module.from_idx]
                )
            if profile_func is not None:
                x = profile_func(module, x, i)
            else:
                x = module(x)
            y.append(x if module.module_idx in self.output_save else None)

        return x


class ModelParser:
    """Generate PyTorch model from the model yaml file."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]] = "./model_configs/show_case.yaml",
        verbose: bool = False,
    ) -> None:
        """Generate PyTorch model from the model yaml file.

        Args:
            cfg: model config file or dict values read from the model config file.
            verbose: print the parsed model information.
        """

        self.verbose = verbose
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            with open(cfg) as f:
                self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.input_size = None

        if (
            self.cfg
            and "input_size" in self.cfg
            and len(self.cfg["input_size"]) == 2  # type: ignore
        ):
            self.input_size = self.cfg["input_size"]

        self.custom_module_paths: Optional[Union[List[str], str]]
        if "custom_module_paths" in self.cfg:
            self.custom_module_paths = self.cfg["custom_module_paths"]  # type: ignore
        else:
            self.custom_module_paths = None

        self.in_channel = self.cfg["input_channel"]

        self.depth_multiply = self.cfg["depth_multiple"]
        self.width_multiply = self.cfg["width_multiple"]

        self.backbone_cfg: List[List] = self.cfg["backbone"]  # type: ignore
        if "head" in self.cfg:
            self.head_cfg: Optional[List[List]] = self.cfg["head"]  # type: ignore
        else:
            self.head_cfg = None

        if "channel_divisor" in self.cfg:
            self.channel_divisor: int = self.cfg["channel_divisor"]  # type: ignore
        else:
            self.channel_divisor = GeneratorAbstract.CHANNEL_DIVISOR

        self.model, self.output_save = self._parse_model()

    def log(self, msg: str):
        """Log."""
        if self.verbose:
            print(msg)

    def _log_parse(  # pylint: disable=too-many-locals
        self,
        info: Optional[Tuple[int, int, int]] = None,
        module: Optional[nn.Module] = None,
        module_generator: Optional[GeneratorAbstract] = None,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        in_size: Optional[Union[np.ndarray, List]] = None,
        out_size: Optional[List[int]] = None,
        head: bool = False,
    ) -> None:
        """Print log message for parsing modules.

        Args:
            info: (i, idx, repeat) Current parsing information.
            module: Parsed module.
            module_generator: Module generator used to generate module.
            args: Arguments of the module.
            in_size: Input size of the module.
                    Only required when {self.input_size} is not None.
            out_size: Output size of the module
                    Only required when {self.input_size} is not None.
            head: Print head(Column names) message only.
        """
        if head:
            log = [
                f"{'idx':>3} | {'from':>10} | {'n':>3} | {'params':>8} |"
                f" {'module':>15} | {'arguments':>35} |"
                f" {'in_channel':>10} | {'out_channel':>11} |"
            ]
            if self.input_size is not None:
                log[0] += f" {'in shape':>15} | {'out shape':>15} |"
            log.append(f"{len(log[0]) * '-'}")
        else:
            assert (
                info is not None
                and module is not None
                and module_generator is not None
                and args is not None
            ), "info, module, and module_generator must be provided to generate log string."
            i, idx, repeat = info

            args = args.copy()
            if module.name == "YamlModule":
                args[0] = args[0].split(os.sep)[-1].split(".")[0]

            args_str = str(args)
            if kwargs is not None:
                args_str += ", "
                for key, val in kwargs.items():
                    args_str += f"{key}: {val}, "

                args_str = args_str[:-2]

            args_str_list = split_str_line(args_str, line_limit=35)

            log = [
                f"{i:3d} | {str(idx):>10} | {repeat:3d} |"
                f" {module.n_params:8,d} | {module.name:>15} | {args_str_list[0]:>35} |"
                f" {module_generator.in_channel:>10} | {module_generator.out_channel:>11} |"
            ]
            for j in range(1, len(args_str_list)):
                log.append(
                    f"{'':>3} | {'':>10} | {'':>3} |"
                    f" {'':>8} | {'':>15} | {args_str_list[j]:>35} |"
                    f" {'':>10} | {'':>11} |"
                )

            if (
                self.input_size is not None
                and in_size is not None
                and out_size is not None
            ):
                in_size_str = str(in_size).replace("\n", ",")
                in_size_str_list = split_str_line(in_size_str, line_limit=15)

                log[0] += f" {in_size_str_list[0]:>15} | {str(out_size):>15} |"

                for j in range(1, len(in_size_str_list)):
                    append_msg = f" {in_size_str_list[j]:>15} | {'':>15} |"
                    if j < len(log):
                        log[j] += append_msg
                    else:
                        append_msg = (
                            f"{'':>3} | {'':>10} | {'':>3} |"
                            f" {'':>8} | {'':>15} | {'':>35} |"
                            f" {'':>10} | {'':>11} |"
                        ) + append_msg

                        log.append(append_msg)

        self.log("\n".join(log))

    def _parse_model(  # pylint: disable=too-many-locals
        self,
    ) -> Tuple[nn.Sequential, List[int]]:
        """Parse model."""
        in_channels: List[int] = []
        in_sizes: List[int] = []
        layers: List[nn.Module] = []
        output_save: List[int] = []
        self._log_parse(head=True)

        if self.head_cfg is not None:
            model_cfg = self.backbone_cfg + self.head_cfg
        else:
            model_cfg = self.backbone_cfg

        channel_divisor = GeneratorAbstract.CHANNEL_DIVISOR
        GeneratorAbstract.CHANNEL_DIVISOR = self.channel_divisor
        for i, module_cfg in enumerate(model_cfg):  # type: ignore
            if len(module_cfg) > 4:
                idx, repeat, module, args, kwargs = module_cfg
            else:
                idx, repeat, module, args = module_cfg
                kwargs = None

            if i >= len(self.backbone_cfg):
                GeneratorAbstract.CHANNEL_DIVISOR = 1
                width_multiply = 1.0
                depth_multiply = 1.0
            else:
                width_multiply = float(self.width_multiply)  # type: ignore
                depth_multiply = float(self.depth_multiply)  # type: ignore

            module_generator = ModuleGenerator(
                module, custom_module_paths=self.custom_module_paths
            )(
                *args,
                keyword_args=kwargs,
                from_idx=idx,
                in_channels=tuple(in_channels) if i > 0 else (self.in_channel,),  # type: ignore
                width_multiply=width_multiply,
            )
            repeat = max(round(repeat * depth_multiply), 1) if repeat > 1 else repeat

            if isinstance(module_generator, FlattenGenerator):
                if self.input_size is not None:
                    module_generator.in_shape = in_sizes[idx]  # type: ignore
                else:
                    module_generator.in_shape = [in_channels[idx], 1, 1]  # type: ignore

            module = module_generator(repeat=repeat)
            module.module_idx, module.from_idx = i, idx

            if self.input_size is not None:
                in_size = (
                    np.array(in_sizes, dtype=np.object_)[idx]
                    if i > 0
                    else [self.in_channel] + self.input_size
                )
                out_size = module_generator.compute_out_shape(in_size, repeat=repeat)
                in_sizes.append(out_size)
            else:
                in_size, out_size = None, None

            in_channels.append(module_generator.out_channel)
            layers.append(module)

            output_save.extend(
                [x % i for x in ([idx] if isinstance(idx, int) else idx) if x != -1]
            )

            self._log_parse(
                info=(i, idx, repeat),
                module=module,
                module_generator=module_generator,
                args=args,
                kwargs=kwargs,
                in_size=in_size,
                out_size=out_size,
            )

        GeneratorAbstract.CHANNEL_DIVISOR = channel_divisor
        parsed_model = nn.Sequential(*layers)
        n_param = sum([x.numel() for x in parsed_model.parameters()])
        n_grad = sum([x.numel() for x in parsed_model.parameters() if x.requires_grad])

        self.log(
            f"Model Summary: {len(list(parsed_model.modules())):,d} "
            f"layers, {n_param:,d} parameters, {n_grad:,d} gradients"
            f"\n"
        )

        return parsed_model, output_save
