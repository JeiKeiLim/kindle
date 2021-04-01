"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import os

import torch

from kindle import Model


def test_profiler():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = Model(os.path.join("tests", "test_configs", "example.yaml"), verbose=True)
    model.to(device)
    profiler = model.profile(n_run=1000, batch_size=32)

    profiler.print_result(sort_by_rank=True)

    assert profiler.result_array.sum() != 0


if __name__ == "__main__":
    test_profiler()
