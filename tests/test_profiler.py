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
    profiler = model.profile(n_run=100, batch_size=32, input_size=(224, 224))
    mac = profiler.get_macs(verbose=True)
    print(f"Total MACs: {mac:,.0f}")

    profiler.print_result(sort_by_rank=True)

    assert mac == 1616970
    assert profiler.result_array.sum() != 0


if __name__ == "__main__":
    test_profiler()
