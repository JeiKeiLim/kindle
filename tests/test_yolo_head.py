"""YOLO Head module test.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os

import torch

from kindle.model import Model


def test_yolo_head():
    model = Model(
        os.path.join("tests", "test_configs", "yolo_sample.yaml"), verbose=True
    )

    model.eval()
    in_tensor = torch.rand((1, 3, 480, 380))
    out_tensor = model(in_tensor)

    assert out_tensor[0].shape == (1, 11340, 15)
    assert out_tensor[1][0].shape == (1, 3, 60, 48, 15)
    assert out_tensor[1][1].shape == (1, 3, 30, 24, 15)
    assert out_tensor[1][2].shape == (1, 3, 15, 12, 15)


if __name__ == "__main__":
    test_yolo_head()
