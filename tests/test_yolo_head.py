"""YOLO Head module test.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os
import random

import torch

from kindle.model import Model


def test_yolo_head_initialize_bias_class_probability(n_test: int = 5):
    for n in range(n_test):
        model = Model(
            os.path.join("tests", "test_configs", "yolo_sample.yaml"), verbose=n == 0
        )
        class_probability = random.random()
        n_object_per_image = (random.randint(4, 16), random.randint(320, 1024))

        model.model[-1].initialize_biases(
            class_probability=class_probability, n_object_per_image=n_object_per_image
        )

        class_prob = torch.log(
            torch.tensor(class_probability / (model.model[-1].n_classes - 0.99))
        )

        for i in range(model.model[-1].n_layers):
            class_bias = model.model[-1].conv[i].bias.view(3, -1)[:, 5:].mean(0)
            assert torch.isclose(class_bias, class_prob, rtol=0.1).sum() == 10

        for i in range(model.model[-1].n_layers):
            obj_bias = model.model[-1].conv[i].bias.view(3, -1)[:, 4].mean()
            obj_log = torch.log(
                n_object_per_image[0]
                / (n_object_per_image[1] / model.model[-1].stride[i]) ** 2
            )
            assert torch.isclose(obj_bias, obj_log, rtol=0.1)


def test_yolo_head_initialize_bias(n_test: int = 5):
    for n in range(n_test):
        model = Model(
            os.path.join("tests", "test_configs", "yolo_sample.yaml"), verbose=n == 0
        )
        class_frequency = torch.randint(100, 5000, (10,))
        n_object_per_image = (random.randint(4, 16), random.randint(320, 1024))
        model.model[-1].initialize_biases(
            class_frequency=class_frequency, n_object_per_image=n_object_per_image
        )

        freq_log = torch.log(class_frequency / class_frequency.sum())

        for i in range(model.model[-1].n_layers):
            class_bias = model.model[-1].conv[i].bias.view(3, -1)[:, 5:].mean(0)
            assert torch.isclose(class_bias, freq_log, rtol=0.1).sum() == 10

        for i in range(model.model[-1].n_layers):
            obj_bias = model.model[-1].conv[i].bias.view(3, -1)[:, 4].mean()
            obj_log = torch.log(
                n_object_per_image[0]
                / (n_object_per_image[1] / model.model[-1].stride[i]) ** 2
            )
            assert torch.isclose(obj_bias, obj_log, rtol=0.1)


def test_yolo_head():
    model = Model(
        os.path.join("tests", "test_configs", "yolo_sample.yaml"), verbose=True
    )
    model.model[-1].initialize_biases()
    in_tensor = torch.rand((1, 3, 480, 380))

    out_tensor = model(in_tensor)
    assert out_tensor[0].shape == (1, 3, 60, 48, 15)
    assert out_tensor[1].shape == (1, 3, 30, 24, 15)
    assert out_tensor[2].shape == (1, 3, 15, 12, 15)

    model.eval()

    out_tensor = model(in_tensor)
    assert out_tensor[0].shape == (1, 11340, 15)
    assert out_tensor[1][0].shape == (1, 3, 60, 48, 15)
    assert out_tensor[1][1].shape == (1, 3, 30, 24, 15)
    assert out_tensor[1][2].shape == (1, 3, 15, 12, 15)


if __name__ == "__main__":
    # test_yolo_head()
    # test_yolo_head_initialize_bias()
    test_yolo_head_initialize_bias_class_probability()
