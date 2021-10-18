"""YOLO Head module test.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os
import random

import torch
from torch import nn

from kindle.model import YOLOModel


def test_yolo_head_initialize_bias_class_probability(n_test: int = 5):
    for n in range(n_test):
        model = YOLOModel(
            os.path.join("tests", "test_configs", "yolo_sample.yaml"), verbose=n == 0
        )
        class_probability = random.random()
        n_object_per_image = (random.randint(4, 16), random.randint(320, 1024))

        model.initialize_biases(
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
        model = YOLOModel(
            os.path.join("tests", "test_configs", "yolo_sample.yaml"), verbose=n == 0
        )
        class_frequency = torch.randint(100, 5000, (10,))
        n_object_per_image = (random.randint(4, 16), random.randint(320, 1024))
        model.initialize_biases(
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


def test_yolo_model_v2():
    model = YOLOModel(
        os.path.join("tests", "test_configs", "yolo_samplev2.yaml"),
        verbose=True,
        init_bias=True,
    )
    in_tensor = torch.rand((1, 3, 480, 380))

    out_tensor = model(in_tensor)
    assert out_tensor[0].shape == (1, 3, 60, 48, 15)
    assert out_tensor[1].shape == (1, 3, 30, 24, 15)
    assert out_tensor[2].shape == (1, 3, 15, 12, 15)

    profiler = model.profile(n_run=1)
    n_param = profiler.get_parameter_numbers()
    n_macs = profiler.get_macs()

    assert n_param == 7087815
    assert n_macs == 1316596800


def test_yolo_head():
    model = YOLOModel(
        os.path.join("tests", "test_configs", "yolo_sample.yaml"), verbose=True
    )
    model.initialize_biases()
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

    profiler = model.profile(n_run=1)
    n_param = profiler.get_parameter_numbers()
    n_macs = profiler.get_macs()

    assert (
        n_param == 7279367
    ), f"Number of parameter is not matched! Current: {n_param:,d}"
    assert n_macs == 1350937664, f"Number of MACs is not matched! Current: {n_macs}"


def test_yolo_head_fused():
    model = YOLOModel(
        os.path.join("tests", "test_configs", "yolo_sample.yaml"), verbose=True
    )
    model.initialize_biases()

    in_tensor = torch.rand((1, 3, 480, 380))
    model.eval()
    out_tensor = model(in_tensor)
    model.fuse().eval()

    out_tensor_fused = model(in_tensor)

    assert torch.all(torch.isclose(out_tensor[0], out_tensor_fused[0], rtol=1e-6))
    for i in range(model.model[-1].n_layers):
        assert torch.all(
            torch.isclose(out_tensor[1][i], out_tensor_fused[1][i], rtol=1e-6)
        )


def test_yolo_head_xyxy():
    model = YOLOModel(
        os.path.join("tests", "test_configs", "yolo_samplev2.yaml"),
        verbose=True,
        init_bias=True,
    )

    in_tensor = torch.rand((1, 3, 480, 380))
    model.eval()
    model.model[-1].out_xyxy = False
    out_tensor_xywh = model(in_tensor)
    model.model[-1].out_xyxy = True
    out_tensor_xyxy = model(in_tensor)

    xywh = out_tensor_xywh[0][0, :, :4]
    xyxy = out_tensor_xyxy[0][0, :, :4]

    x1y1_from_xywh = xywh[:, :2] - (xywh[:, 2:4] / 2)
    x2y2_from_xywh = xywh[:, :2] + (xywh[:, 2:4] / 2)
    wh_from_xyxy = xyxy[:, 2:] - xyxy[:, :2]
    cxcy_from_xyxy = xyxy[:, :2] + (wh_from_xyxy / 2)

    assert torch.isclose(torch.cat((x1y1_from_xywh, x2y2_from_xywh), -1), xyxy).all()
    assert torch.isclose(torch.cat((cxcy_from_xyxy, wh_from_xyxy), -1), xywh).all()


def test_yolo_head_export():
    model = YOLOModel(
        os.path.join("tests", "test_configs", "yolo_samplev2.yaml"),
        verbose=True,
        init_bias=True,
    )

    in_tensor = torch.rand((1, 3, 480, 380))
    model.eval()
    out_tensor = model(in_tensor)
    model.export(verbose=True)

    out_tensor_export = model(in_tensor)

    assert torch.all(torch.isclose(out_tensor[0], out_tensor_export[0], rtol=1e-6))


def test_yolo_model_v3():
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = YOLOModel(
        os.path.join("tests", "test_configs", "yolo_samplev3.yaml"),
        verbose=True,
        init_bias=True,
    ).to(device)
    in_tensor = torch.rand((1, 3, 512, 512)).to(device)

    out_tensor = model(in_tensor)
    assert out_tensor[0].shape == (1, 3, 64, 64, 15)
    assert out_tensor[1].shape == (1, 3, 32, 32, 15)
    assert out_tensor[2].shape == (1, 3, 16, 16, 15)

    profiler = model.profile(n_run=100)
    n_param = profiler.get_parameter_numbers()
    n_macs = profiler.get_macs()

    assert n_param == 7087815
    assert n_macs == 1316629568


def test_yolo_model_mobilevit():
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = YOLOModel(
        os.path.join("tests", "test_configs", "yolo_mobilevit_sample.yaml"),
        verbose=True,
        init_bias=True,
    ).to(device)
    in_tensor = torch.rand((1, 3, 512, 512)).to(device)

    out_tensor = model(in_tensor)
    assert out_tensor[0].shape == (1, 3, 64, 64, 15)
    assert out_tensor[1].shape == (1, 3, 32, 32, 15)
    assert out_tensor[2].shape == (1, 3, 16, 16, 15)

    profiler = model.profile(n_run=100)
    n_param = profiler.get_parameter_numbers()
    n_macs = profiler.get_macs()

    assert n_param == 4910455
    assert n_macs == 1747833760


if __name__ == "__main__":
    # test_yolo_head()
    # test_yolo_head_initialize_bias()
    # test_yolo_head_initialize_bias_class_probability()
    # test_yolo_head_fused()
    # test_yolo_model_v2()
    # test_yolo_head_xyxy()
    # test_yolo_head_export()
    test_yolo_model_v3()
    test_yolo_model_mobilevit()
