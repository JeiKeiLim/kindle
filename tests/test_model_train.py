"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from kindle import Model, TorchTrainer


def prepare_cifar10():
    batch_size = 16

    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10(
        "./data/cifar10", train=True, download=True, transform=preprocess
    )

    test_dataset = datasets.CIFAR10(
        "./data/cifar10", train=False, download=True, transform=preprocess
    )

    subset_sampler = SubsetRandomSampler(np.arange(0, len(train_dataset), 10))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=subset_sampler
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def get_trainer(path: str) -> Tuple[Model, TorchTrainer]:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = Model(path, verbose=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    trainer = TorchTrainer(model, criterion, optimizer, device=device)

    return model, trainer


def validate_export_model(
    model: Model, trainer: TorchTrainer, test_loader: DataLoader
) -> Tuple[bool, float, float]:
    model.eval()
    test_loss, test_accuracy = trainer.test(test_loader)

    # model.fuse().eval()
    model.export().eval()
    test_loss_fused, test_accuracy_fused = trainer.test(test_loader)

    is_loss_close = (
        np.isclose(test_loss, test_loss_fused, rtol=0.001)
        or test_loss_fused < test_loss
    )
    is_accuracy_close = (
        np.isclose(test_accuracy, test_accuracy_fused, rtol=0.001)
        or test_accuracy_fused > test_accuracy
    )

    return (is_loss_close and is_accuracy_close), test_accuracy, test_loss


def test_model_example():
    epochs = 1
    model, trainer = get_trainer(os.path.join("tests", "test_configs", "example.yaml"))

    train_loader, test_loader = prepare_cifar10()

    trainer.train(train_loader, n_epoch=epochs)

    is_fused_valid, test_accuracy, test_loss = validate_export_model(
        model, trainer, test_loader
    )

    assert is_fused_valid
    assert test_accuracy > 0.33 and test_loss < 1.8


def test_model_showcase():
    epochs = 1

    model, trainer = get_trainer(
        os.path.join("tests", "test_configs", "show_case.yaml")
    )
    train_loader, test_loader = prepare_cifar10()
    trainer.train(train_loader, n_epoch=epochs)

    is_fused_valid, test_accuracy, test_loss = validate_export_model(
        model, trainer, test_loader
    )
    print(test_loss, test_accuracy)

    assert is_fused_valid
    assert test_accuracy > 0.18 and test_loss < 2.3


def test_model_nn_model():
    epochs = 1

    model, trainer = get_trainer(os.path.join("tests", "test_configs", "nn_model.yaml"))
    train_loader, test_loader = prepare_cifar10()
    trainer.train(train_loader, n_epoch=epochs)
    is_fused_valid, test_accuracy, test_loss = validate_export_model(
        model, trainer, test_loader
    )

    print(test_loss, test_accuracy)
    assert is_fused_valid
    assert test_accuracy > 0.18 and test_loss < 2.3


def test_model_gap_model():
    epochs = 1

    model, trainer = get_trainer(
        os.path.join("tests", "test_configs", "gap_test_model.yaml")
    )
    train_loader, test_loader = prepare_cifar10()
    trainer.train(train_loader, n_epoch=epochs)
    is_fused_valid, test_accuracy, test_loss = validate_export_model(
        model, trainer, test_loader
    )

    print(test_loss, test_accuracy)
    assert is_fused_valid
    assert test_accuracy > 0.30 and test_loss < 2.1


def test_model_pretrained(force: bool = False):
    if not torch.cuda.is_available() and not force:
        return

    epochs = 1

    model, trainer = get_trainer(
        os.path.join("tests", "test_configs", "pretrained_example.yaml")
    )
    train_loader, test_loader = prepare_cifar10()
    trainer.train(train_loader, n_epoch=epochs)
    is_fused_valid, test_accuracy, test_loss = validate_export_model(
        model, trainer, test_loader
    )

    print(test_loss, test_accuracy)
    assert is_fused_valid
    assert test_accuracy > 0.50 and test_loss < 1.4


if __name__ == "__main__":
    # test_model_pretrained(force=True)
    # test_model_nn_model()
    test_model_showcase()
    # test_model_example()
    # test_model_gap_model()
