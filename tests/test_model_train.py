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


def test_model_example():
    epochs = 1
    model, trainer = get_trainer(os.path.join("tests", "test_configs", "example.yaml"))

    train_loader, test_loader = prepare_cifar10()

    trainer.train(train_loader, n_epoch=epochs)
    test_loss, test_accuracy = trainer.test(test_loader)

    print(test_loss, test_accuracy)
    assert test_accuracy > 0.35 and test_loss < 1.7


def test_model_showcase():
    epochs = 1

    model, trainer = get_trainer(
        os.path.join("tests", "test_configs", "show_case.yaml")
    )
    train_loader, test_loader = prepare_cifar10()
    trainer.train(train_loader, n_epoch=epochs)
    test_loss, test_accuracy = trainer.test(test_loader)

    print(test_loss, test_accuracy)
    assert test_accuracy > 0.18 and test_loss < 2.3


def optuna_sample():
    import optuna

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.CIFAR10(
        "./data/cifar10", train=True, download=True, transform=preprocess
    )
    test_dataset = datasets.CIFAR10(
        "./data/cifar10", train=False, download=True, transform=preprocess
    )
    subset_sampler = SubsetRandomSampler(np.arange(0, len(train_dataset), 2))

    def objective(trial: optuna.Trial):
        model_cfg = {
            "input_size": [32, 32],
            "input_channel": 3,
            "depth_multiple": 1.0,
            "width_multiple": 1.0,
        }
        conv_type = trial.suggest_categorical("conv_type", ["Conv", "DWConv"])
        kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
        n_channel_01 = trial.suggest_int("n_channel_01", 8, 64, step=8)
        n_channel_02 = trial.suggest_int("n_channel_02", 8, 128, step=8)

        linear_activation = trial.suggest_categorical(
            "linear_activation", ["ReLU", "SiLU"]
        )
        n_channel_03 = trial.suggest_int("n_channel_03", 64, 256, step=8)
        n_channel_04 = trial.suggest_int("n_channel_04", 32, 128, step=8)
        n_repeat = trial.suggest_int("n_repeat", 1, 3)

        backbone = [
            [-1, n_repeat, conv_type, [n_channel_01, kernel_size, 1]],
            [-1, 1, "MaxPool", [2]],
            [-1, n_repeat, conv_type, [int(n_channel_02), kernel_size, 1]],
            [-1, 1, "MaxPool", [2]],
            [-1, 1, "Flatten", []],
            [-1, 1, "Linear", [n_channel_03, linear_activation]],
            [-1, 1, "Linear", [n_channel_04, linear_activation]],
            [-1, 1, "Linear", [10]],
        ]
        model_cfg.update({"backbone": backbone})

        model = Model(model_cfg, verbose=True).to(device)
        batch_size = trial.suggest_int("batch_size", 8, 256)
        epochs = trial.suggest_int("epochs", 5, 20)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=subset_sampler
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        trainer = TorchTrainer(model, criterion, optimizer, device=device)
        trainer.train(train_loader, n_epoch=epochs, test_dataloader=test_loader)
        test_loss, test_accuracy = trainer.test(test_loader)

        return test_loss

    study = optuna.create_study(study_name="Sample AutoML", direction="minimize")
    study.optimize(objective)


if __name__ == "__main__":
    test_model_showcase()
    test_model_example()
    # optuna_sample()
