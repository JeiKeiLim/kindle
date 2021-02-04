"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from kindle import Model, TorchTrainer


def test_model_train():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = Model(os.path.join("tests", "test_configs", "example.yaml"), verbose=True)
    batch_size = 16
    epochs = 1

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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=subset_sampler
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    trainer = TorchTrainer(model, criterion, optimizer, device=device)
    trainer.train(train_loader, n_epoch=epochs, test_dataloader=test_loader)
    test_loss, test_accuracy = trainer.test(test_loader)

    print(test_loss, test_accuracy)
    assert test_accuracy > 0.45 and test_loss < 1.5


if __name__ == "__main__":
    test_model_train()
