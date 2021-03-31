"""Module Description.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from kindle import Model
from tests.test_model_train import prepare_cifar10


@torch.no_grad()
def aug_flip_lr(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


@torch.no_grad()
def aug_flip_ud(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


@torch.no_grad()
def aug_flip_lrud(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2, -1])


@torch.no_grad()
def aug_random_scale(x: torch.Tensor) -> torch.Tensor:
    ratio = np.random.uniform(0.8, 1.2)
    xr = F.interpolate(x, scale_factor=ratio)

    if xr.shape[2:] != x.shape[2:]:
        h1, w1 = x.shape[2:]
        h2, w2 = xr.shape[2:]

        xr = F.pad(xr, [0, w1 - w2, 0, h1 - h2], value=0)

    return xr


def test_tta_model():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    epochs = 1
    model = Model(os.path.join("tests", "test_configs", "example.yaml"), verbose=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_loader, test_loader = prepare_cifar10()

    aug_funcs = [aug_flip_ud, aug_flip_lr, aug_flip_lrud, aug_random_scale]

    for epoch in range(epochs):
        seen = 0
        tta_correct = 0
        running_loss = 0

        pbar = tqdm(enumerate(train_loader))
        for batch, (data, labels) in pbar:
            data, labels = data.to(device), labels.to(device)
            aug_idx = np.random.randint(len(aug_funcs) + 1)
            if aug_idx < len(aug_funcs):
                data = aug_funcs[aug_idx](data)

            optimizer.zero_grad()
            tta_out = model(data)
            # tta_out = model(data, augment_func=aug_funcs)
            # tta_out = F.softmax(tta_out, dim=-1).sum(dim=0)
            tta_loss = criterion(tta_out, labels)
            tta_loss.backward()
            optimizer.step()

            _, tta_predicted = torch.max(tta_out, 1)
            seen += labels.size(0)
            tta_correct += (tta_predicted == labels).sum().item()
            running_loss += tta_loss.item()

            pbar.set_description(
                f"[Train]: [{epoch + 1:03d}] "
                f"Loss: {(running_loss / (batch + 1)):.7f}, "
                f"Accuracy: {(tta_correct / seen) * 100:.2f}%"
            )
        pbar.close()

        with torch.no_grad():
            seen = 0
            tta_correct = 0
            model_correct = 0
            tta_running_loss = 0
            model_running_loss = 0

            pbar = tqdm(enumerate(test_loader))
            for batch, (data, labels) in pbar:
                data, labels = data.to(device), labels.to(device)
                tta_out = model(data, augment_func=aug_funcs)
                # tta_out = model(data, augment_func=aug_random_scale)
                # tta_out = tta_out.sum(dim=0)
                tta_out = F.softmax(tta_out, dim=-1).sum(dim=0)
                tta_out = F.softmax(tta_out, dim=-1)

                model_out = model(data)

                tta_loss = criterion(tta_out, labels)
                model_loss = criterion(model_out, labels)

                _, tta_predicted = torch.max(tta_out, 1)
                _, model_predicted = torch.max(model_out, 1)
                seen += labels.size(0)
                tta_correct += (tta_predicted == labels).sum().item()
                model_correct += (model_predicted == labels).sum().item()

                tta_running_loss += tta_loss.item()
                model_running_loss += model_loss.item()

                pbar.set_description(
                    f"[Test]: [{epoch + 1:03d}] "
                    f"Loss(TTA, No-TTA): {(tta_running_loss / (batch + 1)):.7f}, {(model_running_loss / (batch + 1)):.7f}, "
                    f"Accuracy(TTA, No-TTA): {(tta_correct / seen) * 100:.2f}%, {(model_correct / seen) * 100:.2f}%"
                )


if __name__ == "__main__":
    test_tta_model()
