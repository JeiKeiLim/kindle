"""PyTorch trainer module.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from tqdm import tqdm


def _get_n_data_from_dataloader(dataloader: DataLoader) -> int:
    """Get a number of data in dataloader.

    Args:
        dataloader: torch data loader

    Returns:
        A number of data in dataloader
    """
    if isinstance(dataloader.sampler, SubsetRandomSampler):
        n_data = len(dataloader.sampler.indices)
    elif isinstance(dataloader.sampler, SequentialSampler):
        n_data = len(dataloader.sampler.data_source)
    else:
        n_data = len(dataloader) * dataloader.batch_size if dataloader.batch_size else 1

    return n_data


def _get_n_batch_from_dataloader(dataloader: DataLoader) -> int:
    """Get a batch number in dataloader.

    Args:
        dataloader: torch data loader

    Returns:
        A batch number in dataloader
    """
    n_data = _get_n_data_from_dataloader(dataloader)
    n_batch = dataloader.batch_size if dataloader.batch_size else 1

    return n_data // n_batch


class TorchTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: Union[str, torch.device] = "cpu",
        verbose: int = 1,
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
        """

        if isinstance(device, str):
            device = torch.device(device)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.verbose = verbose
        self.device = device

    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        shuffle: bool = False,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            test_dataloader: test data loader
            shuffle: shuffle train data on every epoch.
                     Sampler must be SubsetRandomSampler to apply shuffle.

        Returns:
            loss and accuracy
        """
        average_loss, accuracy = -1.0, -1.0
        n_batch = _get_n_batch_from_dataloader(train_dataloader)

        for epoch in range(n_epoch):
            if shuffle and isinstance(train_dataloader.sampler, SubsetRandomSampler):
                np.random.shuffle(train_dataloader.sampler.indices)

            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(enumerate(train_dataloader), total=n_batch)
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                model_out = self.model(data)
                loss = self.criterion(model_out, labels)
                loss.backward()
                self.optimizer.step()

                # TODO: Modify for multi-label classification.
                _, predicted = torch.max(model_out, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.7f}, "
                    f"Accuracy: {(correct / total) * 100:.2f}%"
                )
            pbar.close()

            if test_dataloader is not None:
                self.test(test_dataloader)

            average_loss = running_loss / n_batch
            accuracy = correct / total

        return average_loss, accuracy

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader) -> Tuple[float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, accuracy
        """

        n_batch = _get_n_batch_from_dataloader(test_dataloader)

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(enumerate(test_dataloader), total=n_batch)
        for batch, (data, labels) in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            model_out = self.model(data)
            running_loss += self.criterion(model_out, labels).item()

            # TODO: Modify for multi-label classification.
            _, predicted = torch.max(model_out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.update()
            pbar.set_description(
                f" Test: {'':5} Loss: {(running_loss / (batch + 1)):.7f}, "
                f"Accuracy: {(correct / total) * 100:.2f}%"
            )

        loss = running_loss / n_batch
        accuracy = correct / total
        return loss, accuracy
