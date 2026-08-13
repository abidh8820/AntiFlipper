from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CNNMNIST(nn.Module):
    """Architecture used by the submitted AntiFLipper experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


class TinyCNN(nn.Module):
    """Fast offline-only model for pipeline smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.fc = nn.Linear(4 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.net(x), 1))


def create_model(dataset: str, requested: str = "auto") -> nn.Module:
    if dataset == "SYNTHETIC":
        return TinyCNN()
    if dataset == "MNIST":
        return CNNMNIST()
    if dataset == "CIFAR10":
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    raise ValueError(dataset)
