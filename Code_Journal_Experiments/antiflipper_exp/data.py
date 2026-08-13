from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from .attacks import IndexedDataset
from .config import ExperimentConfig


@dataclass
class FederatedData:
    train: Dataset
    test: Dataset
    clients: dict[int, IndexedDataset]
    root: IndexedDataset | None
    partition_indices: dict[int, list[int]]


def _targets(dataset: Dataset) -> np.ndarray:
    values = getattr(dataset, "targets", None)
    if values is not None:
        return np.asarray(values, dtype=np.int64)
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)


def _load(cfg: ExperimentConfig) -> tuple[Dataset, Dataset]:
    if cfg.dataset == "SYNTHETIC":
        generator = torch.Generator().manual_seed(cfg.seed)
        x_train = torch.randn(240, 1, 28, 28, generator=generator)
        y_train = torch.arange(240) % 10
        x_test = torch.randn(100, 1, 28, 28, generator=generator)
        y_test = torch.arange(100) % 10
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)

    from torchvision import datasets, transforms

    root = Path(cfg.data_root)
    if cfg.dataset == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return (
            datasets.MNIST(root / "mnist", train=True, download=True, transform=transform),
            datasets.MNIST(root / "mnist", train=False, download=True, transform=transform),
        )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return (
        datasets.CIFAR10(root / "cifar10", train=True, download=True, transform=transform),
        datasets.CIFAR10(root / "cifar10", train=False, download=True, transform=transform),
    )


def _reserve_root(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> tuple[list[int], np.ndarray]:
    root: list[int] = []
    keep = np.ones(len(labels), dtype=bool)
    for cls in np.unique(labels):
        choices = np.flatnonzero(labels == cls)
        selected = rng.choice(choices, size=min(per_class, len(choices)), replace=False)
        root.extend(int(x) for x in selected)
        keep[selected] = False
    return root, np.flatnonzero(keep)


def _iid(indices: np.ndarray, n: int, rng: np.random.Generator) -> dict[int, list[int]]:
    shuffled = rng.permutation(indices)
    return {i: [int(x) for x in split] for i, split in enumerate(np.array_split(shuffled, n))}


def _dirichlet(indices: np.ndarray, labels: np.ndarray, n: int, alpha: float, rng: np.random.Generator) -> dict[int, list[int]]:
    # Retry until every client has enough data to form at least one normal batch.
    for _ in range(100):
        result = {i: [] for i in range(n)}
        for cls in np.unique(labels[indices]):
            class_indices = rng.permutation(indices[labels[indices] == cls])
            proportions = rng.dirichlet(np.full(n, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for client_id, split in enumerate(np.split(class_indices, cuts)):
                result[client_id].extend(int(x) for x in split)
        if min(map(len, result.values())) >= 2:
            for value in result.values():
                rng.shuffle(value)
            return result
    raise RuntimeError("Could not construct a non-empty Dirichlet partition; increase alpha or reduce clients")


def build_federated_data(cfg: ExperimentConfig) -> FederatedData:
    train, test = _load(cfg)
    labels = _targets(train)
    rng = np.random.default_rng(cfg.seed + 101)
    # Reserve the same balanced root subset for every method so matched-seed client
    # partitions contain exactly the same examples. Only FLTrust consumes the root set.
    root_indices, available = _reserve_root(labels, cfg.root_samples_per_class, rng)
    if cfg.distribution == "IID":
        partitions = _iid(available, cfg.num_clients, rng)
    else:
        partitions = _dirichlet(available, labels, cfg.num_clients, cfg.dirichlet_alpha, rng)
    clients = {i: IndexedDataset(train, idx) for i, idx in partitions.items()}
    root = IndexedDataset(train, root_indices)
    return FederatedData(train, test, clients, root, partitions)
