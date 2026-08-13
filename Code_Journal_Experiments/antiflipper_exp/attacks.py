from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import LABEL_MAP


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base[self.indices[index]]


class PartiallyPoisonedDataset(Dataset):
    """Deterministically permutes exactly a configured subset of local labels."""

    def __init__(self, base: Dataset, poison_fraction: float, seed: int) -> None:
        self.base = base
        count = int(round(len(base) * poison_fraction))
        rng = np.random.default_rng(seed)
        self.poisoned = set(int(i) for i in rng.choice(len(base), size=count, replace=False)) if count else set()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        label = int(y.item()) if torch.is_tensor(y) else int(y)
        if index in self.poisoned:
            label = LABEL_MAP[label]
        return x, label

    @property
    def poisoned_examples(self) -> int:
        return len(self.poisoned)
