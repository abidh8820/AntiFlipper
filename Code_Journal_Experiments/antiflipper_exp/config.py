from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


LABEL_MAP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8}


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    dataset: str
    distribution: str
    method: str
    seed: int
    rounds: int
    num_clients: int
    participation: float = 1.0
    dirichlet_alpha: float = 1.0
    malicious_client_fraction: float = 0.4
    attack_round_probability: float = 1.0
    label_poison_fraction: float = 1.0
    local_epochs: int = 3
    local_batch_size: int = 64
    test_batch_size: int = 512
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    eta: float = 0.1
    threshold_mode: str = "relative"
    rho_threshold: float = 0.05
    absolute_threshold: float = 0.0005
    counter_threshold: int = 6
    grace_rounds: int = 2
    counter_mode: str = "cumulative"
    trust_update: str = "quadratic"
    eval_fraction: float = 1.0
    root_samples_per_class: int = 10
    checkpoint_every: int = 1
    model: str = "auto"
    device: str = "auto"
    data_root: str = "data"
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.dataset not in {"MNIST", "CIFAR10", "SYNTHETIC"}:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.distribution not in {"IID", "NON_IID"}:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        for name in ("participation", "malicious_client_fraction", "attack_round_probability", "label_poison_fraction", "eval_fraction"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.threshold_mode not in {"relative", "absolute"}:
            raise ValueError("threshold_mode must be relative or absolute")
        if self.counter_mode not in {"cumulative", "consecutive", "none"}:
            raise ValueError("counter_mode must be cumulative, consecutive, or none")
        if self.trust_update not in {"quadratic", "linear"}:
            raise ValueError("trust_update must be quadratic or linear")
        if self.counter_threshold < 1 or self.num_clients < 2 or self.rounds < 1:
            raise ValueError("counter_threshold, rounds, and num_clients are invalid")
        if self.method == "multi_krum":
            f = int(self.malicious_client_fraction * self.num_clients)
            if self.num_clients <= 2 * f + 2:
                raise ValueError("Multi-Krum requires n > 2f + 2")

    def canonical(self) -> dict[str, Any]:
        value = asdict(self)
        value["tags"] = list(self.tags)
        return value

    @property
    def config_hash(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    @property
    def job_id(self) -> str:
        fields = [self.experiment, self.dataset, self.distribution, self.method, f"s{self.seed}", self.config_hash]
        return "__".join(str(x).lower().replace(".", "p") for x in fields)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        value = dict(value)
        value["tags"] = tuple(value.get("tags", ()))
        cfg = cls(**value)
        cfg.validate()
        return cfg

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.canonical(), indent=2, sort_keys=True), encoding="utf-8")
