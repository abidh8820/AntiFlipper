from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


LABEL_MAP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8}

# Methods whose server logic is the AntiFLipper trust pipeline.
#   antiflipper     accuracy reports only (submitted method)
#   antiflipper_cc  accuracy reports + update-consistency cross-check (proposed)
#   cc_only         ablation: consistency cross-check only, reports ignored, uniform weights
#   antiflipper_cc_v2  strike-based soft weighting driven by the cross-check (GPU-side variant)
#   antiflipper_cr     class-conditional residual: a client is compared with the accuracy
#                      expected for its own label mix, not with the population mean
#   antiflipper_cr_cc  class-conditional residual + update-consistency cross-check
CC_V2 = "antiflipper_cc_v2"
CR_METHODS = {"antiflipper_cr", "antiflipper_cr_cc"}
HARD_EXCLUSION_METHODS = {"antiflipper", "antiflipper_cc", "cc_only"} | CR_METHODS
ANTIFLIPPER_METHODS = HARD_EXCLUSION_METHODS | {CC_V2}
CONSISTENCY_METHODS = {"antiflipper_cc", "cc_only", CC_V2, "antiflipper_cr_cc"}
REPORT_METHODS = {"antiflipper", "antiflipper_cc", CC_V2} | CR_METHODS
# Methods whose inconsistent clients get their report replaced by the round minimum.
REPORT_REPLACEMENT_METHODS = {"antiflipper_cc", "antiflipper_cr_cc"}
# Threat model: at most 40% of clients are malicious.
MAX_MALICIOUS_FRACTION = 0.4
KNOWN_METHODS = ANTIFLIPPER_METHODS | {"fedavg", "median", "trimmed_mean", "foolsgold", "tolpegin", "multi_krum", "flame", "lfighter", "fltrust"}

# Fields added after the first experiments were run. They are omitted from the
# canonical form (and therefore from the config hash / job id) while they hold
# their default value, so already-completed jobs keep their identity.
_LATE_FIELDS_DEFAULTS = {"report_strategy": "honest", "consistency_z": 2.5, "max_exclusion_fraction": 1.0}


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
    # "honest": every client reports the accuracy it actually measured (submitted threat model).
    # "clean_labels": adaptive attacker trains on flipped labels but reports the accuracy
    # measured on its *unflipped* local labels, i.e. it lies to look honest.
    report_strategy: str = "honest"
    # antiflipper_cc only: robust z-score cut-off for the update-consistency cross-check.
    consistency_z: float = 2.5
    # Upper bound on the fraction of all clients that may ever be excluded. Tied to the
    # threat model (<=40% malicious): beyond it, exclusions are more likely to be honest
    # skewed clients, and each one removes classes from training (exclusion cascade).
    # 1.0 = no budget (all methods before antiflipper_cr).
    max_exclusion_fraction: float = 1.0

    def validate(self) -> None:
        if self.method not in KNOWN_METHODS:
            raise ValueError(f"Unknown method: {self.method}")
        if self.report_strategy not in {"honest", "clean_labels"}:
            raise ValueError("report_strategy must be honest or clean_labels")
        if self.consistency_z <= 0:
            raise ValueError("consistency_z must be positive")
        if not 0 < self.max_exclusion_fraction <= 1:
            raise ValueError("max_exclusion_fraction must be in (0, 1]")
        if self.malicious_client_fraction > MAX_MALICIOUS_FRACTION + 1e-9:
            raise ValueError(f"malicious_client_fraction above the {MAX_MALICIOUS_FRACTION:.0%} threat-model bound")
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
        for name, default in _LATE_FIELDS_DEFAULTS.items():
            if value.get(name) == default:
                value.pop(name)
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
