from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig


SEEDS = (7, 19, 31)
HEADLINE_METHODS = ("antiflipper", "fedavg", "median", "trimmed_mean", "foolsgold", "tolpegin", "multi_krum", "flame", "lfighter", "fltrust")
SWEEP_METHODS = ("antiflipper", "fedavg", "lfighter", "fltrust")


def base(dataset: str, distribution: str, method: str, seed: int, experiment: str) -> ExperimentConfig:
    if dataset == "MNIST":
        return ExperimentConfig(experiment, dataset, distribution, method, seed, rounds=200, num_clients=100, local_batch_size=64)
    return ExperimentConfig(experiment, dataset, distribution, method, seed, rounds=100, num_clients=20, local_batch_size=32)


def smoke_jobs() -> list[ExperimentConfig]:
    value = ExperimentConfig(
        experiment="smoke", dataset="SYNTHETIC", distribution="IID", method="antiflipper", seed=7,
        rounds=2, num_clients=4, local_epochs=1, local_batch_size=16, test_batch_size=50,
        malicious_client_fraction=0.25, counter_threshold=1, grace_rounds=0, checkpoint_every=1,
        tags=("smoke",),
    )
    return [value]


def core_jobs() -> list[ExperimentConfig]:
    jobs: list[ExperimentConfig] = []
    # A1: headline table, including the reviewer-requested FLTrust baseline.
    for dataset, distribution, method, seed in itertools.product(("MNIST", "CIFAR10"), ("IID", "NON_IID"), HEADLINE_METHODS, SEEDS):
        jobs.append(replace(base(dataset, distribution, method, seed, "A1_headline"), tags=("tier2", "headline")))

    # A2: AntiFLipper clean false-positive stress test. Baselines cannot emit its detection metrics.
    for dataset, seed in itertools.product(("MNIST", "CIFAR10"), SEEDS):
        jobs.append(replace(base(dataset, "IID", "antiflipper", seed, "A2_clean_heterogeneity"), malicious_client_fraction=0.0, tags=("tier2", "clean-control")))
        for alpha in (0.1, 0.3, 1.0):
            jobs.append(replace(base(dataset, "NON_IID", "antiflipper", seed, "A2_clean_heterogeneity"), dirichlet_alpha=alpha, malicious_client_fraction=0.0, tags=("tier2", "clean-control")))

    # A3: boundary sweep on CIFAR-10 non-IID, scoped to four methods.
    for method, ratio, seed in itertools.product(SWEEP_METHODS, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5), SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A3_malicious_ratio"), malicious_client_fraction=ratio, tags=("tier2", "boundary")))

    # A5-lite: one factor at a time plus matched no-attack controls. Seeds 101/113/127
    # are held out from the main comparisons and must not be used for headline reporting.
    sensitivity_seeds = (101, 113, 127)
    variants: list[dict] = []
    variants += [{"eta": x, "notes": f"eta={x}"} for x in (0.05, 0.1, 0.2)]
    variants += [{"rho_threshold": x, "notes": f"rho={x}"} for x in (0.025, 0.05, 0.1)]
    variants += [{"counter_threshold": x, "notes": f"counter={x}"} for x in (3, 6, 9)]
    seen: set[tuple] = set()
    for variant, seed, attack_ratio in itertools.product(variants, sensitivity_seeds, (0.0, 0.4)):
        signature = (seed, attack_ratio, variant.get("eta", 0.1), variant.get("rho_threshold", 0.05), variant.get("counter_threshold", 6))
        if signature in seen: continue
        seen.add(signature)
        cfg = base("MNIST", "NON_IID", "antiflipper", seed, "A5_sensitivity")
        jobs.append(replace(cfg, malicious_client_fraction=attack_ratio, tags=("tier2", "validation", "clean-control" if attack_ratio == 0 else "attack"), **variant))
    return jobs


def optional_jobs() -> list[ExperimentConfig]:
    jobs: list[ExperimentConfig] = []
    # A4-lite: true partial-example poisoning, not intermittent full attacks.
    for method, fraction, seed in itertools.product(SWEEP_METHODS, (0.1, 0.25, 0.5, 0.75, 1.0), SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A4_partial_flip"), label_poison_fraction=fraction, tags=("tier3",)))
    # A6-lite: method ablations, including the legacy absolute threshold.
    variants = (
        {"trust_update": "linear", "notes": "linear trust update"},
        {"threshold_mode": "absolute", "notes": "legacy absolute threshold"},
        {"counter_mode": "none", "notes": "no cumulative counter"},
        {"counter_mode": "consecutive", "notes": "consecutive counter"},
    )
    for variant, seed in itertools.product(variants, SEEDS):
        jobs.append(replace(base("MNIST", "NON_IID", "antiflipper", seed, "A6_ablation"), tags=("tier3",), **variant))
    return jobs


def build(profile: str) -> list[ExperimentConfig]:
    if profile == "smoke": return smoke_jobs()
    if profile == "core": return core_jobs()
    if profile == "full": return core_jobs() + optional_jobs()
    raise ValueError(f"Unknown profile: {profile}")


def save_manifest(path: Path, profile: str, jobs: list[ExperimentConfig]) -> None:
    payload = {"schema_version": 1, "profile": profile, "job_count": len(jobs), "jobs": [x.canonical() for x in jobs]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path) -> tuple[str, list[ExperimentConfig]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["profile"], [ExperimentConfig.from_dict(x) for x in value["jobs"]]
