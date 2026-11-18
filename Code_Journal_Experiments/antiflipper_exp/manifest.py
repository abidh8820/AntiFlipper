from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig


SEEDS = (7, 19, 31)
# Held out from every reported comparison; used only for hyperparameter sensitivity.
SENSITIVITY_SEEDS = (101, 113, 127)

# Reduced matrix (v2, Nov 2026). Rationale: reviewers asked for multi-seed statistics,
# FLTrust, ratio/heterogeneity sweeps, sensitivity, and an adaptive report-faking
# attacker. They did not ask for every legacy baseline under every setting.
CIFAR_HEADLINE_METHODS = ("antiflipper", "antiflipper_cc", "fedavg", "median", "multi_krum", "flame", "lfighter", "fltrust")
MNIST_HEADLINE_METHODS = ("antiflipper", "fedavg", "lfighter", "fltrust")
SWEEP_METHODS = ("antiflipper", "fedavg", "lfighter", "fltrust")
LEGACY_BASELINES = ("trimmed_mean", "foolsgold", "tolpegin")
# Tags are part of the config hash; A1 keeps its v1 tags so completed runs keep their ids.
HEADLINE_TAGS = ("tier2", "headline")


def base(dataset: str, distribution: str, method: str, seed: int, experiment: str) -> ExperimentConfig:
    # Client counts are kept from the submitted paper (MNIST 100, CIFAR-10 20) and are
    # stated explicitly in the manuscript; changing them would orphan completed runs.
    if dataset == "MNIST":
        return ExperimentConfig(experiment, dataset, distribution, method, seed, rounds=200, num_clients=100, local_batch_size=64)
    return ExperimentConfig(experiment, dataset, distribution, method, seed, rounds=100, num_clients=20, local_batch_size=32)


def smoke_jobs() -> list[ExperimentConfig]:
    common = dict(
        dataset="SYNTHETIC", distribution="IID", seed=7, rounds=2, num_clients=4, local_epochs=1,
        local_batch_size=16, test_batch_size=50, malicious_client_fraction=0.25, counter_threshold=1,
        grace_rounds=0, checkpoint_every=1, tags=("smoke",),
    )
    return [
        ExperimentConfig(experiment="smoke", method="antiflipper", **common),
        ExperimentConfig(experiment="smoke", method="antiflipper_cc", report_strategy="clean_labels", **common),
        ExperimentConfig(experiment="smoke", method="lfighter", **common),
    ]


def _dedupe(jobs: list[ExperimentConfig]) -> list[ExperimentConfig]:
    seen: set[str] = set(); result: list[ExperimentConfig] = []
    for job in jobs:
        if job.job_id not in seen:
            seen.add(job.job_id); result.append(job)
    return result


def core_jobs() -> list[ExperimentConfig]:
    """Priority-ordered. The launcher runs top to bottom, so stopping early still
    leaves the most important tables complete."""
    jobs: list[ExperimentConfig] = []

    # P1 / A1: CIFAR-10 non-IID headline (main operating point, 40% attackers).
    for method, seed in itertools.product(CIFAR_HEADLINE_METHODS, SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A1_headline"), tags=HEADLINE_TAGS))

    # P1 / A7: adaptive attacker that reports accuracy on its clean labels.
    # Baselines do not use reports, so their matched runs are the A1 jobs above.
    for method, seed in itertools.product(("antiflipper", "antiflipper_cc"), SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A7_adaptive_report"), report_strategy="clean_labels", tags=("P1", "adaptive")))

    # P2 / A3: malicious-ratio boundary (0.4 is A1; 0.0 is the A2 alpha=1 clean control).
    for method, ratio, seed in itertools.product(SWEEP_METHODS, (0.2, 0.3, 0.5), SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A3_malicious_ratio"), malicious_client_fraction=ratio, tags=("P2", "boundary")))

    # P2 / A2: clean false-positive stress test under heterogeneity (no attackers).
    for method, alpha, seed in itertools.product(("antiflipper", "antiflipper_cc"), (0.1, 0.5, 1.0), SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A2_clean_heterogeneity"), dirichlet_alpha=alpha, malicious_client_fraction=0.0, tags=("P2", "clean-control")))

    # P2 / A2b: attacked, extreme heterogeneity (alpha=1.0 is A1).
    for method, seed in itertools.product(SWEEP_METHODS, SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A2_attacked_heterogeneity"), dirichlet_alpha=0.1, tags=("P2", "heterogeneity")))

    # P3 / A1: CIFAR-10 IID headline.
    for method, seed in itertools.product(CIFAR_HEADLINE_METHODS, SEEDS):
        if method == "antiflipper_cc": continue
        jobs.append(replace(base("CIFAR10", "IID", method, seed, "A1_headline"), tags=HEADLINE_TAGS))

    # P3 / A1: MNIST (secondary dataset; 6 IID jobs already completed under v1).
    for distribution, method, seed in itertools.product(("IID", "NON_IID"), MNIST_HEADLINE_METHODS, SEEDS):
        jobs.append(replace(base("MNIST", distribution, method, seed, "A1_headline"), tags=HEADLINE_TAGS))

    # P3 / A5: one-factor sensitivity of the two thresholds reviewers asked about,
    # with matched no-attack controls, on held-out seeds.
    variants = [{}]
    variants += [{"rho_threshold": x, "notes": f"rho={x}"} for x in (0.025, 0.1)]
    variants += [{"counter_threshold": x, "notes": f"counter={x}"} for x in (3, 9)]
    for variant, seed, ratio in itertools.product(variants, SENSITIVITY_SEEDS, (0.0, 0.4)):
        cfg = base("MNIST", "NON_IID", "antiflipper", seed, "A5_sensitivity")
        jobs.append(replace(cfg, malicious_client_fraction=ratio, tags=("P3", "validation", "clean-control" if ratio == 0 else "attack"), **({"notes": "default"} | variant)))
    return _dedupe(jobs)


def optional_jobs() -> list[ExperimentConfig]:
    jobs: list[ExperimentConfig] = []
    # A4: true partial-example label flipping (1.0 is A1).
    for method, fraction, seed in itertools.product(("antiflipper", "antiflipper_cc", "lfighter", "fltrust"), (0.25, 0.5, 0.75), SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A4_partial_flip"), label_poison_fraction=fraction, tags=("P4",)))
    # A6: ablations of AntiFLipper design choices.
    variants = (
        {"trust_update": "linear", "notes": "linear trust update"},
        {"threshold_mode": "absolute", "notes": "legacy absolute threshold"},
        {"counter_mode": "none", "notes": "no cumulative counter"},
        {"counter_mode": "consecutive", "notes": "consecutive counter"},
    )
    for variant, seed in itertools.product(variants, SEEDS):
        jobs.append(replace(base("MNIST", "NON_IID", "antiflipper", seed, "A6_ablation"), tags=("P4",), **variant))
    # Legacy baselines from the submitted paper, multi-seed on CIFAR-10.
    for distribution, method, seed in itertools.product(("NON_IID", "IID"), LEGACY_BASELINES, SEEDS):
        jobs.append(replace(base("CIFAR10", distribution, method, seed, "A1_headline"), tags=("P4", "legacy-baseline")))
    return jobs


def build(profile: str) -> list[ExperimentConfig]:
    if profile == "smoke": return smoke_jobs()
    if profile == "core": return core_jobs()
    if profile == "full": return _dedupe(core_jobs() + optional_jobs())
    raise ValueError(f"Unknown profile: {profile}")


def save_manifest(path: Path, profile: str, jobs: list[ExperimentConfig]) -> None:
    payload = {"schema_version": 2, "profile": profile, "job_count": len(jobs), "jobs": [x.canonical() for x in jobs]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path) -> tuple[str, list[ExperimentConfig]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["profile"], [ExperimentConfig.from_dict(x) for x in value["jobs"]]
