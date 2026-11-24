from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

from .config import CC_V2, ExperimentConfig


SEEDS = (7, 19, 31)
PROFILES = ("smoke", "core", "full", "cc_v2", "cr", "fltrust_fix", "rev")
# Held out from every reported comparison; used only for hyperparameter sensitivity.
SENSITIVITY_SEEDS = (101, 113, 127)

# Reduced matrix (v2, Nov 2026). Rationale: reviewers asked for multi-seed statistics,
# FLTrust, ratio/heterogeneity sweeps, sensitivity, and an adaptive report-faking
# attacker. They did not ask for every legacy baseline under every setting.
# AntiFLipper-CC is the proposed method; plain AntiFLipper (reports only) and cc_only
# (cross-check only) are its two halves, kept to show what each part contributes.
CIFAR_HEADLINE_METHODS = ("antiflipper_cc", "antiflipper", "cc_only", "fedavg", "median", "multi_krum", "flame", "lfighter", "fltrust")
MNIST_HEADLINE_METHODS = ("antiflipper_cc", "antiflipper", "fedavg", "lfighter", "fltrust")
SWEEP_METHODS = ("antiflipper_cc", "antiflipper", "fedavg", "lfighter", "fltrust")
ABLATION_METHODS = ("antiflipper_cc", "antiflipper", "cc_only")
# Threat model bound is 40%; 0.4 itself is the A1 headline operating point.
RATIO_SWEEP = (0.1, 0.2, 0.3)
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
        ExperimentConfig(experiment="smoke", method="cc_only", **common),
        ExperimentConfig(experiment="smoke", method="lfighter", **common),
        ExperimentConfig(experiment="smoke", method="antiflipper_cr", max_exclusion_fraction=0.4, **common),
        ExperimentConfig(experiment="smoke", method="antiflipper_cr_cc", max_exclusion_fraction=0.4, consistency_z=3.0, report_strategy="clean_labels", **common),
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
    # Baselines and cc_only ignore reports, so their matched runs are the A1 jobs above.
    for method, seed in itertools.product(("antiflipper_cc", "antiflipper"), SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A7_adaptive_report"), report_strategy="clean_labels", tags=("P1", "adaptive")))

    # P2 / A2: clean false-positive stress test under heterogeneity (no attackers).
    # This is where the cross-check could hurt honest skewed clients, so all three variants.
    for method, alpha, seed in itertools.product(ABLATION_METHODS, (0.1, 0.5, 1.0), SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A2_clean_heterogeneity"), dirichlet_alpha=alpha, malicious_client_fraction=0.0, tags=("P2", "clean-control")))

    # P2 / A2b: attacked, extreme heterogeneity (alpha=1.0 is A1).
    for method, seed in itertools.product(SWEEP_METHODS + ("cc_only",), SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A2_attacked_heterogeneity"), dirichlet_alpha=0.1, tags=("P2", "heterogeneity")))

    # P2 / A3: malicious-ratio sweep up to the 40% threat-model bound (0.4 is A1;
    # 0.0 is the A2 alpha=1 clean control).
    for method, ratio, seed in itertools.product(SWEEP_METHODS, RATIO_SWEEP, SEEDS):
        jobs.append(replace(base("CIFAR10", "NON_IID", method, seed, "A3_malicious_ratio"), malicious_client_fraction=ratio, tags=("P2", "boundary")))

    # P3 / A1: CIFAR-10 IID headline.
    for method, seed in itertools.product(CIFAR_HEADLINE_METHODS, SEEDS):
        if method == "cc_only": continue
        jobs.append(replace(base("CIFAR10", "IID", method, seed, "A1_headline"), tags=HEADLINE_TAGS))

    # P3 / A1: MNIST (secondary dataset; 6 IID jobs already completed under v1).
    for distribution, method, seed in itertools.product(("IID", "NON_IID"), MNIST_HEADLINE_METHODS, SEEDS):
        jobs.append(replace(base("MNIST", distribution, method, seed, "A1_headline"), tags=HEADLINE_TAGS))

    # P3 / A5: one-factor sensitivity of the proposed method's thresholds (rho, counter,
    # consistency z), with matched no-attack controls, on held-out seeds.
    variants = [{}]
    variants += [{"rho_threshold": x, "notes": f"rho={x}"} for x in (0.025, 0.1)]
    variants += [{"counter_threshold": x, "notes": f"counter={x}"} for x in (3, 9)]
    variants += [{"consistency_z": x, "notes": f"z={x}"} for x in (2.0, 3.0)]
    for variant, seed, ratio in itertools.product(variants, SENSITIVITY_SEEDS, (0.0, 0.4)):
        cfg = base("MNIST", "NON_IID", "antiflipper_cc", seed, "A5_sensitivity")
        jobs.append(replace(cfg, malicious_client_fraction=ratio, tags=("P3", "validation", "clean-control" if ratio == 0 else "attack"), **({"notes": "default"} | variant)))
    return _dedupe(jobs)


def optional_jobs() -> list[ExperimentConfig]:
    jobs: list[ExperimentConfig] = []
    # A4: true partial-example label flipping (1.0 is A1).
    # Partial flips move the last layer less, so this stresses the cross-check most.
    for method, fraction, seed in itertools.product(ABLATION_METHODS + ("lfighter", "fltrust"), (0.25, 0.5, 0.75), SEEDS):
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


def cc_v2_jobs(reference: list[ExperimentConfig]) -> list[ExperimentConfig]:
    matched = {
        (job.experiment, job.seed): job for job in reference
        if job.method == "antiflipper_cc" and job.dataset == "CIFAR10" and job.distribution == "NON_IID"
        and (job.experiment != "A2_clean_heterogeneity" or job.dirichlet_alpha == 0.1)
    }
    # Pair the two key comparisons by seed so both have early results.
    return [
        replace(matched[experiment, seed], method=CC_V2)
        for experiments in (("A2_clean_heterogeneity", "A7_adaptive_report"), ("A1_headline", "A2_attacked_heterogeneity"))
        for seed in SEEDS
        for experiment in experiments
    ]


# --- Class-conditional residual (Nov 22) --------------------------------------------
# Core results showed report-based AntiFLipper excluding up to 92% of honest clients at
# Dirichlet 0.1 (clients holding hard classes look like attackers; each exclusion removes
# classes and drags the next clients down). antiflipper_cr compares each client with the
# accuracy expected for its own label mix and caps exclusions at the 40% threat bound.
CR = {"method": "antiflipper_cr", "max_exclusion_fraction": 0.4}
CR_CC = {"method": "antiflipper_cr_cc", "max_exclusion_fraction": 0.4, "consistency_z": 3.0}
# Ablations: is it the residual or the budget? Does z=3 alone fix the old CC?
BUDGET_ONLY = {"method": "antiflipper", "max_exclusion_fraction": 0.4, "notes": "budget only"}
CC_Z3 = {"method": "antiflipper_cc", "consistency_z": 3.0}


def cr_jobs() -> list[ExperimentConfig]:
    """Priority-ordered. The first 14 jobs (seed 7) are a pilot that answers whether the
    fix works in ~6 GPU hours; then the same settings for seeds 19 and 31; then coverage."""
    def job(dataset, distribution, seed, experiment, variant, tags, **overrides):
        return replace(base(dataset, distribution, variant["method"], seed, experiment),
                       tags=tags, **{**{k: v for k, v in variant.items() if k != "method"}, **overrides})

    key_settings = (  # (experiment, overrides, variants)
        ("A2_clean_heterogeneity", {"dirichlet_alpha": 0.1, "malicious_client_fraction": 0.0}, (CR, CR_CC, BUDGET_ONLY, CC_Z3)),
        ("A2_attacked_heterogeneity", {"dirichlet_alpha": 0.1}, (CR, CR_CC, BUDGET_ONLY)),
        ("A1_headline", {}, (CR, CR_CC, CC_Z3)),
        ("A7_adaptive_report", {"report_strategy": "clean_labels"}, (CR, CR_CC)),
        ("A2_clean_heterogeneity", {"dirichlet_alpha": 0.5, "malicious_client_fraction": 0.0}, (CR, CR_CC)),
    )
    jobs: list[ExperimentConfig] = []
    for seed in SEEDS:  # seed-major: seed 7 = pilot
        for experiment, overrides, variants in key_settings:
            for variant in variants:
                tags = ("CR", "pilot" if seed == SEEDS[0] else "key")
                jobs.append(job("CIFAR10", "NON_IID", seed, experiment, variant, tags, **overrides))
    for variant, seed in itertools.product((CR, CR_CC), SEEDS):
        jobs.append(job("CIFAR10", "NON_IID", seed, "A2_clean_heterogeneity", variant, ("CR", "coverage"), dirichlet_alpha=1.0, malicious_client_fraction=0.0))
    for variant, ratio, seed in itertools.product((CR, CR_CC), RATIO_SWEEP, SEEDS):
        jobs.append(job("CIFAR10", "NON_IID", seed, "A3_malicious_ratio", variant, ("CR", "coverage"), malicious_client_fraction=ratio))
    for variant, seed in itertools.product((CR, CR_CC), SEEDS):
        jobs.append(job("CIFAR10", "IID", seed, "A1_headline", variant, ("CR", "coverage")))
    for distribution, variant, seed in itertools.product(("NON_IID", "IID"), (CR, CR_CC), SEEDS):
        jobs.append(job("MNIST", distribution, seed, "A1_headline", variant, ("CR", "coverage")))
    return _dedupe(jobs)


def fltrust_fix_jobs() -> list[ExperimentConfig]:
    """Re-run every CIFAR-10 FLTrust job of core with the BatchNorm-safe aggregator.
    The core runs rescaled BN running statistics and had 1-7 NaN-loss rounds per run.
    notes="bn-safe" gives them distinct job ids; MNIST (no BatchNorm) is unaffected."""
    return [replace(job, notes="bn-safe", tags=("FLTRUST_FIX",))
            for job in core_jobs() if job.method == "fltrust" and job.dataset == "CIFAR10"]


CC_MATCHED = {"method": "cc_only", "max_exclusion_fraction": 0.4, "consistency_z": 3.0, "notes": "matched z3 cap"}
FEDAVG = {"method": "fedavg"}


def rev_jobs() -> list[ExperimentConfig]:
    """Experiments requested by the internal review of paper v10 (priority order).

    P1 (seed-major): report faking under strong skew; CC only with the same z and cap as
    AntiFLipper; clean FedAvg controls. P2: partial flips. P3: sensitivity of the full method.
    """
    def job(dataset, distribution, seed, experiment, variant, tags, **overrides):
        return replace(base(dataset, distribution, variant["method"], seed, experiment),
                       tags=tags, **{**{k: v for k, v in variant.items() if k != "method"}, **overrides})

    p1 = (
        ("A8_adaptive_skew", {"dirichlet_alpha": 0.1, "report_strategy": "clean_labels"}, (CR_CC, CR, CC_MATCHED)),
        ("A1_headline", {}, (CC_MATCHED,)),
        ("A2_attacked_heterogeneity", {"dirichlet_alpha": 0.1}, (CC_MATCHED,)),
        ("A2_clean_heterogeneity", {"dirichlet_alpha": 0.1, "malicious_client_fraction": 0.0}, (CC_MATCHED, FEDAVG)),
        ("A2_clean_heterogeneity", {"dirichlet_alpha": 0.5, "malicious_client_fraction": 0.0}, (FEDAVG,)),
        ("A2_clean_heterogeneity", {"dirichlet_alpha": 1.0, "malicious_client_fraction": 0.0}, (FEDAVG,)),
    )
    jobs: list[ExperimentConfig] = []
    for seed in SEEDS:
        for experiment, overrides, variants in p1:
            for variant in variants:
                jobs.append(job("CIFAR10", "NON_IID", seed, experiment, variant, ("REV", "P1"), **overrides))
    for seed, fraction, variant in itertools.product(SEEDS, (0.5, 0.25), (CR_CC, CR)):
        jobs.append(job("CIFAR10", "NON_IID", seed, "A4_partial_flip", variant, ("REV", "P2"), label_poison_fraction=fraction))
    sensitivity = [{"notes": "default"}, {"consistency_z": 2.5, "notes": "z=2.5"}, {"consistency_z": 3.5, "notes": "z=3.5"},
                   {"counter_threshold": 3, "notes": "counter=3"}, {"counter_threshold": 9, "notes": "counter=9"}]
    for seed, ratio, change in itertools.product(SENSITIVITY_SEEDS, (0.4, 0.0), sensitivity):
        jobs.append(job("MNIST", "NON_IID", seed, "A9_sensitivity_full", {**CR_CC, **change}, ("REV", "P3"),
                        malicious_client_fraction=ratio))
    return _dedupe(jobs)


def build(profile: str) -> list[ExperimentConfig]:
    if profile == "smoke":
        return smoke_jobs()
    if profile == "cr":
        return cr_jobs()
    if profile == "fltrust_fix":
        return fltrust_fix_jobs()
    if profile == "rev":
        return rev_jobs()
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    reference = core_jobs()
    cc_v2 = cc_v2_jobs(reference)
    if profile == "cc_v2":
        return cc_v2
    first_sweep = next(i for i, job in enumerate(reference) if job.experiment == "A2_clean_heterogeneity")
    core = reference[:first_sweep] + cc_v2 + reference[first_sweep:]
    return core if profile == "core" else _dedupe(core + optional_jobs())


def save_manifest(path: Path, profile: str, jobs: list[ExperimentConfig]) -> None:
    payload = {"schema_version": 2, "profile": profile, "job_count": len(jobs), "jobs": [x.canonical() for x in jobs]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path) -> tuple[str, list[ExperimentConfig]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["profile"], [ExperimentConfig.from_dict(x) for x in value["jobs"]]
