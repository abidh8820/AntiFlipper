from __future__ import annotations

import copy
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .aggregators import AggregatorState, aggregate, update_consistency
from .config import ANTIFLIPPER_METHODS, CC_V2, CONSISTENCY_METHODS, ExperimentConfig
from .models import create_model


METHODS = ("antiflipper", "antiflipper_cc", "antiflipper_cr", "antiflipper_cr_cc", CC_V2, "cc_only", "fedavg", "median", "trimmed_mean", "foolsgold", "tolpegin", "multi_krum", "flame", "lfighter", "fltrust")


def _antiflipper_server_work(method: str, states: list[dict], base: dict, count: int, iteration: int) -> list[float]:
    """Server-side work AntiFLipper does before aggregation, so the benchmark is not
    just a weighted average: (cc) consistency check + trust update + thresholding."""
    from .engine import ClientResult, NUM_CLASSES, _class_residuals, _update_cc_v2, _update_trust

    inconsistent = set()
    z_cut = 3.0 if method == "antiflipper_cr_cc" else 2.5
    if method in CONSISTENCY_METHODS:
        _, flags = update_consistency(states, base, z_cut)
        inconsistent = set(np.flatnonzero(flags))
    cfg = ExperimentConfig("bench", "MNIST", "IID", method, 0, rounds=1, num_clients=count,
                           **({"max_exclusion_fraction": 0.4} if method.startswith("antiflipper_cr") else {}))
    rng = np.random.default_rng(iteration)
    reports = {i: float(x) for i, x in enumerate(rng.uniform(0.5, 1.0, count))}
    trust = {i: 1.0 / count for i in range(count)}
    if method.startswith("antiflipper_cr"):
        # Class-conditional residual from synthetic per-class counts (same work as in training).
        results = []
        for i in range(count):
            n = rng.integers(0, 300, NUM_CLASSES)
            results.append(ClientResult(i, {}, 0.0, reports[i], False, 0,
                                        class_correct=np.floor(n * rng.uniform(0.3, 1.0, NUM_CLASSES)), class_count=n))
        reports = _class_residuals(results, trust)
    counters = dict.fromkeys(reports, 0)
    if method == CC_V2:
        _update_cc_v2(cfg, reports, inconsistent, reports.copy(), trust, counters, cfg.grace_rounds)
    else:
        if method in ("antiflipper_cc", "antiflipper_cr_cc") and inconsistent:
            floor = min(reports.values())
            for client_id in inconsistent:
                reports[client_id] = floor
        _update_trust(cfg, reports, trust, counters, set(), cfg.grace_rounds, {}, inconsistent)
    return list(trust.values())


def _states(dataset: str, count: int) -> tuple[dict, list[dict], dict, tuple[str, ...]]:
    model = create_model(dataset)
    parameter_keys = tuple(name for name, param in model.named_parameters() if param.requires_grad)
    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    states: list[dict] = []
    generator = torch.Generator().manual_seed(991 + count)
    for index in range(count):
        value = copy.deepcopy(base)
        for key, tensor in value.items():
            if torch.is_floating_point(tensor):
                # Small deterministic differences keep clustering/cosine methods well-defined.
                value[key] = tensor + torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * (1e-5 + index * 1e-7)
        states.append(value)
    root = copy.deepcopy(base)
    for key, tensor in root.items():
        if torch.is_floating_point(tensor): root[key] = tensor + torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * 1e-5
    return base, states, root, parameter_keys


def fixed_workload_benchmark(output: Path, iterations: int = 5) -> list[dict]:
    """Time every method on identical model states and submitted-update counts."""
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    # ResNet18 is kept at its paper workload (20 clients); MNIST supplies N scaling.
    for dataset, counts in (("MNIST", (20, 50, 100)), ("CIFAR10", (20,))):
        for count in counts:
            base, states, root, parameter_keys = _states(dataset, count)
            parameters = sum(v.numel() for v in base.values())
            client_ids = list(range(count))
            for method in METHODS:
                times: list[float] = []
                error = ""
                try:
                    runtime = AggregatorState()
                    for iteration in range(iterations + 1):
                        started = time.perf_counter()
                        trust = _antiflipper_server_work(method, states, base, count, iteration) if method in ANTIFLIPPER_METHODS else None
                        aggregate(method, states, base, client_ids, 0.4, runtime, trust_weights=trust,
                                  root_state=root, seed=iteration, parameter_keys=parameter_keys)
                        elapsed = time.perf_counter() - started
                        if iteration: times.append(elapsed)
                except Exception as exc:
                    error = repr(exc)
                rows.append({
                    "dataset": dataset, "method": method, "parameters": parameters, "updates": count,
                    "iterations": len(times), "mean_seconds": statistics.fmean(times) if times else "",
                    "std_seconds": statistics.stdev(times) if len(times) > 1 else (0.0 if times else ""), "error": error,
                    "workload": "identical deterministic synthetic model states on CPU; malicious_fraction=0.4; AntiFLipper rows include trust update (+class-conditional residual for _cr, +consistency check for _cc); FLTrust excludes root training",
                })
            del states, base, root
    with (output / "fixed_workload_timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "fixed_workload_timing.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
