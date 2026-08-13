from __future__ import annotations

import copy
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .aggregators import AggregatorState, aggregate
from .models import create_model


METHODS = ("antiflipper", "fedavg", "median", "trimmed_mean", "foolsgold", "tolpegin", "multi_krum", "flame", "lfighter", "fltrust")


def _states(dataset: str, count: int) -> tuple[dict, list[dict], dict]:
    model = create_model(dataset)
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
    return base, states, root


def fixed_workload_benchmark(output: Path, iterations: int = 5) -> list[dict]:
    """Time every method on identical model states and submitted-update counts."""
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    # ResNet18 is kept at its paper workload (20 clients); MNIST supplies N scaling.
    for dataset, counts in (("MNIST", (20, 50, 100)), ("CIFAR10", (20,))):
        for count in counts:
            base, states, root = _states(dataset, count)
            parameters = sum(v.numel() for v in base.values())
            client_ids = list(range(count)); trust = np.ones(count) / count
            for method in METHODS:
                times: list[float] = []
                error = ""
                try:
                    runtime = AggregatorState()
                    for iteration in range(iterations + 1):
                        started = time.perf_counter()
                        aggregate(method, states, base, client_ids, 0.4, runtime, trust_weights=trust, root_state=root, seed=iteration)
                        elapsed = time.perf_counter() - started
                        if iteration: times.append(elapsed)
                except Exception as exc:
                    error = repr(exc)
                rows.append({
                    "dataset": dataset, "method": method, "parameters": parameters, "updates": count,
                    "iterations": len(times), "mean_seconds": statistics.fmean(times) if times else "",
                    "std_seconds": statistics.stdev(times) if len(times) > 1 else (0.0 if times else ""), "error": error,
                    "workload": "identical deterministic synthetic model states on CPU; malicious_fraction=0.4",
                })
            del states, base, root
    with (output / "fixed_workload_timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "fixed_workload_timing.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
