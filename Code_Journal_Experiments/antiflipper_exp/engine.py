from __future__ import annotations

import copy
import csv
import json
import os
import platform
import random
import time
import traceback
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .aggregators import AggregatorState, aggregate
from .attacks import PartiallyPoisonedDataset
from .config import ExperimentConfig, LABEL_MAP
from .data import FederatedData, build_federated_data
from .models import create_model


@dataclass
class ClientResult:
    client_id: int
    state: dict[str, torch.Tensor]
    local_loss: float
    reported_accuracy: float
    attacked: bool
    poisoned_examples: int


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=max(1, min(batch_size, len(dataset))), shuffle=shuffle, generator=generator, num_workers=0)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = total = 0
    loss_total = 0.0
    mapped_success = 0
    mapped_eligible = 0
    class_correct = np.zeros(10, dtype=np.int64)
    class_total = np.zeros(10, dtype=np.int64)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for data, target in _loader(dataset, batch_size, False, 0):
        data, target = data.to(device), torch.as_tensor(target, device=device, dtype=torch.long)
        output = model(data)
        prediction = output.argmax(dim=1)
        loss_total += float(criterion(output, target))
        correct += int((prediction == target).sum())
        total += len(target)
        mapped = torch.as_tensor([LABEL_MAP[int(x)] for x in target], device=device)
        mapped_success += int((prediction == mapped).sum())
        mapped_eligible += len(target)
        for cls in range(10):
            mask = target == cls
            class_total[cls] += int(mask.sum())
            class_correct[cls] += int(((prediction == target) & mask).sum())
    valid = class_total > 0
    per_class = np.divide(class_correct[valid], class_total[valid]) if valid.any() else np.asarray([0.0])
    return {
        "loss": loss_total / max(total, 1),
        "accuracy": correct / max(total, 1),
        "asr": mapped_success / max(mapped_eligible, 1),
        "macro_class_accuracy": float(per_class.mean()),
        "worst_class_accuracy": float(per_class.min()),
    }


def _evaluation_subset(dataset: Dataset, fraction: float, seed: int) -> Dataset:
    if fraction >= 1:
        return dataset
    count = max(1, int(round(len(dataset) * fraction)))
    rng = np.random.default_rng(seed)
    return Subset(dataset, sorted(int(x) for x in rng.choice(len(dataset), count, replace=False)))


def train_local(
    cfg: ExperimentConfig,
    model_template: nn.Module,
    global_state: dict[str, torch.Tensor],
    dataset: Dataset,
    client_id: int,
    round_index: int,
    attacked: bool,
    device: torch.device,
) -> ClientResult:
    local_seed = cfg.seed * 1_000_003 + round_index * 10_007 + client_id * 101
    seed_everything(local_seed)
    training_data: Dataset = dataset
    poisoned_examples = 0
    if attacked:
        poisoned = PartiallyPoisonedDataset(dataset, cfg.label_poison_fraction, local_seed + 17)
        training_data = poisoned
        poisoned_examples = poisoned.poisoned_examples

    # The report evaluates the pre-training global model using the same labels visible
    # to local training in this client-round, matching the submitted method.
    model = copy.deepcopy(model_template).to(device)
    model.load_state_dict(global_state)
    eval_data = _evaluation_subset(training_data, cfg.eval_fraction, local_seed + 23)
    reported_accuracy = evaluate(model, eval_data, cfg.test_batch_size, device)["accuracy"]

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    for local_epoch in range(cfg.local_epochs):
        loader = _loader(training_data, cfg.local_batch_size, True, local_seed + local_epoch)
        for data, target in loader:
            data, target = data.to(device), torch.as_tensor(target, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    del model
    return ClientResult(client_id, state, float(np.mean(losses)), reported_accuracy, attacked, poisoned_examples)


def _attack_occurs(cfg: ExperimentConfig, client_id: int, round_index: int) -> bool:
    rng = np.random.default_rng(cfg.seed * 2_000_033 + round_index * 20_011 + client_id * 211)
    return bool(rng.random() < cfg.attack_round_probability)


def _confusion(detected: set[int], attackers: set[int], num_clients: int) -> dict[str, float]:
    universe = set(range(num_clients))
    tp = len(detected & attackers); fp = len(detected - attackers)
    fn = len(attackers - detected); tn = len((universe - attackers) - detected)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "fpr": fp / (fp + tn) if fp + tn else 0.0}


def _normalize_trust(trust: dict[int, float], excluded: set[int]) -> None:
    for client_id in excluded:
        trust[client_id] = 0.0
    eligible = [i for i in trust if i not in excluded]
    total = sum(max(0.0, trust[i]) for i in eligible)
    if not eligible:
        raise RuntimeError("AntiFLipper excluded every client")
    if total <= 1e-15:
        uniform = 1.0 / len(eligible)
        for i in eligible:
            trust[i] = uniform
        return
    for i in eligible:
        trust[i] = max(0.0, trust[i]) / total


def _update_trust(
    cfg: ExperimentConfig,
    reports: dict[int, float],
    trust: dict[int, float],
    counters: dict[int, int],
    excluded: set[int],
    round_index: int,
    detection_round: dict[int, int],
) -> None:
    if not reports:
        raise RuntimeError("No eligible accuracy reports")
    average = float(np.mean(list(reports.values())))
    for client_id, accuracy in reports.items():
        difference = accuracy - average
        magnitude = abs(difference) if cfg.trust_update == "linear" else difference * difference
        trust[client_id] = max(0.0, trust[client_id] + (cfg.eta * magnitude if difference >= 0 else -cfg.eta * magnitude))
    _normalize_trust(trust, excluded)

    if round_index < cfg.grace_rounds:
        return
    n_eligible = len(reports)
    newly_excluded: set[int] = set()
    for client_id in reports:
        below = (n_eligible * trust[client_id] < cfg.rho_threshold) if cfg.threshold_mode == "relative" else (trust[client_id] < cfg.absolute_threshold)
        if cfg.counter_mode == "none":
            counters[client_id] = cfg.counter_threshold if below else 0
        elif below:
            counters[client_id] += 1
        elif cfg.counter_mode == "consecutive":
            counters[client_id] = 0
        if counters[client_id] >= cfg.counter_threshold:
            newly_excluded.add(client_id)
            detection_round.setdefault(client_id, round_index)
    excluded.update(newly_excluded)
    _normalize_trust(trust, excluded)


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
    else:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _checkpoint_payload(
    round_index: int,
    model: nn.Module,
    trust: dict[int, float],
    counters: dict[int, int],
    excluded: set[int],
    detection_round: dict[int, int],
    rows: list[dict[str, Any]],
    aggregator_state: AggregatorState,
) -> dict[str, Any]:
    return {
        "next_round": round_index + 1,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "trust": trust, "counters": counters, "excluded": excluded,
        "detection_round": detection_round, "rows": rows,
        "aggregator_state": aggregator_state.state_dict(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def run_experiment(cfg: ExperimentConfig, job_dir: Path, resume: bool = True) -> dict[str, Any]:
    cfg.validate(); job_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(job_dir / "config.json")
    checkpoint_path = job_dir / "checkpoint.pt"
    metrics_path = job_dir / "round_metrics.csv"
    state_path = job_dir / "state.json"
    state_path.write_text(json.dumps({"status": "running", "job_id": cfg.job_id, "started_at": time.time()}, indent=2), encoding="utf-8")
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    data: FederatedData = build_federated_data(cfg)
    model = create_model(cfg.dataset, cfg.model).to(device)
    attackers_count = int(cfg.malicious_client_fraction * cfg.num_clients)
    attacker_rng = np.random.default_rng(cfg.seed + 307)
    attackers = set(int(x) for x in attacker_rng.choice(cfg.num_clients, attackers_count, replace=False)) if attackers_count else set()

    # Identity and partition invariants required by the methodology audit.
    assert set(data.clients) == set(range(cfg.num_clients))
    assert set(data.partition_indices) == set(data.clients)
    assert all(data.clients[i].indices == data.partition_indices[i] for i in data.clients)
    partition_metadata = {
        "client_sizes": {str(i): len(indices) for i, indices in data.partition_indices.items()},
        "client_index_sha256": {str(i): hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest() for i, indices in data.partition_indices.items()},
        "root_size": len(data.root) if data.root is not None else 0,
        "root_index_sha256": hashlib.sha256(np.asarray(data.root.indices, dtype=np.int64).tobytes()).hexdigest() if data.root is not None else None,
        "attackers": sorted(attackers),
    }
    (job_dir / "partition_metadata.json").write_text(json.dumps(partition_metadata, indent=2), encoding="utf-8")

    trust = {i: 1.0 / cfg.num_clients for i in range(cfg.num_clients)}
    counters = {i: 0 for i in range(cfg.num_clients)}
    excluded: set[int] = set()
    detection_round: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    aggregator_state = AggregatorState()
    start_round = 0
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_round = int(payload["next_round"]); model.load_state_dict(payload["model"])
        trust = payload["trust"]; counters = payload["counters"]; excluded = set(payload["excluded"])
        detection_round = payload["detection_round"]; rows = payload["rows"]
        aggregator_state.load_state_dict(payload["aggregator_state"])
        random.setstate(payload["python_rng"]); np.random.set_state(payload["numpy_rng"]); torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        _write_rows(metrics_path, rows)

    run_started = time.perf_counter()
    try:
        for round_index in range(start_round, cfg.rounds):
            round_started = time.perf_counter()
            eligible = sorted(set(range(cfg.num_clients)) - excluded)
            count = max(1, int(round(cfg.participation * len(eligible))))
            selection_rng = np.random.default_rng(cfg.seed * 3_000_017 + round_index)
            selected = sorted(int(x) for x in selection_rng.choice(eligible, count, replace=False))
            global_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            client_results: list[ClientResult] = []
            for client_id in selected:
                attacked = client_id in attackers and _attack_occurs(cfg, client_id, round_index)
                client_results.append(train_local(cfg, model, global_cpu, data.clients[client_id], client_id, round_index, attacked, device))

            reports = {x.client_id: x.reported_accuracy for x in client_results}
            if cfg.method == "antiflipper":
                _update_trust(cfg, reports, trust, counters, excluded, round_index, detection_round)
                admitted = [x for x in client_results if x.client_id not in excluded]
            else:
                admitted = client_results
            if not admitted:
                raise RuntimeError("No client updates remain for aggregation")

            root_state = None
            if cfg.method == "fltrust":
                if data.root is None:
                    raise RuntimeError("FLTrust root data missing")
                root_state = train_local(cfg, model, global_cpu, data.root, -1, round_index, False, device).state

            if device.type == "cuda": torch.cuda.synchronize()
            aggregation_started = time.perf_counter()
            new_state, method_weights = aggregate(
                cfg.method, [x.state for x in admitted], global_cpu, [x.client_id for x in admitted],
                cfg.malicious_client_fraction, aggregator_state,
                trust_weights=[trust[x.client_id] for x in admitted] if cfg.method == "antiflipper" else None,
                root_state=root_state, seed=cfg.seed + round_index,
            )
            if device.type == "cuda": torch.cuda.synchronize()
            aggregation_seconds = time.perf_counter() - aggregation_started
            model.load_state_dict(new_state)
            test = evaluate(model, data.test, cfg.test_batch_size, device)
            confusion = _confusion(excluded if cfg.method == "antiflipper" else set(), attackers, cfg.num_clients)
            malicious_delays = [detection_round[i] for i in attackers if i in detection_round]
            row = {
                "round": round_index, **test,
                "local_loss": float(np.mean([x.local_loss for x in client_results])),
                "selected_clients": len(selected), "aggregated_clients": len(admitted), "excluded_clients": len(excluded),
                "attacked_client_rounds": sum(x.attacked for x in client_results),
                "poisoned_examples": sum(x.poisoned_examples for x in client_results),
                "aggregation_seconds": aggregation_seconds,
                "round_seconds": time.perf_counter() - round_started,
                "mean_method_weight": float(np.mean(method_weights)),
                **confusion,
                "mean_detection_round": float(np.mean(malicious_delays)) if malicious_delays else "",
            }
            rows.append(row); _write_rows(metrics_path, rows)
            (job_dir / "latest_detection.json").write_text(json.dumps({
                "attackers": sorted(attackers), "excluded": sorted(excluded), "detection_round": detection_round,
                "trust": trust, "counters": counters,
            }, indent=2), encoding="utf-8")
            if (round_index + 1) % cfg.checkpoint_every == 0 or round_index + 1 == cfg.rounds:
                _atomic_torch_save(_checkpoint_payload(round_index, model, trust, counters, excluded, detection_round, rows, aggregator_state), checkpoint_path)

        final = rows[-1]
        last_window = rows[-min(10, len(rows)):]
        summary = {
            "job_id": cfg.job_id, "status": "completed", "config_hash": cfg.config_hash,
            "final_accuracy": final["accuracy"], "last10_accuracy_mean": float(np.mean([x["accuracy"] for x in last_window])),
            "last10_accuracy_std": float(np.std([x["accuracy"] for x in last_window], ddof=1)) if len(last_window) > 1 else 0.0,
            "final_asr": final["asr"], "accuracy_auc": float(np.mean([x["accuracy"] for x in rows])),
            "final_precision": final["precision"], "final_recall": final["recall"], "final_f1": final["f1"], "final_fpr": final["fpr"],
            "mean_aggregation_seconds": float(np.mean([x["aggregation_seconds"] for x in rows])),
            "mean_round_seconds": float(np.mean([x["round_seconds"] for x in rows])),
            "total_seconds": time.perf_counter() - run_started, "attackers": sorted(attackers), "excluded": sorted(excluded),
            "detection_round": detection_round, "device": str(device), "torch_version": torch.__version__, "platform": platform.platform(),
        }
        (job_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state_path.write_text(json.dumps({"status": "completed", "job_id": cfg.job_id, "finished_at": time.time()}, indent=2), encoding="utf-8")
        return summary
    except KeyboardInterrupt:
        state_path.write_text(json.dumps({"status": "interrupted", "job_id": cfg.job_id, "interrupted_at": time.time()}, indent=2), encoding="utf-8")
        raise
    except BaseException as exc:
        (job_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        state_path.write_text(json.dumps({"status": "failed", "job_id": cfg.job_id, "error": repr(exc), "failed_at": time.time()}, indent=2), encoding="utf-8")
        raise
