from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def folder_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    if seconds < 60:
        return f"{round(seconds)}s"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    days, remainder = divmod(minutes, 1440)
    hours, minutes = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    return f"{hours}h {minutes}m"


def format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return "0 B"


def metrics_for(job_dir: Path) -> list[dict]:
    path = job_dir / "round_metrics.csv"
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def snapshot(profile: str) -> str:
    manifest = read_json(ROOT / "manifests" / f"{profile}.json")
    jobs = manifest.get("jobs", [])
    result_root = ROOT / "results" / profile
    counts = Counter()
    unfinished = []
    dataset_rates: dict[str, list[float]] = defaultdict(list)
    method_rates: dict[tuple[str, str], list[float]] = defaultdict(list)

    for config in jobs:
        job_dir = result_root / job_id(config)
        status = read_json(job_dir / "state.json").get("status", "pending")
        counts[status] += 1
        rows = metrics_for(job_dir)
        recent = [float(row["round_seconds"]) for row in rows[-10:] if row.get("round_seconds")]
        dataset = config["dataset"]
        method = config["method"]
        if recent:
            dataset_rates[dataset].extend(recent)
            method_rates[dataset, method].extend(recent)
        if status != "completed":
            unfinished.append((config, status, rows, recent))

    all_rates = [rate for rates in dataset_rates.values() for rate in rates]
    estimated_work = []
    active_lines = []
    unmeasured_datasets = set()
    for config, status, rows, recent in unfinished:
        dataset = config["dataset"]
        method = config["method"]
        total_rounds = config["rounds"]
        done_rounds = len(rows)
        remaining_rounds = max(0, total_rounds - done_rounds)
        dataset_sample = dataset_rates.get(dataset)
        samples = recent or method_rates.get((dataset, method)) or dataset_sample or all_rates
        average = fmean(samples) if samples else None
        eta = remaining_rounds * average if average is not None else None
        if eta is not None:
            estimated_work.append(eta)
            if remaining_rounds and not dataset_sample:
                unmeasured_datasets.add(dataset)

        if status != "running":
            continue
        latest = rows[-1] if rows else {}
        client_eval = latest.get("client_eval_seconds")
        client_train = latest.get("client_train_seconds")
        client_timing = ""
        if client_eval and client_train:
            client_timing = (
                f"\n  Last round per client: {float(client_train):.2f}s training, "
                f"{float(client_eval):.2f}s accuracy report (averages)"
            )
        round_time = f"{average:.1f}s" if average is not None else "collecting data"
        job_eta = format_duration(eta) if eta is not None else "waiting for round timing"
        active_lines.append(
            f"{config['experiment']} | {dataset} {config['distribution']} | {method} | seed {config['seed']}\n"
            f"  Rounds: {done_rounds}/{total_rounds} ({100 * done_rounds / max(1, total_rounds):.1f}%)\n"
            f"  Estimated round time: {round_time}\n"
            f"  Estimated time left: {job_eta}{client_timing}"
        )

    active_text = "\n\n".join(active_lines) if active_lines else "No job is currently running."
    if not unfinished and jobs:
        queue_eta = "complete"
    elif not active_lines:
        queue_eta = "queue idle"
    elif not estimated_work:
        queue_eta = "waiting for the first timed round"
    else:
        # Workers share the queue, but a single job cannot use more than one worker.
        total_eta = max(max(estimated_work), sum(estimated_work) / len(active_lines))
        queue_eta = f"{format_duration(total_eta)} (rough estimate)"
        if unmeasured_datasets:
            queue_eta += f"\n  Using other datasets' timings for {', '.join(sorted(unmeasured_datasets))}."

    total = len(jobs)
    completed = counts.get("completed", 0)
    progress = 100 * completed / max(1, total)
    usage = shutil.disk_usage(ROOT.anchor or ROOT)
    tracked_bytes = sum(folder_bytes(ROOT / name) for name in ("results", "analysis", "packages"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "=" * 72 + "\n"
        f"AntiFLipper monitor | {profile} | updated {now}\n"
        + "=" * 72 + "\n"
        f"Jobs: {completed}/{total} complete ({progress:.1f}%) | "
        f"{counts.get('running', 0)} running | {counts.get('pending', 0)} pending | "
        f"{counts.get('failed', 0)} failed | {counts.get('interrupted', 0)} interrupted\n"
        f"Current jobs:\n{active_text}\n"
        f"Estimated time left for all jobs: {queue_eta}\n"
        f"Stored results/checkpoints/archives: {format_bytes(tracked_bytes)}\n"
        f"Free space on {ROOT.anchor or 'drive'}: {format_bytes(usage.free)}\n"
        "\nPress Ctrl+C to close this monitor. It does not stop the experiment.\n"
    )


def job_id(config: dict) -> str:
    job_dir_name = config.get("job_id")
    if job_dir_name:
        return job_dir_name
    import hashlib

    canonical = dict(config)
    canonical["tags"] = list(canonical.get("tags", ()))
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(raw.encode()).hexdigest()[:12]
    fields = [config["experiment"], config["dataset"], config["distribution"], config["method"], f"s{config['seed']}", config_hash]
    return "__".join(str(value).lower().replace(".", "p") for value in fields)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live AntiFLipper experiment progress monitor")
    parser.add_argument("--profile", choices=("smoke", "core", "full", "cc_v2", "cr", "fltrust_fix"), default="core")
    parser.add_argument("--watch", action="store_true", help="refresh continuously")
    parser.add_argument("--interval", type=int, default=10, help="refresh interval in seconds")
    args = parser.parse_args()
    ansi_watch = args.watch and os.name != "nt"
    alternate_screen = ansi_watch and sys.stdout.isatty()
    if alternate_screen:
        sys.stdout.write("\033[?1049h")
    try:
        while True:
            frame = snapshot(args.profile)
            if ansi_watch:
                frame = "\033[2J\033[H" + frame
            elif args.watch and sys.stdout.isatty():
                os.system("cls")
            print(frame, flush=True)
            if not args.watch:
                return 0
            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        return 0
    finally:
        if alternate_screen:
            sys.stdout.write("\033[?1049l")
            sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
