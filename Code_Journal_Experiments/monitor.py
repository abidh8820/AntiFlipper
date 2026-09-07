from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

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
    active: list[dict] = []
    rates: dict[str, list[float]] = defaultdict(list)

    for config in jobs:
        job_dir = result_root / job_id(config)
        state = read_json(job_dir / "state.json")
        status = state.get("status", "pending")
        counts[status] += 1
        rows = metrics_for(job_dir)
        if rows:
            values = [float(row["round_seconds"]) for row in rows if row.get("round_seconds")]
            if values:
                rates[config.get("dataset", "unknown")].extend(values)
        if status == "running":
            active.append({"config": config, "state": state, "rows": rows, "job_dir": job_dir})

    current_eta = None
    total_eta = None
    active_text = "No job is currently running.\n"
    if active:
        active_lines: list[str] = []
        total_work_seconds = 0.0
        fallback_values: list[float] = []
        for item in active:
            config = item["config"]
            rows = item["rows"]
            values = [float(row["round_seconds"]) for row in rows if row.get("round_seconds")]
            average = sum(values) / len(values) if values else None
            if values:
                fallback_values.extend(values)
            done_rounds = len(rows)
            total_rounds = int(config.get("rounds", 0))
            remaining_rounds = max(0, total_rounds - done_rounds)
            current_eta = remaining_rounds * average if average is not None else None
            if current_eta is not None:
                total_work_seconds += current_eta
            active_lines.append(
                f"{config.get('experiment')} | {config.get('dataset')} {config.get('distribution')} | "
                f"{config.get('method')} | seed {config.get('seed')}\n"
                f"  Rounds: {done_rounds}/{total_rounds} ({100 * done_rounds / max(1, total_rounds):.1f}%)\n"
                f"  Average round: {average:.1f}s\n"
                f"  Remaining: {format_duration(current_eta)}"
                if average is not None else
                f"{config.get('experiment')} | {config.get('dataset')} | {config.get('method')}\n"
                f"  Rounds: {done_rounds}/{total_rounds}\n"
                f"  Average round: collecting data"
            )
        active_text = "\n\n".join(active_lines) + "\n"

        for pending in jobs:
            pending_dir = result_root / job_id(pending)
            pending_state = read_json(pending_dir / "state.json").get("status", "pending")
            if pending_state in {"completed", "running"}:
                continue
            dataset = pending.get("dataset", "unknown")
            dataset_values = rates.get(dataset) or fallback_values
            dataset_average = sum(dataset_values) / len(dataset_values) if dataset_values else None
            if dataset_average is not None:
                total_work_seconds += int(pending.get("rounds", 0)) * dataset_average
        # Approximate wall time by sharing the remaining work among active workers.
        total_eta = total_work_seconds / max(1, len(active))

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
        f"Current job:\n{active_text}"
        f"Estimated total remaining: {format_duration(total_eta)} (rough estimate)\n"
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
    parser.add_argument("--profile", choices=("smoke", "core", "full"), default="core")
    parser.add_argument("--watch", action="store_true", help="refresh continuously")
    parser.add_argument("--interval", type=int, default=10, help="refresh interval in seconds")
    args = parser.parse_args()
    while True:
        if args.watch:
            os.system("cls")
        print(snapshot(args.profile), flush=True)
        if not args.watch:
            return 0
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
