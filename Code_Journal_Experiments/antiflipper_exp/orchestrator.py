from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .analysis import analyze
from .config import ExperimentConfig
from .engine import run_experiment
from .manifest import build, load_manifest, save_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
RESULTS = ROOT / "results"
ANALYSIS = ROOT / "analysis"
PACKAGES = ROOT / "packages"


class Tee:
    def __init__(self, *handles): self.handles = handles
    def write(self, value):
        for handle in self.handles: handle.write(value); handle.flush()
        return len(value)
    def flush(self):
        for handle in self.handles: handle.flush()


def prepare_manifest(profile: str, force: bool = False) -> Path:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS / f"{profile}.json"
    if force or not path.exists():
        save_manifest(path, profile, build(profile))
    return path


def job_status(job_dir: Path) -> str:
    state = job_dir / "state.json"
    if not state.exists(): return "pending"
    try: return json.loads(state.read_text(encoding="utf-8")).get("status", "unknown")
    except (json.JSONDecodeError, OSError): return "unknown"


def status_rows(manifest_path: Path) -> tuple[str, list[ExperimentConfig], list[dict]]:
    profile, jobs = load_manifest(manifest_path)
    result_root = RESULTS / profile
    rows = [{"index": i + 1, "job_id": cfg.job_id, "experiment": cfg.experiment, "dataset": cfg.dataset, "distribution": cfg.distribution, "method": cfg.method, "seed": cfg.seed, "status": job_status(result_root / cfg.job_id)} for i, cfg in enumerate(jobs)]
    return profile, jobs, rows


def write_checklist(manifest_path: Path) -> Path:
    profile, _, rows = status_rows(manifest_path)
    counts = {name: sum(r["status"] == name for r in rows) for name in ("completed", "running", "interrupted", "failed", "pending", "unknown")}
    lines = [f"# Experiment checklist — {profile}", "", f"Updated: {datetime.now(timezone.utc).isoformat()}", "", f"Progress: **{counts['completed']}/{len(rows)} complete**, {counts['failed']} failed, {counts['interrupted']} interrupted, {counts['running']} running.", "", "A checked item has a valid completed state file. Re-running the launcher skips it. Interrupted jobs resume from `checkpoint.pt`.", "", "| Done | # | Experiment | Dataset | Split | Method | Seed | Status |", "|---|---:|---|---|---|---|---:|---|"]
    for row in rows:
        mark = "x" if row["status"] == "completed" else " "
        lines.append(f"| [{mark}] | {row['index']} | {row['experiment']} | {row['dataset']} | {row['distribution']} | {row['method']} | {row['seed']} | {row['status']} |")
    path = ROOT / "CHECKLIST.md"
    temporary = path.with_suffix(".tmp"); temporary.write_text("\n".join(lines) + "\n", encoding="utf-8"); os.replace(temporary, path)
    with (ROOT / "checklist.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return path


def capture_environment(profile: str) -> None:
    output = RESULTS / profile / "environment"; output.mkdir(parents=True, exist_ok=True)
    info = {"captured_at": datetime.now(timezone.utc).isoformat(), "python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    try:
        import torch
        info.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception as exc: info["torch_error"] = repr(exc)
    (output / "system.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    try:
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60)
        (output / "pip_freeze.txt").write_text(frozen.stdout + frozen.stderr, encoding="utf-8")
    except Exception as exc: (output / "pip_freeze_error.txt").write_text(repr(exc), encoding="utf-8")


def run_manifest(manifest_path: Path, stop_on_error: bool = False, retry_failed: bool = True) -> int:
    profile, jobs = load_manifest(manifest_path)
    result_root = RESULTS / profile; result_root.mkdir(parents=True, exist_ok=True)
    capture_environment(profile); write_checklist(manifest_path)
    failures = 0
    try:
        for index, cfg in enumerate(jobs, 1):
            job_dir = result_root / cfg.job_id
            current = job_status(job_dir)
            if current == "completed":
                print(f"[{index}/{len(jobs)}] SKIP completed {cfg.job_id}")
                continue
            if current == "failed" and not retry_failed:
                print(f"[{index}/{len(jobs)}] SKIP failed {cfg.job_id}")
                continue
            job_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{index}/{len(jobs)}] RUN {cfg.job_id}")
            with (job_dir / "console.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                try: run_experiment(cfg, job_dir, resume=True)
                except KeyboardInterrupt: raise
                except BaseException as exc:
                    failures += 1; print(f"FAILED: {exc!r}")
                    if stop_on_error: raise
            write_checklist(manifest_path)
    except KeyboardInterrupt:
        print("Interrupted. The last completed round is checkpointed; run the same command to resume.")
        return 130
    finally:
        write_checklist(manifest_path)
        analyze(result_root, ANALYSIS / profile)
        package(manifest_path)
    return 1 if failures else 0


def package(manifest_path: Path) -> Path:
    profile, _, rows = status_rows(manifest_path)
    complete = sum(r["status"] == "completed" for r in rows) == len(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:10]
    PACKAGES.mkdir(parents=True, exist_ok=True)
    destination = PACKAGES / f"AntiFlipper_Q1_{profile}_{'COMPLETE' if complete else 'PARTIAL'}_{stamp}_{manifest_hash}.zip"
    include_roots = [manifest_path, ROOT / "CHECKLIST.md", ROOT / "checklist.csv", ANALYSIS / profile, RESULTS / profile]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in include_roots:
            if not source.exists(): continue
            paths = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
            for path in paths:
                # Checkpoints are for local resumption and are intentionally excluded from analysis archives.
                if path.name in {"checkpoint.pt"} or path.suffix == ".tmp": continue
                archive.write(path, path.relative_to(ROOT))
        for path in (ROOT / "antiflipper_exp").rglob("*.py"):
            archive.write(path, path.relative_to(ROOT))
        for name in ("README.md", "EXPERIMENTS.md", "requirements.txt", "run_all.cmd", "run_all.ps1", "cli.py", "self_test.py"):
            path = ROOT / name
            if path.exists(): archive.write(path, path.relative_to(ROOT))
    latest = PACKAGES / f"LATEST_{profile}.txt"; latest.write_text(destination.name, encoding="utf-8")
    print(f"Packaged analysis archive: {destination}")
    return destination
