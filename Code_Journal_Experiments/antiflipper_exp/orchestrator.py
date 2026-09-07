from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis import analyze
from .config import ExperimentConfig
from .engine import run_experiment
from .manifest import build, load_manifest, save_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
RESULTS = ROOT / "results"
ANALYSIS = ROOT / "analysis"
PACKAGES = ROOT / "packages"
_WORKER_STOP_EVENT: Any = None


class Tee:
    def __init__(self, *handles): self.handles = handles
    def write(self, value):
        for handle in self.handles: handle.write(value); handle.flush()
        return len(value)
    def flush(self):
        for handle in self.handles: handle.flush()


def _initialize_worker(stop_event: Any, worker_count: int) -> None:
    global _WORKER_STOP_EVENT
    _WORKER_STOP_EVENT = stop_event
    # Prevent N worker processes from each claiming every host CPU thread.
    try:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // worker_count))
    except Exception:
        pass


def _worker_stop_requested() -> bool:
    return _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set()


def _run_one_job(index: int, total: int, cfg: ExperimentConfig, result_root: Path) -> dict[str, Any]:
    """Run one uniquely assigned job. Shared reporting stays in the parent."""
    if _worker_stop_requested():
        return {"index": index, "job_id": cfg.job_id, "status": "cancelled"}
    job_dir = result_root / cfg.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / "console.log").open("a", encoding="utf-8") as log:
        stream = Tee(sys.stdout, log)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            print(f"[{index}/{total}] RUN {cfg.job_id}")
            try:
                run_experiment(cfg, job_dir, resume=True, stop_requested=_worker_stop_requested)
                return {"index": index, "job_id": cfg.job_id, "status": "completed"}
            except KeyboardInterrupt:
                print("INTERRUPTED at a safe round boundary")
                return {"index": index, "job_id": cfg.job_id, "status": "interrupted"}
            except BaseException as exc:
                print(f"FAILED: {exc!r}")
                return {"index": index, "job_id": cfg.job_id, "status": "failed", "error": repr(exc)}
            finally:
                # Release cached allocations before this process accepts another job.
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass


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


def capture_environment(profile: str, workers: int = 1) -> None:
    output = RESULTS / profile / "environment"; output.mkdir(parents=True, exist_ok=True)
    info = {"captured_at": datetime.now(timezone.utc).isoformat(), "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "experiment_workers": workers}
    try:
        import torch
        info.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception as exc: info["torch_error"] = repr(exc)
    (output / "system.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    try:
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60)
        (output / "pip_freeze.txt").write_text(frozen.stdout + frozen.stderr, encoding="utf-8")
    except Exception as exc: (output / "pip_freeze_error.txt").write_text(repr(exc), encoding="utf-8")


@contextlib.contextmanager
def _profile_run_lock(profile: str):
    """Prevent two launchers from scheduling the same profile concurrently."""
    lock_root = ROOT / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{profile}.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        details = ""
        try:
            details = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        suffix = f" ({details})" if details else ""
        raise RuntimeError(
            f"Another {profile!r} launcher appears to be active{suffix}. "
            f"Do not start two launchers. If the previous process crashed, verify it is stopped and delete {lock_path}."
        ) from exc
    try:
        payload = {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}
        os.write(descriptor, json.dumps(payload).encode("utf-8"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def run_manifest(manifest_path: Path, stop_on_error: bool = False, retry_failed: bool = True, workers: int = 1) -> int:
    profile, _ = load_manifest(manifest_path)
    with _profile_run_lock(profile):
        return _run_manifest(manifest_path, stop_on_error, retry_failed, workers)


def _run_manifest(manifest_path: Path, stop_on_error: bool = False, retry_failed: bool = True, workers: int = 1) -> int:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    profile, jobs = load_manifest(manifest_path)
    result_root = RESULTS / profile; result_root.mkdir(parents=True, exist_ok=True)
    capture_environment(profile, workers); write_checklist(manifest_path)
    failures = 0
    interrupted = False
    try:
        runnable: list[tuple[int, ExperimentConfig]] = []
        for index, cfg in enumerate(jobs, 1):
            job_dir = result_root / cfg.job_id
            current = job_status(job_dir)
            if current == "completed":
                print(f"[{index}/{len(jobs)}] SKIP completed {cfg.job_id}")
                continue
            if current == "failed" and not retry_failed:
                print(f"[{index}/{len(jobs)}] SKIP failed {cfg.job_id}")
                continue
            runnable.append((index, cfg))

        if not runnable:
            print("No runnable jobs remain.")
        elif workers == 1:
            # Preserve the original in-process behavior for the default mode.
            for index, cfg in runnable:
                job_dir = result_root / cfg.job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                print(f"[{index}/{len(jobs)}] RUN {cfg.job_id}")
                with (job_dir / "console.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                    try: run_experiment(cfg, job_dir, resume=True)
                    except KeyboardInterrupt: raise
                    except BaseException as exc:
                        failures += 1; print(f"FAILED: {exc!r}")
                        if stop_on_error: raise
                write_checklist(manifest_path)
        else:
            worker_count = min(workers, len(runnable))
            print(f"Running {len(runnable)} jobs with {worker_count} parallel workers on the available CUDA device(s).")
            context = mp.get_context("spawn")
            stop_event = context.Event()
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(stop_event, worker_count),
            )
            futures = {
                executor.submit(_run_one_job, index, len(jobs), cfg, result_root): (index, cfg)
                for index, cfg in runnable
            }
            try:
                for future in as_completed(futures):
                    index, cfg = futures[future]
                    try:
                        result = future.result()
                    except BaseException as exc:
                        failures += 1
                        print(f"[{index}/{len(jobs)}] WORKER FAILED {cfg.job_id}: {exc!r}")
                        result = {"status": "failed"}
                    status = result["status"]
                    if status == "failed":
                        failures += 1 if "error" in result else 0
                    print(f"[{index}/{len(jobs)}] {status.upper()} {cfg.job_id}")
                    write_checklist(manifest_path)
                    error_text = str(result.get("error", "")).lower()
                    resource_failure = status == "failed" and (
                        "out of memory" in error_text or "cuda error" in error_text
                    )
                    if resource_failure:
                        print("CUDA resource failure detected; stopping the queue. Resume with fewer workers.")
                    if (stop_on_error and status == "failed") or resource_failure:
                        stop_event.set()
                        for pending in futures:
                            pending.cancel()
                        break
            except KeyboardInterrupt:
                interrupted = True
                stop_event.set()
                for pending in futures:
                    pending.cancel()
                print("Stopping workers after their current round is checkpointed...")
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted. The last completed round is checkpointed; run the same command to resume.")
    finally:
        write_checklist(manifest_path)
        analyze(result_root, ANALYSIS / profile)
        package(manifest_path)
    if interrupted:
        print("Interrupted. Run the same command to resume all unfinished jobs.")
        return 130
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
        for name in ("README.md", "RUN_INSTRUCTIONS.md", "EXPERIMENTS.md", "requirements.txt", "run_all.cmd", "run_all.ps1", "cli.py", "self_test.py"):
            path = ROOT / name
            if path.exists(): archive.write(path, path.relative_to(ROOT))
    latest = PACKAGES / f"LATEST_{profile}.txt"; latest.write_text(destination.name, encoding="utf-8")
    print(f"Packaged analysis archive: {destination}")
    return destination
