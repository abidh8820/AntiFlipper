from __future__ import annotations

import argparse
import json
import sys

from antiflipper_exp.analysis import analyze
from antiflipper_exp.benchmark import fixed_workload_benchmark
from antiflipper_exp.manifest import load_manifest
from antiflipper_exp.orchestrator import ANALYSIS, RESULTS, package, prepare_manifest, run_manifest, status_rows, write_checklist


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Resumable AntiFLipper journal experiment pipeline")
    sub = value.add_subparsers(dest="command", required=True)
    for name in ("plan", "status", "run", "analyze", "benchmark", "package"):
        command = sub.add_parser(name)
        command.add_argument("--profile", choices=("smoke", "core", "full"), default="core")
        if name == "plan": command.add_argument("--force", action="store_true")
        if name == "run":
            command.add_argument("--stop-on-error", action="store_true")
            command.add_argument("--no-retry-failed", action="store_true")
            command.add_argument("--workers", type=int, default=1, help="number of experiment processes (default: 1)")
    return value


def main() -> int:
    args = parser().parse_args(); manifest = prepare_manifest(args.profile, getattr(args, "force", False))
    if args.command == "plan":
        profile, jobs = load_manifest(manifest); checklist = write_checklist(manifest)
        print(json.dumps({"profile": profile, "jobs": len(jobs), "manifest": str(manifest), "checklist": str(checklist)}, indent=2)); return 0
    if args.command == "status":
        _, _, rows = status_rows(manifest); write_checklist(manifest)
        counts = {name: sum(x["status"] == name for x in rows) for name in sorted({x["status"] for x in rows})}
        print(json.dumps({"total": len(rows), **counts}, indent=2)); return 0
    if args.command == "run": return run_manifest(manifest, args.stop_on_error, not args.no_retry_failed, args.workers)
    if args.command == "analyze":
        print(json.dumps(analyze(RESULTS / args.profile, ANALYSIS / args.profile), indent=2)); return 0
    if args.command == "benchmark":
        print(json.dumps(fixed_workload_benchmark(ANALYSIS / args.profile / "timing"), indent=2)); return 0
    if args.command == "package": print(package(manifest)); return 0
    return 2


if __name__ == "__main__": raise SystemExit(main())
