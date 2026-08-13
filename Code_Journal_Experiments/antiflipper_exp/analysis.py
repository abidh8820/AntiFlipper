from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("final_accuracy", "last10_accuracy_mean", "final_asr", "accuracy_auc", "final_precision", "final_recall", "final_f1", "final_fpr", "mean_aggregation_seconds", "mean_round_seconds")


def _read_completed(results_root: Path) -> list[tuple[dict, dict]]:
    rows = []
    for summary_path in results_root.glob("*/summary.json"):
        config_path = summary_path.parent / "config.json"
        if config_path.exists():
            rows.append((json.loads(config_path.read_text(encoding="utf-8")), json.loads(summary_path.read_text(encoding="utf-8"))))
    return rows


def analyze(results_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    completed = _read_completed(results_root)
    per_run: list[dict] = []
    for config, summary in completed:
        row = {**config, **{key: summary.get(key) for key in METRICS}, "job_id": summary["job_id"]}
        row["tags"] = ",".join(config.get("tags", []))
        per_run.append(row)
    if per_run:
        with (output / "per_run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_run[0])); writer.writeheader(); writer.writerows(per_run)

    identity = ("experiment", "dataset", "distribution", "method", "dirichlet_alpha", "malicious_client_fraction", "label_poison_fraction", "eta", "threshold_mode", "rho_threshold", "counter_threshold", "counter_mode", "trust_update", "notes")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_run: groups[tuple(row.get(k) for k in identity)].append(row)
    aggregate_rows: list[dict] = []
    for key, values in groups.items():
        result = dict(zip(identity, key)); result["n_seeds"] = len(values)
        for metric in METRICS:
            observations = [float(v[metric]) for v in values if v.get(metric) is not None]
            if not observations: continue
            mean = statistics.fmean(observations)
            std = statistics.stdev(observations) if len(observations) > 1 else 0.0
            result[f"{metric}_mean"] = mean; result[f"{metric}_std"] = std
            result[f"{metric}_ci95"] = 1.96 * std / math.sqrt(len(observations)) if len(observations) > 1 else 0.0
        aggregate_rows.append(result)
    if aggregate_rows:
        fields = list(dict.fromkeys(k for row in aggregate_rows for k in row))
        with (output / "aggregate_mean_std_ci95.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(aggregate_rows)

    report = _markdown_report(per_run, aggregate_rows)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    _plots(per_run, output)
    result = {"completed_runs": len(per_run), "aggregate_groups": len(aggregate_rows)}
    (output / "analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _markdown_report(per_run: list[dict], aggregate: list[dict]) -> str:
    experiments = sorted({x["experiment"] for x in per_run})
    lines = ["# AntiFLipper automated analysis", "", f"Completed runs analyzed: **{len(per_run)}**.", "", "## Included experiment groups", ""]
    lines += [f"- {name}" for name in experiments] or ["- None yet"]
    lines += ["", "## Interpretation notes", "", "- Seed-level runs are the independent observations; rounds are not treated as independent replicates.", "- `*_ci95` uses the normal approximation `1.96 * s / sqrt(n)` and should be interpreted cautiously for three seeds.", "- ASR is the fraction of test predictions equal to the fixed paired-label permutation; interpret it together with clean/no-attack controls.", "- AntiFLipper detection metrics are meaningful for AntiFLipper. Other aggregators do not explicitly classify clients and therefore retain default empty-detection values.", "- `mean_aggregation_seconds` is observed aggregation time during training. Run the fixed-workload benchmark separately for matched-workload timing.", ""]
    lines += _automated_findings(aggregate)
    if aggregate:
        lines += ["## Headline summary", "", "| Experiment | Dataset | Distribution | Method | Seeds | Final accuracy | Recall | FPR |", "|---|---|---|---|---:|---:|---:|---:|"]
        for row in aggregate:
            if row["experiment"] != "A1_headline": continue
            lines.append(f"| {row['experiment']} | {row['dataset']} | {row['distribution']} | {row['method']} | {row['n_seeds']} | {row.get('final_accuracy_mean', float('nan')):.4f} ± {row.get('final_accuracy_std', 0):.4f} | {row.get('final_recall_mean', float('nan')):.4f} | {row.get('final_fpr_mean', float('nan')):.4f} |")
    lines += ["", "## Files", "", "- `per_run_metrics.csv`: one row per completed seed/configuration.", "- `aggregate_mean_std_ci95.csv`: mean, standard deviation, and 95% CI half-width across seeds.", "- `figures/`: automatically generated experiment plots when matplotlib is available.", ""]
    return "\n".join(lines)


def _automated_findings(aggregate: list[dict]) -> list[str]:
    lines = ["## Automated findings", ""]
    headline = [x for x in aggregate if x["experiment"] == "A1_headline"]
    for dataset in sorted({x["dataset"] for x in headline}):
        for distribution in sorted({x["distribution"] for x in headline if x["dataset"] == dataset}):
            candidates = [x for x in headline if x["dataset"] == dataset and x["distribution"] == distribution and "final_accuracy_mean" in x]
            if candidates:
                best = max(candidates, key=lambda x: x["final_accuracy_mean"])
                anti = next((x for x in candidates if x["method"] == "antiflipper"), None)
                if anti:
                    lines.append(f"- {dataset} {distribution}: best mean final accuracy was {best['method']} ({best['final_accuracy_mean']:.4f}); AntiFLipper was {anti['final_accuracy_mean']:.4f}.")
    clean = [x for x in aggregate if x["experiment"] == "A2_clean_heterogeneity" and "final_fpr_mean" in x]
    if clean:
        worst = max(clean, key=lambda x: x["final_fpr_mean"])
        lines.append(f"- Clean heterogeneity: maximum observed mean AntiFLipper FPR was {worst['final_fpr_mean']:.4f} on {worst['dataset']} {worst['distribution']} (Dirichlet alpha={worst['dirichlet_alpha']}).")
    boundary = sorted([x for x in aggregate if x["experiment"] == "A3_malicious_ratio" and x["method"] == "antiflipper" and x["malicious_client_fraction"] > 0 and "final_recall_mean" in x], key=lambda x: x["malicious_client_fraction"])
    failed = next((x for x in boundary if x["final_recall_mean"] < 0.9), None)
    if boundary:
        if failed: lines.append(f"- Detection boundary: mean recall first fell below 0.90 at malicious fraction {failed['malicious_client_fraction']:.2f} (recall {failed['final_recall_mean']:.4f}).")
        else: lines.append(f"- Detection boundary: mean recall remained at least 0.90 through the largest completed malicious fraction ({boundary[-1]['malicious_client_fraction']:.2f}).")
    partial = sorted([x for x in aggregate if x["experiment"] == "A4_partial_flip" and x["method"] == "antiflipper" and "final_recall_mean" in x], key=lambda x: x["label_poison_fraction"])
    if partial:
        lines.append(f"- Partial flipping: AntiFLipper mean recall ranged from {min(x['final_recall_mean'] for x in partial):.4f} to {max(x['final_recall_mean'] for x in partial):.4f} across completed poison fractions; interpret this together with ASR and accuracy degradation.")
    if len(lines) == 2: lines.append("- Insufficient completed experiment groups for automated comparisons.")
    lines.append("")
    return lines


def _plots(rows: list[dict], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    specifications = (
        ("A3_malicious_ratio", "malicious_client_fraction", "final_accuracy", "Malicious client fraction", "Final accuracy", "malicious_ratio_accuracy.png"),
        ("A3_malicious_ratio", "malicious_client_fraction", "final_recall", "Malicious client fraction", "Detection recall", "malicious_ratio_recall.png"),
        ("A2_clean_heterogeneity", "dirichlet_alpha", "final_fpr", "Dirichlet alpha", "False-positive rate", "heterogeneity_fpr.png"),
        ("A4_partial_flip", "label_poison_fraction", "final_recall", "Poisoned-label fraction", "Detection recall", "partial_flip_recall.png"),
    )
    for experiment, xkey, ykey, xlabel, ylabel, filename in specifications:
        selected = [r for r in rows if r["experiment"] == experiment]
        if not selected: continue
        plt.figure(figsize=(6.4, 4.2))
        for method in sorted({r["method"] for r in selected}):
            method_rows = [r for r in selected if r["method"] == method]
            points: dict[float, list[float]] = defaultdict(list)
            for row in method_rows: points[float(row[xkey])].append(float(row[ykey]))
            xs = sorted(points); ys = [statistics.fmean(points[x]) for x in xs]
            errors = [statistics.stdev(points[x]) if len(points[x]) > 1 else 0 for x in xs]
            plt.errorbar(xs, ys, yerr=errors, marker="o", capsize=3, label=method)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(figures / filename, dpi=200); plt.close()
