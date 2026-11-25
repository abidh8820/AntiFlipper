"""Generate every result table, figure and in-text number of the AntiFLipper paper (v11).

Usage:  python Paper/v11/make_results.py
Reads the complete result set (core, cr and fltrust_fix profiles) and writes
Paper/v11/generated/*.tex and Paper/v11/fig/gen_*.pdf. No number in the paper is typed by
hand: tables come from here and in-text numbers are macros in generated/numbers.tex.

Conventions
- Accuracy = mean test accuracy over the last 10 rounds (single rounds are noisy under skew).
- FLTrust: original implementation (the stronger one); BatchNorm-safe re-runs only in a footnote.
- Timing: rounds that ran while another workload shared the machine are dropped
  (Results_Backup_2026-11-23/round_timing_tagged.csv); other jobs use summary.json.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "generated"
FIG = HERE / "fig"
RESULT_ROOTS = (
    REPO / "Results_Backup_2026-11-24_NDAG" / "repo" / "AntiFlipper" / "Code_Journal_Experiments" / "results",
    REPO / "Results_Backup_2026-11-23" / "results",
    REPO / "Code_Journal_Experiments" / "results",
)
TIMING_TAGS = REPO / "Results_Backup_2026-11-23" / "round_timing_tagged.csv"
PROFILES = ("core", "cr", "fltrust_fix", "rev", "rev2", "rev3")
DASH = "--"

PROPOSED = "antiflipper_cr_cc"
NAMES = {
    "fedavg": "FedAvg", "fedavg_weighted": "FedAvg (sample-weighted)", "median": "Median", "multi_krum": "Multi-Krum", "flame": "FLAME",
    "lfighter": "LFighter", "fltrust": r"FLTrust (reference)$^{\dagger}$",
    "fltrust_bnsafe": r"FLTrust (BN-safe)$^{\dagger}$",
    "antiflipper": "Raw reports", "antiflipper_budget": "Raw reports + cap",
    "antiflipper_cc": "Raw reports + CC", "antiflipper_cc_z3": "Raw reports + CC ($z{=}3$)",
    "cc_only": r"CC only$^{\ddagger}$", "cc_only_m": r"CC only$^{\ddagger}$", "antiflipper_cr": "CR + cap (no CC)",
    PROPOSED: r"\textbf{AntiFLipper}",
}
PLAIN = {"fedavg": "FedAvg", "fedavg_weighted": "FedAvg, sample-weighted", "lfighter": "LFighter", "fltrust": "FLTrust", "antiflipper": "Raw reports",
         "antiflipper_cc": "Raw reports + CC", "cc_only": r"CC only$^{\ddagger}$", "antiflipper_cr": "CR + cap (no CC)",
         PROPOSED: "AntiFLipper"}
BASELINES = ["fedavg", "median", "multi_krum", "flame", "lfighter", "fltrust", "fltrust_bnsafe"]
# Okabe-Ito colour-blind-safe palette; the proposed method is always black and thick.
COLORS = {"fedavg": "#999999", "fedavg_weighted": "#555555", "lfighter": "#E69F00", "fltrust": "#56B4E9", "antiflipper": "#D55E00",
          "antiflipper_cc": "#CC79A7", "cc_only": "#009E73", "antiflipper_cr": "#0072B2", PROPOSED: "#000000"}


# ----------------------------------------------------------------------------- data

def load_runs() -> list[dict]:
    runs, seen = [], set()
    for root in RESULT_ROOTS:
        for profile in PROFILES:
            for summary_path in sorted((root / profile).glob("*/summary.json")):
                job = summary_path.parent.name
                if job in seen or not (summary_path.parent / "config.json").exists():
                    continue
                seen.add(job)
                config = {"report_strategy": "honest", "consistency_z": 2.5, "max_exclusion_fraction": 1.0, "notes": "",
                          **json.loads((summary_path.parent / "config.json").read_text(encoding="utf-8"))}
                if config["method"] == "antiflipper" and config["max_exclusion_fraction"] < 1:
                    config["method"] = "antiflipper_budget"
                elif config["method"] == "antiflipper_cc" and abs(config["consistency_z"] - 2.5) > 1e-9:
                    config["method"] = "antiflipper_cc_z3"
                elif config["method"] == "cc_only" and config["max_exclusion_fraction"] < 1:
                    config["method"] = "cc_only_m"  # matched to AntiFLipper: z=3 and the 0.4N cap
                elif config["method"] == "fltrust" and config["notes"] == "bn-safe":
                    config["method"] = "fltrust_bnsafe"
                run = {**config, **json.loads(summary_path.read_text(encoding="utf-8")),
                       "_dir": summary_path.parent, "_job": job, "_profile": profile}
                add_round_metrics(run)
                runs.append(run)
    apply_free_round_timing(runs)
    return runs


def add_round_metrics(run: dict) -> None:
    """Last-10-round ASR (same window as accuracy) and the number of non-finite-loss rounds."""
    import math
    path = run["_dir"] / "round_metrics.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    asr = [float(r["asr"]) for r in rows[-10:] if r.get("asr") not in ("", None)]
    run["last10_asr"] = statistics.fmean(asr) if asr else None
    run["nonfinite_rounds"] = sum(1 for r in rows if not math.isfinite(float(r.get("loss") or "nan")))
    run["full_detected"] = 1.0 if run.get("full_detection_round") not in (None, "") else 0.0


TIMING_FIELDS = ("defense_seconds", "client_train_seconds", "client_eval_seconds", "root_training_seconds", "round_seconds")


def apply_free_round_timing(runs: list[dict]) -> None:
    if not TIMING_TAGS.exists():
        return
    free, shared = defaultdict(lambda: defaultdict(list)), set()
    with TIMING_TAGS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["gpu_shared"] == "1":
                shared.add(row["job"])
                continue
            for field in TIMING_FIELDS:
                if row.get(field) not in ("", None):
                    free[row["job"]][field].append(float(row[field]))
    for run in runs:
        if run["_job"] not in shared:
            continue
        for field in TIMING_FIELDS:
            values = free.get(run["_job"], {}).get(field, [])
            run[f"mean_{field}"] = statistics.fmean(values) if values else None
        if run.get("mean_client_eval_seconds") and run.get("mean_client_train_seconds"):
            run["mean_client_eval_overhead"] = run["mean_client_eval_seconds"] / run["mean_client_train_seconds"]


def select(runs, **criteria) -> list[dict]:
    def match(run):
        for key, want in criteria.items():
            got = run.get(key)
            if isinstance(want, float):
                if got is None or abs(float(got) - want) > 1e-9:
                    return False
            elif got != want:
                return False
        return True
    return [r for r in runs if match(r)]


def stat(rows, metric, scale=1.0):
    values = [float(r[metric]) * scale for r in rows if r.get(metric) not in (None, "")]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def headline(runs, method, dataset="CIFAR10", distribution="NON_IID"):
    return select(runs, experiment="A1_headline", dataset=dataset, distribution=distribution, method=method,
                  report_strategy="honest", malicious_client_fraction=0.4)


def experiment_rows(runs, experiment, method):
    return select(runs, experiment=experiment, method=method)


def clean(runs, method, alpha):
    return select(runs, experiment="A2_clean_heterogeneity", method=method, dirichlet_alpha=alpha)


def attacked_skew(runs, method):
    if method == "fltrust_bnsafe":
        return select(runs, experiment="A2_attacked_heterogeneity", method=method)
    return select(runs, experiment="A2_attacked_heterogeneity", method=method)


def adaptive(runs, method):
    if method in ("lfighter", "fltrust", "fltrust_bnsafe", "cc_only", "cc_only_m", "fedavg"):
        return headline(runs, method)  # ignores reports: faking has no effect
    return select(runs, experiment="A7_adaptive_report", method=method)


def full_detect_text(rows) -> str:
    """'k/n' seeds in which every attacker was excluded, with the mean round over those seeds."""
    if not rows:
        return DASH
    hits = [r for r in rows if r.get("full_detection_round") not in (None, "")]
    if not hits:
        return f"0/{len(rows)}"
    mean = statistics.fmean(float(r["full_detection_round"]) for r in hits)
    return f"{len(hits)}/{len(rows)} ({mean:.0f})"


def ratio(runs, method, p):
    if abs(p - 0.4) < 1e-9:
        return headline(runs, method)
    return select(runs, experiment="A3_malicious_ratio", method=method, malicious_client_fraction=p)


def write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------- cells

class Column:
    """One numeric table column: collects (mean, std) per row, then formats with the best value bold."""

    def __init__(self, metric, scale=1.0, digits=1, std=False, best="max"):
        self.metric, self.scale, self.digits, self.std, self.best = metric, scale, digits, std, best

    def cells(self, row_sets: list[list[dict] | None]) -> list[str]:
        stats = [stat(rows, self.metric, self.scale) if rows is not None else (None, None) for rows in row_sets]
        means = [round(m, self.digits) for m, _ in stats if m is not None]
        target = (max(means) if self.best == "max" else min(means)) if means and self.best else None
        out = []
        for rows, (mean, sd) in zip(row_sets, stats):
            if rows is None:
                out.append("n/a")
                continue
            if mean is None:
                out.append(DASH)
                continue
            text = f"{mean:.{self.digits}f}"
            if target is not None and round(mean, self.digits) == target:
                text = rf"\textbf{{{text}}}"
            if self.std and sd is not None and len(rows) > 1:
                text += rf"{{\scriptsize$\pm${sd:.{self.digits}f}}}"
            out.append(text)
        return out


def render(rows_labels: list[str], columns: list[tuple[Column, list]], rule_after: set[int] = frozenset()) -> list[str]:
    grid = [col.cells(sets) for col, sets in columns]
    lines = []
    for i, label in enumerate(rows_labels):
        lines.append(label + " & " + " & ".join(g[i] for g in grid) + r" \\")
        if i in rule_after:
            lines.append(r"\midrule")
    return lines


# ----------------------------------------------------------------------------- tables

def table_main(runs) -> None:
    methods = BASELINES + [PROPOSED]
    detects = lambda m: m not in ("fedavg", "median")
    H = lambda d, s: [headline(runs, m, d, s) for m in methods]
    det = [headline(runs, m) if detects(m) else None for m in methods]
    columns = [
        (Column("last10_accuracy_mean", 100, 1, std=True), H("CIFAR10", "NON_IID")),
        (Column("last10_asr", 100, 1, best="min"), H("CIFAR10", "NON_IID")),
        (Column("final_precision", 1, 2), det),
        (Column("final_recall", 1, 2), det),
        (Column("final_fpr", 1, 2, best="min"), det),
        (Column("last10_accuracy_mean", 100, 1, std=True), H("CIFAR10", "IID")),
        (Column("last10_accuracy_mean", 100, 1, std=True), H("MNIST", "NON_IID")),
        (Column("last10_accuracy_mean", 100, 1, std=True), H("MNIST", "IID")),
    ]
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Comparison with existing defenses under 40\% label-flipping clients (mean$\pm$std over three seeds).}",
        r"\label{tab:main}", r"\footnotesize", r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{@{}l ccccc c cc@{}}", r"\toprule",
        r"& \multicolumn{5}{c}{\textbf{CIFAR-10, non-IID} ($\alpha_D{=}1$)} & \textbf{CIFAR-10} & \multicolumn{2}{c}{\textbf{MNIST}} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-7}\cmidrule(lr){8-9}",
        r"\textbf{Method} & Acc.\,(\%)\,$\uparrow$ & ASR\,(\%)\,$\downarrow$ & Prec.\,$\uparrow$ & Rec.\,$\uparrow$ & FPR\,$\downarrow$ & IID Acc. & non-IID Acc. & IID Acc. \\",
        r"\midrule",
    ]
    lines += render([NAMES[m] for m in methods], columns, rule_after={len(methods) - 2})
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\parbox{\linewidth}{\scriptsize $^{\dagger}$Uses a 100-sample clean root dataset at the server. Reference: rescales all tensors including BatchNorm statistics; BN-safe: trainable parameters only.}", r"\end{table*}", ""]
    write("table_main.tex", "\n".join(lines))


def table_ablation(runs) -> None:
    variants = [("fedavg", "--", "--", "--"), ("fedavg_weighted", "--", "--", "--"),
                ("antiflipper", r"\ding{55}", r"\ding{55}", r"\ding{55}"),
                ("antiflipper_budget", r"\ding{55}", r"\ding{55}", r"\ding{51}"),
                ("antiflipper_cc", r"\ding{55}", r"\ding{51}", r"\ding{55}"),
                ("cc_only_m", "--", r"\ding{51}", r"\ding{51}"),
                ("antiflipper_cr", r"\ding{51}", r"\ding{55}", r"\ding{51}"),
                (PROPOSED, r"\ding{51}", r"\ding{51}", r"\ding{51}")]
    methods = [v[0] for v in variants]
    det = lambda sets: [None if m.startswith("fedavg") else rows for m, rows in zip(methods, sets)]
    main = [headline(runs, m) for m in methods]
    fake = [adaptive(runs, m) for m in methods]
    cln = [clean(runs, m, 0.1) for m in methods]
    skew = [attacked_skew(runs, m) for m in methods]
    columns = [
        (Column("last10_accuracy_mean", 100, 1), main),
        (Column("final_fpr", 1, 2, best="min"), det(main)),
        (Column("last10_accuracy_mean", 100, 1), fake),
        (Column("final_recall", 1, 2), det(fake)),
        (Column("final_fpr", 1, 2, best="min"), det(fake)),
        (Column("last10_accuracy_mean", 100, 1, std=True), cln),
        (Column("final_accuracy", 100, 1, std=True), cln),
        (Column("honest_excluded", 1, 1, best="min"), det(cln)),
        (Column("last10_accuracy_mean", 100, 1), skew),
        (Column("final_recall", 1, 2), det(skew)),
    ]
    labels = [f"{NAMES[m]} & {cr} & {cc} & {cap}" for m, cr, cc, cap in variants]
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Ablation on CIFAR-10: AntiFLipper with components removed (mean over three seeds).}",
        r"\label{tab:ablation}", r"\footnotesize", r"\setlength{\tabcolsep}{3.6pt}",
        r"\begin{tabular}{@{}l ccc cc ccc ccc cc@{}}", r"\toprule",
        r"& & & & \multicolumn{2}{c}{\textbf{40\% attack}} & \multicolumn{3}{c}{\textbf{40\% report faking}} & \multicolumn{3}{c}{\textbf{No attack}} & \multicolumn{2}{c}{\textbf{40\% attack}} \\",
        r"& & & & \multicolumn{2}{c}{$\alpha_D{=}1$} & \multicolumn{3}{c}{$\alpha_D{=}1$} & \multicolumn{3}{c}{$\alpha_D{=}0.1$} & \multicolumn{2}{c}{$\alpha_D{=}0.1$} \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}\cmidrule(lr){13-14}",
        r"\textbf{Variant} & CR & CC & Cap & Acc. & FPR & Acc. & Rec. & FPR & Acc.\,(last 10) & Acc.\,(final) & Excl.\,$\downarrow$ & Acc. & Rec. \\",
        r"\midrule",
    ]
    lines += render(labels, columns, rule_after={1, len(methods) - 2})
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\parbox{\linewidth}{\scriptsize Acc.: last-10-round mean unless marked. Excl.: honest clients excluded (of 20). FedAvg rows: no defense, for reference; the sample-weighted one weights clients by dataset size and was run only at $\alpha_D{=}0.1$. --: not run. CC uses $z{=}2.5$ in ``Raw reports + CC'' and $z{=}3$ elsewhere. $^{\ddagger}$Uniform trust; it ignores reports, so faking has no effect.}", r"\end{table*}", ""]
    write("table_ablation.tex", "\n".join(lines))


def table_skew(runs) -> None:
    methods = ["fedavg", "fedavg_weighted", "lfighter", "fltrust", "antiflipper", PROPOSED]
    rows = [attacked_skew(runs, m) for m in methods]
    det = [r if not m.startswith("fedavg") else None for m, r in zip(methods, rows)]
    columns = [(Column("last10_accuracy_mean", 100, 1, std=True), rows),
               (Column("last10_asr", 100, 1, best="min"), rows),
               (Column("final_recall", 1, 2), det),
               (Column("final_fpr", 1, 2, best="min"), det)]
    grid = [col.cells(sets) for col, sets in columns]
    short = {"fedavg_weighted": "FedAvg (weighted)"}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Strong skew with attackers (CIFAR-10, $\alpha_D{=}0.1$, 40\% full flips, honest reports).}",
        r"\label{tab:skew}", r"\footnotesize", r"\setlength{\tabcolsep}{3.6pt}",
        r"\begin{tabular}{@{}l ccccc@{}}", r"\toprule",
        r"\textbf{Method} & Acc.\,(\%)\,$\uparrow$ & ASR\,$\downarrow$ & Rec. & FPR & All found \\", r"\midrule",
    ]
    for i, method in enumerate(methods):
        found = full_detect_text(rows[i]) if method in ("antiflipper", PROPOSED) else "n/a"
        name = short.get(method, NAMES[method].replace(" (reference)", ""))
        lines.append(f"{name} & " + " & ".join(g[i] for g in grid) + f" & {found}" + r" \\")
        if i == 3:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\parbox{\linewidth}{\scriptsize FedAvg: equal client weights; weighted: by dataset size, as in~\cite{fedavg-pmlr-v54-mcmahan17a}. All found: seeds in which every attacker was excluded (mean round).}", r"\end{table}", ""]
    write("table_skew.tex", "\n".join(lines))


def partial(runs, method, fraction):
    return select(runs, experiment="A4_partial_flip", method=method, label_poison_fraction=fraction)


def fake_skew(runs, method):
    if method in ("fedavg", "lfighter", "fltrust", "cc_only_m"):
        return attacked_skew(runs, method)  # ignores reports: faking has no effect
    return select(runs, experiment="A8_adaptive_skew", method=method)


def table_adaptive(runs) -> None:
    """Attackers that try to hide: partial flips and report faking."""
    fmt = lambda rows, metric, scale, d, sd=False: (
        DASH if not rows or stat(rows, metric, scale)[0] is None else
        f"{stat(rows, metric, scale)[0]:.{d}f}" + (rf"{{\scriptsize$\pm${stat(rows, metric, scale)[1]:.{d}f}}}" if sd else ""))
    scenarios = [
        ("Full flip", "1", headline(runs, PROPOSED), headline(runs, "fedavg")),
        (r"Partial, 50\%", "1", partial(runs, PROPOSED, 0.5), partial(runs, "fedavg", 0.5) or None),
        (r"Partial, 25\%", "1", partial(runs, PROPOSED, 0.25), partial(runs, "fedavg", 0.25) or None),
        ("Faking", "1", adaptive(runs, PROPOSED), headline(runs, "fedavg")),
        ("Faking", "0.1", fake_skew(runs, PROPOSED), attacked_skew(runs, "fedavg")),
        (r"\quad CR only$^{*}$", "0.1", fake_skew(runs, "antiflipper_cr"), None),
        (r"\quad CC only$^{\ddagger}$", "0.1", fake_skew(runs, "cc_only_m"), None),
    ]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Attackers that try to stay hidden (CIFAR-10, 40\% attackers).}",
        r"\label{tab:adaptive}", r"\footnotesize", r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{@{}l c cc ccccc@{}}", r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{FedAvg}} & \multicolumn{5}{c}{\textbf{AntiFLipper}} \\", r"\cmidrule(lr){3-4}\cmidrule(l){5-9}",
        r"\textbf{Attack} & $\alpha_D$ & Acc. & ASR & Acc.\,(\%) & ASR & Rec. & FPR & Found \\", r"\midrule",
    ]
    for i, (name, alpha, ours, base) in enumerate(scenarios):
        lines.append(f"{name} & {alpha} & {fmt(base, 'last10_accuracy_mean', 100, 1) if base else DASH} & "
                     f"{fmt(base, 'last10_asr', 100, 1) if base else DASH} & "
                     f"{fmt(ours, 'last10_accuracy_mean', 100, 1, True)} & {fmt(ours, 'last10_asr', 100, 1)} & "
                     f"{fmt(ours, 'final_recall', 1, 2)} & {fmt(ours, 'final_fpr', 1, 2)} & {full_detect_text(ours)}" + r" \\")
        if i in (0, 2):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\parbox{\linewidth}{\scriptsize Partial: share of each attacker's samples relabelled, a fresh random subset in every round. FedAvg ignores reports, so its faking rows repeat the honest-reporting runs. $^{*}$Residual and cap, no consistency check. $^{\ddagger}$Consistency check and cap with uniform trust; it ignores reports, so its row is the honest-reporting run. Found: seeds in which every attacker was excluded (mean round).}",
              r"\end{table}", ""]
    write("table_adaptive.tex", "\n".join(lines))


MATCHED_TIMING = REPO / "Results_Backup_2026-11-23" / "timing_matched" / "fixed_workload_timing.csv"


def matched_timing() -> dict[str, float]:
    if not MATCHED_TIMING.exists():
        return {}
    with MATCHED_TIMING.open(encoding="utf-8") as handle:
        return {r["method"]: float(r["mean_seconds"]) * 1000 for r in csv.DictReader(handle)
                if r["dataset"] == "CIFAR10" and r["mean_seconds"] not in ("", None)}


def matched_std_text() -> str:
    """Spread of the matched benchmark for the two methods compared in the text."""
    if not MATCHED_TIMING.exists():
        return DASH
    with MATCHED_TIMING.open(encoding="utf-8") as handle:
        std = {r["method"]: float(r["std_seconds"]) * 1000 for r in csv.DictReader(handle)
               if r["dataset"] == "CIFAR10" and r["std_seconds"] not in ("", None)}
    return f"{std[PROPOSED]:.0f}\\,ms for AntiFLipper and {std['fedavg']:.0f}\\,ms for FedAvg"


def client_size_range(runs, alpha) -> tuple[int, int]:
    """Smallest and largest client dataset over the CIFAR-10 non-IID splits at this alpha."""
    sizes = []
    for r in runs:
        if r.get("dataset") == "CIFAR10" and r.get("distribution") == "NON_IID" and float(r.get("dirichlet_alpha", 0)) == alpha:
            meta = r["_dir"] / "partition_metadata.json"
            if meta.exists():
                sizes += json.loads(meta.read_text(encoding="utf-8"))["client_sizes"].values()
    return (min(sizes), max(sizes)) if sizes else (0, 0)


def residual_auc(runs, experiment, alpha, first_rounds=10, method=PROPOSED, column="effective_report") -> list[float]:
    """Per seed: probability that a random attacker has a lower residual than a random honest
    client, averaged over the first rounds (active clients the consistency check did not mark)."""
    import numpy as np
    out = []
    sel = [r for r in runs if r.get("experiment") == experiment and r["method"] == method
           and r.get("dataset") == "CIFAR10" and r.get("distribution") == "NON_IID"
           and float(r.get("dirichlet_alpha", 0)) == alpha]
    for r in sorted(sel, key=lambda r: r["seed"]):
        by = defaultdict(lambda: ([], []))
        with (r["_dir"] / "client_history.csv").open(encoding="utf-8") as handle:
            for x in csv.DictReader(handle):
                if x["excluded"] == "1" or x["inconsistent"] == "1" or x["effective_report"] == "":
                    continue
                by[int(x["round"])][0 if x["attacker"] == "1" else 1].append(float(x[column]))
        aucs = []
        for t in range(first_rounds):
            att, hon = by[t]
            if att and hon:
                a, h = np.array(att)[:, None], np.array(hon)[None, :]
                aucs.append(float((a < h).mean() + 0.5 * (a == h).mean()))
        if aucs:
            out.append(statistics.fmean(aucs))
    return out


def table_cost(runs) -> None:
    methods = ["fedavg", "multi_krum", "flame", "lfighter", "fltrust", PROPOSED]
    matched = matched_timing()
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cost per round on CIFAR-10 (ResNet-18, 20 clients).}",
        r"\label{tab:cost}", r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}l ccccc@{}}", r"\toprule",
        r"& \textbf{Server} & \multicolumn{2}{c}{\textbf{Server time (ms)}} & \textbf{Client} & \textbf{Report} \\",
        r"\cmidrule(lr){3-4}",
        r"\textbf{Method} & \textbf{data} & matched & in training & \textbf{overhead} & \textbf{payload} \\", r"\midrule",
    ]
    for method in methods:
        rows = headline(runs, method)
        server, _ = stat(rows, "mean_defense_seconds", 1000)
        root, _ = stat(rows, "mean_root_training_seconds", 1000)
        observed = f"{server:.0f}" + (f" + {root:.0f}" if method == "fltrust" else "")
        bench = f"{matched[method]:.0f}" if method in matched else DASH
        overhead, _ = stat(rows, "mean_client_eval_overhead", 100)
        client = rf"{overhead:.1f}\%" if method == PROPOSED else "0"
        data = r"\ding{51}" if method == "fltrust" else r"\ding{55}"
        payload = "84\\,B" if method == PROPOSED else "0"
        if method == PROPOSED:
            lines.append(r"\midrule")
        lines.append(f"{NAMES[method].replace(' (reference)', '')} & {data} & {bench} & {observed} & {client} & {payload}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\parbox{\linewidth}{\scriptsize Matched: identical 20 updates on one CPU (Intel i7-8700, 6 threads, PyTorch 2.8), mean of 5 timed iterations after one warm-up; standard deviation " + matched_std_text() + r". In training: measured per round on an RTX 3090 GPU; excluded clients are not averaged. FLTrust: aggregation + root training.}", r"\end{table}", ""]
    write("table_cost.tex", "\n".join(lines))


def table_sensitivity(runs) -> None:
    settings = [("default", r"Default ($c_{\max}{=}6$, $z{=}3$)"),
                ("counter=3", r"$c_{\max}=3$"), ("counter=9", r"$c_{\max}=9$"),
                ("z=2.5", r"$z=2.5$"), ("z=3.5", r"$z=3.5$")]
    attack = [select(runs, experiment="A9_sensitivity_full", notes=n, malicious_client_fraction=0.4) for n, _ in settings]
    cln = [select(runs, experiment="A9_sensitivity_full", notes=n, malicious_client_fraction=0.0) for n, _ in settings]
    columns = [(Column("last10_accuracy_mean", 100, 1, best=None), attack),
               (Column("final_recall", 1, 2, best=None), attack),
               (Column("final_fpr", 1, 2, best="min"), cln)]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Parameter sensitivity of AntiFLipper (MNIST non-IID, held-out seeds).}",
        r"\label{tab:sensitivity}", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}l ccc@{}}", r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{40\% attack}} & \textbf{No attack} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}",
        r"\textbf{Setting} & Acc.\,(\%) & Rec. & FPR \\", r"\midrule",
    ]
    lines += render([label for _, label in settings], columns, rule_after={0})
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_sensitivity.tex", "\n".join(lines))


# ----------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linestyle": ":", "legend.fontsize": 6.8, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                         "pdf.fonttype": 42})
    return plt


def _style(method):
    return dict(color=COLORS[method], lw=2.0 if method == PROPOSED else 1.2,
                marker="o" if method == PROPOSED else "s", ms=3.5 if method == PROPOSED else 2.8,
                zorder=5 if method == PROPOSED else 3)


def figure_ratio(runs) -> None:
    plt = _plt()
    methods = ["fedavg", "lfighter", "fltrust", "antiflipper", PROPOSED]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75))
    for ax, metric, scale, ylabel in ((axes[0], "last10_accuracy_mean", 100, "Accuracy (%)"),
                                      (axes[1], "final_recall", 1, "Recall")):
        for method in methods:
            if metric == "final_recall" and method == "fedavg":
                continue
            pts = [(p * 100, *stat(ratio(runs, method, p), metric, scale)) for p in (0.1, 0.2, 0.3, 0.4)]
            pts = [x for x in pts if x[1] is not None]
            x, y, e = zip(*pts)
            ax.errorbar(x, y, yerr=e, capsize=1.5, elinewidth=0.6, label=PLAIN[method], **_style(method))
        ax.set_xlabel("Malicious clients (%)"); ax.set_ylabel(ylabel); ax.set_xticks([10, 20, 30, 40])
    axes[1].set_ylim(-0.05, 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.10), columnspacing=0.8, handlelength=1.6)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(FIG / "gen_ratio.pdf", bbox_inches="tight"); plt.close(fig)


def figure_heterogeneity(runs) -> None:
    """No attackers: honest clients excluded and accuracy as the split becomes more skewed."""
    plt = _plt()
    methods = ["fedavg", "antiflipper", "antiflipper_cc", "antiflipper_cr", PROPOSED]
    alphas = (1.0, 0.5, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75))
    width = 0.19
    for i, method in enumerate(methods):
        acc = [stat(clean(runs, method, a), "last10_accuracy_mean", 100) for a in alphas]
        style = _style(method)
        if method == "fedavg":
            style.update(ls="--", marker=None)
        axes[1].errorbar(range(len(alphas)), [m for m, _ in acc], yerr=[s for _, s in acc], capsize=1.5, elinewidth=0.6,
                         label=PLAIN[method] + (" (no defense)" if method == "fedavg" else ""), **style)
        if method == "fedavg":
            continue
        excl = [stat(clean(runs, method, a), "honest_excluded")[0] or 0 for a in alphas]
        xs = [k + (i - 2.5) * width for k in range(len(alphas))]
        axes[0].bar(xs, excl, width, color=COLORS[method], edgecolor="white", linewidth=0.3)
    fw = stat(clean(runs, "fedavg_weighted", 0.1), "last10_accuracy_mean", 100)
    if fw[0] is not None:
        axes[1].errorbar([alphas.index(0.1) + 0.12], [fw[0]], yerr=[fw[1]], fmt="D", ms=3.2, capsize=1.5, elinewidth=0.6,
                         color=COLORS["fedavg_weighted"], label=PLAIN["fedavg_weighted"], zorder=4)
    for ax in axes:
        ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"{a:g}" for a in alphas]); ax.set_xlabel(r"Dirichlet $\alpha_D$")
    axes[0].set_ylabel("Honest clients excluded"); axes[0].set_ylim(0, 20)
    axes[1].set_ylabel("Accuracy (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16), columnspacing=0.8, handlelength=1.6)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(FIG / "gen_heterogeneity.pdf", bbox_inches="tight"); plt.close(fig)


def figure_trajectories(runs) -> None:
    """Per-round accuracy and surviving clients without attackers at alpha_D = 0.1 (one representative run)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.55))
    for method in ("antiflipper", PROPOSED):
        candidates = sorted(clean(runs, method, 0.1), key=lambda r: r["seed"])
        run_dir = candidates[1]["_dir"] if len(candidates) > 1 else candidates[0]["_dir"]
        with (run_dir / "round_metrics.csv").open(encoding="utf-8") as handle:
            rm = list(csv.DictReader(handle))
        style = dict(color=COLORS[method], lw=1.4 if method == PROPOSED else 1.1, ls="-" if method == PROPOSED else "--")
        xs = [int(r["round"]) for r in rm]
        acc = [100 * float(r["accuracy"]) for r in rm]
        smooth = [statistics.fmean(acc[max(0, i - 4):i + 1]) for i in range(len(acc))]  # 5-round moving average
        axes[0].plot(xs, acc, color=COLORS[method], lw=0.5, alpha=0.25)
        axes[0].plot(xs, smooth, label=PLAIN[method], **style)
        surviving = defaultdict(int)
        with (run_dir / "client_history.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                surviving[int(row["round"])] += row["excluded"] == "0"
        rounds = sorted(surviving)
        axes[1].plot(rounds, [surviving[r] for r in rounds], **style)
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Accuracy (%, 5-round avg.)"); axes[0].set_ylim(0, 75)
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Active clients"); axes[1].set_ylim(0, 21.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.09), handlelength=2.0)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(FIG / "gen_trajectories.pdf", bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- in-text numbers

def override_share(rows) -> str:
    """Share of client-rounds in which the consistency check marked a client inconsistent."""
    flagged = total = 0
    for run in rows:
        path = run["_dir"] / "client_history.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total += 1
                flagged += row.get("inconsistent") == "1"
    return DASH if not total else f"{100 * flagged / total:.1f}"


def numbers(runs) -> None:
    P, R = PROPOSED, "antiflipper"
    timing = matched_timing()
    mt = lambda m: f"{timing[m]:.0f}" if m in timing else DASH
    v = lambda rows, metric, scale=1.0, d=1: (lambda m: DASH if m is None else f"{m:.{d}f}")(stat(rows, metric, scale)[0])
    s = lambda rows, metric, scale=1.0, d=1: (lambda m: DASH if m is None else f"{m:.{d}f}")(stat(rows, metric, scale)[1])
    acc = "last10_accuracy_mean"
    def wins(a_rows, b_rows, metric=acc):
        """Seeds (matched by seed) in which a is higher than b."""
        b = {r["seed"]: float(r[metric]) for r in b_rows if r.get(metric) is not None}
        return str(sum(1 for r in a_rows if r["seed"] in b and r.get(metric) is not None and float(r[metric]) > b[r["seed"]]))

    def lowest_asr_seeds(rows, others):
        by = [{r["seed"]: float(r["last10_asr"]) for r in o} for o in others]
        return str(sum(1 for r in rows if all(r["seed"] in d and float(r["last10_asr"]) < d[r["seed"]] for d in by)))

    skew_others = [attacked_skew(runs, m) for m in ("fedavg", "fedavg_weighted", "lfighter", "fltrust", "antiflipper")]
    auc_range = lambda vals: (f"{min(vals):.2f}", f"{max(vals):.2f}") if vals else (DASH, DASH)
    fk_res_hi = auc_range(residual_auc(runs, "A7_adaptive_report", 1.0, method="antiflipper_cr"))
    fk_raw_lo = auc_range(residual_auc(runs, "A8_adaptive_skew", 0.1, method="antiflipper_cr", column="raw_report"))
    fk_res_lo = auc_range(residual_auc(runs, "A8_adaptive_skew", 0.1, method="antiflipper_cr"))
    macros = {
        "headAcc": v(headline(runs, P), acc, 100), "headAccRaw": v(headline(runs, R), acc, 100),
        "headAccFLTrust": v(headline(runs, "fltrust"), acc, 100), "headAccLFighter": v(headline(runs, "lfighter"), acc, 100),
        "headAccFLAME": v(headline(runs, "flame"), acc, 100), "headAccFedAvg": v(headline(runs, "fedavg"), acc, 100),
        "headAccKrum": v(headline(runs, "multi_krum"), acc, 100), "headAccKrumStd": s(headline(runs, "multi_krum"), acc, 100),
        "headFPRFLTrust": v(headline(runs, "fltrust"), "final_fpr", 1, 2), "headFPRRaw": v(headline(runs, R), "final_fpr", 1, 2),
        "headFPRcc": v(headline(runs, "antiflipper_cc"), "final_fpr", 1, 2),
        "headFPRcconly": v(headline(runs, "cc_only_m"), "final_fpr", 1, 2),
        "headRecallLFighter": v(headline(runs, "lfighter"), "final_recall", 1, 2),
        "headRecallFLAME": v(headline(runs, "flame"), "final_recall", 1, 2),
        "headASR": v(headline(runs, P), "last10_asr", 100), "headASRFedAvg": v(headline(runs, "fedavg"), "last10_asr", 100),
        "headDet": v(headline(runs, P), "full_detection_round", 1, 0),
        "defMs": v(headline(runs, P), "mean_defense_seconds", 1000, 0),
        "defMsFedAvg": v(headline(runs, "fedavg"), "mean_defense_seconds", 1000, 0),
        "defMsFLTrust": v(headline(runs, "fltrust"), "mean_defense_seconds", 1000, 0),
        "rootMsFLTrust": v(headline(runs, "fltrust"), "mean_root_training_seconds", 1000, 0),
        "defMsFLAME": v(headline(runs, "flame"), "mean_defense_seconds", 1000, 0),
        "defMsKrum": v(headline(runs, "multi_krum"), "mean_defense_seconds", 1000, 0),
        "defMsLFighter": v(headline(runs, "lfighter"), "mean_defense_seconds", 1000, 0),
        "evalOverhead": v(headline(runs, P), "mean_client_eval_overhead", 100, 1),
        "cleanHonRaw": v(clean(runs, R, 0.1), "honest_excluded", 1, 1),
        "cleanHon": v(clean(runs, P, 0.1), "honest_excluded", 1, 1),
        "cleanHonBudget": v(clean(runs, "antiflipper_budget", 0.1), "honest_excluded", 1, 1),
        "cleanHonCC": v(clean(runs, "antiflipper_cc", 0.1), "honest_excluded", 1, 1),
        "cleanHonZthree": v(clean(runs, "antiflipper_cc_z3", 0.1), "honest_excluded", 1, 1),
        "cleanAccRaw": v(clean(runs, R, 0.1), acc, 100), "cleanAcc": v(clean(runs, P, 0.1), acc, 100),
        "cleanAccCROnly": v(clean(runs, "antiflipper_cr", 0.1), acc, 100),
        "cleanAccBudget": v(clean(runs, "antiflipper_budget", 0.1), acc, 100),
        "cleanAccMid": v(clean(runs, P, 0.5), acc, 100), "cleanAccMidRaw": v(clean(runs, R, 0.5), acc, 100),
        "cleanHonMidRaw": v(clean(runs, R, 0.5), "honest_excluded", 1, 1),
        "skewAcc": v(attacked_skew(runs, P), acc, 100), "skewAccStd": s(attacked_skew(runs, P), acc, 100),
        "skewAccFedAvg": v(attacked_skew(runs, "fedavg"), acc, 100), "skewAccFedAvgStd": s(attacked_skew(runs, "fedavg"), acc, 100),
        "skewAccLFighter": v(attacked_skew(runs, "lfighter"), acc, 100), "skewAccFLTrust": v(attacked_skew(runs, "fltrust"), acc, 100),
        "skewAccRaw": v(attacked_skew(runs, R), acc, 100), "skewFPRRaw": v(attacked_skew(runs, R), "final_fpr", 1, 2),
        "skewRecall": v(attacked_skew(runs, P), "final_recall", 1, 2),
        "skewRecallLFighter": v(attacked_skew(runs, "lfighter"), "final_recall", 1, 2),
        "skewRecallFLTrust": v(attacked_skew(runs, "fltrust"), "final_recall", 1, 2),
        "skewDet": v(attacked_skew(runs, P), "full_detection_round", 1, 0),
        "fakeRecallRaw": v(adaptive(runs, R), "final_recall", 1, 2),
        "fakeRecallCROnly": v(adaptive(runs, "antiflipper_cr"), "final_recall", 1, 2),
        "fakeRecall": v(adaptive(runs, P), "final_recall", 1, 2), "fakeAcc": v(adaptive(runs, P), acc, 100),
        "ratioAccLow": v(ratio(runs, P, 0.1), acc, 100), "ratioAccFedAvgLow": v(ratio(runs, "fedavg", 0.1), acc, 100),
        "ratioAccThirty": v(ratio(runs, P, 0.3), acc, 100), "ratioAccFedAvgThirty": v(ratio(runs, "fedavg", 0.3), acc, 100),
        "ratioAccLFighterThirty": v(ratio(runs, "lfighter", 0.3), acc, 100),
        "ratioAccFLTrustThirty": v(ratio(runs, "fltrust", 0.3), acc, 100),
        "bnsafeHead": v(select(runs, experiment="A1_headline", method="fltrust_bnsafe", dataset="CIFAR10", distribution="NON_IID"), acc, 100),
        "mnistAcc": v(headline(runs, P, "MNIST", "NON_IID"), acc, 100),
        "iidAcc": v(headline(runs, P, "CIFAR10", "IID"), acc, 100),
        "skewFull": full_detect_text(attacked_skew(runs, P)).split(" ")[0],
        "skewDetHits": (lambda h: DASH if not h else f"{statistics.fmean(float(r['full_detection_round']) for r in h):.0f}")(
            [r for r in attacked_skew(runs, P) if r.get("full_detection_round") not in (None, "")]),
        "skewMissed": str(sum(round((1 - float(r["final_recall"])) * len(r.get("attackers") or [])) for r in attacked_skew(runs, P))),
        "fltrustNanRuns": str(sum(1 for r in headline(runs, "fltrust") if r.get("nonfinite_rounds"))),
        "fltrustNanRounds": str(sum(int(r.get("nonfinite_rounds") or 0) for r in headline(runs, "fltrust"))),
        "bnsafeFPR": v(headline(runs, "fltrust_bnsafe"), "final_fpr", 1, 2),
        "cleanFinal": v(clean(runs, P, 0.1), "final_accuracy", 100),
        "cleanFinalMin": f"{min(100 * float(r['final_accuracy']) for r in clean(runs, P, 0.1)):.1f}",
        "cleanAccStd": s(clean(runs, P, 0.1), acc, 100), "cleanAccCROnlyStd": s(clean(runs, "antiflipper_cr", 0.1), acc, 100),
        "ccOverrideShare": override_share(clean(runs, P, 0.1)),
        "ratioFPRmax": f"{max(float(r['final_fpr']) for p_ in (0.1, 0.2, 0.3) for r in ratio(runs, P, p_)):.2f}",
        "matchedOurs": mt(PROPOSED), "matchedFedAvg": mt("fedavg"), "matchedFLTrust": mt("fltrust"),
        "matchedKrum": mt("multi_krum"), "matchedFLAME": mt("flame"), "matchedLFighter": mt("lfighter"),
        "matchedRaw": mt("antiflipper"),
        "nRuns": str(len([r for r in runs if r["method"] != "fltrust_bnsafe"])),
        # matched consistency check alone (z=3, cap), FedAvg without attackers, partial flips, faking under skew
        "ccHeadAcc": v(headline(runs, "cc_only_m"), acc, 100), "ccHeadFPR": v(headline(runs, "cc_only_m"), "final_fpr", 1, 2),
        "ccSkewAcc": v(attacked_skew(runs, "cc_only_m"), acc, 100),
        "ccSkewRecall": v(attacked_skew(runs, "cc_only_m"), "final_recall", 1, 2),
        "ccCleanAcc": v(clean(runs, "cc_only_m", 0.1), acc, 100),
        # attackers found under strong skew with report faking, counted over all seeds
        "fsFound": str(round(sum(float(r["final_recall"]) * round(r["num_clients"] * r["malicious_client_fraction"])
                                 for r in fake_skew(runs, P)))),
        "fsTotal": str(sum(round(r["num_clients"] * r["malicious_client_fraction"]) for r in fake_skew(runs, P))),
        # sample-weighted FedAvg controls (rev2); dashes until the runs exist
        "fedWCleanAcc": v(clean(runs, "fedavg_weighted", 0.1), acc, 100), "fedWCleanStd": s(clean(runs, "fedavg_weighted", 0.1), acc, 100),
        "fedWCleanFinal": v(clean(runs, "fedavg_weighted", 0.1), "final_accuracy", 100),
        "fedWSkewAcc": v(attacked_skew(runs, "fedavg_weighted"), acc, 100), "fedWSkewStd": s(attacked_skew(runs, "fedavg_weighted"), acc, 100),
        "fedWSkewASR": v(attacked_skew(runs, "fedavg_weighted"), "last10_asr", 100),
        "pfAccFedAvgHalf": v(partial(runs, "fedavg", 0.5), acc, 100), "pfAccFedAvgQuarter": v(partial(runs, "fedavg", 0.25), acc, 100),
        "fedCleanAcc": v(clean(runs, "fedavg", 0.1), acc, 100), "fedCleanStd": s(clean(runs, "fedavg", 0.1), acc, 100),
        "fedCleanFinal": v(clean(runs, "fedavg", 0.1), "final_accuracy", 100),
        "fedCleanMid": v(clean(runs, "fedavg", 0.5), acc, 100), "fedCleanHigh": v(clean(runs, "fedavg", 1.0), acc, 100),
        "pfAccHalf": v(partial(runs, P, 0.5), acc, 100), "pfAccQuarter": v(partial(runs, P, 0.25), acc, 100),
        "pfDetHalf": v(partial(runs, P, 0.5), "full_detection_round", 1, 0),
        "pfDetQuarter": v(partial(runs, P, 0.25), "full_detection_round", 1, 0),
        "fakeDet": v(adaptive(runs, P), "full_detection_round", 1, 0),
        "fsAcc": v(fake_skew(runs, P), acc, 100), "fsRecall": v(fake_skew(runs, P), "final_recall", 1, 2),
        "fsRecallCR": v(fake_skew(runs, "antiflipper_cr"), "final_recall", 1, 2),
        # data split and residual separation (review response)
        "sizeLowMin": f"{client_size_range(runs, 0.1)[0]:,}".replace(",", "{,}"),
        "sizeLowMax": f"{client_size_range(runs, 0.1)[1]:,}".replace(",", "{,}"),
        "sizeHighMin": f"{client_size_range(runs, 1.0)[0]:,}".replace(",", "{,}"),
        "sizeHighMax": f"{client_size_range(runs, 1.0)[1]:,}".replace(",", "{,}"),
        "aucHighMin": f"{min(residual_auc(runs, 'A1_headline', 1.0)):.2f}",
        "aucHighMax": f"{max(residual_auc(runs, 'A1_headline', 1.0)):.2f}",
        "aucLowMin": f"{min(residual_auc(runs, 'A2_attacked_heterogeneity', 0.1)):.2f}",
        "aucLowMax": f"{max(residual_auc(runs, 'A2_attacked_heterogeneity', 0.1)):.2f}",
        "trajSeed": str(sorted(r["seed"] for r in clean(runs, P, 0.1))[1]),
        "fsAccFedAvg": v(fake_skew(runs, "fedavg"), acc, 100),
        # rev2 controls and the separation of faked reports
        "cleanWinsFedW": wins(clean(runs, P, 0.1), clean(runs, "fedavg_weighted", 0.1)),
        "skewWinsFedW": wins(attacked_skew(runs, P), attacked_skew(runs, "fedavg_weighted")),
        "skewWinsFed": wins(attacked_skew(runs, P), attacked_skew(runs, "fedavg")),
        "skewASR": v(attacked_skew(runs, P), "last10_asr", 100),
        "skewASRFedAvg": v(attacked_skew(runs, "fedavg"), "last10_asr", 100),
        "skewASRBest": lowest_asr_seeds(attacked_skew(runs, P), skew_others),
        "pfASRHalf": v(partial(runs, P, 0.5), "last10_asr", 100), "pfASRQuarter": v(partial(runs, P, 0.25), "last10_asr", 100),
        "pfASRFedAvgHalf": v(partial(runs, "fedavg", 0.5), "last10_asr", 100),
        "pfASRFedAvgQuarter": v(partial(runs, "fedavg", 0.25), "last10_asr", 100),
        "fakeAucResHighMin": fk_res_hi[0], "fakeAucResHighMax": fk_res_hi[1],
        "fakeAucRawLowMin": fk_raw_lo[0], "fakeAucRawLowMax": fk_raw_lo[1],
        "fakeAucResLowMin": fk_res_lo[0], "fakeAucResLowMax": fk_res_lo[1],
        "sensAccMin": (lambda rows: DASH if not rows else f"{min(100 * float(r['last10_accuracy_mean']) for r in rows):.1f}")(
            select(runs, experiment="A9_sensitivity_full", malicious_client_fraction=0.4)),
        "sensAccMax": (lambda rows: DASH if not rows else f"{max(100 * float(r['last10_accuracy_mean']) for r in rows):.1f}")(
            select(runs, experiment="A9_sensitivity_full", malicious_client_fraction=0.4)),
        "sensFPRmax": (lambda rows: DASH if not rows else f"{max(float(r['final_fpr']) for r in rows):.2f}")(
            select(runs, experiment="A9_sensitivity_full", malicious_client_fraction=0.0)),
        "sensRecallMin": (lambda rows: DASH if not rows else f"{min(float(r['final_recall']) for r in rows):.2f}")(
            select(runs, experiment="A9_sensitivity_full", malicious_client_fraction=0.4)),
    }
    write("numbers.tex", "% generated by make_results.py -- do not edit\n" +
          "".join(rf"\newcommand{{\{k}}}{{{val}}}" "\n" for k, val in macros.items()))


def main() -> None:
    runs = load_runs()
    FIG.mkdir(parents=True, exist_ok=True)
    for stale in ("table_headline.tex", "table_detection.tex", "table_heterogeneity.tex"):
        (OUT / stale).unlink(missing_ok=True)
    for build in (table_main, table_ablation, table_skew, table_adaptive, table_cost, table_sensitivity,
                  figure_ratio, figure_heterogeneity, figure_trajectories, numbers):
        build(runs)
    counts = defaultdict(int)
    for r in runs:
        counts[r["_profile"]] += 1
    print(f"{len(runs)} runs {dict(counts)} -> {OUT}, {FIG}/gen_*.pdf")


if __name__ == "__main__":
    main()
