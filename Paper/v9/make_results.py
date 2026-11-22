"""Generate every results table and figure of the v9 paper from the experiment outputs.

Usage (from anywhere):
    python Paper/v9/make_results.py            # uses Code_Journal_Experiments/{analysis,results}/core
    python Paper/v9/make_results.py --profile full

Writes Paper/v9/generated/*.tex and Paper/v9/fig/gen_*.pdf. Cells without completed runs
are printed as "--" and missing figures as a "pending" placeholder, so the paper always
compiles and never shows a number that was not produced by the pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "Code_Journal_Experiments"
OUT = HERE / "generated"
FIG = HERE / "fig"

NAMES = {
    "antiflipper_cc": r"AntiFLipper-CC",
    "antiflipper": r"AntiFLipper (reports)",
    "cc_only": r"CC only",
    "fedavg": "FedAvg", "median": "Median", "multi_krum": "Multi-Krum", "flame": "FLAME",
    "lfighter": "LFighter", "fltrust": r"FLTrust$^\dagger$",
    "trimmed_mean": "Trimmed Mean", "foolsgold": "FoolsGold", "tolpegin": "Tolpegin",
    "antiflipper_cr": r"\textbf{AntiFLipper-CR}", "antiflipper_cr_cc": r"AntiFLipper-CR+CC",
    "antiflipper_budget": r"AntiFLipper + budget", "antiflipper_cc_z3": r"AntiFLipper-CC ($z{=}3$)",
    "antiflipper_cc_v2": r"CC-v2 (soft)",
}
HEADLINE_ORDER = ["fedavg", "median", "multi_krum", "flame", "lfighter", "fltrust", "cc_only", "antiflipper_cc", "antiflipper", "antiflipper_cr_cc", "antiflipper_cr"]
SWEEP_ORDER = ["fedavg", "lfighter", "fltrust", "antiflipper", "antiflipper_cr"]
PROFILES = ("core", "cr")
DASH = "--"


# ----------------------------------------------------------------------------- data

RESULT_ROOTS = (EXP / "results", REPO / "Results_Backup_2026-11-22" / "Code_Journal_Experiments" / "results")


def load_runs(profile: str) -> list[dict]:
    """One dict per completed job: config fields + summary fields. Reads the live
    results/ tree and the 2026-11-22 backup; a job present in both is read once."""
    runs, seen = [], set()
    paths = [p for root in RESULT_ROOTS for p in (root / profile).glob("*/summary.json")]
    for summary_path in paths:
        if summary_path.parent.name in seen:
            continue
        seen.add(summary_path.parent.name)
        config_path = summary_path.parent / "config.json"
        if not config_path.exists():
            continue
        config = {"report_strategy": "honest", "consistency_z": 2.5, "max_exclusion_fraction": 1.0,
                  **json.loads(config_path.read_text(encoding="utf-8"))}
        # Ablations reuse method names; give them distinct labels so they never mix.
        if config["method"] == "antiflipper" and config["max_exclusion_fraction"] < 1:
            config["method"] = "antiflipper_budget"
        elif config["method"] == "antiflipper_cc" and abs(config["consistency_z"] - 2.5) > 1e-9:
            config["method"] = "antiflipper_cc_z3"
        runs.append({**config, **json.loads(summary_path.read_text(encoding="utf-8")), "_dir": summary_path.parent})
    return runs


def select(runs, **criteria) -> list[dict]:
    def match(run):
        for key, value in criteria.items():
            got = run.get(key)
            if isinstance(value, float):
                if got is None or abs(float(got) - value) > 1e-9: return False
            elif got != value:
                return False
        return True
    return [r for r in runs if match(r)]


def stat(runs, metric, scale=1.0):
    values = [float(r[metric]) * scale for r in runs if r.get(metric) not in (None, "")]
    if not values:
        return None, None, 0
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0), len(values)


def fmt(runs, metric, scale=1.0, digits=2, pm=True):
    mean, std, n = stat(runs, metric, scale)
    if mean is None:
        return DASH
    text = f"{mean:.{digits}f}"
    if pm and n > 1:
        text += rf"$\pm${std:.{digits}f}"
    return text


def headline(runs, dataset, distribution, method, **extra):
    return select(runs, experiment="A1_headline", dataset=dataset, distribution=distribution, method=method,
                  report_strategy="honest", malicious_client_fraction=0.4, **extra)


# ----------------------------------------------------------------------------- tables

def write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8")


def table_headline(runs) -> None:
    settings = [("CIFAR10", "NON_IID"), ("CIFAR10", "IID"), ("MNIST", "NON_IID"), ("MNIST", "IID")]
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Final test accuracy (\%, mean$\pm$std over 3 seeds) with 40\% label-flipping clients, and server-side defense time per round on CIFAR-10 non-IID. $^\dagger$FLTrust additionally uses a 100-sample clean root dataset at the server. -- : not run in this setting or not yet completed.}",
        r"\label{tab:headline}", r"\renewcommand{\arraystretch}{1.12}", r"\small",
        r"\begin{tabular}{l|cc|cc|c}", r"\hline",
        r"& \multicolumn{2}{c|}{\textbf{CIFAR-10 (ResNet-18, 20 clients)}} & \multicolumn{2}{c|}{\textbf{MNIST (CNN, 100 clients)}} & \textbf{Defense time} \\",
        r"\textbf{Method} & non-IID ($\alpha_D{=}1$) & IID & non-IID ($\alpha_D{=}1$) & IID & (ms, CIFAR non-IID) \\", r"\hline",
    ]
    for method in HEADLINE_ORDER:
        cells = [fmt(headline(runs, d, s, method), "final_accuracy", 100) for d, s in settings]
        cells.append(fmt(headline(runs, "CIFAR10", "NON_IID", method), "mean_defense_seconds", 1000, 1))
        if method == "cc_only":
            lines.append(r"\hline")
        lines.append(f"{NAMES[method]} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    write("table_headline.tex", "\n".join(lines))


def table_detection(runs) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Attack impact and attacker identification on CIFAR-10 non-IID (40\% attackers, mean over 3 seeds). ASR: share of test predictions equal to the flipped label. P/R/FPR: final precision, recall and false-positive rate of the excluded (AntiFLipper variants) or zero-weighted (baselines) clients. Det.: mean round at which an attacker is excluded.}",
        r"\label{tab:detection}", r"\renewcommand{\arraystretch}{1.1}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l|c|ccc|c}", r"\hline",
        r"\textbf{Method} & \textbf{ASR (\%)} & \textbf{P} & \textbf{R} & \textbf{FPR} & \textbf{Det.} \\", r"\hline",
    ]
    for method in HEADLINE_ORDER:
        rows = headline(runs, "CIFAR10", "NON_IID", method)
        detects = method not in ("fedavg", "median")
        cells = [fmt(rows, "final_asr", 100, 1, pm=False)]
        cells += [fmt(rows, m, 1, 2, pm=False) if detects else "n/a" for m in ("final_precision", "final_recall", "final_fpr")]
        cells.append(fmt(rows, "mean_detection_round", 1, 1, pm=False) if method.startswith("antiflipper") or method == "cc_only" else "n/a")
        lines.append(f"{NAMES[method]} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    write("table_detection.tex", "\n".join(lines))


def table_adaptive(runs) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Report-faking adaptive attacker on CIFAR-10 non-IID: attackers train on flipped labels but report the accuracy measured on their clean labels. Methods that do not use reports are unaffected, so their honest-report runs apply to both columns.}",
        r"\label{tab:adaptive}", r"\renewcommand{\arraystretch}{1.1}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l|cc|cc}", r"\hline",
        r"& \multicolumn{2}{c|}{\textbf{Honest reports}} & \multicolumn{2}{c}{\textbf{Faked reports}} \\",
        r"\textbf{Method} & Acc. (\%) & Recall & Acc. (\%) & Recall \\", r"\hline",
    ]
    for method in ("lfighter", "fltrust", "cc_only", "antiflipper", "antiflipper_cc", "antiflipper_cr", "antiflipper_cr_cc"):
        honest = headline(runs, "CIFAR10", "NON_IID", method)
        uses_reports = method not in ("lfighter", "fltrust", "cc_only")
        faked = select(runs, experiment="A7_adaptive_report", dataset="CIFAR10", distribution="NON_IID", method=method) if uses_reports else honest
        cells = [fmt(honest, "final_accuracy", 100), fmt(honest, "final_recall", 1, 2, pm=False),
                 fmt(faked, "final_accuracy", 100), fmt(faked, "final_recall", 1, 2, pm=False)]
        lines.append(f"{NAMES[method]} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    write("table_adaptive.tex", "\n".join(lines))


def table_cost(runs) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Measured cost per round for AntiFLipper-CR (mean over seeds). Client: report evaluation vs.\ local training time per client and their ratio. Server: defense time (cross-check, trust update and aggregation) and peak GPU memory of the job. Upload: extra bytes over FedAvg per client (accuracy plus per-class correct and count).}",
        r"\label{tab:cost}", r"\renewcommand{\arraystretch}{1.1}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{l|ccc|cc|c}", r"\hline",
        r"& \multicolumn{3}{c|}{\textbf{Client}} & \multicolumn{2}{c|}{\textbf{Server}} & \\",
        r"\textbf{Setting} & Eval & Train & Ratio & Defense & GPU & \textbf{Upl.} \\",
        r" & (s) & (s) & (\%) & (ms) & (MB) & \\", r"\hline",
    ]
    for dataset, distribution, label in (("CIFAR10", "NON_IID", "CIFAR non-IID"), ("CIFAR10", "IID", "CIFAR IID"), ("MNIST", "NON_IID", "MNIST non-IID"), ("MNIST", "IID", "MNIST IID")):
        rows = headline(runs, dataset, distribution, "antiflipper_cr")
        cells = [fmt(rows, "mean_client_eval_seconds", 1, 3, pm=False), fmt(rows, "mean_client_train_seconds", 1, 2, pm=False),
                 fmt(rows, "mean_client_eval_overhead", 100, 1, pm=False), fmt(rows, "mean_defense_seconds", 1000, 1, pm=False),
                 fmt(rows, "peak_gpu_memory_mb", 1, 0, pm=False), "+84\\,B" if rows else DASH]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    write("table_cost.tex", "\n".join(lines))


def table_sensitivity(runs) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{One-factor sensitivity of AntiFLipper-CC on MNIST non-IID (held-out seeds). Attack: 40\% flippers; clean: no attackers, so every exclusion is a false positive.}",
        r"\label{tab:sensitivity}", r"\renewcommand{\arraystretch}{1.1}", r"\footnotesize",
        r"\begin{tabular}{l|cc|c}", r"\hline",
        r"& \multicolumn{2}{c|}{\textbf{Attack}} & \textbf{Clean} \\",
        r"\textbf{Setting} & Acc. (\%) & Recall & FPR \\", r"\hline",
    ]
    labels = {"default": r"default ($\rho{=}0.05$, $c_{\max}{=}6$, $z{=}2.5$)", "rho=0.025": r"$\rho=0.025$", "rho=0.1": r"$\rho=0.1$",
              "counter=3": r"$c_{\max}=3$", "counter=9": r"$c_{\max}=9$", "z=2.0": r"$z=2.0$", "z=3.0": r"$z=3.0$"}
    for note, label in labels.items():
        attack = select(runs, experiment="A5_sensitivity", notes=note, malicious_client_fraction=0.4)
        clean = select(runs, experiment="A5_sensitivity", notes=note, malicious_client_fraction=0.0)
        cells = [fmt(attack, "final_accuracy", 100), fmt(attack, "final_recall", 1, 2, pm=False), fmt(clean, "final_fpr", 1, 3, pm=False)]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    write("table_sensitivity.tex", "\n".join(lines))


# ----------------------------------------------------------------------------- figures

def _placeholder(path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.text(0.5, 0.5, f"{title}\n(results pending)", ha="center", va="center", fontsize=9, color="0.4")
    ax.set_axis_off(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 7, "figure.dpi": 150})
    return plt


def figure_ratio(runs) -> None:
    plt = _style()
    path = FIG / "gen_ratio.pdf"
    points = defaultdict(lambda: defaultdict(list))  # metric -> method -> [(ratio, mean, std)]
    for method in SWEEP_ORDER:
        for ratio in (0.1, 0.2, 0.3, 0.4):
            rows = (headline(runs, "CIFAR10", "NON_IID", method) if ratio == 0.4 else
                    select(runs, experiment="A3_malicious_ratio", method=method, malicious_client_fraction=ratio))
            for metric, scale in (("final_accuracy", 100), ("final_recall", 1)):
                mean, std, _ = stat(rows, metric, scale)
                if mean is not None:
                    points[metric][method].append((ratio * 100, mean, std))
    if not points["final_accuracy"]:
        return _placeholder(path, "Accuracy / recall vs. malicious ratio")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for ax, metric, ylabel in ((axes[0], "final_accuracy", "Final accuracy (%)"), (axes[1], "final_recall", "Detection recall")):
        for method in SWEEP_ORDER:
            series = sorted(points[metric].get(method, []))
            if not series or (metric == "final_recall" and method == "fedavg"):
                continue
            x, y, e = zip(*series)
            ax.errorbar(x, y, yerr=e, marker="o", ms=3, capsize=2, lw=1.2 if method != "antiflipper_cc" else 2,
                        label=NAMES[method].replace(r"\textbf{", "").replace("}", "").replace(r"$^\dagger$", "†"))
        ax.set_xlabel("Malicious clients (%)"); ax.set_ylabel(ylabel); ax.set_xticks([10, 20, 30, 40])
    axes[0].legend(loc="lower left", ncol=1)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def figure_heterogeneity(runs) -> None:
    plt = _style()
    path = FIG / "gen_heterogeneity.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    plotted = False
    for method in ("antiflipper", "antiflipper_budget", "cc_only", "antiflipper_cc", "antiflipper_cr", "antiflipper_cr_cc"):
        series = []
        for alpha in (0.1, 0.5, 1.0):
            mean, std, _ = stat(select(runs, experiment="A2_clean_heterogeneity", method=method, dirichlet_alpha=alpha), "final_fpr")
            if mean is not None:
                series.append((alpha, mean, std))
        if series:
            plotted = True
            x, y, e = zip(*series)
            axes[0].errorbar(x, y, yerr=e, marker="o", ms=3, capsize=2, label=NAMES[method].replace(r"\textbf{", "").replace("}", ""))
    labels, values, errors = [], [], []
    for method in ("fedavg", "lfighter", "fltrust", "cc_only", "antiflipper", "antiflipper_budget", "antiflipper_cc", "antiflipper_cr", "antiflipper_cr_cc"):
        mean, std, _ = stat(select(runs, experiment="A2_attacked_heterogeneity", method=method), "final_accuracy", 100)
        if mean is not None:
            labels.append(NAMES[method].replace(r"\textbf{", "").replace("}", "").replace(r"$^\dagger$", "†")); values.append(mean); errors.append(std)
    if values:
        plotted = True
        axes[1].bar(range(len(values)), values, yerr=errors, capsize=2, color="0.6")
        axes[1].set_xticks(range(len(values))); axes[1].set_xticklabels(labels, rotation=30, ha="right")
    if not plotted:
        plt.close(fig); return _placeholder(path, "Heterogeneity: clean FPR and attacked accuracy")
    axes[0].set_xscale("log"); axes[0].set_xlabel(r"Dirichlet $\alpha_D$ (lower = more skewed)"); axes[0].set_ylabel("False-positive rate (no attack)")
    axes[0].legend(); axes[1].set_ylabel("Final accuracy (%), $\\alpha_D=0.1$, 40% attackers")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def figure_trajectories(runs) -> None:
    """Per-round mean trust and consistency similarity for attackers vs honest clients,
    honest and faked reports, from client_history.csv (first seed available)."""
    plt = _style()
    path = FIG / "gen_trajectories.pdf"
    panels = [("AntiFLipper, no attack, $\\alpha_D=0.1$", select(runs, experiment="A2_clean_heterogeneity", method="antiflipper", dirichlet_alpha=0.1)),
              ("AntiFLipper-CR, no attack, $\\alpha_D=0.1$", select(runs, experiment="A2_clean_heterogeneity", method="antiflipper_cr", dirichlet_alpha=0.1))]
    panels = [(t, sorted(r, key=lambda x: x["seed"])) for t, r in panels]
    usable = [(t, r[0]["_dir"] / "client_history.csv") for t, r in panels if r and (r[0]["_dir"] / "client_history.csv").exists()]
    if not usable:
        return _placeholder(path, "Trust / consistency trajectories")
    fig, axes = plt.subplots(1, len(usable), figsize=(3.5 * len(usable), 2.2), squeeze=False)
    for ax, (title, history_path) in zip(axes[0], usable):
        by_round = defaultdict(lambda: {"a": [], "h": []})
        with history_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["trust"] in ("", None): continue
                group = "a" if row["attacker"] == "1" else "h"
                by_round[int(row["round"])][group].append(float(row["trust"]) * 100 if row["excluded"] == "0" else 0.0)
        rounds = sorted(by_round)
        for group, label, style in (("h", "honest", "-"), ("a", "attackers", "--")):
            ax.plot(rounds, [statistics.fmean(by_round[r][group]) if by_round[r][group] else 0 for r in rounds], style, label=label)
        ax.set_title(title, fontsize=8); ax.set_xlabel("Round"); ax.set_ylabel("Mean trust (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    runs = [run for profile in PROFILES for run in load_runs(profile)]
    FIG.mkdir(parents=True, exist_ok=True)
    for build in (table_headline, table_detection, table_adaptive, table_cost, table_sensitivity,
                  figure_ratio, figure_heterogeneity, figure_trajectories):
        build(runs)
    seeds = defaultdict(int)
    for r in runs: seeds[r["experiment"]] += 1
    write("status.tex", f"% generated from {len(runs)} completed runs: {dict(seeds)}\n"
          rf"\newcommand{{\completedRuns}}{{{len(runs)}}}" "\n")
    print(f"{len(runs)} completed runs -> {OUT} and {FIG}/gen_*.pdf")


if __name__ == "__main__":
    main()
