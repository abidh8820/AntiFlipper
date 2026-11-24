"""Generate every result table, figure and in-text number of the v10 paper.

Usage:  python Paper/v10/make_results.py
Reads the complete result set copied from the GPU machine (core, cr and fltrust_fix profiles)
and writes Paper/v10/generated/*.tex and Paper/v10/fig/gen_*.pdf. No number in the paper is
typed by hand: tables come from here, and in-text numbers are macros in generated/numbers.tex.

Conventions
- Accuracy = mean test accuracy over the last 10 rounds (final-round accuracy is noisy under
  strong skew; both are in summary.json).
- FLTrust: the original core runs are the baseline (stronger of the two implementations).
  The BatchNorm-safe re-runs are reported in numbers.tex for a footnote.
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
    REPO / "Code_Journal_Experiments" / "results",
)
TIMING_TAGS = REPO / "Results_Backup_2026-11-23" / "round_timing_tagged.csv"
PROFILES = ("core", "cr", "fltrust_fix")
DASH = "--"

# Proposed method = antiflipper_cr_cc. Everything else is a baseline or an ablation.
NAMES = {
    "fedavg": "FedAvg", "median": "Median", "multi_krum": "Multi-Krum", "flame": "FLAME",
    "lfighter": "LFighter", "fltrust": r"FLTrust$^\dagger$",
    "antiflipper": "AntiFLipper-v1 (raw reports)",
    "antiflipper_budget": r"\quad v1 + exclusion cap",
    "antiflipper_cc": r"\quad v1 + CC",
    "antiflipper_cc_z3": r"\quad v1 + CC ($z{=}3$)",
    "cc_only": r"\quad CC only (no reports)",
    "antiflipper_cr": r"\quad CR only (no CC)",
    "antiflipper_cr_cc": r"\textbf{AntiFLipper (CR+CC)}",
}
PLAIN = {k: v.replace(r"\quad ", "").replace(r"\textbf{", "").replace("}", "").replace(r"$^\dagger$", "")
          .replace(r"$z{=}3$", "z=3") for k, v in NAMES.items()}
BASELINES = ["fedavg", "median", "multi_krum", "flame", "lfighter", "fltrust"]
ABLATIONS = ["antiflipper", "antiflipper_cc", "cc_only", "antiflipper_cr"]
PROPOSED = "antiflipper_cr_cc"
SHORT = {**NAMES, "antiflipper": "v1 (raw reports)", "antiflipper_budget": r"\quad v1 + cap",
         "cc_only": r"\quad CC only", "antiflipper_cr": r"\quad CR only", PROPOSED: r"\textbf{Ours (CR+CC)}"}


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
                elif config["method"] == "fltrust" and config["notes"] == "bn-safe":
                    config["method"] = "fltrust_bnsafe"
                runs.append({**config, **json.loads(summary_path.read_text(encoding="utf-8")),
                             "_dir": summary_path.parent, "_job": job, "_profile": profile})
    apply_free_round_timing(runs)
    return runs


def apply_free_round_timing(runs: list[dict]) -> None:
    """Replace mean timings by the mean over rounds measured on an otherwise idle machine."""
    if not TIMING_TAGS.exists():
        return
    free = defaultdict(lambda: defaultdict(list))
    shared_jobs = set()
    with TIMING_TAGS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["gpu_shared"] == "1":
                shared_jobs.add(row["job"])
                continue
            for field in ("defense_seconds", "client_train_seconds", "client_eval_seconds", "root_training_seconds", "round_seconds"):
                if row.get(field) not in ("", None):
                    free[row["job"]][field].append(float(row[field]))
    for run in runs:
        if run["_job"] not in shared_jobs:
            continue  # job never overlapped another workload: summary.json is already clean
        timings = free.get(run["_job"], {})
        for field in ("defense_seconds", "client_train_seconds", "client_eval_seconds", "root_training_seconds", "round_seconds"):
            values = timings.get(field, [])
            run[f"mean_{field}"] = statistics.fmean(values) if values else None
        if run.get("mean_client_eval_seconds") and run.get("mean_client_train_seconds"):
            run["mean_client_eval_overhead"] = run["mean_client_eval_seconds"] / run["mean_client_train_seconds"]


def select(runs, **criteria) -> list[dict]:
    def match(run):
        for key, value in criteria.items():
            got = run.get(key)
            if isinstance(value, float):
                if got is None or abs(float(got) - value) > 1e-9:
                    return False
            elif got != value:
                return False
        return True
    return [r for r in runs if match(r)]


def stat(rows, metric, scale=1.0):
    values = [float(r[metric]) * scale for r in rows if r.get(metric) not in (None, "")]
    if not values:
        return None, None, 0
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0), len(values)


def fmt(rows, metric, scale=1.0, digits=1, pm=True):
    mean, std, n = stat(rows, metric, scale)
    if mean is None:
        return DASH
    text = f"{mean:.{digits}f}"
    if pm and n > 1:
        text += rf"$\pm${std:.{digits}f}"
    return text


def value(rows, metric, scale=1.0, digits=1):
    mean, _, _ = stat(rows, metric, scale)
    return DASH if mean is None else f"{mean:.{digits}f}"


def headline(runs, method, dataset="CIFAR10", distribution="NON_IID", **extra):
    return select(runs, experiment="A1_headline", dataset=dataset, distribution=distribution, method=method,
                  report_strategy="honest", malicious_client_fraction=0.4, **extra)


def clean(runs, method, alpha):
    return select(runs, experiment="A2_clean_heterogeneity", method=method, dirichlet_alpha=alpha)


def attacked_skew(runs, method):
    return select(runs, experiment="A2_attacked_heterogeneity", method=method)


def adaptive(runs, method):
    return select(runs, experiment="A7_adaptive_report", method=method)


def ratio(runs, method, p):
    if abs(p - 0.4) < 1e-9:
        return headline(runs, method)
    return select(runs, experiment="A3_malicious_ratio", method=method, malicious_client_fraction=p)


def write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------- tables

def table_headline(runs) -> None:
    settings = [("CIFAR10", "NON_IID"), ("CIFAR10", "IID"), ("MNIST", "NON_IID"), ("MNIST", "IID")]
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Test accuracy (\%, mean of the last 10 rounds, mean$\pm$std over 3 seeds) with 40\% label-flipping clients, and server-side defense time per round on CIFAR-10 non-IID. CR: class-conditional residual; CC: update-consistency check. $^\dagger$FLTrust additionally trains on a 100-sample clean root dataset at the server (root training is not included in its defense time; see Table~\ref{tab:cost}). --: not run in this setting.}",
        r"\label{tab:headline}", r"\renewcommand{\arraystretch}{1.1}", r"\small",
        r"\begin{tabular}{l|cc|cc|c}", r"\hline",
        r"& \multicolumn{2}{c|}{\textbf{CIFAR-10 (ResNet-18, 20 clients)}} & \multicolumn{2}{c|}{\textbf{MNIST (CNN, 100 clients)}} & \textbf{Defense time} \\",
        r"\textbf{Method} & non-IID ($\alpha_D{=}1$) & IID & non-IID ($\alpha_D{=}1$) & IID & (ms/round) \\", r"\hline",
    ]
    for group in (BASELINES, ABLATIONS, [PROPOSED]):
        for method in group:
            cells = [fmt(headline(runs, method, d, s), "last10_accuracy_mean", 100) for d, s in settings]
            cells.append(fmt(headline(runs, method), "mean_defense_seconds", 1000, 0, pm=False))
            lines.append(f"{NAMES[method]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    write("table_headline.tex", "\n".join(lines))


def table_detection(runs) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Attack impact and attacker identification on CIFAR-10 non-IID with 40\% attackers (mean over 3 seeds). ASR: share of test predictions equal to the flipped label. P/R/FPR: precision, recall and false-positive rate of the clients that are excluded (AntiFLipper variants) or given zero weight (baselines) in the last round. Hon.: honest clients excluded (of 12). Det.: round by which all attackers are excluded.}",
        r"\label{tab:detection}", r"\renewcommand{\arraystretch}{1.1}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l|c|ccc|cc}", r"\hline",
        r"\textbf{Method} & \textbf{ASR} & \textbf{P} & \textbf{R} & \textbf{FPR} & \textbf{Hon.} & \textbf{Det.} \\", r"\hline",
    ]
    for method in ["fedavg", "multi_krum", "flame", "lfighter", "fltrust", "antiflipper", "antiflipper_cc", "cc_only", "antiflipper_cr", PROPOSED]:
        rows = headline(runs, method)
        detects = method not in ("fedavg", "median")
        excludes = method.startswith("antiflipper") or method == "cc_only"
        cells = [fmt(rows, "final_asr", 100, 1, pm=False)]
        cells += [fmt(rows, m, 1, 2, pm=False) if detects else "n/a" for m in ("final_precision", "final_recall", "final_fpr")]
        cells.append(fmt(rows, "honest_excluded", 1, 1, pm=False) if excludes else "n/a")
        cells.append(fmt(rows, "full_detection_round", 1, 0, pm=False) if excludes else "n/a")
        lines.append(f"{NAMES[method]} & " + " & ".join(cells) + r" \\")
        if method == "fltrust":
            lines.append(r"\hline")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    write("table_detection.tex", "\n".join(lines))


def table_heterogeneity(runs) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Strong heterogeneity on CIFAR-10. Left: no attackers, so every exclusion is a false positive (Hon.: honest clients excluded, of 20). Right: 40\% attackers at $\alpha_D{=}0.1$ (accuracy, recall, honest clients excluded of 12). Accuracy: last-10-round mean, 3 seeds.}",
        r"\label{tab:het}", r"\renewcommand{\arraystretch}{1.1}", r"\scriptsize", r"\setlength{\tabcolsep}{1.8pt}",
        r"\begin{tabular}{l|cc|cc|c||ccc}", r"\hline",
        r"& \multicolumn{5}{c||}{\textbf{No attack}} & \multicolumn{3}{c}{\textbf{40\% attack, $\alpha_D{=}0.1$}} \\",
        r"& \multicolumn{2}{c|}{$\alpha_D{=}0.1$} & \multicolumn{2}{c|}{$\alpha_D{=}0.5$} & $1$ & & & \\",
        r"\textbf{Method} & Acc. & Hon. & Acc. & Hon. & Acc. & Acc. & R & Hon. \\", r"\hline",
    ]
    for method in ["fedavg", "lfighter", "fltrust", "antiflipper", "antiflipper_budget", "antiflipper_cc", "antiflipper_cc_z3", "cc_only", "antiflipper_cr", PROPOSED]:
        c01, c05, c1, att = clean(runs, method, 0.1), clean(runs, method, 0.5), clean(runs, method, 1.0), attacked_skew(runs, method)
        detects = method not in ("fedavg",)
        excludes = method.startswith("antiflipper") or method == "cc_only"
        cells = [fmt(c01, "last10_accuracy_mean", 100, 1), fmt(c01, "honest_excluded", 1, 1, pm=False),
                 fmt(c05, "last10_accuracy_mean", 100, 1), fmt(c05, "honest_excluded", 1, 1, pm=False),
                 fmt(c1, "last10_accuracy_mean", 100, 1, pm=False),
                 fmt(att, "last10_accuracy_mean", 100, 1),
                 fmt(att, "final_recall", 1, 2, pm=False) if detects else "n/a",
                 fmt(att, "honest_excluded", 1, 1, pm=False) if excludes else "n/a"]
        lines.append(f"{SHORT[method]} & " + " & ".join(cells) + r" \\")
        if method == "fltrust":
            lines.append(r"\hline")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    write("table_heterogeneity.tex", "\n".join(lines))


def table_adaptive(runs) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Report-faking attacker (CIFAR-10 non-IID, 40\%): attackers train on flipped labels but report the statistics of their clean labels. Methods that ignore reports are unaffected, so their honest-report runs apply to both columns.}",
        r"\label{tab:adaptive}", r"\renewcommand{\arraystretch}{1.1}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l|cc|ccc}", r"\hline",
        r"& \multicolumn{2}{c|}{\textbf{Honest reports}} & \multicolumn{3}{c}{\textbf{Faked reports}} \\",
        r"\textbf{Method} & Acc. & R & Acc. & R & FPR \\", r"\hline",
    ]
    for method in ("lfighter", "fltrust", "cc_only", "antiflipper", "antiflipper_cc", "antiflipper_cr", PROPOSED):
        honest = headline(runs, method)
        uses_reports = method not in ("lfighter", "fltrust", "cc_only")
        faked = adaptive(runs, method) if uses_reports else honest
        cells = [fmt(honest, "last10_accuracy_mean", 100), fmt(honest, "final_recall", 1, 2, pm=False),
                 fmt(faked, "last10_accuracy_mean", 100), fmt(faked, "final_recall", 1, 2, pm=False),
                 fmt(faked, "final_fpr", 1, 2, pm=False)]
        lines.append(f"{NAMES[method]} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    write("table_adaptive.tex", "\n".join(lines))


def table_cost(runs) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Measured cost per round on CIFAR-10 non-IID (20 clients, ResNet-18 with 11.2\,M parameters) and MNIST (100 clients, 21.8\,k parameters). Client: time of the report evaluation relative to local training, measured within each run. Server: defense time (checks, trust update and aggregation) and FLTrust's root-model training. Upload: bytes per client and round beyond the model update (44.7\,MB on CIFAR-10).}",
        r"\label{tab:cost}", r"\renewcommand{\arraystretch}{1.1}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l|c|cc|c}", r"\hline",
        r"& \textbf{Client} & \multicolumn{2}{c|}{\textbf{Server (ms)}} & \textbf{Upload} \\",
        r"\textbf{Method} & eval/train & Defense & Root & (bytes) \\", r"\hline",
    ]
    extra = {"fedavg": "0", "multi_krum": "0", "flame": "0", "lfighter": "0", "fltrust": "0", "antiflipper": "4", PROPOSED: "84"}
    for method in ("fedavg", "multi_krum", "flame", "lfighter", "fltrust", "antiflipper", PROPOSED):
        rows = headline(runs, method)
        cells = [fmt(rows, "mean_client_eval_overhead", 100, 1, pm=False) + r"\%" if method in ("antiflipper", PROPOSED) else "0",
                 fmt(rows, "mean_defense_seconds", 1000, 0, pm=False),
                 fmt(rows, "mean_root_training_seconds", 1000, 0, pm=False) if method == "fltrust" else "0",
                 extra[method]]
        lines.append(f"{NAMES[method]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    for method in ("lfighter", "fltrust", PROPOSED):
        rows = headline(runs, method, "MNIST", "NON_IID")
        cells = [fmt(rows, "mean_client_eval_overhead", 100, 1, pm=False) + r"\%" if method == PROPOSED else "0",
                 fmt(rows, "mean_defense_seconds", 1000, 1, pm=False),
                 fmt(rows, "mean_root_training_seconds", 1000, 1, pm=False) if method == "fltrust" else "0", extra[method]]
        lines.append(f"{NAMES[method]}, MNIST & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    write("table_cost.tex", "\n".join(lines))


def table_sensitivity(runs) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{One-factor sensitivity of the shared trust/counter/consistency pipeline (v1 + CC, MNIST non-IID, held-out seeds 101, 113, 127). Attack: 40\% flippers. Clean: no attackers, so every exclusion is a false positive.}",
        r"\label{tab:sensitivity}", r"\renewcommand{\arraystretch}{1.1}", r"\footnotesize",
        r"\begin{tabular}{l|cc|c}", r"\hline",
        r"& \multicolumn{2}{c|}{\textbf{Attack}} & \textbf{Clean} \\",
        r"\textbf{Setting} & Acc. (\%) & R & FPR \\", r"\hline",
    ]
    labels = {"default": r"default ($\rho{=}0.05$, $c_{\max}{=}6$, $z{=}2.5$)", "rho=0.025": r"$\rho=0.025$", "rho=0.1": r"$\rho=0.1$",
              "counter=3": r"$c_{\max}=3$", "counter=9": r"$c_{\max}=9$", "z=2.0": r"$z=2.0$", "z=3.0": r"$z=3.0$"}
    for note, label in labels.items():
        attack = select(runs, experiment="A5_sensitivity", notes=note, malicious_client_fraction=0.4)
        cln = select(runs, experiment="A5_sensitivity", notes=note, malicious_client_fraction=0.0)
        cells = [fmt(attack, "last10_accuracy_mean", 100), fmt(attack, "final_recall", 1, 2, pm=False), fmt(cln, "final_fpr", 1, 3, pm=False)]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    write("table_sensitivity.tex", "\n".join(lines))


# ----------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 7, "figure.dpi": 150})
    return plt


def figure_ratio(runs) -> None:
    plt = _plt()
    methods = ["fedavg", "lfighter", "fltrust", "antiflipper", PROPOSED]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for ax, metric, scale, ylabel in ((axes[0], "last10_accuracy_mean", 100, "Accuracy (%)"), (axes[1], "final_recall", 1, "Recall")):
        for method in methods:
            if metric == "final_recall" and method == "fedavg":
                continue
            pts = [(p * 100, *stat(ratio(runs, method, p), metric, scale)[:2]) for p in (0.1, 0.2, 0.3, 0.4)]
            pts = [x for x in pts if x[1] is not None]
            if not pts:
                continue
            x, y, e = zip(*pts)
            ax.errorbar(x, y, yerr=e, marker="o", ms=3, capsize=2, lw=2.0 if method == PROPOSED else 1.1,
                        label=PLAIN[method].replace(" (raw reports)", "").replace(" (CR+CC)", ""))
        ax.set_xlabel("Malicious clients (%)"); ax.set_ylabel(ylabel); ax.set_xticks([10, 20, 30, 40])
    axes[0].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "gen_ratio.pdf", bbox_inches="tight"); plt.close(fig)


def figure_trajectories(runs) -> None:
    """Per-round accuracy and mean trust of honest clients without attackers at alpha_D = 0.1, seed 19."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for method, style in (("antiflipper", "--"), (PROPOSED, "-")):
        rows = [r for r in clean(runs, method, 0.1) if r["seed"] == 19]
        if not rows:
            continue
        run_dir = rows[0]["_dir"]
        with (run_dir / "round_metrics.csv").open(encoding="utf-8") as handle:
            rm = list(csv.DictReader(handle))
        axes[0].plot([int(r["round"]) for r in rm], [100 * float(r["accuracy"]) for r in rm], style,
                     label=PLAIN[method].replace(" (raw reports)", "").replace(" (CR+CC)", ""))
        surviving = defaultdict(int)
        with (run_dir / "client_history.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                surviving[int(row["round"])] += row["excluded"] == "0"
        rounds = sorted(surviving)
        axes[1].plot(rounds, [surviving[r] for r in rounds], style,
                     label=PLAIN[method].replace(" (raw reports)", "").replace(" (CR+CC)", ""))
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Test accuracy (%)"); axes[0].legend(loc="lower right")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Clients not excluded"); axes[1].set_ylim(0, 21)
    fig.tight_layout(); fig.savefig(FIG / "gen_trajectories.pdf", bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- in-text numbers

def numbers(runs) -> None:
    P, V1 = PROPOSED, "antiflipper"
    macros = {
        "headAcc": value(headline(runs, P), "last10_accuracy_mean", 100),
        "headAccVone": value(headline(runs, V1), "last10_accuracy_mean", 100),
        "headAccFLTrust": value(headline(runs, "fltrust"), "last10_accuracy_mean", 100),
        "headAccLFighter": value(headline(runs, "lfighter"), "last10_accuracy_mean", 100),
        "headAccFedAvg": value(headline(runs, "fedavg"), "last10_accuracy_mean", 100),
        "headAccCR": value(headline(runs, "antiflipper_cr"), "last10_accuracy_mean", 100),
        "headFPRFLTrust": value(headline(runs, "fltrust"), "final_fpr", 1, 2),
        "headFPRVone": value(headline(runs, V1), "final_fpr", 1, 2),
        "headFPRcc": value(headline(runs, "antiflipper_cc"), "final_fpr", 1, 2),
        "headRecallLFighter": value(headline(runs, "lfighter"), "final_recall", 1, 2),
        "headRecallFLAME": value(headline(runs, "flame"), "final_recall", 1, 2),
        "headDet": value(headline(runs, P), "full_detection_round", 1, 0),
        "defMs": value(headline(runs, P), "mean_defense_seconds", 1000, 0),
        "defMsVone": value(headline(runs, V1), "mean_defense_seconds", 1000, 0),
        "defMsFedAvg": value(headline(runs, "fedavg"), "mean_defense_seconds", 1000, 0),
        "defMsFLTrust": value(headline(runs, "fltrust"), "mean_defense_seconds", 1000, 0),
        "rootMsFLTrust": value(headline(runs, "fltrust"), "mean_root_training_seconds", 1000, 0),
        "defMsFLAME": value(headline(runs, "flame"), "mean_defense_seconds", 1000, 0),
        "defMsKrum": value(headline(runs, "multi_krum"), "mean_defense_seconds", 1000, 0),
        "evalOverhead": value(headline(runs, P), "mean_client_eval_overhead", 100, 1),
        "cleanHonVone": value(clean(runs, V1, 0.1), "honest_excluded", 1, 1),
        "cleanHonCR": value(clean(runs, P, 0.1), "honest_excluded", 1, 1),
        "cleanHonCROnly": value(clean(runs, "antiflipper_cr", 0.1), "honest_excluded", 1, 1),
        "cleanHonBudget": value(clean(runs, "antiflipper_budget", 0.1), "honest_excluded", 1, 1),
        "cleanHonZthree": value(clean(runs, "antiflipper_cc_z3", 0.1), "honest_excluded", 1, 1),
        "cleanAccVone": value(clean(runs, V1, 0.1), "last10_accuracy_mean", 100),
        "cleanAcc": value(clean(runs, P, 0.1), "last10_accuracy_mean", 100),
        "cleanAccCROnly": value(clean(runs, "antiflipper_cr", 0.1), "last10_accuracy_mean", 100),
        "cleanAccCConly": value(clean(runs, "cc_only", 0.1), "last10_accuracy_mean", 100),
        "cleanAccMid": value(clean(runs, P, 0.5), "last10_accuracy_mean", 100),
        "cleanAccMidVone": value(clean(runs, V1, 0.5), "last10_accuracy_mean", 100),
        "skewAcc": value(attacked_skew(runs, P), "last10_accuracy_mean", 100),
        "skewAccStd": f"{stat(attacked_skew(runs, P), 'last10_accuracy_mean', 100)[1]:.1f}",
        "skewAccFedAvg": value(attacked_skew(runs, "fedavg"), "last10_accuracy_mean", 100),
        "skewAccFedAvgStd": f"{stat(attacked_skew(runs, 'fedavg'), 'last10_accuracy_mean', 100)[1]:.1f}",
        "skewAccLFighter": value(attacked_skew(runs, "lfighter"), "last10_accuracy_mean", 100),
        "skewAccFLTrust": value(attacked_skew(runs, "fltrust"), "last10_accuracy_mean", 100),
        "skewAccVone": value(attacked_skew(runs, V1), "last10_accuracy_mean", 100),
        "skewRecall": value(attacked_skew(runs, P), "final_recall", 1, 2),
        "skewDet": value(attacked_skew(runs, P), "full_detection_round", 1, 0),
        "fakeRecallVone": value(adaptive(runs, V1), "final_recall", 1, 2),
        "fakeRecallCROnly": value(adaptive(runs, "antiflipper_cr"), "final_recall", 1, 2),
        "fakeRecall": value(adaptive(runs, P), "final_recall", 1, 2),
        "fakeAcc": value(adaptive(runs, P), "last10_accuracy_mean", 100),
        "ratioAccLow": value(ratio(runs, P, 0.1), "last10_accuracy_mean", 100),
        "ratioAccFedAvgLow": value(ratio(runs, "fedavg", 0.1), "last10_accuracy_mean", 100),
        "ratioAccThirty": value(ratio(runs, P, 0.3), "last10_accuracy_mean", 100),
        "ratioAccFedAvgThirty": value(ratio(runs, "fedavg", 0.3), "last10_accuracy_mean", 100),
        "ratioAccLFighterThirty": value(ratio(runs, "lfighter", 0.3), "last10_accuracy_mean", 100),
        "bnsafeHead": value(select(runs, experiment="A1_headline", method="fltrust_bnsafe", dataset="CIFAR10", distribution="NON_IID"), "last10_accuracy_mean", 100),
        "mnistAcc": value(headline(runs, P, "MNIST", "NON_IID"), "last10_accuracy_mean", 100),
        "nRuns": str(len([r for r in runs if r["method"] != "fltrust_bnsafe"])),
    }
    write("numbers.tex", "% generated by make_results.py -- do not edit\n" +
          "".join(rf"\newcommand{{\{k}}}{{{v}}}" "\n" for k, v in macros.items()))


def main() -> None:
    runs = load_runs()
    FIG.mkdir(parents=True, exist_ok=True)
    for build in (table_headline, table_detection, table_heterogeneity, table_adaptive, table_cost, table_sensitivity,
                  figure_ratio, figure_trajectories, numbers):
        build(runs)
    counts = defaultdict(int)
    for r in runs:
        counts[r["_profile"]] += 1
    print(f"{len(runs)} runs {dict(counts)} -> {OUT}, {FIG}/gen_*.pdf")


if __name__ == "__main__":
    main()
