# AntiFLipper paper (v10)

`AntiFlipper_v10.tex`: 8 pages, IEEE conference format, written as a standalone paper (no revision history).
The method is **AntiFLipper = class-conditional residual (CR) + update-consistency check (CC) + 40% exclusion cap**
(`antiflipper_cr_cc` in the code). The original report-only design appears only as the **"Raw reports"** ablation.

No result number is typed by hand. Tables and figures come from `make_results.py`, in-text numbers are macros in
`generated/numbers.tex`, and the architecture figure is TikZ (`fig/architecture.tex`).

| Element | Content | Source |
|---|---|---|
| Fig. 1 | Architecture (TikZ) | `fig/architecture.tex` |
| Table I | Comparison with 6 defenses (CIFAR-10 non-IID, IID, MNIST) | A1 |
| Table II | Ablation: raw, raw+cap, raw+CC, CC only (matched z=3 + cap), CR+cap, full method | A1, A2, A7, A8 |
| Table III | Strong skew (α_D=0.1, 40% attackers) vs. baselines | A2 |
| Table IV | Attackers that try to stay hidden: partial flips 50/25%, report faking at α_D=1 and 0.1 (incl. CR-only and CC-only rows) | A1, A4, A7, A8 |
| Table V | Cost: matched CPU benchmark (i7-8700, 6 threads, 5 iterations, std in footnote) and in-training GPU time | `timing_matched/`, A1 |
| Table VI | Sensitivity of the full method (MNIST non-IID, held-out seeds) | A9 |
| Fig. 2 | No attack: honest clients excluded and accuracy vs. α_D, with the FedAvg control | A2 |
| Fig. 3 | One clean α_D=0.1 run (seed 19, the middle seed by number): cascade with raw reports vs. none | A2 |
| Fig. 4 | Accuracy and recall vs. share of attackers (10–40%) | A1, A3 |

## Reproduce

```bash
python Paper/v10/make_results.py      # loads profiles core, cr, fltrust_fix, rev
cd Paper/v10 && latexmk -pdf AntiFlipper_v10.tex
```

`make_results.py` searches, in order: `Results_Backup_2026-11-24_NDAG/.../results` (core),
`Results_Backup_2026-11-23/results` (cr: 84, fltrust_fix: 18, rev: 69 runs) and `Code_Journal_Experiments/results`.
It loads 402 runs; 384 enter the paper. The matched benchmark is `Results_Backup_2026-11-23/timing_matched/`
(`benchmark.py fixed_workload_benchmark`, run on the local CPU). Result backups are local and not committed.

Conventions: accuracy and ASR are means over the last 10 rounds; results are mean ± std over three seeds (7, 19, 31);
"FedAvg" is equal-weight averaging (as is AntiFLipper's initial trust), not sample-weighted; FLTrust uses the original
implementation, with the BatchNorm-safe variant also reported; in-training timings exclude rounds that shared the GPU
with other workloads. Partial flips redraw the flipped subset every round (`engine.py` seeds it with the round index).

## External-style review of v10 (2026-11-25, "weak reject") → changes

| Point | Change |
|---|---|
| 1. FedAvg is equal-weight; Dirichlet split misdescribed | Setup now says the split divides each class among clients, and gives client sizes (143–10,704 at α_D=0.1; 1,244–4,835 at α_D=1). FedAvg defined as equal-weight; limitation added. **Open:** sample-weighted FedAvg runs at α_D=0.1 (6 jobs, needs a small code change). |
| 2. Claim narrower than the title | Abstract and conclusion scope the accuracy ranking to full flips + honest reporting at α_D=1. Table IV adds CR-only (recall 0.00) and CC-only (0.04) for faking at α_D=0.1. Title kept by author decision. |
| 3. Partial-flip protocol misdescribed; no baseline | Threat model and Table IV footnote now say the subset is redrawn each round. Text says these rows show detection, not an accuracy advantage. **Open:** FedAvg with partial flips (6 jobs). |
| 4. Residual estimator includes the scored client | M2 statement made conditional; analysis paragraph covers self-inclusion; absent classes and the MAD ε specified in M2/M4/Algorithm 1. Residual separation from logs: AUC 0.73–0.91 (α_D=1) vs. 0.56–0.65 (α_D=0.1), first 10 rounds, per seed. |
| Minor | Cost footnote: hardware, threads, iterations, std. "Comparable in this benchmark" wording. Fig. 3 seed selection stated. This README updated. |

## GLOBECOM 2026 reviews → v10 (summary)

Addressed: malicious ratio 10–40% (Fig. 4); FLAME, FLTrust, LFighter, Multi-Krum, Median baselines; 3 seeds with
std; sensitivity (Table VI); report faking (Tables II, IV); partial flips (Table IV); 84 B overhead (Table V);
α_D ∈ {0.1, 0.5, 1}; honest clients under severe skew (Fig. 2–3, Table II); consistent client counts.
Partly addressed: privacy of reports (secure-aggregation variant described, not implemented); edge cost (measured
client overhead, no edge hardware); convergence (cap keeps ≥ 0.6N clients, trajectories; no proof); recent references.
Not addressed (stated as limitations): partial participation/dropout, other domains, colluding attackers, backdoors
(out of scope), CRFL, Bulyan (infeasible: needs N ≥ 4f+3 = 35 at 40% of 20 clients).

## Remaining gaps

1. Sample-weighted FedAvg at α_D=0.1, clean and attacked (review point 1).
2. FedAvg with partial flips 50/25% (review point 3).
3. Partial participation (sample e.g. 50% of clients per round); small code change.
4. Report faking under strong skew: unsolved by design (Prop. 3 + diverse honest updates).
5. Privacy variant, a third domain, edge-hardware timing: larger efforts.
