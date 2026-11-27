# AntiFLipper paper (v11)

Two versions built from the same generated tables, figures and macros:

| File | Format | Length |
|---|---|---|
| `AntiFlipper_v11.tex` | IEEE conference (GLOBECOM style) | **8 pages** including references (7 of body); no overfull boxes |
| `AntiFlipper_v11_journal.tex` | IEEE Transactions (`journal` class) | **12 pages** including references; no overfull boxes |

v11 is a full rewrite of v10 (`../v10`, left unchanged). It adds the rev2 controls, the rev3 runs (partial participation,
baseline partial flips) and a corrected account of report faking.

The journal version extends the conference text. It adds:
- research questions Q1–Q4;
- a related-work comparison table (Table I), a notation table and a hyperparameter table;
- a formal self-inclusion-bias result (Prop. 4, Cor. 1) with proof;
- a privacy protocol sketch;
- per-seed results under strong skew (`table_seeds`);
- a report-separation AUC table (`table_auc`);
- a participation trajectory figure (`gen_participation.pdf`);
- the sensitivity table;
- a Discussion section (deployment guidance, threats to validity).

The method is **AntiFLipper = class-conditional residual (CR) + update-consistency check (CC) + 40% exclusion cap**
(`antiflipper_cr_cc` in the code).

No result number is typed by hand. Tables and figures come from `make_results.py`, in-text numbers are macros in
`generated/numbers.tex`, and the architecture figure is TikZ (`fig/architecture.tex`).

| Element | Content | Source |
|---|---|---|
| Fig. 1 | Architecture (TikZ) | `fig/architecture.tex` |
| Table I | Comparison with 6 defenses (CIFAR-10 non-IID, IID, MNIST) | A1 |
| Table II | Ablation, plus FedAvg and sample-weighted FedAvg reference rows | A1, A2, A7, A8, rev2 |
| Table III | Strong skew with attackers: accuracy, **ASR**, recall, FPR; includes sample-weighted FedAvg | A2, rev2 |
| Table IV (`table_partial`, table*) | Partial flips 50/25% and **50% participation** for FedAvg, LFighter, FLTrust (BN-safe) and AntiFLipper | rev, rev2, rev3 |
| Table V (`table_adaptive`) | Report faking at α_D=1 and 0.1 with CR-only and CC-only rows; FedAvg columns | A1, A7, A8 |
| Table VI | Cost: matched CPU benchmark and in-training GPU time | `timing_matched/`, A1 |
| Fig. 2 | No attack: exclusions and accuracy vs. α_D; sample-weighted FedAvg marker at α_D=0.1 | A2, rev2 |
| Fig. 3 | One clean α_D=0.1 run (seed 19): cascade with raw reports vs. none | A2 |
| Fig. 4 | Accuracy and recall vs. share of attackers (10–40%) | A1, A3 |

Sensitivity (A9) is now reported in the text through the macros `sensAccMin/Max`, `sensRecallMin` and `sensFPRmax`.
`generated/table_sensitivity.tex` is still produced but is not included in the paper.

## Reproduce

```bash
python Paper/v11/make_results.py      # loads profiles core, cr, fltrust_fix, rev, rev2, rev3
cd Paper/v11 && latexmk -pdf AntiFlipper_v11.tex && latexmk -pdf AntiFlipper_v11_journal.tex
```

Result backups are local and not committed. Pull them with `Results_Backup_2026-11-23/pull_results.sh`.

## What the rev2 results (2026-11-25) showed, and how v11 reports them

All figures below are on CIFAR-10 at α_D=0.1 unless stated otherwise.

| Comparison | Result | Where in the paper |
|---|---|---|
| No attack: sample-weighted FedAvg vs. AntiFLipper | 59.7±10.5 vs. 54.3±9.8. Weighted FedAvg is ahead **in all 3 seeds** | VII-B, stated plainly: AntiFLipper costs a few points without attackers under strong skew |
| 40% attackers: sample-weighted FedAvg vs. AntiFLipper | 44.8 vs. 49.1. AntiFLipper is ahead in 2 of 3 seeds (3 of 3 against equal-weight FedAvg). ASR is 21.4 vs. 7.5; **AntiFLipper has the lowest ASR of all methods in every seed** | VII-C, Table III |
| Partial flips at α_D=1: FedAvg vs. AntiFLipper | 71.9/73.6 vs. 71.3/72.1. FedAvg is slightly ahead because these attacks are weak; ASR 5.0 → 3.2 at 50% | VII-D: "detection survives a weaker signal, not that it improves accuracy" |

## What the rev3 results (2026-11-26) showed

The rev3 runs are all CIFAR-10 at α_D=1 with 40% attackers and 3 seeds. FLTrust in rev3 is the BN-safe variant;
its full-participation accuracy is 57.8 (the reference variant reaches 69.2).

**Partial participation (half of the active clients sampled per round):**
- AntiFLipper reaches 71.0±1.0 (72.0 with full participation). It excludes every attacker with no false positives
  in 3/3 seeds, although detection is slower (round 34 vs. 15).
- LFighter reaches 63.6, FLTrust (BN-safe) 62.1 and FedAvg 46.1.
- Baseline detection is measured on sampled clients over the last 10 rounds, because the final-round metric counts
  unsampled attackers as misses. On that measure LFighter catches 0.84 of sampled attackers, with FPR 0.13.

**Partial flips for the baselines:**
- LFighter and FedAvg land at about 70–74%, AntiFLipper at 71.3/72.1, and FLTrust (BN-safe) at 70.5/66.1. FLTrust's
  66.1 includes one unstable run at 25% flips (56.3). The attack is weak.
- Detection at 50%/25% flips: LFighter finds 0.46/0.17 of attackers and FLTrust (BN-safe) 0.67/0.50, both with false
  positives. AntiFLipper finds them all, with no false positives.

**Code change:** in `engine.py`, `_update_trust` now applies the relative threshold to the number of non-excluded
clients rather than the number sampled this round. Full-participation runs are unchanged.

## New finding: report faking (corrects v10)

v10 said that faked reports cannot be told apart from honest ones (Prop. 3), yet report-only variants still caught
62–67% of fakers at α_D=1. The client logs explain this:

- **At α_D=1:** faked reports still rank attackers lower (residual AUC 0.70–0.73). The model has already trained on
  the attackers' flipped samples. Prop. 3 now states this assumption and the text calls the effect "leakage".
- **At α_D=0.1:** the raw faked reports separate attackers (AUC 0.64–0.73), but **the residual removes that signal**
  (0.46–0.58). Classes held mostly by attackers have a reference Â_k made largely of the attackers' own reports, which
  is the self-inclusion bias of Section V.
- **What this means:** faking under strong skew succeeds because of the label-mix correction, not because faked
  reports look honest. A leave-one-out or robust per-class reference is the natural fix (future work, would need runs).

## Review points → v11

| Source | Point | v11 |
|---|---|---|
| GLOBECOM R1 | ratio 10–50% | 10–40% (Fig. 4); the threat model caps at 40% |
| GLOBECOM R1 | Bulyan, FLAME, CRFL, trust-based baselines | FLAME, FLTrust, LFighter, Multi-Krum, Median. Bulyan is inapplicable (N≥4f+3) and CRFL targets backdoors; both are stated |
| GLOBECOM R1/R2 | seeds and std, sensitivity, communication cost | 3 seeds with std; sensitivity in VII-G; 84 B report (Table V) |
| GLOBECOM R1 | other domains, edge hardware, dropout | Partial participation run (rev3, Table IV); data-dependent dropout, domains and hardware in Limitations |
| GLOBECOM R1/R3 | privacy of reports, including repeated rounds | Theoretical Analysis "Privacy" (secure-aggregation variant, not implemented) |
| GLOBECOM R2 | fake reports | Threat model, Prop. 3, Table IV, plus the leakage and residual analysis |
| GLOBECOM R2 | α_D=0.1 | Fig. 2, Fig. 3, Tables II–IV |
| GLOBECOM R2 | convergence | Exclusion bound plus trajectories; no proof (stated) |
| GLOBECOM R2 | α symbol reuse, section outline, client counts | Distinct symbols (η, α_D); no outline paragraph; the same client count for IID and non-IID |
| GLOBECOM R3 | partial and adaptive flips; FLTrust with precision and recall on the same splits | Table IV; Table I |
| Ext. review 1 | sample-weighted FedAvg | Run (rev2) and reported in Tables II–III, Fig. 2 and the text |
| Ext. review 3 | partial-flip baseline | FedAvg run (rev2), Table IV |
| Ext. review 4 | self-inclusion in the residual | M2, the estimation-bias paragraph, and the new faking analysis |
| Ext. follow-up | narrower claims | Abstract and conclusion scoped by setting; "1 of 24"; "18.3 of 20 on average to none" |

## Remaining gaps (stated in Limitations)

1. Report faking under strong skew (1 of 24). Fix candidate: a leave-one-out or robust per-class reference.
2. Without attackers under strong skew, AntiFLipper is less accurate than FedAvg (both variants). Sample-weighted trust is untested.
3. Participation tested at one rate with uniform sampling only; data-dependent dropout; colluding attackers or attackers that align with CC; other domains; edge hardware.
4. The privacy variant is not implemented. There is no convergence proof. Three seeds only.
