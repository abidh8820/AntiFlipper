# AntiFLipper paper (v10)

`AntiFlipper_v10.tex`: 7 pages, IEEE conference format, written as a standalone paper (no revision history).
The method is **AntiFLipper = class-conditional residual (CR) + update-consistency check (CC) + 40% exclusion cap**
(`antiflipper_cr_cc` in the code). The original report-only design appears only as the **"Raw reports"** ablation.

No result number is typed by hand. Tables and figures come from `make_results.py`, in-text numbers are macros in
`generated/numbers.tex`, and the architecture figure is TikZ (`fig/architecture.tex`).

| Element | Content |
|---|---|
| Fig. 1 | Architecture (TikZ) |
| Table I | Comparison with 6 defenses: CIFAR non-IID accuracy, ASR, precision, recall, FPR; CIFAR IID and MNIST accuracy; server time |
| Table II | Ablation (CR / CC / cap checkmarks) on four scenarios: main, report faking, no attack at α_D=0.1, attack at α_D=0.1 |
| Table III | Strong heterogeneity vs. baselines (α_D=0.1, 40%) |
| Table IV | Measured cost (server data, server time, client overhead, extra upload) |
| Table V | Sensitivity (ρ, c_max, z) |
| Fig. 2 | No attack: honest clients excluded and accuracy vs. α_D |
| Fig. 3 | Representative run: cascade with raw reports vs. none with AntiFLipper |
| Fig. 4 | Accuracy and recall vs. malicious ratio |

## Rebuild

```bash
python Paper/v10/make_results.py      # reads Results_Backup_2026-11-24_NDAG (core, cr, fltrust_fix)
cd Paper/v10 && latexmk -pdf AntiFlipper_v10.tex
```

Conventions: accuracy is the mean over the last 10 rounds; results are mean ± std over three seeds (seed values are not
named in the paper); FLTrust uses the original implementation (the BatchNorm-safe variant is in a footnote); timings
exclude rounds that shared the machine with other workloads.

## Reviewer comments (GLOBECOM 2026) → v10

✅ addressed with new experiments or analysis · ◐ partly addressed · ✗ not addressed (stated as a limitation)

| # | Comment | Status | Where in v10 |
|---|---|---|---|
| R1-1 | Failure threshold vs. malicious ratio (10–50%) | ✅ (10–40%) | Fig. 4: recall 1.0 at 10–40%; Prop. 2: the residual gap does not shrink with p. 50% is outside the threat model (median breakdown), and §II says why. |
| R1-2 | Compare with Bulyan, FLAME, CRFL, trust-based methods | ◐ | FLAME, FLTrust, LFighter, Multi-Krum, Median: all run. Bulyan is infeasible at 40% with 20 clients (needs N ≥ 4f+3 = 35), explained in §III. CRFL (a certified backdoor defense) was not run. |
| R1-2 | General model poisoning, backdoors | ✗ (scope) | Out of scope in the threat model; combinable with robust aggregation. |
| R1-3 | Seeds, mean ± std | ✅ | 3 seeds everywhere, mean ± std in every accuracy cell. |
| R1-3 | Hyperparameter sensitivity | ✅ | Table V (ρ, c_max, z; held-out seeds). CR adds no tuned hyperparameter. |
| R1-3 | Dropout, stragglers, partial participation | ✗ | Limitations. |
| R1-4 | Other domains, lightweight models | ✗ | Limitations (two image datasets, CNN and ResNet-18). |
| R1-5, R3-4 | Privacy of accuracy reports | ◐ | §V Privacy. Per-class counts reveal label histograms (worse than v1); proposed fix: secure aggregation of the pooled sums plus a client-side residual. Not implemented, and no formal analysis. |
| R1-6, R3 | Novelty vs. prior validation-score work | ✅ | New contributions: the label-mix diagnosis (Prop. 1), the class-conditional residual (Prop. 2), and the faking impossibility (Prop. 3) plus the consistency check. Positioning paragraph in §III. |
| R1-7 | Edge-device memory, energy, latency | ◐ | Measured client eval/train ratio (6.8% on CIFAR-10) and server cost (Table II, Table III, Fig. 2); no edge hardware, no energy. |
| R1-1 | Detection when the global model is poisoned | ◐ | Prop. 2; the ratio sweep shows recall 1.0 up to 40%. No formal bound on global-model corruption. |
| R2-1 | Client reports a fake accuracy | ✅ | Report-faking attacker (§II, Prop. 3, Table II): raw reports recall 0.67, CR alone 0.62, full method 1.00. |
| R2-2 | Communication overhead | ✅ | 84 bytes per client and round vs. a 44.7 MB update (§V, Table IV). |
| R2-3 | Single runs | ✅ | 3 seeds, 315 runs. |
| R2-4 | Only Dirichlet α=1 | ✅ | α_D ∈ {0.1, 0.5, 1}, with and without attackers (Table II, Table III, Fig. 2). |
| R2-5 | τ_thresh / cnt_max values and sensitivity | ✅ | §VI and Table I list all values; Table V. |
| R2-6 | Why faster than FedAvg | ✅ | §VII-F: after exclusion it averages ~12 instead of 20 models; the defense time now includes all checks. |
| R2-7 | Inconsistent client counts | ✅ | 20 (CIFAR) and 100 (MNIST) for both IID and non-IID, stated in §VI. |
| R2-8 | Convergence | ◐ | §V: FedAvg guarantees apply once exclusions stop, and the cap keeps ≥ 0.6N clients; Fig. 3 shows per-round accuracy and surviving clients. No new proof. |
| R2-9 | Few recent references | ◐ | 9 references from 2023–2026 (HSCS, data quality, Confident FL, LFighter, LFR-PPFL, systematic review, FLGT, AI-to-protect, FedAlign). More could be added. |
| R2-10 | α with three meanings | ✅ | Notation table: η trust rate, α_D Dirichlet; the evaluation fraction γ is dropped (always 1). |
| R2-11, R3-3 | Outline skips Section VII | ✅ | Outline built from `\ref`. |
| R3 | Adaptive attackers mimicking honest accuracy | ✅ | Report faking (Table II). |
| R3 | Colluding attackers | ✗ | Limitations. |
| R3-1 | Partial-flip attacks | ✗ | Limitations. The `full` profile has a partial-flip block (A4), but only for the earlier variants; it was not run. |
| R3 | Honest clients with severe skew penalized | ✅ | The main new result: at α_D=0.1 with no attackers, honest clients excluded drop from 18.3 to 0 of 20 (Table II, Figs. 2–3). |
| R3-2 | FLTrust on identical splits, detection P/R | ✅ | Table I (P/R/FPR for every method that drops clients); all methods share partitions, attackers and seeds. |

## Remaining gaps, by value for a resubmission

1. **Partial-flip attack** (R3-1): about 18 GPU hours for CR+CC, CR and v1 at flip fractions 0.25/0.5/0.75 × 3 seeds on CIFAR non-IID. This is the most-requested missing experiment.
2. **Partial participation / dropout** (R1-3): sample, say, 50% of clients per round. This needs a small code change.
3. **Privacy variant** (R1-5, R3-4): implement the secure-aggregation sums plus the client-side residual. Detection is unchanged by construction; only the protocol changes.
4. CRFL comparison, a third dataset or domain, and edge-hardware timing: larger efforts.
