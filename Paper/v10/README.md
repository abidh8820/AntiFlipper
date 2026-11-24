# AntiFLipper v10

`AntiFlipper_v10.tex` (7 pages, IEEE conference format). The proposed method is **AntiFLipper = class-conditional
residual (CR) + update-consistency check (CC) + 40% exclusion cap** (`antiflipper_cr_cc` in the code). The
original method is called **v1** and appears as an ablation.

No result number is typed by hand. Tables and figures are produced by `make_results.py`, and in-text numbers
are macros in `generated/numbers.tex`.

## Rebuild

```bash
python Paper/v10/make_results.py      # reads Results_Backup_2026-11-24_NDAG (core, cr, fltrust_fix)
cd Paper/v10 && latexmk -pdf AntiFlipper_v10.tex
```

Conventions used by the generator:
- Accuracy is the mean of the last 10 rounds; under strong skew a single round can swing by tens of points.
- FLTrust uses the original core runs, the stronger of the two implementations. The BatchNorm-safe re-runs go in a footnote.
- Timings drop rounds that ran while another workload shared the GPU machine (`Results_Backup_2026-11-23/round_timing_tagged.csv`).

## Reviewer comments (GLOBECOM 2026) → v10

✅ addressed with new experiments or analysis · ◐ partly addressed · ✗ not addressed (stated as a limitation)

| # | Comment | Status | Where in v10 |
|---|---|---|---|
| R1-1 | Failure threshold vs. malicious ratio (10–50%) | ✅ (10–40%) | Fig. 2: recall 1.0 at 10–40%; Prop. 2: the residual gap does not shrink with p. 50% is outside the threat model (median breakdown), and §II says why. |
| R1-2 | Compare with Bulyan, FLAME, CRFL, trust-based methods | ◐ | FLAME, FLTrust, LFighter, Multi-Krum, Median: all run. Bulyan is infeasible at 40% with 20 clients (needs N ≥ 4f+3 = 35), explained in §III. CRFL (a certified backdoor defense) was not run. |
| R1-2 | General model poisoning, backdoors | ✗ (scope) | Out of scope in the threat model; combinable with robust aggregation. |
| R1-3 | Seeds, mean ± std | ✅ | 3 seeds everywhere, mean ± std in every accuracy cell. |
| R1-3 | Hyperparameter sensitivity | ✅ | Table VII (ρ, c_max, z; held-out seeds). CR adds no tuned hyperparameter. |
| R1-3 | Dropout, stragglers, partial participation | ✗ | Limitations. |
| R1-4 | Other domains, lightweight models | ✗ | Limitations (two image datasets, CNN and ResNet-18). |
| R1-5, R3-4 | Privacy of accuracy reports | ◐ | §V Privacy. Per-class counts reveal label histograms (worse than v1); proposed fix: secure aggregation of the pooled sums plus a client-side residual. Not implemented, and no formal analysis. |
| R1-6, R3 | Novelty vs. prior validation-score work | ✅ | New contributions: the label-mix diagnosis (Prop. 1), the class-conditional residual (Prop. 2), and the faking impossibility (Prop. 3) plus the consistency check. Positioning paragraph in §III. |
| R1-7 | Edge-device memory, energy, latency | ◐ | Measured client eval/train ratio (6.8% on CIFAR-10) and server cost (Table VI); no edge hardware, no energy. |
| R1-1 | Detection when the global model is poisoned | ◐ | Prop. 2; the ratio sweep shows recall 1.0 up to 40%. No formal bound on global-model corruption. |
| R2-1 | Client reports a fake accuracy | ✅ | Report-faking attacker (§II, Prop. 3, Table V): v1 recall 0.67, CR alone 0.62, with CC 1.00. |
| R2-2 | Communication overhead | ✅ | 84 bytes per client and round vs. a 44.7 MB update (§V, Table VI). |
| R2-3 | Single runs | ✅ | 3 seeds, 315 runs. |
| R2-4 | Only Dirichlet α=1 | ✅ | α_D ∈ {0.1, 0.5, 1}, with and without attackers (Table IV). |
| R2-5 | τ_thresh / cnt_max values and sensitivity | ✅ | §VI-A and Table I list all values; Table VII. |
| R2-6 | Why faster than FedAvg | ✅ | §VI-F: after exclusion it averages ~12 instead of 20 models; the defense time now includes all checks. |
| R2-7 | Inconsistent client counts | ✅ | 20 (CIFAR) and 100 (MNIST) for both IID and non-IID, stated in §VI-A. |
| R2-8 | Convergence | ◐ | §V: FedAvg guarantees apply once exclusions stop, and the cap keeps ≥ 0.6N clients; Fig. 3 shows per-round accuracy and surviving clients. No new proof. |
| R2-9 | Few recent references | ◐ | 9 references from 2023–2026 (HSCS, data quality, Confident FL, LFighter, LFR-PPFL, systematic review, FLGT, AI-to-protect, FedAlign). More could be added. |
| R2-10 | α with three meanings | ✅ | Notation table: η trust rate, α_D Dirichlet; the evaluation fraction γ is dropped (always 1). |
| R2-11, R3-3 | Outline skips Section VII | ✅ | Outline built from `\ref`. |
| R3 | Adaptive attackers mimicking honest accuracy | ✅ | Report faking (Table V). |
| R3 | Colluding attackers | ✗ | Limitations. |
| R3-1 | Partial-flip attacks | ✗ | Limitations. The `full` profile has a partial-flip block (A4), but only for the v1 variants; it was not run. |
| R3 | Honest clients with severe skew penalized | ✅ | The main new result: at α_D=0.1 with no attackers, honest clients excluded drop from 18.3 to 0 of 20 (Table IV, Fig. 3). |
| R3-2 | FLTrust on identical splits, detection P/R | ✅ | Table III (P/R/FPR for every method that drops clients); all methods share partitions, attackers and seeds. |

## Remaining gaps, by value for a resubmission

1. **Partial-flip attack** (R3-1): about 18 GPU hours for CR+CC, CR and v1 at flip fractions 0.25/0.5/0.75 × 3 seeds on CIFAR non-IID. This is the most-requested missing experiment.
2. **Partial participation / dropout** (R1-3): sample, say, 50% of clients per round. This needs a small code change.
3. **Privacy variant** (R1-5, R3-4): implement the secure-aggregation sums plus the client-side residual. Detection is unchanged by construction; only the protocol changes.
4. CRFL comparison, a third dataset or domain, and edge-hardware timing: larger efforts.
