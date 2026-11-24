# Experiment list (v2 — reduced matrix, Nov 2026)

The generated manifest is the source of truth. Each row is an independent, checkpointed job.
Run `python cli.py plan --profile core` to regenerate the human-readable checklist.
The v1 258-job manifest is archived at `manifests/core_v1_258jobs.json`; its 6 completed
MNIST-IID jobs keep their ids in v2 and are skipped automatically.

**Threat model:** at most 40% of clients are malicious (enforced in `config.validate`).
**Proposed method:** AntiFLipper-CC. Plain AntiFLipper (reports only) and `cc_only`
(cross-check only) are its two halves, run as an ablation.

Jobs run **in priority order**. Stopping after P1 or P2 still leaves complete tables.

## Core profile — 231 jobs (completed 2026-11-22)

219 jobs as listed below, plus 12 `antiflipper_cc_v2` jobs added on the GPU machine (the `cc_v2` profile).
All results are in `Results_Backup_2026-11-22/`.

| Priority | ID | Purpose | Matrix | Jobs |
|---|---|---|---|---:|
| P1 | A1 | CIFAR-10 non-IID headline + ablation (40% attackers) | 9 methods × 3 seeds | 27 |
| P1 | A7 | Adaptive attacker: reports accuracy on its clean labels | AntiFLipper-CC, AntiFLipper × 3 seeds | 6 |
| P2 | A2 | Clean false-positive test, no attackers | CC, AntiFLipper, cc_only × Dirichlet {0.1, 0.5, 1.0} × 3 seeds | 27 |
| P2 | A2b | Attacked, extreme heterogeneity (Dirichlet 0.1) | 5 sweep methods + cc_only × 3 seeds | 18 |
| P2 | A3 | Malicious ratio {0.1, 0.2, 0.3} (0.4 = A1) | 5 sweep methods × 3 × 3 seeds | 45 |
| P3 | A1 | CIFAR-10 IID headline | 8 methods × 3 seeds | 24 |
| P3 | A1 | MNIST IID + non-IID (6 already complete) | 5 methods × 2 splits × 3 seeds | 30 |
| P3 | A5 | Sensitivity of CC: ρ {0.025, 0.1}, counter {3, 9}, z {2, 3} + default; attack and clean | 7 × 2 × 3 held-out seeds (MNIST non-IID) | 42 |

- Headline methods (CIFAR-10): AntiFLipper-CC, AntiFLipper, cc_only, FedAvg, Median, Multi-Krum, FLAME, LFighter, FLTrust (cc_only is non-IID only).
- Sweep methods: AntiFLipper-CC, AntiFLipper, FedAvg, LFighter, FLTrust.
- Baselines and cc_only ignore accuracy reports, so for the adaptive attacker their matched runs are the A1 jobs.

## Full profile — 306 jobs

Core plus:

| ID | Purpose | Matrix | Jobs |
|---|---|---|---:|
| A4 | Partial-example label flipping (1.0 = A1); hardest case for the cross-check | CC, AntiFLipper, cc_only, LFighter, FLTrust × {0.25, 0.5, 0.75} × 3 seeds | 45 |
| A6 | Ablations: linear update, absolute threshold, no/consecutive counter | 4 × 3 seeds (MNIST non-IID) | 12 |
| Legacy | Trimmed Mean, FoolsGold, Tolpegin on CIFAR-10 | 3 × 2 splits × 3 seeds | 18 |

## CR profile — 84 jobs (class-conditional residual, Nov 22)

**Why.** The core results showed report-based AntiFLipper failing under strong skew. At Dirichlet 0.1 with
**no attackers** it excluded 92% of honest clients (accuracy 16.9%; FedAvg 44%). Clients holding hard or rare
classes report low accuracy and look like attackers. Each exclusion removes those classes from training, which
drags the next clients down (a cascade). `cc_v2` (update check only) had 0% false positives there, but caught no attackers
at all in the headline setting (recall 0, attackers keep 35% of the weight). So the report signal is needed; it
only has to stop penalizing label mix.

**`antiflipper_cr`.** Clients report per-class correct and sample counts from the same evaluation pass
(+84 bytes per round). The server pools per-class accuracy, weighted by current trust, computes each client's
*expected* accuracy for its own label mix, and runs the unchanged trust/counter pipeline on the residual
`reported - expected`. An honest client holding hard classes has residual ~0. A flipper, whose images of class
k carry label pi(k), scores ~0 on labels the model otherwise gets right, so its residual is strongly negative.
Exclusions are capped at the 40% threat-model bound (`max_exclusion_fraction=0.4`, lowest trust first).
On a constructed skewed case, honest hard-class clients have raw deviation -0.275 (penalized every round)
but residual 0.000.

**`antiflipper_cr_cc`.** Adds the update-consistency check with `z=3.0`: in the MNIST sensitivity runs, z=3
gave 1–5% FPR at recall 1.0, vs 13–15% FPR for z=2.5.

**Order (seed-major).** Jobs 1–14 are the seed-7 pilot (~6 GPU hours). Check them before waiting for the rest.

| Block | Settings (CIFAR-10 non-IID unless noted) | Variants | Jobs |
|---|---|---|---:|
| Key, 3 seeds | clean α_D=0.1 | CR, CR+CC, budget only, CC z=3 | 12 |
| | attacked α_D=0.1 | CR, CR+CC, budget only | 9 |
| | headline (α_D=1, 40%) | CR, CR+CC, CC z=3 | 9 |
| | report-faking attacker | CR, CR+CC | 6 |
| | clean α_D=0.5 | CR, CR+CC | 6 |
| Coverage, 3 seeds | clean α_D=1; ratio 10/20/30%; CIFAR IID; MNIST IID and non-IID | CR, CR+CC | 42 |

"Budget only" (plain AntiFLipper with the 40% cap) and "CC z=3" are ablations: they show whether the residual,
the budget or the threshold produces any improvement. All other settings reuse the completed core runs as baselines.

## FLTrust-fix profile — 18 jobs

The core FLTrust aggregator rescaled every floating tensor, including BatchNorm running statistics, by
`root_norm / client_norm`. On ResNet-18 this distorted the BN statistics and gave 1–7 NaN-loss rounds in most
CIFAR-10 runs (accuracy recovered, but the baseline was not clean). Scores and rescaling now use trainable
parameters only, and BN buffers are the score-weighted average of the clients' buffers. All 18 CIFAR-10 FLTrust
jobs are re-run with `notes="bn-safe"`; `Paper/v9/make_results.py` prefers them over the core runs. MNIST's CNN
has no BatchNorm and is unaffected.

**Queue on the GPU machine:** `antiflipper-cr.service` (systemd user service, 3 workers) runs `run_queue.sh cr fltrust_fix`.

## Rev profile — 69 jobs (internal review of paper v10, Nov 24)

Experiments a reviewer of v10 asked for, in priority order (seed-major within P1).

| Priority | Block | Settings | Variants | Jobs |
|---|---|---|---|---:|
| P1 | A8 report faking + strong skew | CIFAR-10, α_D=0.1, 40%, `report_strategy=clean_labels` | CR+CC, CR, CC only (z=3, cap 0.4) | 9 |
| P1 | Matched CC only | main (α_D=1), attacked α_D=0.1, clean α_D=0.1 | CC only with the same z=3 and cap as AntiFLipper | 9 |
| P1 | Clean FedAvg control | no attackers, α_D ∈ {0.1, 0.5, 1} | FedAvg | 9 |
| P2 | A4 partial flips | label_poison_fraction ∈ {0.5, 0.25} | CR+CC, CR | 12 |
| P3 | A9 sensitivity of the full method | MNIST non-IID, held-out seeds, attack and clean | CR+CC default, z ∈ {2.5, 3.5}, c_max ∈ {3, 9} | 30 |

P1 is ~12 GPU hours with 3 workers, P2 ~5.5 h, and P3 ~7.5 h. Queue: `antiflipper-rev.service` runs `run_queue.sh rev`.

## AntiFLipper-CC

Same as AntiFLipper, plus a server-side cross-check of each report. The last-layer update of each
client is compared (cosine) with the coordinate-wise median last-layer update. A robust (median/MAD)
low-side outlier beyond `consistency_z` (default 2.5) has its accuracy report replaced by the round's
worst report before the trust update. It is also flagged for that round, exactly as if its trust were below the
threshold, so the same counter that tolerates transient non-IID deviations decides exclusion.
It needs no server data, and its cost is O(n·|last layer|).

`cc_only` keeps the cross-check and the counter but ignores the reports: there is no trust update, and
weights are uniform over the clients not yet excluded.

The first CC version only replaced the report. In a sanity run (MNIST non-IID, 10 clients, 8 rounds) that slowed trust
too little: 4/4 attackers were flagged every round, but none were excluded. Flagging toward the counter fixed this
(4/4 excluded, FPR 0, with both honest and report-faking attackers).

## What every job records

`results/<profile>/<job>/`:

| File | Content |
|---|---|
| `config.json` | full configuration |
| `partition_metadata.json` | client sizes, index hashes, attacker ids, FLTrust root set |
| `round_metrics.csv` | one row per round (below) |
| `client_history.csv` | one row per client per round: attacker, selected, excluded, raw and effective report, consistency similarity, inconsistent flag, trust, counter, aggregation weight, local loss |
| `latest_detection.json` | current trust, counters, exclusions |
| `summary.json` | per-job aggregates (below) |
| `console.log`, `error.txt`, `state.json` | logs and status |

Per round:
- **Accuracy:** test accuracy, loss, ASR, macro and worst class accuracy, mean local loss.
- **Detection:** cumulative and per-round TP/FP/FN/TN, precision, recall, F1 and FPR; excluded and aggregated clients; inconsistent reports; mean detection round.
- **Server time:** pre-aggregation defense (trust update and cross-check), aggregation, total defense, FLTrust root training, test evaluation, full round.
- **Client time:** report evaluation and local training (mean and max), evaluation overhead (eval/train), evaluation samples.
- **Memory:** peak GPU allocated and reserved MB, peak process RSS MB (the server aggregates on CPU).
- **Communication:** download and upload bytes this round.

Per job (`summary.json`):
- final and last-10-round accuracy, accuracy AUC, final ASR
- final and mean per-round detection metrics; honest vs. attacker exclusions; mean and full detection round; first honest exclusion
- means of every timing above, plus maxima of client times
- peak memory, total upload and download bytes, parameter count, total wall time
- device, torch version, platform

Profile-level: `results/<profile>/environment/` (GPU, CUDA, `pip freeze`), `analysis/<profile>/` (mean ± std ± 95% CI across seeds, plots, REPORT.md), and `analysis/<profile>/timing/` (matched-N aggregation benchmark).

Jobs that ran before a metric was added lack its column; analysis skips missing values.

## Metrics added in v2

- Per-round detection (`round_precision/recall/f1/fpr`) for every method that drops clients (zero weight), so
  LFighter, FLTrust and Multi-Krum get comparable detection numbers.
- `defense_seconds` = trust update/consistency check + aggregation. AntiFLipper is no longer timed on the weighted average alone.
  FLTrust root training is reported separately as `root_training_seconds`.
- `upload_bytes_per_client_round`: analytic communication cost (model + 4-byte scalar for report-based methods).
- `honest_excluded`, `attackers_excluded`, `final_surviving_clients`, `inconsistent_reports`.

## Smoke profile — 6 jobs

An offline two-round synthetic run (AntiFLipper, AntiFLipper-CC with the adaptive attacker, cc_only, LFighter).
It only verifies the pipeline and is not publishable evidence.
