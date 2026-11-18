# Experiment list (v2 — reduced matrix, Nov 2026)

The generated manifest is the source of truth. Each row is an independent, checkpointed job.
Run `python cli.py plan --profile core` to regenerate the human-readable checklist.
The v1 258-job manifest is archived at `manifests/core_v1_258jobs.json`; its 6 completed
MNIST-IID jobs keep their ids in v2 and are skipped automatically.

Jobs run **in priority order**. Stopping after P1 or P2 still leaves complete tables.

## Core profile — 171 jobs

| Priority | ID | Purpose | Matrix | Jobs |
|---|---|---|---|---:|
| P1 | A1 | CIFAR-10 non-IID headline (main operating point) | 8 methods × 3 seeds | 24 |
| P1 | A7 | Adaptive attacker: reports accuracy on its clean labels | AntiFLipper, AntiFLipper-CC × 3 seeds | 6 |
| P2 | A3 | Malicious-ratio boundary (0.4 comes from A1) | 4 methods × {0.2, 0.3, 0.5} × 3 seeds | 36 |
| P2 | A2 | Clean false-positive test, no attackers | AntiFLipper, -CC × Dirichlet {0.1, 0.5, 1.0} × 3 seeds | 18 |
| P2 | A2b | Attacked, extreme heterogeneity (Dirichlet 0.1) | 4 methods × 3 seeds | 12 |
| P3 | A1 | CIFAR-10 IID headline | 7 methods × 3 seeds | 21 |
| P3 | A1 | MNIST IID + non-IID (6 already complete) | 4 methods × 2 splits × 3 seeds | 24 |
| P3 | A5 | Sensitivity: ρ ∈ {0.025, 0.1}, counter ∈ {3, 9} + default, attack and clean | 5 × 2 × 3 held-out seeds (MNIST non-IID) | 30 |

Headline methods (CIFAR-10): AntiFLipper, AntiFLipper-CC, FedAvg, Median, Multi-Krum, FLAME, LFighter, FLTrust.
Sweep methods: AntiFLipper, FedAvg, LFighter, FLTrust.

## Full profile — 237 jobs

Core plus:

| ID | Purpose | Matrix | Jobs |
|---|---|---|---:|
| A4 | Partial-example label flipping (1.0 comes from A1) | 4 methods × {0.25, 0.5, 0.75} × 3 seeds | 36 |
| A6 | Ablations: linear update, absolute threshold, no/consecutive counter | 4 × 3 seeds (MNIST non-IID) | 12 |
| Legacy | Trimmed Mean, FoolsGold, Tolpegin on CIFAR-10 | 3 × 2 splits × 3 seeds | 18 |

## AntiFLipper-CC (new)

Same as AntiFLipper, plus a server-side cross-check of each report. The last-layer update of each
client is compared (cosine) with the coordinate-wise median last-layer update. A robust (median/MAD)
low-side outlier beyond `consistency_z` (default 2.5) has its accuracy report replaced by the round's
worst report before the trust update. It is also flagged for that round, exactly as if its trust were below the
threshold, so the same counter that tolerates transient non-IID deviations decides exclusion.
It needs no server data, and its cost is O(n·|last layer|).

The first version only replaced the report. In a sanity run (MNIST non-IID, 10 clients, 8 rounds) that slowed trust
too little: 4/4 attackers were flagged every round, but none were excluded. Flagging toward the counter fixed this.

## Metrics added in v2

- Per-round detection (`round_precision/recall/f1/fpr`) for every method that drops clients (zero weight), so
  LFighter, FLTrust and Multi-Krum get comparable detection numbers.
- `defense_seconds` = trust update/consistency check + aggregation. AntiFLipper is no longer timed on the weighted average alone.
  FLTrust root training is reported separately as `root_training_seconds`.
- `upload_bytes_per_client_round`: analytic communication cost (model + 4-byte scalar).
- `honest_excluded`, `attackers_excluded`, `final_surviving_clients`, `inconsistent_reports`.

## Smoke profile — 3 jobs

An offline two-round synthetic run (AntiFLipper, AntiFLipper-CC with the adaptive attacker, LFighter).
It only verifies the pipeline and is not publishable evidence.
