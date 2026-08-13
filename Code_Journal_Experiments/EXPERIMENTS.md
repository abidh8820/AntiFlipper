# Experiment list

The generated manifest is the source of truth. Each row is an independent, checkpointed job. Run `python cli.py plan --profile core` to regenerate the human-readable checklist.

## Core profile — 258 jobs

| ID | Purpose | Matrix | Jobs |
|---|---|---:|---:|
| A1 | Multi-seed headline replication and FLTrust addition | 10 methods × 2 datasets × 2 distributions × 3 seeds | 120 |
| A2 | Clean false-positive stress test | AntiFLipper × 2 datasets × (IID + Dirichlet 0.1/0.3/1.0) × 3 seeds | 24 |
| A3 | Malicious-ratio/failure-boundary sweep | 4 methods × 6 ratios × CIFAR-10 non-IID × 3 seeds | 72 |
| A5-lite | One-factor sensitivity with matched clean controls | 7 unique settings × attack/no-attack × 3 held-out seeds | 42 |

The ten A1 methods are AntiFLipper, FedAvg, Median, Trimmed Mean, FoolsGold, Tolpegin, Multi-Krum, FLAME, LFighter, and FLTrust. The A3 subset is AntiFLipper, FedAvg, LFighter, and FLTrust.

## Full profile — 330 jobs

The full profile contains every core job plus:

| ID | Purpose | Matrix | Jobs |
|---|---|---:|---:|
| A4-lite | True partial-example label-flipping | 4 methods × 5 poison fractions × 3 seeds | 60 |
| A6-lite | Key method ablations | 4 ablations × 3 seeds | 12 |

The convergence analysis B5 does not create extra jobs. It uses the per-round logs from the experiments above.

## Smoke profile — 1 job

An offline two-round synthetic run used only to verify checkpointing, logging, analysis, and packaging. It is not publishable evidence.

## Status semantics

- `pending`: never started;
- `running`: process began but has not finalized;
- `completed`: summary and final state were written;
- `failed`: exception details are in the job's `error.txt`;
- `interrupted`: the launcher was stopped; rerunning resumes from `checkpoint.pt`.

Completed jobs are never rerun unless their result directory is deliberately moved elsewhere. Failed and interrupted jobs are retried by default.
