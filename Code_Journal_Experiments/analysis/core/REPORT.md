# AntiFLipper automated analysis

Completed runs analyzed: **0**.

## Included experiment groups

- None yet

## Interpretation notes

- Seed-level runs are the independent observations; rounds are not treated as independent replicates.
- `*_ci95` uses the normal approximation `1.96 * s / sqrt(n)` and should be interpreted cautiously for three seeds.
- ASR is the fraction of test predictions equal to the fixed paired-label permutation; interpret it together with clean/no-attack controls.
- Detection metrics: AntiFLipper variants are scored on their cumulative exclusion set; per-round selectors (Multi-Krum, FLAME, LFighter, FLTrust, FoolsGold, Tolpegin) on clients given zero weight (`final_*` = last round, `mean_round_*` = averaged over rounds). FedAvg/Median/Trimmed Mean do not classify clients and have empty detection values.
- `mean_aggregation_seconds` is the aggregation call only; `mean_defense_seconds` adds server-side pre-aggregation work (trust update, consistency check). FLTrust root-model training is reported separately as `mean_root_training_seconds`. Run the fixed-workload benchmark for matched-N timing.
- `upload_bytes_per_client_round` is analytic: float32 model size, plus 4 bytes for the AntiFLipper accuracy scalar.

## Automated findings

- Insufficient completed experiment groups for automated comparisons.


## Files

- `per_run_metrics.csv`: one row per completed seed/configuration.
- `aggregate_mean_std_ci95.csv`: mean, standard deviation, and 95% CI half-width across seeds.
- `figures/`: automatically generated experiment plots when matplotlib is available.
