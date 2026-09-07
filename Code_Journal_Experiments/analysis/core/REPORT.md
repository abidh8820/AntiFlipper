# AntiFLipper automated analysis

Completed runs analyzed: **0**.

## Included experiment groups

- None yet

## Interpretation notes

- Seed-level runs are the independent observations; rounds are not treated as independent replicates.
- `*_ci95` uses the normal approximation `1.96 * s / sqrt(n)` and should be interpreted cautiously for three seeds.
- ASR is the fraction of test predictions equal to the fixed paired-label permutation; interpret it together with clean/no-attack controls.
- AntiFLipper detection metrics are meaningful for AntiFLipper. Other aggregators do not explicitly classify clients and therefore retain default empty-detection values.
- `mean_aggregation_seconds` is observed aggregation time during training. Run the fixed-workload benchmark separately for matched-workload timing.

## Automated findings

- Insufficient completed experiment groups for automated comparisons.


## Files

- `per_run_metrics.csv`: one row per completed seed/configuration.
- `aggregate_mean_std_ci95.csv`: mean, standard deviation, and 95% CI half-width across seeds.
- `figures/`: automatically generated experiment plots when matplotlib is available.
