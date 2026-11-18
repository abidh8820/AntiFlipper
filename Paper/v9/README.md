# AntiFLipper v9

`AntiFlipper_v9.tex` is the reframed paper (7 pages, IEEE conference format, standard IEEEtran margins).
Every results table and figure is generated from the experiment pipeline. No result number is typed by hand.

## Refresh results

```bash
git pull                                  # bring results/ and analysis/ back from the GPU machine
python Paper/v9/make_results.py           # writes Paper/v9/generated/*.tex and fig/gen_*.pdf
cd Paper/v9 && latexmk -pdf AntiFlipper_v9.tex
```

Cells show `--` until their runs complete; figures show "results pending".
Red `\pending{...}` notes mark sentences that interpret results. Rewrite each one once the numbers exist,
then make `\pending` empty (`\renewcommand{\pending}[1]{}`) for a clean copy.

## What changed from v8 (reviewer comment → change)

| Reviewer comment (GLOBECOM 2026) | v9 |
|---|---|
| R1/R3 novelty: accuracy-deviation detection is related to prior validation-score work | New contribution: the report-faking attacker (Prop. 2) and AntiFLipper-CC, plus a positioning paragraph against FLTrust, HSCS and BaFFLe |
| R2-1 a client can fake its accuracy; R3 adaptive attackers | Formal attacker in §II; Table IV (honest vs. faked reports); consistency check M3 |
| R1-1 failure threshold vs. malicious ratio | Ratio sweep 10–40% (Fig. 2); Prop. 1 gives the (1−p)g gap; the threat model caps p at 40% |
| R1-2/R3-2 compare with FLTrust; report detection P/R | FLTrust baseline (with root-data caveat); Table III P/R/FPR for every method that drops clients |
| R1-3/R2-3 single runs, no std | 3 seeds (7, 19, 31), mean±std everywhere; sensitivity on held-out seeds |
| R1-3/R2-5 hyperparameter values and sensitivity | All values in §VI-A and Table I; Table V varies ρ, c_max and z |
| R2-4 only Dirichlet α=1 | Clean FPR at α_D ∈ {0.1, 0.5, 1}; attacked accuracy at α_D = 0.1 (Fig. 3) |
| R3 honest clients with severe skew may be penalized | Same clean-control experiment; Prop. 3 states the condition |
| R2-2 communication overhead | §V Cost (analytic) and Table VI (measured); every run logs upload/download bytes |
| R2-6 why faster than FedAvg | Timing now includes the trust update and cross-check; text separates exclusion effects from method cost; a fixed-workload benchmark is available |
| R2-7 inconsistent client counts | Same counts for IID and non-IID (CIFAR 20, MNIST 100), stated explicitly |
| R2-8 convergence | §V Convergence paragraph; per-round accuracy and surviving clients are logged |
| R2-10 α used with three meanings | Notation table: η trust rate, α_D Dirichlet, γ eval fraction |
| R2-11/R3-3 outline skipped Section VII | Outline built from `\ref`, so it always matches |
| R2-9 few recent references | Added BaFFLe; recent label-flipping work kept in §III. More can be added for a journal version |
| R1-5/R3-4 privacy of repeated accuracy reports | §V Privacy; M3 does not use reports, so noisy reports leave it unaffected |
| R1-4/R1-7 other domains, edge hardware | Stated as limitations (§VII); out of scope for this version |
| R1-2 general model poisoning, backdoors | Explicitly out of scope in the threat model |

Removed from v8 because no logs back them: the dynamic-attack table and the γ=0.1 accuracy numbers
(see `../v8/AUTHOR_INPUT_NEEDED.md`). The γ/(3E) overhead stays as analysis, and Table VI will measure it.
