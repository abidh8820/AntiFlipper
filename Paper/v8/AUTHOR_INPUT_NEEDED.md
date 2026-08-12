# AntiFLipper: remaining author input before submission

I checked `AntiFlipper-main/AntiFlipper-main/` against the manuscript and saved result files. The items below marked resolved have been incorporated into `AntiFlipper_v7_revised.tex`; they do not require new experiments.

## Resolved from the repository

- [x] **Trust update:** the implemented coefficient is `eta = 0.1` in all checked-in AntiFLipper runs. Accuracy is represented on `[0,1]`.
- [x] **Trust initialization and normalization:** all clients start at `1.0`, then are normalized to `1/N`. Negative values are clipped to zero, detected clients are assigned zero, and each remaining value is divided by the sum over non-detected clients. There is no separate upper clip.
- [x] **Detection threshold:** the implementation uses the fixed absolute condition `trust < 0.0005`, beginning at zero-based epoch index 2 (communication round 3). It does not directly calculate `1/(kN)`. The fixed value is equivalent to `k=20` for `N=100` and `k=100` for `N=20`.
- [x] **Flag counter:** violations are cumulative and never reset. A client is classified when `bad_cnt > 5`, i.e., after six counted violations (`cnt_max = 6`).
- [x] **Local evaluation:** the pre-update global model is evaluated using the client's local training loader; there is no held-out local validation split. Samples used for evaluation can also be used for training. A malicious client's evaluation uses the same flipped labels as its training.
- [x] **Attack used in the checked-in runs:** every label on every malicious client is flipped with the fixed pairwise permutation `0<->1`, `2<->3`, `4<->5`, `6<->7`, and `8<->9`, in every selected round. Although source/target arguments exist, the current dataset wrapper does not use them for MNIST/CIFAR-10; an unconditional `or True` also makes the configured malicious-behavior rate ineffective.
- [x] **Models:** MNIST uses `CNNMNIST` (two convolutional and two fully connected layers; 21,840 trainable parameters). The checked-in CIFAR notebooks/results use ResNet18. No result-backed ShuffleNetV2 experiment was found.
- [x] **Participation and optimization:** MNIST uses 100 clients, all selected each round, for 200 rounds, batch size 64, three local epochs, SGD learning rate 0.01 and momentum 0.9. CIFAR-10 uses 20 clients, all selected each round, for 100 rounds, batch size 32, three local epochs, learning rate 0.01 and momentum 0.9. Non-IID runs use Dirichlet `alpha=1`.
- [x] **Randomness:** the checked-in experiment notebooks use seed 7 and set Python, NumPy, PyTorch, CUDA, and `PYTHONHASHSEED`. Client selection is without replacement; with participation fraction 1, it merely permutes all clients.
- [x] **Table I semantics:** the saved arrays reproduce every AntiFLipper table entry as the arithmetic mean over communication rounds from one run. They are not final-round or best-round results. Global accuracy and cross-entropy test loss are computed each round on the standard centralized 10,000-example test set.
- [x] **Aggregation timing:** CUDA events bracket only the aggregation branch and are synchronized with `torch.cuda.synchronize()`. Local training/evaluation, data transfer, and global testing are excluded. There is one measurement per round, no separate warm-up or timing repetitions, and the table is the arithmetic mean over rounds. After detection, AntiFLipper aggregates 60 models on MNIST and 12 on CIFAR-10, whereas FedAvg retains all clients.
- [x] **Static detection:** the detected sets exactly equal the configured malicious sets. MNIST: 40 TP, 0 FP, 0 FN, precision=recall=1.0; all detected by round 9 (IID) and round 11 (non-IID). CIFAR-10: 8 TP, 0 FP, 0 FN, precision=recall=1.0; all detected by round 15 (IID) and round 18 (non-IID).
- [x] **Figure 2 provenance:** the plotting notebook averages trust across clients within the detected-malicious group and remaining-client group, per round, for one run; it does not average multiple seeds.
- [x] **Missing figures:** `fig/sys_arc.pdf` and `fig/antiflipper_combined_subplots_trust_updated.pdf` were recovered from `AntiFlipper_Rejected.zip` and restored under `AntiFlipper/v8/fig/`.

## Answers still needed from the author

1. **Dynamic-attack results:** Where are the code/configurations and logs for the two dynamic schedules in Table II? I could not find files supporting their exact schedules, detection counts/rounds, false positives/negatives, or aggregation times. If those files no longer exist, confirm whether the table should remain with an explicit “accuracy only; other logs unavailable” limitation or be removed.
2. **10% local-evaluation experiment:** Where are the code and logs supporting the four accuracies currently reported for `gamma=0.1`? I could not verify the subset sampling rule, seed, detection results, or measured latency. If they are unavailable, I recommend removing those empirical accuracy claims and retaining only the analytical 1.1% overhead calculation.
3. **Architecture history:** Was ShuffleNetV2 actually used for any reported row? If yes, specify the row and provide its configuration/log. Otherwise its removal from the evaluated-model claim is correct.
4. **Authors:** Please confirm author order, affiliation indices, and each author's full email address. The grouped `.edu` suffix appears inconsistent with `sherajul@cse.uiu.ac.bd`.
5. **Submission format:** What is the target venue and page limit, and does it allow the custom `geometry` margins? The manuscript overrides standard IEEEtran margins.
6. **Final presentation:** Is this a marked revision for reviewers or a camera-ready manuscript? Before camera-ready submission, the blue revision coloring, red author notes, and response-only language should be removed.

## Important implementation caveats now documented

- A newly detected client has already supplied a model for that round; code-level exclusion takes effect in subsequent rounds.
- Trust weights used for aggregation are collected before that round's trust update, producing a one-round lag in the applied weights.
- The normalization function has no explicit guard for the edge case in which all remaining raw trust values sum to zero.
- All verified results are single-seed runs. No standard deviations, confidence intervals, malicious-ratio sweep, or hyperparameter sensitivity sweep are available.

## Highest-value optional experiment

If only one new experiment is feasible, repeat the headline configurations with at least three fixed seeds and report mean plus/minus standard deviation. Otherwise retain the explicit single-run limitation and avoid statistical-significance or “state-of-the-art” claims.
