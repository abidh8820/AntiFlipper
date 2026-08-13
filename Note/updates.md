# AntiFLipper — Q1 Journal Experiment Plan and Methodology Audit

This plan is based on the GLOBECOM reviews, `Paper/v8/AntiFlipper_v7_revised.tex`, and the current implementation in `Code/`. The revised manuscript already addresses several presentation-level comments: it states the actual client counts and trust parameters, explains the grace period and communication overhead, reports detection results for the existing static runs, and explicitly limits the threat model. Those changes are useful, but they do not replace the experiments below.

## Overall assessment

I agree with the main direction of the earlier plan: multi-seed results, malicious-ratio and non-IID sweeps, evasive attacks, sensitivity analysis, and stronger baselines are necessary for a credible journal submission.

I would change the prioritization and experimental definitions. The most important missing evidence is not another large table of average accuracy values. It is evidence that AntiFLipper (1) distinguishes attackers from naturally difficult honest clients, (2) has a measurable detection boundary, (3) remains useful when clients can behave adaptively or participate intermittently, and (4) is compared fairly using attack-specific, detection, utility, and cost metrics.

---

## 1. Correctness and reproducibility issues to resolve first

Do not launch a large experiment campaign until these items are fixed and covered by small tests. Otherwise, expensive runs may not be reproducible or may test a different attack than their labels claim.

### 1.1 Make every experimental quantity an explicit configuration value

The implementation hardcodes the trust update coefficient (`0.1`), trust threshold (`0.0005`), flag rule (`bad_cnt > 5`), and grace period (`epoch > 1`) inside `environment_federated.py`. Move them into named configuration arguments and save the complete configuration with every result.

Ground truth in the current implementation:

| Quantity | Current behavior |
|---|---|
| Trust update coefficient, `eta` | `0.1` in `adjustment = 0.1 * diff^2` |
| Exclusion threshold | fixed normalized trust `< 0.0005` |
| Counter threshold | exclusion on the sixth accumulated violation (`bad_cnt > 5`) |
| Counter semantics | cumulative; it does not reset after a clean round |
| Grace period | no counter increments in rounds 0–1 |

There is also an unused variable named `alphaAntiFlipper = 0.09`. Remove it to prevent later confusion. The revised manuscript's `eta=0.1` is consistent with the expression that actually updates trust.

### 1.2 Replace the absolute trust threshold with a population-relative definition

`normalize()` makes the non-excluded clients' trust values sum to one, so a typical trust value scales as `1/N_active`. A fixed threshold of `0.0005` therefore means different things for 100-client MNIST and 20-client CIFAR-10.

Use a dimensionless threshold such as

```text
rho_i = N_eligible * tau_i
flag client i when rho_i < rho_thresh
```

where `N_eligible` is defined precisely (preferably the number of non-excluded clients whose reports are included in that round). Tune `rho_thresh` only on a validation protocol, not on test results. Retain the old absolute-threshold method as an ablation so the paper can demonstrate why the change is justified.

Also handle the edge case in which all surviving unnormalized trust values are zero; the current normalizer would divide by zero.

### 1.3 Separate three different attack-strength concepts

The current line

```python
if r <= malicious_behavior_rate or True:
```

makes every malicious client attack in every round. After removing `or True`, `malicious_behavior_rate` will be the **probability of a full label-permutation attack in a client-round**. It is not a partial-label-flip rate.

Implement and name these independently:

- `malicious_client_fraction`: fraction of clients controlled by the adversary.
- `attack_round_probability`: probability that a malicious client attacks in a participating round.
- `label_poison_fraction`: fraction of that client's examples whose labels are permuted.

Use a deterministic per-run random generator and record the realized number of poisoned client-rounds and examples. The earlier proposed “partial-flip sweep” cannot be produced by changing `malicious_behavior_rate` alone.

### 1.4 Verify client identity, attacker assignment, and participation bookkeeping

The peer objects are appended after shuffling IDs, while later code indexes `self.peers` using sampled integer positions. Before new runs, add assertions/tests showing that the following remain aligned: sampled position, `peer_id`, local partition, trust dictionary key, attacker label, reported accuracy, and submitted update.

The initial malicious-node log is constructed from one call to `choose_peers()`. Under partial participation this may not enumerate all attackers. Store ground truth directly from the complete peer registry, and compute detection metrics over eligible/participating clients with clearly defined denominators.

### 1.5 Create a reproducible experiment runner

Replace notebook-state-driven runs with a script or configuration-based runner that:

- records seed, dataset version, partition, client IDs, attacker IDs, attack schedule, hyperparameters, model, and code revision;
- writes to a unique run directory rather than appending to an existing file;
- supports resume without losing trust scores, counters, attacker schedule, or RNG state;
- saves per-round global metrics, per-client reports/trust, detected IDs, wall-clock components, and failures;
- does not average incompatible runs or silently use an old checkpoint.

This is a prerequisite for defensible multi-seed statistics and an artifact reviewers can reproduce.

### 1.6 Correct small manuscript inconsistencies before submission

- Proposition 2 still refers to a parameter `k`, although the revised method now documents a fixed threshold. Remove or redefine `k`.
- Algorithm 2 uses `<=` while the implementation uses `<`; make them identical.
- The abstract says clients evaluate and report accuracy under an attacker that cannot tamper with evaluation. State whether this relies on trusted execution/attestation. A client that controls its software can otherwise fabricate the scalar.
- The proposed challenge-set/commit-reveal discussion does not by itself prove that a reported local accuracy is truthful. Commit-reveal only prevents changing a committed value; it does not establish correct computation. Any claimed mitigation needs a precise protocol and threat assumption.

---

## 2. Metrics and statistical protocol for all new experiments

### 2.1 Repeat runs and report uncertainty

Use at least five independent seeds for the principal AntiFLipper and strongest-baseline comparisons; more seeds are preferable if variance is high. Seeds must change model initialization, data partitioning, attacker assignment, participation, minibatch order, and attack randomness through controlled RNG streams.

Report mean, standard deviation, and a 95% confidence interval across independent runs. For direct method comparisons, use matched seeds/partitions and report the paired difference with its confidence interval. Do not treat communication rounds from one run as independent samples.

### 2.2 Report four metric families

**Model utility**

- final-round and mean-of-last-10-rounds global accuracy;
- convergence curve and area under the accuracy-versus-round curve;
- worst-class and macro class accuracy, especially under skew;
- time/rounds to reach a fixed accuracy, where applicable.

The current table's mean accuracy over every communication round strongly depends on convergence speed and should not be the only headline metric.

**Attack effectiveness**

- attack success rate (ASR) and source-class accuracy for targeted flips;
- global accuracy degradation relative to a matched no-attack run;
- realized poisoned-example and poisoned-client-round counts.

Overall accuracy can conceal successful targeted corruption.

**Detection quality**

- precision, recall, F1, false-positive rate, and false-negative rate;
- detection delay (rounds or malicious participations until exclusion);
- fraction of honest training data/clients incorrectly excluded;
- metrics over time, not only the final detected set.

**Systems cost**

- server aggregation latency at matched numbers of submitted updates;
- client evaluation latency, end-to-end round latency, bytes sent/received, and peak memory;
- throughput/scaling versus client count and model size.

### 2.3 Include matched clean controls

Every heterogeneity, participation, and hyperparameter setting needs a no-attack AntiFLipper run. This is essential for measuring whether naturally low-accuracy honest clients are filtered. A defense that preserves global accuracy while excluding a minority distribution is not necessarily behaving correctly.

Pre-register a primary configuration and a small number of primary outcomes. Use a held-out validation setup for threshold selection so the test configurations are not repeatedly tuned until they look favorable.

---

## 3. Prioritized experiment program

### Priority A — required for a journal submission

#### Experiment A1: Multi-seed replication and fair baseline timing

Rerun the current IID and non-IID configurations with matched seeds for AntiFLipper, FedAvg, and a focused set of relevant strong baselines. Report the metrics in Section 2.

For aggregation timing, benchmark all methods with the same hardware, warm-up, synchronization, update tensors, precision, and number of submitted updates. Report both:

1. a fixed-workload microbenchmark at `N = 20, 50, 100` updates, and
2. actual end-to-end round cost.

AntiFLipper currently becomes faster partly because detected clients are removed and it aggregates fewer updates. That is a valid operational effect, but it is not an apples-to-apples aggregation-complexity comparison. Separate the two effects. Include local evaluation in end-to-end cost even if the server-only cost remains a separate metric.

#### Experiment A2: Clean false-positive stress test under heterogeneity

Run **no attack** with Dirichlet concentration values such as `alpha_D in {0.05, 0.1, 0.3, 1.0}` plus a truly IID partition. Do not call `alpha_D=100` IID; it is only approximately IID.

Measure false-positive rate, worst-client/per-class accuracy, excluded-data fraction, and utility. Include at least one quantity-skew setting and, if possible, a pathological label-skew setting (for example, 1–2 classes per client). This directly tests Reviewer 3's central failure concern.

#### Experiment A3: Malicious-client fraction and failure boundary

Sweep `p_mal in {0, 0.1, 0.2, 0.3, 0.4, 0.5}` under IID and at least one severe non-IID setting. Plot utility, ASR, detection precision/recall, detection delay, and the empirical honest-versus-malicious accuracy gap.

Do not merely state that the gap shrinks with `p_mal`; show where detection and utility fail. At 50%, describe the symmetry/majority limitation rather than implying a guarantee. Some robust baselines have Byzantine-fraction validity constraints, so mark invalid cells as “outside assumptions” instead of running them as if theoretically supported.

#### Experiment A4: Adaptive label-flipping attacks

Evaluate these separately:

- **partial-example flipping:** `label_poison_fraction in {0.1, 0.25, 0.5, 0.75, 1.0}`;
- **intermittent behavior:** `attack_round_probability in {0.25, 0.5, 0.75, 1.0}`;
- **delayed/on-off attacks:** contiguous benign and malicious phases;
- **threshold-aware attack:** choose the greatest poisoning rate that keeps the reported/true accuracy deviation near the flag boundary;
- **heterogeneous collusion:** malicious clients use different mappings or schedules rather than an identical permutation.

Plot attack impact against detectability. A low-recall result is not necessarily a defense failure if the attack also causes negligible harm, so report ASR/utility and recall together.

#### Experiment A5: Hyperparameter sensitivity and threshold scaling

Study `eta`, relative trust threshold `rho_thresh`, `cnt_max`, and grace period. A full Cartesian grid across every dataset is wasteful. First use one-factor sweeps around a registered default on a validation configuration, then evaluate a small set of selected combinations on held-out seeds and distributions.

Suggested initial values:

- `eta in {0.025, 0.05, 0.1, 0.2}`;
- `rho_thresh in {0.01, 0.025, 0.05, 0.1}`, where `rho=N_eligible*tau`;
- `cnt_max in {3, 6, 9, 12}`;
- grace period in `{0, 2, 5, 10}` rounds.

Compare the relative threshold against the original absolute `0.0005` threshold across `N in {20, 50, 100, 200}`. The goal is a stable operating region, not a single best point.

#### Experiment A6: Ablation of AntiFLipper's components

Compare:

- uniform FedAvg;
- accuracy-weighting without hard exclusion;
- linear/absolute versus quadratic deviation updates;
- exclusion without the cumulative counter;
- cumulative counter versus consecutive counter with decay/reset;
- absolute versus population-relative threshold;
- full local evaluation versus subsets such as `{1%, 5%, 10%, 25%, 100%}`.

Report detection and cost as well as accuracy. The existing 10% subset result is useful but one single run/configuration does not establish the trade-off.

#### Experiment A7: Strong, correctly scoped baselines

Prioritize a direct label-flipping/client-selection baseline (HSCSFL if reproducible) and FLTrust because the reviewers explicitly named them. FLTrust requires a clean server root dataset; that is a different deployment assumption, but the comparison remains informative if root-set size and composition are disclosed.

Also retain representative update-based robust aggregators already implemented (Median, Trimmed Mean, Multi-Krum, FLAME) rather than adding many weakly relevant methods.

**Correction regarding Bulyan:** Bulyan is not simply the existing Multi-Krum selection followed by the repository's generic `trimmed_mean()`. Its second stage performs coordinate-wise selection around a median, and its standard resilience condition requires roughly `n >= 4f + 3`. It is therefore not valid for `f/n = 0.4` (and not for many of the proposed ratio cells). Add Bulyan only in configurations satisfying its assumptions and only with a verified implementation. It is lower priority than FLTrust/HSCSFL for this paper.

### Priority B — strongly recommended

#### Experiment B1: Partial participation, dropout, and stragglers

Sweep participation fractions such as `{0.1, 0.25, 0.5, 1.0}` with random and biased participation. Add transient dropout and delayed-client scenarios. Define whether trust decay/counters advance only when a client participates. Report detection delay in both wall-clock rounds and number of malicious participations.

This experiment is especially important because the current evidence uses full participation, while real cross-device FL does not.

#### Experiment B2: Accuracy-report integrity / fabricated scores

This is the method's most consequential systems assumption. Test attackers that submit:

- the round mean or median accuracy;
- an honest-looking sampled score;
- a clipped score just above the inferred threshold;
- coordinated scores from colluding attackers.

The current method should be expected to fail if arbitrary reports are accepted. The scientific contribution would be to quantify this failure and evaluate one explicit mitigation, such as a trusted execution/remote-attestation assumption or a server-verifiable challenge protocol. Account for the mitigation's computation, communication, privacy, and reference-data assumptions. Do not claim that commit-reveal alone verifies correctness.

If no mitigation is implemented, frame report integrity as an explicit trusted-client assumption and avoid general deployment-security claims.

#### Experiment B3: Privacy–utility trade-off for reported accuracy

Add calibrated noise or randomized response to the scalar report and sweep a clearly defined privacy/noise budget. Measure attack detection, honest false positives, global utility, and detection delay. The privacy unit and neighboring-dataset definition must be stated; merely adding arbitrary Gaussian noise is not a differential-privacy evaluation.

This experiment turns the manuscript's current qualitative privacy discussion into evidence and directly answers Reviewer 1.

#### Experiment B4: A genuinely relevant third dataset/domain

Add one dataset that supports the IoT/edge motivation, rather than adding a third image benchmark only for count. A human-activity-recognition sensor dataset with a lightweight 1D CNN/MLP is a natural choice; an industrial tabular dataset is another option. Partition by real user/device where possible, because naturally heterogeneous clients are more convincing than only synthetic Dirichlet splits.

Use the same statistical and detection protocol as above. If this cannot be done well, narrow the paper's IoT claims instead of adding a superficial dataset.

#### Experiment B5: Empirical convergence and population bias

Plot per-seed accuracy/loss trajectories, time to stable performance, oscillation after exclusions, number of surviving clients, and divergence/failure rate. In clean heterogeneous runs, compare performance on the original client population with performance on the surviving population. This does not replace a formal convergence theorem, but it directly exposes instability or objective shift due to false exclusions.

### Priority C — optional or claim-dependent

#### Experiment C1: On-device profiling

If the final manuscript continues to claim suitability for resource-constrained IoT devices, measure local evaluation latency, peak memory, energy, and communication on at least one representative constrained platform. Otherwise, present the analytical overhead and workstation measurements as motivation only and narrow the deployment claim.

#### Experiment C2: Broader attacks

AntiFLipper is a label-flipping-specific defense. Model poisoning and backdoors are not required if the title, abstract, claims, and threat model remain narrow. A small out-of-scope stress test may be informative, but it should not displace the required label-flipping experiments above. Do not market AntiFLipper as a general Byzantine defense without such evidence.

---

## 4. Recommended minimal experiment matrix

The full cross-product is too expensive. Use a staged design:

1. **Debug/smoke stage:** one seed, MNIST, short runs, with assertions and metric validation.
2. **Selection stage:** tune threshold/counter only on held-out validation seeds using MNIST plus one CIFAR-10 configuration.
3. **Primary stage:** at least five matched seeds for AntiFLipper and selected baselines on MNIST, CIFAR-10, and the chosen third dataset; IID plus severe non-IID; clean plus full attack.
4. **Boundary stage:** AntiFLipper, FedAvg, and the two strongest relevant baselines for malicious-ratio, heterogeneity, and adaptive-attack sweeps.
5. **Systems stage:** fixed-workload latency scaling and end-to-end cost, using fewer repetitions but many timing iterations after warm-up.

The primary paper should emphasize a small number of readable plots:

- attack harm and detection recall versus malicious-client fraction;
- false-positive rate and utility versus heterogeneity;
- ASR versus detection recall for partial/intermittent attacks;
- sensitivity/scalability of relative versus absolute trust thresholds;
- utility/detection/cost Pareto plot for local evaluation fraction or privacy noise.

Put the larger numerical matrix, per-seed values, and secondary baselines in an appendix or artifact.

---

## 5. What should change in the paper after the experiments

- Replace single-run table entries with seed-level summaries and confidence intervals.
- Make final/last-window accuracy and ASR primary; retain mean-over-round accuracy only as a convergence-speed/AUC-style secondary metric.
- Report detection precision, recall, false-positive rate, and delay for every AntiFLipper attack experiment.
- State the empirical failure boundary instead of implying robustness at arbitrary malicious fractions.
- Explain and validate the population-relative threshold, or explicitly retain the absolute threshold as a limitation.
- Distinguish partial-example, intermittent, delayed, targeted, and full-permutation attacks precisely.
- Separate fixed-workload server aggregation cost from end-to-end operational savings caused by client exclusion.
- Describe report integrity as an assumption unless a verified mitigation is implemented and evaluated.
- Update Related Work and the result table only for baselines actually run under compatible assumptions.
- Narrow IoT/privacy/general-security claims when the corresponding experiments are not completed.

---

## 6. Practical execution order

1. Refactor configuration, logging, identity bookkeeping, and attack controls; add unit/smoke tests.
2. Run clean heterogeneity tests and threshold-scaling/sensitivity tests. These may reveal a design problem that should be fixed before all other runs.
3. Freeze the revised method and evaluation protocol.
4. Run multi-seed primary configurations and fair timing benchmarks.
5. Run malicious-ratio and adaptive-attack boundary sweeps.
6. Add FLTrust/HSCSFL (subject to reproducible implementations and disclosed assumptions) and rerun the primary comparisons.
7. Run participation/dropout, fabricated-report, and privacy-noise studies.
8. Add one meaningful IoT/edge-domain dataset and on-device profiling if those claims remain central.
9. Rewrite the paper only after results are frozen; release configurations, per-seed outputs, and analysis scripts with the artifact.

The minimum convincing journal package is Steps 1–6 plus detection metrics and clean controls. Steps 7–8 would materially strengthen a Q1 submission, especially if the target journal emphasizes security, privacy, or edge/IoT systems.
