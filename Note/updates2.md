# AntiFLipper — Experiment Plan v2 (2-month budget)
*Supersedes `Experiment_Plan_and_Methodology_Audit.md`. Consolidates: the original code audit, Codex's independent review (`updates.md`), and corrections made after discussion. Target: Q1 journal resubmission with real new experiments, ~2 months.*

---

## 0. Before anything: run the timing benchmark

Compute estimates below are **unverified guesses** based on typical RTX 3090 throughput for models this size. Run `benchmark_timing.py` (provided alongside this file) from inside `Code/` first:

```
python benchmark_timing.py
```

It runs a short and a slightly longer session per dataset and backs out a clean per-round time estimate (separating one-time setup cost from per-round cost). Send me the printed summary and I'll recompute the schedule in §6 against your real hardware instead of the placeholder numbers here.

---

## 1. Methodology / correctness fixes — do these first, before any new run

### 1.1 Make hardcoded quantities explicit configuration values
Currently hardcoded in `environment_federated.py`:

| Quantity | Current value | Fix |
|---|---|---|
| Trust update coefficient (η) | `0.1` in `adjustment = 0.1 * diff^2` | Pull out as a named `run_exp` argument |
| Exclusion threshold | fixed normalized trust `< 0.0005` | See §1.2 |
| Flag/counter rule | `bad_cnt > 5` (exclusion on the **6th** violation) | Pull out as named argument; note it's **cumulative, not consecutive** — it does not reset after a clean round |
| Grace period | no increments while `epoch <= 1` | Pull out as named argument; state explicitly in the paper |

Also remove the unused dead variable `alphaAntiFlipper = 0.09` (leftover from an earlier, unused linear trust-update formula) to avoid confusion. Save the complete configuration alongside every result file.

### 1.2 Fix the population-size dependency in the exclusion threshold
`normalize()` makes all non-excluded clients' trust values sum to 1, so a typical honest client's trust scales as `~1/N_eligible`. A **fixed absolute threshold** (`0.0005`) therefore means something very different for MNIST (N=100 → mean trust ~0.01) than for CIFAR-10 (N=20 → mean trust ~0.05) — the detector is effectively ~5x stricter on MNIST purely due to client count, not dataset difficulty.

**Fix:** switch to a population-relative threshold:
```
rho_i = N_eligible * tau_i
flag client i when rho_i < rho_thresh
```
Define `N_eligible` precisely (number of non-excluded clients whose reports are included that round). Also guard the edge case where all surviving trust values are zero (current `normalize()` would divide by zero).

Keep the old absolute threshold as an ablation/) so you can show *why* the change was necessary — this is a legitimate mini-result, not just a bug fix.

### 1.3 Separate three different attack-strength concepts
Currently:
```python
r = np.random.random()
if r <= malicious_behavior_rate or True:   # "or True" makes this always fire
```
Removing `or True` alone is not enough — `malicious_behavior_rate` as currently named conflates several independent ideas. Implement and name these separately:
- `malicious_client_fraction` — fraction of clients controlled by the adversary (this is what `attackers_ratio` already does — no change needed)
- `attack_round_probability` — probability a malicious client attacks in a given participating round
- `label_poison_fraction` — fraction of *that client's own examples* whose labels are permuted

Use a deterministic per-run RNG stream and log the realized number of poisoned client-rounds/examples so results are exactly reproducible.

### 1.4 Sanity-check identity/bookkeeping before trusting new results
Peers are appended after ID shuffling, and later code indexes into `self.peers` by sampled position. Add a quick assertion/test confirming these stay aligned across a run: sampled position ↔ `peer_id` ↔ local data partition ↔ trust-dict key ↔ attacker label ↔ reported accuracy ↔ submitted update. Also confirm the malicious-node ground-truth log is built from the full peer registry, not just from one `choose_peers()` call (which may not enumerate all attackers under partial participation — not currently relevant since you're keeping full participation, but cheap to fix while you're in this code anyway).

### 1.5 Resolve the CIFAR-10 client-count discrepancy
Paper text says "100 clients, 20 active per round" for CIFAR-10 IID. The actual notebook (`Experiments_CIFAR10.ipynb`) has `NUM_PEERS=20`, `FRAC_PEERS=1`. **Find whichever config actually produced your existing Table I numbers and make the paper match reality exactly.** Also double check `LOCAL_LR` — the MNIST notebook cell shows `0.01`, while the paper currently states `0.001`; verify which was actually used.

### 1.6 Manuscript-only fixes (no experiments needed)
- Proposition 2 still references a parameter `k` that the current method no longer uses — remove or redefine it.
- Algorithm 2 uses `<=` where the code uses `<` — make them match exactly.
- State explicitly whether client-side evaluation integrity relies on a trusted-execution/attestation assumption. As written, a client that controls its own software could fabricate the reported accuracy scalar.
- The commit-reveal/challenge-set discussion added in the last revision prevents *changing* a committed value after the fact — it does **not** prove the value was computed correctly. Reword this claim precisely, or drop it in favor of explicitly framing report integrity as a trusted-client assumption (see Experiment B2 below if you want to actually close this gap empirically).

---

## 2. Design decisions settled during planning (with the reasoning, so you don't re-litigate them later)

| Decision | Resolution | Why |
|---|---|---|
| **Full vs. partial client participation** | **Keep full participation (`frac_peers=1`)** | This is the norm in the cross-silo label-flipping-defense literature (FLTrust, FoolsGold, LFighter, Tolpegin, FLAME all use full or near-full participation at this client scale). Every baseline you're comparing against was designed and evaluated under that same convention — switching now would break the apples-to-apples comparison. Partial participation is a legitimate *optional* add-on (see Priority C), not a requirement. |
| **Number of seeds** | **3 seeds as the default across all new experiments.** Use 5 only for the single headline comparison table if compute allows once everything else is done. | 3 is a widely accepted minimum in published ML/FL work under real compute constraints ("mean ± std over 3 runs" is a standard, defensible phrase). 2 is too thin to defend if challenged; 5 is stronger but not required everywhere. |
| **Which methods get the full sweep treatment** | **Keep all 9 methods for the core headline table only.** For every other new sweep (ratio, heterogeneity, sensitivity, partial-flip, ablation), narrow to **4 methods: AntiFLipper, FedAvg (floor), LFighter (closest prior work), FLTrust (new baseline reviewers explicitly requested)**. | Re-running every sweep against all 9 methods is not standard practice and isn't what makes the paper credible — reviewers care about seeing the sweep exist and behave sensibly for the strongest comparators, not exhaustive coverage. State the scoping explicitly in the paper; this is normal, not a weakness. FoolsGold in particular already has a lopsided, clear story (near-zero accuracy under non-IID) and doesn't need multi-seed treatment to make its point. |
| **Bulyan as a baseline** | **Dropped, or added only at malicious ratios where it's theoretically valid.** | Bulyan's resilience condition requires roughly `n ≥ 4f+3`, which is **not satisfiable at your 40% malicious ratio** (your main operating point). It is also not simply Krum + your existing generic `trimmed_mean()` — its second stage does coordinate-wise trimmed selection around a median. Not worth the implementation cost given it can't even be run at your headline setting. |
| **FLTrust vs. HSCSFL as the added baseline** | **Prioritize FLTrust.** Add HSCSFL only if you find a reproducible implementation. | Both were explicitly named by reviewers. FLTrust is better-documented and more commonly reproduced in follow-up papers; HSCSFL's availability is uncertain. |

---

## 3. Prioritized experiment list

### Tier 1 — Do regardless of anything else (cheap, prerequisite)
All of §1.1–1.6 above. Don't run anything below until these are done — otherwise expensive runs may need to be redone.

### Tier 2 — Core (directly answers your actual GLOBECOM/ICC reviews)

| # | Experiment | Config | Seeds |
|---|---|---|---|
| A1 | Multi-seed replication of the headline table + fair aggregation timing | All 9 methods × 2 datasets × 2 distributions | 3 |
| A2 | Clean false-positive stress test under heterogeneity (**no attack**) | Dirichlet α ∈ {0.1, 0.3, 1.0} + true IID (don't call α=100 "IID" — it's only approximate) | 3 |
| A3 | Malicious-ratio sweep with honest failure-boundary framing | `p_mal ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}`, 4-method subset, 1 dataset (pick the more heterogeneous one, i.e. non-IID) | 3 |
| A5-lite | One-factor hyperparameter sensitivity (not a full grid) | η ∈ {0.05, 0.1, 0.2}, ρ_thresh ∈ {0.025, 0.05, 0.1}, cnt_max ∈ {3, 6, 9} — one factor varied at a time around the current default, tuned on **held-out validation seeds**, then confirmed on separate test seeds | 3 |
| A7 | Add FLTrust as a baseline, rerun core comparisons with it included | 4-method subset | 3 |

**A1's timing benchmark must separate two effects that are currently conflated:** AntiFLipper appears faster partly because excluded clients shrink the aggregation workload over time — that's a real operational benefit, but it's not an apples-to-apples per-update aggregation-complexity comparison. Report both (a) a fixed-workload microbenchmark at matched N, and (b) actual end-to-end cost including the shrinking-workload effect.

### Tier 3 — Add if time remains after Tier 2 (good ROI, reuses infrastructure)

| # | Experiment | Config |
|---|---|---|
| A4-lite | Partial-flip sweep only (skip the full adaptive-attack taxonomy) | `label_poison_fraction ∈ {0.1, 0.25, 0.5, 0.75, 1.0}`, 4-method subset |
| A6-lite | 2–3 key ablations, not the full list | Suggest: (i) quadratic vs. linear trust-deviation update, (ii) absolute vs. relative threshold (this doubles as validating §1.2's fix), (iii) exclusion with vs. without the cumulative counter |
| B5 | Empirical convergence plots | Reuses existing per-round logs — per-seed accuracy/loss trajectories, oscillation after exclusions, number of surviving clients over time. Cheap, no new runs needed beyond what Tier 2 already produces |

### Tier 4 — Cut for this round (real value, but out of scope for 2 months)
- B1: participation/dropout sweep (contradicts the "keep full participation" decision above; revisit only if a future reviewer specifically asks)
- B2: fabricated-accuracy-report attack + mitigation (substantial new sub-project — attack design, mitigation design, and evaluation)
- B3: differential-privacy noise/utility tradeoff for the reported scalar
- B4: a third, IoT-relevant dataset/domain
- C1: on-device (e.g., Raspberry Pi) profiling
- C2: broader attack types beyond label-flipping (explicitly out of scope per your own threat model)

If your Q1 reviewers come back asking specifically for any of these, revisit — don't preemptively build them now.

---

## 4. Metrics to report (beyond raw accuracy)

Your current headline metric (accuracy averaged over *every* communication round) is confounded by convergence speed and shouldn't be the only thing reported. For Tier 2/3 experiments, report:

**Utility:** final-round accuracy, mean of last-10-rounds accuracy (keep as a secondary convergence-speed indicator, not the headline)
**Attack effectiveness:** attack success rate (ASR) / source-class accuracy for the flipped classes, accuracy degradation relative to a matched no-attack run
**Detection quality:** precision, recall, F1, false-positive rate, detection delay (rounds until exclusion), fraction of *honest* clients incorrectly excluded
**Systems cost:** fixed-workload aggregation latency (matched N) reported separately from end-to-end cost including the exclusion effect

Every heterogeneity/sensitivity setting needs a matched **no-attack control run** — this is what actually demonstrates the detector isn't just filtering "naturally different" honest clients (directly answers Reviewer 3's core concern).

---

## 5. What changes in the paper once this is done

- Replace single-run Table I/II entries with mean ± std (3 seeds; 5 for the headline table if compute allows).
- Add detection precision/recall/F1/FPR alongside accuracy for every attack experiment.
- State the empirical malicious-ratio failure boundary rather than implying robustness at arbitrary ratios.
- Report the relative-threshold fix and its ablation against the old absolute threshold.
- Correct the CIFAR-10 client-count and learning-rate discrepancies (§1.5).
- Fix Proposition 2 / Algorithm 2 inconsistencies (§1.6).
- Reword or remove the commit-reveal correctness claim (§1.6); state report integrity as an explicit trusted-client assumption.
- Update Related Work: FLTrust moves from "not empirically compared" to an actual compared baseline; Bulyan/CRFL stay as "discussed but not compared, and not valid at our operating ratio" — now with the precise reason why.
- New figures: accuracy/detection-recall vs. malicious ratio; false-positive rate vs. heterogeneity; detection recall vs. partial-flip fraction; relative- vs. absolute-threshold ablation.

---

## 6. Suggested 8-week schedule (placeholder — will refine once benchmark numbers are in)

| Week | Work |
|---|---|
| 1 | Run `benchmark_timing.py`; do all of Tier 1 (§1.1–1.6) |
| 2 | Validation-stage sensitivity tuning (A5-lite) on held-out seeds; resolve CIFAR-10 config question |
| 3–4 | Tier 2 core: A1 (multi-seed headline table), A2 (clean heterogeneity stress test) |
| 5 | A3 (malicious-ratio sweep), start A7 (implement FLTrust) |
| 6 | Finish A7, run with FLTrust included |
| 7 | Tier 3 if time allows: A4-lite (partial-flip), A6-lite (key ablations), B5 (convergence plots) |
| 8 | Regenerate figures/tables, rewrite evaluation section, finalize for submission |

This schedule assumes your real per-round timing (from the benchmark) keeps the Tier 2 compute load under roughly 10-14 days of GPU time. **Send me the benchmark output and I'll tell you precisely whether this fits, or what to trim further.**