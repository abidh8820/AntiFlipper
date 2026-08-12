# AntiFLipper — Experiment Plan & Methodology Audit
*For Q1 journal resubmission. Based on a direct audit of `github.com/abidh8820/AntiFlipper` (Code/environment_federated.py, aggregation.py, utils.py, sampling.py, and the MNIST/CIFAR-10 notebooks).*

---

## 1. Methodology issues to resolve BEFORE running new experiments

These aren't new experiments — they're correctness/consistency issues found in the current code that a Q1 reviewer is likely to catch if left unaddressed.

### 1.1 CIFAR-10 client-count mismatch (⚠️ must resolve first)
- The paper states CIFAR-10 IID used "100 clients, 20 active per round."
- The actual notebook (`Experiments_CIFAR10.ipynb`) has `NUM_PEERS = 20`, `FRAC_PEERS = 1` (i.e., 20 total clients, all active) hardcoded for the run labeled IID.
- **Action:** find whichever run/checkpoint actually produced Table I's CIFAR-10 numbers and confirm the true `NUM_PEERS`/`FRAC_PEERS` used. Update the paper's Implementation section to match reality exactly — don't leave this as a guess.

### 1.2 The trust-exclusion threshold (τ_thresh) is implicitly dependent on the number of peers
- `normalize()` in `utils.py` divides every peer's trust value by the sum of all trust values each round, so trust weights always sum to 1 across the population.
- That means the *average* honest peer's trust weight is roughly `1/N_honest`, which shrinks as `N` grows.
- Your code uses a **fixed absolute threshold** (`τ_thresh = 0.0005`) to decide when a peer gets flagged, regardless of `N`.
- Consequence: for `N=100` (MNIST), `0.0005` is about 5% of the average honest weight (`~0.01`); for `N=20` (CIFAR-10), it's about 1% of the average honest weight (`~0.05`). **The detector is effectively stricter on MNIST than on CIFAR-10, purely because of the different client counts — not because of anything intrinsic to the datasets.**
- **Action, pick one:**
  - (a) Keep `N` fixed across all datasets/experiments so this confound doesn't apply, or
  - (b) Redefine the threshold relative to `N` (e.g., `τ_thresh = c / N` for a tunable constant `c`), and re-tune, or
  - (c) At minimum, explicitly discuss this scale-dependency in the paper as a known limitation of the current threshold design.
  - Given you're already doing a hyperparameter sensitivity sweep (Experiment 4 below), this is a natural time to test (b) empirically and pick whichever is more defensible.

### 1.3 The `malicious_behavior_rate` parameter currently does nothing
```python
r = np.random.random()
if r <= malicious_behavior_rate or True:   # "or True" always fires
```
- Every attacker performs a **full label flip, every round, unconditionally**, regardless of the configured rate.
- This means every result you've reported so far is implicitly "100% full-flip attacker" — that's fine and doesn't need to be redone, just make sure the paper doesn't imply you tested weaker attackers when you haven't.
- **Action:** delete `or True` before running Experiment 2 (partial-flip sweep) below. Don't touch it if you're not running that experiment.

### 1.4 Grace period before flagging starts
- Flagging only begins after round 2 (`epoch > 1`). This is a real design choice (avoids penalizing clients before trust scores stabilize) but isn't currently mentioned in the paper.
- **Action:** state this explicitly in the methodology text — it's a one-sentence fix, not new experimentation.

---

## 2. Ground-truth hyperparameters (replace the paper's placeholders with these)

| Symbol (paper) | Code location | Actual value |
|---|---|---|
| η (trust learning rate) | `adjustment = 0.1 * (diff * diff)` | **0.1** |
| τ_thresh (exclusion threshold) | `if trust_w[peerID] < 0.0005` | **0.0005** (absolute constant — see §1.2 caveat) |
| cnt_max (flag count before exclusion) | `if bad_cnt[peerID] > 5` | **6** (i.e., the 6th consecutive/cumulative violation triggers exclusion) |
| Grace period | `and epoch > 1` | Flagging inactive for rounds 0–1 |

Correct the paper's Implementation section with these exact values instead of the `[VALUE]` placeholders, and fix the τ_thresh formula (it is **not** `1/(kN)` as speculated in the last revision — that was a guess made before the code audit).

---

## 3. Experiment list (prioritized, cheapest/highest-value first)

| # | Experiment | What it answers | Code change required | Relative effort |
|---|---|---|---|---|
| 1 | **Multi-seed reporting** — rerun each existing config (dataset × IID/non-IID × defense) across 5 seeds, report mean ± std | Statistical significance (flagged by every reviewer) | None — `seed` is already a clean parameter, and seeding correctly precedes data partitioning and attacker assignment | Low (compute-bound only) |
| 2 | **Malicious ratio sweep**: attackers_ratio ∈ {10%, 20%, 30%, 40%, 50%} | Validates (or refutes) the analytical malicious-ratio argument added in the last paper revision | None — `attackers_ratio` already a parameter | Low |
| 3 | **Non-IID severity sweep**: Dirichlet α ∈ {0.1, 0.5, 1, 5, 100≈IID} | Tests the false-positive risk under extreme heterogeneity (flagged by reviewers) | None — `alpha` already a parameter | Low |
| 4 | **Partial-flip / stealthy-attacker sweep**: malicious_behavior_rate ∈ {25%, 50%, 75%, 100%} | Operationalizes the "adaptive/evasive attacker" discussion added in the last revision | Fix the `or True` bug first (§1.3) | Low-Medium |
| 5 | **Hyperparameter sensitivity**: η ∈ {0.05, 0.1, 0.2}, τ_thresh ∈ {0.0002, 0.0005, 0.001} (or the relative version from §1.2), cnt_max ∈ {3, 6, 9} | Answers the most-repeated "no sensitivity analysis" comment; also resolves §1.2 | Pull `0.1`, `0.0005`, `5` out of `environment_federated.py` as named `run_exp` arguments | Medium (small refactor + compute) |
| 6 | **Resolve CIFAR-10 client-count mismatch** (§1.1) | Fixes a correctness/consistency issue, not a new capability | Investigation of old run logs/checkpoints; possibly a re-run with corrected `NUM_PEERS` | Medium (mostly investigation) |
| 7 | **Add Bulyan as a baseline** | Answers reviewer requests for a general-purpose Byzantine-robust baseline | You already have `Krum(multi=True)` and `trimmed_mean()` in `aggregation.py` — Bulyan is Krum-based selection followed by trimmed-mean aggregation of the selected subset. Mostly composition of existing functions | Medium |
| 8 | *(Optional, only if time allows)* **Add FLTrust as a baseline** | Directly answers "compare to FLTrust" requests from two reviewers | Needs a small server-held root dataset + cosine-similarity trust scoring; you already have cosine-similarity utilities (used by FoolsGold) to build from | Medium-High |
| 9 | *(Optional, stretch)* On-device timing on constrained hardware (e.g., Raspberry Pi) | Answers the resource-constrained deployment claim | No code change to the FL logic; needs physical/emulated hardware access | High (logistics-bound, not code-bound) |

**Suggested order:** 1 → 3 → 2 → 6 → 4 → 5 → 7 → (8, 9 only if time allows).

Items 1–3 alone require **no new code**, only more compute time, and directly answer the single most-repeated complaint across all three of your original reviews (no statistical rigor, no ratio/heterogeneity sweep). That's the highest-leverage place to start.

---

## 4. What changes in the paper once these are done

- Replace single-run numbers in Table I/II with **mean ± std** across 5 seeds.
- Add 2–3 new figures: accuracy/detection-recall vs. malicious ratio; accuracy vs. Dirichlet α; detection recall vs. partial-flip fraction.
- Replace the hyperparameter placeholders with the real values from §2, and correct the τ_thresh formula.
- Update the Implementation section to state the actual CIFAR-10 client configuration once §1.1 is resolved.
- If Bulyan (and/or FLTrust) is added, extend Table I with the new baseline columns and revise the Related Work differentiation paragraph accordingly (it currently states these were *not* empirically compared — that sentence needs to change once you add them).
- Explicitly state the grace period (§1.4) and the τ_thresh scaling decision (§1.2) in the methodology text.

---

## 5. Suggested timeline (rough)

- **Week 1–2:** Refactor hyperparameters into named arguments (§1.2, §5 item), fix the `or True` bug (§1.3), resolve the CIFAR-10 client-count question (§1.1).
- **Week 2–4:** Run Experiments 1–3 (multi-seed, ratio sweep, heterogeneity sweep) — no further code changes needed, just compute.
- **Week 4–5:** Run Experiment 4 (partial-flip sweep) and Experiment 5 (hyperparameter sensitivity).
- **Week 5–7:** Implement and run Experiment 7 (Bulyan baseline); attempt Experiment 8 (FLTrust) if time allows.
- **Week 7–8:** Rewrite the paper's evaluation section with the new results, regenerate figures, submit.

This fits comfortably inside a "put in real effort over a couple of months" plan, well short of needing a full new research cycle.