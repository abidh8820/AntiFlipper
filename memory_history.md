# Memory History

Running log of conversations and context for AntiFlipper. Newest entries at the bottom.
Each entry: date, what was discussed/done, decisions, and next steps.

---

## 2026-11-18

- Discussed how to keep conversation context available across machines (repo notes + `CLAUDE.md`, syncing the `~/.claude` memory folder, or cloud sessions).
- **Decision:** maintain this `memory_history.md` in the repo root after every conversation.
- Created `CLAUDE.md` with the rule to update this file at the end of each conversation.
- **Next:** none pending.

## 2026-11-18 (session 2)

- Reviewed `Paper/v8/reviews.txt` (GLOBECOM 2026, 3rd rejection; scores mostly 3/3/3, presentation 2–3). Main criticisms: limited novelty ("well investigated"), single-seed results, no malicious-ratio sweep, only Dirichlet α=1, no hyperparameter values/sensitivity, fake-accuracy-report attacker not addressed, old references, α symbol overloaded, section outline skips VII, unexplained 3.01ms < FedAvg 4.61ms, inconsistent client counts.
- Found the existing plan (`Note/updates2.md`) and the experiment framework (`Code_Journal_Experiments`, 258-job core manifest). Only 6 of the 258 jobs are done (MNIST IID: AntiFlipper and FedAvg; Median is partial).
- Advised: the core problem is novelty, not seed count. Cut the matrix hard (drop MNIST from sweeps, use CIFAR-10 non-IID as the main setting, about 6 baselines, sweeps with 4 methods). Add one real new idea (an accuracy-faking adaptive attacker plus a cheap mitigation). Target a venue that fits the work (journal or TPS letter) instead of GLOBECOM a 4th time.
- **Next:** user to pick a direction (venue and reframe); run a per-job timing benchmark to size the reduced matrix.

## 2026-11-18 (session 3)

- User decision: **not submitting anywhere for now**. The paper is kept as proof of work, target 6 pages (6–8 allowed); a detailed journal version comes later. This session: code only. Prepare the scheduler but do **not** run or schedule it.
- New branch `reduced-matrix-2026` (from `update-2026`).
- Code changes in `Code_Journal_Experiments/`:
  - Reduced, priority-ordered manifest: core = 171 jobs (was 258), full = 237. Old manifest archived as `manifests/core_v1_258jobs.json`. The 6 completed MNIST-IID jobs keep their ids (late config fields are omitted from the hash at defaults; A1 keeps the v1 tags `("tier2","headline")`).
  - New adaptive attacker `report_strategy="clean_labels"` (trains on flipped labels, reports accuracy on clean labels).
  - New method `antiflipper_cc`: last-layer cosine to the coordinate-median update, with a robust z-score; inconsistent reports are replaced by the round's worst report.
  - Timing now includes the trust update (`defense_seconds`); FLTrust root training is separate; per-round detection P/R/F1/FPR for all selecting baselines; analytic upload bytes.
  - Removed the stale `.locks/core.lock` (PID from Sept, dead).
- Default `python` on this machine is CPU-only (torch 2.8.0+cpu). The GPU env must be set through `ANTIFLIPPER_PYTHON`.
- **Next:** paper revision (6–8 pages); later, run `run_all.cmd core <workers>` from the GPU env.

## 2026-11-18 (session 4)

- Sanity check (MNIST non-IID, 10 clients, 8 rounds, CPU): the adaptive report-faking attacker fully evades AntiFLipper (0/4 excluded, 95.6% → 79.8%). The first CC version flagged 4/4 attackers but excluded none, because replacing the report barely moved trust. Fix: CC-inconsistent clients count as below-threshold toward the counter. After the fix CC excluded 4/4, FPR 0, 95.7%.
- Git hygiene: added `Code_Journal_Experiments/.gitignore` (pycache, locks, checkpoints, zips); untracked pycache and lock. `To run.txt` was rewritten for the GPU machine (path `C:\Projeccts\AntiFlipper`).
- Committed and pushed branch `reduced-matrix-2026` to origin. Nothing has been run on GPU yet.
- **Next:** user pulls on the GPU device and runs `run_all.cmd core 2`; then the paper revision.

## 2026-11-18 (session 5)

- User decisions:
  - **Threat model caps malicious clients at 40%**; never sweep above it.
  - "Do whatever gets it accepted." Code changes are delegated to Claude; the user only runs on GPU.
- Added the `cc_only` ablation (cross-check only, reports ignored, uniform weights). **AntiFLipper-CC is now the proposed method**; AntiFLipper and cc_only are its two halves. Reason: the cross-check is in the LFighter/FLAME family, so the paper must show that the combination beats each part.
- Manifest: core 219, full 294. The ratio sweep is {0.1, 0.2, 0.3} (0.4 = headline); `validate()` rejects anything above 0.4. Sensitivity now runs on CC and adds `consistency_z` {2, 3}. cc_only is in the headline, clean/attacked heterogeneity, and partial flip.
- Open risks to check first in the results:
  - CC false positives at Dirichlet 0.1 (A2).
  - Whether cc_only ≈ CC; if so, the accuracy report adds little.
  - Old paper numbers may shrink under multi-seed final-accuracy reporting.
- Pushed `reduced-matrix-2026`.
- Saved a project-status memory. **Next:** the user runs `run_all.cmd core 2` on the GPU machine and sends back `analysis/core/REPORT.md`; then the 6–8 page paper revision.
