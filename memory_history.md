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
- Saved a project-status memory.

## 2026-11-18 (session 6)

- The GPU run started on the other machine: Linux, RTX 3090, 2 workers, about 35 s per round, so about 1 h per CIFAR job and an estimated 4–5 days in total. The user's commit `c3e21f1` from that machine fixed a ResNet BatchNorm crash on size-1 batches (singleton batches use BN eval mode) and added durable writes plus a posix flock.
- First real CIFAR non-IID CC result (round 27): all 8 attackers excluded, 0 honest excluded, accuracy 69%.
- Added client-side metrics: `client_eval_seconds`, `client_train_seconds`, `client_eval_overhead` (eval/train), `client_eval_samples`, `peak_gpu_memory_mb`. The report now uses a lean `_report_accuracy`, which gives identical values (verified), so the timing reflects a real client. Jobs finished before this change lack these columns.
- Metrics audit (same session): added `client_history.csv` (per-client per-round raw and effective report, consistency similarity, trust, counter, weight, loss; truncated on resume, so no duplicate rounds, verified). Also added peak GPU reserved, peak process RSS, per-round upload/download bytes, test-eval time, max client times, and summary detection delay (`mean_detection_round`, `full_detection_round`, `first_honest_exclusion_round`). Confirmed the manifest covers LFighter, FLTrust, FLAME, Multi-Krum, Median, FedAvg plus the 3 variants (core) and Trimmed Mean, FoolsGold, Tolpegin (full). EXPERIMENTS.md now lists every recorded metric.

## 2026-11-18 (session 7) — paper v9

- Created `Paper/v9/AntiFlipper_v9.tex`: 7 pages, standard IEEEtran margins, compiles cleanly. Retitled "Lightweight Detection of Label-Flipping and Report-Faking Clients". Contributions:
  - the report-faking attacker (Prop. 2: report-only detection cannot separate faked reports);
  - AntiFLipper-CC (M3 consistency check, Algorithm 1);
  - a multi-seed evaluation.
- Notation table fixes the α overload (η, α_D, γ, z, ρ). Removed the dynamic-attack table and the γ=0.1 numbers (no logs back them). Added bib entries `lamport1982byzantine` and `andreina2021baffle`.
- `Paper/v9/make_results.py` generates all tables (II headline, III detection, IV adaptive, V sensitivity, VI cost) and figures (gen_ratio, gen_heterogeneity, gen_trajectories) from `Code_Journal_Experiments/results/core`; it shows "--"/pending until runs exist. Red `\pending{}` notes mark interpretation to write after the results arrive. `Paper/v9/README.md` maps every reviewer comment to the change.
- Note: GPU-side commit `df84084` deleted the 6 old MNIST results and CC s31; they will be re-run (better, since they now get the full metrics). They are recoverable from `c3e21f1`.
- **Next:** when runs complete, pull, run make_results.py, rewrite the \pending sentences.
- **Next:** the user runs `run_all.cmd core 2` on the GPU machine and sends back `analysis/core/REPORT.md`; then the 6–8 page paper revision.

## 2026-11-18 (session 6)

- Started the 219-job AntiFlipper core queue with two RTX 3090 workers on `NDAG-M-Lab`. Enabled the `antiflipper-core.service` user unit at boot and verified that the user manager has lingering enabled.
- Made Linux launcher locks release on exit and synced checkpoint writes to disk. Fixed the CIFAR-10 ResNet BatchNorm crash on a one-image batch by using running statistics for that batch while keeping the example in training.
- Verified automatic recovery with a forced service stop. Both workers loaded their saved rounds and wrote new checkpoints. Six jobs were complete, two running, and 211 waiting, including one interrupted job with a saved checkpoint.
- Ran the existing PTPC cron watchdogs. The Guard queue has all 30 cell logs and a completed separability result. The local g14 queue is exhausted, with eight local logs and two placeholders delegated to `bio` whose results remain unverified.
- **Next:** let the core service finish and review `analysis/core/REPORT.md`.

## 2026-11-18 (session 7)

- Added executable `Code_Journal_Experiments/monitor.sh` for live Linux monitoring and `--once` snapshots. The monitor refreshes in a Linux terminal and exits cleanly on Ctrl+C.
- Verified both modes against the live core queue. At the time of the check, six jobs were complete and two workers were training.

## 2026-11-18 (session 8)

- Updated the Linux watch display to use the terminal’s alternate screen and clear it before each redraw. Verified a 10-second refresh and restoration of the original screen on Ctrl+C.

## 2026-11-18 (session 9)

- Checked the core ETA at 15:00 Dhaka time. Six of 219 jobs were complete and two workers were active. The monitor projected roughly nine days with continuous uptime, using live CIFAR-10 timing and historical MNIST timing from Windows runs.

## 2026-11-18 (session 10)

- Confirmed the interrupted core job is CIFAR-10 AntiFLipper-CC seed 31. It paused during the BatchNorm fix, and its checkpoint records two completed rounds. It is next in the queue behind seeds 7 and 19.

## 2026-11-18 (session 11)

- Confirmed two parallel core runs were active at 15:03 Dhaka time. Seeds 7 and 19 had completed 18 and 21 rounds, respectively.

## 2026-11-18 (session 12)

- Checked whether a third worker could speed up the core queue. The RTX 3090 showed 99–100% utilization across the sample while two AntiFlipper workers and another GPU workload were active. Kept the boot service at two workers because the GPU was already busy despite available VRAM.

## 2026-11-18 (updated runner resumed)

- Re-enabled `antiflipper-core.service` at boot with lingering enabled and started the updated 219-job core queue with two GPU workers. The pulled progress summaries were older than the local checkpoints. Seeds 7 and 19 resumed from rounds 38 and 41, then saved rounds 40 and 43 with the new metrics.
- Synced client history before writing each checkpoint and verified replay removes uncheckpointed entries. The live monitor now shows the latest round's average client training and report times. Verified screen clearing at 10 seconds and terminal restoration on Ctrl+C.
- Per-client history starts when the updated runner takes over an existing checkpoint. Let the queue continue from its saved progress.

## 2026-11-18 (new metric coverage)

- Checked whether resuming covered the new instrumentation. Client history starts at round 39 for seed 7 and round 42 for seed 19. The six completed MNIST runs also lack the new client timings. Recommended a fresh core run if every round needs the new measurements, with existing results and checkpoints archived first.

## 2026-11-18 (fresh core queue)

- At the user's request, deleted the local previous core results, checkpoints, analysis, and packages and started all 219 core jobs from scratch at 15:27 Dhaka time. Verified both GPU workers wrote client history and durable checkpoints starting at round 1. Boot startup before login remains enabled.
- Updated the monitor to estimate each active job and the full queue from recent round times, remaining rounds, and active workers. It flags datasets whose rates are still estimated from other datasets.

## 2026-11-18 (boot and sleep behavior)

- Verified the core service is active and enabled, systemd lingering is on, and the desktop's plugged-in idle action is set to do nothing. The service starts before login after Ubuntu boots. Screen lock leaves training active, while suspend pauses training until wake.

- Clarified that signing out of GitHub has no effect on local training, monitoring, or checkpoint recovery. GitHub credentials are only relevant when accessing a remote that requires authentication.

## 2026-11-18 (checkpoint cleanup)

- Added cleanup after final outputs and completion state are saved. Unfinished jobs retain their checkpoint, and the launcher retries completed-job cleanup after an interruption. Verified retention, resume, and cleanup on a two-round synthetic job.
- Reloaded the core service at safe round boundaries. Both GPU workers resumed from round 13, and boot startup remains enabled.

## 2026-11-22 — core results reviewed (Results_Backup_2026-11-22)

- The core run completed: 231/231 jobs (the GPU side extended the manifest to 231). The 3 "failed" states were early FLAME attempts that were re-run successfully. A `cc_v2` profile (method `antiflipper_cc_v2`, 12 jobs) exists only as a manifest; nothing has run and its code is not in the repo. Ask the user what it is.
- Findings (CIFAR-10 non-IID, 40% attackers, 3 seeds):
  - **Plain AntiFLipper is the strongest result.** It ties or tops accuracy in all 4 settings (71.1±0.3 CIFAR non-IID vs FLTrust 71.2 with root data; LFighter 50.5±24.4, Krum 49.0±37.2, FLAME 58.9±14.1) with the lowest variance, recall 1.0, FPR 0.08, and 23 ms defense time vs 42 (LFighter), 352+115 root (FLTrust), 875 (Krum), 1058 (FLAME).
  - **AntiFLipper-CC hurts at z=2.5:** FPR 0.25 non-IID / 0.36 IID, slightly lower accuracy. The MNIST sensitivity runs show z=3.0 brings FPR to 0.01–0.05 with recall 1.0, so the default z was too aggressive.
  - **Report faking on CIFAR only partly evades plain AntiFLipper:** recall 0.67, accuracy 72.6, ASR 3.5%. Likely cause: ResNet memorizes attackers' flipped training samples, so clean-label reports still come out lower. Prop. 2 in v9 must be qualified. CC raises recall to 0.88 at the cost of FPR 0.17.
  - **Strong heterogeneity breaks the report signal.** At α_D=0.1, clean AntiFLipper FPR is 0.92 (acc 16.9%) and attacked accuracy 13.2% vs FedAvg 43.9%; at α_D=0.5, FPR is 0.80. cc_only is robust when clean (FPR 0.05) but does not detect attacks at α_D=0.1.
  - Final-round accuracy is noisy (one p=0.3 seed: final 23%, last-10 mean 67.6%); consider last-10 as the headline metric.
  - Client report overhead measured at 9% (CIFAR) and 24% (MNIST), vs 11% analytic.
- **Next:** decide the paper framing (plain AntiFLipper as the main method; CC as an extension/trade-off) and whether to run CC with z=3 on CIFAR and/or a heterogeneity-aware report fix.
- User asked whether the results are publishable. Verdict given: yes at mid-tier (IEEE Access, ICC/GLOBECOM workshops, regional IEEE conferences) if the paper leads with plain AntiFLipper and states the α_D≤0.5 failure honestly; still borderline at main-track GLOBECOM (novelty concern unresolved). Fixing the heterogeneity failure (self-referenced trust) is the change that would make it clearly publishable.
- User asked whether another dataset helps. Answer: only a little for acceptance (it addresses R1's domain comment, not novelty or the α_D≤0.5 failure). If adding one, use a naturally partitioned, IoT-relevant dataset: UCI HAR (30 subjects, 6 classes, tiny model, cheap) or FEMNIST. Run a cheap pilot first, because natural non-IID may trigger the same false-positive failure. Code assumes 10 classes (LABEL_MAP, evaluate per-class arrays), so it needs a small generalization.
- Heterogeneity-fix design discussed (not implemented yet):
  - Diagnosis: comparing each client with the population mean penalizes honest clients that hold hard or rare classes, and permanent exclusion removes those classes from training, which pushes the next clients down (cascade, FPR 0.92 at α_D=0.1).
  - Correction: the earlier "self-referenced trust" idea does NOT catch static attackers (present from round 0); keep it only for dynamic attacks.
  - Proposal:
    1. Class-conditional residual: clients report per-class (correct, count); the server pools per-class accuracy robustly (per-class median or trust-weighted); the signal is acc_j minus the expected acc given the client's own label mix. Cost: 2K small ints (≈80 B for K=10), O(NK) on the server. Privacy option: broadcast the pooled per-class accuracy and let the client report only the scalar residual (label histogram stays local).
    2. Exclusion budget equal to the 40% threat-model bound, plus soft down-weighting before exclusion, to stop the cascade.
    3. Keep CC (z=3.0) as the backstop against report faking.
  - Validation: pilot ~6 CIFAR jobs (α_D=0.1 clean and attacked, α_D=1 headline, 1 seed), then about 60 jobs.

## 2026-11-22 (session 2) — class-conditional residual (AntiFLipper-CR) implemented

- Found that the GPU machine had unpushed code: a FLAME fix (BatchNorm buffers no longer clipped or noised), `antiflipper_cc_v2` (strike-based soft weights), and 12 cc_v2 jobs inserted into core (hence 231). Recovered it from the source snapshot in the results package and committed it as "Sync GPU-machine code" (840ac4e).
- cc_v2 results:
  - clean α_D=0.1: FPR 0, acc 62.7% (≈ the no-defense reference);
  - headline: recall 0, attackers keep 35% of the weight, acc 49±33%.
  - ⇒ The update check alone does not detect attackers; the report signal is needed but must stop penalizing label mix.
- Implemented `antiflipper_cr` and `antiflipper_cr_cc` (z=3.0):
  - clients report per-class (correct, count) (+84 B/round);
  - the server pools trust-weighted per-class accuracy, computes each client's expected accuracy for its label mix, and feeds the residual into the unchanged trust/counter pipeline;
  - new `max_exclusion_fraction` (default 1.0 = old behavior; CR uses 0.4, lowest trust excluded first).
  - Unit check: honest hard-class clients go from −0.275 raw deviation to 0.000 residual.
  - All 231 completed core job ids are unchanged (self-test checks this against the backup).
- New `cr` profile (84 jobs, ~34 GPU h, seed-major): the first 14 = seed-7 pilot (~6 h); includes the budget-only and CC-z=3 ablations. Added the Linux launcher `run_all.sh`; the monitor and run_all.ps1 know the `cr` and `cc_v2` profiles.
- `Paper/v9/make_results.py` now reads core + cr from both results/ and the backup, and keeps the ablation variants separate. Paper text is NOT reframed yet (wait for CR results).
- .gitignore: `Results_Backup_*/` and `ssh.txt` (local only).
- GPU machine: run `git checkout --` on the code paths before pulling (their local edits are already in 840ac4e), then `./run_all.sh cr 2`.
- CPU sanity (MNIST α_D=0.1, 10 clients, 8 rounds): no attack: AntiFLipper 86.2% with 1 honest excluded vs **CR 93.8% with 0 excluded**; 40% attackers: AntiFLipper 75.3%, 0/4 caught vs **CR 81.8%, 1/4 caught, 0 honest excluded**.
- GPU machine notes (from its uncommitted memory_history): FLAME was fixed there on 11-19; **FLTrust had the same BatchNorm bug** (1–7 NaN-loss rounds in most CIFAR runs). Fixed here: FLTrust scales trainable params only and averages BN buffers by score. New `fltrust_fix` profile (18 CIFAR FLTrust re-runs, notes="bn-safe"); make_results prefers them.
- **Remote access set up:** SSH key `~/.ssh/id_ed25519_antiflipper` (alias `antiflipper-gpu` in ~/.ssh/config → ndag@100.75.182.25, Tailscale). The user installed the public key; the password is not stored. The GPU machine has no GitHub access, so code goes over as a git bundle (scp + fetch + ff-merge).
- Deployed on GPU: its local edits are saved on branch `gpu-local-backup-2026-11-22` (4c67959); the machine is now at 76b6d2a. Self-test and smoke pass there. Started the systemd user service `antiflipper-cr` (3 workers, restart on failure, boot-enabled, linger on): runs `run_queue.sh cr fltrust_fix`. Started 2026-11-22 12:14 Dhaka; ~52 s/round per job with 3 parallel; pilot (14 jobs) ~7–8 h, all of cr ~40 h, fltrust_fix ~10 h more.
- Monitor: `ssh antiflipper-gpu` then `journalctl --user -u antiflipper-cr -f` or `Code_Journal_Experiments/monitor.sh --profile cr`.
- **Next:** check the pilot after ~8 h (clean α_D=0.1 FPR, attacked α_D=0.1 acc vs FedAvg 44%, headline ≈71%), then reframe Paper/v9.
- Gave the user the monitoring commands (`ssh -t ndag@100.75.182.25 "cd ~/Desktop/AAR/AntiFlipper/Code_Journal_Experiments && ./monitor.sh --profile cr"`, `journalctl --user -u antiflipper-cr -f`). Verified: service active, enabled, 0 restarts, linger on, checkpoints written every round; 3 workers; monitor estimate ~1d 21h for cr.
- First pilot results (2026-11-22 ~13:50, clean α_D=0.1, seed 7, CIFAR):
  - **CR: 0/20 honest excluded**, final 55.1%, last-10 52.0% (old AntiFLipper: 18/20 excluded, 16.9%; keep-everyone cc_v2 reference 62.7 final / 60.4 last-10 over 3 seeds ±6).
  - **Budget only: 8/20 excluded (the cap), 13.4%** ⇒ the residual, not the budget, fixes it.
  - CR+CC: 0 excluded, last-10 45.1%, final 10.3% = a one-round oscillation dip (56% at round 98).
  - CR weights stay ~uniform (≈5% each). Confirms using the last-10 mean as the headline metric.

## 2026-11-22 (monitor default)
- User reported `monitor.sh` over SSH showed the old run: it defaulted to `--profile core`.
- Fixed: monitor.py now defaults to the profile with the most recent `state.json` activity (cr now, fltrust_fix later); `--profile` still overrides. Deployed to GPU PC via bundle (GPU at 3f9d9e5); service untouched.
- CR status at 15:57: 10/84 done, 0 failed; A7 adaptive seed-7 jobs running.
- Next: check attacked α0.1 + headline pilot results (already complete on GPU).

## 2026-11-22 (seed-7 pilot complete, 17/84 done)
Last-10 accuracy, seed 7 unless noted; core baselines from Results_Backup_2026-11-22.
- Headline (α1, 40%): CR+CC 72.6%, recall 1, FPR 0 (best headline so far; AntiFLipper 70.9 FPR .17, CC 69.9 FPR .33). CR alone 69.3, recall 1, FPR 0 (final-round dip to 43.8, so report last-10). CC z=3 alone 71.0, FPR .25 (z alone does not fix FPR).
- Attacked α0.1: CR / CR+CC 35.8%, 7/8 attackers caught, 0 honest excluded. Old AntiFLipper 15.6 (FPR .92), budget only 13.6, FedAvg s7 33.9 (mean 43.8), cc_v2 s7 36.0 (catches 0).
- Adaptive report-faking: CR alone recall .375 (evadable by faking, as expected); CR+CC recall 1, FPR 0, 72.2%.
- Clean α0.5: CR 68.5 FPR 0; CR+CC 69.3 FPR .15 (old AntiFLipper 34.6, CC 66.5).
- Clean α0.1 seed 19: CR 61.9, CR+CC 53.1, both FPR 0 (old AntiFLipper 22.8).
- Decision: the paper's proposed method = AntiFLipper-CR+CC (CR fixes heterogeneity FPs; CC needed against report faking). Watch the CR+CC vs CR accuracy gap at clean α0.1 across seeds.

## 2026-11-22 21:45 (24/84 done, seed 19 under way)
- Seed 19 attacked α0.1: CR and CR+CC 59.0%, 8/8 caught, 0 honest excluded. Beats FedAvg s19 51.7, LFighter 50.2, cc_v2 49.4, old AntiFLipper 13.3. So the "only level with FedAvg" result was seed 7 only.
- Seed 19 headline: CR 71.4, CR+CC 72.0, both recall 1 FPR 0; CC z3 70.6 FPR .17.
- Service healthy, 0 restarts, ~27h left (cr), then fltrust_fix.

## 2026-11-23 08:38 (49/84 done; key block complete for all 3 seeds)
3-seed last-10 means (core baselines in brackets):
- Headline: CR+CC 72.0 (recall 1, FPR 0 every seed); CR 70.6; CC z3 70.9 (FPR .14). [AntiFLipper 71.0 FPR .08, FLTrust 69.2, FedAvg 44.2]
- Attacked α0.1: CR 49.1, CR+CC 49.1 (recall .96, FPR 0). [FedAvg 43.8, cc_v2 43.6, LFighter 41.6, FLTrust 39.3, AntiFLipper 13.0, budget-only 12.5]
- Clean α0.1: CR 60.2 (FPR .02), CR+CC 54.2 (FPR 0). [cc_v2 60.4, cc_only 57.5, AntiFLipper 16.3, CC z3 21.3]
- Clean α0.5: CR 64.8, CR+CC 66.5. [AntiFLipper 40.7, CC 59.7, cc_only 65.5]; clean α1: CR 75.5, CR+CC 75.4.
- Adaptive faking: CR+CC recall 1 all seeds, FPR 0, 71.9; CR alone recall .63.
- Remaining: ratio 10/20/30%, CIFAR IID, MNIST coverage (~19h), then fltrust_fix.

## 2026-11-23 16:07 (64/84 done, ~13h left in cr)
- Ratio sweep (3-seed last-10): CR 74.3/73.5/72.8 at 10/20/30%, recall 1, FPR ≤.02. CR+CC 74.1/73.3 (30%: 1 seed so far, 72.1), recall 1, FPR ~.04-.05.
  Core baselines: AntiFLipper 73.2/72.4/69.2 (FPR .09-.17), FLTrust 71.8/71.8/70.8 (FPR .31-.43), LFighter 72.4/71.4/64.9, FedAvg 74.3/73.5/61.0.
- CR/CR+CC are the most accurate defense at every ratio, and at 10-20% match FedAvg (no cost of defending).
- Remaining: CR+CC 30% seeds, CIFAR IID, MNIST coverage; then fltrust_fix.

## 2026-11-23 18:04 (GPU contention)
- 17:31: another user's GPU workload on the NDAG PC took ~21 GB of the RTX 3090 -> CUDA OOM. The orchestrator stopped cr safely (checkpoints kept), the queue moved on to fltrust_fix, which also hit OOM; the service exited 1 and systemd restarted it (NRestarts=1). cr hit OOM again, so fltrust_fix is running now.
- State: cr 68/84 (16 left: CIFAR IID CR+CC, MNIST), fltrust_fix 3 running (~18%). Rounds ~115 s instead of ~50 s because the GPU is shared.
- Self-healing: fltrust_fix finishes -> queue exits 1 (cr failed) -> systemd restarts after 60 s -> cr resumes, completed jobs skipped. No action needed unless OOM repeats.
- ETA: ~1-1.5 days if the other job keeps running, ~15 h if it stops.

## 2026-11-23 18:15 (other GPU job)
- The other GPU workload (another lab user's, running since 07:49) caused the 17:31 OOM; expected end ~20:30 Nov 23. Not touched. Details in local-only memory_history.local.md.
- Impact: accuracy/detection unaffected (seeded, RNG restored on resume). Timing is inflated for jobs run during contention (fltrust_fix, last cr jobs). Take paper timing claims from core + the timing benchmark; flag overlapping jobs, re-run the timing benchmark on an idle GPU if needed.
- ETA: fltrust_fix ~04:00-05:00 Nov 24, remaining 16 cr jobs ~11:00-13:00 Nov 24.

## 2026-11-23 (results snapshot copied + analyzed)
- Copied results/cr (74 dirs, 68 complete) and results/fltrust_fix (partial) from the GPU, without checkpoints, into Results_Backup_2026-11-23/ (gitignored). Tables: Results_Backup_2026-11-23/ANALYSIS_tables.md (script analyze.py next to it).
- New observations beyond earlier entries: CIFAR IID CR 73.6 (2 seeds) ≈ AntiFLipper 73.3; CR detection is slow under α0.1 attack (full detection ~round 70 vs ~15 at α1); CR defense cost ~30-40 ms/round vs 23 ms for plain AntiFLipper (still ≪ FLTrust ~360 ms, Multi-Krum ~900, FLAME ~1000).

## 2026-11-23 (timing: keep only GPU-free rounds)
- User: sample timing only from rounds when the GPU was free. Built Results_Backup_2026-11-23/contention.py: reconstructs each round's wall-clock end from journal COMPLETED timestamps (service_events.log) minus cumulative round_seconds, tags rounds inside the other job's window (07:49 Nov 23 .. the other workload's end) as gpu_shared -> round_timing_tagged.csv.
- CR CIFAR medians, free vs shared: client train 2.27 vs 2.74 s (+20%), test eval 1.12 vs 1.97 s, server defense 0.035 vs 0.034 s (CPU, unaffected). Core run client train median 2.18 s (consistent with free).
- Plan: paper timing tables use GPU-free rounds only; FLTrust re-runs are fully inside the shared window, so FLTrust timing comes from core (accuracy from the bn-safe re-runs). Re-run contention.py after the queue ends (add fltrust_fix + the resumed jobs; resumed jobs need anchoring at their Resuming timestamp).

## 2026-11-23 21:03 status
- The other GPU workload ended 19:31. New load since 17:45: another lab user's CPU-heavy job (load avg 67-92 on 24 cores). FLTrust rounds still ~100-130 s (vs ~49 s free); GPU only 58% busy -> CPU-bound.
- fltrust_fix 3/18 done (A1 headline, 3 seeds), A2 attacked jobs running. cr 68/84 (resumes after fltrust_fix).
- Timing filter generalized: Results_Backup_2026-11-23/busy_windows.txt (local) lists GPU and CPU workload windows; contention.py tags rounds in any window. Server defense timing is CPU-side, so CPU contention can affect it -> the CPU window is also excluded.
- ETA: ~midday Nov 24 for fltrust_fix and evening Nov 24 for cr if the CPU load continues; ~8 h less if it stops.
- 21:08: the CPU job is about half done; expected end ~00:30-01:00 Nov 24. Revised ETA: fltrust_fix ~05:30, cr ~09:00 Nov 24 (if nothing else starts).
- Scheduling check on NDAG PC: nothing else scheduled that can start new work; the other workloads were started by hand.

## 2026-11-23 ~21:30 (decision pending)
- User asked whether to stop the other lab user's CPU job. Advised against it without the owner's consent: it is expected to end ~00:30-01:00 anyway, would only save ~2-3 h, and results are unaffected. Options: wait, ask the owner, or renice with their OK.
- Decision deferred to ~01:00 Nov 24: check whether the CPU job has ended; if not, the user decides.

## 2026-11-24 00:48 status + FLTrust finding
- The other CPU workload has ended (load avg 3); FLTrust rounds are back to ~50 s. The 01:00 decision is moot.
- fltrust_fix 6/18 done, 3 at ~40%; ETA ~05:50. Then the 16 cr jobs: all done ~09:00-10:00 Nov 24.
- **The BN-safe FLTrust is worse, not better**: headline last-10 57.7 (s7 57.9, s19 56.5, s31 58.8) vs core 69.2. No NaN rounds any more (core had 1-3), but single-round collapses to 5-12% accuracy, then recovery. Attacker weight share is the same as core (20-30%), so detection did not change. Likely cause: parameters move by the root-norm-scaled update while BN buffers jump to the clients' fully trained statistics -> parameter/statistics mismatch. Attacked α0.1 bn-safe: 29.9/37.1/46.4 (core 32.1/41.2/44.7).
- Decision (recommended): the paper uses the core FLTrust as the main baseline (the stronger version: the fairer comparison, and still below CR+CC 72.0), notes its 1-3 NaN rounds, and reports the BN-safe variant in a footnote. Let fltrust_fix finish as is (the total finish time does not change). Possible alternative later: rescale BN deltas like the parameters and clamp running_var > 0, or take BN statistics from the root model.

## 2026-11-24 ~01:00 (FLTrust timing)
- contention.py now covers cr + fltrust_fix (anchor: journal COMPLETED, else round_metrics.csv mtime for running jobs) and reports per method. The CPU workload window was closed at its end, 00:33 Nov 24 (local busy_windows.txt).
- Server defense per round (median), free-machine rounds: CR 36 ms, CR+CC 36 ms, AntiFLipper 25 ms, FLTrust 403 ms (+178 ms root training; n=67 free rounds so far). Shared-machine FLTrust: 2.95 s (7x inflated by CPU contention); CR barely affected (32-35 ms). Core-run FLTrust 352-399 ms agrees with the free-round value.
- Paper: FLTrust timing = free rounds of the bn-safe re-runs (the 12 jobs from 00:09 on are all free) cross-checked with core; plus the matched-N aggregation benchmark on an idle GPU as the controlled comparison. FLTrust ≈ 11x the server cost of CR.

## 2026-11-24 01:57 (automatic result pulls)
- Results_Backup_2026-11-23/pull_results.sh copies results/cr + fltrust_fix (no checkpoints) from the NDAG PC via temp dir (a failed pull keeps the old copy), refreshes service_events.log, logs to pull_results.log.
- Windows Task Scheduler task "AntiFlipper pull results": 06:30, 11:30, 13:30 Nov 24 (StartWhenAvailable, so a missed run happens on wake). Tested: OK. Pulled now: cr 68/84, fltrust_fix 9/18.
- Remove later with: Unregister-ScheduledTask -TaskName 'AntiFlipper pull results' -Confirm:$false

## 2026-11-24 09:36 — ALL RUNS COMPLETE
- cr 84/84 and fltrust_fix 18/18 done; service inactive (Result=success). Final pull OK -> Results_Backup_2026-11-23/ (ANALYSIS_tables.md regenerated).
- Coverage: CIFAR IID CR+CC 73.3, CR 73.5 (AntiFLipper 73.3, LFighter 73.2), FPR 0. MNIST IID/non-IID CR & CR+CC 98.7, recall 1, FPR 0 (CC 8/20 honest excl. on non-IID).
- FLTrust old vs bn-safe (3-seed): headline 69.2 vs 57.8; IID 63.1 vs 63.9; α0.1 attacked 39.3 vs 37.8; ratio 10/20/30% 71.8/71.8/70.8 vs 69.7/68.1/63.0. bn-safe: 0 NaN but lower in 5/6 -> paper uses original FLTrust as main baseline, bn-safe in footnote.
- Timing (GPU/CPU-free rounds, median server defense): CR 36 ms, CR+CC 36 ms, FLTrust 415 ms + 174 ms root training (n=1136 free rounds); MNIST CR 3-4 ms.
- NDAG PC: NVIDIA driver/library mismatch (driver updated overnight) -> needs a reboot before new GPU work.
- Next: update Paper/v9/make_results.py for CR rows + FLTrust choice + free-round timing, then rewrite the paper around AntiFLipper-CR+CC.

## 2026-11-24 10:15 (verification + full NDAG copy)
- Local results verified against NDAG by md5: cr + fltrust_fix 822/822 files present; only 3 environment files differ (rewritten when the service re-ran at boot). Core: 693/693 result files identical to Results_Backup_2026-11-22.
- NDAG PC rebooted ~09:48 (driver mismatch fixed: driver 580.196.07, GPU idle). On boot antiflipper-cr started, skipped all jobs, re-ran analysis/packaging, exited (inactive). Nothing of ours running. The service is still enabled and will repeat this no-op on every boot.
- Full NDAG copy: Results_Backup_2026-11-24_NDAG/ (gitignored): repo/ = whole AntiFlipper tree minus datasets (data/, 242 MB) and .pt checkpoints (396 MB: results incl. core/cr/fltrust_fix/failed_attempts, analysis, packages zips, manifests, Paper, notes); ndag_all_branches.bundle (verified); ndag_git_status.txt. NDAG-only branches fetched locally as ndag/gpu-local-backup-2026-11-22 and ndag/publish-aashnan-completed-20261120 (not pushed).

## 2026-11-24 (paper v10)
- Wrote Paper/v10 (7 pages, compiles clean): proposed method = AntiFLipper (CR+CC+40% cap); the original method is "v1" and an ablation. New Props: raw reports penalize label mix (1), the residual removes it (2), faked reports defeat any report-only rule incl. CR (3); exclusion cap bound; cost; privacy (per-class counts leak label histograms -> secure-aggregation variant proposed, not implemented).
- make_results.py v10 reads Results_Backup_2026-11-24_NDAG; last-10 accuracy; original FLTrust (bn-safe in a footnote); free-round timing; all in-text numbers are macros (generated/numbers.tex). Tables: headline, detection, heterogeneity, adaptive, cost, sensitivity; figs: ratio, trajectories.
- README maps all reviewer comments: most addressed; open: partial flips (R3-1), collusion, dropout/partial participation, other domains/edge hardware, CRFL, formal privacy. The best next experiment: partial-flip block (~18 GPU h).

## 2026-11-24 (paper v10 rewrite)
- User feedback: write it as a real paper (no "v1"/revision history), more professional, better tables, short captions referenced from the text, say "three seeds" without seed values, redraw the figure.
- Rewrote Paper/v10: standalone narrative (accuracy feedback -> two weaknesses -> AntiFLipper); original design = "Raw reports" ablation. New TikZ architecture (fig/architecture.tex). Tables rebuilt with booktabs, best values in bold, one-line captions: I main comparison (accuracy+detection+time), II ablation with CR/CC/cap checkmarks over 4 scenarios, III strong skew, IV cost, V sensitivity. Figs: heterogeneity sweep, trajectory (5-round moving average), ratio. 7 pages, compiles clean.

## 2026-11-24 (internal review of v10 -> revision)
- User pasted a "major revision" review of v10. Verified each point against the data: strong-skew recall 0.96 with full detection 2/3 seeds (60, 81; seed 7 misses 1/8); clean α0.1 full-method seed 7 final 10.3% with all clients active; CC marks only 0.6% of client-rounds inconsistent there (so the old "CC overrides honest residuals" explanation was wrong); ratio-sweep FPR up to 0.14; sensitivity table was raw+CC.
- Revised the paper: narrowed abstract/intro/conclusion (the reviewer's lead claim), fixed theory (assumptions, w_k bias, no cosine guarantee, no convergence claim), both FLTrust implementations + NaN counts (6 rounds/3 runs), last-10 ASR, final-round accuracy + std in ablation, "All found k/3" in Table III, table notes moved below tabulars, matched timing column.
- Added antiflipper_cr/_cr_cc to benchmark.py; ran the matched benchmark locally (CPU, 20 updates): AntiFLipper 173 ms, FedAvg 183, LFighter 214, FLTrust 1269, FLAME 3465, Multi-Krum 4109 -> Results_Backup_2026-11-23/timing_matched/.
- Prepared but NOT launched: `rev` profile (69 jobs: P1 faking+strong skew, matched CC-only z3+cap, clean FedAvg α 0.1/0.5/1; P2 partial flips 0.5/0.25; P3 full-method sensitivity on MNIST) + antiflipper-rev.service. Needs the user's go-ahead (shared GPU PC) and a code sync via bundle.

## 2026-11-24 11:31 (rev profile launched)
- User: "schedule it". Synced code to the NDAG PC via bundle (GPU now at 596cd7e). Installed antiflipper-rev.service (enabled, linger) running `run_queue.sh rev`; disabled antiflipper-cr.service (finished; avoided its no-op re-run at every boot). Started 11:31; first jobs = A8 faking + strong skew, seed 7. GPU was idle; someone was at the desktop.
- ETA ~25 GPU h at full speed: P1 ~23:30 Nov 24, P2 ~05:00, P3 ~12:30 Nov 25 (monitor's 1d22h is pessimistic, it uses CIFAR timings for MNIST).
- pull_results.sh now also copies results/rev and the antiflipper-rev journal (service_events_rev.log). Task Scheduler pulls: Nov 24 18:00; Nov 25 00:30, 07:00, 13:30, 20:00.
- Monitor: ssh -t ndag@100.75.182.25 "cd ~/Desktop/AAR/AntiFlipper/Code_Journal_Experiments && ./monitor.sh" ; logs: journalctl --user -u antiflipper-rev -f ; stop: systemctl --user stop antiflipper-rev
