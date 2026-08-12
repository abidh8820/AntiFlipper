# Response to Reviewers — AntiFLipper (GC 2026)

We thank the reviewers for their careful and constructive feedback. Below we address each point raised in Reviews 1–3. Changes are marked in **blue** in the revised manuscript (new/modified text) and **red** (values the authors must confirm before final submission — see note at the end).

---

## Review 1

**W1 / R1: No analysis of detection under varying malicious-client ratios; Lemma-1 assumption weakens as poisoning ratio grows.**
Added Section VI-C ("Dependence on the Malicious Client Ratio"), which analytically derives how the detectable accuracy gap shrinks as the malicious fraction *p* grows, and explains why our fixed 40% setting matches the standard worst case used by comparably-scoped defenses (e.g., Multi-Krum). We are explicit that we have **not** empirically swept *p*, and list this in the new Limitations section (§VIII) as follow-up work, including deriving the precise failure threshold *p**.

**W2 / R2: Narrow threat-model scope vs. general-purpose defense claims; no comparison to Bulyan, CRFL, FLTrust.**
Softened the framing throughout (abstract, related work) to be explicit that AntiFLipper targets the label-flipping threat model specifically, and added a new Related Work paragraph (§III-A) positioning AntiFLipper against Bulyan, CRFL, and FLTrust, explaining precisely how it differs (no server-held reference data, no update/gradient inspection, single scalar vs. full update). We candidly state that we have not benchmarked against these methods in this version and list it as a scoping limitation (§VIII).

**W3 / R3, R7: No statistical rigor (single-run results); no hyperparameter sensitivity analysis; no resource-constrained hardware measurements.**
Added explicit acknowledgment in §VIII (Limitations) that all results are single-run, and that hyperparameter sensitivity ($\eta$, $k$, $cnt_{\max}$) and on-device (memory/energy/latency) measurements were not performed. The revised manuscript now identifies the exact hyperparameters that must be recovered from the experiment configuration; their values remain author-action items and must be filled before submission.

**R4: Expand to non-image domains and lightweight architectures.**
Listed explicitly as an unaddressed gap in §VIII ("Domain and architecture coverage"). No new experiments were run.

**R5: Formal privacy analysis of accuracy-score leakage; consider differential privacy.**
Added §VI-E ("Privacy Considerations of Accuracy Reporting"), which discusses the membership-inference-style risk qualitatively, contrasts it with gradient-based leakage, and proposes a concrete DP mitigation (Laplace/Gaussian noise on the scalar) as a specific, tractable next step. We do not claim to have quantified the resulting privacy-utility trade-off empirically.

**R6: Clarify novelty relative to prior trust/validation-score defenses.**
Rewrote §III-A to explicitly differentiate AntiFLipper from FLTrust, HSCS, MCDFL, FLGT, and other 2023–2025 trust-based defenses (see reference list changes below), identifying the single-scalar, no-reference-data design as the specific point of departure.

**R7: Overhead analysis assumes full-capacity devices; provide on-device measurements.**
Acknowledged explicitly in §VIII as an unaddressed gap; the existing overhead analysis (§VII) is now explicitly labeled as analytical rather than measured on constrained hardware.

---

## Review 2

**R1: Threat model doesn't consider clients that falsify their reported accuracy to evade detection.**
Added a dedicated discussion in the new §II-B ("Threat Model Boundary and Adaptive Adversaries") that treats falsified accuracy reporting as an explicit extension of the threat model, explains why it falls outside our stated scope, and proposes a challenge-set-based mitigation direction.

**R2: No communication overhead analysis.**
Added §VII-A ("Communication Overhead"), an analytical treatment showing AntiFLipper's per-round communication is essentially identical to vanilla FedAvg (one extra 4-byte scalar), contrasted with the payload expansion of cryptographic defenses like Shen et al.

**R3: No repeated-run / mean ± std results.**
Acknowledged explicitly and listed in §VIII as the top limitation; we did not fabricate multi-seed numbers.

**R4: Only Dirichlet α = 1 tested; no extreme heterogeneity.**
Acknowledged explicitly in the new §VI-D ("Robustness to Non-IID Skew and Adaptive Adversaries") and listed in §VIII.

**R5: τ_thresh and cnt_max values not reported; no sensitivity analysis.**
The revised Implementation section (§V) now explicitly states these as named parameters with placeholders for the authors to insert the actual values used, rather than leaving them unstated as before. A full sensitivity sweep was not run (see §VIII).

**R6: No explanation for AntiFLipper being faster than FedAvg baseline.**
Reframed the result conservatively in §V-C: the small difference must not be interpreted as an algorithmic speedup over averaging. The authors must confirm from code and logs exactly what the timer includes, synchronization/warm-up/repetition details, and whether AntiFLipper aggregates fewer clients after filtering. The meaningful result is its much lower measured time than the more computationally intensive FLAME/LFighter implementations.

**R7: Inconsistent client counts between MNIST and CIFAR-10, IID vs non-IID.**
Explicitly clarified in §V (MNIST uses 100 clients throughout; CIFAR-10 uses 100 for IID and 20 for non-IID). We flag with a red marker that the authors should confirm/state the actual rationale (we propose a plausible one — sample-per-client sizing for Dirichlet partitioning under ResNet18/ShuffleNetV2 — pending confirmation) and consider adding a matched 100-client non-IID run if resources allow.

**R8: No convergence analysis under progressive client exclusion.**
Added §VI-F ("Convergence Under Progressive Client Exclusion"), an informal argument that each round is structurally a weighted-FedAvg instance on the current honest subset, and that exclusion (being monotonic and reserved for persistently malicious clients) does not worsen the honest-set heterogeneity bound. We are explicit this is not a formal proof.

**R9: Only 3 of 21 references from 2023–2025.**
Expanded the bibliography with recent label-flipping defenses and relevant general-purpose baselines, including Jiang et al. (2023), Ovi et al. (2023), LFighter (2024), FLGT (2025), Abroshan (2025), and FedAlign (2026), as well as Bulyan, CRFL, FLTrust, and FedAvg convergence work. Metadata for the recent entries was checked and corrected.

**R10: Symbol α used with three different meanings.**
Fixed throughout: α is now reserved for the Dirichlet concentration parameter only; the trust-update learning rate is renamed η; the local-evaluation fraction is renamed γ.

**R11: Structural overview in the introduction skips Section VII.**
Fixed — the introduction's roadmap paragraph now correctly enumerates all sections including the new §VII (overhead) and §VIII (limitations).

---

## Review 3

**W1: Core idea has precedent in trust-based/validation-score literature; originality limited.**
Addressed via the expanded Related Work differentiation paragraph (§III-A), which explicitly positions AntiFLipper's single-scalar, no-reference-data design against FLTrust, HSCS, MCDFL, and FLGT.

**W2: Security lemmas are informal; don't address adaptive attackers who mimic honest accuracy, or colluding clients.**
Added §II-B and §VI-D, which discuss partial-flip and colluding strategies explicitly designed to evade average-based detection. The revision no longer claims an unproved bound on the stealth-vs-impact trade-off and is explicit that empirical stress-testing was not performed.

**W3: Honest clients with severe non-IID skew could be penalized; not stress-tested.**
Addressed directly in §VI-D and flagged in §VIII as an evaluation gap (only α = 1 tested).

**R1: Evaluate adaptive/partial-flip strategies designed to evade detection.**
Discussed analytically in §II-B / §VI-D; not empirically evaluated (§VIII).

**R2: Compare directly with FLTrust/HSCS-style defenses under identical CIFAR-10 non-IID splits; report precision/recall.**
Discussed in the Related Work differentiation paragraph and explicitly listed as an unaddressed baseline-comparison gap in §VIII.

**R3: Fix section numbering (Section VII before VIII in outline).**
Fixed — see response to Review 2, R11.

**R4: Discuss privacy leakage from repeated accuracy scalars.**
Addressed in the new §VI-E (see response to Review 1, R5).

---

## Summary of manuscript changes

- **New/expanded sections:** §II-B (Threat Model Boundary and Adaptive Adversaries), §VI-C–F (malicious-ratio dependence, non-IID/adaptive robustness discussion, privacy considerations, convergence discussion), §VII-A (Communication Overhead), §VIII (Limitations and Threats to Validity).
- **New figure:** accuracy-vs-aggregation-time trade-off plot (Fig. 3), built from the existing Table I data, giving a clearer visual case for the efficiency claim.
- **Notation fix:** α / η / γ disambiguated throughout.
- **References:** 12 new citations added, most from 2023–2026, including the general-purpose baselines (Bulyan, CRFL) and FLTrust requested by reviewers.
- **Honesty on scope:** rather than claiming coverage we don't have, every reviewer request that requires new experimentation (multi-seed runs, malicious-ratio sweep, extreme non-IID, SOTA baseline comparisons, on-device measurements) is explicitly named as a limitation with a concrete description of what the follow-up experiment would look like.

## ⚠️ Outstanding items requiring author input before resubmission

The following are marked in **red** directly in the .tex file and must be resolved before submission—we did not fabricate values or implementation details:

1. The actual trust learning rate η, trust threshold scale *k*, and flag threshold `cnt_max` used to produce Table I's numbers.
2. Confirmation (or correction) of the proposed rationale for why CIFAR-10 non-IID uses 20 clients while IID uses 100.
3. Timing-protocol details needed to explain the FedAvg/AntiFLipper difference: timed code region, repetitions, warm-up, device synchronization, and number of models actually aggregated.
4. Dynamic-attack detection logs: detected-client counts, detection rounds, false positives/negatives, and scenario aggregation times.
5. The 10% evaluation-subset implementation: subset sampling rule, seed, detection results, and measured evaluation latency.
6. If time permits before resubmission, even a minimal 3-seed re-run of the headline MNIST/CIFAR-10 numbers (to report mean ± std) would meaningfully strengthen the response to Reviews 1 and 2's statistical-rigor concerns; this is the single highest-leverage new experiment if any experimentation is possible.
