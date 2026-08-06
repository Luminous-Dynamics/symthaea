# MA-001R-delta run logs — preserved raw output

Per `ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md` §9 step 5 (user-approved
2026-07-26) and the `../ma001-runs/`/`../ma001r-runs/`/`../ma001l-runs/` non-erasure precedent.

- **`ma001r-delta-run-2026-07-26.txt`** — `DeltaRuleLearner` (MA-001L, passed all 7 gates on
  prerecorded synthetic data) wired into a real, continuously-evolving `Organism` via
  `Ma001rProbe::run_with_delta_rule`/`run_shuffled_with_delta_rule`, repeating MA-001R's exact
  protocol (baseline/post-training counterfactual reading, held-out check, equal-outcome control,
  shuffled-context control, reversal condition). Seed 1, `Ma001rConfig::default()`. Run via
  `cargo run -p symthaea-alife --example ma001r_delta_run --release`. Build noise excluded.

**Result** (see the plan doc's §12 for the full write-up): **the delta rule does not replicate its
clean MA-001L success on a live Organism — and fails in a specific, diagnosed way, not merely a
weaker version of success.** The trained model learns the *opposite* of the true relationship
(predicts *low* energy for the context that truly maps to *high* energy), which makes its held-out
predictions *worse* than an untrained model's, and the reversal condition shows non-monotonic
swinging rather than a clean flip. Root cause (code-verified, not merely inferred):
`ActiveInferenceAgent::update_belief` is a genuine incremental gradient step pulled toward a fixed
prior, not an instantaneous snap to the current observation — so a real organism's belief does not
cleanly track "which context is present this specific tick" the way MA-001L's fixed synthetic
placeholder did by construction. This is a real architectural mismatch between what MA-001L
validated (an idealized, instantaneous-signal data-generating process) and what a live organism's
own perceptual dynamics can actually supply as input to the same rule — not a re-opening of
MA-001L's own result, which remains valid on its own terms.

- **`ma001r-delta-run-v2-2026-07-26.txt`** — the diagnosed fix (§13): both the original
  belief-based mechanism and a new gated-observation-based one run side by side via
  `Ma001rProbe::run_with_delta_rule_from_observation`/`run_shuffled_with_delta_rule_from_observation`
  (built from a new `OrganismTick::gated_observation` field). Run via the same command.

**Result** (see the plan doc's §13): **partial success, not full resolution.** The fix genuinely
resolves the worst problem — `direction_correct` flips from false (confidently backwards) to true,
and the post-training effect size collapses from a wild, pathological 0.7666 to a modest, plausible
0.1955 — but the remaining gates still fail: the effect is ~4× weaker than MA-001L's own clean
result, the shuffled control does not collapse (it slightly *exceeds* the bound arm), the held-out
check improves for only one of the two contexts, and reversal shows almost no movement over 2,000
further ticks. Disclosed, untested candidate factor: `gate_observation`'s own blanket-permeability
attenuation (a real mechanism absent from MA-001L's idealized synthetic tuples) may be weakening
the signal below what's needed for full replication.

- **`ma001r-delta-belief-trajectory-2026-07-26.txt`** — direct measurement (§14) of
  `self.organism.agent.belief.mean[2..6]`'s actual trajectory during the original belief-based
  training run, tick by tick, to test the "smearing" hypothesis directly rather than infer it from
  code alone. New example `ma001r_delta_belief_trajectory.rs`.

**Result** (see the plan doc's §14): **the "belief smears to a blind constant" framing was
incomplete — corrected.** Belief does not cleanly track the true per-tick context (tracking ratios
0.63–0.77, i.e. it recovers only 63–77% of the true context separation), but it also does not
smear to a single context-blind value: at every steady-state tick, Context A's belief is reliably
*higher* than Context B's in every informative dimension — a stable, damped, but correctly-ordered
two-point oscillation. Since v1's actual failure was a **sign flip** (the ordering itself
inverted), a signal whose ordering the measurement shows is preserved cannot by itself be the full
explanation — the true cause of v1's sign flip most likely lies elsewhere (the resource/energy
dims, dims 0–1, which this measurement didn't cover — or the delta rule's own accumulation
dynamics over many steps).

- **`ma001r-delta-raw-observation-2026-07-26.txt`** — the raw (pre-permeability-gate) observation
  variant (§14): new `OrganismTick::raw_observation` field, `Ma001rProbe::
  run_with_delta_rule_from_raw_observation`/`run_shuffled_with_delta_rule_from_raw_observation`.

**Result** (see the plan doc's §14): **4 of 5 gates pass** — direction_correct,
separates_from_equal_outcome, held_out_confirms (both contexts improve, unlike v2 where context A
got worse), and reversal_flips_and_holds are all true. Only `shuffled_collapses` still fails.
Substantially stronger than v2's gated-observation result — supports permeability attenuation
being a real, independent contributing factor beyond belief inertia.

- **`ma001r-delta-hyperparameter-sweep-2026-07-26.txt`** — 6 `DeltaRuleConfig` variants against the
  already-committed gated-observation mechanism (§14). New example
  `ma001r_delta_hyperparameter_sweep.rs`, no source changes.

**Result** (see the plan doc's §14): best configs (higher `eta`, with or without weight decay)
reach 4/5 gates, cross-validated against the already-recorded plan-doc numbers (config 1 exactly
reproduces the committed 0.1955 result). But `reversal_flips_and_holds` fails at **every** tested
hyperparameter setting, including the two highest learning rates — reversal-resistance looks
structural, not simply an undertuned learning rate.

This line is now at a well-diagnosed, multiply-triangulated stopping point (three independent
follow-up experiments, all adversarially re-verified) — see the plan doc's §14 for the full
synthesis and what remains open.

- **`ma001r-delta-methodology-check-FLAWED-FIRST-ATTEMPT-2026-07-26.txt`** — a first attempt at
  isolating why `shuffled_collapses` fails everywhere (always-Context-B control), found
  mechanistically degenerate *before* being reported (Context B's own social fields are all zero
  except `partner_present`, so 3 of 4 coefficients structurally cannot move, and the one that can
  cancels out symmetrically in `delta_predicted`). Preserved per this project's non-erasure
  precedent, not silently discarded.
- **`ma001r-delta-methodology-check-corrected-2026-07-26.txt`** (§15) — the corrected
  balanced-decorrelated control (period-4 block context schedule, zero true correlation, real
  varying field values). New `Ma001rProbe::
  run_with_delta_rule_from_raw_observation_balanced_decorrelated`.

**Result** (see the plan doc's §15): **ambiguous, partial support.** The balanced-decorrelated
control's post-training `delta_predicted` (0.1206) sits strictly between Equal-outcome's (0.0714)
and Shuffled's (0.1821) — matching neither. Part of Shuffled's elevated value over Equal-outcome
is attributable to alternating-target difficulty alone (partially supporting the hypothesis that
`shuffled_collapses`'s baseline is too strict), but Shuffled still shows real additional movement
beyond even this decorrelated-but-varying control (refuting it as the *full* explanation) — a
genuine, still-unexplained residual (~56% of the Equal-outcome-to-Shuffled gap), candidate cause
untested: per-tick-random assignment vs. a fixed periodic schedule.

- **`ma001r-delta-rng-bias-check-2026-07-26.txt`** (§16) — direct measurement of whether the
  shuffled control's xorshift64 low-order bit correlates with the deterministic outcome schedule
  for the seed used throughout this arc. **Ruled out**: Pearson correlation 0.017 (noise-level),
  run-length and period-2 checks both match a fair coin.
- **`ma001r-delta-shuffled-multiseed-2026-07-26.txt`** (§16) — the shuffled control run across 10
  independent RNG seeds.

**Result** (see the plan doc's §16): **resolved.** Mean post-training `delta_predicted` across 10
seeds = 0.1235 (std_dev 0.0283, range 0.0833–0.1821) — essentially matching the
balanced-decorrelated control's 0.1206 (within 2.4%). **Seed 1, used throughout this entire
research arc, was the maximum of all 10 seeds** — a genuine outlier, not representative. §15's
"unexplained residual" was single-seed finite-sample noise, not a systematic effect. The
alternating-target-difficulty hypothesis fully explains the gap; `shuffled_collapses`'s comparison
baseline (a constant-target control) is confirmed unfair for judging an alternating-target
mechanism. Consequence: the raw-observation variant's own 4/5 result (§14b) very plausibly becomes
5/5 once `shuffled_collapses` is re-baselined correctly — the mechanism was never falling short;
the gate was comparing against the wrong reference.
