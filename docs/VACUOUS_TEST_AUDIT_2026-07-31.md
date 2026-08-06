# Vacuous-test audit — 18 confirmed (2026-07-31)

**A vacuous test is one whose name promises a relational property, but whose assertions cannot
fail when that property is violated.** Not a flaky test, not a weak test — a test that provides
*zero* evidence for the thing it is named after, while appearing in the green count as if it did.

## Why this audit happened

Two independent instances surfaced in a single CI run on 2026-07-30, which is what made it look
like a class rather than an accident:

1. **`ArcChain::test_degradation_with_length`** (psych-bench) reads `chain_2_similarity` and
   `chain_4_similarity`, then asserts only that both `.is_finite()`. It never compares them. It
   passed while the underlying metric was **fully inverted** — a 3-step task scoring 0.0167
   against a 4-step task's 0.1500 — for as long as that defect has existed.
2. **`test_ne_phasic_no_effect_below_threshold`** (neuromodulators), a *negative control*, was
   confirmed vacuous: Michaelis-Menten clearance drives the level below threshold within ~4 cycles
   from any admissible start, so `high_exposure_cycles == 0` after 30 cycles held no matter what
   the threshold logic did.

A sweep was then run over the measurement-bearing crates. 20 candidates found, **18 confirmed**
after adversarial verification in which each verifier defaulted to `refuted=true` and was
instructed to refute if the comparison happened *anywhere* — via a helper, a computed delta
metric, or inside a loop.

## Why this matters more than a normal test gap

These are concentrated in exactly the places where the project's **scientific claims** live: Φ
measurement, model learning, drug-effect direction, and negative controls. A vacuous test is worse
than a missing one, because a missing test is visibly missing. This class inflates the green count
and creates false confidence precisely where the claims are hardest to check by eye.

The `ArcChain` case is the proof that this is not hypothetical: the vacuous test is *why* a
98.3% metric collapse with an impossible internal ordering went unnoticed.

## Two spot-checks, verified by hand rather than taken from the sweep

**`test_iit_partition_decreases_phi`** — the test computes the quantity its name is about, then
throws it away:

```rust
let _partition_total = phi_a.phi + phi_b.phi;   // underscore-prefixed: deliberately discarded
...
assert!(full_result.system_ei > phi_a.system_ei + phi_b.system_ei - 0.1, ...)
```

There is exactly **one** assertion in the whole test, and it is about `system_ei`, not `phi`. The
named property — that partitioning reduces Φ — is never checked at all.

**`test_ne_phasic_no_effect_below_threshold`** — the entire body:

```rust
let bath = NeuromodulatorBath::default(); // phasic = 0.0
assert!(bath.ne_phasic() < 0.3);
```

It asserts that its *input* is below threshold. It never measures an effect of any kind, so "no
effect" is not tested — it is assumed. (Now fixed, with a positive contrast arm.)

## Status

Two are already repaired: the neuromodulator negative control (positive contrast arm added,
`3487dbeec7`) and `ArcChain`, which now has real monotonicity and anti-correlation guards
(`822c070348`) — those guards **fail on purpose**, because the defect they encode is still open.

The remaining 16 are recorded, not fixed. Each needs a judgement about what the correct assertion
is, and several will likely fail once made real — which is the point. **Do not batch-fix these.**
A test that starts genuinely asserting its own name is a new signal, and the interesting outcome
is which ones go red.

## Method note

The sweep keyed on test names containing a relational word (`degrad`, `increas`, `decreas`,
`monoton`, `correlat`, `improv`, `above_chance`, `negative_control`, …) and then required a human
read of each body. Existence and sanity checks — `is_finite`, `is_some`, `!is_empty`, `> 0.0`,
does-not-panic — were classified as **not** relational. Tests that genuinely compare were left
alone even where the tolerance is generous; a loose test is a different problem from a vacuous one.

---

# Confirmed instances

### `crates/core/symthaea-core` (4)

**`test_iit_partition_decreases_phi`** — `crates/core/symthaea-core/src/consciousness_metrics/tests/core_tests.rs:703`
- Name promises: Phi of the full system exceeds the Phi of its partitions (partitioning destroys integration) -- per the doc comment, 'IIT Property: Phi decreases when system is partitioned'.
- Actually asserts: Computes full_result.phi, phi_a.phi, phi_b.phi; binds `let _partition_total = phi_a.phi + phi_b.phi;` (deliberately discarded via underscore); printlns all three phi values; then the SINGLE assertion is `full_result.system_ei > phi_a.system_ei + phi_b.system_ei - 0.1`.
- Slips past: Any regression in the MIP search or in Φ's composition across a real subsystem split passes unnoticed, because the sole assertion never reads `.phi` or `.mip_ei`. Concretely: (a) `find_true_mip` degrading to a non-minimal partition — it is a greedy/annealing search (`calculator.rs:640-685`) whose quality is never asserted here — inflates `mip_ei` and silently deflates every Φ in the crate, including the one that feeds `MotorSafetyLevel::from_phi` on all robotics platforms; (b) Φ becoming sub-additive across an explicit split (`full.phi <= phi_a.phi + phi_b.phi`), i.e. the exact IIT property named in the doc comment being inverted, is undetectable — that comparison is computed into `_partition_total` and thrown away; (c) since `system_ei` is a bare sum of clamped-non-negative pairwise MI over a superset of pairs, the assertion would still pass even if `compute_true_phi` returned `phi: 0.0` for every input. The only mutations this assertion can detect are ones unrelated to its stated purpose: removing the `.max(0.0)` clamp on MI, or changing `effective_information` from a sum to a mean.

**`test_pyphi_phi_scales_with_size`** — `crates/core/symthaea-core/src/consciousness_metrics/tests/pyphi_tests.rs:143`
- Name promises: Phi increases with system size (n=2 -> n=3 -> n=4) for integrated systems.
- Actually asserts: Computes phi_2, phi_3, phi_4 (full results for n=2,3,4); printlns all three `.phi` and `.system_ei` values; then the SINGLE assertion is `phi_4.system_ei > phi_2.system_ei`.
- Slips past: Any regression in the MIP search — the only machinery that distinguishes Φ from raw effective information — slips past silently. `Φ = (system_ei - mip_ei).max(0.0)`, and the test constrains only `system_ei`. If `find_true_mip`/`exhaustive_mip_search`/`partition_effective_information` degenerated (returning the trivial partition, flipping the min/max convention, or enumerating no partitions), `mip_ei` → `system_ei` and Φ would read 0.0 at every system size. `system_ei` is untouched by that bug, so the sole assertion still passes and the test reports green while the crate's headline metric is identically zero. Secondarily, the whole n=3 rung is computed and printed but never asserted, so the "n=2 → n=3 → n=4" monotonic chain the test name promises is unguarded at its only interior point — a non-monotonic Φ (or EI) dip at n=3 is invisible. Because `test_iit_size_effect_on_phi` (core_tests.rs:782) repeats the same EI-for-Φ substitution, the Φ-vs-size property has zero test coverage in symthaea-core, not merely weak coverage.

**`test_iit_size_effect_on_phi`** — `crates/core/symthaea-core/src/consciousness_metrics/tests/core_tests.rs:783`
- Name promises: System size has an effect on Phi -- doc comment: 'More components can increase Phi ... Phi generally increases with system size for integrated systems'.
- Actually asserts: Builds integrated systems of size 3 and 5, computes phi_small and phi_medium, printlns both `.system_ei` and `.phi`; then the SINGLE assertion is `phi_medium.system_ei > phi_small.system_ei`.
- Slips past: Any regression that flattens or inverts Φ's dependence on system size passes this test untouched, because it only reads `system_ei`, which is computed by a separate function from everything Φ adds on top. Concretely: change `find_true_mip` to return the trivial whole-system partition (part_a = all indices, part_b = empty) — a plausible off-by-one or early-return bug in `exhaustive_mip_search`'s bitmask loop (`for mask in 1..(1 << n) - 1`). Then mip_ei == system_ei for every n, `phi = (system_ei - mip_ei).max(0.0)` == 0.0 for both the size-3 and size-5 system, and Φ has exactly zero size effect — the precise property this test is named for. `system_ei` is unchanged (still 3 pairs vs 10 pairs of MI), so the assertion still passes and the test stays green. The same applies to a regression where the `.max(0.0)` clamp fires for all n, or where mip_ei drifts to track system_ei proportionally. Blast radius is the `TruePhiCalculator` module and its downstream consumers (`physics/consciousness_bridge.rs` uses `.system_ei` per emergence level; `bounds.rs:58` validates `phi <= system_ei`) — note the live robotics safety path uses `SpectralMIPFinder`, not this calculator, so this is a correctness rather than a safety-gate hole. Sibling `test_pyphi_phi_scales_with_size` would not catch it either; nor would `test_validation_suite`, which discards the one `passed` boolean that checks Φ monotonicity.

**`test_bound_vs_bundle_different_phi`** — `crates/core/symthaea-core/src/consciousness_metrics/tests/core_tests.rs:212`
- Name promises: Binding and bundling the same components yield DIFFERENT Phi (i.e. the Phi measure is sensitive to structural difference).
- Actually asserts: Computes phi_bundled and phi_bound. Asserts: phi_bundled.phi.is_finite(); phi_bound.phi.is_finite(); phi_bundled.phi >= 0.0; phi_bound.phi >= 0.0; phi_bundled.system_ei.is_finite(); phi_bound.system_ei.is_finite(). Then a comment 'They should have different Phi values (bind creates orthogonal structure) / This test verifies that our entropy measure is sensitive to structural differences' followed by a println of both -- and the function ends.
- Slips past: Any regression that destroys the Phi measure's sensitivity to structure passes this test silently. Three concrete ones:

1. MOST PLAUSIBLE — Phi pinned to zero for all 2-component systems. If find_true_mip's n==2 branch (calculator.rs:178-186) were changed to return `system_ei` instead of the singleton-partition EI (an easy "fix" for someone who finds mip_ei==0.0 suspicious), phi would become exactly 0.0 for EVERY 2-element system, permanently. All six assertions still pass (0.0 is finite and >= 0.0).

2. Estimator collapse. If the binned entropy/MI estimator lost discrimination (e.g. num_bins driven to 1, or mutual_information returning a constant), MI == 0 for both arms, so phi == 0 for both. All six assertions pass. This is the exact failure the test's own comment claims to guard ("sensitive to structural differences").

3. LIVE RISK — bind/bundle collapsing into each other. If ContinuousHV::bind were refactored to delegate to bundle (or the two unified during an HDC algebra cleanup), the two arms become bit-for-bit identical and a test literally named `different_phi` passes. This is not hypothetical: MASTER_ROADMAP records an active HDC binding-algebra workstream that already corrected the self-inverse claim at the source and lists Tier-A migration follow-ups touching cross_modal_binding.rs and symthaea-fep/markov_blanket.rs. Bind semantics are being actively changed right now, and this test names that exact regression class while guarding none of it.

Blast-radius bound (keeps this honest): TruePhiCalculator is NOT the production robotics-safety Phi. Per CORE_SUBSTRATE.md the live consciousness_level -> MotorSafetyLevel::from_phi path uses SpectralMIPFinder. TruePhiCalculator feeds IIT4, temporal, parallel, approximate, and the physics emergence_chain/consciousness_bridge paths. So this is a real research/measurement-integrity gap, not a motor-safety hole — it would corrupt Phi-based experimental results rather than endanger a robot.


### `crates/core/symthaea-fep` (2)

**`test_model_improves_prediction_accuracy`** — `crates/core/symthaea-fep/src/tests.rs:618`
- Name promises: After training, the model's prediction error should be LOWER than before training (accuracy improves).
- Actually asserts: 1) initial_error.is_finite() && initial_error >= 0.0; 2) final_error.is_finite() && final_error >= 0.0; 3) final_error < 10.0 (a fixed upper bound); 4) agent.stats.td_updates > 0. It reads BOTH initial_error and final_error into locals and never compares them to each other.
- Slips past: Any regression that makes FEP learning neutral or actively anti-learning slips through undetected. Concretely: a sign flip or wrong-direction update in TemporalDifferenceLearner::update_model (td_learning.rs), or a broken gradient in ActiveInferenceAgent::update_belief (agent.rs:243-290), such that 50 cycles of training on a constant observation leaves prediction error unchanged or HIGHER than the untrained model. All four assertions still pass:

- Assertions 1 and 2 (finite && >= 0.0) hold by construction -- the quantity is an L2 norm (sqrt of a sum of squares) of finite values, so it is non-negative unless a NaN is introduced upstream.
- Assertion 3 (final_error < 10.0) is unreachable-loose: belief.mean is clamped to [0,1] in update_belief and the observation values here are 0.5-0.7, so the 4-dimensional L2 norm has a plausible ceiling around 2.0. The bound sits ~5x above the worst physically reachable value; no realistic degradation can trip it.
- Assertion 4 (td_updates > 0) only proves the TD branch was entered at least once (it fires ~50 times here), not that any update improved the model. It would only fail if TD learning were entirely disconnected.

Net effect: the test's real content is a smoke test -- "the agent runs 51 perceive cycles without producing NaN, and the TD code path executes." That has some value, but it is strictly weaker than its name implies, and as a result the entire symthaea-fep crate has NO test guarding that its learning actually reduces prediction error. Given symthaea-fep is described in CORE_SUBSTRATE.md as the predictive engine central to the cognitive loop, the core learning claim of the crate is unguarded.

**`test_transition_matrix_converges_to_true_dynamics`** — `crates/core/symthaea-fep/src/tests.rs:563`
- Name promises: After training, the learned transition matrix approaches the known true dynamics (action 0 advances state index by 1 mod 4; action 1 holds it).
- Actually asserts: Runs 50 episodes against a deterministic simulated environment, then asserts only: `agent.stats.td_updates > 0` and `td_stats.avg_prediction_accuracy > 0.0`. The comment above them reads 'Check that model has learned something'.
- Slips past: Any regression that corrupts WHAT is learned while leaving the TD path live passes silently. Concretely: (a) `update_model` writing to the wrong `action_idx` (td_learning.rs:515) so action 0's shift-by-1 dynamics and action 1's identity dynamics are swapped or merged into one matrix; (b) a sign flip in `let gradient = observed_prob - model_prob` (td_learning.rs:523), which drives the matrix AWAY from observed transitions; (c) the row-normalisation loop (td_learning.rs:544-552) flattening learned structure back toward uniform; (d) `trace_scale` collapsing to 0 (td_learning.rs:526-536) so the matrix stays frozen at its random `GenerativeModel::new()` initialisation — the exact failure mode symthaea-futures-ensemble/src/ecological.rs:23 documents as a real hazard it had to guard against separately. In every case `td_updates > 0` still holds and `1/(1+prediction_error)` is still strictly positive, so both assertions pass. The test cannot distinguish "learned the true dynamics" from "learned nothing" from "learned the exact inverse". This matters beyond hygiene because `GenerativeModel::transition_matrices` is the FEP substrate's forward model, consumed by symthaea-alife (ma001/ma001l/ma001r), symthaea-culinary/palate.rs, and symthaea-futures-ensemble — and this is the only test in the repo whose name claims the learned model is correct rather than merely non-static.


### `crates/domains/symthaea-alife` (2)

**`delta_rule_moves_the_transfer_coefficient_toward_the_target_relationship`** — `crates/domains/symthaea-alife/src/ma001l.rs:405`
- Name promises: After training on the Bound stream, the learned Transfer transition coefficient moves TOWARD the target context->outcome relationship - i.e. the post-training model is measurably closer to the target relationship than the untrained model was.
- Actually asserts: Exactly one assertion in the whole body: `assert!(error.is_finite());`, where `error = held_out_energy_error(&model, &held_out)` after 2000 `learner.update(...)` calls and a 200-tuple held-out stream. There is no other assert. The body never reads `model.transition_matrices`, never snapshots the model before training, and never compares `error` against any baseline (untrained model, shuffled arm, or numeric threshold).
- Slips past: `assert!(error.is_finite())` can only fail on NaN/inf poisoning, and even that is nearly unreachable: every coefficient written by `update()` passes through `.clamp(-5.0, 5.0)` (ma001l.rs:279-280), so non-finite values can only arrive via NaN propagating in from `predict_next_state`. Every substantive regression in `DeltaRuleLearner::update` slips past silently, because a fresh or a wrongly-trained model both yield a finite held-out error (~0.35 for the untouched model against outcomes 0.9/0.2):

1. SIGN FLIP — computing `predicted - actual` instead of `actual - predicted` (line 269). The rule then learns AWAY from the target: coefficients move to the clip bound in the wrong direction, the counterfactual A>B ordering inverts, and gates C and D would both fail. Error stays finite. Test passes. This is the exact regression the test's name promises to catch.
2. TRANSPOSED INDEX — writing `matrix[i][j]` instead of `matrix[j][i]` (line 281). The doc comment at lines 253-257 explicitly flags this indexing convention as the thing to get right, since it must match `predict_next_state`'s `transition[j][i] * state.mean[j]`. A transpose destroys the context->outcome mapping entirely. Finite. Passes.
3. NO LEARNING AT ALL — `eta` silently forced to 0.0, an early return in `update()`, or `decay` swamping `eta` so the model is pinned at `initial_transition_matrices`. Finite. Passes.
4. CONTEXT COLLAPSE — `social_dims_for` (line 53) returning a constant, so the four social dims carry no context signal. This nulls the entire contingency MA-001L exists to probe. Finite. Passes.

Blast radius is larger than one module: `DeltaRuleLearner` is consumed by `src/ma001.rs`, `src/ma001r.rs`, and 8 example drivers (`ma001r_delta_run`, `ma001r_delta_hyperparameter_sweep`, `ma001r_delta_shuffled_multiseed`, `ma001a_delta_*`, etc.). Because the only real gates live in a never-executed example binary, there is currently ZERO automated regression protection on the learning rule underpinning the whole MA-001*-delta result line — a sign or index regression would silently invalidate every downstream finding while the suite stayed fully green.

**`neither_arm_leaves_the_model_at_its_initial_values`** — `crates/domains/symthaea-alife/src/ma001l.rs:421`
- Name promises: For BOTH experimental arms (Bound and Shuffled schedules), the model's state after the arm runs differs from its initial values - i.e. post-arm state != initial state, so neither arm is a silent no-op.
- Actually asserts: Body is three lines: `let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);`, `let fresh = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);`, `assert_eq!(model.transition_matrices, fresh.transition_matrices);`. That is the only assertion.
- Slips past: Any regression that makes the "Neither (no-learning control)" arm actually learn slips past silently — e.g. adding a stray `model.learn(...)` into `apply_tuple`'s `LearnerKind::Neither => {}` branch, or replacing the exhaustive match with a `_ =>` catch-all that routes Neither into a learner. That contaminates the experiment's own baseline: Gate B (`bound_error < neither_error`) and Gate E (drift measured against the `untouched` snapshot) both depend on Neither being a true no-op, so a contaminated control would corrupt the headline MA-001L result — "the delta rule reconstructs the conditional transition better than no learning at all" — in the direction that makes it look supported. Under the alternate "both arms" reading, a silently no-op `DeltaRuleLearner::update` (eta defaulted to 0.0, or clip_bound collapsed to 0) would leave both Bound and Shuffled models at their initial values, and neither this test nor the `error.is_finite()` sibling would notice — the learner would appear to run while learning nothing.


### `crates/domains/symthaea-broca` (1)

**`test_directional_loss_decreases_pe`** — `crates/domains/symthaea-broca/src/temporal_projection.rs:2554`
- Name promises: Training with the directional loss DECREASES roundtrip prediction error (final_pe < initial_pe).
- Actually asserts: assert!(final_pe.is_finite(), "PE should be finite after directional training"); assert!((final_pe - initial_pe).abs() > 1e-6, "Directional loss should modify the projection"). That is the complete assertion set. Both initial_pe (line 2559) and final_pe (line 2566) are computed via tp.roundtrip_pe(&thought), before and after 20 compute_directional_gradients/apply_gradients steps.
- Slips past: The named relation is ALREADY violated in main and the test is green — so the regression has in effect already slipped past. Concretely, anything that keeps the perturbation finite and |Δ| > 1e-6 passes, including sign-flipping the cosine gradient at temporal_projection.rs:1219 (`ssm_grad[j] = -(target_ssm[j]*inv_ab - cos_sim*ssm_vec[j]*inv_a2)` → `+(...)`), which converts the loss from cosine-alignment into cosine-ANTI-alignment. That inversion makes the projection actively worse, still perturbs weights, and the test stays green. Same for a wrong learning-rate sign, a mis-indexed target, or dropping the LayerNorm backprop term. The only regression this test can catch is one that makes the loss a complete no-op (Δ == 0) or produces NaN/Inf. Practical blast radius is limited today because `temporal_directional_loss` is never read, so nothing in production calls this loss — but the `--directional-loss` CLI flag is documented in help text (broca_projection_train.rs:1127), so the first person to wire it up will be relying on a test that has never verified its central claim, in a module that default-feature CI never even compiles.


### `crates/domains/symthaea-broca/tests/quality_validation.rs` (1)

**`test_channel_sensitivity_gradient`** — `crates/domains/symthaea-broca/tests/quality_validation.rs:487`
- Name promises: Varying the valence channel across [0.0, 0.5, 1.0] produces a graded/differentiated response — i.e. distinct outputs across the three levels.
- Actually asserts: Inside the generation loop: assert!(result.final_coherence.is_finite(), "Coherence must be finite for valence={valence}"). Then `let mut distinct_count = 1;` followed by a loop that only ever does `distinct_count += 1;`, then the single relational assertion: assert!(distinct_count >= 1, "At least 1 of 3 valence levels should produce output. Got {distinct_count} distinct sequences.").
- Slips past: A regression in which the valence channel stops influencing generated text passes silently. Concretely: the ThoughtEncoder dropping or zeroing channel 9, a normalization change that collapses [-1,1] to a constant, the emotional-modulation path being disconnected, or the generator ceasing to condition on the emotion channels of thought_hv. In every one of those cases all three valence levels return identical token_ids, `distinct_count` stays 1, and `assert!(distinct_count >= 1)` passes. Even the degenerate case of all three generations returning empty token_ids passes, despite the assert message claiming to check that output was produced. This is not hypothetical dead weight: channel 9 carries live vision prediction-error in production (src/cognitive_loop.rs:162) and the test's own config enables emotional modulation, so the encoder→generation emotion path this test names is exactly the path that is unguarded. Compounding it, the nearest sibling coverage cannot backstop the loss — src/encoder.rs's two emotion tests vary valence, arousal and warmth together, so they remain green when valence specifically is lost, and tests/quality_validation.rs:1097 computes a difference flag and then never asserts on it.


### `crates/domains/symthaea-neuromodulators` (3)

**`test_clinical_impact_tacrolimus_most_affected`** — `crates/domains/symthaea-neuromodulators/src/pgx_health_equity.rs:2091`
- Name promises: For tacrolimus/CYP3A5, a specific ancestry group is the MOST affected — a superlative/ranking claim over the per-ancestry at-risk counts.
- Actually asserts: assert!(estimate.total_at_risk > 0, "Should have at-risk patients for tacrolimus/CYP3A5"). That is the only assertion in the body.
- Slips past: Any regression in the argmax that selects `most_affected` slips through completely, as does any drift in the inputs that decide it. Concretely, all of these keep the test green:

1. Inverting the argmax comparison (`if at_risk_count > max_risk_count` -> `<`). Since `max_risk_count` starts at 0 and `most_affected` is initialized to `AncestryGroup::European`, nothing ever satisfies `< 0`, so `most_affected` silently pins to European forever — the exact opposite of the module's stated equity finding.

2. Data drift flipping the winner. The Mixed/African margin is only ~2.2% (43.42M vs 42.48M). Changing `POP_MIXED` from 72.0 to anything below ~70.4, or nudging the Mixed CYP3A5*3 frequency from 0.60 to ~0.68 — a value the source itself annotates as a soft "Weighted average" — flips `most_affected` from Mixed to African. No test in the repo pins either constant.

3. Deleting or renaming the entire CYP3A5 allele-frequency table. The degenerate empty path yields at_risk_fraction = 1.0 for every ancestry, so total_at_risk ~331M and the assertion still passes.

The blast radius is larger than one field: `most_affected` and `max_risk_count` are interpolated into the user-facing `risk_description` string ("Most affected: {:?} ({:.1}M at-risk individuals, {:.1}% of that population)"), and `ClinicalImpactEstimate` derives Serialize/Deserialize, so a wrong value propagates into serialized output and any downstream consumer. For a health-equity module whose entire purpose is identifying which ancestry group is underserved, the superlative is the load-bearing claim, and it is unasserted everywhere in the repo.

**`anti_inflammatory_cytokines_boost_recovery`** — `crates/domains/symthaea-neuromodulators/src/pni_coupling.rs:502`
- Name promises: Anti-inflammatory cytokines (IL-10) BOOST the recovery rate — a causal/comparative link between cytokine level and recovery_rate.
- Actually asserts: After a single pni.update_from_sleep(true, 1.0): assert!(pni.state.cytokines.il10 > 0.3, "sleep should boost IL-10, got {}"); assert!(pni.state.recovery_rate > 0.0, "should have positive recovery rate during sleep").
- Slips past: Any regression in the cytokine→recovery coupling passes silently, including an outright SIGN INVERSION. Concretely, changing line 282 to `recovery_rate = (0.1 * quality * (1.0 - il10)).clamp(0.0, 1.0)` — so that HIGHER anti-inflammatory tone yields SLOWER recovery, the exact opposite of the test's name — gives recovery_rate = 0.065 > 0.0 and il10 = 0.35 > 0.3. Both assertions pass; test green.

Second, the cortisol pathway is entirely unguarded: `update_from_stress` (line 246) suppresses IL-10 via `il10 -= CORTISOL_SUPPRESSION_RATE * cortisol`, yet nothing anywhere asserts that recovery falls with it. Chronic stress can crush IL-10 to 0.0 while recovery_rate stays pinned at its full sleep value — which is in fact the CURRENT behavior — and no test in the repo notices.

Third and most consequential: the named relation is absent from the implementation altogether, and the field's own doc (line 161, "modulated by sleep and anti-inflammatory tone") claims it. A reader trusting the test suite would believe the PNI model couples anti-inflammatory tone to recovery. It does not. Degenerate values also pass (il10 = 0.301, recovery_rate = 1e-38).

Mitigating: recovery_rate has zero production consumers — one write site, one read site (this test's own assertion), so the live blast radius today is nil. The risk is that the false green ratifies the doc'd-but-unimplemented coupling for whoever wires it up next.

**`test_ne_phasic_no_effect_below_threshold`** — `crates/domains/symthaea-neuromodulators/src/lib.rs:2922`
- Name promises: Phasic noradrenaline below the 0.3 threshold produces NO attentional or exploratory effect (a negative-control claim relating sub-threshold NE to downstream attention/exploration).
- Actually asserts: assert!(bath.ne_phasic() < 0.3). That is the entire body apart from a bare comment `// No attention or exploration effect`. No other statement executes.
- Slips past: Any regression that removes or weakens the sub-threshold gate on phasic-NE attentional reorienting passes silently. Concretely: deleting the `else { 0.0 }` arm, flipping `>` to `>=`, lowering the comparison to `if ne_ph > 0.0`, or dropping the gate entirely at cycle_neuromod_phase.rs:105 — all still green. The consequence is that a resting/default bath (phasic NE = 0.0, i.e. no surprise signal at all) would multiply `behavior.adaptive_behavior.attention_sensitivity` and call `adjust_exploration("ne_phasic_reorient", ...)` on every cycle, producing permanent baseline attentional reorienting and monotonic exploration-urge drift untied to any input. Equally, breaking the in-crate gate at lib.rs:513 would sensitize Beta receptors (ne_subtypes.inhibitory) forever at rest. Because the only other negative control (cycle_neuromod_phase.rs:405, `assert!(boost >= 0.0)`) is vacuous in the same way and phase_coverage.rs:151 only checks the above-threshold direction, NO test in either crate fails on any of these. This is precisely the class of silent always-on-neuromodulation bug the roadmap is already chasing elsewhere (the load-dependent consciousness_level result and the dopamine_effective == 0.0000 reward-wiring investigation).


### `crates/domains/symthaea-psych-bench` (5)

**`test_sensitivity_prioritized_over_specificity`** — `crates/domains/symthaea-psych-bench/src/benchmarks/clinical/crisis_detection.rs:1152`
- Name promises: crisis_sensitivity >= crisis_specificity (detection threshold deliberately set low to favour sensitivity over specificity)
- Actually asserts: Nothing. The entire body is: read `crisis_sensitivity` and `crisis_specificity` means, then `if sens < spec { eprintln!("WARNING: sensitivity ... < specificity ...") }`. There is not a single assert! / assert_eq! in the function.
- Slips past: An inversion of the benchmark's stated clinical priority passes the whole suite silently. Concretely: raise `threshold` at crisis_detection.rs:296 from 0.42 to ~0.9 (or weaken the crisis-keyword set, or regress the HDC similarity encoder). Sensitivity falls toward the keyword-only floor while specificity climbs toward 1.0 — i.e. real missed crises (false negatives) on the Suicidal/SelfHarm/Psychosis/DomesticViolence scenarios, the exact failure mode the module's header (line 27) calls "far more" costly. As long as sensitivity stays above 0.5, every test in the module still passes: the range tests are satisfied, `sens > 0.5` is satisfied, f_beta_2 stays within 0..1 (and in fact RISES, since fewer false positives raise precision), and this test merely takes its `if` branch and prints to a captured stderr that cargo never displays for a passing test. Nothing in CI turns red. The confusion-matrix asymmetry that the benchmark exists to measure would be silently reversed.

**`test_moral_oxytocin_positive_correlation`** — `crates/domains/symthaea-psych-bench/src/benchmarks/neuromod/moral_oxytocin.rs:176`
- Name promises: moral_oxytocin_r is POSITIVE, i.e. moral evaluation and oxytocin move together
- Actually asserts: `assert!(r.is_finite(), "Correlation should be finite, got {r}")` -- that is the only assertion.
- Slips past: The benchmark's headline exported metric `moral_oxytocin_r` could drop to exactly 0.0 or invert to strongly negative while every test in the file stays green. Concrete regressions that slip past: (a) the observation loop feeding `make_inputs(None)` instead of `Some(moral)` (a plausible copy-paste from the warmup loop 6 lines above) — oxytocin then has near-zero variance, pearson_r's `denom < 1e-12` guard returns exactly 0.0, which is finite and passes; (b) recording the trace before `bath.update()` rather than after, which under the 4-phase alternating stimulus introduces a one-cycle lag that can invert the sign; (c) tracing the wrong transmitter, e.g. `bath.dopamine.effective()` — dopamine is driven in the very same moral branch at lib.rs:403, so this is an easy slip and stays finite; (d) a sign error in pearson_r's covariance accumulation, giving r = -0.6775, finite, passes; (e) widening the moral deadzone from the current +/-0.3 to +/-0.6, which silently kills the 0.5 stimulus and halves the benchmark's dynamic range while the neuromodulator unit tests (which use moral = 0.8/-0.8) stay green. This is worse than ordinary test hygiene because psych-bench results feed reported figures (examples/psych_bench_paper_data.rs), so a silently zeroed or inverted moral-to-oxytocin correlation would be published as a Zak-2012-validating result with no test failing. Severity is bounded by the fact that the underlying bath mechanism itself is separately protected by the two directional tests in symthaea-neuromodulators, so a regression in the bath (as opposed to in this benchmark harness) would still be caught elsewhere.

**`test_sedative_decreases_consciousness`** — `crates/domains/symthaea-psych-bench/src/benchmarks/neuromod/consciousness_pharmacology.rs:396`
- Name promises: a sedative (adenosine) drives the consciousness proxy BELOW its baseline
- Actually asserts: `assert!(trace.mean().is_finite(), "Sedative should produce finite consciousness values")` -- the only assertion.
- Slips past: The test cannot fail for any violation of its named relation, and the relation is already violated in production: the adenosine condition's consciousness trace is bit-for-bit identical to running no injection at all (verified against the real NeuromodulatorBath), because adenosine has no code path into `sht_2a_signal()`/`gaba_a_signal()`, the only two inputs to `consciousness_proxy()`. Regressions that slip past: (1) the entire adenosine→consciousness coupling being absent or removed — the exact state today; (2) `run_condition` ceasing to inject at all, or `inject()` silently dropping the target (the `_ => None` arm in `NeuromodulatorBath::inject` swallows unrecognised names, so renaming the "aden" alias makes it a silent no-op) or the `active_injections.len() >= 4` cap silently discarding it; (3) the sign of `consciousness_proxy()` inverting so a sedative *raises* the proxy — the formula is documented as replicating consciousness_engine.rs:381-389, so a sign flip there would go undetected here; (4) the proxy collapsing to a constant. All four leave `trace.mean().is_finite()` true. Net effect: the benchmark publishes four `sedative_proxy_*` metrics that are indistinguishable from baseline drift, under a test name and doc-comment asserting "consciousness DROPS".

**`test_reversal_learning_initial_faster_than_reversal`** — `crates/domains/symthaea-psych-bench/src/benchmarks/cogbench/reversal_learning.rs:416`
- Name promises: initial acquisition is faster than reversal, i.e. reversal_cost > 0
- Actually asserts: `assert!(cost.is_finite(), "reversal cost should be finite")` -- the only assertion.
- Slips past: Any regression that destroys the reversal cost — the single quantity that defines this paradigm — slips through silently. Concretely: a change to the asymmetric learning rates (learning_rate 0.35 / loss_learning_rate 0.90, lines 50-51), the change-point surprise threshold (`consecutive_errors >= 2`, line 150), the post_reversal_countdown fast-learning window (lines 153/221), or the softmax temperature (line 87) that makes the agent adapt to a reversal as fast as, or FASTER than, it acquires the original contingency would leave reversal_cost at 0 or negative and the test still passes. A sign flip to cost = -1.0 (reversal learned faster than initial acquisition — the exact opposite of the documented human finding and of the test's own name) passes both this assertion and the full_battery regression guard, whose trip point for this metric is -1.088. The only way to fail the assertion is NaN/inf, which requires avg_reversal_trials or initial to be non-finite — impossible here since both are bounded u32-derived counts with max_trials=200 fallbacks (lines 238, 240). Practical impact: this is a construct-validity guard for a published psych-bench paradigm; the measured cost is currently ~0.16 trials against human norms implying ~+4 to +6, and one committed baseline (v0.5.1) records exactly 0.0 — the failure this test is named for has already happened once in-repo and went unreported. One honest caveat: the specific degenerate case where the criterion machinery breaks entirely so no reversal ever completes IS caught, but by a different sibling (test_reversal_learning_completes_reversals, line 386), not by this one.

**`test_psi_consciousness_mod_correlation`** — `crates/domains/symthaea-psych-bench/src/benchmarks/neuromod/consciousness_feedback.rs:164`
- Name promises: psi and the consciousness modulation signal are correlated (that a relationship exists at all)
- Actually asserts: `assert!(r.is_finite(), "Correlation should be finite, got {r}")` -- the only assertion.
- Slips past: Verified by direct mutation, not inference. (1) Deleting the entire consciousness→neuromodulator feedback pathway — `symthaea-neuromodulators/src/lib.rs:391-395`, the exact code this file's doc comment claims to validate end-to-end — changes the asserted value from -0.01377422 to -0.01377377 and the test still passes. (2) Making `consciousness_modulation()` return a hardcoded constant passes, because `pearson_r`'s `if denom < 1e-12 { 0.0 }` guard converts zero-variance into a *finite* 0.0 — total signal collapse reads green. (3) Inverting or reweighting the ACh/NE contribution in `consciousness_modulation()` passes. The only failure mode `is_finite()` can catch is NaN propagation, which sibling `test_consciousness_feedback_all_finite` already asserts more directly, so this test's marginal coverage is exactly zero. Compounding: the benchmark feeds `psi_consciousness_mod_r` into paper-data CSV (`examples/psych_bench_paper_data.rs:714`), so a metric that is ~0 and causally disconnected from psi is exported as published evidence with no test able to notice.

