# Integrated Prequential PIT Audits in Symbolic-Law Discovery v0.1

**Status:** exploratory integration in the Symthaea physics-bridge simulator; exact-head Rust compile/test/format status is pending. It is a wiring and measurement-cost pilot, not evidence that the monitoring-enabled policy beats nominal EIG or state of the art.

## 1. Scope

This follow-on connects `PrequentialPitMartingale` to the existing active symbolic-law discovery loop in `integrated_discovery.rs`. It is intentionally narrower than a production scientific monitor: the simulator has a declared heteroscedastic Gaussian noise model and can therefore issue a conditional predictive distribution for a provisional singleton.

New selector path: `InformationGainWithPrequentialPitAudits`.

- While multiple hypotheses remain, select experiments by the existing nominal EIG-per-cost selector.
- Once exactly one provisional hypothesis remains, do **not** accept it immediately. Select the next admissible unused experiment randomly from the predeclared pool and score the result as a validation audit.
- Before each audit measurement, evaluate the provisional candidate and its predictive scale
  (\sigma(x)=\text{NOISE\_SIGMA\_FRACTION}\,(1+|\hat y(x)|)). After observing (y), compute the simulator's standard-normal PIT approximation
  (u=\Phi((y-\hat y(x))/\sigma(x))).
- Pass the PIT plus four bounded, pre-outcome context channels (normalized selected (m,v,r,t) coordinates) into the fixed-weight PIT martingale.
- Stop the provisional singleton run on a monitor alarm. The evaluator records detection separately from hidden truth; the detector receives only candidate predictions, observations, and the predeclared query contexts.

The simulator rejects the PIT-audit policy if CfC smoothing is enabled or adaptive selection is disabled. Smoothing changes the measurement distribution; treating its state estimate as one Gaussian draw would break the declared predictive-CDF contract. An unsupported configuration fails closed instead of silently reusing the PIT guarantee.

## 2. Structured refinement after a PIT alarm

The follow-up branch `research/residual-guided-typed-refinement-v0-1` extends bounded reseeding with three explicit, unit-checked correction templates around the rejected provisional model:

- **Additive correction:** `base + delta`, where `delta` has ENERGY dimensions.
- **Multiplicative correction:** `base * (1 + factor)`, where `factor` is dimensionless.
- **Interaction correction:** `base + x * delta_x`, where the generated `delta_x` has dimensions ENERGY divided by the dimensions of `x`.

Proposals are now represented by a unit-checked correction basis plus a scalar coefficient fitted by weighted least squares on previously observed measurements. Additive corrections use an ENERGY-dimensional basis; multiplicative corrections use the base expression times a dimensionless factor; interaction corrections multiply a typed term by a variable with compensating dimensions. The fitted candidate is scored using normalized squared residuals plus a small, fixed expression-complexity penalty. Structured proposals are retained only if their empirical score improves against the rejected candidate. The coefficient is bounded to [-10, 10] by a predeclared rule. This is still candidate ranking, not validation or a statistical guarantee: proposals can overfit accumulated observations, so post-alarm claims must still use fresh unused measurements.

All previously alarmed candidates remain excluded. The search has a fixed generation-attempt cap, accepts at most four new candidates per restart, allows at most two restarts, and returns terminal unresolved status if it cannot construct a usable candidate set. The proposal routine has no access to the evaluator's discrepancy amplitude or ground-truth expression. New tests check dimensional validity for all three correction families and verify that a structured correction can improve empirical fit on a controlled residual fixture. A deterministic, evaluator-only holdout design is generated from a separate RNG stream and used only after the policy terminates to test whether the final singleton numerically matches the reference expression. The points are sampled from the same input domain as discovery, so this is a leakage-resistant same-domain equivalence score—not extrapolation evidence or real-world validation. The score is emitted separately for clean and misspecified runs and never enters candidate generation, selection, stopping, or monitoring. The newest follow-up adds a second evaluator-only split whose every variable is sampled strictly below or above the discovery interval. It uses a bounded attempt count, rejects non-finite/extreme reference evaluations, records how many usable points were obtained, and reports OOD-equivalence separately for clean and misspecified worlds. Clean-world success now requires both same-domain and OOD agreement. This deliberately makes the success definition stricter; the small existing synthetic pilot is not evidence that this OOD gate is well powered, and the new check remains only numerical agreement on the declared extrapolation design.

## 3. Truth-absent recovery track

The prior integrated simulator is an **oracle-present candidate-discrimination** task: it seeds the true expression into the initial hypothesis pool and also constructs near-miss candidates from that expression. This is valid for measuring experiment-selection quality when the correct candidate is supplied, but it is not evidence of open-ended law discovery. The two results must not be collapsed into one success rate. The truth-absent pilot reports unresolved outcomes separately from false singletons, counts a recovered singleton as false confidence if it fails either same-domain or OOD equivalence, and records measurement cost and proposal counts for both the stop/unresolved baseline and recovery-enabled policy.

The new truth-absent pilot uses an independently seeded generator to create every initial candidate from the declared grammar and variable dimensions. The candidate generator receives neither the reference expression nor ground-truth values and does not build near-misses by modifying the reference. The experiment design is generated before the reference; the task generator only admits a reference law that is finite and sane on those predeclared measurement points. After the policy has stopped, evaluator-only checks report whether the initial candidate set accidentally contained a numerically equivalent law.

When all current candidates are eliminated, the policy may enter bounded recovery: choose a base expression using only historical residual fit, archive the entire falsified pool, generate fitted typed correction proposals, and resume from the novel candidates. The reference and evaluator holdouts are not visible to this branch. Disabling recovery provides the stop/unresolved comparator. We report initial reference-equivalent pool hits, recovery attempts, generated candidates, same-domain agreement, OOD agreement, and full success separately.

This is a smoke/diagnostic benchmark, not yet a confirmatory measure of open-ended discovery. A formula absent from the initial pool may still be hard to recover; if so, the correct outcome is unresolved, not a forced singleton. A truth-absent run can also fail because the selected physical law is not evaluable at a measurement point or because the bounded generator finds no useful candidate. These outcomes are retained as failures and must not be silently dropped. The task generator uses a predeclared finite training design and validates reference admissibility against it; that design is fixed independently of the reference law. Reference equivalence is scored only after the policy terminates.

This remains short-term **single-coefficient structured correction search**, not a complete mechanism-discovery system. It does not jointly estimate multiple physical parameters, introduce new operator libraries, use boosting, or provide a long-term reusable structure library. Those are separate research steps; the current code does not claim parity with RESTART or other 2026 discovery systems.

The PIT monitor is only valid if the conditional predictive CDF is issued before the observation, its parameters/uncertainty protocol are frozen without leakage, and the context channels are based only on the selected action and permitted history. The current synthetic fixture has a known Gaussian noise model; this does not establish validity for real physics measurements, unknown noise, fitted-scale uncertainty, censored data, or adaptive repeated measurements.

## 3. Integration smoke test

The unit test `prequential_pit_audits_are_wired_into_adaptive_discovery` runs paired clean and discrepancy worlds for the same fixed seeds:

- nominal EIG baseline on the same generated world/seed;
- the new EIG-plus-prequential-PIT-audit selector;
- a declared discrepancy amplitude of 0.12 in the simulator's existing synthetic discrepancy basis;
- a 20-round cap and no smoothing, with audit measurement cost counted as a round/query;
- receipts reporting clean recovery, clean PIT alarms, false singleton outcomes, misspecification detection, actual audit queries, and protocol failures.

The test makes no assertion that the monitor improves detection rate or recovery. It asserts that the new branch reaches audit queries, fails closed on protocol defects, and reports recovery-expansion/terminal-alarm counts without treating them as evidence of superiority. Separate tests reject smoothing and check the bounded post-alarm candidate-recovery invariants.

The following focused commands are added to the expected-information-gain workflow:

```sh
cargo test -p symthaea-physics-bridge prequential_pit_audits_are_wired_into_adaptive_discovery --lib -- --nocapture
cargo test -p symthaea-physics-bridge prequential_pit_policy_rejects_smoothed_measurements --lib
cargo test -p symthaea-physics-bridge post_alarm_recovery_reopens_archived_candidates_and_adds_typed_candidates --lib
cargo test -p symthaea-physics-bridge fitted_correction_coefficient_recovers_known_scalar_on_synthetic_residuals --lib
cargo test -p symthaea-physics-bridge evaluator_equivalence_requires_independent_heldout_points --lib
cargo test -p symthaea-physics-bridge truth_absent_recovery_is_reported_separately_from_oracle_present_discrimination --lib -- --nocapture
```

No Rust run is claimed until the exact-head checks finish.

## 4. What the comparison must measure next

Current smoke scale (32 paired seeds) only verifies integration. For a confirmatory test:

- Expand to at least 60 held-out structural worlds across six or more families and 20 paired seeds per world.
- Include null controls, omitted terms, scale and tail misspecification, context-specific shifts, query-region extrapolation, and inaccurate predictive uncertainty.
- Compare nominal EIG, scenario-robust EIG, scheduled random audits, batch conformal audits, finite-horizon spending, standalone PRM where its assumptions hold, and prequential PIT on a conditional-CDF-valid protocol.
- Keep identical candidate pools and noise streams paired where possible; bootstrap worlds first, seeds within worlds second.
- Publish clean structural recovery, false confident singleton rate, calibrated alarm rate, sensitivity by discrepancy magnitude, total measurement cost to valid recovery or explicit unresolved status, median/90th-percentile detection delay, compute and wall time. An alarm that stops forever is not a win unless valid recovery or an explicit unresolved report is counted.
- Preserve the model/query/score/CDF/config digests and exact source/toolchain receipts. Keep hidden truth and discrepancy labels in evaluator-only artifacts.

## 5. Comparison with published scientific-discovery systems

The intended external framing is not a single symbolic-accuracy leaderboard. SciLaws-Bench separates held-out predictive fit, scientific validity, and structural recovery for novel terms, with both fixed-data and active-query settings: https://yiyihum.github.io/SciLaws-Bench/. LLM-AutoSciLab / ActiveSciBench emphasizes a closed loop of hypothesis generation, query selection, and mechanism recovery; reported metrics include symbolic equation accuracy and exact graph recovery: https://scientific-discovery.github.io/llm-autoscilab-project/. LLM-SRBench evaluates transformed familiar equations and synthetic discovery tasks to reduce memorization effects: https://proceedings.mlr.press/v267/shojaee25a.html.

These scores are not directly comparable to the current Symthaea micro-simulator. Pin exact benchmark versions/splits, licenses, task prior knowledge, query/compute budgets and evaluator before a direct comparison. Use the scientific benchmarks to define the endpoint taxonomy; use the paired simulation to isolate the monitoring mechanism; then run a held-out active-discovery comparison to support any performance claim.

## 6. Advancement criteria (targets, not results)

Retain the PIT-audit selector as a candidate for broader evaluation only if the exact-head code passes its focused checks and subsequent matched-budget results show:

- no more than a 5-percentage-point decline in clean-world structural recovery;
- at least 20% relative reduction in false confident singletons under the preregistered misspecification mix;
- detection sensitivity at least 80% on alternatives chosen before opening the held-out split;
- at least 20% lower median measurement cost to valid recovery or explicit unresolved status than the best simple baseline, with paired world-clustered uncertainty intervals supporting the effect;
- no hidden-truth leakage, no invalid smoothed-PIT inputs, and no use of a non-alarm as proof of adequacy.

If a gate fails, report the negative result and retain nominal EIG or another audited policy where it does better. Do not adjust thresholds to manufacture a positive result.
