# SPECTRA UVSD likelihood fitter

This crate fits the single-observer unequal-variance Gaussian signal-detection model to ordered type-1 response-category counts. It is deliberately separate from the SPECTRA observation adapter, shared ROC implementation, and Symthaea's HDC optimizer.

## Model

The noise evidence distribution is N(0, 1). Signal evidence is N(mu, sigma^2), with sigma strictly positive. For K ordered response categories, the K-1 criteria are strictly increasing. Each category is an interval between adjacent criteria (with infinite outer bounds).

The un-smoothed multinomial log likelihood is

    sum_j noise_j * log(Phi(c_j) - Phi(c_(j-1)))
  + sum_j signal_j * log(Phi((c_j-mu)/sigma) - Phi((c_(j-1)-mu)/sigma))

The estimator reports d_a = mu / sqrt((1 + sigma^2) / 2). It does not silently add pseudo-counts. Empty observed cells contribute zero; an impossible positive-count cell makes the objective non-finite and is not hidden.

## Inputs and analysis unit

Construct a CategoryTable with one row for every predeclared ordered category, including categories with zero observed counts. The order field is an opaque strictly increasing category rank; it is not a numeric score spacing. Preserve the same category definitions for all fits. For SPECTRA adapter output, group the stratified category counts by participant and session and fit each table separately. Do not fit one pooled curve to a heterogeneous population and then call it a single-observer estimate.

If a category is absent from an adapter's emitted sparse count vector, insert its zero-count row from the frozen observation profile before constructing the table. The estimator cannot infer an omitted category's definition.

## Numerical and optimizer design

- Uses shared symthaea-statistics log-CDF, log-survival and log-interval primitives; lower-tail probabilities are evaluated by a Mills-ratio continued fraction.
- Ordered thresholds are parameterized by a free first threshold and exponentiated positive gaps.
- The primary likelihood has no continuity correction or arbitrary pseudo-count smoothing. An explicit `CountCorrection::MiyoshiReferencePerCategory` compatibility mode adds `1/K` to each cell in each stimulus class, matching the public R source and `rt-da` Python implementation; it is never enabled by default.
- Multi-start BFGS uses an analytic gradient in natural parameters, transformed to the ordered-cut parameterization. An independent central finite-difference test is included; finite differences remain a fallback for non-finite analytic-gradient evaluations.
- Nelder-Mead is provided as a derivative-free cross-check, not as an assumed truth oracle.
- Reports objective, gradient norm, start count and dispersion, selected optimizer, expected-Fisher eigenvalues/rank/condition, and explicit Converged / Boundary / NotIdentifiable / WeaklyIdentified / NotConverged / NotEvaluable dispositions.
- For raw-likelihood, converged, interior fits with full-rank expected information, reports local asymptotic standard errors and 95% Wald intervals for sigma and d_a. These are approximations, not profile-likelihood or bootstrap intervals; they are suppressed for boundary, non-converged, weakly identified and reference-corrected fits.
- Boundary and non-convergence are not converted into finite-point success.
- This prototype does not fit hierarchical UVSD, compute profile-likelihood intervals, or claim calibrated interval coverage.

The fit crate intentionally does not depend on symthaea-core or HDC-specific optimization types. A statistical likelihood should have a transparent numerical contract independent of consciousness/HDC state or optimizer telemetry.

## Interpretation and fail-closed boundaries

- `NotIdentifiable` is an early conservative category-support screen. `WeaklyIdentified` is assigned after fitting when the expected Fisher matrix in internal coordinates is rank-deficient under a relative `1e-10` eigenvalue threshold or has condition number above `1e10`. These are engineering diagnostics, not universal inferential cutoffs or proofs of global identifiability.
- Expected-Fisher intervals use an inverse local information matrix and delta method. They are not robust to unmodeled dependence, misspecification, boundary effects or poor finite-sample asymptotics; separate profile-likelihood/bootstrap calibration remains required.
- Numerical parameter bounds are optimizer guardrails, not scientifically justified parameter limits. A fit touching one is returned as `Boundary`.
- `Converged` means the selected optimizer met its numerical stopping rule. It does not certify a global optimum, model adequacy, the Gaussian assumption, or statistical identifiability.
- `solution_parameter_spread` is a coordinate-wise max-minus-min across converged starts in internal coordinates (mu, log-sigma, first threshold, log-gaps). It is a multistart stability diagnostic, not a confidence interval.
- `diagonal_curvature` is a coordinate-wise second derivative of the objective. It is not a full Hessian, covariance matrix, or uncertainty interval.
- Expected Fisher information is computed from the fitted Gaussian category probabilities and the observed class totals. Wald intervals are emitted only for an interior converged fit using the unsmoothed primary likelihood.
- The lowest finite objective is selected even when its run did not converge; in that case the result is `NotConverged`. A higher-objective converged start cannot mask a lower-objective failed start.
- The primary likelihood has no pseudo-counts. The separately named `MiyoshiReferencePerCategory` mode matches the external `1/K` compatibility convention; exact comparisons must still pin category order, preprocessing and correction mode.

## Reproduction and benchmark programme

1. Known-answer parameter recovery over mu, sigma, criteria, category count, trial budget, class imbalance and sparse cells.
2. Equal-variance negative controls (sigma=1) to measure false UVSD declarations separately from parameter RMSE.
3. Unequal-variance alternatives on both sides of one, including small departures near the practical-identifiability boundary.
4. Held-out category log likelihood, threshold recovery, boundary and non-convergence rates, expected-information rank/condition, Wald/profile/bootstrap interval coverage, and optimizer disagreement.
5. Stress conditions: missing categories, lapses, response ties, speed-accuracy effects, participant heterogeneity and dependence.
6. Reproduce all 11 visual-detection data sets from the public Miyoshi, Rahnev & Lau reference implementation; report per-dataset results, not only a pooled average. Treat recognition-memory dissociations as a domain-transfer negative control.
7. Compare separately with the published RT/confidence UVSD workflow, SDT Assistant's rating-scale UVSD fits, equal-variance SDT, and Lages' hierarchical binary UVSD. These methods have different data and pooling assumptions.
8. Match the independent `rt-da` Python package on its published six-cell example and its 80 subject-fit comparisons before claiming estimator parity; then reproduce all 11 Miyoshi visual-detection datasets per dataset, not only in aggregate. Also include its RT-validity checks, bootstrap intervals and group-level summaries as parity targets.

Reference work:
- Miyoshi, Rahnev & Lau (2026), iScience 29(3), 114998: https://doi.org/10.1016/j.isci.2026.114998
- Their public implementation: https://github.com/kiyomiyoshi/rt_type1_roc
- Independent Python implementation and published-example/subject-fit validation: https://github.com/trevcaru/rt-da
- Lages (2024), Psychonomic Bulletin & Review 31, 2534-2557: https://doi.org/10.3758/s13423-024-02504-5
- SDT Assistant V3.1: https://hautus.org/sdt-assistant/

### Published-example compatibility regression

The R reference function defaults to adding `1/K` to every category cell in each stimulus class (`add_constant = TRUE`); the Rust estimator makes this an explicit `CountCorrection::MiyoshiReferencePerCategory` compatibility mode, never a default. This distinction matters for sparse cells and must be pinned when comparing numerical fits. The independent `rt-da` package also documents matching the R output on 80 subject fits (with per-fit d_a differences no larger than 2.5e-3); that is a stronger benchmark than a single example.

The public R example uses `nr_s1 = [10, 7, 16, 27, 29, 10]` and `nr_s2 = [43, 21, 10, 12, 8, 3]`. Its cell order is strongest-Yes to strongest-No, whereas this crate uses ascending evidence order, so the vectors are reversed before fitting. The explicit reference correction adds `1/K` to each of the K cells in each stimulus class. The regression test targets the published `mu=1.231417`, `sigma=1.252275`, `d_a=1.086691`, and `logL=-315.8781`.

An independent Python reference replay of the same six-start design obtained `mu=1.231422`, `sigma=1.252292`, `d_a=1.086687`, and `logL=-315.878077`. This is a Python algorithm check, not execution of the Rust test.

## Pre-study design search

The public `expected_fisher_information_for_design` API computes local expected information for proposed category boundaries before observations exist. The `design_search` binary explores all 1,287 five-cut designs from a fixed [-3, 3] grid for six ordinal categories, scoring the worst-case expected standard error of d_a across mu={0.5, 1.0, 1.5, 2.0} and sigma={0.70, 1.0, 1.30, 1.60}, with 250 signal and 250 noise trials per design cell. It reports equal-probability and fixed-spacing baselines plus the top 20 minimax candidates.

This is a **model-based design sensitivity analysis**, not evidence that an optimized category scale improves real participant measurements. It assumes the specified Gaussian distributions and known design parameters; the next qualification step is Monte Carlo recovery under lapses, RT/confidence noise, criterion changes, class imbalance and missingness, with frozen design choices and disjoint held-out seeds. Use worst-case and average precision together; do not choose a design solely from one parameter point.

## Latent-threshold finite-sample design recovery

The `latent_design_recovery` binary compares four six-category evidence-threshold vectors: equal-probability, fixed-spacing, the current minimax candidate, and the trade-off candidate. Every design is simulated at the same signal/noise trial totals. It reports per-cell fit dispositions, parameter bias/RMSE, expected-Fisher diagnostics at the generating values, and local Wald interval availability, conditional coverage, Wilson bounds and widths.

All four design arms use common random numbers: the same underlying latent Gaussian draws for a given (mu, sigma, replicate) are categorized by each threshold vector. This permits paired comparisons and reduces avoidable Monte Carlo noise between candidate designs. The seed root and PRNG are emitted in the CSV receipt. Use a seed root reserved for evaluation; do not use the same campaigns to select the design and claim its performance.

Example:

```bash
cargo run --release --manifest-path tools/spectra-uvsd-fit/Cargo.toml --bin latent_design_recovery -- --outer=2000 --trials-per-class=250 --seed=0x5350454344455349 > latent-design-recovery.csv
```

The default 50 outer campaigns are a smoke pilot, not a coverage qualification. Coverage is conditional on returned local Wald intervals; availability and Wilson intervals are reported separately, along with the fit disposition denominator. The output labels every row `LATENT_CONDITIONAL_ONLY`: the candidate evidence thresholds are not controlled participant settings, and this runner does not operationalize confidence criteria or RT measurement. A future operational design must add the observer/channel mapping and burden qualification specified by #10464. These are Gaussian simulation results only, not participant data, a protocol recommendation, or evidence of superiority over external methods.

## Run

    cargo test --manifest-path tools/spectra-uvsd-fit/Cargo.toml
    cargo run --release --manifest-path tools/spectra-uvsd-fit/Cargo.toml --bin recovery_sweep > uvsd-recovery.csv
    cargo run --release --manifest-path tools/spectra-uvsd-fit/Cargo.toml --bin design_search > spectra-design.csv

The recovery runner emits one CSV row per frozen synthetic campaign across mu, sigma and independent seeds, including optimizer and Fisher-information diagnostics. It is a parameter-recovery smoke benchmark only; its Wald intervals are not yet coverage-calibrated and the runner does not estimate false UVSD declaration rates, power, or empirical performance.

This code is an estimator prototype, not an empirical result. It has not been compiled or run in the current environment. Human-subject collection, physical actuation and causal/anomalous information claims are out of scope.


## Sequential policy core (experimental)

The crate exports an experimental `sequential` module for choosing among a frozen set of **experimenter-controlled** trial actions. It intentionally has no action field for participant confidence criteria, internal evidence thresholds or latent response bias.

The selector currently provides:
- typed action-space validation and canonical action ordering;
- hard exclusion of actions without qualified action-to-observation mappings or qualified model adequacy;
- minimum representativeness and model-discrimination floors;
- forced, randomized audit-stratum coverage before ordinary exploration/exploitation;
- deterministic SplitMix64 research randomization;
- exploration plus exploitation of a caller-supplied worst-case target utility;
- marginal propensity for every eligible action, including the combined exploration and exploitation probability mass;
- replay receipts containing policy configuration, model-set version, action-set/history/assessment digests, complete candidate assessments, random draws, exclusions and selected action.

The selector is intentionally a **policy primitive**, not a Bayesian adaptive inference engine. Its caller must supply and freeze the assessments, robust utility, model-adequacy decisions, digest commitments, estimand, ambiguity set and action-to-observation calibration. The module does not compute EIG, posterior updates, model-discrepancy diagnostics, confidence/RT validity, or sequential p-values. A qualified action set and complete receipt do not establish statistical calibration.

SplitMix64-v1 is reproducible simulation randomness, not cryptographic randomness or a deployment-grade randomization service. Do not use this prototype for live human-subject adaptive presentation. See the [SPECTRA sequential-design specification PR #10491](https://github.com/Luminous-Dynamics/symthaea/pull/10491) and [tracking issue #10490](https://github.com/Luminous-Dynamics/symthaea/issues/10490).


## Scenario assessment aggregation (experimental)

The crate also exports an experimental `scenario` module. It converts a complete action × scenario matrix of independently computed outcomes into the `CandidateAssessment` records consumed by the sequential policy.

For each candidate action, it computes conservative policy inputs from caller-supplied score bounds:
- normalized scenario-weighted **lower bounds** for target information and model-discrimination information;
- minimum representativeness lower bound across all scenarios (not an average that can hide a weak subgroup);
- maximum participant-burden upper bound across scenarios;
- operational-mapping and model-adequacy qualification only when every declared scenario is qualified;
- robust target utility from per-scenario utility lower bounds, using maximin or weighted lower-tail CVaR.

Each `ScenarioOutcome` retains both point estimates and conservative lower/upper bounds. Validation requires information and representativeness lower bounds not to exceed their point estimates, utility lower bounds not to exceed utility point estimates, and burden upper bounds not to be lower than point burden. The score-contract digest identifies the caller's frozen score-generation and uncertainty-bound method. **The module does not itself certify that those bounds have nominal statistical coverage.**

The lower-tail CVaR implementation integrates fractional scenario probability mass exactly at the tail boundary. For example, with scenario masses 0.7 and 0.3 and utility lower bounds 10 and -5, the worst 0.5 mass has CVaR 1.0: all 0.3 mass at -5 plus 0.2 mass at 10, divided by 0.5.

Scenario weights are normalized once when the frozen scenario set is built. Every action must have exactly one finite outcome for every scenario. Missing, duplicate or unknown action/scenario pairs reject the whole batch; a failed mapping/model-adequacy flag is preserved and becomes a hard eligibility gate in the sequential selector. No failed scenario is silently removed or used to renormalize the remaining weights.

The returned `ScenarioAssessmentBatch` preserves normalized scenario definitions, caller-supplied scenario-set and score-contract digests, robust objective, canonically sorted raw outcomes and sorted candidate assessments. The digests are opaque caller commitments, not hashes or signatures generated by this module.

This module does **not** generate scenarios, fit candidate models, estimate EIG, diagnose model adequacy or certify coverage. Those scores and qualification flags must come from independently specified and calibrated model/diagnostic code. It must not be used for live adaptive experiments.


## Known-truth observer and measurement-channel generator (experimental)

The crate exports `scenario_dgp`, a deterministic synthetic-data generator for controlled stress tests. It accepts the frozen experimenter action set plus a schedule of (action ID, participant ID, within-participant trial index) requests. Signal/noise class balance and stimulus allocation come from that schedule.

All observer-side quantities live in `ObserverScenario`, not in the action schema:
- signal mean and signal/noise variance ratio;
- latent criteria attached to the experimenter-presented response-scale ID;
- criterion offset and within-session drift;
- lapse and evidence-dependent confidence/category missingness;
- participant-specific sensitivity, log-sigma and criterion random effects;
- Gaussian or variance-standardized Student-t evidence.

An optional RT channel uses an explicitly declared evidence-speed/lognormal latency model, additive non-decision time, quantization, action deadline and timeout disposition. This is a synthetic stress-test model, not a universal law of response time.

The generator returns a canonical request/trial receipt. Each trial retains observed fields separately from `oracle_*` truth fields. Oracle evidence, true category, lapse and participant-level parameters are for simulation diagnostics only; never pass them to a fitted estimator. Missing confidence/category responses, RT timeouts and out-of-range RTs remain explicit rows and counts.

Random streams are keyed by root seed, scenario, participant and trial identity, so permuting the input request vector leaves each generated trial unchanged. The versioned SplitMix64 generator is for reproducible simulation, not cryptographic randomness or production random assignment. Caller-supplied scenario/action digests are opaque commitments; this module does not hash or sign the dataset.

The v0.1 DGP does not model correlated participant random effects, sequential learning beyond linear criterion drift, a validated cognitive RT model, or human behavior. The Student-t family is a controlled heavy-tail alternative. Synthetic data cannot validate a human protocol or establish empirical performance.


## Simultaneous candidate-score confidence bounds (experimental)

The `score_bounds` module converts bounded independent Monte Carlo replicate vectors into simultaneous finite-sample score bounds for candidate decisions. For each action × scenario pair, callers provide replicate vectors for target information, model discrimination, representativeness, target utility and participant burden, with a frozen finite range for each family.

For R action × scenario records it allocates `familywise_alpha` across six one-sided tail claims per record: lower bounds for target information, model discrimination, representativeness and target utility; an upper bound for participant burden; and a second upper bound for target utility. Thus `delta = familywise_alpha / (6R)`. The one-sided Hoeffding radius is:

`(upper - lower) * sqrt(log(1 / delta) / (2 * n))`.

Information, model-discrimination and representativeness use lower bounds; participant burden uses an upper bound; utility receives both lower and upper bounds so a separate indifference-zone selector can construct a robust-utility interval. The resulting point estimates and bounds map directly into `ScenarioOutcome` and can be passed to `aggregate_scenario_assessments`. The batch preserves every raw replicate vector, declared range, familywise/per-claim alpha, seed, source digest, score-contract digest, the frozen action snapshot and the normalized scenario snapshot.

Bonferroni does not require independence across action/metric claims, but Hoeffding does require independent bounded Monte Carlo replications within each score series. This method is intentionally conservative; it is not an empirical Bernstein estimate. It quantifies Monte Carlo uncertainty about the supplied score-generating process only. It cannot repair a biased simulator, an invalid score definition, a wrong estimand or an incomplete scenario bank. A simultaneous bound around a biased point-generating process is still biased.

All bounds and ranges are caller-contractual: the implementation validates sample membership and range ordering, but it does not prove that the declared ranges or independent-replication assumptions are scientifically correct. The method ID is `hoeffding-bonferroni-one-sided-v1`. The module is experimental and is not authorized for live adaptive experiments.


## Robust indifference-zone selection certificate (experimental)

The crate also exports an experimental `ranking` module. It consumes the target-utility lower and upper bounds in a `ScoreBoundsReceipt`, aggregates them over the frozen scenario set using maximin or weighted lower-tail CVaR, and chooses the action with the largest robust lower bound.

It reports a certificate only when

```text
max_action(robust_utility_upper_bound)
  - selected_action(robust_utility_lower_bound)
  <= epsilon
```

Because the score-bound intervals are simultaneous over all action × scenario utility tails, this inequality provides a familywise-confidence statement that the selected action is within the preregistered indifference-zone epsilon of the best action **over the frozen scenario set**, conditional on the bounded-IID Monte Carlo assumptions and declared score ranges.

A failed certificate still identifies the most conservative candidate (highest robust lower bound), but returns `NoSelectionCertificate`; it does not pretend the candidate is proven best. Epsilon is measured in the caller's declared utility units and must be fixed before opening the evaluation seed bank.

This is not a guarantee that the frozen scenarios cover reality, a participant-sampling interval, a causal effect estimate, or an automatic authorization to run the recommended experiment. The score model, scenario bank, objective and utility ranges remain external scientific assumptions.


## Anytime-valid robust simulation allocation (experimental)

The `score_bounds` module exposes `bound_candidate_score_samples_anytime` in addition to the fixed-look Hoeffding estimator.

For each action × scenario cell, the six one-sided tails (four lower score bounds, upper utility, upper burden) receive base per-claim alpha `delta = familywise_alpha / (6 × actions × scenarios)`. At replication count n, the look-specific tail budget is

```text
delta_n = delta / (n * (n + 1))
radius_n = range_width * sqrt(log(n * (n + 1) / delta) / (2 * n))
```

Since `sum_{n>=1} 1/(n(n+1)) = 1`, a union bound over every look and every claim gives time-uniform coverage under independent bounded replication streams within each action × scenario cell. This allows the score allocator to request data-dependent new replications and stop on a robust certificate without reusing fixed-look intervals as though they were anytime-valid. Cross-cell dependence from common random numbers is allowed for the union-bound statement; this module does not estimate paired action-difference intervals.

The `allocation` module provides a first deterministic **robust interval-width heuristic**:
- it requires the time-uniform receipt, a frozen robust objective and a preregistered epsilon;
- it stops when the existing robust indifference-zone certificate succeeds;
- otherwise it considers actions whose robust upper bound can still be within epsilon of the selected robust lower bound;
- it allocates more score simulations to the competitive action/scenario cell with the largest contribution to unresolved maximin or CVaR utility uncertainty;
- it returns `NO_SELECTION_CERTIFICATE` when the compute budget is exhausted or cannot fund an atomic paired batch.

Common-random-number mode requests every action in the selected scenario at the same replication indices and with the same per-index seed. It requires equal current sample counts across actions in that scenario and will not issue a partial paired batch when the remaining budget cannot fund all actions. Independent-per-cell mode permits unequal per-cell counts. The caller supplies a digest of the complete current score receipt; the allocation receipt stores that caller commitment, action/scenario digests, per-cell sample counts, allocation policy, root seed, seed mode, seed-method ID (`spectra-splitmix64-seed-v1`), chosen cell priorities, current certificate and all next-batch seeds. The allocator does not compute or authenticate the digest.



The `verify_score_bounds_receipt` function independently reconstructs a fixed-look or anytime receipt from its raw score vectors, frozen action/scenario snapshots, declared ranges and method schedule. The robust ranking certificate calls this verifier before it accepts the score intervals. It detects changed bounds, output rows, multiplicity, method metadata or raw values outside the declared ranges. Caller-supplied source, score-contract and snapshot digest strings remain opaque commitments, not cryptographic signatures.

The current Hoeffding alpha-spending schedule is a safe baseline but can be conservative at moderate sample counts. Compare it with betting / empirical-Bernstein confidence sequences in [#10598](https://github.com/Luminous-Dynamics/symthaea/issues/10598). For common-random-number runs, the current DGP reuses the action-independent trial seed stream for a shared scenario/participant/trial identity. Do not turn that fact into paired action-difference inference until explicit replication identities and pair-integrity receipts exist; that is tracked separately in [#10597](https://github.com/Luminous-Dynamics/symthaea/issues/10597).

This is **not OCBA, Track-and-Stop, Top Two or a sample-optimal ranking-and-selection algorithm**. It is a transparent baseline to benchmark against robust OCBA, fixed allocation, sequential elimination, Track-and-Stop where assumptions apply, and Top Two where assumptions apply. Seeds are deterministic research-simulation seeds, not cryptographic randomization. The policy does not run human-subject experiments.
