# Epistemic adjudication contract harness

This standalone Rust crate compiles the repository's production
`src/web_research/types.rs` and `src/web_research/meta_learning.rs` by path.
It does not copy their logic or replace the full Symthaea crate.

## Contract

- `VerificationOutcome.was_correct` is `Some(true)` or `Some(false)` only when an independent correctness outcome is available.
- `None` means the claim has been assessed by the system's verifier but has not been independently adjudicated.
- Unadjudicated records remain countable, but do not update source credibility, domain expertise, empirical accuracy, readiness or Meta-Phi.
- `LearnerStats` reports total, adjudicated and unadjudicated record counts separately. It also reports adjudication-only Brier score and ten-bin expected calibration error (ECE); both remain `None` until valid independent labels exist, and pending claims cannot alter them.
- History is bounded to 1,000 records. Pending records are evicted before adjudicated records so an unbounded stream of pending claims cannot erase measured calibration history.
- The former serialized boolean representation remains compatible: a legacy boolean deserializes as `Some(bool)`; a missing correctness field deserializes as `None`.

The high-level web-research bridge now records each newly verified claim as unadjudicated. It no longer maps `confidence > 0.7` to a ground-truth correctness label.

## Run

```sh
cargo test --locked --manifest-path tools/sym-rsi-epistemic-adjudication/Cargo.toml
cargo clippy --locked --manifest-path tools/sym-rsi-epistemic-adjudication/Cargo.toml --all-targets -- -D warnings
```

This harness compiles the repository's production type and meta-learning source modules and now runs **20 tests** (five type tests, twelve learner/calibration tests, and three serialization-compatibility tests). It does not compile the entire web-research runtime or prove that external adjudication references are authentic. The caller remains responsible for ensuring any supplied `Some(bool)` is backed by an independent outcome rather than the verifier's own confidence.

## Interpretation boundary

A high epistemic confidence score is still a prediction about a claim, not an observed correctness outcome. The web-research bridge records its claims as pending/unadjudicated, preserving the predicted confidence separately. Only an explicit independent adjudication may call `record_outcome` with `Some(true)` or `Some(false)`.

The current verifier remains a source-evidence heuristic. This change prevents its confidence estimate from training the calibration learner as ground truth; it does not turn multi-source agreement into independent scientific validation.

## Calibration metrics

Brier score is the mean squared error between predicted confidence and the independent binary correctness outcome. ECE uses ten fixed confidence bins and weights each bin's absolute confidence/accuracy gap by its sample proportion. Both metrics use only adjudicated observations with finite probabilities in [0,1]; they are not computed from pending claims or the verifier's own confidence labels. The valid calibration sample count is reported alongside the scores, and both scores remain absent when that count is zero. A missing score is reported as `None`, not as an artificial zero or a neutral calibration claim.


## Selective prediction and abstention

The production module `src/web_research/selective_prediction.rs` implements the separately versioned `symthaea-risk-coverage-v1` evaluator. The standalone harness imports this exact source file; the metric logic is not copied into the tests.

- Only independently adjudicated binary truth values with finite confidence in [0, 1] enter risk, AURC, operating points or confidence intervals. Pending and invalid records remain visible as separate counts.
- Risk is the fraction of wrong accepted predictions. Confidence is confidence in the predicted label, not a correctness label.
- Equal binary64 confidence values are admitted as one group. The requested coverage is rounded upward to the next complete tie group; no record is arbitrarily selected within a tie.
- AURC is the conventional item-wise empirical area, with equal-confidence groups handled by the expected item-wise risk over all within-tie permutations. It therefore reduces to standard AURC for unique confidence values while remaining deterministic and input-order invariant.
- Fixed 50%, 80% and 90% targets are accompanied by achieved coverage, selected sample count and a two-sided 95% Wilson interval when the selected sample meets the configured minimum. Always-answer, random-abstention expectation and inclusive confidence-threshold baselines are also reported.
- Reports stratify by domain, source family, delayed/censored status and adjudicator. Unknown values remain an explicit group.
- Decision costs are absent unless the evaluation supplies explicit false-positive, false-negative and abstention costs. Costs are normalized per original eligible case.
- The default minimum eligible sample size is 20. Tests may lower it to exercise small-cohort edge cases; doing so is not an evaluation recommendation.

Run the same locked harness commands above. The metric evaluates predictions against supplied labels; it does not authenticate adjudicator independence, establish truth, choose a production abstention policy or prove that the live researcher uses the metric.


## Selective adjudication and population-risk estimation

The production module `src/web_research/selective_observation.rs` adds an opt-in population-risk lane without changing the ordinary adjudicated-cohort metrics.

A frozen cohort row wraps its prediction, any adjudicator/panel assignment made before the outcome is observed, and a known probability that the independent label is observed by the evaluation cutoff. This pre-label assignment is distinct from the actual adjudicator on a completed label. The caller must also identify the sampling design (randomized audit or known propensity conditional on pre-label history) and reference its frozen manifest.

- Every record in the target population must have a finite inclusion probability in (0, 1]. Zero or missing support blocks population estimation; the evaluator will not extrapolate into never-adjudicated regions.
- A pending record with inclusion probability 1.0 is inconsistent with a design defined at the frozen evaluation cutoff and is rejected.
- The implementation uses Horvitz-Thompson estimates of error totals divided by known full-cohort accepted counts. Finite-sample risk estimates are deliberately not clipped and can exceed 1.
- Kish effective sample size gates the report and each fixed operating point. Domain, source-family, delayed/censored and pre-label adjudicator-assignment strata are evaluated separately; unsupported or underpowered strata remain visible without scores. The completed label's actual adjudicator is not used as a population stratum because it may be unknown on pending rows.
- No confidence interval is generated. Marginal inclusion probabilities alone do not determine a design-valid variance under arbitrary dependence or clustered sampling. A future confirmatory estimator must use joint inclusion probabilities or an independently qualified cluster/design-aware variance procedure.
- Conditional ignorability and the declared pre-label sampling design are assumptions supplied by the evaluator. The module does not authenticate the manifest or prove those assumptions.

This is an exploratory population estimator, not the finite-sample martingale risk-control procedure in the recent SOWRC literature. It does not inherit SOWRC's guarantees. Its role is to expose selection bias and support failures before making population-level selective-risk claims.

The standalone Rust 1.96 harness imports the production source by path. Exact-head formatting, tests and strict Clippy are required before qualification.


## Custodian-randomized stratified audit design

The production module `src/web_research/stratified_audit.rs` turns an externally randomized, complete pre-label cohort into a fixed-size stratified simple-random-sampling-without-replacement (SRSWOR) audit plan.

- A separate custodian supplies independent, uniformly generated 128-bit random priorities and a frozen randomization-manifest reference. This module does not generate entropy, assign reviewers, authenticate the manifest, or expose evaluator outcomes.
- The builder selects the lowest-priority units in each pre-label stratum according to an explicit quota. Priority collisions are rejected rather than broken by record ID.
- Every record in the complete cohort is returned with a positive first-order inclusion probability `pi_i = n_h / N_h`. Every nonempty stratum must receive a positive quota; zero-support strata fail closed.
- Within-stratum second-order probabilities are `pi_ij = n_h(n_h-1)/(N_h(N_h-1))` for distinct records. Cross-stratum probabilities are the product of first-order probabilities, conditional on independent randomization across strata.
- The plan reports whether every stratum supports a within-stratum sample variance: a non-census stratum needs at least two sampled records. This is a design diagnostic, not a claim that a variance estimator has already been implemented.
- The output composes with `SelectiveObservationRecord.adjudication_inclusion_probability`; selected records receive their label through the independent adjudicator, while nonselected records remain pending in the complete cohort.

The random priorities, pre-label strata, quotas and selected IDs must be preserved in an independently controlled manifest. The selection guarantee is conditional on the custodian's randomization and the stated SRSWOR design.


## Design-based stratified audit risk

The production module src/web_research/stratified_risk.rs combines the complete frozen prediction cohort with the custodian-supplied SRSWOR audit plan.

- Requires an exact one-to-one match between predictions and audit-plan records. Selected records must have independent labels; nonselected records remain pending in this audit lane.
- Evaluates always-answer, the observed abstention policy, confidence thresholds and tie-safe risk–coverage prefixes.
- For a fixed acceptance set, the binary loss is 1 only when the candidate answers and is wrong. Per stratum, the Horvitz-Thompson error total is N_h multiplied by the sample mean of loss.
- The unbiased SRSWOR total variance is estimated as N_h squared times (1 - n_h/N_h) times s_h squared divided by n_h. Stratum variances add under independent stratum randomization; risk variance divides by the exact full-cohort accepted count squared.
- A full census has zero sampling variance. A non-census stratum with one sampled unit has no estimable within-stratum variance and withholds the overall interval. A stratum where the candidate accepts no units has identically zero loss and contributes zero variance.
- The point estimate is retained even when a confidence interval is unavailable. Normal 95% intervals are pointwise asymptotic approximations, gated by the minimum accepted audit-label count. They are not finite-sample, anytime-valid or simultaneous guarantees, and are not clipped to [0, 1].
- AURC is reported as a tie-aware point estimate only; its covariance across nested prefixes is not yet estimated. No AURC confidence interval is claimed.
- The risk–coverage curve is updated incrementally across descending confidence groups; each prefix is scored from per-stratum counts rather than rescanning the full cohort.

The standalone harness imports the production source by path. Exact Rust 1.96 formatting, tests and strict Clippy remain required.


## Fixed-horizon selective-observation weighted risk control (SOWRC)

The production module `src/web_research/sowrc.rs` adds a separate fixed-horizon finite-sample selective-risk control lane, inspired by Bai, Fang and Chen, *Risk-Controlling Predictive Sets for Time-Series Events Under Selective Observation with Finite-Sample Guarantees*, Axioms 15(9), 706 (2026), DOI 10.3390/axioms15090706.

It differs from the adjudicated-cohort risk/coverage evaluator, the inverse-propensity population point estimator, and the fixed-quota SRSWOR variance estimator.

- Each chronological event carries the pre-label conditional label-inclusion probability q_t, independent-label-observed flag, and the evaluator-only true-label nonconformity score.
- The v1 restriction is constant predictable weights a_t=1, deterministic envelopes c_n=1 and b_n=1/q_min, and HT loss I_t * 1{r_t > lambda}/q_t.
- It uses fixed betting fractions and mixture weights, computes the observable martingale-mixture E-value, and inverts by log-space bisection for every threshold on a predeclared nested grid.
- The total failure budget delta is split across the fixed threshold grid. The selection rule chooses the smallest grid threshold with upper bound <= alpha; if none qualifies, lambda=1 returns the full label space with structurally zero miscoverage.
- Calibration protocol, inclusion design, and prediction artifact each have separate frozen manifest references and validated 64-character lowercase SHA-256 digest commitments, echoed into the report. The evaluator does not fetch or authenticate signatures over those manifests.
- Observed nonconformity scores are sorted once; suffix weights and a monotone cursor compute all threshold totals in O(N log N + N + J).
- Fixed-quota SRSWOR marginal pi_i values are not substituted for predictable q_t. Sequential design assumptions must be justified separately.

Assumptions/limitations:
- Requires predictable q_t known before labels, conditional ignorability, q_min positivity, bounded scores, fixed horizon/envelopes, frozen predictor and precommitted grid/mixture.
- The module cannot authenticate the custodian/randomization or prove conditional ignorability.
- The risk control applies to the weighted calibration population under those assumptions. Future deployment risk under drift needs a separate certified drift envelope and failure-probability allocation.
- It is not an anytime/time-uniform, adaptive-budget, estimated-propensity or multi-risk implementation.
- No accuracy, scientific truth, live benchmark or SOTA claim follows from this evaluator.

The production source currently contains **18 authored tests**, including exact Bernoulli mask enumeration weighted by the actual design probabilities, a Markov outcome process with history-adaptive predictable audit probabilities, and a deliberately invalid outcome-dependent selection negative control showing why conditional ignorability is necessary. The negative control is not a theorem proof. Exact Rust 1.96 formatting, tests and strict Clippy remain required.




## Predictable Bernoulli audit receipt

The production module `src/web_research/predictable_audit.rs` adds the eventwise audit-selection mechanism required by the SOWRC input contract. It is separate from the fixed-quota stratified SRSWOR design.

- The custodian supplies each q_t, an independent u64 random word, the exact pre-label history digest, and frozen audit/random-stream manifest references and SHA-256 digests.
- Inclusion probabilities use an exact dyadic grid: numerator over 2^64, restricted to multiples of 2^11, so q_t is exactly representable in binary64. A numerator of 2^64 is certainty.
- A unit is selected exactly when random_word < q_numerator. The returned custodian receipt includes selection status, q_t, pre-label history digest, pre-label priority score and random-stream index, but omits the raw random word.
- The full receipt is custodian-private. Do not provide it or its selected flags to the candidate researcher.
- q_t may vary with pre-label history. The new `LinearPrelabelPriority` policy maps an exact pre-label priority score in [0, 2^53] to q_min..q_max, rounds down to the exact dyadic grid, and verifies every supplied q_t against the frozen rule. The policy remains external to the candidate. The code cannot prove the score was computed without current-outcome leakage; this still requires audit of the pre-label history and policy manifest.
- The adapter to SOWRC requires exactly one true-label nonconformity score for every selected audit event and rejects missing labels, nonselected labels, unknown records and invalid scores. Selected-label nonresponse is not silently treated as successful observation.
- The module does not generate entropy, choose the q-policy, authenticate the custodian, or implement a nonresponse model.

The production module includes **14 authored tests** covering exact dyadic conversion, boundary selection, q=1, probability-floor controls, digest validation, record sequencing, raw-draw privacy, history-adaptive q-values, monotone pre-label priority-to-q mapping, exact q-policy replay/mismatch rejection, invalid score/range rejection, and the fail-closed SOWRC adapter. Tests are imported by path into the Rust 1.96 focused harness; no execution result is claimed until exact-head qualification.
