# Sequential falsification e-process reference v0

**Date:** 2026-11-15  
**Status:** implementation candidate; exact-head Rust qualification pending  
**Owner:** Symthaea #12402  
**Stacked implementation:** PR #12399 branch research/falsification-power-reference-v0-1

## 1. Objective

The fixed-design power reference answers whether a frozen design could distinguish a declared alternative under idealized assumptions. It does not address optional stopping, repeated looks or experiment actions chosen from earlier observations.

This slice adds a narrowly scoped time-uniform reference. The evaluator must commit a predicted discrepancy and decision identity before consuming its corresponding observation. The process returns a p-value based on the maximum e-value seen so far, so later evidence moving against the alternative cannot erase an earlier boundary crossing.

## 2. Conditional Gaussian reference

Let F_(t-1) contain all information before observation t, including prior observations and the selected action A_t.

Under the simple null:

    Y_t | F_(t-1), A_t ~ Normal(0, sigma_t^2), where sigma_t is known and predictable

For one predeclared alternative, the evaluator chooses a discrepancy d_t using only information available before Y_t. Under the alternative the conditional mean is d_t, with the same per-step known variance sigma_t. The action may change sigma_t, provided its value is committed before the observation.

The conditional likelihood-ratio increment is:

    e_t = exp(d_t * Y_t / sigma_t^2 - d_t^2 / (2 * sigma_t^2))

Under the stated null and predictable d_t:

    E_0[e_t | F_(t-1), A_t] = 1

Thus the product E_t = product_(s<=t) e_s is a nonnegative martingale under the simple null. Ville's inequality gives:

    P_0(sup_t E_t >= 1 / alpha) <= alpha

The running-maximum anytime p-value is:

    p_t = min(1, 1 / max_(s<=t) E_s)

This is the time-uniform boundary implemented in the Rust reference. It is not the current fixed-time p-value and it is not permitted to reset the running maximum after an adverse observation.

## 3. Predict-before-observe protocol

The API uses a two-stage transition:

1. commit_prediction(observation_id, decision_digest, d_t) (or the mixture variant) records the discrepancy for the next observation. Use commit_prediction_with_sigma / commit_predictions_with_sigma to bind a known action-dependent sigma_t in the same step.
2. observe(observation_id, decision_digest, Y_t) consumes only the matching observation.

The implementation rejects:
- observation before prediction;
- a second prediction while one is pending;
- a mismatched observation ID or decision digest;
- duplicate consumed observation IDs;
- non-finite inputs and numerical overflow;
- a mixture with missing, duplicate or malformed alternative declarations.

This is a structural chronology check, not cryptographic proof of data isolation. A caller could still leak the outcome into its own prediction through an out-of-band path. A production evaluator must preserve and independently audit the action/prediction commit event before outcome release.

## 4. Fixed mixture across alternatives

For a finite alternative set j=1..K, choose weights w_j > 0 with sum_j w_j = 1 before the sequence. Each alternative produces its own predictable sequence d_(j,t) and e-process E_(j,t).

The fixed mixture:

    E_t_mix = sum_j w_j * E_(j,t)

is also an e-process under the same null. This avoids reporting the largest post-hoc component as though it were a predeclared test. The mixture weights and alternative identities must be frozen. Dilution from assigning weight to many irrelevant alternatives is a power cost and must be reported.

This does not provide a generic family-wise test for an adaptively selected alternative set. If the alternative family changes after observing data, the fixed-mixture argument no longer applies without additional predictable betting or error-control theory.

## 5. Assumptions and non-claims

v0 requires:
- a simple conditional Gaussian null with zero residual mean;
- known, strictly positive per-step sigma_t committed with each prediction;
- conditional variance and distribution correctly specified at every step;
- discrepancy predictions committed before their scored outcomes;
- a fixed alternative identity/mixture and predeclared weights;
- finite, correctly aligned observation/action records.

It does not qualify:
- unknown or estimated variance;
- composite nulls and nuisance-parameter estimation;
- autocorrelation or cross-observation dependence not captured by the conditional model;
- unmodelled heteroskedasticity, heavy tails, censoring, missing-not-at-random or sensor drift;
- data-dependent post-hoc alternative selection;
- simulator nulls without a valid null simulator;
- physical trajectories merely because their one-step residuals look Gaussian.

Those require separate test constructions, calibration and independent validation. In particular, iid confidence-sequence guarantees must not be transferred to a correlated physical time series by analogy.

## 6. Literature comparators

- Koning & van Meer, *Anytime validity is free: inducing sequential tests*, JRSS B (2026-02-21): a Doob-martingale/terminal-test induction comparator. It shows anytime-valid procedures can match a valid terminal test at the maximum horizon; therefore e-processes should not be treated as the only sequential paradigm.
- Cortinovis, Battaglia & Caron, *SLICE: Sequential Likelihood-Free Inference via Conformal E-Values* (2026-10-09): a more ambitious path for simulator models with intractable likelihoods. It uses fresh null simulations to conformalize positive predictable scores into conditional e-values and confidence sequences.
- Abdulsamad et al., *Maximin Robust Bayesian Experimental Design* (2026-03-14): a distinct experiment-selection approach addressing model misspecification with max-min / Sibson information and PAC-Bayes bounds. Robust design does not itself supply sequential test validity.

The small Gaussian likelihood-ratio reference in this crate is not an implementation of SLICE, terminal-test induction, or robust Bayesian design.

## 7. Benchmark protocol

Compare, on the same task families and inclusive observation/compute/cost budgets:
- fixed-horizon Neyman–Pearson test (terminal reference);
- sequentialized terminal test / alpha spending;
- single-alternative Gaussian likelihood-ratio e-process;
- fixed weighted mixture e-process;
- post-hoc best-alternative selection as an explicitly invalid negative control;
- fixed-grid, random, EIG/model disagreement, D-optimal and support-aware experiment policies.

Report separately:
- type-I error by null family, including confidence intervals;
- empirical probability of ever crossing alpha under repeated looks;
- terminal-horizon parity with the matched fixed test;
- alternative-specific power curves and detection delay;
- false-adequacy and false-rejection rates;
- unresolved / underpowered frequency;
- observations, simulator calls, monetary/resource cost and human review;
- malformed, invalid, timed-out and failed runs.

Use independent null calibration and sealed alternative confirmation. A green unit-test workflow does not establish empirical type-I calibration. Never pool away a weak alternative stratum.

## 8. Cross-project ownership

- **Symthaea #12375 / #12402:** alternative semantics, power accounting, predictable policy interface and sequential qualification.
- **Symthaea #11060 / #11171:** blind episodes and actual research-policy adapter; the finite baseline is not the learned researcher.
- **Symtropy #3784 / #3973:** physical parameter identifiability, covariance-aware observation models and numerical/physical qualification.
- **Symtropy #5328 / Mycelix #6031:** sealed alternative custody and purpose-bound access evidence.
- **Sol-Atlas #853:** applied claim classes and evidence ceilings.

The sequential evaluator records evidence; it does not promote itself to a scientific authority.

## 9. Acceptance gates

Before calling the implementation slice qualified:
- exact-head Rust format, all-target tests and strict Clippy;
- known-answer likelihood-ratio increment;
- fixed-mixture equality to weighted component e-values;
- predictable commit/observe chronology and negative controls;
- maximum-e-value anytime p-value behavior;
- no-mutation behavior for invalid updates and numerical overflow.

Before claiming statistical validity in a campaign:
- independent derivation review of conditional model assumptions;
- differential reference implementation;
- null simulation under the actual adaptive action process, with confidence intervals;
- empirical crossing probability under repeated monitoring;
- terminal-horizon comparison to a fixed test;
- sealed alternative-specific power and detection-delay evaluation;
- independent review of the alternative custody and action-prediction event log.

A successful implementation test is not a scientific result, an external benchmark result, or physical validation.
