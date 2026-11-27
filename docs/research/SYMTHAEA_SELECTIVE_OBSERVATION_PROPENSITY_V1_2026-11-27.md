# Symthaea selective-observation and propensity-aware risk v1

Status: research implementation; exact-head Rust qualification pending. This is an offline population-risk estimator, not a production adjudication service or a finite-sample risk-control guarantee.

## Motivation: adjudication-only metrics can still be biased

PR #13969 prevents the model's own confidence from becoming a correctness label. PR #14036 evaluates selective risk using independent adjudications. That closes the self-labeling loop, but it does not guarantee that the adjudicated cohort represents the full population.

For example, a review process may adjudicate high-confidence claims more frequently, inspect only claims that trigger a warning, or preferentially send certain scientific domains to experts. Risk measured only on completed adjudications can then differ systematically from risk among all candidate decisions. A large number of selected labels does not solve a zero-support region.

The 2026 paper *Risk-Controlling Predictive Sets for Time-Series Events Under Selective Observation with Finite-Sample Guarantees* (Bai, Fang and Chen, Axioms 15(9), 706, 21 September 2026; DOI 10.3390/axioms15090706) makes this identification boundary explicit: when a region has zero label-inclusion probability, population risk is not identifiable there. It discusses randomized audits, Horvitz-Thompson losses and martingale-mixture boundaries under stated predictability, ignorability and positivity assumptions.

This Symthaea increment adopts the support diagnostic and inverse-inclusion point-estimation idea. It does **not** implement that paper's e-process, martingale boundary, finite-sample certificate, or future-drift theorem. No such guarantee is claimed.

## Input contract

Each row includes:
- the exact candidate prediction, confidence, abstention status and optional independent truth;
- any adjudicator/panel assignment made before the outcome is observed (separate from the actual adjudicator on a completed label);
- a known adjudication-inclusion probability for the chance that an independent label is observed by the frozen evaluation cutoff;
- the sampling design ID and a reference to its independently frozen manifest.

Two design declarations are supported:
1. randomized audit;
2. known propensity conditional on recorded pre-label history, with the conditioning variables named.

The target population is the complete frozen prediction cohort. Candidate predictions must therefore exist for every row, including rows whose outcomes are pending. The input must not be restricted to only adjudicated examples.

## Identification and admissibility

All records must have finite, strictly positive inclusion probabilities in (0, 1]. A missing or zero probability blocks the population estimate: the implementation never imputes unseen outcomes in a region where labels are impossible to obtain.

The propensity must describe the probability of an independent label being observed by the analysis cutoff, not merely the probability that a reviewer was assigned. A pending record with probability 1.0 is contradictory under this definition and fails closed.

Conditional ignorability is not mechanically verifiable. The design must assign probabilities using only variables observed before the label. For a randomized audit this means the random audit selection must be independent of the protected outcome within its declared sampling frame. For a modelled propensity it means the recorded pre-label conditioning variables must be sufficient to justify conditional independence. The evaluator stores the declaration and manifest reference; a separate custodian must audit them.

## Estimator

For candidate policy acceptance set A, let N_A be the known number of accepted predictions in the full frozen cohort. For adjudicated record i, let I_i indicate that its independent label is observed, let pi_i be its known inclusion probability, and let L_i be 1 when the candidate prediction is wrong and 0 otherwise.

The population error total is estimated by the Horvitz-Thompson sum:

`estimated_errors(A) = sum_i [ I_i * L_i / pi_i ]`.

The estimated selective risk is:

`estimated_risk(A) = estimated_errors(A) / N_A`.

This is an inverse-inclusion point estimator under the declared design and assumptions. It can fall outside [0, 1] in a finite sample; it is never silently clipped. Such a result is a high-variance warning, not a valid probability. Kish effective sample size is reported from the observed inverse-probability weights, and low effective sample sizes withhold population summaries.

AURC uses the same item-wise convention as the selective-prediction lane. Equal-confidence groups are averaged over within-tie permutations so input order cannot change the result. The full-cohort rank/coverage denominator includes pending labels because candidate predictions are available for every cohort row. AURC is withheld if the labelled sample has no support in any confidence prefix; no confidence interval is produced.

Fixed 50%, 80% and 90% operating points use full tie groups and report achieved coverage. The report also contains always-answer, observed-policy and inclusive confidence-threshold summaries. The same support and effective-sample-size checks are applied separately to domain, source-family, delayed/censored and pre-label adjudicator-assignment strata. The completed label's actual adjudicator is not used as a population stratum because it may be unknown precisely for pending rows.

## Uncertainty limitation

Marginal inclusion probabilities are not enough to calculate a generally valid variance under arbitrary dependence, fixed-size sampling, clustering, or adaptive audit designs. This v1 therefore does not attach Wilson intervals to weighted estimates and does not pretend that the adjudicated-only Wilson intervals apply.

Confirmatory evaluation needs joint inclusion probabilities or an independently qualified design-aware variance procedure. If tasks or source families are clustered, bootstrap and variance estimation must operate at the task/family cluster rather than the individual-claim level. A finite-sample guarantee would require a separately implemented and qualified procedure; this module is not SOWRC.

## Paradigm shift and next research

The field should stop treating "we have independent labels" as equivalent to "we have representative labels". Symthaea should measure both:
1. adjudication integrity (is the label independent of the candidate?);
2. adjudication design (could every relevant part of the target population have been labelled?).

The next experiment should randomize a small audit sample across the whole candidate cohort, including abstained and low-confidence cases, stratify by domain and source family, freeze the audit probabilities before adjudication, and compare:
- observed adjudicated-only risk;
- inverse-propensity population risk;
- complete-label oracle risk, retained only by the evaluator after candidate outputs are frozen.
- complete-label oracle risk, available only to the evaluator after the prediction cohort is frozen.

This permits a direct estimate of selection bias and audit-efficiency trade-offs without exposing hidden outcomes to the candidate.

## Tests authored

The production module covers:
- zero/missing/invalid propensity blocks population estimates;
- positive-probability pending rows remain in the full-cohort denominator;
- known finite-population Horvitz-Thompson arithmetic;
- un-clipped estimates outside [0, 1];
- inconsistent pending records with inclusion probability 1;
- invalid candidate confidence and duplicate IDs;
- required pre-label covariates for propensity declarations;
- tie-order invariance;
- stratum-level support failures.

All tests are authored and imported from the production module in the standalone Rust 1.96 harness. They have not been claimed as executed until the exact-head workflow reports formatting, tests and strict Clippy.
