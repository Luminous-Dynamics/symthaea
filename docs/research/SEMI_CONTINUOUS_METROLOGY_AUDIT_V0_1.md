# SEMI-CONTINUOUS-AUDIT-001 — continuous population error under adaptive metrology selection v0.1

**Issue:** #13514  
**Implementation:** `symthaea-semiconductor::benchmark::continuous_metrology_audit`  
**Base:** validated evidence/selection code at #13465  
**Status:** Rust implementation and seven unit tests authored; exact-head validation pending.

## What already existed

The existing `metrology_audit` module implements:
- deterministic exploit selection (inclusion probability 1);
- fixed-quota randomized audit selection using source-defined strata;
- simple random sampling without replacement (SRSWOR) within each residual stratum;
- per-row first-order inclusion probability;
- binary defect-prevalence and Brier totals using Horvitz–Thompson;
- finite-population variance for the residual audit sample.

This increment deliberately reuses that design. It does not introduce another sampler or incorrectly apply independent-Bernoulli variance to fixed-size sampling.

## New continuous outcome contract

For every candidate in the frozen frame, the new input binds:
- candidate, subject and stratum IDs;
- continuous target measurand and physical unit;
- frozen VM point prediction;
- frozen predicted absolute error for model-assisted correction;
- immutable prediction-receipt digest;
- complete finite-frame digest.

Every selected candidate has exactly one explicit result state:
- `ReturnedQualified`;
- `ReturnedUnqualified`;
- `SelectedButNotReturned`.

For a qualified result, the estimator checks:
- K1/current frontier and downstream eligibility;
- label ID and non-zero revision;
- full label payload digest and eligibility receipt digest;
- measurement-event identity with the frozen prediction;
- graph maturity at the declared cutoff;
- exact graph subject, measurand, unit, revision and physical label value;
- finite non-negative qualification uncertainty.

No point value contributes if any of these checks fail. If a selected result is missing or unqualified, the report returns UNKNOWN and no population MAE estimate. It does not shrink the denominator or silently impute a result.

## Estimators

For candidate i, observed absolute VM error is:

`e_i = |y_i - \hat y_i|`

For deterministic exploit candidates (census), their errors are added exactly.

For residual stratum h with population size N_h and a simple random audit of n_h items:

- Horvitz–Thompson residual total:
  `T_hat_HT,h = N_h * mean_sample(e_i)`
- Model-assisted residual total:
  `T_hat_MA,h = sum_frame(m_i) + N_h * mean_sample(e_i - m_i)`

where m_i is the frozen predicted absolute error for each candidate in the residual frame.

The model-assisted residual correction is design-unbiased under the declared SRSWOR design and a complete frame; the prediction model only supplies a variance-reduction/control-variate term.

For residual sample values z_i (either e_i for HT or e_i-m_i for model-assisted), the design variance estimator is:

`V_hat_h = N_h^2 * (1 - n_h/N_h) * s_h^2 / n_h`

The implementation sums stratum variances and divides the standard error by the full population N to obtain an SE for mean absolute error. Exploit census strata contribute zero sampling variance. If a residual stratum has fewer than two sampled audit results and is not a census, its variance is unqualified rather than fabricated.

## Design validation

The evaluator verifies the selection report digest and its exact SRSWOR propensity structure:
- deterministic exploit rows are selected with pi=1;
- randomized audit rows are selected;
- not-selected residual rows remain unselected;
- all residual rows in a stratum have the same pi=n_h/N_h;
- each residual stratum has positive randomized audit support.

It also requires one prediction per population candidate and exactly one result state per selected candidate. The estimator reports:
- model-only predicted mean absolute error;
- HT mean absolute error;
- model-assisted residual-corrected mean absolute error;
- their respective standard errors where design-qualified;
- per-stratum totals/variance;
- effective sample size and maximum inverse-probability weight;
- measured fraction, nonresponse and unqualified counts.

## Authored tests

Seven unit tests cover:
1. a two-stratum known-answer HT and model-assisted estimate;
2. full census equals exact population MAE with zero variance;
3. selected-but-not-returned outcome returns UNKNOWN without denominator shrinkage;
4. blocked K1 label is nonresponse, not a zero error;
5. incomplete frame and invalid audit propensity fail closed;
6. unit mismatch and label-payload tampering reject;
7. deterministic replay returns the same receipt and estimates.

Exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 reference workflow are required before a pass can be claimed.

## Competitive benchmark protocol

On identical candidate frames and selection schedules compare:
- model-only estimated error;
- HT estimate using the audit lane;
- model-assisted residual correction;
- full-census synthetic truth;
- risk/EVSI exploit + randomized audit policy;
- pure SRSWOR audit baseline.

Sweep metrology budgets and audit quotas. Report:
- population MAE bias/RMSE and interval coverage;
- worst-stratum estimation error;
- metrology cost and reduction;
- ESS and maximum weight;
- audit fraction and nonresponse;
- selected-set MAE separately from population MAE.

Never pool different physical target units or combine nm, cm⁻³ and overlay error into one score.

## Claim ceiling

This estimates a continuous finite-population error under the existing logged exploit + stratified SRSWOR audit design and complete K1-qualified outcomes. It does not identify arbitrary MNAR outcomes, prove that a production frame is complete, authenticate physical source issuers, establish causal impact of a sampling policy, or authorize manufacturing action.
