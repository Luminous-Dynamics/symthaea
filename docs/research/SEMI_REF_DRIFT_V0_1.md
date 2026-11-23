# SEMI-REF-DRIFT-001 — time-dependent reference-standard correction v0.1

**Issue:** [#13442](https://github.com/Luminous-Dynamics/symthaea/issues/13442)  
**Implementation:** `symthaea-semiconductor::benchmark::reference_standard_drift`  
**Status:** pre-fitted linear drift model evaluator; exact-head validation pending.

## Research question

Can the value of a reference standard, and the correction derived from it, be predicted at the actual time of a wafer measurement with an uncertainty budget that includes the fitted drift parameters, residual behaviour, model discrepancy and uncertainty in the measurement time?

JCGM GUM-6:2020 §10.6 discusses measurement-standard drift and estimating the standard value and uncertainty at the time of use from calibration history. It is the metrological motivation for this work, not a claim that every standard follows a linear drift model.

Reference: https://www.bipm.org/documents/20126/50065290/JCGM_GUM_6_2020.pdf

## Initial model

The first benchmark deliberately limits itself to an externally fitted linear correction model:

`c(t) = beta_0 + beta_1 (t - t_0)`

where time is expressed in days relative to the declared anchor.

The covariance of `[beta_0, beta_1]` contributes time-dependent parameter uncertainty:

`u_param^2(t) = [1, d] Cov(beta) [1, d]^T`

The module evaluates the maximum parameter variance across the measurement-time interval endpoints. It then adds residual, model-discrepancy and measurement-time-width contributions linearly as a conservative no-independence upper bound:

`u_upper(t) = sqrt(max(u_param^2(t_lower), u_param^2(t_upper))) + u_residual + u_model + |beta_1| * half_time_width_days`

This is an intentionally conservative synthetic contract. It is not a universal GUM uncertainty model, and it does not replace a qualified joint likelihood/covariance model.

## Implemented Rust interface

The module consumes:
- physical measurement evidence;
- the calibration certificate;
- the original #13382 measurement qualification policy and report;
- a pre-fitted reference-standard drift model packet;
- an explicit policy for minimum calibration-history epochs, maximum extrapolation, uncertainty limit, coverage factor and anchor agreement;
- a measurement-time interval and a knowledge frontier.

The model packet binds:
- reference standard and reference certificate digest;
- measurand and unit;
- anchor time, fit interval and prediction-validity interval;
- model-available time;
- calibration-epoch count and history / holdout manifest digests;
- declared model qualification state;
- correction intercept/slope and 2×2 parameter covariance;
- residual and model-discrepancy standard uncertainties;
- deterministic model digest.

The function:
- independently replays the original static #13382 qualification;
- refuses a tampered base qualification report;
- rejects identity/scope mismatch, certificate-anchor inconsistency, invalid chronology, malformed covariance and model-digest mismatch;
- returns UNKNOWN for unsupported prediction horizon, excessive extrapolation, insufficient history, missing holdout evidence or a model not available at the chosen knowledge frontier;
- withholds predicted correction values when the model is UNKNOWN or REJECTED;
- predicts the additive correction and corrected measurement only when the model passes the declared support/qualification checks;
- returns a deterministic evidence receipt;
- never mutates the certificate, measurement, label, model, training snapshot or equipment state.

The model's declared `Qualified` state and manifest digests do not authenticate the history or issuer. Independent physical evidence and source trust remain outside this module.

## Authored tests

Thirteen Rust tests cover:
1. known linear drift correction at measurement time;
2. parameter covariance, residual and model uncertainty propagation;
3. time-interval-width uncertainty;
4. model not yet available at the selected knowledge frontier;
5. prediction outside the declared horizon;
6. excessive extrapolation;
7. insufficient calibration epochs and missing holdout validation manifest;
8. correction-anchor mismatch;
9. reference-standard identity mismatch;
10. malformed/non-PSD covariance rejection;
11. failed static #13382 qualification blocking dynamic prediction;
12. deterministic report replay.

Tests are authored and are not claimed as passing until the exact branch head completes rustfmt, strict Clippy, crate tests and the SEMI-BENCH-001 reference workflow.

## Baselines and next steps

Compare matched held-out calibration epochs against:
- static certificate correction / last calibrated value;
- linear point extrapolation without parameter uncertainty;
- linear drift with residual uncertainty only;
- linear drift with parameter covariance and model discrepancy;
- the qualified shared-nuisance/correlated model when #9234 is available;
- oracle parameters as evaluation-only upper bound.

Keep prediction error, interval coverage/width, decision-margin ratio and eligibility outcomes separate. Add abrupt steps, nonlinear drift, common-reference causes and independent reference chains only after the linear small-instance oracle is validated.

## Claim ceiling

This is a synthetic model-evaluation component. It does not prove that any physical reference standard follows a linear drift model, authenticate its calibration history, establish traceability, qualify instruments, accredit a laboratory, or authorize label promotion, wafer disposition, calibration adjustment or fab process control.
