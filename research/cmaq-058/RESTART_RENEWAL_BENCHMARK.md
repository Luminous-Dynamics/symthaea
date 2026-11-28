# CMAQ-058 — Capital-restart renewal benchmark contract v0.1

Status: preregistration draft. No benchmark result or superiority claim is implied.

## Arms

1. CMAQ-055 full-reset all-start oracle: every start has independent policy and capital state.
2. CMAQ-058 capital-only exact renewal: one shared prequential factor process per kernel, but capital is restarted for every possible start.
3. CMAQ-054 fixed kernel-family monitor.
4. CMAQ-057 shared predictable adaptive monitor.
5. Faithful SCPT with its adaptive KNN neighborhoods and historical betting fractions.
6. ICML 2026 kernel-family mixture with posterior reweighting.
7. Post-hoc best-fixed-kernel oracle as diagnostic only, not an alpha-valid test.

## Statistical conditions

All primary arms use direct conditional paired equality:
Law(X_t | Z_t,F_(t-1)) = Law(Y_t | Z_t,F_(t-1)).
The covariate is available before the current pair. Every per-expert map/direction/bet is predictable before current pair values. Every expert factor has conditional expectation at most one and is strictly positive.

CMAQ-058 reuses the same per-expert factor f_(k,t) across all start components. It does not reset the factor producer at a start time. This is the defining distinction from CMAQ-055.

## Deterministic exactness checks

Before statistical experiments, compare the renewal recurrence against the naive all-start sum for multiple K and horizons, including factors below/above one, zero log-return, alternating returns, extreme but finite returns, and dyadic epoch boundaries. Verify per-kernel wealth, total mixture, exact future mass and posterior normalization.

## Null/alternative families

Null: exact pair equality, stationary conditional equality with serial dependence, predictable heteroskedasticity preserving the conditional paired null, changing predictable covariates, replay/common-cause negative controls.

Alternatives: low/medium/high frequency conditional differences, equal-marginal conditional-pairing change, sparse covariate region changes, oscillating effects, late shifts, and feature-bank blind-spot controls.

Invalid assumptions (conditional mean drift, reference/test mismatch, feature leakage) are reported as contract violations, not valid-null false alarms.

## Grid and counts

Initial horizons H={128,256,512,1024}. Change points t0 in {0,16,64,128,256,512} where t0<H. At least 10,000 independent full-horizon null paths and 5,000 alternatives per primary cell. Alpha in {0.01,0.05,0.10}. Misses count H+1.

## Metrics

- Full-horizon ever-false-alarm with exact binomial confidence intervals.
- Unconditional detection probability, including misses.
- Restricted delay, p50/p90/p95, worst-family regressions.
- p99 policy factor production vs restart bookkeeping cost, separately.
- Peak memory and per-expert / per-kernel restart state.
- Capital-only versus full-reset posterior and detection-delay differences.

Paired inference resamples independent source/common-cause blocks, never overlapping clips as independent trials.

## Prospective target

At matched alpha, seek at least 20% lower restricted delay than the full-reset oracle on three preregistered shift families without violating the null bound or worsening worst-family power by more than two percentage points. Treat this as a target, not an achieved result.
