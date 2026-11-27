# SEMI-COVERAGE-BACKTEST-001 — dependence-aware realized coverage v0.1

**Issue:** #14014  
**Implementation:** `symthaea-semiconductor::benchmark::realized_coverage_backtest`  
**Status:** named HAC/Bartlett baseline with unit tests authored; exact-head validation pending.

## Estimand

The v0.1 estimand is **qualified observed-stream realized coverage**:

[
widehat C_{mathrm{qual}}
=
rac{1}{n_{mathrm{qual}}}
sum_{t in mathcal Q}
mathbf 1{Y_t in [L_t,U_t]},
]

where (mathcal Q) contains outcomes whose frozen prediction interval was issued before the outcome, whose exact label/revision/payload is bound by a current K1 eligibility receipt, and whose physical result and eligibility report were available by the declared evaluation cutoff.

The denominator is not all wafers, all attempted measurements, or the unobserved population. UNKNOWN, BLOCKED, stale, not-yet-available and post-cutoff results are counted separately; they are not silently interpreted as covered or missed.

A population coverage claim requires the distinct selection/observation-process identification design and independent audit assumptions in #13650/#13984. Qualified-stream backtesting alone does not authorize a population claim.

## Implemented baseline: Newey–West / Bartlett HAC

Given the ordered miss process (M_t = 1 - mathbf 1{Y_t in [L_t,U_t]}), v0.1 computes the truncated Bartlett long-run variance estimate:

[
widehatOmega_L
=
widehatgamma_0
+
2sum_{ell=1}^{L}
left(1-rac{ell}{L+1}ight)widehatgamma_ell,
]

with

[
widehatgamma_ell
=
rac{1}{n}sum_{t=ell+1}^{n}
(M_t-ar M)(M_{t-ell}-ar M),
qquad
widehat{mathrm{SE}}_{mathrm{HAC}}(widehat C)
=
sqrt{widehatOmega_L/n}.
]

The report includes:
- empirical qualified-stream coverage;
- IID Bernoulli standard error as a **diagnostic comparator only**;
- HAC/Bartlett standard error;
- clipped normal-approximation confidence interval;
- standardized deviation from target and critical-value decision;
- qualified/covered/missed/excluded/late counts and prediction-sequence range;
- declared short-memory assumption state and exact HAC lag.

A negative finite-sample truncated long-run variance is not silently clipped. The evaluator returns an error/no inferential report instead.

## Important limitation

This is an explicitly named HAC asymptotic baseline. It is **not** the functional-dependence estimator or theorem in Zhai, Cheng & Wu (arXiv:2609.33868). Their paper derives non-asymptotic marginal coverage-error bounds under a functional-dependence measure, a Bahadur representation/CLT for realized coverage, a consistent block-based standard-error estimator, and a separate long-memory Gaussian-process result with potentially non-Gaussian limiting law and estimated normalization.

No exact-paper estimator or long-memory claim is implemented here. The paper-specific methods remain a reproduction gate for a future module. The current short-memory flag is an operator declaration, not evidence that the process meets short-memory assumptions. If it is not declared, status is `AssumptionNotSupported`, and the report emits no HAC standard error, interval or test.

## Evidence bindings

Each observation includes:
- immutable streaming prediction receipt and its digest;
- frozen lower/upper interval bounds;
- exact label ID, revision, full-payload digest and corrected value;
- full K1 eligibility report and its digest;
- label availability, eligibility-report availability, delivery time and clock domain;
- evaluation cutoff.

The evaluator checks prediction/report digests, label/event/revision/value/payload identity, K1 frontier, availability chronology, clock-domain equality, uniqueness of sample/event/revision/prediction sequence, and monotone event-time order. It does not authenticate issuers or prove physical traceability.

## Baselines and planned extensions

**v0.1 baseline:** IID Bernoulli standard error as a deliberately naive control and Newey–West/Bartlett HAC.

**Next reproduction layer:** exact functional-dependence/block-SE method from arXiv:2609.33868, after transcribing assumptions, formula, finite-sample bound, bandwidth/block conditions, and long-memory normalization from the paper.

**Separate comparators:** moving/circular/stationary block bootstrap; Kupiec unconditional coverage; Christoffersen conditional coverage/independence; Dynamic Quantile backtest; equal-cohort and per-tool/chamber coverage.

These methods have different assumptions and estimands. Do not call a bootstrap/HAC implementation “the paper estimator” unless it exactly matches the source.

## Frozen benchmark design

- IID Bernoulli miss sequence (sanity only);
- Gaussian AR(1) positive/negative autocorrelation;
- GARCH scale persistence and Markov regime switching;
- Student-t5 shared common-scale shocks;
- short-memory and long-memory Gaussian linear processes;
- chamber-common shocks and tool/chamber turnover;
- abrupt mean/variance shifts;
- fixed/variable label delay and cutoff censoring;
- K1 missingness related to extreme outcomes;
- operationally selected labels versus separate randomized-audit sample.

Report empirical size/Type I error, power under 1/2/5/10 percentage-point undercoverage, confidence-interval coverage for realized coverage, interval width, worst-context coverage, effective sample size, cohort count, block/bandwidth selection and delay/cutoff exclusions.

Do not treat failure to reject a nominal-coverage null as proof that the system has correct coverage. Do not claim conditional per-wafer guarantees from long-run marginal coverage.

## Claim ceiling

v0.1 is a synthetic statistical-evaluation baseline. It neither generates prediction intervals nor establishes a manufacturing population guarantee. It does not establish per-wafer conditional coverage, support long-memory inference, repair selective metrology bias, authenticate sources, qualify physical measurement systems, or authorize process/equipment action.
