# AOBENCH-013A: two-arm Bernoulli anytime-valid sequential assay

**Date:** 2026-11-15  
**Parent programme:** [AOBENCH-013 / #12197](https://github.com/Luminous-Dynamics/symthaea/issues/12197)  
**Implementation issue:** [#12213](https://github.com/Luminous-Dynamics/symthaea/issues/12213)  
**Code:** `crates/domains/symthaea-organizational-closure/src/bernoulli_sequential.rs`

## Purpose and claim boundary

This is the first executable stochastic slice of AOBENCH-013. It evaluates a two-arm Bernoulli assay with an explicit minimum practically relevant risk difference, evidence-qualified observations, finite sample/cost/risk budgets and optional stopping.

It does **not** establish general causal-discovery capability, identify arbitrary mechanisms, validate physical randomization receipts, or provide empirical type-I error / power / coverage results. Its assumptions and limitations are intentionally narrow.

## Agent-facing versus evaluator-only data

The agent-facing scenario contains alpha, minimum effect, sample ceilings, per-sample cost and per-sample risk. Each trial records an arm, binary outcome and four independent evidence states:

- allocation / assignment;
- intervention delivery;
- isolation;
- observation.

The evaluator-only oracle holds the true control and treatment Bernoulli probabilities. It is not serializable and is not embedded in the input scenario or trace. Oracle effect, final/interim coverage and claim correctness appear only in the evaluator report after the episode closes. The report preserves a per-look coverage flag and an all-looks coverage indicator, rather than treating final-look coverage as evidence of time-uniform coverage.

Evidence statuses are supplied by a harness; this module does not authenticate their producer. A production evaluator must bind them to independent assignment and measurement receipts.

## Time-uniform confidence sequence

Let the total two-sided error budget be `alpha`. Allocate `alpha_arm = alpha / 2` to each arm. For qualified sample count `n >= 1`, allocate:

```text
alpha_(arm,n) = alpha_arm * 6 / (pi^2 * n^2)
```

Since `sum_(n=1..infinity) 1/n^2 = pi^2/6`, the per-arm allocated error budget sums to `alpha_arm`. Two-sided Hoeffding gives:

```text
radius(n) = sqrt(log(2 / alpha_(arm,n)) / (2*n))
CI_arm(n) = [max(0, mean - radius), min(1, mean + radius)]
```

With no qualified observations, the interval is [0, 1]. A union bound over both arms and all sample counts gives simultaneous coverage of at least `1-alpha` at every interim look, provided each arm's qualified observations are iid Bernoulli with a stable probability. Predictable adaptive allocation between arms is permitted when the within-arm observation streams retain that assumption.

For the treatment-minus-control risk difference:

```text
CI_effect = [lower_treatment - upper_control,
             upper_treatment - lower_control]
```

The boundary is declared only when the **entire** interval:
- lies above `+minimum_effect` (Benefit);
- lies below `-minimum_effect` (Harm); or
- lies within the equivalence region [`-minimum_effect`, `+minimum_effect`] (NoMaterialEffect).

Otherwise the only valid disposition is to continue within budget or abstain. A fixed-time interval repeatedly inspected at arbitrary times would not provide this guarantee; this implementation uses explicit alpha spending across all per-arm sample counts.

## Evidence, stopping and resource accounting

Every attempted sample consumes total-sample, per-arm-sample, cost and risk budgets whether or not its observation can be used. Assignment, delivery, isolation and observation must all be Verified before a binary outcome updates a confidence sequence. Rejected evidence invalidates qualification; unavailable evidence leaves the episode inconclusive. The implementation fails closed on claims made before a boundary, wrong claims, and additional sampling after a boundary.

Cost and risk are separate non-compensatory limits. An attempt that exceeds one limit is still charged against both dimensions.

## Deterministic tests added

- zero and small sample intervals remain wide;
- unavailable assignment does not update the sequence but consumes resources;
- rejected isolation cannot support a claim;
- benefit and harm controls stop only after their respective time-uniform boundaries;
- no-material-effect requires the full interval inside the equivalence region;
- premature and wrong claims fail;
- post-boundary sampling fails;
- evaluator-oracle scoring distinguishes a false positive on a null episode;
- unqualified attempts consume both sample and per-arm budgets;
- cost and risk are both charged and checked independently.

These are deterministic contract tests, not Monte Carlo calibration results.

## Required next validation

Before any statistical-performance claim:
1. Run the exact-head nested-workspace format, check, test and Clippy gates.
2. Generate many independent null, minimum-effect, benefit, harm and equivalence episodes using sealed random seeds.
3. Estimate false-positive/type-I rate, false-negative rate, power, time-uniform interval coverage and expected sample count with episode-level confidence intervals.
4. Compare with fixed-sample randomized designs, sequential probability ratio tests under matched likelihood assumptions, and a valid confidence-sequence baseline.
5. Repeat under family-disjoint effect sizes and noise settings. Add outcome-dependent missingness, allocation failures, noncompliance, interference and drift as *separate* invalidation or extension tracks, not as if the current guarantee covers them.

Initial targets in #12197 (5% null false-positive rate, at least 80% power at the minimum effect, and calibrated coverage) are proposed qualification goals, not measured outcomes. The conservative Hoeffding sequence may be expensive; efficiency improvements must preserve the stated error guarantee and be independently calibrated.

## Limitations

The method assumes stable iid Bernoulli outcomes within arms and a valid allocation contract. It does not cover non-Bernoulli likelihoods, outcome-dependent evidence qualification, informative missingness, noncompliance estimands, interference, drift, multiple endpoints or arbitrary adaptive model selection. It is an offline benchmark evaluator, not an operational experiment authorization system.
