# Temporal distributional risk reference v0.1

**Tracking:** NEURO-SILICON-015 / #9930  
**Implementation:** `research/temporal-intervention-audit/src/distributional.rs`  
**Related:** temporal intervention audit #9881; family adapter #9932/#9929; randomized family inference #9465; controller convergence #9882.  
**Status:** exact finite-support empirical comparator authored; no Rust build, statistical test, benchmark, or population-level dominance claim.

## 1. Motivation

A temporal intervention can reduce average wrong-belief exposure and still produce a much worse upper tail. Mean TED, beneficial-sign frequency, p95/p99, catastrophic episode frequency and expected exceedance are different estimands.

This module adds a deterministic finite-support reference for two distinct empirical orderings:

1. **First-order lower-cost dominance:** the treatment empirical CDF is at least the control empirical CDF at every support threshold, with strict inequality somewhere.
2. **Upper-tail stop-loss ordering:** for every threshold t, the treatment empirical expected exceedance E[(X-t)+] is no greater than control, with strict inequality somewhere.

The second comparison directly exposes long-duration tails. It must not be collapsed into a mean difference or a single quantile.

## 2. Exact finite-support contract

Both empirical distributions use the exact same strictly increasing support grid in microseconds and a count for each grid value. Every count vector must have a positive total and at most one million observations; the grid is capped at 100,000 bins.

No interpolation, smoothing, parametric tail extrapolation, or floating-point tolerance is used. The mean, empirical CDFs, stop-loss numerators and nearest-rank quantiles are computed using checked integer arithmetic.

For treatment T and control C:

- Lower-cost first-order dominance is T ≼₁ C when F_T(x) ≥ F_C(x) for all x, with at least one strict inequality.
- Lower-cost upper-tail stop-loss ordering is T ≼SL C when E[(T-t)+] ≤ E[(C-t)+] for every declared threshold t, with at least one strict inequality.

For unequal sample counts, normalized quantities are compared by exact cross multiplication. Overflow fails closed.

The module reports dominance in both directions. If neither direction holds, the result is crossing/incomparable under that criterion. **Failure to establish dominance is not equivalence.**

## 3. Why both criteria matter

- First-order CDF dominance is a strong ordering: at every threshold, treatment has at least as much probability mass at or below that exposure.
- Stop-loss ordering compares expected excess exposure beyond every threshold. It directly tests upper-tail cost and is sensitive to rare large outcomes.
- A lower mean does not imply first-order or stop-loss dominance.
- A lower p99 at one quantile does not imply that the whole CDF or every stop-loss threshold is better.
- Equal means do not imply equal tail shape.

The finite-support empirical result is a description of the observed sample. It is not a confidence statement about the population distribution.

## 4. Adversarial reference cases

The eight authored tests include:

1. strict lower-cost first-order dominance;
2. crossing CDFs, where neither first-order direction is asserted;
3. equal-mean distributions with different tail shapes, where first-order CDFs cross but stop-loss ordering still identifies the lower-tail-risk distribution;
4. lower treatment mean but worse p99, with neither first-order nor stop-loss dominance;
5. identical distributions, which are not labelled strict dominance;
6. nonuniform support gaps;
7. mismatched, unsorted and zero-count malformed distributions;
8. exact mean and nearest-rank quantile calculation.

These are authored Rust tests, not executed tests.

## 5. Relationship to sequential stochastic dominance

Arnold, Choe, Scarsini and Tsetlin (2026), *Betting on Bets: Anytime-Valid Tests for Stochastic Dominance*, arXiv:2604.21851, develops e-processes for first- and higher-order stochastic dominance:
https://arxiv.org/abs/2604.21851

That is a promising next statistical layer, but the paper's exact process and assumptions must be reproduced before claiming anytime-valid dominance. This module is only the deterministic empirical reference and is not a port of that paper.

Next research steps:
1. reproduce the paper's first-order e-process on its stated finite-support setting;
2. establish exact finite-null enumeration and optional-stopping controls;
3. add higher-order/utility interpretations only after their estimands and direction conventions are independently checked;
4. bind all policy/axis comparisons to the #9465 family ledger;
5. compare against fixed-sample permutation/KS-style methods and naive repeated-look tests;
6. retain source-family, regime, failure and missingness strata.

## 6. Benchmark outcomes

For every temporal exposure axis, report separately:

- mean and median;
- full empirical CDF on the frozen support;
- p90, p95 and p99 nearest-rank empirical quantiles;
- stop-loss curve E[(X-t)+] over all support thresholds;
- first-order and stop-loss dominance direction;
- number of observations and failed/missing outcomes;
- resource-vector cost and physically measured energy, when available.

Do not publish a single scalar "temporal intelligence score". Report the full quality, reliability, tail-risk and resource frontier.

## 7. Claim ceiling

A software PASS establishes exact empirical comparison semantics over a declared finite support. It does not establish a population stochastic-dominance relation, statistical significance, causality, stationarity, safety, representativeness, hardware efficiency or superiority over SOTA.

A valid anytime-valid claim requires a separate sequential inference construction and its declared assumptions, not just repeated use of this empirical comparator.
