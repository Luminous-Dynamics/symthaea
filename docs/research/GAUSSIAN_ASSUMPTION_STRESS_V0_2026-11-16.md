# Gaussian observation-model assumption stress v0

**Date:** 2026-11-16  
**Tracking issue:** #12540  
**Composition:** #12426, #12472, #12442, #8607  
**Implementation:** gaussian_assumption_stress in research/falsification-power-reference

## Purpose

The fixed-design Gaussian terminal test is mathematically qualified under an independent, known-variance Gaussian observation model. The existing five-seed campaign establishes Monte Carlo stability inside that model. It does not establish robustness when the observation process changes.

This campaign holds the test, alpha, discrepancy, horizon, design sigma and seed protocol fixed while changing only the synthetic null residual process. The objective is to map an assumption-failure surface, not to make the existing Gaussian theorem more general.

## Frozen test contract

- nominal alpha: 0.05;
- fixed horizon: 80;
- alternative discrepancy: 0.22 per look;
- design standard deviation: 0.80 at every look;
- test: existing GaussianTerminalInducedTest, unchanged;
- five unique root seeds;
- 10,000 paired null/alternative paths per seed and regime;
- 50,000 paths per scenario/regime;
- PRNG: SplitMix64 / Box-Muller v1, for simulation only.

Null and alternative for a regime share the same residual realization; the alternative adds the frozen discrepancy to every look. This produces a paired sensitivity comparison.

## Observation-process strata

1. Matched iid Gaussian: independent N(0, sigma^2). This is the only in-contract reference condition.
2. Stationary AR(1): rho = 0.65 and marginal variance sigma^2. The tested statistic continues to assume independent residuals.
3. Standardized Student-t(5): scaled by sqrt((df-2)/df) to preserve unit variance.
4. Smooth heteroscedastic ramp: per-look variance factor from 0.5+(t+0.5)/80; the mean factor over the horizon is exactly one.
5. Abrupt variance shift: first 40% of looks use variance factor 0.25, final 60% use 1.5; aggregate average variance factor is exactly one.
6. Sparse gross contamination: 2% component with variance factor 37.75 and 98% base component with variance factor 0.25; the mixture's marginal variance factor is exactly one.

Each stressed stratum is marked OUT_OF_CONTRACT even if its empirical rejection rate happens to be close to 5%. Only matched iid Gaussian is marked IN_CONTRACT.

## Outputs

For each regime and each frozen root seed, report:
- terminal null rejection count/rate and Wilson 95% interval;
- terminal declared-alternative rejection count/rate and Wilson 95% interval;
- residual mean and marginal variance;
- within-path pooled lag-one autocorrelation.

The pooled record also reports false-alarm excess relative to 0.05, and keeps analytic iid terminal power as a reference. The lag-one and residual-moment fields are evaluator diagnostics, never selector-visible truth.

## Interpretation

The in-contract iid positive control must be near 5% size and its alternative power must agree with the analytic Gaussian terminal power under a predeclared Monte Carlo tolerance.

The other regimes are deliberately outside the theorem's assumptions. Their rejection rates are observations of sensitivity/misspecification, not acceptance targets. The workflow does not demand a 5% false-alarm rate on them, and it must not call them calibrated simply because a particular finite Monte Carlo estimate is near nominal.

An AR(1) model with a known predictable conditional mean can be handled by a distinct, correctly specified conditional-innovation method (#12442). This fixed-independence terminal test does not inherit that method's guarantee. Unknown rho, estimated variance, heavy tails, non-Gaussian innovations and changing variance require separate valid treatment.

## Claim ceiling

A passing workflow validates the generator, deterministic replay, in-contract positive control and the stress-report schema for this finite synthetic campaign. It does not create a robust test, identify the true noise model from blind data, qualify optional stopping under these stressed conditions, prove real sensor behavior, or extend the Gaussian result beyond its stated assumptions.
