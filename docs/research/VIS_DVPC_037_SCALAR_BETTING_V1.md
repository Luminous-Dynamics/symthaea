# VIS-DVPC-037 — Exact-path scalar betting e-process audit

**Status:** implementation draft; exact finite-sample synthetic audit only.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8819

## Research question

VIS-DVPC-036 shows the trade-off between repeated Wald intervals and a conservative alpha-spending Hoeffding confidence sequence. The next question is whether a predictable betting process can give a more useful decision gate while maintaining an anytime-valid false-promotion guarantee.

This benchmark focuses on the decision **“Can we reject that the target policy value is at or below a predeclared threshold?”** It does not build a full two-sided confidence sequence and does not emit an interval width for the betting method.

## Scalar betting rule

For a fixed threshold v and logged IPS sample X_t = w_t r_t, define xi_t = X_t - v. Let

- S_{t-1} = sum_{i<t} xi_i;
- Q_{t-1} = sum_{i<t} xi_i^2;
- lambda_t = max(0, S_{t-1}/(S_{t-1}+Q_{t-1})) when S_{t-1} > 0 and the denominator is positive; otherwise lambda_t = 0;
- K_t(v) = product_{i<=t} (1 + lambda_i xi_i).

The code clamps lambda just below one only to guard floating-point rounding and rejects any non-finite/non-positive factor. Since X_t >= 0 and v in [0,1], xi_t >= -1. With predictable lambda_t in [0,1), each factor is nonnegative. Under the null H0: V(pi) <= v and correctly specified propensities, the conditional mean of xi_t is non-positive; K_t(v) is therefore a nonnegative supermartingale. By Ville's inequality, rejecting the null when K_t(v) > 1/alpha controls the probability of ever rejecting under the declared assumptions.

This is the **one-sided scalar-betting test** inspired by Section 5.2 of Karampatziakis, Mineiro & Ramdas (ICML 2021). It is not the full MOPE algorithm: in particular, it does not invert a two-sided hedged wealth process to produce a confidence sequence, and no confidence interval width is imputed to it.

## Exact frozen design

All six scenarios have horizon 8 and enumerate every ordered action/reward path, 4^8 = 65,536 paths per scenario. The logger and target policy are fixed, known and have positive support. Rewards are stationary Bernoulli variables with means in [0,1]. Each path gets the product of its eight event probabilities; reported totals use the exact finite distribution up to floating-point summation.

| Scenario | Logger P(A) | Reward means (A, B) | True target value | Threshold |
|---|---:|---:|---:|---:|
| healthy null | 0.50 | (0.20, 0.80) | 0.50 | 0.50 |
| healthy small alternative | 0.50 | (0.30, 0.80) | 0.55 | 0.50 |
| healthy clearer alternative | 0.50 | (0.40, 0.80) | 0.60 | 0.50 |
| weak-overlap null | 0.99 | (0.20, 0.80) | 0.50 | 0.50 |
| weak-overlap small alternative | 0.99 | (0.30, 0.80) | 0.55 | 0.50 |
| weak-overlap clearer alternative | 0.99 | (0.50, 0.80) | 0.65 | 0.50 |

The benchmark compares the scalar one-sided betting test with two interval-based gates:
- `WALD_FIXED_TIME`: reused nominal 95% Wald intervals, known not to retain sequence-wide coverage under optional stopping;
- `ALPHA_SPENDING_HOEFFDING_CS`: a conservative, transparent confidence sequence with alpha_t = alpha/[t(t+1)] and a known maximum IPS contribution bound.

The threshold-crossing event means different things by scenario:
- when the known target value is at or below threshold, crossing is a **false-promotion** event;
- when the target value exceeds threshold, crossing probability is **decision power**.

These are reported separately. Conditional mean crossing time is paired with the crossing probability so a method that almost never decides cannot appear artificially fast.

## Independent exact-enumeration reference values

Independent Python path enumeration (a reference check, not a Rust test run) gives:

| Scenario | Method | Ever-crossing / rejection probability |
|---|---|---:|
| healthy null | Wald fixed-time intervals | 0.281250000000 |
| healthy null | alpha-spending Hoeffding CS | 0.000000000000 |
| healthy null | scalar betting one-sided | 0.000000000000 |
| weak-overlap null | Wald fixed-time intervals | 0.039414394874 |
| weak-overlap null | alpha-spending Hoeffding CS | 0.000000004230 |
| weak-overlap null | scalar betting one-sided | 0.001975038788 |
| weak-overlap small alternative | scalar betting one-sided | 0.003379111912 |
| weak-overlap clearer alternative | scalar betting one-sided | 0.007784483627 |

The exact scalar-test false-promotion probabilities are below 0.05 in the listed null controls. These tiny-sample observations are not a proof of validity; the supermartingale argument supplies the theoretical guarantee only under the stated assumptions. The values show that betting can be more decision-active than the conservative Hoeffding gate in the weak-overlap alternative at this short horizon, while keeping an anytime-valid threshold test. The absolute decision power is still low, and the method has not been compared with the full two-sided MOPE confidence sequence.

## Output and qualification

The machine-readable TSV includes per-look pointwise interval coverage and width for Wald/Hoeffding, threshold-crossing probabilities and first-crossing probabilities for all three gates, scalar e-process mean wealth and mean log wealth, sequence coverage for interval methods, conditional first crossing time, path counts, and probability mass checks. Under each frozen null, expected e-process wealth is checked against 1 at every look; this is a martingale invariant in addition to the final false-promotion probability. The artifact receipt binds the exact source, frozen fixture and output hashes, commit SHA and Rust version.

Limitations:
- two actions, fixed target policy, fixed logger, stationary Bernoulli rewards, known propensities;
- no contextual dependence, changing logger, reward drift, hidden confounding or propensity misspecification;
- no full MOPE confidence-sequence inversion;
- no real-trace or production claim, and no state-of-the-art claim based on this synthetic fixture.

## Research references

- Karampatziakis, Mineiro & Ramdas, *Off-Policy Confidence Sequences*, ICML 2021. Section 5.2 describes scalar betting; the full MOPE method includes additional machinery: https://proceedings.mlr.press/v139/karampatziakis21a.html
- Waudby-Smith et al., *Anytime-valid off-policy inference for contextual bandits* (2022), including adaptive logging/context processes and empirical-variance-adaptive confidence sequences: https://arxiv.org/abs/2210.10768
