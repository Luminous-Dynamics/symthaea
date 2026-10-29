# VIS-DVPC-038 — Exact-path audit of hedged scalar-betting confidence-set inversion

**Status:** implementation draft; exact finite-sample synthetic audit only.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8833

## Research question

VIS-DVPC-037 evaluated a one-sided scalar betting process at one fixed threshold. The Scalar Betting section of Karampatziakis, Mineiro & Ramdas (ICML 2021) describes a hedged process for candidate values v, betting against low values with w r - v and against high values with w(1-r) - (1-v).

This benchmark implements that candidate-value scalar-betting component on a frozen grid, rather than just one threshold. It is a closer comparator to the paper than the one-sided test, but it is **not the full two-dimensional, no-grid MOPE algorithm**.

## Test construction

For every candidate v and time t, let

- xi_t^+(v) = w_t r_t - v;
- xi_t^-(v) = w_t(1-r_t) - (1-v);
- S_{t-1}^+ = sum_{i<t} xi_i^+(v), Q_{t-1}^+ = sum_{i<t} [xi_i^+(v)]^2;
- S_{t-1}^- = sum_{i<t} xi_i^-(v), Q_{t-1}^- = sum_{i<t} [xi_i^-(v)]^2.

The predictable scalar bets are lambda_t^+ = max(0, S_{t-1}^+/(S_{t-1}^+ + Q_{t-1}^+)) and similarly for lambda_t^-, with the bet set to zero when the numerator or denominator is non-positive and capped only to guard floating-point rounding. Wealth updates are

K_t^+(v) = product_i [1 + lambda_i^+ xi_i^+(v)],
K_t^-(v) = product_i [1 + lambda_i^- xi_i^-(v)],
K_t^hedged(v) = 0.5 * [K_t^+(v) + K_t^-(v)].

The candidate v is rejected at the first time the hedged wealth exceeds 1/alpha. Since both increments are bounded below by -1 for v in [0,1] and w>=0, each factor stays nonnegative for lambda in [0,1). Betting decisions at t are based only on earlier observations.

## Frozen finite-sample experiment

All scenarios use horizon 8, fixed target policy q=0.5, fixed known logger, stationary Bernoulli rewards, bounded reward range [0,1], and full enumeration of the 4^8 = 65,536 ordered action/reward paths.

| Scenario | Logger P(A) | Reward means (A, B) | True target value |
|---|---:|---:|---:|
| healthy null | 0.50 | (0.20, 0.80) | 0.50 |
| healthy small alternative | 0.50 | (0.30, 0.80) | 0.55 |
| healthy clear alternative | 0.50 | (0.40, 0.80) | 0.60 |
| weak-overlap null | 0.99 | (0.20, 0.80) | 0.50 |
| weak-overlap small alternative | 0.99 | (0.30, 0.80) | 0.55 |
| weak-overlap clear alternative | 0.99 | (0.50, 0.80) | 0.65 |

The candidate grid is v=0,0.05,...,1.00. All six true values lie exactly on the grid. This is deliberate: a coverage claim is only made at a registered true grid value. This benchmark does not establish continuous-parameter coverage for arbitrary values between grid points.

## Metrics

At every look and for every candidate value, output:
- pointwise rejection probability at that look;
- first-rejection probability;
- cumulative probability of ever having rejected that candidate.

For the registered true-value candidate, report coverage as one minus its cumulative rejection probability. This is the finite-grid analogue of sequence coverage. Under a correctly specified fixed logger with positive support, the candidate's scalar-betting process has an anytime-valid type-I error argument; the exact numerical audit checks the finite DGP.

The program also reports the expected number of retained candidates, convex-hull width of the retained candidate set, and empty-set probability. The hull width is explicitly **diagnostic only**: candidate-specific scalar bets can produce non-contiguous accepted sets, so a hull is not presented as a full continuous confidence interval.

## Independent exact-enumeration reference results

Independent vectorized path enumeration (reference calculation, not a Rust test run) provides these final-look controls:

| Scenario | Candidate | Interpretation | Ever-rejection probability |
|---|---:|---|---:|
| healthy null | 0.50 | true-value false rejection | 0.000000000000 |
| weak-overlap null | 0.50 | true-value false rejection | 0.008101857334 |
| weak-overlap small alternative | 0.50 | reject a false candidate | 0.007547226594 |
| weak-overlap clear alternative | 0.50 | reject a false candidate | 0.009730604526 |
| weak-overlap small alternative | 0.55 | true-value false rejection | 0.004168114688 |
| weak-overlap clear alternative | 0.65 | true-value false rejection | 0.002021686820 |

All registered true-candidate rejection rates are below the 5% nominal error budget in these small frozen controls. This is evidence for these enumerated cases, not a replacement for the proof or for testing more pathological inputs. The very low rejection rates on some alternative tests also show that a valid method may be conservative at horizon 8; power must be reported instead of inferred.

The Rust workflow pins the selected known answers, checks every-look probability mass, confirms that every scenario's true value is on the grid, and emits source/fixture/output hashes with the compiler version.

## Evidence boundaries

- Synthetic two-action Bernoulli bandit; no real-trace or production result.
- Fixed target policy, fixed known logging policy, stationary reward means, correct propensities, positive support, bounded rewards.
- The 95% grid-valued coverage claim only applies when the true value is on the frozen grid.
- This implements the scalar-betting component with an explicit candidate grid; it is **not** full MOPE, does not use the paper's quadratic lower-bound/common-bet machinery, and does not claim state-of-the-art performance.
- Next step is to port the full hedged two-dimensional MOPE construction (quadratic log-wealth lower bound, feasible-set optimization, common bets, and running confidence-set intersection) and compare it on the same controls.

## Research source

Karampatziakis, Mineiro & Ramdas, *Off-Policy Confidence Sequences*, ICML 2021:
https://proceedings.mlr.press/v139/karampatziakis21a.html
Paper source, Scalar Betting / MOPE algorithm:
https://github.com/n17s/off-policy-confidence-sequences/blob/7a25831a9ba5763e6a166a0d74ed61e54a7a7e4a/opecs.tex
