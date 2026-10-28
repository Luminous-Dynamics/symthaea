# VIS-DVPC-035 — Exact optional-stopping audit for sequential OPE

**Status:** implementation draft; exact synthetic finite-sample audit only.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8791

## Why pointwise intervals are not enough

VIS-DVPC-034 evaluates intervals at fixed sample size. In adaptive policy evaluation, a team may inspect results repeatedly and stop when a lower bound first clears a target. Reusing a 95% fixed-time interval at every look does not make the whole sequence a 95% confidence sequence. The relevant risk is the probability that any look leads to a false conclusion, not only the final-look coverage.

## v1 comparator

`tools/vision-dvpc-sequential-ope.rs` enumerates every length-eight action/reward sequence in a two-action Bernoulli bandit DGP: 4^8 = 65,536 possible paths per supported scenario. In addition to fixed loggers, the fixture includes a rotating *known* logging-propensity schedule while reward means remain stationary, so observations are independent but not identically distributed. It evaluates:

- `WALD_FIXED_TIME`: nominal 95% normal intervals at each look n=2..8, intentionally reused without multiplicity correction.
- `ALPHA_SPENDING_HOEFFDING_CS`: a simple anytime-valid baseline using the logged IPS contribution and known maximum importance weight W. At look t, alpha_t = alpha / (t(t+1)) and radius W * sqrt(log(2/alpha_t) / (2t)). Since the sum over t>=1 of alpha_t is alpha, the union bound controls the probability of any two-sided Hoeffding failure by alpha under the stated assumptions.

The alpha-spending method uses the maximum importance-weight bound across the full known schedule and is deliberately transparent and conservative. It is not an implementation of the published Off-Policy Confidence Sequences algorithm; it provides a known-answer baseline to test sequential composition and optional stopping before adopting a sharper method.

## Exact null-control results

The threshold for false promotion equals the true target policy value, 0.5. A false promotion occurs when any lower confidence bound exceeds the true value. The following results are independently enumerated under the frozen independent-row DGPs. Reward means stay fixed; the rotating logger has known row-specific propensities. The current Rust implementation pins the healthy values in unit tests.

| Logging probability for action A | Reused Wald simultaneous coverage, n=2..8 | Wald ever-false-promotion probability | Alpha-spending CS simultaneous coverage, n=1..8 | CS ever-false-promotion probability |
|---:|---:|---:|---:|---:|
| 0.50 | 43.7500% | 28.1250% | 100.0000% | 0.0000% |
| 0.90 | 19.4543% | 4.5079% | 99.9996% | 0.00045% |
| 0.99 | 1.9879% | 3.9414% | 99.9999996% | 0.000000423% |
| Rotating schedule (0.5, 0.9, 0.99, 0.5, 0.95, 0.8, 0.99, 0.5) | 47.1781% | 0.0271% | 100.0000% | 0.0000% |

These are exact probabilities up to floating point summation, not Monte Carlo estimates. The rotating-schedule case holds the reward means fixed but varies the known logger propensity by row; this is not a reward-drift or outcome-dependent adaptive-logger test. They illustrate two things: repeatedly checking fixed-time intervals can produce very poor sequence-wide coverage, and a correct anytime-valid bound can become almost vacuous under weak overlap. The false-promotion result is specifically the null control where the threshold equals truth; it is not a generic production false-discovery rate.

## Controls and evidence boundary

- Exact probability mass must sum to one for every supported eight-step DGP.
- Logging propensity misspecification is marked `ASSUMPTION_VIOLATION_DESCRIPTIVE_ONLY`; the formal confidence-sequence guarantee does not apply to that case.
- Zero support returns `NOT_IDENTIFIED` and suppresses all intervals before estimator output.
- The tool reports pointwise coverage, interval width, false-promotion probability by look, simultaneous coverage and ever-false-promotion probability.
- The rotating schedule only tests a predeclared, known logger change. It does not cover contexts selected as a function of past outcomes, reward drift, clustering, hidden confounding or unbounded weights.

## Reference implementations to compare next

- Karampatziakis, Mineiro & Ramdas, *Off-Policy Confidence Sequences*, ICML 2021: https://proceedings.mlr.press/v139/karampatziakis21a.html
- Waudby-Smith et al., *Anytime-valid off-policy inference for contextual bandits* (2022): https://arxiv.org/abs/2210.10768

The next research step is to implement/port a faithful published OFCS algorithm and compare it on the identical frozen paths for coverage and width. No claim of SOTA, real-trace validity or production safety follows from this synthetic baseline.