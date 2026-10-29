# VIS-DVPC-045 — Predictor-adjusted MOPE confidence sequence

**Status:** implementation under exact-head Rust qualification; no qualified coverage or efficiency claim until the focused workflow passes.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8898  
Preceding proof obligation: https://github.com/Luminous-Dynamics/symthaea/pull/8895

## Why this is the next research step

The public MOPE implementation includes reward-predictor control variates in `opebetrp.py`; the authors' `Reward-Predictor.ipynb` forms the correction `c = w q(a) - E_pi[q]`. A well-chosen predictor can reduce residual variance under weak overlap. But its adjusted observation has different support from the raw importance-weighted reward, so the original bet constraints cannot be reused without a fresh factor-floor proof.

This PR combines the control-variate expectation identity, the support-aware optimizer and a two-sided sequential confidence sequence. It compares raw and predictor-adjusted MOPE on identical action/reward streams.

Upstream source:
- https://github.com/n17s/mope/blob/main/opebet.py
- https://github.com/n17s/mope/blob/main/opebetrp.py
- https://github.com/n17s/mope/blob/main/Reward-Predictor.ipynb

## Estimand and alpha accounting

For known categorical logger h(a), target policy pi(a), and a predictor q(a) frozen before evaluation outcomes:

- w(a) = pi(a) / h(a)
- q_pi = sum_a pi(a) q(a)
- c(a) = w(a) q(a) - q_pi
- y = w(a) R - c(a) = w(a)(R-q(a)) + q_pi

When target support is covered, E_h[y] = V(pi). A misspecified predictor does not change the estimand when q is fixed; it changes the distribution and range of y.

The lower stream uses y. The upper stream uses 1-y and targets 1-V(pi). Each side has its own action-specific feasible region. Use total alpha=0.05 for each half-initialized wealth process: initial capital 1/2 gives per-side crossing threshold ln(1/alpha)=ln(20), equivalently the uninitialized quadratic lower bound threshold ln(2/alpha)=ln(40). The two tails together have a 0.05 union-bound budget when the martingale assumptions hold. Do not use alpha/2 inside the stream and apply the half-wealth hedge a second time.

## The factor-floor constraints

For each action a, the support of y is

y_min(a) = q_pi - w(a) q(a) = -c(a),
y_max(a) = w(a) (1-q(a)) + q_pi.

For the lower stream, since lambda_2 >= 0 and candidate v lies in [0,1], the worst factor occurs at y_min and v=1:

1 + lambda_1(w(a)-1) + lambda_2(y_min(a)-1) >= 1/2.

For the reflected upper stream, the minimum score is 1-y_max(a), and the same inequality is built with that reflected support. The support-aware polytope is derived from every action in the logging support. Any next bet that fails an action/reward/value-corner check returns an error; the sequence does not quietly widen its feasible set.

At every step the current bet was selected from prior observations. Only after the reward arrives does the code update the sufficient statistics and quadratic lower-bound coefficients, solve the candidate root, verify it against direct product log wealth, update the nested interval, and choose the next bet.

## Benchmark design

Twelve frozen DGP/predictor cases cover:
- healthy support h=pi=(0.5,0.5), W=1;
- weak overlap h=(0.01,0.99), pi=(0.5,0.5), W=50;
- null value 0.50 (reward means (0.20,0.80));
- alternative value 0.65 (reward means (0.50,0.80));
- oracle q equal to reward means, constant q=(0.5,0.5), and reversed q.

For each case, 2,000 seeded trajectories run to horizon 512, with looks 32, 128 and 512. Each raw and adjusted interval sees the same sampled actions and rewards. The PR reports pointwise coverage, sequence-wide coverage and false-exclusion, mean width/endpoints, false promotion under nulls, promotion power under alternatives at threshold 0.55, promotion-time quantiles, root checks/failures/endpoint collisions, method-specific elapsed time, and Wilson 95% intervals for estimated probabilities.

A fixed SplitMix64 seed schedule and replay gate make the statistical output deterministic; wall-clock rows are excluded from the replay comparison. Source, parent kernels, output and summary receive SHA-256 receipts.

## What constitutes a useful improvement

The predictor path earns an efficiency claim only if it narrows intervals or improves decision power **without** losing sequence coverage or violating the wealth floor. If the support-aware feasible set removes most of the variance advantage, the correct result is that this predictor does not help under the current guarantee. The next research directions would be tighter predeclared residual envelopes, cross-fitting with a predictor trained on independent data, and variance-aware betting/freezing—each with a new validity proof.

No real-traffic, production, or SOTA claim is made by these synthetic controls.

## Comparison to current practice

SCOPE-RL includes IPS-family estimators, DM/TIS/PDIS/DR, and high-confidence bounds such as Hoeffding and empirical Bernstein; its evaluation framework reports MSE, Regret@k, ranking, Type-I/II errors and safety violations. This PR adds sequence-wide calibration and anytime decision metrics for the predictor path:
https://scope-rl.readthedocs.io/en/latest/documentation/evaluation_implementation.html

Ryu, Kwon, Koppe & Jun, *Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing* (COLT 2025), is a relevant next comparator for variance-adaptive policy selection; it is not implemented here:
https://proceedings.mlr.press/v291/ryu25a.html

The original MOPE paper is the mathematical reference for the lower-bound confidence-sequence construction:
https://proceedings.mlr.press/v139/karampatziakis21a.html
