# VIS-DVPC-044 — Reward-predictor control-variate wealth-floor audit

**Status:** finite deterministic algebra and adversarial-feasibility audit; it is not yet a reward-predictor confidence sequence.

Tracking work item: https://github.com/Luminous-Dynamics/symthaea/issues/8893

## Research motivation

The public MOPE implementation includes reward-predictor control-variate variants in `opebetrp.py`, with predictor data generated in `Reward-Predictor.ipynb`:
- https://github.com/n17s/mope/blob/main/opebetrp.py
- https://github.com/n17s/mope/blob/main/Reward-Predictor.ipynb

The predictor can reduce residual variance. The critical correctness condition is that the control variate is frozen / predictable before the evaluation outcome, its expectation correction is exact, and every bet preserves the minimum wealth factor required by the martingale argument. A small variance is not sufficient if a bet can make wealth negative or break the half-wealth lower bound.

## Control-variate identity

For action distributions h(a) and pi(a), predictor q(a) in [0,1], q_pi = sum_a pi(a)q(a), and w(a)=pi(a)/h(a), define

c(a) = w(a)q(a) - q_pi,
y = w(a)R - c(a) = w(a)(R-q(a)) + q_pi.

When h(a)>0 for target-supported actions and q is fixed independently of the evaluation outcomes,

E_h[y] = sum_a h(a)[w(a)E[R|a] - w(a)q(a) + q_pi]
       = sum_a pi(a)E[R|a] = V(pi).

The tests enumerate action/reward outcomes exactly under Bernoulli rewards. Oracle and misspecified predictors must preserve this expected value; a predictor affects variance and bounds, not the estimand.

## The feasibility proof obligation

For each supported action, the adjusted variable has possible reward range

y_min(a) = -c(a) = q_pi - w(a)q(a),
y_max(a) = w(a) - c(a).

For any candidate value v in [0,1] and nonnegative second bet lambda_2, the minimum wealth factor for that action occurs at R=0 and v=1:

K_factor_min(a) = 1 + lambda_1(w(a)-1) + lambda_2(y_min(a)-1).

Therefore, a sufficient **action-specific** half-wealth constraint is

lambda_2 >= 0,
lambda_1(w(a)-1) + lambda_2(y_min(a)-1) >= -1/2  for every action with h(a)>0.

This audit encodes those inequalities separately from the upstream `solve2` feasible region. It intentionally does not implement the constrained optimizer or claim a finished confidence sequence; the next implementation step is to optimize the same concave quadratic objective over this generally multi-face polygon, then verify every emitted bet against the exact factor floor.

## Exploratory counterexample to verify before porting

A frozen two-action case uses h=(0.01,0.99), pi=(0.5,0.5), predictor q=(0.999,0.001), max importance ratio W=50, candidate v=1, and action 0 with observed reward R=0. The public `solve2` inequalities are

lambda_2 >= 0,
lambda_1 + lambda_2 <= 1/2,
(W-1)lambda_1 - W lambda_2 >= -1/2.

The fixture uses lambda_2=0.1 and lambda_1=(W*lambda_2 - 1/2)/(W-1). It satisfies those encoded inequalities at the boundary, yet direct substitution into the factor using y=wR-c returns 0.455, below 1/2. The action-specific constraints reject this witness.

**Interpretation boundary:** this flags that the encoded feasible set alone does not certify the claimed factor floor for this frozen configuration. It does not, by itself, prove that every upstream MOPE reward-predictor use is invalid; the intended `wmax` semantics and any additional assumptions must be reconciled with the exact source / paper before drawing that broader conclusion. The counterexample is preserved as a regression obligation rather than waved away.

## Frozen predictor regimes

- healthy logger h=(0.5,0.5), pi=(0.5,0.5), reward means (0.2,0.8), oracle q=(0.2,0.8);
- same balanced logger with constant q=(0.5,0.5);
- weak overlap h=(0.01,0.99), pi=(0.5,0.5), W=50, oracle q=(0.2,0.8);
- weak overlap with reversed q=(0.8,0.2);
- weak overlap with constant q=(0.5,0.5);
- additional adversarial near-boundary predictor q=(0.999,0.001) for the wealth-floor witness.

The report includes the exact adjusted mean, raw IPS variance, adjusted residual variance, residual range, action-level weights and control terms, and the explicit factor-floor witness.

## Hard gates

- action support and probability vectors validate; declared W bounds every w;
- E_h[y] equals the target value for every predictor profile;
- oracle predictor variance reduction is measured rather than assumed;
- an upstream-constraint witness that violates the half-wealth floor stays visible;
- the action-specific feasible region rejects the witness and has no factor-floor violation over the deterministic feasible-grid audit;
- deterministic TSV, focused Rust tests, and exact-source/output SHA-256 receipt.

This PR is a necessary audit before porting reward-predictor betting. It does not yet generate confidence intervals, compare statistical coverage, or claim better-than-SOTA performance. The follow-on must benchmark coverage, width, false promotion, power, runtime, predictor misspecification and support shift together.