# VIS-DVPC-048 — IPCW, informative censoring and identification ceilings

**Status:** deterministic synthetic benchmark under exact-head Rust qualification. No empirical result is qualified before CI completes.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8935

## Research question

Off-policy evaluation often assumes that a reward is available for every logged decision. In delayed-feedback systems, analysis is usually performed after a fixed follow-up deadline: some logged decisions have matured outcomes and some remain censored. Dropping the unavailable outcomes, or replacing them with zero without correcting the inclusion probability, can change the estimand.

This benchmark distinguishes three situations:

1. **MAR / independent censoring given context and action:** if outcome maturity is conditionally independent of reward and its probability q(x,a) is known and positive on target-policy support, IPCW can recover the target value.
2. **Positivity failure:** if a target-supported context/action cell never matures, weighting cannot recover its outcome mean from this dataset. The benchmark emits `NOT_IDENTIFIED_POSITIVITY` and suppresses IPCW point estimates and intervals.
3. **Outcome-dependent censoring (MNAR):** if the chance that a reward matures depends on the unseen reward itself, q(x,a) alone is insufficient. The benchmark emits `NOT_IDENTIFIED_MNAR` and suppresses IPCW metrics rather than treating a wrong weighting formula as a correction.

The maturity indicator in this benchmark is finalized at a **predeclared follow-up deadline**. Horizons 128/512/4096 mean increasing sample sizes of decisions with their deadline-availability indicator known. This is an observation/censoring benchmark, not an implementation of asynchronous survival analysis for a time-varying hazard. The fixed deadline keeps the estimator's score sequence well-defined for the confidence-sequence proof.

## Frozen DGP

There are two equally likely contexts, two actions, and a known uniform behavior policy h(a|x)=0.5. The target policy is deterministic: action 1 in context 0, action 0 in context 1.

| Context | Reward mean, action 0 | Reward mean, action 1 | Target action |
|---|---:|---:|---:|
| 0 | 0.20 | 0.80 | 1 |
| 1 | 0.75 | 0.30 | 0 |

The exact target value is

`V = 0.5(0.80 + 0.75) = 0.775`.

For one logged decision, the target-policy IPS score is `X = 2R` if the logged action matches the target action for its context and zero otherwise. Thus X lies in [0,2] and E[X]=V when there is no censoring.

Five fixed-deadline scenarios:
- `no_censoring`: q(x,a)=1 for all cells;
- `uniform_censoring`: q(x,a)=0.5 for all cells;
- `context_action_dependent_censoring`: q=[[0.95,0.20],[0.25,0.90]], independent of reward conditional on the cell;
- `target_cell_never_matures`: q(0,1)=0, so a target-supported reward cell is never observed;
- `reward_dependent_mnar_censoring`: P(mature|R=1)=0.20 and P(mature|R=0)=0.90. This depends on the unobserved outcome, so the q(x,a) correction is not identified.

## Compared estimators

**Available-case IPS** averages X only over matured rows. Its denominator is the number of matured outcomes, not all assignments. Under cell-dependent q, the mature sample's context/action mixture shifts and this complete-case mean is generally not V.

**Zero-filled IPS** sums mature X and divides by all assignments. This remains unbiased only when all outcomes are observed; ordinary censoring causes a downward shift.

**IPCW IPS**, on identified MAR scenarios, uses

`Z_i = 1(mature_i) X_i / q(x_i,a_i)`

and estimates `V` by `n^{-1} sum_i Z_i`. Conditional independent censoring and positivity give E[Z_i]=V. Since X is bounded by 2, the score is bounded by

`B = max_{x,a: pi(a|x)>0} 2 / q(x,a)`.

The registered scenarios use B=2 (no censoring), B=4 (uniform censoring), and B=10 (cell-dependent censoring). Any missing target-cell support or MNAR status suppresses these metrics.

## Anytime-valid IPCW interval

For identified MAR fixtures, the code maintains a two-sided Hoeffding confidence sequence over the **full assigned-decision prefix**. Missing outcomes contribute a known zero to the IPCW score; matured outcomes contribute X/q. Under the frozen independent-observation assumptions, each score is in [0,B] and has mean V.

At sample size n, the per-tail allocation is

`delta_n = alpha / [2 n(n+1)]`, with total alpha=0.05.

The radius is `B * sqrt(log(1/delta_n)/(2n))`; every interval is intersected with the previous one to enforce nesting. The total of the two tails' spending over n is 0.05. This is time-uniform in sample size under the stated assumptions. It is not a valid correction for MNAR censoring or q=0 on target support.

The available-case Wald interval is kept only as a pointwise diagnostic. It is not used as evidence of sequential validity.

## Metrics

For every fixture and each horizon 128, 512 and 4,096, common metrics report available-case IPS estimate, bias, absolute error, pointwise Wald coverage/width, zero-filled estimate/bias, overall maturity fraction, target-action maturity fraction and mature sample count.

Only in `IDENTIFIED_MAR` fixtures, additional rows report IPCW estimate, bias, absolute error, pointwise and sequence-wide CS coverage, CS width, effective sample size, and the registered IPCW score bound.

Results summarize 1,024 fixed-seed replicates and include Monte Carlo standard errors. These are synthetic averages, not a claim that any estimator dominates on real traffic.

## Hard gates

- exact fixture truth `V=0.775` and exact estimand/weight bounds;
- analytic expectation identity E[Z]=V for all three MAR fixtures;
- an analytic regression that complete-case IPS is biased under the cell-dependent q table;
- positivity-failure and reward-dependent-censoring fixtures emit their corresponding `NOT_IDENTIFIED` status and have **no IPCW metric rows**;
- all IPCW CSs are nested and bounded by [0,1];
- observed IPCW scores respect B;
- pointwise and sequence coverage for the valid IPCW CS pass 0.90 empirical regression threshold in every registered fixture/look;
- deterministic replay, exact metric counts and source/output SHA-256 receipt.

The 0.90 coverage threshold is a simulation gate, not the theorem. Formal validity follows from the bounded-score Hoeffding confidence sequence only when the target cells have positive known q and censoring is independent of reward conditional on context/action.

## Industry comparison and next step

SCOPE-RL documents fixed-sample high-confidence OPE intervals including Hoeffding and empirical-Bernstein bounds, along with off-policy selection metrics:
https://scope-rl.readthedocs.io/en/latest/documentation/evaluation_implementation.html

VIS-DVPC-048 adds a sequential IPCW-specific diagnostic and explicit fail-closed identification states. It is not a replacement for survival models or doubly robust estimators. The next benchmark should add age-dependent hazard / inverse-probability-of-censoring weighting, cross-fitted outcome models, propensity estimation error, and sensitivity bounds for MNAR rather than pretending those cases are identified.
