# CMAQ-008: Sequential Support-Drift Detection and Model-Use Revocation

**Status:** protocol v0.1 + executable Rust smoke harness; draft  
**Tracking issue:** [#9058](https://github.com/Luminous-Dynamics/symthaea/issues/9058)  
**Claim ceiling:** synthetic sequential-monitor benchmark only; no production, safety certification or SOTA claim

## Why this is a separate slice

CMAQ-007 demonstrated two related failure modes in deliberately constructed panels: larger planner search can select unsupported optimistic actions even when catalog-average prediction MAE changes little, and a support threshold calibrated on one score distribution can exceed its nominal unsupported-selection and fallback-downside targets after the unsupported actions' scores shift upward. A static calibration rule cannot, on its own, know that its assumptions have stopped holding.

CMAQ-003 measures causal regime-change detection. CMAQ-006 makes model use conditional on freshness and held-out decision-value evidence. CMAQ-008 tests a narrower online control contract: **monitor the stream of post-action support outcomes, revoke model authority when sequential evidence contradicts the calibrated baseline, and measure the cost of both delayed detection and false revocation**.

This v0.1 smoke harness models the outcome feedback itself rather than a learned support estimator. It assumes a trusted evaluator can reveal whether the selected action was unsupported after that action executes. That is a strong, often unrealistic observability assumption; later work must test delayed, noisy, missing and adversarially corrupted feedback.

## Executable panel

The dependency-free Rust CLI runs **512 paired deterministic episodes per scenario**, each 128 decisions long. Every policy sees the same exogenous random tape for a given episode, allowing paired utility comparisons. A candidate action under the model gate has utility +0.03 when supported and −0.20 when unsupported; direct fallback utility is 0. The unsupported-outcome probability is:

- **Stationary null:** 1% through all 128 decisions.
- **Abrupt support shift:** 1% through step 63, then 25% from step 64.
- **Gradual support shift:** 1% through step 31, rises linearly from step 32 through step 95, and reaches 25% at step 96.

These rates and payoffs define a synthetic falsification mechanism. They are not learned or measured estimates from Symthaea, and numbers must not be transferred to a real environment.

## Policies

- **Direct fallback:** never delegates a decision to the model. This exposes its zero-risk / zero-coverage utility baseline.
- **Fixed gate:** uses model actions for all 128 steps. This isolates the cost of never revoking stale model authority.
- **First unsupported revocation:** immediately falls back after the first unsupported outcome. This is an intentionally overreactive baseline; it tests the cost of treating every adverse event as a shift.
- **CUSUM revocable gate:** maintains a one-sided log-likelihood CUSUM for a frozen null rate (p_0=0.01) against a shifted rate (p_1=0.25). Each supported/unsupported post-action outcome updates the score. It revokes future model use at statistic (ge 5.0). Threshold and likelihoods are fixed before running the panel.
- **Oracle changepoint:** knows the hidden shift onset and switches to fallback immediately at that point. It is an evaluator-only upper-bound comparator, not an eligible method.

A revocation is applied **after** the observation that triggered it; that action's utility and unsupported outcome remain in the totals. Detection delay includes the triggering decision. The monitor does not receive a changepoint label.

## Metrics

Every scenario/policy row reports:
- mean cumulative utility and paired utility delta versus fixed gate, with a normal-approximation 95% interval over episode-level paired differences;
- model-action coverage;
- unsupported outcomes per model action and per 1,000 total decisions;
- episode revocation rate and false-revocation rate (pre-shift alarms on changing scenarios, all monitor revocations in stationary scenarios);
- shift-detection coverage and conditional p90 detection delay (−1 means no detections);
- descriptive mean regret, CVaR95 regret (average of the worst ceil(5% of episodes)) and maximum regret versus the oracle policy on paired exogenous tapes.

The oracle's per-episode realised outcome is not guaranteed to dominate every stochastic policy on every finite random tape, so regret uses the positive part of oracle-minus-policy cumulative utility. Tail metrics are descriptive on this finite synthetic panel, not confidence bounds for a population.

## Initial frozen targets (hypotheses, not results)

1. Stationary false-revocation rate ≤ 5% of episodes for CUSUM.
2. Abrupt-shift detection coverage ≥ 90%, with conditional p90 detection delay ≤ 16 decisions.
3. CUSUM improves mean paired cumulative utility versus the fixed gate under the abrupt shift; report the paired interval and retain the result if the interval includes zero.
4. First-error revocation has higher stationary false-revocation rate than CUSUM on the same tape; preserve this baseline to quantify overreaction.
5. Report gradual-shift results separately. Passing the abrupt target does not qualify the gradual lane.

These are smoke-panel qualification checks, not industry/SOTA targets. Any missed target is an engineering finding, not a reason to tune on the same seeds.

## Run

From this directory:

```sh
cargo test --locked
cargo run --quiet --locked
```

The focused workflow validates strict report schema, row uniqueness, fixed/fallback behavior, stationary false-alarm budget, abrupt detection and delay, paired utility lift, gradual-lane output and risk/coverage accounting. Rust must compile and the exact-head workflow must pass before reporting benchmark outputs as Rust evidence.

## Limitations and next work

- Feedback labels are instantaneous and trusted; real systems may have delayed, partial or disputed outcomes.
- The simulator has no state transitions, candidate ranking or learned world model.
- Only one fixed likelihood pair and threshold are tested; no adaptive threshold is tuned on qualification episodes.
- Three synthetic lanes do not cover recurrent regimes, covariate drift, correlated feedback, adversarial support-score changes or hidden mechanism families.
- The harness establishes no production guarantee and has not yet integrated with CMAQ-003/006 or the live Symthaea planner.

Next stage: add feedback delay/drop/corruption, threshold-independent anytime-valid evidence monitoring, recurrent-shift/recovery lanes, and a trusted event-envelope boundary. Then compare to a sequential Bayesian changepoint or Page–Hinkley baseline on held-out environment families, with identical action, compute and observation budgets.
