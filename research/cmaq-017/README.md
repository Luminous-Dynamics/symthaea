# CMAQ-017 — Endogenous Distribution Shift and Multi-Step Exploitability

**Status:** synthetic v0.1 implementation slice; exact-head Rust qualification is still required.

## Question

Can an optimizer create its own out-of-distribution state/action distribution by repeatedly choosing actions that look favorable under an imperfect predictive model? Can the resulting one-step error compound into optimistic rollouts, false-safe decisions, and worse realized outcomes?

CMAQ-016 isolates a static, one-step reward-optimism pocket. CMAQ-017 adds state, dynamics, rollout horizon, repeated replanning, and hidden regime changes.

## What v0.1 implements

A dependency-free Rust benchmark with a deterministic 2D state `(x, v)`, bounded continuous actions, fixed per-episode disturbance tapes, and an evaluator-only transition law.

Five separately reported lanes:

1. `benign`: matched model / environment control.
2. `state_action_pocket`: a narrow optimistic reward residual activates only for a local state/action conjunction.
3. `compounding_drift`: a small action-correlated transition bias accumulates over rollout.
4. `termination_seam`: the model predicts continuation where the environment terminates and applies a penalty.
5. `regime_migration`: the environment changes its action response after a hidden transition time while the model remains stale.

The report matrix varies:

- Prediction/planning horizons: 1, 4, 8 steps.
- Candidate-sequence budgets: 8, 32, 64.
- Execution blocks: 1 or 4 steps, where the block does not exceed the horizon.
- 16 paired deterministic episodes per condition.

The v0.1 candidate ceiling is deliberately 64 so the first mechanism slice remains inexpensive to reproduce. The preregistered 128/512 stress tiers remain a required extension before making claims about large-search asymptotics.

## Policy arms

- `direct_policy`: feedback controller with no planner candidate/model queries (D0-style baseline).
- `direct_search_control`: action-sequence search around a fixed direct-control prior; it consumes the same candidate-sequence count but does not call a world model. This is a candidate-budget control only, not a true compute-matched D1 baseline because FLOPs, latency, and hardware work are not equalized.
- `random_point_search`: random candidate sequences scored by the point model.
- `adaptive_point_cem`: cross-entropy search using the point model.
- `pessimistic_ensemble_cem`: minimum predicted return across three heads.
- `uncertainty_lcb_cem`: mean predicted return minus 1.25 times the three-head return standard deviation.
- `shuffled_future_negative_control`: scores the time-reversed action sequence, then executes the original sequence; this deliberately breaks action/future alignment.
- `oracle_cem`: evaluator-only mean transition law, with the same candidate and planning horizon budgets. It does not reveal the fixed disturbance realization; treat it as an oracle-dynamics comparator, not a mathematical global upper bound or a deployable policy.

The three ensemble heads have distinct local pocket centers in the pocket lane. They share the other misspecifications by construction. This is important: ensemble size is not treated as evidence of independent error. The LCB is a synthetic ablation, not a calibrated uncertainty method.

## Metrics

Every row is keyed by scenario, horizon, execution block, candidate budget, and policy. No pooling across these dimensions is allowed.

- Realized episode return and a paired-episode 95% normal-approximation interval.
- Lower-decile mean and worst-episode return. Sixteen episodes are not enough to make a stable CVaR95 estimate; these are tail diagnostics, not a CVaR95 claim.
- Paired return deltas versus the direct policy (D0), candidate-budget direct-search control, and shuffled-future negative-control (W2). True D1 compute matching remains unimplemented and requires measured FLOPs/forward passes/wall-clock budgets.
- Paired return gap to the oracle arm.
- Predicted-minus-realized return over the executed block, where the planner emitted a model prediction.
- Fraction of visited states outside a declared rectangular training-support proxy.
- Structured 201-point one-step state-transition probe RMSE and selected-trajectory one-step state RMSE against the reference point model. The probe set has uniform marginal coverage on x, velocity and action, but is a one-dimensional modular design rather than a full Cartesian state-action grid; joint coverage is limited. The two errors are intentionally reported side by side.
- Reference-model false-safe rate when the point model predicts nontermination but the environment terminates.
- Candidate-sequence queries, model-head evaluations, learned-model transition evaluations, oracle transition evaluations, and evaluator-only diagnostic transitions, separately. Transition calls are counted at the rollout step where they occur, so terminal truncation reduces the recorded count rather than being charged as a full horizon.

The false-safe and one-step error fields are evaluator-side diagnostics applied to every policy trajectory. They are **not** claims about a policy's own alarm or calibration. The support box is a controlled synthetic proxy, not a learned density estimate or a formal OOD detector.

## Reproduction

From this directory:

```sh
cargo test --locked
cargo run --release --locked > /tmp/cmaq-017-report.json
```

The focused workflow runs formatting, tests, Clippy, report generation, and structural/report invariants on the exact pull-request head.

## Integrity and comparison rules

- All policies receive paired episode seeds, the same true environment law, action bounds, environment horizon, and candidate-sequence budget where they use search.
- Candidate-query counts are separate from model-head evaluations. A three-head ensemble costs three model-head evaluations per candidate.
- Oracle transition calls are separate from learned-model calls.
- Direct and direct-search control arms report no model forecast or optimism gap; they are not assigned invented prediction scores. The direct-search arm matches the number of candidate sequences, not actual compute; do not cite it as D1.
- Early termination is retained in the episode denominator. No failed/terminal rollout is silently dropped.
- The benchmark emits each error-family/horizon/execution-block slice separately. A strong mean cannot hide a failed slice.
- The benchmark is synthetic and does not include a trained Symthaea model, a real simulator, hardware timing, a robot, or an external baseline reproduction.

## Claim ceiling

A passing synthetic run can establish that this controlled generator, scorer, metrics, and negative controls behave as specified. It cannot establish physical safety, world-model superiority, generalization, industry leadership, or SOTA.

A comparative claim requires the CMAQ-011 external bridge: pinned task and simulator revisions, paper-native and wrapper-normalized modes kept distinct, common action cadence/reward semantics, matched training and inference budgets, held-out tasks/error families, paired seeds, and independent reproduction.
