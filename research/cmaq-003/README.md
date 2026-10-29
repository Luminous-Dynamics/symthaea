# CMAQ-003: Uncertainty-Aware Planning and Selective Trust

**Status:** standalone Rust evaluator prototype; no Symthaea agent results or qualification claims.

## Research question

Can a planner improve closed-loop decision quality by treating uncertainty as a planning constraint rather than a report-only diagnostic?

CMAQ-003 compares two receding-horizon planners on the same finite deterministic environment:

- **Point planner:** selects from mean reward predictions and point safety probabilities; it does not consider support counts or reward uncertainty.
- **Trust-region planner:** requires an evaluator-owned minimum support count, subtracts a configurable fraction of the predicted reward uncertainty radius, and requires the lower safety score (predicted safety probability minus its uncertainty radius) to clear the safety threshold.

Both planners execute only the first action of a predicted plan in the ground-truth environment, observe the true next state and replan. This is a controlled mechanism experiment, not a universal planning algorithm.

## Run

From this directory with Rust 1.96.0:

- Tests: `cargo test --locked`
- Exact-model fixture: `cargo run --locked -- --truth fixtures/truth.tsv --predictions fixtures/predictions-exact.tsv --support fixtures/support.tsv --starts fixtures/starts.tsv`
- Optimism trap: replace `predictions-exact.tsv` with `predictions-exploit.tsv` in the command above.

The evaluator emits both policy rows for every task. It rejects duplicate/missing action keys, malformed booleans, out-of-range probabilities/risk settings, and predicted successor states absent from the evaluator-defined task state space.

## Input and leakage boundary

- `truth.tsv`: evaluator-owned deterministic next state, reward, safety and terminal flag for every state-action pair.
- `predictions-*.tsv`: candidate-provided next-state/reward/safety point forecasts and reward/safety uncertainty radii.
- `support.tsv`: evaluator-owned support count for each state-action pair. It must not be candidate self-report.
- `starts.tsv`: initial state, horizon, discount, safety threshold, support floor, and reward-risk coefficient, frozen before scoring.

The truth file is used only by the evaluator and is never passed to the planner. Counterfactual truth for unexecuted actions is legitimate only in this synthetic simulator lane; for real environments, record observed outcomes only and do not invent unchosen outcomes.

## Metrics

- **Realized return** and **constrained return**; an actually unsafe action contributes zero to constrained return and still fails a hard gate.
- **Oracle safe return**: evaluator-only upper bound using ground-truth transitions and rewards, safe actions only, and abstention utility zero.
- **Constrained regret**: positive part of oracle-safe return minus realized constrained return.
- **Risk-adjusted plan value** and **model-to-rollout gap**: predicted risk-adjusted value before execution versus actual constrained return after closed-loop replanning.
- **Unsafe executed actions** (non-compensatory hard gate) and **unsupported executed actions** (using evaluator-owned support metadata).
- **Executed-action reward MAE**, executed transition mismatch count and an all-action safety Brier score.
- **Reward interval coverage**: proportion of all per-task action rewards lying within prediction mean ± radius, plus average interval width.
- **False-safe predictions**: count of actual unsafe actions accepted by that planner's point or conservative safety rule.

The uncertainty radii are candidate-supplied heuristic intervals. This prototype does not establish conformal coverage or a formal safety guarantee. Observed interval coverage is descriptive only; calibration data and qualification data must be separated in any learned system.

## Negative controls and qualification protocol

The fixtures include an exact-model control and an optimizer trap in which a point planner sees a reward of 9,000 for an unsupported branch whose real first-step reward is 2,000. A larger uncertainty radius and the support gate allow the trust-region planner to avoid the branch. That is a hand-constructed known-answer sanity test, not evidence that a learned uncertainty estimator will behave this way.

Before evaluating Symthaea, pair all eligible systems on identical tasks, seeds, action budgets and horizons. Required comparisons: constant/abstain prior; nearest-support lookup; point planner; support-gated planner; uncertainty-only planner; support plus uncertainty; learned dynamics model; and the current Symthaea integration. Include HDC-only, recurrence-only, explicit-memory-only, world-model-only and planner ablations.

Pre-register:
1. At least 10 development seeds and three independent causal-law families, then an evaluator-controlled qualification split.
2. Horizons 1, 3, 5 and 10; support floors and reward-risk multipliers selected on development data only.
3. Task/mechanism-clustered confidence intervals, worst-family results and complete resource costs.
4. Shuffled reward labels, corrupted transitions, overconfident false-safe predictions, and artificially inflated uncertainty on safe supported actions.
5. Risk-coverage curves: plot unsafe rate and constrained regret against coverage/abstention rate, not a single preferred threshold.

Do not claim OOD safety from lower average error or nominal interval coverage. Explicitly measure calibration under shift, selective risk, utility, and abstention costs.

## External comparison and research basis

This work complements CMAQ-001's receipt-bound one-step scorecard and CMAQ-002's closed-loop exploitation evaluator. The 2026 world-model literature emphasizes that visual/prediction quality is not equivalent to closed-loop utility, and calls out compounding error, uncertainty calibration and planner exploitation. See the ICLR 2026 `World-In-World` closed-loop benchmark (https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html), the 2026 work on robust conformal prediction for safe planning under environment shift (https://proceedings.mlr.press/v331/rahaman26a.html), and the 2026 world-model survey (https://link.springer.com/article/10.1007/s44163-026-02122-1).

Run published external benchmarks under their own protocols and report scores separately. CMAQ-003 is a compact causal mechanism test, not a replacement for external robotics/world-model benchmarks.

## Limitations

The harness currently uses deterministic finite discrete state/action spaces, candidate-supplied uncertainty radii, an evaluator-known simulator and a hand-built support floor. It has no stochastic transition kernel, calibration fit, online conformal set update, continuous control, model ensemble, external benchmark results, or real-world receipts. There is no local compile/test result in this authoring environment. The smoke fixtures are expected outputs only; no Symthaea, SOTA, production-safety or empirical improvement claim is made.
