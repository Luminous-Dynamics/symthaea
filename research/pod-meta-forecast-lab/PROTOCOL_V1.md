# POD-Meta Forecast Calibration Protocol v1

**Status:** synthetic protocol and qualification harness  
**Seed:** 1729  
**Training:** 2,048 observations  
**Evaluation:** 1,280 observations  
**Selection:** 256 episodes per evidence regime, seven policy arms  
**Resource constraint:** 3 abstract work units per episode

## 1. Hypothesis

A scheduler should forecast the probability that a candidate operation will discharge an unresolved proof obligation, and should allocate compute using that forecast only to the extent that the forecast is calibrated on data outside its fitting set.

This is distinct from asking whether an operation's advertised claim is true. The forecast is made before execution; an independent evaluator supplies the binary outcome afterward.

## 2. Splits and information boundaries

Training covers four regimes: stable, temporal drift, conflicting sources, and thin evidence. Evaluation uses a separate deterministic seed and includes those regimes plus held_out_shift, which is absent from training.

The fitting model stores only aggregate training counts. No held-out evaluation label enters the predictor. The held-out regime forces hierarchical fallback to operation-family priors and makes the cost of context novelty visible.

This v1 is seed-held-out and context-held-out, not task-family-held-out. It uses a public simulator, not a protected evaluator. It is suitable for checking metrics and causal plumbing, not for a generalization claim.

## 3. Forecasting arms

- Constant 0.5 probability.
- Global Beta-smoothed rate.
- Operation-family rate shrunk toward the global posterior.
- Context-by-operation rate shrunk toward the family posterior.

The hierarchical estimates are intentionally simple and auditable. A learned predictor, online calibration, conformal bounds, and regime-shift detector are future variants, not part of this v1.

## 4. Proper scores and calibration

For binary outcome y and forecast p:

- Brier score: mean of (p - y)^2.
- Log loss: mean of -[y log(p) + (1-y) log(1-p)], with probabilities clamped away from exactly zero and one.
- ECE-10: sum over ten fixed-width probability bins of bin frequency multiplied by the absolute gap between mean forecast and observed frequency.
- Reliability table: each bin's count, mean forecast, and observed outcome frequency.

Brier and log loss are strictly proper scoring rules for probabilistic forecasts. ECE is a descriptive, bin-dependent calibration statistic, not a proper scoring rule and not a standalone promotion criterion. Scores are reported for all evaluation cases, seen regimes only, and held-out shift only. Sparse-bin uncertainty is not hidden by the aggregate.

## 5. Decision experiment

Each episode has one unresolved logical proof obligation and four alternative operation families. Each family has a synthetic cost and an independently sampled success outcome. Policies rank operations by predicted probability divided by cost, then attempt only actions that fit in the 3-unit hard budget. A success is counted only when the simulator outcome is true; a forecast alone cannot create progress.

The same episode outcomes are reused across all policies. The report includes verified proof rate, proofs per work unit, work units per verified proof, total work, and budget violations. Fixed-order and random are controls. The oracle uses the simulator's true probability table and is an unattainable diagnostic upper bound, not a deployable baseline.

## 6. Relationship to external work

- [BAGEN: Towards Budget-Aware Agents](https://ragen-ai.github.io/bagen/) measures progressive interval estimates of remaining internal/external budgets and feasibility. This lab measures operation-success probabilities and a small decision problem; it does not reproduce BAGEN's tasks or interval metrics.
- [HARP: Dynamic Budget Allocation for LLM Evaluation under Hard Resource Constraints](https://arxiv.org/abs/2610.07362) studies predictive bounds and adaptive reallocation under deterministic aggregate budgets. This lab enforces a per-episode hard cap but does not implement HARP's finite-sample coverage guarantees or time-to-event estimator.
- [How Proper Scoring Rules Shape LLM Forecasting](https://arxiv.org/abs/2608.28482) reports that different proper scoring objectives can produce different calibration/discrimination profiles. This motivates reporting Brier and log loss separately; this lab does not train an LLM forecast model.
- [METR: Metrics of Agent Ability](https://evals.alignment.org/notes/2026-07-24-metrics-of-model-ability/) recommends examining score as a function of expenditure rather than relying on one unconstrained success score. This lab reports separate cost and outcome dimensions, but its synthetic work units are not real expenditure.

## 7. External comparison protocol

Before claims against strong systems:

1. Pin one external benchmark version and its official evaluator commit.
2. Match the underlying model, prompt, tool permissions, context, decoding, deadline, and cost cap.
3. Collect candidate forecasts before tool execution and append immutable outcome receipts only after independent verification.
4. Train/calibrate on earlier task families and evaluate on disjoint task families and later time windows.
5. Report Brier, log loss, reliability curves, calibration by operation family, and selective risk/coverage.
6. Report full-episode cost: model calls, input/output tokens, tool calls, verifier calls, wall-clock latency, CPU/GPU time, memory, energy method, API spend, and human correction.
7. Plot native task success and safety against measured expenditure as a Pareto frontier. Do not collapse everything into a scalar.
8. Preserve hard constraints: no unverified proof progress, no unsafe action, no budget overshoot, and no test-label leakage.

No numerical claim in this protocol should be presented as a result on BAGEN, HARP, OSWorld, SWE-bench, or any other external benchmark.
