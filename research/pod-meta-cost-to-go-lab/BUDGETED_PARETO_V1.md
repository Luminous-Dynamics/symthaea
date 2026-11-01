# Budget-Feasible Pareto Action Selection V1

**Status:** deterministic scheduler-side selection primitive for [#9822](https://github.com/Luminous-Dynamics/symthaea/issues/9822). It returns a feasible non-dominated set; it does not execute or authorize an operation.

## Inputs and identity binding

`BudgetedActionContext` binds one decision to the exact pre-state, proof frontier, candidate pool, frozen model, and policy digests. It also contains the complete sorted admissible operation-family list, remaining six-dimensional hard budget, and a minimum completed-proof support count.

Every `CandidateOperationForecast` must match all five context digests. Forecasts must be unique and cover the candidate set exactly. q50/q90/q95 must be monotone in every resource dimension.

## Feasibility

A candidate abstains when its shift status is not `InDomain`, or its support count is below the configured floor. Otherwise the selector computes `remaining_q90 + mandatory_verification_reserve` with checked arithmetic. It is budget-feasible only if this complete vector fits inside the remaining budget in all six dimensions: tokens, tool calls, CPU ms, GPU ms, wall ms, and micro-USD.

Resource dimensions are never summed or converted into a weighted score. Overflow fails closed. The selector does not treat missing support or shift as an infeasibility classification; those are separate abstention dispositions.

## Pareto relation

Among budget-feasible candidates, A dominates B only if A predicts at least as much proof progress and its q90-plus-verifier-reserve cost is no greater in every resource dimension, with at least one strict improvement. Return every non-dominated candidate in canonical operation-family order.

The caller receives the complete assessments and Pareto set. It must apply any further priority/tie-break policy explicitly. This module does not commit an action, reserve provider resources, verify proof completion, or replace the independent budget ledger.

## Qualification fixture

The deterministic CLI matrix emits two `BUDGETED_PARETO_ROW` records. The first asserts that four candidate families result in a three-operation frontier `[2, 3, 5]`; family 4 is rejected because its token q90 exceeds the hard budget. The second asserts exactly one budget-infeasible option and proves that the rejected candidate is not in the returned frontier.

Unit tests additionally cover shift and support abstention, exact context identity matching, duplicate/unknown/missing candidate detection, non-monotone quantiles, all-six-dimension budget checking, mandatory verifier reserve, and checked arithmetic overflow.

## Claim limits

The feasibility gate uses marginal per-resource q90 values. It does not imply 90% simultaneous six-dimensional coverage. That requires a separately qualified joint prediction method and its own calibration assumptions. This selector is not a causal effect estimator or an optimality guarantee; the quality of its frontier depends on the candidate predictions and proof-progress estimates. A synthetic matrix demonstrates invariants, not decision quality or SOTA performance.