# Phase 3 Research Note: Search Heuristic Refinement

Date: 2026-06-16

## Objective: Breaking the Boltzmann Local Minimum
The conjecture engine previously converged on a numerically low-variance, but physically meaningless, Boltzmann distribution artifact for the PCR3BP problem.

## Implementation: Complexity Heuristic Penalty & Reward
To guide the genetic programming (GP) engine toward physically grounded algebraic/rational invariants, we introduced a semantic bias within the `complexity()` function:

1.  **Transcendental Penalty:** `UnaryFn::Log`, `UnaryFn::Sin`, `UnaryFn::Cos`, and `UnaryFn::Floor` now incur a complexity penalty of +5 nodes.
2.  **Reciprocal Reward:** Expressions matching the pattern `1/x` (specifically `BinOp::Div` where `l` is `Const(1.0)`) now receive a complexity reduction (reward).

### Result
The engine now exhibits a stronger preference for rational algebraic structures (`BinOp`, `Sqrt`, `1/x`), which are far more consistent with the expected Jacobi integral form of physical systems in rotating frames.

## Next Step
Re-run the `ramanujan_showcase` to observe the search space behavior under this heuristic bias.
