# Temporal Joint Control (candidate side)

**Package:** `symthaea-temporal-joint-control`  
**Status:** research reference implementation; Rust tests authored but not compiled or executed in the current authoring environment  
**Tracking:** NEURO-SILICON-009 / #9666

## Boundary

This crate receives candidate-owned predictions, candidate action plans and typed resource estimates. It contains no hidden future stream, evaluator outcome, oracle seed, receipt signing key or evaluator oracle. It is safe to import from a candidate process only after the caller has constructed its plans from candidate-visible data.

The empty action list is the no-op baseline. Action IDs must be strictly increasing and stable across all alternatives for the same episode.

## Joint plan contract

A plan is a complete proposed combination of observation, retention/eviction, refresh, computation, verification, abstention or deferral actions. Each complete plan carries:

- predicted reductions in wrong-belief, abstention and missing-output exposure, kept as three separate signed microsecond estimates;
- a twelve-dimensional estimated resource vector;
- a predicted maximum observation gap.

The cost dimensions are persistent bytes, retained byte-microseconds, read bytes, write bytes, compute microseconds, refresh operations, observations, sensor-active microseconds, latency microseconds, energy nanojoules, verification operations and receipt bytes.

All dimensions must be populated by adapters. Zero means a declared zero estimate; unknown is not zero and must be rejected before constructing a plan.

## Selection

1. Reject plans that exceed any resource ceiling or the maximum observation gap.
2. Compute the non-dominated frontier across all three predicted exposure reductions and all twelve resource dimensions.
3. Select deterministically by the caller's explicit exposure-axis priority. Resource use, observation gap and plan ID are deterministic tie-breakers.
4. Return all feasible, Pareto and rejected plan IDs so the choice is auditable.

Hard budgets are never scalarized into a soft penalty. A plan with an enormous predicted gain cannot buy its way through a memory, energy, latency, sensing or verification ceiling. Exposure axes are not silently collapsed into a single leaderboard score.

This v0.1 controller does **not** implement online dual-price adaptation, learned value calibration, a plan generator, or a globally optimal schedule over continuous action spaces. It selects among the complete alternatives supplied by the candidate policy.

## Local validation

From the repository root:

```sh
cargo test --manifest-path research/temporal-joint-control/Cargo.toml
```

The package intentionally has no third-party dependencies. This research workspace does not yet contain a committed Cargo.lock; before confirmatory use, generate and commit the lockfile, then run with --locked. A successful test run is still required before any test is described as passing.
