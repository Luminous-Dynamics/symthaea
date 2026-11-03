# Counterexample-preserving intersection scenario minimization v0.1

**Status:** deterministic minimizer implementation; no Symtropy execution results reported  
**Code:** `symthaea_vehicle::scenario_discovery::minimize_intersection_counterexample`  
**Execution owner:** Symtropy #3387  
**Search / interpretation owner:** Symthaea #10043

## Purpose

A discovered failure should be reduced to a small, stable, understandable reproducer before it becomes a regression case. This module adds a bounded coordinate shrinker for the seven-factor synthetic pedestrian-crossing candidate.

The caller provides a named failure predicate and an evaluator callback. In the intended architecture, that callback submits each proposal to Symtropy's execution/replay service and returns true only when the exact registered failure predicate still holds. Symthaea does not duplicate simulator or actor physics.

## Algorithm

1. Verify the source candidate's schema, factor bounds and content identity.
2. Ask the evaluator whether the original source still exhibits the registered predicate. Reject minimization if it does not.
3. Switch the candidate's generator policy identity to `counterexample-coordinate-minimize-v0.1`, preserving source seed, ordinal, generation budget, geometry and actor script.
4. Visit factors in a fixed order: ego speed, crossing onset, occlusion, sensor delay, dropout, localization bias, brake actuation delay.
5. Move each factor toward its declared nominal target. When the target preserves the predicate, accept it. Otherwise perform up to 12 deterministic bisection evaluations between the last preserving and first non-preserving values.
6. Stop when the coordinate pass completes or the total evaluation budget (maximum 256) is consumed.
7. Emit a new candidate identity and a report binding parent candidate ID, predicate ID, budget, evaluation count, changed factors, minimized candidate and report digest.

Nominal targets are:
- midpoint for ego speed and crossing onset;
- zero occlusion, sensor delay, sensor dropout, localization bias and brake delay.

These are minimization preferences, not probability assumptions or declarations of physical plausibility.

## Guarantees and non-guarantees

The minimizer preserves the failure predicate only to the extent that the caller's evaluator accurately returns the named predicate for each exact candidate. The final accepted candidate was evaluated as preserving that predicate.

It is a **local coordinate shrinker**, not a global optimizer or proof of minimality. Non-monotonic predicates can have disconnected failure regions; a one-dimensional bisection can only find a boundary along the current coordinate segment. The fixed factor order can also affect the final local result.

The output digest is a BLAKE3 integrity fingerprint, not a signature, authenticated execution receipt or proof of independent replay. Candidate parent lineage is explicit, but the caller must retain the Symtropy execution receipt associated with each evaluation.

## Controls

Unit tests are authored for:
- shrinking multiple factors while preserving a synthetic predicate;
- parent lineage and candidate identity;
- invalid source, empty predicate and invalid budget rejection;
- report tampering even after recomputing the report digest.

These tests are not compiled or executed in this session.

## Next progression

- Integrate the evaluator callback with Symtropy #3387 and retain one exact run artifact per queried candidate.
- Add fresh-process replay of the minimized result and its parent.
- Add failure-family and trigger-predicate schemas before using minimized cases as a benchmark.
- Compare coordinate minimization with delta debugging and hierarchical / group-wise shrinkers under equal execution budgets.
- Add a holdout replay set for checking whether minimization overfits one simulator/runtime.
