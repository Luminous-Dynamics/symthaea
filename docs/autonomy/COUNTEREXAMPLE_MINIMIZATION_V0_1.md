# Counterexample-preserving intersection scenario minimization v0.1

**Status:** deterministic minimizer implementation; no Symtropy execution results reported  
**Code:** `symthaea_vehicle::scenario_discovery::minimize_intersection_counterexample`  
**Execution owner:** Symtropy #3387  
**Search / interpretation owner:** Symthaea #10043

## Purpose

A discovered failure should be reduced to a small, stable, understandable reproducer before it becomes a regression case. This module adds a bounded coordinate shrinker for the seven-factor synthetic pedestrian-crossing candidate.

The caller provides a named failure predicate and an evaluator callback. In the intended architecture, that callback submits each proposal to Symtropy's execution/replay service and returns a receipt binding the exact candidate ID, predicate ID, evaluator identity, execution-receipt ID, execution digest and predicate result. Symthaea does not duplicate simulator or actor physics.

## Algorithm

1. Verify the source candidate's schema, factor bounds and content identity.
2. Ask the evaluator whether the original source still exhibits the registered predicate. Reject minimization if it does not.
3. Preserve the source generator identity; record `counterexample-coordinate-minimize-v0.1` as the separate minimization policy in the result report. Candidate identity continues to encode origin policy plus exact current factor values.
4. Visit factors in a fixed order: ego speed, crossing onset, occlusion, sensor delay, dropout, localization bias, brake actuation delay.
5. Move each factor toward its declared nominal target. When the target preserves the predicate, accept it. Otherwise perform up to 12 deterministic bisection evaluations between the last preserving and first non-preserving values.
6. Stop when the coordinate pass completes or the total evaluation budget (maximum 256) is consumed.
7. Emit a new candidate identity (when factors changed) and a report binding the full parent candidate, parent ID, predicate ID, factor changes, every exact candidate probe and its receipt, evaluation count, minimized candidate and report digest.

Nominal targets are:
- midpoint for ego speed and crossing onset;
- zero occlusion, sensor delay, sensor dropout, localization bias and brake delay.

These are minimization preferences, not probability assumptions or declarations of physical plausibility.

## Guarantees and non-guarantees

The minimizer preserves the failure predicate only to the extent that the caller's evaluator accurately returns the named predicate for each exact candidate. The final accepted candidate was evaluated as preserving that predicate.

It is a **local coordinate shrinker**, not a global optimizer or proof of minimality. Non-monotonic predicates can have disconnected failure regions; a one-dimensional bisection can only find a boundary along the current coordinate segment. The fixed factor order can also affect the final local result.

The report stores one receipt per evaluated candidate, including rejected probes. Each receipt's candidate and predicate IDs must match the exact query and it carries evaluator and execution identifiers plus a 64-hex execution digest. The report validator checks receipt lineage and that the final minimized candidate was observed with the predicate preserved. These fields still do not authenticate an executor, validate a signature or prove independent replay. BLAKE3 output digests are integrity fingerprints, not signatures.

## Controls

Unit tests are authored for:
- shrinking multiple factors while preserving a synthetic predicate;
- parent lineage and candidate identity;
- invalid source, empty predicate and invalid budget rejection;
- rejection of an evaluator receipt bound to a different candidate, including a matching ID with mutated embedded scenario fields;
- rejection of mixed evaluator identities within one minimization run;
- report tampering even after recomputing the report digest.

These tests are not compiled or executed in this session.

## Next progression

- Integrate the evaluator callback with Symtropy #3387 and retain one exact run artifact per queried candidate.
- Add fresh-process replay of the minimized result and its parent.
- Add failure-family and trigger-predicate schemas before using minimized cases as a benchmark.
- Compare coordinate minimization with delta debugging and hierarchical / group-wise shrinkers under equal execution budgets.
- Add a holdout replay set for checking whether minimization overfits one simulator/runtime.
