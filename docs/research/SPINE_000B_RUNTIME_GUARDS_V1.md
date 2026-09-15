# SPINE-000B — Runtime Guard Semantics v1

**Status:** preregistered measurement contract; runtime witness capture pending

**Authority:** measurement-only

**Issues:** #3367, #3369, #3370, #3372, #3373, #3374, #3375, #3376, #3377

This contract freezes the runtime predicates that may remain after an R2 source/feature projection reaches a registered Phase-C operation path. It does not execute or re-evaluate those guards.

## 1. Core separation

```text
R2 source / lowering projection
!=
runtime guard satisfaction
!=
actual operation execution
!=
observed destination state change
!=
subsystem causal attribution
```

G1 exists so a guarded operation that legitimately does not execute cannot be misclassified as a missing A1 receipt.

## 2. Guard kinds

Every operation in the frozen Phase-C registry appears exactly once in the overlay as either:

- `UNCONDITIONAL_AFTER_SOURCE` — after the R2 source/feature condition is satisfied, no additional runtime predicate guards the registered operation;
- `ALL_RUNTIME_PREDICATES` — every listed runtime predicate must be witnessed true before the operation is expected to execute.

Unknown guard kinds fail closed.

## 3. Witness status is not just Boolean

Future runtime guard witnesses use four semantic states:

```text
TRUE
FALSE
NOT_EVALUATED
UNRESOLVED
```

`NOT_EVALUATED` is not `FALSE`. If an upstream predicate prevents a downstream production expression from being evaluated, the downstream predicate must remain `NOT_EVALUATED`.

`UNRESOLVED` means the evidence profile did not obtain a qualified witness. It cannot be promoted to true or false from downstream inference.

## 4. Frozen predicates

The v1 overlay assigns append-only `u16` predicate IDs:

1. `vision_bridge_present`
2. `geodesic_path_nonempty`
3. `decoded_geodesic_frames_nonempty`
4. `node_id_present`
5. `consciousness_hv_present`
6. `intent_hv_present`
7. `network_service_present`

Existing IDs never change meaning and are never reused.

## 5. Geodesic dependency chain

The guarded path is:

```text
vision_bridge_present
  -> vision.select_best_geodesic
  -> geodesic_path_nonempty
  -> decode_geodesic_to_frames_improved
  -> decoded_geodesic_frames_nonempty
  -> vision.populate_mental_movie
```

`quality.set_request_geodesic` is unconditional after its source/feature condition. `quality.clear_request_geodesic` is likewise unconditional after its clear-source/feature condition.

A false or unresolved upstream predicate does not authorize re-evaluation of downstream guards by the observer.

## 6. Broadcast evaluation semantics

The first three broadcast resource expressions are elements of one Rust tuple expression:

```text
self.node_id()
self.consciousness_hv()
self.last_intent_hv()
```

Rust evaluates those tuple elements before the tuple pattern match; they are therefore treated as one eager evaluation group, not a left-to-right short-circuit chain.

Only after all three are present does production evaluate:

```text
self.network_service()
```

Therefore `network_service_present` depends on predicate IDs 4, 5 and 6 all being true. If the tuple match fails, predicate 7 is `NOT_EVALUATED`, not false.

## 7. Candidate classification

Given an R2 operation projection and feature status:

- no source projection -> `NO_SOURCE_PROJECTION`;
- inactive Cargo/profile feature -> `FEATURE_INACTIVE`;
- unconditional operation -> `UNCONDITIONAL_CANDIDATE`;
- all required runtime predicates true -> `GUARDED_TRUE_CANDIDATE`;
- an evaluated required predicate false -> `GUARDED_FALSE_NOT_EXPECTED`;
- required guard evidence unresolved or not evaluated without a decisive false result -> `GUARD_OUTCOME_UNRESOLVED`.

A dependent predicate recorded TRUE/FALSE while one of its dependencies is not witnessed true is invalid evidence and fails closed.

## 8. Runtime witness capture law

A future witness layer must capture predicate status from the same production-local value/result that controls the real branch. It must not:

- call the resource accessor a second time;
- recompute a geodesic or decoded frame set;
- infer guard truth from downstream state;
- execute user-controlled predicate expressions;
- allocate unbounded data, hash, serialize or perform I/O at the branch seam;
- consume RNG merely for evidence.

Predicates that become knowable only after a real operation use the real operation result, e.g. path and decoded-frame emptiness.

## 9. A1/A2 interpretation

A1 actual application receipts prove only actual execution capture.

A missing A1 receipt counts as an observer-completeness discrepancy only when:

1. R2 projects the source/feature path; and
2. G1 classifies the operation as `UNCONDITIONAL_CANDIDATE` or `GUARDED_TRUE_CANDIDATE`.

`GUARDED_FALSE_NOT_EXPECTED` is a legitimate non-execution path. `GUARD_OUTCOME_UNRESOLVED` is neither a completeness PASS nor FAIL for that operation.

The counterfactual `I_withoutS` branch remains a deterministic projection. G1 does not turn it into executed counterfactual evidence.

## 10. Qualification boundary

The checked JSON overlay and verifier may establish only:

- exact coverage of the frozen Phase-C operation registry;
- stable predicate identities and dependencies;
- source-anchor consistency for guarded predicates;
- deterministic candidate classification on synthetic truth-table controls.

A green G1 exact-head run does **not** establish runtime predicate truth, application execution, state change, subsystem causality, benefit, or authority.
