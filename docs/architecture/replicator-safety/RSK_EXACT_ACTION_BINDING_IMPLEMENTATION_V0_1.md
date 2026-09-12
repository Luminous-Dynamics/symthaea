# RSK Exact Action Binding — Implementation Mapping v0.1

**Status:** candidate executable mapping; not production evidence until CI executes  
**Class:** A safety-critical authority semantics  
**Scope:** abstract authorization/resource accounting only

This document maps the exact-action contract and ADR-003 to the candidate Rust implementation. It does not specify any physical replication process.

## Invariant

```text
committed_action == authorized_action == evaluated_action
```

The concrete field implemented in this tranche is:

```text
committed_resource_units == authorization.requested_resource_units
                             == request.budget.requested_resource_units
```

## Evaluation binding

`evaluate_bound_replication_authority` receives the already constructed `ReplicationAuthorityRequest`. Only after the constitutional evaluator returns `Allow` does it construct `BoundedReplicationAuthorization`.

The opaque authorization now retains:

```text
requested_resource_units: request.budget.requested_resource_units
```

The field is private. Callers can inspect it through the read-only `requested_resource_units()` accessor but cannot construct or mutate the authorization value directly.

## Commit binding

`ReplicationLedger::commit_descendant` performs, in order:

1. stale-cursor / duplicate-mutation validation;
2. exact resource equality check;
3. only if equal, runtime witness and negative-state checks;
4. hard/ancestral budget checks;
5. event append and authoritative state mutation.

A mismatch returns:

```text
LedgerError::ActionResourceMismatch {
    authorized,
    committed,
}
```

After equality succeeds, the implementation shadows the caller parameter and uses the authorization-bound amount as the source of truth for all later accounting and event recording.

## No-mutation theorem for mismatch

The unit and black-box tests establish the intended pre-commit theorem for both downward and upward substitution:

```text
cursor_after          == cursor_before
event_count_after     == event_count_before
subject_state_after   == subject_state_before
lineage_state_after   == lineage_state_before
child_exists_after    == false
mutation_id_consumed  == false
```

The last property is exercised by successfully committing the exact evaluated action with the same mutation ID after the rejected substitution.

## Evidence surfaces

Candidate executable evidence is present in two locations:

- crate unit test: `exact_resource_binding_rejects_substitution_without_state_mutation`;
- public-API integration tests in `tests/exact_action_binding.rs`.

The integration lane additionally verifies that a successful `DescendantCommitted` event records exactly the amount retained in the authorization.

## Deliberate non-claims

This tranche does not establish complete production action identity. It does not yet bind or verify:

- trusted monotonic commit time;
- authenticated grant/quorum/runtime witness provenance;
- verified deployment/risk policy;
- versioned capability semantics;
- versioned multidimensional resource-accounting semantics;
- canonical action/event digest encoding;
- durable transactional storage/CAS;
- runtime artifact identity;
- any physical containment mechanism.

Those remain separate Class A admission gates tracked under #1335 and its child issues.

## Promotion rule

Do not mark this invariant executed or close its #1335 acceptance items until the focused Rust 1.96 compile/test/Clippy lane has actually run on the exact candidate commit. Authored tests and static diff review remain design evidence only.
