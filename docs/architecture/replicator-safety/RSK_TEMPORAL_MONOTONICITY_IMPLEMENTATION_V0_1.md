# RSK Temporal Monotonicity — Implementation Mapping v0.1

**Status:** candidate executable mapping; not production evidence until CI executes  
**Class:** A safety-critical authority semantics  
**Scope:** abstract authorization-time ordering only

This document maps ADR-004's temporal validity rule to the reference Rust implementation. It does not establish trusted clock provenance and does not specify any physical replication mechanism.

## Reference invariant

For one bound authorization:

```text
authorization.evaluated_at <= commit_time < authorization.expires_at
```

The lower bound is inclusive and the upper bound is exclusive.

## Implementation mapping

`BoundedReplicationAuthorization` already retains the successful request's evaluation time:

```text
evaluated_at_unix_secs: request.now_unix_secs
```

and the underlying authorized replication retains the grant expiry.

`ReplicationLedger::commit_descendant` now checks, before authoritative mutation:

```text
if commit_time < evaluated_at:
    deny AuthorityTimeBeforeEvaluation

if commit_time >= expires_at:
    deny AuthorityExpired
```

The pre-evaluation error retains both values for audit:

```text
AuthorityTimeBeforeEvaluation {
    evaluated_at,
    committed_at,
}
```

## Transition ordering

The relevant commit prefix is:

1. verify authorization cursor and mutation ID are current;
2. verify exact evaluated resource amount matches the commit action;
3. verify temporal ordering and expiry;
4. continue with child uniqueness, runtime witness, negative ancestry, grant/revocation, and budget checks;
5. only then append evidence and mutate authoritative state.

This ordering ensures a temporal denial cannot spend the mutation ID or modify counters/events.

## Candidate executable evidence

### Crate unit tests

- `temporal_monotonicity_rejects_pre_evaluation_without_state_mutation`
  - evaluates at 100;
  - attempts commit at 99;
  - requires the typed rollback error;
  - requires unchanged cursor, event count, subject state, lineage state, and absent child;
  - then commits at 100 with the same mutation ID.

- `authorization_time_interval_is_half_open`
  - commit at evaluation time succeeds;
  - commit at expiry and after expiry fail;
  - expiry denial leaves the cursor unchanged and does not consume the mutation ID.

### Public-API integration tests

`tests/temporal_monotonicity.rs` independently exercises the same contract through exported types and APIs rather than crate-private test access.

## Why this is not trusted time

The current reference API still receives `now_unix_secs` from the caller. Enforcing ordering does not authenticate that fact.

Production admission requires a separate verified-time boundary under #1669, including as applicable:

- source provenance/authentication;
- continuity and anti-rollback state;
- restart semantics;
- freshness/uncertainty bounds;
- disagreement handling for multiple sources;
- durable binding to the admitted epoch/state.

The future production transition should therefore replace or wrap the caller scalar with an opaque verified time/continuity value while preserving the same half-open ordering invariant.

## Formal-model mapping

The TLA+ v0.2 work in #1672 should model at least:

```text
ExpiredAuthorityCannotCommit
CommitCannotPrecedeEvaluation
DeniedTransitionDoesNotMutateAuthorityState
```

It should distinguish transient loss of temporal eligibility from persistent quarantine/revocation/fork states.

## Deliberate non-claims

This tranche does not establish:

- trustworthy wall-clock time;
- cross-restart time continuity;
- distributed clock agreement;
- authenticated time evidence;
- durable anti-rollback storage;
- complete production action identity;
- any physical containment guarantee.

## Promotion rule

Do not describe these properties as executed evidence until the focused Rust 1.96 compile/test/Clippy lane runs on the exact candidate commit. A queued run is not a pass.
