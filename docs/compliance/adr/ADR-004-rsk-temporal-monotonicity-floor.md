# ADR-004: Enforce an RSK Temporal Monotonicity Floor

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The Replicator Safety Kernel (RSK) binds an authorization to `evaluated_at_unix_secs` and an expiry. The reference ledger currently rejects commit at or after expiry, but it does not reject a caller-supplied commit timestamp that is earlier than the authorization's evaluation time.

That permits an impossible temporal ordering inside the reference semantics:

```text
commit_time < authorization.evaluated_at
```

Even with cursor and evidence checks intact, accepting that ordering weakens anti-replay reasoning and makes later trusted-time integration harder to prove.

This ADR contains no physical replication mechanism, fabrication recipe, molecular design, or biological implementation.

## Decision

The reference ledger will enforce the half-open temporal interval:

```text
authorization.evaluated_at <= commit_time < authorization.expires_at
```

A commit timestamp earlier than evaluation returns a dedicated error before authoritative mutation.

The existing expiry rule remains half-open: commit at exactly `expires_at` is denied.

## Important trust boundary

This tranche establishes **ordering semantics only**. It does not make caller-supplied wall-clock time trustworthy.

Production admission still requires #1669's verified time/continuity boundary. A malicious caller that can choose arbitrary timestamps is not converted into a trusted time source merely because the values must be ordered.

Conceptually:

```text
reference semantics:
    evaluated_at <= supplied_commit_time < expires_at

production boundary:
    evaluated_at <= verified_commit_time < expires_at
    AND time source / continuity / freshness is independently verified
```

## Fail-closed behavior

A pre-evaluation timestamp must not:

- advance the ledger cursor;
- append an event;
- consume the mutation ID;
- alter subject/lineage counters;
- create the child;
- install or mutate an authority scope.

If no other state has changed, the exact same authorization and mutation ID may still commit at a temporally valid time.

## Error semantics

The candidate implementation introduces a typed error carrying both temporal facts:

```text
AuthorityTimeBeforeEvaluation {
    evaluated_at,
    committed_at,
}
```

This is distinct from `AuthorityExpired` so audit can differentiate rollback/pre-evaluation anomalies from ordinary expiry.

## Tests

Candidate tests must cover:

1. `commit_time == evaluated_at` is allowed if all other predicates pass;
2. `commit_time < evaluated_at` is denied;
3. denied pre-evaluation time causes no authoritative mutation;
4. denied attempt does not consume the mutation ID;
5. `commit_time == expires_at` remains denied;
6. `commit_time > expires_at` remains denied.

## Alternatives considered

### Only check expiry

Rejected. It allows a caller to move time backward relative to the authorization event.

### Clamp early commit time to `evaluated_at`

Rejected. Silent correction obscures evidence and converts invalid input into an apparently valid event.

### Implement trusted time in the same tranche

Rejected. Trusted time requires source authentication, continuity across restart, uncertainty policy, possibly multiple-source disagreement handling, and durable anti-rollback state. Those are materially larger Class A concerns tracked under #1669 and should remain separately reviewable.

## Evidence discipline

Authored tests and this ADR are design evidence. This ADR remains `Proposed` until the focused Rust 1.96 lane executes the exact candidate commit. A queued workflow is not a pass.

## Consequences

### Positive

- removes an impossible temporal ordering from the reference state machine;
- makes authorization validity a clear half-open interval;
- provides a clean semantic seam for #1669's future verified time type;
- improves replay/rollback fault modeling and TLA+ v0.2 mapping.

### Residual risk

- caller-supplied time remains untrusted in this reference tranche;
- restart continuity, clock rollback detection, uncertainty envelopes, source provenance, and durable time evidence remain production blockers;
- the RSK crates remain `publish = false` and production-admission denied.

## Related work

- #1335 — Class A authority/evidence umbrella blocker
- #1669 — trusted monotonic time and freshness continuity
- #1672 — TLA+ v0.2 temporal/fork/recovery refinement
- #1676 — threat model and clock-tampering cases
- #1724 — exact evaluated action binding
