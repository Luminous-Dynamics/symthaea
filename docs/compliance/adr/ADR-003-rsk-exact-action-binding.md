# ADR-003: Bind RSK Authorization to the Exact Evaluated Action

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The Replicator Safety Kernel (RSK) evaluates a `ReplicationAuthorityRequest` against grant, lineage, budget, monitoring, evidence, containment, and quorum predicates, then returns an opaque `BoundedReplicationAuthorization` for ledger commit.

Static review found one authority-integrity gap tracked by #1335: `ReplicationBudgetSnapshot.requested_resource_units` is evaluated, but the resulting opaque authorization does not retain that exact amount. `ReplicationLedger::commit_descendant` subsequently receives a separate caller-supplied `resource_units` value.

Existing hard and ancestral ceilings still bound total resource consumption, so this is not an unbounded-budget escape. It is nevertheless a Class A defect because the operation committed need not be exactly the operation evaluated.

This ADR contains no physical replication mechanism, fabrication recipe, molecular design, or biological implementation.

## Decision

RSK will enforce the exact-action invariant:

```text
committed_action == authorized_action == evaluated_action
```

For this tranche, the action field closed by executable code is the evaluated resource amount:

```text
authorization.requested_resource_units == commit.resource_units
```

The value is retained inside the opaque `BoundedReplicationAuthorization` created only after a successful authority evaluation. `commit_descendant` compares the caller-supplied commit amount with the bound value before any authoritative state mutation.

A mismatch returns a dedicated denial/error containing both values for audit. It must not:

- advance the ledger cursor;
- append a ledger event;
- consume the mutation ID;
- change subject or lineage counters;
- create the child subject;
- install or alter an authority scope.

The same mutation ID and authorization may therefore still be used for the exact previously evaluated amount after a rejected substitution, provided no other state transition has made the authorization stale.

### Why equality, not `commit <= authorized`

Allowing a smaller or larger amount than evaluated would make the authorization describe a family of operations rather than one evaluated operation. RSK instead chooses exact identity. Callers that want a different amount must obtain a fresh evaluation bound to that amount.

### Remaining exact-action work

This tranche closes the concrete resource-substitution defect only. It does **not** claim the complete production action identity is finished. #1335 and the production-admission gates still require binding every authority-relevant field that can differ between evaluation and commit, including any future operation/derivation identity, policy/schema versions, verified time/evidence, and durable predecessor identity.

## Alternatives considered

### Re-check only against maximum budgets at commit

Rejected. A substituted amount could remain below all ceilings while differing from the evaluated intent.

### Permit any amount less than or equal to the evaluated amount

Rejected. This weakens exact intent-to-commit identity and complicates audit/replay semantics.

### Rely only on an action digest

Rejected for this tranche. A digest is useful durable evidence, but typed enforcement should retain and compare the field whose semantics the kernel enforces. Future canonical action digests should bind those typed fields rather than replace them.

## Safety properties

The implementation and black-box tests must establish:

1. the opaque authorization exposes the evaluated resource amount read-only;
2. a lower substituted amount is denied;
3. a higher substituted amount is denied;
4. mismatch denial occurs before authoritative mutation;
5. cursor and event length are unchanged after mismatch;
6. subject and lineage accounting are unchanged after mismatch;
7. the proposed child does not exist after mismatch;
8. the mutation ID is not consumed by mismatch;
9. the exact evaluated amount can still commit afterward if the authorization remains otherwise valid;
10. a successful event records the exact bound amount.

## Evidence discipline

Authored tests are design evidence, not executed evidence. This ADR remains `Proposed` until the focused RSK Rust 1.96 lane executes the exact candidate commit and reports compile/test/Clippy results. A queued workflow is not a pass.

The parent #1326 focused workflow is independently queued and is intentionally not disturbed by this stacked branch.

## Consequences

### Positive

- closes the concrete intent→authorization→commit substitution path identified in #1335;
- makes resource accounting auditable against the exact evaluated request;
- preserves mutation IDs on pre-commit mismatch, avoiding denial-induced token consumption;
- gives the v0.2 formal model a concrete `CommittedActionWasAuthorizedExactly` transition to mirror.

### Cost / residual risk

- callers must re-evaluate when any bound action field changes;
- this does not solve trusted time, verified grants/quorum/runtime evidence, policy authority, capability-schema versioning, typed resource-accounting semantics, durable replay, or runtime-build identity;
- the underlying crates remain `publish = false` and production-admission denied.

## Related work

- #1335 — exact evaluated action and verified evidence umbrella blocker
- #1668 — verified positive authority evidence
- #1669 — trusted monotonic time
- #1670 — authenticated governance/recovery authority
- #1672 — TLA+ v0.2 refinement
- #1673 — verified risk policy
- #1678 — capability schema binding
- #1679 — typed resource accounting
- #1682 — exact build/runtime admission identity
