# MEM-CORR-001A — provenance-aware memory correction propagation plan

## Purpose

Make explicit correction propagation through memory a separate, deterministic,
content-blind planning step.

The planner does **not** mutate a backing memory store. It computes the exact
per-artifact actions that a later executor must apply against a privacy/memory
graph, then commits to that plan.

This preserves the distinction:

```text
correction observed
!= memory mutated
!= durable memory rewritten
!= index rebuilt
!= stale retrieval eliminated
```

## Inputs

A correction directive binds:

- correction ID;
- correction receipt commitment;
- semantic key;
- exact context;
- durability (`TurnOnly`, `Session`, `DurableExplicit`);
- exact corrected-away source IDs;
- correction epoch.

A memory dependency snapshot binds:

- snapshot ID;
- policy version;
- artifact IDs;
- semantic/context scope;
- role and retention;
- complete-vs-unknown dependency provenance;
- exact source IDs;
- explicit independent-basis sufficiency;
- derivation-policy reference;
- external-copy possibility.

The snapshot is canonicalized by artifact ID and domain-separated with BLAKE3.

## Per-artifact actions

The planner emits one of:

- `NoEffectOutsideScope`;
- `ShadowForCurrentScope`;
- `Invalidate`;
- `RecomputeRequired`;
- `SupersedeWithFreshIdentity`;
- `RetainIndependentBasis`;
- `BlockedUnknownDependency`.

### TurnOnly

A turn-only correction may shadow matching state for the current interaction,
but it never authorizes a durable rewrite.

### Session

A session correction may invalidate session/turn-scoped dependent artifacts.
Durable artifacts are shadowed rather than silently rewritten.

### DurableExplicit

A directly corrected durable source memory requires a fresh identity. Derived
memories whose basis includes corrected-away sources must either:

- explicitly retain a sufficient independent basis; or
- be recomputed with a fresh identity.

Indexes/embeddings and external-export artifacts depending on corrected-away
sources are invalidated rather than treated as corrected memories.

## Dependency rules

Scope match alone is insufficient to change an artifact.

```text
same semantic key/context
+
no dependency on corrected source
→ retain independent basis
```

Unknown dependency metadata fails closed:

```text
scope match
+
dependency provenance unknown
→ BlockedUnknownDependency
```

An artifact may claim `independent_basis_sufficient` only when dependency
metadata is complete.

Recompute planning removes corrected-away sources from the recorded remaining
source set. A later executor must reject any recomputation that reintroduces
those retired sources.

## Fresh identity

`SupersedeWithFreshIdentity` and `RecomputeRequired` both mark
`fresh_identity_required = true`.

This prevents correction from becoming resurrection under the same memory ID.

## External copies

`external_deletion_unproven` is reported only for destructive/recompute actions
on artifacts that may have external copies. Temporary shadowing does not claim
that deletion was attempted or proven.

## Determinism and replay

The plan commitment binds:

- schema;
- operation ID/epoch;
- correction identity/receipt/epoch/durability;
- snapshot identity/commitment/policy;
- canonical per-artifact decisions;
- corrected and remaining source sets;
- fresh-identity requirements;
- external-deletion limitations;
- aggregate blocked/fresh/external counts.

Same inputs are idempotent. Changing the operation epoch changes plan identity,
which gives a later executor a replay-resistant sequencing input without this
planner pretending to maintain global mutable state.

## Tests

The integration suite covers:

- turn-only durable non-rewrite;
- session shadow vs session-state invalidation;
- durable direct-source supersession;
- derived-memory recomputation;
- explicit surviving independent basis;
- unknown dependency fail-close;
- unrelated-context isolation;
- unaffected same-scope dependency retention;
- index/export invalidation;
- external-copy limitation semantics;
- ordering-invariant commitments;
- operation-epoch identity;
- snapshot/plan tamper rejection;
- invalid independent-basis claims.

## Follow-on executor

A later tranche should bind this plan to actual governed mutations:

```text
Correction receipt
      ↓
MEM-CORR propagation plan
      ↓
privacy / reflective-memory graph mutation receipt
      ↓
fresh memory identities where required
      ↓
index rebuild + current snapshot receipt
      ↓
retrieval firewall
```

That executor must verify the actual correction receipt semantics and the live
privacy/memory graph rather than trusting opaque references supplied here.

## Nonclaims

MEM-CORR-001A does not:

- decide whether arbitrary natural-language content is true;
- mutate a real backing store;
- prove the referenced correction receipt was semantically valid;
- prove a recomputation occurred;
- rebuild a vector/index snapshot;
- prove external copies were deleted;
- infer user preference, consent, or physical authority.
