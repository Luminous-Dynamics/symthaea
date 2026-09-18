# VIS-004C — Visual Belief Revision Ledger v1

## Purpose

VIS-004C makes changes to persistent visual beliefs explicit and attributable.

A current belief state is not enough for a trustworthy world model. Symthaea should also be able to answer:

- what did I believe before?
- what changed?
- which evidence caused the change?
- was this an inference from observation or a generative prediction?
- did I become less certain?

## Full before/after state

Every `BeliefRevision` stores:

- monotonically increasing revision number;
- operation kind;
- complete `before` belief snapshot (except the first revision);
- complete `after` belief snapshot;
- evidence-bearing cause.

The ledger validates that each revision's `before` state exactly equals the previous revision's `after` state.

History with gaps or silently rewritten intermediate states is rejected during deserialization.

## Supported belief snapshots

The initial ledger can track:

- `SemanticClassBeliefSet`
- `TrackIdentityBeliefSet`

One ledger is scoped to one exact belief question. It cannot switch entity, vocabulary, track, or belief family halfway through its history.

## Operations

Initial operation vocabulary:

- `Initialized`
- `EvidenceAssimilated`
- `EvidenceRetracted`
- `Reweighted`
- `ResetToUnknown`

The first entry must be revision 1, `Initialized`, with no before-state.

A later `Initialized` entry is rejected.

No-op revisions are rejected.

`ResetToUnknown` is valid only if the resulting competing-belief set has zero explicit candidates and `unassigned_mass = 1.0`.

## Historical evidence boundary

Revision causes accept only `VisualEvidence::Inferred` or `VisualEvidence::Remembered`.

Direct observations can support the inference that causes a revision, but an `Observed` value itself is not a belief revision.

`Predicted`, `Simulated`, and `Counterfactual` evidence cannot rewrite historical belief state.

This keeps:

```text
what I observed
what I inferred from it
what I currently predict
```

as separate epistemic layers.

## Capacity behavior

The ledger is bounded, but reaching capacity is an error.

Old revisions are never evicted silently to make room for new ones.

Future retention/archival work may explicitly checkpoint and seal old history, but VIS-004C does not implement that policy yet.

## Serialization boundary

Deserialization validates:

- capacity;
- first-entry semantics;
- contiguous revision numbers;
- per-revision invariants;
- exact before/after continuity;
- question consistency;
- evidence origin.

## Explicit nonclaims

VIS-004C is an auditable semantic revision history, not a cryptographically tamper-evident ledger. It does not yet provide signed commitments, distributed consensus, calibrated Bayesian updates, canonical world identity, or robotics authority.

## Follow-up

After VIS-004A/B/C qualify, the next integration step should create a persistent `VisualEntityBelief` layer that composes:

- typed identity references;
- competing class/identity hypotheses;
- qualified geometry;
- source observation lineage;
- occlusion and last-observed state;
- belief revision ledgers.

That composite should remain separate from predicted/counterfactual future state.
