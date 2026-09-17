# EKM-031 — Belief Revision Decision-History Persistence Capsule V1

## Purpose

EKM-030 captures the applied epistemic-support mutation chain, but that chain is not the complete authority lineage. EKM-025 records every belief-revision decision, including rejected decisions that never produce a mutation. Restart safety therefore also requires preservation of the complete append-only decision history and its identifier lineage.

EKM-031 adds a read-only persistence contract for that history. It does not restore or mutate `BeliefRevisionHistory`.

## Capsule V1

`BeliefRevisionHistoryCapsuleV1` captures:

- schema version (`V1`),
- capture cycle,
- every immutable `BeliefRevisionReceipt` in append order,
- eligible decisions,
- rejected decisions,
- proposal delta and rationale,
- unique evidence-basis references and duplicate-attempt diagnostics,
- immutable evidence snapshots,
- frozen belief-revision policy,
- calibration snapshot,
- uncertainty assessment,
- typed gate decision/failures,
- evaluation cycle,
- the next receipt ID implied by the complete contiguous receipt lineage,
- the linked EKM-030 mutation-capsule capture cycle and mutation count.

The existing immutable receipt type is retained directly instead of flattening it into a second lossy persistence representation.

## Identifier continuity

Capture requires receipt IDs to be exactly contiguous from `1` in append order. The capsule derives the next ID as `last_id + 1` (or `1` for an empty history).

This is important because a process restart must not reset `BeliefRevisionReceiptId` and reuse an old identifier for a new decision. An ID identifies one historical decision lineage, not merely a vector index in the current process.

## Mutation-history cross-check

Every applied mutation in the linked EKM-030 capsule must resolve to a decision receipt in EKM-031. The source decision must:

- exist,
- be eligible,
- name the same claim,
- carry the same proposed delta,
- be evaluated no later than the mutation authorization.

An eligible decision is allowed to remain unapplied. A mutation is never allowed to exist without its decision lineage.

Rejected decisions remain in the capsule even though they cannot be mutation sources. Their preservation matters because deleting them would rewrite audit history and would shift/reuse later receipt identifiers.

## Temporal integrity

Capture rejects:

- decision receipts evaluated after the capsule capture cycle,
- a linked EKM-030 capsule captured after the revision-history capsule,
- regressing decision evaluation cycles within one append-only history,
- mutation authorization that predates the source revision decision.

## Live validation

A capsule may be checked against a still-live `BeliefRevisionHistory` and linked EKM-030 capsule. Any new decision—eligible or rejected—makes the old capsule stale. Any changed linked mutation capsule also makes it stale.

## Non-claims

EKM-031 does **not** establish:

- history hydration/restoration,
- file or database persistence,
- crash consistency,
- atomic multi-capsule commits,
- cryptographic integrity/signatures,
- correctness of any belief-revision policy,
- correctness of any scientific conclusion,
- migration of legacy `TemporalFact::confidence`.

It exposes no method that inserts a receipt or changes a decision.

## Next boundary

A future restart tranche should restore the ledger, revision-decision history, epistemic-support state, mutation receipts, and replay-protection state as one versioned lineage, then prove post-restore equivalence and replay safety before any live legacy-confidence migration.

## Qualification boundary

This PR is stacked on EKM-030 / PR #3717. CI remains authoritative. Queued, absent, or unexecuted workflow jobs are not qualification evidence.
