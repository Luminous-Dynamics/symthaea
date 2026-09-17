# EKM-030 — Belief Mutation Persistence Capsule V1

## Purpose

EKM-026 introduced explicit epistemic-support state and append-only mutation history. EKM-029 then closed the public mutation surface and imposed one public authorization identity per successful revision receipt.

EKM-030 defines the first deterministic export/validation contract for that state. It is intentionally **not** a restore implementation.

## Capsule V1

`BeliefMutationPersistenceCapsuleV1` captures:

- schema version (`V1`),
- capture cycle,
- the complete caller-declared support-state inventory,
- each claim's baseline epistemic support,
- current epistemic support,
- current monotonic state revision,
- initialization/update cycles,
- last mutation ID,
- every append-only belief mutation receipt,
- source revision receipt IDs,
- signed deltas and exact support-before/support-after values,
- revision-before/revision-after values,
- authorization ID and authority label,
- authorization/application cycles,
- consumed authorization count.

The baseline is reconstructed from the first mutation's `support_before`, or from current support for an unmutated state. This means the capsule contains the information a separately qualified restore layer would need to reconstruct the support-state chain without guessing its starting value.

## Capture validation

Capture fails closed unless:

- the supplied claim-ID inventory is unique,
- the supplied IDs account for every registered support state,
- every supplied ID resolves to a state,
- every mutation references a declared state,
- mutation IDs are unique and globally monotonic,
- source revision receipt IDs are unique,
- authorization IDs are unique,
- authorization precedes application,
- application/state cycles do not postdate capture,
- each claim's revision chain starts at revision 0 and advances exactly +1,
- consecutive support transitions connect exactly,
- each `support_after` equals `support_before + proposed_delta` within the existing floating-point tolerance,
- the final mutation exactly matches the live state's revision, support, last mutation ID, and update cycle,
- an unmutated state remains at revision 0 with no mutation ID,
- consumed authorization count equals mutation receipt count.

That final condition is deliberate. EKM-029 makes it true for the closed public authority path. If lower-level internal code consumes an additional authorization identity during an idempotent replay, the state is rejected as **not persistence-closed** rather than silently omitting replay history.

## Live validation

A capsule can later be checked against a still-live store. The observation cycle may advance, but the state inventory, state values, mutation chain, and consumed authorization count must remain exactly represented by the capsule.

Any later legitimate mutation therefore makes the older capsule stale rather than silently updating it.

## Negative control

The tests deliberately exercise the crate-internal raw EKM-026 firewall to create an alternate replay authorization ID without a second mutation receipt. The capsule rejects that state because `consumed_authorizations != mutation_receipts`.

This demonstrates why the EKM-029 public replay restriction is required for deterministic persistence.

## Revision-decision history boundary

The mutation capsule stores the **source revision receipt IDs used by successful mutations**, but it does not yet persist the full EKM-025 `BeliefRevisionHistory`. That distinction matters because the revision history also contains rejected decisions and owns the monotonic receipt-ID sequence.

A restart implementation must not infer that a support-state capsule alone is sufficient to reset `BeliefRevisionHistory` to receipt ID 1. Reusing an earlier decision ID would collapse distinct authority lineages.

Therefore full restart recovery additionally requires a versioned revision-decision-history capsule (or an equivalent monotonic-ID persistence mechanism) before hydration can be considered safe.

## Non-claims

EKM-030 does **not** establish:

- hydration/restoration of `EpistemicSupportStore`,
- persistence/restoration of full `BeliefRevisionHistory`, including rejected decisions and its next receipt ID,
- SQLite/file persistence,
- crash consistency,
- atomic disk commits,
- cryptographic integrity or signatures,
- rollback execution,
- migration of legacy `TemporalFact::confidence`,
- scientific correctness of a belief revision.

No method in this tranche writes support state or mutation history.

## Next boundary

A future restore program should first persist the complete EKM-025 revision-decision lineage, then consume this versioned support capsule and reconstruct an `EpistemicSupportStore` only after independently checking:

1. revision-receipt ID continuity, including rejected decisions,
2. capsule structural validity,
3. ledger claim identity compatibility,
4. exact support revision-chain continuity,
5. authorization-history closure,
6. post-restore equivalence against both capsules,
7. replay safety after authority reconstruction.

Restore should remain separate from ordinary belief revision authority.

## Qualification boundary

This PR is stacked on EKM-029 / PR #3716. CI remains authoritative. Queued, absent, or unexecuted workflow jobs are not qualification evidence.
