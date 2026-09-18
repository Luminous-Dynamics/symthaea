# EKM-063 — Historical Evidence Projection

Status: **draft / unqualified**

## Purpose

EKM-062 establishes that one exact protected mutation-time evidence census is eligible for isolated historical projection review.

EKM-063 is the first layer that materializes that historical view, but only as immutable claim/evidence projection records. It does **not** construct an `EpistemicLedger`, execute EKM-026, hydrate a support store, mutate revision history, advance trusted state, or authorize activation.

## Membership rule

Historical membership is determined exclusively by the protected EKM-028 census carried through EKM-056 → EKM-057 → EKM-058 → EKM-059 → EKM-060 → EKM-061 → EKM-062.

`observed_at_cycle` is never used to decide whether a record belonged to the ledger at the mutation decision.

This matters because evidence inserted later can legitimately carry an observation timestamp at or before the earlier seal cycle.

## Projection record

Each `HistoricalEvidenceProjectionRecordV1` binds:

- mutation receipt ID;
- source revision receipt ID;
- claim ID;
- seal cycle;
- exact sealed claim snapshot;
- exact protected sealed evidence census;
- final claim evidence count;
- IDs present on the final append-only claim but absent from the protected census;
- count of excluded final records whose `observed_at_cycle <= sealed_at_cycle`;
- domain-separated record digest.

The excluded-record diagnostic demonstrates why observation timestamps cannot substitute for historical membership.

## Construction

`HistoricalEvidenceProjectionV1::project` first re-verifies the exact EKM-062 receipt against:

- restart-v2 snapshot;
- EKM-057 seal sidecar;
- EKM-044 restart validation receipt;
- EKM-059 protected checkpoint;
- EKM-060 protected-sidecar admission;
- EKM-061 currentness receipt.

It then requires:

- protected source equivalence is verified;
- checkpoint currentness is verified;
- isolated projection review is eligible;
- all replay/hydration/activation authority flags remain false;
- projection occurs at or after the EKM-062 review cycle;
- projection occurs before EKM-061 currentness expiry.

## Final-ledger relation

For each protected census:

- the final claim must still exist with identical descriptive semantics;
- sealed evidence IDs must be a subset of the final claim evidence IDs;
- duplicate sealed/final IDs fail closed;
- every final-only evidence ID must resolve to the same claim;
- final-only evidence is excluded from the historical projection regardless of observation timestamp.

EKM-063 therefore represents a protected historical *membership projection*, not a timestamp reconstruction.

## Projection digest

The top-level domain-separated digest binds:

- restart capture cycle;
- projection cycle;
- EKM-062 eligibility receipt digest;
- protected EKM-056 seal-capsule digest;
- restart outer checksum;
- seal-wire checksum;
- every per-mutation projection record digest;
- projected/excluded evidence counts;
- backdated-final-only diagnostic count;
- all membership/authority booleans.

`verify_against` re-runs the complete projection and requires exact equality plus digest equivalence.

## Authority boundary

A successful projection records:

- `membership_derived_from_protected_census = true`;
- `observation_cycle_used_as_membership = false`;
- `ledger_constructed = false`;
- `historical_replay_authorized = false`;
- `writable_hydration_authorized = false`;
- `activation_authorized = false`.

No mutable ledger handle is exposed.

## Non-claims

EKM-063 does not claim:

- that EKM-026 accepted the historical projection;
- that support mutations have been replayed;
- that reconstructed support state equals persisted support state;
- that revision history has been hydrated;
- that a restart is safe to activate;
- that trusted checkpoints may be advanced;
- that legacy confidence semantics are migrated;
- that causal/world-model/action state is changed.

## Qualification boundary

GitHub Actions remains the executable authority for format, compilation, Clippy, tests and runtime checks.

At freeze time, parent EKM-062 exact-head CI #7304 remained queued. Static/API review is not qualification evidence.

## Next safe tranche

After executable qualification, the next narrow step is an **EKM-026 historical replay dry-run** against an isolated temporary ledger assembled only from one EKM-063 projection record.

That tranche should:

- use existing ledger append APIs;
- keep the temporary ledger private;
- execute the existing mutation firewall in dry-run/verification form only;
- compare the resulting support transition with the persisted mutation receipt;
- expose only a replay verification report/digest;
- keep writable hydration and activation false.
