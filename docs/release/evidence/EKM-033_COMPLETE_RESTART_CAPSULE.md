# EKM-033 — Complete Epistemic Restart Capsule

## Status

Draft / unqualified until the exact branch head executes the repository qualification lanes.

## Purpose

EKM-032 binds one atomic epistemic epoch with an integrity manifest, but a digest is not a restore payload. EKM-033 closes that gap by carrying the immutable ledger records required for cold reconstruction beside the already-validated EKM-030 belief-mutation capsule, EKM-031 complete belief-revision decision history, and EKM-032 manifest.

The central invariant is:

> A restart artifact is payload-complete only if it carries the exact ledger semantics and ID lineage needed to reproduce the same manifest from a fresh process, not merely the digest of those semantics.

## Added structures

`PersistedEpistemicLedgerV1` carries:

- capture cycle,
- validated ledger lineage,
- every provenance record,
- every claim record including its exact evidence-ID list,
- every evidence record.

`EpistemicRestartCapsuleV1` binds:

- complete persisted ledger payload,
- EKM-030 support state + successful mutation history,
- EKM-031 eligible and rejected revision-decision history,
- EKM-032 atomic restart manifest.

`QuarantinedEpistemicRestartV1` is a cold reconstruction image containing:

- a freshly rebuilt ledger,
- the exact rebuilt inventory,
- read-only EKM-030 mutation/support snapshots,
- read-only EKM-031 revision receipts,
- a manifest regenerated from the reconstructed state.

## Quarantine reconstruction rules

The persisted ledger is not installed by writing private fields. EKM-033 creates a fresh `EpistemicLedger` and replays only the ordinary append APIs in stable-ID order:

1. provenance records,
2. claims,
3. evidence.

Every generated ID must equal the persisted ID. After evidence reconstruction, every provenance, claim, and evidence value must equal the persisted record exactly. Claim evidence-list ordering therefore has to reproduce naturally rather than being patched after the fact.

The reconstructed ledger is then passed back through EKM-032. The regenerated manifest must match the source manifest digest exactly.

## Additional cross-checks

EKM-033 also checks that every snapshotted basis evidence record inside EKM-025/EKM-031 revision receipts still matches the immutable ledger record with the same evidence ID. A rejected receipt whose basis evidence was unknown at decision time may retain `snapshot = None`; this remains valid audit history and is not rewritten after the fact.

`verify_source_live` can additionally confirm that the current source ledger, support store, and revision history have not diverged from the captured artifact. This is observational only.

## Negative controls

Tests cover:

- complete cold reconstruction in quarantine,
- source-state equivalence checks,
- semantic ledger tampering changing/failing manifest equivalence,
- claim↔evidence census tampering,
- preservation of rejected decision history.

## Authority boundary

EKM-033 has **no activation path**.

It does not:

- hydrate an `EpistemicSupportStore` from persisted support state,
- hydrate a writable `BeliefRevisionHistory`,
- recreate mutation/firewall authorization state as executable authority,
- swap a reconstructed ledger into `KnowledgeManager`,
- mutate legacy `TemporalFact::confidence`,
- alter causal state,
- alter world-model state,
- alter action selection,
- perform file/database I/O,
- define a cross-version serialized wire format.

The current quarantine therefore establishes payload completeness + cold ledger reconstruction + manifest equivalence, not live restart activation.

## Serialization boundary

The capsule is presently an in-memory typed contract. EKM-032 still labels revision receipt hashing as `ExplicitFieldsPlusReceiptDebugV1`; Rust `Debug` is not a durable cross-version wire representation.

A later serialization tranche should define canonical versioned bytes for the complete capsule before disk/network persistence is called stable.

## Next boundary

The next authority-bearing work should remain split:

1. canonical restart wire format / parser with corruption and schema-version controls,
2. quarantine hydration of support/history state with replay-ID preservation,
3. post-hydration manifest equivalence,
4. explicit activation authorization,
5. atomic live-state swap with rollback provenance.

Those steps should not be collapsed into one restore function.
