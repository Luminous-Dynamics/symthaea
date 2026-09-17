# EKM-035 — Independent Restart-Wire Semantic Validation

## Status

Draft / unqualified until the exact branch head executes the repository qualification lanes.

## Purpose

EKM-034 verifies framing, bounds, versioning and a corruption checksum. A malicious producer can still recompute a valid checksum over semantically inconsistent data. EKM-035 therefore treats the decoded DTO as untrusted and independently validates the state graph and revision/mutation state machines.

The central invariant is:

> A checksum-valid restart envelope is still not epistemically admissible until its internal relationships independently satisfy the same lineage, temporal, evidence and authority invariants expected from a genuine capture.

## Ledger checks

The validator requires:

- manifest counts to equal decoded collection sizes,
- provenance, claim and evidence IDs to be contiguous from `1`,
- manifest next IDs to equal the derived next IDs,
- provenance parents to exist and precede their children,
- no claim/evidence/provenance record to postdate capture,
- every evidence record to resolve to a known claim and provenance record,
- claim evidence-ID lists to contain no duplicates,
- every claim evidence-ID list to equal the actual evidence census targeting that claim.

## Support / mutation checks

The validator requires:

- every support state to reference a known claim,
- support states to be unique and ordered,
- support-state time ranges to be coherent,
- consumed authorization count to equal mutation count,
- mutation IDs, authorization IDs and source revision IDs to be unique,
- mutation IDs to be monotonic,
- authorization <= application <= capture,
- finite bounded deltas,
- checked revision increments with overflow rejection,
- exact `support_after = support_before + delta` within the existing tolerance,
- every mutation to map to a persisted support state,
- per-claim revision/support chains to be continuous from baseline to final state,
- final support state, revision, mutation ID and update cycle to match the latest mutation.

## Revision-history checks

The validator requires:

- receipt IDs to be contiguous from `1`,
- next receipt ID to match the derived next ID,
- non-regressing decision cycles,
- no decision to postdate capture,
- V1 opaque policy/decision fields to be non-empty,
- unique revision-basis IDs,
- snapshot evidence ID to equal requested evidence ID,
- basis evidence to exist and exactly match the current immutable wire evidence record,
- decision time to be >= basis-evidence observation time,
- uncertainty assessment time to be <= decision time,
- duplicate-basis diagnostics to be unique,
- declared provenance-root count to be independently recomputed from basis evidence and provenance ancestry.

Provenance-root traversal is iterative rather than recursive so a deeply nested untrusted provenance lineage cannot exhaust the call stack.

## Mutation → decision checks

Every persisted mutation must reference a revision receipt that:

- exists,
- is encoded as eligible,
- targets the same claim,
- carries the exact same proposed-delta bit pattern,
- predates or equals mutation authorization.

## Negative controls

Tests cover:

- valid structurally coherent snapshot,
- claim↔evidence census tampering,
- mutation linked to rejected decision,
- forged provenance-root count,
- authorization predating its decision,
- `u64::MAX` revision input rejected without arithmetic overflow.

## Authority boundary

The validator returns only `EpistemicRestartWireValidationReport` or a typed validation error.

It does **not**:

- construct `EpistemicRestartCapsuleV1`,
- construct `QuarantinedEpistemicRestartV1`,
- construct or mutate `EpistemicLedger`,
- hydrate `EpistemicSupportStore`,
- hydrate `BeliefRevisionHistory`,
- recreate executable authorization state,
- activate or swap state,
- mutate causal/world-model/action state,
- perform file or database I/O.

The EKM-034 opaque policy/decision V1 fields remain a known limitation. This validator does not pretend to re-evaluate the full belief gate without those private policy semantics.

## Next boundary

Before wire data may construct an EKM-033 quarantine capsule, the remaining opaque policy/decision fields should become explicit read-only versioned snapshots and the wire representation should be upgraded accordingly.
