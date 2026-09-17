# EKM-034 — Bounded Epistemic Restart Wire Envelope

## Status

Draft / unqualified until the exact branch head executes repository qualification lanes.

## Purpose

EKM-033 defines a complete typed in-memory restart capsule, but that is not yet a safe disk/network representation. EKM-034 adds explicit framing and a bounded parser while deliberately stopping before parsed bytes can construct a capsule or live epistemic state.

The central invariant is:

> Untrusted restart bytes may be parsed into an inspectable snapshot only; parsing is not restoration and never grants mutation or activation authority.

## Framing

V1 uses:

- fixed magic bytes,
- explicit wire version,
- declared payload length,
- bounded payload size,
- payload bytes,
- domain-separated BLAKE3 checksum covering version, length, and payload.

The checksum detects accidental/corrupt changes. It is not a signature, MAC, trusted timestamp, or authorization proof.

## Parser bounds

The decoder rejects:

- wrong magic,
- unsupported versions,
- oversized payloads,
- oversized strings,
- oversized record counts,
- integer/length overflow,
- truncation,
- envelope length disagreement,
- checksum disagreement,
- invalid UTF-8,
- invalid boolean tags,
- unknown claim/evidence/polarity enum tags,
- invalid bounded support values,
- invalid calibration values,
- invalid uncertainty values,
- envelope/manifest capture-cycle disagreement,
- trailing payload bytes.

All collection counts are bounded before allocation.

## Explicitly typed V1 content

The envelope carries structured fields for:

- EKM-032 manifest summary and component digests,
- provenance records and parent IDs,
- claims and exact claim→evidence lists,
- evidence records,
- persisted epistemic-support state,
- belief-mutation receipts and authorization identities,
- revision receipt IDs/claim/delta/rationale,
- revision evidence-basis snapshots,
- duplicate-basis diagnostics,
- calibration snapshots,
- multidimensional uncertainty values and bases,
- revision eligibility/root-count summaries,
- revision evaluation cycles.

## Known V1 semantic limitation

`BeliefRevisionPolicy` fields and the complete `BeliefRevisionDecision` failure structure are currently private to the gate implementation. EKM-034 does **not** weaken that encapsulation merely for serialization.

V1 therefore carries those two receipt substructures as length-prefixed, version-pinned opaque `Debug` text and names the encoding:

`ExplicitFieldsWithOpaqueRevisionPolicyDecisionV1`

This is intentionally not claimed as a durable cross-version receipt schema.

A future V2 should add explicit read-only policy/decision snapshot types, encode every field and failure variant with stable tags, and remove the opaque text before wire→capsule conversion is allowed.

## Authority boundary

`EpistemicRestartWireV1::decode` returns only `EpistemicRestartWireSnapshotV1`.

There is no API in this tranche that:

- constructs `EpistemicRestartCapsuleV1` from bytes,
- constructs `QuarantinedEpistemicRestartV1` from bytes,
- hydrates an `EpistemicLedger`,
- hydrates `EpistemicSupportStore`,
- hydrates `BeliefRevisionHistory`,
- recreates executable authorization state,
- activates or swaps state into `KnowledgeManager`,
- mutates confidence, causal state, world-model state, or action selection,
- writes files or databases.

## Negative controls

Tests cover:

- encode/decode of a capsule containing both eligible and rejected revision decisions,
- one-byte payload tampering,
- unsupported version,
- truncation,
- trailing bytes.

## Next boundary

Before bytes may become an EKM-033 quarantine capsule, the opaque policy/decision fields should be replaced with a fully explicit schema and the parsed snapshot should receive semantic validation independent of the encoder.
