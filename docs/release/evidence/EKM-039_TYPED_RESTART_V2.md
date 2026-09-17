# EKM-039 — Typed Epistemic Restart Capsule V2

## Purpose

EKM-033/V1 provides a complete cold-reconstruction payload, but its EKM-034 wire lineage predates explicit policy/decision schemas. EKM-039 introduces a separate V2 capsule that preserves V1 unchanged while binding the EKM-038 schema history into restart integrity.

## Added contract

`EpistemicRestartCapsuleV2` contains:

- a complete verified `EpistemicRestartCapsuleV1`
- complete `BeliefRevisionSchemaHistoryCapsuleV1`
- a domain-separated schema-history BLAKE3 digest
- a domain-separated V2 capsule digest over the V1 manifest digest + typed schema digest

`QuarantinedEpistemicRestartV2` wraps the existing V1 read-only quarantine and the typed schema capsule. It has no activation or mutable hydration method.

## Central invariant

**V2 restart identity binds the complete epistemic state and the exact typed policy/decision semantics that produced every persisted revision receipt.**

## Explicit canonical hashing

The V2 schema digest hashes only explicit typed fields:

- policy numeric thresholds
- policy booleans
- canonical uncertainty caps
- receipt identity / claim / delta / cycle
- eligibility
- declared provenance-root count
- every typed decision failure and its payload
- every nested knowledge-weight routing failure

Stable tags are explicit. The digest does not use Rust enum discriminants or `Debug` text.

## V1 compatibility

V1 is not mutated or reinterpreted. V2 composes a verified V1 capsule and adds typed revision semantics as a new integrity layer. This avoids silently changing the meaning of existing V1 wire artifacts.

## Quarantine behavior

V2 quarantine:

1. verifies all V2 links and digests,
2. delegates cold ledger reconstruction to the already-isolated V1 quarantine path,
3. rechecks every schema record against the quarantined revision receipts,
4. retains only read-only state.

## Authority boundary

No writable `EpistemicSupportStore`, writable `BeliefRevisionHistory`, executable authorization, state swap, activation, causal/world-model mutation, evidence ingestion, file I/O, or network I/O is introduced.

The next serialization tranche may define a wire-v2 envelope for this typed capsule. Parsed bytes must still terminate in an untrusted DTO + independent semantic validation before any wire-to-quarantine conversion is considered.

## Qualification boundary

Exact-head CI remains authoritative. Queued runs do not establish format, compile, Clippy, test, or runtime PASS.
