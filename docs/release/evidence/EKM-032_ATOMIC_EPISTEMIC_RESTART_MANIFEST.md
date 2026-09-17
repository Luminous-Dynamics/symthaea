# EKM-032 — Atomic Epistemic Restart Manifest V1

## Purpose

EKM-030 captures validated epistemic-support state and successful belief-mutation history. EKM-031 captures the complete belief-revision decision lineage, including rejected decisions. Those capsules are necessary but are not, by themselves, an atomic restart image: they can be captured at different logical cycles and they do not bind the complete claim/evidence/provenance ledger identity namespace.

EKM-032 introduces a read-only manifest that binds all three epistemic surfaces at one logical capture cycle.

## Atomic capture invariant

A V1 manifest is created only when all of the following cycles are exactly equal:

- the manifest capture cycle,
- the EKM-030 mutation/support capsule capture cycle,
- the EKM-031 decision-history capsule capture cycle,
- the EKM-031 linked EKM-030 capture cycle.

A sequential bundle from different cycles is rejected as `NonAtomicCaptureEpoch`; it is not relabeled an atomic snapshot.

## Ledger lineage

The caller supplies an explicit `EpistemicLedgerInventoryV1` because `EpistemicLedger` deliberately exposes no unrestricted internal-map iterator.

Capture fails unless that inventory:

- contains unique claim, evidence, and provenance IDs,
- exactly matches the ledger counts,
- resolves every declared record,
- uses contiguous ID namespaces beginning at `1`,
- therefore yields deterministic next claim/evidence/provenance IDs,
- contains no claim/evidence/provenance record from after the capture cycle,
- preserves every provenance-parent reference,
- preserves every evidence→claim and evidence→provenance reference,
- exactly reconciles each claim's evidence-ID list against the evidence records that actually target that claim,
- contains no duplicated evidence link inside a claim.

The derived next IDs are included in `EpistemicLedgerLineageV1` and in the manifest digest so a future restore cannot silently reuse a historical identifier namespace.

## Cross-capsule linkage

EKM-032 independently rechecks the EKM-030 → EKM-031 relationship:

- linked mutation count must match,
- every mutation must resolve to its source revision receipt,
- the source receipt must be eligible,
- claim IDs must match,
- proposed deltas must match exactly,
- authorization cannot predate the source decision,
- every persisted support-state claim must exist in the ledger.

This prevents pairing a decision-history capsule with a different mutation capsule merely because the two happen to share a capture cycle and count.

## Integrity digests

V1 uses the repository's existing BLAKE3 dependency with domain-separated component hashes:

- ledger lineage + full claim/evidence/provenance semantics,
- EKM-030 support/mutation capsule,
- EKM-031 revision-history capsule,
- top-level manifest binding the three component digests and derived ledger lineage.

Strings and byte sequences are length-prefixed. Integer IDs/cycles are little-endian. Floating-point support/delta values are hashed by their bit representation. Ledger enums use explicit stable V1 tags. Inventory ordering is normalized before hashing.

### Revision receipt encoding limitation

`BeliefRevisionReceipt` intentionally keeps revision-policy internals encapsulated. EKM-032 therefore labels its V1 encoding explicitly as `ExplicitFieldsPlusReceiptDebugV1`: each immutable receipt is integrity-bound using its complete derived Rust `Debug` representation under a versioned/domain-separated hash.

This is adequate for detecting changes within the V1 implementation, but it is **not claimed to be a cross-version or cross-language wire encoding**. A future on-disk restore format must replace this receipt subencoding with a fully specified canonical byte schema and bump the manifest version.

The limitation is represented in the public type system rather than hidden behind the word "canonical".

## Security boundary

The BLAKE3 digest provides integrity identity, not authentication. It is not a signature, MAC, trusted timestamp, or proof that the party creating the manifest was authorized.

Signer identity, cryptographic authorization, durable storage, and trusted rollback remain separate authority layers.

## Non-claims

EKM-032 does **not** establish:

- ledger hydration,
- `BeliefRevisionHistory` hydration,
- `EpistemicSupportStore` hydration,
- file/SQLite persistence,
- crash consistency or fsync behavior,
- cryptographic signer authentication,
- executable rollback,
- legacy `TemporalFact::confidence` migration,
- causal truth,
- world-model or action authority.

No method in this tranche writes epistemic state.

## Qualification boundary

This PR is stacked on EKM-031 / PR #3718. The parent exact-head CI run #7041 remains queued at authoring time. No format, compile, Clippy, test, or runtime PASS is inferred from queued or absent jobs.

The next authority increase should be a separate restore tranche that consumes one qualified manifest, reconstructs state in isolation, and proves post-restore equivalence before the reconstructed objects can become live.