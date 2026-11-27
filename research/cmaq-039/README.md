# CMAQ-039 — Closure-gated rank-reference admission

## Purpose

CMAQ-036 computes transitive common-cause components and one conservative max-dangerous score per effective block. CMAQ-031 consumes one frozen score per independent calibration unit. CMAQ-039 is the typed admission bridge between them.

The central invariant is:

    number of CMAQ-031 reference scores = CMAQ-036 calibration_effective_block_count

Raw episode count is diagnostic only. It never becomes the sample size.

## Implemented

- Validates the CMAQ-036 closure algorithm identifier and consistency of block, member, split and pseudoreplication inventories.
- Requires a closure authority binding for the exact closure digest, source manifest, calibration profile, calibration manifest, scope and score definition.
- Requires exactly one evaluator attestation per calibration IndependentBlock.
- Verifies exact block maximum, sorted member IDs, member score receipt inventory and dependence-key receipt inventory.
- Verifies every score attestation binds the same closure digest and score definition.
- Requires block-manifest, aggregate-score and tie-break receipt digests.
- Rejects duplicate tie keys, missing or extra attestations, non-calibration attestations, noncanonical inventories and count tampering.
- Produces a ClosureQualifiedRankReference with a private underlying reference; callers cannot move the raw CMAQ-031 reference out of the wrapper.
- Converts that qualified reference into a ClosureQualifiedMonitor, which retains the closure digest and effective-block diagnostics while delegating online observations, quarantine state and wealth reporting to CMAQ-031.
- Preserves closure provenance during the normal calibration-to-monitor handoff rather than dropping it immediately after monitor construction.

The result has two calibration scores for the four-episode adversarial fixture: the three linked episodes contribute one maximum score of 90; the unlinked fourth episode contributes one score of 20.

## Explicit trust boundary

This crate checks typed correspondence and digest syntax. It does not verify signatures, signer authority, canonical serialization, receipt authenticity, dependency inventory completeness or physical independence. The evidence authority must authenticate the closure manifest and all attestations before creating ClosureAuthorityBinding.

CMAQ-036 only closes declared keys. A pair of disjoint dependency inventories is not proof that the physical units are independent. Unknown or unresolved common causes must be merged or blocked by the evidence authority.

The closure digest is carried as an externally authenticated binding because CMAQ-036 v0.1 returns a structured ClosureManifest but does not itself compute or sign a canonical manifest digest. CMAQ-039 does not invent one.

## Tests

The 13-test deterministic suite covers:
- transitive asset/session closure and max-score preservation;
- one score per effective block, not one per raw episode;
- CMAQ-031 monitor compatibility and retention of closure provenance in the runtime wrapper;
- missing and duplicate attestations;
- wrong closure digest;
- aggregate score different from the closure maximum;
- noncanonical member/receipt order;
- duplicate randomized tie-break keys;
- tampered effective count;
- source-manifest mismatch;
- malformed member score receipts and dependence-key receipts;
- duplicated member identity.

## Scope not yet implemented

This first increment bridges closure-qualified aggregate scores into CMAQ-031 monitoring. It does not yet construct CMAQ-029 trajectory calibration manifests or CMAQ-030 raw candidate/reference margin blocks. Those paths need typed block-level score attestations at their own exact scope. In particular, CMAQ-039 must not fabricate candidate/reference margin cells from a scalar maximum. Follow-on integration should add explicit score-attestation APIs in CMAQ-029 and CMAQ-030, then feed those outputs into CMAQ-032 with the closure digest preserved end to end.

## Qualification

The dedicated workflow runs Rust formatting, deterministic tests and strict Clippy on the exact PR head. No signature, physical-independence, calibration-coverage or safety claim follows from a successful Rust workflow.
