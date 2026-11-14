# MEL-MR-EVAL-031 — Rights-gated music-edit byte materializer v1

**Date:** 14 November 2026  
**Status:** Implementation tranche; exact-head qualification pending  
**Parent:** [MEL-MR-EVAL-030 / PR #9284](https://github.com/Luminous-Dynamics/symthaea/pull/9284)  
**Tracking:** [Issue #12003](https://github.com/Luminous-Dynamics/symthaea/issues/12003)

## Objective

The EVAL-030 adapter validates manifest declarations. It does not open files. EVAL-031 adds the missing byte boundary: before an external editor or evaluator receives a dataset, every declared input, paired target, instruction and permission-evidence artifact must be found under an explicit local root and hashed from actual bytes.

The operation matrix remains unchanged: all twelve SMEBench operation families are NotYetQualified. Materializing data is not evidence that Muse can perform an edit.

## Paradigm

Treat a benchmark sample as a lineage-bound bundle of bytes and permissions, not a row of metadata. The materializer binds dataset ID, immutable version and benchmark track; manifest and artifact-map commitments; source recording, instruction and paired target; optional symbolic source, target selector and annotation metadata; source-lineage ID and split; local rights/access evidence when declared; and actual byte count and SHA-256 for every artifact role.

A deterministic receipt binds the complete materialization result. Its integrity verifier checks receipt structure, canonical ordering, manifest/map commitments, coverage and receipt digest. Re-running materialization reopens and rehashes files; receipt replay alone does not re-read them.

## Implemented contract

- Typed artifact-path map with exact sample-ID coverage.
- Rights/access evidence paths are required exactly when corresponding evidence digests exist; symbolic paths are required exactly when the source symbolic digest exists.
- Root-relative paths only. Rejects absolute paths, traversal components, empty segments, backslashes, colon-bearing paths, symlinks at any path component, non-regular files and canonical root escapes.
- Streaming SHA-256 over the opened file handle, with per-artifact and total-byte limits and before/after file metadata checks.
- Any mismatch between declared digest and actual bytes rejects the entire materialization.
- Exact byte-identical source/target audio under distinct lineage IDs rejects as a conservative duplicate signal.
- Domain-separated deterministic receipt with relative paths only; no host absolute paths are included.
- Standalone command example writes a receipt only after materialization and structural replay succeed.
- Ten authored adversarial/positive test functions cover successful receipt, source mutation, traversal/missing map, symlink, rights-evidence mutation, receipt/map tampering, exact cross-lineage duplicate and size limits.

## Explicit limits

A digest match establishes only that bytes read in this invocation match the declared digest. It does **not** authenticate the supplier, issuer, license, annotation author or legal permission. A copied permission document whose bytes match the manifest is not a legal rights review.

Exact byte equality does not detect alternate encodings, transcoding, cropping, remixes, time-stretching or pitch-shifted near-duplicates. Those require a separate acoustic fingerprinting programme with measured false-positive and false-negative controls. No perceptual fingerprint or similarity threshold is claimed here.

The path walk rejects observed symlinks and canonical escapes, but this is not a hostile-local-user sandbox or cryptographic filesystem snapshot. The caller must provide a controlled, quiescent local root. It does not provide secure archive extraction, download, audio decoding, scorer execution or editor execution.

No real benchmark assets are committed or fetched by this PR. The synthetic artifact-map fixture is a schema/contract fixture only; its referenced paths are not present and it is not materializable.

## Benchmark implications

Keep these evaluation tracks separate:

1. **SMEBench controlled MIDI-synthesized stratum:** operation-specific target-property agreement and source preservation, all twelve operations reported independently.
2. **SMEBench real multitrack complement:** separate reporting; synthetic success does not imply generalization to real multitrack recordings.
3. **MMAE public Agent development scope:** reproduce the official 2,000-example Agent baseline and native scorer first. The published baseline reports IFR 44.06%, CR 74.63% and EMR 7.45% on that full public scope. The 1,003-example Single Model subset is a different comparison scope.
4. **Human evaluation:** edit execution, preservation, plausibility, audible artifacts, instruction scope, listener preference and confidence remain separate judgment dimensions. Cluster uncertainty by source lineage/item and participant; use unseen renderer/instrument families for metric calibration.

Prospective targets (not measurements): at least 95% target-property success on supported held-out controlled operations; zero false acceptance on deterministic protected-content mutation controls; on the identical full MMAE Agent development scope, EMR at least 12%, CR at least 76.63% and IFR at least 44.06%, subject to a reproduced baseline and paired source-lineage-clustered intervals. Publish per-operation and per-stratum failures; retain abstentions and failures in the denominator.

## Validation state

The production adapter extension, schemas, synthetic map fixture, CLI example, standalone harness and workflow integration are part of this implementation tranche. Ten materializer test functions are authored. Compilation, formatting, tests and Clippy are not claimed until the exact-head Rust 1.96 workflow finishes. No local Rust execution result is asserted.

## Next boundary

After exact-head tests pass, separate increments are needed for acoustic near-duplicate/fingerprint review with calibrated error bounds; an authorized public MMAE scorer adapter and exact-scope baseline reproduction; renderer invocation/output qualification and operation-specific symbolic edit validators; and blind human calibration and musician-led agency studies.

No hidden-set credentials, registration status, dataset license, benchmark score or SOTA superiority is inferred.
