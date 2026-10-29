# PITB signed coverage receipts: design and open qualification work

Date: 2026-10-29

Status: first-stage implementation on the research branch; **no real independent receipt exists and no component manifest is qualified**.

## Why this is a critical blocker

The current family-component manifest builder intentionally emits `qualified_for_confirmatory_resampling = false`. It sees a digest pointer and self-declared `independently_reviewed_complete` status, but it does not authenticate the bytes, reviewer keys, or independent provenance. Treating a digest as a signature would turn a provenance label into an authorization boundary.

The previous inventory shape also placed the receipt digest inside the relationship inventory, which makes signing the entire inventory byte digest and storing the signed receipt digest inside that same inventory circular. The verifier avoids that cycle by having reviewers sign the normalized relationship-edge payload digest, while the inventory points to the final receipt digest. The exact review-methodology and trust-store byte digests are included in the signed message.

## Implemented in this increment

- Added the Rust binary `verify_component_coverage_receipt` with strict serde input types and Ed25519 signature verification.
- Recomputes the fixed source partition from the pinned upstream commit, split seed, algorithm, assignments, and count contract before accepting the receipt.
- Canonicalizes the edge payload by sorting edges by `edge_id` and reviewer IDs lexicographically before hashing; validates allowed relation/status values, digest syntax, unique IDs/pairs, and references to known families.
- Requires at least two valid endorsements from distinct organizations, where reviewer IDs/key IDs resolve through the pinned trust-store bytes.
- Requires both the trust-store digest and methodology digest to match explicit expected digests (which must be frozen in the versioned run config by the future integration), in addition to matching the receipt's signed values.
- Emits a verification report whose contract explicitly says `requires_component_and_analyzer_validation`. Receipt verification **does not** set the component manifest's qualification flag or authorize a confirmatory claim.
- Added Draft 2020-12 schemas and shape-only synthetic fixtures for receipts, trust stores, and verification reports. These fixtures do not contain a valid signature set and must never be mistaken for auditor evidence.

## Remaining integration gates

1. Add the coverage trust-store digest and receipt artifact to the frozen run config and analysis report; bump the schema version rather than silently changing an existing config contract.
2. Make the component-manifest builder consume the exact verified report/receipt and keep qualification false for incomplete inventory or any non-dismissed hard relationship crossing splits.
3. Make `analyze_pitb_results` reverify the receipt and trust-store bytes itself, or verify a verifier report whose full inputs and executable digest are pinned. It must not trust a bare boolean or a user-editable status field.
4. Add an end-to-end workflow case that generates deterministic test keys in Rust, creates a signed synthetic receipt in memory, runs the CLI through the exact frozen partition path, and exercises bad-key, changed-methodology, changed-edge, same-organization, incomplete-inventory, and wrong-trust-root negatives.
5. Obtain and independently audit a genuine reviewer trust root and two reviewers from distinct organizations. Until this occurs, the inventory stays unqualified regardless of the code's synthetic positive tests.
6. Run the new Rust tests and exact-head workflow. The current environment has no local Rust toolchain, and no CI result for the new head has been observed. A green schema test would not establish cryptographic verification, reviewer independence, or scientific validity.

## External comparison and benchmark targets

This is infrastructure to qualify evidence, not a prover score improvement. The relevant targets remain:

- zero confirmatory passes in the registered false-receipt, wrong-key, mutated-edge, changed-methodology, same-organization, and incomplete-inventory controls;
- exact report replay from the same pinned files and trust-store bytes;
- no qualified manifest when a non-dismissed hard component crosses train/validation/test;
- no confirmatory claim unless statistical Type-I error and power gates pass using the component clusters;
- an independently reviewed receipt whose provenance and reviewer organizations are not controlled solely by the party running the benchmark.

A signed completeness claim is still an attestation, not a mathematical proof that no relationship was missed. The review logs must be auditable, and a fresh-source/temporal replication remains necessary.
