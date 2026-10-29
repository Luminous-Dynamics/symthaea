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

## Remaining integration and qualification gates

1. **Implemented in this branch:** run-config v3 all-or-none artifact pins, shared receipt verifier, component-builder signature verification, analyzer-side re-verification in both `analyze` and `power` modes, and manifest state reconciliation. The old v2 schema is retained as historical contract; the workflow validates v3.
2. **Still required:** an end-to-end workflow test must generate deterministic synthetic Ed25519 keys, sign a receipt over the exact pinned partition/inventory/methodology/trust-store, drive the real builder and analyzer CLIs, and reject bad-key, changed-methodology, changed-edge, same-organization, incomplete-inventory, wrong-trust-root, missing-input, and changed-manifest controls.
3. Obtain and independently audit a genuine reviewer trust root and two reviewers from distinct organizations. Until this occurs, the checked-in fixture cannot qualify the actual inventory regardless of passing synthetic tests.
4. Run the new Rust tests and exact-head workflow. The current environment has no local Rust toolchain, and no passing CI result for the current head has been observed. Static inspection and JSON fixture checks are not substitutes for a successful compiler/test run.
5. Independently audit retained review logs for the full 372-family review and verify that trust-store provisioning was not controlled solely by the experiment author. Signatures authenticate endorsements under a trust root; they do not prove review competence or logical exhaustiveness.

## External comparison and benchmark targets

This is infrastructure to qualify evidence, not a prover score improvement. The relevant targets remain:

- zero confirmatory passes in the registered false-receipt, wrong-key, mutated-edge, changed-methodology, same-organization, and incomplete-inventory controls;
- exact report replay from the same pinned files and trust-store bytes;
- no qualified manifest when a non-dismissed hard component crosses train/validation/test;
- no confirmatory claim unless statistical Type-I error and power gates pass using the component clusters;
- an independently reviewed receipt whose provenance and reviewer organizations are not controlled solely by the party running the benchmark.

A signed completeness claim is still an attestation, not a mathematical proof that no relationship was missed. The review logs must be auditable, and a fresh-source/temporal replication remains necessary.
