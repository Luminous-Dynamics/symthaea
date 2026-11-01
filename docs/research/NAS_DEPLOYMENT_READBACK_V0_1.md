# Artifact-bound physical flash readback verifier v0.1

Tracking: NAS-READBACK-001 / #9601  
Build artifact identity: #9594 / PR #9595  
Success quota: #9598 / PR #9646  
Fleet scope: #9511

## Purpose

A successful compiler or flashing-tool exit code does not prove that a target contains the expected bytes. This module verifies exact raw readback payload bytes against the model and firmware binary bytes that produced a VerifiedDeployableArtifact.

## Evidence chain

1. Hash the source model and firmware byte vectors and compare them to the exact VerifiedDeployableArtifact binary digests.
2. Bind the readback session to that artifact, pipeline, build receipt, target profile, board ID and board revision.
3. Pin device-identity evidence, probe ID, flash-tool executable, readback protocol and session identity/sequence by digest or stable ID.
4. Hash the raw readback capture and the independently observed address/offset/length map separately.
5. Compare every observed region ID, flash address and byte length to the expected build/linker mapping.
6. Compare every readback byte directly with the exact source binary slice; digest equality is additional evidence, not a substitute for byte equality.
7. Require complete, gap-free and non-overlapping coverage of both model and firmware binaries, the capture payload, and target flash address ranges.
8. Issue a non-serializable VerifiedDeploymentReadback capability and canonical receipt only after all byte checks pass.
9. Append only that capability to a per-device hash-chained session ledger; require unique session IDs and strictly increasing session sequences.
10. Bind subsequent physical measurements to both the exact deployable-artifact digest and exact readback-receipt digest.

## Capture representation

The capture is a compact concatenation of region bytes in the offsets supplied by the acquisition adapter. The capture hash covers the raw bytes. The region-map hash covers the strictly region-ID-sorted observed region descriptors. The verifier independently checks every region and exact source coverage.

## Claim ceiling

This validates the *supplied readback bytes* against supplied source bytes and declared mapping. It does not authenticate the physical probe, device serial, flash tool, acquisition channel or caller. Device identity evidence is digest-pinned but not cryptographically validated here. The local serialized session ledger detects duplicate IDs and non-monotonic sequences relative to its supplied history, but it is not a durable atomic store, trusted monotonic clock, hardware-backed anti-rollback mechanism or externally witnessed global ledger. A party able to rewrite the entire ledger and recompute its hashes can replace its history unless an external checkpoint is retained.

Protected/encrypted flash needs a separately qualified measured-boot attestation path; do not call that ordinary byte readback. Secure-element evidence, trust roots, keys, probe custody, actual hardware acquisition and independent operator verification remain separate.

## Negative controls

Twelve Rust tests are authored for exact match, altered readback byte, altered address even after rehashing the region map, truncated capture, source-coverage gaps, overlapping flash regions, target/profile mismatch, receipt recomputation, exact readback-bound measurement, unknown receipt fields, duplicate/non-monotonic session rejection and session-ledger tamper detection. Tests are authored, not compiled or run in this environment.

## Benchmark integration

A physical metric promoted to fleet qualification should reference:
- exact VerifiedDeployableArtifact;
- exact VerifiedDeploymentReadback;
- target profile and device identity evidence;
- same session identity/sequence;
- measurement protocol/instrument evidence;
- raw trace capture digest and replicate identity.

A compiler or programmer success must never count as verified deployment success.

## Aspirational acceptance targets (not achieved)

- 100% of deployable binary bytes accounted for by exact address/offset/length readback evidence.
- Zero altered, partial, truncated, wrong-target or address-mismatched captures accepted.
- Zero build-only or programmer-success-only attempts counted as verified deployment.
- 100% of promoted physical metrics reference an artifact-bound readback receipt.
- Readback overhead, mismatch rate, incomplete capture rate and physical deployment yield reported separately.
