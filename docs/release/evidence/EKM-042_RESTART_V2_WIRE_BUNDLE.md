# EKM-042 — Restart V2 Wire Bundle

## Purpose

EKM-034 already owns base epistemic-state serialization and EKM-040 owns typed policy/decision sidecar serialization. EKM-042 composes those two complete envelopes without duplicating either format.

## Central invariant

**A restart-v2 transport bundle is framing, not admission: outer checksum validity and valid sub-envelopes do not by themselves prove that the base state, typed schema state, and claimed V2 digest describe the same epistemic image.**

## Bundle contents

The outer envelope carries:

- complete EKM-034 V1 base-state wire bytes
- complete EKM-040 typed schema-wire bytes
- claimed EKM-039 V2 capsule digest
- outer domain-separated BLAKE3 corruption checksum

Each sub-envelope remains independently checksummed and parsed by its own implementation.

## Framing and bounds

The bundle uses:

- fixed V2 magic
- explicit outer version and encoding tag
- explicit payload length
- 300 MiB maximum base sub-envelope
- 80 MiB maximum schema sub-envelope
- 384 MiB maximum outer payload
- checked integer conversion and length arithmetic
- exact trailing-byte rejection

## Decode boundary

`EpistemicRestartWireV2::decode` performs:

1. outer framing and checksum validation,
2. bounded extraction of both sub-envelopes,
3. EKM-034 parsing/checksum validation,
4. EKM-040 parsing/checksum validation,
5. extraction of the claimed V2 digest.

It returns only `EpistemicRestartWireSnapshotV2` containing two untrusted snapshots and the untrusted claimed digest.

It deliberately does **not** cross-check capture epochs, receipt identities, typed policies against base receipts, evidence-dependent decision failures, or recompute the V2 digest. Those are semantic admission checks for the next tranche.

## Authority boundary

No wire-to-capsule construction, quarantine construction, writable hydration, authorization, activation, evidence/belief mutation, causal/world-model/action changes, file I/O, or network I/O.

## Qualification boundary

Exact-head CI remains authoritative. Queued runs do not establish format, compile, Clippy, test, or runtime PASS.
