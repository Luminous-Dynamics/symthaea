# TLSNotary Beta.1 / RISC Zero Presentation Probe

**Status:** Source-authored feasibility probe. Not compiled and not executed.

## Purpose

Build a separate RISC Zero guest that verifies a serialized TLSNotary v0.1.0-beta.1 Presentation using its official `Presentation::verify()` API, then checks the expected notary key, server DNS name, TLS version, received transcript length and exact trust-root profile.

Pinned versions:
- TLSNotary: `v0.1.0-beta.1`, commit `40a069565a2eaecbb5e691dbc634be20e8498daf`;
- RISC Zero: `3.0.6` for this isolated experiment;
- guest `std` feature enabled because Beta.1 attestation/core use `std` APIs.

The live TLS MPC/MPZ protocol is not a guest dependency. Only `tlsn-attestation` and `tlsn-core` verification crates are selected.

## Important limitation

Beta.1's public `PresentationOutput` exposes the authenticated `PartialTranscript`, server identity and connection information, but not the typed transcript commitment list. That list is hidden behind `pub(crate)` accessors on `Attestation::Body`. This probe therefore verifies the presentation and journals the authenticated byte ranges, but does not prove a hidden value opens the exact signed commitment or run a private field predicate. See Symthaea #12373.

The host verifier must compare every journal field with its own request and trust profile. A root set, notary key, origin or TLS profile supplied by the prover is not authority by itself.

## Why 3.0.6

RISC Zero 3.1.0 (released 10 November 2026) regenerated the guest and recursion circuits; receipts are not mutually compatible across 3.0.x and 3.1.0. The probe pins the 3.0.6 line as an isolated compatibility experiment. Do not change the legacy guest's version or ImageID as part of this probe.

## Reproducibility

`methods/build.rs` embeds the guest ELF and distinct ImageID. This branch has no host prover runner yet. Compile and verify status are not claimed. The next build step must use the exact RISC Zero 3.0.6 toolchain and record the guest ELF digest, ImageID, guest memory/cycles, receipt size and native prove/verify timings.
