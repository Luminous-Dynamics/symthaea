# MEL Audio Transform Qualification v1

**Status:** standalone Rust reference validator with synthetic byte fixtures; no human equivalence study or audio transform is claimed.

## Purpose

A transformation is not a nuisance by name. It is eligible only for a declared task dimension, after a separately supplied independent listening panel has demonstrated equivalence using a preregistered margin and an interval method that accounts for listener/item dependence.

For example, static gain normalization may be acceptable for arrangement preference but invalid for a dynamics judgment. Stereo-to-mono can erase spatial information. Tempo changes are invalid for tempo-sensitive prompts. Lossy encoding may erase transient or artifact evidence.

## Contract

The manifest binds exact source and variant SHA-256 values, an implementation/build digest, a complete parameter digest, task and rubric digests, audio metadata, rights/consent receipt digests, and a separate equivalence receipt. The validator recomputes source/variant file hashes with sha256sum and fails on byte mismatch.

The transformation must list the task dimension as preserved and must not list that dimension as possibly altered. The independent receipt must:
- use a paired cluster bootstrap or crossed listener/item mixed-effects method;
- match the preregistered confidence level and equivalence margin;
- include a confidence interval wholly inside [-margin, +margin];
- meet the preregistered minimum listener and item counts; and
- carry digests for the panel protocol, cohort commitment, analysis plan and role-independence attestation.

An underpowered, inconclusive, non-equivalent or out-of-margin receipt is reported as **ineligible**, never as a pass. Non-significance alone is not equivalence.

## Run

    cargo +1.96.0 run --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml \
      --bin mel-audio-transform-qualification -- audit \
      tools/mel-judge-grounding-audit/fixtures/valid-transform-qualification-v1.json \
      tools/mel-judge-grounding-audit/fixtures/source-audio.bin \
      tools/mel-judge-grounding-audit/fixtures/variant-audio.bin \
      report.json

The CLI requires sha256sum on PATH. It validates file bytes, not decoded audio streams. The fixture files are deliberately non-audio placeholders so no one mistakes synthetic conformance for perceptual evidence.

## Claim ceiling

This validator does not authenticate human receipts, prove role separation, open/decode audio, verify sample-rate/channel/duration metadata against the file, or prove perceptual equivalence. A digest is not a signature. Signed/transparency-anchored provenance and isolated judge execution remain in MEL-SEALED-ATTEST-001 (#13254). A transform qualified for one task dimension is not thereby safe for another.
