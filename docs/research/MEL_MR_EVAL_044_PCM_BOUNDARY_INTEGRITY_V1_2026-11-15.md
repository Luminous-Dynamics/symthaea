# MEL-MR-EVAL-044 — PCM boundary integrity contract v1

**Date:** 2026-11-15  
**Status:** Source, schema, fixture and workflow authored; exact-head qualification pending  
**Issue:** [#12268](https://github.com/Luminous-Dynamics/symthaea/issues/12268)  
**Stack base:** [EVAL-043 / PR #12245](https://github.com/Luminous-Dynamics/symthaea/pull/12245)

## Scope

This is the acoustic complement to EVAL-043's symbolic session contract. It verifies a frozen PCM representation, exact protected regions, authorized edit support and predeclared measurements at edit-window boundaries.

It does **not** infer that audio is musically good from waveform statistics. A passing boundary receipt means only that the declared byte-integrity and signal-feature thresholds passed. Human listening remains an independent evaluation track.

## v1 data model

- PCM is signed 32-bit interleaved little-endian samples, with frozen sample rate and channel count.
- Source and output audio are bound by domain-separated SHA-256 digests.
- The request commits to source lineage, operation plan, frame count, edit windows, protected regions and boundary thresholds before evaluation.
- Edit windows are sorted, nonempty and nonoverlapping. Any sample mutation outside them is rejected.
- Protected regions are sorted by start frame, disjoint from edit support, and verified against both the source and output region digest.
- Every boundary policy names an actual start/end edge of a declared edit window and contains left/right analysis lengths plus fixed thresholds.
- Each output boundary reports metrics per channel. A single bad channel cannot be averaged away by the others.

## Objective boundary features

For each channel and declared boundary, v1 computes:

1. Seam jump: absolute sample delta across the boundary.
2. Slope discontinuity: absolute change in first-difference across the seam.
3. Maximum local sample step within the combined analysis window.
4. Left/right integer-RMS ratio deviation in parts per million.

The threshold vector is part of the frozen request. A threshold miss becomes an explicit negative receipt (passed=false); wrong source/output bytes, changed protected regions and unauthorized support mutations fail closed.

## Deterministic controls

The standalone crate currently authors eleven unit tests covering:
- exact protected-region preservation and authorized change accounting;
- unauthorized changes outside edit support;
- protected-region digest mismatch;
- impulse/click at an edit edge;
- per-channel seam failure that cannot be averaged away;
- protected/edit-support overlap;
- request and PCM digest binding;
- boundary-policy binding to edit-window edges;
- receipt tamper and policy substitution;
- malformed interleaving and metadata mismatch;
- deterministic feature reports and receipt digests.

A frozen 12-frame mono PCM known-answer fixture modifies two samples inside the declared support, preserves the head region exactly, evaluates both edit edges, and validates the content-bound receipt.

## Exact-head workflow

The workflow pins Rust 1.96.0, checks out and verifies the exact pull-request head, validates the JSON Schema and fixture, stages the standalone crate outside the repository workspace, formats and checks the staged copy, resolves the lockfile, runs tests and the fixture example, runs Clippy with warnings denied, and uploads exact source/schema/fixture/dependency identities.

**Until that workflow passes for the exact current PR head, compilation, tests, formatting and Clippy remain pending.** The workflow normalizes the staged copy before checking formatting; it does not claim committed source was already rustfmt-clean.

## Deliberate limitations

- Fixed timeline only: source and output frame count, sample rate and channel layout must match.
- No time-warp, tempo-map or frame-map support yet.
- No STFT/spectral flux, loudness-weighted perceptual model, stereo-image metric or source-separation residual analysis yet.
- No renderer execution or live DAW/plugin integration.
- No proof that the PCM arrays came from a trusted provider; actual file-byte materialization and provenance belong to EVAL-035/EVAL-041.
- No human-rated artifact-free or musical-continuity result.
- No licensed/consented real-audio benchmark result.

## Next stages

1. Extend deterministic negatives to phase/latency offset, channel swap, dropout, transient duplication and old-source residual.
2. Bind EVAL-044 receipts to EVAL-043 sequence steps and track boundary/session drift at 1/2/4/8 operations.
3. Add exact audio-byte materialization through EVAL-035/EVAL-041 and explicit time maps for timeline-changing operations.
4. Run a rights-cleared, consented real-audio cohort with blinded boundary, preservation and musical-continuity listening.
5. Compare matched operation/sample cohorts to Harmony, WavCraft, DAW/manual references and replaceable render backends, with every failed attempt and repair included in fully loaded cost.

No audible-quality or commercial-superiority claim is made by this v1 contract.
