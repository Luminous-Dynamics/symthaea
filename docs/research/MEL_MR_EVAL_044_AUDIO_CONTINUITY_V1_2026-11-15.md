# MEL-MR-EVAL-044 — Audio continuity and cumulative drift contract v1

**Date:** 2026-11-15  
**Status:** Stage 1 source/tests authored; exact-head qualification pending  
**Issue:** [#12327](https://github.com/Luminous-Dynamics/symthaea/issues/12327)  
**Stack base:** EVAL-043 [PR #12245](https://github.com/Luminous-Dynamics/symthaea/pull/12245)

## Stage 1 scope

A standalone Rust 2024 crate validates deterministic PCM session properties. The canonical input is interleaved signed 16-bit little-endian PCM plus a pinned sample rate, channel count and frame count.

The artifact digest is domain-separated SHA-256 over the sample format, frame count and every exact sample byte. Every step must consume the exact prior audio digest and return the same sample format and frame count. The verifier does not silently resample, change channel count or alter duration.

Each step declares:
- semantic edit regions;
- additional transition/crossfade support;
- a versioned boundary policy and measurement radius;
- expected output digest and upstream rights, lineage, realization and transaction gates;
- attempt, failure, render, retry and manual-intervention counters.

Only the union of the frozen edit regions and transition support is authorized to change. Every sample value outside that union must remain bit-exact relative to the immediate parent. A crossfade window cannot be silently expanded by the verifier.

## Boundary evidence

For every edge of the combined authorized support, the harness records per-channel:
- adjacent-sample jump in parent and output;
- absolute change in the boundary jump;
- parent and output left/right short-window RMS discontinuity;
- drift in that left/right RMS discontinuity.

Each step also records change relative to the frozen original source, the cumulative authorized frame union, source-relative maximum sample residual and source-relative off-support mutation count. This gives a per-step and terminal drift curve without confusing immediate-parent changes with session-root drift.

These are deterministic diagnostic measurements, not universal perceptual thresholds. A boundary gate becomes Verified only when both calibrated thresholds are present and satisfied. Missing thresholds produce Unknown/Abstain; exceeding either threshold produces Rejected.

No single threshold should be used indiscriminately for hard cuts, crossfades, transients, sustained instruments, stereo phase relationships and different sample rates. Calibrate thresholds by operation, format and listener cohort.

## Implemented controls

Seventeen unit tests are authored for:
- exact preservation outside the support;
- one-sample off-support leakage;
- declared transition support versus undeclared dilation;
- exact parent-digest chaining;
- sample-rate/channel-format substitution;
- frame-count changes;
- digest binding to format and samples;
- missing boundary thresholds as Unknown;
- threshold violations as Rejected;
- refusal to continue after an unknown step;
- two-step parent chaining and cumulative cost;
- canonical window order and support bounds.

The deterministic eight-step example modifies only eight disjoint, predeclared stereo PCM regions and emits a sealed session report.

## Qualification

The exact-head workflow uses Rust 1.96.0, checks out and verifies the exact PR head, validates the JSON Schema, stages the isolated crate outside the large workspace, normalizes and checks formatting, runs tests and the eight-step fixture, runs Clippy with warnings denied, and uploads source/schema/lockfile identities.

**No compilation, test, formatting or Clippy result is claimed until the workflow passes on the exact current head.** The formatter currently normalizes a staged copy; this does not prove committed source was originally rustfmt-clean.

## Limits

This stage has no real audio corpus, backend calls, DAW/plugin invocation or human listening study. It tests synthetic deterministic PCM only. Its adjacent-sample and RMS measures cannot establish that a musical seam sounds natural, that a replacement sounds idiomatic, or that the resulting performance is musically coherent.

It does not cryptographically reverify EVAL-041 receipts; those upstream receipt digests and EVAL-040 decision identities are schema-bound inputs, not independently reauthorized by this harness. EVAL-040 remains the sole terminal transaction authority.

## Prospective targets

- 100% exact off-support sample preservation on deterministic splice fixtures.
- Zero false accepts on frozen off-support, timebase, sample-format and transition-scope controls.
- 100% exact audio-byte restoration for operations declared byte-reversible.
- 98% per-step and 85% complete eight-step acceptance on a powered, rights-cleared real-audio cohort.
- At least 90% detection of preregistered audible seam defects with no more than 5% false alarms on benign controls, calibrated against blinded listeners.
- Lower p95 cost per fully accepted multi-step edit than repeated whole-track regeneration at matched operation mix and listener-rated quality.

All real-audio thresholds remain prospective until a preregistered study with independent listening and held-out renderers is complete.