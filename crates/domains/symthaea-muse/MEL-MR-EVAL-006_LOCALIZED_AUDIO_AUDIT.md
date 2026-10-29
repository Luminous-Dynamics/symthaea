# MEL-MR-EVAL-006 — Localized Rendered-Audio Preservation Audit

**Status:** executable Rust measurement primitive; calibration and end-to-end evaluation remain pending  
**Tracking:** [MEL-MR-EVAL-005 protocol](https://github.com/Luminous-Dynamics/symthaea/blob/research/mel-mr-eval-005-audio-preservation-protocol-20261029/crates/domains/symthaea-music-theory/MEL-MR-EVAL-005_RENDERED_AUDIO_PRESERVATION.md)  
**Implementation:** `crates/domains/symthaea-muse/src/audio_preservation.rs`

## Research change: from global scores to localized invariants

A global similarity score can hide a small but important corruption when most of a recording is unchanged. This module therefore evaluates complete short-time windows outside the authorized edit mask, per channel, and reports the worst values and every violating window. Time and channel are not averaged away. A channel-window violates the operational gate when either the absolute RMS-level change exceeds its configured dB tolerance or the normalized spectral-magnitude distance exceeds its configured threshold.

This is a deliberately narrow, deterministic first detector—not a claim of perceptual equivalence. It does not yet detect every musical invariant (for example, source attribution inside a dense mix), infer expected effect tails, or distinguish all semantically meaningful changes from harmless synthesis variation. Spectral magnitude is phase-insensitive; a separate timing/onset detector and blind listening study remain needed.

## Measurement contract

- Inputs are interleaved `f32` buffers with explicit sample rate and channel count.
- Edit intervals use half-open audio-frame coordinates, not scalar sample-array offsets.
- Both inputs must have the same sample rate, channel count, and frame count. There is no automatic alignment, resampling, trimming, or loudness normalization.
- Frames intersecting the authorized mask expanded by a configured boundary guard are skipped. The guard is an explicit allowance, not a learned reverb-tail model.
- Every other complete window is measured independently per channel.
- Threshold exceedance in any protected channel-window blocks acceptance. Non-finite audio, malformed masks, and incompatible inputs are errors.
- Passing additionally requires an externally maintained calibration ID and a minimum number of protected windows. An identifier is only recorded; the module cannot prove that calibration is real, appropriate, or frozen.
- The report contains SHA-256 commitments over domain-separated sample bits and format metadata. They bind the receipt to supplied buffers but do not authenticate authorship, capture time, or renderer identity.

## Tests included

The Rust unit suite covers identity preservation, one localized protected-region mutation, a mutation confined to the authorized mask, no-calibration abstention, a stereo single-channel mutation, mismatched duration, non-finite samples, and boundary-guard behavior. These tests are committed but **not yet executed in this environment**; a passing CI result must be attached to the exact PR head before implementation validation is claimed.

## Calibration and benchmark plan

1. Pin renderer version, sample rate, channel layout, instrument bank/effects, effect-state resets, and latency handling. Record where exact deterministic replay is not available.
2. Generate matched source/output pairs with known truth: identity, authorized-only mutation, wrong-region edit, wrong-channel edit, gain shift, frequency substitution, onset displacement, clipping, phase-only variation, reverb-tail extension, and renderer-version mismatch.
3. Split by composition/motif family **and** renderer/configuration. Fit thresholds on a calibration partition; freeze them before held-out testing. Deduplicate source identities before splitting.
4. Measure sensitivity and false-accept rate per mutation family, not only overall. Publish counts and confidence intervals. An initial target is zero accepted critical mutations in at least 300 independently generated/annotated negatives, but this is a pilot target—not a safety proof.
5. Add an independent onset/timing channel and source-aware checks before expanding the detector's claim domain.
6. Run blinded listening trials to measure whether detector verdicts track audible collateral changes. Keep technical preservation, requested-edit success, naturalness, and preference as separate endpoints.

## Required reporting

Report raw denominators, calibration set ID, held-out composition and renderer families, thresholds, mutation-family confusion matrix, protected-window coverage, abstentions, p50/p95 runtime, Rust commit SHA, exact invocation, and input/output hashes. Never translate a deterministic fixture pass into a general claim that audio is perceptually preserved or that Symthaea beats a state-of-the-art editor.

## Benchmark gate (targets, not attained results)

| Gate | Target for a future calibrated study |
|---|---|
| Deterministic controls | 100% expected outcomes on the predeclared conformance suite |
| Critical mutation sensitivity | No critical mutation accepted in the test panel; publish confidence bounds |
| Requested edit | Report separately; the preservation detector alone cannot establish it |
| Human-audible collateral change | Blinded human agreement and detector sensitivity/specificity with intervals |
| Distribution shift | Report by unseen composition and unseen renderer strata |
| Reproducibility | Exact receipt recomputation and rejection of cross-input/tampered receipts |
| Efficiency | p50/p95 detector latency, including FFT work, separated from rendering time |

The first paradigm shift is architectural: use typed authorization masks plus independent localized detectors and content-bound receipts, then test the *incremental causal effect* of that guard against the same editor with matched inputs and budgets. This does not remove the need for stronger detectors or external baselines; it makes that comparison falsifiable.
